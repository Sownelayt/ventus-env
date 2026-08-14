"""Two-sample, cost-bounded TensorMap decode probe for H100 and B200."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "cuda_tensormap_decode.cu"
AUDITOR = HERE / "audit_instruction_stream.py"
REMOTE = "/opt/tma-decode"
IMAGE = "nvidia/cuda:13.3.0-devel-ubuntu24.04"
TARGETS = {
    "h100": {"gpu": "H100!", "arch": "sm_90", "cc": "9.0"},
    "b200": {"gpu": "B200", "arch": "sm_100", "cc": "10.0"},
}
CHILD_TIMEOUT = 2
OUTER_TIMEOUT = 10


def build(arch: str) -> str:
    root = f"{REMOTE}/{arch}"
    common = f"nvcc -O3 -lineinfo -std=c++20 -arch={arch} {REMOTE}/{SOURCE.name}"
    return (
        "bash -lc 'set -euo pipefail; "
        f"mkdir -p {root}; {common} -o {root}/probe -lcuda; "
        f"{common} -ptx -o {root}/probe.ptx -lcuda; "
        f"{common} -cubin -o {root}/probe.cubin -lcuda; "
        f"nvdisasm {root}/probe.cubin > {root}/probe.sass; "
        f"python3 {REMOTE}/{AUDITOR.name} {REMOTE}/{SOURCE.name} "
        f"{root}/probe.sass --output {root}/instruction_audit.json; "
        f"grep -q prefetch.tensormap {root}/probe.ptx; "
        f"grep -q cp.async.bulk.tensor.5d {root}/probe.ptx; "
        f"grep -q cp.async.bulk.wait_group {root}/probe.ptx; "
        f"test -s {root}/probe.sass'"
    )


app = modal.App("ventus-tma-decode-v4-cold-then-hot")
base = (
    modal.Image.from_registry(IMAGE, add_python="3.12")
    .add_local_file(SOURCE, f"{REMOTE}/{SOURCE.name}", copy=True)
    .add_local_file(AUDITOR, f"{REMOTE}/{AUDITOR.name}", copy=True)
)


def image(arch: str) -> modal.Image:
    return base.run_commands(build(arch))


def execute(label: str) -> dict[str, Any]:
    target = TARGETS[label]
    command = [f"{REMOTE}/{target['arch']}/probe"]
    started = time.monotonic()
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=CHILD_TIMEOUT, check=False)
        payload: dict[str, Any] = {"classification": "completed", "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except subprocess.TimeoutExpired as error:
        payload = {"classification": "child_timeout", "returncode": 124, "stdout": error.stdout or "", "stderr": error.stderr or ""}
    payload.update({"gpu": label, "command": command, "wall_seconds": time.monotonic() - started, "child_timeout_seconds": CHILD_TIMEOUT, "function_timeout_seconds": OUTER_TIMEOUT, "collected_at_utc": datetime.now(timezone.utc).isoformat()})
    query = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,clocks.current.sm,memory.total", "--format=csv,noheader,nounits"], text=True, capture_output=True, check=False)
    payload["nvidia_smi_query"] = query.stdout + query.stderr
    return payload


@app.function(image=image("sm_90"), gpu="H100!", timeout=OUTER_TIMEOUT, retries=0, max_containers=1)
def run_h100() -> dict[str, Any]:
    return execute("h100")


@app.function(image=image("sm_100"), gpu="B200", timeout=OUTER_TIMEOUT, retries=0, max_containers=1)
def run_b200() -> dict[str, Any]:
    return execute("b200")


@app.function(image=base, timeout=OUTER_TIMEOUT, retries=0, max_containers=1)
def import_probe() -> str:
    return hashlib.sha256(
        Path(f"{REMOTE}/{SOURCE.name}").read_bytes()
        + Path(f"{REMOTE}/{AUDITOR.name}").read_bytes()
    ).hexdigest()


def validate(payload: dict[str, Any], label: str) -> list[dict[str, str]]:
    if payload.get("classification") != "completed" or payload.get("returncode") != 0:
        raise SystemExit(f"{label}: {payload.get('classification')} returncode={payload.get('returncode')}")
    rows = list(csv.DictReader(io.StringIO(str(payload.get("stdout", "")))))
    if not rows:
        raise SystemExit("no CSV rows")
    if any(row["cc"] != TARGETS[label]["cc"] for row in rows):
        raise SystemExit("compute capability mismatch")
    if any(int(row["errors"]) != 0 for row in rows):
        raise SystemExit("correctness failure")
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["direction"], row["rank"], row["scenario"], row["prefetch_gap_cycles"])
        grouped.setdefault(key, []).append(row)
    seen_addresses: set[int] = set()
    seen_fingerprints: set[int] = set()
    for key, pair in grouped.items():
        pair.sort(key=lambda row: int(row["repeat"]))
        if [int(row["repeat"]) for row in pair] != [0, 1]:
            raise SystemExit(f"{key}: not exactly repeat 0/1")
        scenario = key[2]
        if scenario == "bulk_hot":
            continue
        addresses = [int(row["descriptor_va"], 16) for row in pair]
        fingerprints = [int(row["descriptor_fingerprint64"], 16) for row in pair]
        if any(address % 4096 for address in addresses):
            raise SystemExit(f"{key}: TensorMap is not on a dedicated 4 KiB page")
        if len(set(addresses)) != 1 or len(set(fingerprints)) != 1:
            raise SystemExit(f"{key}: cold/hot pair changed TensorMap identity")
        if addresses[0] in seen_addresses or fingerprints[0] in seen_fingerprints:
            raise SystemExit(f"{key}: TensorMap identity reused by another case")
        seen_addresses.add(addresses[0])
        seen_fingerprints.add(fingerprints[0])
        if [int(row["prior_tma_use"]) for row in pair] != [0, 1]:
            raise SystemExit(f"{key}: prior TMA use is not 0/1")
        expected_prefetch = [1, 0] if scenario in {"explicit_prefetch", "prefetch_lead"} else [0, 0]
        if [int(row["tma_prefetch_before_issue"]) for row in pair] != expected_prefetch:
            raise SystemExit(f"{key}: prefetch marker does not match scenario")
        expected_roles = ["prefetched", "hot"] if expected_prefetch[0] else ["cold", "hot"]
        if [row["pair_role"] for row in pair] != expected_roles:
            raise SystemExit(f"{key}: pair role does not match scenario")
        if any(row["descriptor_l2_warm_method"] != "ld.global.cg" for row in pair):
            raise SystemExit(f"{key}: descriptor was not made L2-hot with ordinary loads")
    return rows


@app.local_entrypoint()
def main(gpu: str = "h100", import_only: bool = False, output: str = "") -> None:
    if gpu not in TARGETS:
        raise SystemExit("--gpu must be h100 or b200")
    expected = hashlib.sha256(SOURCE.read_bytes() + AUDITOR.read_bytes()).hexdigest()
    if import_only:
        if import_probe.remote() != expected:
            raise SystemExit("mounted source hash mismatch")
        print(f"CPU-only import probe passed; source SHA-256={expected}")
        return
    payload = (run_h100 if gpu == "h100" else run_b200).remote()
    rows = validate(payload, gpu)
    output_path = Path(output).resolve() if output else HERE / "data" / gpu
    output_path.mkdir(parents=True, exist_ok=True)
    stdout = payload.pop("stdout")
    stderr = payload.pop("stderr")
    (output_path / "attempt_01.csv").write_text(str(stdout), encoding="utf-8")
    (output_path / "attempt_01.stderr.txt").write_text(str(stderr), encoding="utf-8")
    (output_path / "attempt_01.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_path / "manifest.json").write_text(json.dumps({"gpu": gpu, "measurement_release": "ColdThenHotV1", "warmups": 0, "repeats": 2, "paid_attempts": 1, "pairing": "one descriptor per case; same-kernel first use then immediate reuse", "source_bundle_sha256": expected, "rows": len(rows), "completed_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"saved: {output_path}")
