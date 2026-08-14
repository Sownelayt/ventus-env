"""Cost-bounded CUDA cooperative-copy and compute-overlap controls."""

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
SOURCE = HERE / "cuda_control_items.cu"
REMOTE = "/opt/tma-control-items"
TARGETS = {
    "h100": {"gpu": "H100!", "arch": "sm_90", "cc": "9.0"},
    "b200": {"gpu": "B200", "arch": "sm_100", "cc": "10.0"},
}
CHILD_TIMEOUT = 2
OUTER_TIMEOUT = 10
EXPECTED_METHODS = {
    ("g2s", "cooperative32"),
    ("g2s", "cooperative128"),
    ("s2g", "cooperative32"),
    ("s2g", "cooperative128"),
    ("roundtrip", "cooperative32"),
    ("roundtrip", "cooperative128"),
    ("overlap", "compute_only"),
    ("overlap", "tma_serial"),
    ("overlap", "tma_overlap"),
}


def build(arch: str) -> str:
    root = f"{REMOTE}/{arch}"
    common = (
        f"nvcc -O3 -lineinfo -std=c++20 -arch={arch} "
        f"{REMOTE}/{SOURCE.name}"
    )
    return (
        "bash -lc 'set -euo pipefail; "
        f"mkdir -p {root}; "
        f"{common} -o {root}/probe -lcuda; "
        f"{common} -ptx -o {root}/probe.ptx -lcuda; "
        f"{common} -cubin -o {root}/probe.cubin -lcuda; "
        f"nvdisasm {root}/probe.cubin > {root}/probe.sass; "
        f"grep -q cp.async.bulk.tensor {root}/probe.ptx; "
        f"grep -q cp.async.bulk.wait_group {root}/probe.ptx; "
        f"! grep -q prefetch.tensormap {root}/probe.ptx; "
        f"test \"$(grep -c fence.proxy.tensormap {root}/probe.ptx)\" -eq 2; "
        f"test -s {root}/probe.sass'"
    )


app = modal.App("ventus-tma-control-items-v3-cold-then-hot")
base = modal.Image.from_registry(
    "nvidia/cuda:13.3.0-devel-ubuntu24.04", add_python="3.12"
).add_local_file(SOURCE, f"{REMOTE}/{SOURCE.name}", copy=True)


def image(arch: str) -> modal.Image:
    return base.run_commands(build(arch))


def execute(label: str) -> dict[str, Any]:
    command = [
        f"{REMOTE}/{TARGETS[label]['arch']}/probe",
        "--warmups", "0",
        "--repeats", "2",
        "--seeds", "101",
        "--suite", "smoke",
        "--csv",
    ]
    started = time.monotonic()
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=CHILD_TIMEOUT,
            check=False,
        )
        payload: dict[str, Any] = {
            "classification": "completed",
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired as error:
        payload = {
            "classification": "child_timeout",
            "returncode": 124,
            "stdout": error.stdout or "",
            "stderr": error.stderr or "",
        }
    payload.update(
        {
            "gpu": label,
            "command": command,
            "wall_seconds": time.monotonic() - started,
            "child_timeout_seconds": CHILD_TIMEOUT,
            "function_timeout_seconds": OUTER_TIMEOUT,
            "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    payload["nvidia_smi_query"] = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,compute_cap,driver_version,clocks.current.sm",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    ).stdout
    return payload


@app.function(
    image=image("sm_90"), gpu="H100!", timeout=OUTER_TIMEOUT,
    retries=0, max_containers=1,
)
def run_h100() -> dict[str, Any]:
    return execute("h100")


@app.function(
    image=image("sm_100"), gpu="B200", timeout=OUTER_TIMEOUT,
    retries=0, max_containers=1,
)
def run_b200() -> dict[str, Any]:
    return execute("b200")


@app.function(image=base, timeout=OUTER_TIMEOUT, retries=0, max_containers=1)
def import_probe() -> str:
    return hashlib.sha256(Path(f"{REMOTE}/{SOURCE.name}").read_bytes()).hexdigest()


@app.local_entrypoint()
def main(gpu: str = "h100", import_only: bool = False, output: str = "") -> None:
    if gpu not in TARGETS:
        raise SystemExit("--gpu must be h100 or b200")
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if import_only:
        if import_probe.remote() != digest:
            raise SystemExit("mounted source hash mismatch")
        print(f"CPU-only import probe passed; source SHA-256={digest}")
        return

    payload = (run_h100 if gpu == "h100" else run_b200).remote()
    if payload.get("classification") != "completed" or payload.get("returncode") != 0:
        raise SystemExit(
            f"{payload.get('classification')} returncode={payload.get('returncode')}"
        )
    rows = list(csv.DictReader(io.StringIO(str(payload["stdout"]))))
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    descriptor_addresses: set[int] = set()
    descriptor_fingerprints: set[int] = set()
    for row in rows:
        key = (row["direction"], row["method"])
        groups.setdefault(key, []).append(row)
        if row["cc"] != TARGETS[gpu]["cc"] or int(row["errors"]) != 0:
            raise SystemExit(f"CSV validation failed for {key}")
        if row["method"] in {"tma_serial", "tma_overlap"}:
            address = int(row["descriptor_va"], 16)
            fingerprint = int(row["descriptor_fingerprint64"], 16)
            if address % 4096 or row["tma_prefetch_before_issue"] != "0":
                raise SystemExit("overlap TensorMap precondition failed")
    if set(groups) != EXPECTED_METHODS:
        raise SystemExit(
            f"unexpected case set: missing={sorted(EXPECTED_METHODS-set(groups))} "
            f"extra={sorted(set(groups)-EXPECTED_METHODS)}"
        )
    for key, pair in groups.items():
        pair.sort(key=lambda row: int(row["repeat"]))
        if [row["repeat"] for row in pair] != ["0", "1"]:
            raise SystemExit(f"{key}: incomplete repeat pair")
        if key[1] in {"tma_serial", "tma_overlap"}:
            if [row["pair_role"] for row in pair] != ["cold", "hot"] or [row["prior_tma_use"] for row in pair] != ["0", "1"]:
                raise SystemExit(f"{key}: not a cold/hot TensorMap pair")
            addresses = [int(row["descriptor_va"], 16) for row in pair]
            fingerprints = [int(row["descriptor_fingerprint64"], 16) for row in pair]
            if len(set(addresses)) != 1 or len(set(fingerprints)) != 1:
                raise SystemExit(f"{key}: descriptor changed inside pair")
            if addresses[0] in descriptor_addresses or fingerprints[0] in descriptor_fingerprints:
                raise SystemExit(f"{key}: descriptor reused across cases")
            descriptor_addresses.add(addresses[0]); descriptor_fingerprints.add(fingerprints[0])
    if len(rows) != 18:
        raise SystemExit("unexpected row count")

    out = Path(output).resolve() if output else HERE / "data" / gpu
    out.mkdir(parents=True, exist_ok=True)
    stdout, stderr = payload.pop("stdout"), payload.pop("stderr")
    (out / "attempt_01.csv").write_text(str(stdout), encoding="utf-8")
    (out / "attempt_01.stderr.txt").write_text(str(stderr), encoding="utf-8")
    (out / "attempt_01.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "gpu": gpu,
                "warmups": 0,
                "repeats": 2,
                "paid_attempts": 1,
                "measurement_release": "ColdThenHotV1",
                "pairing": "one same-kernel cold command then immediate hot reuse for overlap Tensor paths",
                "source_sha256": digest,
                "rows": len(rows),
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"saved: {out}")
