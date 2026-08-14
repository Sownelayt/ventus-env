"""Cost-bounded 128B payload-residency controls for H100 and B200."""

from __future__ import annotations
import csv, hashlib, io, json, subprocess, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import modal

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "cuda_data_residency_128b.cu"
REMOTE = "/opt/tma-data-residency"
TARGETS = {"h100": {"gpu": "H100!", "arch": "sm_90", "cc": "9.0"}, "b200": {"gpu": "B200", "arch": "sm_100", "cc": "10.0"}}
CHILD_TIMEOUT = 2
OUTER_TIMEOUT = 10

def build(arch: str) -> str:
    root = f"{REMOTE}/{arch}"; common = f"nvcc -O3 -lineinfo -std=c++20 -arch={arch} {REMOTE}/{SOURCE.name}"
    return "bash -lc 'set -euo pipefail; " + f"mkdir -p {root}; {common} -o {root}/probe -lcuda; {common} -ptx -o {root}/probe.ptx -lcuda; {common} -cubin -o {root}/probe.cubin -lcuda; nvdisasm {root}/probe.cubin > {root}/probe.sass; ! grep -q prefetch.tensormap {root}/probe.ptx; ! grep -q UTMACCTL.PF {root}/probe.sass; grep -q cp.async.bulk.wait_group {root}/probe.ptx; test -s {root}/probe.sass'"

app = modal.App("ventus-tma-data-residency-v5-primed-controls")
base = modal.Image.from_registry("nvidia/cuda:13.3.0-devel-ubuntu24.04", add_python="3.12").add_local_file(SOURCE, f"{REMOTE}/{SOURCE.name}", copy=True)
def image(arch: str) -> modal.Image: return base.run_commands(build(arch))

def execute(label: str) -> dict[str, Any]:
    commands = [
        [f"{REMOTE}/{TARGETS[label]['arch']}/probe", path, direction]
        for path in ("bulk", "tensor") for direction in ("g2s", "s2g")
    ]
    started = time.monotonic(); csv_parts: list[str] = []; stderr_parts: list[str] = []
    payload: dict[str, Any] = {"classification": "completed", "returncode": 0}
    for index, command in enumerate(commands):
        shard_started = time.monotonic()
        try:
            result = subprocess.run(command, text=True, capture_output=True,
                                    timeout=CHILD_TIMEOUT, check=False)
            shard = {"command": command, "classification": "completed",
                     "returncode": result.returncode,
                     "wall_seconds": time.monotonic() - shard_started}
            if result.returncode != 0:
                payload.update({"classification": "child_failed",
                                "returncode": result.returncode})
        except subprocess.TimeoutExpired as error:
            result = None
            shard = {"command": command, "classification": "child_timeout",
                     "returncode": 124,
                     "wall_seconds": time.monotonic() - shard_started}
            payload.update({"classification": "child_timeout", "returncode": 124})
            csv_parts.append(str(error.stdout or "")); stderr_parts.append(str(error.stderr or ""))
        else:
            lines = result.stdout.splitlines()
            csv_parts.extend(lines if index == 0 else lines[1:])
            stderr_parts.append(result.stderr)
        payload.setdefault("shards", []).append(shard)
        if payload["returncode"] != 0:
            break
    payload.update({"gpu": label, "command": commands,
                    "stdout": "\n".join(csv_parts) + "\n",
                    "stderr": "".join(stderr_parts),
                    "wall_seconds": time.monotonic() - started,
                    "child_timeout_seconds": CHILD_TIMEOUT,
                    "function_timeout_seconds": OUTER_TIMEOUT,
                    "collected_at_utc": datetime.now(timezone.utc).isoformat()})
    payload["nvidia_smi_query"] = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,clocks.current.sm", "--format=csv,noheader,nounits"], text=True, capture_output=True, check=False).stdout
    return payload

@app.function(image=image("sm_90"), gpu="H100!", timeout=OUTER_TIMEOUT, retries=0, max_containers=1)
def run_h100() -> dict[str, Any]: return execute("h100")
@app.function(image=image("sm_100"), gpu="B200", timeout=OUTER_TIMEOUT, retries=0, max_containers=1)
def run_b200() -> dict[str, Any]: return execute("b200")
@app.function(image=base, timeout=OUTER_TIMEOUT, retries=0, max_containers=1)
def import_probe() -> str: return hashlib.sha256(Path(f"{REMOTE}/{SOURCE.name}").read_bytes()).hexdigest()

@app.local_entrypoint()
def main(gpu: str = "h100", import_only: bool = False, output: str = "") -> None:
    if gpu not in TARGETS: raise SystemExit("--gpu must be h100 or b200")
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if import_only:
        if import_probe.remote() != digest: raise SystemExit("mounted source hash mismatch")
        print(f"CPU-only import probe passed; source SHA-256={digest}"); return
    payload = (run_h100 if gpu == "h100" else run_b200).remote()
    if payload.get("classification") != "completed" or payload.get("returncode") != 0: raise SystemExit(f"{payload.get('classification')} returncode={payload.get('returncode')}")
    rows = list(csv.DictReader(io.StringIO(str(payload["stdout"]))))
    if not rows or any(row["cc"] != TARGETS[gpu]["cc"] or int(row["errors"]) != 0 for row in rows): raise SystemExit("CSV validation failed")
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    addresses: set[int] = set()
    fingerprints: set[int] = set()
    for row in rows:
        groups.setdefault((row["path"], row["direction"], row["scenario"]), []).append(row)
        if row["path"] != "tensor":
            continue
        address = int(row["descriptor_va"], 16)
        fingerprint = int(row["descriptor_fingerprint64"], 16)
        if address % 4096:
            raise SystemExit("TensorMap freshness validation failed")
        if row["tma_prefetch_before_issue"] != "0":
            raise SystemExit("unexpected TensorMap prefetch")
        if row["descriptor_l2_warm_method"] != "ld.global.cg+untimed_same_map_tma_primer":
            raise SystemExit("TensorMap was not fixed hot by an untimed TMA")
    if len(groups) != 12: raise SystemExit(f"expected 12 cases, got {len(groups)}")
    for key, pair in groups.items():
        pair.sort(key=lambda row: int(row["repeat"]))
        expected_roles = {
            "cold_hot": ["cold", "hot"],
            "cold_cold": ["cold", "cold"],
            "hot_hot": ["hot", "hot"],
        }[key[2]]
        if [row["repeat"] for row in pair] != ["0", "1"] or [row["pair_role"] for row in pair] != expected_roles:
            raise SystemExit(f"{key}: incomplete residency control pair")
        if key[0] == "tensor":
            if [row["prior_tma_use"] for row in pair] != ["1", "2"]:
                raise SystemExit(f"{key}: primer/measured use count is not 1/2")
            pair_addresses = [int(row["descriptor_va"], 16) for row in pair]
            pair_fingerprints = [int(row["descriptor_fingerprint64"], 16) for row in pair]
            if len(set(pair_addresses)) != 1 or len(set(pair_fingerprints)) != 1:
                raise SystemExit(f"{key}: descriptor changed inside pair")
            if pair_addresses[0] in addresses or pair_fingerprints[0] in fingerprints:
                raise SystemExit(f"{key}: descriptor reused across cases")
            addresses.add(pair_addresses[0]); fingerprints.add(pair_fingerprints[0])
    out = Path(output).resolve() if output else HERE / "data" / gpu; out.mkdir(parents=True, exist_ok=True)
    stdout, stderr = payload.pop("stdout"), payload.pop("stderr")
    (out / "attempt_01.csv").write_text(str(stdout), encoding="utf-8"); (out / "attempt_01.stderr.txt").write_text(str(stderr), encoding="utf-8"); (out / "attempt_01.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({"gpu": gpu, "measurement_release": "ColdThenHotV1", "warmups": 0, "repeats": 2, "paid_attempts": 1, "pairing": "same-kernel cold-hot, cold-cold and hot-hot payload controls", "descriptor_residency": "an untimed TMA using the exact same descriptor completes before payload preparation and both timed commands", "source_sha256": digest, "rows": len(rows), "completed_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"saved: {out}")
