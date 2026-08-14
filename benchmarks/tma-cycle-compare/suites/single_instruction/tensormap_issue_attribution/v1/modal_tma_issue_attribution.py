"""Cost-bounded fixed-site TMA issue-attribution probe for H100 and B200."""

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
SOURCE = HERE / "cuda_tma_issue_attribution.cu"
AUDITOR = HERE / "audit_instruction_stream.py"
REMOTE = "/opt/tma-issue-attribution"
IMAGE = "nvidia/cuda:13.3.0-devel-ubuntu24.04"
TARGETS = {
    "h100": {"gpu": "H100!", "arch": "sm_90", "cc": "9.0"},
    "b200": {"gpu": "B200", "arch": "sm_100", "cc": "10.0"},
}
MODES = ("none", "predicated_off", "other_tensormap")
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
        f"{root}/probe.ptx {root}/probe.sass --output {root}/instruction_audit.json; "
        f"test -s {root}/probe; test -s {root}/instruction_audit.json'"
    )


app = modal.App("ventus-tma-issue-attribution-v1")
base = (
    modal.Image.from_registry(IMAGE, add_python="3.12")
    .add_local_file(SOURCE, f"{REMOTE}/{SOURCE.name}", copy=True)
    .add_local_file(AUDITOR, f"{REMOTE}/{AUDITOR.name}", copy=True)
)


def image(arch: str) -> modal.Image:
    return base.run_commands(build(arch))


def execute(label: str) -> dict[str, Any]:
    target = TARGETS[label]
    started = time.monotonic()
    outputs: list[str] = []
    errors: list[str] = []
    commands: list[list[str]] = []
    for mode in MODES:
        command = [f"{REMOTE}/{target['arch']}/probe", mode]
        commands.append(command)
        child_started = time.monotonic()
        try:
            result = subprocess.run(
                command, text=True, capture_output=True, timeout=CHILD_TIMEOUT,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            return {
                "classification": "child_timeout", "returncode": 124,
                "stdout": "\n".join(outputs),
                "stderr": "\n".join(errors + [str(error.stderr or "")]),
                "failed_mode": mode, "commands": commands,
                "wall_seconds": time.monotonic() - started,
            }
        if result.returncode != 0:
            return {
                "classification": "child_error", "returncode": result.returncode,
                "stdout": "\n".join(outputs + [result.stdout]),
                "stderr": "\n".join(errors + [result.stderr]),
                "failed_mode": mode, "commands": commands,
                "wall_seconds": time.monotonic() - started,
            }
        lines = result.stdout.strip().splitlines()
        if not lines:
            return {"classification": "no_rows", "returncode": 3,
                    "stdout": "\n".join(outputs), "stderr": result.stderr,
                    "failed_mode": mode, "commands": commands}
        outputs.append("\n".join(lines if not outputs else lines[1:]))
        errors.append(
            f"mode={mode} wall_seconds={time.monotonic() - child_started:.6f}\n"
            + result.stderr
        )
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,clocks.current.sm,memory.total",
         "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=False,
    )
    return {
        "classification": "completed", "returncode": 0,
        "stdout": "\n".join(outputs) + "\n",
        "stderr": "\n".join(errors), "commands": commands,
        "wall_seconds": time.monotonic() - started,
        "child_timeout_seconds": CHILD_TIMEOUT,
        "function_timeout_seconds": OUTER_TIMEOUT,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "nvidia_smi_query": query.stdout + query.stderr,
    }


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
        raise SystemExit(
            f"{label}: {payload.get('classification')} returncode={payload.get('returncode')} "
            f"mode={payload.get('failed_mode')}\n{payload.get('stderr', '')}"
        )
    rows = list(csv.DictReader(io.StringIO(str(payload.get("stdout", "")))))
    if len(rows) != 16:
        raise SystemExit(f"expected 16 rows, got {len(rows)}")
    if any(row["cc"] != TARGETS[label]["cc"] for row in rows):
        raise SystemExit("compute capability mismatch")
    if any(int(row["errors"]) != 0 for row in rows):
        raise SystemExit("correctness failure")
    for mode in ("none", "predicated_off_same_site", "other_tensormap_same_site"):
        for direction in ("g2s", "s2g"):
            selected = [row for row in rows if row["prime_mode"] == mode and row["direction"] == direction]
            measured = sorted(
                (row for row in selected if row["stage"] == "measured"),
                key=lambda row: int(row["repeat"]),
            )
            if [int(row["repeat"]) for row in measured] != [0, 1]:
                raise SystemExit(f"{mode}/{direction}: missing measured 0/1")
            if [int(row["prior_measured_map_tma_use"]) for row in measured] != [0, 1]:
                raise SystemExit(f"{mode}/{direction}: invalid prior-use markers")
            prime = [row for row in selected if row["stage"] == "prime"]
            if len(prime) != (0 if mode == "none" else 1):
                raise SystemExit(f"{mode}/{direction}: invalid prime count")
            if mode == "predicated_off_same_site" and any(
                int(row["polls"]) != 0 for row in prime
            ):
                raise SystemExit("predicate-false prime unexpectedly waited")
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
    (output_path / "attempt_01.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_path / "manifest.json").write_text(
        json.dumps({
            "gpu": gpu, "suite": "tensormap_issue_attribution/v1",
            "warmups": 0, "paid_attempts": 1,
            "processes": len(MODES),
            "pairing": "same fixed 2-D static TMA site; measured map repeat 0 then 1",
            "descriptor_state": "L2-hot and explicit-prefetch-hot before measured pair",
            "source_bundle_sha256": expected, "rows": len(rows),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"saved: {output_path}")
