"""Cost-bounded two-sample TMA multi-context probe for H100 and B200."""

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
SOURCE = HERE / "cuda_multi_context.cu"
REMOTE = "/opt/tma-multi-context"
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
        f"grep -q cp.async.bulk.tensor.2d {root}/probe.ptx; "
        f"grep -q cp.async.bulk.wait_group {root}/probe.ptx; "
        f"grep -q mbarrier.try_wait {root}/probe.ptx; "
        f"grep -q ld.global.cg {root}/probe.ptx; "
        f"! grep -q UTMACCTL.PF {root}/probe.sass; "
        f"test -s {root}/probe.sass'"
    )


app = modal.App("ventus-tma-multi-context-v4-cold-then-hot")
base = modal.Image.from_registry(IMAGE, add_python="3.12").add_local_file(
    SOURCE, f"{REMOTE}/{SOURCE.name}", copy=True
)


def image(arch: str) -> modal.Image:
    return base.run_commands(build(arch))


def execute(label: str, project: str, direction: str) -> dict[str, Any]:
    target = TARGETS[label]
    command = [f"{REMOTE}/{target['arch']}/probe", "--project", project]
    if project == "same":
        command.extend(["--direction", direction])
    started = time.monotonic()
    try:
        result = subprocess.run(
            command, text=True, capture_output=True, timeout=CHILD_TIMEOUT,
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
            "project": project,
            "direction": direction,
            "command": command,
            "wall_seconds": time.monotonic() - started,
            "child_timeout_seconds": CHILD_TIMEOUT,
            "function_timeout_seconds": OUTER_TIMEOUT,
            "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,compute_cap,driver_version,clocks.current.sm,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    payload["nvidia_smi_query"] = query.stdout + query.stderr
    return payload


@app.function(image=image("sm_90"), gpu="H100!", timeout=OUTER_TIMEOUT,
              retries=0, max_containers=1)
def run_h100(project: str, direction: str) -> dict[str, Any]:
    return execute("h100", project, direction)


@app.function(image=image("sm_100"), gpu="B200", timeout=OUTER_TIMEOUT,
              retries=0, max_containers=1)
def run_b200(project: str, direction: str) -> dict[str, Any]:
    return execute("b200", project, direction)


@app.function(image=base, timeout=OUTER_TIMEOUT, retries=0, max_containers=1)
def import_probe() -> str:
    return hashlib.sha256(Path(f"{REMOTE}/{SOURCE.name}").read_bytes()).hexdigest()


def validate(payload: dict[str, Any], label: str) -> list[dict[str, str]]:
    if payload.get("classification") != "completed" or payload.get("returncode") != 0:
        raise SystemExit(
            f"{label}: {payload.get('classification')} "
            f"returncode={payload.get('returncode')}"
        )
    rows = list(csv.DictReader(io.StringIO(str(payload.get("stdout", "")))))
    if not rows:
        raise SystemExit("no CSV rows")
    if any(row["cc"] != TARGETS[label]["cc"] for row in rows):
        raise SystemExit("compute capability mismatch")
    if any(int(row["errors"]) != 0 for row in rows):
        raise SystemExit("correctness failure")
    grouped: dict[tuple[str, ...], set[int]] = {}
    keys = (
        "project", "level", "path", "direction", "order", "mode",
        "map_mode", "bytes", "contexts", "commands",
    )
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), set()).add(
            int(row["repeat"])
        )
    for key, repeats in grouped.items():
        if repeats != {0, 1}:
            continue
        group_rows = [row for row in rows if tuple(row[name] for name in keys) == key]
        if group_rows[0]["path"] != "tensor":
            continue
        ordered = sorted(group_rows, key=lambda row: int(row["repeat"]))
        if [row.get("prior_tma_use") for row in ordered] != ["0", "1"]:
            raise SystemExit("multi-context TensorMap pair is not first-use then reuse")
        if [row.get("pair_role") for row in ordered] != ["cold", "hot"]:
            raise SystemExit("multi-context TensorMap pair is not cold then hot")
        if [row.get("descriptor_state") for row in ordered] != [
            "tmau_cold_bank_first_use", "tmau_hot_bank_reuse"
        ]:
            raise SystemExit("multi-context descriptor state is not cold then hot")
        if any(
            row.get("descriptor_l2_warm_method") != "ld.global.cg"
            for row in group_rows
        ):
            raise SystemExit("TensorMap bank was not warmed by ordinary L2 loads")
        if any(
            row.get("tma_prefetch_before_issue") != "0"
            for row in group_rows
        ):
            raise SystemExit("hidden TensorMap prefetch detected")
        for prefix in ("source", "destination"):
            values = [int(row[f"{prefix}_descriptor_bank_va"], 16) for row in ordered]
            fingerprints = [int(row[f"{prefix}_descriptor_bank_fingerprint64"], 16) for row in ordered]
            if not any(values):
                continue
            if any(value % 4096 for value in values) or len(set(values)) != 1:
                raise SystemExit(f"{prefix} TensorMap bank changed between cold and hot")
            if len(set(fingerprints)) != 1:
                raise SystemExit(f"{prefix} TensorMap contents changed between cold and hot")
        if any(row["repeat_descriptor_bank_unique"] != "0" for row in group_rows):
            raise SystemExit("same-bank cold/hot marker missing")
    if any(repeats != {0, 1} for repeats in grouped.values()):
        raise SystemExit("not every case has repeat 0/1")
    first_use_bank_addresses = [
        int(row[field], 16)
        for row in rows
        if row["path"] == "tensor" and row["repeat"] == "0"
        for field in (
            "source_descriptor_bank_va", "destination_descriptor_bank_va"
        )
        if int(row[field], 16) != 0
    ]
    if len(first_use_bank_addresses) != len(set(first_use_bank_addresses)):
        raise SystemExit(
            "a TensorMap bank address was reused by two measurement cases"
        )
    return rows


@app.local_entrypoint()
def main(
    gpu: str = "h100",
    project: str = "same",
    direction: str = "g2s",
    import_only: bool = False,
    output: str = "",
) -> None:
    if gpu not in TARGETS:
        raise SystemExit("--gpu must be h100 or b200")
    if project not in {"same", "mixed"}:
        raise SystemExit("--project must be same or mixed")
    if direction not in {"g2s", "s2g", "mixed"}:
        raise SystemExit("--direction must be g2s, s2g, or mixed")
    if project == "same" and direction == "mixed":
        raise SystemExit("same-direction project requires g2s or s2g")
    expected = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if import_only:
        if import_probe.remote() != expected:
            raise SystemExit("mounted source hash mismatch")
        print(f"CPU-only import probe passed; source SHA-256={expected}")
        return
    payload = (run_h100 if gpu == "h100" else run_b200).remote(
        project, direction
    )
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
        json.dumps(
            {
                "gpu": gpu,
                "project": project,
                "direction": direction,
                "warmups": 0,
                "repeats": 2,
                "paid_attempts": 1,
                "measurement_release": "ColdThenHotV1",
                "pairing": "one new TensorMap bank per case; repeat 0 first-uses it and repeat 1 immediately reuses the exact same bank in the same kernel",
                "source_sha256": expected,
                "rows": len(rows),
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"saved: {output_path}")
