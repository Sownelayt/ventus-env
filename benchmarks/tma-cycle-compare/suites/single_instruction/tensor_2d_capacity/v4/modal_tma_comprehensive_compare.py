"""Cost-bounded Modal runner for the directed comprehensive TMA matrix.

Background:
  This is intentionally separate from the frozen V3.x/reference runs.  Both
  H100 and B200 execute the same manifest (rank 1--5, capacities, legal global
  address phases, dtypes, Bulk, Tensor, swizzle/interleave, sub-byte and all
  common Reduce operations).

Flow:
  nvcc, PTX/SASS checks and manifest listing happen while the image is built or
  in the CPU-only import probe.  Paid GPU work is split into small sequential
  shards.  Each shard gets a 3 s child timeout (2 s for the isolated risk
  shard), Modal's minimum 10 s outer timeout, retries=0 and max_containers=1.
  Results are saved after every shard, so a later XID cannot erase safe data.

Usage:
  /home/liyb/tma-refer/.venv/bin/modal run \
    benchmarks/tma-cycle-compare/modal_tma_comprehensive_compare.py \
    --import-only
  /home/liyb/tma-refer/.venv/bin/modal run \
    benchmarks/tma-cycle-compare/modal_tma_comprehensive_compare.py \
    --gpu h100 --output benchmarks/tma-cycle-compare/results/comprehensive_cuda_h100

Maintenance:
  One command is one paid attempt; this runner deliberately has no --attempts
  loop.  Unsupported encodings are explicit CSV statuses, never zero cycles.
  Do not raise the timeouts to make a failing case pass: first isolate or split
  the shard and prove its normal body time.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal


HERE = Path(__file__).resolve().parent
ROOT = next(
    (
        parent
        for parent in HERE.parents
        if (parent / "testcases/_get_case/common/tma_model.cc").is_file()
    ),
    None,
)
SOURCE = HERE / "cuda_tma_comprehensive.cu"
MATRIX = HERE / "tma_comprehensive_matrix.h"
AUDITOR = HERE / "audit_fresh_tensormap.py"
if ROOT is None:  # Modal re-imports individually mounted sources under /root.
    REFERENCE = HERE / "tma_ventus_parity_bench_v2.cu"
    MODEL_FILES = tuple(
        HERE / name
        for name in ("tma_model.cc", "tma_model.h", "ventus_tma_v2_spec.h")
    )
else:
    REFERENCE = Path("/home/liyb/tma-refer/src/tma_ventus_parity_bench_v2.cu")
    MODEL_FILES = (
        ROOT / "testcases/_get_case/common/tma_model.cc",
        ROOT / "testcases/_get_case/common/tma_model.h",
        ROOT / "testcases/_get_case/common/ventus_tma_v2_spec.h",
    )
REMOTE = "/opt/tma-comprehensive"
IMAGE = "nvidia/cuda:13.3.0-devel-ubuntu24.04"
OUTER_TIMEOUT_SECONDS = 10  # Modal 1.4.2 minimum.
SAFE_CHILD_TIMEOUT_SECONDS = 3
RISK_CHILD_TIMEOUT_SECONDS = 2
SAFE_SHARDS = (
    "bulk",
    "tensor_capacity",
    "tensor_geometry",
    "dtype",
    "layout",
    "interleave16",
)
RISK_SHARDS = ("subbyte", "interleave32")
PROBE_SHARDS = ("tensor_2d_length",)
TARGETS = {
    "h100": {"gpu": "H100!", "arch": "sm_90", "cc": "9.0"},
    "b200": {"gpu": "B200", "arch": "sm_100", "cc": "10.0"},
}


def remote(name: str) -> str:
    return f"{REMOTE}/{name}"


def build_command(arch: str) -> str:
    directory = remote(f"artifacts/{arch}")
    executable = f"{directory}/cuda_tma_comprehensive"
    source = remote(SOURCE.name)
    common = f"nvcc -O3 -lineinfo -std=c++20 -arch={arch} -I{REMOTE} {source}"
    return (
        "bash -lc 'set -euo pipefail; "
        f"mkdir -p {directory}; "
        f"{common} -o {executable} -lcuda; "
        f"{common} -ptx -o {directory}/bench.ptx -lcuda; "
        f"{common} -cubin -o {directory}/bench.cubin -lcuda; "
        f"nvdisasm {directory}/bench.cubin > {directory}/bench.sass; "
        f"python3 {remote(AUDITOR.name)} {source} {directory}/bench.ptx "
        f"{directory}/bench.sass --output {directory}/instruction_audit.json; "
        f"grep -q cp.async.bulk.shared::cta.global {directory}/bench.ptx; "
        f"grep -q cp.async.bulk.tensor.1d {directory}/bench.ptx; "
        f"grep -q cp.async.bulk.tensor.5d {directory}/bench.ptx; "
        f"grep -q cp.reduce.async.bulk.tensor {directory}/bench.ptx; "
        f"for op in add min max and or xor; do "
        f"grep -Eq \"cp.reduce.async.bulk.tensor.*\\.${{op}}\\.\" "
        f"{directory}/bench.ptx; done; "
        f"test -s {executable}; test -s {directory}/bench.ptx; "
        f"test -s {directory}/bench.cubin; test -s {directory}/bench.sass; "
        f"test -s {directory}/instruction_audit.json'"
    )


app = modal.App("ventus-tma-comprehensive-compare")
base_image = modal.Image.from_registry(IMAGE, add_python="3.12")
for path in (SOURCE, MATRIX, AUDITOR, REFERENCE, *MODEL_FILES):
    base_image = base_image.add_local_file(path, remote(path.name), copy=True)


def image_for(arch: str) -> modal.Image:
    return base_image.run_commands(build_command(arch))


def text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


@app.function(
    image=base_image,
    timeout=OUTER_TIMEOUT_SECONDS,
    retries=0,
    max_containers=1,
)
def run_import_probe() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in (SOURCE.name, MATRIX.name, AUDITOR.name, REFERENCE.name, *(p.name for p in MODEL_FILES)):
        path = Path(remote(name))
        result[name] = {
            "exists": path.is_file(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
            if path.is_file()
            else "",
        }
    return result


def run_shard(
    label: str,
    shard: str,
    case_id: str,
    warmups: int,
    repeats: int,
) -> dict[str, Any]:
    target = TARGETS[label]
    executable = remote(f"artifacts/{target['arch']}/cuda_tma_comprehensive")
    timeout = (
        RISK_CHILD_TIMEOUT_SECONDS
        if shard in RISK_SHARDS
        else SAFE_CHILD_TIMEOUT_SECONDS
    )
    command = [
        executable,
        "--shard",
        shard,
        "--warmups",
        str(warmups),
        "--repeats",
        str(repeats),
        "--csv",
    ]
    if case_id:
        command.extend(["--case", case_id])
    list_command = [executable, "--list", "--shard", shard]
    if case_id:
        list_command.extend(["--case", case_id])
    listing = subprocess.run(
        list_command,
        text=True,
        capture_output=True,
        check=False,
        timeout=1,
    )
    if listing.returncode != 0:
        return {
            "classification": "manifest_failed",
            "returncode": listing.returncode,
            "stdout": "",
            "stderr": listing.stderr,
            "gpu": label,
            "shard": shard,
            "expected_case_ids": [],
        }
    expected_case_ids = [
        row["case_id"]
        for row in csv.DictReader(io.StringIO(listing.stdout))
    ]
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
            env={**os.environ, "CUDA_DEVICE_MAX_CONNECTIONS": "1"},
        )
        result: dict[str, Any] = {
            "classification": "completed",
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except subprocess.TimeoutExpired as error:
        result = {
            "classification": "child_timeout",
            "returncode": 124,
            "stdout": text(error.stdout),
            "stderr": text(error.stderr),
        }
    result.update(
        gpu=label,
        shard=shard,
        command=command,
        child_timeout_seconds=timeout,
        function_timeout_seconds=OUTER_TIMEOUT_SECONDS,
        wall_seconds=time.monotonic() - started,
        collected_at_utc=datetime.now(timezone.utc).isoformat(),
        expected_case_ids=expected_case_ids,
        case_id=case_id,
    )
    return result


def run_batch(
    label: str, shards: list[str], case_id: str, warmups: int, repeats: int
) -> list[dict[str, Any]]:
    """Execute all requested safe shards inside one paid GPU function."""
    results: list[dict[str, Any]] = []
    device_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,compute_cap,driver_version,clocks.current.sm,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    for shard in shards:
        result = run_shard(label, shard, case_id, warmups, repeats)
        result["nvidia_smi_query"] = device_query.stdout + device_query.stderr
        results.append(result)
        if result.get("classification") != "completed" or result.get("returncode") != 0:
            break
    return results


@app.function(
    image=image_for("sm_90"),
    gpu="H100!",
    timeout=OUTER_TIMEOUT_SECONDS,
    retries=0,
    max_containers=1,
)
def run_h100(
    shards: list[str], case_id: str, warmups: int, repeats: int
) -> list[dict[str, Any]]:
    return run_batch("h100", shards, case_id, warmups, repeats)


@app.function(
    image=image_for("sm_100"),
    gpu="B200",
    timeout=OUTER_TIMEOUT_SECONDS,
    retries=0,
    max_containers=1,
)
def run_b200(
    shards: list[str], case_id: str, warmups: int, repeats: int
) -> list[dict[str, Any]]:
    return run_batch("b200", shards, case_id, warmups, repeats)


def parse_rows(data: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(data)))


def validate(result: dict[str, Any], label: str, shard: str, repeats: int) -> list[str]:
    errors: list[str] = []
    if result.get("classification") != "completed":
        errors.append(str(result.get("classification")))
    if result.get("returncode") != 0:
        errors.append(f"returncode={result.get('returncode')}")
    try:
        rows = parse_rows(str(result.get("stdout", "")))
    except csv.Error as error:
        return errors + [f"CSV parse failed: {error}"]
    if not rows:
        return errors + ["no CSV rows"]
    required = {
        "gpu",
        "cc",
        "shard",
        "family",
        "case_id",
        "path",
        "direction",
        "dtype",
        "rank",
        "logical_bytes",
        "global_offset",
        "repeat",
        "cycles",
        "errors",
        "status",
        "descriptor_state",
        "source_descriptor_va",
        "destination_descriptor_va",
        "source_descriptor_fingerprint64",
        "destination_descriptor_fingerprint64",
        "prior_tma_use",
        "descriptor_l2_warm_method",
        "tma_prefetch_before_issue",
        "pair_role",
    }
    if not required.issubset(rows[0]):
        errors.append("CSV header is missing comprehensive fields")
    if any(row.get("cc") != TARGETS[label]["cc"] for row in rows):
        errors.append(f"compute capability is not {TARGETS[label]['cc']}")
    if any(row.get("shard") != shard for row in rows):
        errors.append("row escaped requested shard")
    by_case: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_case.setdefault(row.get("case_id", ""), []).append(row)
    expected_cases = set(result.get("expected_case_ids", []))
    actual_cases = set(by_case)
    if actual_cases != expected_cases:
        missing = sorted(expected_cases - actual_cases)
        extra = sorted(actual_cases - expected_cases)
        errors.append(
            f"manifest mismatch missing={missing[:8]} extra={extra[:8]}"
        )
    for case_id, case_rows in by_case.items():
        statuses = {row.get("status", "") for row in case_rows}
        if statuses <= {"ok", "pass"}:
            if len(case_rows) != repeats:
                errors.append(f"{case_id}: incomplete repeat set")
            if any(int(row.get("cycles", "0") or 0) <= 0 for row in case_rows):
                errors.append(f"{case_id}: non-positive cycle")
            if any(int(row.get("errors", "0") or 0) != 0 for row in case_rows):
                errors.append(f"{case_id}: correctness error")
            tensor_rows = [row for row in case_rows if row.get("path") == "tensor"]
            tensor_rows.sort(key=lambda row: int(row.get("repeat", "-1")))
            for repeat, row in enumerate(tensor_rows):
                expected_state = (
                    "tmau_cold_first_use_l2_hot" if repeat == 0
                    else "tmau_hot_reuse"
                )
                expected_role = "cold" if repeat == 0 else "hot"
                if row.get("descriptor_state") != expected_state:
                    errors.append(f"{case_id}: descriptor state is not cold-then-hot")
                if row.get("pair_role") != expected_role:
                    errors.append(f"{case_id}: pair_role is not cold/hot")
                if row.get("prior_tma_use") != str(repeat):
                    errors.append(f"{case_id}: prior TensorMap TMA use is not 0/1")
                if row.get("tma_prefetch_before_issue") != "0":
                    errors.append(f"{case_id}: hidden TensorMap prefetch detected")
                if row.get("descriptor_l2_warm_method") != "ld.global.cg":
                    errors.append(f"{case_id}: descriptor was not warmed by ordinary L2 load")
            for field in ("source_descriptor_va", "destination_descriptor_va"):
                values = [row.get(field, "") for row in tensor_rows
                          if row.get(field, "") not in {"", "0", "0x0000000000000000"}]
                if values and len(set(values)) != 1:
                    errors.append(f"{case_id}: {field} differs inside cold/hot pair")
                if any(int(value, 16) % 4096 for value in values):
                    errors.append(f"{case_id}: {field} is not 4 KiB aligned")
            for field in ("source_descriptor_fingerprint64", "destination_descriptor_fingerprint64"):
                values = [row.get(field, "") for row in tensor_rows
                          if row.get(field, "") not in {"", "0", "0x0000000000000000"}]
                if values and len(set(values)) != 1:
                    errors.append(f"{case_id}: {field} differs inside cold/hot pair")
        elif not statuses <= {"encode_rejected", "unsupported", "skipped_risk"}:
            errors.append(f"{case_id}: unexpected status {sorted(statuses)}")
    tensor_rows = [
        row for row in rows
        if row.get("path") == "tensor"
        and row.get("status") in {"ok", "pass"}
    ]
    for field in ("source_descriptor_va", "destination_descriptor_va"):
        values = [
            row[field] for row in tensor_rows
            if row.get("repeat") == "0"
            and row.get(field, "") not in {"", "0", "0x0000000000000000"}
        ]
        if len(values) != len(set(values)):
            errors.append(f"{field} was reused by two measurement cases")
    for field in (
        "source_descriptor_fingerprint64",
        "destination_descriptor_fingerprint64",
    ):
        values = [
            row[field] for row in tensor_rows
            if row.get("repeat") == "0"
            and row.get(field, "") not in {"", "0", "0x0000000000000000"}
        ]
        if len(values) != len(set(values)):
            errors.append(f"{field} was reused by two measurement cases")
    return errors


def save_result(output: Path, result: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    shard = str(result["shard"])
    copy = dict(result)
    (output / f"{shard}.csv").write_text(str(copy.pop("stdout", "")), encoding="utf-8")
    (output / f"{shard}.stderr.txt").write_text(
        str(copy.pop("stderr", "")), encoding="utf-8"
    )
    (output / f"{shard}.json").write_text(
        json.dumps(copy, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


@app.local_entrypoint()
def main(
    gpu: str = "h100",
    shard: str = "all",
    case: str = "",
    import_only: bool = False,
    warmups: int = 1,
    repeats: int = 1,
    include_risk: bool = False,
    output: str = "",
) -> None:
    for path in (SOURCE, MATRIX, AUDITOR, REFERENCE, *MODEL_FILES):
        if not path.is_file():
            raise SystemExit(f"missing required source: {path}")
    if import_only:
        probe = run_import_probe.remote()
        for path in (SOURCE, MATRIX, AUDITOR, REFERENCE, *MODEL_FILES):
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            mounted = probe.get(path.name, {})
            if not mounted.get("exists") or mounted.get("sha256") != expected:
                raise SystemExit(f"CPU import mismatch: {path.name}")
        print("CPU-only import probe passed; no GPU allocated")
        return
    if gpu not in TARGETS:
        raise SystemExit("--gpu must be h100 or b200")
    if warmups not in (0, 1):
        raise SystemExit("--warmups must be 0 or 1")
    if not 1 <= repeats <= 3:
        raise SystemExit("--repeats must be in [1, 3]")
    known_shards = {*SAFE_SHARDS, *RISK_SHARDS, *PROBE_SHARDS}
    if shard != "all" and shard not in known_shards:
        raise SystemExit("unknown --shard")
    if case and shard == "all":
        raise SystemExit("--case requires one explicit --shard")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = (
        Path(output).resolve()
        if output
        else HERE / "results" / f"comprehensive_cuda_{gpu}_{stamp}"
    )
    function = run_h100 if gpu == "h100" else run_b200
    shards = (
        [*SAFE_SHARDS, *list(RISK_SHARDS if include_risk else ())]
        if shard == "all"
        else [shard]
    )
    print(
        f"{gpu}: one paid attempt, shards={','.join(shards)}, warmups={warmups}, "
        f"repeats={repeats}, child={SAFE_CHILD_TIMEOUT_SECONDS}s/"
        f"risk={RISK_CHILD_TIMEOUT_SECONDS}s, outer={OUTER_TIMEOUT_SECONDS}s, "
        "retries=0, max_containers=1"
    )

    completed: list[str] = []
    risk_outcomes: dict[str, str] = {
        shard: "not_requested" for shard in RISK_SHARDS
    }
    results = function.remote(shards, case, warmups, repeats)
    for result in results:
        shard = str(result["shard"])
        save_result(output_path, result)
        problems = validate(result, gpu, shard, repeats)
        if problems:
            if shard in RISK_SHARDS:
                # Risk is last.  Preserve the qualification/XID evidence but
                # never retry or lengthen the paid invocation automatically.
                risk_outcomes[shard] = "; ".join(problems)
                print(f"risk shard {shard} stopped as bounded: {risk_outcomes[shard]}")
                continue
            raise SystemExit(
                f"safe shard {shard} failed; stopping immediately: "
                + "; ".join(problems)
            )
        completed.append(shard)
        if shard in RISK_SHARDS:
            risk_outcomes[shard] = "completed"
        print(f"{shard}: passed, GPU body={float(result['wall_seconds']):.3f}s")

    manifest = {
        "gpu": gpu,
        "paid_attempts": 1,
        "warmups": warmups,
        "repeats": repeats,
        "safe_child_timeout_seconds": SAFE_CHILD_TIMEOUT_SECONDS,
        "risk_child_timeout_seconds": RISK_CHILD_TIMEOUT_SECONDS,
        "function_timeout_seconds": OUTER_TIMEOUT_SECONDS,
        "modal_retries": 0,
        "max_containers": 1,
        "sequential": True,
        "requested_shards": shards,
        "requested_case": case,
        "completed_shards": completed,
        "risk_outcomes": risk_outcomes,
        "sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (SOURCE, MATRIX, AUDITOR, REFERENCE, *MODEL_FILES)
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"saved: {output_path}")
