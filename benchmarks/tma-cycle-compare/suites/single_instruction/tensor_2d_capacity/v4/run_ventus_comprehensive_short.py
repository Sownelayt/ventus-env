#!/usr/bin/env python3
"""Run the shared comprehensive Ventus matrix one bounded case at a time.

Background:
  GVM failures can emit enormous logs or wait forever.  The comprehensive
  executable therefore exposes --list/--case and this wrapper gives each case
  an independent process, persists every successful row immediately, and
  keeps only a bounded failure tail.

Usage:
  source ./env.sh
  python3 benchmarks/tma-cycle-compare/run_ventus_comprehensive_short.py \
    --shard all --warmups 1 --repeats 1 \
    --output benchmarks/tma-cycle-compare/results/comprehensive_ventus

Maintenance:
  This is local GVM work, not a Modal runner.  Interleave32 and sub-byte use a
  shorter timeout and run after the safe shards.  Do not hide a timed-out case
  by omitting it from the manifest; its outcome JSON is part of the evidence.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
EXECUTABLE = HERE / "ventus_tma_comprehensive"
SAFE_SHARDS = (
    "bulk",
    "tensor_capacity",
    "tensor_geometry",
    "dtype",
    "layout",
    "interleave16",
    "reduce",
)
RISK_SHARDS = ("subbyte", "interleave32")
PROBE_SHARDS = ("tensor_2d_length",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shard",
        choices=("all", *SAFE_SHARDS, *RISK_SHARDS, *PROBE_SHARDS),
        default="all",
    )
    parser.add_argument("--warmups", type=int, choices=(0, 1), default=1)
    parser.add_argument("--repeats", type=int, choices=(1, 2, 3), default=1)
    # Release-RTL startup plus two no-warmup samples reaches ~3 s even for
    # valid 4 KiB Tensor cases.  This is a free local simulation guard; Modal
    # retains its separate 3 s paid child timeout.
    parser.add_argument("--case-timeout", type=float, default=10.0)
    # 16/32 KiB release-RTL simulation includes host process startup and can
    # exceed five seconds; this local-only guard does not affect Modal billing.
    parser.add_argument("--large-case-timeout", type=float, default=60.0)
    parser.add_argument("--risk-timeout", type=float, default=2.0)
    parser.add_argument("--compile-timeout", type=float, default=12.0)
    parser.add_argument("--only-case", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="record a failed safe case and continue; intended only for bounded diagnostic sweeps",
    )
    parser.add_argument(
        "--output",
        default=str(HERE / "results" / "comprehensive_ventus"),
    )
    args = parser.parse_args()
    if not 0.5 <= args.risk_timeout <= 3:
        parser.error("--risk-timeout must be in [0.5, 3]")
    if not 1 <= args.case_timeout <= 10:
        parser.error("--case-timeout must be in [1, 10]")
    if not 3 <= args.large_case_timeout <= 90:
        parser.error("--large-case-timeout must be in [3, 90]")
    if not 5 <= args.compile_timeout <= 20:
        parser.error("--compile-timeout must be in [5, 20]")
    return args


def list_cases(shard: str) -> list[str]:
    result = subprocess.run(
        [str(EXECUTABLE), "--list", "--shard", shard],
        text=True,
        capture_output=True,
        timeout=2,
        check=False,
    )
    if result.returncode:
        raise SystemExit(f"case listing failed: {result.stderr[-1000:]}")
    cases = [line for line in result.stdout.splitlines() if line]
    if not cases or len(cases) != len(set(cases)):
        raise SystemExit("empty or duplicate case manifest")
    return cases


def csv_only(stdout: str) -> str:
    lines = [
        line
        for line in stdout.splitlines()
        if line.startswith("gpu,") or line.startswith("Ventus,")
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def validate(data: str, case_id: str, repeats: int) -> tuple[list[dict[str, str]], list[str]]:
    try:
        rows = list(csv.DictReader(io.StringIO(data)))
    except csv.Error as error:
        return [], [f"CSV parse failed: {error}"]
    errors: list[str] = []
    if len(rows) != repeats:
        errors.append(f"rows={len(rows)}, expected={repeats}")
    if any(row.get("case_id") != case_id for row in rows):
        errors.append("case_id mismatch")
    if any(row.get("status") != "ok" for row in rows):
        errors.append("non-ok status")
    if any(row.get("errors") != "0" for row in rows):
        errors.append("correctness error")
    try:
        if any(int(row.get("cycles", "0")) <= 0 for row in rows):
            errors.append("non-positive cycles")
    except ValueError:
        errors.append("invalid cycle value")
    tensor_rows = [row for row in rows if row.get("path") == "tensor"]
    if tensor_rows:
        tensor_rows.sort(key=lambda row: int(row.get("repeat", "-1")))
        if [row.get("repeat") for row in tensor_rows] != ["0", "1"]:
            errors.append("Tensor pair is not repeat 0/1")
        if [row.get("pair_role") for row in tensor_rows] != ["cold", "hot"]:
            errors.append("Tensor pair is not cold/hot")
        if [row.get("prior_tma_use") for row in tensor_rows] != ["0", "1"]:
            errors.append("prior TensorMap use is not 0/1")
        if any(row.get("tma_prefetch_before_issue") != "0" for row in tensor_rows):
            errors.append("hidden TensorMap prefetch")
        if [row.get("descriptor_state") for row in tensor_rows] != [
            "tmau_cold_first_use_fresh_allocation", "tmau_hot_reuse"
        ]:
            errors.append("descriptor state is not cold/hot")
        direction = tensor_rows[0].get("direction")
        fields = []
        if direction != "s2g":
            fields += ["source_descriptor_va", "source_descriptor_fingerprint64"]
        if direction != "g2s":
            fields += ["destination_descriptor_va", "destination_descriptor_fingerprint64"]
        for field in fields:
            values = [row.get(field, "") for row in tensor_rows]
            if len(values) != 2 or not values[0] or values[0] in {"0", "0x0"}:
                errors.append(f"missing {field}")
            elif values[0] != values[1]:
                errors.append(f"cold/hot changed {field}")
    return rows, errors


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not EXECUTABLE.is_file():
        raise SystemExit(f"missing {EXECUTABLE}; run make -C {HERE}")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    partial = output / "partial.csv"
    outcomes_path = output / "outcomes.jsonl"

    shards = [args.shard] if args.shard != "all" else [*SAFE_SHARDS]
    cases: list[tuple[str, str]] = []
    for shard in shards:
        cases.extend((shard, case_id) for case_id in list_cases(shard))
    if args.only_case:
        selected = set(args.only_case)
        cases = [item for item in cases if item[1] in selected]
        missing = selected - {case for _, case in cases}
        if missing:
            raise SystemExit("unknown --only-case: " + ",".join(sorted(missing)))

    rows: list[dict[str, str]] = []
    completed: set[str] = set()
    if args.resume and partial.is_file():
        with partial.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["case_id"]] = counts.get(row["case_id"], 0) + 1
        completed = {case for case, count in counts.items() if count == args.repeats}

    environment = {
        **os.environ,
        "VENTUS_BACKEND": os.environ.get("VENTUS_BACKEND", "rtlsim"),
        "VENTUS_ENV_PATH": os.environ.get("VENTUS_ENV_PATH", str(HERE.parents[5])),
    }
    # POCL and the Ventus backend emit fixed-name object0.* files in the
    # process working directory.  Give every runner invocation a private
    # compile/cache directory so independent shard processes cannot corrupt
    # one another; this is also required for reliable retry after a failed
    # compile.
    scratch_context = tempfile.TemporaryDirectory(
        prefix="ventus-tma-comprehensive-pocl-", dir="/tmp"
    )
    scratch = scratch_context.name
    first = True
    for index, (shard, case_id) in enumerate(cases, 1):
        if case_id in completed:
            continue
        if first:
            timeout = args.compile_timeout
        elif shard in RISK_SHARDS:
            timeout = args.risk_timeout
        elif (
            (size_match := re.search(r"_b(\d+)$", case_id)) is not None
            and int(size_match.group(1)) >= 16384
        ):
            timeout = args.large_case_timeout
        else:
            timeout = args.case_timeout
        first = False
        command = [
            str(EXECUTABLE),
            "--case",
            case_id,
            "--shard",
            shard,
            "--warmups",
            str(args.warmups),
            "--repeats",
            str(args.repeats),
            "--csv",
        ]
        # Keep each generated OpenCL object and POCL cache private to one
        # parameter point.  The long 2D sweep otherwise reproducibly polluted
        # the shared cache at case 81 while that same case passed in isolation.
        case_scratch = Path(scratch) / case_id
        case_scratch.mkdir(parents=True, exist_ok=True)
        case_environment = {
            **environment,
            "POCL_CACHE_DIR": str(case_scratch),
        }
        started = time.monotonic()
        try:
            result = subprocess.run(
                command,
                env=case_environment,
                text=True,
                capture_output=True,
                check=False,
                timeout=timeout,
                cwd=case_scratch,
            )
            classification = "completed"
            stdout, stderr, returncode = result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired as error:
            classification = "child_timeout"
            stdout = error.stdout.decode(errors="replace") if isinstance(error.stdout, bytes) else (error.stdout or "")
            stderr = error.stderr.decode(errors="replace") if isinstance(error.stderr, bytes) else (error.stderr or "")
            returncode = 124
        data = csv_only(stdout)
        case_rows, problems = validate(data, case_id, args.repeats)
        if classification != "completed":
            problems.insert(0, classification)
        if returncode != 0:
            problems.insert(0, f"returncode={returncode}")
        outcome = {
            "case_id": case_id,
            "shard": shard,
            "classification": classification,
            "returncode": returncode,
            "timeout_seconds": timeout,
            "wall_seconds": time.monotonic() - started,
            "problems": problems,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "diagnostics": [
                line for line in (stderr + "\n" + stdout).splitlines()
                if line.startswith("MISMATCH,")
                or "STATUS_MISMATCH" in line
                or "FIRST32" in line
            ][-128:],
            "failure_tail": "\n".join((stderr + "\n" + stdout).splitlines()[-60:]),
        }
        with outcomes_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(outcome, sort_keys=True) + "\n")
        if problems:
            print(f"{index}/{len(cases)} {case_id}: " + "; ".join(problems))
            if shard not in RISK_SHARDS and not args.continue_on_failure:
                raise SystemExit("safe case failed; stopped immediately")
            continue
        rows.extend(case_rows)
        write_rows(partial, rows)
        print(f"{index}/{len(cases)} {case_id}: passed in {outcome['wall_seconds']:.3f}s")

    write_rows(output / "attempt_01.csv", rows)
    manifest = {
        "attempts": 1,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "shards": shards,
        "cases_requested": len(cases),
        "cases_completed": len({row["case_id"] for row in rows}),
        "safe_timeout_seconds": args.case_timeout,
        "large_timeout_seconds": args.large_case_timeout,
        "risk_timeout_seconds": args.risk_timeout,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    scratch_context.cleanup()
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
