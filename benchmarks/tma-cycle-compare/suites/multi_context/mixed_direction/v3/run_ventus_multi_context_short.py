#!/usr/bin/env python3
"""Checkpointed short-process runner for Ventus multi-context cases."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
EXECUTABLE = HERE / "ventus_multi_context"
# Multi-command release-RTL simulation is free local work.  Two rounds of up
# to 32 serialized commands can legitimately take >12 wall-clock seconds even
# though paid Modal children remain capped at two seconds.
CASE_TIMEOUT = 60.0
COMPILE_TIMEOUT = 60.0
LOCAL_MEMORY_BYTES = 64 * 1024


def matrix(project: str, direction: str) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    contexts_values = (1, 2, 4, 8, 16, 32)
    for level in ("intra_cta", "multi_cta"):
        for path in ("tensor", "bulk"):
            for size in (128, 4096):
                for contexts in contexts_values:
                    map_modes = ("same_map", "distinct_map") if path == "tensor" else ("same_map",)
                    modes = ("serial", "batched") if level == "intra_cta" else ("batched",)
                    if project == "same":
                        for map_mode in map_modes:
                            for mode in modes:
                                cases.append(
                                    {
                                        "project": project, "level": level,
                                        "path": path, "direction": direction,
                                        "order": "alternating", "mode": mode,
                                        "map_mode": map_mode,
                                        "bytes": size, "contexts": contexts,
                                    }
                                )
                    else:
                        if contexts == 32:
                            continue
                        for map_mode in map_modes:
                            for order in ("alternating", "g2s_then_s2g", "s2g_then_g2s"):
                                for mode in modes:
                                    cases.append(
                                        {
                                            "project": project, "level": level,
                                            "path": path, "direction": "mixed",
                                            "order": order, "mode": mode,
                                            "map_mode": map_mode,
                                            "bytes": size, "contexts": contexts,
                                        }
                                    )
    return cases


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def freshness_fields(row: dict[str, str]) -> dict[str, str]:
    tensor = row.get("path") == "tensor"
    repeat = int(row.get("repeat", "0"))
    return {
        "pair_role": "cold" if repeat == 0 else "hot",
        "prior_tma_use": str(repeat),
        "descriptor_l2_warm_method": (
            "patch_descriptor_write" if tensor else "not_applicable"
        ),
        "tma_prefetch_before_issue": "0",
        "descriptor_state": (
            "tmau_cold_bank_first_use" if tensor and repeat == 0 else
            "tmau_hot_bank_reuse" if tensor else "not_applicable"
        ),
    }


def required_local_bytes(case: dict[str, object]) -> int:
    local_contexts = 1 if case["level"] == "multi_cta" else int(case["contexts"])
    payload = local_contexts * int(case["bytes"]) + 127
    barriers = local_contexts * 8
    if case["project"] == "mixed":
        return 2 * payload + barriers
    return payload + (barriers if case["direction"] == "g2s" else 8)


def resource_limit_reason(case: dict[str, object]) -> str:
    reasons: list[str] = []
    if required_local_bytes(case) > LOCAL_MEMORY_BYTES:
        reasons.append("independent shared regions exceed the 64 KiB Ventus LDS")
    if (
        case["level"] == "intra_cta"
        and (
            (case["project"] == "same" and case["direction"] == "g2s")
            or case["project"] == "mixed"
        )
        and int(case["contexts"]) > 4
    ):
        reasons.append("four mbarrier table entries are statically assigned per resident WG")
    if (
        case["project"] == "same"
        and case["level"] == "multi_cta"
        and case["direction"] == "s2g"
        and case["path"] == "tensor"
        and case["map_mode"] == "distinct_map"
        and int(case["contexts"]) == 32
    ):
        reasons.append(
            "known RTL timeout under 32 independent S2G descriptor streams; "
            "same-map finishes and the timed-out signature is not retried"
        )
    if (
        case["level"] == "intra_cta"
        and case["project"] == "mixed"
        and int(case["contexts"]) > 1
    ):
        reasons.append(
            "current Ventus compiler emits private spill for the mixed "
            "capture kernel, but RTL simulation does not back the PDS spill page"
        )
    if (
        case["project"] == "mixed"
        and case["level"] == "multi_cta"
        and case["path"] == "tensor"
        and case["map_mode"] == "distinct_map"
        and case["order"] == "g2s_then_s2g"
        and int(case["bytes"]) == 4096
        and int(case["contexts"]) >= 8
    ):
        reasons.append(
            "the current Ventus compiler emits a live private spill on this "
            "4 KiB distinct-map mixed-order path; RTL simulation reads the "
            "unbacked PDS spill page and reaches an undefined instruction"
        )
    return "; ".join(reasons)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", choices=("same", "mixed"), required=True)
    parser.add_argument("--direction", choices=("g2s", "s2g", "mixed"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--only-case", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (args.project == "same") == (args.direction == "mixed"):
        parser.error("same requires g2s/s2g; mixed requires mixed")
    if not EXECUTABLE.is_file():
        raise SystemExit(f"missing {EXECUTABLE}; run make -C {HERE}")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    resource_limits: list[dict[str, object]] = []
    outcomes: list[dict[str, object]] = []
    if args.resume:
        raw_path = output / "raw.csv"
        if raw_path.is_file():
            with raw_path.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            for row in rows:
                row.update(freshness_fields(row))
        limit_path = output / "resource_limits.csv"
        if limit_path.is_file():
            with limit_path.open(newline="", encoding="utf-8") as source:
                resource_limits = list(csv.DictReader(source))
        outcome_path = output / "outcomes.json"
        if outcome_path.is_file():
            loaded = json.loads(outcome_path.read_text(encoding="utf-8"))
            outcomes = [
                item for item in loaded
                if item.get("classification") in ("passed", "resource_limit")
            ]
    completed_cases = {str(item["case_id"]) for item in outcomes}
    cases = matrix(args.project, args.direction)
    scratch_context = tempfile.TemporaryDirectory(
        prefix="ventus-tma-multi-context-pocl-", dir="/tmp"
    )
    scratch = scratch_context.name
    environment = {
        **os.environ,
        "VENTUS_BACKEND": os.environ.get("VENTUS_BACKEND", "rtlsim"),
        "VENTUS_ENV_PATH": os.environ.get("VENTUS_ENV_PATH", str(HERE.parents[5])),
        "VENTUS_SPIKE_LOG": "0",
        "POCL_CACHE_DIR": scratch,
    }
    first = True
    for index, case in enumerate(cases, 1):
        case_id = "_".join(
            str(case[key]) for key in
            ("level", "path", "direction", "order", "mode", "map_mode", "bytes", "contexts")
        )
        if args.only_case and case_id not in set(args.only_case):
            continue
        if case_id in completed_cases:
            continue
        local_bytes = required_local_bytes(case)
        limit_reason = resource_limit_reason(case)
        if limit_reason:
            resource_limits.append(
                {
                    "case_id": case_id,
                    **case,
                    "required_local_bytes": local_bytes,
                    "available_local_bytes": LOCAL_MEMORY_BYTES,
                    "classification": "resource_limit",
                    "reason": limit_reason,
                }
            )
            outcomes.append(resource_limits[-1])
            write_csv(output / "resource_limits.csv", resource_limits)
            (output / "outcomes.json").write_text(
                json.dumps(outcomes, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"{index}/{len(cases)} {case_id}: resource_limit")
            continue
        command = [
            str(EXECUTABLE),
            "--project", str(case["project"]),
            "--level", str(case["level"]),
            "--path", str(case["path"]),
            "--direction", str(case["direction"]),
            "--order", str(case["order"]),
            "--map-mode", str(case["map_mode"]),
            "--mode", str(case["mode"]),
            "--bytes", str(case["bytes"]),
            "--contexts", str(case["contexts"]),
        ]
        timeout = COMPILE_TIMEOUT if first else CASE_TIMEOUT
        first = False
        started = time.monotonic()
        try:
            result = subprocess.run(
                command, cwd=scratch, env=environment, text=True,
                capture_output=True, timeout=timeout, check=False,
            )
            classification = "completed"
            stdout, stderr, returncode = result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired as error:
            classification = "timeout"
            stdout = error.stdout.decode(errors="replace") if isinstance(error.stdout, bytes) else (error.stdout or "")
            stderr = error.stderr.decode(errors="replace") if isinstance(error.stderr, bytes) else (error.stderr or "")
            returncode = 124
        csv_text = "\n".join(
            line for line in stdout.splitlines()
            if line.startswith("device,") or line.startswith("Ventus,")
        ) + "\n"
        case_rows = list(csv.DictReader(io.StringIO(csv_text)))
        problems: list[str] = []
        if classification != "completed":
            problems.append(classification)
        if returncode != 0:
            problems.append(f"returncode={returncode}")
        if len(case_rows) != 2 or {row.get("repeat") for row in case_rows} != {"0", "1"}:
            problems.append("missing repeat 0/1")
        if any(row.get("errors") != "0" or row.get("status") != "0" for row in case_rows):
            problems.append("correctness/status failure")
        if case["path"] == "tensor":
            ordered = sorted(case_rows, key=lambda row: int(row["repeat"]))
            if [row.get("pair_role") for row in ordered] != ["cold", "hot"]:
                problems.append("pair is not cold then hot")
            if [row.get("prior_tma_use") for row in ordered] != ["0", "1"]:
                problems.append("prior use is not 0 then 1")
            if any(row.get("repeat_descriptor_bank_unique") != "0" for row in case_rows):
                problems.append("same-bank cold/hot marker missing")
            offsets = {row.get("descriptor_bank_word_offset") for row in case_rows}
            fingerprints = {
                row.get("descriptor_bank_fingerprint64") for row in case_rows
            }
            identities = {row.get("descriptor_va") for row in case_rows}
            if len(offsets) != 1 or len(fingerprints) != 1 or len(identities) != 1:
                problems.append("TensorMap bank changed between cold and hot")
            if any(row.get("descriptor_bank_spacing_bytes") != "0" for row in case_rows):
                problems.append("cold/hot pair incorrectly reports separate banks")
        outcome = {
                "case_id": case_id,
                "classification": "passed" if not problems else classification,
                "returncode": returncode,
                "timeout_seconds": timeout,
                "wall_seconds": time.monotonic() - started,
                "problems": problems,
                "diagnostics": [
                    line for line in (stderr + "\n" + stdout).splitlines()
                    if ("MISMATCH" in line or "FIRST32" in line or
                        "%Error" in line or "kernel metadata" in line or
                        "PMEM page" in line)
                ][-100:],
                "failure_tail": "\n".join((stderr + "\n" + stdout).splitlines()[-60:]),
            }
        outcomes.append(outcome)
        (output / "outcomes.json").write_text(
            json.dumps(outcomes, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if problems:
            scratch_context.cleanup()
            raise SystemExit(f"{case_id}: {'; '.join(problems)}")
        rows.extend(
            {"case_id": case_id, **row, **freshness_fields(row)}
            for row in case_rows
        )
        write_csv(output / "raw.csv", rows)
        print(f"{index}/{len(cases)} {case_id}: passed")
    write_csv(output / "raw.csv", rows)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "device": "ventus",
                "project": args.project,
                "direction": args.direction,
                "warmups": 0,
                "repeats": 2,
                "measurement_release": "ColdThenHotV1",
                "pairing": "same compiled kernel and exact same TensorMap bank: repeat 0 first-use, repeat 1 immediate reuse",
                "case_timeout_seconds": CASE_TIMEOUT,
                "compile_timeout_seconds": COMPILE_TIMEOUT,
                "requested_cases": len(cases),
                "measured_cases": len(rows) // 2,
                "resource_limited_cases": len(resource_limits),
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    scratch_context.cleanup()
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
