#!/usr/bin/env python3
"""Run each Ventus TensorMap PMU case in a fresh, short-lived process."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
EXECUTABLE = HERE / "ventus_tensormap_decode"
# Local RTL process startup/teardown is not paid GPU time and rank-5 PMU dumps
# consistently take just over two wall-clock seconds.  Modal microbenchmarks
# remain capped at two seconds; use three here to avoid truncating valid PMU.
CASE_TIMEOUT = 5.0
COMPILE_TIMEOUT = 12.0
COUNT_PATTERN = re.compile(
    r"compiled hit/miss/compile/coalesce/kill/evict:\s*"
    r"(\d+)/\s*(\d+)/\s*(\d+)/\s*(\d+)/\s*(\d+)/\s*(\d+)"
)
CYCLE_PATTERN = re.compile(r"compile/bind cycles:\s*(\d+)/\s*(\d+)")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--only-case", action="append", default=[])
    args = parser.parse_args()
    if not EXECUTABLE.is_file():
        raise SystemExit(f"missing {EXECUTABLE}; run make -C {HERE}")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict[str, object]] = []
    pmu_rows: list[dict[str, object]] = []
    outcomes: list[dict[str, object]] = []
    environment = {
        **os.environ,
        "VENTUS_BACKEND": os.environ.get("VENTUS_BACKEND", "rtlsim"),
        "VENTUS_ENV_PATH": os.environ.get("VENTUS_ENV_PATH", str(HERE.parents[5])),
        "VENTUS_SPIKE_LOG": "0",
    }
    scratch_context = tempfile.TemporaryDirectory(
        prefix="ventus-tma-decode-pocl-", dir="/tmp"
    )
    scratch = scratch_context.name
    environment["POCL_CACHE_DIR"] = scratch
    first = True
    for direction in ("g2s", "s2g"):
        for rank in range(1, 6):
            for scenario in ("cold_then_hot", "explicit_prefetch"):
                case_id = f"{direction}_rank{rank}_{scenario}"
                if args.only_case and case_id not in set(args.only_case):
                    continue
                command = [
                    str(EXECUTABLE), "--direction", direction,
                    "--rank", str(rank), "--scenario", scenario,
                ]
                started = time.monotonic()
                timeout = COMPILE_TIMEOUT if first else CASE_TIMEOUT
                first = False
                try:
                    result = subprocess.run(
                        command,
                        cwd=scratch,
                        env=environment,
                        text=True,
                        capture_output=True,
                        timeout=timeout,
                        check=False,
                    )
                except subprocess.TimeoutExpired as error:
                    outcomes.append(
                        {
                            "case_id": case_id,
                            "classification": "timeout",
                            "wall_seconds": time.monotonic() - started,
                            "returncode": 124,
                            "stdout_sha256": hashlib.sha256(
                                (error.stdout or "").encode()
                                if isinstance(error.stdout, str)
                                else (error.stdout or b"")
                            ).hexdigest(),
                        }
                    )
                    (output / "outcomes.json").write_text(
                        json.dumps(outcomes, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    scratch_context.cleanup()
                    raise SystemExit(
                        f"{case_id}: {timeout:g}-second timeout; stopping"
                    )
                stdout_text = result.stdout
                # Ventus RTL prints the PMU summary on stderr while the benchmark
                # CSV is emitted on stdout.  Keep the two channels separate for
                # CSV parsing, but search both for the final PMU snapshot.
                pmu_text = result.stderr + "\n" + stdout_text
                csv_text = "\n".join(
                    line for line in stdout_text.splitlines()
                    if line.startswith("device,") or line.startswith("Ventus,")
                ) + "\n"
                rows = list(csv.DictReader(io.StringIO(csv_text)))
                counts = COUNT_PATTERN.findall(pmu_text)
                cycle_counts = CYCLE_PATTERN.findall(pmu_text)
                problems: list[str] = []
                if result.returncode != 0:
                    problems.append(f"returncode={result.returncode}")
                if len(rows) != 2 or {row.get("repeat") for row in rows} != {"0", "1"}:
                    problems.append("missing repeat 0/1")
                if any(row.get("errors") != "0" or row.get("status") != "0" for row in rows):
                    problems.append("correctness/status failure")
                descriptor_addresses = [row.get("descriptor_va") for row in rows]
                descriptor_fingerprints = [
                    row.get("descriptor_fingerprint64") for row in rows
                ]
                if len(set(descriptor_addresses)) != 1:
                    problems.append("cold/hot pair changed TensorMap address")
                if len(set(descriptor_fingerprints)) != 1:
                    problems.append("cold/hot pair changed TensorMap contents")
                ordered = sorted(rows, key=lambda row: int(row.get("repeat", "-1")))
                if [row.get("prior_tma_use") for row in ordered] != ["0", "1"]:
                    problems.append("prior TMA use is not 0/1")
                expected_prefetch = ["1", "0"] if scenario == "explicit_prefetch" else ["0", "0"]
                if [row.get("tma_prefetch_before_issue") for row in ordered] != expected_prefetch:
                    problems.append("prefetch marker does not match scenario")
                expected_roles = ["prefetched", "hot"] if expected_prefetch[0] == "1" else ["cold", "hot"]
                if [row.get("pair_role") for row in ordered] != expected_roles:
                    problems.append("pair role does not match scenario")
                if not counts or not cycle_counts:
                    problems.append("missing PMU summary")
                outcome = {
                    "case_id": case_id,
                    "classification": "passed" if not problems else "failed",
                    "wall_seconds": time.monotonic() - started,
                    "returncode": result.returncode,
                    "timeout_seconds": timeout,
                    "stdout_sha256": hashlib.sha256(stdout_text.encode()).hexdigest(),
                    "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
                    "stdout_tail": stdout_text[-1000:],
                    "stderr_tail": result.stderr[-1000:],
                    "problems": problems,
                }
                outcomes.append(outcome)
                if problems:
                    (output / "outcomes.json").write_text(
                        json.dumps(outcomes, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    raise SystemExit(f"{case_id}: {'; '.join(problems)}")
                for row in rows:
                    raw_rows.append({"case_id": case_id, **row})
                hits, misses, compiles, coalesces, kills, evictions = counts[-1]
                compile_cycles, bind_cycles = cycle_counts[-1]
                command_count = 2
                pmu_rows.append(
                    {
                        "case_id": case_id,
                        "direction": direction,
                        "rank": rank,
                        "scenario": scenario,
                        "commands": command_count,
                        "compiled_hits": hits,
                        "compiled_misses": misses,
                        "descriptor_compiles": compiles,
                        "descriptor_coalesces": coalesces,
                        "invalidate_kills": kills,
                        "evictions": evictions,
                        "descriptor_compile_cycles_total": compile_cycles,
                        "descriptor_compile_cycles_per_compile": (
                            int(compile_cycles) / int(compiles) if int(compiles) else 0
                        ),
                        "bind_cycles_total": bind_cycles,
                        "bind_cycles_per_command": int(bind_cycles) / command_count,
                        "scope": "one descriptor; two same-kernel consecutive data commands",
                    }
                )
                write_csv(output / "raw.csv", raw_rows)
                write_csv(output / "pmu.csv", pmu_rows)
                (output / "outcomes.json").write_text(
                    json.dumps(outcomes, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "device": "ventus",
                "warmups": 0,
                "repeats": 2,
                "measurement_release": "ColdThenHotV1",
                "pairing": "one descriptor per case; same-kernel first use then immediate reuse",
                "case_timeout_seconds": CASE_TIMEOUT,
                "compile_timeout_seconds": COMPILE_TIMEOUT,
                "cases": len(pmu_rows),
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
