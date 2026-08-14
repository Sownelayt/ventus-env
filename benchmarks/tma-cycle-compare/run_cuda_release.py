#!/usr/bin/env python3
"""Plan or execute every CUDA run in the frozen TMA measurement release.

Background:
  A complete FreshTensorMapV1 device result consists of ten formal run keys
  across eight pinned suites.  Running ad-hoc Modal commands makes it easy to
  omit a variant or accidentally retain an earlier result.

Flow:
  ``--plan`` prints the exact CPU/import checks and formal commands.  With
  ``--formal``, each suite first executes one GPU-free import/build audit, then
  its variants run sequentially.  The first failure stops the release.  Modal
  timeouts, retries and container limits remain owned by each versioned suite.

Usage:
  python3 benchmarks/tma-cycle-compare/run_cuda_release.py --device h100 --plan
  python3 benchmarks/tma-cycle-compare/run_cuda_release.py --device h100 \
    --formal --reason final-FreshTensorMapV1-retest

Maintenance:
  Discover work from the pinned release metadata; never duplicate a hand-made
  suite list here.  This wrapper must not retry a failed paid invocation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import run as run_tool


HERE = Path(__file__).resolve().parent
RUN = HERE / "run.py"
DATA = HERE / "data"


def work_items(device: str) -> list[tuple[str, str]]:
    release = run_tool.current_release()
    suites = run_tool.discover_suites()
    items: list[tuple[str, str]] = []
    for suite_id in release["suites"]:
        suite = suites.get(suite_id)
        if suite is None or device not in suite["commands"]:
            continue
        items.extend((suite_id, str(variant)) for variant in suite["variants"])
    return items


def command(
    device: str, suite: str, variant: str, mode: str, reason: str
) -> list[str]:
    result = [
        sys.executable, str(RUN), "--suite", suite,
        "--device", device, "--variant", variant,
    ]
    if mode == "import":
        result.append("--import-only")
    elif mode == "formal":
        result.extend(
            ["--formal", "--force-rerun", "--reason", reason]
        )
    else:
        result.append("--plan")
    return result


def completed_signature(device: str, suite_id: str, variant: str) -> str:
    """Return the exact frozen-suite signature expected for one run key."""
    suite = run_tool.discover_suites()[suite_id]
    sources = run_tool.source_hashes(suite, device)
    payload = run_tool.signature_payload(suite, device, variant, sources)
    return run_tool.test_signature(payload)


def current_passed_signatures() -> set[str]:
    """Return only passed results created after the release audit cutoff."""
    release = run_tool.current_release()["release"]
    cutoff = (
        HERE / "releases" / f"{release}.RESULT_NOT_BEFORE"
    ).read_text(encoding="utf-8").strip()
    signatures: set[str] = set()
    for path in DATA.rglob("run-manifest.json"):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            manifest.get("status") == "passed"
            and manifest.get("measurement_release") == release
            and str(manifest.get("completed_at_utc", "")) >= cutoff
        ):
            signatures.add(str(manifest.get("test_signature", "")))
    return signatures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("h100", "b200"), required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", action="store_true")
    action.add_argument("--formal", action="store_true")
    parser.add_argument("--reason", default="")
    parser.add_argument(
        "--skip-import", action="store_true",
        help="skip CPU/import audits only when they were completed for these exact source hashes",
    )
    parser.add_argument(
        "--skip-completed", action="store_true",
        help="skip exact-signature passed runs; useful after a fail-fast release continuation",
    )
    args = parser.parse_args()
    if args.formal and not args.reason.strip():
        parser.error("--formal requires --reason")

    items = work_items(args.device)
    if args.skip_completed:
        registered = current_passed_signatures()
        remaining: list[tuple[str, str]] = []
        for suite, variant in items:
            signature = completed_signature(args.device, suite, variant)
            if signature in registered:
                print(f"SKIP passed {suite}/{variant} signature={signature[:12]}")
            else:
                remaining.append((suite, variant))
        items = remaining
    print(json.dumps({"device": args.device, "formal_run_keys": items}, indent=2))
    if args.plan:
        seen: set[str] = set()
        for suite, variant in items:
            if suite not in seen:
                print("IMPORT", json.dumps(command(args.device, suite, variant, "import", "")))
                seen.add(suite)
            print("FORMAL", json.dumps(command(args.device, suite, variant, "formal", "REASON")))
        return 0

    if not args.skip_import:
        seen = set()
        for suite, variant in items:
            if suite in seen:
                continue
            seen.add(suite)
            completed = subprocess.run(
                command(args.device, suite, variant, "import", ""),
                cwd=HERE.parents[1], check=False,
            )
            if completed.returncode != 0:
                raise SystemExit(
                    f"{suite}: import/build audit failed; no paid runs follow"
                )

    for suite, variant in items:
        completed = subprocess.run(
            command(args.device, suite, variant, "formal", args.reason),
            cwd=HERE.parents[1], check=False,
        )
        if completed.returncode != 0:
            raise SystemExit(
                f"{suite}/{variant}: formal run failed; stopped without retry"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
