#!/usr/bin/env python3
"""Remove superseded TMA tests/results after a release is complete.

Background:
  The release named by CURRENT_MEASUREMENT_RELEASE is the only authoritative
  cycle dataset.  Historical tests and results must not remain where report
  tools can find them, but deletion is unsafe until every pinned
  device/suite/variant has a current-source, validated replacement.

Flow:
  The default dry run validates all replacement candidates and prints the one
  newest run selected for each required key.  ``--apply`` copies those runs to
  a temporary recovery directory, rebuilds ``data/`` from only those copies,
  removes unpinned suite versions and historical report/archive directories,
  and verifies byte-for-byte tree hashes after restoration.

Usage:
  python3 benchmarks/tma-cycle-compare/prune_to_release.py
  python3 benchmarks/tma-cycle-compare/prune_to_release.py --apply

Maintenance:
  Never weaken the complete-key or current-source checks.  A new measurement
  method must create a new release JSON rather than reusing this release name.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import run as run_tool


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RELEASE_NAME = (HERE / "CURRENT_MEASUREMENT_RELEASE").read_text(
    encoding="utf-8"
).strip()
RELEASE = json.loads(
    (HERE / "releases" / f"{RELEASE_NAME}.json").read_text(encoding="utf-8")
)
# The cross-case descriptor-identity audit was completed at this cutover.
# Earlier FreshTensorMapV1 probes helped develop the method but are not final
# release measurements and must not survive pruning.
RESULT_NOT_BEFORE_UTC = (
    HERE / "releases" / f"{RELEASE_NAME}.RESULT_NOT_BEFORE"
).read_text(encoding="utf-8").strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(sha256_file(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def resource_limit_count(directory: Path) -> int:
    path = directory / "resource_limits.csv"
    if not path.is_file():
        return 0
    with path.open(newline="", encoding="utf-8") as source:
        return sum(1 for _ in csv.DictReader(source))


def pinned_suites() -> dict[str, dict[str, Any]]:
    suites = run_tool.discover_suites()
    pinned = RELEASE.get("suites", {})
    selected = {
        suite_id: suite for suite_id, suite in suites.items()
        if pinned.get(suite_id) == suite.get("suite_version")
    }
    if set(selected) != set(pinned):
        raise SystemExit("pinned suite set is incomplete; refusing to prune")
    return selected


def expected_keys(
    suites: dict[str, dict[str, Any]],
) -> set[tuple[str, str, str]]:
    return {
        (suite_id, str(device), str(variant))
        for suite_id, suite in suites.items()
        for device in suite["commands"]
        for variant in suite["variants"]
    }


def valid_candidate(
    manifest_path: Path, suites: dict[str, dict[str, Any]]
) -> tuple[tuple[str, str, str] | None, list[str]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    suite_id = str(manifest.get("suite_id", ""))
    device = str(manifest.get("device", {}).get("class", ""))
    variant = str(manifest.get("variant", ""))
    key = (suite_id, device, variant)
    problems: list[str] = []
    suite = suites.get(suite_id)
    if suite is None:
        return None, ["suite is not pinned"]
    if manifest.get("measurement_release") != RELEASE_NAME:
        return None, ["measurement release differs"]
    if manifest.get("suite_version") != suite["suite_version"]:
        return None, ["suite version differs"]
    if manifest.get("status") != "passed":
        return None, ["status is not passed"]
    if str(manifest.get("completed_at_utc", "")) < RESULT_NOT_BEFORE_UTC:
        return None, ["run predates the final cross-case identity audit"]
    recorded_sources = {
        key: value for key, value in manifest.get("sources", {}).items()
        if not key.startswith("benchmarks/tma-cycle-compare/releases/")
    }
    current_sources = {
        key: value for key, value in run_tool.source_hashes(suite, device).items()
        if not key.startswith("benchmarks/tma-cycle-compare/releases/")
    }
    if recorded_sources != current_sources:
        return None, ["source hashes are stale"]
    measurement = manifest.get("measurement", {})
    if measurement.get("warmups") != 0 or measurement.get("repeats") != 2:
        problems.append("measurement is not warmups=0/repeats=2")
    if measurement.get("paid_attempts") != (0 if device == "ventus" else 1):
        problems.append("paid_attempt count differs")
    count, matrix_hash, raw_problems = run_tool.validate_raw(
        manifest_path.parent, 2
    )
    problems.extend(raw_problems)
    problems.extend(
        run_tool.validate_fresh_tensormap(
            manifest_path.parent, suite, device
        )
    )
    expected = suite.get("expected_cases", {}).get(device)
    if isinstance(expected, dict):
        expected = expected.get(variant)
    if expected is not None and count != int(expected):
        problems.append(f"case_count={count}, expected={expected}")
    if count != manifest.get("case_count"):
        problems.append("manifest case count differs")
    if matrix_hash != manifest.get("case_matrix_sha256"):
        problems.append("case matrix hash differs")
    limits = suite.get("resource_limited_cases", {}).get(device, 0)
    if isinstance(limits, dict):
        limits = limits.get(variant, 0)
    if resource_limit_count(manifest_path.parent) != int(limits):
        problems.append("resource-limit count differs")
    return (key if not problems else None), problems


def select_runs(
    suites: dict[str, dict[str, Any]]
) -> dict[tuple[str, str, str], Path]:
    candidates: dict[tuple[str, str, str], list[tuple[str, Path]]] = defaultdict(list)
    for path in sorted(DATA.rglob("run-manifest.json")):
        key, problems = valid_candidate(path, suites)
        if key is None:
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        candidates[key].append((str(manifest.get("completed_at_utc", "")), path.parent))
    expected = expected_keys(suites)
    missing = expected - set(candidates)
    if missing:
        raise SystemExit(
            "validated replacements are incomplete; refusing to prune: "
            + ", ".join("/".join(key) for key in sorted(missing))
        )
    return {key: sorted(values)[-1][1] for key, values in candidates.items()}


def remove_unpinned_suites() -> None:
    pinned = RELEASE.get("suites", {})
    targets: list[Path] = []
    for metadata in sorted((HERE / "suites").rglob("suite.yaml")):
        suite = run_tool.load_suite(metadata)
        if pinned.get(suite["suite_id"]) != suite["suite_version"]:
            targets.append(metadata.parent)
    for target in targets:
        shutil.rmtree(target)
    for directory in sorted(
        (path for path in (HERE / "suites").rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts), reverse=True,
    ):
        if not any(directory.iterdir()):
            directory.rmdir()


def remove_generated_artifacts() -> None:
    generated_names = {
        "ventus_tma_comprehensive", "ventus_tensormap_decode",
        "ventus_multi_context", "cuda_g2s_fresh_map",
    }
    for path in sorted((HERE / "suites").rglob("*")):
        if not path.is_file():
            continue
        if (
            path.name in generated_names
            or path.suffix in {".ptx", ".sass", ".cubin"}
            or "__pycache__" in path.parts
            or path.suffix == ".pyc"
        ):
            path.unlink()


def apply(selected: dict[tuple[str, str, str], Path]) -> None:
    staging = Path(tempfile.mkdtemp(prefix="tma-fresh-release-recovery-"))
    staged_data = staging / "data"
    staged_data.mkdir()
    expected_hashes: dict[Path, str] = {}
    for path in selected.values():
        relative = path.relative_to(DATA)
        destination = staged_data / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(path, destination)
        expected_hashes[relative] = tree_hash(destination)

    # DATA is an exact, resolved project subdirectory and every retained run
    # has already been copied to the recovery directory above.
    shutil.rmtree(DATA)
    DATA.mkdir(parents=True)
    for relative, expected_hash in expected_hashes.items():
        source = staged_data / relative
        destination = DATA / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)
        if tree_hash(destination) != expected_hash:
            raise SystemExit(
                f"restored hash mismatch for {relative}; recovery is {staging}"
            )

    for name in ("archive", "baselines", "results", "reports"):
        target = HERE / name
        if target.is_dir():
            shutil.rmtree(target)
    (HERE / "reports").mkdir()
    remove_unpinned_suites()
    remove_generated_artifacts()
    for name in (
        "checkpoints.csv", "disabled_suites.json", "legacy_path_map.csv",
        "migration_inventory.csv", "migration_summary.json", "runs.csv",
        "repository_validation.json",
    ):
        path = HERE / "catalog" / name
        if path.is_file():
            path.unlink()
    checksum = HERE / "releases" / f"{RELEASE_NAME}.SHA256SUMS"
    if checksum.is_file():
        checksum.unlink()
    for suffix in (".json", ".RESULT_NOT_BEFORE", ".SHA256SUMS"):
        for path in (HERE / "releases").glob(f"*{suffix}"):
            if path.name != f"{RELEASE_NAME}{suffix}":
                path.unlink()
    shutil.rmtree(staging)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    suites = pinned_suites()
    selected = select_runs(suites)
    print(f"release={RELEASE_NAME} validated_runs={len(selected)}")
    for key, path in sorted(selected.items()):
        print(f"KEEP {'/'.join(key)}: {path.relative_to(HERE)}")
    if not args.apply:
        print("dry run only; pass --apply after reviewing the complete set")
        return 0
    apply(selected)
    print("pruned historical tests/data/reports; retained runs verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
