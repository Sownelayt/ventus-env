#!/usr/bin/env python3
"""Validate the one authoritative TMA measurement release.

Background:
  The release named by CURRENT_MEASUREMENT_RELEASE replaces every historical
  TMA cycle result.  This checker
  deliberately rejects old suite versions, old-release data, diagnostics,
  duplicate formal runs, stale source hashes, incomplete repeat pairs and
  TensorMap reuse.  Passing this command means reports can only see the
  frozen method and its one validated run per device/variant.

Usage:
  python3 benchmarks/tma-cycle-compare/validate_repository.py

Maintenance:
  Create a new measurement release instead of weakening these checks.  The
  release JSON is the only source of truth for pinned suite versions.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import run as run_tool


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "data"
REPORTS = HERE / "reports"
CATALOG = HERE / "catalog"
CURRENT_RELEASE = (HERE / "CURRENT_MEASUREMENT_RELEASE").read_text(
    encoding="utf-8"
).strip()
RELEASE = json.loads(
    (HERE / "releases" / f"{CURRENT_RELEASE}.json").read_text(encoding="utf-8")
)
FORMAL = {"warmups": 0, "repeats": 2, "paid_attempts": 1}
RAW_EXCLUDED = {
    "partial.csv", "summary.csv", "coverage.csv", "ratios.csv", "trends.csv",
    "resource_limits.csv", "pmu.csv", "completion_cycles.csv",
}
GENERATED_SUFFIXES = {".ptx", ".sass", ".cubin"}
GENERATED_NAMES = {
    "ventus_tma_comprehensive", "ventus_tensormap_decode",
    "ventus_multi_context", "cuda_g2s_fresh_map",
}
RESULT_NOT_BEFORE_UTC = (
    HERE / "releases" / f"{CURRENT_RELEASE}.RESULT_NOT_BEFORE"
).read_text(encoding="utf-8").strip()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_run_keys(
    suites: dict[str, dict[str, Any]],
) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    for suite_id, suite in suites.items():
        for device in suite["commands"]:
            for variant in suite["variants"]:
                keys.add((suite_id, str(device), str(variant)))
    return keys


def check_release_and_suites(
    errors: list[str], summary: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    if RELEASE.get("release") != CURRENT_RELEASE:
        errors.append("CURRENT_MEASUREMENT_RELEASE and release JSON disagree")
    if RELEASE.get("status") not in {"frozen-method", "frozen-results"}:
        errors.append(f"invalid release status {RELEASE.get('status')!r}")
    contract = RELEASE.get("formal_contract", {})
    for key, expected in FORMAL.items():
        if contract.get(key) != expected:
            errors.append(f"release formal_contract.{key}={contract.get(key)!r}")

    suites: dict[str, dict[str, Any]] = {}
    actual_directories: set[tuple[str, str]] = set()
    for path in sorted((HERE / "suites").rglob("suite.yaml")):
        try:
            suite = run_tool.load_suite(path)
        except SystemExit as error:
            errors.append(str(error))
            continue
        suite_id = str(suite["suite_id"])
        version = str(suite["suite_version"])
        actual_directories.add((suite_id, version))
        pinned = RELEASE.get("suites", {}).get(suite_id)
        if pinned != version:
            errors.append(f"unreleased suite remains: {suite_id}@{version}")
            continue
        if suite_id in suites:
            errors.append(f"duplicate pinned suite: {suite_id}@{version}")
            continue
        suites[suite_id] = suite
        if suite.get("measurement_release") != CURRENT_RELEASE:
            errors.append(f"{suite_id}: wrong measurement_release")
        if suite.get("formal") != FORMAL:
            errors.append(f"{suite_id}: formal contract mismatch")
        for field in (
            "wait_method", "data_residency", "descriptor_residency",
            "direction_definition", "context_definition",
        ):
            if not suite.get(field):
                errors.append(f"{suite_id}: missing {field}")
        for source in suite.get("sources", []):
            if not (path.parent / str(source)).is_file():
                errors.append(f"{suite_id}: missing source {source}")

    expected_directories = {
        (str(suite_id), str(version))
        for suite_id, version in RELEASE.get("suites", {}).items()
    }
    if actual_directories != expected_directories:
        errors.append(
            "suite directory set mismatch: "
            f"missing={sorted(expected_directories - actual_directories)} "
            f"extra={sorted(actual_directories - expected_directories)}"
        )
    summary["suite_count"] = len(suites)
    summary["expected_formal_runs"] = len(expected_run_keys(suites))
    return suites


def resource_limit_count(directory: Path) -> int:
    path = directory / "resource_limits.csv"
    if not path.is_file():
        return 0
    with path.open(newline="", encoding="utf-8") as source:
        return sum(1 for _ in csv.DictReader(source))


def check_formal_data(
    errors: list[str], summary: dict[str, Any], suites: dict[str, dict[str, Any]]
) -> None:
    expected = expected_run_keys(suites)
    observed: dict[tuple[str, str, str], list[Path]] = defaultdict(list)
    manifests = sorted(DATA.rglob("run-manifest.json")) if DATA.is_dir() else []
    for path in manifests:
        try:
            manifest = load_json(path)
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"invalid manifest {path}: {error}")
            continue
        suite_id = str(manifest.get("suite_id", ""))
        device = str(manifest.get("device", {}).get("class", ""))
        variant = str(manifest.get("variant", ""))
        key = (suite_id, device, variant)
        observed[key].append(path)
        suite = suites.get(suite_id)
        if suite is None:
            errors.append(f"{path}: suite is not pinned")
            continue
        if manifest.get("measurement_release") != CURRENT_RELEASE:
            errors.append(f"{path}: old measurement release")
        if manifest.get("suite_version") != suite.get("suite_version"):
            errors.append(f"{path}: suite version is not pinned")
        if manifest.get("status") != "passed":
            errors.append(f"{path}: status is not passed")
        if str(manifest.get("completed_at_utc", "")) < RESULT_NOT_BEFORE_UTC:
            errors.append(f"{path}: predates final descriptor-identity audit")
        measurement = manifest.get("measurement", {})
        if measurement.get("warmups") != 0 or measurement.get("repeats") != 2:
            errors.append(f"{path}: measurement is not warmups=0/repeats=2")
        expected_paid = 0 if device == "ventus" else 1
        if measurement.get("paid_attempts") != expected_paid:
            errors.append(f"{path}: paid_attempts is not {expected_paid}")

        # Release JSON is a mutable freeze index, not executable test input.
        # Runs created while the release was being assembled may record an
        # earlier hash for it.  Continue to compare every suite/RTL/runner
        # source exactly, while ignoring only release-index files.
        recorded_sources = {
            key: value for key, value in manifest.get("sources", {}).items()
            if not key.startswith("benchmarks/tma-cycle-compare/releases/")
        }
        current_sources = {
            key: value for key, value in run_tool.source_hashes(suite, device).items()
            if not key.startswith("benchmarks/tma-cycle-compare/releases/")
        }
        if recorded_sources != current_sources:
            errors.append(f"{path}: source hashes differ from the frozen suite")
        case_count, matrix_hash, problems = run_tool.validate_raw(path.parent, 2)
        problems.extend(
            run_tool.validate_fresh_tensormap(path.parent, suite, device)
        )
        if case_count != manifest.get("case_count"):
            problems.append(
                f"case_count={case_count}, manifest={manifest.get('case_count')}"
            )
        if matrix_hash != manifest.get("case_matrix_sha256"):
            problems.append("case matrix hash mismatch")
        expected_cases: Any = suite.get("expected_cases", {}).get(device)
        if isinstance(expected_cases, dict):
            expected_cases = expected_cases.get(variant)
        if expected_cases is not None and case_count != int(expected_cases):
            problems.append(f"case_count={case_count}, expected={expected_cases}")
        expected_limits: Any = suite.get("resource_limited_cases", {}).get(device, 0)
        if isinstance(expected_limits, dict):
            expected_limits = expected_limits.get(variant, 0)
        actual_limits = resource_limit_count(path.parent)
        if actual_limits != int(expected_limits):
            problems.append(
                f"resource_limits={actual_limits}, expected={expected_limits}"
            )
        for problem in problems:
            errors.append(f"{path}: {problem}")

    for key in sorted(expected | set(observed)):
        count = len(observed.get(key, []))
        if key not in expected:
            errors.append(f"unexpected formal run key {key}")
        elif count != 1:
            errors.append(f"formal run key {key} has {count} manifests, expected 1")

    diagnostics = sorted(DATA.rglob("diagnostic-manifest.json")) if DATA.is_dir() else []
    if diagnostics:
        errors.append(f"current data tree still contains {len(diagnostics)} diagnostic runs")
    checkpoints = [
        path for path in DATA.rglob("202*__v*__*")
        if path.is_dir()
        and not (path / "run-manifest.json").is_file()
        and not (path / "diagnostic-manifest.json").is_file()
    ] if DATA.is_dir() else []
    if checkpoints:
        errors.append(f"current data tree still contains {len(checkpoints)} checkpoints")
    summary["formal_run_count"] = len(manifests)
    summary["diagnostic_run_count"] = len(diagnostics)
    summary["checkpoint_count"] = len(checkpoints)


def check_reports(errors: list[str], summary: dict[str, Any]) -> None:
    report_root = REPORTS / "comparisons" / CURRENT_RELEASE
    required = {
        report_root / "REPORT.md",
        report_root / "report-manifest.json",
        report_root / "single_instruction_g2s.csv",
        report_root / "single_instruction_s2g.csv",
        report_root / "single_instruction_roundtrip.csv",
        report_root / "completion_threshold.csv",
        report_root / "data_residency_128b.csv",
        report_root / "multi_context_samples.csv",
        report_root / "run_inventory.csv",
    }
    for path in required:
        if not path.is_file():
            errors.append(f"missing report {path.relative_to(HERE)}")
    old_reports = [
        path for path in REPORTS.rglob("*")
        if path.is_file() and CURRENT_RELEASE not in path.as_posix()
    ] if REPORTS.is_dir() else []
    if old_reports:
        errors.append(f"reports tree still contains {len(old_reports)} old-report files")

    report_manifest = report_root / "report-manifest.json"
    if report_manifest.is_file():
        try:
            report = load_json(report_manifest)
            if report.get("report_status") != "final":
                errors.append("current report is not final")
            if report.get("missing") or report.get("missing_unaccounted"):
                errors.append("current report still has missing formal runs")
            if report.get("diagnostics"):
                errors.append("current report still includes diagnostics")
            if len(report.get("formal_runs", {})) != summary.get(
                "expected_formal_runs"
            ):
                errors.append("current report formal-run count differs")
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"invalid report manifest: {error}")

    checked = 0
    documents = [HERE / "README.md", *sorted(REPORTS.rglob("*.md"))]
    for document in documents:
        if not document.is_file():
            continue
        for target in re.findall(
            r"\[[^]]*\]\(([^)]+)\)", document.read_text(encoding="utf-8")
        ):
            if target.startswith(("http://", "https://", "#")):
                continue
            clean = target.split("#", 1)[0]
            if clean and not (document.parent / clean).resolve().exists():
                errors.append(
                    f"broken link {document.relative_to(HERE)} -> {target}"
                )
            checked += 1
    summary["local_links_checked"] = checked


def check_generated_files(errors: list[str], summary: dict[str, Any]) -> None:
    tracked = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--", "benchmarks/tma-cycle-compare"],
        text=True, capture_output=True, check=False,
    ).stdout.splitlines()
    banned = [
        name for name in tracked
        if Path(name).suffix in GENERATED_SUFFIXES
        or Path(name).name in GENERATED_NAMES
        or "__pycache__" in Path(name).parts
    ]
    if banned:
        errors.append("tracked generated artifacts: " + ", ".join(sorted(banned)))
    summary["tracked_generated_artifacts"] = len(banned)


def check_release_hashes(errors: list[str], summary: dict[str, Any]) -> None:
    path = HERE / "releases" / f"{CURRENT_RELEASE}.SHA256SUMS"
    if not path.is_file():
        errors.append("missing release SHA256SUMS")
        summary["release_hashes_checked"] = 0
        return
    checked = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            errors.append(f"invalid checksum line: {line!r}")
            continue
        target = HERE / relative
        if not target.is_file():
            errors.append(f"checksum target missing: {relative}")
        elif sha256_file(target) != expected:
            errors.append(f"checksum mismatch: {relative}")
        checked += 1
    summary["release_hashes_checked"] = checked


def main() -> int:
    errors: list[str] = []
    summary: dict[str, Any] = {
        "schema_version": 2,
        "measurement_release": CURRENT_RELEASE,
    }
    suites = check_release_and_suites(errors, summary)
    check_formal_data(errors, summary, suites)
    check_reports(errors, summary)
    check_generated_files(errors, summary)
    check_release_hashes(errors, summary)
    summary["errors"] = errors
    summary["passed"] = not errors
    CATALOG.mkdir(parents=True, exist_ok=True)
    path = CATALOG / "repository_validation.json"
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
