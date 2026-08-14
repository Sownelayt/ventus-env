#!/usr/bin/env python3
"""Unified, provenance-first entry point for versioned TMA cycle suites.

Background:
  Formal CUDA and Ventus measurements previously used unrelated entry points
  and ad-hoc output names.  This dispatcher resolves an immutable suite
  package, enforces the no-warmup/two-sample contract, rejects duplicate test
  signatures, and writes a common run manifest beside the raw output.

Flow:
  ``--plan`` is read-only and prints the exact command/output/signature.
  ``--import-only`` performs CPU/build probes without creating formal data.
  ``--formal`` executes the suite with warmups=0, repeats=2 and one paid CUDA
  attempt, then registers the result only after validating its raw rows.

Usage:
  python3 benchmarks/tma-cycle-compare/run.py --list
  python3 benchmarks/tma-cycle-compare/run.py \
    --suite single_instruction.common_matrix --device h100 --plan
  python3 benchmarks/tma-cycle-compare/run.py \
    --suite single_instruction.common_matrix --device ventus --formal

Maintenance:
  Suite directories are immutable.  Change a test by adding v2, not by editing
  an already used v1 package.  A forced duplicate requires a written reason.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SUITES = HERE / "suites"
DATA = HERE / "data"
CATALOG = HERE / "catalog"
RUNS_CSV = CATALOG / "runs.csv"
SUITES_CSV = CATALOG / "suites.csv"
DISABLED_SUITES = CATALOG / "disabled_suites.json"
CURRENT_RELEASE_FILE = HERE / "CURRENT_MEASUREMENT_RELEASE"
RELEASES = HERE / "releases"
MODAL = Path("/home/liyb/tma-refer/.venv/bin/modal")
FORMAL_WARMUPS = 0
FORMAL_REPEATS = 2
FORMAL_PAID_ATTEMPTS = 1


def raw_csv_paths(output: Path) -> list[Path]:
    return sorted({*output.rglob("*.csv"), *output.rglob("*.csv.gz")})


def open_csv(path: Path):
    if path.name.endswith(".csv.gz"):
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return path.open(newline="", encoding="utf-8")


def current_release() -> dict[str, Any]:
    name = CURRENT_RELEASE_FILE.read_text(encoding="utf-8").strip()
    path = RELEASES / f"{name}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("release") != name or not isinstance(data.get("suites"), dict):
        raise SystemExit(f"invalid current measurement release: {path}")
    return data


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_suite(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot read suite metadata {path}: {error}") from error
    required = {"suite_id", "suite_version", "scope", "project", "variants", "formal", "sources", "commands"}
    missing = required - set(data)
    if missing:
        raise SystemExit(f"{path}: missing {sorted(missing)}")
    data["metadata_path"] = path
    data["directory"] = path.parent
    return data


def disabled_suites() -> dict[str, str]:
    if not DISABLED_SUITES.is_file():
        return {}
    data = json.loads(DISABLED_SUITES.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{DISABLED_SUITES}: expected an object")
    return {str(key): str(value) for key, value in data.items()}


def discover_suites() -> dict[str, dict[str, Any]]:
    suites: dict[str, dict[str, Any]] = {}
    disabled = disabled_suites()
    for path in sorted(SUITES.rglob("suite.yaml")):
        suite = load_suite(path)
        key = str(suite["suite_id"])
        version_key = f"{key}@{suite['suite_version']}"
        if version_key in disabled:
            continue
        current = suites.get(key)
        version = int(str(suite["suite_version"])[1:])
        if current is None or version > int(str(current["suite_version"])[1:]):
            suites[key] = suite
    return suites


def source_hashes(suite: dict[str, Any], device: str) -> dict[str, str]:
    directory = Path(suite["directory"])
    paths = [directory / str(name) for name in suite["sources"]]
    paths.append(Path(suite["metadata_path"]))
    paths.append(CURRENT_RELEASE_FILE)
    # The release JSON is an index that is completed as suites/runs freeze; it
    # is not executable test input.  Including it made every earlier run look
    # source-stale whenever a later suite version was pinned.  The immutable
    # suite.yaml and every actual CUDA/RTL/runner source remain hashed below.
    if device == "ventus":
        paths.extend(sorted((ROOT / "gpgpu/ventus/src/pipeline").glob("TmaV2*.scala")))
        paths.extend(sorted((ROOT / "gpgpu/ventus/src/pipeline").glob("TmaV34*.scala")))
        paths.extend(
            [
                ROOT / "gpgpu/ventus/src/pipeline/pipe.scala",
                ROOT / "gpgpu/ventus/src/top/parameters.scala",
            ]
        )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise SystemExit("missing suite sources: " + ", ".join(map(str, missing)))
    return {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in paths
    }


def implementation(device: str, sources: dict[str, str]) -> dict[str, Any]:
    if device == "ventus":
        gpgpu = ROOT / "gpgpu"
        commit = subprocess.run(
            ["git", "-C", str(gpgpu), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=False,
        ).stdout.strip()
        source_fingerprint = sha256_bytes(
            "\n".join(f"{name} {digest}" for name, digest in sorted(sources.items())).encode()
        )
        patch = subprocess.run(
            ["git", "-C", str(gpgpu), "diff", "--binary", "HEAD", "--", "ventus/src/pipeline", "ventus/src/top/parameters.scala"],
            capture_output=True,
            check=False,
        ).stdout
        libraries = [
            ROOT / "install/lib/libVentusRTL-withcache.so",
            ROOT / "install/lib/librtlsim_driver.so",
            ROOT / "install/lib/libVentusGVM-withcache.so",
            ROOT / "install/lib/libgvm_driver.so",
            ROOT / "install/lib/libgvmref.so",
        ]
        return {
            "version": "V3.10",
            "release_state": "development-unfrozen",
            "git_commit": commit or "unavailable",
            "dirty_patch_sha256": sha256_bytes(patch),
            "tma_source_tree_sha256": source_fingerprint,
            "simulator_sha256": {
                path.name: sha256_file(path) for path in libraries if path.is_file()
            },
        }
    return {
        "version": "cuda-13.3-sm90" if device == "h100" else "cuda-13.3-sm100",
        "release_state": "reference",
    }


def signature_payload(
    suite: dict[str, Any], device: str, variant: str, sources: dict[str, str]
) -> dict[str, Any]:
    return {
        "measurement_release": suite.get("measurement_release"),
        "suite_id": suite["suite_id"],
        "suite_version": suite["suite_version"],
        "device": device,
        "variant": variant,
        "warmups": FORMAL_WARMUPS,
        "repeats": FORMAL_REPEATS,
        "paid_attempts": 0 if device == "ventus" else FORMAL_PAID_ATTEMPTS,
        "pairing": suite.get("pairing", "same-kernel-consecutive"),
        "sources": sources,
    }


def test_signature(payload: dict[str, Any]) -> str:
    return sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def existing_signatures() -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in DATA.rglob("run-manifest.json") if DATA.is_dir() else []:
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        signature = manifest.get("test_signature")
        if isinstance(signature, str):
            result[signature] = path
    return result


def output_path(suite: dict[str, Any], device: str, variant: str, run_id: str) -> Path:
    version = "V3.10" if device == "ventus" else (
        "cuda-13.3-sm90" if device == "h100" else "cuda-13.3-sm100"
    )
    return DATA / device / version / str(suite["scope"]) / str(suite["project"]) / variant / run_id


def format_command(
    suite: dict[str, Any], device: str, variant: str, output: Path
) -> list[str]:
    template = suite["commands"].get(device)
    if not isinstance(template, list):
        raise SystemExit(f"{suite['suite_id']} does not support {device}")
    replacements = {
        "modal": str(MODAL),
        "suite": str(Path(suite["directory"])),
        "output": str(output),
        "variant": variant,
    }
    return [str(token).format(**replacements) for token in template]


def write_suite_catalog(suites: dict[str, dict[str, Any]]) -> None:
    CATALOG.mkdir(parents=True, exist_ok=True)
    with SUITES_CSV.open("w", newline="", encoding="utf-8") as output:
        fields = ("suite_id", "suite_version", "scope", "project", "variants", "path")
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for suite in sorted(suites.values(), key=lambda item: str(item["suite_id"])):
            writer.writerow(
                {
                    "suite_id": suite["suite_id"],
                    "suite_version": suite["suite_version"],
                    "scope": suite["scope"],
                    "project": suite["project"],
                    "variants": ";".join(suite["variants"]),
                    "path": Path(suite["metadata_path"]).relative_to(HERE).as_posix(),
                }
            )


def validate_raw(output: Path, repeats: int) -> tuple[int, str, list[str]]:
    rows: list[dict[str, str]] = []
    for path in raw_csv_paths(output):
        if path.name in {"partial.csv", "summary.csv", "coverage.csv", "ratios.csv", "trends.csv"}:
            continue
        try:
            with open_csv(path) as source:
                reader = csv.DictReader(source)
                if reader.fieldnames and "repeat" in reader.fieldnames:
                    rows.extend(reader)
        except (OSError, csv.Error):
            continue
    problems: list[str] = []
    grouped: dict[str, set[int]] = {}
    for row in rows:
        if row.get("case_id"):
            # Some preserved control programs intentionally reuse a size-based
            # case ID across directions/methods.  These fields are part of the
            # measurement identity, while remaining empty for suites whose
            # case_id is already globally unique.
            case = "|".join(
                (row["case_id"], row.get("direction", ""), row.get("method", ""))
            )
        else:
            case = "|".join(
                row.get(field, "")
                for field in (
                    "project", "path", "direction", "scenario", "rank",
                    "contexts", "pairs", "bytes", "dtype",
                    "prefetch_gap_cycles", "level", "map_mode", "issue_mode",
                    "mode", "order", "data_state"
                )
            )
        try:
            repeat = int(row.get("repeat", "-1"))
        except ValueError:
            problems.append(f"{case}: invalid repeat")
            continue
        status = row.get("status", "ok").lower()
        if status not in {"ok", "pass", "0", ""}:
            problems.append(f"{case}: status={status}")
            continue
        observed = grouped.setdefault(case, set())
        if repeat in observed:
            problems.append(f"{case}: duplicate repeat={repeat}")
        observed.add(repeat)
        try:
            errors = int(row.get("errors", "0") or 0)
        except ValueError:
            problems.append(f"{case}: invalid errors field")
        else:
            if errors != 0:
                problems.append(f"{case}: correctness error")
    expected = set(range(repeats))
    for case, observed in grouped.items():
        if observed != expected:
            problems.append(f"{case}: repeats={sorted(observed)}, expected={sorted(expected)}")
    if not rows:
        problems.append("no repeat-bearing CSV rows")
    matrix_hash = sha256_bytes(("\n".join(sorted(grouped)) + "\n").encode())
    return len(grouped), matrix_hash, problems


def repeat_rows(output: Path) -> list[tuple[Path, dict[str, str]]]:
    result: list[tuple[Path, dict[str, str]]] = []
    excluded = {
        "partial.csv", "summary.csv", "coverage.csv", "ratios.csv",
        "trends.csv", "resource_limits.csv", "pmu.csv",
        "completion_cycles.csv",
    }
    for path in raw_csv_paths(output):
        if path.name in excluded:
            continue
        try:
            with open_csv(path) as source:
                reader = csv.DictReader(source)
                if reader.fieldnames and "repeat" in reader.fieldnames:
                    result.extend((path, row) for row in reader)
        except (OSError, csv.Error):
            continue
    return result


def validate_fresh_tensormap(
    output: Path, suite: dict[str, Any], device: str
) -> list[str]:
    """Enforce the frozen one-cold-then-one-hot TensorMap contract."""
    suite_id = str(suite["suite_id"])
    rows = repeat_rows(output)
    problems: list[str] = []

    def group(fields: tuple[str, ...]) -> dict[tuple[str, ...], list[dict[str, str]]]:
        grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
        for _, row in rows:
            grouped.setdefault(tuple(row.get(field, "") for field in fields), []).append(row)
        return grouped

    def unique_nonzero(
        case: str, sample_rows: list[dict[str, str]], field: str,
        *, hexadecimal: bool = True,
    ) -> None:
        values = [row.get(field, "") for row in sample_rows]
        missing_values = {""} if not hexadecimal else {
            "", "0", "0x0", "0x0000000000000000"
        }
        values = [value for value in values if value not in missing_values]
        if not values:
            problems.append(f"{case}: missing {field}")
            return
        if len(values) != len(set(values)):
            problems.append(f"{case}: reused {field}")
        if hexadecimal:
            try:
                parsed = [int(value, 16) for value in values]
            except ValueError:
                problems.append(f"{case}: invalid {field}")
                return
            if device != "ventus" and "descriptor" in field and "fingerprint" not in field:
                if any(value % 4096 for value in parsed):
                    problems.append(f"{case}: {field} is not 4 KiB aligned")

    def same_nonzero(
        case: str, sample_rows: list[dict[str, str]], field: str,
        *, hexadecimal: bool = True,
    ) -> None:
        """Require one nonzero identity reused by both members of a pair."""
        values = [row.get(field, "") for row in sample_rows]
        missing_values = {""} if not hexadecimal else {
            "", "0", "0x0", "0x0000000000000000"
        }
        if len(values) != 2 or any(value in missing_values for value in values):
            problems.append(f"{case}: missing {field} in cold/hot pair")
            return
        if values[0] != values[1]:
            problems.append(f"{case}: cold/hot pair changed {field}")
        if hexadecimal:
            try:
                parsed = [int(value, 16) for value in values]
            except ValueError:
                problems.append(f"{case}: invalid {field}")
                return
            if device != "ventus" and "descriptor" in field and "fingerprint" not in field:
                if any(value % 4096 for value in parsed):
                    problems.append(f"{case}: {field} is not 4 KiB aligned")

    def validate_pair(case: str, case_rows: list[dict[str, str]]) -> list[dict[str, str]]:
        """Return rows ordered cold then hot, reporting every pairing error."""
        try:
            ordered = sorted(case_rows, key=lambda row: int(row.get("repeat", "-1")))
        except ValueError:
            problems.append(f"{case}: invalid repeat index")
            return case_rows
        if [row.get("repeat") for row in ordered] != ["0", "1"]:
            problems.append(f"{case}: expected exactly repeat 0 and repeat 1")
        if [row.get("pair_role") for row in ordered] != ["cold", "hot"]:
            problems.append(f"{case}: expected pair_role cold then hot")
        if [row.get("prior_tma_use") for row in ordered] != ["0", "1"]:
            problems.append(f"{case}: expected prior_tma_use 0 then 1")
        if any(row.get("tma_prefetch_before_issue") != "0" for row in ordered):
            problems.append(f"{case}: hidden TensorMap prefetch")
        expected_states = (
            ["tmau_cold_first_use_fresh_allocation", "tmau_hot_reuse"]
            if device == "ventus"
            else ["tmau_cold_first_use_l2_hot", "tmau_hot_reuse"]
        )
        if [row.get("descriptor_state") for row in ordered] != expected_states:
            problems.append(f"{case}: descriptor_state is not cold then hot")
        return ordered

    def unique_across_fields(
        case: str, sample_rows: list[dict[str, str]], fields: tuple[str, ...],
        *, hexadecimal: bool = True,
    ) -> None:
        values = [
            row.get(field, "")
            for row in sample_rows
            for field in fields
            if row.get(field, "") not in (
                {"", "0", "0x0", "0x0000000000000000"}
                if hexadecimal else {""}
            )
        ]
        if not values:
            problems.append(f"{case}: missing {'/'.join(fields)}")
            return
        if len(values) != len(set(values)):
            problems.append(f"{case}: descriptor identity reused across cases")
        if hexadecimal:
            try:
                parsed = [int(value, 16) for value in values]
            except ValueError:
                problems.append(f"{case}: invalid {'/'.join(fields)}")
                return
            if (
                device != "ventus"
                and all("fingerprint" not in field for field in fields)
                and any(value % 4096 for value in parsed)
            ):
                problems.append(f"{case}: descriptor address is not 4 KiB aligned")

    def rows_by_file() -> dict[Path, list[dict[str, str]]]:
        result: dict[Path, list[dict[str, str]]] = {}
        for path, row in rows:
            result.setdefault(path, []).append(row)
        return result

    if suite_id in {
        "single_instruction.common_matrix",
        "single_instruction.tensor_2d_capacity",
    }:
        tensor = [row for _, row in rows if row.get("path") == "tensor"]
        first_uses_by_file: dict[Path, list[dict[str, str]]] = {}
        for case_rows in group(("case_id",)).values():
            case_rows = [row for row in case_rows if row.get("path") == "tensor"]
            if not case_rows:
                continue
            case = case_rows[0].get("case_id", "unknown")
            case_rows = validate_pair(case, case_rows)
            direction = case_rows[0].get("direction")
            if direction != "s2g":
                same_nonzero(case, case_rows, "source_descriptor_va")
                same_nonzero(case, case_rows, "source_descriptor_fingerprint64")
            if direction != "g2s":
                same_nonzero(case, case_rows, "destination_descriptor_va")
                same_nonzero(case, case_rows, "destination_descriptor_fingerprint64")
            # CUDA runs many cases in one persistent process, so repeat 0 must
            # also prove that each case allocated a new TensorMap.  Ventus
            # launches an isolated simulator process per case; the TMAU is
            # reset there, so a repeated virtual address is not reuse.
            if device != "ventus" and case_rows:
                row_path = next(
                    (path for path, row in rows if row is case_rows[0]), None
                )
                if row_path is not None:
                    first_uses_by_file.setdefault(row_path, []).append(case_rows[0])
        if device != "ventus":
            for path, tensor_rows in first_uses_by_file.items():
                # Pure Bulk shards intentionally have no TensorMap identity.
                # They are part of the common matrix, but are not descriptor-
                # cold evidence and must not be rejected for zero descriptor
                # fields. Tensor-containing shards are still audited globally
                # below so identities cannot be reused across measurement cases.
                if not tensor_rows:
                    continue
                unique_across_fields(
                    f"{path.name}: all cold Tensor first uses", tensor_rows,
                    ("source_descriptor_va", "destination_descriptor_va"),
                )
                unique_across_fields(
                    f"{path.name}: all cold Tensor first-use fingerprints", tensor_rows,
                    (
                        "source_descriptor_fingerprint64",
                        "destination_descriptor_fingerprint64",
                    ),
                )
        if not tensor:
            problems.append("fresh-map suite has no Tensor rows")

    elif suite_id == "single_instruction.tensormap_decode":
        for key, case_rows in group(
            ("direction", "rank", "scenario", "prefetch_gap_cycles")
        ).items():
            scenario = key[2]
            if scenario in {"bulk_hot", ""}:
                continue
            case = "/".join(key)
            ordered = sorted(case_rows, key=lambda row: int(row.get("repeat", "-1")))
            if [row.get("repeat") for row in ordered] != ["0", "1"]:
                problems.append(f"{case}: expected exactly repeat 0/1")
            same_nonzero(case, ordered, "descriptor_va")
            same_nonzero(case, ordered, "descriptor_fingerprint64")
            if [row.get("prior_tma_use") for row in ordered] != ["0", "1"]:
                problems.append(f"{case}: prior_tma_use is not 0/1")
            expected_prefetch = ["1", "0"] if scenario in {"prefetch_lead", "explicit_prefetch"} else ["0", "0"]
            if [row.get("tma_prefetch_before_issue") for row in ordered] != expected_prefetch:
                problems.append(f"{case}: prefetch marker does not match scenario")
            expected_roles = ["prefetched", "hot"] if expected_prefetch[0] == "1" else ["cold", "hot"]
            if [row.get("pair_role") for row in ordered] != expected_roles:
                problems.append(f"{case}: pair_role does not match scenario")
        if device != "ventus":
            for path, file_rows in rows_by_file().items():
                tensor_rows = [
                    row for row in file_rows
                    if row.get("descriptor_va") and row.get("repeat") == "0"
                ]
                unique_across_fields(
                    f"{path.name}: all decode cases", tensor_rows,
                    ("descriptor_va",),
                )
                unique_across_fields(
                    f"{path.name}: all decode first-use fingerprints", tensor_rows,
                    ("descriptor_fingerprint64",),
                )

    elif suite_id == "single_instruction.data_residency_128b":
        residency_version = str(suite.get("suite_version", ""))
        residency_controls = residency_version in {"v4", "v5"}
        grouping = ("path", "direction", "scenario") if residency_controls else ("path", "direction")
        for key, case_rows in group(grouping).items():
            if key[0] != "tensor":
                continue
            case = "/".join(key)
            ordered = sorted(case_rows, key=lambda row: int(row.get("repeat", "-1")))
            expected_roles = (
                {
                    "cold_hot": ["cold", "hot"],
                    "cold_cold": ["cold", "cold"],
                    "hot_hot": ["hot", "hot"],
                }.get(key[2], [])
                if residency_controls else ["cold", "hot"]
            )
            if [row.get("pair_role") for row in ordered] != expected_roles:
                problems.append(f"{case}: payload residency roles do not match scenario")
            expected_prior = ["1", "2"] if residency_version == "v5" else ["0", "1"]
            if [row.get("prior_tma_use") for row in ordered] != expected_prior:
                problems.append(f"{case}: prior use does not match the primer contract")
            expected_prefetch = "1" if residency_version == "v4" else "0"
            if any(row.get("tma_prefetch_before_issue") != expected_prefetch for row in ordered):
                problems.append(f"{case}: TensorMap prefetch marker does not match suite")
            expected_state = {
                "v4": "tmau_explicit_prefetch_hot",
                "v5": "tmau_hot_after_untimed_tma_primer",
            }.get(residency_version)
            if expected_state and any(
                row.get("descriptor_state") != expected_state for row in ordered
            ):
                problems.append(f"{case}: descriptor was not fixed to the expected state")
            same_nonzero(case, ordered, "descriptor_va")
            same_nonzero(case, ordered, "descriptor_fingerprint64")
        for path, file_rows in rows_by_file().items():
            tensor_rows = [
                row for row in file_rows
                if row.get("path") == "tensor" and row.get("repeat") == "0"
            ]
            unique_across_fields(
                f"{path.name}: all residency first uses", tensor_rows,
                ("descriptor_va",),
            )

    elif suite_id == "single_instruction.control_items":
        for key, case_rows in group(("direction", "method")) .items():
            if key[1] not in {"tma_serial", "tma_overlap"}:
                continue
            case = "/".join(key)
            ordered = sorted(case_rows, key=lambda row: int(row.get("repeat", "-1")))
            if [row.get("pair_role") for row in ordered] != ["cold", "hot"]:
                problems.append(f"{case}: pair is not cold/hot")
            if [row.get("prior_tma_use") for row in ordered] != ["0", "1"]:
                problems.append(f"{case}: prior use is not 0/1")
            if any(row.get("tma_prefetch_before_issue") != "0" for row in ordered):
                problems.append(f"{case}: hidden TensorMap prefetch")
            same_nonzero(case, ordered, "descriptor_va")
            same_nonzero(case, ordered, "descriptor_fingerprint64")
        for path, file_rows in rows_by_file().items():
            tensor_rows = [
                row for row in file_rows
                if row.get("method") in {"tma_serial", "tma_overlap"}
                and row.get("repeat") == "0"
            ]
            unique_across_fields(
                f"{path.name}: all pointer-Tensor first uses", tensor_rows,
                ("descriptor_va",),
            )

    elif suite_id.startswith("multi_context."):
        fields = (
            "project", "level", "path", "direction", "order", "mode",
            "map_mode", "bytes", "contexts", "commands",
        )
        for key, case_rows in group(fields).items():
            if key[2] != "tensor":
                continue
            case = "/".join(key)
            ordered = sorted(case_rows, key=lambda row: int(row.get("repeat", "-1")))
            if [row.get("repeat") for row in ordered] != ["0", "1"]:
                problems.append(f"{case}: expected exactly repeat 0/1")
            if [row.get("pair_role") for row in ordered] != ["cold", "hot"]:
                problems.append(f"{case}: pair is not cold then hot")
            if [row.get("prior_tma_use") for row in ordered] != ["0", "1"]:
                problems.append(f"{case}: prior use is not 0 then 1")
            if any(
                row.get("descriptor_state")
                not in {"tmau_cold_bank_first_use", "tmau_hot_bank_reuse"}
                for row in ordered
            ):
                problems.append(f"{case}: descriptor state is not cold then hot")
            if any(row.get("tma_prefetch_before_issue") != "0" for row in ordered):
                problems.append(f"{case}: hidden TensorMap prefetch")
            if device == "ventus":
                if any(
                    row.get("descriptor_l2_warm_method")
                    != "patch_descriptor_write"
                    for row in case_rows
                ):
                    problems.append(f"{case}: descriptor preparation marker differs")
                same_nonzero(case, ordered, "descriptor_bank_word_offset", hexadecimal=False)
                same_nonzero(case, ordered, "descriptor_bank_fingerprint64", hexadecimal=False)
                same_nonzero(case, ordered, "descriptor_va")
            else:
                if any(
                    row.get("descriptor_l2_warm_method") != "ld.global.cg"
                    for row in case_rows
                ):
                    problems.append(f"{case}: descriptor was not warmed by ordinary L2 loads")
                for prefix in ("source", "destination"):
                    field = f"{prefix}_descriptor_bank_va"
                    if any(row.get(field, "") not in {"", "0", "0x0000000000000000"} for row in ordered):
                        same_nonzero(case, ordered, field)
                        same_nonzero(case, ordered, f"{prefix}_descriptor_bank_fingerprint64")
        if device != "ventus":
            for path, file_rows in rows_by_file().items():
                tensor_rows = [
                    row for row in file_rows
                    if row.get("path") == "tensor" and row.get("repeat") == "0"
                ]
                unique_across_fields(
                    f"{path.name}: all multi-context cases", tensor_rows,
                    (
                        "source_descriptor_bank_va",
                        "destination_descriptor_bank_va",
                    ),
                )

    elif suite_id == "single_instruction.completion_latency":
        first_uses: list[dict[str, str]] = []
        for key, case_rows in group(("case_id",)).items():
            if not case_rows or case_rows[0].get("path") != "tensor":
                continue
            case = key[0]
            ordered = sorted(case_rows, key=lambda row: int(row.get("repeat", "-1")))
            if [row.get("repeat") for row in ordered] != ["0", "1"]:
                problems.append(f"{case}: expected repeat 0/1")
                continue
            same_nonzero(case, ordered, "descriptor_va")
            same_nonzero(case, ordered, "descriptor_fingerprint64")
            if [row.get("prior_tma_use") for row in ordered] != ["0", "1"]:
                problems.append(f"{case}: prior use is not 0/1")
            expected_roles = (
                ["prefetched", "hot"]
                if ordered[0].get("descriptor_mode") == "tmau_prefetched_l2_hot"
                else ["cold", "hot"]
            )
            if [row.get("pair_role") for row in ordered] != expected_roles:
                problems.append(f"{case}: cold/hot role mismatch")
            first_uses.append(ordered[0])
        if not first_uses:
            problems.append("completion suite has no Tensor rows")
        else:
            by_context: dict[str, list[dict[str, str]]] = {}
            for row in first_uses:
                by_context.setdefault(row.get("cuda_context_shard", "0"), []).append(row)
            for context, context_rows in by_context.items():
                unique_across_fields(
                    f"completion context {context} first uses",
                    context_rows, ("descriptor_va",),
                )
                unique_across_fields(
                    f"completion context {context} first-use fingerprints",
                    context_rows, ("descriptor_fingerprint64",),
                )

    return problems


def child_metadata(output: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for raw in raw_csv_paths(output):
        try:
            with open_csv(raw) as source:
                first = next(csv.DictReader(source), None)
        except (OSError, csv.Error):
            continue
        if not first:
            continue
        if first.get("gpu"):
            result["name"] = first["gpu"]
        if first.get("cc"):
            result["compute_capability"] = first["cc"]
        if first.get("l2_bytes"):
            try:
                result["l2_bytes"] = int(first["l2_bytes"])
            except ValueError:
                pass
        break
    child_manifest = output / "manifest.json"
    if child_manifest.is_file():
        try:
            result["suite_manifest"] = json.loads(child_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            result["suite_manifest"] = {"parse_error": True}
    attempts: list[dict[str, Any]] = []
    for path in sorted(output.glob("*.json")):
        if path.name in {"manifest.json", "run-manifest.json"}:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        selected = {
            key: payload[key]
            for key in (
                "gpu", "nvidia_smi_query", "command", "child_timeout_seconds",
                "function_timeout_seconds", "wall_seconds", "classification"
            )
            if key in payload
        }
        if selected:
            selected["file"] = path.name
            attempts.append(selected)
    if attempts:
        result["attempts"] = attempts
        query = str(attempts[0].get("nvidia_smi_query", "")).strip().splitlines()
        if query:
            fields = [field.strip() for field in query[0].split(",")]
            if len(fields) >= 3:
                result.setdefault("name", fields[0])
                result.setdefault("compute_capability", fields[1])
                result["driver"] = fields[2]
            if len(fields) >= 4:
                try:
                    result["sm_clock_mhz"] = int(fields[3])
                except ValueError:
                    result["sm_clock_mhz"] = None
    return result


def append_run_catalog(manifest: dict[str, Any], path: Path) -> None:
    fields = (
        "run_id", "suite_id", "suite_version", "device", "implementation_version",
        "variant", "test_signature", "completed_at_utc", "path", "status",
        "reason", "superseded_by"
    )
    existing: list[dict[str, str]] = []
    if RUNS_CSV.is_file():
        with RUNS_CSV.open(newline="", encoding="utf-8") as source:
            existing = list(csv.DictReader(source))
    row = {
            "run_id": manifest["run_id"],
            "suite_id": manifest["suite_id"],
            "suite_version": manifest["suite_version"],
            "device": manifest["device"]["class"],
            "implementation_version": manifest["implementation"]["version"],
            "variant": manifest["variant"],
            "test_signature": manifest["test_signature"],
            "completed_at_utc": manifest["completed_at_utc"],
            "path": path.relative_to(HERE).as_posix(),
            "status": manifest["status"],
            "reason": manifest.get("forced_rerun_reason", manifest.get("reason", "")),
            "superseded_by": "",
        }
    relative_path = row["path"]
    existing = [item for item in existing if item.get("path") != relative_path]
    if row["status"] == "passed":
        for item in existing:
            same_series = all(
                item.get(key) == row[key]
                for key in ("suite_id", "device", "variant")
            )
            if same_series and item.get("status") == "passed":
                item["status"] = "superseded"
                item["superseded_by"] = relative_path
                if not item.get("reason"):
                    item["reason"] = "newer validated suite/run in the same device series"
    existing.append(row)
    with RUNS_CSV.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing)


def reconcile_run_catalog() -> None:
    """Mark older validated formal runs without changing their raw manifests."""
    if not RUNS_CSV.is_file():
        return
    with RUNS_CSV.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    fields = (
        "run_id", "suite_id", "suite_version", "device", "implementation_version",
        "variant", "test_signature", "completed_at_utc", "path", "status",
        "reason", "superseded_by"
    )
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        if row.get("status") not in {"passed", "superseded"}:
            continue
        if not row.get("path", "").endswith("run-manifest.json"):
            continue
        key = (row.get("suite_id", ""), row.get("device", ""), row.get("variant", ""))
        groups.setdefault(key, []).append(row)
    for group in groups.values():
        newest = max(
            group,
            key=lambda item: (
                int(item.get("suite_version", "v0").removeprefix("v") or 0),
                item.get("completed_at_utc", ""),
            ),
        )
        for row in group:
            if row is newest:
                row["status"] = "passed"
                row["superseded_by"] = ""
                if row.get("reason") == "newer validated suite/run in the same device series":
                    row["reason"] = ""
            else:
                row["status"] = "superseded"
                row["superseded_by"] = newest["path"]
                if not row.get("reason"):
                    row["reason"] = "newer validated suite/run in the same device series"
    with RUNS_CSV.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def record_failed_formal_run(
    *, suite: dict[str, Any], device: str, variant: str, output: Path,
    run_id: str, signature: str, sources: dict[str, str], command: list[str],
    started: datetime, returncode: int, environment: dict[str, str],
) -> Path:
    """Persist one failed paid/local attempt as diagnostic evidence."""
    finished = datetime.now(timezone.utc)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "measurement_release": suite.get("measurement_release", "unversioned"),
        "run_id": run_id,
        "suite_id": suite["suite_id"],
        "suite_version": suite["suite_version"],
        "variant": variant,
        "device": {
            "class": device,
            "backend": environment.get("VENTUS_BACKEND", "nvidia-hardware"),
            **({"cuda": "13.3.0"} if device != "ventus" else {}),
            **child_metadata(output),
        },
        "implementation": implementation(device, sources),
        "measurement": {
            "formal": True,
            "warmups": FORMAL_WARMUPS,
            "repeats": FORMAL_REPEATS,
            "pairing": suite.get("pairing", "same-kernel-consecutive"),
            "paid_attempts": 0 if device == "ventus" else FORMAL_PAID_ATTEMPTS,
            "started_at_utc": started.isoformat(),
            "duration_seconds": (finished - started).total_seconds(),
        },
        "sources": sources,
        "test_signature": signature,
        "command": command,
        "returncode": returncode,
        "status": "diagnostic",
        "failure_class": "suite_command_failed",
        "reason": "stopped immediately; no automatic retry or timeout extension",
        "completed_at_utc": finished.isoformat(),
    }
    path = output / "diagnostic-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_run_catalog(manifest, path)
    reconcile_run_catalog()
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--suite")
    parser.add_argument(
        "--suite-version",
        help="select an immutable historical suite version such as v5",
    )
    parser.add_argument("--device", choices=("h100", "b200", "ventus"))
    parser.add_argument("--variant")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--import-only", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument(
        "--register-existing",
        type=Path,
        help=(
            "validate and register a completed suite output after the local "
            "entrypoint was interrupted; never launches a benchmark"
        ),
    )
    parser.add_argument(
        "--repair-validation",
        type=Path,
        help="re-run local validation for an existing failed-validation run",
    )
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--reason", default="")
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="seed a new Ventus formal run from a verified partial checkpoint",
    )
    args = parser.parse_args()
    if not args.list and not args.repair_validation and (
        not args.suite or not args.device
    ):
        parser.error("--suite and --device are required unless --list is used")
    if args.repair_validation and any(
        (
            args.list, args.plan, args.import_only, args.formal,
            args.force_rerun, bool(args.register_existing),
        )
    ):
        parser.error("--repair-validation cannot be combined with run modes")
    if (
        sum((args.plan, args.import_only, args.formal, bool(args.register_existing))) != 1
        and not args.list
        and not args.repair_validation
    ):
        parser.error(
            "choose exactly one of --plan, --import-only, --formal, or "
            "--register-existing"
        )
    if args.force_rerun and not args.reason.strip():
        parser.error("--force-rerun requires --reason")
    if args.register_existing and (args.force_rerun or args.resume_from):
        parser.error(
            "--register-existing cannot be combined with rerun/resume options"
        )
    if args.resume_from and (not args.formal or args.device != "ventus"):
        parser.error("--resume-from is only valid for Ventus --formal runs")
    return args


def register_existing_output(
    *, output: Path, suite: dict[str, Any], device: str, variant: str,
    release: dict[str, Any],
) -> int:
    """Adopt a fully completed paid run without launching the suite again."""
    output = output.resolve()
    try:
        output.relative_to(DATA.resolve())
    except ValueError as error:
        raise SystemExit("--register-existing must be inside the data tree") from error
    if not output.is_dir():
        raise SystemExit(f"completed output does not exist: {output}")
    manifest_path = output / "run-manifest.json"
    if manifest_path.exists():
        raise SystemExit(f"completed output is already registered: {manifest_path}")
    child_manifest_path = output / "manifest.json"
    if not child_manifest_path.is_file():
        raise SystemExit("completed output has no suite manifest.json")
    child_manifest = json.loads(child_manifest_path.read_text(encoding="utf-8"))
    if child_manifest.get("measurement_release") != release["release"]:
        raise SystemExit("completed output belongs to a different measurement release")
    if child_manifest.get("variant") != variant:
        raise SystemExit("completed output variant does not match --variant")
    if child_manifest.get("gpu") != device:
        raise SystemExit("completed output GPU does not match --device")

    sources = source_hashes(suite, device)
    payload = signature_payload(suite, device, variant, sources)
    signature = test_signature(payload)
    if not output.name.endswith(f"__{signature[:12]}"):
        raise SystemExit("completed output path does not match current test signature")
    duplicate = existing_signatures().get(signature)
    if duplicate is not None:
        raise SystemExit(f"duplicate formal signature already exists at {duplicate}")

    case_count, matrix_hash, problems = validate_raw(output, FORMAL_REPEATS)
    problems.extend(validate_fresh_tensormap(output, suite, device))
    expected_cases = suite.get("expected_cases", {}).get(device)
    if isinstance(expected_cases, dict):
        expected_cases = expected_cases.get(variant)
    if expected_cases is not None and case_count != int(expected_cases):
        problems.append(
            f"formal case count={case_count}, expected={int(expected_cases)}"
        )
    if problems:
        raise SystemExit(
            "completed output failed local validation: " + "; ".join(problems[:12])
        )

    completed_at = str(
        child_manifest.get("completed_at_utc")
        or datetime.now(timezone.utc).isoformat()
    )
    command = format_command(suite, device, variant, output)
    manifest = {
        "schema_version": 1,
        "measurement_release": release["release"],
        "run_id": output.name,
        "suite_id": suite["suite_id"],
        "suite_version": suite["suite_version"],
        "variant": variant,
        "device": {
            "class": device,
            "backend": "nvidia-hardware",
            "cuda": "13.3.0",
            **child_metadata(output),
        },
        "implementation": implementation(device, sources),
        "measurement": {
            "formal": True,
            "warmups": FORMAL_WARMUPS,
            "repeats": FORMAL_REPEATS,
            "pairing": suite.get("pairing", "suite-defined-independent-A-B"),
            "paid_attempts": FORMAL_PAID_ATTEMPTS,
            "wait_method": suite.get("wait_method", "suite-defined"),
            "data_residency": suite.get("data_residency", "suite-defined"),
            "descriptor_residency": suite.get(
                "descriptor_residency", "suite-defined"
            ),
            "direction_definition": suite.get(
                "direction_definition", "suite-defined"
            ),
            "context_definition": suite.get("context_definition", "single-command"),
            "started_at_utc": None,
            "duration_seconds": None,
            "registration_note": (
                "GPU suite completed successfully; the local dispatcher was "
                "interrupted before registration, so only local validation "
                "and registration were recovered"
            ),
        },
        "sources": sources,
        "case_matrix_sha256": matrix_hash,
        "test_signature": signature,
        "command": command,
        "case_count": case_count,
        "resource_limited_case_count": 0,
        "validation_problems": [],
        "status": "passed",
        "completed_at_utc": completed_at,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    append_run_catalog(manifest, manifest_path)
    reconcile_run_catalog()
    print(f"validated and registered existing output: {manifest_path}")
    return 0


def repair_validation(
    path: Path, suites: dict[str, dict[str, Any]]
) -> int:
    output = path.resolve()
    try:
        output.relative_to(DATA.resolve())
    except ValueError as error:
        raise SystemExit("--repair-validation must be inside the data tree") from error
    manifest_path = output / "run-manifest.json"
    if not manifest_path.is_file():
        raise SystemExit("repair target has no run-manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "failed_validation":
        raise SystemExit("repair target is not a failed-validation run")
    suite = suites.get(str(manifest.get("suite_id")))
    if suite is None or suite["suite_version"] != manifest.get("suite_version"):
        raise SystemExit("repair target suite/version is unavailable")
    device = str(manifest["device"]["class"])
    variant = str(manifest["variant"])
    sources = source_hashes(suite, device)
    if sources != manifest.get("sources"):
        raise SystemExit("suite source hashes changed; existing data cannot be repaired")
    case_count, matrix_hash, problems = validate_raw(output, FORMAL_REPEATS)
    problems.extend(validate_fresh_tensormap(output, suite, device))
    expected = suite.get("expected_cases", {}).get(device)
    if isinstance(expected, dict):
        expected = expected.get(variant)
    if expected is not None and case_count != int(expected):
        problems.append(f"formal case count={case_count}, expected={int(expected)}")
    expected_limited = suite.get("resource_limited_cases", {}).get(device, 0)
    if isinstance(expected_limited, dict):
        expected_limited = expected_limited.get(variant, 0)
    limited_path = output / "resource_limits.csv"
    actual_limited = 0
    if limited_path.is_file():
        with limited_path.open(newline="", encoding="utf-8") as source:
            actual_limited = sum(1 for _ in csv.DictReader(source))
    if actual_limited != int(expected_limited):
        problems.append(
            f"resource-limited case count={actual_limited}, "
            f"expected={int(expected_limited)}"
        )
    manifest["case_count"] = case_count
    manifest["case_matrix_sha256"] = matrix_hash
    manifest["resource_limited_case_count"] = actual_limited
    manifest["validation_problems"] = problems
    manifest["status"] = "passed" if not problems else "failed_validation"
    manifest["validation_repair"] = {
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "case identity validator includes direction/method dimensions",
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    append_run_catalog(manifest, manifest_path)
    if problems:
        raise SystemExit("validation repair failed: " + "; ".join(problems[:12]))
    print(f"repaired and registered: {manifest_path}")
    return 0


def main() -> int:
    args = parse_args()
    release = current_release()
    suites = discover_suites()
    write_suite_catalog(suites)
    reconcile_run_catalog()
    if args.repair_validation:
        return repair_validation(args.repair_validation, suites)
    if args.list:
        for suite in sorted(suites.values(), key=lambda item: str(item["suite_id"])):
            pinned = release["suites"].get(str(suite["suite_id"]))
            if pinned == suite["suite_version"]:
                print(f"{suite['suite_id']} {suite['suite_version']} release={release['release']} variants={','.join(suite['variants'])}")
        return 0
    if args.suite not in suites:
        raise SystemExit(f"unknown suite: {args.suite}")
    suite = suites[args.suite]
    if args.suite_version:
        matches = [
            load_suite(path)
            for path in SUITES.rglob("suite.yaml")
            if path.parent.name == args.suite_version
            and load_suite(path).get("suite_id") == args.suite
        ]
        if len(matches) != 1:
            raise SystemExit(
                f"cannot resolve {args.suite} version {args.suite_version}: "
                f"found {len(matches)} matches"
            )
        suite = matches[0]
        disabled_reason = disabled_suites().get(
            f"{suite['suite_id']}@{suite['suite_version']}"
        )
        if disabled_reason:
            raise SystemExit(
                f"{suite['suite_id']} {suite['suite_version']} is disabled: "
                f"{disabled_reason}"
            )
    pinned_version = release["suites"].get(str(suite["suite_id"]))
    if pinned_version != suite["suite_version"]:
        raise SystemExit(
            f"{suite['suite_id']} {suite['suite_version']} is not part of "
            f"current release {release['release']} (pinned={pinned_version})"
        )
    if suite.get("measurement_release") != release["release"]:
        raise SystemExit(
            f"{suite['suite_id']}: missing measurement_release="
            f"{release['release']}"
        )
    formal = suite.get("formal", {})
    expected_formal = {
        "warmups": FORMAL_WARMUPS,
        "repeats": FORMAL_REPEATS,
        "paid_attempts": FORMAL_PAID_ATTEMPTS,
    }
    if any(int(formal.get(key, -1)) != value for key, value in expected_formal.items()):
        raise SystemExit(f"{suite['suite_id']}: suite formal contract does not match {expected_formal}")
    variants = [str(value) for value in suite["variants"]]
    variant = args.variant or (variants[0] if len(variants) == 1 else "")
    if variant not in variants:
        raise SystemExit(f"--variant must be one of {variants}")
    if args.register_existing:
        return register_existing_output(
            output=args.register_existing,
            suite=suite,
            device=args.device,
            variant=variant,
            release=release,
        )
    sources = source_hashes(suite, args.device)
    payload = signature_payload(suite, args.device, variant, sources)
    signature = test_signature(payload)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}__{suite['suite_version']}__{signature[:12]}"
    output = output_path(suite, args.device, variant, run_id)
    command = format_command(suite, args.device, variant, output)
    if args.import_only:
        if args.device == "ventus":
            command = ["make", "-C", str(suite["directory"]), "validate"]
        else:
            command.append("--import-only")
    print(json.dumps({"signature": signature, "output": str(output), "command": command}, indent=2))
    if args.plan:
        return 0
    duplicate = existing_signatures().get(signature) if args.formal else None
    if duplicate is not None and not args.force_rerun:
        raise SystemExit(f"duplicate formal signature already exists at {duplicate}")
    if args.device == "ventus" and args.formal:
        subprocess.run(["make", "-C", str(suite["directory"]), "all"], check=True, cwd=ROOT)
    resume_from: Path | None = None
    if args.resume_from:
        resume_from = args.resume_from.resolve()
        try:
            resume_from.relative_to(DATA.resolve())
        except ValueError as error:
            raise SystemExit("--resume-from must be inside the benchmark data tree") from error
        comprehensive = any(
            token.endswith("run_ventus_comprehensive_short.py") for token in command
        )
        multi_context = any(
            token.endswith("run_ventus_multi_context_short.py") for token in command
        )
        if not comprehensive and not multi_context:
            raise SystemExit("selected suite runner does not support checkpoint resume")
        output.mkdir(parents=True, exist_ok=True)
        if comprehensive:
            partial_source = resume_from / "partial.csv"
            outcomes_source = resume_from / "outcomes.jsonl"
            if not partial_source.is_file() or not outcomes_source.is_file():
                raise SystemExit(
                    "--resume-from is missing partial.csv or outcomes.jsonl"
                )
            shutil.copy2(partial_source, output / partial_source.name)
            shutil.copy2(outcomes_source, output / outcomes_source.name)
        else:
            if not (resume_from / "raw.csv").is_file() or not (
                resume_from / "outcomes.json"
            ).is_file():
                raise SystemExit(
                    "--resume-from is missing raw.csv or outcomes.json"
                )
            for name in ("raw.csv", "outcomes.json", "resource_limits.csv"):
                source = resume_from / name
                if source.is_file():
                    shutil.copy2(source, output / name)
        command.append("--resume")
    started = datetime.now(timezone.utc)
    environment = {**os.environ, "VENTUS_ENV_PATH": str(ROOT)}
    if args.device == "ventus":
        install = ROOT / "install"
        environment["VENTUS_INSTALL_PREFIX"] = str(install)
        environment["PATH"] = (
            f"{install / 'bin'}:{environment.get('PATH', '')}"
        ).rstrip(":")
        environment["LD_LIBRARY_PATH"] = (
            f"{install / 'lib'}:{environment.get('LD_LIBRARY_PATH', '')}"
        ).rstrip(":")
        environment["POCL_DEVICES"] = "ventus"
        environment["OCL_ICD_VENDORS"] = str(install / "lib/libpocl.so")
        environment["POCL_ENABLE_UNINIT"] = "1"
        environment["VENTUS_BACKEND"] = os.environ.get("VENTUS_BACKEND", "rtlsim")
    completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    if completed.returncode != 0:
        if args.formal:
            diagnostic = record_failed_formal_run(
                suite=suite, device=args.device, variant=variant, output=output,
                run_id=run_id, signature=signature, sources=sources,
                command=command, started=started, returncode=completed.returncode,
                environment=environment,
            )
            print(f"diagnostic registered: {diagnostic}")
        raise SystemExit(f"suite command failed with returncode={completed.returncode}")
    if args.import_only:
        return 0
    finished = datetime.now(timezone.utc)
    case_count, matrix_hash, problems = validate_raw(output, FORMAL_REPEATS)
    problems.extend(validate_fresh_tensormap(output, suite, args.device))
    expected_cases = suite.get("expected_cases", {}).get(args.device)
    if isinstance(expected_cases, dict):
        expected_cases = expected_cases.get(variant)
    if expected_cases is not None and case_count != int(expected_cases):
        problems.append(
            f"formal case count={case_count}, expected={int(expected_cases)}"
        )
    expected_limited_value = suite.get("resource_limited_cases", {}).get(
        args.device, 0
    )
    if isinstance(expected_limited_value, dict):
        expected_limited_value = expected_limited_value.get(variant, 0)
    expected_limited = int(expected_limited_value)
    limited_path = output / "resource_limits.csv"
    actual_limited = 0
    if limited_path.is_file():
        with limited_path.open(newline="", encoding="utf-8") as source:
            actual_limited = sum(1 for _ in csv.DictReader(source))
    if actual_limited != expected_limited:
        problems.append(
            f"resource-limited case count={actual_limited}, expected={expected_limited}"
        )
    manifest = {
        "schema_version": 1,
        "measurement_release": release["release"],
        "run_id": run_id,
        "suite_id": suite["suite_id"],
        "suite_version": suite["suite_version"],
        "variant": variant,
        "device": {
            "class": args.device,
            "backend": environment.get("VENTUS_BACKEND", "nvidia-hardware"),
            **({"cuda": "13.3.0"} if args.device != "ventus" else {}),
            **child_metadata(output),
        },
        "implementation": implementation(args.device, sources),
        "measurement": {
            "formal": True,
            "warmups": FORMAL_WARMUPS,
            "repeats": FORMAL_REPEATS,
            "pairing": suite.get("pairing", "suite-defined-independent-A-B"),
            "paid_attempts": 0 if args.device == "ventus" else FORMAL_PAID_ATTEMPTS,
            "wait_method": suite.get("wait_method", "suite-defined"),
            "data_residency": suite.get("data_residency", "suite-defined"),
            "descriptor_residency": suite.get("descriptor_residency", "suite-defined"),
            "direction_definition": suite.get("direction_definition", "suite-defined"),
            "context_definition": suite.get("context_definition", "single-command"),
            "started_at_utc": started.isoformat(),
            "duration_seconds": (finished - started).total_seconds(),
            "resumed_from": (
                resume_from.relative_to(HERE).as_posix() if resume_from else None
            ),
        },
        "sources": sources,
        "case_matrix_sha256": matrix_hash,
        "test_signature": signature,
        "command": command,
        "case_count": case_count,
        "resource_limited_case_count": actual_limited,
        "validation_problems": problems,
        "status": "passed" if not problems else "failed_validation",
        "completed_at_utc": finished.isoformat(),
    }
    if args.force_rerun:
        manifest["forced_rerun_reason"] = args.reason
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_run_catalog(manifest, manifest_path)
    if problems:
        raise SystemExit("formal validation failed: " + "; ".join(problems[:12]))
    print(f"registered: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
