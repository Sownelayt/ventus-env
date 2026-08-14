#!/usr/bin/env python3
"""Inventory, migrate, and verify the TMA cycle-result repository.

Background:
  The original results directory mixed formal runs, development probes,
  reports, CUDA devices, and many Ventus revisions.  This command creates a
  content-addressed migration ledger before any move, then places each result
  in the device/version/scope/project hierarchy documented by README.md.

Flow:
  ``inventory`` hashes every legacy file and writes a deterministic proposed
  mapping. ``migrate`` refuses to run unless the current tree matches that
  inventory, moves each top-level result directory exactly once, and writes a
  post-migration inventory. ``verify`` proves that every old file has exactly
  one new path with the same SHA-256 and size.

Usage:
  python3 benchmarks/tma-cycle-compare/manage_repository.py inventory
  python3 benchmarks/tma-cycle-compare/manage_repository.py migrate
  python3 benchmarks/tma-cycle-compare/manage_repository.py verify

Maintenance:
  Add classification rules before migrating a newly introduced legacy naming
  family.  Never silently classify an unknown directory as formal data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent
LEGACY_RESULTS = HERE / "results"
DATA = HERE / "data"
REPORTS = HERE / "reports"
ARCHIVE = HERE / "archive"
CATALOG = HERE / "catalog"
BEFORE = CATALOG / "file_inventory_before.csv"
AFTER = CATALOG / "file_inventory_after.csv"
MAP = CATALOG / "legacy_path_map.csv"
SUMMARY = CATALOG / "migration_summary.json"
RUNS = CATALOG / "runs.csv"
CHECKPOINTS = CATALOG / "checkpoints.csv"

FILE_FIELDS = (
    "legacy_dataset",
    "old_path",
    "new_path",
    "size_bytes",
    "sha256",
)
MAP_FIELDS = (
    "legacy_dataset",
    "old_path",
    "new_path",
    "device",
    "implementation_version",
    "scope",
    "project",
    "variant",
    "status",
    "provenance",
    "superseded_by",
    "file_count",
    "size_bytes",
    "tree_sha256",
)

FORMAL_NAMES = {
    "comprehensive_cuda_h100",
    "comprehensive_cuda_b200",
    "comprehensive_ventus_v3_10",
    "tensor_2d_length_cuda_h100",
    "tensor_2d_length_cuda_b200",
    "tensor_2d_length_ventus_v3_10_r1",
    "cuda_h100_descriptor_latency_waitgroup",
    "features_cuda_h100_interleave_fixed",
    "features_cuda_b200_full_sharded_v2",
    "uarch_cuda_h100",
    "uarch_cuda_b200_full",
    "cuda_h100_command_overlap",
    "cuda_h100_command_overlap_issue_timeline",
    "features_ventus_capacity_v2",
    "features_ventus_descriptor",
    "features_ventus_interleave_fixed",
    "uarch_ventus",
    "contention_ventus_core_v2",
    "contention_ventus_pressure",
    "features_ventus_v3_10_capacity",
}

REPORT_NAMES = {
    "comprehensive_compare_v3_10",
    "comprehensive_report",
    "comprehensive_report_cuda",
    "tensor_2d_length_compare_v3_10",
}

SUPERSEDED = {
    "features_cuda_h100_safe": "features_cuda_h100_interleave_fixed",
    "features_cuda_b200_full": "features_cuda_b200_full_sharded_v2",
    "features_cuda_b200_full_sharded": "features_cuda_b200_full_sharded_v2",
    "features_ventus_capacity": "features_ventus_capacity_v2",
    "features_ventus_v3_3_capacity": "features_ventus_v3_3_capacity_final",
    "features_ventus_v3_5_capacity": "features_ventus_v3_5_capacity_final",
    "features_ventus_v3_5_capacity_release": "features_ventus_v3_5_capacity_final",
}

DIAGNOSTIC_TOKENS = (
    "candidate",
    "diagnose",
    "failure",
    "invalid",
    "pre_",
    "probe",
    "retry",
    "smoke",
    "stability",
    "timeout",
    "collision",
    "recheck",
)


@dataclass(frozen=True)
class Classification:
    device: str
    implementation_version: str
    scope: str
    project: str
    variant: str
    status: str
    provenance: str
    superseded_by: str = ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_digest(files: list[tuple[str, int, str]]) -> str:
    digest = hashlib.sha256()
    for relative, size, file_hash in sorted(files):
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(str(size).encode())
        digest.update(b"\0")
        digest.update(file_hash.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def read_manifest(directory: Path) -> dict[str, object]:
    for name in ("manifest.json", "run_manifest.json"):
        path = directory / name
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return {}
    return {}


def classify(name: str, directory: Path) -> Classification:
    lower = name.lower()
    manifest = read_manifest(directory)

    if "h100" in lower:
        device = "h100"
    elif "b200" in lower:
        device = "b200"
    elif "ventus" in lower:
        device = "ventus"
    elif name in REPORT_NAMES or "compare" in lower or "report" in lower:
        device = "cross-device"
    else:
        device = str(manifest.get("gpu") or manifest.get("platform") or "unknown")
        device = device.lower().replace(" ", "-")

    version_match = re.search(r"v(\d+)_(\d+)", lower)
    if device == "ventus":
        if version_match:
            version = f"V{version_match.group(1)}.{version_match.group(2)}"
        else:
            version = "V3.0"
    elif device == "h100":
        version = "cuda-13.3-sm90"
    elif device == "b200":
        version = "cuda-13.3-sm100"
    else:
        version = "mixed"

    if "tensor_2d_length" in lower:
        project = "tensor_2d_capacity"
        scope = "single_instruction"
    elif "descriptor_latency" in lower:
        project = "tensormap_decode"
        scope = "single_instruction"
    elif "command_overlap" in lower:
        project = "same_direction"
        scope = "multi_context"
    elif "contention" in lower:
        project = "mixed_direction"
        scope = "multi_context"
    elif "comprehensive" in lower:
        project = "common_matrix"
        scope = "single_instruction"
    elif "uarch" in lower:
        project = "descriptor_reuse"
        scope = "single_instruction"
    elif "interleave" in lower:
        project = "interleave"
        scope = "single_instruction"
    elif "capacity" in lower:
        project = "tensor_capacity"
        scope = "single_instruction"
    elif "descriptor" in lower:
        project = "descriptor_features"
        scope = "single_instruction"
    elif "dtype" in lower:
        project = "dtype"
        scope = "single_instruction"
    elif "layout" in lower:
        project = "layout"
        scope = "single_instruction"
    elif "oob" in lower:
        project = "oob"
        scope = "single_instruction"
    elif "subbox" in lower:
        project = "subbox"
        scope = "single_instruction"
    elif "reduce" in lower:
        project = "reduce"
        scope = "excluded"
    else:
        project = "cycle_capacity"
        scope = "single_instruction"

    if "g2s" in lower and "s2g" not in lower:
        variant = "g2s"
    elif "s2g" in lower and "g2s" not in lower:
        variant = "s2g"
    else:
        variant = "all_directions"

    if name in REPORT_NAMES or device == "cross-device":
        status = "report"
    elif name == "features_cuda_h100":
        status = "known_xid"
    elif name in SUPERSEDED:
        status = "superseded"
    elif name in FORMAL_NAMES:
        status = "formal"
    elif any(token in lower for token in DIAGNOSTIC_TOKENS):
        status = "diagnostic"
    else:
        status = "historical"

    source_keys = ("sha256", "source_sha256", "gvm_sha256")
    if any(key in manifest for key in source_keys):
        provenance = "verified_hash"
    elif any(directory.glob("*.sha256")):
        provenance = "hash_only"
    elif manifest:
        provenance = "manifest_without_source_hash"
    else:
        provenance = "unverified"

    return Classification(
        device=device,
        implementation_version=version,
        scope=scope,
        project=project,
        variant=variant,
        status=status,
        provenance=provenance,
        superseded_by=SUPERSEDED.get(name, ""),
    )


def destination(name: str, classification: Classification) -> Path:
    if classification.status == "report":
        return REPORTS / "legacy" / classification.project / name
    if classification.status in {"diagnostic", "known_xid", "superseded"}:
        return (
            ARCHIVE
            / "results"
            / classification.device
            / classification.implementation_version
            / classification.project
            / name
        )
    return (
        DATA
        / classification.device
        / classification.implementation_version
        / classification.scope
        / classification.project
        / classification.variant
        / f"legacy--{name}"
    )


def top_level_datasets() -> list[Path]:
    if not LEGACY_RESULTS.is_dir():
        return []
    return sorted(path for path in LEGACY_RESULTS.iterdir() if path.is_dir())


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def inventory() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    file_rows: list[dict[str, object]] = []
    map_rows: list[dict[str, object]] = []
    for directory in top_level_datasets():
        name = directory.name
        classification = classify(name, directory)
        target = destination(name, classification)
        if target.exists():
            raise SystemExit(f"destination already exists: {target}")
        files: list[tuple[str, int, str]] = []
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            relative = path.relative_to(directory).as_posix()
            size = path.stat().st_size
            digest = sha256_file(path)
            files.append((relative, size, digest))
            file_rows.append(
                {
                    "legacy_dataset": name,
                    "old_path": path.relative_to(HERE).as_posix(),
                    "new_path": (target / relative).relative_to(HERE).as_posix(),
                    "size_bytes": size,
                    "sha256": digest,
                }
            )
        map_rows.append(
            {
                "legacy_dataset": name,
                "old_path": directory.relative_to(HERE).as_posix(),
                "new_path": target.relative_to(HERE).as_posix(),
                **asdict(classification),
                "file_count": len(files),
                "size_bytes": sum(item[1] for item in files),
                "tree_sha256": tree_digest(files),
            }
        )
    write_csv(BEFORE, FILE_FIELDS, file_rows)
    write_csv(MAP, MAP_FIELDS, map_rows)
    print(f"datasets={len(map_rows)} files={len(file_rows)}")
    print(f"inventory={BEFORE}")
    print(f"mapping={MAP}")
    return file_rows, map_rows


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"missing required catalog: {path}")
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def verify_legacy_matches_inventory(rows: list[dict[str, str]]) -> None:
    for row in rows:
        path = HERE / row["old_path"]
        if not path.is_file():
            raise SystemExit(f"legacy file disappeared before migration: {path}")
        if path.stat().st_size != int(row["size_bytes"]) or sha256_file(path) != row["sha256"]:
            raise SystemExit(f"legacy file changed after inventory: {path}")


def migrate() -> None:
    file_rows = load_csv(BEFORE)
    map_rows = load_csv(MAP)
    if len(map_rows) != len(top_level_datasets()):
        raise SystemExit("legacy dataset count changed after inventory")
    verify_legacy_matches_inventory(file_rows)
    for row in map_rows:
        source = HERE / row["old_path"]
        target = HERE / row["new_path"]
        if not source.is_dir():
            raise SystemExit(f"missing legacy dataset: {source}")
        if target.exists():
            raise SystemExit(f"refusing to overwrite: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
    LEGACY_RESULTS.mkdir(parents=True, exist_ok=True)
    (LEGACY_RESULTS / "README.md").write_text(
        "# Legacy path compatibility\n\n"
        "Historical result directories were moved into `../data`, `../reports`, "
        "or `../archive`. See `../catalog/legacy_path_map.csv`.\n",
        encoding="utf-8",
    )
    verify()


def adopt_loose_files() -> None:
    """Adopt legacy files that lived directly under results/."""
    before = load_csv(BEFORE)
    mapping = load_csv(MAP)
    known_old_paths = {row["old_path"] for row in before}
    loose = sorted(
        path
        for path in LEGACY_RESULTS.iterdir()
        if path.is_file() and path.name != "README.md"
    )
    for path in loose:
        old_path = path.relative_to(HERE).as_posix()
        if old_path in known_old_paths:
            continue
        target = REPORTS / "legacy" / "tensor_capacity" / path.name
        if target.exists():
            raise SystemExit(f"refusing to overwrite loose result: {target}")
        digest = sha256_file(path)
        size = path.stat().st_size
        before.append(
            {
                "legacy_dataset": "__loose_files__",
                "old_path": old_path,
                "new_path": target.relative_to(HERE).as_posix(),
                "size_bytes": str(size),
                "sha256": digest,
            }
        )
        mapping.append(
            {
                "legacy_dataset": f"loose--{path.stem}",
                "old_path": old_path,
                "new_path": target.relative_to(HERE).as_posix(),
                "device": "ventus",
                "implementation_version": "V3.7-to-V3.8",
                "scope": "single_instruction",
                "project": "tensor_capacity",
                "variant": "all_directions",
                "status": "report",
                "provenance": "unverified",
                "superseded_by": "",
                "file_count": "1",
                "size_bytes": str(size),
                "tree_sha256": tree_digest([(path.name, size, digest)]),
            }
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(target))
    write_csv(BEFORE, FILE_FIELDS, before)
    write_csv(MAP, MAP_FIELDS, mapping)
    verify()


def verify() -> None:
    before = load_csv(BEFORE)
    mapping = load_csv(MAP)
    after_rows: list[dict[str, object]] = []
    errors: list[str] = []
    for row in before:
        path = HERE / row["new_path"]
        if not path.is_file():
            errors.append(f"missing:{row['new_path']}")
            continue
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != int(row["size_bytes"]) or digest != row["sha256"]:
            errors.append(f"mismatch:{row['new_path']}")
        after_rows.append(
            {
                "legacy_dataset": row["legacy_dataset"],
                "old_path": row["old_path"],
                "new_path": row["new_path"],
                "size_bytes": size,
                "sha256": digest,
            }
        )
    write_csv(AFTER, FILE_FIELDS, after_rows)
    directory_entries = [
        row for row in mapping if not row["legacy_dataset"].startswith("loose--")
    ]
    loose_entries = [
        row for row in mapping if row["legacy_dataset"].startswith("loose--")
    ]
    summary = {
        "schema_version": 1,
        "legacy_directory_count": len(directory_entries),
        "legacy_loose_file_count": len(loose_entries),
        "migration_entry_count": len(mapping),
        "legacy_file_count": len(before),
        "verified_file_count": len(after_rows),
        "verified_bytes": sum(int(row["size_bytes"]) for row in after_rows),
        "errors": errors,
        "passed": not errors and len(after_rows) == len(before),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not summary["passed"]:
        raise SystemExit(f"migration verification failed: {errors[:8]}")
    print(json.dumps(summary, indent=2, sort_keys=True))


def catalog_legacy_runs() -> None:
    """Seed runs.csv with one traceability row per migrated legacy dataset."""
    mapping = load_csv(MAP)
    fields = (
        "run_id", "suite_id", "suite_version", "device", "implementation_version",
        "variant", "test_signature", "completed_at_utc", "path", "status", "reason",
        "superseded_by"
    )
    existing: list[dict[str, str]] = []
    if RUNS.is_file():
        with RUNS.open(newline="", encoding="utf-8") as source:
            existing = list(csv.DictReader(source))
    retained = [row for row in existing if not row.get("run_id", "").startswith("legacy--")]
    for row in mapping:
        project = row["project"]
        scope = row["scope"] if row["scope"] != "excluded" else "single_instruction"
        retained.append(
            {
                "run_id": f"legacy--{row['legacy_dataset']}",
                "suite_id": f"{scope}.{project}",
                "suite_version": "legacy-unversioned",
                "device": row["device"],
                "implementation_version": row["implementation_version"],
                "variant": row["variant"],
                "test_signature": row["tree_sha256"],
                "completed_at_utc": "unknown",
                "path": row["new_path"],
                "status": row["status"],
                "reason": ";".join(
                    value for value in (
                        row["provenance"],
                        f"superseded_by={row['superseded_by']}" if row["superseded_by"] else "",
                    ) if value
                ),
                "superseded_by": row["superseded_by"],
            }
        )
    write_csv(RUNS, fields, sorted(retained, key=lambda item: item["run_id"]))
    print(f"legacy_runs={len(mapping)} catalog={RUNS}")


def reconcile_run_catalog() -> None:
    """Rebuild modern manifest rows and mark older validated series entries."""
    fields = (
        "run_id", "suite_id", "suite_version", "device", "implementation_version",
        "variant", "test_signature", "completed_at_utc", "path", "status", "reason",
        "superseded_by"
    )
    existing = load_csv(RUNS) if RUNS.is_file() else []
    rows = [row for row in existing if row.get("run_id", "").startswith("legacy--")]
    manifests = [
        *sorted(DATA.rglob("run-manifest.json")),
        *sorted(DATA.rglob("diagnostic-manifest.json")),
        *sorted(ARCHIVE.rglob("diagnostic-manifest.json")),
    ]
    modern: list[dict[str, str]] = []
    for path in manifests:
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(f"invalid modern run manifest {path}: {error}") from error
        modern.append(
            {
                "run_id": str(manifest.get("run_id", "")),
                "suite_id": str(manifest.get("suite_id", "")),
                "suite_version": str(manifest.get("suite_version", "")),
                "device": str(manifest.get("device", {}).get("class", "")),
                "implementation_version": str(manifest.get("implementation", {}).get("version", "")),
                "variant": str(manifest.get("variant", "")),
                "test_signature": str(manifest.get("test_signature", "")),
                "completed_at_utc": str(manifest.get("completed_at_utc", "")),
                "path": path.relative_to(HERE).as_posix(),
                "status": str(manifest.get("status", "")),
                "reason": str(manifest.get("reason", manifest.get("forced_rerun_reason", ""))),
                "superseded_by": "",
            }
        )
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in modern:
        if row["status"] != "passed":
            continue
        groups.setdefault((row["suite_id"], row["device"], row["variant"]), []).append(row)
    for group in groups.values():
        newest = max(
            group,
            key=lambda row: (
                int(row["suite_version"].removeprefix("v") or 0),
                row["completed_at_utc"],
            ),
        )
        for row in group:
            if row is newest:
                continue
            row["status"] = "superseded"
            row["reason"] = "newer validated suite/run in the same device series"
            row["superseded_by"] = newest["path"]
    rows.extend(modern)
    write_csv(RUNS, fields, sorted(rows, key=lambda item: item["run_id"]))
    newest_by_series = {
        key: max(group, key=lambda row: row["completed_at_utc"])["path"]
        for key, group in groups.items()
    }
    checkpoint_fields = (
        "run_id", "suite_id", "device", "implementation_version", "variant",
        "path", "status", "superseded_by", "file_count", "size_bytes", "tree_sha256"
    )
    checkpoint_rows: list[dict[str, object]] = []
    for directory in sorted(
        path for path in DATA.rglob("202*__v*__*") if path.is_dir()
    ):
        if (directory / "run-manifest.json").is_file() or (
            directory / "diagnostic-manifest.json"
        ).is_file():
            continue
        relative = directory.relative_to(DATA)
        if len(relative.parts) < 6:
            continue
        device, version, scope, project, variant = relative.parts[:5]
        suite_id = f"{scope}.{project}"
        files = [path for path in directory.rglob("*") if path.is_file()]
        file_records = [
            (path.relative_to(directory).as_posix(), path.stat().st_size, sha256_file(path))
            for path in files
        ]
        checkpoint_rows.append(
            {
                "run_id": directory.name,
                "suite_id": suite_id,
                "device": device,
                "implementation_version": version,
                "variant": variant,
                "path": directory.relative_to(HERE).as_posix(),
                "status": "diagnostic_checkpoint",
                "superseded_by": newest_by_series.get((suite_id, device, variant), ""),
                "file_count": len(files),
                "size_bytes": sum(record[1] for record in file_records),
                "tree_sha256": tree_digest(file_records),
            }
        )
    write_csv(CHECKPOINTS, checkpoint_fields, checkpoint_rows)
    print(f"runs={len(rows)} modern={len(modern)} catalog={RUNS}")
    print(f"diagnostic_checkpoints={len(checkpoint_rows)} catalog={CHECKPOINTS}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("inventory", "migrate", "adopt-loose", "verify", "catalog-runs", "reconcile-runs")
    )
    return parser.parse_args()


def main() -> int:
    action = parse_args().action
    if action == "inventory":
        inventory()
    elif action == "migrate":
        migrate()
    elif action == "adopt-loose":
        adopt_loose_files()
    elif action == "catalog-runs":
        catalog_legacy_runs()
    elif action == "reconcile-runs":
        reconcile_run_catalog()
    else:
        verify()
    return 0


if __name__ == "__main__":
    sys.exit(main())
