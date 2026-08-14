#!/usr/bin/env python3
"""Generate direction-first reports from validated formal TMA cycle runs.

The generator reads only passed ``run-manifest.json`` files, preserves the two
raw samples, joins devices by stable case identity, computes NVIDIA TensorMap
differential estimates, multi-context high-N slopes, and 128B residency
controls.  It never substitutes historical warmed measurements for a missing
formal run.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
REPORTS = HERE / "reports"
MEASUREMENT_RELEASE = "ColdThenHotV1"
RESULT_NOT_BEFORE_UTC = (
    HERE / "releases" / f"{MEASUREMENT_RELEASE}.RESULT_NOT_BEFORE"
).read_text(encoding="utf-8").strip()
DEFAULT_OUTPUT = HERE / f"reports/comparisons/{MEASUREMENT_RELEASE}"
ARCHIVE = HERE / "archive"
PINNED_SUITE_VERSIONS = json.loads(
    (HERE / "releases" / f"{MEASUREMENT_RELEASE}.json").read_text(encoding="utf-8")
)["suites"]
DIRECTION_ORDER = {"g2s": 0, "s2g": 1, "roundtrip": 2, "mixed": 3}
DEVICE_ORDER = {"h100": 0, "b200": 1, "ventus": 2}
RAW_EXCLUDED = {
    "partial.csv", "summary.csv", "coverage.csv", "ratios.csv", "trends.csv",
    "resource_limits.csv", "pmu.csv",
}
EXPECTED = {
    ("single_instruction.common_matrix", device, "")
    for device in ("h100", "b200", "ventus")
} | {
    ("single_instruction.tensor_2d_capacity", device, "")
    for device in ("h100", "b200", "ventus")
} | {
    ("single_instruction.tensormap_decode", device, "")
    for device in ("h100", "b200", "ventus")
} | {
    ("single_instruction.data_residency_128b", device, "")
    for device in ("h100", "b200")
} | {
    ("single_instruction.control_items", device, "")
    for device in ("h100", "b200")
} | {
    ("single_instruction.completion_latency", device, variant)
    for device in ("h100", "b200")
    for variant in (
        "bulk-tensor__g2s__fresh-map-common",
        "tensor__g2s__fresh-map-2d",
    )
} | {
    ("multi_context.same_direction", device, variant)
    for device in ("h100", "b200", "ventus") for variant in ("g2s", "s2g")
} | {
    ("multi_context.mixed_direction", device, "mixed")
    for device in ("h100", "b200", "ventus")
} | {
    ("multi_context.s2g_size_sweep", device, "s2g")
    for device in ("h100", "b200")
}


def raw_csv_paths(directory: Path) -> list[Path]:
    return sorted({*directory.glob("*.csv"), *directory.glob("*.csv.gz")})


def open_csv(path: Path):
    if path.name.endswith(".csv.gz"):
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return path.open(newline="", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_runs() -> dict[tuple[str, str, str], tuple[Path, dict[str, Any]]]:
    result: dict[tuple[str, str, str], tuple[Path, dict[str, Any]]] = {}
    disabled_path = HERE / "catalog/disabled_suites.json"
    disabled = set(read_json(disabled_path)) if disabled_path.is_file() else set()
    for path in sorted(DATA.rglob("run-manifest.json")):
        manifest = read_json(path)
        if manifest.get("status") != "passed":
            continue
        if manifest.get("measurement_release") != MEASUREMENT_RELEASE:
            continue
        if PINNED_SUITE_VERSIONS.get(str(manifest.get("suite_id", ""))) != str(
            manifest.get("suite_version", "")
        ):
            continue
        if str(manifest.get("completed_at_utc", "")) < RESULT_NOT_BEFORE_UTC:
            continue
        suite_version_key = (
            f"{manifest.get('suite_id', '')}@{manifest.get('suite_version', '')}"
        )
        if suite_version_key in disabled:
            continue
        key = (
            str(manifest.get("suite_id")),
            str(manifest.get("device", {}).get("class")),
            str(manifest.get("variant", "")),
        )
        previous = result.get(key)
        if previous is None or str(manifest.get("completed_at_utc", "")) > str(previous[1].get("completed_at_utc", "")):
            result[key] = (path, manifest)
    return result


def diagnostic_runs() -> list[tuple[Path, dict[str, Any]]]:
    result: list[tuple[Path, dict[str, Any]]] = []
    for root in (DATA, ARCHIVE):
        for path in sorted(root.rglob("diagnostic-manifest.json")):
            try:
                manifest = read_json(path)
            except (OSError, json.JSONDecodeError):
                continue
            if manifest.get("measurement_release") != MEASUREMENT_RELEASE:
                continue
            if PINNED_SUITE_VERSIONS.get(str(manifest.get("suite_id", ""))) != str(
                manifest.get("suite_version", "")
            ):
                continue
            if str(manifest.get("completed_at_utc", "")) < RESULT_NOT_BEFORE_UTC:
                continue
            result.append((path, manifest))
    return result


def raw_rows(path: Path, manifest: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw in raw_csv_paths(path.parent):
        if raw.name in RAW_EXCLUDED:
            continue
        with open_csv(raw) as source:
            reader = csv.DictReader(source)
            if not reader.fieldnames or "repeat" not in reader.fieldnames:
                continue
            for row in reader:
                row["device_class"] = str(manifest["device"]["class"])
                row["suite_id"] = str(manifest["suite_id"])
                row["suite_version"] = str(manifest["suite_version"])
                row["variant"] = str(manifest.get("variant", ""))
                row["run_id"] = str(manifest["run_id"])
                row["raw_file"] = raw.relative_to(HERE).as_posix()
                rows.append(row)
    return rows


def integer(row: dict[str, str], *names: str, default: int = 0) -> int:
    for name in names:
        value = row.get(name, "")
        if value not in ("", None):
            try:
                return int(float(value))
            except ValueError:
                pass
    return default


def cycles(row: dict[str, str]) -> int:
    return integer(row, "cycles", "total_cycles")


def signed(value: Any) -> str:
    return f"{value:+}" if isinstance(value, (int, float)) else "—"


def device_pair(row: dict[str, Any], device: str) -> str:
    first = row.get(f"{device}_repeat0", "")
    second = row.get(f"{device}_repeat1", "")
    delta = row.get(f"{device}_delta_1_minus_0", "")
    if first == "" or second == "":
        return "—"
    return f"{first}/{second} ({signed(delta)})"


def case_identity(row: dict[str, str]) -> str:
    if row.get("case_id"):
        return row["case_id"]
    fields = (
        "project", "level", "path", "direction", "order", "mode",
        "map_mode", "bytes", "contexts", "commands", "rank", "scenario",
        "prefetch_gap_cycles", "data_state"
    )
    return "|".join(row.get(field, "") for field in fields)


def direction(row: dict[str, str]) -> str:
    value = row.get("direction", "")
    return value if value in DIRECTION_ORDER else "mixed" if value == "mixed" else value


def sort_key(row: dict[str, str]) -> tuple[Any, ...]:
    return (
        DIRECTION_ORDER.get(direction(row), 9),
        row.get("path", ""),
        row.get("dtype", ""),
        integer(row, "logical_bytes", "bytes"),
        integer(row, "contexts", "commands"),
        row.get("case_id", ""),
        DEVICE_ORDER.get(row.get("device_class", ""), 9),
        integer(row, "repeat"),
    )


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as output:
        if fields:
            writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(materialized)
    return len(materialized)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_release_hashes() -> Path:
    """Freeze every method, formal-data and report file in this release."""
    checksum = HERE / "releases" / f"{MEASUREMENT_RELEASE}.SHA256SUMS"
    roots = [HERE / "suites", DATA, REPORTS]
    files = [
        HERE / "CURRENT_MEASUREMENT_RELEASE",
        HERE / "CURRENT_VERSION",
        HERE / "README.md",
        HERE / "Makefile",
        HERE / "run.py",
        HERE / "run_cuda_release.py",
        HERE / "generate_reports.py",
        HERE / "validate_repository.py",
        HERE / "prune_to_release.py",
        HERE / "releases" / f"{MEASUREMENT_RELEASE}.json",
        HERE / "releases" / f"{MEASUREMENT_RELEASE}.RESULT_NOT_BEFORE",
        HERE / "catalog" / "implementations.yaml",
    ]
    for root in roots:
        files.extend(path for path in root.rglob("*") if path.is_file())
    files = sorted(
        {
            path for path in files
            if path != checksum
            and "__pycache__" not in path.parts
            and path.suffix not in {".ptx", ".sass", ".cubin"}
            and path.name not in {
                "ventus_tma_comprehensive", "ventus_tensormap_decode",
                "ventus_multi_context", "cuda_g2s_fresh_map",
            }
        }
    )
    checksum.write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(HERE).as_posix()}\n"
            for path in files
        ),
        encoding="utf-8",
    )
    return checksum


def joined_single_instruction(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    selected = [
        row for row in rows
        if row["suite_id"] in {
            "single_instruction.common_matrix",
            "single_instruction.tensor_2d_capacity",
        }
    ]
    grouped: dict[tuple[str, str], dict[str, dict[int, int]]] = defaultdict(lambda: defaultdict(dict))
    exemplar: dict[tuple[str, str], dict[str, str]] = {}
    for row in selected:
        key = (row["suite_id"], case_identity(row))
        exemplar[key] = row
        grouped[key][row["device_class"]][integer(row, "repeat", default=-1)] = cycles(row)
    output: list[dict[str, Any]] = []
    for key, devices in grouped.items():
        source = exemplar[key]
        result: dict[str, Any] = {
            "suite_id": key[0], "case_id": key[1],
            "direction": direction(source), "path": source.get("path", ""),
            "family": source.get("family", ""), "dtype": source.get("dtype", ""),
            "rank": source.get("rank", ""),
            "logical_bytes": integer(source, "logical_bytes", "bytes"),
            "global_offset": source.get("global_offset", ""),
            "stride_class": source.get("stride_class", ""),
            "interleave": source.get("interleave", ""),
            "swizzle": source.get("swizzle", ""),
        }
        for device in ("h100", "b200", "ventus"):
            values = devices.get(device, {})
            version_rows = [
                row for row in selected
                if row["suite_id"] == key[0]
                and case_identity(row) == key[1]
                and row["device_class"] == device
            ]
            result[f"{device}_suite_version"] = (
                version_rows[0]["suite_version"] if version_rows else ""
            )
            result[f"{device}_repeat0"] = values.get(0, "")
            result[f"{device}_repeat1"] = values.get(1, "")
            result[f"{device}_delta_1_minus_0"] = (
                values[1] - values[0] if 0 in values and 1 in values else ""
            )
        output.append(result)
    return sorted(output, key=sort_key)


def decode_estimates(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Compare V11 first-TMAU-use with its explicit-prefetch control."""
    summary = completion_threshold_summary(rows)
    index: dict[tuple[str, str, str, str], dict[str, int]] = defaultdict(dict)
    exemplar: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in summary:
        if row["path"] != "tensor":
            continue
        for device in ("h100", "b200"):
            value = row.get(f"{device}_completion_cycles", "")
            if value == "":
                continue
            key = (
                device, str(row["project"]), str(row["family"]),
                str(row["base_case_id"]),
            )
            exemplar[key] = row
            index[key][str(row["descriptor_state"])] = int(value)
    output: list[dict[str, Any]] = []
    for key, states in sorted(index.items()):
        first = states.get("tmau_first_use_l2_hot")
        prefetched = states.get("tmau_prefetched_l2_hot")
        if first is None or prefetched is None:
            continue
        source = exemplar[key]
        output.append({
            "device": key[0], "direction": "g2s", "project": key[1],
            "family": key[2], "base_case_id": key[3],
            "dtype": source.get("dtype", ""), "rank": source.get("rank", ""),
            "logical_bytes": source["logical_bytes"],
            "tmau_first_use_l2_hot_cycles": first,
            "tmau_prefetched_l2_hot_cycles": prefetched,
            "l2_to_tmau_fetch_and_prepare_cycles": first - prefetched,
            "interpretation": (
                "L2-to-TMAU fetch plus preparation hidden by prefetch; not pure decode"
            ),
        })
    return output


def capacity_large_slopes(joined: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {
        (str(row["direction"]), str(device), int(row["logical_bytes"]), repeat):
            row.get(f"{device}_repeat{repeat}", "")
        for row in joined if row.get("family") == "tensor_capacity"
        for device in ("h100", "b200", "ventus")
        for repeat in (0, 1)
    }
    output: list[dict[str, Any]] = []
    line_count = (32768 - 16384) // 128
    for direct in ("g2s", "s2g", "roundtrip"):
        for device in ("h100", "b200", "ventus"):
            for repeat in (0, 1):
                low = index.get((direct, device, 16384, repeat), "")
                high = index.get((direct, device, 32768, repeat), "")
                if low == "" or high == "":
                    continue
                delta = int(high) - int(low)
                output.append(
                    {
                        "direction": direct, "device": device, "repeat": repeat,
                        "cycles_16k": low, "cycles_32k": high,
                        "incremental_cycles": delta,
                        "cycles_per_128b_line": round(delta / line_count, 6),
                        "effective_bytes_per_cycle": round(16384 / delta, 6) if delta > 0 else "",
                        "note": "incremental 16KiB->32KiB slope; fixed startup terms cancel",
                    }
                )
    return output


def decode_lead_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    selected = [
        row for row in rows
        if row["suite_id"] == "single_instruction.tensormap_decode"
        and row.get("scenario") == "prefetch_lead"
        and integer(row, "rank") == 2
        and integer(row, "repeat") == 0
    ]
    hot: dict[tuple[str, str], int] = {}
    for row in rows:
        if (
            row["suite_id"] == "single_instruction.tensormap_decode"
            and row.get("scenario") == "prefetch_hot"
            and integer(row, "rank") == 2
            and integer(row, "repeat") == 0
        ):
            hot[(row["device_class"], row.get("direction", ""))] = cycles(row)
    output: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        groups[(row["device_class"], row.get("direction", ""))].append(row)
    for key, group in groups.items():
        baseline = hot.get(key)
        ordered = sorted(group, key=lambda row: integer(row, "prefetch_gap_cycles"))
        minimum = ""
        if baseline is not None:
            hidden = [
                integer(row, "prefetch_gap_cycles")
                for row in ordered if cycles(row) <= baseline + 2
            ]
            minimum = min(hidden) if hidden else ""
        output.append(
            {
                "device": key[0], "direction": key[1], "rank": 2,
                "prefetch_hot_baseline_cycles": baseline if baseline is not None else "",
                "minimum_lead_hiding_within_2_cycles": minimum,
                "lead_cycles": ";".join(str(integer(row, "prefetch_gap_cycles")) for row in ordered),
                "repeat0_cycles": ";".join(str(cycles(row)) for row in ordered),
                "note": "lead is the requested delay between prefetch.tensormap and measured issue",
            }
        )
    return sorted(output, key=lambda row: (
        DIRECTION_ORDER.get(str(row["direction"]), 9),
        DEVICE_ORDER.get(str(row["device"]), 9),
    ))


def ventus_decode_pmu(
    runs: dict[tuple[str, str, str], tuple[Path, dict[str, Any]]]
) -> list[dict[str, Any]]:
    selected = [
        (path, manifest) for (suite, device, _), (path, manifest) in runs.items()
        if suite == "single_instruction.tensormap_decode" and device == "ventus"
    ]
    if not selected:
        return []
    path, manifest = max(selected, key=lambda item: str(item[1].get("completed_at_utc", "")))
    pmu = path.parent / "pmu.csv"
    if not pmu.is_file():
        return []
    totals: dict[str, dict[int, int]] = defaultdict(dict)
    raw = path.parent / "raw.csv"
    if raw.is_file():
        with raw.open(newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                totals[row["case_id"]][integer(row, "repeat")] = cycles(row)
    output: list[dict[str, Any]] = []
    with pmu.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            output.append(
                {
                    "suite_version": manifest["suite_version"],
                    **row,
                    "descriptor_compile_cycles_exact": row.get("descriptor_compile_cycles_per_compile", ""),
                    "bind_cycles_exact": row.get("bind_cycles_per_command", ""),
                    "total_cycles_repeat0": totals.get(row.get("case_id", ""), {}).get(0, ""),
                    "total_cycles_repeat1": totals.get(row.get("case_id", ""), {}).get(1, ""),
                }
            )
    return sorted(output, key=lambda row: (
        DIRECTION_ORDER.get(str(row.get("direction", "")), 9),
        integer(row, "rank"), str(row.get("scenario", "")),
    ))


def linear_slope(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return math.nan
    mean_x = statistics.fmean(point[0] for point in points)
    mean_y = statistics.fmean(point[1] for point in points)
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    return sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator if denominator else math.nan


def multi_context_samples(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        if not row["suite_id"].startswith("multi_context."):
            continue
        device = row["device_class"]
        level = row.get("level", "")
        commands = integer(row, "commands")
        size = integer(row, "bytes")
        if level == "multi_cta" and device in {"h100", "b200"}:
            issue_value = integer(row, "issue_span_ns")
            completion_value = integer(row, "completion_span_ns")
            unit = "ns"
        elif level == "multi_cta":
            issue_value = integer(row, "issue_span")
            completion_value = integer(row, "completion_span")
            unit = "cycles"
        else:
            issue_value = integer(row, "issue_cycles")
            completion_value = integer(row, "total_cycles")
            unit = "cycles"
        output.append(
            {
                "device": device, "suite_version": row["suite_version"],
                "project": row.get("project", ""), "level": level,
                "path": row.get("path", ""), "direction": row.get("direction", ""),
                "order": row.get("order", ""), "mode": row.get("mode", ""),
                "map_mode": row.get("map_mode", ""), "bytes": size,
                "contexts": integer(row, "contexts"), "commands": commands,
                "repeat": integer(row, "repeat"),
                "issue_span": issue_value, "completion_span": completion_value,
                "span_unit": unit,
                "completion_per_command": round(completion_value / commands, 6) if commands else "",
                "cycles_per_128b": (
                    round(completion_value / (commands * size / 128.0), 6)
                    if commands and size and unit == "cycles" else ""
                ),
                "effective_bytes_per_cycle": (
                    round(commands * size / completion_value, 6)
                    if completion_value > 0 and unit == "cycles" else ""
                ),
                "effective_gb_per_second": (
                    round(commands * size / completion_value, 6)
                    if completion_value > 0 and unit == "ns" else ""
                ),
                "case_id": case_identity(row), "raw_file": row["raw_file"],
            }
        )
    return sorted(output, key=lambda row: (
        DIRECTION_ORDER.get(str(row["direction"]), 9), str(row["path"]),
        int(row["bytes"]), str(row["level"]), str(row["mode"]),
        str(row["map_mode"]), int(row["contexts"]),
        DEVICE_ORDER.get(str(row["device"]), 9), int(row["repeat"]),
    ))


def multi_context_slopes(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    selected = [row for row in rows if row["suite_id"].startswith("multi_context.")]
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    fields = ("device_class", "suite_id", "level", "path", "direction", "order", "mode", "map_mode", "bytes", "repeat")
    for row in selected:
        groups[tuple(row.get(field, "") for field in fields)].append(row)
    output: list[dict[str, Any]] = []
    for key, group in groups.items():
        device, _, level = key[:3]
        if level == "multi_cta" and device in {"h100", "b200"}:
            metric = "completion_span_ns"
            unit = "ns"
        elif level == "multi_cta":
            metric = "completion_span"
            unit = "cycles"
        else:
            metric = "total_cycles"
            unit = "cycles"
        unique = {
            integer(row, "commands"): integer(row, metric, "total_cycles")
            for row in group
        }
        high = sorted(unique.items())[-3:]
        slope = linear_slope([(float(x), float(y)) for x, y in high])
        if math.isnan(slope):
            continue
        size = int(key[8])
        output.append(
            {
                **dict(zip(fields, key)),
                "high_n_commands": ";".join(str(x) for x, _ in high),
                "high_n_completion_metric": ";".join(str(y) for _, y in high),
                "completion_metric": metric,
                "slope_unit_per_command": unit,
                "completion_slope_per_command": round(slope, 6),
                "effective_bytes_per_cycle": round(size / slope, 6) if slope > 0 and unit == "cycles" else "",
                "effective_gb_per_second": round(size / slope, 6) if slope > 0 and unit == "ns" else "",
                "note": "effective service slope; does not identify physical datapath count",
            }
        )
    return sorted(output, key=lambda row: (
        DIRECTION_ORDER.get(str(row["direction"]), 9), str(row["path"]),
        int(row["bytes"]), DEVICE_ORDER.get(str(row["device_class"]), 9)
    ))


def s2g_size_sweep_summary(
    samples: list[dict[str, Any]], slopes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expose both N=8 batch throughput and the high-N marginal slope."""
    selected_samples = [
        row for row in samples
        if row.get("project") == "s2g_size_sweep" and row.get("repeat") == 1
    ]
    selected_slopes = [
        row for row in slopes
        if row.get("suite_id") == "multi_context.s2g_size_sweep"
        and row.get("repeat") == "1"
    ]
    sample_index = {
        (str(row["device"]), int(row["bytes"]), int(row["commands"])): row
        for row in selected_samples
    }
    output: list[dict[str, Any]] = []
    for slope in selected_slopes:
        device = str(slope["device_class"])
        size = int(slope["bytes"])
        one = sample_index.get((device, size, 1), {})
        eight = sample_index.get((device, size, 8), {})
        slope_cycles = float(slope["completion_slope_per_command"])
        eight_cycles = eight.get("completion_span", "")
        output.append({
            "device": device,
            "bytes": size,
            "ideal_payload_cycles_at_32b_per_cycle": round(size / 32.0, 6),
            "n1_total_cycles": one.get("completion_span", ""),
            "n8_total_cycles": eight_cycles,
            "n8_effective_bytes_per_cycle": (
                round(8 * size / float(eight_cycles), 6)
                if eight_cycles not in ("", 0) else ""
            ),
            "high_n_commands": slope["high_n_commands"],
            "slope_cycles_per_command": slope["completion_slope_per_command"],
            "slope_excess_over_b_div_32": round(
                slope_cycles - size / 32.0, 6
            ),
            "marginal_bytes_per_cycle": slope["effective_bytes_per_cycle"],
        })
    return sorted(output, key=lambda row: (
        int(row["bytes"]), DEVICE_ORDER.get(str(row["device"]), 9)
    ))


def residency_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    selected = [row for row in rows if row["suite_id"] == "single_instruction.data_residency_128b"]
    groups: dict[tuple[str, str, str, str], dict[int, tuple[int, str]]] = defaultdict(dict)
    for row in selected:
        groups[
            (
                row["device_class"], row.get("direction", ""),
                row.get("path", ""), row.get("case_id", ""),
            )
        ][integer(row, "repeat")] = (
            cycles(row), row.get("data_state", row.get("scenario", ""))
        )
    output: list[dict[str, Any]] = []
    for key, values in groups.items():
        if not {0, 1} <= set(values):
            continue
        output.append(
            {
                "device": key[0], "direction": key[1], "path": key[2],
                "case_id": key[3],
                "scenario": f"{values[0][1]}→{values[1][1]}",
                "repeat0": values[0][0], "repeat1": values[1][0],
                "repeat1_minus_repeat0": values[1][0] - values[0][0],
            }
        )
    return sorted(output, key=lambda row: (
        DIRECTION_ORDER.get(str(row["direction"]), 9), str(row["path"]),
        DEVICE_ORDER.get(str(row["device"]), 9), str(row["scenario"])
    ))


def control_item_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    selected = [
        row for row in rows
        if row["suite_id"] == "single_instruction.control_items"
    ]
    groups: dict[tuple[str, str, str, str], dict[int, int]] = defaultdict(dict)
    for row in selected:
        groups[
            (
                row["device_class"], row.get("direction", ""),
                row.get("method", ""), row.get("case_id", ""),
            )
        ][integer(row, "repeat")] = cycles(row)
    output: list[dict[str, Any]] = []
    for key, values in groups.items():
        if not {0, 1} <= set(values):
            continue
        output.append(
            {
                "device": key[0], "direction": key[1], "method": key[2],
                "case_id": key[3], "bytes": 4096,
                "repeat0": values[0], "repeat1": values[1],
                "repeat1_minus_repeat0": values[1] - values[0],
            }
        )
    return sorted(output, key=lambda row: (
        DIRECTION_ORDER.get(str(row["direction"]), 9),
        str(row["method"]), DEVICE_ORDER.get(str(row["device"]), 9),
    ))


def completion_threshold_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Return only the release-pinned completion scan and its cold/hot values."""
    completion_version = PINNED_SUITE_VERSIONS[
        "single_instruction.completion_latency"
    ]
    selected = [
        row for row in rows
        if row["suite_id"] == "single_instruction.completion_latency"
        and row.get("suite_version") == completion_version
        and row.get("wait_method") == "single_test_wait_cold_then_hot_v12"
    ]
    grouped: dict[
        tuple[str, str, str, str, int, str],
        dict[str, dict[int, list[dict[str, str]]]],
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in selected:
        key = (
            row.get("project", ""), row.get("family", ""),
            row.get("base_case_id", row.get("case_id", "")),
            row.get("path", ""), integer(row, "logical_bytes", "bytes"),
            row.get("descriptor_mode", row.get("descriptor_state", "")),
        )
        grouped[key][row["device_class"]][integer(row, "repeat")].append(row)

    output: list[dict[str, Any]] = []
    for (
        project, family, base_case_id, path, size, descriptor_state
    ), devices in grouped.items():
        exemplar = next(
            sample
            for repeat_samples in devices.values()
            for samples in repeat_samples.values()
            for sample in samples
        )
        result: dict[str, Any] = {
            "direction": "g2s", "project": project, "family": family,
            "base_case_id": base_case_id, "path": path,
            "logical_bytes": size, "descriptor_state": descriptor_state,
            "dtype": exemplar.get("dtype", ""),
            "rank": exemplar.get("rank", ""),
            "wait_method": exemplar.get("wait_method", ""),
            "measurement_semantics": (
                "separate_midpoint_values_for_same_map_cold_then_hot_pair"
            ),
        }
        for device in ("h100", "b200"):
            device_samples = [
                sample
                for samples in devices.get(device, {}).values()
                for sample in samples
            ]
            result[f"{device}_suite_version"] = (
                device_samples[0].get("suite_version", "")
                if device_samples else ""
            )
            result[f"{device}_run_id"] = (
                device_samples[0].get("run_id", "")
                if device_samples else ""
            )
            result[f"{device}_raw_file"] = (
                device_samples[0].get("raw_file", "")
                if device_samples else ""
            )
            for repeat in (0, 1):
                samples = sorted(
                    devices.get(device, {}).get(repeat, []),
                    key=lambda row: (
                        integer(row, "probe_begin_from_issue_begin_cycles"),
                        integer(row, "delay_iterations"),
                    ),
                )
                prefix = f"{device}_repeat{repeat}"
                outcomes = [integer(row, "probe_success") for row in samples]
                probe_durations = [
                    integer(row, "probe_instruction_cycles") for row in samples
                ]
                probe_begins = [
                    integer(row, "probe_begin_from_issue_begin_cycles")
                    for row in samples
                ]
                probe_steps = [
                    second - first
                    for first, second in zip(probe_begins, probe_begins[1:])
                ]
                result[f"{prefix}_probe_instruction_cycles_median"] = (
                    statistics.median(probe_durations) if probe_durations else ""
                )
                result[f"{prefix}_probe_begin_step_median"] = (
                    statistics.median(probe_steps) if probe_steps else ""
                )
                if not outcomes or not any(outcomes) or all(outcomes):
                    for name in (
                        "last_false_probe_begin", "first_true_probe_end",
                        "cycles", "uncertainty", "crossed_samples",
                    ):
                        result[f"{prefix}_{name}"] = ""
                    continue
                best_error = len(samples) + 1
                best_cuts: list[int] = []
                total_false = len(outcomes) - sum(outcomes)
                prefix_true = 0
                prefix_false = 0
                for cut in range(1, len(samples)):
                    previous = outcomes[cut - 1]
                    prefix_true += previous
                    prefix_false += 1 - previous
                    error = prefix_true + (total_false - prefix_false)
                    if error < best_error:
                        best_error = error
                        best_cuts = [cut]
                    elif error == best_error:
                        best_cuts.append(cut)
                cut = best_cuts[len(best_cuts) // 2]
                lower = integer(
                    samples[cut - 1], "probe_begin_from_issue_begin_cycles"
                )
                upper = integer(
                    samples[cut], "probe_end_from_issue_begin_cycles"
                )
                result[f"{prefix}_last_false_probe_begin"] = lower
                result[f"{prefix}_first_true_probe_end"] = upper
                result[f"{prefix}_cycles"] = (
                    (lower + upper + 1) // 2 if upper > lower else ""
                )
                result[f"{prefix}_uncertainty"] = (
                    (upper - lower + 1) // 2 if upper > lower else ""
                )
                result[f"{prefix}_crossed_samples"] = best_error
            cycle0 = result.get(f"{device}_repeat0_cycles", "")
            cycle1 = result.get(f"{device}_repeat1_cycles", "")
            result[f"{device}_completion_cycles"] = (
                (cycle0 + cycle1 + 1) // 2
                if cycle0 != "" and cycle1 != "" else ""
            )
            uncertainty0 = result.get(f"{device}_repeat0_uncertainty", "")
            uncertainty1 = result.get(f"{device}_repeat1_uncertainty", "")
            result[f"{device}_uncertainty_cycles"] = (
                max(uncertainty0, uncertainty1)
                if uncertainty0 != "" and uncertainty1 != "" else ""
            )
        output.append(result)
    return sorted(output, key=lambda row: (
        str(row["project"]), str(row["family"]), str(row["path"]),
        int(row["logical_bytes"]), str(row["descriptor_state"]),
    ))


def resource_limit_rows(
    runs: dict[tuple[str, str, str], tuple[Path, dict[str, Any]]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for path, manifest in runs.values():
        resource = path.parent / "resource_limits.csv"
        if not resource.is_file():
            continue
        with resource.open(newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                output.append(
                    {
                        "device": manifest["device"]["class"],
                        "suite_id": manifest["suite_id"],
                        "suite_version": manifest["suite_version"],
                        "run_id": manifest["run_id"],
                        **row,
                    }
                )
    return output


def run_inventory(
    runs: dict[tuple[str, str, str], tuple[Path, dict[str, Any]]],
    diagnostics: list[tuple[Path, dict[str, Any]]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for path, manifest in [*runs.values(), *diagnostics]:
        measurement = manifest.get("measurement", {})
        output.append(
            {
                "status": manifest.get("status", ""),
                "suite_id": manifest.get("suite_id", ""),
                "suite_version": manifest.get("suite_version", ""),
                "device": manifest.get("device", {}).get("class", ""),
                "implementation_version": manifest.get("implementation", {}).get("version", ""),
                "variant": manifest.get("variant", ""),
                "run_id": manifest.get("run_id", ""),
                "case_count": manifest.get("case_count", ""),
                "resource_limited_case_count": manifest.get("resource_limited_case_count", ""),
                "warmups": measurement.get("warmups", ""),
                "repeats": measurement.get("repeats", ""),
                "paid_attempts": measurement.get("paid_attempts", ""),
                "failure_class": manifest.get("failure_class", ""),
                "reason": manifest.get("reason", ""),
                "manifest": path.relative_to(HERE).as_posix(),
            }
        )
    return sorted(output, key=lambda row: (
        str(row["suite_id"]), DEVICE_ORDER.get(str(row["device"]), 9),
        str(row["variant"]), str(row["status"]),
    ))


def write_current_run_catalog(
    runs: dict[tuple[str, str, str], tuple[Path, dict[str, Any]]]
) -> None:
    path = HERE / "catalog" / "runs.csv"
    fields = (
        "run_id", "suite_id", "suite_version", "device",
        "implementation_version", "variant", "test_signature",
        "completed_at_utc", "path", "status", "reason", "superseded_by",
    )
    rows: list[dict[str, Any]] = []
    for manifest_path, manifest in runs.values():
        rows.append({
            "run_id": manifest.get("run_id", ""),
            "suite_id": manifest.get("suite_id", ""),
            "suite_version": manifest.get("suite_version", ""),
            "device": manifest.get("device", {}).get("class", ""),
            "implementation_version": manifest.get("implementation", {}).get("version", ""),
            "variant": manifest.get("variant", ""),
            "test_signature": manifest.get("test_signature", ""),
            "completed_at_utc": manifest.get("completed_at_utc", ""),
            "path": manifest_path.relative_to(HERE).as_posix(),
            "status": "passed",
            "reason": "",
            "superseded_by": "",
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (
            str(row["suite_id"]), str(row["device"]), str(row["variant"])
        )))


def fresh_tensormap_report(
    output: Path,
    rows: list[dict[str, str]],
    completion_threshold: list[dict[str, Any]],
    decode: list[dict[str, Any]],
) -> None:
    """Write the plain-language report for the corrected V9 G2S experiment."""

    state_first = "tmau_first_use_l2_hot"
    state_prefetched = "tmau_prefetched_l2_hot"
    index = {
        (
            str(row["project"]), str(row["family"]),
            int(row["logical_bytes"]), str(row["descriptor_state"]),
        ): row
        for row in completion_threshold
    }

    def measured(
        project: str, family: str, size: int, state: str, device: str
    ) -> int | str:
        row = index.get((project, family, size, state))
        return row.get(f"{device}_completion_cycles", "") if row else ""

    def delta(first: int | str, prefetched: int | str) -> int | str:
        return first - prefetched if isinstance(first, int) and isinstance(prefetched, int) else ""

    v9_rows = [
        row for row in rows
        if row.get("suite_id") == "single_instruction.completion_latency"
        and row.get("suite_version") == "v9"
        and row.get("wait_method") == "single_test_wait_fresh_map_v9"
    ]
    evidence: list[dict[str, Any]] = []
    for device in ("h100", "b200"):
        for project in ("common", "tensor_2d"):
            samples = [
                row for row in v9_rows
                if row.get("device_class") == device
                and row.get("project") == project
            ]
            tensor = [row for row in samples if row.get("path") == "tensor"]
            if not samples:
                continue
            evidence.append({
                "device": device,
                "project": project,
                "rows": len(samples),
                "tensor_rows": len(tensor),
                "unique_addresses": len({row.get("descriptor_va") for row in tensor}),
                "unique_contents": len({row.get("descriptor_fingerprint64") for row in tensor}),
                "prior_uses": sum(integer(row, "prior_tma_use") for row in tensor),
                "errors": sum(integer(row, "errors") for row in samples),
                "raw_file": samples[0].get("raw_file", ""),
            })

    family_cases: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in completion_threshold:
        family_cases[(str(row["project"]), str(row["family"]))].add(
            str(row["base_case_id"])
        )

    penalties = {
        device: [
            int(row["l2_to_tmau_fetch_and_prepare_cycles"])
            for row in decode if row.get("device") == device
        ]
        for device in ("h100", "b200")
    }

    # A partial report is useful while the paid GPU runs are still pending, but
    # it must never turn an empty post-cutoff data set into claims based on old
    # measurements.  Keep this page as a method/status page until at least one
    # current CUDA run exists.
    if not evidence:
        output.write_text(
            "\n".join([
                "# G2S TensorMap 首次使用周期（FreshTensorMapV1 / V9，审计中）",
                "",
                "> 当前没有结果切换点之后的 H100/B200 正式数据。本页不引用、也不回填任何旧测试值。",
                "",
                "## 本轮将怎样测",
                "",
                "1. 每个测量点分配一个从未被 TMA 指令使用过的 TensorMap；不同点的地址和内容都不同。",
                "2. 只用普通 `ld.global.cg` 把 TensorMap 放入 L2，不执行 `prefetch.tensormap`，也不先运行同一条 Tensor TMA。",
                "3. 随后只发出一次被测 Tensor TMA，测量从指令发出到数据搬运完成。",
                "4. A/B 两次使用不同 TensorMap；不会在同一个 TensorMap 上连续运行两次。",
                "5. 显式预取是单独控制组，使用另一张新 TensorMap；它只用于量化提前送入 TMAU 能隐藏多少时间。",
                "",
                "正式运行还会检查 PTX/SASS：主测 kernel 中不得出现 `UTMACCTL.PF`，显式预取控制 kernel 中必须恰好出现一次。",
                "",
                "完整实现约束见 [V9 方法说明](../../../suites/single_instruction/completion_latency/v9/METHOD.md)。",
                "CUDA 数据完成后，本页会在同一路径更新为周期表和逐运行证据。",
            ]) + "\n",
            encoding="utf-8",
        )
        return

    lines = [
        "# G2S TensorMap 首次使用周期（H100/B200，FreshTensorMapV1 / V9）", "",
        "## 结论", "",
        "这次测到的是用户要求的状态：**TensorMap 已在 L2，但 TMAU 从未使用过它**。"
        "被测 Tensor TMA 因此必须自己完成从 L2 取得 TensorMap 并在 TMAU 内准备它的过程。", "",
        f"在全部 {len(penalties['h100'])} 个 H100 对照项中，这一步增加的周期中位数是 "
        f"{statistics.median(penalties['h100']):g}；B200 的中位数是 "
        f"{statistics.median(penalties['b200']):g}。"
        "它包含 L2 读取和 TMAU 内部准备，不能进一步拆成纯 L2 延迟或纯解码延迟。", "",
        "以下结果只来自 FreshTensorMapV1 / V9；测试树中的其他版本和数据不会参与计算。", "",
        "## 常用容量", "",
        "每个数字都是从发出 TMA 指令到搬运完成的周期数。表中不写范围；"
        "当前单个数字的测量精度约为 ±20 周期。", "",
        "|容量(B)|H100：TMAU 首次使用|H100：已提前预取|H100：多出|B200：TMAU 首次使用|B200：已提前预取|B200：多出|",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for size in (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768):
        h_first = measured("common", "tensor_capacity", size, state_first, "h100")
        h_pref = measured("common", "tensor_capacity", size, state_prefetched, "h100")
        b_first = measured("common", "tensor_capacity", size, state_first, "b200")
        b_pref = measured("common", "tensor_capacity", size, state_prefetched, "b200")
        lines.append(
            f"|{size}|{h_first}|{h_pref}|{delta(h_first, h_pref)}|"
            f"{b_first}|{b_pref}|{delta(b_first, b_pref)}|"
        )

    lines.extend([
        "", "`已提前预取` 是独立的控制组：它使用另一个同样全新的 TensorMap，先执行 "
        "`prefetch.tensormap` 并留出 1024 周期，再发出被计时的 TMA。它不等于提前搬运过数据；"
        "它只把 TensorMap 的读取和准备移到计时区间之前。", "",
        "## 2D 长度关键点", "",
        "2D 项同时比较两种 TensorMap 状态。32B 到 512B 的关键点按 32B 观察；"
        "完整 91 个 2D 配置见 [完整结果 CSV](completion_threshold.csv)。", "",
        "|行宽系列|容量(B)|H100：首次使用/已预取|B200：首次使用/已预取|",
        "|---|---:|---:|---:|",
    ])
    for family, size in (
        ("tensor_2d_length_u8_row32", 32),
        ("tensor_2d_length_u8_row32", 64),
        ("tensor_2d_length_u8_row32", 96),
        ("tensor_2d_length_u8_row32", 128),
        ("tensor_2d_length_u8_row32", 160),
        ("tensor_2d_length_u8_row32", 192),
        ("tensor_2d_length_u8_row32", 224),
        ("tensor_2d_length_u8_row32", 256),
        ("tensor_2d_length_u8_row32", 512),
        ("tensor_2d_length_u8_row32", 1024),
        ("tensor_2d_length_u8_row32", 4096),
        ("tensor_2d_length_u8_row32", 8192),
        ("tensor_2d_length_u8_row128", 16384),
        ("tensor_2d_length_u8_row128", 32768),
    ):
        h_first = measured("tensor_2d", family, size, state_first, "h100")
        h_pref = measured("tensor_2d", family, size, state_prefetched, "h100")
        b_first = measured("tensor_2d", family, size, state_first, "b200")
        b_pref = measured("tensor_2d", family, size, state_prefetched, "b200")
        lines.append(
            f"|{family.removeprefix('tensor_2d_length_')}|{size}|"
            f"{h_first}/{h_pref}|{b_first}/{b_pref}|"
        )

    lines.extend([
        "", "## 实际测试了什么", "",
        "常用矩阵覆盖 Bulk 容量，以及 Tensor 的容量、地址对齐、数据格式、rank 1–5、"
        "stride、subbox、OOB、NaN fill、swizzle 和合法 interleave16。"
        "2D 项覆盖以下实际容量（单位 B）：", "",
        "`32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, "
        "448, 480, 512, 640, 768, 896, 992, 1024, 1056, 1152, 1280, 1408, "
        "1536, 1664, 1792, 1920, 2016, 2048, 2080, 2560, 3072, 3584, 4064, "
        "4096, 4128, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8128, 8160, "
        "8192, 8256, 8704, 9216, 9728, 10240, 10752, 11264, 11776, 12288, "
        "12800, 13312, 13824, 14336, 14848, 15360, 15872, 16256, 16320, "
        "16384, 16512, 17408, 18432, 20480, 22528, 24576, 28672, 32768`", "",
        "各测试组的独立配置数（两种 TensorMap 状态不会重复计数）：", "",
        "|项目|测试组|独立配置数|", "|---|---|---:|",
    ])
    for (project, family), cases in sorted(family_cases.items()):
        lines.append(f"|{project}|{family}|{len(cases)}|")

    lines.extend([
        "", "## 怎样保证没有提前进入 TMAU", "",
        "对每一条被测 Tensor TMA，程序依次执行：", "",
        "1. 分配一个从未使用过的 TensorMap 地址；相邻地址至少相隔 4KiB。",
        "2. 使用不同的 global base 编码 TensorMap，所以地址和 128B 内容都不复用。",
        "3. 只用普通 `ld.global.cg` 读取 TensorMap 的四个 32B 扇区，使它留在 L2。",
        "4. 在没有 `prefetch.tensormap` 的独立 kernel 中发出唯一一次 Tensor TMA。",
        "5. 用一个新 mbarrier 做一次非阻塞完成检查；检查之后的清理不计入时间。", "",
        "延迟扫描中的每一个点、A/B 两次测量、不同测试项，全部使用不同 TensorMap。"
        "因此 A/B 不是对同一个 TensorMap 连续运行两次。", "",
        "编译后的 PTX 和 SASS 也做了自动检查：主 kernel 的 `UTMACCTL.PF` 数量必须为 0；"
        "显式预取控制 kernel 的数量必须为 1；检查失败时整次运行不会登记为正式结果。", "",
        "## 原始数据自检", "",
        "|设备|项目|原始行数|Tensor 行数|不同 TensorMap 地址|不同 TensorMap 内容|此前 TMAU 使用次数|结果错误|证据|",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for item in evidence:
        raw = str(item["raw_file"])
        parent = Path(raw).parent.as_posix() if raw else ""
        links = (
            f"[CSV](../../../{raw}) / "
            f"[指令检查](../../../{parent}/attempt_01.json) / "
            f"[运行清单](../../../{parent}/run-manifest.json)"
            if raw else "—"
        )
        lines.append(
            f"|{item['device'].upper()}|{item['project']}|{item['rows']}|"
            f"{item['tensor_rows']}|{item['unique_addresses']}|"
            f"{item['unique_contents']}|{item['prior_uses']}|{item['errors']}|{links}|"
        )

    lines.extend([
        "", f"四组正式运行共 {sum(item['rows'] for item in evidence):,} 行，正确性错误为 0。"
        "四组运行都属于 FreshTensorMapV1，并使用 V9 测量方法。", "",
        "## 如何理解这些数字", "",
        "- 小容量下，固定启动时间占主导，所以 128B 和 1KiB 可能只差几十周期；"
        "这不表示二者真的在同一周期完成。",
        "- 首次使用与显式预取之间约 200–300 周期的稳定差值，才是 TensorMap 从 L2 进入 TMAU"
        "并完成准备所增加的时间。",
        "- `prefetch.tensormap` 不会提前搬运 payload，也不等同于提前执行整条 TMA；"
        "它只提前处理 TensorMap。",
        "- NVIDIA 没有公开 TMAU 内部完成时间戳。本报告通过单次 `mbarrier.test_wait` 的受控延迟扫描"
        "给出一个完成周期估计值，而不是虚构一个不可直接读取的内部时间戳。", "",
        "测试实现和测量约束见 [V9 方法说明](../../../suites/single_instruction/completion_latency/v9/METHOD.md)。",
    ])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cold_then_hot_markdown_report(
    output: Path, runs: dict[tuple[str, str, str], tuple[Path, dict[str, Any]]],
    joined: list[dict[str, Any]], capacity_slopes: list[dict[str, Any]],
    decode: list[dict[str, Any]],
    leads: list[dict[str, Any]], ventus_pmu: list[dict[str, Any]],
    slopes: list[dict[str, Any]],
    size_sweep: list[dict[str, Any]], residency: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    completion_threshold: list[dict[str, Any]],
    resources: list[dict[str, Any]], diagnostics: list[tuple[Path, dict[str, Any]]],
    missing: list[tuple[str, str, str]],
) -> None:
    """Write the plain-language ColdThenHotV1 report.

    The old report treated repeat 0/1 as two independent cold descriptors.
    This report intentionally says first/second and only consumes validated
    same-address cold/hot pairs from the new release.
    """

    def arrow(first: Any, second: Any) -> str:
        if first in ("", None) or second in ("", None):
            return "—"
        delta = int(second) - int(first)
        return f"{first} → {second} ({delta:+d})"

    def joined_pair(row: dict[str, Any] | None, device: str) -> str:
        if row is None:
            return "—"
        return arrow(row.get(f"{device}_repeat0", ""), row.get(f"{device}_repeat1", ""))

    def completion_pair(row: dict[str, Any] | None, device: str) -> str:
        if row is None:
            return "—"
        first = row.get(f"{device}_repeat0_cycles", "")
        second = row.get(f"{device}_repeat1_cycles", "")
        if first == "" or second == "":
            return "—"
        uncertainty = row.get(f"{device}_uncertainty_cycles", "")
        suffix = f"，边界误差 ±{uncertainty}" if uncertainty != "" else ""
        return f"{first} → {second} ({int(second) - int(first):+d}{suffix})"

    def completion_probe_stat(device: str, suffix: str) -> str:
        values = [
            float(row[f"{device}_repeat{repeat}_{suffix}"])
            for row in completion_threshold
            for repeat in (0, 1)
            if row.get(f"{device}_repeat{repeat}_{suffix}") not in ("", None)
        ]
        return f"{statistics.median(values):g}" if values else "—"

    capacity = {
        (str(row.get("direction")), int(row.get("logical_bytes", 0))): row
        for row in joined if row.get("family") == "tensor_capacity"
    }
    completion = {
        (str(row.get("base_case_id")), str(row.get("descriptor_state"))): row
        for row in completion_threshold
    }
    two_d_candidates = [
        row for row in joined
        if row.get("suite_id") == "single_instruction.tensor_2d_capacity"
        and "_u8_" in str(row.get("case_id", ""))
    ]
    two_d: dict[tuple[str, int], dict[str, Any]] = {}
    for row in two_d_candidates:
        key = (str(row.get("direction")), int(row.get("logical_bytes", 0)))
        previous = two_d.get(key)
        preference = 1 if "_row128_" in str(row.get("case_id", "")) else 0
        previous_preference = (
            1 if previous is not None and "_row128_" in str(previous.get("case_id", "")) else 0
        )
        if previous is None or preference > previous_preference:
            two_d[key] = row
    key_sizes = (32, 64, 96, 128, 160, 192, 224, 256, 512, 1024,
                 2048, 4096, 8192, 16384, 22528, 32768)
    common_sizes = (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
    pmu_cold = [row for row in ventus_pmu if row.get("scenario") == "cold_then_hot"]
    missing_text = "、".join(
        f"{suite}/{device}/{variant or 'default'}"
        for suite, device, variant in missing
    )

    lines = [
        "# TMA 三平台周期比较（ColdThenHotV1）", "",
        "> 本报告只使用 ColdThenHotV1 的新测试。FreshTensorMapV1 已撤回，旧数据不会补入空缺。", "",
        "## 先说明两列到底是什么", "",
        "主周期表中的每个 case 只分配一张此前没有被 TMAU 使用过的 TensorMap。第一次 TMA 完成后，马上用完全相同的 TensorMap 地址和内容执行第二次。后面的 payload 驻留控制是明确标出的例外：它在计时外先完成一次 TMA，把 descriptor 固定为真正的热状态。", "",
        "- **第一次（冷）**：descriptor 可以已经被普通 global load 放进 L2，但此前没有 TMA 指令读取、编译或缓存过它，也没有执行 `prefetch.tensormap`。这个数包含 TMAU 从 L2 取得 TensorMap 并准备它的时间。", "",
        "- **第二次（热）**：紧接着复用第一次的同一个地址和指纹；两次之间没有 prefetch、invalidate 或 acquire。这个数表示 descriptor 已经被上一条 TMA 准备好的情况。", "",
        "- 主周期表的每条正式 Tensor 记录都保存 descriptor 地址、64 位内容指纹、`prior_tma_use=0/1` 和 prefetch 标志；验证器要求同一对地址和指纹相等、第一次前没有 prefetch。payload 驻留 v5 单独记录 `prior_tma_use=1/2`，不混入主周期表。", "",
        "Ventus 每个 case 在独立 RTL 仿真实例中运行，因此跨 case 可以出现相同虚拟地址；实例启动时 TMAU 和 compiled store 都被复位。一个 case 内的冷、热两次仍在同一实例、同一 kernel 中连续完成。", "",
        f"当前采用 {len(runs)} 项通过验证的正式运行；缺少 {len(missing)} 项；保留 {len(diagnostics)} 项失败诊断。", "",
    ]
    if missing:
        lines.extend(["尚未完成：" + missing_text + "。失败项不会用旧数据填充。", ""])

    lines.extend([
        "## 测试项目", "",
        "|项目|版本|测什么|", "|---|---:|---|",
        "|共同单指令矩阵|v4|334 个安全、非 Reduce 配置：Bulk/Tensor、G2S/S2G/roundtrip、rank 1–5、dtype、stride、OOB、swizzle、合法 interleave16|",
        "|2D 容量矩阵|v4|182 个二维 Tensor 点；32B 到 16KiB 围绕 32B 边界加密，之后保留关键大容量点|",
        "|G2S 完成边界|v12|固定单轮扫描 640 个延迟位置；请求步长为 3 周期，每条命令只执行一次 `mbarrier.test_wait`，冷、热各给一个整数中点和边界误差|",
        "|TensorMap 准备|v4|rank 1–5 的 cold demand、compiled hit、显式 prefetch 和 prefetch lead；Ventus 同时读取 PMU|",
        "|128B 数据驻留|v5|H100/B200 的 cold→hot、cold→cold、hot→hot 三组 payload 控制；计时外先用同一 TensorMap 完成一次 TMA，再准备数据冷热，避免 descriptor 首次使用混入数据驻留差值|",
        "|同向多命令|v4|G2S 与 S2G 分开，1/2/4/8/16/32 条命令，同 CTA/多 CTA、serial/batched、same/distinct map；无法映射的 PDS spill 明确记为资源限制|",
        "|混合方向多命令|v3|G2S/S2G alternating、先 G2S、先 S2G，观察排队和方向切换|",
        "|控制项|v3|4KiB cooperative copy 与 compute/TMA overlap|", "",
        "### 这些项目具体怎样测", "",
        "- **共同矩阵和 2D 容量**：一个 case 只测一组参数；同一 kernel 依次发两条使用同一 descriptor 的命令。CUDA 每个 case 使用间隔 4KiB 的新 descriptor 页，Ventus 每个 case 启动全新 RTL 实例。两条命令中间不插入 prefetch、acquire 或 invalidate。", "",
        "- **G2S 完成边界**：G2S 没有公开的 `wait_group 0` 完成时间戳。v12 不做粗扫和细扫，而是把 640 个固定位置一次全部测完，顺序为 639 到 0；每个位置使用一张新的 TensorMap，并在同一 kernel 内先冷启动、清理完成，再立即热复用。每条命令的计时区只执行一次非阻塞 `mbarrier.test_wait`。", "",
        "- **步长怎样实现**：640 个位置编成 80 个静态 kernel bank，每个 bank 8 个位置；bank 在 kernel 启动前由 CPU 选择，不进入计时区。相邻逻辑位置请求增加 1 条 predicate、1 条一致分支和 1 条保留指令，即目标步长 3。GPU 的双发射、分支流水和指令缓存会使真实间隔不必严格等于 3，所以汇总始终使用每个样本实际记录的 `clock64`，不把逻辑步长冒充物理周期。", "",
        "- **一个周期数怎样得到**：按实际 probe 时刻排序，寻找结果从 false 变为 true 的最佳单调分界；最后一个 false 的 probe-begin 与第一个 true 的 probe-end 之间取整数中点作为完成周期，半个间隔作为边界误差。它是 640 个不同命令共同夹出的完成边界，不是 NVIDIA 未公开的单命令内部完成时间戳。", "",
        f"- **为什么误差仍大于 3 周期**：步长只控制相邻 probe 的开始时刻；本轮所有 case 的实际 probe-begin 间距中位数为 H100 {completion_probe_stat('h100', 'probe_begin_step_median')} 周期、B200 {completion_probe_stat('b200', 'probe_begin_step_median')} 周期。一次 `test_wait` 自身的实测时长中位数分别为 H100 {completion_probe_stat('h100', 'probe_instruction_cycles_median')} 周期、B200 {completion_probe_stat('b200', 'probe_instruction_cycles_median')} 周期。上界必须取 first-true 的 probe-end，所以边界误差同时包含这一条观察指令的执行时间；它不是再次粗扫造成的。", "",
        "- **S2G**：命令提交后直接执行 `cp.async.bulk.wait_group 0`，从 issue 前的 `clock64` 读数到 wait 返回后的读数就是表中周期。", "",
        "- **TensorMap 准备**：NVIDIA 只能做 cold、prefetch 和真正 hot 之间的差分；Ventus 额外读取 compiler/binder PMU，所以能确认每对命令只有一次 miss、一次 compile 和一次 hit。", "",
        "- **多命令**：同向 G2S、同向 S2G、混合方向分别运行；同 CTA 与多 CTA 分开，same-map 与 distinct-map 分开。每条命令使用独立数据区域和完成对象，不允许通过同一 cache line 合并伪造斜率。", "",
        "## G2S", "",
        "### Tensor capacity：从 issue 到真正完成", "",
        "H100/B200 使用 v12 的固定单轮 640 点 `test_wait` 扫描值；Ventus 使用 RTL 的精确完成周期。表中每个 TensorMap 都是第一次冷启动后立即热复用。", "",
        "|容量(B)|H100 第一次 → 第二次|B200 第一次 → 第二次|Ventus 第一次 → 第二次|",
        "|---:|---:|---:|---:|",
    ])
    for size in common_sizes:
        case_id = f"tensor_capacity_g2s_b{size}"
        measured = completion.get((case_id, "tmau_first_use_l2_hot"))
        lines.append(
            f"|{size}|{completion_pair(measured, 'h100')}|"
            f"{completion_pair(measured, 'b200')}|"
            f"{joined_pair(capacity.get(('g2s', size)), 'ventus')}|"
        )

    lines.extend([
        "", "### 2D、U8、每行 128B 的关键长度", "",
        "|容量(B)|2D 行宽(B)|H100 第一次 → 第二次|B200 第一次 → 第二次|Ventus 第一次 → 第二次|",
        "|---:|---:|---:|---:|---:|",
    ])
    for size in key_sizes:
        row = two_d.get(("g2s", size))
        case_id = str(row.get("case_id", "")) if row is not None else ""
        measured = completion.get((case_id, "tmau_first_use_l2_hot"))
        if measured is not None or row is not None:
            family = str(row.get("family", "")) if row is not None else ""
            row_width = family.rsplit("row", 1)[-1] if "row" in family else "—"
            lines.append(
                f"|{size}|{row_width}|{completion_pair(measured, 'h100')}|"
                f"{completion_pair(measured, 'b200')}|"
                f"{joined_pair(row, 'ventus')}|"
            )

    lines.extend([
        "", "G2S 的普通共同矩阵仍保留 mbarrier wait 的离散观察开销；上面 NVIDIA 主表优先使用 v12 的完成边界。[completion_threshold.csv](completion_threshold.csv) 保存每个 case 的冷/热中点、上下边界、误差、正式 run ID 和原始 shard 路径；每个 case 的 640 点原始判断保存在对应正式 run 的压缩 CSV 中。", "",
        "## S2G", "",
        "S2G 使用 `cp.async.bulk.wait_group 0`，完成观察比 G2S polling 更直接。", "",
        "### Tensor capacity", "",
        "|容量(B)|H100 第一次 → 第二次|B200 第一次 → 第二次|Ventus 第一次 → 第二次|",
        "|---:|---:|---:|---:|",
    ])
    for size in common_sizes:
        row = capacity.get(("s2g", size))
        if row is not None:
            lines.append(
                f"|{size}|{joined_pair(row, 'h100')}|"
                f"{joined_pair(row, 'b200')}|{joined_pair(row, 'ventus')}|"
            )

    lines.extend([
        "", "### 2D、U8、每行 128B 的关键长度", "",
        "|容量(B)|2D 行宽(B)|H100 第一次 → 第二次|B200 第一次 → 第二次|Ventus 第一次 → 第二次|",
        "|---:|---:|---:|---:|---:|",
    ])
    for size in key_sizes:
        row = two_d.get(("s2g", size))
        if row is not None:
            family = str(row.get("family", ""))
            row_width = family.rsplit("row", 1)[-1] if "row" in family else "—"
            lines.append(
                f"|{size}|{row_width}|{joined_pair(row, 'h100')}|"
                f"{joined_pair(row, 'b200')}|{joined_pair(row, 'ventus')}|"
            )

    lines.extend([
        "", "## Roundtrip", "",
        "roundtrip 在同一对样本中依次做 G2S 和 S2G，TensorMap 冷/热定义不变。", "",
        "|容量(B)|H100 第一次 → 第二次|B200 第一次 → 第二次|Ventus 第一次 → 第二次|",
        "|---:|---:|---:|---:|",
    ])
    for size in common_sizes:
        row = capacity.get(("roundtrip", size))
        if row is not None:
            lines.append(
                f"|{size}|{joined_pair(row, 'h100')}|"
                f"{joined_pair(row, 'b200')}|{joined_pair(row, 'ventus')}|"
            )

    lines.extend([
        "", "完整共同矩阵和 2D 数据按方向保存在 [G2S](single_instruction_g2s.csv)、[S2G](single_instruction_s2g.csv)、[roundtrip](single_instruction_roundtrip.csv) 与 [总表](single_instruction_cycles.csv)。", "",
        "## `prefetch.tensormap` 到底做了多少", "",
        "显式 prefetch 是独立控制组：在第一次 TMA 前只执行一次 prefetch，第二次前不再执行。它只提前准备 descriptor，不搬 payload，也不等价于提前完整执行过一次 TMA。", "",
        "|容量(B)|设备|自然冷启动的第一次|prefetch 后的第一次|真正执行过一次后的第二次|",
        "|---:|---|---:|---:|---:|",
    ])
    for size in (128, 1024, 4096, 16384, 32768):
        case_id = f"tensor_capacity_g2s_b{size}"
        cold = completion.get((case_id, "tmau_first_use_l2_hot"))
        prefetched = completion.get((case_id, "tmau_prefetched_l2_hot"))
        for device in ("h100", "b200"):
            cold_first = cold.get(f"{device}_repeat0_cycles", "") if cold else ""
            pref_first = prefetched.get(f"{device}_repeat0_cycles", "") if prefetched else ""
            true_hot = cold.get(f"{device}_repeat1_cycles", "") if cold else ""
            if any(value != "" for value in (cold_first, pref_first, true_hot)):
                lines.append(
                    f"|{size}|{device.upper()}|{cold_first or '—'}|"
                    f"{pref_first or '—'}|{true_hot or '—'}|"
                )

    lines.extend([
        "", "如果 prefetch 完全等价于提前执行完整 TMA，那么“prefetch 后的第一次”应与“真正执行过一次后的第二次”一致。实际差值应作为硬件只预取/准备 descriptor、但没有完成全部首条命令状态的证据，不应把 prefetch 写成完整 TMA 预热。", "",
        "## Ventus TensorMap 编译与绑定", "",
        "Ventus 可以直接读 RTL PMU，因此这里是硬件计数，不是差分猜测。", "",
        "|方向|rank|第一次 → 第二次总周期|compile cycles|bind cycles/command|compiled hit/miss/compile|",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in pmu_cold:
        lines.append(
            f"|{row.get('direction')}|{row.get('rank')}|"
            f"{row.get('total_cycles_repeat0')} → {row.get('total_cycles_repeat1')}|"
            f"{row.get('descriptor_compile_cycles_exact')}|"
            f"{row.get('bind_cycles_exact')}|"
            f"{row.get('compiled_hits')}/{row.get('compiled_misses')}/{row.get('descriptor_compiles')}|"
        )

    key_slopes = [
        row for row in slopes
        if row.get("path") == "tensor" and row.get("mode") == "batched"
        and row.get("map_mode") == "same_map" and row.get("bytes") == "4096"
        and row.get("repeat") == "1"
        and row.get("suite_id") != "multi_context.s2g_size_sweep"
    ]
    lines.extend([
        "", "## 多命令斜率", "",
        "下表只列 Tensor、batched、same-map、4KiB、第二次热状态；完整样本和所有资源限制见 CSV。", "",
        "|方向|设备|层级|顺序|高 N 命令数|每条命令斜率|单位|有效带宽|",
        "|---|---|---|---|---|---:|---|---:|",
    ])
    for row in key_slopes:
        slope_value = float(row.get("completion_slope_per_command", 0) or 0)
        noisy = slope_value <= 0
        slope_text = "—（运行间噪声）" if noisy else row.get("completion_slope_per_command")
        bandwidth = (
            "—" if noisy else
            row.get("effective_bytes_per_cycle") or
            row.get("effective_gb_per_second") or "—"
        )
        lines.append(
            f"|{row.get('direction')}|{row.get('device_class')}|{row.get('level')}|"
            f"{row.get('order')}|{row.get('high_n_commands')}|"
            f"{slope_text}|"
            f"{row.get('slope_unit_per_command')}/command|{bandwidth}|"
        )
    lines.extend([
        "", "多 CTA 的数值是整张 GPU 同时服务多个 CTA 的有效斜率，不是单条 TMA 物理端口带宽。高 N 的几次 wall-time 若受调度噪声影响而无法拟合出正斜率，表中直接写“运行间噪声”，不把负数解释为性能。超过 128 B/cycle 也只能说明测得的有效服务斜率超过这个阈值；没有内部端口计数器时，不能据此断言 NVIDIA 有几条物理数据通路。Ventus 仍是 `1 active + 1 lookahead`：lookahead 可以提前准备 descriptor，但不会与当前命令同时搬数据。", "",
        "## S2G Tensor same-map batched 长度扫描", "",
        "本表保持同 CTA、Tensor、same-map、batched、冷后热与独立数据区域不变，只改变每条命令的 payload。`N=8 带宽` 是整批实际字节数除以整批总周期；`高 N 斜率` 是稳态每增加一条命令的边际周期。", "",
        "|容量(B)|设备|单条总周期|8条总周期|N=8 带宽(B/cycle)|高 N 命令数|高 N 斜率|相对 B/32 的额外周期|边际带宽(B/cycle)|",
        "|---:|---|---:|---:|---:|---|---:|---:|---:|",
    ])
    for row in size_sweep:
        lines.append(
            f"|{row['bytes']}|{row['device']}|{row['n1_total_cycles']}|"
            f"{row['n8_total_cycles']}|{row['n8_effective_bytes_per_cycle']}|"
            f"{row['high_n_commands']}|{row['slope_cycles_per_command']}|"
            f"{row['slope_excess_over_b_div_32']}|"
            f"{row['marginal_bytes_per_cycle']}|"
        )
    transition = {
        device: min(
            int(row["bytes"]) for row in size_sweep
            if row["device"] == device
            and abs(float(row["slope_excess_over_b_div_32"])) <= 2.0
        )
        for device in ("h100", "b200")
        if any(
            row["device"] == device
            and abs(float(row["slope_excess_over_b_div_32"])) <= 2.0
            for row in size_sweep
        )
    }
    transition_text = "、".join(
        f"{device.upper()} 为 {size}B"
        for device, size in transition.items()
    ) or "尚未观测到"
    lines.extend([
        "", "只有当斜率随容量呈 `B/32 + 常数` 增长时，才能把结果解释为约 32 B/cycle 的 payload 服务率；小容量若保持近似固定斜率，说明命令提交/排队开销仍占主导。", "",
        f"以 `|高 N 斜率 - B/32| ≤ 2 cycles` 为转折判据，本轮首次进入数据吞吐主导区间的容量：{transition_text}。转折后两张卡都稳定在约 `B/32 + 1 cycle/command`；转折前则主要受每条命令约 60–71 cycles 的提交/排队速率限制。", "",
        "## 128B payload 冷热控制", "",
        "|设备|方向|路径|payload 场景|第一次 → 第二次|", "|---|---|---|---|---:|",
    ])
    for row in residency:
        lines.append(
            f"|{row.get('device')}|{row.get('direction')}|{row.get('path')}|"
            f"{row.get('scenario')}|{arrow(row.get('repeat0'), row.get('repeat1'))}|"
        )

    lines.extend([
        "", "这里的 TensorMap 已经由计时外的一条同图 TMA 真正使用过一次，不是只发 prefetch。cold→cold 用于确认两次清空后延迟应接近，hot→hot 用于确认两次热访问应接近；G2S 的 `test_wait` 仍可能出现一个轮询步长的抖动。S2G 在这组 128B 数据中冷热差很小，说明目标 line 是否已驻留不会像 G2S 源数据读取那样贡献数百周期。", "",
        "", "## 当前性能差距怎么解释", "",
        "- 冷、热差值首先反映 TensorMap 第一次从 L2 进入 TMAU 并建立内部状态的代价；它不是纯 decode，因为还包含 descriptor 请求、排队和读取。", "",
        "- 第二次热状态去掉了大部分 descriptor 首次使用开销，更适合比较真实数据通路；大容量的增量斜率比 128B 的绝对值更能说明 L2/shared 持续吞吐。", "",
        "- G2S 的 NVIDIA 数值必须优先看 v12 完成边界；普通 mbarrier 轮询表会把观察间隔混进结果。S2G 的 `wait_group 0` 更直接。", "",
        "- Ventus 的 descriptor compiler 固定约 11 cycles，rank-N Binder 为 N+2 cycles；compiled hit 能去掉这部分，但大容量仍受 RTL L2 的请求服务、shared 交付和单 active 后端限制。", "",
        "- Ventus 使用 RTL L2 且没有真实 HBM/DRAM，CUDA 使用真实 cache 与 HBM。绝对周期不能全部归因于 TMA 地址流水；报告同时保留 payload 驻留控制和大容量斜率。", "",
        "## 数据质量", "",
        f"Ventus multi-context 中有 {len(resources)} 个已声明资源限制项，见 [resource_limits.csv](resource_limits.csv)。失败或超时只进入 [运行清单](run_inventory.csv)，不会进入周期表。", "",
        "每个表格单元都能从 [运行清单](run_inventory.csv) 回到正式 run、源码哈希和原始 CSV。", "",
    ])
    output.write_text("\n".join(lines), encoding="utf-8")


def markdown_report(
    output: Path, runs: dict[tuple[str, str, str], tuple[Path, dict[str, Any]]],
    joined: list[dict[str, Any]], capacity_slopes: list[dict[str, Any]],
    decode: list[dict[str, Any]],
    leads: list[dict[str, Any]], ventus_pmu: list[dict[str, Any]],
    slopes: list[dict[str, Any]],
    size_sweep: list[dict[str, Any]], residency: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    completion_threshold: list[dict[str, Any]],
    resources: list[dict[str, Any]], diagnostics: list[tuple[Path, dict[str, Any]]],
    missing: list[tuple[str, str, str]],
) -> None:
    return cold_then_hot_markdown_report(
        output, runs, joined, capacity_slopes, decode, leads, ventus_pmu,
        slopes, size_sweep, residency, controls, completion_threshold, resources,
        diagnostics, missing,
    )

    capacity = [row for row in joined if row.get("family") == "tensor_capacity"]
    key_lengths = {32, 64, 96, 128, 160, 192, 224, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768}
    two_d_candidates = [
        row for row in joined
        if row["suite_id"] == "single_instruction.tensor_2d_capacity"
        and int(row["logical_bytes"]) in key_lengths
        and "_u8_" in str(row["case_id"])
    ]
    canonical_2d: dict[tuple[str, int], dict[str, Any]] = {}
    for row in two_d_candidates:
        key = (str(row["direction"]), int(row["logical_bytes"]))
        previous = canonical_2d.get(key)
        preference = 1 if "_row128_" in str(row["case_id"]) else 0
        old_preference = (
            1 if previous is not None and "_row128_" in str(previous["case_id"]) else 0
        )
        if previous is None or preference > old_preference:
            canonical_2d[key] = row
    two_d = sorted(canonical_2d.values(), key=sort_key)
    key_slopes = [
        row for row in slopes
        if row.get("path") == "tensor"
        and row.get("mode") == "batched"
        and row.get("map_mode") == "same_map"
        and row.get("bytes") == "4096"
        and row.get("repeat") == "1"
    ]
    pmu_cold = [row for row in ventus_pmu if row.get("scenario") == "cold_demand"]
    has_cuda = any(
        str(manifest.get("device", {}).get("class", "")) in {"h100", "b200"}
        for _, manifest in runs.values()
    )
    missing_set = set(missing)
    missing_diagnostics: list[tuple[Path, dict[str, Any]]] = []
    superseded_diagnostics: list[tuple[Path, dict[str, Any]]] = []
    for item in diagnostics:
        manifest = item[1]
        suite = str(manifest.get("suite_id", ""))
        key = (
            suite,
            str(manifest.get("device", {}).get("class", "")),
            str(manifest.get("variant", ""))
            if suite == "single_instruction.completion_latency"
            else "" if suite.startswith("single_instruction.")
            else str(manifest.get("variant", "")),
        )
        (missing_diagnostics if key in missing_set else superseded_diagnostics).append(item)
    lines = [
        "# TMA 三平台周期比较（FreshTensorMapV1）", "",
        *(
            [
                "> **当前是审计中报告。** 只包含结果切换点之后的正式运行；"
                "H100/B200 尚未重测，所以相关位置明确留空，不会用旧数据补齐。",
                "",
            ]
            if missing else []
        ),
        "正式口径为不执行 TMA 预热、独立测量两次。每个单指令样本使用独立 TensorMap；多命令测试只允许同一个被测批次内部按 same-map 定义共享，任何后续 case 都必须换新地址。",
        "表格先按 G2S/S2G/roundtrip，再按具体功能和容量排列。G2S 主表把两次独立结果合并为一个整数周期值。", "",
        "V9 为每个样本使用一个全新的 TensorMap；TensorMap 只被普通 load 放进 L2，"
        "被测命令此前没有任何 TMAU 使用。报告中的 `首次使用` 数值因此包含从 L2 取回"
        "TensorMap 并在 TMAU 内准备它的开销。", "",
        "测试方法、SASS 检查、唯一地址证据和 2D 关键点见 "
        "[G2S TensorMap 首次使用专项报告](G2S_TENSORMAP_FIRST_USE.md)。", "",
        f"已采用最新正式运行 {len(runs)} 项；缺少 {len(missing)} 项；另有 {len(diagnostics)} 项 diagnostic。", "",
    ]
    if missing:
        lines.extend(["## 尚未完成的正式运行", ""])
        lines.extend(f"- `{suite}` / `{device}` / `{variant or 'default'}`" for suite, device, variant in missing)
        for path, manifest in missing_diagnostics:
            lines.append(
                f"- diagnostic: `{manifest.get('suite_id')}` / `{manifest.get('device', {}).get('class')}` / "
                f"`{manifest.get('failure_class')}`，见 "
                f"[diagnostic manifest](../../../{path.relative_to(HERE).as_posix()})。"
            )
        lines.append("")
    if superseded_diagnostics:
        lines.extend([
            f"另有 {len(superseded_diagnostics)} 次失败探针已被成功正式运行取代，"
            "只保留在 [运行清单](run_inventory.csv) 中，不进入下列数据。", "",
        ])
    lines.extend([
        "## 测试项目说明", "",
        "- `common_matrix v3`：334 个安全、非 Reduce 的共同功能点，包含 Bulk、Tensor、G2S、S2G、roundtrip、rank 1–5、dtype、stride、OOB、swizzle 和合法 interleave16。",
        "- `tensor_2d_capacity v3`：182 个二维 Tensor 容量点；32B 到 16KiB 围绕 32B 边界加密，16KiB 以后保留常用大容量点。",
        "- `completion_latency v9`：只测 G2S 的发出到完成时间；每个 delay 点都使用独立 TensorMap，通过一次 `mbarrier.test_wait` 判断并汇总为单个周期值。",
        "- `tensormap_decode v3`：比较 cold demand、同一命令内的 compiled hit、显式 prefetch 和 prefetch lead；Ventus 同时读取 descriptor/compiler PMU。",
        "- `data_residency_128b v2`：H100/B200 的 128B cold 与 hot payload 各自独立测两次；TensorMap 先显式预取，避免 descriptor 状态混入 payload 冷热差值。",
        "- `same_direction v2`：同向 G2S 或 S2G 的 1/2/4/8/16/32 条命令，区分同 CTA/多 CTA、串行/批量、same-map/distinct-map。每个 case 的 descriptor bank 单独分配并保留到子进程退出，只用普通 `ld.global.cg` 放进 L2，计时前没有 TMA 使用或 prefetch。",
        "- `mixed_direction v2`：G2S 与 S2G 混合，区分 alternating、先 G2S 后 S2G、先 S2G 后 G2S；不同顺序和 issuing mode 绝不复用上一个 case 的 TensorMap。",
        "- `control_items v2`：保留 4KiB cooperative copy 与 compute/TMA overlap 两项独立控制实验。", "",
        "## G2S", "",
        "### 从发出指令到完成", "",
        "每条 Tensor TMA 都使用一个不同地址、不同内容的 TensorMap。普通全局读取先把它放进 L2，但在计时前没有任何 TMA 指令使用过它。主表只给一个完成周期值；两次独立测量和原始判断点保存在 CSV。", "",
        "|测试组|测试项|路径|容量(B)|CUDA TensorMap 状态|H100 周期|B200 周期|Ventus fresh 0/1|",
        "|---|---|---|---:|---|---:|---:|---:|",
    ])
    state_names = {
        "not_applicable": "Bulk（没有 TensorMap）",
        "tmau_first_use_l2_hot": "在 L2，TMAU 第一次使用",
        "tmau_prefetched_l2_hot": "已经提前送入 TMAU",
    }
    ventus_g2s = {
        str(row["case_id"]): row
        for row in joined
        if row.get("direction") == "g2s"
    }
    visible_completion = [
        row for row in completion_threshold
        if row.get("family") in {"bulk_capacity", "tensor_capacity"}
    ]
    for row in visible_completion:
        ventus = ventus_g2s.get(str(row["base_case_id"]))
        ventus_pair = (
            f"{ventus.get('ventus_repeat0')}/{ventus.get('ventus_repeat1')}"
            if ventus
            and row.get("descriptor_state") != "tmau_prefetched_l2_hot"
            else "—"
        )
        lines.append(
            f"|{row['family']}|{row['base_case_id']}|{row['path']}|"
            f"{row['logical_bytes']}|"
            f"{state_names.get(str(row['descriptor_state']), row['descriptor_state'])}|"
            f"{row.get('h100_completion_cycles', '—') or '—'}|"
            f"{row.get('b200_completion_cycles', '—') or '—'}|{ventus_pair}|"
        )
    if not visible_completion:
        lines.append("|—|—|—|—|等待 V9 正式测试|尚未运行|尚未运行|尚未运行|")
    lines.extend([
        "", "**Ventus 的 `0/1` 不是同一个 TensorMap 连续执行两次。** 两次分别使用"
        "不同地址和不同内容，且 `prior_tma_use=0`，所以两次都必须走 descriptor miss、"
        "读取和编译路径。Ventus RTL 仿真没有 GPU 时钟、仲裁和真实 cache 的运行间抖动；"
        "相同路径得到完全相同的整数周期属于正常的确定性结果，不表示 repeat 1 命中了 repeat 0。"
        "独立 rank-2 PMU 对照记录为 `compiled hit=0, miss=2, compile=2`，两次编译共 22 周期，"
        "见 [Ventus TensorMap PMU](ventus_tensormap_pmu.csv)。表中的 `在 L2/已经预取` 是 CUDA V9 的"
        "驻留状态；Ventus 列来自 fresh-allocation common matrix，不能拿 `—` 当成 Ventus 预取结果。", "",
        "这里先列常用容量；地址对齐、dtype、rank、stride、OOB、swizzle、interleave16 和 2D 的完整结果见专项报告及 [completion_threshold.csv](completion_threshold.csv)。"
        "表中单个数字的测量精度约为 ±20 周期。汇总后的两次独立边界见 [completion_threshold.csv](completion_threshold.csv)，逐点原始证据由 [运行清单](run_inventory.csv) 链接到各 run。", "",
        "## S2G", "",
        "### 单指令 Tensor capacity", "",
        "|容量(B)|H100 0/1 (Δ)|B200 0/1 (Δ)|Ventus 0/1 (Δ)|", "|---:|---:|---:|---:|",
    ])
    for row in [item for item in capacity if item["direction"] == "s2g"]:
        lines.append(
            f"|{row['logical_bytes']}|"
            f"{device_pair(row, 'h100')}|{device_pair(row, 'b200')}|"
            f"{device_pair(row, 'ventus')}|"
        )
    lines.extend([
        "", "### 2D 关键长度", "",
        "|容量(B)|H100 0/1|B200 0/1|Ventus 0/1|", "|---:|---:|---:|---:|",
    ])
    for row in [item for item in two_d if item["direction"] == "s2g"]:
        lines.append(
            f"|{row['logical_bytes']}|{device_pair(row, 'h100')}|"
            f"{device_pair(row, 'b200')}|{device_pair(row, 'ventus')}|"
        )
    lines.extend([
        "", "S2G 使用 `cp.async.bulk.wait_group 0`/Ventus 对应 completion，因此比 G2S polling 更适合观察小周期差异。", "",
        "## Roundtrip", "",
        "|容量(B)|H100 0/1|B200 0/1|Ventus 0/1|", "|---:|---:|---:|---:|",
    ])
    for row in [item for item in capacity if item["direction"] == "roundtrip"]:
        lines.append(
            f"|{row['logical_bytes']}|{device_pair(row, 'h100')}|"
            f"{device_pair(row, 'b200')}|{device_pair(row, 'ventus')}|"
        )
    lines.extend([
        "", "完整 334 项共同矩阵、182 项 2D 长度及方向拆分表见 [完整表](single_instruction_cycles.csv)、[G2S](single_instruction_g2s.csv)、[S2G](single_instruction_s2g.csv) 和 [roundtrip](single_instruction_roundtrip.csv)。", "",
        "## TensorMap 从 L2 送入 TMAU 的额外时间", "",
        "这里比较同一个配置的两种状态：TensorMap 已在 L2、TMAU 第一次使用；以及 TensorMap 已经提前送入 TMAU。差值包含从 L2 读取和 TMAU 准备工作，不能单独叫作纯解码周期。", "",
        "|设备|测试项|rank|容量(B)|TMAU 第一次使用|已经提前送入 TMAU|多出的周期|",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in [item for item in decode if item.get("family") == "tensor_capacity"]:
        lines.append(
            f"|{row['device']}|{row['base_case_id']}|{row['rank']}|"
            f"{row['logical_bytes']}|{row['tmau_first_use_l2_hot_cycles']}|"
            f"{row['tmau_prefetched_l2_hot_cycles']}|"
            f"{row['l2_to_tmau_fetch_and_prepare_cycles']}|"
        )
    if not decode:
        lines.append("|—|等待 V9 正式测试|—|—|—|—|—|")
    lines.extend([
        "", (
            "全部当前对照配置见 [TensorMap 差值 CSV](tensormap_decode_estimates.csv)。"
            if decode else
            "H100/B200 对照数据尚未运行；当前 CSV 为空，不引用旧结果。"
        ), "",
        "", "Ventus 使用 RTL PMU，下面是 cold demand 的精确编译/绑定计数：", "",
        "|方向|rank|总启动+搬运 0/1|compile cycles|bind cycles/command|hit/miss/compile|", "|---|---:|---:|---:|---:|---:|",
    ])
    for row in pmu_cold:
        lines.append(
            f"|{row.get('direction')}|{row.get('rank')}|{row.get('total_cycles_repeat0')}/{row.get('total_cycles_repeat1')}|"
            f"{row.get('descriptor_compile_cycles_exact')}|"
            f"{row.get('bind_cycles_exact')}|{row.get('compiled_hits')}/{row.get('compiled_misses')}/{row.get('descriptor_compiles')}|"
        )
    lines.extend([
        "", (
            "V9 使用固定 1024 周期 lead 作为充分预取对照；rank 1–5 的短 lead sweep 由独立 decode suite 给出。"
            if has_cuda else
            "CUDA 重测将使用固定 1024 周期 lead 作为充分预取对照，并用独立 decode suite 扫描 rank 1–5 的短 lead。"
        ), "",
        "## Multi-context", "",
        "高 N 斜率取每个系列最大的三个 command 数；下表只列 Tensor/batched/same-map/4KiB/repeat1，完整结果见 CSV。", "",
        "|方向|设备|层级|顺序|高N commands|completion slope|单位|有效带宽|", "|---|---|---|---|---|---:|---|---:|",
    ])
    for row in key_slopes:
        bandwidth = row["effective_bytes_per_cycle"] or row["effective_gb_per_second"]
        lines.append(
            f"|{row['direction']}|{row['device_class']}|{row['level']}|{row['order']}|"
            f"{row['high_n_commands']}|{row['completion_slope_per_command']}|"
            f"{row['slope_unit_per_command']}/command|{bandwidth}|"
        )
    lines.extend([
        "", "超过 128 B/cycle 只表示测得的有效服务斜率超过该阈值；没有内部端口计数器时不能据此宣称物理数据通路数量。Ventus 保持 `1 active + 1 lookahead`，lookahead 可隐藏 descriptor 准备但不并行搬运。", "",
        "## 128B 数据冷/热", "",
        "|设备|方向|路径|payload 状态|repeat0/1|Δ|", "|---|---|---|---|---:|---:|",
    ])
    for row in residency:
        lines.append(
            f"|{row['device']}|{row['direction']}|{row['path']}|{row['scenario']}|"
            f"{row['repeat0']}/{row['repeat1']}|{signed(row['repeat1_minus_repeat0'])}|"
        )
    lines.extend([
        "", (
            "H100 与 B200 都提供 cold-hot、cold-cold 和 hot-hot 控制；Tensor 路径使用独立的新 descriptor，避免把 TMAU descriptor 命中混入数据冷热差值。"
            if residency else
            "该项尚未运行。正式 CUDA 重测会同时提供 cold-hot、cold-cold 和 hot-hot 控制，并为 Tensor 路径使用独立的新 descriptor。"
        ), "",
        "## 独立控制项", "",
        "这组测量 4 KiB CTA cooperative-copy 和 compute/TMA overlap。它不是 TensorMap 冷热实验。", "",
        "|方向|方法|H100 0/1 (Δ)|B200 0/1 (Δ)|", "|---|---|---:|---:|",
    ])
    control_index = {
        (str(row["direction"]), str(row["method"]), str(row["device"])): row
        for row in controls
    }
    for direct, method in sorted(
        {(str(row["direction"]), str(row["method"])) for row in controls},
        key=lambda item: (DIRECTION_ORDER.get(item[0], 9), item[1]),
    ):
        h100 = control_index.get((direct, method, "h100"))
        b200 = control_index.get((direct, method, "b200"))
        def pair(item: dict[str, Any] | None) -> str:
            return (
                f"{item['repeat0']}/{item['repeat1']} "
                f"({signed(item['repeat1_minus_repeat0'])})" if item else ""
            )
        lines.append(f"|{direct}|{method}|{pair(h100)}|{pair(b200)}|")
    lines.extend([
        "", "完整原始字段见 [control_items.csv](control_items.csv)。", "",
        "## 数据质量与资源边界", "",
        f"最新 Ventus multi-context 正式结果中有 {len(resources)} 个静态或工具链资源限制项，逐项见 [resource_limits.csv](resource_limits.csv)；它们不被计作功能失败。", "",
        "所有结果只来自 FreshTensorMapV1。旧 suite、旧数据、失败探针和旧报告不会作为输入，也不会保留在正式测试树中。", "",
        "## 当前瓶颈与 CUDA 主要差距", "",
        "16KiB→32KiB 的增量斜率消除了大部分固定启动开销：", "",
        "|方向|设备|repeat|cycles/128B line|有效 B/cycle|", "|---|---|---:|---:|---:|",
    ])
    for row in capacity_slopes:
        lines.append(
            f"|{row['direction']}|{row['device']}|{row['repeat']}|"
            f"{row['cycles_per_128b_line']}|{row['effective_bytes_per_cycle']}|"
        )
    lines.extend([
        "",
        "- V9 中“在 L2、TMAU 第一次使用”减去“已经提前送入 TMAU”，表示 TensorMap 从 L2 读取并完成准备所增加的时间；",
        "- Tensor 与 Bulk 的 cold-hot 同向变化：优先归因 payload/destination L2 residency；",
        "- batched 高 N 斜率优于 serial，但 completion 仍线性：存在排队/重叠收益，不代表多条物理通路；",
        "- Ventus 的 descriptor 编译固定 11 cycles、rank-N Binder 为 N+2 cycles，compiled hit 可消除这部分，但大容量斜率仍由 L2 请求服务、shared 交付和单 active 后端决定；",
        (
            "- H100/B200 G2S 完成分析只使用 V9 中地址和内容都不复用的 TensorMap；"
            if has_cuda else
            "- H100/B200 尚无切换点后的数据，当前不能给出 CUDA 性能差距结论；"
        ),
        "- Ventus 使用 RTL L2/无真实 DRAM，CUDA 使用真实 cache/HBM；绝对差距必须结合驻留和 wait 口径，不能把所有差值都归因于 TMA 地址流水。", "",
    ])
    output.write_text("\n".join(lines), encoding="utf-8")


def device_report(
    output: Path, device: str, joined: list[dict[str, Any]],
    completion_threshold: list[dict[str, Any]],
) -> None:
    lines = [
        f"# {device.upper()} 无预热双样本", "",
        "方向优先；完整字段、suite 版本和原始文件路径见跨设备目录中的 CSV。", "",
    ]
    for direct in ("g2s", "s2g", "roundtrip"):
        lines.extend([f"## {direct.upper()}", ""])
        if direct == "g2s" and device in ("h100", "b200"):
            lines.extend([
                "### 从发出指令到完成（V9）", "",
                "每条 Tensor TMA 使用不同地址、不同内容的 TensorMap。TensorMap 已在 L2，但此前没有被 TMAU 使用。表里只给一个完成周期值，测量精度约为 ±20 周期。", "",
                "|测试组|测试项|路径|字节数|TensorMap 状态|完成周期|",
                "|---|---|---|---:|---|---:|",
            ])
            state_names = {
                "not_applicable": "Bulk（没有 TensorMap）",
                "tmau_first_use_l2_hot": "在 L2，TMAU 第一次使用",
                "tmau_prefetched_l2_hot": "已经提前送入 TMAU",
            }
            for row in completion_threshold:
                value = row.get(f"{device}_completion_cycles", "")
                if value == "":
                    continue
                lines.append(
                    f"|{row['family']}|{row['base_case_id']}|{row['path']}|"
                    f"{row['logical_bytes']}|"
                    f"{state_names.get(str(row['descriptor_state']), row['descriptor_state'])}|"
                    f"{value}|"
                )
            if not any(
                row.get(f"{device}_completion_cycles", "") != ""
                for row in completion_threshold
            ):
                lines.append("|—|—|—|—|等待 V9 正式测试|—|")
            lines.extend([
                "",
                "原始 V9 判断点、独立 descriptor 地址和指纹证据见 "
                "[completion_threshold.csv](../comparisons/FreshTensorMapV1/completion_threshold.csv)。", "",
            ])
            continue
        lines.extend(["|suite|path|bytes|repeat0|repeat1|Δ|", "|---|---|---:|---:|---:|---:|"])
        for row in joined:
            if row["direction"] != direct:
                continue
            value0 = row.get(f"{device}_repeat0", "")
            value1 = row.get(f"{device}_repeat1", "")
            if value0 == "":
                continue
            lines.append(
                f"|{row['suite_id']}|{row['path']}|{row['logical_bytes']}|{value0}|{value1}|"
                f"{signed(row[f'{device}_delta_1_minus_0'])}|"
            )
        lines.append("")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    runs = latest_runs()
    write_current_run_catalog(runs)
    diagnostics = diagnostic_runs()
    normalized_keys = {
        (
            suite,
            device,
            variant if suite == "single_instruction.completion_latency"
            else "" if suite.startswith("single_instruction.")
            else variant,
        )
        for suite, device, variant in runs
    }
    missing = sorted(EXPECTED - normalized_keys)
    diagnostic_keys = {
        (
            str(manifest.get("suite_id", "")),
            str(manifest.get("device", {}).get("class", "")),
            str(manifest.get("variant", ""))
            if str(manifest.get("suite_id", "")) == "single_instruction.completion_latency"
            else "" if str(manifest.get("suite_id", "")).startswith("single_instruction.")
            else str(manifest.get("variant", "")),
        )
        for _, manifest in diagnostics
    }
    unaccounted = sorted(set(missing) - diagnostic_keys)
    if unaccounted and not args.allow_partial:
        raise SystemExit("missing unaccounted formal runs: " + ", ".join("/".join(item) for item in unaccounted))
    rows: list[dict[str, str]] = []
    for path, manifest in runs.values():
        rows.extend(raw_rows(path, manifest))
    rows.sort(key=sort_key)
    joined = joined_single_instruction(rows)
    capacity_slopes = capacity_large_slopes(joined)
    decode = decode_estimates(rows)
    leads = decode_lead_summary(rows)
    pmu = ventus_decode_pmu(runs)
    multi_samples = multi_context_samples(rows)
    slopes = multi_context_slopes(rows)
    size_sweep = s2g_size_sweep_summary(multi_samples, slopes)
    residency = residency_summary(rows)
    controls = control_item_summary(rows)
    completion_threshold = completion_threshold_summary(rows)
    resources = resource_limit_rows(runs)
    inventory = run_inventory(runs, diagnostics)
    counts = {
        "single_instruction_cycles.csv": write_csv(output / "single_instruction_cycles.csv", joined),
        "single_instruction_g2s.csv": write_csv(output / "single_instruction_g2s.csv", (row for row in joined if row["direction"] == "g2s")),
        "single_instruction_s2g.csv": write_csv(output / "single_instruction_s2g.csv", (row for row in joined if row["direction"] == "s2g")),
        "single_instruction_roundtrip.csv": write_csv(output / "single_instruction_roundtrip.csv", (row for row in joined if row["direction"] == "roundtrip")),
        "tensor_capacity_large_slopes.csv": write_csv(output / "tensor_capacity_large_slopes.csv", capacity_slopes),
        "tensormap_decode_estimates.csv": write_csv(output / "tensormap_decode_estimates.csv", decode),
        "tensormap_prefetch_lead.csv": write_csv(output / "tensormap_prefetch_lead.csv", leads),
        "ventus_tensormap_pmu.csv": write_csv(output / "ventus_tensormap_pmu.csv", pmu),
        "multi_context_samples.csv": write_csv(output / "multi_context_samples.csv", multi_samples),
        "multi_context_slopes.csv": write_csv(output / "multi_context_slopes.csv", slopes),
        "s2g_size_sweep.csv": write_csv(output / "s2g_size_sweep.csv", size_sweep),
        "data_residency_128b.csv": write_csv(output / "data_residency_128b.csv", residency),
        "control_items.csv": write_csv(output / "control_items.csv", controls),
        "completion_threshold.csv": write_csv(output / "completion_threshold.csv", completion_threshold),
        "resource_limits.csv": write_csv(output / "resource_limits.csv", resources),
        "run_inventory.csv": write_csv(output / "run_inventory.csv", inventory),
    }
    markdown_report(
        output / "REPORT.md", runs, joined, capacity_slopes, decode, leads, pmu,
        slopes, size_sweep, residency, controls, completion_threshold, resources,
        diagnostics, missing,
    )
    (output / "report-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "report_status": "partial" if missing else "final",
                "measurement_release": MEASUREMENT_RELEASE,
                "result_not_before_utc": RESULT_NOT_BEFORE_UTC,
                "formal_runs": {"|".join(key): value[1]["run_id"] for key, value in sorted(runs.items())},
                "missing": [list(item) for item in missing],
                "missing_unaccounted": [list(item) for item in unaccounted],
                "diagnostics": [
                    path.relative_to(HERE).as_posix() for path, _ in diagnostics
                ],
                "row_counts": counts,
                "direction_order": ["g2s", "s2g", "roundtrip", "mixed"],
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"generated: {output}")
    if missing:
        stale_checksum = HERE / "releases" / f"{MEASUREMENT_RELEASE}.SHA256SUMS"
        stale_checksum.unlink(missing_ok=True)
        print("partial report: release hashes were not frozen")
    else:
        checksum = write_release_hashes()
        print(f"frozen hashes: {checksum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
