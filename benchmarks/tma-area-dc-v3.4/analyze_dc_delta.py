#!/usr/bin/env python3
"""Compare mapped V3.3 and V3.4 standalone-TMA DC reports.

Background:
  RTL register-bit audits do not show whether storage compression has moved
  area into muxes, shifters, or timing-driven cells.  This comparison reads
  the two matched N12/1.5 GHz reports and records the mapped deltas.

Flow:
  Parse top-level cell/area counts, hierarchical local and absolute areas,
  estimated DesignWare operator areas, and the reported setup paths.  Emit a
  JSON record and a module CSV used by the V3.4 report.

Usage:
  python3 benchmarks/tma-area-dc-v3.4/analyze_dc_delta.py

Maintenance:
  Inputs must remain final ``report_area``/``report_timing`` outputs produced
  with the same library, corner, clock constraints, and compile recipe.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
V33 = HERE / "results/remote-v3_3-converged/v3_3_release/report"
V34 = HERE / "results/v3_4_release_40x6_desc4/report"
LEGACY = HERE / "results/remote-v2-completion-reduce/report"
OUTPUT = HERE / "results/v3_3_v3_4_mapped_delta.json"
MODULE_CSV = HERE / "results/v3_3_v3_4_module_area.csv"

# Matched logical hierarchy.  V3.3 has one extra TmaV2DmaCore wrapper between
# tma and the two functional children; V3.4 has the same functional split
# directly below tma.  The wrapper's *local* area is therefore folded into the
# logical tma row so that every parent can be reconstructed as:
#   hierarchical total = self total + direct-child hierarchical totals.
LOGICAL_HIERARCHY = (
    {
        "path": "tma",
        "design": "tma",
        "parent": "",
        "children": ("tma/Ingress", "tma/WindowSubsystem"),
        "v3_3_folded_local": ("TmaV2DmaCore",),
    },
    {
        "path": "tma/Ingress",
        "design": "TmaV3Ingress",
        "parent": "tma",
        "children": (
            "tma/Ingress/CommandBinder",
            "tma/Ingress/WindowPlanner",
        ),
    },
    {
        "path": "tma/Ingress/CommandBinder",
        "design": "TmaV2CommandBinder",
        "parent": "tma/Ingress",
        "children": (),
    },
    {
        "path": "tma/Ingress/WindowPlanner",
        "design": "TmaV2WindowPlanner",
        "parent": "tma/Ingress",
        "children": (),
    },
    {
        "path": "tma/WindowSubsystem",
        "design": "TmaV2WindowSubsystem",
        "parent": "tma",
        "children": (
            "tma/WindowSubsystem/DescriptorService",
            "tma/WindowSubsystem/WindowEngine",
        ),
    },
    {
        "path": "tma/WindowSubsystem/DescriptorService",
        "design": "TmaV2DescriptorService",
        "parent": "tma/WindowSubsystem",
        "children": (
            "tma/WindowSubsystem/DescriptorService/DescriptorCompiler",
        ),
    },
    {
        "path": (
            "tma/WindowSubsystem/DescriptorService/DescriptorCompiler"
        ),
        "design": "TmaV2DescriptorCompiler",
        "parent": "tma/WindowSubsystem/DescriptorService",
        "children": (),
    },
    {
        "path": "tma/WindowSubsystem/WindowEngine",
        "design": "TmaV2WindowEngine",
        "parent": "tma/WindowSubsystem",
        "children": (),
    },
)


def read(report: Path, name: str) -> str:
    path = report / name
    if not path.is_file():
        raise SystemExit(f"missing DC report: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def number(text: str, label: str, integer: bool = False) -> float | int:
    match = re.search(rf"^{re.escape(label)}:\s+([0-9.]+)", text, re.M)
    if not match:
        raise SystemExit(f"cannot parse {label}")
    return int(float(match.group(1))) if integer else float(match.group(1))


def parse_top(report: Path) -> dict[str, float | int]:
    text = read(report, "area.rpt")
    return {
        "ports": number(text, "Number of ports", True),
        "nets": number(text, "Number of nets", True),
        "cells": number(text, "Number of cells", True),
        "combinational_cells":
            number(text, "Number of combinational cells", True),
        "sequential_cells":
            number(text, "Number of sequential cells", True),
        "buf_inv": number(text, "Number of buf/inv", True),
        "combinational_area": number(text, "Combinational area"),
        "noncombinational_area": number(text, "Noncombinational area"),
        "total_cell_area": number(text, "Total cell area"),
    }


def parse_hierarchy(report: Path) -> dict[str, dict[str, float]]:
    text = read(report, "area_hierarchy.rpt")
    result: dict[str, dict[str, float]] = {}
    row = re.compile(
        r"^(\S+)\s+([0-9.]+)\s+([0-9.]+)\s+"
        r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\S+)\s*$",
        re.M,
    )
    for match in row.finditer(text):
        path, absolute, percent, comb, noncomb, _, design = match.groups()
        result[design] = {
            "path": path,
            "absolute_total": float(absolute),
            "percent_total": float(percent),
            "local_combinational": float(comb),
            "local_noncombinational": float(noncomb),
        }
    return result


def parse_synthetic(report: Path) -> dict[str, dict[str, float | int]]:
    text = read(report, "area_designware.rpt")
    result: dict[str, dict[str, float | int]] = {}
    row = re.compile(
        r"^\s*(DW[A-Za-z0-9_]+)\s+\S+\s+(\d+)\s+"
        r"([0-9.]+)\s+[0-9.]+%\s*$",
        re.M,
    )
    for name, count, area in row.findall(text):
        entry = result.setdefault(name, {"count": 0, "area": 0.0})
        entry["count"] = int(entry["count"]) + int(count)
        entry["area"] = float(entry["area"]) + float(area)
    return result


def parse_timing(report: Path) -> dict[str, object]:
    text = read(report, "timing_setup.rpt")
    starts = re.findall(r"^  Startpoint: (.+)$", text, re.M)
    endpoints = re.findall(r"^  Endpoint: (.+)$", text, re.M)
    slacks = [
        float(value)
        for value in re.findall(
            r"^  slack \((?:MET|VIOLATED)\)\s+(-?[0-9.]+)",
            text,
            re.M,
        )
    ]

    def owner(point: str) -> str:
        for module in (
            "windowPlanner", "engine", "descriptor", "binder", "ingress"
        ):
            if f"/{module}/" in point or f"_{module}/" in point:
                return module
        return point.split("/", 1)[0]

    return {
        "wns_ns": min(slacks),
        "reported_paths": len(slacks),
        "start_owner_counts": dict(Counter(map(owner, starts))),
        "endpoint_owner_counts": dict(Counter(map(owner, endpoints))),
        "first_path": {
            "startpoint": starts[0],
            "endpoint": endpoints[0],
            "slack_ns": slacks[0],
        },
    }


def delta(old: float | int, new: float | int) -> dict[str, float]:
    old_value = float(old)
    new_value = float(new)
    return {
        "absolute": new_value - old_value,
        "percent": (new_value / old_value - 1.0) * 100.0,
    }


def build_accounting_rows(
    old_hier: dict[str, dict[str, float]],
    new_hier: dict[str, dict[str, float]],
) -> list[dict[str, object]]:
    """Build non-overlapping hierarchy rows with explicit area semantics."""
    by_path = {entry["path"]: entry for entry in LOGICAL_HIERARCHY}
    rows: list[dict[str, object]] = []

    for entry in LOGICAL_HIERARCHY:
        design = str(entry["design"])
        old = old_hier[design]
        new = new_hier[design]
        children = tuple(str(path) for path in entry["children"])
        child_designs = [
            str(by_path[path]["design"])
            for path in children
        ]
        old_folded = tuple(
            str(name) for name in entry.get("v3_3_folded_local", ())
        )

        old_self_comb = float(old["local_combinational"]) + sum(
            float(old_hier[name]["local_combinational"])
            for name in old_folded
        )
        old_self_seq = float(old["local_noncombinational"]) + sum(
            float(old_hier[name]["local_noncombinational"])
            for name in old_folded
        )
        new_self_comb = float(new["local_combinational"])
        new_self_seq = float(new["local_noncombinational"])
        old_self_total = old_self_comb + old_self_seq
        new_self_total = new_self_comb + new_self_seq
        old_children_total = sum(
            float(old_hier[name]["absolute_total"])
            for name in child_designs
        )
        new_children_total = sum(
            float(new_hier[name]["absolute_total"])
            for name in child_designs
        )
        old_total = float(old["absolute_total"])
        new_total = float(new["absolute_total"])
        old_report_path = str(old["path"])
        if old_folded:
            old_report_path += " + " + " + ".join(
                f"{old_hier[name]['path']} local-only wrapper"
                for name in old_folded
            )

        if not entry["parent"]:
            row_kind = "root_parent"
        elif children:
            row_kind = "parent"
        else:
            row_kind = "leaf"

        rows.append({
            "logical_path": entry["path"],
            "depth": str(entry["path"]).count("/"),
            "row_kind": row_kind,
            "module_design": design,
            "parent_logical_path": entry["parent"],
            "direct_children": ";".join(children),
            "area_accounting": (
                "hierarchical_total=self_total+direct_children_total;"
                "do_not_sum_parent_and_child_rows"
                if children else
                "leaf:hierarchical_total=self_total"
            ),
            "v3_3_report_path": old_report_path,
            "v3_4_report_path": new["path"],
            "v3_3_hierarchical_total": old_total,
            "v3_4_hierarchical_total": new_total,
            "hierarchical_total_delta": new_total - old_total,
            "hierarchical_total_delta_percent":
                (new_total / old_total - 1.0) * 100.0,
            "v3_3_self_comb": old_self_comb,
            "v3_4_self_comb": new_self_comb,
            "self_comb_delta": new_self_comb - old_self_comb,
            "v3_3_self_seq": old_self_seq,
            "v3_4_self_seq": new_self_seq,
            "self_seq_delta": new_self_seq - old_self_seq,
            "v3_3_self_total": old_self_total,
            "v3_4_self_total": new_self_total,
            "self_total_delta": new_self_total - old_self_total,
            "v3_3_direct_children_total": old_children_total,
            "v3_4_direct_children_total": new_children_total,
            "v3_3_reconstruction_error":
                old_total - old_self_total - old_children_total,
            "v3_4_reconstruction_error":
                new_total - new_self_total - new_children_total,
        })

    return rows


def format_accounting_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Use report-level precision in CSV instead of binary-float tails."""
    formatted: list[dict[str, object]] = []
    for row in rows:
        output: dict[str, object] = {}
        for key, value in row.items():
            if isinstance(value, float):
                digits = 3 if key.endswith("_percent") else 4
                output[key] = f"{value:.{digits}f}"
            else:
                output[key] = value
        formatted.append(output)
    return formatted


def main() -> None:
    old_top, new_top = parse_top(V33), parse_top(V34)
    legacy_top = parse_top(LEGACY)
    old_hier, new_hier = parse_hierarchy(V33), parse_hierarchy(V34)
    old_syn, new_syn = parse_synthetic(V33), parse_synthetic(V34)
    legacy_syn = parse_synthetic(LEGACY)
    accounting_rows = build_accounting_rows(old_hier, new_hier)
    modules = sorted(set(old_hier) & set(new_hier))
    module_rows: list[dict[str, object]] = []
    for module in modules:
        old = old_hier[module]
        new = new_hier[module]
        module_rows.append({
            "module": module,
            "v3_3_total": old["absolute_total"],
            "v3_4_total": new["absolute_total"],
            "total_delta": (
                new["absolute_total"] - old["absolute_total"]
            ),
            "v3_3_local_comb": old["local_combinational"],
            "v3_4_local_comb": new["local_combinational"],
            "local_comb_delta": (
                new["local_combinational"]
                - old["local_combinational"]
            ),
            "v3_3_local_seq": old["local_noncombinational"],
            "v3_4_local_seq": new["local_noncombinational"],
            "local_seq_delta": (
                new["local_noncombinational"]
                - old["local_noncombinational"]
            ),
        })

    synthetic: dict[str, object] = {}
    for operator in sorted(set(old_syn) | set(new_syn)):
        old = old_syn.get(operator, {"count": 0, "area": 0.0})
        new = new_syn.get(operator, {"count": 0, "area": 0.0})
        synthetic[operator] = {
            "v3_3": old,
            "v3_4": new,
            "area_delta": float(new["area"]) - float(old["area"]),
        }

    result = {
        "conditions": {
            "process": "TSMC N12",
            "library": "tcbn12ffcllbwp16p90cpdtt1v85c",
            "corner": "TT 1.0V 85C",
            "frequency_mhz": 1500,
        },
        "v3_3": old_top,
        "v3_4": new_top,
        "legacy_v2_unmatched_corner": {
            **legacy_top,
            "library": "tcbn12ffcllbwp16p90cpdtt0p8v25c",
            "corner": "TT 0.8V 25C",
            "synthesis_frequency_mhz": 1600,
            "report_frequency_mhz": 1500,
            "line_entries": 4,
            "descriptor_entries": 2,
        },
        "top_delta": {
            key: delta(old_top[key], new_top[key])
            for key in old_top
        },
        "v3_4_vs_legacy_unmatched_corner": {
            key: delta(legacy_top[key], new_top[key])
            for key in legacy_top
        },
        "hierarchy": module_rows,
        "hierarchy_accounting": accounting_rows,
        "synthetic_operators": synthetic,
        "legacy_synthetic_operators": legacy_syn,
        "timing": {
            "legacy_unmatched_corner": parse_timing(LEGACY),
            "v3_3": parse_timing(V33),
            "v3_4": parse_timing(V34),
        },
    }
    OUTPUT.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    with MODULE_CSV.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output, fieldnames=list(accounting_rows[0])
        )
        writer.writeheader()
        writer.writerows(format_accounting_rows(accounting_rows))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
