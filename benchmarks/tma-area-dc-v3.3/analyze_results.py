#!/usr/bin/env python3
"""Parse and gate V3.2/V3.3 standalone and Descriptor Service DC results.

The emitted JSON is the machine-readable DC release gate consumed by the
V3.3 baseline freezer.  A result is not releasable unless all mapped designs
meet 1.5 GHz, the release standalone/WindowEngine area reductions meet their
targets, and the desc4 increment remains a low-single-digit fraction of the
release TMA.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def match_number(text: str, label: str) -> float:
    match = re.search(
        rf"^{re.escape(label)}:\s+([0-9.]+)", text, re.MULTILINE
    )
    if not match:
        raise ValueError(f"missing {label!r}")
    return float(match.group(1))


def parse_variant(name: str) -> dict[str, float | int | str]:
    report = RESULTS / name / "report"
    area_text = (report / "area.rpt").read_text(encoding="utf-8")
    qor_text = (report / "qor_summary.rpt").read_text(encoding="utf-8")
    timing_text = (report / "timing_setup.rpt").read_text(encoding="utf-8")
    path_slacks = [
        float(value)
        for value in re.findall(
            r"slack \((?:MET|VIOLATED)\)\s+(-?[0-9.]+)", timing_text
        )
    ]
    qor_wns = re.findall(r"Design\s+WNS:\s+(-?[0-9.]+)", qor_text)
    return {
        "variant": name,
        "ports": int(match_number(area_text, "Number of ports")),
        "cells": int(match_number(area_text, "Number of cells")),
        "combinational_cells": int(
            match_number(area_text, "Number of combinational cells")
        ),
        "sequential_cells": int(
            match_number(area_text, "Number of sequential cells")
        ),
        "combinational_area_um2": match_number(
            area_text, "Combinational area"
        ),
        "sequential_area_um2": match_number(
            area_text, "Noncombinational area"
        ),
        "cell_area_um2": match_number(area_text, "Total cell area"),
        "wns_ns": (
            min(path_slacks)
            if path_slacks
            else (float(qor_wns[-1]) if qor_wns else math.nan)
        ),
    }


def parse_hierarchical_area(name: str, leaf: str) -> float:
    path = RESULTS / name / "report" / "area_hierarchy.rpt"
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"^\s*(?:\S*/)?{re.escape(leaf)}\s+"
        r"([0-9.]+)\s+[0-9.]+\s+[0-9.]+\s+[0-9.]+\s+[0-9.]+",
        re.MULTILINE,
    )
    values = [float(value) for value in pattern.findall(text)]
    if not values:
        raise ValueError(f"missing hierarchical cell {leaf!r} in {path}")
    # A uniquified hierarchy may expose more than one matching leaf.  The
    # standalone architecture instantiates one engine, so select the largest
    # aggregate rather than a possible nested/generated helper with that name.
    return max(values)


def main() -> None:
    configs = json.loads((ROOT / "configs.json").read_text(encoding="utf-8"))
    names = [item["name"] for item in configs["variants"]]
    rows = [parse_variant(name) for name in names]
    RESULTS.mkdir(exist_ok=True)
    output = RESULTS / "area_summary.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_name = {str(row["variant"]): row for row in rows}

    def area(name: str) -> float:
        return float(by_name[name]["cell_area_um2"])

    reference = area("v3_2_matched_reference")
    release = area("v3_3_release")
    desc2 = area("descriptor2")
    desc4 = area("descriptor4")
    old_engine = parse_hierarchical_area(
        "v3_2_matched_reference", "engine"
    )
    new_engine = parse_hierarchical_area("v3_3_release", "engine")
    release_reduction = (1.0 - release / reference) * 100.0
    engine_reduction = (1.0 - new_engine / old_engine) * 100.0
    descriptor_delta = desc4 - desc2
    descriptor_release_percent = descriptor_delta / release * 100.0
    failed_timing = [
        str(row["variant"]) for row in rows if float(row["wns_ns"]) < 0.0
    ]
    gates = {
        "all_1p5ghz_wns_nonnegative": not failed_timing,
        "standalone_area_reduction_at_least_25_percent":
            release_reduction >= 25.0,
        "window_engine_area_reduction_at_least_35_percent":
            engine_reduction >= 35.0,
        "desc4_increment_at_most_5_percent_of_release":
            descriptor_release_percent <= 5.0,
    }
    summary = {
        "schema_version": 1,
        "technology": "TSMC N12 CLN12FFCLL",
        "library": "tcbn12ffcllbwp16p90cpdtt1v85c",
        "corner": "TT 1.0V 85C",
        "frequency_mhz": 1500,
        "variants": by_name,
        "v3_2_matched_area_um2": reference,
        "v3_3_release_area_um2": release,
        "standalone_area_reduction_percent": release_reduction,
        "v3_2_window_engine_area_um2": old_engine,
        "v3_3_window_engine_area_um2": new_engine,
        "window_engine_area_reduction_percent": engine_reduction,
        "descriptor2_area_um2": desc2,
        "descriptor4_area_um2": desc4,
        "descriptor2_to_4_delta_um2": descriptor_delta,
        "descriptor2_to_4_delta_percent_of_release":
            descriptor_release_percent,
        "failed_timing_variants": failed_timing,
        "gates": gates,
        "area_gate_passed": all(gates.values()),
    }
    summary_path = RESULTS / "area_gate_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {output}")
    print(f"wrote {summary_path}")
    print(
        "V3.3 release vs V3.2 matched: "
        f"{release - reference:.3f} um^2 "
        f"({release / reference:.3f}x, "
        f"{release_reduction:.2f}% reduction)"
    )
    print(
        "WindowEngine V3.3 release vs V3.2 matched: "
        f"{new_engine - old_engine:.3f} um^2 "
        f"({new_engine / old_engine:.3f}x, "
        f"{engine_reduction:.2f}% reduction)"
    )
    print(
        "Descriptor 2->4: "
        f"{descriptor_delta:.3f} um^2 "
        f"({desc4 / desc2:.3f}x, "
        f"{descriptor_release_percent:.2f}% of release TMA)"
    )
    print(
        "V3.3 area-min/release/capacity: "
        f"{area('v3_3_area_min'):.3f}/"
        f"{release:.3f}/"
        f"{area('v3_3_capacity'):.3f} um^2"
    )
    failed_gates = [name for name, passed in gates.items() if not passed]
    if failed_gates:
        raise SystemExit(f"V3.3 DC release gate failed: {failed_gates}")


if __name__ == "__main__":
    main()
