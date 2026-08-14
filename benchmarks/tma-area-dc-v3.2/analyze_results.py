#!/usr/bin/env python3
"""Parse the three standalone TMA V3.2 DC points."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def match_number(text: str, label: str) -> float:
    match = re.search(rf"^{re.escape(label)}:\s+([0-9.]+)", text, re.MULTILINE)
    if not match:
        raise ValueError(f"missing {label!r}")
    return float(match.group(1))


def parse_variant(name: str) -> dict[str, float | int | str]:
    area_text = (RESULTS / name / "report" / "area.rpt").read_text()
    qor_text = (RESULTS / name / "report" / "qor_summary.rpt").read_text()
    wns_matches = re.findall(r"Design\s+WNS:\s+(-?[0-9.]+)", qor_text)
    return {
        "variant": name,
        "ports": int(match_number(area_text, "Number of ports")),
        "nets": int(match_number(area_text, "Number of nets")),
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
        "wns_ns": float(wns_matches[-1]) if wns_matches else float("nan"),
    }


def main() -> None:
    configs = json.loads((ROOT / "configs.json").read_text())
    names = [item["name"] for item in configs["variants"]]
    rows = [parse_variant(name) for name in names]
    RESULTS.mkdir(exist_ok=True)
    output = RESULTS / "area_summary.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_name = {str(row["variant"]): row for row in rows}
    legacy = by_name["legacy_capacity"]
    descriptor = by_name["descriptor4"]
    full = by_name["v3_2_full"]

    def delta(new: dict[str, float | int | str],
              old: dict[str, float | int | str]) -> float:
        return float(new["cell_area_um2"]) - float(old["cell_area_um2"])

    print(f"wrote {output}")
    print(
        "descriptor 2->4 delta: "
        f"{delta(descriptor, legacy):.3f} um^2 "
        f"({float(descriptor['cell_area_um2']) / float(legacy['cell_area_um2']):.3f}x)"
    )
    print(
        "backend capacity delta: "
        f"{delta(full, descriptor):.3f} um^2 "
        f"({float(full['cell_area_um2']) / float(descriptor['cell_area_um2']):.3f}x)"
    )
    print(
        "full vs legacy-capacity: "
        f"{delta(full, legacy):.3f} um^2 "
        f"({float(full['cell_area_um2']) / float(legacy['cell_area_um2']):.3f}x)"
    )


if __name__ == "__main__":
    main()
