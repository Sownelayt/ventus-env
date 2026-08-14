#!/usr/bin/env python3
"""Compare the one V3.4 mapped result with the completed V3.3 release.

Background:
  The long N12 run is intentionally not a sweep. V3.3 release is the formal
  mapped baseline; V3.4 must reduce total cell area by at least 15% and close
  setup timing at 1.5 GHz.

Flow:
  Parse total cell area and worst setup slack from copied DC reports. The
  baseline-only mode fixes V3.3 evidence before the V3.4 run completes.

Usage:
  python3 benchmarks/tma-area-dc-v3.4/analyze_dc_result.py --baseline-only
  python3 benchmarks/tma-area-dc-v3.4/analyze_dc_result.py

Maintenance:
  Copy reports verbatim from each independent VM project. Never substitute an
  intermediate compile area for the final report_area result.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V33_REPORT = (
    ROOT / "benchmarks/tma-area-dc-v3.3/results/"
    "v3_3_release/report"
)
V34_REPORT = (
    HERE / "results/v3_4_release_40x6_desc4/report"
)
RELEASE_REPORT = ROOT / "docs/TMA_V3_4_RELEASE_40X6_20260731.md"
REPORT_START = "<!-- V3.4_DC_RESULT_START -->"
REPORT_END = "<!-- V3.4_DC_RESULT_END -->"


def parse_area(report: Path) -> float:
    text = report.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Total cell area:\s+([0-9.]+)", text)
    if not match:
        raise SystemExit(f"cannot parse total cell area: {report}")
    return float(match.group(1))


def parse_wns(report: Path) -> float:
    text = report.read_text(encoding="utf-8", errors="replace")
    values = [
        float(value)
        for value in re.findall(
            r"slack\s+\((?:MET|VIOLATED)\)\s+(-?[0-9.]+)", text
        )
    ]
    if not values:
        raise SystemExit(f"cannot parse setup slack: {report}")
    return min(values)


def extract(report_dir: Path) -> dict[str, object]:
    area = report_dir / "area.rpt"
    timing = report_dir / "timing_setup.rpt"
    run_summary = report_dir / "run_summary.rpt"
    for path in (area, timing, run_summary):
        if not path.is_file():
            raise SystemExit(f"missing DC report: {path}")
    summary_text = run_summary.read_text(
        encoding="utf-8", errors="replace"
    )
    if "1500" not in summary_text or "tt1v85c" not in summary_text:
        raise SystemExit(f"unexpected DC condition: {run_summary}")
    return {
        "report_dir": str(report_dir.relative_to(ROOT)),
        "total_cell_area": parse_area(area),
        "wns_ns": parse_wns(timing),
        "frequency_mhz": 1500,
        "corner": "tcbn12ffcllbwp16p90cpdtt1v85c",
    }


def update_release_report(result: dict[str, object]) -> None:
    if not RELEASE_REPORT.is_file():
        raise SystemExit(f"missing release report: {RELEASE_REPORT}")
    text = RELEASE_REPORT.read_text(encoding="utf-8")
    if text.count(REPORT_START) != 1 or text.count(REPORT_END) != 1:
        raise SystemExit("V3.4 release report DC marker is missing or ambiguous")
    baseline = result["v3_3_release"]
    measured = result["v3_4_release"]
    passed = result["all_gates_passed"]
    state = "release_candidate" if passed else "mapped_dc_gate_failed"
    replacement = f"""{REPORT_START}
DC 最终结果：

| 项目 | V3.3 | V3.4 | V3.4 门槛 |
|---|---:|---:|---:|
| total cell area | {baseline["total_cell_area"]:.6f} | {measured["total_cell_area"]:.6f} | ≤158134.725343 |
| 1.5 GHz WNS | {baseline["wns_ns"]:.4f} ns | {measured["wns_ns"]:.4f} ns | ≥0 ns |
| area reduction | — | {result["area_reduction_percent"]:.4f}% | ≥15% |

映射门槛状态：`{state}`。
{REPORT_END}"""
    before, tail = text.split(REPORT_START, 1)
    _, after = tail.split(REPORT_END, 1)
    RELEASE_REPORT.write_text(
        before + replacement + after,
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    baseline = extract(V33_REPORT)
    baseline_path = HERE / "results/v3_3_release_baseline.json"
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(
        json.dumps(baseline, indent=2) + "\n", encoding="utf-8"
    )
    if args.baseline_only:
        print(json.dumps(baseline, indent=2))
        return

    measured = extract(V34_REPORT)
    baseline_area = float(baseline["total_cell_area"])
    measured_area = float(measured["total_cell_area"])
    area_reduction = (1.0 - measured_area / baseline_area) * 100.0
    gates = {
        "area_not_above_v3_3":
            measured_area <= baseline_area,
        "area_reduction_ge_15_percent":
            area_reduction >= 15.0,
        "wns_non_negative":
            float(measured["wns_ns"]) >= 0.0,
    }
    result = {
        "v3_3_release": baseline,
        "v3_4_release": measured,
        "area_delta": measured_area - baseline_area,
        "area_reduction_percent": round(area_reduction, 4),
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }
    output = HERE / "results/area_gate_summary.json"
    output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    update_release_report(result)
    print(json.dumps(result, indent=2))
    if not result["all_gates_passed"]:
        raise SystemExit("V3.4 mapped area/timing gate failed")


if __name__ == "__main__":
    main()
