#!/usr/bin/env python3
"""Audit Ventus TMA V3.6 generated RTL before the final DC synthesis.

Background:
  V3.6 retains the V3.4/V3.5 40+6 storage organization and adds deliberate
  timing boundaries across every material V3.5 negative-slack path family.
  The extra pipeline registers must not reintroduce duplicated payload banks,
  response broadcast networks, PMU state, or runtime divide/remainder logic.

Flow:
  Parse CIRCT SystemVerilog register declarations and module instances,
  compute hierarchical state, classify the V3.4 engine fields, count wide
  shift sites and forbidden structures, cross-check the selected payload
  read/permuter implementation, then enforce the release gates.

Usage:
  python3 benchmarks/tma-area-dc-v3.6/analyze_rtl_structure.py

Maintenance:
  This is a pre-DC structural audit, not a mapped-area estimator. Keep field
  patterns synchronized with TmaV34WindowEngine.scala when arrays are renamed.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RTL = ROOT / "gpgpu/generated-tma-v2-v3.6-work/tma.sv"
DEFAULT_BASELINE_RTL = ROOT / "gpgpu/generated-tma-v2-v3.3/tma.sv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "rtl_structure_summary.json"


def clean(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    return re.sub(r'"(?:\\.|[^"\\])*"', '""', text, flags=re.DOTALL)


def module_bodies(text: str) -> dict[str, str]:
    return {
        match.group(1): match.group(0)
        for match in re.finditer(
            r"^module\s+([A-Za-z_][A-Za-z0-9_$]*)\b.*?^endmodule\b",
            text,
            flags=re.DOTALL | re.MULTILINE,
        )
    }


def registers(body: str) -> list[tuple[int, str]]:
    result: list[tuple[int, str]] = []
    for match in re.finditer(r"\breg\b(.*?);", body, flags=re.DOTALL):
        declaration = " ".join(match.group(1).split())
        width_match = re.match(
            r"\s*(?:signed\s+)?\[(\d+)\s*:\s*(\d+)\]\s+(.*)",
            declaration,
        )
        if width_match:
            width = abs(int(width_match.group(1))
                        - int(width_match.group(2))) + 1
            names = width_match.group(3)
        else:
            width = 1
            names = declaration
        for name in names.split(","):
            normalized = name.split("=", 1)[0].strip()
            if normalized:
                result.append((width, normalized))
    return result


def hierarchy(bodies: dict[str, str], top: str) -> dict[str, object]:
    own = {
        name: sum(width for width, _ in registers(body))
        for name, body in bodies.items()
    }
    instance_pattern = re.compile(
        r"^\s*([A-Za-z_][A-Za-z0-9_$]*)\s+"
        r"([A-Za-z_][A-Za-z0-9_$]*)\s*\(",
        flags=re.MULTILINE,
    )
    children = {
        name: [
            module_type
            for module_type, _ in instance_pattern.findall(body)
            if module_type in bodies
        ]
        for name, body in bodies.items()
    }

    @lru_cache(maxsize=None)
    def subtree(name: str) -> int:
        return own[name] + sum(subtree(child) for child in children[name])

    counts: Counter[str] = Counter()

    def visit(name: str) -> None:
        counts[name] += 1
        for child in children[name]:
            visit(child)

    visit(top)
    return {
        "total_bits": subtree(top),
        "own_bits": own,
        "subtree_bits": {name: subtree(name) for name in counts},
        "instance_counts": dict(counts),
        "contribution_bits": {
            name: own[name] * count
            for name, count in counts.items()
            if own[name]
        },
    }


def matching_bits(declarations: list[tuple[int, str]], pattern: str) -> int:
    regex = re.compile(pattern)
    return sum(width for width, name in declarations if regex.search(name))


def matching_count(declarations: list[tuple[int, str]], pattern: str) -> int:
    regex = re.compile(pattern)
    return sum(1 for _, name in declarations if regex.search(name))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtl", default=str(DEFAULT_RTL))
    parser.add_argument(
        "--baseline-rtl",
        default=str(DEFAULT_BASELINE_RTL),
        help="frozen PMU-off V3.3 release RTL",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--standalone-max", type=int, default=35000,
        help="maximum allowed standalone TMA sequential state bits",
    )
    parser.add_argument(
        "--planner-queue-max", type=int, default=5000,
        help="maximum allowed Planner elastic-queue state bits",
    )
    parser.add_argument(
        "--cache-egress-relocated",
        action="store_true",
        help=(
            "treat Queue1_TmaV2CacheRequest as the existing standalone "
            "egress boundary relocated under WindowEngine, and report the "
            "engine data-core gate separately from that boundary"
        ),
    )
    parser.add_argument(
        "--shared-egress-relocated",
        action="store_true",
        help=(
            "treat Queue1_TmaV2SharedRequest as the existing standalone "
            "shared egress boundary relocated under WindowEngine, and "
            "report the engine data-core gate separately from that boundary"
        ),
    )
    args = parser.parse_args()

    rtl = Path(args.rtl).resolve()
    if not rtl.is_file():
        raise SystemExit(f"missing generated RTL: {rtl}")
    text = clean(rtl.read_text(encoding="utf-8"))
    bodies = module_bodies(text)
    engine_name = (
        "TmaV34WindowEngine"
        if "TmaV34WindowEngine" in bodies
        else "TmaV2WindowEngine"
    )
    planner_name = (
        "TmaV34WindowPlanner"
        if "TmaV34WindowPlanner" in bodies
        else "TmaV2WindowPlanner"
    )
    required = {
        "tma", engine_name, "TmaV3Ingress",
        planner_name, "TmaV2CommandBinder",
        "TmaV2DescriptorService",
    }
    missing = sorted(required - bodies.keys())
    if missing:
        raise SystemExit(f"missing required modules: {missing}")

    audit = hierarchy(bodies, "tma")
    engine = registers(bodies[engine_name])
    engine_bits = audit["subtree_bits"][engine_name]
    standalone_bits = audit["total_bits"]
    cache_egress_bits = (
        audit["contribution_bits"].get("Queue1_TmaV2CacheRequest", 0)
        if args.cache_egress_relocated
        else 0
    )
    shared_egress_bits = (
        audit["contribution_bits"].get("Queue1_TmaV2SharedRequest", 0)
        if args.shared_egress_relocated
        else 0
    )
    engine_core_bits = (
        engine_bits - cache_egress_bits - shared_egress_bits
    )
    line_entries = matching_count(engine, r"^lineState_\d+$")
    payload_entries = matching_count(engine, r"^payloadData_\d+$")
    line_context_bits = sum(
        matching_bits(engine, pattern) // max(line_entries, 1)
        for pattern in (
            r"^lineState_\d+$",
            r"^(?:lineVirtualTag|linePpn|lineAddressTag)_\d+$",
            r"^lineRoute_\d+$",
        )
    )
    payload_slot_bits = sum(
        matching_bits(engine, pattern) // max(payload_entries, 1)
        for pattern in (
            r"^payloadState_\d+$", r"^payloadData_\d+$",
            r"^payloadMeta_\d+$", r"^payloadPending_\d+$",
            r"^payloadOutstanding_\d+$",
        )
    )
    payload_1024_copies = matching_count(
        engine, r"^payloadData_\d+$"
    )
    planner_queue_bits = sum(
        contribution
        for name, contribution in audit["contribution_bits"].items()
        if name.startswith("Queue1_TmaV2Cursor")
        or name.startswith("Queue1_TmaV2CoordinateMapToken")
        or name.startswith("Queue1_TmaV2GlobalMapToken")
        or name.startswith("Queue1_TmaV34CompactWindow")
    )
    wide_dynamic_shifts = len(re.findall(
        r"(?:>>|<<)\s*[A-Za-z_$][A-Za-z0-9_$]*", bodies[engine_name]
    ))
    shifted_declarations: list[tuple[int, int]] = []
    for declaration in re.finditer(
        r"\bwire\s+\[(\d+)\s*:\s*0\].*?;",
        bodies[engine_name],
        flags=re.DOTALL,
    ):
        width = int(declaration.group(1)) + 1
        shifts = len(re.findall(r"(?:>>|<<)", declaration.group(0)))
        if shifts:
            shifted_declarations.append((width, shifts))
    oversized_dynamic_shifts = sum(
        shifts for width, shifts in shifted_declarations if width > 1024
    )
    max_dynamic_shift_result_width = max(
        (width for width, _ in shifted_declarations), default=0
    )
    baseline_rtl = Path(args.baseline_rtl).resolve()
    if not baseline_rtl.is_file():
        raise SystemExit(f"missing V3.3 baseline RTL: {baseline_rtl}")
    baseline_bodies = module_bodies(clean(
        baseline_rtl.read_text(encoding="utf-8")
    ))
    baseline_audit = hierarchy(baseline_bodies, "tma")
    baseline_engine_bits = baseline_audit[
        "subtree_bits"
    ]["TmaV2WindowEngine"]
    baseline_standalone_bits = baseline_audit["total_bits"]
    engine_reduction_percent = (
        1.0 - engine_core_bits / baseline_engine_bits
    ) * 100.0
    standalone_reduction_percent = (
        1.0 - standalone_bits / baseline_standalone_bits
    ) * 100.0
    baseline_shifted_declarations: list[tuple[int, int]] = []
    for declaration in re.finditer(
        r"\bwire\s+\[(\d+)\s*:\s*0\].*?;",
        baseline_bodies["TmaV2WindowEngine"],
        flags=re.DOTALL,
    ):
        width = int(declaration.group(1)) + 1
        shifts = len(re.findall(r"(?:>>|<<)", declaration.group(0)))
        if shifts:
            baseline_shifted_declarations.append((width, shifts))
    baseline_oversized_shifts = sum(
        shifts
        for width, shifts in baseline_shifted_declarations
        if width > 1024
    )
    baseline_max_shift_width = max(
        (width for width, _ in baseline_shifted_declarations), default=0
    )
    engine_source_path = (
        ROOT / "gpgpu/ventus/src/pipeline/TmaV34WindowEngine.scala"
    )
    engine_source = engine_source_path.read_text(encoding="utf-8")
    selected_payload_reads = len(re.findall(
        r"\bval\s+\w+\s*=\s*payloadData\s*\(",
        engine_source,
    ))
    selected_line_permuters = len(re.findall(
        r"\b(?:permuterInput|commonPermuterInput)\s*>>\s*Cat\s*\(\s*"
        r"(?:permuterOffset|permuterByteOffset)",
        engine_source,
    ))
    separate_direction_permuters = len(re.findall(
        r"outgoingG2SRaw\s*>>|laneData\.pad\s*\(\s*1024\s*\)\s*<<",
        engine_source,
    ))
    forbidden_patterns = {
        "runtime_divide": r"(?<!/)/(?!/)|\$div\b|\bdiv(?:ider)?\b",
        "runtime_remainder": r"%(?![%=])|\$rem\b|\brem(?:ainder)?\b",
        "pmu_output": r"\bio_perf_s2g\b",
        "old_window_payload": r"\b(?:windowData|slotPayload|requestData|sharedData)_",
        "owner_bitmap": r"\b(?:requestConsumer|consumerBitmap|ownerBitmap)\b",
        "wide_response_broadcast": r"\b(?:responseBroadcast|broadcastWindow)\b",
        "old_fast_path": r"\b(?:emitMode|wideLineCount|wideGlobalBase)\b",
        "duplicate_descriptor_result": r"\bcompilerTag\b|\bcompilerAsid\b",
    }
    forbidden = {
        name: len(re.findall(pattern, text, flags=re.IGNORECASE))
        for name, pattern in forbidden_patterns.items()
    }

    gates = {
        "line_entries_40": line_entries == 40,
        "payload_entries_6": payload_entries == 6,
        "payload_1024_copies_6": payload_1024_copies == 6,
        "line_context_le_256": line_context_bits <= 256,
        "payload_slot_le_1472": payload_slot_bits <= 1472,
        "window_engine_core_le_19500": engine_core_bits <= 19500,
        "standalone_within_limit":
            standalone_bits <= args.standalone_max,
        "window_engine_reduction_ge_40_percent":
            engine_reduction_percent >= 40.0,
        "standalone_reduction_ge_30_percent":
            standalone_reduction_percent >= 30.0,
        "planner_queues_within_limit":
            planner_queue_bits <= args.planner_queue_max,
        "binder_timing_pipeline_le_768":
            audit["subtree_bits"]["TmaV2CommandBinder"] <= 768,
        "ingress_timing_pipeline_le_1600":
            audit["own_bits"]["TmaV3Ingress"] <= 1600,
        "descriptor_timing_pipeline_le_3600":
            audit["subtree_bits"]["TmaV2DescriptorService"] <= 3600,
        "one_selected_payload_read":
            selected_payload_reads == 1,
        "one_bidirectional_line_permuter":
            selected_line_permuters == 1
            and separate_direction_permuters == 0,
        "no_oversized_dynamic_shift":
            oversized_dynamic_shifts == 0
            and max_dynamic_shift_result_width <= 1024,
        "forbidden_absent": all(value == 0 for value in forbidden.values()),
    }
    result = {
        "rtl": str(rtl),
        "state_bits": {
            "standalone_tma": standalone_bits,
            "window_engine": engine_bits,
            "window_engine_data_core": engine_core_bits,
            "relocated_cache_egress": cache_egress_bits,
            "relocated_shared_egress": shared_egress_bits,
            "line_context_per_entry": line_context_bits,
            "payload_slot_per_entry": payload_slot_bits,
            "planner_elastic_queues": planner_queue_bits,
            "binder": audit["subtree_bits"]["TmaV2CommandBinder"],
            "ingress": audit["own_bits"]["TmaV3Ingress"],
            "descriptor_service": audit[
                "subtree_bits"
            ]["TmaV2DescriptorService"],
        },
        "storage": {
            "line_context_entries": line_entries,
            "payload_entries": payload_entries,
            "payload_1024_bit_copies": payload_1024_copies,
        },
        "limits": {
            "standalone_tma_state_bits": args.standalone_max,
            "planner_elastic_queue_bits": args.planner_queue_max,
        },
        "wide_dynamic_shift_sites": wide_dynamic_shifts,
        "v3_3_comparison": {
            "rtl": str(baseline_rtl),
            "standalone_tma_state_bits": baseline_standalone_bits,
            "window_engine_state_bits": baseline_engine_bits,
            "standalone_reduction_percent":
                round(standalone_reduction_percent, 4),
            "window_engine_reduction_percent":
                round(engine_reduction_percent, 4),
            "oversized_dynamic_shift_sites":
                baseline_oversized_shifts,
            "max_dynamic_shift_result_width":
                baseline_max_shift_width,
        },
        "wide_datapath": {
            "selected_payload_read_expressions": selected_payload_reads,
            "bidirectional_line_permuter_expressions":
                selected_line_permuters,
            "separate_direction_permuter_expressions":
                separate_direction_permuters,
            "oversized_dynamic_shift_sites":
                oversized_dynamic_shifts,
            "max_dynamic_shift_result_width":
                max_dynamic_shift_result_width,
        },
        "forbidden_hits": forbidden,
        "module_contribution_bits": dict(sorted(
            audit["contribution_bits"].items(),
            key=lambda item: item[1],
            reverse=True,
        )),
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["all_gates_passed"]:
        raise SystemExit("V3.6 structural gate failed")


if __name__ == "__main__":
    main()
