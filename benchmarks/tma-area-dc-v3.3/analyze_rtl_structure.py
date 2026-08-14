#!/usr/bin/env python3
"""Measure generated TMA state bits and enforce V3.3 structural exclusions.

This is a pre-DC structural check, not a substitute for mapped cell area.
Generated CIRCT files declare sequential state with ``reg``. The standalone
totals recursively expand module instances from ``tma`` so repeated queue
module types are counted once per instance; per-module declaration sums are
retained separately for audit. This gives a reproducible comparison of the
old and replacement WindowEngine before technology mapping.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
V32 = ROOT / (
    "benchmarks/tma-area-dc-v3.2/variants/v3_2_full/rtl"
)
V33 = ROOT / "gpgpu/generated-tma-v2-v3.3"
V33_AREA_MIN = ROOT / "gpgpu/generated-tma-v2-v3.3-area-min"
V33_CAPACITY = ROOT / "gpgpu/generated-tma-v2-v3.3-capacity"


def without_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    # CIRCT embeds Scala assertion source snippets in SystemVerilog strings;
    # operators such as Scala's =/= are diagnostic text, not RTL arithmetic.
    return re.sub(r'"(?:\\.|[^"\\])*"', '""', text, flags=re.DOTALL)


def register_declarations(text: str) -> list[tuple[int, str]]:
    declarations: list[tuple[int, str]] = []
    for match in re.finditer(r"\breg\b(.*?);", text, flags=re.DOTALL):
        declaration = " ".join(match.group(1).split())
        width_match = re.match(
            r"\s*(?:signed\s+)?\[(\d+)\s*:\s*(\d+)\]\s+(.*)",
            declaration,
        )
        if width_match:
            width = abs(int(width_match.group(1)) -
                        int(width_match.group(2))) + 1
            names = width_match.group(3)
        else:
            width = 1
            names = declaration
        for name in names.split(","):
            normalized = name.split("=", 1)[0].strip()
            if normalized:
                declarations.append((width, normalized))
    return declarations


def register_bits(path: Path) -> int:
    text = without_comments(path.read_text(encoding="utf-8"))
    return sum(width for width, _ in register_declarations(text))


def module_bodies(path: Path) -> dict[str, str]:
    text = without_comments(path.read_text(encoding="utf-8"))
    return {
        match.group(1): match.group(0)
        for match in re.finditer(
            r"^module\s+([A-Za-z_][A-Za-z0-9_$]*)\b.*?^endmodule\b",
            text,
            flags=re.DOTALL | re.MULTILINE,
        )
    }


def module_register_bits(path: Path) -> dict[str, int]:
    return {
        name: sum(width for width, _ in register_declarations(body))
        for name, body in module_bodies(path).items()
        if register_declarations(body)
    }


def hierarchical_register_audit(
    path: Path, top: str = "tma"
) -> dict[str, object]:
    bodies = module_bodies(path)
    if top not in bodies:
        raise ValueError(f"missing top module {top!r} in {path}")
    own_bits = {
        name: sum(width for width, _ in register_declarations(body))
        for name, body in bodies.items()
    }
    instance_pattern = re.compile(
        r"^\s*([A-Za-z_][A-Za-z0-9_$]*)\s+"
        r"([A-Za-z_][A-Za-z0-9_$]*)\s*\(",
        re.MULTILINE,
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
    def subtree_bits(module: str) -> int:
        return own_bits[module] + sum(
            subtree_bits(child) for child in children[module]
        )

    counts: Counter[str] = Counter()

    def visit(module: str, stack: tuple[str, ...] = ()) -> None:
        if module in stack:
            raise ValueError(
                f"recursive module hierarchy in {path}: "
                f"{' -> '.join((*stack, module))}"
            )
        counts[module] += 1
        for child in children[module]:
            visit(child, (*stack, module))

    visit(top)
    contributions = {
        module: own_bits[module] * count
        for module, count in counts.items()
        if own_bits[module]
    }
    return {
        "total_bits": subtree_bits(top),
        "module_own_bits": own_bits,
        "module_subtree_bits": {
            module: subtree_bits(module) for module in counts
        },
        "module_instance_counts": dict(counts),
        "module_contribution_bits": contributions,
    }


def window_engine_field_bits(path: Path) -> dict[str, int]:
    body = module_bodies(path).get("TmaV2WindowEngine")
    if body is None:
        raise ValueError(f"missing TmaV2WindowEngine in {path}")
    groups = {
        "slot_payload": r"^slotPayload_",
        "slot_global_address": r"^slotGlobalAddress_",
        "slot_global_run": r"^slotGlobal(?:Offset|Bytes)_",
        "slot_shared_base": r"^slotSharedBase_",
        "slot_shared_atom_delta": r"^slotSharedAtomDelta_",
        "slot_shared_run": r"^slotShared(?:Offset|Bytes)_",
        "slot_pending": r"^slotPending_",
        "slot_outstanding": r"^slotOutstanding_",
        "slot_state": r"^slotState_",
        "request_table": r"^request",
        "shared_table": r"^shared",
        "write_ack_table": r"^writeAck",
    }
    declarations = register_declarations(body)
    result = {
        group: sum(
            width for width, name in declarations
            if re.search(pattern, name)
        )
        for group, pattern in groups.items()
    }
    result["other"] = (
        sum(width for width, _ in declarations) - sum(result.values())
    )
    return result


def main() -> None:
    files = {
        "v3_2_window_engine": V32 / "TmaV2WindowEngine.sv",
        "v3_3_window_engine": V33 / "TmaV2WindowEngine.sv",
        "v3_2_descriptor_service": V32 / "TmaV2DescriptorService.sv",
        "v3_3_descriptor_service": V33 / "TmaV2DescriptorService.sv",
        "v3_2_tma_raw_pmu_on": V32 / "tma.sv",
        "v3_3_tma_pmu_off": V33 / "tma.sv",
        "v3_3_area_min_tma_pmu_off": V33_AREA_MIN / "tma.sv",
        "v3_3_capacity_tma_pmu_off": V33_CAPACITY / "tma.sv",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise SystemExit(f"missing RTL inputs: {missing}")
    definition_bits = {
        name: register_bits(path) for name, path in files.items()
    }
    v32_audit = hierarchical_register_audit(
        V32 / "tma.sv"
    )
    release_audit = hierarchical_register_audit(V33 / "tma.sv")
    area_min_audit = hierarchical_register_audit(
        V33_AREA_MIN / "tma.sv"
    )
    capacity_audit = hierarchical_register_audit(
        V33_CAPACITY / "tma.sv"
    )
    bits = {
        "v3_2_window_engine": v32_audit[
            "module_subtree_bits"
        ]["TmaV2WindowEngine"],
        "v3_3_window_engine": release_audit[
            "module_subtree_bits"
        ]["TmaV2WindowEngine"],
        "v3_2_descriptor_service": v32_audit[
            "module_subtree_bits"
        ]["TmaV2DescriptorService"],
        "v3_3_descriptor_service": release_audit[
            "module_subtree_bits"
        ]["TmaV2DescriptorService"],
        "v3_2_tma_raw_pmu_on": v32_audit["total_bits"],
        "v3_3_tma_pmu_off": release_audit["total_bits"],
        "v3_3_area_min_tma_pmu_off": area_min_audit["total_bits"],
        "v3_3_capacity_tma_pmu_off": capacity_audit["total_bits"],
        "module_definition_sum": definition_bits,
    }
    old_engine = bits["v3_2_window_engine"]
    new_engine = bits["v3_3_window_engine"]
    bits["window_engine_reduction_percent"] = round(
        (1.0 - new_engine / old_engine) * 100.0, 4
    )
    modules = release_audit["module_contribution_bits"]
    area_min_modules = area_min_audit["module_contribution_bits"]
    capacity_modules = capacity_audit["module_contribution_bits"]
    engine_fields = window_engine_field_bits(
        V33 / "TmaV2WindowEngine.sv"
    )
    payload_bits = engine_fields["slot_payload"]
    release_engine_bits = release_audit[
        "module_subtree_bits"
    ]["TmaV2WindowEngine"]
    capacity_engine_bits = capacity_audit[
        "module_subtree_bits"
    ]["TmaV2WindowEngine"]
    added_capacity_slots = 32 - 24
    per_added_slot_bits = (
        capacity_engine_bits - release_engine_bits
    ) // added_capacity_slots
    state_audit = {
        "release_module_register_bits": dict(
            sorted(modules.items(), key=lambda item: item[1], reverse=True)
        ),
        "release_module_instance_counts": dict(
            sorted(
                release_audit["module_instance_counts"].items(),
                key=lambda item: (-item[1], item[0]),
            )
        ),
        "window_engine_field_register_bits": engine_fields,
        "payload_percent_of_window_engine": round(
            payload_bits / release_engine_bits * 100.0, 4
        ),
        "payload_percent_of_standalone_tma": round(
            payload_bits / bits["v3_3_tma_pmu_off"] * 100.0, 4
        ),
        "capacity_point_register_bits": {
            "slot16_desc2_standalone": bits[
                "v3_3_area_min_tma_pmu_off"
            ],
            "slot16_desc2_window_engine": area_min_audit[
                "module_subtree_bits"
            ]["TmaV2WindowEngine"],
            "slot24_desc4_standalone": bits["v3_3_tma_pmu_off"],
            "slot24_desc4_window_engine": release_engine_bits,
            "slot32_desc4_standalone": bits[
                "v3_3_capacity_tma_pmu_off"
            ],
            "slot32_desc4_window_engine": capacity_engine_bits,
            "slot24_to_32_delta": (
                bits["v3_3_capacity_tma_pmu_off"]
                - bits["v3_3_tma_pmu_off"]
            ),
            "bits_per_added_slot": per_added_slot_bits,
            "payload_bits_per_slot": 1024,
            "payload_percent_of_added_slot": round(
                1024 / per_added_slot_bits * 100.0, 4
            ),
        },
    }

    release_text = without_comments(
        (V33 / "tma.sv").read_text(encoding="utf-8")
    )
    forbidden = {
        "$div": r"\$div\b",
        "$rem": r"\$rem\b",
        "runtime_divide_operator": r"(?<!/)/(?!/)",
        "compiled_legal": r"\bcompiled_legal\b",
        "compiled_stepOffsets": r"\bcompiled_stepOffsets\b",
        "wide_fast_path": r"\b(?:emitMode|wideLineCount|wideGlobalBase)\b",
        "per_window_fill": r"\bfillData\b",
        "per_window_line_tag": r"\blineTag\b",
        "per_slot_global_mask": r"\bslotGlobalMask\b",
        "per_slot_shared_mask": r"\bslotSharedMask\b",
        "per_slot_full_shared_addresses": r"\bslotSharedAddress\b",
        "full_request_physical_address": r"\brequestPhysical\b",
        "descriptor_response_payload_copy": r"\bdemandResponseData\b",
        "lookahead_compiled_duplicate": r"\bdescriptorCompiled\b",
        "transform_payload_pipeline": r"\btransformStages_[0-9]+_.*data\b",
        "PMU_output": r"\bio_perf_s2g",
    }
    hits = {
        name: len(re.findall(pattern, release_text))
        for name, pattern in forbidden.items()
    }
    output = {
        "register_bits": bits,
        "state_audit": state_audit,
        "forbidden_hits_in_v3_3_release_tma": hits,
        "window_engine_state_gate_passed":
            bits["window_engine_reduction_percent"] >= 45.0,
        "structural_exclusion_gate_passed": all(value == 0
                                                for value in hits.values()),
    }
    result = HERE / "rtl_structure_summary.json"
    result.write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2))
    if not output["window_engine_state_gate_passed"]:
        raise SystemExit("WindowEngine state reduction is below 45%")
    if not output["structural_exclusion_gate_passed"]:
        raise SystemExit("forbidden V3.3 RTL structure detected")


if __name__ == "__main__":
    main()
