#!/usr/bin/env python3
"""Audit the FreshTensorMapV1 CUDA instruction and source contract."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def function_sections(text: str) -> list[str]:
    starts: list[tuple[int, str]] = []
    for match in re.finditer(
        r"^\s*Function\s*:\s*(\S+)\s*$", text, re.MULTILINE
    ):
        starts.append((match.start(), match.group(1)))
    for match in re.finditer(
        r"^//-+\s+\.text\.(\S+)\s+-+\s*$", text, re.MULTILINE
    ):
        starts.append((match.start(), match.group(1)))
    starts.sort()
    return [
        text[start : starts[index + 1][0] if index + 1 < len(starts) else len(text)]
        for index, (start, symbol) in enumerate(starts)
        if "fresh_tensor_" in symbol
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("ptx", type=Path)
    parser.add_argument("sass", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    source = args.source.read_text(encoding="utf-8")
    ptx = args.ptx.read_text(encoding="utf-8")
    sass = args.sass.read_text(encoding="utf-8")
    sections = function_sections(sass)
    checks = {
        "source_has_4k_descriptor_pages": "kTensorMapPageBytes = 4096" in source,
        "source_has_monotonic_descriptor_pool": "++next_" in source,
        "source_marks_cold_then_hot": 'repeat == 0 ? "tmau_cold_first_use_l2_hot" : "tmau_hot_reuse"' in source,
        "source_reuses_one_map_per_pair": "source_maps[repeat] = source_maps[0]" in source,
        "source_uses_pointer_tensor_kernels": "const CUtensorMap* map" in source,
        "source_warms_with_ld_global_cg": "ld.global.cg.u32" in source,
        "fresh_ptx_present": ptx.count("fresh_tensor_") >= 15,
        "fresh_sass_functions_present": len(sections) >= 15,
        "fresh_sass_has_tensor_ops": sum(section.count("UTMALD") + section.count("UTMAST") for section in sections) >= 15,
        "source_acquires_once_before_pair": "acquire_fresh_maps_once<<<1, 1>>>" in source,
        "measured_fresh_sass_has_no_reacquire": all("UTMACCTL.IV" not in section for section in sections),
        "sass_has_one_time_acquire": "UTMACCTL.IV" in sass,
        "fresh_sass_has_no_prefetch": all("UTMACCTL.PF" not in section for section in sections),
    }
    result = {"schema_version": 1, "checks": checks, "passed": all(checks.values())}
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if not result["passed"]:
        failed = ", ".join(name for name, value in checks.items() if not value)
        raise SystemExit(f"fresh TensorMap audit failed: {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
