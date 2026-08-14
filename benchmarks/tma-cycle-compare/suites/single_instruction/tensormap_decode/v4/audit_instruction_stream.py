#!/usr/bin/env python3
"""Audit cold-demand and explicit-prefetch TensorMap decode kernels."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("sass", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = args.source.read_text(encoding="utf-8")
    sass = args.sass.read_text(encoding="utf-8")
    # ``cuobjdump --dump-sass`` labels a body as ``Function : <symbol>``;
    # ``nvdisasm`` (used in the Modal image build) labels the same body with a
    # ``//----- .text.<symbol> -----`` banner.  Audit the instruction body in
    # either representation instead of treating disassembler formatting as a
    # property of the generated code.
    starts: list[tuple[int, str]] = []
    for match in re.finditer(r"^\s*Function\s*:\s*(\S+)\s*$", sass, re.MULTILINE):
        starts.append((match.start(), match.group(1)))
    for match in re.finditer(
        r"^//-+\s+\.text\.(\S+)\s+-+\s*$", sass, re.MULTILINE
    ):
        starts.append((match.start(), match.group(1)))
    starts.sort()
    cold: list[str] = []
    prefetched: list[str] = []
    for index, (start, symbol) in enumerate(starts):
        template = re.search(
            r"probe_kernelILi\d+ELb([01])ELb([01])ELb([01])EE", symbol
        )
        if not template or template.group(1) != "1":
            continue
        end = starts[index + 1][0] if index + 1 < len(starts) else len(sass)
        section = sass[start:end]
        # Template arguments end in Tensor, G2S, Prefetch.  The final Lb0/Lb1
        # before EE is therefore the compile-time prefetch selector.
        (prefetched if template.group(3) == "1" else cold).append(section)
    checks = {
        "source_uses_4k_pages": "kMapPageBytes = 4096" in source,
        "source_has_cold_then_hot_label": '"cold_then_hot"' in source,
        "source_has_same_kernel_pair_loop": "iteration < kRepeats" in source,
        "source_emits_prior_tma_use": "prior_tma_use" in source,
        "cold_tensor_functions_present": len(cold) >= 10,
        "prefetch_tensor_functions_present": len(prefetched) >= 10,
        "cold_functions_have_no_prefetch": all("UTMACCTL.PF" not in section for section in cold),
        "prefetch_functions_have_prefetch": all("UTMACCTL.PF" in section for section in prefetched),
        "cold_functions_have_tensor_op": all("UTMALD" in section or "UTMAST" in section for section in cold),
    }
    result = {"schema_version": 1, "checks": checks, "passed": all(checks.values())}
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if not result["passed"]:
        raise SystemExit("TensorMap decode instruction audit failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
