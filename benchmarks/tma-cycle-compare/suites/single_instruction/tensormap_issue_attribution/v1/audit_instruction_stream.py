#!/usr/bin/env python3
"""Reject binaries that do not retain one guarded fixed-rank TMA site."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def sections(sass: str) -> dict[str, str]:
    starts: list[tuple[int, str]] = []
    for match in re.finditer(r"^//-+\s+\.text\.(\S+)\s+-+\s*$", sass, re.MULTILINE):
        starts.append((match.start(), match.group(1)))
    starts.sort()
    return {
        symbol: sass[start : starts[index + 1][0] if index + 1 < len(starts) else len(sass)]
        for index, (start, symbol) in enumerate(starts)
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("ptx", type=Path)
    parser.add_argument("sass", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.read_text(encoding="utf-8")
    ptx = args.ptx.read_text(encoding="utf-8")
    sass = args.sass.read_text(encoding="utf-8")
    kernel_sections = {
        name: body for name, body in sections(sass).items()
        if "issue_attribution_kernel" in name
    }
    g2s = [body for name, body in kernel_sections.items() if "ILb1" in name]
    s2g = [body for name, body in kernel_sections.items() if "ILb0" in name]
    checks = {
        "runtime_prime_mode": "int prime_mode" in source,
        "single_nonunrolled_loop": "#pragma unroll 1" in source,
        "explicit_ptx_predicate": "@issue cp.async.bulk.tensor.2d" in source,
        "fixed_rank_2d_only": "tensor.2d" in source and "tensor.3d" not in source,
        "ptx_has_predicated_g2s": bool(re.search(r"@issue\s+cp\.async\.bulk\.tensor\.2d\.shared", ptx)),
        "ptx_has_predicated_s2g": bool(re.search(r"@issue\s+cp\.async\.bulk\.tensor\.2d\.global", ptx)),
        "one_g2s_kernel": len(g2s) == 1,
        "one_s2g_kernel": len(s2g) == 1,
        "one_g2s_tma_site": len(g2s) == 1 and len(re.findall(r"\bUTMALD", g2s[0])) == 1,
        "one_s2g_tma_site": len(s2g) == 1 and len(re.findall(r"\bUTMAST", s2g[0])) == 1,
        # Hopper/Blackwell ptxas lowers the PTX predicate into a uniform branch
        # around the single physical UTMA site.  Require that actual SASS shape
        # instead of claiming the hardware UTMA itself remains predicated.
        "g2s_site_has_guard_branch": len(g2s) == 1 and bool(
            re.search(r"BRA[^\n]*\n(?:[^\n]*\n){0,16}[^\n]*UTMALD", g2s[0])
        ),
        "s2g_site_has_guard_branch": len(s2g) == 1 and bool(
            re.search(r"BRA[^\n]*\n(?:[^\n]*\n){0,16}[^\n]*UTMAST", s2g[0])
        ),
    }
    result = {"schema_version": 1, "checks": checks, "passed": all(checks.values())}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit("instruction-stream audit failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
