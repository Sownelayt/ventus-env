#!/usr/bin/env python3
"""Audit all 20 static banks of V12's fixed 640-point completion scan."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


KERNELS = {
    "tensor_first_use": "completion_tensor_no_prefetch_probe_v12",
    "tensor_prefetched": "completion_tensor_explicit_prefetch_probe_v12",
    "bulk": "completion_bulk_probe_v12",
}
POINTS = 640
BANK_SIZE = 8
BANKS = POINTS // BANK_SIZE
STEP = 3


def ptx_sections(text: str, symbol: str) -> dict[int, tuple[str, str]]:
    output: dict[int, tuple[str, str]] = {}
    pattern = re.compile(rf"^\.entry\s+(\S*{re.escape(symbol)}\S*)\(", re.M)
    for match in pattern.finditer(text):
        name = match.group(1)
        bank_match = re.search(r"ILi(\d+)EEE", name)
        if not bank_match:
            raise SystemExit(f"cannot read bank from PTX symbol {name}")
        bank = int(bank_match.group(1))
        end = text.find("\n}", match.end())
        if end < 0:
            raise SystemExit(f"cannot isolate PTX entry {name}")
        output[bank] = (name, text[match.start():end + 2])
    return output


def sass_sections(text: str, symbol: str) -> dict[int, tuple[str, str]]:
    matches = list(re.finditer(
        rf"^\s*Function\s*:\s*(\S*{re.escape(symbol)}\S*)\s*$",
        text, re.M,
    ))
    output: dict[int, tuple[str, str]] = {}
    for index, match in enumerate(matches):
        name = match.group(1)
        bank_match = re.search(r"ILi(\d+)EEE", name)
        if not bank_match:
            raise SystemExit(f"cannot read bank from SASS symbol {name}")
        bank = int(bank_match.group(1))
        following = re.search(r"^\s*Function\s*:", text[match.end():], re.M)
        end = match.end() + following.start() if following else len(text)
        output[bank] = (name, text[match.start():end])
    return output


def counts(ptx: str, sass: str) -> dict[str, int]:
    slot_labels = len(re.findall(
        r"^\s*delay_v12_bank_\d+_slot_\d+:", ptx, re.M
    ))
    slot_mentions = len(re.findall(
        r"delay_v12_bank_\d+_slot_\d+(?![_a-zA-Z0-9])", ptx
    ))
    return {
        "ptx_arrive_expect_tx": ptx.count("mbarrier.arrive.expect_tx"),
        "ptx_test_wait": ptx.count("mbarrier.test_wait"),
        "ptx_cleanup_try_wait": ptx.count("mbarrier.try_wait"),
        "ptx_tensor_prefetch": ptx.count("prefetch.tensormap"),
        "ptx_tensormap_acquire": ptx.count("fence.proxy.tensormap"),
        "ptx_tensor_g2s": ptx.count("cp.async.bulk.tensor."),
        "ptx_bulk_g2s": ptx.count("cp.async.bulk.shared::cta.global"),
        "ptx_brx_idx": ptx.count("brx.idx.uni"),
        "ptx_slot_predicates": ptx.count("setp.eq.u32 delay_v12_slot_pred"),
        "ptx_slot_labels": slot_labels,
        "ptx_pmevents": len(re.findall(r"^\s*pmevent ", ptx, re.M)),
        "ptx_slot_target_list_mentions": slot_mentions - slot_labels,
        "sass_tensormap_prefetch": sass.count("UTMACCTL.PF"),
        "sass_tensormap_invalidate": sass.count("UTMACCTL.IV"),
        "sass_tensor_g2s": sass.count("UTMALDG."),
        "sass_test_wait_total": sass.count("SYNCS.PHASECHK"),
        "sass_cleanup_try_wait": sass.count("SYNCS.PHASECHK.TRANS64.TRYWAIT"),
        "sass_cleanup_suspend": sass.count("NANOSLEEP.SYNCS"),
        "sass_call_rel": sass.count("CALL.REL"),
        "sass_brx": len(re.findall(r"\bBRX(?:U)?\b", sass)),
        "sass_instruction_count": len(re.findall(r"/\*[0-9a-fA-F]+\*/", sass)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ptx", type=Path)
    parser.add_argument("sass", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    all_ptx = args.ptx.read_text(encoding="utf-8")
    all_sass = args.sass.read_text(encoding="utf-8")
    result: dict[str, object] = {"kernels": {}, "passed": True}
    mismatches: dict[str, object] = {}

    for family, symbol in KERNELS.items():
        ptx_banks = ptx_sections(all_ptx, symbol)
        sass_banks = sass_sections(all_sass, symbol)
        if set(ptx_banks) != set(range(BANKS)):
            mismatches[f"{family}.ptx_banks"] = sorted(ptx_banks)
        if set(sass_banks) != set(range(BANKS)):
            mismatches[f"{family}.sass_banks"] = sorted(sass_banks)
        bank_counts: dict[int, dict[str, int]] = {}
        for bank in sorted(set(ptx_banks) & set(sass_banks)):
            actual = counts(ptx_banks[bank][1], sass_banks[bank][1])
            bank_counts[bank] = actual
            expected = {
                "ptx_arrive_expect_tx": 1,
                "ptx_test_wait": 1,
                "ptx_cleanup_try_wait": 1,
                "ptx_brx_idx": 0,
                "ptx_slot_predicates": BANK_SIZE - 1,
                "ptx_slot_labels": BANK_SIZE,
                "ptx_slot_target_list_mentions": BANK_SIZE,
                "ptx_pmevents": bank * BANK_SIZE * STEP + (BANK_SIZE - 2) + STEP,
                "sass_cleanup_try_wait": 1,
                "sass_cleanup_suspend": 1,
                "sass_call_rel": 0,
            }
            if family == "tensor_first_use":
                expected.update({
                    "ptx_tensor_prefetch": 0, "ptx_tensormap_acquire": 1,
                    "ptx_tensor_g2s": 5, "ptx_bulk_g2s": 0,
                    "sass_tensormap_prefetch": 0,
                    "sass_tensormap_invalidate": 1, "sass_tensor_g2s": 5,
                })
            elif family == "tensor_prefetched":
                expected.update({
                    "ptx_tensor_prefetch": 1, "ptx_tensormap_acquire": 1,
                    "ptx_tensor_g2s": 5, "ptx_bulk_g2s": 0,
                    "sass_tensormap_prefetch": 1,
                    "sass_tensormap_invalidate": 1, "sass_tensor_g2s": 5,
                })
            else:
                expected.update({
                    "ptx_tensor_prefetch": 0, "ptx_tensormap_acquire": 0,
                    "ptx_tensor_g2s": 0, "ptx_bulk_g2s": 1,
                    "sass_tensormap_prefetch": 0,
                    "sass_tensormap_invalidate": 0, "sass_tensor_g2s": 0,
                })
            for name, expected_value in expected.items():
                if actual[name] != expected_value:
                    mismatches[f"{family}.bank{bank}.{name}"] = {
                        "actual": actual[name], "expected": expected_value,
                    }
            if actual["sass_test_wait_total"] < 2:
                mismatches[f"{family}.bank{bank}.sass_test_wait_total"] = {
                    "actual": actual["sass_test_wait_total"], "expected_minimum": 2,
                }

        aggregate = {
            name: sum(values[name] for values in bank_counts.values())
            for name in next(iter(bank_counts.values()), {})
        }
        aggregate.update({
            "kernel_bank_count": len(bank_counts),
            "bank_pmevents_min": min(
                (value["ptx_pmevents"] for value in bank_counts.values()),
                default=0,
            ),
            "bank_pmevents_max": max(
                (value["ptx_pmevents"] for value in bank_counts.values()),
                default=0,
            ),
        })
        result["kernels"][family] = {"symbol": symbol, "checks": aggregate}

    result["ladder"] = {
        "points": POINTS,
        "banks": BANKS,
        "bank_size": BANK_SIZE,
        "requested_step_cycles": STEP,
        "indexed_dispatches_per_timed_path": 0,
    }
    result["passed"] = not mismatches
    if mismatches:
        result["mismatches"] = mismatches
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if mismatches:
        raise SystemExit("V12 PTX/SASS instruction contract failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
