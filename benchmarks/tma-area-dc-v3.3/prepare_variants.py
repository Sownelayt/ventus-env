#!/usr/bin/env python3
"""Package independent, self-contained TMA V3.3 Design Compiler projects.

Background:
  The release area sweep must compare immutable RTL points without sharing a
  WORK library, log, or report directory between concurrent DC processes.

Flow:
  Validate every generated RTL input, copy one top-level SystemVerilog file
  plus the common DC scripts into each variant, and emit a shell environment
  file and SHA-256 manifest. Existing variant directories are never replaced.

Usage:
  python3 benchmarks/tma-area-dc-v3.3/prepare_variants.py
  python3 benchmarks/tma-area-dc-v3.3/prepare_variants.py \
    --variants-dir benchmarks/tma-area-dc-v3.3/variants-final \
    --manifest benchmarks/tma-area-dc-v3.3/RTL_MANIFEST_FINAL.json

Maintenance:
  Add parameter points only in configs.json. The V3.2 reference deliberately
  removes its PMU output ports inside DC so unreachable PMU state is optimized;
  V3.3 RTL is elaborated with both PMU environment flags disabled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VARIANTS = HERE / "variants"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants-dir",
        default=str(VARIANTS),
        help="independent project output directory",
    )
    parser.add_argument(
        "--manifest",
        default=str(HERE / "RTL_MANIFEST.json"),
        help="SHA-256 manifest output path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = Path(args.variants_dir).resolve()
    manifest_path = Path(args.manifest).resolve()
    configs = json.loads((HERE / "configs.json").read_text(encoding="utf-8"))
    scripts = (
        HERE / "dc" / "run_dc_n12_tt1v85c_1500.tcl",
        HERE / "dc" / "run_dc_remote.sh",
    )
    for script in scripts:
        if not script.is_file():
            raise SystemExit(f"missing DC script: {script}")

    variants.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for config in configs["variants"]:
        name = config["name"]
        source = ROOT / config["rtl"]
        if not source.is_file():
            raise SystemExit(f"{name}: missing generated RTL {source}")
        target = variants / name
        if target.exists():
            raise SystemExit(
                f"refusing to overwrite independent project: {target}"
            )
        (target / "rtl").mkdir(parents=True)
        (target / "dc" / "filelists").mkdir(parents=True)
        rtl_target = target / "rtl" / source.name
        shutil.copy2(source, rtl_target)
        for script in scripts:
            shutil.copy2(script, target / "dc" / script.name)
        (target / "dc" / "filelists" / "tma_core_sv.f").write_text(
            f"rtl/{source.name}\n", encoding="utf-8"
        )
        env_lines = (
            f"export VARIANT={name}\n"
            f"export DESIGN_TOP={config['design_top']}\n"
            f"export DROP_PMU_OUTPUTS="
            f"{1 if config['drop_pmu_outputs'] else 0}\n"
        )
        (target / "variant.env").write_text(env_lines, encoding="utf-8")
        manifest.append(
            {
                "variant": name,
                "source": config["rtl"],
                "packaged": rtl_target.relative_to(ROOT).as_posix(),
                "bytes": rtl_target.stat().st_size,
                "sha256": sha256(rtl_target),
                "design_top": config["design_top"],
                "drop_pmu_outputs": config["drop_pmu_outputs"],
            }
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"packaged {len(manifest)} independent projects under {variants}")


if __name__ == "__main__":
    main()
