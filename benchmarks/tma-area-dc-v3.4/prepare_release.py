#!/usr/bin/env python3
"""Package the one final, independent Ventus TMA V3.4 DC project.

Background:
  V3.4 selects 40 LineContexts plus six PayloadSlots after RTL and cycle
  gates. Only this release point is allowed to enter the expensive N12 run.

Flow:
  Validate the PMU-off standalone RTL and structural audit, copy the RTL and
  DC scripts into a new self-contained project, and emit a SHA-256 manifest.
  An existing project is never overwritten.

Usage:
  python3 benchmarks/tma-area-dc-v3.4/prepare_release.py

Maintenance:
  Keep configs.json synchronized with TmaV2Spec defaults. Do not add a second
  variant here; a failed mapped gate requires analysis before another run.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROJECT = HERE / "release-project" / "v3_4_release_40x6_desc4"
MANIFEST = HERE / "RTL_MANIFEST_FINAL.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    config = json.loads(
        (HERE / "configs.json").read_text(encoding="utf-8")
    )["variant"]
    structure = json.loads(
        (HERE / "rtl_structure_summary.json").read_text(encoding="utf-8")
    )
    if not structure.get("all_gates_passed"):
        raise SystemExit("refusing to package RTL with failed structure gates")
    source = ROOT / config["rtl"]
    if not source.is_file():
        raise SystemExit(f"missing generated RTL: {source}")
    rtl_text = source.read_text(encoding="utf-8")
    if "io_perf_s2g" in rtl_text:
        raise SystemExit("release RTL unexpectedly contains DMA PMU outputs")
    if PROJECT.exists():
        raise SystemExit(
            f"refusing to overwrite independent DC project: {PROJECT}"
        )

    (PROJECT / "rtl").mkdir(parents=True)
    (PROJECT / "dc" / "filelists").mkdir(parents=True)
    target = PROJECT / "rtl" / "tma.sv"
    shutil.copy2(source, target)
    for name in ("run_dc_n12_tt1v85c_1500.tcl", "run_dc_remote.sh"):
        shutil.copy2(HERE / "dc" / name, PROJECT / "dc" / name)
    (PROJECT / "dc" / "filelists" / "tma_core_sv.f").write_text(
        "rtl/tma.sv\n", encoding="utf-8"
    )
    (PROJECT / "variant.env").write_text(
        "export VARIANT=v3_4_release_40x6_desc4\n"
        "export DESIGN_TOP=tma\n",
        encoding="utf-8",
    )
    manifest = {
        "variant": config,
        "source": config["rtl"],
        "packaged": target.relative_to(ROOT).as_posix(),
        "bytes": target.stat().st_size,
        "sha256": sha256(target),
        "structure_summary_sha256": sha256(
            HERE / "rtl_structure_summary.json"
        ),
        "project": PROJECT.relative_to(ROOT).as_posix(),
    }
    MANIFEST.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"packaged one release project: {PROJECT}")


if __name__ == "__main__":
    main()
