#!/usr/bin/env python3
"""Package two independent Ventus TMA V3.5 DC projects.

Background:
  The standalone TMA has no floorplan, extracted interconnect, neighbouring
  blocks, or compiled SRAM models. V3.5 therefore keeps the historical
  ZeroWireload comparison and adds a second, explicit block-boundary timing
  envelope without inventing internal net lengths.

Flow:
  Validate the PMU-off RTL and structural audit, then package byte-identical
  RTL into independent zero-wire and boundary-typical projects. Each project
  has its own scripts, filelist, variant, WORK directory, logs, and reports.
  Existing projects are never overwritten.

Usage:
  python3 benchmarks/tma-area-dc-v3.5/prepare_release.py

Maintenance:
  Keep configs.json and the Tcl boundary defaults synchronized. Both projects
  must always contain the same RTL SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROJECT_ROOT = HERE / "release-project"
MANIFEST = HERE / "RTL_MANIFEST_FINAL.json"
MODELS = {
    "zero_wire": "v3_5_release_40x6_desc4_zero_wire",
    "boundary_typical": "v3_5_release_40x6_desc4_boundary_typical",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate() -> tuple[dict[str, object], Path, Path]:
    config_path = HERE / "configs.json"
    structure_path = HERE / "rtl_structure_summary.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    structure = json.loads(structure_path.read_text(encoding="utf-8"))
    if not structure.get("all_gates_passed"):
        raise SystemExit("refusing to package RTL with failed structure gates")
    source = ROOT / str(config["variant"]["rtl"])
    if not source.is_file():
        raise SystemExit(f"missing generated RTL: {source}")
    rtl_text = source.read_text(encoding="utf-8")
    forbidden = (
        "io_perf_s2g",
        " io_perf_",
        "TmaV2WindowTask",
        "windowData",
        "requestData",
        "sharedData",
    )
    hits = [token for token in forbidden if token in rtl_text]
    if hits:
        raise SystemExit(
            f"release RTL contains forbidden PMU/legacy state: {hits}"
        )
    existing = [
        PROJECT_ROOT / project
        for project in MODELS.values()
        if (PROJECT_ROOT / project).exists()
    ]
    if existing:
        paths = ", ".join(str(path) for path in existing)
        raise SystemExit(f"refusing to overwrite DC project(s): {paths}")
    return config, source, structure_path


def package_one(
    source: Path,
    model: str,
    project_name: str,
) -> tuple[Path, Path]:
    project = PROJECT_ROOT / project_name
    (project / "rtl").mkdir(parents=True)
    (project / "dc" / "filelists").mkdir(parents=True)
    target = project / "rtl" / "tma.sv"
    shutil.copy2(source, target)
    for name in ("run_dc_n12_tt1v85c_1500.tcl", "run_dc_remote.sh"):
        destination = project / "dc" / name
        shutil.copy2(HERE / "dc" / name, destination)
        destination.chmod(destination.stat().st_mode | 0o100)
    (project / "dc" / "filelists" / "tma_core_sv.f").write_text(
        "rtl/tma.sv\n", encoding="utf-8"
    )
    (project / "variant.env").write_text(
        f"export VARIANT={project_name}\n"
        "export DESIGN_TOP=tma\n"
        f"export TIMING_MODEL={model}\n"
        "export MAX_CORES=12\n"
        "export DC_TIMEOUT_SECONDS=0\n",
        encoding="utf-8",
    )
    return project, target


def main() -> None:
    config, source, structure_path = validate()
    projects: dict[str, object] = {}
    expected_hash = sha256(source)
    for model, project_name in MODELS.items():
        project, target = package_one(source, model, project_name)
        target_hash = sha256(target)
        if target_hash != expected_hash:
            raise SystemExit(f"packaged RTL hash mismatch: {target}")
        projects[model] = {
            "project": project.relative_to(ROOT).as_posix(),
            "packaged_rtl": target.relative_to(ROOT).as_posix(),
            "bytes": target.stat().st_size,
            "sha256": target_hash,
        }

    manifest = {
        "variant": config["variant"],
        "technology": config["technology"],
        "timing_models": config["timing_models"],
        "source": source.relative_to(ROOT).as_posix(),
        "source_sha256": expected_hash,
        "structure_summary_sha256": sha256(structure_path),
        "projects": projects,
        "rtl_identity_verified": (
            len({entry["sha256"] for entry in projects.values()}) == 1
        ),
    }
    MANIFEST.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print("packaged two independent V3.5 DC projects")
    for model, entry in projects.items():
        print(f"  {model}: {entry['project']}")
    print(f"  shared RTL SHA-256: {expected_hash}")


if __name__ == "__main__":
    main()
