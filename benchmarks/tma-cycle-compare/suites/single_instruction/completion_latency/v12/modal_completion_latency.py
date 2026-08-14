"""Run the ColdThenHotV1 V12 G2S completion test on one NVIDIA GPU."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal


HERE = Path(__file__).resolve().parent
# The same module is imported again inside Modal as /root/*.py, where walking
# parents from __file__ no longer reaches the local checkout.
ROOT = Path("/home/liyb/ventus-env")
SOURCE = HERE / "cuda_g2s_fresh_map.cu"
METHOD = HERE / "METHOD.md"
AUDITOR = HERE / "audit_instruction_stream.py"
LADDER = HERE / "delay_ladder_v12.inc"
LADDER_GENERATOR = HERE / "generate_delay_ladder.py"
COMMON = HERE.parent.parent / "common_matrix/v3/tma_comprehensive_matrix.h"
REFERENCE = Path("/home/liyb/tma-refer/src/tma_ventus_parity_bench_v2.cu")
MODEL_DIR = ROOT / "testcases/_get_case/common"
MODEL_FILES = [
    MODEL_DIR / "tma_model.cc",
    MODEL_DIR / "tma_model.h",
    MODEL_DIR / "ventus_tma_v2_spec.h",
]

REMOTE = "/opt/tma-completion-latency-v12"
IMAGE = "nvidia/cuda:13.3.0-devel-ubuntu24.04"
TARGETS = {
    "h100": {"gpu": "H100!", "arch": "sm_90", "cc": "9.0"},
    "b200": {"gpu": "B200", "arch": "sm_100", "cc": "10.0"},
}
VARIANTS = {
    "smoke": {
        "project": "smoke", "series": 5, "shards": 1,
    },
    "bulk-tensor__g2s__fresh-map-common": {
        "project": "common", "series": 243, "shards": 8,
    },
    "tensor__g2s__fresh-map-2d": {
        "project": "tensor_2d", "series": 182, "shards": 8,
    },
}
SCAN_POINTS = 640
REQUESTED_STEP_CYCLES = 3
CHILD_TIMEOUT = 5
OUTER_TIMEOUT = 10
WAIT_METHOD = "single_test_wait_cold_then_hot_v12"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


LOCAL_FILES = {
    SOURCE.name: SOURCE,
    METHOD.name: METHOD,
    AUDITOR.name: AUDITOR,
    LADDER.name: LADDER,
    LADDER_GENERATOR.name: LADDER_GENERATOR,
    "tma_comprehensive_matrix.h": COMMON,
    REFERENCE.name: REFERENCE,
    **{path.name: path for path in MODEL_FILES},
}


def build(arch: str) -> str:
    artifact = f"{REMOTE}/artifacts/{arch}"
    include = (
        f"-I{REMOTE}/refer -I{REMOTE}/common -I{REMOTE}/model"
    )
    common = (
        f"nvcc -O3 -std=c++20 -arch={arch} {include} "
        f"{REMOTE}/v12/{SOURCE.name}"
    )
    return (
        "bash -lc 'set -euo pipefail; "
        f"mkdir -p {artifact}; "
        f"{common} -o {artifact}/probe -lcuda; "
        f"{common} -ptx -o {artifact}/probe.ptx -lcuda; "
        f"{common} -cubin -o {artifact}/probe.cubin -lcuda; "
        f"cuobjdump --dump-sass {artifact}/probe.cubin > {artifact}/probe.sass; "
        f"python3 {REMOTE}/v12/{AUDITOR.name} "
        f"{artifact}/probe.ptx {artifact}/probe.sass "
        f"--output {artifact}/instruction_audit.json; "
        f"sha256sum {artifact}/probe.ptx {artifact}/probe.cubin "
        f"{artifact}/probe.sass {artifact}/instruction_audit.json "
        f"> {artifact}/artifact_hashes.txt; "
        f"test -s {artifact}/probe; test -s {artifact}/artifact_hashes.txt'"
    )


app = modal.App("ventus-tma-completion-latency-v12")
base = (
    modal.Image.from_registry(IMAGE, add_python="3.12")
    .add_local_file(SOURCE, f"{REMOTE}/v12/{SOURCE.name}", copy=True)
    .add_local_file(METHOD, f"{REMOTE}/v12/{METHOD.name}", copy=True)
    .add_local_file(AUDITOR, f"{REMOTE}/v12/{AUDITOR.name}", copy=True)
    .add_local_file(LADDER, f"{REMOTE}/v12/{LADDER.name}", copy=True)
    .add_local_file(
        LADDER_GENERATOR, f"{REMOTE}/v12/{LADDER_GENERATOR.name}", copy=True
    )
    .add_local_file(COMMON, f"{REMOTE}/common/{COMMON.name}", copy=True)
    .add_local_file(REFERENCE, f"{REMOTE}/refer/{REFERENCE.name}", copy=True)
    .add_local_file(MODEL_FILES[0], f"{REMOTE}/model/{MODEL_FILES[0].name}", copy=True)
    .add_local_file(MODEL_FILES[1], f"{REMOTE}/model/{MODEL_FILES[1].name}", copy=True)
    .add_local_file(MODEL_FILES[2], f"{REMOTE}/model/{MODEL_FILES[2].name}", copy=True)
)


def image(arch: str) -> modal.Image:
    return base.run_commands(build(arch))


def execute(
    label: str, project: str, shard_index: int, shard_count: int
) -> dict[str, Any]:
    artifact = Path(f"{REMOTE}/artifacts/{TARGETS[label]['arch']}")
    command = [
        str(artifact / "probe"), "--project", project,
        "--shard-index", str(shard_index), "--shard-count", str(shard_count),
    ]
    started = time.monotonic()
    try:
        result = subprocess.run(
            command, capture_output=True, timeout=CHILD_TIMEOUT, check=False
        )
        payload: dict[str, Any] = {
            "classification": "completed",
            "returncode": result.returncode,
            "stdout_gzip": gzip.compress(result.stdout, compresslevel=6),
            "stderr": result.stderr.decode("utf-8", errors="replace"),
        }
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or b""
        stderr = error.stderr or b""
        payload = {
            "classification": "child_timeout", "returncode": 124,
            "stdout_gzip": gzip.compress(stdout, compresslevel=6),
            "stderr": stderr.decode("utf-8", errors="replace"),
        }
    payload.update({
        "gpu": label,
        "project": project,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "command": command,
        "wall_seconds": time.monotonic() - started,
        "child_timeout_seconds": CHILD_TIMEOUT,
        "function_timeout_seconds": OUTER_TIMEOUT,
        "artifact_hashes": (artifact / "artifact_hashes.txt").read_text(
            encoding="utf-8"
        ),
        "instruction_audit": json.loads(
            (artifact / "instruction_audit.json").read_text(encoding="utf-8")
        ),
        "wait_audit": {
            "timed_test_wait_sites_per_selected_kernel": 1,
            "timed_commands_per_kernel": 2,
            "cleanup_try_wait_sites_per_selected_kernel": 1,
            "timed_poll_loop": False,
            "cleanup_after_probe_timestamps": True,
        },
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,compute_cap,driver_version,clocks.current.sm",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True, check=False,
    )
    payload["nvidia_smi_query"] = (
        query.stdout + query.stderr
    ).decode("utf-8", errors="replace")
    return payload


@app.function(
    image=image("sm_90"), gpu="H100!", timeout=OUTER_TIMEOUT,
    retries=0, max_containers=1,
)
def run_h100(
    project: str, shard_index: int, shard_count: int
) -> dict[str, Any]:
    return execute("h100", project, shard_index, shard_count)


@app.function(
    image=image("sm_100"), gpu="B200", timeout=OUTER_TIMEOUT,
    retries=0, max_containers=1,
)
def run_b200(
    project: str, shard_index: int, shard_count: int
) -> dict[str, Any]:
    return execute("b200", project, shard_index, shard_count)


@app.function(image=base, timeout=OUTER_TIMEOUT, retries=0, max_containers=1)
def import_probe() -> dict[str, str]:
    remote_files = {
        SOURCE.name: Path(f"{REMOTE}/v12/{SOURCE.name}"),
        METHOD.name: Path(f"{REMOTE}/v12/{METHOD.name}"),
        AUDITOR.name: Path(f"{REMOTE}/v12/{AUDITOR.name}"),
        LADDER.name: Path(f"{REMOTE}/v12/{LADDER.name}"),
        LADDER_GENERATOR.name: Path(f"{REMOTE}/v12/{LADDER_GENERATOR.name}"),
        COMMON.name: Path(f"{REMOTE}/common/{COMMON.name}"),
        REFERENCE.name: Path(f"{REMOTE}/refer/{REFERENCE.name}"),
        **{
            path.name: Path(f"{REMOTE}/model/{path.name}")
            for path in MODEL_FILES
        },
    }
    return {name: sha256(path) for name, path in remote_files.items()}


def expected_hashes() -> dict[str, str]:
    return {name: sha256(path) for name, path in LOCAL_FILES.items()}


def decode_stdout(payload: dict[str, Any]) -> str:
    compressed = payload.pop("stdout_gzip", b"")
    return gzip.decompress(compressed).decode("utf-8")


def validate(
    payload: dict[str, Any], label: str, variant: str, raw: str,
    expected_series: int,
) -> list[dict[str, str]]:
    config = VARIANTS[variant]
    if payload.get("classification") != "completed" or payload.get("returncode") != 0:
        raise SystemExit(
            f"{label}: {payload.get('classification')} "
            f"returncode={payload.get('returncode')}"
        )
    rows = list(csv.DictReader(io.StringIO(raw)))
    expected_cases = expected_series * SCAN_POINTS
    expected_rows = expected_cases * 2
    if len(rows) != expected_rows:
        raise SystemExit(
            f"expected {expected_rows} rows, got {len(rows)}"
        )
    if any(row["cc"] != TARGETS[label]["cc"] for row in rows):
        raise SystemExit("compute capability mismatch")
    if any(row["project"] != config["project"] for row in rows):
        raise SystemExit("project mismatch")
    if any(row["wait_method"] != WAIT_METHOD for row in rows):
        raise SystemExit("wait method mismatch")
    error_rows = [row for row in rows if int(row["errors"]) != 0]
    if error_rows:
        details = "; ".join(
            f"{row['base_case_id']}/{row['descriptor_mode']}/"
            f"d{row['delay_iterations']}/r{row['repeat']}:"
            f"expected={row.get('expected_payload_checksum', '?')},"
            f"actual={row.get('payload_checksum', '?')}"
            for row in error_rows[:8]
        )
        raise SystemExit(f"payload correctness failure: {details}")

    grouped_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped_rows[row["case_id"]].append(row)
        success = int(row["probe_success"])
        cleanup = int(row["cleanup_tries"])
        if (success and cleanup != 0) or (not success and cleanup < 1):
            raise SystemExit("cleanup ran on the wrong side of the timed probe")
        if int(row["probe_end_from_issue_begin_cycles"]) < int(
            row["probe_begin_from_issue_begin_cycles"]
        ):
            raise SystemExit("probe timestamps are not ordered")
    if len(grouped_rows) != expected_cases:
        raise SystemExit("measurement-point count mismatch")
    first_global_bases: set[int] = set()
    first_descriptor_addresses: set[int] = set()
    first_descriptor_fingerprints: set[int] = set()
    for case_id, case_rows in grouped_rows.items():
        ordered = sorted(case_rows, key=lambda row: int(row["repeat"]))
        if [row["repeat"] for row in ordered] != ["0", "1"]:
            raise SystemExit(f"{case_id}: missing repeat 0/1")
        global_bases = {int(row["global_base_va"], 16) for row in ordered}
        if len(global_bases) != 1:
            raise SystemExit(f"{case_id}: cold/hot global base changed")
        global_base = next(iter(global_bases))
        if global_base in first_global_bases:
            raise SystemExit(f"{case_id}: global base reused by another case")
        first_global_bases.add(global_base)
        if ordered[0]["path"] != "tensor":
            continue
        expected_roles = (
            ["prefetched", "hot"]
            if ordered[0]["descriptor_mode"] == "tmau_prefetched_l2_hot"
            else ["cold", "hot"]
        )
        if [row["pair_role"] for row in ordered] != expected_roles:
            raise SystemExit(f"{case_id}: wrong cold/hot role")
        if [row["prior_tma_use"] for row in ordered] != ["0", "1"]:
            raise SystemExit(f"{case_id}: prior use is not 0 then 1")
        addresses = {int(row["descriptor_va"], 16) for row in ordered}
        fingerprints = {
            int(row["descriptor_fingerprint64"], 16) for row in ordered
        }
        if len(addresses) != 1 or len(fingerprints) != 1:
            raise SystemExit(f"{case_id}: TensorMap changed between pair")
        address = next(iter(addresses))
        fingerprint = next(iter(fingerprints))
        if address % 4096:
            raise SystemExit(f"{case_id}: TensorMap is not 4 KiB aligned")
        if address in first_descriptor_addresses:
            raise SystemExit(f"{case_id}: TensorMap address reused by another case")
        if fingerprint in first_descriptor_fingerprints:
            raise SystemExit(f"{case_id}: TensorMap contents reused by another case")
        first_descriptor_addresses.add(address)
        first_descriptor_fingerprints.add(fingerprint)
        if any(row["descriptor_l2_warm_method"] != "ld.global.cg" for row in ordered):
            raise SystemExit(f"{case_id}: descriptor not made L2-hot by ordinary load")
        expected_prefetch = ["1", "0"] if expected_roles[0] == "prefetched" else ["0", "0"]
        if [row["tma_prefetch_before_issue"] for row in ordered] != expected_prefetch:
            raise SystemExit(f"{case_id}: prefetch marker is wrong")
    if not payload.get("instruction_audit", {}).get("passed"):
        raise SystemExit("PTX/SASS instruction audit failed")
    no_prefetch = payload["instruction_audit"]["kernels"]["tensor_first_use"][
        "checks"
    ]
    if no_prefetch["ptx_tensor_prefetch"] or no_prefetch["sass_tensormap_prefetch"]:
        raise SystemExit("the first-use kernel contains a TensorMap prefetch")
    return rows


def summarize(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], dict[int, list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    fields = (
        "project", "family", "base_case_id", "path", "dtype", "dtype_bits",
        "rank", "logical_bytes", "global_span", "global_offset", "interleave",
        "swizzle", "oob_fill", "descriptor_mode",
    )
    for row in rows:
        groups[tuple(row[name] for name in fields)][int(row["repeat"])].append(row)

    output: list[dict[str, Any]] = []
    for key, repeats in sorted(groups.items()):
        result: dict[str, Any] = dict(zip(fields, key))
        repeat_estimates: list[int] = []
        uncertainties: list[int] = []
        for repeat in (0, 1):
            samples = sorted(
                repeats[repeat],
                key=lambda row: int(row["probe_begin_from_issue_begin_cycles"]),
            )
            outcomes = [int(row["probe_success"]) for row in samples]
            if not any(outcomes) or all(outcomes):
                raise SystemExit(f"completion is not bracketed: {key}, repeat={repeat}")
            best_error = len(samples) + 1
            best_cuts: list[int] = []
            total_false = len(outcomes) - sum(outcomes)
            prefix_true = 0
            prefix_false = 0
            for cut in range(1, len(samples)):
                previous = outcomes[cut - 1]
                prefix_true += previous
                prefix_false += 1 - previous
                error = prefix_true + (total_false - prefix_false)
                if error < best_error:
                    best_error = error
                    best_cuts = [cut]
                elif error == best_error:
                    best_cuts.append(cut)
            # The 640 points are independent commands.  Fine spacing exposes
            # normal command-to-command latency variation as crossings around
            # the monotone cut; retain the exact count instead of rerunning or
            # cherry-picking.  The acceptance ratio remains V11's 18/96.
            allowed_error = (len(samples) * 18 + 95) // 96
            if best_error > allowed_error:
                raise SystemExit(
                    f"completion transition is too noisy: {key}, "
                    f"repeat={repeat}, crossed_samples={best_error}/{len(samples)}, "
                    f"allowed={allowed_error}/{len(samples)}"
                )
            cut = best_cuts[len(best_cuts) // 2]
            lower = int(samples[cut - 1]["probe_begin_from_issue_begin_cycles"])
            upper = int(samples[cut]["probe_end_from_issue_begin_cycles"])
            if upper <= lower:
                raise SystemExit(f"invalid completion boundary: {key}, repeat={repeat}")
            estimate = (lower + upper + 1) // 2
            uncertainty = (upper - lower + 1) // 2
            prefix = f"repeat{repeat}"
            result[f"{prefix}_lower"] = lower
            result[f"{prefix}_upper"] = upper
            result[f"{prefix}_cycles"] = estimate
            result[f"{prefix}_uncertainty"] = uncertainty
            result[f"{prefix}_crossed_samples"] = best_error
            result[f"{prefix}_allowed_crossed_samples"] = allowed_error
            result[f"{prefix}_equally_good_cuts"] = len(best_cuts)
            begin_positions = sorted({
                int(sample["probe_begin_from_issue_begin_cycles"])
                for sample in samples
            })
            gaps = [
                right - left
                for left, right in zip(begin_positions, begin_positions[1:])
            ]
            result[f"{prefix}_max_probe_begin_gap"] = max(gaps, default=0)
            repeat_estimates.append(estimate)
            uncertainties.append(uncertainty)
        result["completion_cycles"] = (
            repeat_estimates[0] + repeat_estimates[1] + 1
        ) // 2
        result["measurement_uncertainty_cycles"] = max(uncertainties)
        result["repeat_difference_cycles"] = (
            repeat_estimates[1] - repeat_estimates[0]
        )
        result["high_variance"] = int(
            max(
                int(result["repeat0_crossed_samples"]),
                int(result["repeat1_crossed_samples"]),
            ) > 4
        )
        output.append(result)
    return output


def encode_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def plain_report(device: str, variant: str, rows: list[dict[str, Any]]) -> str:
    state_names = {
        "not_applicable": "Bulk（没有 TensorMap）",
        "tmau_first_use_l2_hot": "在 L2，TMAU 第一次使用",
        "tmau_prefetched_l2_hot": "已经提前送入 TMAU",
    }
    lines = [
        f"# {device.upper()} G2S 完成周期",
        "",
        "每个测试点只分配一个新的 TensorMap。第一次测量是该地址第一次进入 TMAU，完成后在同一个 kernel 内立刻用完全相同的地址和内容测第二次。主测试没有 TensorMap prefetch；普通全局读取只把 descriptor 放进 L2。",
        "",
        "表格分别显示第一次和第二次，不再把两次平均成一个数字。每个数字来自一轮固定的 640 点扫描：80 个静态 kernel bank 各负责 8 个点，bank 在计时外选择；计时路径没有索引跳转，相邻点静态增加一条比较、一条统一分支和一条不访问数据通路的 PM-event。请求步长为三周期，实际物理间隔由 clock64 记录，原始测量点、地址和指纹保存在同目录压缩 CSV。",
        "",
        f"测试组：`{variant}`",
        "",
    ]
    current_family = None
    for row in rows:
        family = str(row["family"])
        if family != current_family:
            if current_family is not None:
                lines.append("")
            lines.extend([
                f"## {family}", "",
                "|测试项|类型|数据格式|rank|字节数|TensorMap 状态|第一次|第二次|第二次-第一次|",
                "|---|---|---|---:|---:|---|---:|---:|---:|",
            ])
            current_family = family
        lines.append(
            f"|{row['base_case_id']}|{row['path']}|{row['dtype']}|{row['rank']}|"
            f"{row['logical_bytes']}|{state_names[str(row['descriptor_mode'])]}|"
            f"{row['repeat0_cycles']}|{row['repeat1_cycles']}|"
            f"{row['repeat_difference_cycles']}|"
        )
    lines.extend([
        "", "## 怎样理解两种 Tensor 结果", "",
        "“在 L2，TMAU 第一次使用”的第一列包含从 L2 取回 TensorMap并交给 TMAU 处理的时间；第二列是同一个 TensorMap 的立即复用。“已经提前送入 TMAU”是独立控制项，第一次发令前显式 prefetch，第二次仍是相同地址的热复用。", "",
    ])
    return "\n".join(lines)


@app.local_entrypoint()
def main(
    gpu: str = "h100", variant: str = "bulk-tensor__g2s__fresh-map-common",
    import_only: bool = False, output: str = "",
) -> None:
    if gpu not in TARGETS:
        raise SystemExit("--gpu must be h100 or b200")
    if variant not in VARIANTS:
        raise SystemExit("unknown --variant")
    expected = expected_hashes()
    if import_only:
        if import_probe.remote() != expected:
            raise SystemExit("mounted source hash mismatch")
        print(f"CPU-only import probe passed; source SHA-256={expected[SOURCE.name]}")
        return

    project = str(VARIANTS[variant]["project"])
    total_series = int(VARIANTS[variant]["series"])
    shard_count = int(VARIANTS[variant]["shards"])
    output_path = Path(output).resolve() if output else HERE / "data" / gpu / variant
    output_path.mkdir(parents=True, exist_ok=True)
    runner = run_h100 if gpu == "h100" else run_b200
    rows: list[dict[str, str]] = []
    shard_metadata: list[dict[str, Any]] = []
    for shard_index in range(shard_count):
        expected_series = (
            total_series + shard_count - 1 - shard_index
        ) // shard_count
        payload = runner.remote(project, shard_index, shard_count)
        raw = decode_stdout(payload)
        shard_rows = validate(
            payload, gpu, variant, raw, expected_series
        )
        rows.extend(shard_rows)
        suffix = f"_shard_{shard_index:02d}"
        with gzip.open(
            output_path / f"attempt{suffix}.csv.gz",
            "wt", encoding="utf-8", newline="",
        ) as sink:
            sink.write(raw)
        stderr = payload.pop("stderr")
        (output_path / f"attempt{suffix}.stderr.txt").write_text(
            str(stderr), encoding="utf-8"
        )
        (output_path / f"attempt{suffix}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        shard_metadata.append({
            "shard_index": shard_index,
            "series": expected_series,
            "wall_seconds": payload["wall_seconds"],
            "classification": payload["classification"],
        })
    summary = summarize(rows)
    (output_path / "completion_cycles.csv").write_text(
        encode_csv(summary), encoding="utf-8"
    )
    (output_path / "REPORT.md").write_text(
        plain_report(gpu, variant, summary), encoding="utf-8"
    )
    manifest = {
        "gpu": gpu,
        "variant": variant,
        "project": project,
        "warmups": 0,
        "repeats": 2,
        "paid_attempts": 1,
        "measurement_release": "ColdThenHotV1",
        "pairing": "same-kernel cold-then-hot on the exact same TensorMap",
        "source_hashes": expected,
        "raw_rows": len(rows),
        "case_count": total_series * SCAN_POINTS,
        "summary_rows": len(summary),
        "wait_method": WAIT_METHOD,
        "scan": {
            "kind": "single_nonadaptive_80x8_static_kernel_bank_ladder",
            "requested_step_cycles": REQUESTED_STEP_CYCLES,
            "banks": 80,
            "positions_per_bank": 8,
            "indexed_dispatches_per_timed_path": 0,
            "scan_index": {
                "minimum": 0, "maximum": SCAN_POINTS - 1,
                "points": SCAN_POINTS,
            },
            "total_probe_positions": SCAN_POINTS,
        },
        "paid_shards": shard_count,
        "case_shards": shard_metadata,
        "primary_metric": "one integer completion cycle estimate",
        "tensormap_contract": {
            "l2_residency": "ordinary ld.global.cg over all four 32-byte sectors",
            "prior_tma_use": [0, 1],
            "address_reuse_within_pair": True,
            "content_reuse_within_pair": True,
            "address_reuse_between_cases": False,
            "address_spacing_bytes": 4096,
            "no_prefetch_kernel_verified_in_ptx_and_sass": True,
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"saved: {output_path}")
