# Ventus TMA V3.1 Design Compiler area sweep

> **已停止，不可作为 V3.2 面积结果。** 2026-07-30 已精确终止虚拟机
> `/home/liyb/tma-dc/20260730_12_tma_v3_1_area_n12_tt1v85c_1500/variants`
> 下三个独立 variant 的 DC 进程，并确认该目录剩余进程数为 0。随后
> V3.2 删除了 elementStride 数据通路及所有运行时除法/取余，因此这里
> 的 V3.1 RTL、运行目录和未完成报告只保留为历史输入；不得外推或冒充
> V3.2 面积。V3.2 如需面积数据，应从新的生成 RTL 建立全新 DC 项目。

This directory measures the standalone `tma` synthesis boundary at the
requested TSMC N12 corner and clock:

- library: `tcbn12ffcllbwp16p90cpdtt1v85c`
- process corner: TT, 1.0 V, 85 °C
- synthesis and reporting clock: 1.5 GHz (0.666667 ns)
- Design Compiler: X-2025.06
- wire-load model: `ZeroWireload`
- PMU: disabled by the existing Ventus elaboration configuration
- maximum host CPUs per synthesis point: 12

The sweep deliberately uses three measured points:

| Variant | Window | G2S request | Shared-ready | S2G ack | Descriptor |
| --- | ---: | ---: | ---: | ---: | ---: |
| `legacy_capacity` | 8 | 4 | 4 | 16 | 2 |
| `descriptor4` | 8 | 4 | 4 | 16 | 4 |
| `v3_1_full` | 24 | 16 | 8 | 32 | 4 |

`legacy_capacity` is V3.1 logic with the old small queue capacities; it is not
the V3.0 RTL.  The first delta isolates descriptor-store scaling and the second
captures the aggregate backend-capacity increase. Other points are estimates,
not DC measurements.

## Generate RTL

From `gpgpu/`:

```bash
./mill -i 'ventus[6.4.0].runMain' top.TmaV2_gen \
  generated-tma-v3.1-area-legacy-capacity \
  --window-entries 8 --request-entries 4 --shared-entries 4 \
  --write-ack-entries 16 --descriptor-entries 2
```

Repeat with the parameters in `configs.json`. The generated `tma.sv` files are
copied into `variants/<name>/rtl/tma.sv`; the remote package is therefore
self-contained.

## Run on the DC virtual machine

Copy this directory to `/home/liyb/tma-dc/`, enter each variant directory, and
run:

```bash
MAX_CORES=12 DC_TIMEOUT_SECONDS=7200 bash dc/run_dc_remote.sh
```

The runner first sources `/eda/synopsys/env.sh`, checks whether it selected
the `tools_patched` binary, and otherwise sources
`/eda/synopsys/env_crack.sh`. It invokes the patched DC by absolute path.
It rejects `MAX_CORES > 12`, performs no automatic license retry by default,
and bounds each `dc_shell` invocation with a two-hour timeout.

## Analyze

After copying each `DC_log/<variant>/report/` directory into
`results/<variant>/report/`, run:

```bash
python3 analyze_results.py
```

This writes `results/area_summary.csv` and prints the measured deltas. Exact
cell area is meaningful only for this library, synthesis recipe, boundary, and
wire-load assumption. No SRAM macros are inferred: Chisel `Reg(Vec(...))`
storage maps to standard-cell flops and muxes, which is intentional here
because it exposes the current RTL cost.
