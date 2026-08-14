# Ventus TMA V3.2 Design Compiler area sweep

This project synthesizes only the standalone `tma` boundary. It does not
include the GPU core, L1/L2 cache, shared memory implementation, scheduler or
execution units.

Common conditions:

- TSMC N12 CLN12FFCLL;
- `tcbn12ffcllbwp16p90cpdtt1v85c`;
- TT, 1.0 V, 85 °C;
- synthesis and reporting at 1.5 GHz (`0.666667 ns`);
- Design Compiler X-2025.06;
- `ZeroWireload`;
- PMU disabled by the existing standalone elaboration;
- at most 12 host CPUs per variant.

The three measured points match the earlier V3.1 sweep:

| Variant | Window | G2S request | Shared-ready | S2G ack | Descriptor |
| --- | ---: | ---: | ---: | ---: | ---: |
| `legacy_capacity` | 8 | 4 | 4 | 16 | 2 |
| `descriptor4` | 8 | 4 | 4 | 16 | 4 |
| `v3_2_full` | 24 | 16 | 8 | 32 | 4 |

Every directory below `variants/` is self-contained and has its own RTL,
Tcl, file list, logs and `DC_log`. Running the three variants concurrently
therefore does not share a DC WORK library or overwrite reports.

The remote run root is:

```text
/home/liyb/tma-dc/20260730_13_tma_v3_2_area_n12_tt1v85c_1500
```

From one variant directory:

```bash
MAX_CORES=12 DC_TIMEOUT_SECONDS=7200 DC_LICENSE_RETRIES=1 \
  bash dc/run_dc_remote.sh
```

After copying each `DC_log/<variant>/report` directory into
`results/<variant>/report`, run:

```bash
python3 benchmarks/tma-area-dc-v3.2/analyze_results.py
```

The generated V3.2 RTL must have zero runtime division/remainder and zero
`compiled_elementStrides` before upload. V3.1 results are historical inputs
only and are not written by this project.
