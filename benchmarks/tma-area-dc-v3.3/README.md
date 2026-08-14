# Ventus TMA V3.3 Design Compiler sweep

This directory synthesizes only the standalone TMA boundary. It does not
modify or include L2, shared memory, TLB, the scheduler, or unrelated GPU
execution units.

All points use TSMC N12
`tcbn12ffcllbwp16p90cpdtt1v85c`, TT 1.0 V 85 °C, 1.5 GHz
(`0.666667 ns`) and at most 12 host CPUs per process. V3.3 RTL is generated
with both DMA PMU flags set to zero. The archived V3.2 matched reference uses
the same 24/16/8/32/4 capacities; its top-level PMU output ports are removed
before optimization so its otherwise dead PMU cone does not contaminate the
comparison.

Prepare six independent projects. For a release/DC handoff, use an explicit
immutable directory and manifest:

```bash
python3 benchmarks/tma-area-dc-v3.3/prepare_variants.py \
  --variants-dir benchmarks/tma-area-dc-v3.3/variants-final \
  --manifest benchmarks/tma-area-dc-v3.3/RTL_MANIFEST_FINAL.json
```

Each `variants-final/<name>` directory contains its own RTL, file list, Tcl, WORK
path, log and report path. From an individual variant:

```bash
MAX_CORES=12 DC_TIMEOUT_SECONDS=14400 bash dc/run_dc_remote.sh
```

The four standalone points are V3.2 matched reference, V3.3 slot16/desc2
area-min, slot24/desc4 release, and slot32/desc4 capacity. Two additional
small projects synthesize Descriptor Service alone at desc2 and desc4.

After copying each `DC_log/<variant>/report` directory into the corresponding
local `results/<variant>/report`, run:

```bash
python3 benchmarks/tma-area-dc-v3.3/analyze_results.py
```
