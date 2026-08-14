# Ventus TMA V3.4 single release DC

This directory packages and launches exactly one standalone TMA synthesis:
`40 LineContext + 6 PayloadSlot + 4 Descriptor`. It does not include or
modify L2, shared memory, TLB, scheduler, or other GPU blocks.

The fixed conditions are TSMC N12
`tcbn12ffcllbwp16p90cpdtt1v85c`, TT 1.0 V 85 °C, 1.5 GHz and at most 12 host
CPUs. The input RTL is elaborated with both DMA PMU flags disabled.

After all RTL and cycle gates pass:

```bash
python3 benchmarks/tma-area-dc-v3.4/prepare_release.py
```

Copy `release-project/v3_4_release_40x6_desc4` to the VM as an independent
project. Launch it detached with:

```bash
MAX_CORES=12 DC_TIMEOUT_SECONDS=0 bash dc/run_dc_remote.sh
```

`DC_TIMEOUT_SECONDS=0` invokes `dc_shell` directly and deliberately installs
no timeout. A positive value remains available only for manual diagnostics.
The release gate is total cell area no greater than the completed V3.3
release (target at least 15% lower) and non-negative 1.5 GHz WNS. A failure
must be analyzed before any second DC point is started.
