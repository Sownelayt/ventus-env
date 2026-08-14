# Ventus TMA V3.5 area/timing synthesis

V3.5 keeps the V3.4 `40 LineContext + 6 PayloadSlot + 4 Descriptor`
configuration and optimizes only the standalone TMA. It does not modify L2,
shared memory, TLB, scheduler, ISA, or TensorMap format.

The release creates two independent DC projects from exactly the same PMU-off
RTL:

- `zero_wire`: historical matched comparison with V3.4;
- `boundary_typical`: an explicit block-level timing envelope for the missing
  physical context. It uses a N12 `BUFFD2` input driver, 10 fF output load,
  0.12 ns maximum I/O delay, 0.12 ns maximum transition, and a maximum
  fanout of 16.

Both use TSMC N12 `tcbn12ffcllbwp16p90cpdtt1v85c`, TT 1.0 V 85 °C,
1.5 GHz, and at most 12 host CPUs per independent project. The boundary model
is an engineering estimate, not a substitute for post-placement extraction.
ZeroWireload remains enabled in both because no floorplan or wire-load model
exists; the second run adds realistic boundary drive/load/budget instead of
inventing internal net lengths. The transition/fanout limits also force DC
to pay for required internal buffering, but this remains a pre-layout
estimate rather than an extracted interconnect result.

After RTL and cycle regression:

```bash
python3 benchmarks/tma-area-dc-v3.5/prepare_release.py
```

On the VM, launch each copied project in a separate detached session with
`DC_TIMEOUT_SECONDS=0`. The two projects have separate `WORK`, `DC_log`,
reports, and logs and cannot overwrite one another.
