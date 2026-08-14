# Ventus TMA V3.6 all-path timing release

V3.6 is a TMA-only timing pass over the V3.5 `40 LineContext + 6
PayloadSlot + 4 Descriptor` design.  It does not change L2, shared memory,
TLB, the scheduler, ISA, or TensorMap format.

The optimization target is **all setup paths at 1.6 GHz**, not only the
single worst path.  The RTL therefore cuts every path family that contributed
material V3.5 TNS:

- LineContext route/address updates and payload metadata/data;
- cache and shared request/response block boundaries;
- Planner cursor, coordinate, global-map, and lane-map stages;
- Binder multiplication and result assembly;
- Descriptor validation, compiler working state, and store commit;
- command ingress, TLB issue, status, and completion boundaries.

The DC project creates separate path groups for Engine, Planner, Binder,
Descriptor, Ingress, the remaining register paths, and all four I/O path
classes.  Release acceptance requires 1.6 GHz setup `WNS >= 0`,
`TNS = 0`, and zero setup violating paths in every path group.  Hold timing
is reported but is not a synthesis-release criterion: without CTS, routed
clock skew, or extracted wires, standalone ZeroWireload hold repair is not a
physical prediction.

The boundary model is the same engineering envelope as V3.5: N12
`tcbn12ffcllbwp16p90cpdtt1v85c`, TT 1.0 V 85 °C, BUFFD2 input drive,
10 fF output load, 0.12 ns maximum I/O delay and transition, and maximum
fanout 16.  Internal wires still use ZeroWireload because no floorplan or
wire-load model is available.

The only release project is:

```text
release-project/v3_6_release_40x6_desc4_boundary_allpaths_1600
```

It uses at most 12 host CPUs and `DC_TIMEOUT_SECONDS=0`.  The remote run is
started only after Chisel, GVM, cycle, and structure checks finish.

