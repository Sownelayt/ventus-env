# Ventus TMA V3.7 all-path timing release

V3.7 is a TMA-only timing-closure pass over the V3.6 `40 LineContext + 6
PayloadSlot + 4 Descriptor` organization.  It does not modify L2, shared
memory, TLB, the scheduler, ISA, or TensorMap memory format.

The implementation attacks every material V3.6 setup path family rather than
adding a register only to the single WNS path:

- Planner cursor expansion now uses parallel rank/lane prefix arithmetic and
  an elastic coordinate-map boundary;
- Binder and Descriptor Compiler replace wide monolithic products with
  balanced 16-bit-limb products and registered result assembly;
- Engine cache/shared route formation uses low-bit carry/tag arithmetic, one
  selected-lane shared shifter, and engine-local one-entry cache/shared egress
  boundaries;
- the S2G shared-response path captures data before line decomposition while a
  compact precomputed route seed preserves the no-regression first-fragment
  schedule;
- DmaCore no longer duplicates the two egress queues outside the Engine.

## Frozen pre-DC evidence

The PMU-off RTL is `generated-tma-v2-v3.7-work/tma.sv`, SHA-256
`8c8ba2136a408f655aa471ff6552802808e620d2d12b81a06c29c2bb7bbc0368`.
The release-project copy has the same hash.

The structural audit is in `rtl_structure_summary.json` and passes every gate:

| Item | V3.7 |
|---|---:|
| standalone TMA sequential state | 34,593 bit |
| WindowEngine hierarchy | 21,832 bit |
| WindowEngine data core, excluding relocated egress boundaries | 19,216 bit |
| cache egress boundary | 1,196 bit |
| shared egress boundary | 1,420 bit |
| LineContext | 40 × 220 bit |
| PayloadSlot | 6 × 1,465 bit |
| resident 1024-bit payload copies | 6 |
| Planner elastic queues | 2,250 bit |

The egress registers are excluded only from the *data-core diagnostic*; both
remain included in the 34,593-bit standalone total.  Runtime divide/remainder,
PMU state, old fast-path state, oversized dynamic shifts, response broadcast,
and duplicated payload banks are all absent.

The fixed capacity performance result is
`../tma-cycle-compare/results/features_ventus_v3_7_timing_allpaths_capacity`.
All 27 G2S/S2G/roundtrip cases pass.  Relative to V3.6, single-direction cases
improve by 1–2 cycles and roundtrip improves by 2–3 cycles; no case regresses.

## Final DC sweep

The final target was changed to 1.5 GHz.  Four independent projects run in
parallel:

```text
release-project/v3_7_line40_payload6_boundary_1500
release-project/v3_7_line40_payload6_zerowire_1500
release-project/v3_7_line63_payload6_boundary_1500
release-project/v3_7_line40_payload8_boundary_1500
```

The first two isolate the boundary drive/load contribution using exactly the
same 40+6 RTL.  The third measures maximum legal LineContext capacity: the
6-bit source ID value 63 is reserved for descriptor refill, and the RTL
therefore enforces `requestEntries <= 63`; a nominal 64+6 point would be
functionally invalid.  The fourth measures the mapped cost of two additional
1024-bit PayloadSlots.  Pre-map sequential state is 34,593 bit for 40+6,
39,653 bit for 63+6, and 37,523 bit for 40+8.

All projects use TSMC N12 `tcbn12ffcllbwp16p90cpdtt1v85c`, TT 1.0 V 85 °C,
1.5 GHz, no PMU, at most 12 CPUs per project, and no process timeout.
Boundary projects use BUFFD2 input drive, 10 fF output load, 0.12 ns maximum
I/O delay and transition, and maximum fanout 16.  The ZeroWireload project
omits only the external boundary envelope; internal wires remain ZeroWireload
in both models because no floorplan/extracted wire model exists.

Unlike V3.6, path groups are reconstructed by module `ref_name` after retime,
before incremental compile, and again after final renaming.  Planner, Engine,
Binder, Descriptor, Ingress, and DmaCore-local control/queue endpoints are
mutually assigned and each material group must contain a real timing path or
the run fails immediately.  A compile-before dry-run returned 0 and confirmed
all six material groups.  Final acceptance is setup WNS >= 0, TNS = 0, and
zero setup violations at 1.5 GHz in every material group.

Launch metadata and remote paths are recorded in `DC_LAUNCH.json`.  All four
jobs reached the Tcl flow with `dc_exit.status=RUNNING` and no early
`Error:`/`Fatal:`; recurring monitoring is intentionally left to the next-day
result collection.

## Completed DC results (2026-08-01)

All four `dc_shell` processes reached final DDC, mapped-Verilog and SDF
writeout, but none closes setup timing at 1.5 GHz.  The only failing setup
group is `PLANNER`:

| Configuration | Total cell area | WNS | TNS | Setup violations |
|---|---:|---:|---:|---:|
| 40+6 boundary-typical | 107,239.232109 | -0.2115 ns | -84.86 ns | 724 |
| 40+6 ZeroWireload boundary | 105,260.706639 | -0.2192 ns | -90.06 ns | 662 |
| 63+6 boundary-typical | 115,310.305451 | -0.2162 ns | -87.28 ns | 692 |
| 40+8 boundary-typical | 112,413.849130 | -0.2159 ns | -88.07 ns | 608 |

The boundary envelope adds 1,978.525470 area (+1.8796%) to the same 40+6
RTL, entirely in combinational timing cells.  Expanding 40+6 to 63+6 adds
8,071.073342 (+7.5262%), while two additional PayloadSlots add 5,174.617021
(+4.8253%).  Neither capacity change improves WNS, so 40+6 remains the
capacity/area choice.

The worst path is not in Engine storage or arbitration.  It starts at
`elementsPerLane`, `cursorIndex` or compiled `boxDims`, crosses the
CursorExpand mixed-radix carry/component network, and ends in
`cursorStage_ram`.  The first boundary-typical path has 0.8833 ns data arrival
against 0.6718 ns required time.  A capacity sweep therefore cannot close the
approximately 0.21 ns deficit; CursorExpand needs another architectural
boundary or a different next-window formulation.  Before adding cursor
contexts, the least invasive timing experiment is to replace `smallDivRem`'s
eight-deep descending `Mux` quotient selector with a thermometer comparison
vector plus balanced popcount, and to form `quotient * divisor` from balanced
binary shift/add terms.  The current RTL calls the comparisons parallel, but
the generated quotient selection is still a serial priority-mux cone repeated
through the rank recurrence.  If that does not remove roughly 0.21 ns, the
rank recurrence itself must be split using an elastic, single-command cursor
pipeline; buffering or increasing LineContext/PayloadSlot capacity cannot fix
this path.

The launcher status is `2` even though mapped outputs exist.  Final report
refresh attempted to remove path groups using ephemeral collection handles
such as `_sel2383`; it then looked for a `TmaV2DmaCore` ref after that wrapper
had been flattened.  These produced `UID-250`/`CMD-013`.  No unresolved
reference, black box, latch or combinational-loop error was found.  Mapped
area is valid, and timing is useful diagnostically, but this is not a release
pass because setup WNS is negative and the final path-group audit failed.

Machine-readable results and exact deltas are frozen in
`DC_RESULTS_20260801.json`.
