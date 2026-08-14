# Ventus TMA V3.8 cursor pipeline

V3.8 replaces the remaining combined Tensor WindowPlanner cursor cone with a
real depth-one elastic pipeline boundary. It preserves V3.7's single active
command, 40 `LineContext`, 6 `PayloadSlot`, 4-entry compiled descriptor store,
L2/shared interfaces and `II=1` steady-state contract.

## Architecture change

The old CursorExpand cycle updated the mixed-radix index cursor and formed the
64-bit address-component recurrence in one combinational cone. V3.8 separates
those recurrences:

1. `CursorCarry` forms nine lane snapshots, rank carries, next cursor row and
   logical window index. Its recurrence advances only on `carryStage.enq.fire`.
2. A 614-bit, depth-one, `pipe=true/flow=false` queue is the physical timing
   boundary.
3. `ComponentMap` consumes that token, forms component deltas and advances its
   independent component cursor only on `cursorStage.enq.fire`.

Both sides may fire every cycle. Backpressure holds the token and the
corresponding recurrence stable, so this is an elastic `II=1` pipeline rather
than a fixed-delay FSM. The bounded divide/remainder helper also uses parallel
thermometer comparisons, balanced popcount and balanced shift/add product;
the generated RTL contains no runtime `/`, `%`, divider or remainder unit.

## RTL and functional evidence

The PMU-off standalone RTL is
`gpgpu/generated-tma-v2-v3.8-work/tma.sv`, SHA-256
`b97fd25aba79773ef35940351dfaca84d5faf842de15984cc07ba00c31decc1c`.
The source SHA-256 is
`e0bbb2192063a7db9f5a29d944e1cd4075defd8a44228f03503ccd4b09818fdc`.

Three Chisel suites passed:

| Suite | Result |
|---|---:|
| Frontend | 8/8 |
| Backend | 17/17 |
| DmaCore | 19/19 |

Frontend includes random backpressure, rank 1–5 C-model comparison and the
strict 32 KiB test: after fill, 256 windows are accepted on 256 consecutive
cycles, so issue span remains 255 cycles. The local GVM capacity matrix is
27/27, and the safe non-Reduce matrix is 334/334 across Bulk, capacity,
geometry, dtype, swizzle and interleave16.

## Cycle cost

The measured cost is exact and independent of transfer size:

| Command sequence | V3.7 -> V3.8 |
|---|---:|
| Bulk G2S/S2G | +0 cycle |
| one Tensor G2S or S2G | +1 cycle |
| Tensor G2S + S2G roundtrip | +2 cycles |

The 27-point table is in
`../tma-cycle-compare/results/v3_7_v3_8_cursor_pipeline_capacity.csv`.
For 32 KiB, G2S/S2G/roundtrip change from `690/887/1560` to
`691/888/1562`. There is no capacity-dependent slope change: the extra stage
adds latency once per Tensor command and does not change throughput.

## Structural cost

The structural audit in `rtl_structure_summary.json` passes every gate:

| Item | V3.7 | V3.8 | Delta |
|---|---:|---:|---:|
| standalone sequential state | 34,593 bit | 35,207 bit | +614 (+1.775%) |
| Planner elastic queues | 2,250 bit | 2,864 bit | +614 |
| WindowEngine hierarchy | 21,832 bit | 21,832 bit | 0 |
| LineContext | 40 x 220 bit | 40 x 220 bit | 0 |
| PayloadSlot | 6 x 1,465 bit | 6 x 1,465 bit | 0 |
| resident 1024-bit payloads | 6 | 6 | 0 |

There is still one selected payload read and one bidirectional line permuter;
no wide response broadcast, 40-way payload mux, PMU state, old fast path or
duplicate descriptor result was introduced. Mapped cell-area change remains
pending DC and must not be inferred directly from these bit counts.

## DC

One independent formal project was launched:

```text
/home/liyb/tma-dc/20260801_23_tma_v3_8_cursor_pipeline_40x6_boundary_n12_tt1v85c_1500
tmux: v38_cursor40p6_boundary_1500
```

It uses N12 `tcbn12ffcllbwp16p90cpdtt1v85c`, TT 1.0 V 85 C, 1.5 GHz,
boundary-typical I/O constraints, at most 12 CPUs, PMU off and no timeout. A
compile-before dry-run passed before launch. The V3.7 final-report defect was
fixed in this copied project: path groups are removed by persistent name, and
top-level DmaCore endpoints no longer depend on a wrapper `ref_name` that can
disappear after flattening. Launch metadata is in `DC_LAUNCH.json`.

### Matched 6T P96 rerun

The frozen V3.8 RTL was also launched as a library-only matched comparison:

```text
/home/liyb/tma-dc/20260801_24_tma_v3_8_p96_matched_40x6_boundary_n12_tt1v85c_1500
tmux: v38_p96_matched_40p6_1500
```

This run keeps the RTL SHA-256, 1.5 GHz clock, TT 1.0 V 85 C corner,
boundary delays/load, `ZeroWireload`, compile options and 12-CPU limit
unchanged. It changes only the standard-cell architecture to
`tcbn12ffcllbwp6t16p96cpdtt1v85c`, the matching NLDM database and the boundary
driver to `BUFFD2BWP6T16P96CPD`. A separate analyze/elaborate/link dry-run
passed with exit status 0 before the no-timeout formal run was launched.

The original P90 run remains the historical microarchitecture result. P90 and
P96 absolute area/timing numbers must not be mixed across RTL versions; the
new run is compared directly against frozen V3.8 P90 to isolate the library
effect. Full launch metadata is in `P96_DC_LAUNCH.json`.
