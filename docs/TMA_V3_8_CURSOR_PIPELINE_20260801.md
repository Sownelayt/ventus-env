# Ventus TMA V3.8 Cursor Pipeline

## Decision

The remaining WindowPlanner timing problem is handled with a real elastic
pipeline, not another combinational-only rewrite. This is the more reliable
architecture: it turns the V3.7 critical cone into two separately registered
recurrences while retaining one window per cycle after fill.

The observable price is one fixed cycle per Tensor data command. It is not one
cycle per window and therefore does not grow with bytes transferred.

## Dataflow and state transition

```text
ActiveCommand (stable)
       |
       v
CursorCarry -- fire --> 614-bit elastic register --> ComponentMap
    |                                                |
    +-- advances index cursor                        +-- advances component cursor
                                                       on next-stage fire
       v
CoordinateMap -> GlobalMap -> SharedMap/SlotEncode -> WindowEngine 40+6
```

`CursorCarry` produces lane 0–7 plus lane 8 as the next-window cursor. The
token stores five 9-bit indices and five 4-bit carries for each of nine lane
snapshots, a 3-bit row and a 25-bit logical-window index. It does not copy the
compiled descriptor, direction, codec, barrier, payload or command context.
All of those remain in the unique immutable `ActiveCommand`.

This split preserves the important invariants:

- exactly one active data command and one stable command snapshot;
- cursor state advances only when ownership of a token moves forward;
- arbitrary downstream backpressure cannot repeat or skip a window;
- both recurrence stages retain `II=1`;
- the 40+6 backend and external L2/shared interfaces are unchanged.

## Measured performance

The V3.8 capacity run is
[`features_ventus_v3_8_cursor_pipeline_capacity`](../benchmarks/tma-cycle-compare/results/features_ventus_v3_8_cursor_pipeline_capacity/attempt_01.csv).
The exact V3.7 comparison is
[`v3_7_v3_8_cursor_pipeline_capacity.csv`](../benchmarks/tma-cycle-compare/results/v3_7_v3_8_cursor_pipeline_capacity.csv).

| Bytes | V3.7 G2S/S2G/RT | V3.8 G2S/S2G/RT | Delta |
|---:|---:|---:|---:|
| 128 | 143/106/232 | 144/107/234 | +1/+1/+2 |
| 1 KiB | 157/138/278 | 158/139/280 | +1/+1/+2 |
| 4 KiB | 217/210/410 | 218/211/412 | +1/+1/+2 |
| 8 KiB | 294/320/597 | 295/321/599 | +1/+1/+2 |
| 16 KiB | 426/502/911 | 427/503/913 | +1/+1/+2 |
| 32 KiB | 690/887/1560 | 691/888/1562 | +1/+1/+2 |

Across the complete safe non-Reduce matrix:

- 34 Bulk cases: all `+0` cycles;
- 208 single-direction Tensor cases: all `+1` cycle;
- 92 Tensor roundtrip cases: all `+2` cycles;
- 334/334 results are functionally correct.

This exact distribution is strong evidence that only pipeline fill latency was
added. If throughput had fallen, the delta would grow with window count; it
does not. The 32 KiB RTL test independently observes 256 consecutive output
fires with an issue span of 255 cycles.

## Area and timing expectation

The only new resident state is the 614-bit carry token. Standalone PMU-off TMA
state rises from 34,593 to 35,207 bits (+1.775%); WindowEngine state, the six
1024-bit payload slots and all 40 LineContexts are unchanged. This is a small,
predictable sequential-area cost.

V3.7 missed 1.5 GHz by about 0.21 ns exclusively in Planner paths that crossed
mixed-radix carry and component formation. V3.8 physically cuts that exact
cross-recurrence path. It should materially improve WNS, but only mapped DC can
prove whether each half is below the 0.6667 ns target and whether another
Planner family becomes limiting. No positive WNS is claimed before that run
finishes.

The formal DC project, constraints and initial health are recorded in
[`DC_LAUNCH.json`](../benchmarks/tma-area-dc-v3.8/DC_LAUNCH.json). It is an
independent 40+6 boundary-typical project at N12 TT 1.0 V 85 C, 1.5 GHz, with
no timeout.
