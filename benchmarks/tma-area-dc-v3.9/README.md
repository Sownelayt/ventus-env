# Ventus TMA V3.9 rank 2+3 Planner experiment

V3.9 splits the five-rank mixed-radix cursor recurrence in
`TmaV34WindowPlanner` into two elastic systolic stages: ranks 0–1 and ranks
2–4. Each stage owns its corresponding cursor state, so adjacent windows can
remain in flight and the steady-state contract stays `II=1`.

The first evaluation is deliberately module-only. Frozen V3.8 and experimental
V3.9 Planner RTL are synthesized with the same N12 6T P96 TT 1.0 V 85 C
library, 1.5 GHz clock, boundary constraints, compile options and six-CPU
limit. This separates the rank-pipeline timing/area delta from Descriptor,
WindowEngine and top-level optimization effects.

## Functional and structural evidence

The complete `TmaV2Frontend_test` suite passes 8/8, including rank 1–5
comparison against the independent model, random backpressure and the strict
32 KiB test. After pipeline fill, all 256 windows are still accepted on 256
consecutive cycles; the 2+3 cut adds one command-start cycle without changing
steady-state throughput.

The new low-rank elastic token is 263 bits. PMU-off standalone state changes
from 35,207 bits in frozen V3.8 to 35,470 bits in V3.9, an increase of 0.747%.
WindowEngine, payload storage, LineContext, Descriptor and Binder state are
unchanged. The generated RTL contains no runtime divide/remainder, PMU state,
old fast path, response broadcast or duplicated payload array. The complete
audit is in `rtl_structure_summary.json`.

## Planner-only DC

Two matched module-level projects were launched in parallel:

```text
/home/liyb/tma-dc/20260801_25_tma_v3_8_planner_module_p96_tt1v85c_1500
/home/liyb/tma-dc/20260801_26_tma_v3_9_rank2p3_planner_module_p96_tt1v85c_1500
```

Both elaborate only `TmaV34WindowPlanner` and use the same P96 TT 1.0 V 85 C
library, 1.5 GHz boundary constraints, compile script, six-CPU limit and no
timeout. Both precompile dry-runs passed with exit status 0. Launch metadata is
in `DC_LAUNCH.json`.

Both formal runs completed with exit status 0 and no DC `Error`/`Fatal`.

| Metric | V3.8 P96 | V3.9 2+3 P96 | Change |
|---|---:|---:|---:|
| Setup WNS | -0.3021 ns | -0.0410 ns | +0.2611 ns |
| Setup TNS | -127.18 ns | -9.66 ns | 92.40% less |
| Violating paths | 838 | 391 | 53.34% less |
| Total cell area | 21,102.10 | 26,004.79 | +23.23% |
| Combinational area | 18,607.10 | 23,284.63 | +25.14% |
| Sequential area | 2,494.99 | 2,720.16 | +9.03% |

The 2+3 cut removes 86.43% of the WNS deficit but does not close 1.5 GHz.
The critical path moves from the complete rank chain
`elementsPerLane -> cursorIndex[4]` to the remaining high-rank chain
`boxDims[2] -> cursorIndex[4]`. This directly motivates a second cut after
rank 3, producing the previously proposed 2+2+1 organization.

Only 225.17 area units of the 4,902.69 increase are sequential. The other
95.4% is combinational mapping cost: DC adds 23,471 combinational cells and
4,077 buffers/inverters while trying to close the aggressive 1.5 GHz target.
The 2+3 experiment is therefore a useful timing proof but is not a release
candidate. A 2+2+1 trial should be evaluated module-first before any full-TMA
DC or version freeze. Exact values and report paths are in
`results/planner_v3_8_v3_9_compare.csv`.
