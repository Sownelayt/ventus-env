# Ventus TMA V3.10 rank 2+2+1 full-module DC

V3.10 replaces the V3.9 Planner rank split `2+3` with three elastic systolic
stages: `rank 0..1 | rank 2..3 | rank 4`. Each stage owns only its corresponding
`cursorIndex` slice, so the added register is a real recurrence cut rather than
a multi-cycle feedback loop. Adjacent windows may occupy all three carry stages
at once and the steady-state contract remains `II=1`.

The data backend remains V3.4's `40 LineContext + 6 PayloadSlot` organization.
No Engine, L2, shared-memory, ISA or external-interface behavior is changed.

## Verification

All three Chisel suites passed:

- `TmaV2Frontend_test`: 8/8, including rank 1–5 C-model comparison, random
  backpressure and the strict 32 KiB contiguous test. The latter still emits
  exactly 256 windows on 256 consecutive cycles after pipeline fill.
- `TmaV2Backend_test`: 17/17.
- `TmaV2DmaCore_test`: 19/19, including Bulk/Tensor G2S/S2G, interleave16,
  cross-line transfers, prefetch/invalidate, reduce and one-lookahead ordering.

The additional MidCarry queue is 497 bits. PMU-off full-TMA sequential state is
35,967 bits: +497 bits versus V3.9 and +760 bits versus V3.8. All Planner
elastic queues together are 4,203 bits. The generated RTL retains exactly six
1024-bit resident payload copies and contains no runtime divider/remainder,
PMU state, old fast path or wide response-broadcast structure. See
`rtl_structure_summary.json`.

The extra carry stage adds one Tensor command startup cycle versus V3.9 (two
versus V3.8), but does not change steady-state window throughput. Bulk commands
do not use the Tensor rank planner and therefore do not pay this latency.

## Full standalone TMA DC

The independent VM project is:

```text
/home/liyb/tma-dc/20260801_27_tma_v3_10_rank2p2p1_full_p96_tt1v85c_1500
```

The synthesis top is the complete standalone `tma`, not a Planner-only module.
It uses:

- `tcbn12ffcllbwp6t16p96cpdtt1v85c` (TSMC N12 6T P96);
- TT 1.0 V, 85 C, 1.5 GHz;
- `BUFFD2BWP6T16P96CPD` boundary driver and the matched boundary-typical
  input/output envelope;
- internal `ZeroWireload`, because no routed interconnect model is available;
- `compile_ultra -retime -timing_high_effort_script`, followed by incremental
  compile;
- PMU-off RTL, 12 CPUs and `DC_TIMEOUT_SECONDS=0`.

The precompile dry-run completed with status 0, no DC Error/Fatal and nonempty
INGRESS, PLANNER, BINDER, DESCRIPTOR, ENGINE and DMA_CORE_CTRL path groups.
The formal run was then launched in tmux session
`v310_rank2p2p1_full_p96_1500`; its initial state was `RUNNING`, the patched
`dc_shell` process was alive, PRESTO analysis had started, and the older V3.8
P96 job remained independently `RUNNING`.

Exact launch metadata and hashes are in `DC_LAUNCH.json`.

## Completed result

Both the frozen V3.8 P96 reference and V3.10 completed with exit status zero
and no DC Error/Fatal. The results below are directly comparable: both use the
same complete `tma` top, PMU-off configuration, P96 library, boundary model,
1.5 GHz constraint and compile script.

| Metric | V3.8 P96 | V3.10 2+2+1 P96 | Change |
|---|---:|---:|---:|
| Setup WNS | -0.2910 ns | -0.0147 ns | deficit reduced 94.95% |
| Setup TNS | -125.10 ns | -1.79 ns | reduced 98.57% |
| Setup violating paths | 988 | 233 | reduced 76.42% |
| Total cell area | 79,899.33 | 85,624.31 | +7.17% |
| Combinational area | 57,774.77 | 62,954.46 | +8.97% |
| Sequential area | 22,124.56 | 22,669.85 | +2.46% |
| Planner hierarchical area | 21,327.48 | 26,607.85 | +24.76% |
| Engine hierarchical area | 45,686.18 | 46,200.80 | +1.13% |

The result is a large timing improvement, but it does **not** meet the strict
1.5 GHz acceptance rule: the only remaining negative-setup group is PLANNER.
ENGINE, BINDER, DESCRIPTOR, DMA_CORE_CTRL and INGRESS have zero setup TNS.
Using WNS only as a first-order estimate, the present mapped result corresponds
to about 1.468 GHz under these exact margins, not a signed-off 1.5 GHz block.
The report also contains 2,394 min-delay/hold violations and 241 max-fanout
violations. The hold population is essentially unchanged from V3.8 and, under
the current ideal-clock/ZeroWireload DC model, remains a place/CTS/routing
closure item rather than evidence for another functional pipeline stage.

The V3.8 worst path ran from `elementsPerLane` through the complete rank chain
to `cursorIndex[4]` and missed by 291 ps. V3.10 moves the worst endpoint to
`cursorIndex[1]` and misses by only 14.7 ps. The path is now the remaining
rank-0/rank-1 LowCarry chain: lane offset generation, rank-0 `smallDivRem`, its
quotient feeding rank 1, and the second `smallDivRem`.

All 100 detailed worst paths start at an `elementsPerLane` bit. Across all 233
violating endpoints, the same command-stable value also reaches LowCarry,
ComponentMap/CursorStage and later mapping registers. This is important:
`elementsPerLane` is stored as a 9-bit numeric value even though its legal
values are only the powers of two 2, 4, 8, 16 and 32, while the Planner already
stores the equivalent 3-bit `laneElementShift`. The residual failure is
therefore not evidence that the new rank-2/rank-3 or rank-4 cuts are too long;
it is dominated by a redundant wide representation and its fanout, plus the
two serial low-rank carry operations.

Area growth is likewise concentrated in Planner. It contributes 5,280.36 of
the total 5,724.98 area-unit increase. Only 563.02 of Planner's increase is
sequential; 4,717.34 is combinational mapping. Thus the 497-bit MidCarry queue
is not the primary area cost. Most growth is DC's larger mapped logic and
buffer/upsize network around the aggressively timed Planner. Engine changes by
only +1.13%, Descriptor by -1.15%, and Binder by +0.43%.

The next low-risk step should be to remove the live 9-bit `elementsPerLane`
state and use `laneElementShift` directly:

1. Generate each lane increment as `lane << laneElementShift`.
2. Reconstruct the local clipping limit as `1 << laneElementShift`, or carry a
   compact predecoded value only at the consuming stage.
3. Re-run Frontend and a Planner-only P96 DC first.

This attacks the common startpoint of every reported worst path and can reduce
area as well as delay. If it still leaves negative slack, the deterministic
fallback is to split LowCarry from `2` ranks into `1+1`, giving a
`1+1+2+1` recurrence. That costs one additional Tensor startup cycle and a
small elastic token, but should not be the first response to a 14.7 ps miss.

Exact metric deltas are in `results/v3_8_v3_10_full_compare.csv`; the copied DC
reports are retained under `results/v3_8_full_p96/` and
`results/v3_10_rank2p2p1_full_p96/`.
