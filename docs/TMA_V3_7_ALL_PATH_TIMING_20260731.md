# Ventus TMA V3.7：全路径时序优化（最终 1.5 GHz）

## 当前结论

V3.7 已完成 RTL、Backend、GVM 周期和结构预检，最终 N12 DC 尚未返回。
它不是只修 V3.6 的一条 WNS 路径，而是同时切断 Planner、Engine、Binder、
Descriptor 和 DmaCore 队列五类大规模负裕量路径。外部 ISA、TensorMap、L2、
shared memory、TLB 和调度器均未修改。

V3.6 在 N12 TT 1.0 V 85 °C、1.6 GHz、boundary-typical 条件下为：

| 指标 | V3.6 |
|---|---:|
| WNS | -0.2801 ns |
| TNS | -897.41 ns |
| setup violations | 15,059 |
| total cell area | 88,420.53 |

负裕量终点中 Engine 占 9,528 条（64.3%），DmaCore 本地队列占 2,599 条
（17.5%），Planner 占 2,294 条（15.5%）。因此只优化最差的一条
Planner 路径不会闭合 TNS。

V3.6 的另一个流程问题是：路径组只在 retime 前按易失的层次名创建；retime
和 `change_names` 后，最终分组报告实际显示 `No paths`。这不会导致功能错误，
但会让 incremental compile 不再按模块均衡优化，也让“逐组通过”的验收失去
意义。V3.7 的 DC 流程会在三个边界重建路径组：初始 compile 前、retime 后且
incremental 前、最终改名后。分组从模块 `ref_name` 推导，并要求所有实质 TMA
组至少包含一条真实 timing path，否则立即终止。

## RTL 改动

### Planner

- 8 lane × rank 的串行递推改成并行前缀和；小常数乘法和除余由比较、移位、
  加法表达，不生成运行时 divider/remainder。
- 在 CursorExpand 后增加弹性 CoordinateMap 边界；cursor 只在 token 接受时
  前进，保持背压下不重复、不遗漏。
- 修正小乘数扩展宽度，负坐标显式使用符号/二补码路径。
- 256-window 连续布局仍保持填充后每周期一个 window。

### Binder 与 Descriptor Compiler

- Binder 的 33×33 位长乘法拆成 16-bit limb 乘积和 68-bit 平衡加法树，结果
  在下一阶段组装，不增加命令 II。
- Descriptor 的 64×33 位乘法拆成 12 个 16×16 位部分积和平衡 96-bit 树；
  工作状态只保存结果低 64 位及高位溢出标志，不复制完整 descriptor。
- Compiler 仍是单路，4 项 compiled store 和跨命令复用语义保持不变。

### WindowEngine 与 DmaCore 边界

- cache line tag 使用 `25-bit tag + low-7-bit carry` 生成，删除完整 32-bit
  `address + 15` 再右移的重复网络。
- shared response 的协议 offset 最多只作用一个 lane，因此由八套 128-bit
  动态移位器改为一套选中 lane 移位器，再做 lane 选择。
- S2G 在发 shared read 时预计算首 fragment 的紧凑 route seed；shared
  response 的 1024-bit 数据先写入现有 PayloadSlot，首 line 可以同周期分配，
  后续 fragment 由常规 decomposer 处理。
- cache 和 shared 各自只有一个 engine-local、深度 1 的最终 egress queue。
  PayloadSlot/LineContext 在请求捕获时释放，响应所有权由 ack/shared tag 接管；
  DmaCore 外侧原有两份 queue 被删除，因此没有串接两级出口队列。

尝试过但被周期证据否决的方案包括：cache candidate 额外寄存一级（32 KiB
S2G 回退 172 cycle）、在 shared data 返回前完整预分解 line（32 KiB S2G
由 888 回退到 997 cycle）、route 再寄存一级（容量回归）。这些改动没有进入
V3.7。

## 周期结果

固定结果位于
[`features_ventus_v3_7_timing_allpaths_capacity`](../benchmarks/tma-cycle-compare/results/features_ventus_v3_7_timing_allpaths_capacity/attempt_01.csv)。

| bytes | G2S | S2G | roundtrip |
|---:|---:|---:|---:|
| 128 B | 143 | 106 | 232 |
| 256 B | 145 | 108 | 236 |
| 512 B | 149 | 112 | 244 |
| 1 KiB | 157 | 138 | 278 |
| 2 KiB | 180 | 160 | 323 |
| 4 KiB | 217 | 210 | 410 |
| 8 KiB | 294 | 320 | 597 |
| 16 KiB | 426 | 502 | 911 |
| 32 KiB | 690 | 887 | 1560 |

27/27 用例通过。相对 V3.6，每个单方向快 1–2 cycle，roundtrip 快 2–3
cycle；没有周期回退。16→32 KiB 的边际速率仍为 G2S 2.063、S2G 3.008、
roundtrip 5.070 cycle/128B，所以本轮时序切分没有伪造数据面吞吐改善；稳态
性能仍由既有 L2/request/ack 模型主导。

## 面积结构预检

PMU-off 生成 RTL 的 SHA-256 为
`8c8ba2136a408f655aa471ff6552802808e620d2d12b81a06c29c2bb7bbc0368`。
结构审计见
[`rtl_structure_summary.json`](../benchmarks/tma-area-dc-v3.7/rtl_structure_summary.json)。

| 项目 | V3.7 |
|---|---:|
| standalone TMA 状态 | 34,593 bit |
| WindowEngine hierarchy | 21,832 bit |
| WindowEngine data core | 19,216 bit |
| relocated cache/shared egress | 1,196 / 1,420 bit |
| LineContext | 40 × 220 bit |
| PayloadSlot | 6 × 1,465 bit |
| 1024-bit 驻留 payload | 6 份 |
| Planner elastic queues | 2,250 bit |
| Binder / Ingress / Descriptor Service | 735 / 1,526 / 3,574 bit |

`WindowEngine data core` 仅用于判断核心状态是否膨胀；两个 egress queue 仍
完整计入 `standalone TMA`。所有结构门槛通过，且不存在运行时 `/`、`%`、
PMU 状态、旧 fast path、宽 response broadcast 或超过 1024 bit 的动态移位
结果。

## 最终 DC 验收

最终目标按要求改为 1.5 GHz，并行启动四个独立项目：

- 40+6 boundary-typical：发布候选；
- 40+6 ZeroWireload：量化外部 drive/load envelope；
- 63+6 boundary-typical：最大合法 LineContext 容量；
- 40+8 boundary-typical：量化两个额外宽 PayloadSlot。

64+6 没有直接综合，因为 6-bit source ID 的 63 固定保留给 descriptor
refill，RTL 也明确 `require(requestEntries <= 63)`；用 64 项会发生 source
冲突，不是有效面积点。四个项目均为 TSMC N12、
`tcbn12ffcllbwp16p90cpdtt1v85c`、TT 1.0 V 85 °C、1.5 GHz、PMU off、
每项目最多 12 CPU、无 timeout。详细远端路径和哈希见
[`DC_LAUNCH.json`](../benchmarks/tma-area-dc-v3.7/DC_LAUNCH.json)。

compile 前 dry-run 已返回 0，并确认 Ingress、Planner、Binder、Descriptor、
Engine、DmaCore control 六个互斥 endpoint 组均有真实路径。四个正式任务均
进入 Tcl、状态为 RUNNING，且初始日志无 `Error:`/`Fatal:`。最终冻结要求为
1.5 GHz 下 setup WNS >= 0、TNS = 0、零 setup violation；DC 返回前 V3.7
只记为 preflight，不覆盖历史 V3.0–V3.6 结果。
