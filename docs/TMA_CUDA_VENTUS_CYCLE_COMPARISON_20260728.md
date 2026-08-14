# CUDA H100/B200 与 Ventus 非 Reduce TMA 周期对比

> 当前评估版本：**Ventus V3.7 preflight**；H100/B200 为固定硬件参考。
> V3.7 的功能、GVM 周期和结构门槛已经通过，正在等待 N12 1.6 GHz DC
> 面积/时序结果，因此尚未冻结，`CURRENT_VERSION` 仍为 V3.0。V3.0 的
> 原始周期、聚合表、运行 metadata、源码快照和 SHA-256 保留在
> [`../benchmarks/tma-cycle-compare/baselines/V3.0/`](../benchmarks/tma-cycle-compare/baselines/V3.0/)。
> 本文顶部使用 V3.7 当前结果；后半部分保留 V3.3 和 2026-07-28 的 V3.0
> 完整覆盖记录，不覆盖历史基线。
>
> CUDA 测试日期：2026-07-28；V3.7 更新：2026-07-31  
> 状态：**V3.7 PRE-FLIGHT COMPLETE / DC PENDING**  
> 正式范围：H100/B200 同一完整非 Reduce 矩阵、Ventus GVM 同参数交集  
> 排除：Reduce；会触发 XID 的 UINT8 interleave16 与未闭环的 interleave32

> 架构解释与建模边界见：
> [Ventus TMA 与 CUDA TMA 微架构差异及 L2/DRAM 建模失真分析](TMA_RTL_CUDA_ARCHITECTURE_AND_L2_MODEL_ANALYSIS_20260729.md)；
> G2S credit 根因、Hopper 公开结构证据和实际 RTL 深度扫描见
> [Hopper TMA 高吞吐原因与 Ventus V3.3 G2S 瓶颈](TMA_V3_3_HOPPER_G2S_ROOT_CAUSE_AND_OPTIMIZATION_20260730.md)。

## V3.7 当前结论

V3.7 保持 40 个轻量 LineContext、6 个 PayloadSlot、4 项 Descriptor Store
和严格单 active 数据命令。本轮只重构 TMA 内部算术和出口时序边界；L2、
shared memory、TLB、调度器和外部协议不变。正式 capacity 结果为：

| bytes | path | H100 | B200 | V3.7 |
|---:|---|---:|---:|---:|
| 128 B | G2S | 460 | 461 | 143 |
| 128 B | S2G | 62 | 93 | 106 |
| 128 B | roundtrip | 553 | 568 | 232 |
| 1 KiB | G2S | 460 | 461 | 157 |
| 1 KiB | S2G | 90 | 121 | 138 |
| 1 KiB | roundtrip | 581 | 596 | 278 |
| 4 KiB | G2S | 460 | 461 | 217 |
| 4 KiB | S2G | 186 | 217 | 210 |
| 4 KiB | roundtrip | 678.5 | 692 | 410 |
| 8 KiB | G2S | 608 | 609 | 294 |
| 8 KiB | S2G | 314 | 345 | 320 |
| 8 KiB | roundtrip | 805 | 820 | 597 |
| 16 KiB | G2S | 608 | 609 | 426 |
| 16 KiB | S2G | 570 | 601 | 502 |
| 16 KiB | roundtrip | 1209 | 1224 | 911 |
| 32 KiB | G2S | 756 | 759 | 690 |
| 32 KiB | S2G | 1082 | 1113 | 887 |
| 32 KiB | roundtrip | 1869 | 1884 | 1560 |

用 16→32 KiB 增量计算稳态边际，V3.7 为 G2S `2.063`、S2G
`3.008`、roundtrip `5.070` cycle/128B；H100/B200 的 G2S 为
`1.156/1.172`。因此当前绝对周期较小仍主要来自 RTL L2 的低固定延迟，
真正可外推的 G2S line 吞吐仍比 H100 慢约 `1.78×`。S2G 表面快于 CUDA
则是 write/ack 模型偏乐观，不能作为流片性能结论。

相对 V3.3，32 KiB G2S 从 765 降到 690（改善 9.8%），roundtrip 从
1626 降到 1560；S2G 从 878 增到 887，而 128 B 三方向从 131/96/210
增到 143/106/232。这是 40+6 面积重构和严格出口所有权带来的固定开销，
尚未被大容量吞吐收益完全抵消。相对直接的 V3.6 时序基线，V3.7 的 27 个
capacity case 全部快 1–3 cycle，没有新回退。

功能证据已由 V3.7 GVM 重新运行：Bulk、capacity、geometry、dtype、
swizzle 和 interleave16 六个安全非 Reduce 分片共 334/334 通过；Frontend、
Backend、DmaCore Chisel 回归分别为 8/8、17/17、19/19。详细时序改动和结构
审计见 [V3.7 全路径时序报告](TMA_V3_7_ALL_PATH_TIMING_20260731.md)。

## V3.3 当前结论

CUDA 使用 GPU `clock64()`，Ventus 使用 GVM `mcycle`，两边不是同一物理
频率；V/H 只比较周期曲线、固定开销和容量拐点，不代表同频流片速度。

### 结论摘要

1. **单条大容量 G2S 的主要瓶颈已经不是地址计算。**32 KiB 命令产生
   256 个 128B window，Planner 全部 `produced=fire=256`，transform
   stall 和 shared-full 都为 0；但 Planner 因下游资源背压 342 cycle，
   TransferSlot/request table 峰值达到 24/16，分别 full 343/359
   cycle。当前瓶颈是 cache request/response 服务速度。
2. **16→32 KiB 的 G2S 边际成本暴露了与 CUDA 的主要单命令差距。**
   Ventus 为 2.500 cycle/128B，H100 为 1.156，B200 为 1.172；Ventus
   的边际 line 吞吐约慢 2.16×/2.13×。由于 Ventus 固定启动时间远小于
   CUDA 模型，32 KiB 总周期才刚出现交叉：765 对 H100 756、B200 759。
3. **S2G 不是 shared 或 transform 限制，而是 cache write/ack。**
   32 KiB 时 shared tag 只用到 3/8，transform stall 为 0；但 32 个
   write-ack tag 全满，测量 repeat 中 ack-full 373 cycle、request-full
   404 cycle。增加 shared 带宽不会改善这个点。
4. **小容量 S2G 的主要差距是固定开销。**128 B 为 Ventus 96、H100
   62、B200 93 cycle；相对 H100 多 34 cycle，但到 8 KiB 已变成
   300/314/345。当前测量可精确分解为 G2S `103 + active cycles`、
   S2G `68 + active cycles`；S2G 的 H100 曲线则精确为
   `58 + 4 × 128B-line`。
5. **Ventus 大容量 S2G 看起来比 H100/B200 更快，主要反映模型偏乐观。**
   16→32 KiB 的边际成本为 Ventus 3.016、H100/B200 4.000
   cycle/128B。Ventus 的 L2/write-ack 是 RTL 时序模型，不含真实
   HBM/DRAM 行为，不能把 `V/H<1` 解释为实际芯片更快。
6. **Descriptor Compiler、Planner 算术和 transform 已不是当前首要
   瓶颈。**五个 common shape 中 Tensor 相对 Bulk 的固定增量始终是
   G2S +44、S2G +43 cycle；显式 prefetch 或 compiled hit 把冷 demand
   的 213 cycle 降至 189，节省 24 cycle。32 KiB 数据阶段则有
   662/810 active cycles，descriptor 优化无法改变其稳态斜率。
7. **与 CUDA 的另一项主要差距是跨数据命令重叠。**H100 同 CTA、
   同 TensorMap 的 8 条 4 KiB G2S，serial/batched 为 3547/991 cycle，
   batched 快 3.579×。V3.3 明确保持一个 active data command；
   Lookahead 只提前 descriptor/Binder，不启动第二条数据搬运。这是为
   面积选择的设计边界，不是当前单命令 pipeline 的 bug。

### 同参数容量周期

Tensor Map 均为 rank 2、UINT8、contiguous、坐标 0、unit stride，
interleave/swizzle/promotion NONE。CUDA 为固定实机 median，Ventus 为
V3.3 GVM 周期：

| bytes | path | H100 | B200 | V3.3 | V/H | V/B |
|---:|---|---:|---:|---:|---:|---:|
| 128 | G2S | 460 | 461 | 131 | 0.285× | 0.284× |
| 128 | S2G | 62 | 93 | 96 | 1.548× | 1.032× |
| 128 | roundtrip | 553 | 568 | 210 | 0.380× | 0.370× |
| 1 KiB | G2S | 460 | 461 | 145 | 0.315× | 0.315× |
| 1 KiB | S2G | 90 | 121 | 128 | 1.422× | 1.058× |
| 1 KiB | roundtrip | 581 | 596 | 256 | 0.441× | 0.430× |
| 4 KiB | G2S | 460 | 461 | 206 | 0.448× | 0.447× |
| 4 KiB | S2G | 186 | 217 | 200 | 1.075× | 0.922× |
| 4 KiB | roundtrip | 678.5 | 692 | 389 | 0.573× | 0.562× |
| 8 KiB | G2S | 608 | 609 | 285 | 0.469× | 0.468× |
| 8 KiB | S2G | 314 | 345 | 300 | 0.955× | 0.870× |
| 8 KiB | roundtrip | 805 | 820 | 568 | 0.706× | 0.693× |
| 16 KiB | G2S | 608 | 609 | 445 | 0.732× | 0.731× |
| 16 KiB | S2G | 570 | 601 | 492 | 0.863× | 0.819× |
| 16 KiB | roundtrip | 1209 | 1224 | 920 | 0.761× | 0.752× |
| 32 KiB | G2S | 756 | 759 | 765 | 1.012× | 1.008× |
| 32 KiB | S2G | 1082 | 1113 | 878 | 0.811× | 0.789× |
| 32 KiB | roundtrip | 1869 | 1884 | 1626 | 0.870× | 0.863× |

完整 27 项见
[`data/TMA_V3_3_CUDA_CAPACITY_COMPARISON.csv`](data/TMA_V3_3_CUDA_CAPACITY_COMPARISON.csv)。

### 边际吞吐

用 16→32 KiB 的增量消除大部分固定启动周期，并除以新增的 128 个
128B line：

| path | H100 cycle/128B | B200 cycle/128B | V3.3 cycle/128B | 解释 |
|---|---:|---:|---:|---|
| G2S | 1.156 | 1.172 | 2.500 | Ventus 单 cache-request/response 路径吞吐不足 |
| S2G | 4.000 | 4.000 | 3.016 | Ventus write/ack 模型相对实机偏乐观 |
| roundtrip | 5.156 | 5.156 | 5.516 | G2S 劣势抵消了 S2G 模型优势 |

因此“32 KiB G2S 总周期只差 9”并不表示微架构吞吐已经相同。Ventus
依靠更小的建模启动成本抵消了较差的稳态 line slope；继续扩大容量后，
G2S 差距仍会增长。

### PMU 瓶颈归因

下表使用 measured repeat，而不是 testcase 累加值。`full S/R` 分别为
TransferSlot/request-full 周期：

| path | bytes | total | active/fixed | Planner stall | full S/R | max slot/req/shared/ack | ack-full | transform stall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G2S | 4 KiB | 206 | 103/103 | 8 | 9/25 | 24/16/2/0 | 0 | 0 |
| G2S | 16 KiB | 445 | 342/103 | 149 | 151/167 | 24/16/2/0 | 0 | 0 |
| G2S | 32 KiB | 765 | 662/103 | 342 | 343/359 | 24/16/2/0 | 0 | 0 |
| S2G | 4 KiB | 200 | 132/68 | 5 | 5/17 | 24/16/3/16 | 0 | 0 |
| S2G | 16 KiB | 492 | 424/68 | 131 | 132/152 | 24/16/3/32 | 108 | 0 |
| S2G | 32 KiB | 878 | 810/68 | 385 | 391/404 | 24/16/3/32 | 373 | 0 |

32 KiB G2S 的平均 cache response latency 为 37.77 cycle/line，最长连续
window fire 只有 24，恰好对应 slot 容量；S2G 的平均 write-ack latency
为 83.83 cycle/line。完整计数见
[`data/TMA_V3_3_BOTTLENECK_PMU.csv`](data/TMA_V3_3_BOTTLENECK_PMU.csv)
和
[`data/TMA_V3_3_RESOURCE_PEAKS.csv`](data/TMA_V3_3_RESOURCE_PEAKS.csv)。

新增的受控诊断实例化真实 WindowEngine，并把 cache response latency
固定为 38 cycle。在同一 48-entry 引擎上只改变有效在途读 credit：

| effective credit | finite issue cycle/line | asymptotic credit bound | command cycles |
|---:|---:|---:|---:|
| 16 | 2.294 | 2.375 | 631 |
| 24 | 1.549 | 1.583 | 441 |
| 32 | 1.165 | 1.188 | 343 |
| 40 | 1.000 | 1.000 | 301 |
| 48 | 1.000 | 1.000 | 301 |

40/48-credit 点连续 256 次每周期发出一个 L2 request，证明 RTL
request/data pipeline 没有固有的 2-cycle 或 3-cycle issue 限制。
当前发布参数的主要问题是 16 个 request credit 无法覆盖约 38-cycle
返回延迟。测试和机器可读结果见
[`data/TMA_V3_3_G2S_CREDIT_DEPTH_DIAGNOSTIC.csv`](data/TMA_V3_3_G2S_CREDIT_DEPTH_DIAGNOSTIC.csv)。

### 常用 Bulk/Tensor 命令

| shape | path | H100 | B200 | V3.3 |
|---|---|---:|---:|---:|
| 4×4 | Bulk G2S | 507 | 440 | 83 |
| 4×4 | Bulk S2G | 43 | 43 | 53 |
| 4×4 | Tensor G2S | 628 | 577 | 127 |
| 4×4 | Tensor S2G | 92 | 93 | 96 |
| 32×32 | Bulk G2S | 507 | 590 | 158 |
| 32×32 | Bulk S2G | 171 | 171 | 145 |
| 32×32 | Tensor G2S | 628 | 577 | 202 |
| 32×32 | Tensor S2G | 215 | 220 | 188 |

完整五个 shape × 四条路径见
[`data/TMA_V3_3_COMMON_COMMAND_COMPARISON.csv`](data/TMA_V3_3_COMMON_COMMAND_COMPARISON.csv)。
V3.3 的 Tensor−Bulk 恒定为 G2S 44、S2G 43 cycle，说明 TensorMap
前端已经成为固定小项，而不是随 payload 放大的瓶颈。

### 当前主要差距与下一步

按优先级排序：

1. **G2S latency-credit 已确认为第一瓶颈。**37.77-cycle 平均返回延迟
   配 16 个 request entry，渐近上限是 2.36 cycle/line；实际为 2.50。
   40-credit 受控点已严格达到每周期一个 request，因此不需要修改
   Planner 或 L2 才能先取得明显改善。
2. **增加轻量 context，而不是直接增加宽 slot。**把约 32–40 个
   compact line request context 与 4–8 个 1024-bit response payload
   entry 分离。当前每个完整 TransferSlot 为 1,541 bit，直接从 24
   增到 48 会增加 36,984 bit 状态并扩大 mux，不符合面积目标。
3. **S2G ack 回压。**当前 ack table 只有 valid bit，面积很小；可在不改
   L2 协议的前提下做 32→48/64 项参数实验，判断能否减少 373 个
   ack-full cycle。但由于当前 S2G 模型已经比 H100 边际 slope 乐观，
   该项优先级低于 G2S。
4. **小 S2G 固定 68 cycle。**显式 prefetch/compiled hit 已能省 24
   cycle；剩余优化应检查 active 晋升、completion/mbarrier/group
   边界，而不是再展开 Planner。
5. **跨命令并发差距是有意保留。**若未来必须追求 CUDA batched
   throughput，就需要重新允许多个 data command 在途；这会恢复
   context、buffer 和完成状态面积，与 V3.3 的单 context 目标冲突。

目前没有证据支持继续复制 transform、扩大 shared queue 或展开地址
计算；这些路径的 stall 都是 0。24 个 slot 在 4 KiB 起已实际用满，
直接缩到 16 项也必须重新测周期，不能视为无效寄存器删除。

## V3.0 历史完整矩阵记录

以下内容保留 2026-07-28 的 V3.0 完整 feature/uarch/pressure 分析，用于
追溯测试覆盖、CUDA XID 修复和多命令实测。其 Ventus 周期不是当前 V3.3
结果；当前数字以上述 V3.3 表和机器可读文件为准。

## V3.0 历史结论

本轮扩展并对齐了容量、tile、rank、stride、OOB、swizzle、outstanding、
descriptor working-set/prefetch、multi-CTA、compute overlap 和 Ventus
双向争用。所有进入正式周期表的 CUDA/Ventus 样本 correctness error
均为 0。

“全面”的判定限定为本项目原先约定的常用非 Reduce 路径：bulk G2S/S2G、
bulk.tensor G2S/S2G，加上上述 feature/uarch 变量。这个范围内 H100 已
完整，因此 B200 本轮补成同样的 5 档命令基线、183-config feature 和
16-config uarch。它不是整个 TMA ISA 的穷举；cluster multicast、
TensorMap replace、未验证的 interleave32 等仍在范围外。逐项状态见
[`data/TMA_THREE_PLATFORM_COVERAGE_MATRIX_20260728.csv`](data/TMA_THREE_PLATFORM_COVERAGE_MATRIX_20260728.csv)。

最重要的结果如下。

1. **同参数单 CTA 的最差正向 cycle 差出现在 32 KiB G2S。**H100 为
   756 cycle，Ventus 为 1013 cycle，`V−H=+257`、`V/H=1.340×`。
   16 KiB G2S 仍为 608/565 cycle，说明 G2S 的 crossover 位于本矩阵的
   16–32 KiB 区间。
2. **相对倍率最大的点是小包 S2G，而不是大包。**128 B S2G 为
   H100 62、Ventus 88 cycle，即 `1.419×`，但绝对差只有 26 cycle。
   S2G 到 8 KiB 已基本相等（314/315），16/32 KiB 时 Ventus cycle
   分别比 H100 少 47/143。
3. **32 KiB roundtrip 轻微越过 H100。**H100/Ventus 为
   1869/1935 cycle，差 66 cycle；16 KiB 仍为 1209/1071。
4. **H100 最明显的容量压力来自“大 tile × 多 CTA”。**32 KiB TMA
   G2S 从 1 CTA 的 756 cycle 增到 528 CTA 的 4030 cycle，
   mean-CTA latency 增长 `5.331×`。4 KiB 同样压力下只增长
   `1.327×`。
5. **H100 已用严格 serial/batched 对照确认同 CTA 的多条 G2S Tensor
   TMA 可以重叠。**每条都是不同 global/shared tile 的 4 KiB 搬运，
   两种模式只改变 wait 位置。N=1 为 454/454 cycle；N=2/4/8 的
   serial 对 batched 分别为 898/552、1781/640、3547/991 cycle，即
   batched 快 1.627/2.783/3.579 倍。该结果证明这一路径存在多条在途
   TMA 的执行重叠，但不等于公开了硬件队列深度，也不外推到不同
   TensorMap、S2G 或跨 CTA。Ventus N=2/4/8 的旧精确 GVM case 在独立
   5 秒短跑中均超时，V3.1/V3.2 则明确采用单 active data command。
6. **descriptor working-set 在两张 CUDA 卡上都没有单调恶化。**
   H100 中位数落在 703 或 851 cycle，prefetch 开关通常不改变中位数。
   Ventus 1–256 maps 的 pf0/pf1 固定为 236/222 cycle，但 1024 maps
   两个 case 都在 5 秒内未完成。B200 完整矩阵在 64/256 maps 出现
   1131→983 cycle 的 prefetch 改善，其余点多为约 687–690.5 cycle。
7. **Ventus 3 秒短预算争用矩阵暴露了单事务基线看不到的风险。**
   58 个独立 case 中 27 个完成、31 个超时；16 KiB、4 KiB G2S、
   4-WG same-set 和部分双向顺序是主要长尾。这里的 timeout 是
   GVM 墙钟预算分类，不是可与 GPU cycle 并列的性能数字。
8. **interleave16 已用合法同参数正控闭环。**CUDA Driver API 文档示例
   对应的 UINT16 NC/8HWC8、4096 B roundtrip 在 H100 六次为
   678–730 cycle，中位数 681；B200 中位数 693；Ventus 同描述符为
   422 cycle，三边 errors/status 均为 0。原 UINT8 case 仍排除。
9. **B200 的单 CTA feature G2S 与 H100 基本重合，但短 S2G 固定开销
   更高。**128 B S2G 为 93/62 cycle（B/H=1.50×），差距随容量增大，
   到 32 KiB 收敛为 1113/1082（1.029×）；G2S 九档的 B/H 均在
   0.994–1.004×，roundtrip 在 1.008–1.027×。
10. **B200 的大容量 TMA multi-CTA 压力明显小于 H100。**32 KiB、
    4xSM 的 TMA mean-CTA latency 为 B200 1876、H100 4030 cycle；
    这里实际 CTA 数是 592 对 528，因此这是同 xSM 规模语义，不是同
    数值 CTA 的严格比较。

这些都是平台自己的原始 cycle：CUDA 使用 `clock64()`，Ventus 使用
GVM `mcycle`。两边没有共同物理频率，所以本文比较“周期压力和拐点”，
不把 `V/H<1` 宣称为流片后的绝对时间更快。

## 正式数据规模

| 数据集 | 配置 | attempts × repeats | 样本 | errors |
|---|---:|---:|---:|---:|
| H100 feature validated-core | 183 | 2×3 | 1098 | 0 |
| B200 feature validated-core | 183 | 2×3 | 1098 | 0 |
| H100 uarch | 16 | 2×9 | 288 | 0 |
| B200 uarch | 16 | 2×9 | 288 | 0 |
| Ventus capacity | 27 | 1×1 | 27 | 0；status 0 |
| Ventus descriptor（成功项） | 29 | 1×1 | 29 | 0；status 0 |
| Ventus validated interleave | 1 | 1×1 | 1 | 0；status 0 |
| Ventus uarch（成功项） | 11 | 1×1 | 11 | 0；status 0 |
| H100↔Ventus feature 精确交集 | 55 | — | — | 全部正确 |
| H100↔Ventus uarch 精确交集 | 11 | — | — | 全部正确 |
| H100↔B200 feature 同语义键 | 183 | 2×3 / 2×3 | 2196 | 全部正确 |
| H100↔B200 uarch 同参数键 | 16 | 2×9 / 2×9 | 576 | 全部正确 |

279 行覆盖列表按 suite/case 给出每个平台的参数、样本数、中位数和状态。
H100/B200 各有 219 行 `passed`、2 行已知非法 interleave 排除、58 行
Ventus-only contention 未运行；Ventus 有 115 行 `passed`、37 行短超时、
127 行不在其测试列表。这里的行包含平台专属方法，不能把 219 与 115
理解成相同配置数；精确三方交集分别是 common-command 20、
feature 55、uarch 11。

修复后 H100 feature 两次远端 benchmark wall time 为 2.674/2.416 秒。
B200 完整 core 首次单进程运行在 4 秒 child timeout 内未完成，因此没有
延长 timeout，而是拆成 9 个独立短分片；正式 18 个分片 body 均为
2.043–3.490 秒。H100 uarch 为 0.848/1.783 秒，B200 完整 uarch 为
3.303/2.935 秒。Modal 本地显示的更长时间包含 GPU 排队、容器启动和
结果回传，不是 benchmark 子进程 wall time。

## 四条常用非 Reduce 命令

这组是最直接的命令级比较：bulk G2S/S2G 与 bulk.tensor G2S/S2G，
五个完全相同的 shape，每张 CUDA 卡 3 attempts × 9 repeats，Ventus
3 attempts × 1 repeat。周期包含对应异步协议的完成等待和正确性闭环。

| bytes/shape | path | H100 | B200 | Ventus |
|---|---|---:|---:|---:|
| 16 / 1×4 | bulk G2S | 507 | 440 | 86 |
| 16 / 1×4 | bulk S2G | 43 | 43 | 56 |
| 16 / 1×4 | tensor G2S | 628 | 577 | 119 |
| 16 / 1×4 | tensor S2G | 86 | 90 | 88 |
| 64 / 4×4 | bulk G2S | 507 | 440 | 86 |
| 64 / 4×4 | bulk S2G | 43 | 43 | 56 |
| 64 / 4×4 | tensor G2S | 628 | 577 | 119 |
| 64 / 4×4 | tensor S2G | 92 | 93 | 88 |
| 256 / 4×16 | bulk G2S | 507 | 440 | 88 |
| 256 / 4×16 | bulk S2G | 51 | 51 | 58 |
| 256 / 4×16 | tensor G2S | 628 | 577 | 121 |
| 256 / 4×16 | tensor S2G | 93 | 98 | 90 |
| 1024 / 16×16 | bulk G2S | 507 | 440 | 108 |
| 1024 / 16×16 | bulk S2G | 75 | 75 | 70 |
| 1024 / 16×16 | tensor G2S | 628 | 589 | 141 |
| 1024 / 16×16 | tensor S2G | 117 | 122 | 102 |
| 4096 / 32×32 | bulk G2S | 507 | 590 | 192 |
| 4096 / 32×32 | bulk S2G | 171 | 171 | 161 |
| 4096 / 32×32 | tensor G2S | 628 | 577 | 225 |
| 4096 / 32×32 | tensor S2G | 215 | 220 | 188 |

B200 相对 H100 的命令级特征很清楚：bulk G2S 在 1 KiB 以内少
13.2% cycle，但 4 KiB 多 16.4%；bulk S2G 五档完全相同；
tensor G2S 少 6.2–8.1%，tensor S2G 多 1.1–5.4%。完整 min/mean/max
和三方差值见
[`data/TMA_COMMON_PATH_CYCLE_COMPARISON_20260728.csv`](data/TMA_COMMON_PATH_CYCLE_COMPARISON_20260728.csv)。

## 容量：精确参数周期

Tensor Map 参数统一为 rank 2、UINT8、contiguous、坐标 0、
element stride 1、interleave/swizzle/promotion NONE。每个 CUDA
单元格为两次 attempt 共 6 个样本的中位数；Ventus 当前为一个独立
正确样本。

| bytes | direction | H100 | B200 | Ventus | B/H | V/H |
|---:|---|---:|---:|---:|---:|---:|
| 128 | G2S | 460 | 461 | 123 | 1.002× | 0.267× |
| 128 | S2G | 62 | 93 | 88 | 1.500× | 1.419× |
| 128 | roundtrip | 553 | 568 | 194 | 1.027× | 0.351× |
| 256 | G2S | 464 | 461 | 125 | 0.994× | 0.269× |
| 256 | S2G | 66 | 97 | 90 | 1.470× | 1.364× |
| 256 | roundtrip | 557 | 572 | 198 | 1.027× | 0.355× |
| 512 | G2S | 460 | 461 | 129 | 1.002× | 0.280× |
| 512 | S2G | 74 | 105 | 94 | 1.419× | 1.270× |
| 512 | roundtrip | 565 | 580 | 206 | 1.027× | 0.365× |
| 1024 | G2S | 460 | 461 | 145 | 1.002× | 0.315× |
| 1024 | S2G | 90 | 121 | 120 | 1.344× | 1.333× |
| 1024 | roundtrip | 581 | 596 | 248 | 1.026× | 0.427× |
| 2048 | G2S | 460 | 461 | 173 | 1.002× | 0.376× |
| 2048 | S2G | 122 | 153 | 159 | 1.254× | 1.303× |
| 2048 | roundtrip | 613 | 628 | 315 | 1.024× | 0.514× |
| 4096 | G2S | 460 | 461 | 229 | 1.002× | 0.498× |
| 4096 | S2G | 186 | 217 | 202 | 1.167× | 1.086× |
| 4096 | roundtrip | 678.5 | 692 | 414 | 1.020× | 0.610× |
| 8192 | G2S | 608 | 609 | 341 | 1.002× | 0.561× |
| 8192 | S2G | 314 | 345 | 315 | 1.099× | 1.003× |
| 8192 | roundtrip | 805 | 820 | 639 | 1.019× | 0.794× |
| 16384 | G2S | 608 | 609 | 565 | 1.002× | 0.929× |
| 16384 | S2G | 570 | 601 | 523 | 1.054× | 0.918× |
| 16384 | roundtrip | 1209 | 1224 | 1071 | 1.012× | 0.886× |
| 32768 | G2S | 756 | 759 | 1013 | 1.004× | 1.340× |
| 32768 | S2G | 1082 | 1113 | 939 | 1.029× | 0.868× |
| 32768 | roundtrip | 1869 | 1884 | 1935 | 1.008× | 1.035× |

完整 min/median/max 和样本数见
[`data/TMA_EXTENDED_EXACT_CYCLE_COMPARISON_20260728.csv`](data/TMA_EXTENDED_EXACT_CYCLE_COMPARISON_20260728.csv)。

## Shape、rank、stride、OOB 与 swizzle

相同 payload 的 tile shape 与容量表基本一致，说明当前 contiguous
case 的主变量仍是字节数，而不是命名 shape。几个有区分度的点如下。

| case | direction | H100 | Ventus | 观察 |
|---|---|---:|---:|---|
| rank1 contiguous，128 B | G2S | 460 | 120 | 小容量固定开销主导 |
| rank2 contiguous，4 KiB | G2S | 460 | 229 | 基线 |
| rank3 contiguous，4 KiB | G2S | 460 | 232 | Ventus 相对 rank2 +3 |
| rank4 contiguous，4 KiB | G2S | 460 | 235 | Ventus 相对 rank2 +6 |
| rank5 contiguous，4 KiB | G2S | 460 | 238 | Ventus 相对 rank2 +9 |
| row stride 2，4 KiB | G2S | 460 | 229 | 本 case 无额外 cycle |
| OOB 25%/50%，4 KiB | G2S | 460 | 229 | Ventus 与全 in-bounds 相同 |
| OOB 100%，4 KiB | G2S | 460 | 146 | 全 OOB 减少实际搬运 |
| swizzle 32/64/128，4 KiB | roundtrip | 822/675/677.5 | 414/414/414 | 32B 模式在本轮多 144.5 cycle |

`rank2_element_stride2` 的 CUDA 正确样本为 460 cycle，但 Ventus GVM
在独立 5 秒和 10 秒探针中均未完成，所以不进入精确交集。它与
outstanding、1024-map 一起应优先做进度/死锁定位。

## H100/B200 outstanding 与 multi-CTA 压力

### Outstanding

feature suite 使用 grid-constant Tensor Map，由一个 CTA 发射 N 个独立
4 KiB G2S 请求。表中周期是整个 batch 的完成周期。

| outstanding | H100 batch | H100/request | B200 batch | B200/request |
|---:|---:|---:|---:|---:|
| 1 | 507.5 | 507.5 | 469 | 469.0 |
| 2 | 646 | 323.0 | 685 | 342.5 |
| 4 | 1050 | 262.5 | 1212.5 | 303.1 |
| 8 | 1951 | 243.9 | 2487 | 310.9 |

N=8 的 batch latency 相对 N=1，H100/B200 分别增长 3.844/5.303 倍，
而工作量是 8 倍；两边都降低了每请求摊销，但 B200 在 N=4 后收益趋平。

上表本身只有 batched 曲线，单凭“每请求摊销下降”不能严格区分执行重叠、
wait 固定成本和其他批处理效应。为消除这个歧义，2026-07-30 新增 H100
同 kernel 严格对照：相同 N 条指令、地址、barrier 和字节数，只移动
wait 的位置。

| N | serial | batched | serial/batched | batched/request |
|---:|---:|---:|---:|---:|
| 1 | 454 | 454 | 1.000x | 454.000 |
| 2 | 898 | 552 | 1.627x | 276.000 |
| 4 | 1781 | 640 | 2.783x | 160.000 |
| 8 | 3547 | 991 | 3.579x | 123.875 |

两次 attempt 的 N=8 batched 中位数都为 991 cycle，serial 分别为
3548/3546 cycle；所有 144 个样本 errors=0。SASS 有精确 30 条
`UTMALDG.2D`，并确认 batched N=8 在第一个 barrier wait 前发出 8 条，
serial 则每条后立即 wait。因此本表可以归因为多条 TMA 搬运显著重叠。
它测的是同 CTA、同 TensorMap、不同坐标的热层次 G2S，不是公开队列深度
或冷 DRAM 并发测试。原始 CSV、源码、PTX、SASS、哈希和费用约束见
[`../benchmarks/tma-cycle-compare/results/cuda_h100_command_overlap/`](../benchmarks/tma-cycle-compare/results/cuda_h100_command_overlap/)。

为避免把“重叠”误写成“N 套搬运器同时工作”，又将 N 扩到 48，并使用
固定 runtime kernel、固定 192 KiB shared/global slab 和固定 TensorMap
进行 issue/complete 双计时。代表点如下：

| N | serial | batched issue 完成 | batched 全部完成 | serial/batched |
|---:|---:|---:|---:|---:|
| 8 | 3819 | 520 | 1282 | 2.979x |
| 16 | 7637 | 962 | 2452 | 3.115x |
| 32 | 15274.5 | 1846 | 4792 | 3.188x |
| 33 | 15750 | 1937 | 5118 | 3.077x |
| 48 | 22908 | 2730 | 7348 | 3.118x |

N=12–32 时，serial、batched issue、batched completion 的边际斜率分别
为 477.375、55.250、146.250 cycles/command；N=33–48 的 batched
completion 斜率为 148.667 cycles/command。说明发射端能比数据服务端
更快地产生请求，多个异步操作确实会同时处于未完成状态，但服务吞吐最终
稳定为约 146–149 cycles/4 KiB，加速在约 3.1–3.2x 饱和。它更像单个或
少数数据通道的请求队列与深流水，不是 48 套完整搬运器。

N=32→33 有约 180-cycle 的一次性 completion 台阶，但 issue 端只增加
91 cycles，且相同 runtime loop 在其他余数位置也有 issue 波动；所以
不能据此声称 CUDA TMA queue depth 就是 32。完整 21 点、两次 attempt、
源码、SASS 与边界分析见
[`../benchmarks/tma-cycle-compare/results/cuda_h100_command_overlap_issue_timeline/`](../benchmarks/tma-cycle-compare/results/cuda_h100_command_overlap_issue_timeline/)。

### Multi-CTA

H100 本次分配有 132 SM；1x/2x/4xSM 分别为 132/264/528 CTA。
数字是所有 CTA 每 repeat 周期的平均值再跨样本取中位数。

| bytes/CTA | 1 CTA | 132 CTA | 264 CTA | 528 CTA | 528/1 CTA |
|---:|---:|---:|---:|---:|---:|
| 4096 | 460 | 524 | 575.5 | 610.5 | 1.327× |
| 16384 | 608 | 775 | 1026.5 | 2042.5 | 3.359× |
| 32768 | 756 | 1292 | 2326 | 4030 | 5.331× |

32 KiB、528 CTA 是本轮 H100 的最大 pressure latency。它并不是单 CTA
命令延迟，也不能直接换算为 aggregate bandwidth；它表明容量和并发
必须交叉测试，分别只测一个维度会漏掉最差点。

B200 本次有 148 SM，因此对应 148/296/592 CTA：

| bytes/CTA | 1 CTA | 148 CTA | 296 CTA | 592 CTA | 592/1 CTA |
|---:|---:|---:|---:|---:|---:|
| 4096 | 461 | 571 | 599 | 607 | 1.317× |
| 16384 | 611 | 639 | 753 | 939 | 1.537× |
| 32768 | 757 | 980 | 1170 | 1876 | 2.478× |

虽然 B200 的 xSM 点实际 CTA 更多，它在 16/32 KiB 的 TMA mean-CTA
pressure 增长仍显著小于 H100。由于数值 CTA 不相同，这里只用于比较
按设备规模归一化后的压力曲线，不能解释成相同并发度的延迟差。

### Compute overlap

4 KiB overlap case：

| mode | H100 | B200 |
|---|---:|---:|
| compute only | 28739 | 23617 |
| TMA then compute（serial） | 29176 | 24108.5 |
| TMA 与 compute overlap | 28917 | 23928 |

相对 compute-only，serial 增加 437 cycle，overlap 只增加 178 cycle，
即 H100 在该人工计算段中隐藏约 `59.3%` 的额外周期。B200 对应增量为
491.5/311 cycle，隐藏约 `36.7%`。

完整 H100 pressure 数据见
[`data/TMA_H100_PRESSURE_CYCLES_20260728.csv`](data/TMA_H100_PRESSURE_CYCLES_20260728.csv)。

## Descriptor working-set 与 prefetch

这一套 uarch benchmark 使用 device-memory Tensor Map 地址，并在 host
launch 间轮转 descriptor，与 Ventus uarch case 保持相同参数。两边
`warmups=0`，以避开 GVM 同 kernel 重复 TMA 的已知问题；CUDA 使用两次
attempt 共 18 个样本的中位数，Ventus 为一个正确样本。

| maps | H100 pf0/pf1 | Ventus pf0/pf1 | B200 pf0/pf1 |
|---:|---:|---:|---:|
| 1 | 703 / 703 | 236 / 222 | 687 / 687 |
| 4 | 703 / 703 | 236 / 222 | 687 / 687 |
| 16 | 851 / 851 | 236 / 222 | 687 / 687 |
| 64 | 851 / 851 | 236 / 222 | 1131 / 983 |
| 256 | 703 / 703 | 236 / 222 | 1131 / 983 |
| 1024 | 703 / 705.5 | 5 秒未完成 | 687 / 690.5 |

H100 的结果没有随 map 数量单调增长，也没有稳定的 prefetch 中位数收益。
Ventus 1–256 maps 没有 working-set 增长，但 prefetch 稳定少 14 cycle；
1024 maps 的 host descriptor 准备和 kernel 整体在短预算内未完成，
当前不能确定慢在 patch、cache 还是 TMA 进度。B200 的 64/256 maps
同为 1131/983，prefetch 少 148 cycle；但 1024 maps 又回到约 690，
同样不是单调 working-set 曲线。

同一 uarch 路径的 outstanding：

| outstanding | H100 batch cycle | B200 batch cycle | Ventus |
|---:|---:|---:|---:|
| 1 | 899.5 | 916 | 260 |
| 2 | 891 | 1049 | 5 秒未完成 |
| 4 | 1183 | 1199 | 5 秒未完成 |
| 8 | 1465 | 1621 | 5 秒未完成 |

完整数据见
[`data/TMA_EXTENDED_UARCH_CYCLE_COMPARISON_20260728.csv`](data/TMA_EXTENDED_UARCH_CYCLE_COMPARISON_20260728.csv)。

## B200 完整矩阵结论

B200 已不再是常用子集：命令基线、183-config feature、16-config uarch
均与 H100 对齐。分析器逐项验证了 `direction/method/case_id` 的 183 个
语义键，并要求 bytes、rank、layout、outstanding、batch 相同；只有
`1x/2x/4xSM` 的 CTA 数按实际 SM 数自然不同（H100 132 SM、B200
148 SM）。

| 特性 | H100 | B200 | 观察 |
|---|---:|---:|---|
| 128 B feature S2G | 62 | 93 | B200 固定开销高 31 cycle |
| 32 KiB feature S2G | 1082 | 1113 | 差距收敛到 31 cycle |
| 32 KiB feature G2S | 756 | 759 | 基本相同 |
| 32 KiB roundtrip | 1869 | 1884 | B200 +0.8% |
| interleave16_u16 roundtrip | 681 | 693 | B200 +1.8% |
| 32 KiB TMA 4xSM | 4030（528 CTA） | 1876（592 CTA） | B200 压力曲线更缓 |
| feature outstanding=8 | 1951 | 2487 | B200 batch +27.5% |
| uarch outstanding=8 | 1465 | 1621 | B200 +10.6% |

最稳定的共同特性是单 CTA G2S：九个容量点的 B/H 都在
0.994–1.004×。最显著的架构差异集中在短 S2G、descriptor 64/256 maps、
高 outstanding 和多 CTA 压力，而不是单请求大容量 G2S。

## Ventus 双向与争用短预算

`run_ventus_contention_short.py` 对 58 个 case 每个启动独立进程。第一个
case 最多 8 秒用于编译，后续每个 case 最多 3 秒；timeout 后继续下一项，
日志只保留结构化结果。

| 子矩阵 | 完成 | 3 秒 timeout |
|---|---:|---:|
| core：isolated、same-set、bidirectional | 17 | 27 |
| pressure：dualwarp、burst | 10 | 4 |
| 合计 | 27 | 31 |

已观察到的结构：

- 128 B bulk/tensor、G2S/S2G、1/4 WG 全部完成；
- 4 KiB spread S2G 在 bulk/tensor、1/4 WG 下完成，但 G2S 超时；
- 4 KiB、4 WG 的四个 same-set case 全部超时；
- bulk 4 KiB 的 `S2G→G2S` 完成，而 `G2S→S2G` 超时；
- tensor 4 KiB 的 `G2S→S2G` 在 1/4 WG 完成，
  `S2G→G2S` 只有 1 WG 完成；
- 4 KiB dualwarp 在 bulk/tensor、1/2 WG 全部完成；
- 16 KiB core 和 dualwarp 点均超时；
- 128 B/1 KiB、depth 2/4 的已选 burst 点全部完成。

完成项示例：bulk 128 B G2S 为 56 cycle，S2G 为 37 cycle；4 KiB
bulk S2G 从 1 WG 的 146 cycle 增至 4 WG 的 311.25 cycle；
4 KiB tensor dualwarp 从 1 WG 的 258 cycle 增至 2 WG 的 412 cycle。

这些 timeout 不能自动解释成 RTL deadlock：GVM 仿真墙钟受编译、checker
和并发日志影响。本表的价值是把危险组合定位出来，并证明 same-set、
方向顺序、多 WG 和容量是单事务 benchmark 没覆盖的重要维度。

原始结果：

- [`../benchmarks/tma-cycle-compare/results/contention_ventus_core_v2/`](../benchmarks/tma-cycle-compare/results/contention_ventus_core_v2/)
- [`../benchmarks/tma-cycle-compare/results/contention_ventus_pressure/`](../benchmarks/tma-cycle-compare/results/contention_ventus_pressure/)

## tma-refer CUDA interleave 故障、根因收敛与修复

最初的完整 H100 core 在第一个 attempt 触发：

```text
XID 13: Graphics SM Warp Exception, Out Of Range Address
XID 43: channel stopped
```

4 秒子进程 timeout 随即终止，第二个 attempt 没有启动。原故障证据保留在
[`../benchmarks/tma-cycle-compare/results/features_cuda_h100/`](../benchmarks/tma-cycle-compare/results/features_cuda_h100/)。

CUDA Driver API 明确定义 `CU_TENSOR_MAP_INTERLEAVE_16B/32B`，并以
UINT16 的 NC/8HWC8（C8 占 16 B）和 NC/16HWC16（C16 占 32 B）为
示例；PTX ISA 还说明 channel slice 按 16/32 B 分组。因此原故障不能
解释成“CUDA/H100 不支持 interleave”：

- [CUDA Driver API：Tensor Map Object Management](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)
- [PTX ISA：Interleave layout](https://docs.nvidia.com/cuda/archive/12.2.2/parallel-thread-execution/index.html#interleave-layout)

tma-refer 的 2026-07-12 H100/B200/B300
formal run 使用相同参考源码，三张卡都在完成 139/184 配置后、首个
`interleave16` 之前以 illegal address 结束。139 的顺序分解恰好是：

```text
movement 81 + tiles 45 + rank/stride/OOB 10 + swizzle 3 = 139
```

本轮新增独立探针，每个 paid function 只执行一个阶段，子进程 timeout
3 秒、function timeout 10 秒、retries=0，并把底层 allocation 从
4096 B 扩大到 64 KiB 以排除简单分配过小：

| H100 case | 阶段 | host encode | device 结果 | remote wall |
|---|---|---|---|---:|
| 原 UINT8 `[16,8,32]`, stride `[16,128]` | G2S | 接受 | illegal access；XID 13/43 | 1.396 s |
| 同上 | S2G | 接受 | illegal access；XID 13/43 | 1.487 s |
| UINT8 canonical-C8 `[8,8,32]` | G2S | `CUDA_ERROR_INVALID_VALUE` | 未启动 | 0.529 s |
| UINT16、保留原 C/W/N `[16,8,32]` | G2S | 接受 | 完成 | 1.237 s |
| UINT16 canonical NC/8HWC8 `[8,8,32]` | roundtrip | 接受 | 完成；逐字节 errors=0 | 1.006 s |
| UINT16 canonical interleave32 候选 | roundtrip | 接受 | illegal access；XID 13/43 | 1.196 s |

这把原 XID 从“roundtrip 某处”收敛到：H100 + CUDA 13.3 + driver
580.95.05 上，旧 UINT8/non-NONE interleave 描述符虽然被 host encode
接受，但 load 和 store 两条 device 路径都会 fault。由于公开 API 没有
写明 UINT8 禁止与 interleave 组合，本文不把它升级成“CUDA 明确不支持
UINT8”，更准确的分类是 host 校验未拦截的运行时无效组合或实现缺口。

修复不是延长 timeout，而是用文档给出的 UINT16 NC/8HWC8 正控替换旧
UINT8 性能项。CUDA 与 Ventus 参数均为 rank 3、UINT16、dims/box
`[8,8,32]`、global strides `[16,128]`、4096 B、
interleave16、swizzle NONE、roundtrip：

| 平台 | 样本 | min | median | max | errors/status |
|---|---:|---:|---:|---:|---|
| H100 | 6 | 678 | 681 | 730 | errors=0 |
| B200 | 6 | 693 | 693 | 849 | errors=0 |
| Ventus GVM | 1 | 422 | 422 | 422 | errors=0；status=0 |

该点已进入
[`data/TMA_EXTENDED_EXACT_CYCLE_COMPARISON_20260728.csv`](data/TMA_EXTENDED_EXACT_CYCLE_COMPARISON_20260728.csv)。
原 UINT8 interleave16 与 interleave32 候选仍不进入性能表。探针证据在
[`../benchmarks/tma-cycle-compare/results/interleave_probe_h100/`](../benchmarks/tma-cycle-compare/results/interleave_probe_h100/)，
修复后的正式 H100 数据在
[`../benchmarks/tma-cycle-compare/results/features_cuda_h100_interleave_fixed/`](../benchmarks/tma-cycle-compare/results/features_cuda_h100_interleave_fixed/)。
B200 在放开完整矩阵前还单独通过了同一 canonical roundtrip 探针
（2.665 秒、errors=0），证据在
[`../benchmarks/tma-cycle-compare/results/interleave_probe_b200/`](../benchmarks/tma-cycle-compare/results/interleave_probe_b200/)；
正式完整数据在
[`../benchmarks/tma-cycle-compare/results/features_cuda_b200_full_sharded_v2/`](../benchmarks/tma-cycle-compare/results/features_cuda_b200_full_sharded_v2/)。

实现使用
[`../benchmarks/tma-cycle-compare/cuda_tma_feature_safe.cu`](../benchmarks/tma-cycle-compare/cuda_tma_feature_safe.cu)
保留原参考 kernel，只替换有问题的 case 参数；interleave32 因正控候选
仍 fault，继续明确排除。

## 费用与超时控制

所有正式 Modal 脚本遵循：

- CUDA binary 在 CPU image build 阶段预编译；
- GPU child timeout 4 秒；
- Modal function timeout 10 秒（Modal 1.4.2 最小允许值）；
- `retries=0`、`max_containers=1`；
- attempts 串行；
- child timeout、非零退出、CSV 不完整或 correctness error 后立即停止，
  不启动下一 attempt。

此外增加了 CPU-only import probe，先验证容器导入、source/wrapper mount
和 SHA-256，再申请 GPU。一次早期脚本错误在容器导入时检查宿主绝对路径，
造成平台重复拉起容器；该检查已移到 local entrypoint，并用 probe 验证
修复。

修复后正式 H100 core 的两次远端 body 为 2.674/2.416 秒。B200 单进程
完整 core 首次在 4 秒内未完成，脚本立即停止且未启动第二 attempt；没有
提高 timeout，而是先拆 5 片，再依据一次 3.896 秒边界值细分为 9 片。
最终 2 attempts × 9 片全部通过，每片 2.043–3.490 秒，合并后仍严格
校验为 183 configurations × 3 repeats。分片只改变进程分组，不改变
case 参数或 kernel 内 `clock64()` 计时区间。

预算门控过程的失败证据保留在
[`features_cuda_b200_full`](../benchmarks/tma-cycle-compare/results/features_cuda_b200_full/)
和
[`features_cuda_b200_full_sharded`](../benchmarks/tma-cycle-compare/results/features_cuda_b200_full_sharded/)；
正式数据只读取 `features_cuda_b200_full_sharded_v2`。

## 可比性与限制

1. CUDA cycle 和 Ventus GVM cycle 没有共同物理频率。
2. H100/B200 是实机；Ventus 是 RTL/GVM 周期模型。
3. H100 feature 在一个进程内运行，B200 为守住 4 秒预算拆成 9 个短
   进程；两边各 case 都使用一轮 warmup，但进程/cache 初态不能证明
   完全一致。Ventus feature 也是每个 case 独立进程。
4. CUDA CTA 为 128 threads，Ventus workgroup 为 32 work-items；
   TMA 都只由一个 lane 发射。
5. descriptor encode、host/kernel launch、初始化和 correctness 校验均在
   feature 计时窗外；uarch descriptor working-set 在 host launch 间轮转。
6. multi-CTA 表是 mean-CTA completion latency，不是 aggregate bandwidth。
7. Ventus feature/uarch 当前只有一个正确 cycle 样本；它在 GVM 上具有
   确定性，但统计强度低于 CUDA。
8. contention timeout 是墙钟预算结果，不是 cycle 上界。

## 证据与复现

主要 raw data：

- [H100 validated feature](../benchmarks/tma-cycle-compare/results/features_cuda_h100_interleave_fixed/)
- [B200 validated feature](../benchmarks/tma-cycle-compare/results/features_cuda_b200_full_sharded_v2/)
- [H100 common-command baseline](../benchmarks/tma-cycle-compare/results/cuda_h100/)
- [B200 common-command baseline](../benchmarks/tma-cycle-compare/results/cuda_b200_full/)
- [H100 uarch](../benchmarks/tma-cycle-compare/results/uarch_cuda_h100/)
- [B200 uarch](../benchmarks/tma-cycle-compare/results/uarch_cuda_b200_full/)
- [Ventus capacity](../benchmarks/tma-cycle-compare/results/features_ventus_capacity_v2/)
- [Ventus descriptor](../benchmarks/tma-cycle-compare/results/features_ventus_descriptor/)
- [Ventus validated interleave](../benchmarks/tma-cycle-compare/results/features_ventus_interleave_fixed/)
- [H100 interleave isolation probes](../benchmarks/tma-cycle-compare/results/interleave_probe_h100/)
- [B200 interleave positive probe](../benchmarks/tma-cycle-compare/results/interleave_probe_b200/)
- [Ventus uarch](../benchmarks/tma-cycle-compare/results/uarch_ventus/)

重新校验和聚合：

```bash
python3 benchmarks/tma-cycle-compare/analyze_extended_results.py
```

脚本要求 H100/B200 各自精确的 183/183/16/16 个 CUDA 配置和完整
repeat grid，只允许
已验证的 `interleave16_u16`，并拒绝 correctness error、非正 cycle、
其他 unexpected interleave 和不完整数据。
生成：

- [精确 feature 周期表](data/TMA_EXTENDED_EXACT_CYCLE_COMPARISON_20260728.csv)
- [uarch 周期表](data/TMA_EXTENDED_UARCH_CYCLE_COMPARISON_20260728.csv)
- [H100 pressure 周期表](data/TMA_H100_PRESSURE_CYCLES_20260728.csv)
- [三平台逐项覆盖列表](data/TMA_THREE_PLATFORM_COVERAGE_MATRIX_20260728.csv)
- [四条常用命令三平台周期表](data/TMA_COMMON_PATH_CYCLE_COMPARISON_20260728.csv)
- [数据摘要](data/TMA_EXTENDED_CYCLE_SUMMARY_20260728.json)
- [证据 SHA-256](data/TMA_EXTENDED_CYCLE_EVIDENCE_SHA256_20260728.txt)

Reduce 的旧 raw artifact 仍保留以便追溯，但本报告、扩展聚合脚本和正式
结论均不读取或引用 Reduce 周期。
