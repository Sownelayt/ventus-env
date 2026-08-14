# Hopper TMA 高吞吐原因与 Ventus V3.3 G2S 瓶颈

> 日期：2026-07-30
> 范围：单条非 Reduce TMA，重点为 contiguous G2S
> 约束：不修改 L2、shared memory、TLB、调度器、ISA 或 TensorMap 格式
> 状态：分析和诊断测试已完成；V3.3 生产 RTL 未因本文修改，DC 输入不变

## 结论

Ventus 当前并不是“硬件结构上做不到每周期发一个 L2 请求”。实际
`TmaV2WindowEngine` 诊断证明，在下游持续 ready、固定 38-cycle
cache response latency、有效 credit 和 slot 足够时，256 个连续 128B
请求可以做到：

```text
issue_span = 255 cycles
issue_cycles_per_line = 1.000
```

当前 V3.3 发布参数做不到持续 `1 request/cycle`，主要因为只有 16 个
global request entry 和 24 个带 1024-bit payload 的 TransferSlot。
32 KiB G2S 的实测平均 response latency 是 37.77 cycle，16 个 credit
只能覆盖约 42% 的单周期发射带宽：

```text
maximum sustained rate ≈ credits / latency
                       = 16 / 37.77
                       = 0.424 line/cycle

minimum steady interval ≈ latency / credits
                        = 37.77 / 16
                        = 2.361 cycle/line
```

V3.3 实际 16→32 KiB 边际值为 2.500 cycle/line，与这个 credit 上限
只差约 5.9%。因此，当前大容量 G2S 的第一瓶颈已经可以定性为：

**L2 返回延迟没有被足够多的轻量在途请求覆盖，而不是 Planner、地址计算、
transform 或 shared 端口每次必须花 2–3 cycle。**

最有价值的下一步不是复制计算单元，也不是恢复多个 active command，而是
把“在途 line 的轻量跟踪”从“含 1024-bit payload 的 TransferSlot”中
拆开。保留单 active command，同时提供约 32–40 个 compact line context，
只保留少量真正存放 response payload 的 buffer。按当前 RTL 模型估算，
这一项可能把 32 KiB G2S 的 line slope 从 2.5 降到约 1.0–1.2
cycle/line，数据阶段改善约 1.8–2.1 倍；计入固定开销后的整条命令改善
大约 1.4–1.8 倍。它是模型内上限估计，不是流片性能承诺。

## 比较口径

### H100 的 1.156 cycle/line

本文使用的 `1.156 cycle/128B` 是 H100 同参数 G2S 从 16 KiB 增加到
32 KiB 的平均边际成本：

```text
(756 - 608) cycles / ((32768 - 16384) / 128)
= 148 / 128
= 1.15625 cycle/128B
```

它不是 NVIDIA 公开的内部 L2 request issue interval，也不表示每一条
cache line 都严格间隔 1.156 cycle 完成。H100 容量曲线存在台阶，该值
是额外 128 条 line 的平均斜率。

CUDA benchmark 使用 `clock64()` 包围 `expect_tx`、TMA issue 和
mbarrier completion；warmup 后重复访问同一坐标。因此这是 warm
cache/L2-to-shared 路径的完整协议时间，不是 cold HBM latency。
独立的 H100 microbenchmark 也明确把其延迟定义为 warm-path
`expect_tx -> UTMALDG.2D -> mbarrier completion`，见
`/home/liyb/tma-refer/results/h100_2026-07-11/README.md`。

### Ventus 的 2.500 cycle/line

V3.3 的值同样来自 16→32 KiB 容量增量：

```text
(765 - 445) cycles / 128 lines
= 2.500 cycle/128B
```

Ventus 使用 GVM `mcycle`，CUDA 使用 GPU `clock64()`，两者不是同一
物理时钟频率。这里比较的是架构上的“每周期能推进多少 128B line”，不把
周期比直接解释为实际时间比。

### 与 shared 理论带宽的关系

CUDA Programming Guide 给出的 shared memory 组织是 32 bank、每 bank
每周期 32 bit，因此无 bank conflict 的聚合参考上限是
`32 × 32 bit = 128 B/cycle`。这不是 NVIDIA 对 TMA shared 端口的具体
实现承诺，但可作为合理的上限参照：

| 平台 | 边际 cycle/128B | 等效 B/cycle | 相对 128 B/cycle |
|---|---:|---:|---:|
| H100 | 1.156 | 110.7 | 86.5% |
| Ventus V3.3 | 2.500 | 51.2 | 40.0% |

H100 的 warm G2S 已接近一条 128B line/cycle，而 Ventus 只利用约四成
名义 line 带宽。这与 credit 覆盖率 `16/37.77 = 42.4%` 几乎重合。
[CUDA shared-memory bank 说明](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)

## NVIDIA 为什么能做得好

### 1. TMA 是独立异步引擎，而不是线程执行地址循环

NVIDIA Hopper Tuning Guide 将 TMA 描述为更复杂的异步 copy engine：
单线程可以发起大块搬运，搬运不占用寄存器，也不需要用一串 SM 指令完成，
线程可在数据在途时继续工作。
[NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/archive/12.8.0/hopper-tuning-guide/index.html)

Hopper 架构资料还明确说明，TMA 接管 stride、offset、boundary 和
地址生成，并通过异步 transaction barrier 完成同步。
[NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)

Ventus V3.3 的 Descriptor Compiler、Binder 和 Planner 已经实现了相同
方向的功能卸载，而且 Planner 能 II=1。因此这一层已不是当前 2.5
cycle/line 的来源。

### 2. 公开专利显示的是“队列 + 流水 + 独立完成跟踪”

NVIDIA 专利 US12141082B2 描述了一种可能的 TMAU 实现：

- SM 与 TMAU 紧耦合；部分实施例是一 SM 一 TMAU；
- 请求先进入内部 request queue；
- descriptor cache 可在当前请求执行时查询或预取下一条 descriptor；
- setup block 与 request generator 并行工作；
- request generator 将大块搬运拆成不超过一条 L2 cache line 的
  子请求；
- 子请求经 GNIC 发往 memory subsystem；
- response completion circuit 独立跟踪已发请求，response processor
  处理返回。

[NVIDIA TMAU patent US12141082B2](https://patents.google.com/patent/US12141082B2/en)

专利不能证明量产 H100 使用了完全相同的队列深度、位宽或物理连接，但它
可靠地说明了 NVIDIA 的设计方向：**命令准备、line request 生成、memory
request 队列、response completion 和 response data path 是解耦的。**

这正是 Ventus 当前还缺少的一层解耦。V3.3 已经把命令前端与数据搬运
解耦，但仍把“等待 L2 的轻量 line”与“已经返回的 1024-bit payload”
绑定在同一种宽 TransferSlot 中。

### 3. H100 实测确认内部存在深异步生命周期

项目已有单 CTA、单 issuing thread、每条 4 KiB、不同 global/shared
tile 的受控测试。N=48 时：

| 模式 | cycles |
|---|---:|
| 48 条逐条 issue→wait | 22,908 |
| batched 发完 48 条 | 2,730 |
| batched 全部完成 | 7,348 |

48 条命令在远早于全部完成时已经发完，稳定 batched issue 约
53–55 cycle/command，completion 约 146–149 cycle/command。它证明
H100 能保留大量异步工作并让请求生命周期重叠，但不证明存在 48 套完整
搬运引擎，也不能据此断言硬件 command queue 深度正好是 48。原始结果见
[`../benchmarks/tma-cycle-compare/results/cuda_h100_command_overlap_issue_timeline/README.md`](../benchmarks/tma-cycle-compare/results/cuda_h100_command_overlap_issue_timeline/README.md)。

独立 unique-source 测试在 528 CTA 时请求吞吐达到：

| tile | requested GB/s |
|---:|---:|
| 4 KiB | 1,864 |
| 16 KiB | 2,255 |
| 32 KiB | 2,518 |

由于该次 Nsight Compute counter 不可用，这些数字不能标为物理 HBM
带宽；但它们至少说明 H100 的 TMA、L2 分区和内存网络可以在大量并发下
持续服务很多 line，而不是按一条命令的完整延迟串行工作。公开微基准研究
同样报告 Hopper 的 L2 和 TMA 异步路径具有很高吞吐。
[Hopper microbenchmarking study](https://arxiv.org/abs/2501.12084)

### 4. NVIDIA 的优势主要是稳态吞吐，不是小命令启动时间

本项目 H100 G2S 在 128B–4KiB 都约为 460 cycle；Ventus V3.3 的
128B G2S 只有 131 cycle。换言之，H100 并不是所有指标都更小：

- Ventus 的建模固定启动成本更低；
- H100 的真实协议、cache 层次和片上网络启动成本更高；
- H100 一旦进入大块稳态搬运，line slope 约为 Ventus 的 46%。

所以“32 KiB 总周期 765 对 756 只差 9 cycle”具有误导性：Ventus 用
偏小的启动成本抵消了较差的稳态吞吐。继续扩大容量时，G2S 差距会继续
按 line slope 增长。

## Ventus V3.3 的直接 RTL 证据

32 KiB G2S measured repeat 的 PMU 为：

| 指标 | 值 |
|---|---:|
| total / active / fixed cycles | 765 / 662 / 103 |
| Planner produced / fire | 256 / 256 |
| Planner stall | 342 |
| full TransferSlot / request cycles | 343 / 359 |
| max slot / request / shared / ack | 24 / 16 / 2 / 0 |
| average cache response latency | 37.77 cycle/line |
| cache request port stall | 18 |
| transform stall / shared-full | 0 / 0 |
| longest continuous window fire run | 24 |

这些数字排除了几个候选原因：

1. Planner 没有漏发或重复，且具备每周期生成一个 window 的能力；
2. transform 和 shared 并未造成稳态停顿；
3. 外部 cache request 端口只 stall 18 cycle，不足以解释 359 cycle
   的 request-full；
4. request table 和 TransferSlot 同时到达容量上限；
5. 最长连续 window run 恰好是 24，直接对应 slot 深度。

完整 PMU 在
[`data/TMA_V3_3_BOTTLENECK_PMU.csv`](data/TMA_V3_3_BOTTLENECK_PMU.csv)。

## 新增的 credit 深度诊断

测试源码：
[`../gpgpu/ventus/tests/src/DmaTest/TmaV2CreditDepth_test.scala`](../gpgpu/ventus/tests/src/DmaTest/TmaV2CreditDepth_test.scala)

复现命令：

```bash
cd gpgpu
./mill -i 'ventus[6.4.0].tests.testOnly' DmaTest.TmaV2CreditDepth_test
```

测试条件：

- 实例化真实 `TmaV2WindowEngine`，不修改生产 RTL；
- 48 个内部 request entry 和 48 个 TransferSlot，避免内部容量先截断；
- 连续输入 256 个单-line、128B G2S window；
- synthetic cache 每周期可接收一个 request；
- 每个 response 固定在 request 后 38 cycle 返回；
- cache response、transform 和 shared 各允许每周期推进一次；
- 测试端把有效外部在途 credit 限为 16/24/32/40/48；
- 同周期 response 和新 request 可回收同一 credit。

结果：

| 有效 credit | 256-line issue span | 有限长度 cycle/line | 渐近 credit 下限 | command cycles |
|---:|---:|---:|---:|---:|
| 16 | 585 | 2.294 | 2.375 | 631 |
| 24 | 395 | 1.549 | 1.583 | 441 |
| 32 | 297 | 1.165 | 1.188 | 343 |
| 40 | 255 | 1.000 | 1.000 | 301 |
| 48 | 255 | 1.000 | 1.000 | 301 |

机器可读结果见
[`data/TMA_V3_3_G2S_CREDIT_DEPTH_DIAGNOSTIC.csv`](data/TMA_V3_3_G2S_CREDIT_DEPTH_DIAGNOSTIC.csv)。

`finite cycle/line` 用首尾 request 的 255 个间隔计算；由于开头可以先
突发填满 credit，它会略好于无限数据流的 `latency/credits` 渐近值。

最关键的结论有三个：

1. 16-credit 诊断的 631-cycle 数据阶段与生产 PMU 的 662 active
   cycles 很接近，说明固定延迟模型抓住了主要行为；
2. 32-credit 点已达到 1.165 finite cycle/line，与 H100 的 1.156
   容量边际值数值接近，但两者测量定义不同，不能视为内部结构相同；
3. 40/48-credit 点严格实现了连续 256 次每周期一个 L2 request，证明
   WindowEngine 的请求和数据流水不存在固有的 2-cycle 或 3-cycle
   issue 限制。

## 应该怎样改

### 不推荐：直接把完整 TransferSlot 增加到 48

V3.3 结构分析得到每增加一个当前格式的 slot 会增加 1,541 bit 状态，
其中 1,024 bit 是 payload。24→48 会先增加：

```text
24 × 1541 = 36,984 state bits
```

这还不包括更宽的选择、仲裁和 mux 网络。面积来源见
[`TMA_V3_3_SINGLE_CONTEXT_UNIFIED_BUFFER.md`](TMA_V3_3_SINGLE_CONTEXT_UNIFIED_BUFFER.md)。
在当前面积紧张的情况下，这不是合理的发布方案。

### 推荐：Compact Line Context 与 Payload Residency 分离

建议保留严格单 active command，把后端资源改成两个不同池：

1. `LineContext[32–40]`
   - 只跟踪已生成、已翻译、已发出或等待 response 的 line；
   - 保存 source、slot/window token、line/fragment、pending mask 和
     少量状态；
   - 不保存 1024-bit payload；
   - source ID 直接索引，不做大范围 CAM。
2. `ResponsePayload[4–8]`
   - 只在 response 实际到达后保存 128B 数据；
   - response 可直接进入单套 scatter/transform 流水；
   - shared request 接受后立即释放 payload entry；
   - 只用小 skid/FIFO 吸收 transform/shared 的短暂背压。

这样 32–40 个在途 line 的成本主要是 compact metadata，而不是
32–40 份 128B 数据。它保留 V3.3 的单 context、单 transform 和面积目标，
同时获得 NVIDIA 专利所体现的“request tracking 与 response processing
分离”的关键能力。

需要先确认 cache response 协议允许怎样的背压：

- 若 response 是标准 Decoupled 且 L2 可以被短暂 backpressure，小型
  payload FIFO 足够；
- 若 response 一旦发出就不可停止，发 request 前必须预留可证明安全的
  response credit，或按最大 shared/transform 停顿提前限流；
- 无论哪种情况，都不能在 response 到达后才发现无处存放数据。

### 建议参数顺序

先做参数原型，不立即冻结容量：

| 原型 | compact line context | payload entry | 目的 |
|---|---:|---:|---|
| A | 24 | 4 | 验证解耦功能和最小面积 |
| B | 32 | 6–8 | 接近 H100 finite slope |
| C | 40 | 6–8 | 验证长流严格 1 request/cycle |

当前 24 个宽 TransferSlot 不应原样保留再叠加 40 个 context。应根据
response residency 重新压缩或替换宽 slot，否则只会重复存储。

## 可能提高多少

固定 38-cycle response 模型中：

- 16 credit：631 command cycles；
- 32 credit：343 command cycles，数据阶段约改善 1.84 倍；
- 40 credit：301 command cycles，数据阶段约改善 2.10 倍。

把诊断的相对节省映射到 V3.3 的 `662 active + 103 fixed`：

```text
32-credit estimate:
  active ≈ 662 - (631 - 343) = 374
  total  ≈ 374 + 103 = 477 cycles

40-credit estimate:
  active ≈ 662 - (631 - 301) = 332
  total  ≈ 332 + 103 = 435 cycles
```

实际生产路径还有 cache 仲裁、TLB、slot 格式和随机背压，不能直接把
435/477 当验收值。较稳妥的判断是：

- line slope 有机会从 2.5 改善到 1.0–1.2 cycle/line，即约 2.1–2.5 倍；
- 32 KiB 完整命令有机会从 765 降到约 430–550 cycle，即约
  1.4–1.8 倍；
- 如果 compact-context 版本仍停在约 2.5 cycle/line，PMU 应能明确显示
  新瓶颈已经转移到 L2 `ready`、response 间隔或 shared backpressure。

这些数字只适用于当前 GVM/L2 模型。优化后 Ventus 模型可能明显少于
H100 的 756 cycle，这是因为 Ventus 固定启动成本和 L2 模型都偏乐观，
不能据此宣称真实 N12 芯片会快于 H100。

## 不应该优先做什么

- 不应继续展开 Planner 算术：Planner 已能 II=1。
- 不应复制 transform：32 KiB transform stall 为 0。
- 不应扩大 shared tag：G2S 最大只用 2/8。
- 不应为了单命令吞吐恢复多个 active command：credit 深度在单 context
  内就能解决主要问题。
- 不应优先优化 S2G ack：当前 S2G 边际 3.016 cycle/line 已比
  H100/B200 的 4.000 更乐观，先校准模型更重要。
- 不应修改他人的 L2：先把 TMA 自己的在途能力补足；现有 cache port
  stall 只有 18 cycle，尚无证据要求改变 L2 接口。

## 下一步

1. 等当前 V3.3 DC 返回，保留其作为未改动的面积/时序基线。
2. 在独立分支或版本中实现 compact `LineContext` 与小
   `ResponsePayload`，只改 TMA。
3. 复用本诊断，覆盖 response latency 13/24/38/64 cycle、
   response 乱序、shared 随机背压和 credit full 后同周期回收。
4. 新增严格验收：
   - 38-cycle latency、40 credit、256 line 时 request fire 无气泡；
   - 任意 response 到达时一定有合法 owner 和存储位置；
   - 单 active command 断言继续成立；
   - transform/shared 吞吐仍为一条 line/cycle。
5. 跑 128B、1/4/16/32 KiB G2S 回归。重点观察
   `full_request_cycles`、`full_payload_cycles`、最长连续
   `cacheRequest.fire` 和 response-to-shared latency。
6. 做 module-level DC 对比：
   - 当前 slot24/request16；
   - compact32/payload8；
   - compact40/payload8。
7. 只有 TMA credit 已不再 full、cache request 仍无法持续发射时，才把
   剩余差距归因到外部 L2/仲裁模型。

综合判断：**可以明显改善，而且主方向已经由测试确认。**NVIDIA 的核心
优势不是神秘的地址算法，而是用解耦队列和足够多的轻量在途状态覆盖真实
cache/网络延迟，并让 response 数据通路持续流动。Ventus 已经具备 II=1
的计算和单周期端口，下一步要补的是这一层延迟容忍结构，同时避免用大量
完整 payload slot 换吞吐。
