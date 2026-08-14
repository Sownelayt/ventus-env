# Ventus TMA 与 CUDA TMA 微架构差异及 L2/DRAM 建模失真分析

> 日期：2026-07-29  
> 范围：非 Reduce TMA；重点为 bulk、bulk.tensor、L2、共享内存和完成路径  
> 关联实测：
> [CUDA H100/B200 与 Ventus 非 Reduce TMA 周期对比](TMA_CUDA_VENTUS_CYCLE_COMPARISON_20260728.md)  
> 方法：Ventus RTL 代码审阅 + NVIDIA 官方文档 + NVIDIA 专利实施例 +
> 现有精确参数周期数据；本分析未新增付费 GPU 运行

## 结论

Ventus 并不是只在 ISA 表面模仿 CUDA TMA。RTL 中已经能看到与 NVIDIA
公开 TMA 数据通路高度相似的骨架：

1. 一个 lane 发射大块异步传输，warp 不逐元素搬运；
2. 每个 SM 有紧耦合的 TMA/DMA 数据通路；
3. tensor map 经过小型 descriptor cache；
4. setup/iterator 把 1D–5D tensor 请求拆成至多一个 L2 cache line 的请求；
5. 返回数据经过布局变换后写 shared memory；
6. G2S 用 mbarrier、S2G 用 group commit/wait 完成记账。

因此，现有 RTL 对 CUDA 的复刻首先是**协议、地址生成和完成语义的复刻**。
它还不是 GH100 TMAU、L2、片上互连和 HBM 子系统的周期级缩小版。当前最
重要的潜在差异依次是：

- **队列和并行深度很浅。**Ventus 明确只有 2 个 command、8 个 window、
  4 个 global request、4 个 shared request 和 16 个 S2G write ack；
  CUDA 的对应容量没有公开，但专利公开了内部请求队列、descriptor
  cache、setup 和 request generator 可重叠的结构。现有 Ventus
  `outstanding > 1` 超时也说明并发控制仍是主要风险。
- **Ventus tensor setup 偏串行。**它用状态机和复用乘法器逐维计算，
  再交给 window planner；CUDA 专利实施例描述的是能与 descriptor
  获取和后续请求生成重叠的流水。rank、stride 和 layout 的额外固定
  周期更容易在 Ventus 暴露。
- **TMA 不是独立的高带宽内存端口。**每个 Ventus SM 的 I-cache、
  D-cache、DMA/TMA 先经过一个固定优先级仲裁器，DMA 是低优先级；
  shared memory 端也由普通流水端口优先于 DMA。真实 H100 同样要与
  内存系统共享资源，但其具体 TMA 仲裁、端口数和队列深度未公开，不能
  假定与 Ventus 相同。
- **L2 的规模和拓扑差异远大于 TMA 指令本身。**当前 Ventus 是
  2 SM、1 cluster、单个 64 KiB L2、单个串行 SourceD 服务通路；
  测试所用 H100 SXM 是 132 SM、50 MB L2、分区 crossbar、多个 HBM3
  控制器。按 MB/MiB 口径两者的容量约差 760–800 倍，而且并行性、热点
  和公平性模型不同。
- **Ventus 当前没有真实虚拟内存代价。**`MMU_ENABLED=false`，DMA TLB
  是 2-cycle identity bypass，实际数据地址限制在 32 bit；CUDA 需要
  处理 64-bit VA、TLB、页表和可能的 page fault。当前周期因此没有覆盖
  TLB miss 或跨页压力。
- **若干 CUDA 能力尚未进入同等数据通路。**包括 L2 promotion/cache
  policy、cluster multicast/remote shared memory、im2col、数据
  prefetch、完整 TensorMap replace、压缩和 L2 residency control。

L2/DRAM 建模的核心结论是：

> `l2cache_memCycles = 32` **不等于仿真中存在 32-cycle DRAM**。
> 这个参数主要用于分配 32 个 L2 MSHR 和 put/secondary buffer；
> GVM 的 backing memory 是一个固定 `DELAY_DDR = 2`、深度 5 的响应槽，
> 实际数据由 Verilator 直接读写 host 侧平坦 `pmem`。

所以不能给现有 GVM 周期乘一个统一系数来“换算成 H100”。失真是双向的：

- 冷 miss、随机访问、跨页、读写切换和高 HBM 排队时，GVM 会明显
  **过于乐观**；
- 工作集超过 Ventus 64 KiB、但仍远小于 H100 50 MB 时，GVM 会因过早
  eviction 和单 L2 争用而**过于悲观**；
- 单 CTA、同一小缓冲区 warmup 后的测试，DRAM 模型影响较小，主要反映
  TMA 控制、L2 hit、shared memory 和同步路径；
- 多 CTA 或真实流式工作负载中，两种误差会叠加且方向不确定。

## 证据等级

本文用以下等级约束结论边界：

- **[RTL]**：可以从当前仓库 Chisel/C++ 实现直接确认；
- **[NVIDIA-DOC]**：NVIDIA 官方 CUDA/Hopper 文档明确公开；
- **[PATENT]**：NVIDIA 专利中的一个或多个实施例，不等同于确认 GH100
  流片细节；
- **[MEASURED]**：本项目已有 H100/B200/GVM raw cycle；
- **[INFERENCE]**：由上述证据联合推断，仍需要 PMU 或专门实验验证。

专利只用于说明 NVIDIA 公开过怎样的候选数据通路，不把“at least one
embodiment”改写为“GH100 一定如此实现”。CUDA 未公开的 TMA queue
深度、端口数、仲裁策略和内部流水延迟都保留为未知。

## 1. 两边确实采用了相同的 TMA 设计思想

### 1.1 CUDA 公开的数据通路

NVIDIA Hopper Tuning Guide 明确说明，TMA 可在 global/shared memory
间双向搬运 1D–5D tensor，也可访问同 cluster 内其他 SM 的 shared
memory；一个线程就能发射大搬运，数据移动不占用普通寄存器和逐元素 SM
指令。[NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#tensor-memory-accelerator)

PTX 把 TMA 定义在 async proxy 中：G2S bulk/tensor 路径使用 mbarrier
完成，S2G 路径使用 bulk async-group 的 commit/wait；tensor map 是
opaque descriptor，还提供 L2 prefetch 和 cache hint。
[PTX ISA 8.2](https://docs.nvidia.com/cuda/archive/12.2.2/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk)
[CUDA 异步代理内存模型](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#async-thread-and-async-proxy)

NVIDIA 专利 US20240176663A1 的实施例进一步给出：

- 每个 SM 紧耦合一个 TMAU，但专利明确说不限于一对一；
- SM 发出一个 block/tensor 请求，TMAU 将其拆成不超过一条 L2 cache
  line 的多条请求；
- 内部包含 request queue、descriptor cache、setup block、request
  generator、OOB/layout 处理和完成计数；
- descriptor miss 可向 general cache controller 发起预取，并在实施例
  中与其他处理并行以隐藏延迟。

参见
[NVIDIA US20240176663A1，TMAU/SM 与数据通路实施例](https://patents.google.com/patent/US20240176663A1/en#p=22)。

### 1.2 Ventus RTL 中对应的模块

Ventus 的对应关系不是概念猜测，而是 RTL 中的明确结构：

| CUDA 公开概念 | Ventus RTL 对应实现 | 状态 |
|---|---|---|
| bulk/tensor，G2S/S2G | `TmaV2Spec` 五类非 Reduce funct | 已实现 |
| tensor map | 固定 128 B、128 B 对齐 descriptor | 已实现 |
| rank 1–5 | decoder + 5D iterator | 已实现 |
| descriptor cache | 2-entry `TmaV2DescriptorService` | 已实现，容量很小 |
| setup/address generation | `TmaV2SetupMath` + layout mapper + iterator | 已实现 |
| cache-line 请求拆分 | 128 B line/window assembler | 已实现 |
| async window/ROB | 2 command + 8 window + request table | 已实现 |
| G2S mbarrier | `TmaV2MbarrierController` | 已实现 |
| S2G commit/wait group | `TmaV2S2GGroupTracker` | 已实现 |
| shared layout/swizzle/OOB | transform pipeline | 常用子集已实现 |

主要代码证据：

- [`TmaV2Spec.scala`](../gpgpu/ventus/src/pipeline/TmaV2Spec.scala#L7)
  定义 opcode、方向、descriptor、rank、16 B atom 和 128 B line；
- [`TmaV2WindowPipeline.scala`](../gpgpu/ventus/src/pipeline/TmaV2WindowPipeline.scala#L341)
  将 8-entry window ROB 与 4-entry cache request table 分离；
- [`TmaV2DmaCore.scala`](../gpgpu/ventus/src/pipeline/TmaV2DmaCore.scala#L195)
  串接 decoder、setup、window planner 和 subsystem；
- [`pipe.scala`](../gpgpu/ventus/src/pipeline/pipe.scala#L104)
  实例化 S2G group tracker 和 mbarrier controller。

**判断：[RTL]+[PATENT]。**两边的高层拆分非常接近，说明 Ventus TMA
功能复刻选择了正确的微架构方向；差异主要落在每一级资源数量、流水重叠、
互连和存储后端，而不只是 opcode 编码。

## 2. 可以从 RTL 看出的主要微架构差异

### 2.1 并发窗口：Ventus 是明确的小规模实现

Ventus 的 v2 参数固定为：

| 资源 | 数量 |
|---|---:|
| command entries | 2 |
| window ROB entries | 8 |
| global request entries | 4 |
| shared-ready entries | 4 |
| S2G write-ack entries | 16 |
| translations per command | 4 |
| descriptor entries | 2 |

这些常量位于
[`TmaV2Spec.scala`](../gpgpu/ventus/src/pipeline/TmaV2Spec.scala#L49)，
并在
[`TmaV2WindowEngine`](../gpgpu/ventus/src/pipeline/TmaV2WindowPipeline.scala#L349)
中用 `require` 固定为 2/8/4/4，而不是仅仅默认值。

CUDA 对应深度没有公开，不能杜撰数值。不过有两条强证据表明实际设计不应
直接按 Ventus 这些数字理解：

1. NVIDIA 专利实施例有内部请求队列、descriptor prefetch、setup 和
   request generation 等可重叠阶段；
2. 2026-07-30 的 H100 严格 serial/batched 对照使用同一 CTA、同一
   TensorMap 和不同 global/shared tile。N=1 两边均为 454 cycle；
   N=2/4/8 的 serial 对 batched 为 898/552、1781/640、3547/991，
   证明至少这一路径上的多条 Tensor TMA 搬运可以显著重叠。Ventus 旧
   精确 GVM case 在 N>1 时未在短预算内完成，V3.1/V3.2 则明确限制为
   单 active data command。

**判断：[RTL]+[MEASURED]+[INFERENCE]。**Ventus 当前最大的差异不是单
请求能否正确搬运，而是持续接收和退休多个命令的能力。CUDA 更深或更有效
的排队/重叠不再只是从 batch 摊销推断：上述受控 G2S 场景已经实测确认
执行重叠；不同 TensorMap、S2G、跨 CTA 的能力和具体深度仍未知，公开
PTX 语义也没有给出实现容量。

后续固定资源的 N=1–48 issue/complete 双计时进一步收紧了这里的“重叠”
含义：N=12–32 时 batched issue 和 completion 的边际间隔分别约
55.25 和 146.25 cycles/4 KiB，而 serial 约 477.38 cycles/4 KiB；
N=48 时在 2730 cycles 已发完，全部完成于 7348 cycles。它证明多个操作
处于在途状态并共享流水吞吐，但不证明有 N 套完整数据通路。约 3.1–3.2x
的饱和加速更符合小队列加单个/少数流水数据通道。N=32→33 的一次性台阶
不足以单独推定 queue depth=32。

### 2.2 tensor setup：共享乘法器状态机与流水化实现的差异

Ventus `TmaV2SetupMath` 是显式状态机，复用 `mulA * mulB` 完成 slice
stride、各维 origin、dense bytes 和 logical bytes 等运算；decoder
注释也说明 interleave slice product 留给 sequential setup，以避免再
放一组宽乘法器。
参见
[`TmaV2Frontend.scala`](../gpgpu/ventus/src/pipeline/TmaV2Frontend.scala#L322)
和
[`地址溢出/复用乘法器说明`](../gpgpu/ventus/src/pipeline/TmaV2Frontend.scala#L219)。

NVIDIA 专利实施例的 setup block 同样收集和计算参数，但还描述了
descriptor miss/prefetch 与请求处理的并行机会。专利没有给每维周期，
所以不能断言 GH100 使用几组乘法器。

**判断：[RTL]+[PATENT]+[INFERENCE]。**Ventus 选择面积优先的串行 setup
很合理，但更容易让 rank、stride、interleave 和复杂 layout 直接增加
可见延迟；CUDA 更可能用面积换重叠，但精确结构未知。

### 2.3 descriptor cache：2-entry、单 miss 全局阻塞

Ventus descriptor service：

- 只有 2 entries；
- 两个 lookup client 用 RR arbiter；
- 全服务只有一个 `missBusy`；
- miss outstanding 时其他 descriptor 请求全局停止；
- replacement 在两个 entry 间交替。

证据见
[`TmaV2Backend.scala`](../gpgpu/ventus/src/pipeline/TmaV2Backend.scala#L29)。

CUDA 专利也明确有按 descriptor global address 标记的专用 cache，但没有
公开容量或 miss 并发数。现有 CUDA descriptor working-set 从 1 到 1024
maps 没有单调恶化，而 Ventus 1024-map case 短超时，说明当前问题更像
控制进度/仿真规模限制，不能简单归因于“两项 cache 一定 thrash”。

**判断：[RTL]+[MEASURED]。**Ventus cache 容量和 miss serialization
确定不同于一个高并发目标所需的设计；CUDA 实际容量仍未知。

### 2.4 TMA 与 L1/L2/shared 的端口和优先级

每个 Ventus SM 的 I-cache、D-cache 和 DMA 是同一个
`L1Cache2L2Arbiter` 的三个输入。普通 Chisel `Arbiter` 是低 index
优先，DMA 被接在 index 2，因此持续 I/D 流量可延迟 TMA。
[`L1Cache2L2Arbiter.scala`](../gpgpu/ventus/src/L1Cache/L1Cache2L2Arbiter.scala#L28)
[`GPGPU_top.scala`](../gpgpu/ventus/src/top/GPGPU_top.scala#L934)

shared memory 端也明确标注：

- 普通 pipe 是 port 0、高优先级；
- DMA 是 port 1、低优先级；
- 32 个 bank 与 32 lanes 对应，bank conflict 会 replay。

见
[`GPGPU_top.scala`](../gpgpu/ventus/src/top/GPGPU_top.scala#L1025)
和
[`ShareMemParameters.scala`](../gpgpu/ventus/src/L1Cache/ShareMem/ShareMemParameters.scala#L30)。

NVIDIA 官方只保证 TMA 是专门异步引擎，专利实施例说 TMAU 紧耦合 SM、
能读写对应 shared/L1 并访问 global memory；没有公开 shared-memory
物理端口是否独占、L1/L2 仲裁是否固定优先。

**判断：[RTL]。**Ventus 的固定优先级是已知事实；CUDA 是否有独立端口或
怎样做 QoS 是未知，不能把专利的“dedicated TMAU”误解为“独占 L2/HBM”。

### 2.5 L2：单实例串行服务与分区 crossbar

Ventus 当前参数是：

- `num_sm=2`、`num_cluster=1`；
- `num_l2cache=1`；
- 32 sets × 16 ways × 128 B line = 64 KiB；
- I-cache、D-cache、DMA 三类 client 汇入该 L2。

见
[`parameters.scala`](../gpgpu/ventus/src/top/parameters.scala#L15)
和
[`L2 参数`](../gpgpu/ventus/src/top/parameters.scala#L117)。

L2 `SourceD` 用一个 `busy` 和八状态 FSM；`io.req.ready := !busy`，同一时刻
只服务一个进入该返回/写回数据通路的请求。
[`SourceD.scala`](../gpgpu/ventus/src/L2cache/SourceD.scala#L70)
替换 way 则直接取 16-bit LFSR 低位，而不是 LRU/PLRU。
[`Directory_test.scala`](../gpgpu/ventus/src/L2cache/Directory_test.scala#L214)

测试所用 H100 SXM 官方规格是 132 SM、50 MB L2、5 个 HBM3 stack 和
10 个 512-bit memory controller；L2 使用 partitioned crossbar，将
连接到相应 GPC 的数据局部化，并支持 residency control，L2/HBM 还支持
压缩。
[NVIDIA Hopper architecture in-depth](https://developer.nvidia.com/blog/?p=45555)

容量名义比为：

```text
H100 名义 50 MB / Ventus 64 KiB ≈ 760–800×
```

这不是“把 Ventus 扩 800 倍就等于 H100”。更重要的是一个共享服务点与
分区 L2/内存控制器在并发下的行为完全不同。

**判断：[RTL]+[NVIDIA-DOC]。**这是当前最可能主导大容量、多 CTA 和
contention 差异的结构，而不是 TMA opcode 或 tensor map 字段。

### 2.6 地址翻译：当前测试实际上绕过 MMU

顶层明确设置 `MMU_ENABLED=false`。DMA TLB 只是一个 2-cycle
identity-mapping bypass，TMA descriptor/data interface 是 32-bit 地址；
decoder 对 global base high 非零或最大地址超过 `0xffffffff` 报 overflow。
[`parameters.scala`](../gpgpu/ventus/src/top/parameters.scala#L29)
[`GPGPU_top.scala`](../gpgpu/ventus/src/top/GPGPU_top.scala#L1007)
[`TmaV2Frontend.scala`](../gpgpu/ventus/src/pipeline/TmaV2Frontend.scala#L244)

NVIDIA 专利的通用 GPU 实施例包含 MMU、TLB、memory partition 和 page
fault 路径，但这些仍是实施例；CUDA 64-bit virtual address 行为本身则是
公开编程模型的一部分。

**判断：[RTL]+[PATENT]。**现有 Ventus 周期不包含真实 TLB hit/miss、
page walk、跨页合并和 fault 行为，因此地址不规则或大 working-set 的
CUDA/Ventus 差距会被低估。

### 2.7 已知功能范围差异

Ventus decoder 当前会拒绝：

- `l2Promotion != 0`；
- 非 tiled access mode，即尚无 im2col 同等路径；
- 非 16 B swizzle atomicity；
- 部分 element stride、sub-byte/layout 组合；
- 32-bit 范围以外的 global address。

证据见
[`TmaV2Frontend.scala`](../gpgpu/ventus/src/pipeline/TmaV2Frontend.scala#L266)。

仓库有 descriptor prefetch/invalidate，但没有看到 CUDA PTX 所公开的
TMA data-to-L2 prefetch、cluster multicast/remote shared-memory、
L2 residency/cache policy、完整 TensorMap replace 或 HBM/L2 压缩等效
实现。interleave32 也尚未进入本项目正式闭环列表。

这些差异不会使已经通过的同参数 case 无效，但说明“常用子集功能相等”
不能外推为“完整 TMA ISA 和缓存行为相等”。

## 3. L2 后面不是 DRAM：当前 GVM 实际建模了什么

### 3.1 `memCycles=32` 只在配置资源，不产生 32-cycle 延迟

`InclusiveCacheMicroParameters.memCycles` 的注释将它描述为 memory
round-trip cycles，但实现中它用于：

- `mshrs = ceil(memCycles / blockBeats)`；
- `secondary = max(mshrs, memCycles - mshrs)`；
- `putLists/putBeats` 的容量配置。

当前 128 B cache line 也是单 beat，所以 `memCycles=32` 得到 32 个
outstanding MSHR。
[`L2cache/Parameters.scala`](../gpgpu/ventus/src/L2cache/Parameters.scala#L114)
[`MSHR 计算`](../gpgpu/ventus/src/L2cache/Parameters.scala#L551)

代码中没有由这个参数驱动的 32-cycle countdown。换言之，它在假设较长
memory latency 时预留 bandwidth-delay-product 资源，却没有让 backing
memory 真的呈现该延迟。

### 3.2 实际 backing memory 是固定延迟平坦内存

`Mem_SimWrapper` 的真实行为是：

- `DELAY_DDR=2`；
- 最多 5 个 response slots；
- request fire 时立即产生 host memory read/write enable；
- 读数据当时就捕获到 response slot；
- 只对 response 做固定 countdown；
- 多个到期 response 用 priority encoder 选一个返回。

见
[`Mem_SimWrapper.scala`](../gpgpu/ventus/src/top/Mem_SimWrapper.scala#L22)。
L2 到 wrapper 的 request/response 两边各有 2-stage `DecoupledPipe`：
[`GPGPU_SimWrapper.scala`](../gpgpu/ventus/src/top/GPGPU_SimWrapper.scala#L105)。

Verilator 在读使能时直接调用 `pmem->read`，写使能时直接
`pmem->write`：
[`ventus_rtlsim_impl.cpp`](../gpgpu/sim-verilator/ventus_rtlsim_impl.cpp#L256)。

因此，当前外存模型不包含：

- HBM stack/channel/bank/row；
- row-buffer hit/miss；
- 地址到 channel/bank 的映射；
- read/write queue 和 turnaround；
- FR-FCFS 或其他 memory scheduler；
- refresh、ECC、压缩；
- 可变返回延迟、尾延迟；
- L2/NoC/HBM 不同频率域；
- 多 partition 的带宽和热点。

`2-cycle wrapper + 2-stage outbound + 2-stage inbound` 可视为约 6 个配置
级传输阶段，但由于 valid/ready、countdown 和 L2 FSM 的相位关系，不能
把它当成精确的可观察 miss latency。确定无疑的是：它远不是 DRAM timing
model。

### 3.3 写路径还有一个时序抽象

flat `pmem` 在 request 被 wrapper 接受时就执行写入，ack 在固定延迟后才
返回。L2 协议通常会挡住软件过早观察该写，但这仍没有模拟真实 HBM write
queue、合并、写排空和 read-after-write 转发。因此 S2G 在写流压力和
G2S/S2G 混合时的周期只能解释为 Ventus RTL 协议周期，不能解释为真实
DRAM 写完成时间。

## 4. 建模失真到底有多大

### 4.1 没有一个全局百分比

可用下面的分解理解：

```text
C_total
  = C_issue/setup
  + C_TMA_queue/address/layout
  + C_L2_hit/shared/completion
  + N_miss × (L_outer + L_DRAM)
  + C_queue/contention
```

当前 GVM 的问题不是所有项都按同一比例变小：

- 前三项是实际 RTL 周期，正是本项目希望验证的部分；
- `N_miss` 没有随 benchmark 输出，无法知道；
- `L_DRAM` 被固定小延迟替代；
- queue/contention 又可能因单 L2、固定优先级而比 H100 更严重。

所以合理的校准式只能是：

```text
C_corrected ≈ C_GVM
            + N_miss × (L_real_miss - L_model_miss)
            + (Q_real - Q_model)
```

在没有 L2 hit/miss、outer request latency 和 queue stall 计数时，
`N_miss`、`L_model_miss` 和两个 Q 都未知，不能算出可信的单值。

### 4.2 可用于判断数量级的外部实测

一项 H100 PCIe pointer-chase 研究测得：

- GH100 L2 hit 约 273 cycle；
- 31–45 MB 压力下约 508 cycle；
- 超出 L2 后 global memory 约 658.7 cycle。

参见
[Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks](https://arxiv.org/html/2507.10789v2#S6.SS3)。

它使用 H100 PCIe/HBM2e，不是本项目 H100 SXM/HBM3；pointer chase 也
不是 TMA。因此这些数字**不能直接加到本项目每条 TMA 上**。它们只说明
真实芯片中 L2 hit 到 off-chip memory 的增量可达约 386 cycle，而 GVM
backing wrapper 的独立固定延迟参数只有 2 cycle。

如果只做敏感性边界：

```text
386 / 2  ≈ 193×
386 / 6  ≈ 64×
```

这两个 64–193× 不是总体 TMA cycle 误差，也不是修正系数；它们只表示
“缺失的外存延迟来源”相对于 GVM 固定延迟配置的数量级。请求并行后，
许多 miss 会重叠，总体差异不会简单乘 `N_miss`。

### 4.3 不同工作区间的失真方向

| 场景 | 当前 GVM 方向 | 原因 | 当前数据可信内容 |
|---|---|---|---|
| 单 CTA、小 buffer、先 warmup | DRAM 影响小 | 大概率 L2 hit | TMA 固定开销、layout/shared/completion |
| cold cache、随机地址 | 严重乐观 | 固定 2-cycle backing，无 bank/row/TLB | 只能验证功能 |
| working set 64 KiB–50 MB | 容量上偏悲观 | Ventus 已溢出，H100 仍可驻留 | Ventus 小 L2 压力，不代表 H100 |
| 多 CTA/多 SM | 方向不定 | GVM 单 L2 更拥塞，但 DRAM 又过快 | 只能观察本 RTL 进度/争用 |
| stride/irregular/channel hotspot | 过于平滑 | 无地址到 HBM bank/channel 映射 | 地址生成正确性 |
| S2G 长写流或读写混合 | 多半乐观且形状失真 | 无 write queue/turnaround | TMA ack/group 协议 |
| 跨页或大地址 | 严重乐观/未覆盖 | MMU disabled、32-bit address | 不可外推 |

## 5. 现有周期已经暴露出的 RTL 特征

### 5.1 Ventus 大容量斜率是精确线性的

现有 capacity 数据：

| path | 8 KiB | 16 KiB | 32 KiB | 8→16 KiB | 16→32 KiB |
|---|---:|---:|---:|---:|---:|
| Ventus G2S | 341 | 565 | 1013 | +224 | +448 |
| Ventus S2G | 315 | 523 | 939 | +208 | +416 |

按 128 B line 换算：

```text
G2S: 224 / 64 = 448 / 128 = 3.50 cycle/line
S2G: 208 / 64 = 416 / 128 = 3.25 cycle/line
```

这组恰好倍增的斜率与 RTL 中固定 128 B line、有限 request entries、
单 SourceD FSM、固定 shared/L2 仲裁高度一致。它更像内部串行数据通路
吞吐，而不像有 bank、row、refresh 和 queue 波动的真实 HBM。

**判断：[MEASURED]+[RTL]+[INFERENCE]。**3.25/3.50 cycle/line 是当前
Ventus 配置的稳定综合吞吐成本，不能叫作 DRAM latency。

### 5.2 32 KiB roundtrip crossover 很可能是 64 KiB L2 边界

实测：

- 16 KiB roundtrip：H100 1209，Ventus 1071；
- 32 KiB roundtrip：H100 1869，Ventus 1935。

32 KiB roundtrip 同时触及约 32 KiB source 和 32 KiB destination，
已经用满 Ventus 名义 64 KiB L2；再加 descriptor、I-cache、D-cache、
对齐和 replacement 冲突，实际可用容量更小。H100 的 50 MB L2 则没有
这一容量问题。

这很好地解释了 Ventus 为什么在 16 KiB 仍低于 H100、到 32 KiB 却
反超 66 cycle，但目前 benchmark 没有 L2 miss counter，因此只列为
**强推断**，不是已证实归因。

### 5.3 H100 多 CTA 压力不能用 Ventus 单 L2 外推

H100 32 KiB G2S 从 1 CTA 的 756 cycle 增到 528 CTA 的 4030 cycle，
工作集约 16.5 MiB，仍低于 50 MB 名义 L2。该 5.331× 增长主要反映
TMA、shared/L2 partition、crossbar、SM 调度和内存队列的综合并发压力，
不能简单解释为 L2 容量溢出。

Ventus contention suite 中大量 case 在 GVM 短预算内未完成；即使以后
得到周期，其 2 SM + 单 L2 拓扑也只能说明本 RTL 的公平性和前进性，
不能数值预测 132-SM H100 的扩展曲线。

## 6. 对现有对比报告应怎样解读

现有同参数结果仍然有价值，但要分层使用：

1. **可以高置信比较**
   - bulk 与 tensor 的额外控制开销；
   - rank/stride/swizzle/OOB 对各自流水的周期影响；
   - 小/大容量的内部 line throughput；
   - barrier/group 完成协议；
   - Ventus 固定资源满载时的 stall 和进度风险。
2. **只能定性比较**
   - warm L2 下的容量拐点；
   - 单 CTA 下 L2/shared 综合延迟；
   - 不同平台的 contention 形状。
3. **当前不能比较**
   - cold HBM latency；
   - HBM sustained bandwidth；
   - channel/bank/row locality；
   - TLB/page-walk；
   - 132-SM 扩展；
   - 绝对纳秒时间和能效。

尤其不能因为 Ventus 小包 G2S cycle 明显低于 H100，就得出“未来 Ventus
流片 TMA 比 H100 快”。H100 的 `clock64()` 和 Ventus `mcycle` 没有共同
物理频率，H100 计时还包含真实 L2/async proxy/完成路径，而 GVM 外存被
理想化。

## 7. 最小成本的校准方案

不需要先引入完整 DRAM 模型，建议按以下顺序收集可解释证据。

### 7.1 第一优先级：把 miss 和 stall 可观测化

为每个 case 输出：

- L2 hit/miss/writeback 数；
- outer A request fire 到 D response fire 的 latency；
- `SourceD.busy` 周期；
- I/D/DMA L2 arbiter stall；
- TMA `robFull/requestFull/sharedFull`；
- descriptor hit/miss；
- shared bank-conflict replay。

TMA window engine 已经有多项 PMU 输出，新增 L2/arbiter 计数的 RTL 成本
低于直接接 DRAMSim。

### 7.2 第二优先级：做固定延迟 sweep

将 `DELAY_DDR` 参数化，用同一组小矩阵跑：

```text
2, 16, 32, 64, 128, 256, 512 cycle
```

每个进程继续使用短墙钟超时。若某 case 的 cycle 对 `DELAY_DDR` 不敏感，
它主要是 warm-hit/control 路径；若斜率接近 miss count，就能反推出外存
敏感度。该方法仍没有 DRAM bank 行为，但能先把 TMA RTL 与 memory
latency 解耦。

### 7.3 第三优先级：显式区分 warm、cold 和容量

增加三类同参数 case：

- warm：同一地址重复；
- cold：每次 L2 invalidate 后访问；
- streaming：轮转超过 64 KiB 的不复用地址；
- conflict：固定同 set 与均匀 spread。

同时报告 hit/miss，不再用 warmup 猜 cache 状态。

### 7.4 第四优先级：需要系统性能预测时再接 DRAM timing model

若目标从“验证 TMA RTL”升级为“预测流片性能”，再把 outer memory
接口接到 Ramulator 2.0、DRAMsim3 或等价 HBM timing model，并至少建模：

- 多 channel/bank 和地址映射；
- read/write queues、turnaround；
- bank timing、refresh；
- 有限带宽和 backpressure；
- 独立 memory clock；
- 多 L2 partition/NoC 注入。

即使接了 DRAM model，也仍需先决定目标 Ventus 芯片的 SM 数、L2 容量/
分区、时钟和内存规格，否则只是“一个假想配置”的性能，而不是可与 H100
相除的真实比值。

## 8. 设计建议

从“继续复刻 CUDA 功能”转向“缩小 CUDA 行为差异”时，优先级建议如下：

1. **先修 outstanding/progress，再加 ISA 边角功能。**2-command、
   4-request 的正确 backpressure 和公平退休比继续增加 descriptor 字段
   更影响实际可用性。
2. **将 DMA 的固定最低优先级改为可证明无饥饿的仲裁。**可使用 round
   robin、age 或 credit/QoS，并为 I/D/TMA 分别计数等待周期。
3. **拆分 L2 SourceD 的单 busy 服务瓶颈。**至少允许 hit-return、
   miss refill、write ack 分离，或用多个 bank/response queue。
4. **descriptor cache 支持多 miss 或 hit-under-miss。**当前一个
   `missBusy` 会把 descriptor working-set 行为放大成全局串行。
5. **保留面积可配的 setup。**当前复用乘法器适合小核，但应允许
   `setup_lanes` 或多级 pipeline 参数，以评估面积/周期拐点。
6. **在所有性能声明中分开 `TMA-core cycle` 和 `memory-system cycle`。**
   前者可由 RTL 准确回答，后者必须绑定明确的 L2/NoC/DRAM 配置。
7. **MMU 和 64-bit 地址要作为独立里程碑。**否则大 working-set、
   unified memory 和跨页 tensor 与 CUDA 的差异无法被现有小 case 暴露。

## 9. 最终判断

Ventus TMA 的设计骨架与 NVIDIA 公开 TMAU 实施例相当接近：两者都把
descriptor/setup、cache-line request generation、shared layout
transformation 和 async completion 从 warp 普通数据搬运中剥离。这说明
当前设计不是“只做了指令名相同”的浅层复刻。

主要差异是**实现尺度和系统集成**：

```text
CUDA/H100:
  深度未知但高度并行的 TMAU
  → 分区 L2/crossbar
  → 多 HBM controller + 真实 DRAM timing

Ventus 当前 RTL:
  2 cmd / 8 window / 4 request
  → I/D/TMA 固定优先级共享端口
  → 单 64 KiB L2 + 单 SourceD
  → 固定 2-cycle flat backing memory
```

因此，当前周期报告最能回答：

> “Ventus 的 TMA 功能流水在自己的 RTL 配置下需要多少周期，哪些参数会
> 增加控制/布局/共享内存代价？”

它还不能直接回答：

> “采用真实 L2、NoC 和 HBM 的 Ventus 芯片，相对 H100 的绝对延迟和
> 带宽是多少？”

要回答后一个问题，最低必要条件是 L2 miss/stall PMU、cold/warm 分离和
memory-latency sweep；要做流片级预测，还需要目标 L2/NoC/HBM 参数化模型。

## 资料

### 项目内 RTL

- [顶层容量、SM/L2/TMA 参数](../gpgpu/ventus/src/top/parameters.scala)
- [GVM 固定延迟 memory wrapper](../gpgpu/ventus/src/top/Mem_SimWrapper.scala)
- [GVM 外存接口 pipeline](../gpgpu/ventus/src/top/GPGPU_SimWrapper.scala)
- [Verilator flat physical memory](../gpgpu/sim-verilator/ventus_rtlsim_impl.cpp)
- [L2 参数与 MSHR 配置](../gpgpu/ventus/src/L2cache/Parameters.scala)
- [L2 单 SourceD FSM](../gpgpu/ventus/src/L2cache/SourceD.scala)
- [L1/DMA 到 L2 仲裁](../gpgpu/ventus/src/L1Cache/L1Cache2L2Arbiter.scala)
- [TMA v2 规格和资源常量](../gpgpu/ventus/src/pipeline/TmaV2Spec.scala)
- [TMA descriptor service](../gpgpu/ventus/src/pipeline/TmaV2Backend.scala)
- [TMA decoder/setup/iterator](../gpgpu/ventus/src/pipeline/TmaV2Frontend.scala)
- [TMA window ROB/request engine](../gpgpu/ventus/src/pipeline/TmaV2WindowPipeline.scala)

### 外部一手资料和研究

- [NVIDIA Hopper Tuning Guide：TMA](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#tensor-memory-accelerator)
- [NVIDIA PTX ISA 8.2：bulk/tensor async copy](https://docs.nvidia.com/cuda/archive/12.2.2/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk)
- [NVIDIA CUDA Programming Guide：async proxy](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#async-thread-and-async-proxy)
- [NVIDIA Hopper architecture in-depth：H100 L2/HBM/SM](https://developer.nvidia.com/blog/?p=45555)
- [NVIDIA US20240176663A1：Tensor map cache storage/TMAU 实施例](https://patents.google.com/patent/US20240176663A1/en)
- [Jarmusch et al.：H100 PCIe/Blackwell memory microbenchmarks](https://arxiv.org/abs/2507.10789)
