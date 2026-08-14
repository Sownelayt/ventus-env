# GPIDL TMA 与 Ventus 本地 TMA V2 的差异及指令修正规模评估

> 结论先行：两套 TMA **语义相近但 ISA/ABI 不兼容**。如果目标是让 GPIDL 描述本地已经实现的 Ventus TMA V2，不能只改助记符或 opcode；至少要重写 TMA 指令族、完成模型和 TensorMap 契约。只改 GPIDL 规范与文档属于中等修改；做到编码、汇编/编译、模拟器和 RTL 测试全链路可用，属于大修改。反过来让本地 RTL 完整实现当前 GPIDL TMA，则接近一次 TMA 子系统重构。

## 1. 比较范围与版本基线

### 1.1 GPIDL 基线

- 仓库：`/home/liyb/gpidl`
- 已执行 fetch，比较默认远端分支 `origin/mmy`
- 最新提交：`9bbbc0bb3d41146a5ef80ca8510cd6327c049bf8`
- 提交时间：**2026-08-06 23:00:08 +08:00**
- 提交说明：`Merge pull request #10 from reoLantern/gaoxb`
- 当前本地 checkout 比该提交落后 9 个提交，但本文对 GPIDL 指令编码长度和 opcode 的判断取自最新的 `origin/mmy`。当前 checkout 与最新提交的 TMA 指令正文没有实质差异；最近提交主要重新求解/生成了全局编码，因此部分 opcode 有变化。

GPIDL 的主要依据：

- [`spec/gpidl/spec.jsonc`](../../gpidl/spec/gpidl/spec.jsonc)：指令与 modifier 的 single source of truth
- [`09_tensor_dataflow.md`](../../gpidl/spec/gpidl/docs/arch/09_tensor_dataflow.md)：张量数据流
- [`08_synchronization.md`](../../gpidl/spec/gpidl/docs/arch/08_synchronization.md)：依赖计数器、mbarrier 与内存一致性
- [`NVIDIA异步执行与同步机制调研.md`](../../gpidl/spec/gpidl/docs/NVIDIA异步执行与同步机制调研.md)：bulk group、UTMACMDFLUSH 等设计依据
- [`encoding/README.md`](../../gpidl/encoding/README.md)：64/96/128 位可变长编码流程

### 1.2 本地 Ventus TMA V2 基线

- 仓库：`/home/liyb/ventus-env/gpgpu`
- 分支：`develop_dma_area_reduction`
- HEAD：`eb684067061891121887932a7de8f4ad5015b90a`
- HEAD 时间：2026-07-14 21:14:34 +08:00
- 重要说明：最新 TMA V2 是当前工作树中的未提交实现，HEAD 本身不能完整代表本文所说的“最新本地设计”。本文以 **2026-08-12 工作树快照**为准。

本地设计的主要依据：

- [`TmaV2Spec.scala`](../gpgpu/ventus/src/pipeline/TmaV2Spec.scala)：架构常量和资源上限
- [`Instructions.scala`](../gpgpu/ventus/src/pipeline/Instructions.scala)：32 位指令匹配
- [`DecodeUnit.scala`](../gpgpu/ventus/src/pipeline/DecodeUnit.scala)：寄存器字段解释
- [`TmaV2DmaCore.scala`](../gpgpu/ventus/src/pipeline/TmaV2DmaCore.scala)：命令接收、校验、描述符查找与完成绑定
- [`TmaV2Frontend.scala`](../gpgpu/ventus/src/pipeline/TmaV2Frontend.scala)：TensorMap 编译、坐标绑定与合法性检查
- [`TmaV2Completion.scala`](../gpgpu/ventus/src/pipeline/TmaV2Completion.scala)：S2G group 与 mbarrier
- [`warp_schedule.scala`](../gpgpu/ventus/src/pipeline/warp_schedule.scala)：group wait、mbarrier wait 和 proxy fence 阻塞
- [`ventus_tma_v2_spec.h`](../testcases/_get_case/common/ventus_tma_v2_spec.h)：C/C++ 侧 ABI 常量
- [`ventus_tma_v2_opencl.h`](../testcases/_get_case/common/ventus_tma_v2_opencl.h)：当前 OpenCL raw instruction wrapper

为避免后续工作树变化造成误解，本次快照的关键文件 SHA-256 为：

| 文件 | SHA-256 |
|---|---|
| `TmaV2Spec.scala` | `c332e33cb3fbead3ef41a0c9234e3a2c17911a4e9f0375e1cb3ea3446edf1719` |
| `ventus_tma_v2_spec.h` | `8f7a2e237a6cce42ffa92a8a7f0a8ccb6960a633e3541104898e7fee317e7464` |
| `ventus_tma_v2_opencl.h` | `488f051391248fb39cb7c5f9df974f6b4324ba51aed5c7f15732cb5c4a536380` |

## 2. 修改规模结论

“修正指令”有三种不同口径，工作量差别很大：

| 目标 | 修改规模 | 预计人工修改 | 粗略工作量 | 结论 |
|---|---:|---:|---:|---|
| 只修 GPIDL 的 TMA 语义和文档，使其描述本地 V2 能力 | 中 | `spec.jsonc` 约 500–700 行，加 2–4 篇架构文档；生成文档另算 | 3–7 人日 | 可以较快完成，但仍不能直接被本地硬件执行 |
| 增加 Ventus TMA V2 target/profile，打通固定 32 位编码、汇编/反汇编、模拟器和测试 | 大 | 约 10–20 个源文件，生成物可能数千行变化 | 3–6 人周 | **推荐目标**；不要求重写本地 TMA datapath |
| 把本地 RTL 改成完整执行当前 GPIDL TMA | 特大 | 前端解码、寄存器模型、地址、cluster/DSM、模式、同步与错误处理均需重做 | 至少 8–16 人周，且可能更高 | 不建议称为“修指令”，本质是重新定 ISA 并扩建硬件 |

以上估算假定：

1. 以本地已实现的功能集合为边界，不补做 GPIDL 独有的 cluster、S2S、im2col、gather/scatter、数据预取和 64 位地址能力。
2. 保留当前本地 TMA V2 datapath，主要修改 GPIDL、工具链和 ABI 接口。
3. 至少完成 Chisel 单元测试和 OpenCL 端到端测试；若还要求正式编译器调度、性能模型和标准 PDF 发布，取估算上限。

## 3. 两套设计的核心定位

### 3.1 GPIDL TMA

GPIDL 是一套独立 GPU ISA。它的指令有公共控制字段，使用 `sreg`、`vreg`、predicate、`wr_sb`/`rd_sb` 等 GPIDL 架构资源，并由编码求解器分配到 **64/96/128 位可变长容器**。

当前 TMA 数据与控制族有 9 个 mnemonic、10 个 leaf：

- `utmaldg`：descriptor 驱动的 tensor G2S
- `utmastg`：descriptor 驱动的 tensor S2G copy
- `utmaredg`：descriptor 驱动的 tensor S2G reduce
- `utmapf`：descriptor 驱动的 tensor data G2L2 prefetch；它是可被硬件忽略的性能 hint
- `ublkcp`：linear bulk copy，G2S/S2G/S2S
- `ublkpf`：linear data G2L2 prefetch；同样是可被硬件忽略的性能 hint
- `ublkred`：linear bulk reduce，S2G/S2S
- `utmacctl`：按地址或全局失效 descriptor cache
- `utmacmdflush`：Hopper/Blackwell SASS 中 bulk async-group commit 的机器指令名称；GPIDL 当前把它另行解释成独立的 command-dispatch flush，需要修正

与它们配套的还有抽象 `bulk_commit`、通用 `depbar`、`mbar_init`、`mbar_arrive`、`mbar_wait`、`mbar_inval` 和通用 `fence`。其中 `utmacmdflush` 与 `bulk_commit` 不应作为两个独立的架构动作。

最新生成编码中，上述 leaf 除 `ublkred` 为 96 位外，其余均为 64 位。它们不是 RISC-V 32 位指令，也不能被当前 Ventus decoder 原样识别。

### 3.2 Ventus TMA V2

本地 TMA V2 是 Ventus/RISC-V 32 位 custom extension：

- 所有指令固定 32 位。
- opcode 固定为 `0x42`，即 `inst[6:0] = 7'b1000010`。
- `funct3 = inst[14:12]` 在 1–7 之间分配数据、TensorMap、group 和 mbarrier/proxy 操作。
- tensor 的 rank、dtype、stride、box、interleave、swizzle 和 OOB 等主要来自固定 128B TensorMap。
- G2S 通过本地 mbarrier transaction bytes 完成；S2G 通过每 warp 四组的显式 group ring 完成。
- 错误不按 GPIDL 文本所写直接 trap，而是写每 warp 的 sticky `dma_status` CSR `0x814`。

因此，两边共同点是“bulk/tensor 异步搬运 + G2S mbarrier + S2G group”，但共同点停留在功能分类，尚未形成二进制或软件 ABI 兼容。

## 4. 本地 TMA V2 指令编码

### 4.1 通用 R 型位域

本地数据类指令复用 RISC-V R 型外形：

```text
31          25 24      20 19      15 14   12 11       7 6        0
+--------------+----------+----------+-------+-----------+----------+
| funct7/opBits|   rs2    |   rs1    | funct3| rd/src3   |  0x42    |
+--------------+----------+----------+-------+-----------+----------+
```

这里最需要注意的是：对 TMA 指令，decode 把 `rd[11:7]` 当成**第三个读操作数**，而不是写回目的寄存器。`reg_idx3` 对非浮点指令来自 `rd` 字段。因此 `funct7` 不保存普通 FMA 意义上的 `rs3`，可以完整作为 reduce/control 位。

当前 OpenCL wrapper 对 tensor 和 mbarrier 指令使用固定寄存器 `x10`、`x11`、`v12` 并发射 `.word`；这是当前软件 wrapper ABI 的限制，不是 5 位编码字段本身只能表示这三个寄存器。

### 4.2 数据指令

| 建议可读名称 | `funct3` | `opBits=inst[31:27]` | 物理字段/当前语义 | 完成机制 |
|---|---:|---|---|---|
| `cp.async.bulk.g2s` | 1 | 必须为 0 | `rd=shared_dst`, `rs1=global_src`, `rs2=bytes` | 若已绑定 mbarrier，则按 transaction bytes 完成；也允许无绑定发射 |
| `cp.async.tensor.g2s` | 2 | 必须为 0 | `rd=shared_base`, `rs1=tensormap`, `rs2=coordinate vreg` | 同上，字节数来自编译后的 descriptor |
| `cp.async.bulk.s2g` | 3 | reduce/type | `rd=global_dst`, `rs1=shared_src`, `rs2=bytes` | 当前 S2G group |
| `cp.async.tensor.s2g` | 4 | reduce；`inst[28:27]` 必须为 0 | `rd=shared_base`, `rs1=tensormap`, `rs2=coordinate vreg`；global 地址来自 descriptor | 当前 S2G group |

bulk copy 的三个地址/长度都必须 16B 对齐或为 16B 整数倍，长度不能为 0。tensor descriptor 必须 128B 对齐，tensor shared base 和最终 bounding-box origin 也要满足本地实现的对齐约束。

### 4.3 Reduce 位域

S2G 数据指令使用：

```text
inst[31:29] reduce_mode
  0 copy
  1 add
  2 min
  3 max
  4 and
  5 or
  6 xor
  7 reserved

inst[28:27] bulk_reduce_type，仅 bulk S2G 使用
  0 u32
  1 s32
  2 b32
  3 reserved
```

本地实际合法组合为：

| 路径 | 合法操作和类型 |
|---|---|
| bulk S2G copy | `copy + type=0` |
| bulk S2G arithmetic | `add/min/max + u32/s32` |
| bulk S2G bitwise | `and/or/xor + b32` |
| tensor S2G reduce | descriptor dtype 仅 `u32/s32`；`add/min/max/and/or/xor` |

每个 reduce 请求最终通过 L2 atomic endpoint 发出一个完整 32 位元素的原子操作，并等待最终 AMO ack 后才记为 S2G 完成。

### 4.4 TensorMap cache control

`funct3=5`，`rs1=descriptor address`，`inst[31:27]` 为：

- 0：`prefetch.tensormap`，把 128B descriptor 取入并编译到本地 descriptor cache。
- 1：`invalidate.tensormap`，按地址失效 cache entry。

本地没有 invalidate-all 指令。注意 `prefetch.tensormap` 只预取/编译**描述符**，不是把 descriptor 描述的 tensor data 预取到 L2。

### 4.5 S2G group control

`funct3=6`，`rs1[4:0]` 被解释成 zimm：

- 16：`commit_group`
- 24：`wait_group 0`
- 25：`wait_group 1`
- 26：`wait_group 2`
- 27：`wait_group 3`

每 warp 有 4 个 group。S2G 命令 issue 时进入当前 open group；非空 commit 后关闭当前组并推进 ring。`wait_group N` 等到旧组完成，但允许最近 N 个已提交组继续在途。它是专用硬件 group tracker，不是 GPIDL 的通用 scoreboard counter。

### 4.6 mbarrier 与 proxy fence

`funct3=7`，`inst[26:25]` 为 op：

| op | 操作 | `rs1` | `rs2` |
|---:|---|---|---|
| 0 | `mbarrier.init` | 8B 对齐的 shared 地址 | expected arrivals，低 8 位且非 0 |
| 1 | `mbarrier.arrive_expect_tx` | mbarrier 地址 | expected transaction bytes |
| 2 | `mbarrier.wait` | mbarrier 地址 | old phase，使用 bit 0 |
| 3 | `fence.proxy.async.shared` | 不使用 | 不使用 |

本地每个 resident workgroup 固定 4 个 mbarrier table entry。`arrive_expect_tx` 会：

1. 增加 pending transaction bytes；
2. 将 pending arrivals 减 1；
3. 为当前 warp 创建一个隐式 binding；
4. 后续一个或多个 G2S 命令从 binding 中消费字节额度；
5. G2S 完成时递减 pending bytes；arrivals 与 bytes 均归零后 phase 翻转。

`mbarrier.wait` 是 warp 阻塞操作，不返回 predicate。barrier 最新状态的两个 32 位 word 被镜像到 shared memory，只有镜像写确认后 waiter 才能释放；软件侧仍把这个 8B 对象视为 opaque。

本地 proxy fence 进入 warp scheduler，并以 LSU 的 `fence_end` 作为释放条件。它不是 GPIDL 带 `consistency_scope` 的通用 fence 的直接二进制映射。

### 4.7 状态 CSR

`dma_status` 位于 CSR `0x814`，每 warp 保存 16 位：

```text
[7:0]   code
[15:8]  detail
```

它保存第一个非零错误，后续错误不会覆盖，软件写 0 清除。已定义 code：

| code | 含义 |
|---:|---|
| 0 | OK |
| 1 | invalid descriptor |
| 2 | unsupported feature/encoding |
| 3 | address overflow |
| 4 | mbarrier protocol error |
| 5 | invalid group operation |

GPIDL 当前 TMA 没有与此等价的 per-warp first-error sticky CSR 契约。

### 4.8 当前实现容量（不是编码位，但会影响软件契约）

本地 `TmaV2Spec` 还固定了下列默认资源：

| 资源 | 默认值 |
|---|---:|
| active command | 1 |
| data lookahead slot | 1 |
| 独立 TensorMap control queue | 1 |
| payload/window entry | 6 |
| global request entry | 40 |
| S2G write-ack entry | 32 |
| shared-ready entry | 8 |
| 每 command translation entry | 4 |
| compiled descriptor cache entry | 4 |
| 每 warp S2G group | 4 |
| 每 resident WG mbarrier entry | 4 |

这些值大多属于 microarchitecture，不应占用指令位；但 group 数、mbarrier 数、单命令 translation 生命周期以及 descriptor cache 的失效规则会被软件观察到，至少应写进 Ventus target profile。GPIDL 当前主要描述抽象语义，没有这些固定容量契约。

## 5. TensorMap 契约差异

### 5.1 本地 descriptor 是公开且固定的 ABI

本地 descriptor 固定为 128B、128B 对齐，magic 为 `0x544d4103`：

| word | 内容 |
|---:|---|
| 0 | magic |
| 1 | control：dtype `[4:0]`、rank `[7:5]`、interleave `[9:8]`、swizzle `[12:10]`、atomicity `[14:13]`、L2 promotion `[17:16]`、OOB `[18]`、access `[20:19]` |
| 2–3 | 64 位 global base 存储槽；当前 RTL 要求高 32 位为 0 |
| 4–8 | 5 个 global dimension |
| 9–16 | 4 个 64 位 global stride |
| 17–21 | 5 个 box dimension |
| 22–26 | 5 个 element stride 兼容槽 |
| 27–31 | reserved，必须为 0 |

当前实现的关键限制：

- rank 为 1–5。
- descriptor 声明 16 种 dtype 编码，copy 路径可以使用通过校验的 dtype；tensor reduce 只接受 `u32/s32`。
- global base 的高 32 位必须为 0，最终有效地址也必须落在 32 位地址空间。
- access mode 只支持 tiled。
- L2 promotion 必须为 0。
- swizzle atomicity 只支持 16B。
- `Swizzle96` 虽有枚举常量，但当前 descriptor compiler 会把它判为 unsupported；实际接受 none/32B/64B/128B，并带组合约束。
- element stride 数据通路未实现：active dimension 的兼容槽必须为 1，inactive dimension 必须为 0。
- OOB NaN 只对浮点类型合法，且带 OOB fill 的 S2G command 不合法。

### 5.2 GPIDL descriptor 有意保持 opaque

GPIDL 只规定 descriptor 为 host 创建、存放在 constant memory 的 128B tensormap，并明确不把字段布局暴露到 ISA。rank、load/store/reduce mode、multicast、cluster size 和 cache policy 的一部分由指令 modifier 再次提供。

这与本地设计形成根本区别：

- GPIDL：descriptor + 大量 instruction modifiers 共同决定行为。
- 本地 V2：descriptor 决定 rank/layout/type，指令只决定方向、地址/坐标和 reduce op。

要以本地 V2 为准修正 GPIDL，必须选择其一：

1. 把 Ventus 128B TensorMap 正式写入 target ABI，并移除重复 modifier；或
2. 保留 GPIDL opaque descriptor，只把 Ventus descriptor 当 target lowering 私有格式。

若目标是让 GPIDL 成为 Ventus 软硬件契约，建议选择第 1 种；否则 GPIDL validator 无法检查软件实际会被 RTL 拒绝的 descriptor 组合。

## 6. 逐项差异

| 维度 | GPIDL 当前设计 | Ventus 本地 TMA V2 | 修正含义 |
|---|---|---|---|
| 基础 ISA | 独立 GPU ISA，64/96/128 位可变长 | RISC-V 风格固定 32 位 custom opcode `0x42` | 不能直接复用 GPIDL 当前编码结果 |
| 寄存器模型 | 256 项 `sreg/vreg`、pair/span、公共控制字段 | 基础指令 5 位 `x/v` 索引，可配 regext；TMA 用 `rd` 作第三源 | 需要 target operand mapping |
| tensor 坐标 | 1–5 个连续 sreg，由 `.d1`–`.d5` 决定 span | 一个 coordinate vreg 的 lane 0–4，rank 来自 descriptor；当前 wrapper 固定 v12 | 操作数形态不兼容 |
| tensor operand 顺序 | 通常 `desc, coords, shm[/mbar]` | 物理字段为 `rd=shm, rs1=desc, rs2=coords` | 汇编器需重排或定义 target syntax |
| 地址宽度 | global/descriptor 常用 64 位 paired sreg | global、descriptor、TLB virtual interface 为 32 位；descriptor high word 必须 0 | 需要明确 ILP32 限制 |
| 方向 | bulk 支持 G2S/S2G/S2S；tensor G2S/S2G | 只支持 G2S/S2G | S2S/cluster 必须从 Ventus profile 移除或标 unsupported |
| cluster | multicast、cluster size、DSM 路径 | 无 multicast、无 cluster、无 S2S | 不能只忽略 modifier，否则会静默误编译 |
| tensor mode | tile、im2col、gather4、w128/w；store 有 scatter4 | 只支持 tiled；模式主要来自 descriptor | 删除/限制 modifier |
| cache/memory modifiers | cache evict、mem semantics、consistency scope | 数据指令无这些位；只有 descriptor cache control 和独立 proxy fence | 需由 target lowering 拒绝或另行展开 |
| data prefetch | `utmapf` tensor data G2L2、`ublkpf` linear data G2L2；二者都是不影响正确性的性能 hint | 无 data prefetch 指令 | 可以合法降级为 no-op，但不可映射到本地 `prefetch.tensormap` |
| descriptor cache | `utmacctl addr` 和 invalidate-all | 按地址 prefetch/compile、按地址 invalidate；无 invalidate-all | 一增一删 |
| reduce op | add/min/max/and/or/xor/inc/dec | add/min/max/and/or/xor | 删除 inc/dec |
| bulk reduce type | u32/s32/u64/s64/f16/bf16/f32/f64/b32/b64，按组合限制 | arithmetic 为 u32/s32；bitwise 为 b32 | 大幅收窄 |
| tensor reduce type | 文本提到 `tma_red_type`，但实际 leaf 没该 modifier，类型来源含糊 | 明确来自 descriptor，且只允许 u32/s32 | GPIDL 需消除内部歧义 |
| G2S mbarrier | `utmaldg` 操作数直接携带 mbar；完成强制 mbar | 先 `arrive_expect_tx` 建立 warp binding，后续 G2S 隐式消费；也允许无 binding | 软件序列和错误模型不同 |
| S2G group | `bulk_commit` 通过公共 `wr_sb/rd_sb` 记账，`depbar` 等阈值 | 4-entry/warp 专用 ring，显式 commit 和 wait 0–3 | 完成模型必须重写 |
| 完成事件 | GPIDL 允许区分源读完和目的写完两个 scoreboard 事件 | 本地 group 等最终 command/AMO ack，没有公开早期 source-read completion | 不能等价映射 `rd_sb/wr_sb` 双事件 |
| mbarrier API | init 两种 form；SIMT arrive 六种 mode并返回 token；wait/try_wait 返回 predicate；inval | uniform init、arrive_expect_tx、阻塞 wait(old phase)；无 token、try_wait、inval | 只保留严格子集，并改变 operand/result |
| mbarrier 数量 | ISA 未固定为每 WG 4 项 | 每 resident WG 固定 4 项 | 应纳入 Ventus profile 限制 |
| bulk group commit | 同时存在被描述成 command-dispatch flush 的 `utmacmdflush`，以及抽象 `bulk_commit`；但 CUDA 中前者正是后者下沉后的 SASS 名称 | `funct3=6, zimm=16` 显式 commit | 合并成一个语义操作；`utmacmdflush` 只能作为 SASS 风格 alias，不能再有独立 flush 语义 |
| fence | 通用 fence kind + scope | `fence.proxy.async.shared`，实现上等待 LSU fence end | 语义需单独定义 |
| 错误 | 多处 prose 写非法参数 trap | sticky CSR 返回 code/detail | 需要统一成 status CSR 契约 |

## 7. 指令映射建议

下表区分“可以 lowering”“只可作为受限别名”和“无等价项”：

| GPIDL | Ventus V2 映射 | 处理建议 |
|---|---|---|
| `utmaldg` | `funct3=2`, reduce=copy | 保留 source alias；删除 dim/load/multicast/cluster/cache modifier，mbar 改成前置 binding |
| `utmastg` | `funct3=4`, reduce=copy | 可直接 lowering，但坐标与 operand 顺序需 target 化 |
| `utmaredg` | `funct3=4`, reduce=1–6 | 与 store 共用编码；type 来自 descriptor，只接受 u32/s32 |
| `ublkcp.g2s` | `funct3=1` | 可 lowering；移除 counted/mem-sem/scope/cache/multicast |
| `ublkcp.s2g` | `funct3=3`, reduce=copy | 可 lowering；完成改成本地 group |
| `ublkcp.s2s` | 无 | Ventus profile 编译期拒绝 |
| `ublkred.s2g` | `funct3=3`, reduce/type bits | 仅支持本地 9 个合法 reduce 编码 |
| `ublkred.s2s` | 无 | 编译期拒绝 |
| `utmacctl addr` | `funct3=5, subop=1` | 可作为 invalidate alias |
| `utmacctl` invalidate-all | 无 | 删除或报 unsupported |
| `utmapf` | 无直接等价项 | Ventus target 降级为 no-op；它是可忽略的 tensor **data** prefetch，不能映射成 descriptor prefetch |
| `ublkpf` | 无直接等价项 | Ventus target 降级为 no-op；若将来增加 L2 data-prefetch path 再实现优化 |
| `utmacmdflush` | `funct3=6, zimm=16` | 作为 CUDA SASS 风格的 `bulk_commit` alias；不分配第二套独立语义或状态 |
| 无 | `funct3=5, subop=0` | 新增 `prefetch.tensormap` |
| `bulk_commit` | `funct3=6, zimm=16` | 名称可保留，语义改成四组 ring 的 commit |
| `depbar` 等 bulk counter | `funct3=6, zimm=24..27` | 新增 `bulk_wait_group N`；不要假装是通用 depbar |
| `mbar_init` | mbar op 0 | 只保留寄存器 form 或由 assembler 展开 immediate form |
| `mbar_arrive` | mbar op 1 | 只保留 uniform `arrive_expect_tx`，无返回 token |
| `mbar_wait` | mbar op 2 | 改成 old-phase 标量阻塞 wait，无 predicate/try-wait |
| `mbar_inval` | 无 | 删除；对象回收依赖 WG 生命周期/实现状态 |
| `fence.async...` | mbar/proxy op 3 | 单独定义为 Ventus proxy fence，不继承 GPIDL scope modifier |

建议在 GPIDL source 层保留 `utmaldg/utmastg/utmaredg/ublkcp/ublkred` 名称作为可读 alias，避免上层编译器立即大改；但 canonical target operation 应按本地四个数据格式建模。alias 必须经过 capability 检查，不能静默丢弃 GPIDL modifier。

### 7.1 CUDA 中确实存在 `UTMACMDFLUSH`，但它属于 SASS 层

需要严格区分 CUDA 的三个层级：

| 层级 | 程序员看到的接口 | 作用 |
|---|---|---|
| CUDA C++ / libcu++ | `cuda::ptx::cp_async_bulk_commit_group()` | C++ 对 PTX 指令的薄封装 |
| PTX 虚拟 ISA | `cp.async.bulk.commit_group` | 把此前未提交、采用 `bulk_group` 完成机制的 bulk/tensor async 操作提交成一个新 group |
| Hopper/Blackwell SASS | `UTMACMDFLUSH` | 上述 commit 在 NVIDIA 机器 ISA 中的实现名称 |

官方证据：

- NVIDIA 的 [CUDA Binary Utilities 指令表](https://docs.nvidia.com/cuda/cuda-binary-utilities/#hopper-instruction-set) 在 Hopper（Compute Capability 9.0）的 Tensor Memory Access instructions 中明确列出 `UTMACMDFLUSH — TMA Command Flush`；Blackwell 表中也保留该 opcode。
- NVIDIA 的 [PTX ISA `cp.async.bulk.commit_group`](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-commit-group) 规定它创建 per-thread bulk async-group，纳入此前所有尚未提交且使用 `bulk_group` completion 的 bulk/tensor async 操作；空 group 也合法。该指令自 PTX ISA 8.0 引入，要求 `sm_90+`。
- CUDA Programming Guide 的 [TMA S2G 示例](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#asynchronous-data-copies-using-the-tensor-memory-accelerator-tma) 使用 `cp_async_bulk_commit_group()`，随后以 `cp_async_bulk_wait_group_read<0>()` 等待 shared source 已被读完。

本地还使用 CUDA 13.1 对 `sm_90` 做了最小编译/反汇编验证：

```text
PTX/C++ source                          Hopper SASS
cp.async.bulk.global.shared...   ->    UBLKCP.G.S
cp.async.bulk.commit_group       ->    UTMACMDFLUSH
cp.async.bulk.wait_group.read 0  ->    DEPBAR.LE SB0, 0x0
```

连续写两个 `commit_group` 会生成两个 `UTMACMDFLUSH`，说明它是每个 group 的提交边界，而不是编译器在 kernel 尾部偶然插入的一次全局 flush。

所以它的需求是明确的：S2G bulk/tensor 操作采用 bulk-group 完成机制时，软件需要先提交 group，之后 `wait_group N` 才有确定的等待对象。`UTMACMDFLUSH` 本身不等待搬运完成，也不替代 memory fence；真正的完成/源读取完成由后续 wait 指令观察。G2S 使用 mbarrier 完成时则不需要这条 bulk-group commit。

SASS 名称中的 `CMDFLUSH` 可以理解为把当前尚未提交的 TMA command batch 推入一个已提交 group；不能据此在 GPIDL 中再发明一个与 `bulk_commit` 并列、只保证 dispatch 的独立软件可见操作。

## 8. GPIDL 当前 TMA 文本中的内部问题

在对齐本地设计前，GPIDL 自身还有几处需要先澄清：

1. `utmaredg` 的 notes 说明 min/max 等操作要与 `tma_red_type` 配合，但该 leaf 的 modifier 实际只有 `tma_dim`、`tma_reduce_mode`、`tma_red_op`、`cache_evict`，生成编码也没有 type 字段。需要明确类型究竟来自 descriptor，还是漏了 modifier。
2. `utmaldg` notes 写 TMA 完成时给 transaction counter“加 N bytes”，而架构同步文档描述的是 outstanding bytes 被完成事件扣减直至 0。两处方向相反。
3. `tma_bulk_dir` 的说明引用 `isa/spec/tma_completion_analysis.md`，仓库内没有该文件，是失效引用。
4. 官方 SASS 指令表和 CUDA 13.1 实测确认 `cp.async.bulk.commit_group` 下沉为 `UTMACMDFLUSH`；当前 GPIDL 同时保留 `utmacmdflush` 和抽象 `bulk_commit`，又赋予二者不同语义，属于重复/冲突定义。应只保留一个 canonical bulk-group commit，另一个最多作为 alias。
5. 大量 modifier 的合法组合只写在 prose 中，没有进入可机器校验的 constraint。例如 reduce op/type/direction、counted 方向、multicast/cluster 的组合都可能被工具生成出 RTL 不支持的实例。

如果直接在这些定义上做 Ventus lowering，工具链容易把“规范内部未决定”误当成“目标可自由选择”。因此第一阶段应先把这些歧义变成明确的 capability/constraint。

## 9. 推荐实施方案

### 9.1 不建议全局替换 GPIDL 基础编码

GPIDL 的整个编码流程以 64/96/128 位 prefix-free 容器为前提。本地 Ventus 是 32 位 RISC-V custom opcode。若为了 TMA 把 GPIDL 全局编码改成 32 位，会波及 400 多个 leaf、取指、分支 `next_pc`、文档和所有已验收布局，代价远超过本任务。

推荐把 GPIDL 分成两层：

```text
GPIDL TMA semantic operation
        |
        +-- GPIDL native target: 64/96/128b encoding
        |
        +-- ventus-tma-v2 target/profile: fixed 32b opcode 0x42
                                      + capability checks
                                      + TensorMap ABI checks
```

这让“语义对齐”与“二进制编码”分离，也允许 GPIDL native target 将来继续拥有 cluster/S2S 等扩展，而 Ventus profile 对不支持的功能给出编译期错误。

### 9.2 分阶段修改

#### 阶段 A：冻结本地 ABI（1–3 人日）

- 将 `TmaV2Spec.scala`、`ventus_tma_v2_spec.h` 和 OpenCL raw words 收敛到一份机器可读 ABI 描述。
- 自动生成 Scala/C 常量和 raw encoding golden table，避免三份“independently maintained”定义漂移。
- 明确公共 mnemonic、operand 顺序、非法编码行为和 status detail 含义。
- 决定无 mbar binding 的 G2S 是否继续作为正式 ABI，而不是实现容忍行为。

#### 阶段 B：修正 GPIDL 语义（3–7 人日）

主要修改 [`spec.jsonc`](../../gpidl/spec/gpidl/spec.jsonc)：

- TMA instruction leaf：重做约 10 个现有 leaf，并触及约 8 个同步 leaf。
- modifier：删除或 target-gate `tma_dim`、load/store/reduce mode、multicast、cluster、counted、cache/mem-sem/scope 等本地不编码的轴。
- 新增 TensorMap prefetch、按地址 invalidate、S2G wait-group、简化 mbarrier 和 status CSR 契约。
- 把合法 reduce/type、方向、descriptor mode 组合做成机器可校验约束，而不是只写 notes。
- 修订 `08_synchronization.md` 和 `09_tensor_dataflow.md`。
- 从 source 重新生成 `ISA指令参考.md` 和 `standard/生成版.md`，不手改生成物。

#### 阶段 C：增加 Ventus 固定编码 target（5–10 人日）

- 不把这些 instruction 塞进 GPIDL 的 64/96/128 容器分配表。
- 新增 Ventus 固定位域描述和 encode/decode golden test。
- 定义 `rd-as-src3`、tensor coordinate vreg、group zimm、mbar op 与 status CSR。
- 明确 alias lowering 和不支持 modifier 的诊断。

如果项目要求 GPIDL 的 native target 也继续保留这些 TMA 操作，则 native 可变长布局仍需重新运行 `container_assignment`、stage 2 layout 和 stage 3 codebook；这会产生大批机械生成差异，但不是主要人工工作量。

#### 阶段 D：工具链和模型（5–10 人日）

- LLVM/assembler：加入正式 mnemonic 或 target builtin，替换当前 `.word`/`.insn` magic。
- disassembler：输出 reduce/type/group/mbarrier 的可读形式。
- Spike/GVM/功能模型：共享同一 encoding table 与 descriptor validator。
- OpenCL header：从 ABI 描述生成 wrapper；tensor wrapper不再必须手写固定 raw word。
- 对非法 capability 组合在编译期报错，运行时 descriptor 错误继续走 CSR。

#### 阶段 E：验证（5–8 人日）

- GPIDL schema validator：0 error/0 warning。
- encode/decode round-trip：覆盖全部 funct、reduce 合法组合及 reserved 位。
- Chisel：继续覆盖 frontend、backend、completion、capacity、credit depth、group tracker 和 scheduler control。
- OpenCL 端到端：Spike、GVM、RTL，分别覆盖 cache/nocache。
- 负向测试：错对齐、descriptor high address、reserved control、非法 reduce/type、第五个 mbarrier、错误 group zimm、sticky CSR 清除。
- 同步序列测试：G2S binding 跨多命令消费、phase reuse、S2G 四组 wrap-around、wait 0–3。

## 10. 验收标准

完成“按本地设计修正 GPIDL TMA”至少应满足：

1. GPIDL 文档不再宣称 Ventus profile 在硬件上执行 S2S、cluster、multicast、im2col/gather/scatter、data prefetch、invalidate-all 或 64 位地址；其中 `utmapf/ublkpf` 因为是性能 hint，可以被 target 接受并降级为 no-op。
2. `utmapf` 与 `prefetch.tensormap` 被明确区分，任何 lowering 都不会把 data prefetch 错编成 descriptor prefetch。
3. tensor rank/type/layout 的来源只有 TensorMap；指令不再重复携带可能与 descriptor 冲突的 modifier。
4. bulk/tensor reduce 的所有合法组合由机器约束检查，并与 RTL 校验完全一致。
5. G2S 文档准确描述“前置 `arrive_expect_tx` 建 binding、命令消费 bytes、完成扣减 bytes、phase flip”。
6. S2G 文档准确描述四组 ring 和 wait-group 0–3，不再用 GPIDL 双 `rd_sb/wr_sb` 事件解释 Ventus。
7. 所有 Ventus 指令都能由同一份表 encode/decode 为 32 位 word，且与当前 OpenCL header 的 golden words 一致。
8. CSR `0x814` 的 code/detail、first-error sticky 和 clear-on-zero 行为进入规范和测试。
9. GPIDL 生成文档、编译器/汇编器、Spike/GVM 和 RTL 不各自维护不同的 opcode 常量。

## 11. 最终判断

如果“修正”指的是让 GPIDL **准确描述我们当前本地 TMA V2**，RTL 主数据通路不需要大改，主要工作在规范、目标编码、工具链和验证；完整做完约 **3–6 人周**。其中最关键的不是改 opcode，而是把以下三项变成明确 ABI：

1. TensorMap 负责哪些属性，指令负责哪些属性；
2. G2S mbarrier binding 与 S2G 四组 ring 的完成语义；
3. 固定 32 位 Ventus encoding 与 GPIDL native 可变长 encoding 的 target 分层。

若只追求一份正确的 GPIDL 规范和对照文档，不要求编译器/模拟器立即执行，则约 **3–7 人日**。若要求本地硬件反过来实现 GPIDL 当前全部 TMA 能力，则工作量至少上升到 **8–16 人周**，而且会牵连整个取指/解码和地址/cluster 架构，不应作为本轮“修正指令”的默认方案。
