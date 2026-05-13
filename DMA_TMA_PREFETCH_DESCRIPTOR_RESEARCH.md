# DMA/TMA TensorMap Prefetch 与 Descriptor TMA 设计与验证

## 结论摘要

本文将方案冻结为更接近 CUDA TMA 的两级结构：

1. 新增一条用户可见的 tensor map prefetch 指令，只把 global memory 中的 tensor map descriptor 提前拉到 L2。
2. `CP_ASYNC_TENSOR` 本身仍然携带 tensor map 地址，并由 TMA 硬件按需从 L2/显存读取 tensor map。
3. Prefetch 只影响性能，不参与正确性。即使软件没有发 prefetch，或者 TMA 执行时 prefetch 还没返回，TMA 也能独立读取 descriptor 并继续执行。

这样第一条 prefetch 指令才是真正的加速 hint，而不是 TMA 的必需前置状态。它的收益来自“后续 TMA 读 tensor map 时命中 L2”或“命中 L2 MSHR/未完成请求”，而不是来自 ISA 语义上的依赖。

## 设计完成状态

截至 2026-05-13，本文冻结的设计边界如下：

- ISA 层冻结为 `PREFETCH_TENSORMAP desc_ptr` 加 `CP_ASYNC_TENSOR_G2S smem, desc_ptr, coords_ptr`。prefetch 是 best-effort hint，TMA 指令本身必须拥有完整 correctness 输入。
- ABI 层冻结为 128B global tensor map descriptor v0，加可选 dynamic coords block。shared 端地址不放进 descriptor，而由 TMA 指令 operand 传入。
- RTL 层冻结为两条互不依赖的路径：prefetch 发 Get-like L2 fill，request 被 L2 接受后立即释放 issue path，response 由异步 metadata sink 接收并填入 TMA descriptor cache；TMA 自主读取 descriptor/dynamic 参数，且可先查 descriptor cache，再复用现有 tensor address generation、OOB、subbox、elementStride、swizzle 逻辑。
- 一致性模型冻结为 host 在 kernel launch 前写 descriptor、kernel 内只读 descriptor。device-side descriptor update、proxy fence、descriptor invalidate/version check 都列为后续功能。
- 当前仓库已经完成 descriptor-addressed G2S TMA 与 tensor map prefetch 的初版 Spike/GVM RTL 实现。legacy VGPR TMA/DMA directed baseline 仍作为回归对照，证明新路径没有破坏现有 DMA/TMA 语义。
- 最终测试对象和 DMA/TMA 项目结构详表见 `testcases/_get_case/DMA_TMA_PROJECT_STRUCTURE.md`，其中按 app-level directed suite、legacy smoke、RTL unit test、Spike/GVM model 分层列出。

## 外部语义参考

- NVIDIA PTX `prefetch.tensormap` 会把 tensor map 所在 cache line 预取，供后续 `cp.async.bulk.tensor` 使用。PTX 文档把 `.tensormap` 支持列为 PTX ISA 8.0、sm_90 起的能力，并说明其面向 `.const` 或 `.param` tensor map 地址。见 NVIDIA PTX ISA 9.2 文档 `prefetch, prefetchu` 小节：<https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prefetch-prefetchu>
- CUDA Programming Guide 说明 TMA 支持 1D 到 5D，多维 bulk tensor copy 需要 tensor map；tensor map 通常由 host 通过 `cuTensorMapEncode` 创建，再作为 `const __grid_constant__` 参数传到 device。见 CUDA Programming Guide 4.11.2：<https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/04-special-topics/async-copies.html#using-the-tensor-memory-accelerator-tma>
- 同一 CUDA 文档还提到 tensor map 可以放到 global memory，但如果被 host 或 device 修改后再使用，需要 tensor map proxy 的 release/acquire fence。这个点对 Ventus 后续做 device-side descriptor 更新很重要。

## 当前 Ventus 现状

### 参数路径

当前 `CP_ASYNC_TENSOR` 是完整 VGPR 参数模式：

- `gpgpu/ventus/src/pipeline/DecodeUnit.scala:496` 将 `CP_ASYNC_TENSOR` 译码为 `A1_VRS1, A2_VRS2, A3_VRS3`。
- `gpgpu/ventus/src/pipeline/issue.scala:17` 的 `vExeData` 对每个 operand 都保留 `Vec(num_thread, UInt(xLen.W))`。
- `gpgpu/ventus/src/pipeline/DMA_core.scala:159` 起直接从 `reg_save.in1/in2/in3` 解出 `TensorVars`，例如 `BoxAddress = in2(0)`、`swizzleMode = in2(12)`、`globalAddress = in1(2)`。
- `testcases/_get_case/tma_matrix_test/tma_matrix_test.cl:11` 注释描述了当前 testcase 协议：kernel 把 96-word descriptor 拷到 `__local param_buf[96]`，patch 三个 runtime 地址，然后加载 VRS1/VRS2/VRS3 后发 `CP_ASYNC_TENSOR`。
- `testcases/_get_case/tensor_dma_test/tensor_dma_test.cl:6` 也明确写着 RTL 硬件从 VGPR 读 `CP_ASYNC_TENSOR` 参数。

这意味着当前所谓 descriptor 并不是硬件可见的 global tensor map，而是 testcase/kernel 自己用 shared memory 暂存的一组三段 VGPR 初始化数据。

### 新目标模型

目标不是把 prefetch 做成 `CP_ASYNC_TENSOR` 的内部必经状态，而是拆成两条彼此独立的指令：

```text
PREFETCH_TENSORMAP desc_ptr        // optional, only warms L2
CP_ASYNC_TENSOR_G2S smem, desc_ptr, coords_ptr
```

`PREFETCH_TENSORMAP` 只提前读取 tensor map 所在 cacheline，让 L2 更可能已经有 descriptor。`CP_ASYNC_TENSOR` 仍然显式携带同一个 `desc_ptr`，并由 TMA 硬件自己读取 tensor map。这样 prefetch 完成与否不会影响 correctness。

## ISA 结构

### 1. 新增 TensorMap Prefetch 指令

定义一条独立 cache hint 指令：

```text
PREFETCH_TENSORMAP rs1

rs1 : tensor map descriptor pointer in global memory
```

RISC-V 32-bit custom encoding 约束：

```text
opcode/funct : PREFETCH_TENSORMAP_L2
rd           : x0，reserved
rs1          : descriptor pointer
rs2          : x0，reserved
funct bits   : v0 固定 target=L2, size=128B
```

语义：

```text
addr = X[rs1]
prefetch cache lines covering [addr, addr + 128B) into L2
no architectural register write
no shared memory side effect
no fence/completion guarantee
may be dropped on unsupported mode
```

关键点：

- 这条指令只负责提前把 tensor map 放进 L2，不填 `TensorVars`，不启动 DMA copy。
- 它是性能 hint，不是 correctness requirement。
- 如果后续 TMA 到达时 prefetch 已完成，TMA descriptor fetch 应命中 L2。
- 如果后续 TMA 到达时 prefetch 还在飞行中，理想情况是 L2 MSHR 合并请求；即使不能合并，TMA 也可以独立发起自己的 descriptor fetch。
- 如果软件完全不发 prefetch，`CP_ASYNC_TENSOR` 仍然正确，只是 descriptor fetch 可能走 L2 miss。

v0 不让 prefetch 指令携带方向、shared 地址、tile 坐标或 box 参数。它只看 tensor map 地址。

### 2. Descriptor-addressed TMA 指令

TMA 指令继续显式携带所有正确性所需的地址类 operand：

```text
CP_ASYNC_TENSOR_G2S rd, rs1, rs2
CP_ASYNC_TENSOR_S2G rd, rs1, rs2   // 未来 SMEM->L2 路径实现后启用

rd  : shared memory pointer
rs1 : tensor map descriptor pointer in global memory
rs2 : dynamic parameter pointer, or 0
```

方向不应该放进 descriptor，也不应该占用一个通用寄存器。建议用 opcode/funct 区分：

```text
funct = TMA_G2S : global/L2 -> SMEM
funct = TMA_S2G : SMEM -> global/L2
```

这样三个 scalar operand 的含义就清楚了：

- `rd` 给 shared 端地址。
- `rs1` 给 tensor map 的 global memory 地址。
- `rs2` 给动态参数地址，例如 tile 坐标块；如果某个模式不需要动态参数，则为 0。

注意：`rs1` 在 prefetch 和 TMA 中是同一个 tensor map 指针。prefetch 只是让这段 descriptor 数据更可能已经在 L2，TMA 仍然会自己读取 `rs1` 指向的 tensor map。

## Descriptor ABI

v0 固定 128B 对齐、128B 长度，便于一次或两次 L2 cacheline 获取，也方便后续增加字段。布局如下：

```text
word  0: magic/version/flags
word  1: dataType/rank/interleave/swizzle/L2promotion/oobfill
word  2: globalAddress
word  3: reserved or descriptor size
word  4..8:   globalDim[0..4]
word  9..13:  globalStrides[0..4]
word 14..18:  boxDim[0..4]
word 19..23:  elementStrides[0..4]
word 24..31:  reserved for im2col, atom/swizzle extension, future 64-bit addr
```

动态块 v0 可以保持很小：

```text
word 0..4: tensorCoords[0..4]
word 5..7: reserved
```

TMA setup 中根据 descriptor 和动态坐标计算当前 tile 起点：

```text
BoxAddress = globalAddress + sum(tensorCoords[d] * globalStrides[d])
```

其中 dim0 的 byte/element 语义要和当前 RTL 已经验证过的 OOB/subbox/elementStride 逻辑保持一致，尤其不能破坏 `tensor_dim0_start` 和 high-dim valid 的行为。

`smem_dst` 不建议放在 descriptor 里，因为 shared 地址是每个 workgroup/kernel 实例运行时才稳定的值，应由 `rd` 传入。

## 32-bit 指令放不下参数怎么办

Ventus 是 32-bit custom instruction，不可能像 NVIDIA 更宽的编码那样把复杂 operand 都放进指令。推荐原则是：

```text
指令只编码“操作类型和少量指针”，复杂参数放在 memory descriptor 中。
```

因此：

- 方向：放在 `opcode/funct`。
- shared 地址：放在 `rd`。
- tensor map 地址：放在 `rs1`。
- tile 坐标或动态参数：放在 `rs2` 指向的 memory block。
- tensor 维度、stride、swizzle、OOB fill、elementStride：放在 tensor map descriptor。

如果希望最小 bring-up，可以临时定义 `rs2` 为直接 `BoxAddress` override；但冻结 ABI 仍以 `rs2` 指向坐标块为目标，因为 descriptor 才能跨 tile 复用。

## RTL 微结构方案

### 1. Prefetch 指令路径

`PREFETCH_TENSORMAP` 可以做成 DMA/L2 前端的轻量请求：

```text
decode PREFETCH_TENSORMAP
  -> capture desc_ptr
  -> optional TLB translation
  -> issue L2 cacheable read/prefetch for descriptor lines
  -> sink/drop response payload
```

请求类型需要和普通 DMA data/TMA descriptor fetch 区分：

- prefetch response：用于完成 L2 fill 和填充 TMA descriptor cache，payload 不写 shared，不填 `TensorVars`。
- descriptor response：由 TMA 指令发起，payload 解码成 `TensorVars`。
- data response：普通 TMA payload，进入 shared write path。

如果当前 L2 接口没有真正的 prefetch/no-response opcode，可以用普通 read 模拟，但 response 只能进入 metadata sink：可以填 descriptor cache，不能写 shared，不能污染 `TensorVars`。这样 Spike 可以把 prefetch 当 no-op，RTL 则通过 L2 fill 和 descriptor cache 得到性能收益。

### 2. TMA 指令路径

`CP_ASYNC_TENSOR_*` 不依赖 prefetch 状态：

```text
decode CP_ASYNC_TENSOR
  -> capture smem_dst / desc_ptr / dyn_ptr
  -> fetch descriptor from desc_ptr through L2
  -> fetch dynamic coords if needed
  -> fill TensorVars
  -> existing tensor address generation
  -> existing L2 data read + SMEM write + fence
```

如果 `PREFETCH_TENSORMAP` 已经把 descriptor 放进 L2，这里的 descriptor fetch 就更快；如果没有，TMA 自己 miss 到 memory，功能仍正确。

### 3. DMA-local descriptor cache

有 L2 prefetch 后，DMA core 内部 descriptor cache 仍然有价值：

- 同一 warp 连续搬多个 tile 时，desc_ptr 相同，可以避免每条 TMA 都重新从 L2 读 128B。
- 多 warp 共享同一 tensor map 时，L2 已经能提供跨 warp 复用；DMA 内部 buffer 主要减少 L2 端口压力。

当前实现已经增加参数化小缓存，默认 `tma_desc_cache_entries = 2`。cache 以 128B-aligned descriptor line 为 tag，payload 来自 `CP_ASYNC_TENSOR_G2S` 的 descriptor response 或 `PREFETCH_TENSORMAP` 的 prefetch response。命中时 TMA 跳过 descriptor L2 fetch；若还有 dynamic coords，仍然单独读取 coords block。

## 当前项目里的未完成 Prefetch 是否有用

基于当前 Ventus L2 RTL，实现结论是：**有条件有用，但前提是 prefetch 不能用现有 `Hint` 路径实现，而应实现成 Get-like 的 cacheline 读取/填 L2 请求，并丢弃返回 payload**。

当前 L2 `Scheduler` 有 MSHR 和 secondary miss 合并结构：

- `Scheduler.scala` 中按 MSHR 实例保存 outstanding miss，并用 `tagMatches` 比较已有 MSHR 的 `tag/set` 和新 miss 的 `tag/set`。
- 当新 miss 命中已有 outstanding MSHR 时，`requests.io.push.bits.index` 会选择 `tagMatches` 对应的旧 MSHR，而不是新分配 MSHR。
- MSHR 只通过 `schedule.a` 对下游发一次 `Get`；内存返回后，`requests` 队列里的每个 source 会依次经 `SourceD` 返回。

因此，如果流程是：

```text
t0:    PREFETCH_TENSORMAP desc_ptr 作为 Get-like 请求进入 L2，descriptor line miss，分配 MSHR 并向下游内存发 Get
t0+N:  CP_ASYNC_TENSOR 读取同一个 desc_ptr，L2 directory 仍 miss，但 tag/set 匹配已有 MSHR
```

那么 TMA descriptor fetch 会作为 secondary miss 挂到同一个 MSHR 上，不应再向下游内存发第二笔相同 cacheline 读取。它仍然要等内存返回，但只等剩余 latency，所以 prefetch 即使未完成也有部分加速效果。

限制也很明确：

- 当前 `DMA_core.scala` 发往 L2 的 DMA 请求是 `a_opcode := 4.U // Get`，没有已有 prefetch opcode。
- 当前 L2 `Scheduler` 对 `Hint` 的处理是 flush/invalidate：`directory.io.read.valid` 会排除 `Hint`，`directory.io.invalidate/flush` 由 `Hint` 触发。因此不能把 `PREFETCH_TENSORMAP` 直接编码成现有 TL `Hint`，否则语义会错。
- 这里不是标准 TL `Hint` 天生只能 flush，而是 Ventus 当前 L1/L2 已经把 opcode 5 约定为 cache maintenance：`DCacheParameters.scala` 中 `TLAOp_Flush = 5.U`，`param=0/1` 分别表示 flush/invalidate；DCache 发 L2 维护请求时也使用这个编码。因此 L2 侧看到同一个 opcode 值时虽然源码名是 `Hint`，实际项目语义是 flush/invalidate，不是 prefetch fill。
- Prefetch 指令的 DMA 前端不能一直 busy 到 response 返回；当前实现已经在 L2 请求被接受后释放 issue/AddrCalc 路径，只保留 prefetch slot、response sink 和 completion queue。这样后续 TMA 可以在 prefetch 未完成时进入 L2，并和同一 descriptor line 的 outstanding MSHR overlap。
- 如果 L2 MSHR 全满，当前 `request.ready` 仍受 `mshr_free` 限制，新请求可能被挡住，secondary merge 的收益会被资源压力削弱。

所以针对当前项目，准确说法不是“看有没有 MSHR”，而是：

```text
PREFETCH_TENSORMAP 若实现为 Get-like L2 fill，并允许 TMA 在它未完成时继续发出，
则当前 L2 的 tag/set MSHR 合并能提供未完成 prefetch 的部分加速；
若实现为现有 Hint，或前端等 response 才释放，则基本没有这个加速效果。
```

## With-cache / no-cache 可行性

with-cache RTL 中，prefetch 的意义很直接：通过 DMA/L2 路径提前填 L2，后续 TMA descriptor fetch 命中。

no-cache 版本如果只是 L1 bypass、仍然有 L2/SMEM/DMA 路径，则功能上也可以工作：

- `PREFETCH_TENSORMAP` 仍走 DMA 到 L2/内存的路径。
- TMA descriptor fetch 仍然从 `rs1` 发起，不依赖 L1。
- 如果 no-cache 配置不保留 L2 cache allocate 语义，prefetch 可以退化为 no-op，但 TMA 仍正确。

因此这条指令的 ISA 语义应该是 best-effort hint，而不是 guaranteed L2 residency。

## 收益评估

### 明显收益

- 软件侧减少 96-word shared descriptor 初始化、三段 `vlw12.v` VGPR 加载和相关 barrier。
- 减少 VGPR 占用，TMA 指令前的准备代码更短。
- descriptor 可复用，一个 tensor map 可以服务多个 tile 坐标。
- 更接近 NVIDIA/CUDA TMA 抽象，后续移植 CUTLASS/TMA 风格 kernel 更自然。
- `PREFETCH_TENSORMAP` 可以被软件提前调度，用前面的独立计算或上一轮 DMA 覆盖 descriptor miss latency。
- 为 host-side tensor map encode、future im2col/SMEM->L2 统一 ABI。
- descriptor 结构可版本化，测试用例不必继续手工拼 VRS1/VRS2/VRS3 三段布局。

### 可能负收益

- 每次 TMA 都要从 `desc_ptr` 读取 descriptor；如果没有 prefetch 命中或 descriptor buffer，小 tile 可能变慢。
- Prefetch 发得太晚时可能没有收益；如果 L2 不合并 outstanding miss，还可能产生重复请求。
- RTL 前端状态机更复杂，prefetch/descriptor/data response tag 管理更容易出错。
- 需要定义 descriptor 一致性规则。host 写 descriptor 后 kernel 使用比较简单；device 在同一 kernel 内修改 descriptor 会牵涉 fence/proxy 语义。
- Spike、GVM reference、host expected model、OpenCL testcase 都要同步改 ABI。
- 指令编码如果处理不好，会留下 VGPR legacy 和 descriptor mode 两套长期包袱。

总体看，收益大于成本，前提是把 `PREFETCH_TENSORMAP` 做成真正的 L2 warm-up hint，同时让 TMA 自主读取 descriptor。只把参数搬到 global descriptor、但没有 prefetch 或复用策略，会得到更干净的 ABI，却未必有性能收益。

## 一致性与内存模型建议

v0 只支持两种安全场景：

1. Host 在 kernel launch 前写好 descriptor。kernel launch 作为 host/device 边界，DMA 直接读取 descriptor。
2. Kernel 内只读 descriptor，不支持 device-side 修改后立即使用。

`PREFETCH_TENSORMAP` 本身不提供 ordering guarantee。它不能替代 fence，也不能保证后续 TMA 一定看到某次 device write；它只对已经可见的 descriptor 做 best-effort cache warm-up。

后续如果要支持 device-side 修改 descriptor，需要补：

- descriptor writeback 或普通 global store 到 descriptor。
- descriptor acquire/release fence。
- warp/block 内同步规则。
- descriptor cache invalidate 或 version check。

NVIDIA 的 global tensor map 使用需要 tensor map proxy fence。Ventus 可以先不实现 proxy，只在文档里把 device-side 修改列为 unsupported，避免隐含一致性 bug。

## Spike 和 testcase 改动方向

Spike 初版可以把 `PREFETCH_TENSORMAP` 实现为 no-op 或只做地址合法性检查，因为它没有 architectural result。Spike 的 `cp_async_tensor.h` 不再从 `P.VU.elt(... VRS ...)` 读 96 个 word，而是先按 32-bit 指令 operand 取地址：

```text
// PREFETCH_TENSORMAP
desc_ptr = RS1
no architectural state update

// CP_ASYNC_TENSOR
desc_ptr = RS1
dyn_ptr  = RS2
smem_dst = READ_REG(insn.rd())
desc[i]  = MMU.load_uint32(desc_ptr + 4*i)
coords[i] = MMU.load_uint32(dyn_ptr + 4*i)   // if dyn mode
```

testcase kernel 不再创建 `__local param_buf[96]`，也不再 `vlw12.v`。流程改为：

```text
host builds global descriptor
kernel computes smem_dst
kernel passes desc_ptr, dyn_ptr, smem_dst via scalar regs
kernel optionally issues PREFETCH_TENSORMAP desc_ptr early
kernel issues CP_ASYNC_TENSOR
kernel issues CP_ASYNC_FENCE
```

host expected model 继续用同一 descriptor layout 解码，避免 RTL/Spike/host 三边各写一套语义。

## Descriptor TMA 验收计划

后续实现 descriptor-addressed TMA 和 `PREFETCH_TENSORMAP` 时，验收 directed tests 按以下顺序补齐：

1. 最小 2D FP32 dense：不发 prefetch，直接 TMA，验证 correctness。
2. 同一 case 加 `PREFETCH_TENSORMAP`，结果必须完全一致。
3. 同一 descriptor、不同 coords，连续搬 2 到 4 个 tile，验证 descriptor 复用。
4. Subbox/OOB/elementStride/swizzle 全部通过新参数入口覆盖。
5. 多 warp 共用同一个 descriptor，同时发 prefetch、DMA 和 fence。
6. shared bank conflict routing，确认 prefetch/descriptor/dynamic response 不会误进 DMA payload routing。
7. with-cache 和 no-cache 各跑完整 directed suite。
8. Misaligned descriptor、bad magic/version、unsupported rank/interleave 的 assert 或 skip 行为。
9. RTL log 或 performance counter 统计 descriptor fetch L2 hit/miss，比较 prefetch 提前距离的收益。

验收顺序为 Spike 先实现 descriptor-address 模型，GVM 再跟进。这样能先暴露 ABI/testcase 问题，再消耗 RTL 编译时间。

## 2026-05-13 验证记录

本次验证目标分两层：

1. 确认迁移前 legacy VGPR DMA/TMA directed baseline 健康，并把它作为 descriptor-addressed TMA 的回归对照组。
2. 确认新增 `PREFETCH_TENSORMAP` 与 `CP_ASYNC_TENSOR_G2S` 在 Spike、Chisel directed RTL test、重建后的 GVM end-to-end 路径都能工作。

所有有意义的 `.out` 都必须从 testcase 目录内启动；从仓库根目录直接运行会导致 host 找不到同目录 `.cl`，表现为 OpenCL build error `-44` 和 `<test>.cl: No such file or directory`，这是 harness invocation 错误，不是 Spike/GVM 功能失败。

Directed suite 的 case 数量会随着覆盖点增加而变化，所以本文的 pass/skip 数字是 2026-05-13 的一次 verified snapshot，不是永久固定目标。后续验证应以当前日志里的 `[n/m]` 和最终 `pass/fail/skip` summary 为准；如果 case 数量变化但 `fail: 0`，且 Spike/GVM 的 skip/pass 分布符合当前 backend 预期，应按 suite drift 处理并更新文档，而不是直接判成回归。

日志目录：

```text
/tmp/codex-dma-tma-prefetch-verify-20260513-171433
```

Spike 结果：

```text
tma_matrix_test:                     15 pass / 0 fail / 8 skip
bulk_dma_matrix_test:                 4 pass / 0 fail / 0 skip
multi_warp_dma_fence_test:            4 pass / 0 fail / 0 skip
dma_shared_routing_conflict_test:     2 pass / 0 fail / 0 skip
```

GVM 结果：

```text
tma_matrix_test (VENTUS_TMA_RUN_RTL_ONLY=1): 23 pass / 0 fail / 0 skip
bulk_dma_matrix_test:                         4 pass / 0 fail / 0 skip
multi_warp_dma_fence_test:                    4 pass / 0 fail / 0 skip
dma_shared_routing_conflict_test:             2 pass / 0 fail / 0 skip
```

GVM 仍会打印已知 checker noise，但 host verdict 全部为 `PASS`/`OK`：

- 多个 case 有 PC `0x80000014` 的 XREG mismatch。
- TMA OOB/estride 覆盖中可见 VREG mismatch 和一次 `XREG_WB_TYPE` 提示。
- 这些 mismatch 没有变成 host-visible failure，本次按 checker/reference 噪声处理。

因此，当前 legacy VGPR 参数路径、bulk DMA、multi-warp DMA fence、shared-bank-conflict routing 的 directed baseline 是干净的。后续实现 descriptor-addressed TMA 时，至少需要保持上述 Spike/GVM host verdict 不退化；新增 prefetch 只能改变性能或 checker 统计，不能成为 correctness 依赖。

### Descriptor/prefetch 新路径验证

实现后新增验证记录如下。

Spike descriptor smoke：

```text
log: /tmp/codex-dma-tma-desc-impl-20260513-run6-fresh
tma_descriptor_test:
  use_prefetch=0: PASS
  use_prefetch=1: PASS
  final verdict: OK
```

该日志确认 `0x00c5e542` 被执行为 `cp.async.tensor.g2s`，读取 128B descriptor、dynamic coords 和 global payload 后写 shared；PC 从 `0x800001b0` 前进到 `0x800001b4`，不再落到 `0x800001b2` 的 compressed halfword 路径。`0x00004042` 也显示为 `cp.async.fence`，避免 trace 被误读成 `c.lwsp`。

Spike directed regression：

```text
log: /tmp/codex-dma-tma-prefetch-desc-spike-20260513
tma_descriptor_test:                 2 pass / 0 fail
tma_matrix_test:                    15 pass / 0 fail / 8 skip
bulk_dma_matrix_test:                4 pass / 0 fail / 0 skip
multi_warp_dma_fence_test:           4 pass / 0 fail / 0 skip
dma_shared_routing_conflict_test:    2 pass / 0 fail / 0 skip
```

Chisel directed RTL tests：

```text
./mill -i 'ventus[6.4.0].tests.testOnly' DmaTest.TMA_core_test -- -z TMA_T26
  TMA_T26_descriptor_g2s_fetches_descriptor_and_coords: PASS

./mill -i 'ventus[6.4.0].tests.testOnly' DmaTest.TMA_core_test -- -z TMA_T27
  TMA_T27_prefetch_tensormap_drops_payload_and_completes: PASS

./mill -i 'ventus[6.4.0].tests.testOnly' DmaTest.TMA_core_test -- -z TMA_T28
  TMA_T28_prefetch_releases_issue_path_before_response: PASS

./mill -i 'ventus[6.4.0].tests.testOnly' DmaTest.TMA_core_test -- -z TMA_T29
  TMA_T29_descriptor_cache_reuses_fetched_descriptor: PASS

./mill -i 'ventus[6.4.0].tests.testOnly' DmaTest.TMA_core_test -- -z TMA_T1_2d_aligned
  TMA_T1_2d_aligned: PASS
```

T26 覆盖 descriptor fetch、dynamic coords fetch、metadata response source tag 和后续 data response/shared write 路径。T27 覆盖 prefetch Get-like L2 read、metadata response sink 和 fence completion，不应产生 shared write。T28 覆盖 prefetch request fire 后立即释放 issue path，prefetch response 未返回时后续 TMA 仍能发出 descriptor fetch。T29 覆盖同一 descriptor 第二次 TMA 命中 DMA-local descriptor cache，跳过 descriptor L2 fetch。T1 回归 legacy VGPR TMA 路径。

GVM end-to-end：

```text
GVM rebuild:
  make -C gpgpu/sim-verilator -f gvm.mk RELEASE=1 PREFIX=/home/liyb/ventus-env-copilot-test/install install
  result: PASS, installed libVentusGVM.so and libVentusGVM-withcache.so
  latest rebuild after overlap/cache RTL: PASS on 2026-05-13

log: /tmp/codex-tma-desc-gvm-20260513-224527
tma_descriptor_test:
  use_prefetch=0: PASS
  use_prefetch=1: PASS
  final verdict: OK

log: /tmp/codex-tma-matrix-gvm-20260513-224621
tma_matrix_test (VENTUS_TMA_RUN_RTL_ONLY=1): 23 pass / 0 fail / 0 skip

Directed suite after registering descriptor testcase:
log: /tmp/codex-dma-tma-directed-spike-20260513-233547
backend=spike pass=5 fail=0
  tma_descriptor_test: use_prefetch=0 PASS, use_prefetch=1 PASS, final OK
  tma_matrix_test: 15 pass / 0 fail / 8 skip
  bulk_dma_matrix_test: 4 pass / 0 fail / 0 skip
  multi_warp_dma_fence_test: 4 pass / 0 fail / 0 skip
  dma_shared_routing_conflict_test: 2 pass / 0 fail / 0 skip

log: /tmp/codex-dma-tma-directed-gvm-20260513-235608
backend=gvm pass=5 fail=0
  tma_descriptor_test: use_prefetch=0 PASS, use_prefetch=1 PASS, final OK
  tma_matrix_test (VENTUS_TMA_RUN_RTL_ONLY=1): 23 pass / 0 fail / 0 skip
  bulk_dma_matrix_test: 4 pass / 0 fail / 0 skip
  multi_warp_dma_fence_test: 4 pass / 0 fail / 0 skip
  dma_shared_routing_conflict_test: 2 pass / 0 fail / 0 skip

Earlier full directed GVM regression:
log: /tmp/codex-dma-tma-prefetch-desc-gvm-20260513
bulk_dma_matrix_test:                         4 pass / 0 fail / 0 skip
multi_warp_dma_fence_test:                    4 pass / 0 fail / 0 skip
dma_shared_routing_conflict_test:             2 pass / 0 fail / 0 skip
```

GVM 仍会打印既有 checker/reference 噪声，包括 PC `0x80000014` 的 XREG mismatch、TMA OOB/estride 覆盖中的 VREG mismatch 和一次 `XREG_WB_TYPE` 提示。所有这些都没有变成 host-visible failure，本次仍按非 host-failing checker 噪声处理。

`testcases/_get_case/run_dma_tma_rtl.sh` 已把 `tma_descriptor_test` 纳入 directed suite，并只在 GVM 的 `tma_matrix_test` 上注入 `VENTUS_TMA_RUN_RTL_ONLY=1`。runner 不设置 `ulimit -s unlimited`；该设置会让 GVM 下 `tma_descriptor_test` 在 RTL trace 前早退为 host segfault。

## 分阶段实施建议

### Phase 0：文档与 ABI freeze

状态：本文已完成。

- 已固化 `PREFETCH_TENSORMAP rs1` 指令语义与编码约束。
- 已固化 `CP_ASYNC_TENSOR_G2S/S2G rd, rs1, rs2` operand 语义。
- 已固化 128B descriptor v0 布局。
- 已规定 host-only descriptor update 是 v0 唯一 supported coherence model。
- 已完成 legacy VGPR DMA/TMA directed baseline 的 Spike/GVM 验证，作为后续迁移回归基线。

### Phase 1：Spike/testcase 语义迁移

状态：已完成初版。

- Spike 增加 `PREFETCH_TENSORMAP` no-op 模型。
- Spike 新增 `CP_ASYNC_TENSOR_G2S`，从 `rd/RS1/RS2` 读取 `smem_dst/descriptor/dynamic coords`，解码 128B descriptor 后执行 G2S tensor copy。
- Spike fetch/execute/disassemble 均显式处理 Ventus custom opcode `0x42/0x72` 占用 RVC 编码空间的问题，避免 32-bit custom `.word` 被 compressed halfword 路径抢走。
- 新增 `testcases/_get_case/tma_descriptor_test`，覆盖 no-prefetch 与 prefetch 两条 descriptor-addressed G2S 路径。

### Phase 2：RTL TMA 自主 descriptor fetch

状态：已完成初版。

- 新增 `CP_ASYNC_TENSOR_G2S` 译码，使用 scalar `rd/rs1/rs2` pointer operands；legacy VGPR `CP_ASYNC_TENSOR` 路径保留。
- `DMA_core.scala` 新增 descriptor/dynamic metadata fetch 状态；这部分是 TMA 自身 correctness 路径，不依赖 prefetch 指令。
- descriptor response 填 `TensorVars` shadow regs，dynamic coords response 填 tile 坐标。
- metadata response 与 data response 通过 L2 source tag 区分，descriptor/prefetch/dynamic payload 不会误进 shared write routing。
- 复用现有 tensor address generation、OOB、subbox、elementStride、swizzle 逻辑。

### Phase 3：RTL Prefetch 指令

状态：已完成 overlap 初版。

- Decode 增加 `PREFETCH_TENSORMAP`。
- DMA/L2 前端发 descriptor-line Get-like cacheable read。
- L2 request fire 后立即释放 AddrCalc/issue path；prefetch response 之后由异步 metadata sink 接收。
- response payload 不写 shared、不填 `TensorVars`，但会填入 TMA descriptor cache。
- DMA/fence completion 仍等 prefetch response 返回后由 completion queue 发出，因此 correctness/inflight accounting 保持明确。

### Phase 4：性能增强

状态：部分完成，后续继续做 counters/编译器包装。

- 增加 L2 MSHR merge 观察或 counter。
- 已增加参数化 DMA descriptor cache，默认 2-entry，由 `tma_desc_cache_entries` 控制。
- 增加编译器 builtin/inline asm 包装。
- 再往后考虑 global descriptor device-side update、tensor map fence/proxy、im2col、interleave、atom/reduce、SMEM->L2。

## 风险清单

- Prefetch response、descriptor response 和 data response 混淆，导致 shared payload 写错或 `TensorVars` 被污染。
- Prefetch 太近时只能形成 MSHR overlap，未必能命中 descriptor cache；太晚或 slot 满时仍可能重复占用 L2 带宽。
- Descriptor buffer 或 L2 line 没有 invalidate，device-side 修改后读到旧 descriptor。
- 当前 OOB 依赖 `BoxAddress - globalAddress` 计算 subbox 偏移，改 coords 模式时需要重新校验 `tensor_dim0_start` 和 high-dim valid 语义。
- no-cache 版本没有 L1 cache hint 语义，不能把 L1 prefetch 当成功能依赖；`PREFETCH_TENSORMAP` 应只依赖 DMA L2/TLB 路径，必要时退化为 no-op。
- 如果坐标通过 memory pointer 传入，坐标读取本身也需要 TLB/L2 response routing。
- S2G 方向不仅是 ISA bit，还需要未来补 SMEM read、bank conflict backpressure、TL PutFull/PutPartial、TLB、fence 完成计数和 cache coherency 处理。

## 完成判定

设计侧已经完成 Phase 0 到 Phase 3 的功能初版：

- `PREFETCH_TENSORMAP` 被定义为 optional、best-effort、Get-like L2 fill hint。
- descriptor-addressed TMA 被定义为 correctness path，不能依赖 prefetch 完成。
- descriptor ABI、dynamic coords、shared 地址来源、一致性边界、with-cache/no-cache 退化语义均已明确。
- 当前 Ventus L2 的 `Hint` 路径不适合作为 prefetch fill；实现必须走 Get-like read 并丢弃 response payload。
- Spike、RTL decode、DMA metadata fetch、prefetch async response sink、descriptor cache、descriptor testcase 都已落地。
- v0 prefetch 已实现 request-fire 后释放 issue path；response 仍负责填 cache 和释放 DMA/fence inflight。

验证侧已经完成迁移前 baseline 与新增 descriptor/prefetch path：

- Spike/GVM directed suite host verdict 全部为 `PASS`/`OK`。
- `tma_descriptor_test` 在 Spike/GVM 上均验证 no-prefetch 与 prefetch 两种路径。
- Chisel T26/T27 分别验证 descriptor/dynamic fetch 和 prefetch metadata response 路由。
- Chisel T28 验证 prefetch response 未返回时，后续 TMA 可以继续发出同 descriptor line 的 descriptor fetch，覆盖 MSHR overlap 前提。
- Chisel T29 验证同一 descriptor 第二次 TMA 命中 DMA-local descriptor cache，跳过 descriptor L2 fetch。
- GVM checker mismatch 已分类为非 host-failing 噪声。
- case 数量按 2026-05-13 snapshot 记录，后续以当前 summary 为准处理 suite drift。

## 最终设计结论

采用 CUDA 风格结构：

```text
PREFETCH_TENSORMAP desc_ptr        // optional, best-effort, warms L2
CP_ASYNC_TENSOR_G2S smem, desc_ptr, coords_ptr
```

prefetch 指令只加速，不承诺完成；TMA 指令始终能独立从 `desc_ptr` 读取 tensor map。这样既解决了当前 VGPR 参数入口臃肿的问题，也避免把 correctness 绑定到一个可能尚未完成的预取操作上。
