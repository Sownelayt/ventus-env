# DMA/TMA 毕业论文写作思路

本文档面向当前项目中 DMA/TMA 方向的毕业设计写作。参考资料为：

- `gpgpu/doc_tma/本科_徐润南_开源GPGPU张量内存加速器设计与实现.pdf`
- 当前项目中的 DMA/TMA RTL、Spike 模型、OpenCL testcase 与验证文档

## 1. 论文定位

前作《开源 GPGPU 张量内存加速器设计与实现》的主线是：参考 NVIDIA Hopper TMA，在 Ventus/乘影 GPGPU 上完成一套功能相近的 DMA 模块，包括指令设计、Chisel RTL 实现和基本功能验证。

当前项目更适合定位为前作基础上的“功能完善、微结构优化与端到端验证”：

> 面向开源 GPGPU 的异步 DMA/TMA 数据搬运机制，完善张量拷贝边界语义、共享内存写入路径、no-cache 配置支持和端到端验证体系，并对 TMA 地址生成与 swizzle 机制进行优化实现。

可以避免把论文写成“重新实现一遍 DMA 模块”。重点应放在：

- 当前 DMA/TMA 模块在真实 OpenCL 程序、编译器、runtime、RTL pipeline、L2/cache、shared memory、scheduler 中的端到端行为。
- TMA 的复杂语义补全：2D/3D tensor、subbox、elementStride、OOB fill、swizzle。
- DMA fence 的多 warp 异步同步正确性。
- shared bank conflict 场景下 DMA response routing 的正确性。
- with-cache 与 no-cache 两种 GVM 配置下 DMA/TMA 通路的统一支持。
- 测试矩阵和验证方法，而不只是几个波形截图。

## 2. 建议题目

可选题目方向：


1、《开源 GPGPU 张量内存加速器的语义构建与验证体系设计》



## 3. 核心贡献点写法

论文摘要和引言中可以归纳为四点贡献：

1. 完善 DMA/TMA 端到端测试体系  
   在 OpenCL application 层构建 directed testcase，使测试覆盖编译链、POCL/OpenCL runtime、RTL pipeline、DMA core、L2/cache、shared memory 与 fence scheduler，而不只停留在 Chisel unit test。

2. 补全 TMA 张量访存复杂语义  
   支持和验证 1D/2D/3D tensor copy、subbox、dim0/dim1 elementStride、OOB zero/fill、不同 dataType 宽度下的填充值处理，以及 descriptor-driven swizzle。

3. 修复并优化 RTL DMA/TMA 数据通路  
   重点包括 `DMA_core.scala` 中 TMA OOB/subbox 交互修复、张量地址生成优化、shared 写入地址计算优化、`sourceTag` response routing，以及 no-cache GVM 中 DMA cacheline request 到 dcache bypass 的适配。

4. 建立 Spike 到 GVM 的分层验证流程  
   先用 Spike 排除 testcase/model 问题，再用 GVM 验证 RTL 行为；最终覆盖 with-cache 和 no-cache 两类后端，并记录已知 checker noise 与 host verdict 的关系。

## 4. 章节结构建议

### 第 1 章 引言

本章目标是说明为什么要研究 DMA/TMA。

建议结构：

- 研究背景：AI/HPC workload 同时具有高计算量和高访存量，GPGPU 性能受存储层次和数据搬运效率影响。
- 异步数据搬运的意义：把 global memory 到 shared memory 的数据搬运从计算线程中解耦，用计算掩盖访存延迟。
- NVIDIA Hopper TMA 的启发：TMA 支持多维 tensor copy、descriptor、OOB fill、swizzle 等能力。
- 开源 GPGPU 的研究价值：Ventus/乘影作为开源 RISC-V GPGPU 平台，适合研究可公开复现的 GPU 微结构。
- 本文问题：前作已实现基础 DMA/TMA，但复杂 tensor 语义、端到端 RTL 场景、no-cache 配置和系统化验证仍需完善。

可写成如下研究目标：

> 本文围绕 Ventus GPGPU 中异步 DMA/TMA 数据搬运机制，完善其张量边界语义和共享内存写入路径，补齐 with-cache/no-cache 后端支持，并构建覆盖复杂访存场景的端到端验证体系。

### 第 2 章 相关背景与研究现状

本章可承接前作，但不要大篇幅复述。重点服务于你的工作。

建议写：

- GPGPU 存储层次：global memory、L2/cache、shared memory、warp scheduler。
- 异步拷贝机制：`cp.async` 类指令、DMA 和 fence 的关系。
- Hopper TMA：tensor map/descriptor、1D 到 5D tensor copy、OOB fill、swizzle、异步执行。
- Ventus/乘影平台：RISC-V custom instruction、OpenCL 编译执行流程、Spike 与 GVM 两类验证模型。
- 前作基础：已有 `cp.dma`、`cp.dma.bulk`、`cp.dma.tensor`、`cp.dma.fence` 指令设计和基本 RTL。
- 当前工作切入点：不是重新设计 ISA，而是完善语义、修复边界问题、优化地址生成、补齐端到端验证。

和参考论文的差异：

- 前作第 2 章偏 Hopper 架构介绍。
- 你的第 2 章应压缩 Hopper 介绍，把篇幅转给“异步 DMA/TMA 在 RTL 验证中容易出错的场景”，例如 OOB、subbox、多 warp fence、shared bank conflict。

### 第 3 章 Ventus DMA/TMA 系统架构

这一章建议写当前项目中的整体路径，让读者知道一条 DMA 指令如何穿过系统。

建议图：

```text
OpenCL kernel
  -> inline asm CP_ASYNC_*
  -> Ventus clang / object
  -> POCL runtime
  -> issue stage
  -> DMA_core
  -> L2/cache or no-cache bypass
  -> Temp_mem
  -> Addrcalc_shared
  -> SharedMemory
  -> fence_end_dma
  -> warp scheduler
```

可分小节：

- 3.1 软件执行路径：OpenCL testcase、inline asm、descriptor 构造、host expected model。
- 3.2 指令与译码：`CP_ASYNC_COPYSIZE`、`CP_ASYNC_BULK`、`CP_ASYNC_TENSOR`、`CP_ASYNC_FENCE`。
- 3.3 RTL 数据路径：`DMA_core`、`AddrCalc_l2cache`、`Temp_mem`、`Addrcalc_shared`。
- 3.4 同步路径：`fence_end_dma`、per-warp inflight counter、warp fence wait。
- 3.5 验证后端：Spike 同步模型、GVM RTL 模型、with-cache/no-cache 差异。

当前项目可引用文件：

- `gpgpu/ventus/src/pipeline/DMA_core.scala`
- `gpgpu/ventus/src/pipeline/warp_schedule.scala`
- `gpgpu/ventus/src/top/GPGPU_top.scala`
- `gpgpu/ventus/src/top/GPGPU_top_nocache.scala`
- `spike/riscv/insns/cp_async_tensor.h`

### 第 4 章 TMA 语义完善与 RTL 实现

这是论文技术主体之一。

建议主线：

1. 说明 tensor copy 的参数  
   包括 `dataType`、`tensorRank`、`globalAddress`、`globalDim`、`globalStrides`、`boxDim`、`elementStrides`、`BoxAddress`、`swizzleMode`、`oobfill`。

2. 说明旧问题  
   只按 box row bounds 判断 OOB 会导致：
   - dim0 OOB 时可能泄漏下一行数据；
   - subbox 场景可能被误判为 OOB；
   - dim1/dim2 高维 OOB 可能读取 backing buffer 后续 pattern，而不是 zero/fill。

3. 说明修复思想  
   同时维护两类边界：
   - box row mask：决定 requested bytes 中哪些需要写回 shared。
   - full tensor validity：决定 payload 是真实数据还是 OOB fill value。

4. 说明关键实现  
   - `tensor_dim0_start` 表示当前 box row 对应的 full tensor dim0 row 起点。
   - `tensor_high_dim_valid` 表示 dim1 及以上维度是否仍在 `globalDim` 范围内。
   - Temp_mem 根据 dataType 宽度和 oobfill 生成 zero/all-one fill。

5. 说明 swizzle  
   当前实现支持 none/32B/64B/128B swizzle。可以给出公式：

```text
span       = 32B, 64B, or 128B
chunk_bits = 1, 2, or 3
chunk      = (logical_offset >> 4) & ((1 << chunk_bits) - 1)
chunk'     = chunk ^ row_low
low4       = logical_offset[3:0]
swizzled   = high | (chunk' << 4) | low4
```

重点强调：RTL、Spike 模型和 host expected model 使用同一语义，从而避免“测试和硬件各说各话”。

### 第 5 章 DMA 微结构优化与 no-cache 支持

本章写当前项目区别于前作的工程深度。

建议分为三部分。

#### 5.1 TMA 地址生成优化

写作重点：

- 旧路径容易在 per-cacheline issue 阶段反复做复杂多维计算。
- 当前优化把 descriptor/setup 解码和 per-cacheline issue 拆开。
- setup 阶段预计算 row span、source strides、shared strides、subbox offset、copy size。
- issue 阶段只推进寄存器保存的 row state 和 high-dimension carry。
- dim state 只在当前 tensor row 的最后一个 cacheline 发出后推进，避免跨 cacheline row 时提前进位。

可写成“减少组合路径复杂度，提高状态机语义清晰度，同时修复 3D subbox 跨 cacheline 的进位问题”。

#### 5.2 shared memory response routing

问题背景：

- 普通 pipeline shared request 和 DMA shared write 共用 SharedMemory。
- 如果 bank conflict replay 中丢失请求来源，response 可能被错误送回 pipe 或 DMA。

当前设计：

- 请求携带 `sourceTag`。
- `sourceTag=false` 表示普通 pipeline 请求。
- `sourceTag=true` 表示 DMA 请求。
- 该 tag 贯穿 shared memory pipeline 和 replay，response 根据 tag demux。

验证场景：

- `dma_shared_routing_conflict_test`
- 通过普通 shared same-bank conflict 和 DMA response 交叠，检查 payload、checksum 和是否 hang。

#### 5.3 no-cache GVM DMA 通路

问题背景：

- no-cache 配置移除 L1 DCache/L2 cache，但仍保留 SMEM。
- 旧 RTL 中 DMA cache request 被 tie-off，并断言不支持 DMA instruction。

当前实现：

- 将 DMA 的 128B cacheline `Get` 请求映射为 no-cache C++ physical-memory bypass 可处理的 `DCacheCoreReq_np`。
- 使用 response route FIFO 保存 DMA source/address，response 返回后恢复成 `DCacheMemRsp` 给 `DMA_core`。
- SMEM 侧复用 with-cache 的 pipe/DMA 二路仲裁和 `sourceTag` response routing。

这一节可以作为很强的工程贡献：说明设计不是只在一种模拟配置下跑通，而是补齐了平台中另一条后端路径。

### 第 6 章 端到端验证设计

这一章应是你的论文亮点之一。建议不要只放“运行通过”，而是讲清楚为什么这些 case 能覆盖风险。

建议先说明验证策略：

```text
Spike:
  用于验证 testcase、descriptor 构造、host expected model 和指令语义。

GVM:
  用于验证 RTL pipeline、DMA core、cache/shared memory、scheduler 和后端连接。

with-cache / no-cache:
  用于验证不同 memory backend 下 DMA/TMA 路径的一致性。
```

然后按测试集展开。

#### 6.1 TMA matrix test

文件：

- `testcases/_get_case/tma_matrix_test/tma_matrix_test.c`
- `testcases/_get_case/tma_matrix_test/tma_matrix_test.cl`

覆盖：

- 1D/2D/3D tensor copy
- 2D/3D subbox
- dim0/dim1 elementStride
- dim0/dim1/dim2 OOB
- OOB zero 与 floating all-one fill
- subbox + OOB
- elementStride + OOB
- swizzle32/64/128

推荐表格：

| 风险点 | 对应用例 | 验证目标 |
| --- | --- | --- |
| 3D subbox | `FP32_3D_subbox_6x6x6_at_1_1_1` | 高维 offset 和 slice stride |
| dim1 stride | `FP32_3D_estride2_dim1` | 非 dim0 elementStride |
| dim0 OOB | `FP32_2D_oob_zero_dim0` | 不泄漏下一行数据 |
| dim1 OOB | `FP32_2D_oob_zero_dim1` | 高维越界 zero-fill |
| OOB fill | `FP16_2D_oob_fill` | 浮点类 all-one fill |
| swizzle | `FP32_2D_swizzle32_rows` 等 | shared layout swizzle |

#### 6.2 bulk DMA matrix test

文件：

- `testcases/_get_case/bulk_dma_matrix_test/`

覆盖：

- `bulk_32B_at_120`
- `bulk_8B_at_124`
- `bulk_192B_aligned`
- `bulk_64B_dst_offset`

重点说明 128B cacheline 边界，因为当前项目中 `l2cacheline = 128B`。

#### 6.3 multi-warp DMA fence test

文件：

- `testcases/_get_case/multi_warp_dma_fence_test/`

覆盖：

- 同一 workgroup 多 warp 并发 DMA。
- 每个 warp 独立 inflight counter。
- `CP_ASYNC_FENCE` 只阻塞对应 warp，不错误阻塞其他 warp。
- 多 DMA 后 single fence 的 completion accounting。

#### 6.4 shared routing conflict test

文件：

- `testcases/_get_case/dma_shared_routing_conflict_test/`

覆盖：

- DMA shared write 与普通 shared access 交叠。
- bank conflict replay 下 `sourceTag` 不丢失。
- 同时验证 DMA payload 和普通 shared conflict checksum。

#### 6.5 验证结果

可以采用当前项目已有结果：

| Backend | `tma_matrix_test` | `bulk_dma_matrix_test` | `multi_warp_dma_fence_test` | `dma_shared_routing_conflict_test` |
| --- | --- | --- | --- | --- |
| Spike | 15 pass / 0 fail / 8 skip | 4 pass / 0 fail | 4 pass / 0 fail | 2 pass / 0 fail |
| GVM with-cache | 23 pass / 0 fail / 0 skip | 4 pass / 0 fail | 4 pass / 0 fail | 2 pass / 0 fail |
| GVM no-cache | 23 pass / 0 fail / 0 skip | 4 pass / 0 fail | 4 pass / 0 fail | 2 pass / 0 fail |

说明：

- Spike 中 skip 的主要原因是部分 RTL-directed OOB/stride case 不适合在同步简化模型中直接验证。
- GVM 日志中可能存在 PC `0x80000014` XREG mismatch 或 OOB 场景 VREG mismatch 等 reference checker noise，但 host verdict 为 `PASS/OK`，不作为功能失败。

### 第 7 章 总结与展望

总结不要写成“做了测试”。建议归纳为：

- 完善了 Ventus DMA/TMA 的复杂张量拷贝语义。
- 优化了 TMA 地址生成和 shared 写入路径。
- 支持了 no-cache 后端的 DMA/TMA 端到端通路。
- 构建了覆盖多类边界场景的 OpenCL application-level directed test suite。

展望可以写：

1. Descriptor prefetch  
   当前 `CP_ASYNC_TENSOR` 仍主要依赖 VGPR 参数模式。后续可改为 descriptor-addressed 模式：指令只携带 descriptor pointer、dynamic coords pointer 和 shared dst，DMA 前端先从 global memory prefetch descriptor，再进入现有 TMA copy 流程。

2. Descriptor cache  
   为避免每次 TMA 都读取 128B descriptor，可设计 1-entry 或多 entry descriptor buffer，提高小 tile 场景收益。

3. SMEM->L2 反向 DMA  
   当前主要实现 global/L2 -> SMEM。后续可支持 shared memory 到 global/L2 的异步写回，涉及 SMEM read producer、TL PutFull/PutPartial、write response completion 和 cache coherency。

4. 性能评估  
   当前验证以功能正确性为主，后续可加入矩阵乘、卷积或 stencil 等 kernel，比较普通 load/store 与 DMA/TMA 的执行周期、访存 stall、barrier stall 和 overlap 效果。

## 5. 和前作论文的差异化表达

前作已有内容：

- Hopper/TMA 背景。
- Ventus 软件/硬件架构。
- `cp.dma`、`cp.dma.bulk`、`cp.dma.tensor`、`cp.dma.barrier/fence` 指令设计。
- `DMA_core` 基础模块，包括 `AddrCalc_l2cache`、`TempMem`、`AddrCalc_shared`。
- 基础功能波形验证。

你的论文应避免重复堆这些内容。可以这样对比：

| 维度 | 前作重点 | 当前论文重点 |
| --- | --- | --- |
| 目标 | 从 0 到 1 实现 DMA/TMA | 完善复杂语义、优化 RTL、系统验证 |
| 验证 | 模块/波形级功能验证 | OpenCL application 端到端矩阵验证 |
| TMA 语义 | 基础 tensor copy | subbox、elementStride、OOB、swizzle |
| 同步 | 基础 barrier/fence | 多 warp、多 DMA、per-warp fence |
| shared memory | 基础写入路径 | bank conflict 下 response routing |
| 后端 | 主要 with-cache | with-cache 与 no-cache 双路径 |
| 展望 | 增加功能、优化、性能验证 | descriptor prefetch、SMEM->L2、性能量化 |

## 6. 摘要草稿

可以后续改写成正式摘要：

> 随着人工智能和高性能计算负载对访存带宽和数据搬运效率的要求不断提高，异步数据拷贝机制成为 GPGPU 隐藏访存延迟、提升片上存储利用率的重要手段。NVIDIA Hopper 架构中的 Tensor Memory Accelerator 支持多维张量数据在全局内存和共享内存之间异步搬运，为开源 GPGPU 架构提供了重要参考。本文基于 Ventus 开源 GPGPU 平台，对已有 DMA/TMA 模块进行语义完善、RTL 优化和端到端验证。本文首先分析 Ventus 中 DMA/TMA 指令的执行路径和同步机制，随后针对张量拷贝中的 subbox、elementStride、OOB fill 和 swizzle 等复杂语义完善 RTL 实现，并优化 TMA 地址生成和 shared memory 写入路径。同时，本文补齐 no-cache GVM 后端中的 DMA/TMA 通路，并通过 `sourceTag` 机制保证 shared bank conflict 场景下 DMA 与普通 pipeline 响应的正确路由。最后，本文构建了覆盖 TMA、bulk DMA、多 warp fence 和 shared routing conflict 的 OpenCL application-level directed test suite，采用 Spike 到 GVM 的分层验证流程，在 with-cache 和 no-cache 后端下完成端到端验证。实验结果表明，所设计和完善的 DMA/TMA 机制能够正确支持复杂张量数据搬运场景，为后续 descriptor prefetch、反向 DMA 和性能优化奠定基础。

## 7. 需要准备的图表

建议论文中至少准备这些图表：

- Ventus DMA/TMA 端到端执行路径图。
- `DMA_core` 内部模块图：InputFIFO、AddrCalc_l2cache、Temp_mem、Addrcalc_shared。
- TMA tensor、box、subbox、globalDim、boxDim 的关系图。
- OOB fill 判定示意图：box row mask 与 full tensor validity。
- TMA 地址生成状态推进图。
- shared memory `sourceTag` response routing 图。
- no-cache DMA request adapter 图。
- 测试覆盖矩阵表。
- Spike/GVM 验证结果表。

## 8. 写作优先级

建议先写：

1. 第 6 章验证设计  
   因为当前项目材料最完整，表格和 case 名称都明确。

2. 第 4 章 TMA 语义完善  
   这是技术贡献最核心的一章。

3. 第 5 章 RTL 优化与 no-cache 支持  
   这章体现工程深度。

4. 第 3 章系统架构  
   写作时从代码和图反推即可。

5. 第 1、2、7 章  
   最后统一语言和贡献表述。

这样写的好处是先把“你到底做了什么”固定下来，再去包装背景和意义，避免绪论写得很大但正文撑不住。

---

# 详细版：按当前项目实际情况来写

上面那版更像大纲，这一版按当前代码和 testcase 来讲。论文真正要写出味道，不能只写“本文完善了 DMA/TMA 模块”，而是要讲清楚：原来哪里不够，项目里具体怎么补，最后怎么证明它补对了。

这篇论文最稳的定位是：

> 前作已经把 Ventus 的 DMA/TMA 从无到有做出来了；当前工作是在这个基础上，把复杂张量拷贝、RTL 连接路径和端到端测试补完整，让它不只是简单 demo 能跑，而是在一批专门构造的边界场景里也能跑对。

这个定位很重要。不要把论文写成“我设计了一个全新的张量内存加速器”。你现在项目里真正有价值的东西，是把一个已经存在的 DMA/TMA 模块往可用、可测、可维护推进了一步。

## 1. 论文主线怎么讲

可以用一句话贯穿全文：

> 本文围绕 Ventus GPGPU 中已有 DMA/TMA 模块，补全复杂 tensor copy 语义，修复 with-cache/no-cache RTL 路径中的连接问题，并构建从 OpenCL 程序到 GVM RTL 的端到端验证体系。

再拆成人话就是三件事：

1. **TMA 搬 tensor 时，哪些数据该搬、哪些地方该填 0，要算清楚。**  
   这对应 subbox、elementStride、OOB fill、swizzle。

2. **DMA 不是孤立模块，接到 GPU 里之后，cache、shared memory、scheduler 都要配合。**  
   这对应 no-cache DMA 通路、shared `sourceTag` routing、多 warp fence。

3. **不能只看波形说对了，要让 OpenCL 程序真的跑一遍。**  
   这对应 `tma_matrix_test`、`bulk_dma_matrix_test`、`multi_warp_dma_fence_test`、`dma_shared_routing_conflict_test`。

## 2. 当前项目里真实做了什么

### 2.1 TMA 参数和当前实现方式

当前项目里的 `CP_ASYNC_TENSOR` 不是 NVIDIA 那种完整的 tensor map prefetch 模式。它现在是 VGPR 参数模式：

- host 端先构造一个 96 word descriptor。
- kernel 把 descriptor 拷到 `__local param_buf[96]`。
- kernel patch 三个运行时地址：
  - `desc_src[2]`：`globalAddress`
  - `desc_src[32]`：`BoxAddress`
  - `desc_src[64]`：shared buffer 地址
- kernel 用 `vlw12.v` 把 descriptor 分三段灌到 VRS1/VRS2/VRS3。
- 最后用 `.word 0x00C535C2` 发 `CP_ASYNC_TENSOR`，再用 `.word 0x00004042` 发 `CP_ASYNC_FENCE`。

这段逻辑在：

- `testcases/_get_case/tma_matrix_test/tma_matrix_test.cl`

论文里一定要说清楚这个现状。不要写成“本文实现了 descriptor prefetch”。descriptor prefetch 应该放到展望里。

### 2.2 DMA_core 里有哪些模块

当前 `DMA_core.scala` 里主路径是：

```text
InputFIFO
  -> AddrCalc_l2cache
  -> Temp_mem
  -> Addrcalc_shared
  -> SharedMemory
  -> fence_end_dma
```

可以按下面这种说法写：

- `AddrCalc_l2cache`：从 DMA 指令中拿参数，算 global memory 里要读哪些 128B cacheline，并向 L2/TLB 发请求。
- `Temp_mem`：等 L2 response 回来，把 response、tag 和原始 DMA 指令配起来，生成有效 mask，处理 OOB fill，并记录这条 DMA 还有多少数据没写完。
- `Addrcalc_shared`：根据 Temp_mem 给的数据和 mask，算 shared memory 地址，处理 tensor packed layout 和 swizzle，然后发 shared 写请求。
- `warp_schedule`：通过每个 warp 的 inflight counter 支持 `CP_ASYNC_FENCE`。

论文里的图可以画成：

```text
OpenCL inline asm
    |
    v
issueX.out_DMA
    |
    v
DMA_core
    |-- AddrCalc_l2cache -- DMA TLB -- L2/cache or no-cache bypass
    |-- Temp_mem --------- inst/tag/data/finish_cnt
    |-- Addrcalc_shared -- SharedMemory
    |
    v
fence_end_dma -> warp scheduler
```

## 3. TMA 语义：这部分要写细

TMA 难点不是“读一段内存”。如果只是连续地址，那就是 bulk DMA。TMA 真正麻烦的是它搬的是一个多维 box。

当前 `TensorVars` 包括：

| 字段 | 含义 |
| --- | --- |
| `globalAddress` | 整个 tensor 的起始地址 |
| `globalDim[0..4]` | tensor 每一维真实大小 |
| `globalStrides[0..4]` | 高维地址步长 |
| `BoxAddress` | 本次 copy 的 box 起点 |
| `boxDim[0..4]` | 本次 copy 的 box 大小 |
| `elementStrides[0..4]` | 每一维按什么步长取元素 |
| `dataType/datawidth` | 元素类型和字节数 |
| `oobfill` | 越界位置填 0 还是浮点 all-one |
| `swizzleMode` | 搬到 shared 后是否做 32B/64B/128B swizzle |

### 3.1 subbox 为什么容易错

subbox 指的是：本次 copy 不是从 tensor 左上角开始，而是从 tensor 中间某个位置开始。

关键量是：

```text
BoxAddress - globalAddress
```

这个差值不能只当成一个线性 byte offset。对 2D/3D tensor 来说，它要被拆成：

- dim0 offset
- dim1 offset
- dim2 offset
- 更高维 offset

当前 RTL 里做了这个拆分：

- 用 `box_to_global_offset = BoxAddress - globalAddress` 得到 box 相对 tensor 的偏移。
- 用 `globalStrides` 从高维到低维拆出 `box_offset_elems_setup(d)`。
- 用 `tensor_base_global_pos_reg` 和 `tensor_global_pos_reg` 保存当前 row 对应的真实 tensor 坐标。

论文里可以这样解释：

> subbox 让 box 的局部坐标和 tensor 的全局坐标不再相同。硬件地址生成时既要按 box 的形状遍历，又要知道当前元素在原 tensor 中的全局坐标，否则后续 OOB 判断会出错。

### 3.2 OOB fill 是最值得写的 bug

OOB 是指：copy box 可能超过真实 tensor 边界。超过边界的地方，不能从 backing buffer 继续读数据，而是要填 0 或 fill value。

这里有两个范围：

- box 范围：这次用户要求搬哪些位置。
- tensor 范围：真实 tensor 中哪些位置合法。

一个位置如果在 box 里，但不在 tensor 里，就应该写 fill。

旧逻辑的问题可以这样讲：

> 旧实现主要根据 box row 的范围判断当前 cacheline 中哪些 byte 有效。这个判断只能说明“这个 byte 属于本次请求的 box”，不能说明“这个 byte 对应真实 tensor 中的合法元素”。当 box 超过 tensor 边界时，硬件可能把 source buffer 后面的 pattern 当成合法数据搬入 shared。

当前项目里有非常具体的失败例子：

- `FP32_2D_oob_zero_dim1` 在旧 RTL 上失败。
- 失败位置 byte 64。
- expected 是 `0x00`，got 是 `0x87`。
- 这说明 dim1 越界时，RTL 没有填 0，而是继续读了后面的 source pattern。

这个例子建议写进论文。它很直观。

当前修复思路：

1. `AddrCalc_l2cache` 给每个 cacheline tag 记录：
   - `box_dim0_start`
   - `tensor_dim0_start`
   - `tensor_high_dim_valid`
   - `shared_row_base`
   - `dim0_stride_bytes`

2. `Temp_mem` 收到 L2 response 后，对每个元素判断：
   - 是否在 box row 范围内；
   - 是否在真实 tensor dim0 row 范围内；
   - dim1/dim2/dim3/dim4 是否还小于 `globalDim`；
   - dim0 stride 下是不是被选中的元素。

3. 如果合法，就用 L2 payload；否则按 data type 生成 fill。

可以写成伪代码：

```text
in_bounds =
  in_box_row &&
  in_tensor_dim0_row &&
  tensor_high_dim_valid &&
  selected_by_elementStride0

if in_bounds:
  shared_data = l2_payload
else:
  shared_data = fill_value(dataType, oobfill)
```

填充值规则：

- 整数类型：填 0。
- 浮点类型，`oobfill=0`：填 0。
- 浮点类型，`oobfill=1`：填全 1 bit。

### 3.3 elementStride 不只是 dim0

当前项目测试里有：

- `FP32_2D_estride2_cols`
- `FP32_3D_estride2_dim1`
- `FP32_2D_estride2_rows_cols`
- `FP32_2D_oob_estride_dim1`

这些 case 的意义不同：

- dim0 stride 检查一行里跳着取元素。
- dim1 stride 检查高维 row 推进是否按 stride 走。
- OOB + stride 检查 stride 后的全局坐标是否参与 OOB 判断。

论文里要说明：

> 如果只处理 dim0 stride，2D/3D tensor 的高维 stride 会被漏掉。当前实现中，高维推进条件使用 `tensor_dim_pos_reg(d) + elementStrides(d) < boxDim(d)`，并用 `tensor_global_pos_reg(d)` 参与 OOB 判断，从而让 stride 后的坐标进入边界检查。

### 3.4 swizzle 怎么写

当前实现支持：

- none
- 32B swizzle
- 64B swizzle
- 128B swizzle

测试 case：

- `FP32_2D_swizzle32_rows`
- `FP32_2D_swizzle64_rows`
- `FP32_2D_swizzle128_subbox_row1`

swizzle 不改变从 global 读哪些数据，只改变写到 shared 的相对位置。

可以用这段公式讲：

```text
span       = 32B, 64B, or 128B
chunk_bits = 1, 2, or 3
chunk      = (logical_offset >> 4) & ((1 << chunk_bits) - 1)
chunk'     = chunk ^ row_low
low4       = logical_offset[3:0]
swizzled   = high | (chunk' << 4) | low4
```

论文里不用展开太多 NVIDIA 细节，重点讲当前项目：

> `Addrcalc_shared` 先得到 tensor copy 的 logical shared address，再根据 `swizzleMode` 和当前 row 的低位计算 swizzled address。host expected model 和 Spike 模型也实现同样公式，保证验证时三方语义一致。

## 4. RTL 路径：不能只写 DMA_core

DMA 指令要跑通，不只是 `DMA_core.scala` 对。它还依赖顶层连接、shared memory、scheduler。

### 4.1 with-cache 路径

在 `GPGPU_top.scala` 中：

- DMA 的 L2 request 接到 `l1Cache2L2Arb.io.memReqVecIn.get(2)`。
- L2 response 回到 `pipe.io.dma_cache_rsp`。
- DMA shared request 和普通 pipe shared request 经过 arbiter 进入 `SharedMemory`。

可以这样写：

> with-cache 配置下，DMA 使用独立的 L2 request 端口读取 cacheline 数据，但写 shared memory 时与普通 pipeline 共享同一个 SharedMemory。因此系统既要保证 DMA L2 response 能回到 DMA_core，也要保证 shared response 能根据请求来源正确返回。

### 4.2 no-cache 路径

这是当前项目很重要的点。

旧问题：

- no-cache 后端没有完整 cache 层级。
- 原先 DMA request 被 tie-off。
- 跑 DMA/TMA 时会触发 “DMA instructions not supported in this configuration”。

当前项目做法：

- 把 DMA `DCacheMemReq_p` 的 128B cacheline `Get` 转成 no-cache 后端能处理的 `DCacheCoreReq_np`。
- 普通 dcache request 和 DMA request 进同一个 arbiter。
- 用 `dcacheRspRouteQ` 记录 response 属于普通 pipeline 还是 DMA。
- 如果属于 DMA，就恢复成 `DCacheMemRsp`，填回 `d_source`、`d_addr`、`d_data`。

论文里可以画：

```text
DMA_core
  -> pipe.io.dma_cache_req
  -> dmaDcacheReq
  -> dcacheReqArb
  -> no-cache memory bypass
  -> dcacheRspRouteQ
  -> pipe.io.dma_cache_rsp
```

这部分的意义：

> no-cache 支持不是为了性能，而是为了让同一套 directed test 在不同 GVM 后端下都能验证，避免 DMA/TMA 只在一种配置下看起来正确。

### 4.3 shared response routing

问题：

- 普通 pipeline 会访问 shared memory。
- DMA 也会写 shared memory。
- 两者共用 `SharedMemory`。
- 如果发生 bank conflict，SharedMemory 会 replay。
- replay 之后 response 还必须知道自己属于普通 pipe 还是 DMA。

解决：

- `ShareMemCoreReq` 中加入 `sourceTag`。
- `false` 表示 pipe。
- `true` 表示 DMA。
- `SharedMemory` 中 `grantMeta.sourceTag := activeReq.sourceTag`。
- response 出来时继续带 `sourceTag`。
- 顶层根据 `sourceTag` 分发 response。

论文里可以写：

> `sourceTag` 的关键不是简单地在入口处打个标记，而是这个标记必须穿过 bank conflict arbitration 和 response pipeline。否则在 replay 场景下，后续 response 仍然可能丢失来源。

### 4.4 多 warp fence

`warp_schedule.scala` 中维护：

- `dma_inflight_cnt`
- `dma_fence_wait`
- `dma_issue_allow`

逻辑：

- DMA issue：对应 warp inflight 加一。
- DMA complete：对应 warp inflight 减一。
- `CP_ASYNC_FENCE`：如果 inflight 不为 0，设置该 warp 的 wait bit。
- inflight 归零后，清 wait bit。
- wait bit 参与 `warp_ready`，只阻塞对应 warp。

论文要强调 per-warp：

> DMA fence 不能做成全局阻塞。warp0 在等自己的 DMA 时，warp1 如果没有 fence 或者自己的 DMA 已完成，就应该继续调度。当前 scheduler 通过每个 warp 独立的 inflight counter 实现这一点。

## 5. 测试设计要写成论文亮点

当前项目的测试不是“跑了几个程序”，而是有明确覆盖目标。

### 5.1 为什么要 application-level test

Chisel unit test 能验证模块局部逻辑，但不够。DMA/TMA 真正出问题，往往发生在这些地方：

- kernel inline asm 参数不对；
- descriptor 构造和 RTL 解读不一致；
- Spike、host expected model、RTL 三方语义不一致；
- L2 response source/tag 配错；
- shared memory response 送错对象；
- fence completion 的 warp id 错；
- no-cache 后端没有接 DMA 路径。

所以要从 OpenCL 程序跑起：

```text
OpenCL host
  -> build descriptor/source pattern
  -> launch kernel
  -> inline asm issue DMA
  -> runtime/GVM/RTL
  -> read back result
  -> host byte compare
```

### 5.2 TMA matrix test

目录：

- `testcases/_get_case/tma_matrix_test/`

kernel 做的事：

- thread 0 patch descriptor；
- load VRS1/VRS2/VRS3；
- 发 `CP_ASYNC_TENSOR`；
- 发 `CP_ASYNC_FENCE`；
- 把 shared buffer 写回 global。

host 做的事：

- 构造 descriptor；
- 构造 source pattern；
- 用 `compute_expected()` 计算 expected；
- 逐 byte 比较结果。

覆盖表可以这样写：

| 类别 | case |
| --- | --- |
| 基础 copy | `FP32_2D_4x4_full`、`FP32_1D_16`、`FP32_3D_2x2x2`、`FP16_2D_8x4` |
| subbox | `FP32_2D_subbox_8x8_at_2_2`、`FP32_3D_subbox_6x6x6_at_1_1_1` |
| stride | `FP32_2D_estride2_cols`、`FP32_3D_estride2_dim1`、`FP32_2D_estride2_rows_cols` |
| OOB | `FP32_2D_oob_zero_dim0`、`FP16_2D_oob_fill`、`FP32_2D_oob_zero_dim1`、`FP32_3D_oob_zero_dim2` |
| 组合场景 | `FP16_2D_oob_subbox_dim0_dim1_fill`、`FP32_2D_oob_estride_dim1` |
| swizzle | `FP32_2D_swizzle32_rows`、`FP32_2D_swizzle64_rows`、`FP32_2D_swizzle128_subbox_row1` |

### 5.3 bulk DMA matrix test

目录：

- `testcases/_get_case/bulk_dma_matrix_test/`

当前 cacheline 是 128B，所以 bulk DMA 的重点是跨 cacheline。

| case | `src_offset` | `copy_bytes` | `dst_offset` | 覆盖点 |
| --- | ---: | ---: | ---: | --- |
| `bulk_32B_at_120` | 120 | 32 | 0 | 跨 128B cacheline |
| `bulk_8B_at_124` | 124 | 8 | 0 | 最小跨线 |
| `bulk_192B_aligned` | 0 | 192 | 0 | 多 cacheline |
| `bulk_64B_dst_offset` | 64 | 64 | 16 | shared dst 非 0 |

这组测试说明：

- DMA 能拆多个 cacheline request；
- L2 response 能按 source/tag 回来；
- shared 写入偏移正确；
- completion 计数正确。

### 5.4 multi-warp DMA fence test

目录：

- `testcases/_get_case/multi_warp_dma_fence_test/`

case：

| case | 覆盖点 |
| --- | --- |
| `2warp_1dma_each_fence` | 两个 warp 各一条 DMA |
| `2warp_2dma_each_single_fence` | 每个 warp 两条 DMA，一个 fence 等全部完成 |
| `4warp_1dma_each_fence` | 4 个 warp 并发 |
| `2warp_cross_cacheline_each` | 多 warp + 跨 cacheline |

项目细节：

- kernel 用 `CSR_WID = 0x805` 读 warp id。
- 每个 warp leader 根据 warp id 计算自己的 src segment 和 shared segment。
- 1-DMA 和 2-DMA 拆成两个 kernel，避免 GVM 对 runtime branch 包住 custom DMA/fence 序列时出现 reference PC fatal。

这个细节建议写，因为它能说明测试是按当前平台特性认真设计过的。

### 5.5 shared routing conflict test

目录：

- `testcases/_get_case/dma_shared_routing_conflict_test/`

case：

| case | copy bytes | rounds | src offset | 覆盖点 |
| --- | ---: | ---: | ---: | --- |
| `routing_64B_conflict64` | 64 | 32 | 0 | DMA shared write 和普通 shared conflict 交叠 |
| `routing_192B_crossline_conflict64` | 192 | 64 | 120 | 跨 cacheline DMA + shared bank conflict |

kernel 里：

- `dma_shared` 和 `conflict` 都从同一个 `__local uint scratch[]` 手动分区。
- `conflict[lid * 32]` 让多个 lane 落到同一个 bank，制造 bank conflict replay。
- host 同时检查 DMA payload 和 conflict checksum。

失败现象可以分三类：

- DMA payload 错：DMA response routing 或 shared 写坏了。
- conflict checksum 错：普通 shared response routing 或 replay 坏了。
- kernel hang：ready/valid 或 fence completion 卡住。

## 6. 验证结果怎么写

验证顺序：

1. Spike
2. GVM with-cache
3. GVM no-cache

原因：

- Spike 先排除 testcase 和 expected model 的问题。
- GVM 再看 RTL 和后端连接问题。
- with-cache/no-cache 同时跑，说明不是只在一种后端下碰巧正确。

当前结果可以写：

| Backend | `tma_matrix_test` | `bulk_dma_matrix_test` | `multi_warp_dma_fence_test` | `dma_shared_routing_conflict_test` |
| --- | --- | --- | --- | --- |
| Spike | 15 pass / 0 fail / 8 skip | 4 pass / 0 fail | 4 pass / 0 fail | 2 pass / 0 fail |
| GVM with-cache | 23 pass / 0 fail / 0 skip | 4 pass / 0 fail | 4 pass / 0 fail | 2 pass / 0 fail |
| GVM no-cache | 23 pass / 0 fail / 0 skip | 4 pass / 0 fail | 4 pass / 0 fail | 2 pass / 0 fail |

要补一句：

> GVM 日志中仍可能出现 PC `0x80000014` 的 XREG mismatch，或 TMA OOB 场景下的 VREG mismatch。这些属于当前 GVM reference checker 的已知噪声，未导致 host verdict 失败。本文功能判断以 OpenCL host 端 byte compare 的 `PASS/OK` 为准。

这句很重要。否则论文答辩时老师可能会问：你日志里 mismatch 为什么还能说通过。

## 7. 章节建议

建议目录：

```text
第 1 章 绪论
  1.1 研究背景
  1.2 Ventus DMA/TMA 的前期基础
  1.3 当前存在的问题
  1.4 本文主要工作

第 2 章 GPGPU 异步数据搬运与 Ventus 平台
  2.1 GPGPU 存储层次
  2.2 DMA/TMA 异步搬运机制
  2.3 Ventus 软件与硬件执行流程
  2.4 本文使用的验证后端

第 3 章 Ventus DMA/TMA 执行路径
  3.1 DMA 指令与参数
  3.2 OpenCL kernel 中的指令发射
  3.3 DMA_core 内部结构
  3.4 cache/shared/scheduler 连接

第 4 章 TMA 张量拷贝语义完善
  4.1 descriptor 参数组织
  4.2 subbox 和 elementStride
  4.3 OOB fill
  4.4 swizzle
  4.5 RTL、Spike、host expected model 语义对齐

第 5 章 DMA/TMA RTL 通路完善
  5.1 bulk DMA 跨 cacheline
  5.2 Temp_mem mask 与 completion
  5.3 shared sourceTag routing
  5.4 no-cache DMA 支持
  5.5 多 warp DMA fence

第 6 章 测试设计与验证结果
  6.1 验证方法
  6.2 TMA matrix test
  6.3 bulk DMA matrix test
  6.4 multi-warp DMA fence test
  6.5 shared routing conflict test
  6.6 验证结果分析

第 7 章 总结与展望
```

## 8. 和前作的区别

可以在绪论或总结里用自然语言带出来，不一定放表。

前作重点：

- 设计 DMA/TMA 指令；
- 实现基础 DMA_core；
- 用波形和简单程序验证基本功能。

当前论文重点：

- 修复杂 tensor 语义；
- 处理 OOB/subbox/stride/swizzle；
- 补 no-cache 后端；
- 处理 shared response routing；
- 验证多 warp fence；
- 建立 application-level directed test。

最关键的差别：

> 前作解决“有没有”的问题；本文解决“复杂情况下能不能稳定跑对、怎么证明跑对”的问题。

这句话很适合放到引言最后。

## 9. 不足和展望

不要假装所有功能都做完。当前项目里还没做：

- im2col；
- shared -> global writeback；
- descriptor prefetch；
- tensor core FP16/BF16 相关完整路径；
- 系统性性能评估。

可以写 4 个展望：

### 9.1 Descriptor prefetch

当前 `CP_ASYNC_TENSOR` 用 VGPR 参数模式，参数多，kernel 也比较重。后续可以改成：

```text
CP_ASYNC_TENSOR rd, rs1, rs2

rd  = shared dst
rs1 = descriptor pointer
rs2 = dynamic coords pointer
```

DMA 前端先从 global memory 读取 descriptor，再填 `TensorVars`，然后复用现有 TMA 地址生成和 shared 写入路径。

### 9.2 Descriptor cache

如果每次 TMA 都读 128B descriptor，小 tile 会不划算。可以做 1-entry 或多 entry descriptor buffer，用 descriptor base 判断 hit/miss。

### 9.3 SMEM->L2 反向 DMA

当前主要是：

```text
global/L2 -> shared memory
```

后续可以支持：

```text
shared memory -> global/L2
```

这需要 shared read producer、L2 PutFull/PutPartial、write response completion 和 cache coherency 设计。

### 9.4 性能评估

当前论文以功能正确性为主。后续可以用矩阵乘、卷积、stencil 或 tiled copy kernel，比普通 load/store 和 DMA/TMA 的 cycles、memory stall、barrier stall。

注意：如果没有数据，不要写”性能显著提升”。可以写”为后续性能优化和评估打下基础”。

---

# 最终论文状态与核实记录（2026-05-17）

## 代码核实修正项目

以下不一致已在论文中修正（逐项经RTL/Spike/Testcase代码核实）：

| 修正项 | 原论文值 | 代码实际值 | 修正位置 |
|--------|---------|-----------|---------|
| max_dma_tag | 16 | 8 | 第3章表3.1 |
| max_l2cacheline | 16 | 6 | 第3章表3.1 |
| numgroupshared | 1 | 32 (=num_thread) | 第3章表3.1 |
| TMA_core_test测试数 | 29 | 15 | 第6章6.4节 |
| Chisel测试总数 | 51 | 37 | 第6章、第7章 |
| Spike OOB判定存在 | “三方语义对齐含OOB/fill” | Spike不读取globalDim，无OOB判定 | 第4章4.6节 |
| 2B INT16 fill value | 论文未提及 | 代码只处理UINT16，INT16有误填隐患 | 第4章4.3.3节 |

## 关键数据核实

| 数据项 | 核实值 | 来源 |
|--------|--------|------|
| DMA_core.scala行数 | 1343 | wc -l |
| DMA_core_test.scala行数 | 825 | wc -l |
| TMA_core_test.scala行数 | 1403 | wc -l |
| DMA_fence_scheduler_test.scala行数 | 249 | wc -l |
| TMA Matrix Test case数 | 23 | g_cases[]数组 |
| Bulk DMA case数 | 4 | g_cases[]数组 |
| Multi-warp fence case数 | 4 | g_cases[]数组 |
| Shared routing case数 | 2 | g_cases[]数组 |
| Descriptor TMA case数 | 2 | use_prefetch=0/1 |
| Spike tma_matrix: pass/fail/skip | 15/0/8 | run.log |
| GPIO_top_nocache.scala行数 | 374 | wc -l |
| dcacheRspRouteQ条目数 | 16 | 源代码 |
| tma_desc_cache_entries | 2 | parameters.scala |
| tma_prefetch_slots | 2 | parameters.scala |
