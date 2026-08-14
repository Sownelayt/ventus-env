# Ventus TMA 指令集、TensorMap ABI 与参数约束参考

> 文档状态：基于 2026-08-10 当前工作树中的最新 TMA RTL、Spike 模型、C model 与 OpenCL 测试公共头整理；包含当前尚未提交的 TMA 实现和清理结果。  
> 适用实现：Ventus `TmaV2`。`TmaV2` 是当前正式描述符 ABI 名称，不代表仍同时支持 TMA v1。  
> 语义基准：涉及硬件行为时以 RTL 为准；Spike 和 C model 用于功能参考与回归验证。  
> 说明：本设计使用 CUDA TensorMap 的若干布局编码和完成模型概念，但 `opcode=0x42` 是 Ventus 自定义指令编码，不是 NVIDIA PTX/SASS 二进制编码。
    
## 1. 文档目的与规范用语

本文给出当前 Ventus TMA 子系统可以对外交流的完整接口说明，包括：

- 全部公开 TMA 指令和二级操作组合；
- 指令位域、寄存器操作数、立即数和原始编码；
- Bulk G2S/S2G 的地址与长度约束；
- 128B TensorMap 描述符的逐 word、逐 bit ABI；
- rank、dimension、stride、coordinate、interleave、swizzle 和 OOB 的定义；
- 所有 datatype，包括 B4x16、B4x16P64 和 B6 的存储格式；
- G2S mbarrier、S2G group 和 async proxy fence 的完成语义；
- Bulk/Tensor S2G reduce 的原子操作定义；
- 状态 CSR、错误码和 detail 编码；
- 当前硬件容量、流水并发边界和已知实现注意事项。

本文使用以下规范词：

- **必须**：违反后指令或描述符非法，不能依赖其数据结果；
- **应当**：正式 ABI 或推荐同步序列要求，虽然部分情况下硬件可能仍接受其他写法；
- **可以**：当前实现支持但不是所有用例都需要；
- **保留**：软件必须写零，未来实现可以赋予新语义；
- **不公开**：内部防御路径可能识别，但不能作为软件 ABI 使用。

## 2. 术语与地址空间

| 术语 | 定义 |
|---|---|
| G2S | Global-to-Shared，从 global memory 搬运到当前 work-group 的 shared/local memory |
| S2G | Shared-to-Global，从 shared/local memory 搬运到 global memory |
| Bulk | 不使用 TensorMap，按连续字节区间搬运 |
| Tensor | 使用 128B TensorMap、rank 1–5 坐标和布局规则搬运 |
| TensorMap | 位于 global memory 的 128B 描述符对象 |
| Atom | TMA 内部最小布局/掩码单位，16B |
| Line | TMA 内部 cache/shared 窗口，128B，包含 8 个 16B atom |
| Command | 一条 Bulk/Tensor 数据搬运或 reduce 指令形成的数据面命令 |
| Control command | TensorMap prefetch/invalidate、group 或 mbarrier/proxy 控制操作 |
| Warp | 发出 TMA 指令、拥有 group 和 mbarrier binding 的调度实体 |
| WG | Work-group；每个 resident WG 拥有独立的 mbarrier entry 配额 |

当前实现使用 32-bit Ventus 虚拟地址。TensorMap 中保留 64-bit global base/stride 存储槽，是为了 ABI 布局兼容和未来扩展；当前可访问的最终地址仍必须落在 `0x00000000..0xffffffff`。

## 3. 指令编码总览

### 3.1 公共 opcode

所有 TMA 自定义指令使用：

```text
opcode[6:0] = 0x42 = 0b1000010
```

公共语义主要由 `funct3 = inst[14:12]` 区分：

```text
31          29 28  27 26 25 24       20 19       15 14  12 11       7 6       0
+-------------+------+-----+-----------+-----------+------+-----------+---------+
| reduce/subop| type | rsv | rs2/vrs2 | rs1/zimm  |funct3| rd operand|  0x42   |
+-------------+------+-----+-----------+-----------+------+-----------+---------+
```

这些指令没有通用寄存器写回。数据指令中的 `rd` 字段是一个**被读取的地址寄存器编号**，不是普通 RISC-V 指令意义上的结果寄存器。

### 3.2 公开语义组合计数

| funct3 | 指令族 | 二级选择 | 公开组合数 |
|---:|---|---|---:|
| 1 | Bulk G2S | 无 | 1 |
| 2 | Tensor G2S | 无 | 1 |
| 3 | Bulk S2G/Reduce | `inst[31:29]` redOp、`inst[28:27]` type | 10 |
| 4 | Tensor S2G/Reduce | `inst[31:29]` | 7 |
| 5 | TensorMap control | `inst[31:27]` | 2 |
| 6 | S2G group | `zimm=inst[19:15]` | 5 |
| 7 | mbarrier/proxy | `inst[26:25]` | 4 |

因此当前共有：

```text
1 + 1 + 10 + 7 + 2 + 5 + 4 = 30 种公开语义组合
```

### 3.3 总指令表

| funct3 | 二级字段 | 操作 | 数据方向 | 完成观察方式 |
|---:|---:|---|---|---|
| 1 | — | `bulk.g2s` | global → shared | mbarrier |
| 2 | — | `tensor.g2s` | global → shared | mbarrier |
| 3 | reduce=0,type=0 | `bulk.s2g.copy` | shared → global | S2G group |
| 3 | reduce=1,type=U32 | `bulk.s2g.reduce.add.u32` | shared → global | S2G group，等待最终 AMO ack |
| 3 | reduce=1,type=S32 | `bulk.s2g.reduce.add.s32` | shared → global | S2G group，等待最终 AMO ack |
| 3 | reduce=2,type=U32 | `bulk.s2g.reduce.min.u32` | shared → global | S2G group，等待最终 AMO ack |
| 3 | reduce=2,type=S32 | `bulk.s2g.reduce.min.s32` | shared → global | S2G group，等待最终 AMO ack |
| 3 | reduce=3,type=U32 | `bulk.s2g.reduce.max.u32` | shared → global | S2G group，等待最终 AMO ack |
| 3 | reduce=3,type=S32 | `bulk.s2g.reduce.max.s32` | shared → global | S2G group，等待最终 AMO ack |
| 3 | reduce=4,type=B32 | `bulk.s2g.reduce.and.b32` | shared → global | S2G group，等待最终 AMO ack |
| 3 | reduce=5,type=B32 | `bulk.s2g.reduce.or.b32` | shared → global | S2G group，等待最终 AMO ack |
| 3 | reduce=6,type=B32 | `bulk.s2g.reduce.xor.b32` | shared → global | S2G group，等待最终 AMO ack |
| 4 | reduce=0 | `tensor.s2g.copy` | shared → global | S2G group |
| 4 | reduce=1 | `tensor.s2g.reduce.add` | shared → global | S2G group，等待最终 AMO ack |
| 4 | reduce=2 | `tensor.s2g.reduce.min` | shared → global | S2G group，等待最终 AMO ack |
| 4 | reduce=3 | `tensor.s2g.reduce.max` | shared → global | S2G group，等待最终 AMO ack |
| 4 | reduce=4 | `tensor.s2g.reduce.and` | shared → global | S2G group，等待最终 AMO ack |
| 4 | reduce=5 | `tensor.s2g.reduce.or` | shared → global | S2G group，等待最终 AMO ack |
| 4 | reduce=6 | `tensor.s2g.reduce.xor` | shared → global | S2G group，等待最终 AMO ack |
| 5 | subop=0 | `prefetch.tensormap` | descriptor control | 控制请求接收后完成；best-effort |
| 5 | subop=1 | `invalidate.tensormap` | descriptor control | 对应地址失效完成后返回 |
| 6 | zimm=16 | `s2g.commit_group` | completion control | 立即提交当前非空 group |
| 6 | zimm=24 | `s2g.wait_group 0` | completion control | 等全部已提交 group |
| 6 | zimm=25 | `s2g.wait_group 1` | completion control | 最多保留最近 1 个 group |
| 6 | zimm=26 | `s2g.wait_group 2` | completion control | 最多保留最近 2 个 group |
| 6 | zimm=27 | `s2g.wait_group 3` | completion control | 最多保留最近 3 个 group |
| 7 | subop=0 | `mbarrier.init` | G2S completion control | 初始化完成并镜像到 shared |
| 7 | subop=1 | `mbarrier.arrive.expect_tx` | G2S completion control | 建立 transaction binding |
| 7 | subop=2 | `mbarrier.wait` | G2S completion control | phase 翻转且镜像写确认后放行 warp |
| 7 | subop=3 | `fence.proxy.async.shared` | proxy ordering | 等待当前 warp 的相关 LSU fence 条件 |

## 4. 数据搬运指令

### 4.1 Bulk G2S

OpenCL 包装：

```c
VENTUS_TMA_BULK_G2S(shared_dst, global_src, bytes);
```

规范编码：

```asm
.insn r 0x42, 1, 0, rd_shared_dst, rs1_global_src, rs2_bytes
```

操作数定义：

| 字段 | 类型 | 定义 | 合法范围/约束 |
|---|---|---|---|
| `rd` 的寄存器值 | `uint32` shared address | shared destination 起始地址 | 必须 16B 对齐 |
| `rs1` | `uint32` virtual address | global source 起始地址 | 必须 16B 对齐 |
| `rs2` | `uint32` byte count | 连续搬运长度 | 必须非零且为 16B 倍数 |

由于长度字段是 32-bit unsigned 且低 4 位必须为零，`bytes` 的精确可编码集合为：

```text
{ 0x00000010, 0x00000020, ..., 0xfffffff0 }
```

也就是最小 16B、最大 `4 GiB - 16B`。除上述入口校验外，软件还必须保证完整访问区间有效：

```text
global_src + bytes - 1 <= 0xffffffff
shared_dst .. shared_dst + bytes - 1 全部位于该 work-group 已分配的 shared-memory 区域
```

两侧区间均不得发生 32-bit 地址回绕。当前 Bulk 前端只直接检查起始地址对齐和 `bytes` 合法性，不会替软件证明整个目标对象或地址区间有效。

语义：

```text
for i in [0, bytes):
    shared[shared_dst + i] = global[global_src + i]
```

实现支持跨 128B cache line 和跨页搬运。MMU 开启时，global 地址使用发出 warp 的 ASID 翻译；shared 地址不走 global TLB。

任何地址未按 16B 对齐、长度为零或长度不是 16B 倍数时：

- 指令报告 `UNSUPPORTED_FEATURE`；
- `detail=1`，即 `FunctBulkG2S`；
- 不产生 TLB、cache 或 shared-memory 请求；
- 不修改源或目的数据。

### 4.2 Bulk S2G

OpenCL 包装：

```c
VENTUS_TMA_BULK_S2G(global_dst, shared_src, bytes);
```

规范编码：

```asm
.insn r 0x42, 3, 0, rd_global_dst, rs1_shared_src, rs2_bytes
```

| 字段 | 类型 | 定义 | 合法范围/约束 |
|---|---|---|---|
| `rd` 的寄存器值 | `uint32` virtual address | global destination 起始地址 | 必须 16B 对齐 |
| `rs1` | `uint32` shared address | shared source 起始地址 | 必须 16B 对齐 |
| `rs2` | `uint32` byte count | 连续搬运长度 | 必须非零且为 16B 倍数 |

`bytes` 的精确可编码集合与 Bulk G2S 相同，即 16B 至 `0xfffffff0`、步长 16B。软件必须另外保证：

```text
global_dst + bytes - 1 <= 0xffffffff
shared_src .. shared_src + bytes - 1 全部位于该 work-group 已分配的 shared-memory 区域
```

完整源、目的区间均不得发生 32-bit 地址回绕。

语义：

```text
for i in [0, bytes):
    global[global_dst + i] = shared[shared_src + i]
```

S2G 的硬件完成点是所有 PutFull/PutPartial 或 AMO 的**最终响应**返回，而不是 A-channel 请求被接收。每条 S2G 指令在接收时自动分配到当前 warp 的 open group。

非法 Bulk 的行为与 G2S 相同，但 `detail=3`，即 `FunctBulkS2G`。

### 4.3 Bulk S2G Reduce

Bulk reduce 复用 `funct3=3` 和 Bulk S2G 的三个地址/长度操作数：

```text
rd value = global destination
rs1      = shared source
rs2      = byte count
```

二级字段为：

```text
inst[31:29] = reduce mode
inst[28:27] = element type
```

type 编码：

| type | 名称 | 定义 |
|---:|---|---|
| 0 | U32 | 32-bit unsigned integer |
| 1 | S32 | 32-bit signed two's-complement integer |
| 2 | B32 | 32-bit uninterpreted bit string |
| 3 | RESERVED | 非法 |

当前公开且与 CUDA `cp.reduce.async.bulk.global.shared::cta` 类型组合一致的子集为：

| reduce | type | OpenCL 包装 | `.insn` funct7 | 元素语义 |
|---:|---:|---|---:|---|
| 1 ADD | U32 | `VENTUS_TMA_BULK_REDUCE_ADD_U32` | `0x10` | 32-bit 模加法 |
| 1 ADD | S32 | `VENTUS_TMA_BULK_REDUCE_ADD_S32` | `0x14` | 32-bit 模加法 |
| 2 MIN | U32 | `VENTUS_TMA_BULK_REDUCE_MIN_U32` | `0x20` | 无符号最小值 |
| 2 MIN | S32 | `VENTUS_TMA_BULK_REDUCE_MIN_S32` | `0x24` | 有符号最小值 |
| 3 MAX | U32 | `VENTUS_TMA_BULK_REDUCE_MAX_U32` | `0x30` | 无符号最大值 |
| 3 MAX | S32 | `VENTUS_TMA_BULK_REDUCE_MAX_S32` | `0x34` | 有符号最大值 |
| 4 AND | B32 | `VENTUS_TMA_BULK_REDUCE_AND_B32` | `0x48` | 逐位与 |
| 5 OR | B32 | `VENTUS_TMA_BULK_REDUCE_OR_B32` | `0x58` | 逐位或 |
| 6 XOR | B32 | `VENTUS_TMA_BULK_REDUCE_XOR_B32` | `0x68` | 逐位异或 |

例如：

```c
VENTUS_TMA_BULK_REDUCE_ADD_U32(global_dst, shared_src, bytes);
VENTUS_TMA_S2G_COMMIT_GROUP();
VENTUS_TMA_S2G_WAIT_GROUP0();
```

对每个 32-bit element，语义为：

```text
global_dst[i] = reduce(global_dst[i], shared_src[i])
```

每个 element 是独立原子更新。地址和长度约束与 Bulk S2G copy 完全相同：两个起始地址必须 16B 对齐，`bytes` 必须非零且为 16B 倍数。因此元素数必然是 4 的倍数，不存在 Tensor OOB fill/suppress 语义。group completion 必须等待全部 AMO 的最终确认。

以下编码会在产生 TLB、shared 或 cache 请求之前被拒绝：

- COPY 携带非零 type；
- ADD/MIN/MAX 携带 B32 或 RESERVED type；
- AND/OR/XOR 携带 U32、S32 或 RESERVED type；
- reduce mode 7；
- 任意不满足 Bulk 地址/长度约束的组合。

CUDA 还定义 U64/S64、B64、FP16/BF16/FP32/FP64、INC/DEC 等组合；当前 Ventus L2 原子端点没有这些运算/位宽，本 ABI 不公开这些组合并明确拒绝，不能把“CUDA 合法”误写成“Ventus 已实现”。

## 5. Tensor 指令寄存器 ABI

Tensor 指令的操作数含义为：

| 指令字段 | 当前 OpenCL ABI 使用 | 数据 |
|---|---|---|
| `rd` | `x10` | shared base address |
| `rs1` | `x11` | TensorMap descriptor address |
| `rs2` | `v12` | 5 个 signed 32-bit 起始坐标 |

坐标加载包装：

```c
int32_t coords[5] = {x0, x1, x2, x3, x4};
VENTUS_TMA_LOAD_COORDS_V12(coords);
```

只有 `coordinate[0..rank-1]` 有效。软件应将未使用坐标写零，以便调试和未来兼容。坐标是 signed 32-bit，两补码解释。

Tensor 指令的方向、reduce mode、shared base 和 coordinates 都是 command-local 参数，不存入 TensorMap cache；同一个静态 TensorMap 可以被多条不同坐标的命令复用。

### 5.1 Tensor G2S

```c
VENTUS_TMA_TENSOR_G2S(shared_dst, descriptor);
```

固定 ABI 编码：

```text
0x00c5a542
```

执行步骤：

1. 用 `{ASID, descriptor_address}` 查询 compiled TensorMap cache；
2. miss 时读取完整 128B TensorMap；
3. 编译并缓存静态字段和静态错误状态；
4. CommandBinder 将 coordinates 绑定到 global base/strides；
5. WindowPlanner 产生 128B window 和 16B lane 映射；
6. WindowEngine 从 global cache 读取有效字节并写入 shared；
7. OOB lane 根据描述符写 zero 或 NaN fill；
8. 所有 cache/shared 资源退出后生成完成事件；
9. 如果命令绑定了 mbarrier，则完成事件扣减 barrier pending bytes。

TensorMap 地址和 shared base 都必须 128B 对齐。

### 5.2 Tensor S2G copy

```c
VENTUS_TMA_TENSOR_S2G(shared_src, descriptor);
```

固定 ABI 编码：

```text
0x00c5c542
```

执行步骤与 G2S 的 descriptor/bind/plan 相同，但数据面顺序为：

```text
shared read → byte permutation/packing → TLB → cache Put → final ack
```

当前限制：

- shared base 必须 128B 对齐；
- 起始坐标的所有 active dimension 必须非负；
- 正方向超出 global dimension 的元素被抑制，不写 global memory；
- `oobFill=NaN` 不允许用于 S2G；
- 完成必须通过 S2G group 观察。

## 6. Tensor S2G 原子 Reduce

Tensor S2G 使用 `inst[31:29]` 选择 reduce mode：

| mode | 名称 | 固定 ABI 编码 | 支持 dtype | 运算定义 |
|---:|---|---|---|---|
| 0 | COPY | `0x00c5c542` | 除方向受限 sub-byte 外的合法 dtype | `dst = src` |
| 1 | ADD | `0x20c5c542` | U32、S32 | 32-bit 模加法 |
| 2 | MIN | `0x40c5c542` | U32、S32 | U32 无符号；S32 有符号 |
| 3 | MAX | `0x60c5c542` | U32、S32 | U32 无符号；S32 有符号 |
| 4 | AND | `0x80c5c542` | U32、S32 | 逐位与 |
| 5 | OR | `0xa0c5c542` | U32、S32 | 逐位或 |
| 6 | XOR | `0xc0c5c542` | U32、S32 | 逐位异或 |
| 7 | RESERVED | `0xe0c5c542` | 无 | 报 `UNSUPPORTED_FEATURE` |

软件包装：

```c
VENTUS_TMA_TENSOR_REDUCE_ADD(shared_src, descriptor);
VENTUS_TMA_TENSOR_REDUCE_MIN(shared_src, descriptor);
VENTUS_TMA_TENSOR_REDUCE_MAX(shared_src, descriptor);
VENTUS_TMA_TENSOR_REDUCE_AND(shared_src, descriptor);
VENTUS_TMA_TENSOR_REDUCE_OR(shared_src, descriptor);
VENTUS_TMA_TENSOR_REDUCE_XOR(shared_src, descriptor);
```

Reduce 语义：

- 每个 in-bounds tensor element 生成一个完整 32-bit AMO；
- 每次 AMO 的 operand 来自对应 shared element；
- global destination 不要求由当前 command 独占；
- 正方向 OOB element 被抑制，不产生 AMO；
- 与 Tensor S2G copy 相同，任一 active dimension 的起始坐标为负都会拒绝整条命令，且不产生 AMO；
- group completion 等待每个 AMO 的最终 ack；
- sub-byte、浮点和 64-bit dtype 不能用于 reduce；
- TensorMap 的 `swizzleAtomicity` 字段与 reduce 的 L2 AMO 语义不是一回事。当前该描述符字段仍必须编码为 16B atom 模式 0。

## 7. TensorMap 控制指令

TensorMap control 使用 `funct3=5`，`inst[31:27]` 为 5-bit subop。

### 7.1 Prefetch TensorMap

```c
VENTUS_TMA_PREFETCH_TENSORMAP(descriptor);
```

```text
subop[31:27] = 0
encoding      = 0x0005d042
rs1           = descriptor address，当前包装使用 x11
```

约束与行为：

- descriptor address 必须 128B 对齐；
- 请求是 best-effort cache hint；
- hit 时只更新替换状态和 PMU；
- miss 时读取、编译并填充 compiled descriptor cache；
- 软件不直接得到编译结果；
- prefetch 与 demand miss 可以 coalesce；
- prefetch control lane 独立于 active data command 和 LookaheadSlot；
- Spike 不建模 descriptor cache，因此该指令是架构上无副作用的 NOP。

### 7.2 Invalidate TensorMap

```c
VENTUS_TMA_INVALIDATE_TENSORMAP(descriptor);
```

```text
subop[31:27] = 1
encoding      = 0x0805d042
rs1           = descriptor address，当前包装使用 x11
```

语义：

- 按 `{ASID, descriptor_address[31:7]}` 精确失效；
- 可以杀死正在 fetch/compile 的同地址 entry；
- 同地址等待中的 demand 会在失效 ordering point 后内部重试；
- invalidate 返回时，不会再把失效前编译结果交给后续 demand；
- 软件在原地修改 TensorMap 后，必须在再次使用前执行 invalidate；
- PMU reset 不会清空 descriptor cache。

subop 2–31 不属于公开 ABI。

## 8. S2G group 完成模型

S2G group 指令使用 `funct3=6`。`inst[19:15]` 不再表示通用 `rs1` 寄存器，而是 5-bit `zimm`。

| zimm | 操作 | 原始编码 | 语义 |
|---:|---|---:|---|
| 16 | commit group | `0x00086042` | 提交当前非空 open group |
| 24 | wait group 0 | `0x000c6042` | 所有已提交 group 必须完成 |
| 25 | wait group 1 | `0x000ce042` | 最多保留最近 1 个已提交 group 未完成 |
| 26 | wait group 2 | `0x000d6042` | 最多保留最近 2 个已提交 group 未完成 |
| 27 | wait group 3 | `0x000de042` | 最多保留最近 3 个已提交 group 未完成 |

每个 warp 有 4 个循环 group。数据指令在进入 TMA ingress 时被分配到当前 `issuePtr` 指向的 group。

### 8.1 Commit 规则

```c
VENTUS_TMA_S2G_COMMIT_GROUP();
```

- 当前 group 中存在已接收 S2G command 时，commit 将其标记为 committed，并将 issue pointer 移到下一 group；
- 当前 group 为空时，commit 不推进指针；
- 已 committed 且仍有 outstanding command 的 group 不能被重新用于 issue；
- Bulk S2G、Tensor S2G copy 和全部 Reduce 都进入同一 group 机制。

### 8.2 Wait 规则

`wait_group N` 中的 N 表示允许最近 N 个 committed group 继续 outstanding，而等待更老的 group：

```text
wait_group 0：等待全部已提交 S2G command 最终完成
wait_group 1：允许最新 1 个 committed group 尚未完成
wait_group 2：允许最新 2 个 committed group 尚未完成
wait_group 3：允许最新 3 个 committed group 尚未完成
```

只有 committed group 参与 wait 判断。因此软件要等待一批 S2G，必须先 commit，再 wait。

非法 zimm：

- 不影响 group 状态；
- 写 `INVALID_GROUP_OPERATION`；
- `detail=zimm`。

## 9. mbarrier 与 async proxy

### 9.1 mbarrier 对象

软件对象：

```c
typedef struct __attribute__((aligned(8))) {
    ulong opaque;
} ventus_tma_mbarrier_t;
```

参数：

| 参数 | 格式 | 范围/要求 |
|---|---|---|
| address | 32-bit shared address | 必须 8B 对齐 |
| object size | 8B | 软件必须按 opaque object 对待 |
| entries | 每 resident WG 4 个 | 由硬件静态分区 |
| phase | 1 bit | 软件只通过 wait 的 old phase 参数观察 |
| generation | 内部 8 bit | 用于阻止旧 command completion 命中新 barrier 生命周期 |

硬件会把内部 mbarrier 状态镜像到 shared 中的 8B object。该布局是架构不透明的；软件不得直接解码或改写两个 32-bit word。

### 9.2 `mbarrier.init`

```c
VENTUS_TMA_MBARRIER_INIT(address, arrivals);
```

```text
funct3       = 7
inst[26:25] = 0
encoding     = 0x00b57042
rs1/x10      = address
rs2/x11      = arrivals
```

`arrivals` 的正式范围是 1–255。它表示当前 phase 预期执行多少次有效的 `arrive.expect_tx`，不是字节数。

初始化会设置：

```text
pendingBytes    = 0
expectedArrivals= arrivals
pendingArrivals = arrivals
phase           = 0
generation      = 0
```

以下情况是协议错误：未对齐、arrivals 为零、WG 的 4 个 entry 全部占用，或尝试覆盖仍有 pending bytes 的 barrier。

### 9.3 `mbarrier.arrive.expect_tx`

```c
VENTUS_TMA_MBARRIER_ARRIVE_EXPECT_TX(address, transaction_bytes);
```

```text
funct3       = 7
inst[26:25] = 1
encoding     = 0x02b57042
rs1/x10      = address
rs2/x11      = transaction_bytes
```

要求：

- address 必须命中当前 WG 已初始化的 barrier；
- `transaction_bytes > 0`；
- pending byte 加法不能溢出 32 bit；
- `pendingArrivals > 0`；
- 当前 warp 不能已经有另一个未消费完的 binding；
- 上一 phase 必须已被 wait 正确观察。

成功后：

```text
pendingBytes += transaction_bytes
pendingArrivals -= 1
当前 warp binding = {barrier entry, generation, remaining bytes}
```

后续 G2S command 会从 binding 中预约 transaction bytes：

- Bulk 使用 `bytes`；
- Tensor 使用 `logicalBytes`；
- 一个 binding 可以被一个或多个连续 G2S command 消耗；
- 每条命令的 bytes 必须不超过 binding remaining；
- remaining 归零后 warp binding 自动解除；
- S2G command 不消费 mbarrier binding。

Tensor `logicalBytes` 定义为：

```text
rowBytes    = ceil(boxDims[0] * dtypeBits / 8)
logicalBytes= rowBytes * product(boxDims[1 .. rank-1])
```

该数值是逻辑 global payload 大小，不一定等于 padded/swizzled shared footprint。

### 9.4 `mbarrier.wait`

```c
VENTUS_TMA_MBARRIER_WAIT(address, old_phase);
```

```text
funct3       = 7
inst[26:25] = 2
encoding     = 0x04b57042
rs1/x10      = address
rs2/x11      = old_phase，只有 bit0 有效
```

当以下条件全部成立时 wait 完成：

- 当前 barrier phase 不等于 `old_phase`；
- pending bytes 已归零；
- pending arrivals 已归零并触发 phase 翻转；
- 最新 barrier 状态的两个 shared word 已获得 shared-memory ack。

wait 未满足时，warp 被移出 ready 集合，而不是占用执行槽忙等。

### 9.5 `fence.proxy.async.shared`

```c
VENTUS_TMA_FENCE_PROXY_ASYNC_SHARED();
```

```text
funct3       = 7
inst[26:25] = 3
encoding     = 0x06007042
无有效地址/数值操作数
```

该指令是 shared memory 普通访问代理与 TMA async 代理之间的 ordering point，不是数据搬运完成指令。

正确理解：

- 它不替代 G2S 的 `mbarrier.wait`；
- 它不替代 S2G 的 `wait_group`；
- G2S 后在 barrier wait 之后执行，用于让普通 shared load 观察 async-proxy 写入；
- S2G 前执行，用于让 TMA async-proxy 观察之前的普通 shared store。

## 10. 推荐指令序列

### 10.1 Bulk/Tensor G2S

```c
VENTUS_TMA_MBARRIER_INIT(barrier, 1u);
VENTUS_TMA_MBARRIER_ARRIVE_EXPECT_TX(barrier, logical_bytes);

// 二选一
VENTUS_TMA_BULK_G2S(shared_dst, global_src, logical_bytes);
// 或
VENTUS_TMA_TENSOR_G2S(shared_dst, descriptor);

VENTUS_TMA_MBARRIER_WAIT(barrier, old_phase);
VENTUS_TMA_FENCE_PROXY_ASYNC_SHARED();

// 此后普通 shared load 才消费结果
```

### 10.2 Bulk/Tensor S2G

```c
// 普通程序先产生 shared 数据
VENTUS_TMA_FENCE_PROXY_ASYNC_SHARED();

// 可发出一条或多条 S2G
VENTUS_TMA_BULK_S2G(global_dst0, shared_src0, bytes0);
VENTUS_TMA_TENSOR_S2G(shared_src1, descriptor1);

VENTUS_TMA_S2G_COMMIT_GROUP();
VENTUS_TMA_S2G_WAIT_GROUP0();

// 此时所有已提交 Put/AMO 的最终 ack 已返回
```

### 10.3 Tensor Reduce

```c
VENTUS_TMA_FENCE_PROXY_ASYNC_SHARED();
VENTUS_TMA_TENSOR_REDUCE_ADD(shared_operands, descriptor);
VENTUS_TMA_S2G_COMMIT_GROUP();
VENTUS_TMA_S2G_WAIT_GROUP0();
```

### 10.4 原地修改 TensorMap

```text
软件写回 128B TensorMap
        ↓
invalidate.tensormap(descriptor address)
        ↓
后续 tensor command 重新 demand-fetch/compile
```

`prefetch.tensormap` 可以放在第一次 tensor command 之前隐藏部分 descriptor fetch/compile 延迟，但不是正确性必需条件。

## 11. TensorMap 128B ABI

### 11.1 基本格式

| 属性 | 值 |
|---|---|
| 对象大小 | 128B |
| 对齐 | 128B |
| word 数量 | 32 × 32-bit |
| byte order | little-endian |
| magic | `0x544d4103` |
| cache key | `{ASID, descriptor_address[31:7]}` |

描述符自身不包含 direction、shared base、coordinates、reduce mode 或“描述符长度”字段。

### 11.2 逐 word 布局

| Word | Byte offset | 字段 | 格式 | Active dimension 规则 |
|---:|---:|---|---|---|
| 0 | `0x00` | magic | `uint32` | 必须为 `0x544d4103` |
| 1 | `0x04` | control | bit fields | 见下一节 |
| 2 | `0x08` | globalBase low | `uint32` | global base bits[31:0] |
| 3 | `0x0c` | globalBase high | `uint32` | 当前必须为 0 |
| 4 | `0x10` | globalDims[0] | `uint32` encoded | active；0 表示 `2^32` |
| 5 | `0x14` | globalDims[1] | `uint32` encoded | rank>1 时 active，否则必须 0 |
| 6 | `0x18` | globalDims[2] | `uint32` encoded | rank>2 时 active，否则必须 0 |
| 7 | `0x1c` | globalDims[3] | `uint32` encoded | rank>3 时 active，否则必须 0 |
| 8 | `0x20` | globalDims[4] | `uint32` encoded | rank>4 时 active，否则必须 0 |
| 9–10 | `0x24–0x2b` | globalStrides[0] | little-endian `uint64` | rank>1 时使用 |
| 11–12 | `0x2c–0x33` | globalStrides[1] | little-endian `uint64` | rank>2 时使用 |
| 13–14 | `0x34–0x3b` | globalStrides[2] | little-endian `uint64` | rank>3 时使用 |
| 15–16 | `0x3c–0x43` | globalStrides[3] | little-endian `uint64` | rank>4 时使用 |
| 17 | `0x44` | boxDims[0] | `uint32` | active 时 1–256 |
| 18 | `0x48` | boxDims[1] | `uint32` | active 时 1–256，否则 0 |
| 19 | `0x4c` | boxDims[2] | `uint32` | active 时 1–256，否则 0 |
| 20 | `0x50` | boxDims[3] | `uint32` | active 时 1–256，否则 0 |
| 21 | `0x54` | boxDims[4] | `uint32` | active 时 1–256，否则 0 |
| 22 | `0x58` | elementStrides[0] | `uint32` placeholder | active 必须 1，否则 0 |
| 23 | `0x5c` | elementStrides[1] | `uint32` placeholder | active 必须 1，否则 0 |
| 24 | `0x60` | elementStrides[2] | `uint32` placeholder | active 必须 1，否则 0 |
| 25 | `0x64` | elementStrides[3] | `uint32` placeholder | active 必须 1，否则 0 |
| 26 | `0x68` | elementStrides[4] | `uint32` placeholder | active 必须 1，否则 0 |
| 27–31 | `0x6c–0x7f` | reserved | 5 × `uint32` | 必须全部为 0 |

### 11.3 Control word 位域

| Bits | 宽度 | 字段 | 当前合法值 |
|---|---:|---|---|
| `[4:0]` | 5 | dtype | 0–15 |
| `[7:5]` | 3 | rank | 1–5 |
| `[9:8]` | 2 | interleave | 0、1、2 |
| `[12:10]` | 3 | swizzle | 0、1、2、3 |
| `[14:13]` | 2 | swizzle atomicity | 只支持 0，即 16B |
| `[15]` | 1 | reserved | 必须 0 |
| `[17:16]` | 2 | L2 promotion | 只支持 0 |
| `[18]` | 1 | OOB fill | 0=zero，1=NaN |
| `[20:19]` | 2 | access mode | 只支持 0=tiled |
| `[31:21]` | 11 | reserved | 必须 0 |

构造公式：

```c
control = dtype
        | (rank               << 5)
        | (interleave         << 8)
        | (swizzle            << 10)
        | (swizzle_atomicity  << 13)
        | (l2_promotion       << 16)
        | (oob_fill           << 18)
        | (access_mode        << 19);
```

## 12. Datatype 定义与方向矩阵

TMA 只搬运位模式，不执行浮点格式转换。`FP32_FTZ`、TF32 等编码的主要作用是 TensorMap 类型标记和 OOB NaN pattern 选择。

| Code | 名称 | bits/element | G2S copy | S2G copy | Reduce | 备注 |
|---:|---|---:|---:|---:|---:|---|
| 0 | U8 | 8 | 是 | 是 | 否 | 原样搬运 |
| 1 | U16 | 16 | 是 | 是 | 否 | 原样搬运 |
| 2 | U32 | 32 | 是 | 是 | 是 | 支持全部 6 种 reduce |
| 3 | S32 | 32 | 是 | 是 | 是 | MIN/MAX 使用有符号比较 |
| 4 | U64 | 64 | 是 | 是 | 否 | 原样搬运 |
| 5 | S64 | 64 | 是 | 是 | 否 | 原样搬运 |
| 6 | FP16 | 16 | 是 | 是 | 否 | G2S 可 NaN fill |
| 7 | FP32 | 32 | 是 | 是 | 否 | G2S 可 NaN fill |
| 8 | FP32_FTZ | 32 | 是 | 是 | 否 | G2S 可 NaN fill |
| 9 | FP64 | 64 | 是 | 是 | 否 | G2S 可 NaN fill |
| 10 | BF16 | 16 | 是 | 是 | 否 | G2S 可 NaN fill |
| 11 | TF32 | 32 | 是 | 是 | 否 | G2S 可 NaN fill |
| 12 | TF32_FTZ | 32 | 是 | 是 | 否 | G2S 可 NaN fill |
| 13 | B4x16 | 4 | 是 | 是 | 否 | shared/global 都保持 packed |
| 14 | B4x16P64 | 4 | 是 | 否 | 否 | G2S-only padded layout |
| 15 | B6 | 6 | 是 | 是 | 否 | G2S 和 S2G shared 格式不对称 |
| 16–31 | reserved/unsupported | — | 否 | 否 | 否 | `UNSUPPORTED_DTYPE` detail |

## 13. Dimension、Stride、Coordinate 与地址计算

### 13.1 Rank 和 active dimension

```text
rank ∈ [1, 5]
active dimension: d < rank
inactive dimension: d >= rank
```

对 inactive dimension：

- `globalDims[d]` 必须为 0；
- `boxDims[d]` 必须为 0；
- `elementStrides[d]` 必须为 0；
- 对应的未使用 global stride 必须为 0。

### 13.2 Global dimension

active `globalDims[d]` 使用 32-bit encoded unsigned value：

```text
raw != 0 → dimension = raw
raw == 0 → dimension = 2^32
```

因此，0 不是“空维度”。空 tensor 不能通过 active dimension=0 表示。

### 13.3 Box dimension

每个 active `boxDims[d]` 必须满足：

```text
1 <= boxDims[d] <= 256
```

`boxDims` 定义一条 Tensor 指令遍历的逻辑子箱大小，不要求小于 global dimension；超出部分由 OOB 规则处理。

dim0 行字节数：

```text
boxRowBytes = ceil(boxDims[0] * dtypeBits / 8)
```

必须满足：

```text
boxRowBytes != 0
boxRowBytes % 16 == 0
```

整条 Tensor 指令的逻辑搬运字节数为：

```text
logicalBytes = boxRowBytes * product(boxDims[1 .. rank-1])
```

`logicalBytes` 必须能用非零 `uint32` 表示。因为 `boxRowBytes` 必须是 16B 倍数，所以它的最大合法值是 `0xfffffff0`。描述符编译阶段若发现该乘积超过 32 位，会报告地址溢出错误，命令不会产生内存流量。

### 13.4 Element stride

描述符保留 CUDA-compatible element-stride slots，但当前 RTL 没有 element-stride datapath：

```text
active elementStrides[d]   必须为 1
inactive elementStrides[d] 必须为 0
```

这不是“stride=1 的可配置功能”，而是固定占位约束。

### 13.5 Global stride

`globalStrides[k]` 是 dimension `k+1` 增加 1 时的 byte stride。每个 stride 是 64-bit little-endian value。

普通类型（包括紧凑 B4x16）要求 16B 对齐；padded sub-byte
（B4x16P64、B6）或 interleave32 要求 32B 对齐：

```text
alignment = 32, if dtype is B4x16P64/B6 or interleave == 32B
alignment = 16, otherwise
```

不重叠检查递推：

```text
requiredSpan = ceil(globalDims[0] * dtypeBits / 8)

for d = 1 .. rank-1:
    stride = globalStrides[d-1]
    require stride != 0
    require stride % alignment == 0
    require stride >= requiredSpan
    requiredSpan = stride * globalDims[d]
```

乘法或最终访问地址超出当前 32-bit 地址空间时报告 address overflow。

### 13.6 Command coordinate

每个实际元素坐标为：

```text
coord[d] = commandCoordinate[d] + boxIndex[d]
boxIndex[d] ∈ [0, boxDims[d])
```

G2S 允许 signed negative origin，并对越界部分执行 fill。所有 Tensor S2G 命令（copy 和 reduce）都拒绝任一 active 起始 coordinate 为负；正方向越界部分则被抑制，其中 copy 不写 global memory，reduce 不生成 AMO。

坐标绑定后得到的 **bounding-box global 起始地址** 必须能够用 32-bit
地址表示并满足 16B 对齐。这个约束检查的是最终动态地址，而不只是 descriptor
中的静态 `globalBase`。例如 U32 的 `coordinate[0]=3` 会形成 `base+12`，即使
`globalBase` 本身对齐，命令仍以 `BadAlignment` 拒绝且不产生 TLB、cache 或
shared 数据流量。负坐标 G2S 也遵守同一规则；例如 U32 的 `coordinate[0]=-4`
形成 `base-16`，只要没有 32-bit 地址溢出就是合法的对齐起点，越界元素再由
OOB fill 处理。

紧凑 B4x16 还要求 `coordinate[0]` 是 32 的倍数，使 dim0 起点落在完整的
16B packed atom 上；其 `globalDims[0]` 必须为偶数，避免 tensor 行在半字节处
结束。B4x16P64 和 B6 保持更严格的 `coordinate[0] % 128 == 0`。

### 13.7 非 interleave global 地址

对 byte-aligned datatype：

```text
globalByteOffset = coord[0] * elementBytes
                 + Σ(d=1..rank-1) coord[d] * globalStrides[d-1]
globalAddress = globalBase + globalByteOffset
```

对 sub-byte，应在 bit 域理解 dim0：

```text
bitOffset     = coord[0] * dtypeBits
              + 8 * Σ(d=1..rank-1) coord[d] * globalStrides[d-1]
globalByte    = globalBase + floor(bitOffset / 8)
bitInByte     = bitOffset % 8
```

### 13.8 Interleave global 地址

Interleave 只支持 byte-aligned datatype，且 rank 必须至少为 3。

```text
sliceBytes       = 16, if interleave16
                 = 32, if interleave32
channelsPerSlice = sliceBytes / elementBytes
slice            = floor(coord[0] / channelsPerSlice)
inSlice          = coord[0] % channelsPerSlice
sliceStride      = globalStrides[rank-3] * globalDims[rank-2]

globalByteOffset = inSlice * elementBytes
                 + slice * sliceStride
                 + Σ(d=1..rank-1) coord[d] * globalStrides[d-1]
```

## 14. Interleave、Swizzle 与 Shared Layout

### 14.1 Interleave 枚举

| Code | 名称 | 约束 |
|---:|---|---|
| 0 | NONE | 所有 rank 可用 |
| 1 | 16B | rank≥3；sub-byte 禁止 |
| 2 | 32B | rank≥3；必须配 swizzle32；sub-byte 禁止 |
| 3 | reserved | 非法 |

### 14.2 Swizzle 枚举

| Code | 名称 | Shared row span | 当前支持 |
|---:|---|---:|---:|
| 0 | NONE | tight/no fixed swizzle span | 是 |
| 1 | 32B | 32B | 是 |
| 2 | 64B | 64B | 是 |
| 3 | 128B | 128B | 是 |
| 4 | 96B encoding placeholder | — | 否 |
| 5–7 | reserved | — | 否 |

当 `swizzle != NONE` 时：

```text
boxRowBytes <= swizzleSpan
swizzleSpan = 16 << swizzleCode
```

短行不会在 shared 中紧密拼接，而是使用完整的 swizzle span 作为物理 row pitch。

### 14.3 当前 XOR swizzle 映射

定义：

```text
span  = 16 << swizzleCode
mask  = (1 << swizzleCode) - 1
phase = (sharedBase >> 7) & mask
```

对一个 logical shared byte offset 和 outer-row 编号：

```text
swizzledOffset =
    (logical & ~(span - 1))
  | ((((logical >> 4) & mask) ^ ((row + phase) & mask)) << 4)
  | (logical & 15)
```

其中 `row` 是 box 的 outer dimensions，即 dimension 1..rank-1 的 mixed-radix flatten index。

### 14.4 合法布局组合

| Interleave | Swizzle NONE | Swizzle32 | Swizzle64 | Swizzle128 |
|---|---:|---:|---:|---:|
| NONE | 是 | 是，受 row span 限制 | 是，受 row span 限制 | 是，受 row span 限制 |
| 16B | 是 | 是，受 row span 限制 | 是，受 row span 限制 | 是，受 row span 限制 |
| 32B | 否 | 是 | 否 | 否 |

再叠加以下限制：

- interleave 非 NONE 时 rank≥3；
- sub-byte 强制 interleave NONE；
- B4x16P64、B6 只允许 swizzle NONE 或 swizzle128；
- descriptor `swizzleAtomicity` 只能为 `ATOM_16B=0`；
- `ATOM_32B=1`、`ATOM_32B_FLIP_8B=2`、`ATOM_64B=3` 当前均拒绝。

## 15. OOB 语义

### 15.1 OOB 枚举

| Code | 名称 | G2S | S2G |
|---:|---|---|---|
| 0 | ZERO | 越界目的 shared byte 填零 | 越界 global write 被抑制 |
| 1 | NAN | 仅浮点 G2S | 不支持 |

Sub-byte 只允许 ZERO。

### 15.2 NaN fill pattern

当前 little-endian quiet-NaN pattern：

| dtype | 数值 bit pattern | Shared bytes |
|---|---:|---|
| FP16 | `0x7e00` | `00 7e` |
| BF16 | `0x7fc0` | `c0 7f` |
| FP32 | `0x7fc00000` | `00 00 c0 7f` |
| FP32_FTZ | `0x7fc00000` | `00 00 c0 7f` |
| TF32 | `0x7fc00000` | `00 00 c0 7f` |
| TF32_FTZ | `0x7fc00000` | `00 00 c0 7f` |
| FP64 | `0x7ff8000000000000` | `00 00 00 00 00 00 f8 7f` |

整数或 sub-byte descriptor 配置 NaN fill 会被拒绝，而不是退化为 zero。

## 16. Sub-byte 数据格式

### 16.1 通用限制

三种 sub-byte 类型共同要求：

- interleave 必须 NONE；
- OOB fill 必须 ZERO；
- 不能用于 reduce；
- 最终 bounding-box global 起点必须 16B 对齐。

静态地址约束按布局区分：

| dtype | global base | active global stride | dim0 coordinate |
|---|---:|---:|---:|
| B4x16 | 16B | 16B | 32 elements（16B packed） |
| B4x16P64 | 32B | 32B | 128 elements |
| B6 | 32B | 32B | 128 elements |

这些限制使每个普通 Tensor lane 从完整、对齐的 16B atom 开始。padded
sub-byte 是固定布局特例：B4x16P64 每 lane 只有 8B global payload；B6 每 lane
有 12B global payload。它们分别使用固定的 word phase 和唯一的 word-aligned
B6 tail，不恢复任意 byte-offset 路由。

### 16.2 B4x16：紧凑 4-bit

```text
dtype code      = 13
bits/element    = 4
2 elements      = 1 byte
32 elements     = 16B
```

global 和 shared 都保持相同紧凑 packing：

```text
byte 0 = {element1[3:0], element0[3:0]}
byte 1 = {element3[3:0], element2[3:0]}
...
```

G2S 和 S2G 都支持，因此同一 packed shared 数据可以按相同 TensorMap 语义回写。

### 16.3 B4x16P64：4-bit padded 128B shared row

```text
dtype code         = 14
bits/element       = 4
boxDims[0]         = 128
globalDims[0] %128 = 0
coordinate[0] %128 = 0
global payload     = 128 * 4 / 8 = 64B
shared footprint   = 128B
direction          = G2S only
```

128 个元素被分成 8 个 lane，每个 lane 16 个元素：

```text
每 lane global payload = 16 * 4 / 8 = 8B packed
每 lane shared pitch   = 16B
有效 payload 位于 lane[0..7]
lane[8..15] 是 padding，不属于 transaction bytes
```

可用 swizzle：NONE 或 128B。当前不支持 B4x16P64 S2G。

### 16.4 B6：方向不对称

```text
dtype code         = 15
bits/element       = 6
boxDims[0]         = 128
globalDims[0] %128 = 0
coordinate[0] %128 = 0
global payload     = 128 * 6 / 8 = 96B
shared footprint   = 128B
```

每个 128-element row 分为 8 个 lane，每个 lane 16 elements。

G2S：

```text
16 个 6-bit element 在 global 中连续 packed 为 96 bit = 12B
每 lane 将这 12B packed payload 写入 16B shared lane 的 byte[0..11]
shared lane byte[12..15] 是 padding
```

S2G：

```text
每个 shared lane 输入 16B
每个 byte 的低 6 bit 表示一个 element
硬件取 16 个 low-6-bit element，重新打包为 12B global payload
```

所以 B6 的 G2S shared 输出和 S2G shared 输入格式不同：

```text
G2S shared：12B packed + 4B padding
S2G shared：16B，每 byte 一个 low-6-bit element
```

不能把 B6 G2S 的 128B shared 结果不经转换直接交给 B6 S2G 并期待原样 round-trip。

B6 只允许 interleave NONE，swizzle NONE 或 swizzle128。

## 17. 地址与生命周期约束

### 17.1 对齐汇总

| 对象/参数 | 对齐 |
|---|---:|
| Bulk global source/destination | 16B |
| Bulk shared source/destination | 16B |
| TensorMap descriptor address | 128B |
| Tensor shared base | 128B |
| 普通 Tensor global base/stride | 16B |
| B4x16 global base/stride | 16B |
| B4x16P64/B6 global base/stride | 32B |
| interleave32 global base/stride | 32B |
| Tensor 最终 bounding-box global 起点 | 16B |
| mbarrier object | 8B |

### 17.2 ASID 与页表生命周期

每条已接收 TMA command 最多缓存 4 个 payload translation entry。这些 translation 只在该 command 生命周期内有效，并在 command completion 后丢弃。

从 command 被接收到完成可见之前，软件必须保证：

- 不 remap/unmap/migrate command 访问的页面；
- 不回收或改变该 command 的 ASID 语义；
- G2S 要等待 mbarrier completion 可见；
- S2G 要等待对应 group completion 可见。

## 18. 状态 CSR

### 18.1 CSR 格式

```text
CSR address = 0x814
bits[7:0]   = architectural status code
bits[15:8]  = detail
bits[31:16] = 0
```

OpenCL 包装：

```c
VENTUS_TMA_STATUS_CLEAR();
VENTUS_TMA_STATUS_READ(status);
```

该 CSR 是 per-warp sticky first-error：

- 当前值为 0 时，第一个非零 TMA error 被记录；
- 后续 error 不覆盖第一个 error；
- 软件写 0 清除；
- 新 warp 初始化时清零。

### 18.2 Architectural status code

| Code | 名称 | 定义 |
|---:|---|---|
| 0 | OK | 尚未记录错误 |
| 1 | INVALID_DESCRIPTOR | 描述符、坐标或对齐等静态/动态输入非法 |
| 2 | UNSUPPORTED_FEATURE | 编码存在但当前子集不实现 |
| 3 | ADDRESS_OVERFLOW | 地址计算或访问超过 32-bit 可访问范围 |
| 4 | MBARRIER_PROTOCOL | mbarrier lifecycle、binding 或 byte accounting 错误 |
| 5 | INVALID_GROUP_OPERATION | funct6 zimm 不是 commit/wait0–3 |

### 18.3 Tensor detail code

Tensor frontend 使用以下 detail：

| Detail | 名称 | 典型原因 |
|---:|---|---|
| 0 | OK | 无错误 |
| 1 | BadMagic | word0 不等于 `0x544d4103` |
| 2 | Reserved | 已退役 detail，当前保留 |
| 3 | ReservedBits | control/word27–31/unused dimension 非零 |
| 4 | UnsupportedDType | dtype 16–31 |
| 5 | BadRank | rank 不在 1–5 |
| 6 | BadLayout | interleave/swizzle 组合或 span 非法 |
| 7 | BadAlignment | descriptor/global/shared 地址未对齐 |
| 8 | BadDimension | box 范围、row bytes 或 padded-sub-byte 窗口非法 |
| 9 | BadStride | stride 为零、未对齐或 lower dimensions 重叠 |
| 10 | BadCoordinate | 当前方向不接受该 coordinate |
| 11 | AddressOverflow | base、stride 或 dynamic coordinate 地址溢出 |
| 12 | UnsupportedFeature | element stride、L2 promotion、access mode 等不支持 |

### 18.4 mbarrier detail

| Detail | 阶段 | 含义 |
|---:|---|---|
| 0 | init | init 参数、entry 或 lifecycle 非法 |
| 1 | arrive.expect_tx | arrive 参数或 binding 状态非法 |
| 2 | wait | 地址没有命中当前 WG barrier 或未对齐 |
| 3 | completion | completion bytes/generation 与 barrier 不一致 |
| 4 | reserve | G2S command 超出当前 warp binding remaining |

## 19. 当前硬件容量与流水结构

以下是当前默认 RTL 容量，不是 TensorMap 内存 ABI 字段：

| 参数 | 默认值 | 定义 |
|---|---:|---|
| `max_dma_inst` | `max(num_warp, 4)`；当前 JSON 为 8 | group tracker 的每 warp DMA inflight 计数上限 |
| `dma_group_entries` | 4 | 每 warp S2G group ring 深度，也是公开 completion ABI 数量 |
| `DefaultWindowEntries` | 6 | WindowEngine PayloadSlot 数量 |
| `DefaultGlobalRequestEntries` | 40 | G2S LineContext/global request tracking 数量 |
| `DefaultWriteAckEntries` | 32 | S2G Put/AMO 最终 ack tag 数量 |
| `DefaultSharedReadyEntries` | 8 | shared request/response tracking 数量 |
| `DefaultDescriptorEntries` | 4 | compiled TensorMap cache entries；实现也支持综合为 2 |
| `TranslationEntriesPerCommand` | 4 | 单 command payload translation cache 数量 |
| `WindowCacheSourceEntries` | 64 | 复用的 power-of-two cache source namespace |
| `AtomBytes` | 16B | 内部原子布局单位 |
| `LineBytes` | 128B | cache/shared window 单位 |
| `MbarrierEntriesPerWg` | 4 | 每 resident WG mbarrier 配额 |

`DefaultWriteAckEntries=32` 的作用是允许最多 32 个 S2G cache-line Put/AMO 等待最终确认。它跟踪的是 line transaction，不是软件指令数量，也不是每个 element 都固定占一个 entry；reduce AMO 会复用 entry，在 ack 后回收。

### 19.1 Command 并发边界

当前 ingress 结构为：

```text
一个 active data command
        +
一个 LookaheadSlot
```

LookaheadSlot 可以提前执行：

- 指令基本校验；
- descriptor lookup/fetch/compile；
- coordinate binding；
- mbarrier transaction reserve；
- window planner 准备。

但它不能：

- 分配 WindowEngine payload slot；
- 发出 global TLB/cache payload request；
- 发出 shared payload request。

因此数据面仍然只有一个 active command 真正搬运。TensorMap prefetch/invalidate 有独立 control queue，可以与 active data command 和 lookahead 并行。

## 20. 描述符构造示例

以下示例构造 rank-2 FP32 TensorMap：

```c
#include "ventus_tma_v2_spec.h"

uint32_t desc[32] __attribute__((aligned(128))) = {0};

desc[VENTUS_TMA_V2_WORD_MAGIC] = VENTUS_TMA_V2_MAGIC;
desc[VENTUS_TMA_V2_WORD_CONTROL] =
    VENTUS_TMA_DTYPE_FP32 |
    (2u << 5) |                              // rank=2
    (VENTUS_TMA_INTERLEAVE_NONE << 8) |
    (VENTUS_TMA_SWIZZLE_NONE << 10) |
    (VENTUS_TMA_SWIZZLE_ATOM_16B << 13) |
    (VENTUS_TMA_OOB_ZERO << 18);

// 真正运行前写入 16B aligned global buffer address。
desc[VENTUS_TMA_V2_WORD_GLOBAL_BASE + 0] = global_base_low;
desc[VENTUS_TMA_V2_WORD_GLOBAL_BASE + 1] = 0;

desc[VENTUS_TMA_V2_WORD_GLOBAL_DIMS + 0] = 32;
desc[VENTUS_TMA_V2_WORD_GLOBAL_DIMS + 1] = 8;

// dim1 stride=160B，低 32 bit 在 word9，高 32 bit 在 word10。
desc[VENTUS_TMA_V2_WORD_GLOBAL_STRIDES + 0] = 160;
desc[VENTUS_TMA_V2_WORD_GLOBAL_STRIDES + 1] = 0;

desc[VENTUS_TMA_V2_WORD_BOX_DIMS + 0] = 16;
desc[VENTUS_TMA_V2_WORD_BOX_DIMS + 1] = 4;

desc[VENTUS_TMA_V2_WORD_ELEMENT_STRIDES + 0] = 1;
desc[VENTUS_TMA_V2_WORD_ELEMENT_STRIDES + 1] = 1;

// 所有 inactive dimension、unused stride 和 word27..31 已由初始化保持为 0。
```

该 descriptor 的：

```text
boxRowBytes = 16 * 4 = 64B
logicalBytes= 64 * 4 = 256B
```

如果用于 G2S mbarrier，`arrive.expect_tx` 应登记 256B。

## 21. 非法或当前不支持的设计

当前正式接口明确不支持：

- `CP_ASYNC_COPYSIZE`；
- 旧 `CP_ASYNC_FENCE`；
- funct3=0；
- Bulk S2G reduce mode 7、type 3 或任何未列入 4.3 节的 op/type 组合；
- Tensor S2G reduce mode 7；
- TensorMap subop 2–31；
- funct6 中除 zimm 16、24、25、26、27 之外的值；
- rank 0 或 rank 6–7；
- descriptor size/magic 的旧版本兼容值；
- dtype 16–31；
- L2 promotion 非零；
- access mode 非 tiled；
- element stride 非 1；
- swizzle96；
- swizzle atomicity 32B/32B-flip8/64B；
- interleave 与 sub-byte 组合；
- OOB NaN 与整数、sub-byte 或 S2G 组合；
- B4x16P64 S2G；
- sub-byte reduce；
- 非 16B 对齐或非 16B 倍数的 Bulk；
- descriptor/global/shared 动态地址超过 32-bit 范围。

### 21.1 CUDA 对齐边界

这里的“CUDA 对齐”是语义对齐，不是二进制兼容。审阅当前实现后，边界如下：

| 项目 | CUDA/PTX 语义 | Ventus 当前状态 |
|---|---|---|
| Bulk reduce 方向 | shared::cta → global | 一致 |
| Bulk 地址/长度 | 两个地址 16B 对齐，长度为 16B 倍数 | 一致；Ventus 另外把零长度作为显式非法命令拒绝 |
| Bulk reduce 原子性 | 每个 element 对 global destination 做原子 read-modify-write | 一致；每个 32-bit element 生成一条完整 AMO |
| Bulk reduce completion | `bulk_group` final-operation completion | 由 S2G commit/wait group 对齐，等待最终 AMO ack |
| Bulk reduce 类型 | CUDA 还支持 64-bit、浮点、half/bfloat 和 inc/dec | Ventus 当前仅实现 4.3 节九个 32-bit 组合，其余明确拒绝 |
| Tensor coordinates | `.s32` coordinates；global←shared 的起始坐标必须非负 | 一致；Tensor S2G copy/reduce 共用同一负坐标拒绝规则 |
| Tensor bounding-box address | 每次 tensor 操作的 global bounding-box 起点必须 16B 对齐 | 一致；Ventus 在坐标绑定后检查最终动态起点，未对齐命令零内存流量 |
| 紧凑 B4 起始相位 | 4-bit packed dim0 起点落在 16B atom 边界 | Ventus 要求 `coordinate[0] % 32 == 0`，并要求 `globalDims[0]` 为偶数 |
| padded B4/B6 地址 | CUDA tensor map 编码要求 32B base/stride，dim0 坐标按 128 elements 分组 | 一致；Ventus 保留固定 8B/12B payload 特例，不提供任意 byte phase |
| Tensor 正方向 OOB | global←shared 越界 element 不写 destination | 一致；copy suppress Put，reduce suppress AMO |
| CTA scope | 当前指令形式使用 `shared::cta` / `cta_group::1` | Ventus 当前只实现 workgroup/CTA 级 shared 语义 |
| 编码 | PTX 由 NVIDIA 工具链编码为 SASS | Ventus 使用自定义 `opcode=0x42`，不与 PTX/SASS 二进制兼容 |
| TensorMap binary | CUDA Driver API 创建实现定义的 tensor map | Ventus 使用本文第 11 节自定义 128B ABI，不可直接传入 CUDA TensorMap |

因此，“与 CUDA 对齐”不能解释为已经覆盖 CUDA TMA 的全部 dtype、scope、multicast 或平台 ABI。当前承诺是：已公开子集的方向、坐标、OOB、原子性、对齐和完成语义与对应 PTX 操作一致；未实现能力必须显式拒绝，不能静默退化成非原子 copy/reduce。

## 22. 编码规范与当前实现注意事项

### 22.1 非规范高位别名

正式软件 ABI 对非语义高位写零。但是当前 RTL/Spike MATCH mask 对 funct1–6 的部分 `inst[26:25]` 没有强制为零，所以存在若干“不同 raw word、相同语义”的电气可解码别名。

这些别名：

- 不属于公开 ABI；
- OpenCL 包装从不生成；
- 未来可以被收紧为非法编码；
- 软件、编译器和测试生成器不得依赖。

只有 funct7 正式使用 `inst[26:25]` 选择 mbarrier/proxy subop。

### 22.2 funct3=0

- RTL 保留通用 unknown-funct 防御路径：报告 unsupported，保证零 TLB/cache/shared 流量；
- Spike 没有对应 encoding/handler，因此按未知指令处理；
- 两者都不提供 funct3=0 软件兼容语义。

### 22.3 跨模型状态一致性与已知约束

对“Tensor S2G copy/reduce 使用负起始坐标”，RTL、Spike 与 C model 均以 detail `BadCoordinate=10` 报告 architectural `INVALID_DESCRIPTOR`，并保证不产生 global/shared 数据流量。

`mbarrier.init(arrivals)` 的正式范围是 1–255。RTL 当前截取 low 8 bit，而 Spike 显式拒绝大于 255；软件必须遵守正式范围，不能利用 RTL 截断行为。

## 23. 源码索引

当前定义可从以下文件交叉核对：

- RTL 常量与枚举：[`gpgpu/ventus/src/pipeline/TmaV2Spec.scala`](../gpgpu/ventus/src/pipeline/TmaV2Spec.scala)
- 指令 BitPat：[`gpgpu/ventus/src/pipeline/Instructions.scala`](../gpgpu/ventus/src/pipeline/Instructions.scala)
- 解码表：[`gpgpu/ventus/src/pipeline/DecodeUnit.scala`](../gpgpu/ventus/src/pipeline/DecodeUnit.scala)
- Descriptor compiler/CommandBinder：[`gpgpu/ventus/src/pipeline/TmaV2Frontend.scala`](../gpgpu/ventus/src/pipeline/TmaV2Frontend.scala)
- Ingress/DMA core：[`gpgpu/ventus/src/pipeline/TmaV2DmaCore.scala`](../gpgpu/ventus/src/pipeline/TmaV2DmaCore.scala)
- Planner/layout：[`gpgpu/ventus/src/pipeline/TmaV2WindowPipeline.scala`](../gpgpu/ventus/src/pipeline/TmaV2WindowPipeline.scala)
- WindowEngine/ack：[`gpgpu/ventus/src/pipeline/TmaV2WindowEngine.scala`](../gpgpu/ventus/src/pipeline/TmaV2WindowEngine.scala)
- Descriptor cache：[`gpgpu/ventus/src/pipeline/TmaV2Backend.scala`](../gpgpu/ventus/src/pipeline/TmaV2Backend.scala)
- Group/mbarrier：[`gpgpu/ventus/src/pipeline/TmaV2Completion.scala`](../gpgpu/ventus/src/pipeline/TmaV2Completion.scala)
- OpenCL 指令包装：[`testcases/_get_case/common/ventus_tma_v2_opencl.h`](../testcases/_get_case/common/ventus_tma_v2_opencl.h)
- C/C++ ABI 常量：[`testcases/_get_case/common/ventus_tma_v2_spec.h`](../testcases/_get_case/common/ventus_tma_v2_spec.h)
- C model：[`testcases/_get_case/common/tma_model.cc`](../testcases/_get_case/common/tma_model.cc)
- Spike Tensor 模型：[`spike/riscv/insns/ventus_tma_v2_tensor.inc`](../spike/riscv/insns/ventus_tma_v2_tensor.inc)

---

本文描述的是当前工作树中已经实现的 TMA 子集。Spike、GVM、GVM-nocache、RTL 和 RTL-nocache directed suite 是该接口的跨后端一致性验证集合。新增功能或改变任何保留位语义时，应同步修改 RTL、Spike、公共头、C model、功能测试和本文档。
