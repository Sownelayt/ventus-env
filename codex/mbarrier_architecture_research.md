# Ventus CUDA-like mbarrier 深化调研与硬件重构建议

更新时间：2026-07-24

状态：设计调研，不代表已经实现。本文以当前工作区中的 TMA v2 RTL、Spike 模型、OpenCL wrapper、测试和各项目独立维护的 TMA v2 常量文件为分析对象。

## 1. 执行结论

当前 Ventus 的 `TmaV2MbarrierController` 已经是一个独立控制模块，但它并不是理想的 CUDA-like mbarrier 实现。当前结构的核心是：

```text
固定 entry 寄存器表保存权威状态
  -> 状态变化后标记 dirty
  -> 后台把状态分两次写入 shared memory 中的 8B 对象
```

因此，shared memory 中的对象只是延迟镜像；真正决定 phase、pending arrivals 和 pending bytes 的是 controller 内部固定表。这个边界导致以下根本问题：

1. 每个 WG 固定四个 entry，barrier 数量受硬件表限制。
2. shared-memory 地址不是 barrier 的唯一身份，`barrierId/generation` 才是内部身份。
3. TMA 通过 per-warp 隐式 binding 关联 barrier，而不是每个异步命令显式携带 mbarrier 地址。
4. shared memory 没有真正的 mbarrier 原子 RMW 接口，`isMBarrier` 只是响应路由标签。
5. 当前 ISA 只覆盖 `init/arrive_expect_tx/blocking_wait/proxy_fence`，缺失 CUDA/PTX 的完整 token、predicate、inval、arrive、arrive_drop、expect_tx、complete_tx、test_wait/try_wait 语义。

推荐重构为：

```text
核心 mbarrier 指令 ─────┐
                        │
TMA complete_tx 事件 ───┼──> MBarrierUnit
                        │       │
warp wait/唤醒 <────────┘       ├── waiter CAM（只做睡眠加速）
                                │
                                └── SharedMem MBarrier Atomic Port
                                          │
                                  8-byte shared object
                                  （架构唯一真值）
```

关键原则是：

> 所有合法 mbarrier 操作和 TMA 完成通知都通过同一个语义单元，对 shared memory 中的实际 8B 对象执行线性化的读取、状态变换和写回。固定表可以作为 cache 或 waiter 加速结构，但不能再成为架构真值，也不能限制可创建的 barrier 数量。

这比当前结构更符合 CUDA/PTX 的公开架构语义。需要注意，NVIDIA 没有公开 Hopper 内部 mbarrier 数据通路的晶体管级结构，因此本文提出的是满足公开语义、适合 Ventus 的微架构，而不是声称 NVIDIA 芯片内部一定使用完全相同的模块划分。

## 2. 调研范围与证据层级

本文把证据分成三层：

1. **架构语义**：以当前 NVIDIA PTX ISA 和 CUDA Programming Guide 为准。
2. **公开硬件信息**：只采用 NVIDIA 公开说明，例如 shared-memory barrier、TMA 和硬件加速 wait。
3. **Ventus 微架构建议**：根据当前 RTL 约束推导；这一层是设计选择，不冒充 NVIDIA 未公开实现。

主要外部资料：

- [NVIDIA PTX ISA 9.3](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [CUDA Programming Guide：Asynchronous Barriers](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html)
- [CUDA Programming Guide：Device-Callable Memory Barrier Primitives](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/05-appendices/device-callable-apis.html#memory-barrier-primitives-interface)
- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)

主要项目证据：

- `gpgpu/ventus/src/pipeline/TmaV2Completion.scala`
- `gpgpu/ventus/src/pipeline/TmaV2DmaCore.scala`
- `gpgpu/ventus/src/pipeline/TmaV2Backend.scala`
- `gpgpu/ventus/src/pipeline/scoreboard.scala`
- `gpgpu/ventus/src/pipeline/warp_schedule.scala`
- `gpgpu/ventus/src/pipeline/pipe.scala`
- `gpgpu/ventus/src/L1Cache/ShareMem/ShareMem.scala`
- `gpgpu/ventus/src/L1Cache/ShareMem/ShareMemParameters.scala`
- `gpgpu/ventus/src/top/GPGPU_top.scala`
- `spike/riscv/insns/cp_async_mbarrier_proxy.h`
- `testcases/_get_case/common/ventus_tma_v2_opencl.h`
- `gpgpu/ventus/tests/src/DmaTest/TmaV2Completion_test.scala`
- `gpgpu/ventus/src/pipeline/TmaV2Spec.scala`
- `spike/riscv/ventus_tma_v2_spec.h`
- `testcases/_get_case/common/ventus_tma_v2_spec.h`

## 3. CUDA/PTX mbarrier 的架构语义

### 3.1 对象属性与数量

PTX 将 mbarrier 定义为 shared memory 中的 opaque `.b64` 对象：

| 属性 | 要求 |
|---|---|
| 存储空间 | shared memory |
| 大小 | 8B |
| 对齐 | 8B |
| 软件可解释布局 | 不可，opaque |
| 数量限制 | 受 shared-memory 容量限制，不受传统 barrier slot 数量限制 |

PTX 还规定：一个已经初始化的 mbarrier 对象只能通过 mbarrier 指令访问。对有效对象执行普通 load/store，或者对未初始化/已失效对象执行非 `init` 的 mbarrier 操作，行为未定义。

这个规则对硬件实现非常重要：硬件需要保证所有**合法 mbarrier 操作之间**原子，不需要把任意普通 shared load/store 都纳入一个通用事务系统，因为正确程序不会在对象有效期间这样访问它。

### 3.2 对象内容

当前 PTX 定义的 opaque 状态至少包含：

- 当前 primary phase；layout v1 还区分 conditional phase。
- 当前 phase 的 pending arrival count。
- 下一 phase 的 expected arrival count。
- 当前 phase 的 tx-count，用于跟踪异步事务。
- layout v1 还包含 payload report。

layout v0 的计数范围是：

| 字段 | 范围 |
|---|---:|
| expected arrivals | `1 .. 2^20-1` |
| pending arrivals | `0 .. 2^20-1` |
| tx-count | `-(2^20-1) .. +(2^20-1)` |

tx-count 是带符号的，而不是普通 unsigned outstanding byte counter。这意味着实现不能简单规定“只有 expect 先发生、complete 后发生”，也不能在 `completeCount > currentTx` 时必然报错。不同异步代理和执行路径可能让 complete 与 expect 在硬件中的到达顺序不同，只要最终状态仍在合法范围内即可。

### 3.3 phase 完成条件

对 layout v0，当前 phase 只有在以下两个条件同时满足时完成：

```text
pending_arrivals == 0
tx_count == 0
```

完成时必须原子执行：

```text
phase := next phase
pending_arrivals := expected_arrivals
```

因此，下面这种拆分是错误的：

```text
先把 phase 翻转并对外可见
过几个周期再重置 pending arrivals
```

任何 mbarrier 操作看到的都必须是 phase 转换前或转换后的完整状态，不能看到中间状态。

### 3.4 生命周期

基本生命周期是：

```text
未初始化/invalid
  -> mbarrier.init
  -> phase 0
  -> phase 1
  -> phase 2
  -> ...
  -> mbarrier.inval
  -> 内存可被重新用作普通 shared data 或另一个 mbarrier
```

重要规则：

1. 使用前必须 `init`。
2. 不能对一个仍然有效的对象再次 `init`。
3. 重用这 8B 内存前必须 `inval`。
4. phase 自动循环，不需要每个 phase 重新 `init`。
5. 每个 phase 至少应有一次成功的 wait 观察，然后程序才能在下一 phase 继续 arrive；违反者属于程序未定义行为。

第 5 条可以由硬件 debug checker 检查，但不必为了在所有合法程序中工作而维护一张固定的 `phaseObserved` 权威表。

### 3.5 操作语义

建议把 PTX 操作抽象成统一的状态变换：

| 操作 | 对对象的影响 | 返回 |
|---|---|---|
| `init(count)` | phase=0；expected=pending=count；tx=0 | 无 |
| `inval` | 使对象失效 | 无 |
| `expect_tx(n)` | `tx += n` | 无 |
| `complete_tx(n)` | `tx -= n`，然后检查 phase 完成 | 无 |
| `arrive(count)` | `pending -= count`，然后检查 phase 完成 | 更新前的 opaque token |
| `arrive.expect_tx(n)` | 先 `tx += n`，再执行 count=1 的 arrive | 更新前的 opaque token |
| `arrive_drop(count)` | `expected -= count`，并执行同 count 的 arrive | 更新前的 opaque token |
| `test_wait(token/parity)` | 检查 token/parity 对应 phase 是否已经完成 | predicate |
| `try_wait(token/parity)` | 检查完成；未完成时可暂时挂起执行线程 | predicate |

`arrive.expect_tx` 中 expect 与 arrive 是一个 mbarrier 操作的两个有序步骤，不能让另一条对同一对象的操作插在两者中间。

### 3.6 token、parity 与 wait

`arrive` 和 `arrive_drop` 返回 64-bit opaque token，表示更新前的 barrier 状态。软件只能把 token 交给同一对象的 wait，不能解释其 bit layout。

wait 有两种指定 phase 的方式：

1. 使用 arrive 返回的 token。
2. 使用 phase parity，偶数 phase 为 0，奇数 phase 为 1。

`test_wait` 是非阻塞测试，立即返回 true/false。`try_wait` 是潜在阻塞操作；硬件可以让 warp 睡眠，但也允许因为实现定义的超时提前恢复并返回 false。因此，“一直阻塞直到 phase 完成且不返回 predicate”的自定义指令只能算 convenience wait，不能直接等同于完整 PTX `try_wait`。

CUDA Programming Guide 说明 compute capability 8.0 及以上对 shared-memory asynchronous barrier 提供硬件加速。Hopper 公开资料进一步说明 H100 能够让等待线程睡眠，避免持续在 shared-memory barrier object 上自旋。

### 3.7 TMA 与 mbarrier

对 global-to-shared 的 `cp.async.bulk`/TMA，PTX 指令显式包含：

```text
[shared destination], [global source], size, [mbarrier]
```

完成机制是：

```text
mbarrier::complete_tx::bytes
```

含义是：

1. TMA 命令明确知道要通知的 mbarrier 地址。
2. 数据真正完成写入后，对该对象执行 `complete_tx(copied_bytes)`。
3. TMA 完成通知不是“找到这个 warp 最近绑定的 barrier”。
4. G2S 使用 mbarrier completion；shared-to-global 主要使用 bulk async group completion。

异步命令与 barrier 的关联必须在命令被硬件接收时固定下来，不能依赖后续可变化的 per-warp 隐式状态。

### 3.8 内存序与 async proxy

mbarrier 不是单纯的计数器。

需要满足的关键 ordering：

- 默认 `arrive` 具有 release 语义。
- 成功的 `test_wait/try_wait` 默认具有 acquire 语义。
- 成功 wait 后，参与者在 arrive-release 前的普通内存访问对等待者可见。
- 使用同一 mbarrier 跟踪的 bulk async copy，在成功 wait 后其结果可见。
- `cp.async.bulk` 在 async proxy 执行；generic proxy 与 async proxy 访问同一内存时需要 `fence.proxy.async` 建立跨代理顺序。
- TMA 的隐式 complete-tx 只对该异步命令自身的内存操作建立完成关系，不能自动为 issuing thread 更早的任意操作建立传递顺序。

因此，`fence.proxy.async.shared` 不能在规范上简单定义为“等待 LSU 当前没有请求”。Ventus 可以用 drain 实现一个保守版本，但必须明确它在顺序上保护哪些先前操作和哪些后续 async 命令。

### 3.9 公开资料没有说明的内容

公开文档没有回答：

- NVIDIA 是否使用单个集中式 mbarrier execution unit。
- mbarrier 状态是否每次都直接从物理 shared SRAM 读取。
- 芯片内部是否有 barrier cache、shadow state、专用 bank 或旁路。
- wait wakeup CAM 的大小和组织方式。
- 同地址 mbarrier 操作在真实芯片中的具体流水线周期数。

所以，本文要求的是**架构上以 shared object 为真值**。内部可以有 cache，但必须保持与 shared-memory 对象一致，并且不能暴露固定 slot 数量限制。

## 4. 当前 Ventus mbarrier 机制

### 4.1 当前 ISA 与软件接口

当前三个项目各自维护的 TMA v2 常量把 opcode `0x42`、funct3=7 定义为 `mbarrier_proxy`，bits 26:25 选择：

```text
0: mbarrier_init
1: arrive_expect_tx
2: mbarrier_wait
3: fence_proxy_async_shared
```

OpenCL wrapper 把 mbarrier 声明为 8B 对齐的 64-bit opaque 对象，这一 ABI 外形是合理的：

```c
typedef struct __attribute__((aligned(8))) {
  ulong opaque;
} ventus_tma_mbarrier_t;
```

但是：

- `VENTUS_TMA_BULK_G2S(dst, src, bytes)` 没有 mbarrier 参数。
- `VENTUS_TMA_TENSOR_G2S(shared, descriptor)` 没有 mbarrier 参数。
- `VENTUS_TMA_MBARRIER_ARRIVE_EXPECT_TX(addr, bytes)` 是 void wrapper，没有返回 token。
- `VENTUS_TMA_MBARRIER_WAIT(addr, old_phase)` 是自定义 blocking wait，没有 predicate 返回。
- 没有 `inval/arrive/arrive_drop/expect_tx/complete_tx/test_wait/try_wait`。

因此，当前接口只是一个针对已有 TMA 测试的裁剪协议，不是完整 CUDA/PTX mbarrier ISA。

### 4.2 指令执行路径

当前 mbarrier 指令被 decode 为 DMA funct7 路径，然后在 issue 阶段送到 warp scheduler：

```text
DecodeUnit
  -> issue.out_warpscheduler
  -> warp_schedule.dma_sync_cmd
  -> TmaV2MbarrierController.syncCommand
```

`issue.scala` 只把 mbarrier 控制指令送到 scheduler，没有普通执行单元的结果写回路径。这解释了为什么当前 arrive 不能返回 64-bit token、wait 也不能返回 predicate。完整 CUDA-like ISA 必须新增 response/writeback 路径，或者定义一条明确的自定义 blocking convenience 指令并与 PTX 操作分开。

### 4.3 固定权威表

`TmaV2MbarrierController` 强制要求：

```scala
entries == num_block * MbarrierEntriesPerWg
```

且每个 resident WG 固定四个 entry。

每个 entry 保存：

```text
valid
owner
address
pendingBytes
expectedArrivals
pendingArrivals
phase
generation
phaseObserved
dirty
```

查找键是 `{owner, address}`，但命中后所有操作实际上更新 entry 寄存器。shared object 不参与读取，也不决定下一状态。

当前计数位宽：

| 字段 | 当前位宽 |
|---|---:|
| expected arrivals | 8 |
| pending arrivals | 8 |
| pending bytes | unsigned 32 |
| generation | 8 |

这与 PTX v0 的 20-bit arrival 和 signed tx-count 不一致。

### 4.4 per-warp 隐式 binding

执行 `arrive_expect_tx` 后，controller 设置：

```text
bindingValid(wid)
bindingEntry(wid)
bindingGeneration(wid)
bindingRemaining(wid)
```

随后每条 G2S TMA 在 reserve 阶段消费该 binding 的一部分 bytes，并得到：

```text
barrierValid
barrierId
generation
transactionBytes
```

测试明确验证“一次 expectation 持续绑定多个乱序 G2S 操作”。这是一种自定义软件协议，但与 PTX bulk async 指令显式携带 `[mbar]` 不同。

隐式 binding 还有几个潜在问题：

1. 指令重放、异常、flush 或未来多 issue 后更难定义“下一条”命令。
2. 同一个 warp 同时操作多个 barrier 很困难。
3. barrier 地址在 TMA command 中丢失，只剩内部 table id。
4. 需要 generation 防止固定 slot 被复用后的陈旧 completion。

### 4.5 TMA completion 路径

TMA backend 在 command 中保存：

```text
barrierValid
barrierId
barrierGeneration
transactionBytes
```

G2S scatter 对 shared memory 发出写请求，并等待 shared response。最后一个 line 的 shared 写 response 完成后，line 进入 Done；所有 line 完成后才产生 command completion。

这一点是当前设计中应保留的正确性质：

> complete_tx 事件不能早于对应 TMA shared write 的 acknowledgement。

问题在于 completion 最终根据内部 `barrierId/generation` 更新表，而不是根据命令显式保存的 shared mbarrier 地址更新对象。

### 4.6 shared-memory 镜像

表项更新后设置 `dirty`。writer：

1. 选择一个 dirty entry。
2. 锁存 address、pendingBytes 和编码后的状态。
3. 向 shared memory 写低 32 位。
4. 等 response。
5. 再写高 32 位。
6. 再等 response。

wait 只有在 phase 已变化、dirty 清除、writer 不再处理该 entry 时才释放。

这个机制努力保证软件最终看到最新镜像，但仍有三个架构问题：

1. 对象的两个 32-bit word 不是一个原子状态变换。
2. shared memory 中的对象被外部写坏，controller 不会察觉。
3. controller 的固定 entry 才是状态，和“用户定义 shared-memory object”模型不一致。

### 4.7 shared memory 当前能力

`ShareMemCoreReq` 只有：

```text
instrId
isWrite
isMBarrier
setIdx
perLaneAddr
data
sourceTag
```

其中：

- `isWrite` 选择普通读或写。
- `isMBarrier` 只随 response 返回，用于 pipe 中把 DMA response 与 mbarrier mirror writer response 分开。
- 没有 atomic opcode。
- 没有 read-modify-write lock。
- 没有 64-bit object transaction。

shared memory 是 32-bit word banked SRAM，bank 数当前等于 lane 数。每个 bank 有独立读写端口，支持 byte waymask 和 write bypass。普通 bank conflict arbiter 只负责把同一请求中冲突 lane 串行化，不负责跨请求 RMW 原子性。

当前参数：

```text
sharedmem_depth      = 1024
sharedmem_BlockWords = 32
word bytes           = 4
total                = 128 KiB / CU
```

如果完全只存 mbarrier，理论上可容纳 16384 个 8B 对象；真实程序还需要 shared data，因此实际数量由每个 WG 的 shared-memory 分配决定，但显然不应固定成四个。

### 4.8 wait 与 WG 生命周期

当前 controller 为每个 warp 保存：

```text
waitActive
waitEntry
waitPhase
```

`waitMask` 进入 scheduler，和 DMA group wait、proxy wait 一起从 `warp_ready` 中屏蔽 warp。这一思路可以保留为 `try_wait` 睡眠加速。

但当前 waiter 绑定固定 entry。新版应把 waiter key 改为：

```text
{CTA allocation epoch, resolved shared address, token/parity}
```

WG 最后一个 warp 退出时，当前 scheduler 检查 `ownerBusy`，等固定表、binding、waiter 和 writer 都空闲后释放整个 WG 分区。新版仍然必须在 shared allocation 回收前 drain 未完成 TMA，但不能把“内部 barrier entry 是否 valid”当作 WG 生命周期真值。

### 4.9 Spike 模型与 RTL 的差异

Spike 模型直接 `load_uint32/store_uint32` 操作 mbarrier 对象，同时仍使用软件侧 barrier allocation/binding 表。wait 通过重新执行同一个 PC 模拟 deschedule。

因此：

- Spike 比 RTL 更接近“对象中保存状态”，但两个 32-bit store 仍不是统一原子抽象。
- RTL 把表当真值、对象当镜像。
- 两者目前只在自定义测试协议下对齐，没有共享一个明确的 mbarrier reference transition model。

重构时应先建立一个与 RTL/Spike 无关的 64-bit state reference model，再让两端共同对齐它。

## 5. 当前实现与目标模型的差异矩阵

| 维度 | 当前 Ventus | 推荐目标 |
|---|---|---|
| barrier 身份 | `{owner, table entry}`，地址用于查表 | resolved shared address |
| 架构真值 | controller registers | shared 8B object |
| 容量 | 4/WG | shared-memory capacity |
| 状态更新 | 表内单周期更新，稍后镜像 | 对对象线性化 RMW |
| 对象写入 | 两个依次 ack 的 32-bit write | 一个锁区间内的 64-bit 状态提交 |
| arrival count | 8-bit | v0 20-bit |
| tx-count | unsigned pending bytes | signed tx-count |
| phase token | 无 | 64-bit opaque token |
| wait 结果 | 无，直接阻塞 | predicate；try_wait 可睡眠 |
| TMA 关联 | per-warp implicit binding | command 显式 mbar address |
| completion 身份 | barrierId + generation | address + allocation epoch |
| init/inval | init + WG 清表 | init/inval |
| arrive_drop | 无 | 支持 |
| release/acquire | 未完整建模 | 明确建模 |
| proxy fence | LSU drain 近似 | generic/async proxy ordering |
| wait 容量 | 固定 per-warp state | waiter cache 容量独立于 barrier 数 |

## 6. 推荐微架构

### 6.1 模块边界

建议拆成两个逻辑层：

```text
MBarrierUnit
  - 操作仲裁
  - 语义检查
  - 状态 decode/transition/encode
  - token/predicate response
  - waiter 管理
  - TMA completion 接收
  - release/acquire/proxy 协调

SharedMemMBarrierAtomicEndpoint
  - 8B 对齐检查后的物理 bank 访问
  - 64-bit read
  - RMW 锁
  - 64-bit write
  - write acknowledgement
```

这两个层可以先在一个 Chisel Module 内实现，但接口概念应分开，避免以后为了并行化 shared bank 而重写所有 mbarrier 语义。

### 6.2 请求接口

建议统一请求：

```scala
class MBarrierRequest extends Bundle {
  val source          // core / TMA / debug
  val wid
  val ctaSlot
  val allocationEpoch
  val op
  val address         // resolved physical shared address or unambiguous CU-local offset
  val count
  val token
  val parity
  val sem             // relaxed / acquire / release
  val scope           // first phase只实现CTA
}
```

统一响应：

```scala
class MBarrierResponse extends Bundle {
  val source
  val wid
  val success
  val token
  val predicate
  val protocolError
}
```

TMA completion 不需要 token/predicate，但需要响应或 backpressure，保证事件没有丢失。

### 6.3 地址与对象身份

barrier 的架构身份必须是 shared-memory 地址，而不是 entry id。

Ventus 当前 LDS allocator 会给 WG 分配 shared-memory base，普通 shared 地址最后映射到 CU 内的 set/block/word。新版 MBarrierUnit 应使用与普通 LSU/TMA 相同的 resolved shared 地址规则，并验证：

1. 地址落在 shared aperture。
2. 8B 对齐。
3. CTA-scope 操作的对象属于执行 CTA 的 shared allocation。
4. 未来 cluster remote 地址按 cluster address mapping 处理。

`allocationEpoch` 不属于 PTX 对象状态，只用于防止硬件内部陈旧事件：

```text
旧 WG 发出的 TMA completion
  不能在 shared allocation 已回收并分给新 WG 后
  更新相同物理地址的新对象
```

正确系统应在 WG 回收前 drain 所有命令；epoch 是额外保护和断言依据，不应成为 barrier 的软件身份。

### 6.4 64-bit 状态编码

PTX 规定对象 opaque，因此 Ventus 可以自定义编码。

一个可行的 v0 预算是：

```text
expected arrivals  20
pending arrivals   20
signed tx-count    21
phase               1
valid/layout/private 2
----------------------
total               64
```

这只是位宽可行性证明，不建议在设计冻结前直接固化 bit position。需要先决定：

- invalid 如何编码。
- 是否只支持 layout v0。
- token 是返回完整更新前 object state，还是返回包含 phase/cookie 的私有编码。
- protocol debug metadata 是否放对象内。

首版建议：

1. 只支持 layout v0。
2. token 直接返回更新前的 opaque 64-bit state，软件不解释。
3. 不把 `phaseObserved` 放进规范状态；必要时只做仿真断言。
4. layout v1 另设 decode，不挤占 v0 计数位宽。

### 6.5 原子访问 FSM

正确性优先的单发射 FSM：

```text
Idle
  -> SelectRequest
  -> ValidateAddress
  -> IssueRead64
  -> WaitRead64
  -> DecodeAndTransition
  -> [RegisterWaiter | IssueWrite64 | Respond]
  -> WaitWriteAck
  -> WakeWaiters
  -> Respond
  -> Idle
```

线性化点建议定义为：

- 只读 `test_wait`：读取到稳定对象状态的 read response。
- 状态修改操作：64-bit 新状态被 atomic endpoint 接受并保证写入的时刻。
- blocking waiter：成功 phase transition 的状态提交；wake response 不能早于该提交。

首版一个 CU 同时只处理一个 mbarrier RMW，天然序列化所有地址，逻辑简单。吞吐优化时再增加多个 object lock/MSHR：

```text
不同 8B 地址可以并行
相同 8B 地址必须总序
```

### 6.6 当前 banked shared memory 上的实现

8B 对齐对象包含两个相邻 32-bit word。在当前 bank 映射下，可以：

1. read request 激活 lane 0/1，读取两个 word。
2. 保持 mbarrier object lock。
3. transition 后用一个 write request 激活 lane 0/1，同时写两个 word。
4. 收到完整 write response 后解除锁。

即使两个 word 落入不同 bank，也不能只依赖“同周期发出”来声明原子性；atomic endpoint 必须保证：

- 下一条同对象 mbarrier 操作看不到一个新 word 和一个旧 word。
- write ack 覆盖两个 word。
- bank conflict replay 不会把一个 64-bit commit 暴露成两个可观察状态。

实现选择：

| 方案 | 优点 | 风险 |
|---|---|---|
| SharedMemory 内部专用 atomic port | 原子边界清晰，未来易做 bank lock | 修改 SharedMemory 较多 |
| 外部 MBarrierUnit 用现有 vector read/write port并全局串行 | 首版改动较小 | 必须仔细控制 request/response 和双 word commit |
| 固定表 + mirror | 当前已有 | 不满足目标，不建议延续 |

推荐先实现“外部语义单元 + 明确的 shared atomic adapter”，随后把 adapter 下沉到 SharedMemory 内部。

### 6.7 状态变换伪代码

```text
transition(old, request):
    require address_aligned

    switch request.op:
      INIT:
        require !old.valid
        require count in valid range
        new.valid     = true
        new.phase     = 0
        new.expected  = count
        new.pending   = count
        new.tx        = 0

      INVAL:
        require old.valid
        new.valid = false

      EXPECT_TX:
        require old.valid
        new.tx = checked_signed_add(old.tx, count)

      COMPLETE_TX:
        require old.valid
        new.tx = checked_signed_sub(old.tx, count)
        new = maybe_complete_phase(new)

      ARRIVE:
        require old.valid
        require count <= old.pending
        response.token = encode_token(old)
        new.pending = old.pending - count
        new = maybe_complete_phase(new)

      ARRIVE_EXPECT_TX:
        require old.valid
        response.token = encode_token(old)
        temp.tx = checked_signed_add(old.tx, count)
        temp.pending = old.pending - 1
        new = maybe_complete_phase(temp)

      ARRIVE_DROP:
        require old.valid
        response.token = encode_token(old)
        new.expected = old.expected - count
        new.pending = old.pending - count
        new = maybe_complete_phase(new)

      TEST_WAIT:
        require old.valid
        response.predicate = phase_completed(old, token_or_parity)
        no write

maybe_complete_phase(state):
    if state.pending == 0 && state.tx == 0:
        state.phase = next_phase(state.phase)
        state.pending = state.expected
    return state
```

`checked_signed_add/sub` 要检查 PTX layout 对应的合法范围，而不是采用 unsigned underflow 规则。

### 6.8 请求仲裁与 backpressure

新 RMW 是多周期操作，必须解决事件丢失问题。

建议至少有：

```text
core request queue      depth >= 2
TMA completion queue   depth >= 最大可同时完成的 backend command 数，或允许 backend backpressure
response queue          depth >= 2
```

仲裁策略：

- TMA completion 不能饿死，否则 barrier 永远不完成。
- core wait/test 也不能永久饿死。
- 可以使用 round-robin，并对 completion 设置 aging/优先级。
- 同一对象严格按进入 MBarrierUnit 的线性化顺序处理。

当前 controller 直接接收 `Valid[DmaCompletion]`，依靠单周期 table update 吞下事件。改成多周期 RMW 后必须改成 `Decoupled` 或在入口增加不会溢出的 queue，不能继续假定 completion 每周期无条件消费。

### 6.9 wait 加速器

waiter CAM 不是 barrier storage。

建议 entry：

```text
valid
wid
ctaSlot
allocationEpoch
address
tokenOrParity
phaseType
timeout/deadline（可选）
```

工作流程：

```text
TRY_WAIT request
  -> 原子读取对象
  -> 已完成：返回 true
  -> 未完成且有 waiter slot：
       在释放对象串行化之前登记 waiter
       scheduler 屏蔽 warp
  -> 未完成且无 slot：
       返回 false，由软件 polling/replay
```

避免 lost wakeup 的核心是：

```text
“读取为未完成”与“登记 waiter”之间
不能插入同一对象的 phase transition。
```

phase transition 提交后，MBarrierUnit 匹配 waiter 并唤醒。waiter 数量可以有限，但 barrier 对象数量不能因此有限。

首版可以不做 CAM，只实现 non-blocking `test_wait`。这样先验证对象状态和 TMA completion，再增加调度优化。

### 6.10 TMA command 与 completion

建议把当前字段：

```text
barrierValid
barrierId
barrierGeneration
transactionBytes
```

改为概念上的：

```text
completionKind       // none / mbarrier / bulk_group
mbarAddress
completeCount
ctaSlot
allocationEpoch
```

G2S command 接收时锁存 mbar 地址。最后一笔 shared write ack 后：

```text
TMA backend
  -> MBarrier COMPLETE_TX(address, completeCount)
  -> shared object atomic RMW
  -> write ack
  -> phase transition/wakeup
```

不能在以下时刻提前发 complete：

- global read response 到达但 shared write 尚未提交。
- shared write request 仅被仲裁器接受但 SRAM response 尚未返回。
- 只有部分 line 完成。
- OOB/fill 路径尚有未提交的 destination byte。

对 copied bytes 的定义必须与 expect 侧一致。对于当前 TMA，推荐使用命令定义的逻辑传输 byte count，并对 OOB/fill 的计数规则在 spec 中固定，不能让前端按逻辑 bytes、后端按实际 in-bounds bytes 各算一套。

### 6.11 release/acquire 的 Ventus 落地

至少需要定义三个 ordering 点：

#### 核心 arrive-release

同一线程在 arrive 前的普通 shared writes，必须在 arrive 状态提交前完成到可被 CTA 观察的点。

保守实现：

```text
arrive request 不进入状态 RMW
直到该 warp 先前的 shared LSU write 全部 ack
```

更高性能实现可以使用 per-warp shared ordering epoch，而不是完全 drain 所有 LSU 类型。

#### TMA complete-release

TMA completion 只有在该异步命令的所有 shared writes 获得 completion acknowledgement 后才能进入 MBarrierUnit。当前 backend 的 line Done 路径已经接近这一要求。

#### wait-acquire

wait 返回 true 或唤醒 warp，必须晚于：

1. phase transition 的 64-bit 状态提交。
2. 该 phase 跟踪的 TMA shared writes 可见。
3. release arrive 之前的参与者内存访问达到相应可见点。

在当前 CU-local、单 SharedMemory arbitration 的实现中，可以先使用统一 ack/drain 建立保守顺序；未来引入更并行的 shared ports 后必须保留这些 happens-before 边界。

### 6.12 async proxy fence

建议把 proxy fence 从“一个特殊 blocking bit”提升为显式 ordering request：

```text
FENCE_PROXY_GENERIC_TO_ASYNC
FENCE_PROXY_ASYNC_TO_GENERIC
```

CTA-local首版至少保证：

- generic -> async：执行线程在 fence 前的 generic shared writes 全部完成，然后才允许该线程后续 TMA/S2G async shared reads 被接受。
- async -> generic：异步写完成被 mbarrier/bulk-group 观察后，后续 generic shared reads 能看到结果。

如果硬件采用一个强序 shared-memory arbitration domain，drain 可以实现比 PTX 更强但正确的顺序；文档和 spec 仍应描述 proxy 语义，而不是把具体 drain 信号写成架构定义。

### 6.13 WG 回收

WG shared allocation 释放前必须满足：

```text
该 WG 没有未完成 TMA 命令
没有排队的 complete_tx
没有仍引用该 allocation epoch 的 waiter
没有正在执行的 mbarrier RMW
```

不要求所有 mbarrier 对象都显式 `inval` 才能结束 kernel；CTA 结束后整个 shared allocation 被回收是更高层生命周期事件。但任何陈旧 completion 都不能越过回收点。

当前 `ownerBusy` 思路可以保留，但 busy 来源应改为：

```text
TMA inflight + completion queue + mbarrier engine request + waiter
```

而不是“固定 barrier table 中还有 valid entry”。

## 7. ISA/ABI 重构

### 7.1 最低推荐指令集合

第一阶段建议至少支持：

```text
mbarrier.init
mbarrier.inval
mbarrier.arrive
mbarrier.arrive.expect_tx
mbarrier.expect_tx
mbarrier.complete_tx       // TMA内部必须支持；是否先暴露软件指令可另定
mbarrier.test_wait.parity
fence.proxy.async.shared
```

第二阶段增加：

```text
mbarrier.arrive_drop
mbarrier.test_wait token
mbarrier.try_wait token/parity
release/relaxed
acquire/relaxed
```

### 7.2 结果写回

完整语义需要：

- `arrive/arrive_drop/arrive.expect_tx` 返回 64-bit token。
- `test_wait/try_wait` 返回 predicate。

当前 mbarrier 指令只走 warp scheduler，没有 writeback。可选方案：

| 方案 | 优点 | 缺点 |
|---|---|---|
| 新增 MBarrier execution/writeback path | 语义清晰，可返回 token/predicate | pipeline 接口修改较多 |
| 复用 scalar ALU writeback，MBarrierUnit 异步响应 | 少建一套 RF 端口 | 需要 scoreboard 和 response tag |
| 只保留 blocking convenience wait | 改动小 | 不完整，不可冒充 PTX test/try wait |

推荐 MBarrierUnit 响应进入 scalar writeback arbiter，并由 scoreboard 在请求发出到响应返回期间保护 destination register。

### 7.3 TMA 的第四个操作数

当前 bulk G2S R-type 已使用：

```text
dst
src
bytes
```

没有位置放 `mbarAddress`。可选方案：

| 方案 | 说明 | 评价 |
|---|---|---|
| 新长指令/R4-like 编码 | 四个寄存器都在指令中 | 最明确，但编译器/解码改动最大 |
| 固定 ABI scalar registers | 发射时一次性读取 dst/src/size/mbar | 适合当前已有 `.word` 固定寄存器风格 |
| command descriptor | 指令指向包含四个字段的 command block | 灵活，但增加读取和一致性成本 |
| 继续 per-warp binding | 先设置 barrier，再隐式关联后续命令 | 不推荐，保留当前根本问题 |

PTX 是虚拟 ISA，因此 Ventus 机器指令不必逐字段复制 PTX 编码；但命令进入 TMA engine 时必须已经显式、不可变地包含 mbar 地址。固定 ABI register 方案是当前工程下较现实的第一选择。

## 8. 推荐内部不变量

设计和验证应围绕以下 invariant：

### I1：唯一真值

任意时刻，架构状态由 shared 8B object 定义。任何 cache/shadow 都可从对象恢复，不能要求固定 table entry 才能访问。

### I2：同对象总序

对同一 resolved 8B 地址的所有 mbarrier 操作存在一个总顺序，每次操作读取前一操作提交后的完整状态。

### I3：phase 原子转换

`phase` 翻转与 `pending := expected` 不可被分开观察。

### I4：completion 不丢失

每个 TMA command 产生恰好一次 complete-tx，除非命令被定义为失败并走明确错误路径。

### I5：数据先于通知

TMA destination 的最后一个 shared write ack 严格先于对应 complete-tx 的线性化点。

### I6：无 lost wakeup

一个 waiter 观察 incomplete 并进入睡眠后，如果目标 phase 完成，它最终一定被唤醒；phase transition 不能落在检查与登记的缝隙中。

### I7：容量解耦

waiter/cache 满可以降低性能或回退 polling，但不能使一个合法、已分配 shared object 无法执行基本 mbarrier 操作。

### I8：allocation reuse 安全

旧 allocation epoch 的 completion/waiter/RMW 不能修改或唤醒新 WG。

### I9：response 晚于提交

任何返回 token/predicate 或解除 warp wait 的 response 都不能早于其要求的 shared-memory state/data visibility。

### I10：计数范围

所有 arrival 和 signed tx 变换都保持在选定 layout 的合法范围；边界错误只能产生明确 protocol status，不能静默 wrap。

## 9. 与当前代码的迁移映射

| 当前位置 | 推荐变化 |
|---|---|
| `TmaV2Completion.scala` | 用 MBarrierUnit + atomic object FSM 替换固定权威表和 mirror writer |
| `scoreboard.scala:DmaCompletion` | `barrierId/generation` 改为 completion kind、mbar address、allocation epoch、count |
| `TmaV2DmaCore.scala` | 删除 txReserve 对固定 binding 的依赖；命令接收时锁存 mbar address |
| `TmaV2Backend.scala` | 保留 shared ack 后 completion；completion payload 改为地址 |
| `pipe.scala` | 接入 core request、TMA completion、response/writeback；completion 支持 backpressure |
| `ShareMem.scala` | 增加 mbarrier atomic adapter/port，或支持锁定的 read64/write64 |
| `GPGPU_top.scala` | shared requester/sourceTag 从 pipe/DMA 两类扩展为可路由 mbarrier endpoint |
| `warp_schedule.scala` | wait mask 改由 waiter engine 驱动；proxy fence 使用明确 ordering handshake |
| `DecodeUnit.scala` | 扩展 op 编码、返回寄存器和 ordering qualifier |
| OpenCL wrapper | TMA G2S 显式接收 mbar，arrive/test_wait 提供返回值 |
| Spike | 与统一 64-bit reference transition model 对齐 |
| `TmaV2Spec.scala` | 删除固定四 entry 的架构约束，维护 gpgpu 的 layout/op/count/order/completion 常量 |
| Spike/testcases TMA v2 headers | 各自在所属 submodule 中同步公开 ISA/ABI 编码 |
| `tma_v2_vectors.json` | 作为 gpgpu 测试资源独立维护，不再由根仓库脚本生成 |

当前 `ShareMemCoreReq.sourceTag` 只有 Bool，区分 pipe 与 DMA。如果 MBarrierUnit 作为第三个顶层 requester，应改成至少 2-bit source enum；如果 atomic endpoint 直接下沉到 SharedMemory，则可以提供独立 request/response port，避免继续堆叠 `isMBarrier` 路由标签。

## 10. 分阶段落地建议

### 阶段 0：冻结语义

在改 RTL 前完成：

1. 选择首版只支持 layout v0、CTA scope。
2. 固定 64-bit opaque codec。
3. 固定每个操作的输入、输出、错误条件。
4. 固定 TMA completeCount/OOB/fill 计数规则。
5. 固定 release/acquire/proxy 的保守实现边界。

### 阶段 1：独立 reference model

建立无 RTL 依赖的纯状态模型：

```text
decode64
transition
encode64
token/parity wait
range/protocol checks
```

Spike 和 Chisel test 应共用相同测试向量，但不要让模型算法由 RTL 代码生成。

### 阶段 2：单发射 atomic endpoint

实现：

- read64。
- object lock。
- transition。
- write64。
- ack 后 response。

先只做 `init/inval/arrive/expect_tx/complete_tx/test_wait.parity`，不接 TMA。

### 阶段 3：核心 ISA

接入 decode、issue、scoreboard 和 scalar writeback；验证 token 和 predicate。

### 阶段 4：显式 TMA mbarrier

修改 G2S command ABI，删除主路径的 per-warp binding；把 TMA completion 改为地址事件，并在 shared data ack 后提交 complete-tx。

### 阶段 5：wait 睡眠

增加 waiter CAM、warp mask、无 lost wakeup 机制。CAM 满回退 test_wait polling。

### 阶段 6：内存序

接入 core arrive-release、wait-acquire 和 proxy fence ordering。先允许强于 PTX 的保守 drain，再用 PMU 定位性能损失。

### 阶段 7：并行化

在正确性稳定后增加：

- 多个 object MSHR。
- 按地址/bank-pair 锁。
- mbarrier state cache。
- completion batching。

任何性能优化都必须继续满足第 8 节 invariant。

### 阶段 8：cluster/layout v1

最后考虑 remote shared、cluster scope、multicast、conditional phase 和 payload report。不要在 CTA v0 尚未稳定时同时引入这些维度。

## 11. 验证计划

### 11.1 状态单元测试

至少覆盖：

| 类别 | 用例 |
|---|---|
| init/inval | 正常、未对齐、count 边界、重复 init、invalid 后操作 |
| arrive | count=1、多 count、最后一次 arrival、返回 token |
| expect/complete | 正值、负值中间态、最终归零、上下界 |
| phase | arrival 先归零、tx 后归零；tx 先归零、arrival 后归零 |
| arrive_expect_tx | expect 与 arrive 不可插入 |
| arrive_drop | expected/pending 同时变化并影响后续 phase |
| wait token | 当前 phase false、上一 phase true、过旧 token 非法 |
| wait parity | phase 0/1 多轮复用 |

### 11.2 并发交错

对同一个对象随机交错：

```text
多个 warp arrive
多个 TMA complete
expect 与 complete 到达顺序反转
wait 与最后一个 complete 同周期
wait 登记与 phase transition 相邻
```

每次把 RTL 最终 64-bit object、response 顺序和 reference model 比较。

### 11.3 多对象与容量

验证：

- 同 WG 超过四个对象。
- 多 WG 使用相同 CTA-relative offset，但 resolved physical address 不同。
- barrier 数量超过 waiter slot 数仍能 polling 完成。
- 不同地址并发不会串状态。

### 11.4 TMA 集成

覆盖：

- 一条 expect 对应一条 TMA。
- 多条 TMA 指向同一 barrier。
- 同一 warp 的多条 TMA 指向不同 barrier。
- 不同 warp 指向同一 barrier。
- command completion 乱序。
- OOB/fill。
- bulk 和 tensor G2S。
- 最后一笔 shared write ack 前 wait 必须为 false。
- complete-tx 提交并 phase 转换后 wait 才为 true。

### 11.5 生命周期

覆盖：

- `inval` 后内存作为普通 shared data 使用。
- WG 退出时有 inflight TMA，必须阻止 allocation 回收。
- allocation epoch 重用。
- warp reset/flush 时未完成 request、response、waiter 的清理。

### 11.6 内存序 litmus

建议构建小型 litmus：

```text
producer:
  st.shared data
  mbarrier.arrive.release

consumer:
  wait.acquire == true
  ld.shared data
```

以及：

```text
generic shared write
fence.proxy.async.shared
TMA/S2G async read
```

和：

```text
TMA G2S write
complete_tx
wait.acquire
generic shared read
```

这些测试不应只比较最终数据，还要在 RTL assertions 中观察 ack/phase/wakeup 的相对顺序。

### 11.7 推荐 assertions

```text
同一对象最多一个 active RMW lock
write64 ack 前不得发 response/wakeup
phase toggle -> pendingAfter == expectedAfter
phase toggle -> txAfter == 0
TMA complete event -> command 所有 shared writes 已 ack
每个 accepted TMA command 至多一个 complete event
每个 complete event 必须对应 accepted command
waiter wake -> token/parity 对应 phase 已完成
allocation release -> 无引用该 epoch 的 inflight 状态
```

### 11.8 PMU

建议新增：

```text
mbar_req_core
mbar_req_tma_complete
mbar_rmw_cycles
mbar_queue_stall
mbar_shared_port_stall
mbar_same_object_conflict
mbar_wait_sleep
mbar_wait_poll_fallback
mbar_wakeup
mbar_protocol_error
mbar_signed_tx_negative_cycles
```

这些指标用于判断单发射单元是否成为瓶颈，以及 waiter/cache 是否值得优化。

## 12. TMA v2 常量的独立维护决策

2026-07-25 已决定退役根仓库的 `tools/tma_spec.py` 和 `spec/tma_v2.json`。原因是 gpgpu、Spike 和 testcases 本质上是独立 submodule，没有必要为了少量公共常量引入跨子模块生成和构建依赖。

现有生成物保留原路径并转为普通源码：

| 所属项目 | 文件 | 维护范围 |
|---|---|---|
| gpgpu | `gpgpu/ventus/src/pipeline/TmaV2Spec.scala` | RTL 和 Chisel 测试使用的 TMA v2 常量 |
| Spike | `spike/riscv/ventus_tma_v2_spec.h` | ISA 模型使用的 TMA v2 常量 |
| testcases | `testcases/_get_case/common/ventus_tma_v2_spec.h` | OpenCL host 测试和 C++ reference model 常量 |
| gpgpu tests | `gpgpu/ventus/tests/resources/tma_v2_vectors.json` | 独立测试资源 |

维护规则变为：

1. 每个 submodule 的常量文件由该项目直接维护和提交。
2. 对外可见的 opcode、funct、descriptor layout 和 status code 发生变化时，在同一个设计变更中同步三个项目。
3. 不要求任一 submodule 在构建时读取根仓库文件。
4. 不建立跨 submodule include path、symlink 或临时生成步骤。
5. mbarrier object size/alignment、操作编码属于需要跨项目同步的 ABI。
6. waiter entry、state cache、RMW FSM 等微架构参数只属于 gpgpu，不写入 Spike/testcases 公共 header。

这种方案接受少量人工同步成本，换取各项目构建独立、源文件直接可读，并避免把“每 WG 四个 entry”之类临时 RTL 设计错误地固化成跨项目公共规范。

## 13. 关键设计决策建议

建议现在冻结以下决策：

1. **采用独立 MBarrierUnit。**
2. **shared 8B object 是唯一架构真值。**
3. **首版只支持 PTX-like layout v0、CTA scope。**
4. **首版使用单发射原子 RMW，先正确再并行。**
5. **删除固定 4/WG 的架构限制。**
6. **waiter CAM 只做加速，满时允许 polling fallback。**
7. **TMA G2S command 显式锁存 mbar address。**
8. **completion 使用地址和 allocation epoch，不再使用 barrierId 作为架构身份。**
9. **TMA completion channel 可背压，禁止丢 pulse。**
10. **保留当前“shared write ack 后才 command complete”的正确路径。**
11. **tx-count 按 signed 范围实现。**
12. **arrive/token 和 wait/predicate 增加真实 writeback。**
13. **release/acquire/proxy 被写入架构契约，不只写成某个现有 drain 信号。**
14. **Spike、RTL 和测试共享独立 reference model 语义。**

## 14. 最终判断

当前实现适合作为验证 TMA completion 流程的原型，因为它已经具备：

- 独立 controller。
- TMA command completion。
- generation 防陈旧事件。
- wait warp mask。
- shared write ack 后再完成。

但它不适合作为最终 CUDA-like mbarrier 架构，因为：

- 固定表而不是用户 shared object 保存真值。
- 固定四个对象。
- TMA 隐式绑定。
- shared memory 没有 mbarrier 原子 endpoint。
- token、predicate、inval、signed tx 和完整 ordering 缺失。

最合理的演进不是继续强化固定表，而是：

```text
保留 completion/wait 调度经验
  + 删除固定 entry 权威状态
  + 引入 shared-object atomic RMW
  + TMA 显式携带地址
  + waiter/cache 降级为纯加速结构
```

这条路线既符合 CUDA/PTX 对用户定义 shared-memory barrier object 的公开语义，也能利用 Ventus 当前 banked shared memory、DMA completion 和 warp scheduler 的已有基础。
