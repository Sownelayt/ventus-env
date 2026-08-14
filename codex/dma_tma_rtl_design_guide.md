# Ventus TMA RTL 设计与组件交互导读

更新时间：2026-08-11

本文只描述当前未提交工作树中的唯一 TMA 实现。它不按 Git commit 复原设计，
也不保留已删除的 `DMA_core`、旧 S2G backend、copysize 或 v0/v1 兼容路径。
软件可见的指令与 TensorMap ABI 见
[`ventus_tma_instruction_abi_reference.md`](./ventus_tma_instruction_abi_reference.md)。

## 1. 总体结构

```text
Issue / pipe
  | data command                         | group / mbarrier / proxy control
  v                                      v
TmaV2DmaCore                        TmaV2S2GGroupTracker
  |-- TmaV2Ingress                      TmaV2MbarrierController
  |     |-- one-entry data FIFO
  |     |-- Descriptor request client 0/1
  |     |-- TmaV2DescriptorCompiler
  |     |-- TmaV2CommandBinder
  |     `-- TmaV2WindowPlanner
  |
  `-- TmaV2WindowSubsystem
        |-- TmaV2DescriptorService
        `-- TmaV2WindowEngine
              |-- payload slots / line contexts
              |-- per-command translation cache
              |-- shared-ready queue / S2G ack table
              v
          TLB + L2 cache + SharedMemory
```

数据面一次只拥有一条 active command。Ingress 另有一个 depth-1 FIFO，允许后一条
数据命令提前完成校验、descriptor lookup、坐标绑定和 mbarrier transaction reserve。
Lookahead 不分配 window、LineContext 或 cache/shared 请求，因此不会与 active command
的数据流交错。active command 完成后，准备好的下一条命令立即 promotion；FIFO 出队和
下一条输入补入可以同拍发生。

TensorMap prefetch/invalidate 使用独立 depth-1 control queue，不占数据命令 FIFO。
因此 active data command、一个 data lookahead 和一个 control hint 可以同时驻留，但
不存在第三条数据命令。

## 2. 当前关键文件

| 文件 | 职责 |
|---|---|
| `TmaV2Spec.scala` | 指令、dtype、状态、容量默认值和 ABI 常量 |
| `TmaV2Frontend.scala` | descriptor 编译和 command-local 坐标绑定 |
| `TmaV2WindowPipeline.scala` | rank 1–5 cursor 展开、OOB clipping、global/shared lane 映射 |
| `TmaV2WindowEngine.scala` | PayloadSlot、LineContext、TLB/cache/shared 调度和完成 |
| `TmaV2Backend.scala` | descriptor service 以及内部 Decoupled bundle 定义 |
| `TmaV2DmaCore.scala` | Ingress、子系统、真实 TLB/L2/shared 端口适配与仲裁 |
| `TmaV2Completion.scala` | S2G group tracker 和 WG 分区 mbarrier controller |
| `pipe.scala` | 将 issue、completion、group、mbarrier 和 proxy fence 接入流水线 |
| `PerfCounters.scala` | `TmaPerfCounters`，仅在 PMU-on 构建实例化 |

## 3. 从指令到 active command

### 3.1 Compact request

数据指令进入 `TmaV2Ingress` 时只保存一次公共字段：

```text
funct, inst[31:27], wid, asid, group, rd/in1, rs1/in2[0..4], rs2/in3
```

`inst[31:27]` 在使用点按 funct 解释：Tensor/Bulk S2G 的 reduce mode、Bulk
reduce dtype，或 TensorMap control subop。没有为同一位域保留多份历史寄存器。

### 3.2 Bulk validation 和切窗

Bulk G2S/S2G 均要求：

```text
global address % 16 == 0
shared address % 16 == 0
bytes != 0
bytes % 16 == 0
```

Window 按 shared 的 128B set 边界切分，而不是机械地从起点每次取 128B：

```text
firstChunk = min(bytes, 128 - sharedAddress[6:0])
middleChunk = 128
lastChunk = remaining bytes
```

例如 `shared+0x10,size=128` 产生 `112B+16B`；`shared+0x70,size=32`
产生 `16B+16B`。16B 对齐不等于 128B 对齐，首块不完整是合法情形。
global cursor 同步增加相同字节数，但 cache 侧随后按自己的 128B line 边界分解。

每个 Bulk lane 始终是完整 16B atom。RTL 中不再存在 1–15B lane 起始 offset、
signed shared delta、128-bit 任意右移或 `atom/atomCount` 兼容字段。

### 3.3 DescriptorService

Tensor data command 使用 descriptor client 0；prefetch/invalidate 使用 client 1。
`TmaV2DescriptorService` 提供默认 4-entry compiled TensorMap cache：

1. 以 descriptor virtual address 和 ASID 查 tag；
2. hit 直接返回 compiled entry；
3. miss 经 DmaCore 的 TLB 和专用 cache source `all-ones` 读取 128B descriptor；
4. `TmaV2DescriptorCompiler` 校验并生成运行时需要的派生字段；
5. demand response pin 住对应 compiled entry，避免 lookahead 使用期间被 control miss 替换。

Compiler 的主要检查包括 magic、rank 1–5、dtype、global/box dimensions、
base/stride alignment、stride 不重叠、interleave/swizzle/OOB 组合和 65-bit
`requiredSpan={overflow,span[63:0]}`。错误 descriptor 也可缓存，后续 hit 不重复 refill。

### 3.4 CommandBinder

`TmaV2CommandBinder` 不搬运 payload。它把 command 的 5 个 signed 32-bit 坐标与
compiled descriptor 结合，计算每个维度的 origin component、最终 bounding-box
global 起点和 command-local status。

当前动态对齐规则是：

- 最终 bounding-box global 起点必须 16B 对齐并落在 32-bit 地址空间；
- B4x16 `coordinate[0]` 必须是 32 elements 的倍数；
- B4x16P64/B6 `coordinate[0]` 必须是 128 elements 的倍数；
- Tensor S2G copy/reduce 不接受任一 active negative starting coordinate；
- Tensor G2S 允许对齐的负坐标，后续由 Planner 生成 OOB fill。

错误命令进入 seal 路径，只产生 status/completion，不分配 TLB、cache 或 shared 流量。

### 3.5 Mbarrier reserve

合法 G2S command 在进入 Engine 前发送 `txReserve(bytes,wid)`。
`TmaV2MbarrierController` 根据当前 warp/WG 的 `arrive.expect_tx` 状态返回
`barrierValid/barrierId/generation`。这些绑定字段只来自 reserve response，数据
请求本身不携带旧的 `dma_barrier_valid/id/bytes` 副本。

## 4. WindowPlanner 与地址约束

`TmaV2WindowPlanner` 把 rank 1–5 tensor box 展开为最多 8 个 lane 的 128B
window。高维 cursor 通过流水 stage 传递；填满后保持 II=1，不为冷路径节面积而
串行化 DescriptorCompiler 或 CommandBinder 的乘法器。

每个 lane 给 Engine 的信息是：

```text
valid, globalAddress, globalBytes,
sharedDelta, sharedBytes, oobBytes
```

不存在 lane `globalOffset`。公开合法输入保证普通 lane 的 `globalAddress` 为 16B
边界；右边界 OOB 只能缩短有效前缀，不会把有效数据的起点移到 atom 中间。

固定 sub-byte 特例：

- B4x16：32 elements = 16B，global/shared 都紧凑 packed；
- B4x16P64：每 lane 8B packed payload，映射到 16B shared lane 的固定低 8B；
- B6 G2S：每 lane 12B packed payload；最多产生一个额外、word-aligned cache-line tail；
- B6 S2G：从 16 个 shared byte 的 low 6 bits 重打包成 12B。

因此 Engine 只需要 word selector、byte prefix mask 和 B6 唯一 tail，不需要通用
arbitrary-byte LinePermuter。

## 5. WindowEngine 数据结构

默认容量：

| 结构 | 默认 | 作用 |
|---|---:|---|
| PayloadSlot | 6 | 保存完整 128B payload 和 window ownership |
| LineContext | 40 | 追踪窄 cache-line 请求、路由和翻译状态 |
| shared-ready | 8 | 解耦 payload 与 shared request/response |
| S2G write-ack | 32 | 追踪 Put/AMO 的最终 L2 ack |
| descriptor cache | 4 | compiled TensorMap entries |
| translation cache | 每 command 4 | command-local VPN→PPN |

PayloadSlot 和 LineContext 分离的原因是：一扇 128B window 可能涉及多个 cache line，
但大量 in-flight line 不应复制完整 128B payload。G2S response 到达后直接写入目标
PayloadSlot并释放窄 LineContext；S2G 则先占 PayloadSlot 收集 shared data，再逐 line
发写请求。

Engine 始终只保存一条 immutable active command。`seal` 表示该命令不会再收到新
window；完成条件同时要求 seal 已见、所有 PayloadSlot/LineContext/shared/ack 状态
排空。

## 6. G2S 工作流

```text
Window
 -> 按 cache line tag 分成 word-aligned segment
 -> LineContext 分配/同 line 合并
 -> command-local TLB lookup/miss/coalesce
 -> L2 Get
 -> response data 按 cacheWord/destinationWord 写 PayloadSlot
 -> OOB fill 合并
 -> shared-ready
 -> SharedMemory write
 -> 所有 replay beat/write ack 完成
 -> window retire
```

Engine 保留 4 个 recent-line leader，可让相邻、同 cache line 的两个消费者共享一次
L2 Get。它只保存固定 compact pair，不恢复任意多消费者搜索。

Shared 内部请求格式为：

```text
write
source
sharedSetIdx
8 * sharedAtomIndex[2:0]
data[1023:0]
mask[127:0]
```

每个有效 atom 必须属于同一个 128B shared set；`mask` 是唯一 byte 有效定义。
外层适配器将其转换为现有 32-bank SharedMemory request。bank conflict 可能导致
多拍 response，Engine 用 `wordMask[31:0]` 合并 replay，直至所有活动 word 返回。

## 7. S2G copy 与 reduce 工作流

```text
Window
 -> PayloadSlot
 -> shared read request
 -> 合并 replayed shared response
 -> 固定 word route / sub-byte repack
 -> LineContext + TLB
 -> L2 PutFull/PutPart，或每 32-bit element 一条 AMO
 -> write-ack scoreboard
 -> 最终 AccessAck/AMO ack
 -> window/command completion
```

copy 根据 byte mask 选择 PutFull 或 PutPart。Tensor 和 Bulk reduce 都不会退化成
read-copy-write；DmaCore 把已经 scalarize 的完整 32-bit element 映射到 L2
AtomicUnit 的 AMO opcode/param。每条 reduce request 必须恰好有一个完整 4B word。

S2G completion 等待最终 ack，而不是在 Put/AMO 被 L2 接受时提前完成。这个条件决定
group wait 的内存可见性，也是 write-ack table 必须独立存在的原因。

## 8. TLB、cache source 与非 2 次幂容量

`requestEntries` 和 `writeAckEntries` 的合法范围都是 `2..256`，默认 `40/32`。
内部 data source namespace 为：

```text
nextPow2(max(requestEntries, writeAckEntries))
```

默认宽度为 6 bit，256 项配置为 8 bit。由于 active command 只有一个方向，G2S
LineContext 与 S2G ack 可以复用 `0..255`；descriptor refill 始终使用物理 source
`1023`，不能被识别为 data response。

40 等非 2 次幂配置不能依赖 Vec 动态索引截断。free/valid/response 命中均显式限制
`source < configuredEntries`；计数器使用 `log2Ceil(entries+1)`，256 项时可准确表示
0 到 256。257 及以上在 elaboration 阶段失败。

每条 command 的 4-entry translation cache 在 command completion 时清空。软件必须
在 completion 可见前保持页表映射和 ASID 语义稳定。

## 9. Completion、group、mbarrier 与 proxy fence

### 9.1 S2G group

每个 warp 有 4-slot group ring。S2G command 进入 open group，`commit_group` 封闭
当前 slot，`wait_group keep=N` 等待旧 group 排空。tracker 只消费 DmaCore 的最终
completion，因此 Put/AMO 尚未 ack 时 wait 不会错误放行。

### 9.2 Mbarrier

Mbarrier 表静态按 `[WG][4]` 分区，entry id 为 `{owner,localEntry}`。控制器保留
8-bit generation 和 phase 防御。G2S transaction 完成后更新权威表，并向 shared
中的 8B barrier object 发镜像写；wait 必须同时观察 transaction completion 和镜像
write ack，旧 generation 的迟到 completion 不能释放新 phase。

### 9.3 `fence.proxy.async.shared`

该 control 指令不进入 TMA data plane。`pipe.scala`/control scheduler 检查发令 warp
的 LSU ShiftBoard outstanding；只有同一 warp 的相关 LSU 请求排空后 fence 完成。
它建立 generic shared proxy 与 async/TMA shared proxy 的顺序，不等待其他 warp，
也不替代 S2G group 或 G2S mbarrier completion。

## 10. PMU 与面积边界

普通构建默认：

```text
VENTUS_PMU_TMA=0
VENTUS_PMU_TMA_DETAIL=0
```

只有显式设置 `VENTUS_PMU_TMA=1` 才实例化 `TmaPerfCounters`；detail 计数还需
`VENTUS_PMU_TMA_DETAIL=1`。PMU-off 是面积比较和默认综合基线。

当前保留的性能面积投入包括：depth-1 lookahead、Planner II=1、并行 descriptor/
binder 算术、PayloadSlot/LineContext 分离、recent-line pair 和独立最终 ack table。
它们都服务可观测吞吐或延迟。已删除的是无合法输入来源的 sub-16B offset、动态
byte permuter、重复 barrier/subop 状态和固定 64-entry source namespace。

## 11. 调试顺序

遇到“命令不完成”时按 ownership 顺序检查：

1. Ingress `state`：validation、descriptor、binder、reserve 还是 ready；
2. `commandValid/commandSealed`：Engine 是否拥有命令、是否收到 seal；
3. `activeWindows/activeRequests/activeShared/activeWriteAcks`：卡在哪一层；
4. TLB miss owner 与 source 是否匹配；
5. cache response 是 descriptor source 1023、G2S line source，还是 S2G ack source；
6. shared replay `wordMask` 是否覆盖所有活动 word；
7. S2G 最终 ack、group completion 或 G2S mbarrier mirror ack 是否尚未返回。

静态对齐错误应表现为 status + zero memory traffic。若错误命令产生 TLB/cache/shared
请求，属于前端 seal/ownership bug；若数据正确但 completion 不来，优先检查
LineContext、shared-ready、write-ack 和 barrier generation 的释放条件。

## 12. 主要验证入口

```text
TmaV2Frontend_test       descriptor/binder/planner/rank/dtype/golden
TmaV2Backend_test        descriptor service、WindowEngine、replay/credit
TmaV2DmaCore_test        真实 TLB/cache/shared 端口和 bulk/tensor/reduce
TmaV2Completion_test     mbarrier/generation/mirror
TmaV2GroupTracker_test   commit/wait/ring accounting
TmaV2Capacity_test       source 255、乱序返回、256 项容量
TmaControlScheduler_test group/mbarrier/proxy 与 LSU outstanding
AtomicUnit_test          AMO 最终 ack 和互斥
```

端到端唯一功能套件是 `dma_tma_v2_func_test`，应在 spike、gvm、gvm-nocache、
rtl、rtl-nocache 五个 backend 使用同一 verdict。任何新的地址优化都必须同时覆盖
B4/B6、OOB、interleave、swizzle、跨 cache line/page、shared `+0x10/+0x70`
边界以及 S2G final ack。
