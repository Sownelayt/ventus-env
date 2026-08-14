# Ventus TMA V3.1：Descriptor 编译缓存与单命令窗口流水

> **历史候选说明：** 本文记录的是未冻结的 V3.1 实现，其中曾支持
> elementStride 1–8。V3.2 已按设计决策撤销该功能、删除相关硬件并
> 清除生成 RTL 中所有运行时除法/取余；V3.1 的功能和面积结论不能代替
> V3.2。最新结果见 `TMA_V3_2_ELEMENT_STRIDE_REMOVAL_AND_PERFORMANCE.md`。

> 日期：2026-07-29  
> 状态：**实现候选，功能回归通过，性能门槛未通过，未冻结**  
> 当前只读基线：**V3.0**  
> 范围：非 Reduce 的 bulk、bulk.tensor、G2S、S2G 和 roundtrip  
> CUDA 参考：复用已冻结的 H100/B200 数据，没有新增付费 GPU 运行

## 结论

V3.1 已完成计划中的主体结构改造：

- TensorMap 静态计算从每条命令重复执行，改为单路 Descriptor Compiler
  加 4 项 compiled store；
- 删除旧 `emitMode/wide*` fast path 和对应 Setup 周期；
- 动态命令只经过小型 Binder，静态 descriptor 结果直接复用；
- WindowPlanner 改为三级弹性流水，独立测试中 32 KiB 命令在填充后连续
  发出 256 个 window，Planner 本身达到 `II=1`；
- 数据搬运严格限制为单 active command，TensorMap control lane 仍可独立
  prefetch/invalidate；
- 后端资源调整为 24 window、16 global request、8 shared-ready 和
  32 S2G ack，并修正 G2S/S2G credit 与数据所有权释放；
- 补齐外层 `elementStride=1..8`，并同步修正 Spike/GVM 参考模型。

功能结果是完整的：Frontend 8/8、最终 descriptor 31/31、capacity 27/27、
常用非 Reduce 60/60 均通过，with-cache 和 nocache GVM 都成功重建。

但 V3.1 **不能冻结**。32 KiB 的 Planner 虽能连续产生 256 个 window，
端到端最长连续 `window.fire` 只有 G2S 24、S2G 52；4/16/32 KiB 的发布
加速门槛也没有达到。PMU 明确显示瓶颈已经从旧 Setup/Planner 转移到
单端口 L2 请求/响应、window ROB 和 S2G ack：

- 32 KiB G2S：16 个 request 与 24 个 window 全部打满；
- 32 KiB S2G：16 request、24 window、32 ack 全部打满；
- 增大队列的隔离实验没有改善 S2G 860-cycle 探针；
- 当前 L2 `SourceD` 仍是单 busy FSM，无法持续接受一个请求/cycle。

因此本轮正确的发布判断是：**保留 V3.1 候选和全部证据，V3.0 继续作为
当前基线；下一步先改 L2/TMA 端口吞吐，再重测和决定 V3.2。**

## 1. 最终数据通路

```text
TensorMap control lane
  prefetch / targeted invalidate / invalidate-all
              |
              v
  +---------------------------+
  | 4-entry compiled store    |<---- {ASID, address[31:7]}
  | INVALID/COMPILING/VALID   |
  +-------------+-------------+
                | miss / coalesce
                v
       descriptor refill arbiter
                |
                v
  +---------------------------+
  | single Descriptor Compiler|
  | counter + rank + 1 mul    |
  +-------------+-------------+
                |
                +---- commit/kill ----> compiled store

Data command lane: strictly one active command
  ingress -> compiled hit/wait -> Binder
                               -> CursorExpand
                               -> LaneMap
                               -> LineCanonicalize/skid
                               -> 24-entry Window Engine
                                  | 16 request credits
                                  | 8 shared-ready credits
                                  | 32 independent S2G ack tags
                                  v
                         TLB -> L2/cache <-> shared
                                  |
                         completion / mbarrier / group
```

这个拆分去掉了“大状态机串行控制所有阶段”的结构。各级通过
`valid/ready`、有限队列和资源 credit 解耦；在前后无依赖时可同时工作。
保持单 active data command 是有意的面积/控制选择，不等于内部只有一个
window 或一个 memory request。

## 2. Descriptor Compiler 与 compiled store

主要实现位于：

- [`TmaV2Frontend.scala`](../gpgpu/ventus/src/pipeline/TmaV2Frontend.scala)
- [`TmaV2Backend.scala`](../gpgpu/ventus/src/pipeline/TmaV2Backend.scala)
- [`TmaV2DmaCore.scala`](../gpgpu/ventus/src/pipeline/TmaV2DmaCore.scala)

### 2.1 缓存与状态

compiled store 使用 4 项 round-robin，tag 为
`{ASID, TensorMap address[31:7]}`。每项有：

- `INVALID`：可分配；
- `COMPILING`：refill/compile 正在进行，同 tag 请求合并；
- `VALID`：返回静态编译结果，包括非法 descriptor 的 status。

非法 descriptor 也会缓存，因此重复使用同一非法 TensorMap 不会重复
refill 和验证。Compiler 忙时，其他已 `VALID` 的 hit 仍能返回；不同 tag
miss 背压。prefetch 在命中、分配或合并后即可完成，不等待数据命令。

定向 invalidate 同时匹配地址和 ASID。若与同 tag 编译竞争，则设置 kill，
禁止旧结果提交；全局 invalidate 清除 store 并取消当前提交。

payload 与 descriptor refill 共享 cache 端口。payload 正常优先，但连续
推迟 descriptor 7 次后强制给 Compiler 一次机会，避免活动搬运使 Compiler
永久饥饿。

### 2.2 编译内容

`TmaV2CompiledDescriptor` 保存：

- dtype、rank、interleave、swizzle、OOB 与静态 status；
- global dimensions/strides、box dimensions、element strides；
- `stepOffsets`、`logicalBytes`；
- `interleaveSliceStride`、`channelsLog2`、`elementByteShift`。

Compiler 用一个共享的 64×64 乘法器、`busy`、micro-op counter 和 rank
counter，在 rank 1–5 下最多 11 个 busy cycle。外层 element stride 的
1–8 倍步长采用 shift/add，不增加第二个宽乘法器。

### 2.3 elementStride 修复

正式 descriptor 回归最初在 `rank2_element_stride2` 暴露了两个问题：

1. RTL 把所有 `elementStride != 1` 判为 unsupported，软件已登记的
   mbarrier 字节数无法完成，表现为等待；
2. RTL 修复后，Spike/GVM 参考模型仍做相同的旧判定，导致 RTL 已完成而
   reference 没有扣减 barrier，最终发生 PC 分叉。

最终语义为：

- 活跃维的 element stride 支持 1–8；
- non-interleave 的 dim0 stride 按 CUDA 语义忽略；
- 外层输出数量为 `ceil(boxDim / elementStride)`；
- global 坐标按 stride 跳跃，shared 输出保持稠密；
- interleave 的 dim0 stride 大于 1 仍明确 unsupported，需未来增加
  byte-within-lane codec。

RTL、独立 C-model、DmaCore 集成测试与
[`ventus_tma_v2_tensor.inc`](../spike/riscv/insns/ventus_tma_v2_tensor.inc)
现在采用同一规则。修复前诊断日志保留在正式 descriptor 结果目录中。

## 3. 删除 Setup fast path 与动态 Binder

最终源码中已不存在：

- `TmaV2EmitMode`、`emitMode`；
- `wideLineCount`、`wideGlobalBase`、`wideRowShift`；
- `sDense`、`sLogicalDense`；
- `TmaV25DIterator`、`TmaV2LayoutMapper`。

静态 descriptor 只由 Compiler 计算一次。局部 Command Binder 只处理：

- command ID、direction、reduce 元数据和 shared base；
- 坐标相关的 `originComponents`；
- shared-base 对齐、负坐标约束和动态地址溢出。

Binder 不重新计算 `logicalBytes`、dense layout、step offsets 或 fast-path
类型。Planner token 直接携带 command ID、direction、reduce 与 completion
信息，不再读取可变化的 live sideband。

## 4. WindowPlanner 三级弹性流水

[`TmaV2WindowPipeline.scala`](../gpgpu/ventus/src/pipeline/TmaV2WindowPipeline.scala)
中的 Planner 分为：

1. `CursorExpand`：展开 8 个 lane 的 cursor 快照、下一 cursor 和 last；
2. `LaneMap`：计算 global/shared 地址、OOB/fill、codec；
3. `LineCanonicalize`：形成至多 16 个 fragment，去重 line tag，再进入
   输出 skid buffer。

cursor 只在第一级 enqueue 成功时前进。任一级背压时 token 和 cursor
保持稳定。最后 token 进入流水后停止生产；最后输出被接收并清空流水后，
Planner 才接受下一命令。

独立严格吞吐测试的结果是：

- 32 KiB contiguous；
- 256 个 128B window；
- 流水填充后，相邻输出之间没有内部气泡；
- 随机背压下与独立 C-model 逐 window 一致，无重复、无遗漏。

这证明 Planner `II=1`，但不代表现有 L2 后端能持续消费一个
window/cycle。

## 5. 单 active command 与 control lane

接口保留 2-entry command ID 宽度以减少系统修改，但 ingress、DmaCore 和
WindowEngine 均增加断言：活动数据搬运命令数不得超过 1。

第二条 bulk/tensor 数据命令必须等第一条 completion。TensorMap prefetch
和 invalidate 走独立 control lane，因此当前数据搬运期间仍能分配、合并
或编译其他 TensorMap。

旧 uarch 的 `outstanding=2/4/8` kernel 会先发完全部命令，再执行第一个
wait。该软件序列要求硬件同时接受多条数据命令，与 V3.1 的明确契约形成
循环依赖，因此这三项被正式标为 **not applicable**，没有伪装成通过。
单命令阻塞第二条命令，以及活动搬运期间接受 descriptor prefetch，均由
Chisel 定向测试覆盖。

后续 2026-07-30 H100 严格 serial/batched 对照已经确认：同一 CTA、
同一 TensorMap、不同 global/shared tile 的 2/4/8 条 4 KiB G2S Tensor
TMA 可以显著重叠，N=8 从 serial 3547 cycle 降到 batched 991 cycle。
因此，“单 active command”只能视为 Ventus V3.1/V3.2 的面积与实现复杂度
取舍，不能解释为 CUDA 硬件也只允许单条在途。该结论不改变本版本已经冻结
的接口约束，但应作为后续版本是否增加少量 command slot 的量化依据。

扩展到固定资源 N=48 后，加速在约 3.1–3.2x 饱和；稳定 batched
completion 间隔约 146–149 cycles/4 KiB，而 serial 约
477 cycles/4 KiB。这不要求复制几十套数据通路。若后续版本要恢复 CUDA
式小命令重叠，应先评估少量轻量 command scoreboard/credit（例如 4 项）
能否覆盖约 3.2 的 latency/throughput 比，而不是直接引入 32 或 48 个
完整 command context。该项仍需用 Ventus 自身端口和面积数据验证。

## 6. 后端资源、credit 与所有权

最终默认值是：

| 资源 | V3.0 | V3.1 |
|---|---:|---:|
| compiled descriptor | 2 | 4 |
| active data command | 最多 2 | 严格 1 |
| window metadata/data | 8 | 24 |
| global request | 4 | 16 |
| shared-ready | 4 | 8 |
| S2G ack tag | 16 | 32 |

索引、计数器和 PMU 宽度均由参数推导，不再硬编码为 8/4/4。

G2S 在发 read 前预留 response/storage credit；cache response 转交
transform/shared queue 后立即释放 global request entry。这样返回响应
总有存储位置，不靠回压一个已经返回的外部响应。

S2G global write 被 cache 接受后立即释放 128B window data；独立 ack
scoreboard 只保留 command completion 所需的信息。命令 retire 同时等待
window drain 和 ack 归零。测试覆盖了“最后 window 先释放、较早 write
ack 后返回”的所有权边界。

source ID 直接索引 16-entry request 表或 32-entry ack 表；重复 global
line 的合并只使用小型 tag CAM，没有把 16/32 项改成新的全关联数据 mux。

## 7. PMU

新增并接入 GPGPU 顶层的计数包括：

- compiled hit/miss、compile、coalesce、invalidate-kill；
- compile cycles、bind cycles；
- Planner produced/fire/stall；
- 最大 active window/request/shared/ack；
- 最长连续 `window.fire`；
- ROB/request/shared/ack full stall；
- G2S response、S2G ack 数量与延迟总和；
- cache request stall。

capacity 数据使用一次 warmup 加一次 measurement，因此下表的事件总数是
两个 32 KiB 命令，共 512 个 window：

| 32 KiB | G2S | S2G |
|---|---:|---:|
| Planner produced/fire | 512/512 | 512/512 |
| Planner stall cycles | 696 | 780 |
| max window/request/shared/ack | 24/16/2/0 | 24/16/4/32 |
| longest continuous `window.fire` | 24 | 52 |
| ROB full cycles | 698 | 784 |
| request full cycles | 718 | 808 |
| cache request stall cycles | 36 | 143 |
| response/ack count | 512 | 512 |
| response/ack latency sum | 19338 | 43231 |
| average response/ack latency | 37.77 | 84.44 |
| ack-full stall cycles | 0 | 766 |

`produced == fire == 512` 证明没有丢 window。最长连续区间和 full counters
则证明端到端气泡来自后端 credit 被 L2/ack 消耗速度限制，而不是 Planner
漏发。

## 8. 测试与构建结果

| 验证 | 结果 |
|---|---|
| Frontend 完整 Chisel suite | 8/8 pass |
| Backend 既有完整 suite | 11/11 pass |
| 新 S2G 旧 ack/最后 window 所有权测试 | pass |
| descriptor store/coalesce/hit/replace 定向回归 | 2/2 pass |
| descriptor ASID/address invalidate + kill | pass |
| DmaCore 既有完整 suite | 18/18 pass |
| 新 rank2 elementStride=2 DmaCore 集成 | pass |
| descriptor 功能矩阵 | 31/31 pass |
| capacity 128B–32KiB | 27/27 pass |
| 常用 non-Reduce，3 attempts | 60/60 pass |
| uarch 有效项 | 13/13 pass |
| uarch 多 active 旧项 | 3 explicitly not applicable |
| contention core/pressure | 58 outcomes；16 pass、42 short-timeout、0 failure |
| with-cache GVM | build/install pass |
| nocache GVM | build/install pass |
| 独立 V3.1 RTL | 5 个 `.sv` 已生成并固定 SHA-256 |

contention 使用 3 秒 case timeout，目的是让任何多 warp/多 CTA 等待不会
长时间占用仿真资源。timeout 仅作为有界进度观测，不被计作功能通过。

完整结构化结果见：

- [`TMA_V3_1_SUMMARY.json`](data/TMA_V3_1_SUMMARY.json)
- [`TMA_V3_1_CAPACITY_GATE.csv`](data/TMA_V3_1_CAPACITY_GATE.csv)
- [`TMA_V3_1_COMMON_COMMAND_COMPARISON.csv`](data/TMA_V3_1_COMMON_COMMAND_COMPARISON.csv)
- [`TMA_V3_1_RTL_SHA256.txt`](data/TMA_V3_1_RTL_SHA256.txt)
- [`TMA_V3_1_EVIDENCE_SHA256.txt`](data/TMA_V3_1_EVIDENCE_SHA256.txt)

## 9. V3.0、V3.1、H100 与 B200 精确容量周期

H100/B200 是固定实机参考，V3.0/V3.1 是 GVM/RTL 周期。两者没有统一
频率、L2、DRAM 或系统规模，表中的 cycle 只能按同参数观察结构特征，
不能当成同频绝对性能。

| bytes | direction | V3.0 | V3.1 | V3.0/V3.1 | H100 | B200 |
|---:|---|---:|---:|---:|---:|---:|
| 128 | G2S | 123 | 133 | 0.925× | 460 | 461 |
| 128 | S2G | 88 | 97 | 0.907× | 62 | 93 |
| 128 | roundtrip | 194 | 213 | 0.911× | 553 | 568 |
| 1024 | G2S | 145 | 147 | 0.986× | 460 | 461 |
| 1024 | S2G | 120 | 129 | 0.930× | 90 | 121 |
| 1024 | roundtrip | 248 | 259 | 0.958× | 581 | 596 |
| 4096 | G2S | 229 | 208 | 1.101× | 460 | 461 |
| 4096 | S2G | 202 | 201 | 1.005× | 186 | 217 |
| 4096 | roundtrip | 414 | 392 | 1.056× | 678.5 | 692 |
| 16384 | G2S | 565 | 447 | 1.264× | 608 | 609 |
| 16384 | S2G | 523 | 501 | 1.044× | 570 | 601 |
| 16384 | roundtrip | 1071 | 931 | 1.150× | 1209 | 1224 |
| 32768 | G2S | 1013 | 767 | 1.321× | 756 | 759 |
| 32768 | S2G | 939 | 881 | 1.066× | 1082 | 1113 |
| 32768 | roundtrip | 1935 | 1631 | 1.186× | 1869 | 1884 |

V3.1 的收益主要随 G2S 容量增长：4/16/32 KiB 分别为
1.101×、1.264×、1.321×。这符合删除重复 Setup、提高 request/window
并行度后的预期。S2G 只有 1.005×、1.044×、1.066×，说明 ack/L2 write
路径几乎完全掩盖了前端优化。

小容量有固定开销回退：128B G2S/S2G/roundtrip 分别慢
8.1%、10.2%、9.8%。其中 S2G 97 cycle 相对 V3.0 的允许上限
96.8 cycle，窄幅越界。

## 10. 发布门槛

| 门槛 | 结果 |
|---|---|
| compiled hit 不重新编译 | pass，Backend 单元测试证明 |
| 128B 不回退超过 10% | G2S/roundtrip pass；S2G fail |
| 4KiB 至少 1.3× | fail |
| 16KiB G2S/S2G 至少 1.7× | fail |
| 32KiB G2S/S2G 至少 1.7× | fail |
| Planner 32KiB、256 window 内部连续 | pass |
| 端到端 32KiB 连续 256 `window.fire` | fail：G2S 24、S2G 52 |

自动分析器返回码为 2，并写出全部证据。`freeze_baseline.py` 读取
`performance_gate_passed=false` 后拒绝创建 V3.1 baseline，因此
`CURRENT_VERSION` 保持 V3.0，V3.0 文件没有被覆盖。

## 11. 为什么后端没有达到一个 window/cycle

24/16/8/32 的资源扩展解决了旧 8/4/4 很快耗尽的问题，但不能改变下游
服务率。当前 with-cache 路径仍有：

- 单个 64 KiB L2；
- 单 `SourceD` busy/FSM 服务；
- DMA 与 I/D cache 共享仲裁；
- 每周期单 cache-request 端口；
- S2G 写入后的独立 ack 延迟和 32-entry 上限。

连续布局的 Planner 每个 window 只需一个唯一 global line，理论上最适合
达到 II=1；实际仍把 request/ROB 填满，说明限制不是 stride 多 line/window。

隔离实验曾扩大 tags/FIFO/resources，但 32 KiB S2G 探针仍为 860 cycle，
同时最大 ack 占用只有 34/128。这排除了“继续堆更多 ack tag 就会线性改善”
的假设。实验结果保留在
[`features_ventus_v3_1_latency_covering_probe`](../benchmarks/tma-cycle-compare/results/features_ventus_v3_1_latency_covering_probe/)；
最终源码已经恢复计划规定的 24/16/8/32。

## 12. L2 模型对结果的影响

现有 GVM 的 L2/backing memory 不是 H100 的分区 L2/HBM 周期模型。详细
分析见
[`TMA_RTL_CUDA_ARCHITECTURE_AND_L2_MODEL_ANALYSIS_20260729.md`](TMA_RTL_CUDA_ARCHITECTURE_AND_L2_MODEL_ANALYSIS_20260729.md)。

对本轮结论应作如下限制：

- 当前 PMU 能可靠说明**本 RTL**在 request/ROB/ack 上等待；
- 不能把 V3.1 cycle 乘一个固定比例换算为 H100；
- cold miss、随机地址、跨页和 HBM 排队时，固定低延迟 backing memory
  偏乐观；
- 超过 Ventus 64 KiB、但远小于 H100 50 MB 的工作集，单 L2 容量和
  eviction 又可能偏悲观；
- 32 KiB S2G/roundtrip 比 H100 cycle 更低，不代表真实芯片会更快，
  更可能体现了简化 write/DRAM 模型。

但是 L2 模型失真不能解释掉本轮发布失败：端到端连续 window 是针对当前
RTL 自身的验收，PMU 已直接观察到它没有达到。

## 13. 下一步

优先级应是：

1. **把 TMA 到 L2 的服务端改成可持续接收请求的流水。**先拆分
   `SourceD` 的 lookup/response/data 阶段，允许 hit-return 与下一请求
   重叠；目标是连续布局下 cache request `II=1`。
2. **为 TMA 提供独立或分 bank 的 L2 注入/返回通路。**至少避免单 busy
   FSM 把 16 个 request credit 变成仅用于覆盖延迟、不能提高吞吐。
3. **重新设计 S2G ack 聚合。**保留写数据早释放，但让 ack 可以批量或
   每周期退休，避免 32-entry scoreboard 成为新的节拍器。
4. **增加 L2 hit/miss、SourceD stage occupancy、DMA arbiter stall PMU。**
   现有 TMA PMU 已定位到 L2 边界，下一轮需要把 L2 内部再拆开。
5. **完成当前 42 个 contention timeout 的短分片复测。**保持 3 秒上限，
   把大 case 拆小，而不是把 timeout 延长。
6. 在上述改变后创建 **V3.2** 候选，复用相同 H100/B200 固定参考，重跑
   Ventus 128B、1/4/16/32 KiB、完整 non-Reduce 和端到端 256-window
   验收。只有全部门槛通过后才冻结。

不建议下一步继续增加 window/request/ack 数量。当前实验已经说明资源数量
不是主要斜率限制；继续堆表项只会增加面积和索引逻辑。

## 14. 复现

本地回归均在 `source ./env.sh` 后运行。所有 runner 是一 case 一个 child，
超时立即终止，不会因单个死锁无限计费。

```bash
# descriptor：普通 case 5s，大 case 15s
python3 benchmarks/tma-cycle-compare/run_ventus_feature_short.py \
  --suite descriptor --attempts 1 --warmups 1 --repeats 1 \
  --case-timeout 5 --large-case-timeout 15 --compile-timeout 8 \
  --output benchmarks/tma-cycle-compare/results/features_ventus_v3_1_descriptor

# capacity
python3 benchmarks/tma-cycle-compare/run_ventus_feature_short.py \
  --suite capacity --attempts 1 --warmups 1 --repeats 1 \
  --case-timeout 5 --large-case-timeout 15 --compile-timeout 8 \
  --output benchmarks/tma-cycle-compare/results/features_ventus_v3_1_capacity_final

# 常用 non-Reduce
python3 benchmarks/tma-cycle-compare/run_ventus_short.py \
  --profile full --attempts 3 --case-timeout 4 \
  --warmups 0 --repeats 1 \
  --output benchmarks/tma-cycle-compare/results/ventus_v3_1_full

# 分析；当前预期返回 2，表示保留证据但拒绝冻结
python3 benchmarks/tma-cycle-compare/analyze_v31_results.py
```

CUDA H100/B200 结果继续使用 V3.0 已冻结数据，不需要重新运行 Modal。
