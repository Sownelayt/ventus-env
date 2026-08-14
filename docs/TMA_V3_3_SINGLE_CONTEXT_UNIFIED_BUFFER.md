# Ventus TMA V3.3：单 Context、统一 Buffer 与统一 TensorMap 预处理

## 结论

V3.3 已把数据搬运收敛为严格的 `1 ActiveCommand + 1
LookaheadSlot`，没有多个数据 command context。显式
`prefetch.tensormap` 和普通 Tensor TMA 的隐式 demand lookup 共用一套
Descriptor Service、tag、Compiler 和 4 项 compiled store。后端用 24 项
双向 `TransferSlot` 取代宽 Window ROB 及多份 payload，并保持 G2S/S2G
乱序完成。

功能和性能门槛均通过：

- Frontend 8/8、Backend 17/17、DmaCore 19/19、Completion/Group 10/10；
- 30 项 descriptor/layout、20 项非 Reduce Bulk/Tensor、27 项容量测试
  全部正确；
- 所有冻结容量点都快于 V3.2，变化范围为 `-1.80%` 到 `-0.26%`；
- 32 KiB 为 G2S 765、S2G 878、roundtrip 1626 cycles，分别优于门槛
  767/881/1631；
- PMU-off WindowEngine 层次顺序状态由 116,940 bit 降到 38,602 bit，
  减少 66.99%；运行时除法、取余、旧 fast path、每-window fill/line-tag 和
  PMU 输出均为 0。

本次没有修改 L2、shared memory、TLB、调度器、ISA 或 TensorMap 内存
格式。TMA 与这些模块的既有接口保持不变。

## 最终硬件结构

```text
                         +-----------------------------+
prefetch/invalidate ---> | control lane                |
                         |                             v
data command ----------> | LookaheadSlot ------> Descriptor Service
                         | VALIDATE/DESC/BIND/RESERVE   | 4-entry store
                         +--------------+--------------+ 1 compiler
                                        |
                              PreparedCommand snapshot
                                        |
                      previous completion enters FIFO
                                        |
                                        v
                               ActiveCommand (only 1)
                                        |
                         four-stage WindowPlanner, II=1
                                        |
                           24 unified TransferSlots
                       +----------------+----------------+
                       |                                 |
                 G2S: L2 read                       S2G: shared read
                       |                                 |
                  scatter/replay                    transform pipeline
                       |                                 |
                 transform pipeline                   L2 write
                       |                                 |
                  shared write                    valid-only ack table
                       +----------------+----------------+
                                        |
                 planner/slot/request/shared/ack all empty
                                        |
                                  completion FIFO
```

### 组件职责

| 组件 | 数量/容量 | 类型 | 职责 |
|---|---:|---|---|
| `TmaV3Ingress` | 1 | 控制 + 小状态机 | 保存一条 lookahead，做检查、descriptor、Binder 和 reserve；只在上一条完成后晋升 |
| Descriptor Service | 1 | tag store + 单路流水/计数器 | 统一 demand/prefetch/invalidate，2/4 项可配置，发布为 4 项 |
| Command Binder | 1 | rank 计数的时序计算 | 只处理坐标相关动态绑定，不重复静态 descriptor 计算 |
| WindowPlanner | 1 | 四级弹性流水 | `CursorExpand/GlobalMap/SharedMap/SlotEncode`，无背压时 II=1 |
| ActiveCommand | 1 | 紧凑寄存器 | 保存 direction、ASID、barrier/group、dtype/fill/codec 等命令级信息 |
| TransferSlot | 24 | 双向统一 buffer | 每项仅一份 1024-bit payload，G2S/S2G 复用地址、mask、pending 和 phase |
| Global request table | 16 | 紧凑乱序表 | G2S consumer bitmap 或 S2G slot/fragment，不保存 payload 和 command 副本 |
| Shared tag table | 8 | 紧凑乱序表 | 保存 valid、角色、slot 和 partial word mask |
| Write ack table | 32 | valid scoreboard | S2G write 接受后释放 slot，只等待 ack |
| Transform | 1 套、三级 | 弹性流水 | G2S fill/codec 与 S2G pack；没有复制计算单元 |

状态机只保留在“必须等待外部事件”的边界，例如 lookahead 生命周期、
descriptor refill/compile 和单项 slot phase。Planner、transform 以及可连续
计算的映射部分均为弹性流水；模块之间主要依靠 valid/ready、valid bit、
credit 和 source ID 解耦。

## Descriptor Service

compiled descriptor payload 为 612 bit，保留 status、dtype、rank、
interleave、swizzle、OOB、global base/dims/strides、box dims、
interleave slice stride 和 logical bytes。以下字段不再存储：

- `legal`；
- `dtypeBits`、`channelsLog2`、`elementByteShift`；
- `swizzleAtomicity`；
- 全部 `stepOffsets`。

tag 为 `{ASID, address[31:7]}`，entry 状态为
`INVALID/FETCHING/COMPILING/VALID`。非法 descriptor 也以 status 形式
缓存。同 tag 的 demand 与 prefetch 在 FETCHING/COMPILING 阶段合并；
VALID hit 在 Compiler 忙时仍可返回。replacement 使用 invalid-first +
tree-PLRU，未完成或被 demand pin 的 entry 不可替换。

显式 prefetch 只保证命中、分配或合并后即可完成，维持异步 hint 语义；
普通命令等待 compiled result。invalidate 优先于尚未提交的 demand 和
compile；已经进入 PreparedCommand 的快照不被追溯取消。

Demand response 只保存 entry index，并在 Binder 捕获前 pin 对应 store
entry，不再复制一份 612-bit compiled payload。Lookahead 也用
`boundCommand.compiled` 同时承担 descriptor 等待和 Binder 输出存储，
删除另一份 612-bit `descriptorCompiled`。独立 control lane 的 queue 和
saved token 仅保存 address、ASID、wid、subop，不再保存坐标、shared
base、barrier 等普通数据命令字段。

实机 GVM 的三种启动方式结果如下：

| 启动方式 | cycles | PMU 证据 |
|---|---:|---|
| cold demand | 213 | `maps_1_pf0` 合计 demand miss 2、hit 1 |
| 连续相同 TensorMap 的 VALID hit | 189 | `maps_1_pf0` 第三个 repeat |
| explicit prefetch 后 demand | 189 | demand hit 3、miss 0 |

显式 prefetch 与 demand hit 的时间完全相同，说明两条路线确实使用同一
compiled store，而不是两套缓存。4 项预取后逐项 demand、同周期同 tag
合并、Compiler 忙时其他 hit、非法 descriptor 复用、2 项 PLRU 和
invalidate 边界均有 Chisel 定向测试。

跨命令复用的是静态 compiled descriptor：命中时不会再次 refill 或进入
Compiler。每条命令的坐标、shared base 和方向仍可能不同，因此 Binder
仍以约 2 个周期重新生成动态 bound snapshot；不能把 Binder 也跳过，否则
会错误复用上一条命令的坐标。

## 单 Active 与 Lookahead

lookahead 的准备状态为 validation、descriptor、bind、reserve、
ready/error；准备期间不得驱动 Planner、分配 TransferSlot 或发出 payload
TLB/L2/shared 请求。第三条数据命令在 lookahead 被占用时背压。

G2S 在准备阶段完成必要的 mbarrier reserve，并把 barrier ID、generation
和 transaction bytes 放入 PreparedCommand。S2G group 在入口接受时计数，
即使后续 descriptor 报错也保持 FIFO completion。上一条命令的 completion
进入 FIFO 后，准备完成的下一条在下一周期晋升 active。

control lane 与数据 lookahead 独立，所以活动搬运期间仍可显式预取或
invalidate 其他 TensorMap，但不会启动第二条数据搬运。

## 统一 TransferSlot 与 S2G 单写点修正

每个 TransferSlot 只有一份 payload。命令级 direction、ASID、barrier、
group、fill pattern 和 codec 不再复制进每个 slot。G2S cache response
只经过一套 8-lane scatter；合并到多个 consumer 时才预留一个 replay
slot。S2G shared read 在已有 slot 后发出，写入 L2 后立即释放 slot，
write ack 表只保留 valid。

最终面积收敛版进一步只保存可以证明不可推导的 slot 信息：

- 8 个 global address 保留完整 32 位，避免 stride/interleave/rank wrap
  的大跨度地址被错误截断；
- global/shared byte mask 都利用“单个连续 byte run”的协议性质压成每
  lane `offset(4)+bytes(5)`；
- 8 个 shared address 利用“同一 128B window 内的 16B atom”性质压成
  一个 32-bit base 加 8 个 4-bit signed atom delta；
- PA 只在 request table 保存 20-bit PPN，低 12 位直接复用 VA；
- 三级 transform 流水只传 slot ID，transform 后 payload 原位写回
  TransferSlot，不再让 payload 穿过三级 token；
- slot 分配、replay 分配和 S2G shared read 发出时不再无条件清零
  1024-bit payload；mask/response ownership 保证所有会被消费的 byte
  必先被覆盖。

因此单 slot 元数据从 809 bit 收敛到 501 bit，24 项合计少 7,392 bit。
原先第一级 transform queue 为 flow-through，生成 RTL 实际只有后两级
各注册一份 payload；tag-only transform 的顺序状态净减少 2,048 bit，
而不是按三级逻辑 token 简单估算的 3,072 bit。所有 mask 在 slot 接受时
都有连续区间断言，shared address delta 也有对齐和 `[-8, 7]` 范围断言。

初版 V3.3 在 S2G 中把完整 shared response 先写成 raw payload，再在下一
周期读出送 transform。因为 payload 只有一个写点，这会与上一 window 的
transform 回写互斥，4 KiB 恰好增加 32 cycles。最终实现让完整 shared
response 直接进入现有 transform 流水；只有 partial response 才在 slot
内合并。第一级 transform queue 同时作为非 pipe、flow-through 的 response
skid，切断 ready 组合环且保持稳态 II=1。

修正前后：

| 路径 | bytes | 修正前 | 最终 V3.3 | V3.2 |
|---|---:|---:|---:|---:|
| S2G | 4 KiB | 233 | 200 | 201 |
| S2G | 32 KiB | 900 | 878 | 881 |

最终 PMU 中这两个连续路径的 transform stall 为 0。

## 周期结果

### 容量门槛

| bytes | G2S V3.2→V3.3 | S2G V3.2→V3.3 | roundtrip V3.2→V3.3 |
|---:|---:|---:|---:|
| 128 | 133 → 131 | 97 → 96 | 213 → 210 |
| 1 KiB | 147 → 145 | 129 → 128 | 259 → 256 |
| 4 KiB | 208 → 206 | 201 → 200 | 392 → 389 |
| 16 KiB | 447 → 445 | 501 → 492 | 931 → 920 |
| 32 KiB | 767 → 765 | 881 → 878 | 1631 → 1626 |

全部 15 个冻结比较点都改善。详细规则和逐项结论见
[`TMA_V3_3_CAPACITY_GATE.csv`](data/TMA_V3_3_CAPACITY_GATE.csv)。

### 资源峰值与 buffer 容量

现有 PMU 日志按每个容量/方向取全程最大值。下表顺序为
`TransferSlot/global request/shared tag/write ack`：

| bytes | G2S | S2G | roundtrip |
|---:|---:|---:|---:|
| 128 | 1/1/1/0 | 1/1/1/1 | 1/1/1/1 |
| 1 KiB | 8/8/2/0 | 8/4/3/8 | 8/8/3/8 |
| 4 KiB | 24/16/2/0 | 24/16/3/20 | 24/16/3/20 |
| 16 KiB | 24/16/2/0 | 24/16/3/32 | 24/16/3/32 |
| 32 KiB | 24/16/2/0 | 24/16/3/32 | 24/16/3/32 |

因此 release 的 24 slot、16 request 和 32 ack 在正式负载中都真实达到
上限，不是未使用的寄存器空间。slot 保存的是同一条 active command 的
不同在途 window，用来覆盖 L2/shared/ack 延迟，并不等于多 context。
若把 release 直接缩成 slot16，4 KiB 起会更早背压，必须用同一周期矩阵
实测后才能接受，不能只根据面积删除。另一方面，request16 和 ack32 已
先饱和，slot32 未必继续提高性能；capacity 点当前主要用于量化面积斜率。
完整 27 项峰值见
[`TMA_V3_3_RESOURCE_PEAKS.csv`](data/TMA_V3_3_RESOURCE_PEAKS.csv)。

### 常用命令与固定 CUDA 参考

下表为完全相同的 shape/方向参数。CUDA 是此前冻结的 H100/B200 cycle
median；Ventus 是 RTL/GVM 的模型周期。由于时钟域、L2/DRAM 和计时边界
不同，跨平台绝对数字用于结构特征参考，不应解释为实际 GPU 性能倍数。

| case | path | V3.2 | V3.3 | H100 | B200 |
|---|---|---:|---:|---:|---:|
| 4×4 | Bulk G2S | 86 | 83 | 507 | 440 |
| 4×4 | Bulk S2G | 55 | 53 | 43 | 43 |
| 4×4 | Tensor G2S | 129 | 127 | 628 | 577 |
| 4×4 | Tensor S2G | 97 | 96 | 92 | 93 |
| 32×32 | Bulk G2S | 161 | 158 | 507 | 590 |
| 32×32 | Bulk S2G | 146 | 145 | 171 | 171 |
| 32×32 | Tensor G2S | 204 | 202 | 628 | 577 |
| 32×32 | Tensor S2G | 189 | 188 | 215 | 220 |

完整 20 项对照见
[`TMA_V3_3_COMMON_COMMAND_COMPARISON.csv`](data/TMA_V3_3_COMMON_COMMAND_COMPARISON.csv)。

## RTL 结构与面积

### 结构门槛

| 指标 | V3.2 | V3.3 release | 变化 |
|---|---:|---:|---:|
| WindowEngine 层次顺序状态位 | 116,940 | 38,602 | -66.99% |
| DescriptorService 层次顺序状态位 | 9,467 | 3,505 | -62.98% |
| standalone TMA 层次顺序状态位 | 149,101（PMU-on） | 53,543（PMU-off） | -64.09% |

V3.2 standalone 的第三项只能说明原始 RTL 状态量，因为其生成物误开
PMU；正式 DC 比较会在 elaboration 后删除 4160 个 PMU 输出位，使整个
不可观察 PMU cone 在优化前变 dead。

### 剩余状态审计

最终 release `tma.sv` 从顶层 `tma` 展开实例层次后共有 53,543 个顺序
状态位。这里特别按实例计数：例如两个同类型 `WindowTask` queue 会计
两次，而不是只统计一次 module definition。按模块和字段拆分后，
大头如下。这里统计的是 RTL declaration bit，不等同于映射 cell area，
但可以准确判断继续压缩寄存器是否有结构收益。

| 状态 | bit | 占 standalone |
|---|---:|---:|
| 24 项唯一 `slotPayload` | 24,576 | 45.90% |
| 24×8 个完整 global address | 6,144 | 11.47% |
| `CursorExpand` 一级弹性 queue | 4,502 | 8.41% |
| Descriptor Service + Compiler | 3,505 | 6.55% |
| GlobalMap + 两级 WindowTask queue | 2,277 | 4.25% |
| `TmaV3Ingress` | 1,694 | 3.16% |
| Planner 本体（不含 queue） | 1,691 | 3.16% |

payload 在 WindowEngine 层次内部占 63.67%，已经是最主要的顺序面积。它目前
可能在同一周期分别被 cache scatter/replay、transform 原位回写和 partial
shared response 更新不同 slot，因此不能把 `Reg(Vec)` 直接替换为单个
普通 1R1W SRAM。若 DC 证明标准单元 payload 仍是面积第一大头，正确的
下一步是设计 2/4-bank、带 byte write-enable 的同步 payload store，并对
同 bank 写冲突做显式背压；不能为了推 SRAM 而重新复制多份 payload。

三种容量生成物的 standalone 层次顺序状态分别为：slot16/desc2
39,873 bit、slot24/desc4 53,543 bit、slot32/desc4 65,871 bit。保持
desc4 不变时，
24→32 项准确增加 12,328 bit，即每个新增 slot 1,541 bit；其中 payload
固定占 1,024 bit（66.45%），其余 517 bit 是地址、连续 byte-run、
pending、phase 等元数据。因此若 DC 的 capacity 点近似线性增长，缩小
slot 的上限收益也只有约 1.5 Kbit/项，不能解决组合 scatter/mux 本身；
反之若增长明显超线性，优先怀疑寄存器阵列的选择/写入网络。

8 个 global address 不能像 shared address 一样无条件压成小 delta：
rank wrap、合法大 stride 和 interleave 可能让同一 window 内出现大跨度
地址。除非改变 Planner/slot 边界并保留可证明的重算信息，否则继续截短
会损失现有功能。

Planner token 还有一组不改变功能容量的低风险压缩候选：

- `CursorExpand.laneIndex` 在 queue 后不再消费，可直接去掉 360 bit；
- `laneRow` 在 shared map 中只取低 3 bit，Cursor/Global 两级可少
  464 bit；
- 每 128B 递增的 `logicalBase` 低 7 位恒零，两级可少 14 bit；
- ActiveCommand 在 last 接受前保持不变，三层 token 不必重复
  commandId/direction/reduce sideband，可少 39 bit；
- global mask 沿三层都可改成连续 run，约少 168 bit；
- 两个 WindowTask queue 的 shared mask 只需 byte count，约少 176 bit；
- 两个 WindowTask queue 可直接沿用 slot 的 shared-base + atom-delta
  表示，约少 384 bit。

合计约 1,605 bit，占 standalone 3.00%。这些优化仍需补 queue
backpressure 与 compact-token 对照测试，但不会像 payload banking 那样
改变存储端口，也不要求增加 64-bit 算术，适合 V3.4 的第一批清理。

更激进的做法是把 CursorExpand 的 8×5D coordinate/component 改成
`coordinate0 + outerInBounds + global component sum`，可在上述
`laneIndex` 清理之外再少约 3.0 Kbit；代价是把多路比较和 64-bit
reduction 前移到最深的 cursor carry 路径。当前版本不为这点收益冒
1.5 GHz 或 `II=1` 回退风险，先以 DC critical path 决定是否进入 V3.4。

其余 request/shared/ack 表已经是紧凑 scoreboard。仍可利用 line address
低 7 位恒零，以及 copy-replay 与 reduce-element 互斥，再减少约 200 bit；
这类低于 standalone 0.4% 的优化不应优先于 payload banking。

### Design Compiler 条件

- TSMC N12 CLN12FFCLL；
- `tcbn12ffcllbwp16p90cpdtt1v85c`；
- TT 1.0 V、85 °C；
- 1.5 GHz，周期 0.666667 ns；
- `compile_ultra -retime` 后接 incremental；
- 每项最多 12 CPU；
- 每个 variant 使用独立的 RTL、WORK、日志和 report。

Descriptor Service 的 2/4 项精确增量，以及 16/24/32 slot 三个容量点，
均使用本轮面积收敛后的同一份源码重新综合。最终结果如下：

<!-- V3.3_DC_STANDALONE_TABLE -->

## 验证覆盖

| 测试集 | 通过 | 重点 |
|---|---:|---|
| Frontend | 8/8 | rank 1–5、dtype/layout、C-model、随机背压、256-window 无气泡 |
| Backend | 17/17 | 2/4 项缓存、prefetch/demand、direct pin、PLRU/invalidate、replay、乱序、credit |
| DmaCore | 19/19 | Bulk/Tensor 双向、interleave16、Reduce 正确性、single active/lookahead |
| Completion + GroupTracker | 10/10 | mbarrier、S2G group、错误 completion |
| GVM descriptor/layout | 30/30 | stride、OOB、swizzle、interleave16/32、FP4/FP6 |
| GVM common non-Reduce | 20/20 | Bulk/Tensor、G2S/S2G、16 B–4 KiB |
| GVM capacity | 27/27 | 128 B–32 KiB、G2S/S2G/roundtrip |

Reduce 继续做正确性验证，但不进入 V3.3 性能冻结门槛。

## 冻结哈希

| 构件 | SHA-256 |
|---|---|
| release standalone `tma.sv` | `b72c9b2bdb2519809769d35cb771eca62aa3c070c4c6c8c2a6b6971342d183ce` |
| release `TmaV2WindowEngine.sv` | `0f07542deb6ecd5206854e190abf3fd32ff601ba817c08644b7ce64d492eda20` |
| GVM with cache | `3ee23185ce0973789a23023995f28282695df3ab27acd51394251cfcb732fe08` |
| GVM no cache | `0e6377cb35babaea8ba3cdcb15337efa7099d7363af3896a1e946b5dc8eb63b4` |
| final RTL manifest | `48bf492dd0ec6d8dc43df6eb07d787192507ddca3bb85f0468a164158badd802` |

冻结前性能/结构证据哈希见
[`TMA_V3_3_EVIDENCE_SHA256.txt`](data/TMA_V3_3_EVIDENCE_SHA256.txt)。
在 DC 六项结果尚未汇总时，该文件首行明确标记
`preflight_without_mapped_dc`；DC 门槛通过并重跑正式分析后才会改为
`v3.3_release_candidate`，冻结器同时复核 RTL、GVM 和 DC 文件的实际
哈希，不能使用旧 summary 误冻结。结构检查见
[`rtl_structure_summary.json`](../benchmarks/tma-area-dc-v3.3/rtl_structure_summary.json)。

## 限制与后续

V3.3 已解决本轮目标中的主要架构问题：没有多 command 数据 context，
没有宽 WindowTask 和全 window response 广播，没有重复 payload，也没有
descriptor 双路线重复缓存。当前大容量性能仍主要由外部 L2/shared/ack
时序模型决定，而不是 Planner 或 transform 内部气泡。

下一步应先根据 DC 层次面积和关键路径确认：

1. 若 24-slot release 达到面积和 1.5 GHz 门槛，保持 4 项 descriptor；
2. 若面积仍偏大，下一项大头是 24×1024-bit 的唯一 payload array；
   优先评估真实 1R1W SRAM macro 或同步 banked payload，不恢复多
   context，也不再复制 payload；
3. Planner 的 `CursorExpand` 仍保存约 4.4 Kbit 的宽 token；只有在 DC
   层次面积和关键路径证明值得时，才进一步做 token summary 压缩，避免
   破坏 1.5 GHz 和 II=1；
4. 用真实 L2 延迟/带宽参数校准性能模型，但不修改其他人的 L2 RTL；
5. 补充更长时间的随机 cache/shared/ack backpressure soak，作为后续
   tape-in 前验证，不改变本次冻结周期。
