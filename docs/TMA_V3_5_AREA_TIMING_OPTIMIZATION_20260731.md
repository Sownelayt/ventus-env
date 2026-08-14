# Ventus TMA V3.5：组合面积与时序优化

## 结论

V3.5 保持 V3.4 的架构边界和容量：

```text
1 ActiveCommand + 1 LookaheadSlot
40 LineContext + 6 PayloadSlot
4-entry compiled TensorMap store
```

只修改 TMA 源码、TMA 测试、生成 RTL 和综合/结果脚本；没有修改 L2、
shared memory、外部 TLB、调度器、ISA 或 TensorMap 内存格式。

V3.4 的 N12 DC 面积已经从 V3.3 的 186,040.85 降到 102,368.92，
但 1.5 GHz WNS 从 -0.2178 ns 变为 -0.2268 ns，仍未收敛。100 条最差
setup path 都终止于 WindowPlanner；同时 V3.4 的 8 lane 数据重排实际
综合成 8 个 1024-bit 动态右移器。V3.5 因此不再削减必要容量，而是直接
优化这两个组合热点。

最终结果：

- Frontend 8/8、Backend 17/17、DmaCore 19/19 通过；
- capacity 27/27 与 V3.4 精确同周期；
- 常用 Bulk/Tensor 非 Reduce 20/20 与 V3.4 精确同周期；
- descriptor/layout 30/30 正确，swizzle/interleave 精确同周期；
- 10 个 cold S2G/roundtrip 项改善 9–18 cycles；
- OOB25/50 为 190→192 cycles，增加 2 cycles（1.05%），满足短命令
  `max(2 cycles, 2%)` 门槛；
- PMU-off standalone TMA 状态从 V3.4 的 28,712 降到 27,053 bit；
- 最大动态移位结果宽度从 1024 降到 256 bit；
- 运行时除法/取余、PMU、旧宽 payload 和 response 广播仍为零。

## 实现

### Planner OOB 区间算法

V3.4 对每个 lane 的 16 个 byte 逐项执行坐标加法和边界比较，先产生
16-bit mask，再用 PriorityEncoder 和 PopCount 回收成 offset+length。
这同时造成大面积比较/加法网络和最差 setup path。

V3.5 直接计算合法连续区间：

```text
valid_start = clamp(request_start, dimension_start, dimension_end)
valid_end   = clamp(request_end,   dimension_start, dimension_end)
offset      = valid_start - request_start
bytes       = max(valid_end - valid_start, 0)
```

普通 8/16/32/64-bit 类型使用 endpoint、clamp 和移位；packed FP4 对
element count 做向上取整；padded FP4/FP6 保持原有 all-or-none 规则。
stage 数和 Planner II=1 不变。

### 紧凑 Cursor/Window token

Cursor stage 不再为 8 lane 保存 `5×32-bit coordinate`，而是保存实际
范围足够的 `5×9-bit laneIndex`；后级用稳定 ActiveCommand 的 origin
恢复动态坐标。compact window 只携带：

- 一份 shared base；
- 每 lane 的 global address、offset、bytes；
- shared atom delta、offset、bytes和 valid。

Planner 弹性队列从 V3.4 的 2,704 降到 1,784 bit，减少 920 bit
（34.02%）。32 KiB contiguous 仍产生 256 个连续 window，填充后无
内部气泡。

### 256-bit atom-pair LinePermuter

V3.4 虽然源码共用了重排表达式，但表达式仍在 lane 循环内处理完整
1024-bit line。V3.5 分成两级：

1. 根据 cache offset 高位选择相邻两个 128-bit atom；
2. 只对拼接后的 256-bit 数据按低 4-bit byte offset 动态移位。

G2S extract 和 S2G insert 使用同一套编码方式，S2G 通过 byte reverse
复用右移结构。结构审计中最大动态移位结果宽度为 256 bit，不再存在
大于 1024-bit 的移位结果。这里不宣称一套物理运算单元；最终实例数量和
共享程度以 DC 的 DesignWare/综合网表为准。

### 后端窄状态和组合 cone 隔离

- LineContext 的 virtual line tag 与翻译后的 PPN 合并成同一个 25-bit
  `lineAddressTag`，翻译提交后原位覆盖高 20 bit；
- 单 LineContext 从 240 降到 220 bit，40 项共减少 800 bit；
- 192-bit routeCandidate 在进入 40 项 LineContext 写选择前切断
  fragment/tag/interval 组合 cone；
- common single-line window 可直接预编码到 candidate，连续路径仍保持
  每周期一个 line；
- synthetic fill 可以与旧 routeCandidate 分配 LineContext 同周期，
  消除了最初的 OOB 每窗口额外气泡。

曾实验让 fill 完成时下一 OOB window 也穿透补入 candidate。它把 Planner
stall 从31降到10，却把最大在途 cache request 从8推到22，并使周期
192→203。当前 L2 模型下这只是把瓶颈转移为排队，因此该实验已撤回，
不进入最终 RTL。这也说明 40 个 entry 是延迟覆盖容量，不等于所有布局
都应无条件灌满。

## 结构结果

| 项目 | V3.3 | V3.4 | V3.5 |
|---|---:|---:|---:|
| standalone TMA state | 53,543 | 28,712 | 27,053 |
| WindowEngine state | 38,602 | 19,567 | 18,989 |
| Planner elastic queues | 8,361 | 2,704 | 1,784 |
| Command Binder | 1,133 | 325 | 325 |
| Ingress | 1,694 | 1,490 | 1,489 |
| Descriptor Service | 3,505 | 2,884 | 2,884 |
| LineContext/entry | — | 240 | 220 |
| PayloadSlot/entry | — | 1,465 | 1,465 |
| 最大动态移位结果 | 3,071 bit | 1,024 bit | 256 bit |

V3.5 相对 V3.3 的 standalone/Engine 状态分别减少49.47%/50.81%；
相对 V3.4 继续减少5.78%/2.95%。最终 RTL 只有6份1024-bit可驻留
payload，且没有：

- runtime `/`、`%`、divider/rem；
- 40路 payload mux、owner bitmap或response广播；
- PMU counter、timestamp或perf output；
- 旧 WindowTask/requestData/sharedData；
- fast path或第二份 descriptor compile result。

结构原始数据：

- [`../benchmarks/tma-area-dc-v3.5/rtl_structure_summary.json`](../benchmarks/tma-area-dc-v3.5/rtl_structure_summary.json)
- [`data/TMA_V3_5_RTL_TEST_SUMMARY.json`](data/TMA_V3_5_RTL_TEST_SUMMARY.json)

## 周期结果

### Capacity

| 大小 | G2S V3.4→V3.5 | S2G V3.4→V3.5 | Roundtrip V3.4→V3.5 |
|---:|---:|---:|---:|
| 128 B | 128→128 | 92→92 | 203→203 |
| 1 KiB | 142→142 | 124→124 | 249→249 |
| 4 KiB | 203→203 | 196→196 | 382→382 |
| 8 KiB | 279→279 | 296→296 | 558→558 |
| 16 KiB | 411→411 | 488→488 | 882→882 |
| 32 KiB | 675→675 | 874→874 | 1532→1532 |

全部27项逐项相等，32 KiB G2S 的256条 line 仍保持
`issue span=255 cycles`。

### Descriptor/layout

V3.4 descriptor 基线使用 `warmups=0`。一次 V3.5 探针误用
`warmups=1`，使 swizzle/interleave 表面增加11 cycles；匹配参数后它们
分别恢复371/373 cycles，内部 active/line/compile/bind 计数也完全一致。
不同 warmup 的结果不得互相比周期，因为当前确定性 RTL L2 会在同一进程
的重复运行间保留状态。

匹配结果中：

- swizzle32/64/128：371→371；
- interleave16/32/u16：373→373；
- OOB100：145→145；
- OOB25/50：190→192；
- cold S2G/roundtrip tile：10项改善9–18 cycles。

完整数据：

- [`data/TMA_V3_5_CAPACITY_COMPARISON.csv`](data/TMA_V3_5_CAPACITY_COMPARISON.csv)
- [`data/TMA_V3_5_DESCRIPTOR_COMPARISON.csv`](data/TMA_V3_5_DESCRIPTOR_COMPARISON.csv)
- [`data/TMA_V3_5_COMMON_COMMAND_COMPARISON.csv`](data/TMA_V3_5_COMMON_COMMAND_COMPARISON.csv)
- [`data/TMA_V3_5_CUDA_CAPACITY_COMPARISON.csv`](data/TMA_V3_5_CUDA_CAPACITY_COMPARISON.csv)
- [`data/TMA_V3_5_SUMMARY.json`](data/TMA_V3_5_SUMMARY.json)

### 与 CUDA 固定参考

H100/B200 不重跑，继续使用相同参数的冻结数据：

| 大小 | 路径 | H100 | B200 | Ventus V3.5 |
|---:|---|---:|---:|---:|
| 128 B | G2S | 460 | 461 | 128 |
| 128 B | S2G | 62 | 93 | 92 |
| 32 KiB | G2S | 756 | 759 | 675 |
| 32 KiB | S2G | 1082 | 1113 | 874 |
| 32 KiB | Roundtrip | 1869 | 1884 | 1532 |

Ventus L2 是确定性 RTL 模型，这些绝对值不能解释为硅实现比 NVIDIA 快；
它们只适合做 Ventus 版本间回归。真实 CUDA 包含片上互连、cache slice、
HBM/ECC、频率域和动态资源争用。

## 两个 DC 时序模型

两个项目使用完全相同、PMU-off 的最终 RTL，条件均为：

- TSMC N12 `tcbn12ffcllbwp16p90cpdtt1v85c`；
- TT 1.0 V、85°C、1.5 GHz；
- `compile_ultra -retime` 后增量优化；
- 每项目最多12 CPU；
- 无 memory compiler，所有小深度阵列仍由标准单元实现；
- `DC_TIMEOUT_SECONDS=0`，不设置超时。

模型1 `zero_wire` 与 V3.4 完全 matched，用于可信的版本面积/WNS差分。

模型2 `boundary_typical` 在缺少邻接模块和物理 RC 的前提下采用明确的
block-boundary 工程预算：

- N12 `BUFFD2BWP16P90CPD` 输入驱动；
- input/output max delay 各0.12 ns，min delay各0.03 ns；
- output load 0.010 pF（10 fF）；
- max transition 0.12 ns；
- max fanout 16。

由于没有 floorplan、wire-load model或寄生参数，第二个模型内部仍必须
使用 ZeroWireload。它会让 DC 为边界驱动、负载、transition和fanout
支付代价，但只是前布局典型估算，不是实际布线时序。两个模型的差值可
用于判断设计对边界条件的敏感度；不能把第二个模型的 WNS冒充 post-route
WNS。

DC 打包和运行说明：
[`../benchmarks/tma-area-dc-v3.5/README.md`](../benchmarks/tma-area-dc-v3.5/README.md)。

两个项目已经在虚拟机中通过独立 WORK、日志和报告目录启动：

| 模型 | VM 项目 | tmux session | 启动检查 |
|---|---|---|---|
| historical matched | `20260731_18_tma_v3_5_zero_wire_n12_tt1v85c_1500` | `tma_v3_5_zero_wire_dc` | `RUNNING`，目标库加载完成，无 Fatal/Error |
| typical boundary | `20260731_19_tma_v3_5_boundary_typical_n12_tt1v85c_1500` | `tma_v3_5_boundary_typical_dc` | `RUNNING`，边界 driver 被接受，目标库加载完成，无 Fatal/Error |

启动记录见
[`../benchmarks/tma-area-dc-v3.5/DC_LAUNCH.json`](../benchmarks/tma-area-dc-v3.5/DC_LAUNCH.json)。
这里只记录启动时健康状态；在 `dc_exit.status` 产生最终退出码且报告完整前，
不得视为综合通过。

## 最终哈希

```text
standalone PMU-off RTL
9255600bbdf1e72de4a9fd786293f2b469ed4f8e8a91e2f67ffbe50150c3974d

GVM with-cache
a4824fc5dd894ad3ff71a74378e6a24d23d141c9ca83d75ba701c8ffda51f535

GVM no-cache
71147591cd55452e0c9e6c7401d417048574022d18bb3cf5f70b29a130dddb38
```

V3.5 的版本状态在两个 DC 完成并分析前为 `dc_running`，不覆盖历史结果。
核心证据文件哈希见
[`data/TMA_V3_5_EVIDENCE_SHA256.txt`](data/TMA_V3_5_EVIDENCE_SHA256.txt)。
