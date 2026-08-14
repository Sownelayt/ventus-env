# Ventus TMA 默认容量与 256 项容量 DC 面积对比

报告日期：2026-08-12  
综合完成日期：2026-08-11

## 1. 结论

在相同工艺库、约束和 PMU-off 条件下：

- 默认配置 `requestEntries=40, writeAckEntries=32` 的总 cell area 为
  `84,469.027`。
- 最大容量配置 `requestEntries=256, writeAckEntries=256` 的总 cell area
  为 `139,325.018`。
- 256 配置比默认配置增加 `54,855.991`，即 `+64.942%`，面积为默认配置的
  `1.649×`。
- 面积增量中，组合面积占 `51.390%`，时序面积占 `48.610%`。
- `TmaV2WindowEngine` 增加 `58,821.488`，是唯一实质性增长的模块；其增量
  超过全芯片净增量，是因为 DC 在 256 配置中将 Planner 映射得更小，抵消了
  `4,114.686` 的面积。
- 256 配置新增 `39,157` 个时序单元。容量参数直接新增的
  `LineContext + writeAckValid` 状态为 `39,104 bit`，解释了时序单元增量的
  `99.865%`。因此主要时序面积不是版本残留，而是 256 项真实存储成本。
- 默认配置低于脚本的 `100,000` cell-area 上限 `15.531%`；256 配置高出该
  上限 `39.325%`。256 适合参数/功能压力验证，不适合作为当前默认物理实现点。

## 2. 对比条件

两个 DC 使用完全相同的综合条件：

| 条件 | 值 |
|---|---|
| Design top | `tma` |
| 工艺 | TSMC N12 CLN12FFCLL |
| 标准单元库 | `tcbn12ffcllbwp6t16p96cpdtt1v85c` |
| PVT | TT, 1.0 V, 85 °C |
| 目标频率 | 1500 MHz，周期约 0.667 ns |
| 边界模型 | `boundary_typical` |
| 内部线负载 | `ZeroWireload` |
| 最大面积约束 | 100,000 |
| 综合策略 | `compile_ultra -retime -timing_high_effort_script`，随后 `compile_ultra -incremental` |
| DC 并行度 | 12 cores |
| TMA PMU | 未实例化 |
| Payload slots | 6，两版相同 |
| Shared-ready entries | 8，两版相同 |
| Descriptor entries | 4，两版相同 |

唯一有意变化的是：

| 参数 | 默认 DC | 256 DC | 倍数/增量 |
|---|---:|---:|---:|
| `requestEntries` | 40 | 256 | `6.4×`，+216 |
| `writeAckEntries` | 32 | 256 | `8×`，+224 |
| cache source 位宽 | 6 bit | 8 bit | +2 bit |
| request group 数 | 10 | 64 | +54 |

## 3. DC 总面积

DC 在本次 ZeroWireload 设置下不报告 net interconnect area，因此本文统一比较
`Total cell area`，不是布局布线后的 die area。

| 指标 | 默认 40/32 | 256/256 | 差值 | 变化 |
|---|---:|---:|---:|---:|
| Total cell area | 84,469.027 | 139,325.018 | +54,855.991 | +64.942% |
| Combinational area | 63,286.568 | 91,477.279 | +28,190.712 | +44.545% |
| 其中 Buf/Inv area | 5,692.244 | 8,992.420 | +3,300.176 | +57.977% |
| 非 Buf/Inv 组合面积 | 57,594.323 | 82,484.859 | +24,890.536 | +43.217% |
| Noncombinational area | 21,182.459 | 47,847.739 | +26,665.279 | +125.884% |
| Macro/black-box area | 0 | 0 | 0 | — |

面积增量的构成：

| 增量类别 | 面积增量 | 占总增量 |
|---|---:|---:|
| 时序单元 | 26,665.279 | 48.610% |
| 非 Buf/Inv 组合逻辑 | 24,890.536 | 45.374% |
| Buffer/Inverter | 3,300.176 | 6.016% |
| 合计 | 54,855.991 | 100% |

## 4. 单元和网络数量

| 指标 | 默认 40/32 | 256/256 | 差值 | 变化 |
|---|---:|---:|---:|---:|
| Ports | 26,677 | 26,685 | +8 | +0.030% |
| Nets | 389,139 | 583,182 | +194,043 | +49.865% |
| Leaf cells | 360,556 | 554,831 | +194,275 | +53.882% |
| Combinational cells | 328,917 | 484,035 | +155,118 | +47.160% |
| Sequential cells | 31,633 | 70,790 | +39,157 | +123.785% |
| Buffer/Inverter cells | 58,919 | 94,350 | +35,431 | +60.135% |
| Cell references | 44 | 64 | +20 | +45.455% |

端口只增加 8 个，说明面积增长不来自外部接口复制；主要增长发生在 Engine 内部
状态及其选择/回收网络。

## 5. 层次面积归因

下表使用各层级的 exclusive/local area，避免父模块与子模块重复计数。

| Exclusive hierarchy | 默认面积 | 256 面积 | 差值 | 变化 |
|---|---:|---:|---:|---:|
| `tma` 本地逻辑 | 2,712.011 | 2,854.011 | +142.000 | +5.236% |
| `TmaV2Ingress` 本地逻辑 | 1,568.526 | 1,630.384 | +61.858 | +3.944% |
| `TmaV2CommandBinder` | 2,108.289 | 2,098.594 | -9.695 | -0.460% |
| `TmaV2WindowPlanner` | 26,970.845 | 22,856.160 | -4,114.686 | -15.256% |
| `TmaV2DescriptorService` | 6,122.373 | 6,077.399 | -44.974 | -0.735% |
| `TmaV2WindowEngine` | 44,986.983 | 103,808.471 | +58,821.488 | +130.752% |
| 合计 | 84,469.027 | 139,325.018 | +54,855.991 | +64.942% |

`WindowEngine` 在总面积中的占比由 `53.3%` 上升至 `74.5%`。Engine 内部进一步
分解为：

| Engine 面积 | 默认 | 256 | 差值 | 变化 |
|---|---:|---:|---:|---:|
| 组合 | 31,932.851 | 64,709.592 | +32,776.741 | +102.643% |
| 时序 | 13,054.132 | 39,098.879 | +26,044.747 | +199.513% |
| 合计 | 44,986.983 | 103,808.471 | +58,821.488 | +130.752% |

Planner 的 RTL 和容量参数没有缩小。其面积下降来自全设计 `retime`、不同的时序
压力及映射选择，不能解释为 256 配置节省了 Planner 硬件。256 版本在 1.5 GHz
下存在大量 setup 违例，DC 的优化分配与默认版并非逐模块完全同构。因此，真实的
容量增量应主要从 Engine 和寄存器位数判断。

## 6. 为什么时序面积几乎增加一倍半

当前每个 LineContext 的固定状态为：

| 字段 | 位数 |
|---|---:|
| `lineState` | 3 |
| `lineAddressTag` | 25 |
| `lineRoute` | 152 |
| 每项合计 | 180 |

`lineRoute=152 bit` 来自 G2S compact route：

```text
sharedBase                           32
8 × cacheWord[4:0]                  40
8 × destinationWord[1:0]            16
8 × bytes[4:0]                       40
8 × sharedAtomDelta[2:0]             24
合计                                152 bit
```

从 40 增至 256 项新增：

```text
LineContext:    (256 - 40) × 180 = 38,880 bit
writeAckValid:  (256 - 32) × 1   =    224 bit
直接容量状态合计                  = 39,104 bit
```

DC 实际新增时序单元为 `70,790 - 31,633 = 39,157`。直接容量状态解释其中
`39,104 / 39,157 = 99.865%`，剩余 53 个时序单元来自索引宽度、仲裁状态和
retime 差异。

这也说明当前 256 配置的高面积并非 Payload 被复制：PayloadSlot 始终是 6 项，
每个 1024 bit；增长的是窄 LineContext 数量和 ack tag validity。

## 7. 为什么组合面积也增长 44.5%

容量扩大不仅增加寄存器，还扩大或加深以下网络：

- LineContext free/recycle 检测由 40 路扩大到 256 路；
- 四项一组的层次选择由 10 group 扩大到 64 group；
- translate/cache issue 的状态检测和仲裁覆盖更多 group；
- cache response、recycle 和 source 合法性处理由 6-bit source 扩大为 8-bit；
- write-ack free/response waiting 检测由 32 路扩大为 256 路；
- 256 项寄存器 D 输入和使能网络需要更多 mux、decoder、buffer 和 inverter；
- 在 1.5 GHz 时序压力下，DC 进一步复制/增强部分组合锥和驱动网络。

从结果看，Buffer/Inverter 本身只占总面积增量的 `6.016%`，其余组合增量主要
是表项选择、状态比较、编码/解码和仲裁逻辑，而不是纯粹的高扇出 buffer。

## 8. 面积约束和时序背景

| 项目 | 默认 40/32 | 256/256 |
|---|---:|---:|
| 相对 100,000 面积约束 | 低 15.531% | 高 39.325% |
| Setup WNS | -0.01 ns | -0.15 ns |
| Setup TNS | -0.70 ns | -1,415.99 ns |
| Setup violating paths | 179 | 15,528 |

两次 DC 都正常完成，退出码为 0，正式日志中均无 `Error:` 或 `Fatal:`，并生成了
DDC、门级网表和 SDF。但“综合成功”不等同于“约束收敛”：

- 默认配置非常接近 1.5 GHz closure；
- 256 配置明显不满足当前 1.5 GHz 和 100,000 面积目标。

## 9. 设计判断

1. 默认 `40/32` 仍是当前更合理的产品点。它保留流水和足够 outstanding，面积
   处于约束内，时序只剩很小缺口。
2. 参数上保留最大 256 项是合理的，可用于研究长延迟系统、容量压力测试和未来
   实现探索；但不应把 256 视作零成本或默认推荐配置。
3. 若未来确实需要接近 256 的产品容量，优先方向不是删除正常流水，而是研究
   LineContext/ack 表的 SRAM 或专用小型 register-file 实现，以及分 bank 的
   局部选择。当前 DC 报告中 macro/black-box 数量为 0，所有容量状态均落成标准
   单元触发器，这是 256 配置时序面积高达 47,847.739 的直接原因。
4. 不能用 256 版 Planner 面积下降推导“容量增加让 Planner 更小”；这是
   timing-driven 全局映射波动。要比较 Planner 微结构，应采用独立 module-only、
   matched constraint 综合。

## 10. 原始结果位置

默认配置：

```text
/home/liyb/tma-dc/20260811_28_tma_alignment_cleanup_default_p96_tt1v85c_1500/
  DC_log/current_20260811_alignment_cleanup_default_p96_boundary_1500/report/
```

256 配置：

```text
/home/liyb/tma-dc/20260811_29_tma_alignment_cleanup_outstanding256_p96_tt1v85c_1500/
  DC_log/current_20260811_alignment_cleanup_outstanding256_p96_boundary_1500/report/
```

主要原始报告：`area.rpt`、`area_hierarchy.rpt`、`qor_summary.rpt`、
`reference.rpt`、`area_designware.rpt` 和 `run_summary.rpt`。

本地 RTL 参数和状态定义见：

- `gpgpu/ventus/src/pipeline/TmaV2Spec.scala`
- `gpgpu/ventus/src/pipeline/TmaV2WindowEngine.scala`
