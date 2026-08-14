# Ventus TMA V3.2：撤销 elementStride、清除除法与周期复测

> 日期：2026-07-30  
> 状态：功能、生成 RTL、GVM 构建和非 Reduce 周期回归通过  
> 上一候选：V3.1（未冻结）  
> 固定基线：V3.0 保持只读  
> 范围：TMA 硬件、Spike/GVM 语义、单元/系统测试和本地周期模型  
> 费用：本轮没有启动 Modal，也没有重启 Design Compiler

## 结论

V3.2 撤销了 V3.1 新增的 elementStride 1–8 支持。TensorMap 的
`elementStrides[0..4]` 仅保留为内存格式占位符：active 维必须为 `1`，
inactive 维必须为 `0`。非 unit active 值返回 `UnsupportedFeature`，
不会进入 Binder、Planner 或搬运后端。

这次改动确实删除了硬件负担：

- 删除 `TmaV2ElementStrideMath`；
- decoded command、compiled descriptor、compiled store、Binder 和
  Planner token 都不再携带 elementStride；
- `logicalBytes` 直接按 box dimensions 计算；
- 外层 `stepOffsets` 直接采用 global strides；
- Planner 的迭代步长固定为 1；
- V3.1 生成 RTL 中 6 处运行时 `/` 全部消失，V3.2 生成 RTL 中没有
  运行时除法或取余；
- compiled descriptor 相关队列 token 从 1002 bit 降为 982 bit，
  每项减少 20 bit；4 项 compiled store 的原始 payload 共少 80 bit，
  其上下游寄存器和队列也同步变窄；
- 生成 `tma.sv` 文本体积从 5,707,109 B 降到 5,666,780 B。

周期结果没有变化。128 B、1/4/16/32 KiB 的 G2S、S2G 和 roundtrip
共 15 组逐项 `Δ=0 cycles`；完整非 Reduce bulk/bulk.tensor 20 例的
V3.2 CSV 与 V3.1 CSV 字节级一致。因此，V3.1 的 elementStride 支持
没有给 unit-stride 常用路径增加状态周期；它增加的是组合逻辑、缓存
宽度、布线扇出以及综合时序/面积风险。最终面积变化必须用新的 V3.2
RTL 重跑 DC，不能用已停止的 V3.1 工程推断。

## 1. DC 终止状态

已在虚拟机上按进程当前目录精确终止：

```text
/home/liyb/tma-dc/20260730_12_tma_v3_1_area_n12_tt1v85c_1500/variants
```

三个 variant 是互相独立的 DC 工作目录。终止后再次扫描
`/proc/<pid>/cwd`，该根目录下剩余匹配进程数为 `0`。本轮没有删除历史
目录，也没有启动新综合。

旧目录和仓库内 `benchmarks/tma-area-dc/variants/*` 都是 V3.1 输入，
其中仍包含已撤销的 elementStride 逻辑，不能作为 V3.2 面积结果。

## 2. 最终 elementStride 契约

### 2.1 TensorMap 内存格式

为保持 TensorMap 二进制布局稳定，word 22–26 仍是五个 32-bit
placeholder：

| 维度状态 | 合法占位值 | V3.2 行为 |
| --- | ---: | --- |
| `dimension < rank` | 1 | 接受，数据通路固定 unit stride |
| `dimension >= rank` | 0 | 接受，视为未使用字段 |
| active 且不等于 1 | 其他 | `UnsupportedFeature` |
| inactive 且不等于 0 | 其他 | descriptor 非法 |

硬件仍读取这些 raw bits，只用于 descriptor 边界处的合法性/不支持检查；
值不会写入 compiled descriptor，也不会参与任何地址、计数或大小计算。
因此“保留占位符”和“删除功能数据通路”同时成立。

### 2.2 删除的数据通路

`TmaV2DecodedCommand` 和 `TmaV2CompiledDescriptor` 已删除
`elementStrides`。Compiler 的行为改为：

```text
logicalBytes = elementBytes × product(boxDimensions)
stepOffsets[0] = elementBytes
stepOffsets[d>0] = globalStrides[d-1]
```

Planner 各维 cursor 始终加 1。Spike/GVM 参考实现采用相同规则，并在
active placeholder 非 1 时返回 unsupported，避免 RTL 与参考模型对
mbarrier/completion 的理解再次分叉。

CUDA 参考数据中的 `rank2_element_stride2` 可以保留，用来说明 CUDA
能力，但不再属于 Ventus 功能或性能矩阵。

## 3. 除法和取余审计

审计对象是重新生成的最终 SystemVerilog，而不是只搜索 Scala 源码。
V3.1 对照 RTL 中找到 6 处运行时算术：

| 类别 | V3.1 表达式 | V3.2 |
| --- | --- | --- |
| elementStride ceiling divide | `/ 7`、`/ 6`、`/ 5`、`/ 3` | 删除 |
| copy window index | `/ 16` | `>> 4` |
| reduce window index | `/ 32` | `>> 5` |

对 V3.2 `tma.sv` 同时搜索以下模式，命中数均为 0：

- Verilog `/`、`%` 算术操作；
- `$div/$sdiv/$udiv`；
- `$mod/$smod/$umod`；
- `DW_div`、`DW_rem`。

全 GPU 重新 elaboration 后，又对集成态 `verilog-out/TmaV2*.sv` 及
TMA queue 模块做了同样审计，运行时除法/取余与
`compiled_elementStrides` 也都是 0。GPU 的标量/向量 ISA 仍有独立的
`IntDivMod`、`FloatDivSqrt` 等通用执行单元；它们不属于 TMA 地址、
descriptor、Planner 或搬运路径，本轮没有越界删除处理器本身的除法
指令能力。

Scala 源码中仍能看到诸如 `fragment / 2`、`word % 4` 的表达式。这些
操作数全是 Scala `Int` 循环变量，在 Chisel elaboration 时就计算成
常量，不是 `UInt` 硬件运算；最终 RTL 的零命中结果证明它们没有生成
除法器或取余器。

需要注意，DC 对固定常数除法未必实例化通用 divider，可能优化成比较、
移位和加法网络；所以“6→0”不能直接等价为某个面积百分比。它能确定
证明的是相关组合逻辑和路径已经从最终 RTL 中消失。

## 4. 功能验证

### 4.1 Chisel

| 测试集 | 结果 | 本轮关键覆盖 |
| --- | ---: | --- |
| `TmaV2Frontend_test` | 8/8 | rank 1–5、全部 dtype、Compiler/Binder、随机背压、32 KiB 无气泡、非 unit placeholder 拒绝 |
| `TmaV2Backend_test` | 12/12 | compiled store、coalesce、invalidate、仲裁、credit/ROB/ack/TLB |
| `TmaV2DmaCore_test` | 19/19 | bulk/tensor G2S/S2G、interleave16、prefetch、单 active、unsupported sealing |

DmaCore 中旧的 elementStride 正向系统用例已改为负向用例：descriptor
只 refill 一次，不发 payload/shared 请求，不消耗 barrier credit，并以
`UnsupportedFeature` 完成。

### 4.2 GVM 功能矩阵

删除旧正向 elementStride 用例后，descriptor 矩阵为 30/30：

- rank 1–5 与普通 row stride；
- 25/50/100% OOB；
- swizzle 32/64/128；
- interleave16/interleave32 和已验证的 UINT16 interleave16；
- 常用 tile 的 G2S、S2G 和 roundtrip。

### 4.3 RTL 与 GVM

- 新 RTL：`gpgpu/generated-tma-v2-v3.2/tma.sv`
- Verilator/GVM 全量重建成功；
- Verilator 处理 338 个 module、619 个 C++ 文件；
- 最终 V3.2 标签构建 wall time 约 223.5 秒；
- 仅有工程既有的 Chisel dynamic-index warning，没有 TMA 错误。

最终库已安装到项目 `install/lib/libVentusGVM-withcache.so`。安装后
128 B G2S smoke 再次得到 133 cycles，PMU 明确打印 `[TMA V3.2]`。
带完整 debug/trace 的冷启动 wall time 为 10.684 秒；这是本地仿真器
开销，不是硬件周期，也没有使用按秒计费的云资源。

## 5. 性能结果

以下均为本地 GVM 周期，不使用 Modal。容量矩阵配置为 warmup 1、
repeat 1、顺序 fail-fast；普通用例超时 5 秒，大容量用例超时 10 秒。
周期来自硬件模型计数，不是 wall time。

| bytes | path | V3.1 cycles | V3.2 cycles | Δ |
| ---: | --- | ---: | ---: | ---: |
| 128 | G2S | 133 | 133 | 0 |
| 128 | S2G | 97 | 97 | 0 |
| 128 | roundtrip | 213 | 213 | 0 |
| 1,024 | G2S | 147 | 147 | 0 |
| 1,024 | S2G | 129 | 129 | 0 |
| 1,024 | roundtrip | 259 | 259 | 0 |
| 4,096 | G2S | 208 | 208 | 0 |
| 4,096 | S2G | 201 | 201 | 0 |
| 4,096 | roundtrip | 392 | 392 | 0 |
| 16,384 | G2S | 447 | 447 | 0 |
| 16,384 | S2G | 501 | 501 | 0 |
| 16,384 | roundtrip | 931 | 931 | 0 |
| 32,768 | G2S | 767 | 767 | 0 |
| 32,768 | S2G | 881 | 881 | 0 |
| 32,768 | roundtrip | 1,631 | 1,631 | 0 |

完整常用矩阵另覆盖 16 B、64 B、256 B、1 KiB、4 KiB 的 bulk 与
bulk.tensor、G2S 与 S2G，共 20 例；V3.2 和 V3.1 的
`attempt_01.csv` 字节级相同。

这意味着删除功能后没有“找回”架构可见周期。原因是 unit-stride
命令在 V3.1 里虽然穿过了 elementStride 相关组合选择，但 Compiler
micro-op 数量、Binder 周期、Planner token 数、L2/shared/ack 等待都没
因 stride=1 增加状态。预期收益应在 DC 面积、临界路径、功耗和更窄
队列上观察，而不是当前 cycle-accurate 状态机计数上观察。

## 6. 固化文件与哈希

| 文件 | SHA-256 |
| --- | --- |
| V3.2 `tma.sv` | `656cbb94e83ed5078622eb565cd4c33d7b4d049c4a60166f41f1bd72b19bfe03` |
| `TmaV2Frontend.scala` | `1beb0d794d6fc66b191d2471ab0782abd6a81c47f1b5f38f0f157f85fcb7e282` |
| `TmaV2WindowPipeline.scala` | `0b2ed16599f8afac9705ea0e4b95c0d77e736afa8b850a2a59cd7f36bfa27fc6` |
| 15 例容量 CSV | `977b5eb951502d76c575ee723eb4eff214b175ca855391265326b284b33c541e` |
| 20 例完整 CSV | `4c47e3542cec8a667265aa2d31e9c1010ede3ba07f308558f5d11d114659ce7a` |
| 30 例 descriptor CSV | `b5f19deba32ef834b52ffc1dd1b57b175320b7d9f13978718759a5aec76c1166` |
| installed V3.2 GVM | `550df9f236b31e5f50e8883f1610dfc8c7a4acd8507966552e9785330f523f54` |

结果目录：

- `benchmarks/tma-cycle-compare/results/features_ventus_v3_2_capacity_reference_completed`
- `benchmarks/tma-cycle-compare/results/ventus_v3_2_full`
- `benchmarks/tma-cycle-compare/results/features_ventus_v3_2_descriptor`

V3.0 和 V3.1 的既有结果没有覆盖。

## 7. 下一步

下一步应从 V3.2 的 `tma.sv` 新建独立 DC 目录，在
TSMC12、TT 1.0 V、85 °C、1.5 GHz 条件下重跑面积/时序。最少测：

1. 当前完整参数点；
2. compiled store 2 项；
3. compiled store 4 项。

这样可以把“删除 elementStride 算术”和“compiled store 容量”分开。
在新 DC 数字出现前，不应宣称具体面积百分比，也不应继续用停止的
V3.1 三点外推 V3.2。
