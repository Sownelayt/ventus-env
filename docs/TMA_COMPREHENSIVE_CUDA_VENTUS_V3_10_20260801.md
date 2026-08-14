# TMA V3.10 精确周期：H100、B200 与 Ventus

## 结论

最新结构此前只有 Chisel 功能结果，没有安装到 GVM 后的精确周期数据。
本轮用当前 V3.10 `rank 2+2+1` WindowPlanner 重新构建 with-cache GVM，
补测了与固定 H100/B200 数据逐参数一致的 334 个安全非 Reduce 配置。

- Ventus 最终数据 334/334 正确，三方完整参数签名连接 334/334。
- V3.10 相对 V3.8：34 条 Bulk 全部 `+0`；208 条单向 Tensor 全部
  `+2 cycles`；92 条 roundtrip 全部 `+4 cycles`。
- 变化是每条 Tensor 命令固定增加的两级启动延迟，不随容量、rank、dtype、
  stride、swizzle 或 interleave 增长；大容量稳态吞吐没有回退。
- 32 KiB Tensor G2S/S2G/roundtrip 为 `693/890/1566 cycles`；H100 为
  `756/1124/1911`，B200 为 `854/1160/1880`。
- Ventus 绝对周期通常低于 CUDA，主要因为 GVM 的 L2/backing-memory 延迟
  模型远比真实 GPU 乐观，不能据此宣称 Ventus 实芯片快于 NVIDIA。
  更有结构意义的 16→32 KiB G2S 边际速率仍是 Ventus
  `2.063 cycle/128B`，H100/B200 均为 `1.156 cycle/128B`。
- 新增的付费敏感 2D 长度探针在 H100/B200 各完成 182 配置、364 个正确
  样本。H100 的约 460-cycle G2S 平台在两次重复中复现；S2G 则严格呈现
  每 32 B 增加 1 cycle 的斜率。
- 同一新矩阵在 Ventus V3.10 上有 180/182 个有效精确周期；另外两个 G2S
  点每次都恰好错误 128 B，独立复测仍失败，故保留为空白功能缺陷点而不把
  未通过校验的 `402/657` cycles 当成性能数据。

## 版本证据

| 对象 | SHA-256 |
|---|---|
| V3.10 TMA source | `def2b423d807513f97c6d5ae67971cfd84722950cf039fb4077945ff6c1f45f2` |
| installed with-cache GVM | `2fc7e48d3144e464bff13134c2061448cdbc761f52dc772b4e2f3b731009ed1a` |

GVM 安装时间为 2026-08-01 22:35:48（UTC+8）。本轮每项一个 recorded
repeat，`warmups=1`；周期来自设备侧完整命令边界，不包含 host launch。

## 与 NVIDIA 精确比较的项目

CUDA 与 Ventus 由同一份 `CaseSpec` 生成参数，聚合时使用完整参数签名而非
只按名称连接。

| 分片 | 数量 | 三方共同变量 |
|---|---:|---|
| Bulk | 34 | 16 B–32 KiB、G2S/S2G、global mod128=0/16/32/64/112 |
| Tensor capacity | 27 | 128 B–32 KiB、G2S/S2G/roundtrip |
| Tensor geometry | 109 | rank 1–5、容量、地址相位、stride、subbox、OOB |
| dtype | 92 | 13 种普通 dtype、128 B/4 KiB、三方向、浮点 OOB zero/NaN |
| layout | 54 | swizzle 32/64/128、1/4/16 KiB、三方向及 plain 对照 |
| interleave16 | 18 | UINT16 whole-atom、1/4/16 KiB、三方向及 plain 对照 |
| **总计** | **334** | 安全、非 Reduce 共同功能面 |

本轮主表不含 Reduce、sub-byte 和 interleave32。Reduce 当前不是已冻结的
共同性能路径；sub-byte/interleave32 属于 CUDA encode/XID 风险项，混入主表
会把功能缺口或故障处理时间误当成搬运性能。

## G2S

### Tensor capacity：plain U8 2D、128B row

| 长度 | H100 | B200 | Ventus V3.10 |
|---:|---:|---:|---:|
| 128 B | 470 | 570 | 146 |
| 256 B | 460 | 558 | 148 |
| 512 B | 460 | 558 | 152 |
| 1 KiB | 460 | 558 | 160 |
| 2 KiB | 460 | 562 | 183 |
| 4 KiB | 608 | 558 | 220 |
| 8 KiB | 611 | 558 | 297 |
| 16 KiB | 608 | 706 | 429 |
| 32 KiB | 756 | 854 | 693 |

### 2D Tensor 32B 粒度补充探针

新增的 opt-in `tensor_2d_length` 分片在 H100 和 B200 上各完成 182 个配置，
每配置在同一 kernel 内连续记录两次；GPU body 分别只有 0.563 s 和
0.797 s。它不加入历史 `--shard all`，因此不会无意扩大常规付费矩阵。
对应的一次性付费记录为
[H100 Modal run](https://modal.com/apps/sownelayt/main/ap-YvktMeGMJs3QQfaLOWbHFI)
和 [B200 Modal run](https://modal.com/apps/sownelayt/main/ap-cojdXBvw5Ss1Dkvku1R3FK)。

采样按合法 2D box 分层：32–512 B 每 32 B，640 B–2 KiB 每 128 B，
2.5–8 KiB 每 512 B，并在 1/2/4/8 KiB 两侧加入特殊点；8–16 KiB
改用 64B row 每 512 B，16 KiB 后用 128B row 抽样至 32 KiB。另有
U16 对照点。固定 32B row 超过 8 KiB 会让第二维大于 TensorMap 的 256
上限，所以不能简单把 32–16 KiB 的全部 32B 倍数都作为同一种合法 2D
格式。

最关键的重复结果是：H100 `u8_row128` G2S 的 128 B 为 `461;460`，
1 KiB 为 `460;460`，2 KiB 和 4 KiB 也都是 `460;460`。B200 的
128 B 与 1 KiB 均为 `558;461`。因此约 460-cycle 平台不是一次偶然值；
它是热态命令稳定存在的固定延迟平台。B200 的首条命令比紧邻的第二条高
约 97 cycles，H100 多数点首/次相同，但在服务区间边界也会出现首条较高。
上面的旧 capacity 表每点只有一个 recorded sample，其中 128 B 的
`470/570` 应视为单样本历史参考；讨论重复性和平台边界时，以本次原始双样本
为准。

两个样本的准确含义不是两次独立 Modal 运行，也不能直接标成“L2 cold/hot”：
host 先用 `cuTensorMapEncodeTiled` 生成一次 `CUtensorMap`，随后用完全相同的
descriptor 和 source data 运行一次独立 warmup kernel；计时 kernel 内再连续
执行两条相同命令，表格依次保存 iteration 0 和 iteration 1。host TensorMap
编码不在 `clock64()` 区间内，因而 `558→461` 不是省掉 host descriptor
编译。warmup 已读取相同 source，纯粹的 L2 首次冷读也不像是 97-cycle 固定差
的主因。

当前更有力的解释是“计时 kernel/当前 barrier phase 的首次使用固定开销，叠加
mbarrier 轮询落点变化”；CUDA 未公开的 descriptor fetch/decode cache 仍可能
贡献一部分，但本实验把 descriptor 地址和 data 地址同时复用了，不能定量拆分。
要严格拆分，需要增加四个正交用例：相同 descriptor+不同 tile、不同 descriptor+
相同 data、相同 descriptor+相同 data，以及在同一 CTA 内先做不计时 warmup
再记录 iteration 0；同时改用紧凑 inline-PTX try-wait 记录轮询次数。

G2S 随容量并非每 32 B 线性增长，而是出现 `460/608/756`、间隔约
148 cycles 的量化台阶。用项目本地 CUDA 13.1 对同一源代码重编译 sm_90
和 sm_100 并检查 SASS 后，能看到 2D 路径的 `UTMALDG.2D` 之后是
`SYNCS.PHASECHK.TRANS64.TRYWAIT` 轮询；未完成时经过 `YIELD` 回到循环，
连续失败达到阈值后还存在自适应 `NANOSLEEP` 路径。因此这组台阶首先是
`cuda::barrier::arrive_and_wait()` 完成观测的轮询量化，而不是可直接解释为
TMA 内部 wave 宽度的数字。

这也更准确地解释了 128 B 与 1 KiB 为什么能完全相同：warmup 已使 descriptor
和数据路径进入热态，TMA 又会异步并行处理多个 sector；两种容量的真实完成
时刻虽然未必相同，但只要都落在同一次成功的 mbarrier 轮询之前，软件可见的
`clock64()` 差值就会同为约 460 cycles。若要继续分辨平台内部的细小差异，
下一版探针应直接使用更紧凑的 inline-PTX `mbarrier.try_wait` 循环，并同时保留
当前 CUDA C++ barrier 路径作为应用可见参考。

这里的测量边界与 CUDA 官方语义一致：G2S 用 shared-memory barrier 等待
预计字节数完成，S2G 用 bulk async-group commit/wait；TMA 由单线程发起，
数据搬运由异步引擎完成。参考
[CUDA Programming Guide: TMA async copies](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/04-special-topics/async-copies.html)。

全部原始双样本和三方按“方向 → 格式 → 长度”排列的表见
[2D length probe 报告](../benchmarks/tma-cycle-compare/results/tensor_2d_length_compare_v3_10/REPORT.md)。

## S2G

### Tensor capacity：plain U8 2D、128B row

| 长度 | H100 | B200 | Ventus V3.10 |
|---:|---:|---:|---:|
| 128 B | 104 | 140 | 109 |
| 256 B | 108 | 144 | 111 |
| 512 B | 116 | 152 | 115 |
| 1 KiB | 132 | 168 | 141 |
| 2 KiB | 164 | 200 | 163 |
| 4 KiB | 228 | 264 | 213 |
| 8 KiB | 356 | 392 | 323 |
| 16 KiB | 612 | 648 | 505 |
| 32 KiB | 1124 | 1160 | 890 |

补充探针把 S2G 的规律分辨得很清楚。H100 `u8_row32` 的 32/64/96/
128 B 分别为 `98;56`、`99;57`、`100;58`、`101;59`；B200 为
`126;87`、`127;88`、`128;89`、`129;90`。两台 GPU 的两个重复序列
都是每增加 32 B 增加 1 cycle，只是首条命令分别固定多约 42/39 cycles。
这与 NVIDIA 官方模型中“L2 128B cache line 由四个 32B sector 组成，L2
最小访问单位为一个 sector”吻合。它是很强的相关证据，但若没有 NVIDIA
内部 TMA 计数器，不能把每个软件可见周期机械等同为一个物理 sector 写入。
cache line、sector 与 wavefront 定义参考
[Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)。

## Roundtrip

Roundtrip 是同参数完整 G2S 后接 S2G；补充的 32B 探针只测单向路径，避免
翻倍 Modal 成本并混淆两个方向的阶梯。

| 长度 | H100 | B200 | Ventus V3.10 |
|---:|---:|---:|---:|
| 128 B | 605 | 575 | 238 |
| 256 B | 609 | 579 | 242 |
| 512 B | 610 | 576 | 250 |
| 1 KiB | 623 | 592 | 284 |
| 2 KiB | 655 | 772 | 329 |
| 4 KiB | 719 | 836 | 416 |
| 8 KiB | 995 | 964 | 603 |
| 16 KiB | 1251 | 1368 | 917 |
| 32 KiB | 1911 | 1880 | 1566 |

## 大容量边际吞吐

用 16→32 KiB 的周期增量除以新增的 128 个 128B line：

| Path | H100 cycle/128B | B200 cycle/128B | Ventus V3.10 cycle/128B |
|---|---:|---:|---:|
| Tensor G2S | 1.156 | 1.156 | 2.063 |
| Tensor S2G | 4.000 | 4.000 | 3.008 |
| Tensor roundtrip | 5.156 | 4.000 | 5.070 |

这比绝对周期更适合判断结构瓶颈：Ventus 的固定启动延迟较小且 L2 模型
乐观，但 G2S 连续 line 消费仍约比 NVIDIA 慢 `2.063/1.156 = 1.78x`。
V3.10 新增的 rank carry 流水只改变启动周期，没有改善或恶化这条斜率。

## 全矩阵统计

334 个三方连接点中，`Ventus/H100` 周期比中位数为 0.5545，
`Ventus/B200` 为 0.4923。按方向：

| Direction | 配置数 | V/H100 中位数 | V/B200 中位数 | Ventus 慢于 H100 | Ventus 慢于 B200 |
|---|---:|---:|---:|---:|---:|
| G2S | 130 | 0.3478 | 0.3728 | 0 | 0 |
| S2G | 112 | 0.9342 | 0.8068 | 43 | 17 |
| roundtrip | 92 | 0.5638 | 0.4911 | 0 | 0 |

这些绝对比值只能描述当前 GVM 与实测 CUDA 的软件可见周期，不能当作同工艺
硬件性能倍率。特别是 S2G write-ack 和 backing memory 是简化 RTL 模型，
其绝对优势最不可信。

## V3.8 到 V3.10 的精确差分

| 路径 | 配置数 | 每项周期变化 |
|---|---:|---:|
| Bulk G2S | 17 | 0 |
| Bulk S2G | 17 | 0 |
| Tensor G2S | 113 | +2 |
| Tensor S2G | 95 | +2 |
| Tensor roundtrip | 92 | +4 |

这与 V3.9/V3.10 每次为 Tensor cursor carry 增加一级弹性边界的设计预期完全
一致。没有发现随容量增长的额外 delta，因此当前最新结构的性能变化可归因于
固定 pipeline fill，而不是 WindowPlanner II 或后端吞吐退化。

## 正确性复测说明

并行 layout 首轮在 `swizzle64_roundtrip_b16384_plain` 出现一次
`validation_error`；顺序恢复时该项通过，随后
`swizzle128_roundtrip_b16384_swz` 出现一次同类错误。第二次恢复后主矩阵
54/54，且两项分别再做三次独立运行，6/6 通过、周期稳定。

原始失败和恢复记录保留在 layout `outcomes.jsonl`。这种跨不同 case 漂移、
立即独立复测通过的模式更符合 GVM 随机初值/完成观测偶发，而不是固定 layout
映射错误；不过在正式冻结前仍建议把这两项加入多 seed RTL 回归。

## 结果位置

- [V3.10 capacity raw](../benchmarks/tma-cycle-compare/results/features_ventus_v3_10_capacity/attempt_01.csv)
- [V3.10 comprehensive raw](../benchmarks/tma-cycle-compare/results/comprehensive_ventus_v3_10/)
- [H100 raw](../benchmarks/tma-cycle-compare/results/comprehensive_cuda_h100/)
- [B200 raw](../benchmarks/tma-cycle-compare/results/comprehensive_cuda_b200/)
- [三方精确连接表](../benchmarks/tma-cycle-compare/results/comprehensive_compare_v3_10/ratios.csv)
- [覆盖表](../benchmarks/tma-cycle-compare/results/comprehensive_compare_v3_10/coverage.csv)
- [敏感 case 独立复测 1](../benchmarks/tma-cycle-compare/results/comprehensive_ventus_v3_10_layout_stability_1/attempt_01.csv)
- [敏感 case 独立复测 2](../benchmarks/tma-cycle-compare/results/comprehensive_ventus_v3_10_layout_stability_2/attempt_01.csv)
- [敏感 case 独立复测 3](../benchmarks/tma-cycle-compare/results/comprehensive_ventus_v3_10_layout_stability_3/attempt_01.csv)
- [2D length H100 raw](../benchmarks/tma-cycle-compare/results/tensor_2d_length_cuda_h100/tensor_2d_length.csv)
- [2D length B200 raw](../benchmarks/tma-cycle-compare/results/tensor_2d_length_cuda_b200/tensor_2d_length.csv)
- [2D length Ventus raw](../benchmarks/tma-cycle-compare/results/tensor_2d_length_ventus_v3_10_r1/attempt_01.csv)
- [2D length Ventus 失败独立复测](../benchmarks/tma-cycle-compare/results/tensor_2d_length_ventus_v3_10_fail_recheck_1/outcomes.jsonl)
- [2D length 方向优先报告](../benchmarks/tma-cycle-compare/results/tensor_2d_length_compare_v3_10/REPORT.md)
