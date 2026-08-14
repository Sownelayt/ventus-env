# TensorMap 首次发射延迟归因（V1）

## 结论

此前在 completion-latency v12 大 kernel 中观察到的 H100 `257 → 187`（差 70）和 B200 `313 → 221`（差 92），**不是 NVIDIA TMA 固有的一段 70/92 周期硬件流程**，也不能归因于 TensorMap 绑定或 mbarrier 绑定。

固定为 rank-2、只保留一个静态 TMA 指令位置后，两张卡、两个方向的第一次和第二次 issue 完全相等：

|设备|方向|第一次 issue|第二次 issue|差值|
|---|---|---:|---:|---:|
|H100|G2S|111|111|0|
|H100|S2G|39|39|0|
|B200|G2S|96|96|0|
|B200|S2G|22|22|0|

因此旧差值主要属于旧测量程序本身：运行时 rank 1–5 分派、五个 TMA 指令分支和大体积延迟梯形代码共同形成的首次代码路径效应。它可能包含指令缓存首次取入、分支目标首次到达、调度和 uniform-register 准备；它不是 descriptor decode 的计数器，也不是一个可以映射到 NVIDIA 专利某个模块的固定延迟。

## 怎样排除 TensorMap 和 TMAU 首次工作

每个场景都在一个新 CUDA 进程和新 context 中运行。被测 TensorMap 和控制 TensorMap 均满足：

- 位于不同的 4 KiB 页；
- 先用普通 `.cg` load 放入 L2；
- 计时前执行 `prefetch.tensormap`，lead 为 1024 cycles；
- payload/目标 line 同样已用 `.cg` load 变热；
- 被测命令固定为 2D、U8、128B；
- PTX/SASS 审计确认每个方向只有一个物理 `UTMALDG.2D` 或 `UTMASTG.2D` 位置。

随后分别运行三种模式：

|模式|在被测两条命令前做什么|能预热什么|
|---|---|---|
|none|什么都不做|无额外预启动|
|predicated_off_same_site|同一 PTX 位置执行 predicate=false；ptxas 降成绕过唯一 UTMA 指令的分支|周围取指和控制流；不向 TMAU 发请求|
|other_tensormap_same_site|在同一个物理 UTMA 位置真正执行另一张已 prefetch 的 TensorMap，并等到完成|完整 TMA issue/完成通路，但没有执行被测图|

三种模式的被测第一次/第二次 issue 都相同：

|设备|方向|none|false 控制后|另一张图完整 TMA 后|
|---|---|---:|---:|---:|
|H100|G2S|111 → 111|111 → 111|111 → 111|
|H100|S2G|39 → 39|39 → 39|39 → 39|
|B200|G2S|96 → 96|96 → 96|96 → 96|
|B200|S2G|22 → 22|22 → 22|22 → 22|

如果 prefetch 后还存在“第一次真实 Tensor 指令需要额外绑定 TensorMap、激活 TMAU 或绑定 mbarrier”的固定代价，那么 none 模式的第一次应慢于第二次，或者 other-map prime 至少应缩短第一次；实测两者都没有发生。

## 目前真正的 issue 时间包含什么

这里的 `issue_cycles` 是同一线程中 `clock64()` 读数包住物理 UTMA 发射序列的差值，不是从 SM 外部观察到的端口握手周期。

H100 的最终 SASS 显示，G2S 的计时范围包括：

1. 读取开始时钟；
2. 选择 TensorMap 地址并把地址转入 uniform register；
3. 准备 shared/barrier 的 uniform operand；
4. `ELECT`/一致线程选择序列；
5. `UTMALDG.2D` 向 TMA 路径提交，必要时在其短循环中等待接受；
6. 读取结束时钟。

`mbarrier.init` 和 `mbarrier.arrive.expect_tx` 在开始时钟之前。SASS 中对应的 `SYNCS.ARRIVE.TRANS64` 也位于开始 `CS2R` 之前。因此 G2S 的 111/96 cycles **不包含 mbarrier reserve 本身**；UTMALDG 只携带已经准备好的 barrier 地址。

S2G 不使用 mbarrier。其计时范围主要是 TensorMap 地址的 uniform 化、`UTMASTG.2D` 提交以及两个时钟读数，所以只有 H100 39 cycles、B200 22 cycles。`commit_group` 和 `wait_group 0` 位于 issue 结束时钟之后，只进入 completion 时间。

predicate=false 控制路径本身为：H100 G2S/S2G `45/15` cycles，B200 `51/18` cycles。它给出了时钟读取、循环控制、地址选择和保护分支的近似底噪，但不能简单从真实 issue 相减得到纯 UTMA 延迟，因为启用路径还会执行额外的 uniform-register 和 `ELECT` 序列。

## 对旧 70/92 周期说法的修正

旧 v12 kernel 在一个运行时 switch 中保留 rank 1–5 五个 Tensor 指令目标，并携带很大的静态延迟 ladder。该环境中的第一次 issue 比第二次慢 70/92 cycles，但：

- descriptor 已 prefetch 时差值仍在，所以它本来就不像 descriptor L2 refill/decode；
- Bulk 没有相同的固定差值；
- 本次固定 rank、单物理 UTMA 位置后，差值在 H100/B200 上都严格变成 0；
- false 预取指和另一张图完整 TMA 都不能进一步改善已经相等的结果。

所以最稳妥的结论是：旧 70/92 是**大测试 harness 的首次 Tensor 分支路径开销**，其中最可能的主体是较大代码体积下的指令获取/分支目标首次到达及其调度，而不是 TMAU 内一个可见的固定阶段。没有公开内部计数器，不能再把这部分细分成若干精确 NVIDIA 模块周期。

## 原始数据与实现

- [H100 原始 CSV](../../../data/h100/cuda-13.3-sm90/single_instruction/tensormap_issue_attribution/tensor2d-128b__g2s-s2g/20260805T102939Z__v1__964164880590/attempt_01.csv)
- [B200 原始 CSV](../../../data/b200/cuda-13.3-sm100/single_instruction/tensormap_issue_attribution/tensor2d-128b__g2s-s2g/20260805T103025Z__v1__964164880590/attempt_01.csv)
- [汇总 CSV](summary.csv)
- [测试定义](../../../suites/single_instruction/tensormap_issue_attribution/v1/suite.yaml)
- [CUDA 源码](../../../suites/single_instruction/tensormap_issue_attribution/v1/cuda_tma_issue_attribution.cu)

两次 Modal GPU 函数的实际 wall time 分别为 H100 1.574 秒、B200 3.306 秒；每个子进程的 timeout 为 2 秒，函数 timeout 为 10 秒，retries=0。
