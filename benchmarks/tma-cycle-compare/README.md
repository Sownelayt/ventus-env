# TMA 周期测试：ColdThenHotV1

本目录只维护一套权威测试方法和对应结果：`ColdThenHotV1`。它替代已经撤回的
`FreshTensorMapV1`；旧 suite、旧数据、失败探针和旧报告均不参与比较。

## 冷、热两次到底是什么

除明确标注的 payload 驻留与 prefetch 控制外，每个 Tensor case 只创建一张
此前未被 TMAU 使用过的 TensorMap，并在同一个 kernel 中连续执行两次：

1. `repeat=0 / cold`：TensorMap 可以由普通 `.cg` load 放入 L2，但此前没有
   TMA 指令或 `prefetch.tensormap` 消费它；
2. `repeat=1 / hot`：第一条命令完成后，立即使用完全相同的 descriptor 地址和
   128B 内容再次执行；两次之间没有 prefetch、invalidate 或 acquire。

因此第一次包含 descriptor 从 L2 进入 TMAU并建立内部状态的开销，第二次才是
同一 TensorMap 已被真正执行过一次后的热状态。所有 Tensor 原始记录保存地址、
内容指纹、`prior_tma_use=0/1`、`pair_role=cold/hot` 和 prefetch 标志；统一校验器
会拒绝地址/内容不一致或隐藏预取的样本。

Ventus 的每个 case 都使用新启动的 RTL 实例，所以 compiled store 在 case 开始时
为空；case 内仍是同一实例中的冷→热连续执行。

## 当前固定项目

|项目|版本|设备|内容|
|---|---:|---|---|
|共同单指令矩阵|v4|H100/B200/Ventus|334 个安全非 Reduce 配置，含 Bulk/Tensor、G2S/S2G/roundtrip、rank、dtype、stride、OOB、swizzle、interleave16|
|2D 容量矩阵|v4|H100/B200/Ventus|182 个二维 Tensor 点，32B–16KiB 围绕 32B 边界加密，之后保留关键大容量点|
|G2S 完成边界|v12|H100/B200|固定单轮 640 点、请求步长 3 周期的单次 `mbarrier.test_wait` 扫描，冷、热分别给出中点周期与误差|
|TensorMap 准备|v4|H100/B200/Ventus|rank 1–5、cold demand、compiled hit、显式 prefetch、prefetch lead；Ventus 另有 PMU|
|128B payload 驻留|v5|H100/B200|计时外先完整执行一条同图 TMA 固定 descriptor 为热，再测 cold→hot、cold→cold、hot→hot payload|
|同向多命令|v4|H100/B200/Ventus|G2S/S2G 分开，same/distinct map、同 CTA/多 CTA、serial/batched、1–32 条命令|
|S2G 长度扫描|v1|H100/B200|固定 Tensor/same-map/batched/同 CTA，只改变 128B–4KiB payload，1–32 条命令|
|混合方向多命令|v3|H100/B200/Ventus|alternating、G2S→S2G、S2G→G2S|
|控制项|v3|H100/B200|4KiB cooperative copy 与 compute/TMA overlap|

所有正式运行固定 `warmups=0`、`repeats=2`；两次样本在同一 kernel 内连续完成。
CUDA 每个项目只有一次付费运行、`retries=0`、`max_containers=1`；Ventus 本地运行
的 `paid_attempts=0`。Reduce、elementStride、已知 XID 13 组合和 interleave32 不在
当前正式矩阵中。

## 目录和溯源

```text
CURRENT_MEASUREMENT_RELEASE
releases/ColdThenHotV1.json
suites/<scope>/<project>/<version>/    不可变测试源码与 suite.yaml
data/<device>/<version>/...            26 个通过验证的正式运行
reports/comparisons/ColdThenHotV1/      唯一正式报告与派生 CSV
catalog/runs.csv                        run 到原始目录的索引
schemas/                                manifest schema
run.py                                  统一计划、静态检查和正式运行入口
generate_reports.py                     只读取当前发布数据
validate_repository.py                  完整性、哈希和冷/热身份校验
```

每个 run manifest 记录 suite/version、设备、CUDA 或 Ventus 实现哈希、命令行、
wait 方法、case matrix hash 和所有源码 SHA-256。报告中的 `run_inventory.csv` 可从
每个表格项目追溯到原始 CSV。

## 使用

只查看计划或进行不分配 GPU 的导入检查：

```bash
python3 benchmarks/tma-cycle-compare/run.py --list
python3 benchmarks/tma-cycle-compare/run.py \
  --suite single_instruction.common_matrix --device h100 --plan
python3 benchmarks/tma-cycle-compare/run.py \
  --suite single_instruction.completion_latency --device h100 \
  --variant bulk-tensor__g2s__fresh-map-common --import-only
```

正式运行必须显式使用 `--formal`。相同签名默认拒绝重跑；只有
`--force-rerun --reason '原因'` 才能绕过，并把理由写入 manifest。

生成并验证唯一正式报告：

```bash
python3 benchmarks/tma-cycle-compare/generate_reports.py
python3 benchmarks/tma-cycle-compare/validate_repository.py
```

- [三平台正式报告](reports/comparisons/ColdThenHotV1/REPORT.md)
- [全部正式运行清单](reports/comparisons/ColdThenHotV1/run_inventory.csv)
- [G2S 原始汇总](reports/comparisons/ColdThenHotV1/single_instruction_g2s.csv)
- [S2G 原始汇总](reports/comparisons/ColdThenHotV1/single_instruction_s2g.csv)

`releases/FreshTensorMapV1.RETRACTED` 只保留撤回原因，不包含旧测试或旧结果。
