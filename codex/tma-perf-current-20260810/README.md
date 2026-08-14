# TMA 当前未提交工作树本地性能与正确性回归

日期：2026-08-10

## 结论

本轮以 `/home/liyb/ventus-env` 当前未提交工作树为唯一基线完成回归，没有按 commit 回退或重建代码。最终结果如下：

- cached GVM（Verilator RTL）完整性能/压力套件 7/7 通过。
- cached 与 nocache movement 性能门禁均通过；25 个点的平均及最坏退化比例均显著低于 `1.10/1.20` 门限。
- Spike 性能/profile 套件 6/6 通过。
- TMA 与 AtomicUnit Scala 单元测试 9 个 suite、71 个 test 全部通过。
- PMU-on cached/nocache 构建通过；回归结束后又以 PMU-off 全量重建 cached/nocache，生成、Verilator 编译、链接、安装均通过。
- 当前没有观察到由这轮 TMA 修改造成的性能门禁回退或已覆盖功能损坏。

## 测试中发现并修复的问题

### 1. S2G shared bank-conflict 最后一拍可能选择错误 payload

原实现只在非最终 replay beat 上按 `sharedResponse.source` 选择 payload。若最后一拍仍是部分 bank-conflict 响应，可能与错误的 payload 合并，覆盖先前 replay 已收集的数据。

修复位置：

- `/home/liyb/ventus-env/gpgpu/ventus/src/pipeline/TmaV2WindowEngine.scala` 中的 `s2gSharedMerge`。
- 对每个有效 S2G shared response beat 都按 source 选择正确 payload，并在合并拍阻止 cache issue 争用唯一读端口。
- 新增 `/home/liyb/ventus-env/gpgpu/ventus/tests/src/DmaTest/TmaV2Backend_test.scala` 回归：`S2G final shared response beat preserves data from earlier replays`。

### 2. AtomicUnit 可能把同 architectural source 的普通 L2 响应误认成内部 AMO 响应

AtomicUnit 的 Get/Put 内部事务若仅复用 architectural source，普通 cache 响应可能被错误消费；同时旧的额外高位 tag 在 Scheduler 进入真实 L2 source 宽度时会被截断。

修复位置：

- `/home/liyb/ventus-env/gpgpu/ventus/src/L1Cache/AtomicUnit/AtomicUnit.scala` 中的 `atomicTaggedSource`。
- 内部 Get/Put 使用实际 L2 source 中未被合法 cache requester 使用的 requester ID `3`；最终上游 AMO 响应恢复原 source。
- Atomic FSM 明确分为 `waitGetRsp -> issuePutReq -> waitPutRsp`，Put 最终确认后才向上游完成。
- 新增 `/home/liyb/ventus-env/gpgpu/ventus/tests/src/cache/AtomicUnit_test.scala` 回归：`AtomicUnit does not consume an ordinary response with the same architectural source`。

该修复后，原先会在 `128 words × 8 splits = 1024 AMO` 压力点遗漏一次累加的问题消失。

### 3. Reduce 性能测试原地修改同一 descriptor，触发跨 SM descriptor cache 陈旧问题

该问题属于测试构造，不是数据面 RTL：测试在不同尺寸之间反复改写同一 descriptor 地址，却没有执行跨 SM invalidate。现改为给 `{8, 32, 128}` words 各分配一份不可变 descriptor，并在运行前一次性配置。

修复位置：`/home/liyb/ventus-env/testcases/_get_case/dma_tma_v2_reduce_perf_test/dma_tma_v2_reduce_perf_test.cpp`。

## 最终性能结果

| 后端/套件 | 覆盖 | 最终结果 |
|---|---|---|
| cached GVM movement gate | bulk/tensor、G2S/S2G/G2S→S2G，25 点 | PASS；`avg_ratio=0.5402`，`max_ratio=0.8072` |
| nocache GVM movement gate | 同上，25 点 | PASS；`avg_ratio=0.5106`，`max_ratio=0.7854` |
| cached GVM ping-pong | G2S、manual-input S2G、TMA-input G2S→S2G | 3/3 PASS |
| cached GVM Tensor feature stress | 66 case、双向、每项 8 次；rank 2–5、swizzle、interleave、stride、misaligned subbox、OOB | PASS |
| cached GVM contention | bulk/tensor、双向/混合、same-set/spread、multi-WG、dual-warp、burst | 158 cases，0 failure |
| cached GVM reduce | 8/32/128 words × 1/2/4/8 splits | 12 cases，0 failure；1024 AMO 压力点通过 |
| Spike perf/profile | 3 个 ping-pong + feature + contention + reduce | 6/6 PASS |

门禁定义是 `ratio = TMA cycles / reference cycles`，因此数值小于 1 表示 TMA 更快。门限为平均不超过 `1.10`、任一点不超过 `1.20`；当前 cached 与 nocache 最坏点分别为 `0.8072`、`0.7854`。

代表性 ping-pong 数据（64×64，2 buffers，8 stages）：

- G2S：`299568` vs manual `445142` cycles，约快 32.70%。
- manual-input Tensor S2G：`330640` vs manual `444778` cycles，约快 25.66%。
- TMA-input G2S→S2G：`183822` vs manual `444744` cycles，约快 58.67%。

Reduce 最重压力点：`128 words × 8 splits`，共 1024 次 AMO，`tma_cycles=15948`，数据校验通过。Reduce 的性能 crossover 并非所有尺寸都优于 two-kernel baseline；这是 profile 结果而不是门禁失败，12 个点的功能结果全部正确。

## 正确性与容量回归

以下 Scala suite 一次性运行，结果为 9 suite、71 test、0 failure：

- `TmaV2Capacity_test`
- `TmaControlScheduler_test`
- `TmaV2Completion_test`
- `TmaV2Frontend_test`
- `TmaV2Backend_test`
- `TmaV2CreditDepth_test`
- `TmaV2DmaCore_test`
- `TmaV2GroupTracker_test`
- `AtomicUnitTest`

关键覆盖包括：

- 256-entry source namespace 的 source 255 与乱序返回。
- 32 KiB planner 精确生成 256 个 window。
- 默认 40 credit 在模拟 latency 38 下达到 1 cycle/line，提升到 48 不再改善该测试点。
- Bulk 跨 128B shared set 的正确切分。
- G2S/S2G copy、bulk reduce、tensor reduce、4 KiB 跨页、partial mask。
- unknown funct0 与非法对齐均保证零内存流量。
- descriptor prefetch/invalidate/refetch。
- 一条 active + 一条 lookahead，第三条正确反压。
- shared replay 最后一拍、S2G 最终 write ack、group/mbarrier completion。
- Atomic Get/Put ack 顺序、backpressure 和同 source 普通响应隔离。

## 构建与静态检查

- PMU-on cached 和 nocache GVM 全量构建成功，用于本轮性能采样。
- 回归结束后，cached/nocache 均用 `VENTUS_PMU_TMA=0`、`VENTUS_PMU_TMA_DETAIL=0` 全量重建成功。
- 两套最终生成参数均为 `PMU_TMA=0`、`PMU_TMA_DETAIL=0`；生成 SV 中不存在 `io_perf_tma` 或 `[TMA PERF]`，说明默认构建没有实例化 TMA PMU 数据通路。
- `gpgpu` 与 `testcases` 的 `git diff --check` 均通过。
- 未执行 reset/checkout，没有覆盖或清理工作树中其他未提交修改和生成物。

## 结果索引

- Spike 6/6：[`spike/summary.csv`](spike/summary.csv)
- cached ping-pong 3/3：[`gvm-pingpong-rerun/summary.csv`](gvm-pingpong-rerun/summary.csv)
- cached movement + feature 2/2：[`gvm-core-perf-rerun/summary.csv`](gvm-core-perf-rerun/summary.csv)
- cached contention：[`gvm-contention-full-fixed/summary.csv`](gvm-contention-full-fixed/summary.csv)
- cached reduce：[`gvm-reduce-tag-full/summary.csv`](gvm-reduce-tag-full/summary.csv)
- nocache movement：[`gvm-nocache-movement/summary.csv`](gvm-nocache-movement/summary.csv)

这些目录只列最终有效重跑结果；同级其他目录包含定位上述问题时产生的中间失败/诊断记录，不应作为最终门禁结论。

## 已知验证边界

GVM 是 Verilator RTL，但现有 GVM reference/lockstep 模型不能完整建模 TMA 与部分 multi-WG 行为，相关长日志中可能出现 reference mismatch 诊断。上述 directed suite 的 PASS 来自主机端对输出 buffer、status、完成顺序和 PMU 事件的校验；因此这是一轮完整的 RTL directed 功能/性能回归，但不等同于 TMA 指令逐条 lockstep 证明。

本地 256-entry 覆盖包含 Scala 容量、source 255、乱序回收和 256-window 测试；256 outstanding 的 DC 综合是独立远端任务，不计入本地性能数字。
