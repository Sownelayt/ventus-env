# DMA/TMA 验证方法导读

更新时间：2026-07-03

本文解释当前 DMA/TMA 测试体系怎么运行、每类测试验证什么、性能数据怎么解读，以及 RTL 修改后应该如何选择最小和全量 gate。

## 1. 验证目标

当前 DMA/TMA 验证不只是跑 PASS/FAIL，而是同时回答四类问题：

1. 功能正确性：G2S、bulk S2G、tensor S2G、mixed route 和 multi-WG 是否写对数据。
2. completion/fence 正确性：per-warp DMA group ring 是否正确统计 G2S/S2G/prefetch completion，commit_group/wait_group 的年龄判断、wrap 和 slot 复用是否正确。
3. 性能收益：TMA/S2G 是否相对 full-warp manual baseline 有收益，旧路径是否没有明显退化。
4. 高级特性代价：swizzle、interleave、stride、OOB、rank3/4/5 是否只付合理代价，没有 10x 级 worst-case。

## 2. 环境准备

每次运行测试前：

```bash
source env.sh
cd testcases/_get_case
```

项目规则里强调，ventus 相关工具都来自 `./install` 目录，依赖 `source env.sh` 设置环境变量。

## 3. 构建命令

完整构建：

```bash
./build-ventus.sh --build "rtlsim;gvm;driver;pocl"
```

若只在测试目录构建某个 case：

```bash
make -C testcases/_get_case/dma_tma_s2g_func_test -j8
make -C testcases/_get_case/dma_tma_multi_wg_func_test -j8
```

RTL 后端构建较慢，建议一次构建后做一组验证，不要每个小点都重新构建。长构建可以约每 5 分钟查看一次状态。

## 4. 统一运行脚本

DMA/TMA 默认使用：

```bash
testcases/_get_case/run_dma_tma_rtl.sh
```

常用参数：

| 参数 | 含义 |
|---|---|
| `--backend gvm` | 使用 GVM |
| `--backend gvm-nocache` | 使用 GVM nocache 配置 |
| `--backend rtl` | 使用 cache RTL |
| `--backend rtl-nocache` | 使用 nocache RTL |
| `--suite directed` | 运行 directed/function 类回归 |
| `--suite all` | 运行 registry 中所有 DMA/TMA mode |
| `--tag perf-full` | 按 tag 过滤 |
| `--case <name>` | 按 mode label 过滤 |
| `--run-arg <arg>` | 给测试程序额外传参 |
| `--jobs 8` | 单测试内部编译/运行并行度 |
| `--run-jobs 4` | 同时运行最多 4 个测试 |
| `--timeout <sec>` | 单 case timeout |
| `--list` | 列出匹配 case，不实际运行 |

推荐默认：

```bash
./run_dma_tma_rtl.sh --suite directed --backend rtl --run-jobs 4 --jobs 8 --timeout 1800
./run_dma_tma_rtl.sh --suite all --backend rtl --run-jobs 4 --jobs 8 --timeout 7200
```

`--run-jobs 4` 的设计是每个测试大约消耗 8 核，最多 4 个测试约 32 核，对当前服务器压力可接受。

## 5. Registry

当前默认 registry 是：

```text
testcases/_get_case/cases_dma_tma.csv
```

当前默认保留 8 个 DMA/TMA mode：

| 类别 | 目录 | 作用 |
|---|---|---|
| G2S functional | `dma_tma_g2s_func_test` | 保护旧 G2S bulk/TMA 功能 |
| S2G functional | `dma_tma_s2g_func_test` | 保护 bulk/tensor/mixed S2G 功能 |
| multi-WG functional | `dma_tma_multi_wg_func_test` | 保护多 WG route/group/page/descriptor 隔离 |
| G2S pingpong perf | `dma_tma_g2s_pingpong_perf_test` | 保护旧 TMA G2S overlap 性能 |
| tensor S2G perf manual input | `dma_tma_tensor_s2g_pingpong_perf_test sweep manual_tensor_s2g` | 评估 manual input + tensor S2G writeback |
| tensor S2G perf TMA input | `dma_tma_tensor_s2g_pingpong_perf_test sweep tma_tensor_s2g` | 评估 TMA G2S input + tensor S2G writeback |
| DMA/TMA movement profile | `dma_tma_movement_profile_test` | 比较 bulk/tensor G2S/S2G 单次搬运 PMU cycle |
| S2G feature perf | `dma_tma_s2g_feature_perf_test sweep 8` | 诊断 swizzle/interleave/stride/OOB 等 feature 代价 |

已经从默认回归中删除或降级的内容：

| 项 | 当前处理 |
|---|---|
| `tma_roundtrip_pipeline_perf_test` | 已物理删除，不再默认运行 |
| `dma_tma_s2g_pipeline_perf_test` | 已物理删除，bulk 性能不再作为默认主线 |
| quick mode | 不保留单独 quick registry，使用 full 测试参数控制 |
| `tma_gemm_perf_test` | 源码可保留为未来应用级测试，不进入 DMA/TMA 默认回归 |

## 6. 后端策略

建议按下面顺序判断问题：

### 6.1 GVM

GVM 主要用于快速验证 host、kernel、reference 和 descriptor 语义。若 GVM 失败，通常不是 RTL 微结构问题，应先看：

1. host 参数是否设置对。
2. descriptor 生成是否符合 kernel 期望。
3. reference 是否和 OOB/stride/swizzle 语义一致。
4. kernel 是否有未初始化 shared/global 数据。

### 6.2 RTL

RTL 是最终设计验收的主后端。若 GVM 通过但 RTL 失败，优先看：

1. shared response route。
2. TLB owner route。
3. L2 ack source/tag route。
4. completion arbiter 和 group counter。
5. 跨页/TLB owner、L2 ack route 或 group completion。

### 6.3 No-cache

nocache 后端用来排除 cache 假设。若 cache RTL 通过但 rtl-nocache 失败，优先看：

1. 地址翻译是否绕过/命中不一致。
2. L2/TLB route 是否依赖 cache metadata。
3. DMA L1TLB 和 L2/nocache response route 是否完整。
4. partial write mask 是否被 nocache path 正确处理。

## 7. 功能测试方法

### 7.1 单独构建和运行

S2G 单 WG：

```bash
make -C testcases/_get_case/dma_tma_s2g_func_test -j8
cd testcases/_get_case/dma_tma_s2g_func_test
./dma_tma_s2g_func_test.out
```

Multi-WG：

```bash
make -C testcases/_get_case/dma_tma_multi_wg_func_test -j8
cd testcases/_get_case/dma_tma_multi_wg_func_test
./dma_tma_multi_wg_func_test.out
```

G2S：

```bash
make -C testcases/_get_case/dma_tma_g2s_func_test -j8
cd testcases/_get_case/dma_tma_g2s_func_test
./dma_tma_g2s_func_test.out
```

### 7.2 Filter

部分 S2G functional 支持通过环境变量过滤：

```bash
S2G_TENSOR_CASE_FILTER=fuzz_tensor ./dma_tma_s2g_func_test.out
S2G_MIXED_CASE_FILTER=mixed_tensor_outstanding_long ./dma_tma_s2g_func_test.out
```

这类 filter 用于局部复现，不替代完整 directed gate。

### 7.3 Directed gate

RTL directed gate：

```bash
cd testcases/_get_case
./run_dma_tma_rtl.sh --suite directed --backend rtl --run-jobs 4 --jobs 8 --timeout 1800
```

rtl-nocache directed gate：

```bash
cd testcases/_get_case
./run_dma_tma_rtl.sh --suite directed --backend rtl-nocache --run-jobs 4 --jobs 8 --timeout 1800
```

修改 shared/TLB/L2/completion route 后，至少需要跑 RTL 和 rtl-nocache directed。

## 8. 性能测试方法

### 8.1 G2S pingpong

命令：

```bash
cd testcases/_get_case/dma_tma_g2s_pingpong_perf_test
./dma_tma_g2s_pingpong_perf_test.out sweep
```

作用：

1. 保护旧 TMA G2S overlap 路径。
2. 提供 G2S + manual writeback 的参考。
3. 观察 tile/stage sweep 下 TMA input 的收益曲线。

典型 sweep 维度：

| 维度 | 值 |
|---|---|
| tile | `16x16`, `32x16`, `32x32`, `64x32`, `64x64` |
| buffer | 固定 `2` |
| stages | `1`, `2`, `4`, `8`, `16` |

### 8.2 tensor S2G pingpong

命令：

```bash
cd testcases/_get_case/dma_tma_tensor_s2g_pingpong_perf_test
./dma_tma_tensor_s2g_pingpong_perf_test.out sweep manual_tensor_s2g
./dma_tma_tensor_s2g_pingpong_perf_test.out sweep tma_tensor_s2g
```

作用：

1. 当前 tensor S2G 主性能 gate。
2. `manual_tensor_s2g` 隔离 writeback 方向，输入由 warp manual 完成。
3. `tma_tensor_s2g` 同时覆盖 TMA G2S input 和 tensor S2G writeback，是 end-to-end 主路线。
4. baseline 是 full-warp manual 搬运，不是单 lane 弱 baseline。

解读：

| 现象 | 可能原因 |
|---|---|
| 小 tile 差，大 tile 好 | descriptor/setup/fence 固定开销占比高 |
| 所有 tile 平台期 | outstanding/line table/ack 周转限制 |
| PutPart 异常多 | tensor feature 未被 coalesced fast path 覆盖 |
| lineFullStall/readFullStall 高 | backend table 或 shared read 周转限制 |
| no-wait overlap 写同一 byte | 程序未定义行为，应由 kernel 插入 wait/barrier |

### 8.3 Feature perf

命令：

```bash
cd testcases/_get_case/dma_tma_s2g_feature_perf_test
./dma_tma_s2g_feature_perf_test.out sweep 8
./dma_tma_s2g_feature_perf_test.out stress 8
./dma_tma_s2g_feature_perf_test.out single swizzle32_16x16 8
```

作用：

1. 做同参数 base/feature paired A/B。
2. 观察 swizzle/interleave/stride/OOB/high-rank 相对 non-feature baseline 的 cycle ratio。
3. 发现默认 pingpong 不一定覆盖到的高级 feature worst-case。

重要原则：

1. swizzle 必须和同 shape non-swizzle 比。
2. interleave 必须和同 shape non-interleave 比。
3. 不混用不同 tile 或不同 rank 作为 baseline。
4. stress 是诊断工具，不等同默认性能 gate。

### 8.4 DMA/TMA movement profile

命令：

```bash
cd testcases/_get_case/dma_tma_movement_profile_test
./dma_tma_movement_profile_test.out sweep
./dma_tma_movement_profile_test.out single tensor_s2g 32 32
```

作用：

1. 纯 movement microbench，无 compute，无 buffer/stage 参数。
2. 按同 tile baseline 对比 `bulk_g2s`, `tensor_g2s`, `bulk_s2g`, `tensor_s2g`。
3. PMU active cycles 是主指标；host ns 只作辅助诊断。
3. 不直接作为 S2G 性能优化目标。

## 9. PMU 解读

S2G 相关性能优先看这些指标：

| 指标 | 解读 |
|---|---|
| PutFull | 越高说明越多 full cacheline writeback |
| PutPart | 高级 feature 或 partial/OOB 导致的部分写 |
| PutPart/PutFull 比例 | 判断 coalescing 是否有效 |
| lineFullStall | line table 满 |
| readFullStall | shared read table 满 |
| ackFullStall | ack table 满 |
| ackLatencySum / ackCount | L2 ack 平均延迟；`ackLatencySum` 受 `PMU_TMA_DETAIL` 控制 |
| dmaFenceWait | group wait/fence 等待代价 |

不要只看 speedup。高级 feature 的合理目标是：

1. 功能 PASS。
2. paired ratio 不出现极端 10x 角落。
3. PutPart 数和 feature 预期一致。
4. 改动后 G2S pingpong 和 tensor S2G pingpong 没有明显退化。

## 10. 推荐 gate 顺序

### 10.1 文档或 host-only 轻改

```bash
git diff --check -- codex testcases/_get_case
```

如果只改文档，不需要跑 RTL 测试。

### 10.2 修改 functional 测试

```bash
make -C testcases/_get_case/dma_tma_s2g_func_test -j8
make -C testcases/_get_case/dma_tma_multi_wg_func_test -j8
cd testcases/_get_case
./run_dma_tma_rtl.sh --suite directed --backend gvm --run-jobs 4 --jobs 8 --timeout 1800
```

若涉及 RTL 敏感语义，再跑：

```bash
./run_dma_tma_rtl.sh --suite directed --backend rtl --run-jobs 4 --jobs 8 --timeout 1800
./run_dma_tma_rtl.sh --suite directed --backend rtl-nocache --run-jobs 4 --jobs 8 --timeout 1800
```

### 10.3 修改 S2G backend RTL

最小 gate：

```bash
./run_dma_tma_rtl.sh --suite directed --backend rtl --run-jobs 4 --jobs 8 --timeout 1800
./run_dma_tma_rtl.sh --suite directed --backend rtl-nocache --run-jobs 4 --jobs 8 --timeout 1800
```

性能 guard：

```bash
cd testcases/_get_case/dma_tma_tensor_s2g_pingpong_perf_test
./dma_tma_tensor_s2g_pingpong_perf_test.out sweep manual_tensor_s2g
./dma_tma_tensor_s2g_pingpong_perf_test.out sweep tma_tensor_s2g
```

旧路径 guard：

```bash
cd testcases/_get_case/dma_tma_g2s_pingpong_perf_test
./dma_tma_g2s_pingpong_perf_test.out sweep
```

### 10.4 修改 swizzle/interleave/stride/OOB

必须跑 feature perf：

```bash
cd testcases/_get_case/dma_tma_s2g_feature_perf_test
./dma_tma_s2g_feature_perf_test.out sweep 8
./dma_tma_s2g_feature_perf_test.out stress 8
```

再跑 S2G functional：

```bash
cd testcases/_get_case
./run_dma_tma_rtl.sh --suite directed --backend rtl --run-jobs 4 --jobs 8 --timeout 1800
./run_dma_tma_rtl.sh --suite directed --backend rtl-nocache --run-jobs 4 --jobs 8 --timeout 1800
```

### 10.5 发布前 full registry

```bash
cd testcases/_get_case
./run_dma_tma_rtl.sh --suite all --backend gvm --run-jobs 4 --jobs 8 --timeout 7200
./run_dma_tma_rtl.sh --suite all --backend gvm-nocache --run-jobs 4 --jobs 8 --timeout 7200
./run_dma_tma_rtl.sh --suite all --backend rtl --run-jobs 4 --jobs 8 --timeout 7200
./run_dma_tma_rtl.sh --suite all --backend rtl-nocache --run-jobs 4 --jobs 8 --timeout 7200
```

## 11. 失败分析模板

记录失败时建议包含：

| 字段 | 内容 |
|---|---|
| 后端 | gvm/gvm-nocache/rtl/rtl-nocache |
| 命令 | 完整命令 |
| case | registry label 或子 case name |
| 结果 | PASS/FAIL/timeout |
| 用时 | 秒 |
| mismatch | mismatch 数、最大误差、首个错误位置 |
| 相关 PMU | PutFull/PutPart/stall/ack/TLB |
| 判断 | host/reference、kernel、RTL route、backend resource、fence 中哪一类 |

不要直接引用大日志。只摘关键行和搜索结果。
