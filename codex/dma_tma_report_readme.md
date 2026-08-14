# DMA/TMA 设计与验证导读

更新时间：2026-07-03

本文是当前 DMA/TMA 文档集的入口。它不替代主报告，而是把“怎么读、怎么构建、怎么运行、每份文档负责什么”先讲清楚，便于后续继续推进 RTL 设计和测试覆盖。

## 1. 文档结构

| 文档 | 重点 | 适合什么时候读 |
|---|---|---|
| `codex/dma_s2g_design_report.md` | S2G 项目主报告，保留当前结论、历史关键决策、最新性能结果、后续计划 | 想看项目总状态、性能结论、下一步优先级 |
| `codex/dma_tma_rtl_design_guide.md` | RTL 机制导读，覆盖 `DMA_core`、bulk S2G、tensor S2G、group fence、PMU 和面积约束 | 准备改 Chisel RTL 前 |
| `codex/dma_tma_validation_guide.md` | 验证方法导读，覆盖构建、后端选择、registry、functional/perf/feature 运行方式 | 准备跑测试或判断改动有没有破坏旧路径前 |
| `codex/dma_tma_test_catalog.md` | 当前 `testcases/_get_case` 下 DMA/TMA 相关测试目录和 case 含义索引 | 准备补测试或清理测试命名前 |

建议阅读顺序：

1. 先读本导读，确认构建与测试入口。
2. 再读 `dma_tma_rtl_design_guide.md`，建立 RTL 的数据通路和控制通路模型。
3. 然后读 `dma_tma_validation_guide.md`，理解为什么要分 GVM/RTL/cache/nocache。
4. 最后按需要查 `dma_tma_test_catalog.md` 和主报告中的最新结果。

## 2. 项目入口

当前 DMA/TMA RTL 主要位于：

| 路径 | 作用 |
|---|---|
| `gpgpu/ventus/src/pipeline/DMA_core.scala` | DMA 顶层调度、G2S/S2G/Tensor 模块连接、TLB/L2/shared response/completion 路由 |
| `gpgpu/ventus/src/pipeline/DMA_s2g.scala` | bulk S2G 后端，同时承接 tensor S2G line task；负责 shared read、DMA L1TLB translation、L2 Put、ack 和 S2G completion |
| `gpgpu/ventus/src/pipeline/DMA_tma_s2g.scala` | tensor S2G 前端；负责 descriptor fetch/cache、tensor address setup、feature 解析、fast/fallback line task 生成 |
| `gpgpu/ventus/src/pipeline/warp_schedule.scala` | DMA fence、S2G group commit/wait、per-warp group counter |
| `gpgpu/ventus/src/pipeline/pipe.scala` | 指令 issue 时把 `dma_group`、S2G 类型等信息带给 DMA core |
| `gpgpu/ventus/src/pipeline/PerfCounters.scala` | pipeline 和 S2G 相关 PMU 计数结构 |
| `gpgpu/ventus/src/top/parameters.scala` | DMA/TMA 相关硬件参数，包括 line entries、shared read entries、descriptor cache entries、group entries |

当前测试主要位于：

| 路径 | 作用 |
|---|---|
| `testcases/_get_case/cases_dma_tma.csv` | DMA/TMA 默认回归 registry |
| `testcases/_get_case/run_dma_tma_rtl.sh` | DMA/TMA 统一运行脚本，支持 GVM、GVM-nocache、RTL、RTL-nocache |
| `testcases/_get_case/dma_tma_g2s_func_test` | G2S bulk/TMA 功能测试 |
| `testcases/_get_case/dma_tma_s2g_func_test` | S2G bulk/tensor/mixed routing 功能测试 |
| `testcases/_get_case/dma_tma_multi_wg_func_test` | 多 workgroup 下 G2S/S2G/TMA 隔离与并发测试 |
| `testcases/_get_case/dma_tma_g2s_pingpong_perf_test` | G2S pingpong overlap 性能测试 |
| `testcases/_get_case/dma_tma_tensor_s2g_pingpong_perf_test` | tensor S2G pingpong 性能测试，含 tensor S2G 主性能路线 |
| `testcases/_get_case/dma_tma_s2g_feature_perf_test` | swizzle/interleave/stride/OOB/high-rank 等 feature 诊断性能测试 |
| `testcases/_get_case/dma_tma_movement_profile_test` | DMA/TMA movement profile，用于比较 bulk/tensor G2S/S2G 单次搬运 cycle |

## 3. 环境与构建

进入项目后先设置环境：

```bash
source env.sh
```

完整构建入口是：

```bash
./build-ventus.sh --build "rtlsim;gvm;driver;pocl"
```

只重建 RTL/GVM 仿真器时，关注 `build-ventus.sh` 内部的这些路径：

```text
gpgpu/sim-verilator
gpgpu/sim-verilator-nocache
```

脚本会分别构建 cache 和 nocache RTL，并在同一组目录中安装 GVM 后端。RTL 构建时间较长，之前经验是不要频繁等待，可以每 5 分钟左右查看一次构建状态。

OpenCL kernel 编译使用 `install/bin/clang`，项目规则中给出的基本形式是：

```bash
./install/bin/clang -cl-std=CL2.0 -target riscv32 -mcpu=ventus-gpgpu \
  kernel.cl -o kernel.riscv -nodefaultlibs \
  -Wl,${VENTUS_ENV_PATH}/install/lib/crt0.o \
  -Wl,${VENTUS_ENV_PATH}/install/lib/riscv32clc.o \
  -Wl,--gc-sections -L${VENTUS_ENV_PATH}/install/lib -lworkitem \
  -I${VENTUS_ENV_PATH}/installinclude/clc -O1 \
  -Wl,-T,${VENTUS_ENV_PATH}/install/lib/ldscripts/ventus/elf32lriscv.ld \
  -Wl,--init=${KERNEL_FUNC_NAME} -w \
  -D__opencl_c_generic_address_space=1 \
  -D__opencl_c_named_address_space_builtins=1 \
  -D__OPENCL_VERSION__=200
```

反汇编入口：

```bash
./install/bin/llvm-objdump -d --mattr=+v,+zfinx kernel.riscv > kernel.dump
```

## 4. 后端与验证口径

当前 DMA/TMA 验证需要同时看四类后端：

| 后端 | 主要价值 | 注意事项 |
|---|---|---|
| `gvm` | 快速功能参考，适合判断 host/kernel/reference 逻辑是否合理 | 不代表 RTL 时序和硬件资源竞争 |
| `gvm-nocache` | 快速检查 nocache 配置下的软件路径 | 同样不代表 RTL 微结构 |
| `rtl` | cache RTL 真正实现路径，必须作为设计验收主后端 | 日志大、运行慢，性能数据以这里为主 |
| `rtl-nocache` | nocache RTL 实现路径，主要检查地址、TLB/L2 路由和 completion 是否对 cache 假设敏感 | 有些性能结果只作为稳定性参考 |

统一运行脚本示例：

```bash
cd testcases/_get_case
./run_dma_tma_rtl.sh --suite directed --backend rtl --run-jobs 4 --jobs 8 --timeout 1800
./run_dma_tma_rtl.sh --suite all --backend rtl --run-jobs 4 --jobs 8 --timeout 7200
```

当前建议并发策略：

| 参数 | 推荐值 | 原因 |
|---|---:|---|
| `--jobs` | 8 | 单个测试编译/运行大约按 8 核设计 |
| `--run-jobs` | 4 | 当前服务器可承受约 32 核并发，不让默认 functional gate 过慢 |
| `--timeout` directed | 1800 | 功能测试足够 |
| `--timeout` all/perf | 7200 | 全量性能 sweep 和 RTL 后端更稳妥 |

## 5. 当前设计主线

当前 S2G 已经从占位符演进为两级结构：

```text
bulk S2G instruction
  -> DmaS2G backend
  -> shared read / DMA L1TLB / L2 Put / AccessAck / completion

tensor S2G instruction
  -> DmaTensorS2G descriptor and address setup
  -> S2GLineTask
  -> DmaS2G backend
  -> shared read / DMA L1TLB / L2 Put / AccessAck / completion
```

这个结构的核心好处是：

1. bulk 和 tensor 写回共享高性能后端，不复制大规模 line table 或 ack table。
2. tensor 前端可以专心处理 descriptor、rank、stride、swizzle、interleave、OOB 等语义。
3. completion/fence 统一用 `{wid, group, is_s2g}`，便于做 CUDA-style group wait。
4. 面积约束可控，主要资源是少量 table、word-banked line image 和 2-entry descriptor cache，而不是按 tile/stage 放大；S2G translation 直接复用 DMA L1TLB。

## 6. 当前验证主线

功能测试按“语义正确性”分层：

1. G2S 单 WG：确认旧有 bulk/TMA G2S 没被 S2G 改动破坏。
2. S2G 单 WG：覆盖 bulk S2G、tensor S2G、mixed routing、group wait、fixed-seed fuzz。
3. Multi-WG：覆盖多 workgroup 下的地址空间、group、descriptor、page/TLB route 隔离。

性能测试按“性能问题来源”分层：

1. G2S pingpong：旧路径性能基准和 overlap 参考。
2. tensor S2G pingpong：当前 S2G 主性能 gate，特别是 tensor S2G writeback。
3. S2G feature perf：swizzle/interleave/stride/OOB/high-rank 的同参数 paired A/B 诊断。
4. DMA/TMA movement profile：纯 movement microbench，辅助定位 bulk/tensor G2S/S2G 单次搬运固定开销。

这个分层背后的判断是：默认回归不追求覆盖所有排列组合，而是用少量长期稳定测试护住语义；feature 和 stress 用于定位高级特性是否出现极端性能角落。

## 7. 日志和数据注意事项

RTL、rtl-nocache、gvm 会直接把日志输出到 stdout。`VENTUS_SPIKE_LOG=1` 时 spike 会在当前目录输出日志文件。日志通常非常长，处理时应遵守：

1. 不直接整文件读入上下文。
2. 搜索日志时限制输出，例如用 `rg -n "FAIL|mismatch|PASS|S2G" log_file | head` 这类方式。
3. 性能表优先从测试程序输出的 summary 或 markdown report 提取。
4. 每次刷新性能结果时记录后端、命令、用时、points、min/avg/max 和是否 clean build。

## 8. 后续维护方式

后续文档更新建议：

1. 主报告 `dma_s2g_design_report.md` 只记录最新状态、关键结果和设计决策。
2. RTL 细节变动优先更新 `dma_tma_rtl_design_guide.md`。
3. 新增或重命名测试时优先更新 `dma_tma_test_catalog.md`。
4. 运行方式、gate 策略或并发参数变化时更新 `dma_tma_validation_guide.md`。
5. 性能结果如果只是当前状态参考，明确写“当前快照”，不要当最终门限。
