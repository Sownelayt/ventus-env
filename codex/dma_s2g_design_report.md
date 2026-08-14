# DMA S2G RTL 设计与测试报告

更新时间：2026-07-04
范围：`shared memory -> global memory` 的 bulk S2G 与 tensor S2G 路线。本文档记录当前 RTL 实现、测试状态、性能快照、已知约束、功能测试覆盖状态和后续 RTL 设计重点。

## 文档拆分导读

随着 S2G 从占位符实现推进到 bulk/tensor 共用高性能后端、group fence、高级 feature 诊断和四后端回归，单一主报告已经偏长。当前文档集按“主报告 + 专题导读”组织：

| 文档 | 内容定位 |
|---|---|
| `codex/dma_tma_report_readme.md` | DMA/TMA 文档入口，包含项目入口、构建命令、后端选择和阅读顺序 |
| `codex/dma_tma_rtl_design_guide.md` | RTL 设计导读，重点说明 `DMA_core`、`DmaS2G`、`DmaTensorS2G`、group fence、PMU 和面积/时序边界 |
| `codex/dma_tma_validation_guide.md` | 验证方法导读，说明如何构建、如何跑 GVM/RTL/cache/nocache、如何解读功能和性能结果 |
| `codex/dma_tma_test_catalog.md` | 当前 `testcases/_get_case` 中 DMA/TMA 测试目录、suite、case 和维护策略索引 |
| `codex/dma_s2g_design_report.md` | 本主报告，保留当前结论、最新测试结果、关键设计决策和后续优先级 |

推荐读法：新读者先读 `dma_tma_report_readme.md`，准备改 RTL 时读 `dma_tma_rtl_design_guide.md`，准备跑测试或补 case 时读 `dma_tma_validation_guide.md` 和 `dma_tma_test_catalog.md`。本文件继续作为项目主记录，优先保存最新状态和关键结论。

## 0. 阅读方式

这份文档按渐进式披露组织：

| 层级 | 建议读法 | 内容 |
|---|---|---|
| 第一层 | 先读 1 到 4 节 | 当前结论、最新验证、代码入口、下一步判断 |
| 第二层 | 读 5 到 11 节 | RTL 机制、测试体系、性能解释、G2S/S2G 对比、约束、功能覆盖计划 |
| 第三层 | 展开附录 | CUDA 参数对齐、命令、详细测试矩阵、后续硬件优化草案 |

后续继续推进时，优先更新 1、2、4、7、9、11 节；默认只保留最新有效结果。

## 1. 当前结论

S2G 路线已经不是占位符实现。bulk S2G 已经是 line-based、多 outstanding 的 write pipeline；tensor S2G 的 swizzle/interleave/stride/OOB 和普通 no-permute row 都可以通过 `S2GLineTask` 接入 bulk S2G backend。当前删除的是 multi-row linear span fusion、high-rank flatten 和纯 linear 专用 setup/FSM，而不是通用 row-level coalescing。

最新代码关键状态：

| 项目 | 当前状态 |
|---|---|
| bulk S2G | `DmaS2G` 内部有 instruction table、line table、dynamic shared-read table、ack table、direct DMA L1TLB translation；不再为无同步 same-line overlap 提供硬件顺序兜底 |
| tensor S2G | `DmaTensorS2G` 保留 descriptor/setup 前端；swizzle/interleave/stride/OOB 和 ordinary no-permute row-level line-task 由 `DmaS2G` 统一执行 shared read、TLB、L2 Put、ack completion；不再保留跨多行 linear span fusion |
| completion/fence | DMA completion token 带 `{wid, group, is_s2g}`；warp scheduler 维护 per-warp all-DMA group ring，G2S/S2G/prefetch 都进入同一 `group_count[g]`；`is_s2g` 保留为调试/统计来源，不再定义独立 wait 域 |
| descriptor cache | tensor S2G 有 2-entry descriptor-line cache；descriptor hot-update 目前作为软件编程约束处理 |
| perf counters | bulk/tensor S2G 统一汇总 PutFull/PutPart、line、bytes、shared req/rsp、TLB req、ack count；ack latency 与 line/read/ack full stall 属于 `PMU_TMA_DETAIL` 诊断项，当前参数打开 |
| 最新修复 | 删除 S2G L0 page-cache 后统一直接依赖 DMA L1TLB；删除 same destination line hazard 结构；删除纯 linear tensor span/flatten 专用路径但保留 row-level line-task；G2S/S2G 同步统一到 per-warp DMA group ring；详细 S2G PMU 改为参数化 |
| 最新测试补强 | S2G functional 已扩到 tensor matrix、bulk matrix、bulk backend stress、mixed routing；无 wait 的 same-line overlap 不再作为合法 directed case，编程模型要求软件用 wait/barrier 同步重叠写 |

最新测试基线：

| 类型 | 后端 | 结果 |
|---|---|---|
| RTL build | cache/nocache | `source env.sh; ./build-ventus.sh --build "rtlsim"` PASS，安装 `libVentusRTL-withcache.so` 和 `libVentusRTL-nocache.so` |
| directed gate | GVM | `./run_dma_tma_rtl.sh --suite directed --backend gvm --run-jobs 4 --jobs 8 --timeout 1800`，3/3 PASS，约 63s |
| directed gate | GVM-nocache | `./run_dma_tma_rtl.sh --suite directed --backend gvm-nocache --run-jobs 4 --jobs 8 --timeout 1800`，3/3 PASS，约 47s |
| directed gate | RTL | `./run_dma_tma_rtl.sh --suite directed --backend rtl --run-jobs 4 --jobs 8 --timeout 1800`，3/3 PASS，约 52s |
| directed gate | RTL-nocache | `./run_dma_tma_rtl.sh --suite directed --backend rtl-nocache --run-jobs 4 --jobs 8 --timeout 1800`，3/3 PASS，约 48s |
| G2S pingpong full sweep | RTL | 25/25 PASS，243s，min/avg/max = 1.1214x/1.3703x/1.4998x |
| manual input + tensor S2G full sweep | RTL | 25/25 PASS，239s，min/avg/max = 1.0647x/1.2493x/1.3601x |
| TMA G2S input + tensor S2G full sweep | RTL | 25/25 PASS，258s，min/avg/max = 1.2068x/1.9379x/2.4807x |
| DMA/TMA movement profile | GVM/RTL/GVM-nocache/RTL-nocache | standalone sweep 均 PASS；RTL 40 个 child PASS，bulk/tensor G2S 与 bulk/tensor S2G 均按同 tile baseline 做 PMU cycle A/B |
| feature diagnostic | RTL | `dma_tma_s2g_feature_perf_test sweep 8` PASS，17s；PMU total active cycles = 154744；paired host ratio 范围约 0.973x 到 1.144x |

movement profile 当前 standalone sweep 摘要：

| backend | mode | min | avg | max |
|---|---|---:|---:|---:|
| GVM | bulk_g2s | 1.6636x | 3.8040x | 7.1760x |
| GVM | tensor_g2s | 1.3598x | 3.4628x | 6.8179x |
| GVM | bulk_s2g | 1.1535x | 1.2155x | 1.2742x |
| GVM | tensor_s2g | 0.1953x | 0.2168x | 0.2608x |
| RTL | bulk_g2s | 1.6402x | 3.8073x | 7.1862x |
| RTL | tensor_g2s | 1.3628x | 3.4645x | 6.8325x |
| RTL | bulk_s2g | 1.2336x | 1.3412x | 1.4336x |
| RTL | tensor_s2g | 1.1374x | 1.2920x | 1.4237x |
| GVM-nocache | bulk_g2s | 1.5297x | 3.1849x | 5.7705x |
| GVM-nocache | tensor_g2s | 1.3201x | 2.9343x | 5.4830x |
| GVM-nocache | bulk_s2g | 1.2349x | 1.3250x | 1.4038x |
| GVM-nocache | tensor_s2g | 0.3542x | 0.3805x | 0.4319x |
| RTL-nocache | bulk_g2s | 1.5342x | 3.1916x | 5.7793x |
| RTL-nocache | tensor_g2s | 1.3235x | 2.9399x | 5.4910x |
| RTL-nocache | bulk_s2g | 1.2648x | 1.3734x | 1.4649x |
| RTL-nocache | tensor_s2g | 1.1807x | 1.3325x | 1.4564x |

GVM/GVM-nocache 的 tensor_s2g cycle 与 RTL 明显不一致；当前把它作为 GVM 模型诊断差异记录，RTL/rtl-nocache 才是该 profile 的硬件性能判断口径。

当前代码已经删除 linear span/flatten 专用路径、same-line hazard，保留 no-permute row-level line-task，并把 G2S/S2G 同步统一为 all-DMA group ring。源码残留检索、`diff --check`、RTL cache/nocache 重建、四后端 directed gate、RTL perf registry 和 RTL feature diagnostic 均已通过。

当前文档对应的测试体系版本：

| 项 | 最新状态 |
|---|---|
| 默认 registry | `cases_dma_tma.csv` 当前保留 8 个 DMA/TMA mode，其中新增 1 个高级 feature diagnostic perf |
| 已删除目录 | `tma_roundtrip_pipeline_perf_test`, `dma_tma_s2g_pipeline_perf_test` |
| 已改名/重定义目录 | `dma_tma_g2s_pingpong_perf_test`, `dma_tma_tensor_s2g_pingpong_perf_test`, `dma_tma_movement_profile_test` |
| registry 外诊断 | `tma_gemm_perf_test` 源码保留，但不再进入默认 DMA/TMA 回归；`dma_tma_s2g_feature_perf_test stress` 作为手动 feature worst-case 扩展诊断，不进入默认 registry |
| 最新全量结果 | 2026-07-04 已完成四后端 directed gate；RTL `--suite perf` 5/5 PASS；RTL feature diagnostic PASS。四后端 `--suite all` 可作为发布前长回归 |

这些性能数字是当前代码快照，不是最终门限。后续 RTL/FSM 或测试内容继续改动后，需要重新运行 full sweep。

## 2. 最新 RTL 变化

### 2.1 Tensor Line Task 接入 Bulk Backend

`DmaTensorS2G` 现在不再在 fast path 内自己完成整套 shared read、TLB、Put、ack。它对 CUDA-style FP32 tensor tile 做 descriptor/setup/addrgen，然后输出 `S2GLineTask`：

```text
DmaTensorS2G descriptor/setup/line addrgen
  -> S2GLineTask {
       wid, group, asid, src, dst, bytes,
       dstWordStride,
       swizzleMode, swizzleBase, swizzleRow,
       first, last
     }
  -> DmaS2G line/read/TLB/Put/ack backend
  -> DMA completion {wid, group, is_s2g=true}
```

涉及文件：

| 文件 | 作用 |
|---|---|
| `gpgpu/ventus/src/pipeline/DMA_s2g.scala` | 定义 `S2GLineTask`；bulk backend 支持外部 tensor line task；shared read issue 可按 swizzle metadata 做 per-lane source gather，并可用 `dstWordStride` 表示 packed shared -> sparse global word mask |
| `gpgpu/ventus/src/pipeline/DMA_tma_s2g.scala` | tensor descriptor/setup 前端；no-permute row、swizzle/interleave、规则 stride、OOB clipping 等可进入 line-task 路径；复杂非法组合走 fallback；不再做 multi-row linear span fusion |
| `gpgpu/ventus/src/pipeline/DMA_core.scala` | `dmaS2G.io.line_task <> dmaTensorS2G.io.line_task`，并汇总 S2G PMU |

### 2.2 Advanced Feature Fast Path

swizzle/interleave 已经从“只能功能正确的 fallback”提升为常见 FP32 场景下的 coalesced writeback 路径。规则 element stride、OOB suppress 和普通 no-permute row 都通过 line task 复用 `DmaS2G` backend。plain linear tensor 只是不再拥有跨多行 span fusion 或 high-rank contiguous flatten 这种独立硬件主线。

当前 fast path 条件：

| 条件 | 说明 |
|---|---|
| data width | FP32/4B，和 bulk aligned S2G 保持一致 |
| rank/shape | rank 可以大于 2；plain linear/high-rank contiguous 不再做专用 flatten fast path |
| swizzle | `swizzleMode` 可为 32B/64B/128B；tensor 前端限制 line task 不跨 swizzle span，backend 对每个 active lane 计算 swizzled shared source |
| interleave | interleave16/32 按 4 或 8 个 FP32 element 的 slice 发段，每段保持 destination coalesced；跨 slice 时更新 interleave global offset |
| element stride | dim0 stride 使用 `dstWordStride` 生成 sparse destination word mask；dim1+ stride 保留 row/plane 循环，但每个有效 row 进入 line task |
| OOB suppress | setup 阶段同时保存 raw output layout 和 clipped valid output dim；shared tile 地址按 raw layout 走，global 写回只发 clipped in-bounds 子区间 |
| fallback | 非 4B data width、复杂 swizzle+OOB、非法 interleave 或当前未覆盖的不规则组合仍走保守 fallback |

swizzle 的关键点是 destination line 仍然连续，慢的原因原本是 shared source 变成 permutation 后无法用普通 `{src,dst,bytes}` 表示。现在 `S2GLineTask` 携带 `swizzleMode/swizzleBase/swizzleRow`，`DmaS2G` 在 shared read issue 时对每个 lane 做：

```text
logical shared addr = src + lane * 4
physical shared addr = swizzleSharedAddr(logical, swizzleBase, swizzleMode, swizzleRow)
destination word = dst line word + lane
```

这样 line image 仍由 bulk backend 聚合，L2 仍能发 PutFull/PutPart；面积上只增加少量 line-entry metadata 和 per-lane address xor/mux，不增加随 tile/stage 增长的大 buffer。

interleave 的关键点是 destination 地址映射会按 16B/32B slice 跳转，不能简单把整行视作一个连续 global span。当前实现采用小步保守方案：tensor line emitter 每次只覆盖一个 interleave slice，slice 内仍是连续 16B 或 32B coalesced line task；跨 slice 时用 `interleaveDim0DeltaBytes()` 更新 global offset。这个方案没有引入多 outstanding destination-line gather table，因此面积风险低，但比理想的跨 slice line coalescer 仍保守。

stride/OOB 的关键点是不能为了“规则但不连续”的 tensor 直接退回每元素 PutPart。当前实现通过 `S2GLineTask.dstWordStride` 让 backend 在同一 destination line 内按 `dstStartWord + lane * stride` 写稀疏 mask；tensor setup 同时保留 raw output dim 和 clipped output dim，避免 OOB clipping 错改 shared tile row stride。plain linear/high-rank contiguous flatten 已删除。

### 2.3 最新死锁修复

发现的问题：

```text
tensor fast path 为一条 tensor S2G 创建 synthetic instruction slot
bulk addrgen 同时把这个 slot 当成普通 bulk S2G
bulk addrgen 使用 size=0/offset=0 生成异常 zero-byte line
zero-byte line 不产生有效 ack
fence/group wait 无法完成
```

修复方式：

| 机制 | 说明 |
|---|---|
| `instExternal` | 标记 instruction slot 来自 tensor line task |
| bulk addrgen 过滤 | `genInstVec` 只选择 `!instExternal(i)` 的普通 bulk instruction |
| tensor line task 生命周期 | `first` 时分配 synthetic instruction，`last` 时关闭 `extInstActive`，ack completion 后清除 `instExternal` |
| assertions | line task 禁止 `bytes=0`，并要求 src/dst/bytes 4B 对齐 |

这个修复不增加大 data buffer，只增加 instruction metadata bit，对面积影响很小。

### 2.4 Descriptor Cache 测试约束

tensor directed 测试固定了 descriptor cache 的软件契约：host 不应在多个 kernel case 间复用同一 descriptor metadata line 并修改内容。RTL tensor descriptor cache 认为同一 descriptor line 可以复用；如果软件 hot-update 同一 line，就可能命中缓存中的 descriptor。

当前测试通过使用独立 descriptor buffer 或 buffer 内独立 word offset 规避 hot-update：stride/OOB 使用 3072B 偏移，descriptor-mix 使用 1024B 偏移，group-page 使用 2048B 偏移。这个处理是在测试层遵守“不要 hot-update 同一 descriptor metadata line”的编程约束，不代表 RTL 需要为每个 kernel 自动 flush descriptor cache。

当前处理方式：

| 层面 | 处理 |
|---|---|
| RTL | 保持 descriptor cache，不为这个测试场景增加硬件 invalidation |
| 测试 | 每个 directed case 避免复用同一 descriptor line；multi-WG subbox 用真实 guard kernel arg 强制 descriptor 落到新 device vaddr；新增 tensor multi-WG stress 使用不同 descriptor buffer 内偏移 |
| 编程约束 | measured kernel 不在 hot loop 内修改 tensor descriptor metadata line |
| OpenCL source 约束 | RTL 后端下 multi-kernel `.cl` 容易选错入口；directed case 中每个 measured kernel 尽量使用单入口 `.cl` |

这个约束后续可以通过显式 descriptor cache invalidate/prefetch 语义解决，但它不是当前 S2G 数据通路优化的优先项。

### 2.5 FULL-only All-DMA Group Fence 固化

当前 `warp_schedule` 中的 group 机制已经从 S2G-only 计数扩展成 per-warp all-DMA group ring。所有 DMA 指令，包括 G2S、S2G 和 prefetch，issue 后都会归入当前 open group；`commit_group` 只负责封口并推进 issue pointer，不负责启动 DMA 执行。

| 机制 | 当前实现 |
|---|---|
| group 数量 | `dma_group_entries = 4` |
| group counter | `dma_group_count(num_warp, dma_group_entries)` |
| counter 位宽 | `dmaInflightWidth = log2Ceil(max_dma_inst + 1)`；当前 8 warp 配置下为 4 bit |
| issue 归属 | `pipe.scala` 从 scheduler 取 `dma_issue_group` 写入 DMA request |
| 递增 | 任意 DMA issue 时对当前 warp 的 open group 加一 |
| 递减 | 任意 DMA completion 返回后按 `{wid, group}` 减一；S2G completion 仍必须等 L2 Put `AccessAck` |
| wait 语义 | `wait_group N` 只等待超过最近 N 个 committed group 的 FULL completion |
| ring 复用 | `issue_ptr` 指向的 group 若仍 committed 且未完成，则阻塞新的 DMA issue，直到该 slot 清零 |

当前明确不新增 READ-phase completion，也不扩展 `DmaCompletion` 的 phase 字段。`wait_group.read` 暂作为后续高级特性保留；如果未来实现，需要额外的 read-phase outstanding 状态和对应 PMU，但不应混入当前 FULL fence 语义。

### 2.6 Direct DMA L1TLB Translation

S2G destination 是 global address，所有 destination line translation 统一走 DMA 专用 8-entry L1TLB。当前 RTL 不在 S2G 内部保存额外 VPN/PPN 翻译状态，避免和 DMA L1TLB 出现重复 owner、response route 或一致性语义。

| 模块 | 当前实现 |
|---|---|
| `DmaS2G` | 每条 destination line 在 `linePaddrValid=false && lineTlbReq=false` 时直接向 DMA L1TLB 发请求；TLB response 写回该 line 的 physical line base |
| `DmaTensorS2G` | descriptor TLB 与 legacy data TLB 都直接走外部 DMA L1TLB；fast path 通过 `S2GLineTask` 进入 `DmaS2G` 后同样使用统一 line TLB |
| PMU | `tlbReq` 计数真实 TLB request；不再提供额外 page-local translation counter |
| 测试 | `dst_cross_page`、`cross_page_tlb_ABC` 和 multi-WG cross-page cases 覆盖 4KB 边界、TLB owner 与 response route |

验证结论：四后端 directed gate 均通过。RTL `dma_tma_s2g_feature_perf_test sweep 8` 代表点里 `tlb req` 与 line issue 数一致；`manual_tensor_s2g` 和 `tma_tensor_s2g` 两条 S2G writeback full sweep 仍维持当前性能趋势。

## 3. 当前代码入口

主要 RTL 文件：

| 文件 | 当前重点 |
|---|---|
| `gpgpu/ventus/src/pipeline/DMA_s2g.scala` | bulk S2G backend、`S2GLineTask`、line table、dynamic shared read、swizzled shared gather、direct DMA L1TLB translation、ack completion、S2G PMU |
| `gpgpu/ventus/src/pipeline/DMA_tma_s2g.scala` | tensor descriptor/setup、linear/swizzle/interleave line-task fast path、descriptor cache、fallback element path |
| `gpgpu/ventus/src/pipeline/DMA_core.scala` | DMA path routing、completion arbiter、S2G PMU aggregation、line task connection |
| `gpgpu/ventus/src/pipeline/warp_schedule.scala` | per-warp all-DMA group ring、legacy all-DMA wait/count、commit_group/wait_group |
| `gpgpu/ventus/src/pipeline/pipe.scala` | issue 侧传递 DMA direction/group 信息 |
| `gpgpu/ventus/src/pipeline/PerfCounters.scala` | S2G counters 与 pipeline stall counter |
| `gpgpu/ventus/src/top/parameters.scala` | S2G line/read/descriptor/group 等资源参数 |

主要测试目录：

| 目录 | 当前用途 |
|---|---|
| `testcases/_get_case/dma_tma_g2s_func_test/` | G2S/TMA directed、matrix、descriptor、routing 和 mixed async gate |
| `testcases/_get_case/dma_tma_s2g_func_test/` | bulk/tensor S2G directed 和 stress |
| `testcases/_get_case/dma_tma_multi_wg_func_test/` | 多 WG G2S/S2G 功能 gate |
| `testcases/_get_case/dma_tma_g2s_pingpong_perf_test/` | G2S pingpong perf，manual writeback 对照 |
| `testcases/_get_case/dma_tma_tensor_s2g_pingpong_perf_test/` | tensor S2G pingpong perf，含 `manual_tensor_s2g`、`tma_tensor_s2g` |
| `testcases/_get_case/dma_tma_movement_profile_test/` | DMA/TMA movement profile；bulk/tensor G2S/S2G 单次搬运 PMU cycle A/B |
| `testcases/_get_case/dma_tma_s2g_feature_perf_test/` | tensor S2G 高级 feature diagnostic perf，覆盖 no-feature baseline、interleave16/32、swizzle32/64/128、interleave32+swizzle32 组合、padded global stride、element stride2、subbox misalignment、OOB suppress、rank3/4/5 fallback |

构建和环境：

```bash
source env.sh
./build-ventus.sh --build "rtlsim"
```

## 4. 下一步最应该做什么

当前测试体系整理和 S2G functional coverage 已经完成到可作为回归护栏的程度。mixed completion/routing、group/page 边界、multi-WG S2G 压力、deterministic mask/offset、tensor long-shape/datatype alias、4WG group/page、4WG mixed outstanding、tensor multi-WG stride/OOB、descriptor mix、小规模 fixed-seed fuzz 和更长 `G2S + bulk S2G + tensor S2G` mixed outstanding 都已经进入 directed gate。扩展后的 `directed` 功能 gate 在 GVM/GVM-nocache/RTL/RTL-nocache 四后端均通过。

当前已经把 CUDA-style bulk group 的 FULL wait 子集固定下来，而不是马上实现 `wait_group.read`。现有 `warp_schedule` 已经按每 warp、每 group 维护 all-DMA FULL outstanding counter；在当前 `num_warp=8`、`dma_group_entries=4`、`dmaInflightWidth=4` 下，对应 32 个 4-bit group counter。这个代价很小，且与 `max_dma_inst` 参数绑定，不写死为 3-bit。

当前实现边界：

| 项 | 决策 |
|---|---|
| `commit_group` | 将当前 warp 的当前 DMA group 封口；有 outstanding work 时标记 committed 并推进 issue group |
| `wait_group N` | 只等待超过最近 N 个 committed group 的 FULL completion |
| FULL completion | G2S 以 shared tile 可用为完成；S2G 以 L2 Put `AccessAck` 全部返回为完成；二者统一递减同一 group counter |
| READ completion | 暂不实现；`wait_group.read` 作为 Phase 2 高级特性保留 |
| 面积边界 | 不新增 line buffer/descriptor cache，只使用 scheduler 小计数器和已有 completion token |

高级特性性能诊断已经补上。`dma_tma_s2g_feature_perf_test` 的结论必须按 paired A/B 读取；RTL PMU active cycles 是后续更精确的优化依据，host paired ratio 只作为轻量趋势。最新 RTL `sweep 8` 中，feature diagnostic PASS，PMU total active cycles 为 154744，paired host ratio 范围约 0.973x 到 1.144x，未出现数量级退化。

### 4.1 测试体系整理决策

当前性能测试按“测什么路径”组织，而不是按早期目录来源组织。

当前测试目录决策：

| 项 | 处理 | 原因 |
|---|---|---|
| `tma_roundtrip_pipeline_perf_test` | 物理删除 | 与 pingpong 的端到端主语义重复；buffer sweep 对当前 tensor S2G 路径没有足够独立价值 |
| `dma_tma_s2g_pipeline_perf_test` | 物理删除 | bulk perf 不再保留；tensor micro perf 统一收敛到 tensor S2G pingpong 和 directed 功能覆盖 |
| `tma_gemm_perf_test` | 从 DMA/TMA 默认回归移除 | GEMM 是应用级诊断；当前 manual GEMM 自身失败，不适合作为 DMA/TMA gate |
| `tma_pingpong_pipeline_perf_test` | 改名为 `dma_tma_g2s_pingpong_perf_test` | 明确表示只测 G2S 输入优化，writeback 仍是 manual |
| `dma_tma_s2g_pingpong_perf_test` | 改名为 `dma_tma_tensor_s2g_pingpong_perf_test` | 明确表示测 S2G writeback 优化，可组合 manual/TMA G2S 输入 |
| `dma_tma_serial_pipeline_perf_test` | 重命名并重定义为 `dma_tma_movement_profile_test` | 内容改为纯 movement microbench，不再测 compute/writeback；旧 G2S serial 名称不再保留 |

不保留目录兼容 alias。`cases_dma_tma.csv` 和 `run_dma_tma_rtl.sh` 是当前唯一运行入口。

新的默认测试分层：

| 层级 | 测试 |
|---|---|
| functional gate | `dma_tma_g2s_func_test`, `dma_tma_s2g_func_test`, `dma_tma_multi_wg_func_test` |
| perf trend | `dma_tma_g2s_pingpong_perf_test`, `dma_tma_tensor_s2g_pingpong_perf_test` |
| movement diagnostic profile | `dma_tma_movement_profile_test` |
| feature diagnostic | `dma_tma_s2g_feature_perf_test` |
| future perf | `dma_tma_multi_wg_roundtrip_perf_test`，独立新建，不混入功能测试目录 |

后续 multi-WG roundtrip perf 的方向是每个 workgroup 独立执行 `G2S -> compute -> S2G/manual writeback`，`num_wg` 参数化，用来观察多 WG 并发下的 DMA 仲裁、CTA/WG tag 隔离、completion/fence 域隔离和吞吐 scaling。它不是单 WG buffer sweep 的机械替代，而是更贴近 GPGPU 场景的系统级性能测试。

当前完成度：

| 项 | 状态 | 说明 |
|---|---|---|
| gate 与测试分级 | 已完成 | `cases_dma_tma.csv` 当前是 8 个 mode；功能 gate、perf trend、diagnostic profile、高级 feature diagnostic 已分层 |
| tensor S2G matrix | 已完成第五轮 | 37 个子 case：rank1/2/3/4/5、subbox、padded stride、element stride、interleave16/32、swizzle32/64/128、OOB suppress、rank3/subbox OOB、long shape、datatype alias、descriptor cache A/B/C/A、4 个 fixed-seed dims/coords/box/stride/OOB/swizzle 组合 |
| bulk S2G backend stress | 已完成并按新契约精简 | bulk matrix、fixed mask/offset、deterministic mask/offset、6/8 line pressure、empty commit、wait-oldest/wait-group、group wrap、多 issue同 group后 wrap、跨页/TLB owner、多 warp；无 wait same-line overlap 顺序 case 已删除 |
| mixed completion/routing | 已按统一 group ring 更新 | S2G suite 已有 `s2g_routing_conflict`、`dma_group_keep1_preserves_newer_s2g`、`dma_group_wait0_drains_g2s_s2g`、bulk+tensor same fence、bidirectional descriptor mix、`mixed_tensor_outstanding_long` |
| multi-WG S2G | 已完成第四轮 | `bulk_s2g_4wg`、`tma_s2g_4wg`、2WG group wait、2WG tensor subbox、multi-WG cross-page、4WG group/page、4WG mixed G2S+S2G outstanding、2WG tensor stride/OOB、4WG tensor descriptor mix、4WG tensor group/page 已进入 RTL/rtl-nocache default multi-WG gate |

下一步建议：

1. **继续保持 unified group ring 作为同步主线。**
   当前四后端 directed gate 已确认 `dma_group_keep1_preserves_newer_s2g`、`dma_group_wait0_drains_g2s_s2g` 和 mixed long outstanding 能工作。后续如果调整 wait 编码或 group 数量，应优先复跑这些 case，避免重新引入 G2S/S2G 分域语义。

2. **若继续做高级 feature 性能优化，优先级转向 segmented PutPart 合并。**
   当前默认 feature sweep 没有数量级退化，但 stress 角落仍可能比默认 sweep 更差。根因通常不是 backend 容量，而是 feature path 把一个可合并的 128B destination line 拆成多个 16B/32B segment PutPart。下一刀应考虑小型 segmented line task 或 row-group coalescer，但要严格控制 metadata 和 line image 规模。

3. **保持 FULL-only group fence 作为当前正式语义。**
   当前 `commit_group/wait_group` 等待 all-DMA FULL completion，不实现 `wait_group.read`。READ counter 仍是后续高级 fence 特性，只有当 pingpong/writeback 显示 shared buffer 复用等待再次成为主瓶颈时再进入设计。

4. **每次 RTL 修改后固定回归功能和性能护栏。**
   最小抽样包括 `S2G_TENSOR_CASE_FILTER=fuzz_tensor`、`S2G_MIXED_CASE_FILTER=mixed_tensor_outstanding_long`、S2G full directed、multi-WG directed、四后端 directed gate，以及 RTL 下 G2S pingpong、S2G tensor pingpong、movement profile 和 feature sweep。已有性能趋势不能出现显著下降。

一句话：基础 S2G、tensor fallback、fixed-seed fuzz、mixed long outstanding、4WG group/page、tensor multi-WG stride/OOB/group-page、高级 feature perf diagnostic、FULL all-DMA group fence，以及 swizzle/interleave 第一版 feature line-task 都已经落地。后续重点是保持同步语义稳定，再用 feature stress 或应用场景数据决定是否继续做 swizzle/interleave segmented PutPart 合并。

### 4.2 高级 Feature 性能诊断结论

新增 `dma_tma_s2g_feature_perf_test` 的目的不是替代 pingpong full sweep，而是单独回答“高级 tensor descriptor feature 现在有多贵”。其中 swizzle 和 interleave 是一等目标：swizzle 代表 shared-memory tile layout permutation，interleave 代表 global tensor layout remapping。测试固定为一个 workgroup、单 kernel 多次迭代，host expected 校验实际写回字节；RTL 结论使用 PMU active cycles，不使用 host event ns 做性能判断。

当前该测试有两层入口：`sweep [iterations]` 是默认 registry 使用的稳定诊断集合；`stress [iterations]` 是扩展参数矩阵，用于回答高级 descriptor feature 是否存在远差于默认 sweep 的参数角落，不放入默认全量回归。

测试覆盖：

| case | 主要含义 |
|---|---|
| `linear_rank2_16x16` | FP32 rank2 16x16，当前作为 no-feature fallback/feature baseline，不再代表 linear fast path |
| `linear_rank3_8x8x4` | rank3 但无 swizzle/interleave/stride/OOB，用于看 high-rank fallback setup/addrgen 额外成本 |
| `base_interleave16_rank3_C8_W3_N2` / `interleave16_rank3_C8_W3_N2` | 同 shape/stride/coords，只切换 interleave16 |
| `base_interleave32_rank3_C16_W2_N2` / `interleave32_rank3_C16_W2_N2` | 同 shape/stride/coords，只切换 interleave32 |
| `interleave16_rank3_C6_W3_N2` | interleave16 slice-level coalesced line task，使用已在 functional matrix 中验证过的 rank3 shape |
| `interleave32_rank3_C10_W2` | interleave32 slice-level coalesced line task，使用已验证 shape |
| `base_swizzle32_rank2_8x16` / `swizzle32_rank2_8x16` | CUDA-style swizzle32 paired comparison，inner row 为 32B |
| `base_swizzle32/64/128_rank2_16x16` / `swizzle32/64/128_rank2_16x16` | 显式同 shape paired baseline；swizzle32 16x16 是 32B span 压力形态，swizzle64/128 是合法 inner span 对照 |
| `base_swizzle128_rank2_32x8` / `swizzle128_rank2_32x8` | swizzle128 paired comparison，inner row 为 128B |
| `base_interleave32_swizzle32_rank3_C8_W8_N4` / `interleave32_swizzle32_rank3_C8_W8_N4` | interleave32 + swizzle32 合法组合，inner row 为 32B |
| `base_interleave32_swizzle32_rank3_C16_W4_N2` / `interleave32_swizzle32_rank3_C16_W4_N2` | interleave32 + swizzle32 压力组合，row 跨两个 32B swizzle span |
| `swizzle32/64/128_rank2_16x16` | swizzled shared gather + destination coalesced line task；其中 swizzle32 16x16 是诊断压力 shape，不代表 CUDA-style 最优合法 shape |
| `stride2_rank2_16x16` | 规则 element stride line-task masked PutPart |
| `oob_rank2_16x16_at_8_0` | S2G OOB suppress，逻辑 256 元素、实际写 128 元素 |

2026-07-04 当前 RTL `sweep 8` paired A/B 实测。这里的比较口径是同 shape、同 stride、同 coords、同 iterations，只切换 swizzle 或 interleave；`feature/base` 越接近 1 越好。当前表使用 host paired ratio 作为 quick diagnostic，后续若要做 RTL feature 性能优化，应再打开 PMU 逐 case 抽取 active cycles、PutFull/PutPart、line stall 和 ack latency：

| pair | feature ns/iter | base ns/iter | host feature/base |
|---|---:|---:|---:|
| interleave16 C8_W3_N2 | 76998953.00 | 69806344.25 | 1.103x |
| interleave32 C16_W2_N2 | 74837911.25 | 72994445.62 | 1.025x |
| swizzle32 8x16 | 92595509.12 | 92139854.25 | 1.005x |
| swizzle32 16x16 | 103151285.50 | 90193222.38 | 1.144x |
| swizzle64 16x16 | 89727120.12 | 91489770.75 | 0.981x |
| swizzle128 16x16 | 88038093.12 | 89569901.25 | 0.983x |
| swizzle128 32x8 | 81794179.00 | 82668831.12 | 0.989x |
| interleave32+swizzle32 C8_W8_N4 | 98697752.00 | 98577584.62 | 1.001x |
| interleave32+swizzle32 C16_W4_N2 | 81418175.38 | 83665292.12 | 0.973x |

`stress [iterations]` 入口仍保留，用于覆盖 swizzle32/64/128 在 8x32、32x8、64x4 等不同 row/span 压力形态，interleave16/32 在 C4/C8/C16/C32/C64 slice 压力形态，interleave32+swizzle32 的 N=1 干净组合形态，以及 padded global stride、element stride、misaligned subbox、OOB suppress、rank3/4/5 addressing。stress 不属于当前默认性能结论，只有在需要搜索最坏 feature 参数时再运行。

解释：

1. 正确口径必须是 paired A/B，不能拿 interleave 小 shape 去和 rank2 16x16 linear 直接比，也不能只拿优化前后比值当 feature 代价。
2. 当前默认 sweep 中 swizzle32 16x16 是最差 paired 点，host ratio 约 1.144x；interleave16 C8_W3_N2 约 1.103x。
3. swizzle64/128 默认 paired 点没有退化，当前 host ratio 约 0.981x 到 0.989x。
4. interleave32+swizzle32 组合默认 paired 点约 1.001x/0.973x，说明组合路径至少没有回到逐元素级别。
5. stress 入口仍有必要，因为默认 sweep 不保证覆盖最坏 row/span/slice 排列；下一轮如果要判断“最坏能否到 10x”，必须重跑 stress 并按 PMU active cycles 解析。
6. high-rank contiguous fast flatten 已删除；rank3/4/5 no-feature case 现在作为 row-level line-task/setup 成本监控点，而不是 linear 专用 fast path。
7. 这组测试还暴露了一个测试编程约束：不要在同一进程内释放并复用同一 descriptor device line 来测不同 descriptor feature，否则可能测到 descriptor cache hot-update 行为。当前 feature perf host 通过保留 descriptor buffer 到进程结束来避免这个干扰。

设计判断：

| 方向 | 当前判断 |
|---|---|
| 继续扩大 line buffer | 暂不优先；当前 swizzle/interleave 的第一版 fast path 已经证明主要问题不是 backend 容量 |
| swizzle fast path | 已完成第一版和 CUDA-style paired 覆盖；默认 sweep 最差为 swizzle32 16x16，host ratio 约 1.144x。后续若继续优化，可做 swizzle32 多 row 合并成 128B PutFull |
| interleave coalescer | 已完成 slice-level coalesced line task；默认 sweep interleave16/32 为 1.103x/1.025x。若应用重度依赖 interleave16，应通过 stress 确认最坏形态后再做 segmented row-group 合并 |
| stride/OOB fallback coalescing | 规则 stride line-task 化、OOB row/line clipping 和 masked PutPart 已进入默认功能与 feature 诊断 |
| `wait_group.read` | 仍是高级 fence 特性，但这组数据说明高级 feature fallback 本身也足够慢，优先级不应被忽略 |
| perf gate 策略 | `dma_tma_s2g_feature_perf_test` 作为 diagnostic/profile，记录趋势和退化，不作为主线 speedup 门限 |

#### 4.2.1 Swizzle 与 Interleave 的 RTL 差异

当前 RTL 中两者已经进入 tensor line fast path，但 coalescing 方式不同：

| feature | 当前 RTL 行为 | 为什么慢 | 更合理的高性能方向 |
|---|---|---|---|
| swizzle32/64/128 | `S2GLineTask` 携带 `swizzleMode/swizzleBase/swizzleRow`，`DmaS2G` 在 shared read issue 阶段做 per-lane shared address permutation，destination 仍按 line coalesced | swizzle32 诊断 shape 会被 32B span 切分，task 数多于 64/128；当前默认 sweep 最差为 1.144x | 若应用需要继续压低 swizzle32，可研究 128B destination line 内多个 swizzle row segment 的合并 |
| interleave16/32 | tensor front-end 按 16B/32B slice 发 line task，slice 内保持 destination coalesced，跨 slice更新 global offset | 默认 sweep 下 interleave16/32 仅为 1.103x/1.025x；更糟糕的 slice/row 组合需要用 stress 入口确认 | 暂不默认扩大 line table；若 stress 证明必要，再做小型 segmented row-group 合并 |
| padded global stride/subbox | unit element stride 且 in-bounds，可作为 feature/fallback 边界监控 | 默认功能测试已覆盖，当前不是主性能瓶颈 | 保持小规模回归监控 |
| element stride | dim0 stride 用 `dstWordStride` 表示 sparse destination word mask；dim1+ stride仍按有效 row 发 line task | 默认功能与 feature 诊断覆盖，当前没有证据显示是主要风险 | 保持当前 line-task 方案，后续只监控是否退回逐元素 fallback |
| OOB suppress | setup 同时保留 raw shared layout dim 和 clipped valid dim，只对 in-bounds 子区间发 write task | 默认功能与 feature 诊断覆盖，guard 区域保持不变 | 保持当前 clipping 方案，复杂 swizzle+OOB 组合再按真实需求扩展 |
| rank3/4/5 contiguous | 现在作为 no-feature row-level line-task/setup 监控点 | 默认功能与 feature 诊断覆盖 | 后续只按真实应用需求补 high-rank sparse/feature 组合 |

这意味着 swizzle/interleave 第一版已经解决了 element fallback 的大头；规则 stride、OOB clipping 仍保留 line-task 路径。真正需要继续关注的是 segmented PutPart 合并，但是否值得做应由 stress 或真实应用 shape 决定，而不是默认扩大硬件资源。

## 5. RTL 设计机制

### 5.1 Bulk S2G Backend

bulk S2G 的结构：

```text
CP_ASYNC_BULK_S2G
  -> instruction table
  -> line task allocator
  -> dynamic shared-read entries, instrId >= 2
  -> DMA L1TLB
  -> L2 PutFull/PutPart
  -> AccessAck
  -> instruction completion
```

核心语义：

| 机制 | 说明 |
|---|---|
| instruction table | 允许 fence 前多个 S2G instruction outstanding |
| line table | 每个 entry 表示一个 destination 128B cacheline write task |
| dynamic shared read | `instrId>=2`，避免固定 `instrId=0/1` 限制并发 |
| TLB translation | 每条 destination line 直接走 DMA 专用 8-entry L1TLB |
| ack table | L2 Put source tag 跟踪 line completion |
| completion | 只有所有 line 的 L2 AccessAck 返回后才完成 instruction |

面积关键点：

| 资源 | 当前原则 |
|---|---|
| 128B line data | 是最大面积项，保持小 entry 数 |
| line data storage | 用 word-banked `SyncReadMem`，metadata 用寄存器 |
| instruction metadata | 可接受，不为每条 instruction 存完整 128B data |
| address translation | 统一走 DMA L1TLB，避免 S2G 内部复制翻译状态 |
| ack/read table | 受 `max_dma_tag` 和 shared `instrId` 宽度约束 |

### 5.2 Tensor S2G Row-Level Line Task

tensor S2G 现在是 row-level line-task + fallback：

| 路径 | 条件 | 写回形态 |
|---|---|---|
| swizzle fast line task | FP32、swizzle32/64/128、当前覆盖的合法 in-bounds shape | destination 按 line/segment coalesced；shared source在 `DmaS2G` shared read issue 阶段 per-lane swizzle gather |
| interleave fast line task | FP32、interleave16/32、当前覆盖的合法 shape | 按 interleave slice 发 16B/32B coalesced line task，跨 slice 更新 global offset |
| stride/OOB line task | FP32、规则 element stride、OOB clipped 子区间 | dim0 stride 用 sparse word mask，OOB 只发有效子区间 |
| no-permute row line task | 无 swizzle/interleave、普通 FP32 row | 按 row/segment 发 `S2GLineTask`，不做跨多行 linear span fusion |
| fallback | 非 4B data width、复杂 swizzle+OOB、非法 interleave 或未覆盖不规则组合 | 保守 element-level 或 row-level legacy path |

fast path 当前已有机制：

| 机制 | 说明 |
|---|---|
| descriptor cache | 2-entry、每 entry 一条 128B descriptor line |
| data translation | fast path line task 统一由 `DmaS2G` 后端直接走 DMA L1TLB |
| line task domain | `first/last` 标记一条 tensor S2G 的 synthetic instruction 生命周期 |
| swizzle metadata | `S2GLineTask` 携带 `swizzleMode/swizzleBase/swizzleRow`，不增加 128B data buffer |
| interleave slice emitter | tensor 前端按 4/8 个 FP32 element 的 interleave slice 生成 line task |
| `dstWordStride` | 用极小 metadata 支持 packed shared 到 sparse destination word mask |
| raw/clipped dim | raw dim 维护 shared tile layout，clipped dim 只控制 global valid write |
| completion token | line task completion 携带 `{wid, group, is_s2g=true}`，group 同步由统一 DMA group ring 处理 |

注意：tensor fast path 虽然已经复用 bulk backend，但 descriptor/setup/addrgen 仍在 `DmaTensorS2G` 内，是后续 RTL 性能优化点之一。当前 swizzle/interleave/stride/OOB/high-rank 第一版 fast path 已经证明高级 feature 不应长期停在 element fallback；CUDA-style `interleave32+swizzle32` 组合 case 已经落地并四后端 PASS。从最新 paired RTL PMU 看，element stride、OOB 和 high-rank contiguous 已经不再是 10x 级问题；后续高级 feature RTL 优化应优先看 swizzle/interleave segmented row-group 合并，同时继续用 stride/OOB/rank3/4/5 作为“不退回 fallback”的护栏。

### 5.3 Completion 和 Group Wait

当前 completion token：

```text
DmaCompletion {
  wid
  group
  is_s2g
}
```

当前 fence 语义：

| 指令编码语义 | 用途 |
|---|---|
| legacy wait-all / wait-count | 兼容路径，按 all-DMA inflight 计数等待 |
| `commit_group` | 提交当前 per-warp DMA group；G2S、S2G、prefetch 都可进入当前 open group |
| `wait_group keep` | 按 group 年龄等待 all-DMA committed group，允许最近 `keep` 个 committed group 继续后台完成 |

这个模型不再提供 G2S-only 或 S2G-only wait 域。软件如果希望 G2S 和 S2G 不互相过度等待，需要显式安排 group：常见 tensor pingpong 写法是 `issue G2S -> commit_group -> issue S2G -> commit_group -> wait_group1`，这样等待更老的 G2S，同时允许最新 S2G 在后台完成。代价是单 ring、单 FULL counter 无法同时“等待较新的 G2S并保留更老的 S2G”；这是当前面积友好设计的同步语义边界。

## 6. CUDA 参数对齐原则

公开参数建议继续按 CUDA TMA 语义描述，Ventus 内部可以保留兼容 decode：

| CUDA 概念 | Ventus 当前/建议字段 | 约束 |
|---|---|---|
| `tensorDataType` | `TensorVars.dataType` | 公开 ABI 后续应对齐 CUDA enum；现有 legacy enum 可在 decode 层转换 |
| `tensorRank` | `TensorVars.tensorRank` | 1 到 5 |
| `globalAddress` | descriptor base word | 至少 16B 对齐 |
| `globalDim` | `TensorVars.globalDim` | 每维元素数 |
| `globalStrides` | `TensorVars.globalStrides` | byte stride，CUDA 要求对齐 |
| `boxDim` | `TensorVars.boxDim` | tile box 形状 |
| `elementStrides` | `TensorVars.elementStrides` | 主路径优先支持 1 |
| `interleave` | `TensorVars.interleaveMode` | fast path 先只支持 none |
| `swizzle` | `TensorVars.swizzleMode` | fast path 先只支持 none |
| `l2Promotion` | `TensorVars.L2promotion` | 可先 decode 后忽略 |
| `oobFill` | `TensorVars.oobfill` | G2S 是 fill，S2G store 侧应 suppress OOB writes |

参考：

- CUDA Driver API `cuTensorMapEncodeTiled`: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html
- PTX `cp.async.bulk.tensor`: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor

## 7. 最新验证结果

### 7.1 功能回归

当前最可信的功能 gate 是 `run_dma_tma_rtl.sh --suite directed`。它不是性能测试，也不是 smoke 抽样；它现在实际覆盖 3 个 testcase directory。注意这里的“suite verdict”是 host 顶层 suite 计数，S2G suite 内部还有更细的 byte-level 子 case。

| testcase directory | 当前角色 | 顶层 case 数 | 覆盖重点 |
|---|---|---:|---|
| `dma_tma_g2s_func_test` | G2S/TMA 强回归 gate | 10 | bulk/tensor G2S、TMA matrix、descriptor/prefetch、mixed async fence、routing conflict、multi-warp fence |
| `dma_tma_s2g_func_test` | 当前 S2G 主功能 gate | 5 | bulk S2G matrix 25 子 case、small dual 2 子 case、bulk stress 14 子 case、tensor S2G matrix 37 子 case、mixed routing 6 子 case |
| `dma_tma_multi_wg_func_test` | 多 WG mixed gate | 15 | 2WG/4WG bulk G2S/S2G、2WG/4WG tensor G2S/S2G、2WG group wait/subbox/cross-page stress、4WG group/page、4WG mixed G2S+S2G outstanding、2WG tensor stride/OOB、4WG tensor descriptor mix、4WG tensor group/page |

本次并发 gate 实测：

| 命令 | 后端 | testcase dirs | 顶层 case | fail | skip | 耗时明细 |
|---|---|---:|---:|---:|---:|---|
| `./run_dma_tma_rtl.sh --suite directed --backend gvm --run-jobs 4 --jobs 8 --timeout 1800` | GVM | 3/3 pass | 30 pass | 0 | 0 | G2S 47s, S2G 64s, multi-WG 7s |
| `./run_dma_tma_rtl.sh --suite directed --backend gvm-nocache --run-jobs 4 --jobs 8 --timeout 1800` | GVM-nocache | 3/3 pass | 30 pass | 0 | 0 | G2S 47s, S2G 48s, multi-WG 5s |
| `./run_dma_tma_rtl.sh --suite directed --backend rtl --run-jobs 4 --jobs 8 --timeout 1800` | RTL | 3/3 pass | 30 pass | 0 | 0 | G2S 45s, S2G 49s, multi-WG 11s |
| `./run_dma_tma_rtl.sh --suite directed --backend rtl-nocache --run-jobs 4 --jobs 8 --timeout 1800` | rtl-nocache | 3/3 pass | 30 pass | 0 | 0 | G2S 41s, S2G 54s, multi-WG 10s |

per-case summary：

| testcase | GVM | GVM-nocache | RTL | rtl-nocache |
|---|---|---|---|---|
| `dma_tma_g2s_func_test` | pass 10, fail 0, skip 0 | pass 10, fail 0, skip 0 | pass 10, fail 0, skip 0 | pass 10, fail 0, skip 0 |
| `dma_tma_s2g_func_test` | pass 5, fail 0, skip 0 | pass 5, fail 0, skip 0 | pass 5, fail 0, skip 0 | pass 5, fail 0, skip 0 |
| `dma_tma_multi_wg_func_test` | pass 15, fail 0, skip 0 | pass 15, fail 0, skip 0 | pass 15, fail 0, skip 0 | pass 15, fail 0, skip 0 |

这次测试还验证了 runner 并发隔离修复：每个 child 使用独立 `POCL_CACHE_DIR` 和 `TMPDIR/TMP/TEMP` 后，之前的 POCL `mkstemp()` / `CL_BUILD_PROGRAM_FAILURE` 不再出现。因此该错误应归为并发运行环境/cache 问题，不是 RTL S2G/G2S 数据通路错误。

本次 S2G functional 子 case 覆盖：

| suite | 子 case | 主要覆盖 | RTL/rtl-nocache |
|---|---:|---|---|
| `bulk_s2g` | 25 | aligned/partial/tail/cross-dst、cross-source、cross-source+dst、fixed mask/offset、deterministic seed mask/offset、4 个新增 srcOffset/dstOffset/bytes fixed-seed 边界、dst cross 4KB page、G2S->S2G roundtrip | PASS/PASS |
| `bulk_s2g_small_dual` | 2 | 4B/8B/16B/32B 小 copy，两条 S2G 一个 fence drain | PASS/PASS |
| `bulk_s2g_stress` | 14 | 4/6/8 line pressure、empty commit、wait-oldest、wait-group、group wrap、multi-issue group wrap、跨页/TLB route、多 warp fence；same-line overlap 顺序 case 已删除 | PASS/PASS |
| `tma_s2g` | 37 | rank1/2/3/4/5、partial tail、subbox、padded stride、element stride、interleave16/32、swizzle32/64/128、OOB suppress、rank3/subbox OOB、long shape、datatype alias、descriptor cache A/B/C/A、4 个 fixed-seed dims/coords/box/stride/OOB/swizzle 组合 | PASS/PASS |
| `mixed_s2g_routing` | 6 | S2G shared route conflict、unified DMA group keep/wait0、bulk+tensor same fence、bidirectional descriptor mix、`mixed_tensor_outstanding_long` | PASS/PASS |

当前 fixed-seed / mixed long 子 case：

| 组 | 新增 case | 目的 | 验证 |
|---|---|---|---|
| bulk fixed-seed | `fuzz_seed08_src4_dst60_12B`, `fuzz_seed09_src252_dst120_20B`, `fuzz_seed10_src508_dst4092_36B`, `fuzz_seed11_src764_dst188_84B` | 覆盖小 offset、shared line 尾部、destination 4KB page 尾部和非整 line byte mask 组合 | GVM bulk focused PASS；RTL directed PASS；rtl-nocache directed PASS |
| tensor fixed-seed | `fuzz_tensor_rank2_stride_oob_swizzle32`, `fuzz_tensor_rank3_stride_subbox`, `fuzz_tensor_rank3_swizzle64_oob`, `fuzz_tensor_rank5_sparse_stride` | 覆盖 dims/coords/box/stride、fallback/OOB/swizzle/high-rank sparse stride 组合 | GVM `S2G_TENSOR_CASE_FILTER=fuzz_tensor` PASS；RTL focused PASS；RTL/rtl-nocache directed PASS |
| mixed long | `mixed_tensor_outstanding_long` | 在同一窗口混合 bulk G2S、tensor G2S、两条 bulk S2G、两条 tensor S2G，检查 completion、ack route、page/TLB owner 和 marker 顺序 | GVM focused PASS；RTL focused PASS；RTL/rtl-nocache directed PASS |

注意：新增 `fuzz_tensor_rank3_stride_subbox` 曾暴露 host reference 的 outDim 语义问题。RTL 对 dim1+ 的 `elementStride` 使用 `ceil(boxDim / elementStride)` 作为迭代次数，dim0 保持 `boxDim`；文档和 host expected 已按这个实际实现修正。

本次新增的 multi-WG tensor S2G 顶层 case：

| case | 主要覆盖 | RTL/rtl-nocache |
|---|---|---|
| `tma_s2g_2wg_stride_oob` | 2WG 不同 coords、element stride、OOB suppress，校验 guard 不被写坏 | PASS/PASS |
| `tma_s2g_4wg_descriptor_mix` | 4WG rank1/rank2/subbox/stride 混合 descriptor，校验 descriptor/coords/WG offset 隔离 | PASS/PASS |
| `tma_s2g_4wg_group_page` | 4WG tensor S2G 写到 sparse page，混合 group/page/TLB owner 与 completion 隔离 | PASS/PASS |

tensor S2G matrix 的具体 case：

| case | 意义 | 当前结果 |
|---|---|---|
| `rank1_32_fp32_full_line`, `rank1_64_fp32_two_lines`, `rank1_24_fp32_partial_tail` | rank1 full/two-line/tail partial | PASS |
| `rank2_4x4_origin`, `rank2_16x16_contiguous`, `rank2_subbox_8x8_at_2_2` | rank2 origin、主流 contiguous tile、subbox | PASS |
| `rank2_padded_rows_stride64`, `rank2_element_stride2_cols` | non-contiguous stride/fallback boundary | PASS |
| `rank2_swizzle32_rows`, `rank2_swizzle64_rows`, `rank2_swizzle128_subbox_row1` | swizzle32/64/128 地址映射，in-bounds/unit-stride 可走 swizzled line-task fast path | PASS |
| `rank3_plain_2x2x2`, `rank4_plain_small`, `rank5_plain_small` | rank3/4/5 地址生成 | PASS |
| `interleave16_rank3`, `interleave32_rank3` | interleave16/32 地址映射，in-bounds/unit-stride 可走 slice-level line-task fast path | PASS |
| `rank2_oob_dim0`, `rank2_oob_dim1` | S2G OOB suppress，global guard 保持 | PASS |
| `rank3_subbox_2x2x2_at_1_1_1`, `rank3_oob_dim2_suppress`, `rank2_oob_subbox_partial` | rank3 subbox、dim2 OOB suppress、rank2 subbox partial OOB | PASS |
| `rank3_long_4x4x4`, `rank3_stride_oob_long`, `rank4_long_4x4x2x2`, `rank5_long_4x2x2x2x2` | high-rank long shape、stride + OOB fallback | PASS |
| `interleave32_long_rank3`, `swizzle128_oob_stride` | long-shape interleave/swizzle，以及 OOB/stride fallback 组合 | PASS |
| `u32_rank2_4x4_alias`, `i32_rank1_32_alias` | 4B datatype enum alias，确认搬运不依赖 FP32 解释 | PASS |
| `fuzz_tensor_rank2_stride_oob_swizzle32`, `fuzz_tensor_rank3_stride_subbox`, `fuzz_tensor_rank3_swizzle64_oob`, `fuzz_tensor_rank5_sparse_stride` | 少量固定 seed，覆盖 stride/OOB/swizzle/subbox/high-rank sparse stride 组合 | PASS |
| `descriptor_cache_A_fill`, `descriptor_cache_B_fill`, `descriptor_cache_C_replace`, `descriptor_cache_A_reuse` | 2-entry descriptor cache fill/replacement/reuse，不做 hot-update | PASS |

bulk S2G stress 的实现注意点：

| 点 | 说明 |
|---|---|
| 6/8 line pressure | kernel 使用显式展开 issue，不用 loop 生成，避免 GVM/RTL 后端遇到未覆盖的编译器生成指令 |
| case 过滤 | `S2G_STRESS_CASE_FILTER` 可单独定位 stress 子 case |
| tensor case 过滤 | `S2G_TENSOR_CASE_FILTER` 可单独定位 tensor matrix 子 case |

### 7.2 性能代表点

下面这些代表点来自 2026-07-04 当前分支 RTL full sweep。它们是当前代码状态的参考值，不是长期固定门限。

| 测试 | 后端 | 点 | baseline cycles | opt cycles | speedup |
|---|---|---|---:|---:|---:|
| `dma_tma_g2s_pingpong_perf_test` | RTL | 16x16, buffers=2, stages=4 | 16224 | 13166 | 1.2323x |
| `dma_tma_tensor_s2g_pingpong_perf_test manual_tensor_s2g` | RTL | 16x16, buffers=2, stages=4 | 16174 | 14394 | 1.1237x |
| `dma_tma_tensor_s2g_pingpong_perf_test manual_tensor_s2g` | RTL | 64x64, buffers=2, stages=16 | 888664 | 653382 | 1.3601x |
| `dma_tma_tensor_s2g_pingpong_perf_test tma_tensor_s2g` | RTL | 16x16, buffers=2, stages=4 | 16174 | 11486 | 1.4081x |
| `dma_tma_tensor_s2g_pingpong_perf_test tma_tensor_s2g` | RTL | 64x64, buffers=2, stages=16 | 888662 | 358228 | 2.4807x |

对应 report：

| report | 用途 |
|---|---|
| `testcases/_get_case/dma_tma_g2s_pingpong_perf_test/log/dma_tma_g2s_pingpong_perf_report_20260704_183433.md` | G2S pingpong full sweep |
| `testcases/_get_case/dma_tma_tensor_s2g_pingpong_perf_test/log/dma_tma_tensor_s2g_pingpong_perf_manual_tensor_s2g_report_20260704_183429.md` | manual input + tensor S2G full sweep |
| `testcases/_get_case/dma_tma_tensor_s2g_pingpong_perf_test/log/dma_tma_tensor_s2g_pingpong_perf_tma_tensor_s2g_report_20260704_183851.md` | TMA G2S input + tensor S2G full sweep |
| `testcases/_get_case/dma_tma_movement_profile_test/log/dma_tma_movement_profile_report_20260704_194747.md` | RTL DMA/TMA movement profile |

### 7.3 当前 Full Sweep 快照

下面是当前分支上最新 full sweep 快照，用于观察趋势。后续 RTL/FSM 继续改动后应整体重跑。

| 测试 | 后端 | pass | min | avg | max |
|---|---|---:|---:|---:|---:|
| `dma_tma_g2s_pingpong_perf_test` | RTL | 25/25 | 1.1214x | 1.3703x | 1.4998x |
| `dma_tma_tensor_s2g_pingpong_perf_test manual_tensor_s2g` | RTL | 25/25 | 1.0647x | 1.2493x | 1.3601x |
| `dma_tma_tensor_s2g_pingpong_perf_test tma_tensor_s2g` | RTL | 25/25 | 1.2068x | 1.9379x | 2.4807x |

## 8. 性能解释

### 8.1 为什么 `G2S+S2G` 可以超过 2x

当前 baseline 已经不是 lane0 串行搬运，而是 warp/workgroup 协作的 manual path。超过 2x 的原因不是 baseline 故意写弱，而是这个测试里 compute 很轻，主要成本来自两端数据搬运和同步：

```text
manual baseline:
  global -> register -> shared
  compute
  shared -> register -> global

G2S+S2G:
  TMA G2S fills shared
  compute
  tensor/bulk S2G writes back
```

在大 tile、多 stage 下，G2S 输入和 S2G 写回都能在后台推进，manual path 的显式 load/store、barrier 和寄存器中转被大量移除，所以 `G2S+S2G` 大 tile 超过 2x 是合理结果。小 tile 的 speedup 较低，主要受 fixed setup、barrier、descriptor、fence 和 kernel 控制流开销影响。

### 8.2 为什么当前不急着扩大 buffer 或继续做性能 RTL

最新代表点显示：

| 观察 | 含义 |
|---|---|
| 大 tile PutFull/PutPart 主要为 PutFull/0 | data path coalescing 已生效 |
| line/read/ack stall 在代表点很低 | backend line/ack 周转不是当前最显著瓶颈 |
| 小 tile 和 manual input + S2G avg 仍受限 | 更像 setup/FSM 固定开销，是后续硬件优化候选 |
| 三 tile 原地轮转已把 `dmaFenceWait` 从 4 万级降到千级 | wait/fence drain 这个大问题已经被释放 |
| feature paired A/B 没有数量级退化 | 默认 sweep 中 paired host ratio 约 0.973x 到 1.144x；stress 入口保留给后续最坏参数搜索 |

因此当前不应继续加大 line entry、descriptor cache 或 shared tile ring。功能测试护栏已经补到 tensor descriptor、bulk line backend、mixed completion/routing、fixed-seed fuzz 和 multi-WG group/page；后续性能 RTL 可以分三条线看：连续 FP32 主路径保留 fast setup/FSM 候选，高级 feature 先用 stress 找最坏参数再决定是否做 swizzle/interleave segmented row-group 合并，系统级性能则补 multi-WG roundtrip perf 来观察真实并发压力。

## 9. G2S 与 S2G 路线对比

本节专门比较当前 G2S 和 S2G 两条 DMA/TMA 路线。结论先说清楚：二者不是简单的反向镜像。G2S 是 global memory 到 shared memory 的读取填充路径；S2G 是 shared memory 到 global memory 的写回提交路径。方向不同导致数据聚合位置、TLB/L2 交互、shared memory 访问方式、completion/fence 语义和性能瓶颈都不同。因此 S2G 不能直接照搬 G2S 的 temporary memory 结构，而需要独立的 line/read/ack backend。

### 9.1 RTL 数据通路对比

| 项 | G2S / TMA G2S | bulk/tensor S2G |
|---|---|---|
| 主入口 | `DMA_core.scala` 中非 `funct==3/4` 路径进入 `AddrCalc_l2cache` | bulk S2G `funct==3` 进入 `DmaS2G`；tensor S2G `funct==4` 进入 `DmaTensorS2G` |
| 方向 | global -> L2 Get -> temporary memory -> shared write | shared read -> line image -> DMA L1TLB -> L2 Put -> AccessAck |
| 数据来源 | L2 返回 global cacheline data | shared memory per-lane read response |
| 数据汇聚点 | `Temp_mem` 保存 L2 cacheline image 和 tag metadata | `DmaS2G` line table + word-banked `SyncReadMem` 保存待写回 line image |
| 地址生成 | `AddrCalc_l2cache` 负责 descriptor、box、stride、global read 地址和 tag | bulk 由 `DmaS2G` addrgen 切 line；tensor 由 `DmaTensorS2G` descriptor/setup 生成 `S2GLineTask` |
| shared 接口 | 主要是向 shared memory 写入 G2S 数据 | 主要是从 shared memory 读取待写回数据 |
| L2 操作 | `Get` 为主，response 携带数据 | `PutFull` / `PutPart` 为主，response 主要是写 ack |
| completion | shared 写入完成后产生 DMA completion，`is_s2g=false` | L2 Put ack 全部返回后产生 completion，`is_s2g=true` |
| fence 域 | 统一 per-warp DMA group ring；legacy wait-all 按 all-DMA inflight 等待 | 同一 group ring；commit/wait_group 按 all-DMA committed group 年龄等待 FULL completion |
| 主要表项 | temp data/tag、prefetch/descriptor 相关状态 | instruction table、line table、shared read table、ack table |
| 高级 tensor feature | G2S 已有较成熟 descriptor/matrix path | tensor S2G 前端解析 feature，尽量转换成 coalesced line task，再复用 bulk S2G backend |

直观地说，G2S 的中心问题是“怎样把 global cacheline 拉回来并正确写进 shared”；S2G 的中心问题是“怎样从 shared 收集出一个 global cacheline，再正确、可确认地写回 global”。后者多了 write ack、partial mask 和 buffer reuse 的约束；无同步 overlap 写同一 destination byte 交给软件契约处理。

### 9.2 G2S 为什么适合 temporary memory 路线

G2S 的数据天然从 L2 以 cacheline response 形式回来。硬件只要记录每个 L2 response 属于哪个 DMA instruction、哪个 tensor row/box、写到 shared 的哪个位置，就可以在 `Temp_mem` 中暂存 cacheline image，然后由 `AddrCalc_shared` 产生 shared write。这个方向有几个特点：

1. L2 response 自带完整或可切片的 data line，硬件不需要先从 shared 聚合数据。
2. shared memory 是写目标，写入后 compute 才消费，因此 completion 更接近“shared tile 已填好”。
3. G2S TMA 的 descriptor、prefetch、tag reuse 和 temporary cacheline image 已经围绕 L2 Get response 优化多年，适合继续保留。
4. TMA G2S 的性能瓶颈主要是 descriptor/setup、L2 Get/response、shared write 吞吐和 overlap；它不需要处理 global write ack 的可见性问题。

这也是为什么 G2S 的 RTL 中能看到 `AddrCalc_l2cache -> Temp_mem -> AddrCalc_shared` 这条清晰流水。它的 temporary memory 是 global read response 的落点。

### 9.3 S2G 为什么不能简单反向复用 G2S

如果把 G2S 反过来想成“shared -> tempmem -> L2”，看起来似乎可以复用 temporary memory，但实际会遇到几个硬约束：

| 约束 | 为什么影响 S2G 设计 |
|---|---|
| shared 是数据源 | S2G 必须主动发 shared read，把每个 lane/word 的数据收齐；没有天然的 L2 data response 可直接写进 tempmem |
| global 是写目标 | 写回必须发 `PutFull`/`PutPart`，并等待 `AccessAck` 才能确认 global 可见 |
| partial line 很常见 | dst offset、OOB、stride、swizzle/interleave 都可能产生 mask，必须精确维护 word mask |
| 同 global byte 可能多次写 | 若多个 outstanding S2G 覆盖同一 destination byte 且中间无 wait/barrier，结果未定义 |
| buffer reuse 更危险 | kernel 复用 shared output buffer 前，必须确认对应 S2G writeback 已经 FULL completion |
| tensor feature 影响源和目标 | swizzle 改 shared layout，interleave/stride 改 global layout，必须在 line task 中携带 metadata |

因此当前 S2G 使用独立 `DmaS2G` backend：用 instruction table 维护 DMA 指令生命周期，用 line table 维护每条 global line，用 shared read table 跟踪 shared outstanding read，用 ack table 跟踪 L2 Put ack，并直接依赖 DMA 专用 L1TLB 完成 destination translation。这个结构是写路径需要的，不是 G2S temporary memory 的重复发明。

### 9.4 Tensor S2G 为什么拆成 frontend + backend

tensor S2G 当前没有把所有功能塞进一个大 FSM，而是拆成：

```text
DmaTensorS2G:
  descriptor cache / tensor setup / rank-stride-box-OOB-swizzle-interleave
  -> S2GLineTask

DmaS2G:
  shared read / line image / page-TLB / PutFull-PutPart / ack / completion
```

这样设计的原因有三点：

1. bulk S2G 和 tensor S2G 都需要同一套写回 backend。复用 backend 可以避免为 tensor 再复制一份 line table、read table、ack table 和 TLB route。
2. tensor 语义变化快，尤其是 swizzle、interleave、stride、OOB、高 rank。把这些限制在 frontend，backend 只认 line task，RTL 边界更清晰。
3. 面积更可控。当前只增加 line metadata、descriptor cache、少量 address/setup 寄存器和 swizzle/interleave 计算逻辑，不按 tile/stage 扩大大 buffer。

这个拆分也解释了为什么当前 S2G 高级 feature 的性能优化重点是“让 frontend 产生更少、更规整的 line task”，而不是盲目扩大 backend 表项。

### 9.5 性能结果怎么比较

当前 full sweep 里的三个主性能口径不是同一个实验，不能直接读成“G2S 比 S2G 快”或“S2G 比 G2S 快”：

| 测试 | 优化方向 | RTL min/avg/max | 解释 |
|---|---|---:|---|
| `dma_tma_g2s_pingpong_perf_test` | 只优化 input：manual G2S -> TMA G2S，writeback 仍是 manual | 1.1214x / 1.3703x / 1.4998x | G2S 能显著降低输入搬运和同步成本，但输出端仍保留 manual writeback，所以 speedup 有上限 |
| `dma_tma_tensor_s2g_pingpong_perf_test manual_tensor_s2g` | 只优化 output：manual writeback -> tensor S2G，input 仍是 manual | 1.0647x / 1.2493x / 1.3601x | S2G 写回有收益，但仍要付 shared read gather、Put ack、group wait 和 descriptor/setup 成本；manual input 也限制总体 speedup |
| `dma_tma_tensor_s2g_pingpong_perf_test tma_tensor_s2g` | 同时优化 input 和 output：TMA G2S + tensor S2G | 1.2068x / 1.9379x / 2.4807x | 两端搬运都交给 DMA/TMA，manual load/store、寄存器中转和同步成本同时减少，大 tile 多 stage 下可以超过 2x |

这说明当前性能差距更多来自测试分解方式和方向语义，而不是 S2G backend “天然不如 G2S”。G2S-only 的 avg 约 1.37x，高于 manual-input + tensor-S2G 的 avg 约 1.25x，主要原因是：

1. G2S 的 L2 Get response 可以直接形成 cacheline image；S2G 必须先从 shared read 收集 line image。
2. G2S completion 关注 shared tile 已可用；S2G completion 必须等 L2 Put ack，FULL group 语义更重。
3. S2G 需要处理 PutPart、ack tag 和 buffer reuse，控制面比 G2S 写 shared 更复杂。
4. `manual_tensor_s2g` 的 input 仍是 manual，整体时间里仍保留大量输入搬运成本，掩盖了一部分 writeback 优化收益。

反过来，当 `tma_tensor_s2g` 同时使用 G2S 和 S2G 后，avg 接近 2x、max 超过 2.5x，说明两条路线的收益可以叠加，且当前 S2G backend 对主流 contiguous FP32 tile 已经足够有效。

### 9.6 设计取舍总结

当前二者差异可以概括为：

| 设计问题 | G2S 选择 | S2G 选择 | 原因 |
|---|---|---|---|
| 数据暂存 | temporary cacheline memory | line table + word-banked line image | G2S 暂存 L2 response；S2G 暂存 shared read 聚合结果 |
| Tensor 语义 | addressgen 和 tempmem tag 携带 tensor metadata | tensor frontend 生成 line task，backend 不理解完整 tensor | S2G 写回 backend 需要被 bulk/tensor 共用 |
| Completion | shared write 完成 | L2 Put ack 完成 | S2G 必须保证 global 可见和 shared buffer 可复用 |
| 高级 feature | 由成熟 G2S TMA path 处理 | swizzle/interleave/stride/OOB 尽量 coalesced line task，复杂角落 fallback | 避免逐元素 PutPart，同时控制面积 |
| 面积策略 | 保留既有 tempmem/prefetch 结构 | 小 line/read/ack/desc table，不复制大 buffer | S2G 是新增能力，不能无限扩大资源 |
| 性能优化方向 | descriptor/prefetch、L2 Get overlap、shared write | line coalescing、PutFull 比例、ack/fence、segmented feature 合并 | 两个方向的主要瓶颈不同 |

因此，后续优化也应该分开看：G2S 继续作为成熟 baseline 和 input overlap 参考；S2G 重点维护 line-based writeback、FULL group fence 和 feature coalescing。若要提升 S2G 相对 G2S 的单向性能，优先方向不是照搬 G2S tempmem，而是减少 tensor frontend 固定开销、降低 swizzle/interleave 分段数、提高 PutFull 占比、缩短 ack/fence 周转，同时保持 line/read/ack 表项规模受控。

## 10. 已知约束

| 约束 | 当前处理 |
|---|---|
| descriptor hot-update 缺少显式一致性语义 | 软件不在 hot loop 内改 descriptor metadata line；directed test 使用独立 descriptor buffer |
| tensor fallback 路径仍需边界管理 | 规则 stride/OOB/high-rank contiguous 已进入 line-task 路径；非 4B、复杂 swizzle+OOB、非法 interleave 或未覆盖不规则组合仍走 fallback |
| tensor setup 仍是通用 5D FSM | 后续可做 FP32 rank1/rank2 fast setup；对应功能边界 case 已有 directed 覆盖 |
| tensor 前端仍基本单线性发射 | 后续再评估 1 到 2 项 line-task FIFO；当前先用 directed/perf 保证现有语义 |
| full sweep 是当前 registry 快照 | RTL `--suite perf` 当前 5/5 PASS；后续 RTL/FSM 或测试内容变化后再刷新 |
| 面积/时序还未综合确认 | 保持资源小规模，下一轮 RTL 后做复核 |

## 11. 功能测试覆盖完善计划

当前阶段的目标不是继续证明 TMA/S2G 能带来性能收益，而是把已经实现的 RTL 机制用 directed functional case 固定住。后续任何 fast setup、line emitter、fallback coalescing 或小 FIFO 改动，都应该先在这些功能测试下保持回归通过。

### 11.1 调研依据

当前调研基于 RTL 和功能/性能测试的完整路径，而不是只看单个 kernel：

| 类别 | 文件 | 关注点 |
|---|---|---|
| bulk S2G backend | `gpgpu/ventus/src/pipeline/DMA_s2g.scala` | instruction table、line table、dynamic shared-read、direct DMA L1TLB、ack table、external tensor line task |
| tensor S2G frontend | `gpgpu/ventus/src/pipeline/DMA_tma_s2g.scala` | descriptor cache、setup/addrgen、swizzle/interleave/stride/OOB feature line task、fallback path |
| DMA core route | `gpgpu/ventus/src/pipeline/DMA_core.scala` | TLB owner、L2 response/ack route、shared response route、completion arbiter、PMU 汇总 |
| scheduler | `warp_schedule.scala`, `scoreboard.scala`, `pipe.scala` | `{wid, group, is_s2g}` completion、per-warp all-DMA group ring、legacy all-DMA wait/count |
| S2G directed | `testcases/_get_case/dma_tma_s2g_func_test/` | bulk/tensor S2G 当前主功能回归 |
| G2S directed 参考 | `testcases/_get_case/dma_tma_g2s_func_test/` | TMA matrix、descriptor cache/prefetch、routing conflict、mixed async fence 的组织方法 |
| multi-WG directed | `testcases/_get_case/dma_tma_multi_wg_func_test/` | 2WG bulk/tensor S2G 基础功能与 G2S multi-WG 参考 |
| perf trend/full | `dma_tma_g2s_pingpong_perf_test`, `dma_tma_tensor_s2g_pingpong_perf_test` | 只作为长 stage/overlap 性能趋势，不作为 byte-level 功能覆盖替代 |

### 11.2 当前已有功能覆盖

bulk S2G 当前已经不是只测一个简单 copy。`dma_tma_s2g_func_test` 里已有 25 个 bulk matrix 子 case、2 个 small-dual 子 case 和 14 个 stress 子 case：

| 覆盖组 | 已有 case | 当前意义 | 仍然不足 |
|---|---|---|---|
| 基础 copy/mask | `basic_128B_aligned`, `partial_32B`, `partial_tail_96B`, `dst_offset_cross_line`, `fixed_mask_*`, `mask_seed00..07_*`, `fuzz_seed08..11_*` | PutFull/PutPart、tail mask、destination cross-line、固定 mask/offset、8 个 deterministic seed offset/size 组合，以及 4 个新增 srcOffset/dstOffset/bytes fixed-seed 边界 | 后续只在发现真实 RTL 风险时再加少量固定 seed，不做随机长跑 |
| source/dst split | `src_shared_offset_128B`, `cross_source_line_same_dst_line`, `cross_source_and_dst_line` | shared source 跨 128B line，destination 同线/跨线组合 | 还缺不同 lane bank conflict 组合 |
| page/TLB | `dst_cross_page`, `cross_page_tlb_ABC` | destination 4KB page 边界和跨页/TLB owner 回归 | 还缺 multi-WG page 竞争的更大组合 |
| mixed mini-roundtrip | `g2s_wait_s2g_roundtrip` | G2S 后接 S2G，基础 completion 顺序 | 规模小，不能替代完整 mixed completion/routing |
| small dual | `dual_s2g_4B_8B`, `dual_s2g_16B_32B_offset` | 两条小 S2G 用一个 fence drain | 仍是单 WG、低压力 |
| line pressure | `four_issue_one_fence`, `line_entry_pressure_6_lines`, `line_entry_pressure_8_lines` | 4/6/8 line outstanding，覆盖 backpressure/recovery | 还缺与 page/group 混合 |
| group/wait | `group_empty_commit`, `wait_oldest_one_reuse`, `wait_group_reuse`, `group_wrap_wait0123`, `group_multi_issue_wrap` | empty commit、wait-oldest、wait_group0/1/2/3、group wrap 后 source reuse、一个 group 内多条 S2G 后 wrap 回同一 group slot | 还缺更复杂 group/page 组合 |
| multi-warp | `multi_warp_s2g_fence` | 多 warp 独立 S2G fence | 还缺多 warp + group/page 组合 |

tensor S2G matrix 已从 4 个基础 case 扩成 37 个子 case：

| 覆盖组 | 已有 case | 当前意义 | 仍然不足 |
|---|---|---|---|
| rank1 linear | `rank1_32_fp32_full_line`, `rank1_64_fp32_two_lines`, `rank1_24_fp32_partial_tail` | full line、two-line first/last、tail partial | 还缺跨 4KB page 的 tensor line |
| rank2 mainstream | `rank2_4x4_origin`, `rank2_16x16_contiguous`, `rank2_subbox_8x8_at_2_2` | origin、主流 contiguous tile、subbox offset | 还缺更大 subbox 边界组合 |
| non-contiguous | `rank2_padded_rows_stride64`, `rank2_element_stride2_cols`, `fuzz_tensor_rank3_stride_subbox`, `fuzz_tensor_rank5_sparse_stride` | row stride、dim0/dim1+ element stride、subbox 和 high-rank sparse fallback | 还缺 multi-WG high-rank sparse 组合 |
| high rank | `rank3_plain_2x2x2`, `rank3_subbox_2x2x2_at_1_1_1`, `rank4_plain_small`, `rank5_plain_small`, `rank3_long_4x4x4`, `rank4_long_4x4x2x2`, `rank5_long_4x2x2x2x2` | rank3/4/5 descriptor/setup/addrgen、rank3 subbox、更大 high-rank shape | 还缺 multi-WG high-rank 混合 |
| interleave/swizzle | `interleave16_rank3`, `interleave32_rank3`, `interleave32_long_rank3`, `rank2_swizzle32_rows`, `rank2_swizzle64_rows`, `rank2_swizzle128_subbox_row1`, `swizzle128_oob_stride`, `fuzz_tensor_rank2_stride_oob_swizzle32`, `fuzz_tensor_rank3_swizzle64_oob`；feature perf 中已有 CUDA-style paired swizzle/interleave 组合 | interleave 与 swizzle 地址映射，覆盖 fast path、long shape、OOB/stride fallback、fixed-seed swizzle/OOB 组合和单 WG feature perf | 后续若有需求再补 multi-WG swizzle/OOB 组合 |
| OOB suppress | `rank2_oob_dim0`, `rank2_oob_dim1`, `rank3_oob_dim2_suppress`, `rank2_oob_subbox_partial` | S2G OOB 不写，global guard 保持，subbox partial OOB | 还缺更大 shape 的 OOB 边界 |
| datatype alias | `u32_rank2_4x4_alias`, `i32_rank1_32_alias` | 4B datawidth 下 U32/I32/FP32 enum 只影响 descriptor decode，不影响搬运语义 | 非 4B datatype 应放 negative suite，不进默认 gate |
| descriptor cache | `descriptor_cache_A_fill`, `descriptor_cache_B_fill`, `descriptor_cache_C_replace`, `descriptor_cache_A_reuse` | 2-entry cache fill/replacement/reuse；不做 hot-update | 还缺显式 prefetch/invalidate 语义，当前作为软件约束 |

mixed 与 multi-WG 当前状态：

| 路径 | 已有覆盖 | 缺口 |
|---|---|---|
| S2G suite 的 mixed routing | `s2g_routing_conflict`, `dma_group_keep1_preserves_newer_s2g`, `dma_group_wait0_drains_g2s_s2g`, `bulk_tensor_s2g_same_fence`, `bidirectional_descriptor_mix`, `mixed_tensor_outstanding_long` | 覆盖 unified group ring 的 keep=1/0、bulk/tensor same fence、descriptor mix，以及同一窗口内的 G2S + bulk S2G + tensor S2G 长序列 | 后续可选扩 multi-WG 版 tensor mixed long |
| G2S suite 的 mixed async fence | 已有 bulk G2S/S2G、tensor G2S/S2G、prefetch、normal store 混合 kernel | 继续作为 G2S 强回归参考 |
| perf pingpong | 覆盖 `commit G2S group -> commit S2G group -> wait_group1` 的长 stage overlap | perf 输出不做 byte-level mask/OOB/guard 检查 |
| multi-WG bulk S2G | 2WG/4WG 独立段写回、2WG group wait、2WG cross-page、4WG group/page | 后续优先看 group/page 与 tensor mixed，不再补无同步 overlap 顺序 case |
| multi-WG mixed G2S/S2G | 4WG mixed G2S + bulk S2G outstanding | 已覆盖 completion/routing 基础混合；single-WG S2G suite 已有 G2S + bulk S2G + tensor S2G 长 outstanding | 后续可选扩 multi-WG 版 tensor mixed long |
| multi-WG tensor S2G | 2WG/4WG rank2 基础功能、2WG 不同 coords/subbox、2WG stride/OOB、4WG descriptor mix、4WG tensor group/page | 当前足够支撑后续 group fence、feature diagnostic、fast setup/FSM 等 RTL 优化；后续按 bug 反馈再加 high-rank/swizzle multi-WG |

### 11.3 RTL 风险到测试覆盖映射

下面这张表是后续补测试的核心依据。每一行都对应当前 RTL 里真实存在的状态、表项或路由，而不是抽象地“多测一些”。

| RTL 机制 | 当前已有测试 | 风险 | 建议新增 case |
|---|---|---|---|
| `instExternal` synthetic instruction | `tma_s2g` 基础 case 已覆盖不 hang | tensor line task 与普通 bulk addrgen 混淆会造成 zero-byte line 或 completion 泄漏 | back-to-back tensor S2G before fence；tensor S2G 与 bulk S2G 同 fence；tensor first/last 多 line |
| line table 4 entry 左右规模 | `four_issue_one_fence` | 正好等于 entry 数，无法验证 full 后 backpressure 和恢复 | 6/8 line issue before fence，要求最终数据正确，允许 stall |
| dynamic shared-read `instrId>=2` | bulk stress 与 `s2g_routing_conflict` 覆盖 | shared response route 和 normal local memory 访问冲突 | 后续扩随机 mask/offset routing conflict |
| source shared line split | `src_shared_offset_128B` 间接覆盖 | source 跨 128B line 时 lane mask/word mapping 错 | explicit `cross_source_line_same_dst_line`、`cross_source_and_dst_line` |
| destination line mask | partial/cross-line/fixed mask、8 个 deterministic mask cases 和 4 个新增 fixed-seed offset/size cases | PutFull/PutPart 选择、byte mask、tail mask 错 | 当前已足够支撑下一轮 RTL 优化；发现具体 bug 后再定向加 seed |
| overlap 编程契约 | same-line overlap case 已删除 | 无 wait/barrier 覆盖同一 destination byte 属于未定义行为 | 需要顺序覆盖时由 kernel 插入 wait/barrier |
| ack table/source tag | stress 与 `bulk_tensor_s2g_same_fence` 覆盖 | ack route 错会提前 completion 或无法释放 line | 更多 line + mixed bulk/tensor S2G ack 组合 |
| TLB owner / cross-page | `dst_cross_page`, `cross_page_tlb_ABC`, `multi_wg_cross_page`, `bulk_s2g_4wg_group_page`, `tma_s2g_4wg_group_page`, `mixed_tensor_outstanding_long` 覆盖 | 跨页、TLB owner、与 G2S/Tensor descriptor TLB 竞争 | 后续可选扩 multi-WG tensor mixed long |
| unified completion group | `dma_group_keep1_preserves_newer_s2g`, `dma_group_wait0_drains_g2s_s2g`, `mixed_g2s_s2g_4wg`, `mixed_tensor_outstanding_long` 覆盖 | G2S/S2G completion 递减错误 group、group ring wrap 后 slot 过早复用、wait_group keep 年龄判断错误 | 下一轮 RTL 优化后固定复跑 |
| group ring | wait-oldest/wait-group、`group_wrap_wait0123`、`group_multi_issue_wrap` 覆盖 | empty commit、wait_group1/3、wrap/overflow、同一 group 多 outstanding FULL counter | 后续扩 group 与 page/tensor/multi-WG 组合 |
| descriptor cache | 独立 buffer 避免误用 | cache hit/replacement 未显式验证 | desc A hit、desc B fill、desc C replacement、A miss reload；不做 hot-update |
| tensor fast/fallback boundary | 37 个 tensor matrix case、`tma_s2g_2wg_stride_oob`、`tma_s2g_4wg_descriptor_mix` | fast path 误吃 OOB/stride，或 swizzle/interleave line task/fallback element mask 错 | 当前 fixed-seed fuzz 已落地；后续 RTL fast setup 或 feature path 改动后重点复跑 |
| S2G OOB suppress | dim0/dim1/dim2/subbox partial 已覆盖 | 错误写 global guard，或误按 G2S fill 语义处理 | 更大 shape 和多 WG mixed OOB，继续校验 guard 不变 |
| multi-WG CTA isolation | 2WG/4WG 基础、2WG group wait、2WG subbox、cross-page、4WG group/page、4WG mixed outstanding、tensor stride/OOB、tensor descriptor mix、tensor group/page | WG id、descriptor/coords offset、completion wid/group 串扰 | 后续可选做 multi-WG tensor mixed long；当前不阻塞 FULL group fence 和 fallback diagnostic |

### 11.4 P0：优先补的测试

P0-A：扩 `s2g_tensor_matrix_case.c`，把 tensor S2G 从 4 个基础 case 升级成 matrix-driven functional suite。当前第一轮已经落地在 `dma_tma_s2g_func_test` 的 `tma_s2g` suite 中，没有另开独立性能测试，继续复用现有编译和 verdict 机制。

| case | 状态 | 目的/校验 |
|---|---|---|
| `rank1_32_fp32_full_line` | 已落地 | 保留 PutFull 基础 case |
| `rank1_64_fp32_two_lines` | 已落地 | 多 line first/last lifecycle |
| `rank1_24_fp32_partial_tail` | 已落地 | tail partial，guard 不被改 |
| `rank2_4x4_origin` | 已落地 | origin offset 正确 |
| `rank2_16x16_contiguous` | 已落地 | 主流 FP32 linear tile |
| `rank2_subbox_8x8_at_2_2` | 已落地 | coords/subbox global offset 正确 |
| `rank2_padded_rows_stride64` | 已落地 | row stride 不等于 row bytes |
| `rank2_element_stride2_cols` | 已落地 | non-unit element stride 不误走 contiguous fast path |
| `rank3_plain_2x2x2` | 已落地 | rank3 地址生成 |
| `rank4_plain_small` | 已落地 | rank4 decode/setup |
| `rank5_plain_small` | 已落地 | rank5 decode/setup |
| `interleave16_rank3` | 已落地 | interleave16 地址映射和 slice-level line task |
| `interleave32_rank3` | 已落地 | interleave32 地址映射和 slice-level line task |
| `rank2_oob_dim0` | 已落地 | dim0 OOB suppress，guard 保持 |
| `rank2_oob_dim1` | 已落地 | dim1 OOB suppress，guard 保持 |
| `descriptor_cache_A/B/C/A` | 已落地 | 2-entry descriptor cache fill/replacement/reuse |
| `rank2_swizzle32/64/128` | 已落地 | swizzle 地址映射和 swizzled shared gather line task |
| `rank3_subbox_2x2x2_at_1_1_1` | 已落地 | rank3 subbox offset |
| `rank3_oob_dim2_suppress` | 已落地 | 高维 OOB suppress |
| `rank2_oob_subbox_partial` | 已落地 | subbox partial OOB |
| `rank3_long_4x4x4` | 已落地 | rank3 larger shape linearization |
| `rank3_stride_oob_long` | 已落地 | stride + OOB + long shape |
| `rank4_long_4x4x2x2` | 已落地 | rank4 long shape |
| `rank5_long_4x2x2x2x2` | 已落地 | rank5 long shape |
| `interleave32_long_rank3` | 已落地 | interleave32 long shape 地址映射 |
| `swizzle128_oob_stride` | 已落地 | swizzle128 + OOB + stride fallback |
| `u32_rank2_4x4_alias` | 已落地 | U32 datatype enum alias |
| `i32_rank1_32_alias` | 已落地 | I32 datatype enum alias |

tensor S2G matrix 的 host 期望值生成要与 G2S matrix 分开处理：

1. G2S OOB 是 fill；S2G OOB 是 suppress store。
2. 目的 global buffer 先填 guard pattern，例如 `0xcdcdxxxx`。
3. 对每个 logical tensor element，若坐标 in-bound，则把 shared source 对应 word 写到 expected global；若 OOB，则 expected 保持 guard。
4. descriptor cache case 不允许 hot-update 同一 descriptor line；需要显式使用 A/B/C 不同 descriptor line，测试 cache hit/replacement，而不是未定义一致性。
5. full-line/feature case 可以顺带检查 PMU 形态，例如 rank1/rank2 的 PutFull 或 swizzle/interleave 的 PutPart 比例，但 PASS/FAIL 仍以数据正确为准。

P0-B：扩 bulk S2G backend stress。当前第一轮已经在 `shared_to_global_dma_test.cl`、`s2g_bulk_matrix_case.c` 和 host stress driver 中落地。

| case | 状态 | 主要验证 |
|---|---|---|
| `cross_source_line_same_dst_line` | 已落地 | source split 后 byte order/mask 正确 |
| `cross_source_and_dst_line` | 已落地 | 同时跨 source/destination line |
| `line_entry_pressure_6_lines` | 已落地 | line table 满后能 backpressure 并恢复 |
| `line_entry_pressure_8_lines` | 已落地 | 多轮 issue/ack 回收 |
| `group_empty_commit` | 已落地 | group state 不 underflow |
| `dst_cross_page` | 已落地 | global dst 从 page 尾跨到下一页 |
| `group_wrap_wait0123` | 已落地 | group ring wrap、wait_group0/1/2/3 和 source reuse |
| `group_multi_issue_wrap` | 已落地 | 一个 committed group 内 3 条 S2G，wrap 前必须等待该 group 的全部 FULL completion，防止 group slot 过早复用 |
| `cross_page_tlb_ABC` | 已落地 | A/B/C 三页写回后复用 A 页，验证跨页 TLB route 与 completion 顺序 |
| `fixed_mask_dst4_20B` | 已落地 | 非 128B 对齐 dst offset 和短 partial mask |
| `fixed_mask_dst28_100B` | 已落地 | dst offset 28B、100B partial 跨 word/line mask |
| `fixed_mask_src124_dst4_64B` | 已落地 | source line 尾跨线、destination partial line |
| `fixed_mask_dst124_132B` | 已落地 | destination line 尾跨线，跨两条 line 的 partial/full 组合 |
| `mask_seed00..07_*` | 已落地 | 8 个固定 seed 的 srcOffset/dstOffset/bytes 组合，覆盖 4B 到 256B 的 deterministic mask matrix |

same-line overlap directed case 已从测试集中删除。当前契约不再把无 wait/barrier 的同 destination byte 覆盖定义为硬件保证；需要覆盖顺序时由 kernel 显式使用 wait/barrier 划分可见性边界。

P0-C：补 mixed completion/routing functional。当前第一轮已经落地到 `dma_tma_s2g_func_test` 的 `mixed_s2g_routing` suite 中，作为 S2G 自己的 mixed completion/routing 回归入口。

| case | 状态 | 组合 | 主要验证 |
|---|---|---|---|
| `s2g_routing_conflict` | 已落地 | S2G shared grouped read + normal local memory same-bank noise | `instrId>=2` shared response 不被 normal/shared write response 干扰 |
| `dma_group_keep1_preserves_newer_s2g` | 已落地，四后端 directed PASS | G2S group 后接 S2G group，再 `wait_group1` | 统一 group ring 能等待更老 G2S，同时保留最新 S2G 后台完成 |
| `dma_group_wait0_drains_g2s_s2g` | 已落地，四后端 directed PASS | G2S group + S2G group 后 `wait_group0` | wait_group0 drain 所有已提交 DMA group，不按方向分域 |
| `bulk_tensor_s2g_same_fence` | 已落地 | bulk S2G + tensor S2G before one fence | bulk ack route 与 tensor synthetic completion 都正确 |
| `bidirectional_descriptor_mix` | 已落地 | bulk G2S、tensor G2S prefetch、bulk S2G、tensor S2G、normal marker | TLB owner、L2 route、completion arbiter 全路径 |

P0-D：扩 multi-WG S2G。当前第四轮已经把 4WG 基础、2WG group wait、2WG tensor subbox、multi-WG cross-page、4WG group/page、4WG mixed outstanding、tensor stride/OOB、tensor descriptor mix、tensor group/page 放入默认 multi-WG gate。

| case | 状态 | 主要验证 |
|---|---|---|
| `bulk_s2g_4wg` | 已落地 | WG id/offset 不串扰 |
| `tma_s2g_4wg` | 已落地 | 每 WG 独立 descriptor/coords/dst |
| `bulk_s2g_2wg_group_wait` | 已落地 | group state 按 warp/WG 隔离 |
| `tma_s2g_2wg_subbox` | 已落地 | 每 WG 不同 coords/subbox |
| `multi_wg_cross_page` | 已落地 | TLB owner 和跨页 route 隔离 |
| `bulk_s2g_4wg_group_page` | 已落地 | 4WG 各自跨 4 个 sparse page commit/wait，验证 group/page/TLB owner 隔离 |
| `mixed_g2s_s2g_4wg` | 已落地 | 4WG 下 G2S 与 bulk S2G 同时 outstanding，分别检查 S2G 写回和 G2S readback |
| `tma_s2g_2wg_stride_oob` | 已落地 | 2WG tensor S2G element stride 与 OOB suppress，校验未写区域 guard |
| `tma_s2g_4wg_descriptor_mix` | 已落地 | 4WG 使用不同 rank/shape/subbox/stride descriptor，验证 descriptor/coords/WG offset 隔离 |
| `tma_s2g_4wg_group_page` | 已落地 | 4WG tensor S2G sparse page 写回，验证 tensor group/page/TLB owner 隔离 |

### 11.5 P1/P2：稳定后再补

P1 是有真实 RTL 风险、但可以排在 P0 之后的覆盖：

| 类别 | case |
|---|---|
| 随机 mask/offset | 8 个 fixed seed 已落地；后续只保留少量新增 fixed seed，结果写入 deterministic expected |
| tensor datatype alias | U32/I32 4B alias 已落地；后续非 4B datatype 放 negative suite，不进入默认回归 |
| fallback long shape | rank3/4/5 long shape 已落地；后续重点转到 multi-WG tensor fallback/OOB |
| descriptor prefetch 语义 | 如果后续公开 tensor S2G prefetch，再补 prefetch hit/miss/fence case |
| nocache route | 对 P0 中 routing/mixed 选 2 到 3 个重点 case 跑 rtl-nocache |

P2 是更偏压力/随机化的长期项：

| 类别 | case |
|---|---|
| fuzz | 随机 tensor dims/coords/box/stride，限制在当前 RTL 支持范围内 |
| negative/assert | 非 4B datawidth、非法 rank/interleave 只放在专用 negative suite，不进入默认回归 |
| long-run | 多 kernel 连续运行，检查 descriptor cache 和 TLB owner 状态不跨 kernel 产生错误状态 |

### 11.6 建议执行顺序和验收口径

测试可信度审计和分级已经落到当前 registry，P0-A/P0-B/P0-C/P0-D 覆盖项都已经落地；deterministic mask/offset、tensor long-shape/datatype alias、4WG group/page、mixed G2S/S2G outstanding、tensor multi-WG stride/OOB、descriptor mix、小规模 fixed-seed fuzz 和 `mixed_tensor_outstanding_long` 都在当前护栏内。当前不需要再证明“是否能测 S2G”，而是应该把这些测试作为后续 group fence、高级 feature path、fallback coalescing 或 fast setup/FSM 优化的共同护栏。

执行顺序：

1. 保持当前 `directed` gate：RTL/rtl-nocache 下 3 个 testcase directory 必须继续 0 fail、0 skip。
2. FULL-only group fence 或高级特性相关修改前，先跑 focused 抽样：bulk S2G、`S2G_TENSOR_CASE_FILTER=fuzz_tensor`、`S2G_MIXED_CASE_FILTER=mixed_tensor_outstanding_long`。
3. 已有 interleave/swizzle/interleave+swizzle/stride/OOB/high-rank diagnostic perf 应继续作为趋势监控；下一步若做 RTL 性能优化，优先用该测试监控 segmented PutPart 合并是否有效，同时确认 stride/OOB/rank3/4/5 不退回逐元素 fallback。
4. 每完成一组 RTL 改动，都先跑 focused 抽样，再跑 GVM/GVM-nocache/RTL/RTL-nocache `directed` gate。
5. directed gate 稳定后，再抽样跑对应 perf 代表点；只有趋势明显改善且功能稳定时再 full sweep。
6. 后续如果实际 bug 指向 multi-WG tensor mixed long 或 high-rank swizzle/OOB，再定向补测试，不把默认 gate 扩成随机长跑。

验收口径：

```bash
source env.sh
./build-ventus.sh --build "rtlsim"

cd testcases/_get_case
./run_dma_tma_rtl.sh --suite directed --backend rtl --run-jobs 4
./run_dma_tma_rtl.sh --suite directed --backend rtl-nocache --run-jobs 4
```

通过标准：

1. 当前 3 个 directed testcase directory 继续 PASS：`dma_tma_g2s_func_test`、`dma_tma_s2g_func_test`、`dma_tma_multi_wg_func_test`。
2. `bulk_s2g`、`bulk_s2g_small_dual`、`bulk_s2g_stress`、`tma_s2g` 默认 suite 继续 PASS。
3. `dma_tma_s2g_func_test` 子 case 维持当前基线：bulk matrix 25/25、small dual 2/2、bulk stress 14/14、tensor matrix 37/37、mixed routing 6/6。
4. `dma_tma_multi_wg_func_test` 维持当前基线：RTL/rtl-nocache 15/15 suite PASS，其中包含 4WG bulk/tensor S2G、2WG group wait、2WG tensor subbox、cross-page、4WG group/page、mixed outstanding、tensor stride/OOB、tensor descriptor mix 和 tensor group/page。
5. 新增 P0 suite 在 RTL PASS；关键路由/跨页 case 至少抽样在 rtl-nocache PASS。
6. GVM 可用于纯功能快速参考，但不作为 concurrency/性能最终判断；multi-WG 中 4WG group/page、mixed outstanding 与新增 tensor high-stress case 在 GVM/Spike 下按现有策略 skip。
7. perf 测试只作为“没有明显误伤”的趋势参考，不替代 directed functional verdict。
8. 所有新增 case 都要有 guard pattern，检查未写区域保持不变，尤其是 OOB suppress 和 partial Put。

### 11.7 测试可信度与分级

这部分回答当前 DMA/TMA 测试应该如何分层使用。结论不是简单的“全部作为 gate”，而是要按用途分层：功能测试统一作为 gate；性能测试只做趋势；GEMM/serial 这类复杂路径只做诊断或应用级参考。不再单独保留 smoke 设计。

可信度判断标准：

| 维度 | 可靠测试应满足 | 风险信号 |
|---|---|---|
| oracle | host 端独立生成 expected，而不是复用被测路径结果 | 只比较两个硬件路径，或只看没有 crash |
| guard | 未写区域有 sentinel/guard，能发现越界写、OOB 写、partial mask 错 | 只校验有效 payload，不看周边 |
| backend skip | skip 条件明确，RTL gate 不应因为 skip 返回 OK 而误判完整通过 | Spike/GVM/RTL 行为差异被 summary OK 掩盖 |
| baseline | perf baseline 是 warp/workgroup cooperative，不是 lane0 串行 | baseline 与 opt workload 不等价 |
| measurement | PMU cycles 和 host ns 分清，host ns 只作辅助 | PMU parse 失败后仍用 host ns 得出强结论 |
| isolation | perf 子进程隔离，避免 GVM/POCL 临时状态串扰 | 多 case 共用状态导致偶发 |
| scope | 测试目的清楚：functional gate、perf trend、diagnostic 不混用 | perf 测试被当成功能覆盖充分性证明 |

当前 `cases_dma_tma.csv` 中测试的可信度分级：

| 测试 | 建议等级 | 靠谱的部分 | 不能承担的结论 |
|---|---|---|---|
| `dma_tma_g2s_func_test` | functional gate | G2S bulk/tensor/descriptor/matrix/routing/mixed/multi-warp 都有 host expected；matrix 覆盖 datatype、rank、subbox、stride、swizzle、OOB；本次 RTL/rtl-nocache 都是 pass 10, fail 0, skip 0 | 只能证明当前 G2S/TMA directed 形态；不能替代 S2G 专门的 OOB/mask/page stress |
| `dma_tma_s2g_func_test` | functional gate, coverage 已显著补强 | bulk matrix 25 子 case、small dual 2 子 case、bulk stress 14 子 case、tensor matrix 37 子 case、mixed routing 6 子 case；本次四后端 directed gate 都通过 | 已足够作为 FULL group fence、feature path、fallback 边界、后续 tensor S2G fast setup/FSM 优化的功能护栏；perf 结论仍需 pingpong/feature sweep |
| `dma_tma_multi_wg_func_test` | multi-WG functional gate, coverage 已补到 group/page/mixed outstanding/tensor stress 第一轮 | 有 2WG/4WG bulk/tensor G2S/S2G、2WG group wait、2WG tensor subbox、cross-page、4WG group/page、4WG mixed outstanding、2WG tensor stride/OOB、4WG tensor descriptor mix、4WG tensor group/page，按 WG 独立 expected 校验；本次 RTL/rtl-nocache 都是 pass 15, fail 0, skip 0 | 后续可选补 multi-WG tensor mixed long；GVM/Spike 下 5 个高压力路径按策略跳过 |
| `dma_tma_g2s_pingpong_perf_test` | perf trend | child run 隔离；manual/TMA 都做 CPU ref check；tile/stage sweep 与当前 S2G perf 统一；当前 full sweep 25/25 PASS | 只覆盖 rank2 FP32 连续 tile；PMU cycle parse 是性能统计，不是功能覆盖；不能替代 G2S matrix |
| `dma_tma_tensor_s2g_pingpong_perf_test` | S2G perf trend | manual input + tensor S2G、TMA G2S input + tensor S2G 两条默认 sweep 都有 CPU ref；tile/stage 与 G2S pingpong 对齐；当前两条 full sweep 均 25/25 PASS | perf 不是 directed 功能覆盖；主要是 rank2 FP32 linear，不能证明 swizzle/interleave/stride/OOB feature 边界 |
| `dma_tma_movement_profile_test` | movement diagnostic perf | 测试内容是纯 movement microbench，不再复用 pingpong kernel，不再有 buffer/stage 参数或 compute 阶段；bulk/tensor G2S 和 bulk/tensor S2G 都用同 tile manual baseline 做 PMU cycle A/B；当前四后端 standalone sweep PASS | S2G-only kernel 内仍需初始化 shared source；host ns 只作辅助；主要用于定位单次 movement path，不替代 pipeline overlap perf |
| `tma_gemm_perf_test` | application diagnostic / experimental perf，已从默认 DMA/TMA registry 移除 | manual/TMA 都和 CPU GEMM reference 比较；后续可作为应用级单独诊断 | 当前 manual GEMM 自身失败，不能进入 DMA/TMA correctness gate；并行多 WG 路径也不适合作为基础 gate |
| `tma_roundtrip_pipeline_perf_test` | 已物理删除 | 曾用于单 WG buffer count sweep | 与 pingpong 主语义重复；当前 tensor S2G 路径不真正利用 buffer 维度，后续改为独立 `dma_tma_multi_wg_roundtrip_perf_test` 更合适 |
| `dma_tma_s2g_pipeline_perf_test` | 已物理删除 | 曾用于 bulk S2G 单向 perf | bulk perf 不再作为默认趋势项；保留 bulk 功能 stress，把性能趋势收敛到 tensor S2G pingpong |

更细的判断：

1. `dma_tma_g2s_func_test` 是 G2S/TMA 路线的功能基准。RTL/rtl-nocache 下没有 silent skip，可以作为 S2G 修改后的 G2S 强回归。
2. `dma_tma_multi_wg_func_test` 方向正确，当前已经证明 2WG/4WG 的 bulk/tensor G2S/S2G 基本隔离、group/page、mixed outstanding 和 tensor stride/OOB/descriptor mix；后续可选做 multi-WG tensor mixed long，但它不阻塞 FULL group fence 固化和高级特性性能诊断。
3. `dma_tma_g2s_pingpong_perf_test` 和 `dma_tma_tensor_s2g_pingpong_perf_test` 算“可信 perf microbench”：有 CPU ref、child 隔离、PMU 优先。但它们只证明当前测到的连续 FP32 pipeline 没算错，不能替代 directed coverage。
4. `tma_roundtrip_pipeline_perf_test` 和 `dma_tma_s2g_pipeline_perf_test` 已按测试体系整理决策删除；对应语义分别由 tensor S2G pingpong、bulk directed stress 和未来 multi-WG roundtrip perf 承接。
5. `tma_gemm_perf_test` 不应该拿来判断 DMA/TMA RTL 是否整体正确。它仍可作为应用级诊断源码保留，但已经从默认 DMA/TMA registry 中移除。
6. `dma_tma_movement_profile_test` 更像 movement 分析仪表，不是回归门槛。它的价值是比较 bulk/tensor G2S/S2G 的单次搬运 PMU cycle，不证明 pipeline overlap 收益。
7. `run_dma_tma_rtl.sh` 的注册和分流机制是有用的。当前文档已经给出可信等级；后续可以把这个等级落到 CSV 或 runner summary 中，例如 `gate`, `perf-trend`, `diagnostic`。

测试加固建议：

| 优先级 | 加固项 | 目标 |
|---|---|---|
| P0 | 把可信等级表固化到 CSV 或 runner summary | 避免把 GEMM/serial/perf 误当功能 gate |
| P0 | RTL directed gate 继续对 skip 做检查 | RTL/rtl-nocache 下关键 suite 不允许 silent skip；本次 directed gate 已做到 0 skip |
| P0 | 保留 G2S func 作为老路径强回归 | S2G 修改不能误伤 G2S |
| P1 | multi-WG S2G/TMA 继续扩 mixed tensor outstanding | 让 multi-WG 从 group/page 压力覆盖进入 G2S+bulk S2G+tensor S2G 长序列覆盖 |
| P1 | perf 测试报告中明确 `functional_check` 与 `performance_metric` 分离 | 跑慢不等于功能错，算错必须 fail |
| P2 | 给 `cases_dma_tma.csv` 增加 quality tier 或在 runner summary 里输出 tier | 后续自动化更清楚 |

基于这个审计，后续推荐的回归门槛是：

| 门槛 | 测试 |
|---|---|
| 必跑功能 gate | `./run_dma_tma_rtl.sh --suite directed --backend rtl --run-jobs 4` 和 `./run_dma_tma_rtl.sh --suite directed --backend rtl-nocache --run-jobs 4` |
| 默认全量 DMA/TMA registry | `./run_dma_tma_rtl.sh --suite all --backend rtl --run-jobs 4 --jobs 8 --timeout 7200` |
| G2S 性能趋势 | `dma_tma_g2s_pingpong_perf_test sweep` |
| S2G 性能趋势 | `dma_tma_tensor_s2g_pingpong_perf_test sweep manual_tensor_s2g` 与 `sweep tma_tensor_s2g` |
| 非门槛诊断 | `dma_tma_movement_profile_test` movement profile；`tma_gemm_perf_test` 作为 registry 外应用诊断 |

### 11.8 并发执行设计

功能测试本身压力不算大，不需要再设计独立 smoke 层来节省时间。更合理的加速方式是提高测试目录级并发。

当前建议：

| 项 | 设计 |
|---|---|
| 并发粒度 | testcase directory 级并发，不在单个测试进程内部拆 case |
| 单测试资源估计 | 当前一个 RTL/GVM 测试大约消耗 8 个 CPU core |
| 最大并发 | 4 个测试目录 |
| 总资源预算 | 约 32 个 CPU core，对当前服务器压力不高 |
| runner 参数 | `./run_dma_tma_rtl.sh --run-jobs 4 ...` |
| 同目录保护 | runner 已避免同一 testcase directory 同时跑两个 child |
| 运行缓存 | 每个 child 使用独立 `POCL_CACHE_DIR` 和 `TMPDIR/TMP/TEMP`，避免并发 OpenCL build 共享默认 cache |

推荐命令：

```bash
source env.sh
cd testcases/_get_case
./run_dma_tma_rtl.sh --suite directed --backend rtl --run-jobs 4
./run_dma_tma_rtl.sh --suite directed --backend rtl-nocache --run-jobs 4
```

注意：

1. 这个并发设计用于功能 gate；perf full sweep 仍应按实验目的选择并发，否则不同 child 的 host ns 会互相影响。
2. primary perf metric 仍应优先看 PMU active cycles；host ns 在并发下更只能作为辅助。
3. 如果后续新增更重的 case，可以按测试目录给 runner 加 tag 或 quality tier，而不是恢复 smoke 层。
4. `run_dma_tma_rtl.sh` 的 `--run-jobs` 上限已从 3 放宽到 4，和这个资源预算一致。
5. runner 会在每个 testcase 的 `log/` 目录下创建独立 POCL cache/temp 子目录；这是并发 RTL/GVM gate 的必要条件，否则默认 `~/.cache/pocl` 或全局 temp 状态可能导致 `mkstemp()` 失败，表现为 `CL_BUILD_PROGRAM_FAILURE`，不是 RTL 数据通路错误。
6. 不建议把 `rtl` 和 `rtl-nocache` 两个 full gate 同时启动。2026-07-02 实测跨 backend 并发时，RTL `dma_tma_s2g_func_test` 曾在已经通过主要 case 后以 rc=134 abort；随后单独重跑 S2G RTL 和单独重跑完整 RTL directed gate 都通过，因此该现象归为跨 backend 仿真资源冲突，而不是 S2G 功能回归。

当前实测：

| 命令 | testcase dirs | 顶层 case | fail | skip | 耗时明细 |
|---|---:|---:|---:|---:|---|
| `./run_dma_tma_rtl.sh --suite directed --backend gvm --run-jobs 4 --jobs 8 --timeout 1800` | 3/3 pass | 30 pass | 0 | 0 | G2S 63s, S2G 58s, multi-WG 10s |
| `./run_dma_tma_rtl.sh --suite directed --backend gvm-nocache --run-jobs 4 --jobs 8 --timeout 1800` | 3/3 pass | 30 pass | 0 | 0 | G2S 44s, S2G 42s, multi-WG 7s |
| `./run_dma_tma_rtl.sh --suite directed --backend rtl --run-jobs 4 --jobs 8 --timeout 1800` | 3/3 pass | 30 pass | 0 | 0 | G2S 45s, S2G 54s, multi-WG 11s |
| `./run_dma_tma_rtl.sh --suite directed --backend rtl-nocache --run-jobs 4 --jobs 8 --timeout 1800` | 3/3 pass | 30 pass | 0 | 0 | G2S 205s, S2G 222s, multi-WG 31s |

这两个 gate 覆盖 `dma_tma_g2s_func_test`、`dma_tma_s2g_func_test`、`dma_tma_multi_wg_func_test`。其中 `g2s` 是 G2S/TMA 路线强回归，`s2g` 是 S2G 路线主 gate，`multi_wg` 用来检查多工作组下 G2S/S2G 混合行为。

## 12. 全量 DMA/TMA 测试运行记录

本节记录当前 `cases_dma_tma.csv` 注册 DMA/TMA 测试的最新有效运行结果。当前 registry 包含 7 个测试目录、8 个 mode：3 个 directed functional gate、3 个主性能 full sweep、1 个 movement profile 和 1 个 feature diagnostic。

注意：四个后端不建议同时启动四个 `run_dma_tma_rtl.sh`，因为各 testcase directory 会共享 OpenCL/POCL 构建中间文件和部分日志命名。可信跑法是按后端顺序运行，每个后端内部使用 `--run-jobs 4 --jobs 8`。

### 12.1 Directed Gate

| command | backend | result | 参考用时 |
|---|---|---|---:|
| `./run_dma_tma_rtl.sh --suite directed --backend gvm --run-jobs 4 --jobs 8 --timeout 1800` | GVM | 3/3 PASS | 63s |
| `./run_dma_tma_rtl.sh --suite directed --backend gvm-nocache --run-jobs 4 --jobs 8 --timeout 1800` | GVM-nocache | 3/3 PASS | 47s |
| `./run_dma_tma_rtl.sh --suite directed --backend rtl --run-jobs 4 --jobs 8 --timeout 1800` | RTL | 3/3 PASS | 52s |
| `./run_dma_tma_rtl.sh --suite directed --backend rtl-nocache --run-jobs 4 --jobs 8 --timeout 1800` | RTL-nocache | 3/3 PASS | 48s |

Directed gate 覆盖：

| testcase | 主要内容 |
|---|---|
| `dma_tma_g2s_func_test` | bulk/tensor G2S、TMA matrix、descriptor/prefetch、datatype/rank/subbox/stride/swizzle/OOB、routing/mixed/multi-warp fence |
| `dma_tma_s2g_func_test` | bulk S2G matrix、small dual、bulk stress、tensor S2G matrix、mixed routing、unified DMA group keep/wait0、long outstanding |
| `dma_tma_multi_wg_func_test` | 2WG/4WG bulk/tensor G2S/S2G、group/page、mixed outstanding、tensor stride/OOB、descriptor mix |

### 12.2 RTL Perf Registry

最新 RTL 性能全量命令：

```bash
./run_dma_tma_rtl.sh --suite perf --backend rtl --run-jobs 4 --jobs 8 --timeout 2400
```

结果为 5/5 PASS：

| mode label | command | result | time | key metric / note |
|---|---|---|---:|---|
| `dma_tma_g2s_pingpong_perf_full` | `./dma_tma_g2s_pingpong_perf_test.out sweep` | PASS | 243s | 25 点，min/avg/max = 1.1214x/1.3703x/1.4998x |
| `dma_tma_tensor_s2g_pingpong_manual_input_perf_full` | `./dma_tma_tensor_s2g_pingpong_perf_test.out sweep manual_tensor_s2g` | PASS | 239s | 25 点，min/avg/max = 1.0647x/1.2493x/1.3601x |
| `dma_tma_tensor_s2g_pingpong_tma_input_perf_full` | `./dma_tma_tensor_s2g_pingpong_perf_test.out sweep tma_tensor_s2g` | PASS | 258s | 25 点，min/avg/max = 1.2068x/1.9379x/2.4807x |
| `dma_tma_movement_profile_full` | `./dma_tma_movement_profile_test.out sweep` | PASS | 62s | RTL standalone sweep：40 child PASS；bulk_g2s min/avg/max = 1.6402x/3.8073x/7.1862x，tensor_g2s = 1.3628x/3.4645x/6.8325x，bulk_s2g = 1.2336x/1.3412x/1.4336x，tensor_s2g = 1.1374x/1.2920x/1.4237x |
| `dma_tma_s2g_feature_perf_sweep` | `./dma_tma_s2g_feature_perf_test.out sweep 8` | PASS | 17s | PMU total active cycles = 154744；paired host ratio 约 0.973x 到 1.144x |

结论：

1. 当前 RTL 性能 registry 没有相对文档基线的明显下降。
2. 主性能结论仍以 RTL PMU active cycles 为准；host ns 只做辅助。
3. feature diagnostic 不是主性能门限，但能及时暴露 swizzle/interleave/stride/OOB 组合是否退回逐元素 fallback。
4. `tma_gemm_perf_test` 不属于当前 DMA/TMA registry；GEMM 应作为后续应用级诊断单独处理。

## 13. 附录

<details>
<summary>12.1 常用命令</summary>

构建：

```bash
source env.sh
./build-ventus.sh --build "rtlsim"
```

功能测试：

```bash
source env.sh
cd testcases/_get_case/dma_tma_s2g_func_test
make
VENTUS_BACKEND=rtl ./dma_tma_s2g_func_test.out tma_s2g
VENTUS_BACKEND=rtl ./dma_tma_s2g_func_test.out bulk_s2g_stress
VENTUS_BACKEND=rtl-nocache ./dma_tma_s2g_func_test.out tma_s2g
```

性能代表点：

```bash
source env.sh

cd testcases/_get_case/dma_tma_g2s_pingpong_perf_test
VENTUS_BACKEND=rtl ./dma_tma_g2s_pingpong_perf_test.out single 16 16 2 4

cd ../dma_tma_tensor_s2g_pingpong_perf_test
VENTUS_BACKEND=rtl ./dma_tma_tensor_s2g_pingpong_perf_test.out single tma_tensor_s2g 16 16 2 4
VENTUS_BACKEND=rtl ./dma_tma_tensor_s2g_pingpong_perf_test.out single tma_tensor_s2g 64 64 2 16
```

full sweep：

```bash
source env.sh

cd testcases/_get_case/dma_tma_tensor_s2g_pingpong_perf_test
VENTUS_BACKEND=rtl ./dma_tma_tensor_s2g_pingpong_perf_test.out sweep manual_tensor_s2g
VENTUS_BACKEND=rtl ./dma_tma_tensor_s2g_pingpong_perf_test.out sweep tma_tensor_s2g

cd ../dma_tma_g2s_pingpong_perf_test
VENTUS_BACKEND=rtl ./dma_tma_g2s_pingpong_perf_test.out sweep

cd ../dma_tma_movement_profile_test
VENTUS_BACKEND=rtl ./dma_tma_movement_profile_test.out sweep
```

日志很长，建议用 `rg` 过滤：

```bash
rg "PASS|FAIL|summary|REPORT|S2G PERF|dma fence/group wait|Assertion|error" log/*.log
```

</details>

<details>
<summary>12.2 资源预算</summary>

| 资源 | 当前建议 | 原因 |
|---|---:|---|
| bulk line entries | 4 | 128B line image 是主要面积项 |
| dynamic shared read entries | 不超过 `lsu_nMshrEntry - 2` | `instrId` 0/1 留给 legacy/ tensor fallback |
| tensor descriptor cache | 2 | 对齐 G2S `tma_desc_cache_entries`，不复制大 cache |
| tensor line-task FIFO | 0 到 2 | 只在 fast setup 后证明需要时增加 |
| group ring | 小 entry ring | 存 `{group, count}` 类 metadata，不存 data |

不要把 line buffer 默认扩到 16 或 32；如果后续 full sweep 证明 line table 是瓶颈，再考虑从 4 提到 6，而不是直接大幅增加。

</details>

<details>
<summary>12.3 当前性能快照</summary>

当前 RTL full sweep 可信口径：

| path | min | avg | max |
|---|---:|---:|---:|
| G2S + manual writeback | 1.1214x | 1.3703x | 1.4998x |
| manual input + tensor S2G writeback | 1.0647x | 1.2493x | 1.3601x |
| TMA G2S + tensor S2G | 1.2068x | 1.9379x | 2.4807x |

当前 feature diagnostic 可信口径：

| metric | value |
|---|---:|
| RTL feature sweep result | PASS |
| runtime | 17s |
| PMU total active cycles | 154744 |
| paired host ratio range | 0.973x - 1.144x |

GVM 用于功能参考，不作为 RTL tensor perf 结论。主性能判断以 RTL PMU active cycles 和 full sweep speedup 为准。

</details>

<details>
<summary>12.4 Directed Stress 后续清单</summary>

bulk S2G：

| 类别 | 待补 |
|---|---|
| routing conflict | single-WG 长 mixed outstanding 已落地；后续可选 multi-WG tensor mixed long |
| random mask/offset | fixed mask、deterministic mask 和 4 个 fixed-seed offset/size 已落地；后续只按 bug 定向补 |
| group/page | 不同 group 与跨页/TLB owner 组合 |
| group/page | 4WG group wait + sparse page/TLB owner 组合 |

tensor S2G：

| 类别 | 待补 |
|---|---|
| row-level boundary | rank1/rank2 no-permute row、partial row、cross-line row segment |
| fallback boundary | fixed-seed swizzle/OOB/stride 已落地；后续按 feature path 变更定向复跑 |
| OOB | 多 WG mixed OOB suppress，继续检查 guard 不变 |
| rank | rank3/4/5 更大 shape 的 row coalescing 边界 |
| mixed outstanding | single-WG G2S + bulk S2G + tensor S2G 已落地；multi-WG 版可选 |

</details>

<details>
<summary>12.5 最新测试记录</summary>

本节记录当前文档版本引用的最新构建、功能 gate 和 RTL perf registry 结果。

```text
Build:
  ./build-ventus.sh --build "rtlsim"
  RTL cache build:    PASS
  RTL-nocache build: PASS

Directed gate:
  GVM:         3/3 PASS, longest case 63s
  GVM-nocache: 3/3 PASS, longest case 47s
  RTL:         3/3 PASS, longest case 52s
  RTL-nocache: 3/3 PASS, longest case 48s

RTL perf registry:
  command:
    ./run_dma_tma_rtl.sh --suite perf --backend rtl --run-jobs 4 --jobs 8 --timeout 2400
  result:
    5/5 PASS
  runtime:
    dma_tma_g2s_pingpong_perf_full:                243s
    dma_tma_s2g_writeback_manual_tensor_s2g_full:  239s
    dma_tma_s2g_writeback_tma_tensor_s2g_full:     258s
    dma_tma_movement_profile_full:                 62s
    dma_tma_s2g_feature_perf_sweep:                17s

Perf representative:
  G2S pingpong full sweep: 1.1214x/1.3703x/1.4998x
  manual input + tensor S2G full sweep: 1.0647x/1.2493x/1.3601x
  TMA G2S input + tensor S2G full sweep: 1.2068x/1.9379x/2.4807x
  DMA/TMA movement profile: 40 child paths PASS

Feature paired host feature/base from RTL sweep 8:
  interleave16 C8_W3_N2: 1.103x
  interleave32 C16_W2_N2: 1.025x
  swizzle32 8x16: 1.005x
  swizzle32 16x16: 1.144x
  swizzle64 16x16: 0.981x
  swizzle128 16x16: 0.983x
  swizzle128 32x8: 0.989x
  interleave32+swizzle32 C8_W8_N4: 1.001x
  interleave32+swizzle32 C16_W4_N2: 0.973x
  PMU total active cycles: 154744
```

</details>

<details>
<summary>12.6 调研对象与文件地图</summary>

本节保留原调研报告中的文件级定位，方便后续按文档直接改代码。

| 类别 | 文件/目录 | 用途 |
|---|---|---|
| DMA core | `gpgpu/ventus/src/pipeline/DMA_core.scala` | DMA 输入 FIFO、G2S/Tensor G2S、bulk S2G、tensor S2G、TLB/L2/shared/completion 路由 |
| bulk S2G | `gpgpu/ventus/src/pipeline/DMA_s2g.scala` | `CP_ASYNC_BULK_S2G` 的 line-based backend |
| tensor S2G | `gpgpu/ventus/src/pipeline/DMA_tma_s2g.scala` | `CP_ASYNC_TENSOR_S2G` descriptor path、fast path、fallback |
| G2S 参考 | `gpgpu/ventus/src/pipeline/DMA_core.scala` 中 `AddrCalc_l2cache` 与 `Temp_mem` | descriptor cache、prefetch、line response、tempmem completion 是 S2G 设计参考 |
| decode/issue | `DecodeUnit.scala`, `issue.scala`, `pipe.scala` | DMA 指令识别、issue 信息传递、`dma_group`/direction bit |
| scheduler | `warp_schedule.scala`, `scoreboard.scala` | async DMA completion、fence、group wait |
| PMU | `PerfCounters.scala`, `GPGPU_top.scala`, `GPGPU_top_nocache.scala` | S2G counters、pipeline stall、program/testcase summary |
| 参数 | `top/parameters.scala` | line entry、shared read entry、descriptor cache、group entry 等参数 |
| build | `build-ventus.sh` | RTL simulator 构建脚本 |
| env | `env.sh` | Ventus 工具链和运行环境变量 |
| case registry | `testcases/_get_case/cases_dma_tma.csv` | DMA/TMA case/tag/backend 注册 |
| S2G directed | `testcases/_get_case/dma_tma_s2g_func_test/` | bulk/tensor S2G 功能测试 |
| G2S directed | `testcases/_get_case/dma_tma_g2s_func_test/` | G2S 功能和 mixed async 参考 |
| multi-WG directed | `testcases/_get_case/dma_tma_multi_wg_func_test/` | 多 WG G2S/S2G 功能测试 |
| G2S pingpong perf | `testcases/_get_case/dma_tma_g2s_pingpong_perf_test/` | G2S input 优化趋势，manual writeback 对照 |
| tensor S2G pingpong perf | `testcases/_get_case/dma_tma_tensor_s2g_pingpong_perf_test/` | manual/G2S input 与 tensor S2G writeback 组合 |
| DMA/TMA movement profile | `testcases/_get_case/dma_tma_movement_profile_test/` | bulk/tensor G2S/S2G 单次搬运 PMU cycle A/B；无 compute，无 buffer/stage 参数 |
| removed | `dma_tma_s2g_pipeline_perf_test`, `tma_roundtrip_pipeline_perf_test` | 已物理删除；语义由 directed stress、tensor S2G pingpong 和未来 multi-WG roundtrip perf 承接 |

原始需求里特别强调：

```text
项目构建脚本使用 build-ventus.sh
环境变量使用 source env.sh
项目测试部分使用 testcases/_get_case，其中分为性能测试和功能测试两种
```

因此本文档中的所有测试路线都按 `testcases/_get_case` 组织，而不是额外引入独立脚本体系。

</details>

<details>
<summary>12.7 指令、Issue、Fence 与 Completion 约束</summary>

S2G 高性能化不能只看 data path，还必须满足 scheduler/fence 语义。当前 DMA 指令大致分域：

| funct/路径 | 方向 | Completion 含义 |
|---|---|---|
| bulk G2S | global -> shared | shared 可见后完成 |
| tensor G2S | global -> shared | shared 可见后完成 |
| prefetch tensormap | metadata | descriptor/prefetch metadata 返回后完成 |
| bulk S2G | shared -> global | 所有 L2 Put AccessAck 返回后完成 |
| tensor S2G | shared -> global | 所有 L2 Put AccessAck 返回后完成 |

S2G 当前采用 FULL-only group fence。这里的 FULL 指一条 S2G 指令在所有 L2 Put `AccessAck` 返回后才对 scheduler 报 completion；group counter 只统计这种 full completion，不统计 shared read 阶段。

S2G 的硬约束：

1. S2G completion 不能在 shared read 完成时提前返回。
2. S2G completion 不能在 Put issue 后提前返回。
3. 必须等待对应 Put 的 L2 AccessAck，否则 fence 后 global memory 可见性不成立。
4. completion path 不能因为 `fence_end_dma.ready` 或 arbiter ready 条件造成内部 entry 无法释放。
5. group wait 只能改变“等待哪些已提交 group”，不能改变 S2G 写入的完成定义。
6. 本阶段不实现 `wait_group.read`，因此不需要 READ-phase counter；source shared buffer 的提前释放留给后续高级特性。

当前 completion token：

| 字段 | 作用 |
|---|---|
| `wid` | 释放对应 warp 的 DMA wait 状态 |
| `group` | all-DMA group commit/wait 计数 |
| `is_s2g` | 保留为调试/统计来源；scheduler 不再用它划分 wait 域 |

当前 scheduler group 状态：

| 状态 | 参数化规模 | 含义 |
|---|---:|---|
| `dma_group_issue_ptr` | `num_warp` | 每个 warp 当前 open group |
| `dma_group_committed` | `num_warp * dma_group_entries` | group 是否已提交且仍可能被 wait 观察 |
| `dma_group_count` | `num_warp * dma_group_entries * dmaInflightWidth` | all-DMA FULL outstanding counter；当前 8 warp、4 group、4-bit，共 128 bit |
| `dma_group_wait_keep` | `num_warp * log2Ceil(dma_group_entries + 1)` | `wait_group N` 中保留最近 N 个 group |

计数规则：

```text
任意 DMA issue:
  dma_group_count[wid][issue_ptr] += 1

任意 DMA full completion:
  dma_group_count[wid][completion.group] -= 1

commit_group:
  如果当前 group 有 work，则标记 committed 并推进 issue_ptr

wait_group N:
  允许最近 N 个 committed group 保持 pending
  阻塞直到更老 committed group 的 dma_group_count == 0
```

这个设计是 CUDA bulk group 的 FULL wait 子集。PTX 的 `cp.async.bulk.wait_group.read` 可以进一步只等待 tensormap/source read 完成，但它需要另一个 read-phase 状态或等价 token。当前明确不做这部分：好处是 completion token 不需要增加 phase，`dma_inflight_cnt` 不会出现一条 S2G 被 read/full 两次 completion 错减的问题，验证面也更小。

为什么仍保留 `is_s2g` 字段：

```text
mixed kernel:
  issue G2S for next input tile
  compute current tile
  issue S2G writeback for current tile
  wait input ready

如果 wait input ready 使用 legacy wait-all:
  会把后台 S2G writeback 也 drain 掉
  latency hiding 失败

统一 group 写法:
  issue G2S -> commit_group
  issue S2G -> commit_group
  wait_group1 等更老 G2S，允许最新 S2G 后台推进
```

这里 `is_s2g` 不再决定 wait 逻辑，只方便 RTL debug、PMU 归类和未来如果要重新引入方向敏感诊断时少改 token 格式。

当前建议保留两类语义：

| 语义 | 用途 |
|---|---|
| legacy wait-all/count | 兼容 kernel 和 directed 测试，等待 all-DMA inflight |
| all-DMA commit/wait_group | CUDA-style group ring，G2S/S2G/prefetch 共用 FULL completion counter |

保留但暂不实现的语义：

| 语义 | 状态 | 原因 |
|---|---|---|
| `wait_group.read N` | Phase 2 reserve | 可提前释放 shared source buffer，但需要 read/full 分相 completion；当前先把 FULL group wait 做稳 |

</details>

<details>
<summary>12.8 G2S 可参考机制</summary>

G2S 路线已经有较成熟的 RTL 组织，是 S2G 设计的直接参考。

G2S 中值得复用的机制：

| 机制 | G2S 中的作用 | S2G 对应设计 |
|---|---|---|
| descriptor cache | 避免重复 L2/TLB fetch tensormap | tensor S2G 2-entry descriptor-line cache |
| prefetch slot | metadata 提前取回 | 后续可考虑 tensor S2G descriptor prefetch |
| tensor iterator | 根据 descriptor/coords 生成访问序列 | tensor S2G setup/feature addrgen |
| TLB/L2 request 解耦 | metadata/data 分阶段推进 | direct DMA L1TLB + line task |
| tempmem line buffer | 存放 G2S L2 response line | S2G line write buffer |
| completion arbiter | 多 DMA 子模块统一完成 | S2G `DmaCompletion` 域扩展 |

G2S 与 S2G 最大差异：

| 项 | G2S | S2G |
|---|---|---|
| 最终可见动作 | shared write 成功 | L2/global Put ack 返回 |
| 数据方向 | L2 line -> shared lanes | shared lanes -> L2 line |
| OOB | 可 fill | store 侧应 suppress |
| line buffer 内容 | L2 response data | shared read 聚合后的 write data/mask |
| completion 触发 | shared 输出完成 | AccessAck 完成 |

因此不能简单把 G2S 状态机反过来用。S2G 必须额外处理：

1. partial store mask 合并。
2. same destination line overlap 顺序。
3. PutFull/PutPart 选择。
4. AccessAck tag 分配和回收。
5. source shared line 与 destination global line 双边界切分。
6. group wait 不能提前释放 shared source buffer。

</details>

<details>
<summary>12.9 SharedMemory、L2 和 TLB 路由约束</summary>

SharedMemory response 现在用 `instrId` 路由：

| 条件 | 路由 |
|---|---|
| `isWrite=true` | G2S/temp mem shared write response |
| `isWrite=false && instrId==1` | tensor S2G fallback/legacy shared read response |
| `isWrite=false && instrId>=2` | bulk S2G backend dynamic shared read response |

这就是 bulk S2G 必须使用 dynamic shared read entry 的原因。固定 `instrId=0/1` 的实现无法支持多个 outstanding read，也无法和 tensor fallback 清晰区分。

S2G dynamic shared read entry 应保存：

| 字段 | 作用 |
|---|---|
| `readValid` | entry 是否在用 |
| `readLine` | 对应 line table entry |
| `readPendingMask` | 等待哪些 lane response |
| `readLaneWord` | 每个 lane 写入 destination line 的哪个 word |
| `readLaneMask` | 每个 lane 的 byte mask |

L2 source 编码约束：

| 类型 | 编码思路 |
|---|---|
| metadata response | `d_source(0)=1`，再用高位区分 prefetch/desc/tensor |
| bulk S2G ack | 非 metadata、`d_opcode==0`、source low pattern 为 bulk S2G |
| tensor S2G fallback ack | 非 metadata、`d_opcode==0`、source low pattern 为 tensor S2G |
| tensor fast path ack | 现在通过 bulk backend 统一走 bulk S2G ack path |

TLB response 没有 source id，所以 `DMA_core` 仍用 `tlbBusy/tlbOwner` 对外保证同一时刻一个 DMA owner。S2G destination translation 统一依赖 DMA 专用 L1TLB；短期不修改外部 TLB response 接口。

nocache top 额外要求：

1. DMA request 会转成 DCache request。
2. response route queue 保存 `dmaSource/dmaAddr/dmaRspOpcode`。
3. 新 source 编码必须能在 with-cache 和 nocache 下都还原。
4. 因此 S2G source pattern 不应依赖 cache-only 行为。

</details>

<details>
<summary>12.10 Bulk S2G RTL 结构细节</summary>

当前 `DmaS2G` 可以按四类表理解。

Instruction table：

| 字段 | 说明 |
|---|---|
| `instValid` | instruction slot 有效 |
| `instWid` | warp id |
| `instGroup` | DMA group id |
| `instAsid` | address space id |
| `instSrc` | shared source base |
| `instDst` | global destination base |
| `instSize` | copy size |
| `instOffset` | addrgen 当前 offset |
| `instExternal` | 是否来自 tensor line task synthetic instruction |
| `instAddrDone` | 是否已生成全部 line task |
| `instPendingLines` | 尚未完成的 line 数 |
| `instComplete` | 等待 completion arbiter |

Line table：

| 字段 | 说明 |
|---|---|
| `lineValid` | line entry 有效 |
| `lineInst` | 属于哪个 instruction |
| `lineSrc` | shared source address |
| `lineDstVaddr` | destination line virtual address |
| `linePaddr` | translated physical address |
| `linePaddrValid` | DMA L1TLB translation 是否完成 |
| `lineTlbReq` | 是否已经发 TLB request |
| `lineDstStartWord` | destination line 起始 word |
| `lineBytes` | 本 line task 字节数 |
| `lineReadIssued` | shared read 是否已发 |
| `lineSharedDone` | shared data 是否收齐 |
| `linePutIssued` | L2 Put 是否已发 |
| `lineSeq` | line issue 顺序 |
| `lineMask` | 每个 word 的 byte mask |
| `lineDataMem` | word-banked `SyncReadMem`，存放 128B line data |

Read table：

| 字段 | 说明 |
|---|---|
| `readValid` | read entry 有效 |
| `readLine` | 目标 line entry |
| `readPendingMask` | 等待 lane response |
| `readLaneWord` | lane -> destination word 映射 |
| `readLaneMask` | lane byte mask |

Ack table：

| 字段 | 说明 |
|---|---|
| `ackValid` | tag 是否在用 |
| `ackLine` | tag 对应 line entry |
| `ackInst` | tag 对应 instruction |
| issue cycle | PMU 下记录 ack latency |

line task 生成边界：

```text
chunkBytes = min(
  bytesLeft,
  bytesToDstLine,
  bytesToSrcLine,
  numgroupshared * 4B
)
```

这个切分同时尊重 destination L2 line、source shared line 和 shared grouped read 宽度。它避免一条 line task 横跨两个 destination cacheline，也避免一次 shared read 横跨不可表达的 source line 组合。

PutFull/PutPart 选择：

| 条件 | L2 opcode |
|---|---|
| 128B line 每个 word byte mask 全满 | PutFull |
| 边界、不对齐、partial row、OOB 后有效 mask 非满 | PutPart |

overlap 编程契约：

1. 硬件不再比较 same destination line 的 older/younger 顺序。
2. 多个 outstanding S2G/TMA 如果覆盖同一 destination byte 且中间没有 wait/barrier，结果未定义。
3. 需要确定覆盖顺序的软件必须在两批写之间插入 wait_group/barrier。

</details>

<details>
<summary>12.11 Tensor S2G RTL 结构细节</summary>

`DmaTensorS2G` 当前由三段组成：

```text
descriptor fetch/cache
  -> descriptor decode/setup/addrgen
  -> fast line task or fallback element path
```

Descriptor cache：

| 字段 | 说明 |
|---|---|
| valid | entry 是否有效 |
| line tag | descriptor 128B line address |
| data | 32 个 descriptor word |
| replacement | 小规模 round-robin |

命中时：

```text
from_fifo.fire
  -> descriptor cache hit
  -> descWordsReg loaded from cache
  -> skip desc TLB/L2
  -> enter setup
```

未命中时：

```text
from_fifo.fire
  -> desc TLB req/rsp
  -> desc L2 Get
  -> desc response
  -> fill descriptor cache
  -> enter setup
```

Fast path 条件：

| 条件 | 原因 |
|---|---|
| FP32 / 4B element | 当前 shared lane 与 byte mask 逻辑按 4B word 优化 |
| `elementStrides[0]==1` | 保证 dim0 连续 |
| box in-bounds | S2G OOB store suppress 尚未进入 fast path |
| linear/no feature | rank1/rank2 contiguous span 可合成连续 byte span，提高 PutFull 比例 |
| swizzle32/64/128 | destination 仍连续，shared source 用 per-lane swizzled gather |
| interleave16/32 | 按 16B/32B interleave slice 发 coalesced line task |

Fast path 输出：

| 字段 | 来源 |
|---|---|
| `wid/group/asid` | original DMA issue |
| `src` | shared base + shared logical offset + row offset |
| `dst` | global base + global logical offset + row offset |
| `bytes` | 不跨 source/destination line 的 chunk bytes |
| `swizzleMode/swizzleBase/swizzleRow` | swizzle gather 所需的小 metadata；普通 linear/interleave 为 0 |
| `first` | tensor instruction 的第一条 line task |
| `last` | tensor instruction 的最后一条 line task |

Fallback path 保留原因：

1. OOB store 需要 per-element suppress。
2. 非连续 stride 可能导致 destination line mask 稀疏。
3. 非 4B datatype 和未覆盖的 CUDA TMA 组合仍需保守处理。
4. 保留 fallback 可以让 fast path 只覆盖高价值主线，不用一次性实现所有 CUDA TMA 组合。

当前 tensor S2G line-task 路径服务所有 4B 常见 row，包括 descriptor feature 和普通 no-permute row；删除的是跨多行 linear span fusion、high-rank flatten 和纯 linear 专用 setup/FSM。当前可以抽象成：

```text
if dataType == FP32:
     if swizzle32/64/128:
       emit line tasks capped at swizzle span,
       attach swizzle metadata for shared gather
     else if interleave16/32:
       emit one coalesced line task per interleave slice
     else if regular stride or OOB clipping:
       emit feature line tasks with mask/stride metadata
     else:
       emit ordinary row-level line tasks
else:
     use generic setup/fallback
```

</details>

<details>
<summary>12.12 测试组织细节</summary>

测试分为功能测试和性能测试。

功能测试目标：

| 测试 | 目标 |
|---|---|
| small dual bulk | 小尺寸 S2G 基本正确性 |
| four issue one fence | 多 outstanding issue + 单 fence |
| wait-oldest reuse | 只等最老写回后复用 source |
| wait-group reuse | CUDA-style group wait 复用 source |
| multi-warp fence | 多 warp 独立 completion |
| tensor origin/subbox | descriptor coords 和 global offset |
| tensor full-line rank1 | fast path PutFull |
| tensor interleave rank3 | interleave 地址映射和 slice-level line-task 正确性 |

性能测试目标：

| 测试 | 对照路径 | 主要回答的问题 |
|---|---|---|
| `dma_tma_g2s_pingpong_perf_test` | G2S + manual writeback | G2S 路线是否被误伤 |
| `dma_tma_tensor_s2g_pingpong_perf_test manual_tensor_s2g` | manual input + tensor S2G | 只替换写回是否有收益 |
| `dma_tma_tensor_s2g_pingpong_perf_test tma_tensor_s2g` | G2S input + tensor S2G | 双向 DMA 闭环收益 |
| `dma_tma_movement_profile_test` | manual movement vs bulk/tensor DMA/TMA movement | 纯 movement microbench；同 tile 对比 bulk_g2s、tensor_g2s、bulk_s2g、tensor_s2g |

baseline 修正原则：

1. baseline 必须是 warp/workgroup 协作搬运，不允许 lane0 串行。
2. perf shape 与 G2S pingpong 保持一致：`16x16`, `32x16`, `32x32`, `64x32`, `64x64`。
3. pingpong stage sweep 使用 `1, 2, 4, 8, 16`；movement profile 不再有 buffer/stage 参数。
4. primary metric 用 PMU active cycles；host ns 只作辅助。
5. GVM 用于功能参考；tensor descriptor perf 以 RTL/rtl-nocache 为主。movement profile 中 GVM/GVM-nocache 的 tensor S2G cycle 可作为模型诊断，不作为 RTL 性能结论。

推荐的 full sweep 输出：

```text
| path | min | avg | max |
|---|---:|---:|---:|
| G2S + manual writeback | ... | ... | ... |
| manual input + S2G writeback | ... | ... | ... |
| G2S + S2G | ... | ... | ... |

| tile | stages | G2S+manual | manual+S2G | G2S+S2G |
|---|---:|---:|---:|---:|
```

为什么需要 directed + perf 两类测试：

| 只跑 directed 的盲点 | 只跑 perf 的盲点 |
|---|---|
| 看不到 wait/fence drain、lineFullStall、ack latency 平台期 | 很难定位 byte-level 错误 |
| 很难证明高性能路径被实际命中 | 可能把 baseline 写弱得到虚假 speedup |
| 不覆盖多 stage overlap | 不覆盖 OOB/fallback/feature 边界 |

</details>

<details>
<summary>12.13 性能分析细节</summary>

当前可观察 counters：

| counter | 解释 |
|---|---|
| `instIssued` | S2G instruction 数 |
| `lineIssued` | L2 Put line task 数 |
| `PutFull/PutPart` | full-line 与 partial-line write 比例 |
| `bytesWritten` | 实际写出字节 |
| `sharedReadReq/Rsp` | shared read issue/response |
| `tlbReq` | TLB 请求数 |
| `ackCount/LatencySum` | L2 AccessAck 数和累计延迟 |
| `lineFullStall` | line table 满或外部 line task 因无 line entry 阻塞 |
| `readEntryFullStall` | shared read entry 满 |
| `ackTagFullStall` | L2 ack tag 满 |
| `dma fence/group wait` | scheduler 侧 DMA wait stall |

如何解释常见现象：

| 现象 | 判断 |
|---|---|
| 小 tile 差、大 tile 好 | fixed setup/barrier/fence 占比高 |
| 所有 tile 有平台期 | outstanding 或 backend 周转可能是瓶颈 |
| PutPart 比例高 | coalescing 没覆盖 shape，或存在 partial/OOB/stride |
| lineFullStall 高 | line table/Put ack 周转限制 issue |
| ackLatencySum 高 | L2/cache/nocache ack path 成本高 |
| dmaFenceWait 高 | wait 点过早、group 排布不合理或 ring slot 复用受阻 |
| tlbReq 高 | 跨页或 line 数多；当前直接依赖 DMA L1TLB，需结合 end-to-end cycle 判断是否真的成为瓶颈 |
| line/read/ack stall 都低但性能一般 | 前端 setup/FSM/control/barrier 更可疑 |

当前最新代表点对后续硬件优化的启示：

1. 64x64/stage16 的 tensor S2G pingpong 已到 2.4807x，说明 data path 主线是有效的。
2. manual input + tensor S2G full sweep avg 为 1.2493x，说明单独替换 writeback 仍有收益，但 fixed setup/fence/compute 占比更明显。
3. G2S pingpong full sweep avg 为 1.3703x，说明当前 S2G 改动没有明显误伤 G2S 代表路径。
4. directed PMU 中 full-line rank1 已是 1 个 PutFull，证明 fast path 的 PutFull 选择实际命中。
5. feature perf 中 swizzle/interleave 组合未出现数量级退化；后续是否继续压低 swizzle32 多 row PutPart，应由 stress 或真实应用 shape 决定。

</details>

<details>
<summary>12.14 面积与时序边界</summary>

S2G 设计不能靠无限堆资源换性能。RTL 级别不能直接给出面积，但必须约束结构规模。

面积主项：

| 资源 | 面积风险 | 当前策略 |
|---|---|---|
| 128B line image | 高 | bulk 少量 entry，tensor 不复制完整大表 |
| word bank | 中到高 | `SyncReadMem` 推断 memory，而不是全寄存器 |
| descriptor cache | 中 | 2-entry，每 entry 128B |
| ack/read table | 低到中 | 受 tag/instrId 上限约束 |
| group ring | 低 | 只存 metadata |
| setup FSM registers | 中 | 避免为每 element 建表 |

不建议的优化：

1. 默认把 line buffer 扩到 16 或 32。
2. 为 tensor S2G 复制一套完整 bulk line table。
3. 为每个 tile/stage 建寄存器数组。
4. 用大规模全连接比较器处理所有 overlap。
5. 在没有 full sweep 证据前调大所有 queue 参数。

可接受的优化：

| 优化 | 原因 |
|---|---|
| rank1/rank2 fast setup | 减少控制开销，寄存器增加很小 |
| 1 到 2 项 line-task FIFO | 解耦前端和 backend，小 metadata FIFO |
| 2-entry descriptor cache | 与 G2S 对齐，容量小 |
| 直接依赖 DMA L1TLB | 复用已有 DMA translation path，不复制翻译状态 |
| FSM 分段 | 改善时序和可读性，不明显增面积 |

综合后必须复核：

1. bulk `SyncReadMem` 是否被综合成预期 memory。
2. 32 个 word bank + Put read buffer 的代价是否可接受。
3. descriptor cache 是否推断为合适 memory/register 结构。
4. line mask 比较是否落在关键路径。
5. tensor setup 中 multiply/stride/feature 判断是否落在关键路径。
6. completion/group wait ring 是否影响 scheduler 时序。

</details>

<details>
<summary>12.15 已定位问题与修复记录</summary>

### Descriptor hot-update

现象：

```text
tensor directed subbox case 读到 stale descriptor
表现为写入位置/形状像上一个 case
multi-WG subbox 顺序运行时，第一个元素落到 8x8 视图 (5,0)
```

原因：

1. host/后端在同一进程连续 case 中复用同一个 descriptor device vaddr。
2. RTL tensor descriptor cache 按 descriptor line address 命中。
3. 后一个 case 改写同一 descriptor line，但硬件没有 invalidate 语义。
4. tensor S2G 命中 stale descriptor，导致写入 shape/base 错误。
5. Ventus RTL 后端对同一 `.cl` 中多个 kernel 的 `--init` 选择也可能不符合 host 期望，因此 directed measured kernel 不应依赖多入口 source。

当前处理：

| 层面 | 处理 |
|---|---|
| directed test | 每个 case 避免复用同一 descriptor line；必要时用真实 guard kernel arg 推开 device vaddr |
| kernel source | multi-WG subbox 拆成单入口 `.cl`，避免 `--init` 选到同 source 的其他 kernel |
| perf kernel | descriptor 在 setup kernel 写定，hot loop 不改 metadata line |
| RTL | 保留 cache，不加入临时 invalidation |

### Tensor synthetic instruction zero-byte line

现象：

```text
tensor S2G rank2 case hang
debug 显示出现 bytes=0 的 sharedReq/line
fence/group wait 等不到 completion
```

原因：

1. tensor fast path 接入 bulk backend 时，为一条 tensor instruction 分配 synthetic instruction slot。
2. 这个 slot 没有普通 bulk 的 `size/offset` 语义。
3. bulk addrgen 没有排除该 slot。
4. bulk addrgen 从 synthetic slot 生成 `bytes=0` 的 line。

修复：

```text
instExternal(freeInstIdx) := true.B   // tensor synthetic instruction
genInstVec excludes instExternal
completion clears instExternal
```

验证：

| 测试 | 结果 |
|---|---|
| RTL `tma_s2g` | PASS |
| RTL `bulk_s2g_stress` | PASS |
| rtl-nocache `tma_s2g` | PASS |

### Tensor S2G host reference outDim

现象：

```text
新增 fixed-seed case fuzz_tensor_rank3_stride_subbox 首次运行失败
FAIL fuzz_tensor_rank3_stride_subbox at byte 456: got=cd exp=1b
```

原因：

1. RTL tensor S2G 对 dim1+ 的 `elementStride` 使用 `ceil(boxDim / elementStride)` 作为迭代次数。
2. dim0 保持 `boxDim`，由 dim0 element stride 影响 global offset/line mask。
3. host expected 生成逻辑原先把所有维度的 `boxDim` 都当作输出元素个数。
4. 当 dim1 stride 为 2 时，host 期望多生成了 RTL 不会写的坐标，形成假失败。

修复：

```text
fill_expected():
  dim0 out_dim = boxDim[0]
  dim1+ out_dim = ceil(boxDim[d] / elementStride[d])
```

验证：

| 测试 | 结果 |
|---|---|
| GVM `S2G_TENSOR_CASE_FILTER=fuzz_tensor` | PASS |
| RTL `S2G_TENSOR_CASE_FILTER=fuzz_tensor` | PASS |
| RTL `--suite directed` | PASS |
| rtl-nocache `--suite directed` | PASS |

### Baseline 过弱

早期问题：

```text
baseline 是 lane0 串行搬运
导致 S2G perf 出现十几倍 speedup
```

修复：

1. baseline 改成 warp/workgroup 协作搬运。
2. shape 改成 G2S pingpong 同款 tile/stage。
3. 十几倍 speedup 口径不再使用。

### GVM 与 RTL 性能差异

当前判断：

| 后端 | 用途 |
|---|---|
| GVM | 功能参考、快速验证 |
| RTL | 性能结论主依据 |
| rtl-nocache | cache path 对照和 nocache 路由验证 |

GVM tensor descriptor model 在相关 probe 中明显慢于协作 baseline，不作为 tensor S2G 性能评价主依据。

</details>

<details>
<summary>12.16 后续 RTL 性能优化草案</summary>

本节保留后续 RTL 性能优化草案。当前不建议恢复 pure linear span fusion、high-rank flatten 或 rank1/rank2 linear-only setup 捷径；普通 no-permute tensor row 已经通过 row-level `S2GLineTask` 复用 bulk backend。后续优化应优先围绕高级 feature 的 segmented PutPart 合并，并严格保持小状态、小 metadata，不扩大大容量 line buffer。

### Phase 0: 保持当前 row-level line-task 主线

当前保留的基础设计：

| 子项 | 当前设计 |
|---|---|
| ordinary no-permute row | 按 row/segment 生成 `S2GLineTask`，不做跨多行 linear span fusion |
| dim0 element stride | `S2GLineTask.dstWordStride` 支持 sparse destination word mask |
| dim1+ element stride | 保留 row/plane 级循环，每个有效 row 生成 line task |
| OOB suppress | raw dim 驱动 shared tile layout，clipped dim 只控制 global valid write |
| rank3/4/5 | 作为 row-level line-task/setup 覆盖对象，不引入 high-rank flatten 专用路径 |
| fallback | 非 FP32/4B、复杂 swizzle+OOB、非法 interleave 或未覆盖的不规则组合继续走保守 fallback |

面积边界：

- 不新增大 line buffer，不增加 line table 默认深度。
- 不把 tile/stage/rank 转换成寄存器规模随之增长的状态。
- 新增 metadata 必须保持 bit 级或少量字段级。

### Phase 1: swizzle/interleave segmented row-group coalescer

当前默认 feature sweep 没有数量级退化；swizzle32 16x16 是默认 paired worst，host ratio 约 1.144x。若后续 stress 或真实应用 shape 证明 swizzle/interleave segmented PutPart 是瓶颈，目标是在 destination 128B line 内收集多个 16B/32B segment，让 backend 尽量发更少的 PutPart，理想情况下合成 PutFull：

| 项 | 设计要求 |
|---|---|
| 触发条件 | FP32、unit element stride、box in-bounds、多个 swizzle/interleave segment 落在同一 destination 128B line |
| task 形式 | 新增小型 segmented line task metadata，或扩展 `S2GLineTask` 携带 4 段 segment valid/source metadata |
| data buffer | 复用 bulk S2G line image，不新增随 tile/stage 增长的大 buffer |
| shared read | 每段仍按 swizzle row 做 per-lane gather，按 segment 写入同一 line image 的不同 32B lane |
| completion | 仍由 bulk backend ack 聚合，completion token 不变 |
| fallback | 任一 segment OOB、stride 非 1、跨 destination line 或不满 4 段时退回当前 PutPart path |
| 验证 | `dma_tma_s2g_feature_perf_test` paired swizzle32、interleave16/32、interleave32+swizzle32 应不退化；stride/OOB/rank3/4/5 不能退回逐元素 fallback |

### Phase 2: 小 FIFO 评估

只有出现下列证据才加 FIFO：

| 证据 | 说明 |
|---|---|
| feature row emitter 已优化后仍有平台期 | 前端/后端解耦可能有用 |
| `line_task.valid && !line_task.ready` 占比高 | backend 反压前端 |
| line/read/ack stall 低但 active cycles 高 | 可能被逐条交互 latency 限制 |

FIFO 设计：

| 参数 | 建议 |
|---|---|
| depth | 1 或 2 |
| 内容 | `S2GLineTask` metadata，不存 128B data |
| completion | 仍由 `DmaS2G` synthetic instruction 管理 |
| 面积 | 很小，远低于 line buffer |

### Phase D: 测试

新增或强化 case：

| case | 预期 |
|---|---|
| rank1 32 FP32 | 1 PutFull |
| rank1 64 FP32 | 2 PutFull |
| rank2 4x4 origin | row-level PutPart/PutFull 选择正确 |
| rank2 16x16 contiguous | 8 PutFull 或按实现 line count |
| rank2 subbox | 正确 offset，partial line |
| regular element stride | line-task masked PutPart，dim0 用 `dstWordStride` |
| irregular sparse stride | fallback |
| swizzle32/64/128 | shared gather line task，结果与 host swizzle reference 一致 |
| interleave16/32 | slice-level line task，结果与 host interleave reference 一致 |
| interleave32 + swizzle32 | 已进入 feature diagnostic；paired A/B 不应出现数量级退化 |
| OOB | clipped valid region line-task；复杂组合 fallback |

性能先跑代表点，再跑 full sweep：

```text
representative:
  16x16/stage4
  64x64/stage16

full:
  dma_tma_g2s_pingpong_perf_test sweep
  dma_tma_tensor_s2g_pingpong_perf_test sweep manual_tensor_s2g
  dma_tma_tensor_s2g_pingpong_perf_test sweep tma_tensor_s2g
```

</details>
