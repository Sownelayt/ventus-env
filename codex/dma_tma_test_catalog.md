# TMA 测试目录索引

更新时间：2026-08-11

本文只描述当前工作树中仍然存在并由 registry 使用的 TMA 测试。旧 G2S、旧 S2G 和旧 multi-WG 功能目录已经合并到统一功能套件，不再作为兼容入口保留。

## 1. Registry 与运行入口

测试注册表：

```text
testcases/_get_case/cases_dma_tma.csv
```

统一运行入口：

```bash
cd testcases/_get_case
./run_dma_tma_rtl.sh --case dma_tma_v2_func_test --backend spike
./run_dma_tma_rtl.sh --case dma_tma_v2_func_test --backend gvm
./run_dma_tma_rtl.sh --case dma_tma_v2_func_test --backend gvm-nocache
./run_dma_tma_rtl.sh --case dma_tma_v2_func_test --backend rtl
./run_dma_tma_rtl.sh --case dma_tma_v2_func_test --backend rtl-nocache
```

runner 会 source 仓库根目录的 `env.sh`，在 testcase 自己的目录中构建和运行，并把完整日志放入 testcase 的 `log/`，把概要写入指定的 `--log-dir`。

## 2. 唯一功能套件 `dma_tma_v2_func_test`

路径：

```text
testcases/_get_case/dma_tma_v2_func_test
```

主文件：

| 文件 | 作用 |
|---|---|
| `dma_tma_v2_func_test.cpp` | OpenCL host、descriptor 构造、软件 golden 和 verdict |
| `dma_tma_v2_func_test.cl` | 当前 TMA 指令的 directed kernels |
| `../common/ventus_tma_v2_opencl.h` | 唯一 OpenCL 指令包装 |
| `../common/ventus_tma_v2_spec.h` | RTL/Spike/测试共享数值的 C 侧镜像 |
| `../common/tma_model.{h,cc}` | TensorMap decode、bind 和 atom-plan 参考模型 |

该套件注册到 `spike`、`gvm`、`gvm-nocache`、`rtl` 和 `rtl-nocache`，是当前功能正确性的唯一软件入口。

### 2.1 Bulk copy

- 合法 G2S/S2G roundtrip；
- global/shared 地址 16B 对齐；
- 非零且为 16B 倍数的长度；
- 地址不对齐、零长度和非 16B 倍数的拒绝；
- rejected command 后的 ordering 和 sticky status。

### 2.2 Bulk S2G reduce

覆盖当前公开的九个 CUDA-compatible 32-bit 组合：

- `add.u32`、`add.s32`；
- `min.u32`、`min.s32`；
- `max.u32`、`max.s32`；
- `and.b32`、`or.b32`、`xor.b32`。

每个 case 检查逐 32-bit element 的最终 global 值以及 S2G group completion。非法 `ADD+B32` 和 reserved reduce mode 还会检查：

- 整条命令被拒绝；
- global destination 保持不变；
- status 为 `UNSUPPORTED_FEATURE`。

### 2.3 Tensor G2S/S2G

- rank 1–5；
- U8/U16/U32/U64、S8/S16/S32/S64、FP16/BF16/FP32/FP64；
- B4x16、B4x16P64 和 B6 sub-byte 布局；
- 最终 bounding-box global 起点 16B 对齐；
- B4x16 coordinate[0] 的 32-element phase 与偶数 global dim0；
- B4x16P64/B6 coordinate[0] 的 128-element phase；
- global stride、subbox、partial OOB；
- interleave16/interleave32；
- swizzle32/swizzle64/swizzle128；
- zero fill 和浮点 NaN fill；
- S2G 正方向 OOB suppress；
- descriptor prefetch、invalidate 和地址复用。

Tensor G2S 允许 **最终地址仍为 16B 对齐** 的 signed negative origin，并对越界部分 fill。Tensor S2G copy 和 reduce 都拒绝任一 active dimension 的负起始坐标；动态地址不对齐或不合法 sub-byte phase 同样检查 status 和零数据流副作用。

### 2.4 Tensor S2G reduce

- U32/S32 的 add、min、max；
- 32-bit bitwise and、or、xor；
- rank-2 stride；
- 正方向 partial OOB suppress；
- 负起始坐标整条命令拒绝；
- 最终 AMO ack 之后 group 才可完成。

### 2.5 Completion、并发和错误状态

- mbarrier 初始化、`arrive.expect_tx`、wait 和 phase reuse；
- async shared proxy fence；
- S2G commit/wait group 及 ring wrap；
- mixed G2S/Tensor/Bulk 双向传输；
- multi-warp mbarrier；
- multi-workgroup 隔离；
- sticky CSR 首错保留与 clear；
- 非法 magic、reserved word、layout、L2 promotion、sub-byte 方向和 OOB 模式。

## 3. C model 一致性测试

路径：

```text
testcases/_get_case/dma_tma_cmodel_test
```

运行：

```bash
make -C testcases/_get_case/dma_tma_cmodel_test check
```

它不运行硬件，而是验证 128B TensorMap ABI、descriptor validation、地址绑定和 atom plan。`make golden` 会重建 RTL 单测消费的 `gpgpu/ventus/tests/resources/tma_v2_atom_trace.csv`；只有 ABI 或 planner 语义有意变化时才应更新该 golden。

## 4. 当前性能与诊断目录

| 目录 | 目的 | Registry backend |
|---|---|---|
| `dma_tma_g2s_pingpong_perf_test` | Tensor G2S ping-pong 与 overlap | spike、gvm |
| `dma_tma_tensor_s2g_pingpong_perf_test` | manual-input 和 TMA-input 的 Tensor S2G ping-pong | spike、gvm |
| `dma_tma_movement_profile_test` | Bulk/Tensor、G2S/S2G movement microbench 和性能门禁 | gvm、gvm-nocache |
| `dma_tma_tensor_feature_perf_test` | stride/interleave/swizzle/OOB/high-rank 诊断 | spike、gvm |
| `dma_tma_bidirectional_contention_perf_test` | 双向、多命令、多 warp/WG contention | spike、gvm |
| `dma_tma_v2_reduce_perf_test` | Tensor reduce、AMO 和 contention profile | spike、gvm |

性能测试用于观测吞吐、stall 和尾延迟，不替代 `dma_tma_v2_func_test` 的功能 verdict。`dma_tma_movement_profile_test/run_tma_perf_gate.sh` 是当前无版本别名的正式性能门禁入口。

## 5. RTL 单元测试

路径：

```text
gpgpu/ventus/tests/src/DmaTest
```

| 测试 | 主要边界 |
|---|---|
| `TmaV2Frontend_test` | descriptor compile、CommandBinder、负坐标、layout validation |
| `TmaV2Backend_test` | window/line/shared/cache 数据通路 |
| `TmaV2DmaCore_test` | 指令 ingress、Bulk/Tensor、reduce AMO、非法编码零流量 |
| `TmaV2Completion_test` | completion 与 mbarrier |
| `TmaV2GroupTracker_test` | commit/wait group |
| `TmaV2CreditDepth_test` | window/request/shared/write-ack 容量边界 |
| `TmaV2Capacity_test` | 256 line/ack tag、source 255 与乱序回收 |
| `TmaControlScheduler_test` | group、mbarrier、proxy wait 调度 |

Reduce 还依赖 `gpgpu/ventus/tests/src/cache/AtomicUnit_test.scala`，因为 TMA 只生成 AMO 请求，最终原子运算由共享的 L2 AtomicUnit 完成。

## 6. 维护规则

新增或改变 TMA 语义时，应在同一修改中同步：

1. RTL 指令定义、validation 和数据通路；
2. Spike handler；
3. 公共 spec/OpenCL 包装；
4. C model（若涉及 descriptor/plan）；
5. `dma_tma_v2_func_test`；
6. 对应 RTL 单测；
7. `cases_dma_tma.csv` tags；
8. `codex/ventus_tma_instruction_abi_reference.md` 和本文。

不要重新引入按历史版本命名的功能目录、旧指令别名或旧 PMU/环境变量兼容入口。
