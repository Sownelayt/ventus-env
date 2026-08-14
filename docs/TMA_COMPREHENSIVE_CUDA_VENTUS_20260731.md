# TMA 全特性周期矩阵：H100、B200 与 Ventus

## 结论

本轮建立了 **399 个同参数定向用例**，覆盖 Bulk、Tensor rank 1–5、
16 B–32 KiB、global 128B 对齐与合法 16/32/64/112B 相位、dtype、stride、
subbox、OOB、swizzle、interleave、sub-byte 和 Reduce。CUDA 与 Ventus
由同一份 `CaseSpec` 生成参数，并逐字节或逐元素验证结果。

核心结论如下：

1. H100/B200 的 378 个安全配置全部正确；Ventus 的 334 个安全非 Reduce
   配置全部正确。rank 1–5、普通 dtype、三种 swizzle、interleave16、
   对齐/非对齐、stride 与 OOB 的主功能面已经比较完整。
2. Ventus 当前非 Reduce 绝对周期通常小于 CUDA，但这主要来自 GVM 中
   RTL L2/backing-memory 模型的低固定延迟，不能据此宣称真实芯片更快。
   更有意义的 16→32 KiB 边际吞吐显示：G2S 为 **2.063 cycle/128B**，
   H100/B200 为 **1.16–1.19 cycle/128B**。当前主要非 Reduce 性能瓶颈仍是
   G2S line/request 消费速率，而不是 Descriptor、rank 或 dtype。
3. CUDA 的 interleave16 在 16 KiB 上相对同 shape plain 快 1.78–3.27×；
   Ventus 的 int/plain 周期逐项完全相同。这是本轮最明显的非 Reduce
   微架构差异：Ventus 完成了地址语义，但没有体现 interleave 对物理
   transaction/channel 并行度的收益。
4. Ventus 对任意非零 128B 相位统一增加 G2S 49 cycles、S2G 76 cycles；
   CUDA 的代价取决于具体相位和跨线位置。这说明 Ventus 的 fragment/line
   处理仍是较粗的固定串行代价，值得优先解耦第二 fragment。
5. Reduce 是当前最大缺口：Ventus 4 KiB Reduce 为 8574 cycles，H100/B200
   约 264/267 cycles，即慢约 32×；同时 44 项中有 8 项数据错误、2 项
   32 KiB 短超时。Reduce 不能进入性能冻结，必须先修正确性。

本轮没有覆盖或改写冻结的 V3.0 数据。Ventus 列已由当前工作区的 V3.7
preflight GVM 重新运行；六个安全非 Reduce 分片共 334/334 通过。原始结果在
[`comprehensive_ventus_v3_7`](../benchmarks/tma-cycle-compare/results/comprehensive_ventus_v3_7/)，
不是把新数据写回 V3.0 baseline。

## 测试矩阵与计时边界

| 分片 | 数量 | 主要变量 |
|---|---:|---|
| Bulk | 34 | 16 B–32 KiB、G2S/S2G、global mod128 0/16/32/64/112 |
| Tensor capacity | 27 | 128 B–32 KiB、G2S/S2G/roundtrip |
| Tensor geometry | 109 | rank 1–5、容量、相位、stride、subbox、OOB |
| dtype | 92 | 13 种普通 dtype、128 B/4 KiB、三方向、浮点 OOB zero/NaN |
| swizzle | 54 | 32/64/128B、1/4/16 KiB、三方向及 plain 对照 |
| interleave16 | 18 | UINT16 whole-atom、1/4/16 KiB、三方向及 plain 对照 |
| Reduce | 44 | U32/S32、六种 op、rank、容量、相位、stride、OOB |
| sub-byte | 3 | U4 align8、U4 align16、U6 align16 |
| interleave32 | 18 | UINT16 whole-atom 风险探针及 plain 对照 |

CUDA 使用 SM 内 `clock64()`，计时从 issue 前开始，到 mbarrier wait 或
`cp.async.bulk.wait_group 0` 完成后结束。Ventus 使用 `read_cycle_lo()`，
边界同样覆盖 issue、协议 wait 和命令完成，不包括 host launch 和最终 host
校验。CUDA 每个平台只有一个 recorded repeat、一个付费 attempt；主矩阵
`warmups=1`。Ventus 主性能矩阵同样保留一个 recorded repeat；失败诊断和
风险项不作为精确性能倍率。

“非 128B 对齐”仅改变合法的 global data base 相位。TensorMap descriptor
和 shared base 仍满足协议对齐要求，因此比较的不是非法 descriptor 拒绝
时间。

## 覆盖状态

| 平台 | Manifest | 正周期且零错误 | 其他状态 |
|---|---:|---:|---|
| H100 | 399 | 379 | 3 encode rejected；1 illegal address；16 未在故障后继续 |
| B200 | 399 | 380 | 1 sub-byte timeout；1 illegal address；17 未在故障后继续 |
| Ventus | 399 | 385 | 8 validation error；6 bounded timeout |

安全 CUDA 分片共 378 项，H100/B200 均 378/378。Ventus 安全非 Reduce
分片共 334 项，334/334；Reduce 为 34/44 成功。风险分片的失败没有自动
重试，也没有延长上限。

完整机器可读状态见：

- [coverage.csv](../benchmarks/tma-cycle-compare/results/comprehensive_report/coverage.csv)
- [summary.csv](../benchmarks/tma-cycle-compare/results/comprehensive_report/summary.csv)
- [ratios.csv](../benchmarks/tma-cycle-compare/results/comprehensive_report/ratios.csv)
- [trends.csv](../benchmarks/tma-cycle-compare/results/comprehensive_report/trends.csv)

## 容量周期

以下均为同参数完整命令周期；`R/T` 表示 Tensor roundtrip。

| Path | Bytes | H100 | B200 | Ventus |
|---|---:|---:|---:|---:|
| Bulk G2S | 128 | 507 | 428 | 95 |
| Bulk G2S | 4096 | 507 | 578 | 169 |
| Bulk G2S | 16384 | 658 | 578 | 378 |
| Bulk G2S | 32768 | 810 | 728 | 642 |
| Bulk S2G | 128 | 47 | 47 | 57 |
| Bulk S2G | 4096 | 171 | 171 | 161 |
| Bulk S2G | 16384 | 555 | 555 | 453 |
| Bulk S2G | 32768 | 1067 | 1067 | 847 |
| Tensor G2S | 128 | 470 | 570 | 143 |
| Tensor G2S | 4096 | 608 | 558 | 217 |
| Tensor G2S | 16384 | 608 | 706 | 426 |
| Tensor G2S | 32768 | 756 | 854 | 690 |
| Tensor S2G | 128 | 104 | 140 | 106 |
| Tensor S2G | 4096 | 228 | 264 | 210 |
| Tensor S2G | 16384 | 612 | 648 | 502 |
| Tensor S2G | 32768 | 1124 | 1160 | 887 |
| Tensor R/T | 128 | 605 | 575 | 232 |
| Tensor R/T | 4096 | 719 | 836 | 410 |
| Tensor R/T | 16384 | 1251 | 1368 | 911 |
| Tensor R/T | 32768 | 1911 | 1880 | 1560 |

### 大容量边际吞吐

用 16→32 KiB 的增量除以新增的 128 个 cache line，可以弱化固定启动延迟：

| Path | H100 cycle/128B | B200 cycle/128B | Ventus cycle/128B |
|---|---:|---:|---:|
| Bulk G2S | 1.188 | 1.172 | 2.063 |
| Tensor G2S | 1.156 | 1.156 | 2.063 |
| Bulk S2G | 4.000 | 4.000 | 3.078 |
| Tensor S2G | 4.000 | 4.000 | 3.008 |
| Tensor roundtrip | 5.156 | 4.000 | 5.070 |

这张表比绝对周期更接近 TMA 数据面的结构差异：

- Ventus G2S 每条线约 2.06 cycles，仍未达到一个 128B line/cycle；相对
  CUDA 的边际吞吐差约 1.78×。
- Ventus S2G 表面上优于 CUDA，但这里恰好最受简化 L2/write-ack 模型影响，
  不能直接投射到真实芯片。
- Ventus roundtrip 与 H100 边际速率接近、比 B200 慢约 27%；它近似等于
  当前 G2S 与 S2G 两段之和，说明两方向在单命令内没有额外重叠。

## 特性变化规律

### rank 与 dtype

同为 256 B 时，Ventus rank 1→5 的 G2S 为 145/146/147/148/149 cycles，
S2G 为 108/109/110/111/112 cycles；额外 rank 基本只支付每维一个绑定周期。
这证明 Descriptor/Binder 不是大容量瓶颈。

13 种普通 dtype 在 4 KiB 上逐平台同周期：Ventus G2S/S2G/R/T 为
217/210/410，H100 为 608/228/719，B200 为 558/264/836。普通 dtype
转换没有形成吞吐差异；sub-byte 需要单独判断，不能由这个结论外推。

### 128B 相位

4 KiB Ventus 的相位规律非常整齐：

| Path | mod128=0 | 任意 16/32/64/112 | 固定增量 |
|---|---:|---:|---:|
| Bulk G2S | 170 | 219 | +49 |
| Bulk S2G | 161 | 237 | +76 |
| Tensor G2S | 219 | 268 | +49 |
| Tensor S2G | 211 | 287 | +76 |
| Tensor roundtrip | 413 | 538 | +125 |

CUDA 不呈现“所有非零相位同代价”。例如 H100 Bulk S2G 的
mod128=0/16/32/64/112 为 171/267/170/169/266；只有真实跨越不利物理
边界的相位出现约 96-cycle 台阶。Ventus 的统一固定增量表明当前实现只按
“是否跨线”进入第二 fragment 路径，没有充分流水化两个 fragment。

### swizzle 与 interleave

swizzle32/64/128 的 enabled/plain 在绝大多数点同周期；CUDA 通常只在 G2S
启动部分相差约 10–13 cycles，持续吞吐没有明显惩罚。Ventus enabled/plain
逐项同周期，符合“地址置换在 Planner 中完成、数据通路不增加阶段”的目标。

interleave16 则完全不同。16 KiB 的同 shape 对照为：

| Direction | H100 plain→int16 | B200 plain→int16 | Ventus plain→int16 |
|---|---:|---:|---:|
| G2S | 1507→608（2.48×） | 1607→706（2.28×） | 428→428 |
| S2G | 2207→674（3.27×） | 1158→649（1.78×） | 504→504 |
| roundtrip | 3686→1261（2.92×） | 2630→1381（1.90×） | 915→915 |

因此 Ventus 的 interleave 目前更像“功能地址变换”，没有让多个独立 physical
line/channel 更早并行进入 L2。若 L2 接口本身只有一个 request/cycle，TMA
仍应保证连续产生独立请求；若 L2 模型没有 bank/channel 并行度，则必须用
TMA PMU 区分“请求已连续发出”和“模型没有表现出硬件收益”。

## Reduce 与风险特性

### Reduce

U32/S32 的 add/min/max/and/or/xor 在 4 KiB 上均正确，但 Ventus 六种 op
都为 8574 cycles；H100 为 264、B200 为 267。相同周期说明瓶颈不是具体
ALU op，而是 Reduce 的逐 atom/fragment 搬运与 read-modify-write 调度。

Ventus 的失败点为：

- validation error：U32/S32 16 KiB、global mod128=32、stride 144/512、
  OOB right25/right50/outer75；
- 10 秒 bounded timeout：U32/S32 32 KiB。

rank 1–5 和六种 op 的 4 KiB 主路径均正确，因此应围绕 fragment 合并、
非 contiguous address 和大容量计数/slot 回收定位，而不是重写 Reduce opcode
解码。

### sub-byte 与 interleave32

- H100 对三个 sub-byte descriptor 均明确 `encode_rejected`。
- B200 的 U4 align8 roundtrip 正确且为 603 cycles；U4 align16 在 2 秒
  风险上限内未完成，U6 因前项停止未运行。
- Ventus U4 align8 roundtrip 正确；U4 align16 与 U6 align16 在 2 秒内
  未完成。
- H100/B200 都在 `interleave32_g2s_b1024_int` 触发
  `cudaErrorIllegalAddress`，平台日志对应 XID 13；故 interleave32 不能作为
  Ventus 对 CUDA 的性能验收门槛。Ventus 自身 1/4 KiB 和多数 16 KiB
  interleave32 功能正确，只有 16 KiB S2G plain/int 在 2 秒风险上限内未完成。

## 当前瓶颈与下一步

优先级建议如下：

1. **先修 Reduce 正确性。** 分别固定复现 mod32、stride144、stride512、
   OOB 和 16 KiB；检查 fragment pending mask、Reduce operand 对应的 byte
   enable、乱序 slot 回收和完成计数。32 KiB 只在这些点正确后再跑。
2. **把 G2S 边际速率从 2.06 推向 1 cycle/line。** PMU 至少拆出
   Planner candidate、L2 request fire、response fire、scatter/transform、
   shared request fire 和 payload-slot stall。目标不是继续增加完整 1024-bit
   entry，而是让 40 个窄 LineContext 持续承载请求、6 个 PayloadSlot 只覆盖
   实际 response→shared 延迟。
3. **解耦跨线 fragment。** 当前任意非零相位固定 +49/+76 cycles，第二
   fragment 不应阻塞后续 line；让两个 fragment 使用独立 LineContext，
   只在同一 destination atom 的最终 byte-valid 合并处会合。
4. **补齐 interleave 的物理并行性。** 先验证 int16 下 TMA 是否已经做到
   L2 request `fire` 连续无气泡；若做到了，差异属于 L2 模型，应在报告中
   明确而不扩大 TMA。若未做到，则优化 line canonicalization 和 request
   arbitration，使独立 channel/line 可连续发出。
5. **暂不优先优化 Descriptor/rank/dtype/swizzle。** 当前数据表明这些部分
   只贡献固定小周期或零边际代价；继续扩大 descriptor 流水不会解决
   2.06 cycle/line 和 Reduce 32× 差距。
6. **性能冻结同时保留两类数。** 用完整命令周期检查软件可见延迟；用
   16→32 KiB slope 和 TMA 内部 PMU 检查结构吞吐。不要用一个固定比例把
   RTL L2 周期校准成 H100/B200 周期。

## 成本、复现与证据

Modal 使用 10 秒平台最小 function timeout、3 秒安全 child timeout、2 秒
风险 child timeout、`retries=0`、`max_containers=1`，且风险分片最后串行。
H100 九个分片 GPU child wall time 合计约 2.892 秒，B200 合计约 4.665 秒
（其中 sub-byte timeout 消耗 2.086 秒）。没有对 XID/timeout 盲目重试。

- H100 run：<https://modal.com/apps/sownelayt/main/ap-OUuwHuttARE6h58H8mSQrL>
- B200 run：<https://modal.com/apps/sownelayt/main/ap-Z0ml16dmat0QDeE0zTId53>
- [H100 raw](../benchmarks/tma-cycle-compare/results/comprehensive_cuda_h100)
- [B200 raw](../benchmarks/tma-cycle-compare/results/comprehensive_cuda_b200)
- [Ventus raw](../benchmarks/tma-cycle-compare/results/comprehensive_ventus)
- [benchmark README](../benchmarks/tma-cycle-compare/README.md)

本轮关键哈希：

| 对象 | SHA-256 |
|---|---|
| shared matrix | `4be6e05ef083c4f52cd0fd2ea360c5bb30d692ee8563e8000f7b04585fa9f05e` |
| CUDA benchmark | `c49333f4a311ab31dc8afa93991e7e5de1d9c1e7aff4d8c3cc62f530c08d114f` |
| installed GVM with-cache | `1f4c31e9c6312bd4801945e1992fff75b106d6f2ce4ffbe2a1ac5e92c632ac5d` |

结果只有一次 recorded repeat，适合识别数量级、固定台阶和结构 slope，
不用于宣称小于数个 cycle 的统计差异。若后续只复核性能，应只重跑本文列出的
容量、相位和 int16 配对点，不必再次付费运行全部 399 项。
