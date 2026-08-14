# Ventus TMA V3.4：40+6 解耦原型与 Bulk 命令 Context 判断

> 状态更新（2026-07-31）：40+6 已经进入生产 TMA RTL，并通过功能、
> 周期和结构门槛。本文保留早期隔离原型及设计决策过程；最终实现结果见
> [`TMA_V3_4_RELEASE_40X6_20260731.md`](TMA_V3_4_RELEASE_40X6_20260731.md)。

## 结论

`40 LineContext + 6 PayloadSlot` 的第一阶段隔离 RTL 原型已经通过。

- 固定 38-cycle cache response latency、256 条连续 128B line 时，
  cache request 的首尾 255 个间隔全部连续，`issue span=255`，
  即严格 `1 request/cycle`。
- 乱序 response、cache request 随机背压、shared 随机背压和 18-cycle
  shared 突发停顿下，6 个 payload slot 全部真实占满。
- payload 满后 cache response 被安全反压；恢复后 128 条 line 全部只
  完成一次，地址和 1024-bit 数据逐项匹配。
- 原型保持单 active command，没有增加第二个 Tensor/Bulk 数据 context。
- 本节描述的是当时的隔离验证状态；后续生产迁移已完成。历史 V3.3
  生成 RTL和已有 DC 输入仍未被覆盖。

原始结果见
[`data/TMA_V3_4_40X6_PROTOTYPE_20260730.csv`](data/TMA_V3_4_40X6_PROTOTYPE_20260730.csv)。

## 原型结构

源码：

- [`../gpgpu/ventus/tests/src/DmaTest/TmaV34DecoupledBufferPrototype.scala`](../gpgpu/ventus/tests/src/DmaTest/TmaV34DecoupledBufferPrototype.scala)
- [`../gpgpu/ventus/tests/src/DmaTest/TmaV34DecoupledBufferPrototype_test.scala`](../gpgpu/ventus/tests/src/DmaTest/TmaV34DecoupledBufferPrototype_test.scala)

结构为：

```text
10 physical groups × 4 narrow subentries
                  = 40 LineContext
                            |
                    cache response
                            |
                       6 PayloadSlot
                            |
                  3-stage transform tags
                            |
                       shared write
```

LineContext 只保存 global/shared route、token、last 和 valid/issued，
不保存 1024-bit 数据。cache response 被接受时，route 和 data 原子转移
到一个 free/recycled PayloadSlot，原 LineContext 同周期释放。

transform 流水只携带 payload ID。PayloadSlot 在 shared request
真正 `fire` 后才释放；同周期释放的 payload 和 LineContext 都允许重新
分配。

原型包含以下结构断言：

- response source 必须命中一个已经发出的 LineContext；
- response 在 `valid && !ready` 期间 source/data/valid 必须稳定；
- issued context 必须仍然 valid；
- transform 内的 payload 必须仍然 valid；
- `accepted - shared completed` 必须等于
  `active LineContext + active PayloadSlot`；
- completion 前两个池都必须排空。

## 测试结果

复现命令：

```bash
cd /home/liyb/ventus-env/gpgpu
./mill -i 'ventus[6.4.0].tests.testOnly' \
  DmaTest.TmaV34DecoupledBufferPrototype_test
```

### 连续固定延迟

```text
V34_40X6_CONTIG,
lines=256,
latency=38,
issue_span=255,
max_line=39,
max_payload=3,
completion_cycle=298
```

39 而不是 40 是因为 response 与新 LineContext allocation 可以同周期
复用同一个 source credit。持续 ready 时 payload 只需覆盖流水驻留，
因此峰值为 3。

### 乱序与背压

```text
V34_40X6_REORDER,
lines=128,
response_stalls=47,
max_line=40,
max_payload=6,
cycles=217
```

该用例把 response 延迟随机化为 5–50 cycle，并在所有已到期 response
中随机选择返回顺序。shared 突发停顿使 6 个 payload 全满，验证了第
4–6 项不是未使用的冗余状态。

## 当前原型证明了什么

已经证明：

1. 40 个窄 source credit 足以覆盖当前约 38-cycle L2 延迟；
2. 1024-bit payload 无需随每个未返回请求一起分配；
3. 六个 payload owner 可以独立吸收 transform/shared 的短期背压；
4. response backpressure、乱序 source、同周期 free+allocate 和完成
   计数能够同时成立；
5. 10×4 分层选择不妨碍每周期发一个 request。

尚未证明：

1. `TmaV2WindowTask` 到 canonical line route 的完整拆分；
2. stride、跨 line、swizzle、interleave、OOB、FP4/FP6；
3. partial masked shared write；
4. S2G 对同一 payload 的多 line write 与 write ack；
5. Reduce；
6. N12 组合面积和 1.5 GHz 时序。

因此该结果是结构可行性门槛通过，不是 V3.4 功能冻结。

## 生产迁移必须处理的结构

### WindowDecomposer

增加一个不含 payload 的 WindowDecomposer，把一个 window 的最多
16 个原始 fragment 串行 canonicalize 成 line token。连续布局一条
window 只产生一条 line，因此仍能保持每周期接受一个 window；多 line
布局受单 cache 端口限制，本来就不能每周期完成多个 line。

### Partial shared write

G2S response 不等待完整 128B window 拼装，而是按 line route 产生
partial masked shared write。互不重叠的 byte mask 可以乱序写入；
命令 completion 必须等待全部 shared ack。

若某种 codec 必须看到完整 atom，则只让对应 fragment 暂存在六项
payload pool 中，不恢复 40 份完整 window payload。

### TLB

不能把当前 request-table 的全表 coalesce 直接从 16 展开到 40，否则
会形成接近 `40×39` 的 VPN 比较网络。应改为：

- 单个 active TLB miss；
- 现有 4 项 translation cache；
- 每周期 RR 选择一个 LineContext 查表/重试；
- response 只填 translation cache，不广播写 40 项。

### Cache source namespace

当前静态约束是：

```text
requestEntries + writeAckEntries <= 64
```

`40 + 32` 会超过 64。由于严格单 active command、方向在完成前不变，
可以按方向复用 source：

```text
G2S source 0..39 = LineContext
S2G source 0..31 = writeAck
descriptor source 63
```

生产约束改为 `max(lineContexts, writeAcks) < 63`，不修改 L2 接口宽度。

## Bulk DMA 是否需要多数据 Context

Bulk 和 Tensor 的区别只在前端：

- Bulk 做地址/长度检查和必要的 G2S mbarrier reserve；
- Tensor 还需要 descriptor lookup/compile 和 Binder；
- 二者进入同一个 WindowEngine、L2、transform 和 shared 数据面。

现有周期中，Bulk 相对 Tensor 的固定优势约为 43–44 cycle，而不是
后端 line rate 更高：

| bytes | Bulk G2S | Tensor G2S | 差值 | Bulk S2G | Tensor S2G | 差值 |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 83 | 127 | 44 | 53 | 96 | 43 |
| 256 | 85 | 129 | 44 | 55 | 98 | 43 |
| 1 KiB | 97 | 141 | 44 | 67 | 110 | 43 |
| 4 KiB | 158 | 202 | 44 | 145 | 188 | 43 |

数据来自
[`data/TMA_V3_3_COMMON_COMMAND_COMPARISON.csv`](data/TMA_V3_3_COMMON_COMMAND_COMPARISON.csv)。

### 大块 Bulk

单条 4–32 KiB Bulk 已经能连续生成 32–256 条 line。40 个 LineContext
足以让单命令填满 L2 流水。增加第二个 active Bulk 不会提高单 cache
request 端口的稳态带宽，只会增加 command owner、ASID、barrier/group、
completion 和仲裁状态。

因此发布结构应继续让 Bulk 和 Tensor 都保持：

```text
1 active data command + 1 prepared lookahead
```

### 很多小 Bulk

大量独立 128B/512B Bulk 是例外。单 active 要在每条命令边界等待旧
response/shared/ack 排空；lookahead 只能隐藏地址检查和 reserve，不能
让下一条命令提前发数据。

处理顺序应为：

1. 地址连续且同步语义相同的软件请求，直接合成一条更大的 Bulk；
2. 增加 `N=1/2/4/8`、`128B/512B/1KiB/4KiB` 的 batched Bulk 实测；
3. 只有真实工作负载证明命令边界是瓶颈时，再增加 Bulk-only
   `BulkSegment FIFO`，而不是恢复通用多 context。

受限 BulkSegment chaining 仍需给每个 line/payload/ack 增加 segment
owner，并独立追踪 barrier/group completion。它本质上是多个 active
data epoch，面积和验证复杂度都不是零。

## 当前决策

- V3.4 数据面继续以单 active command 为前提。
- 40+6 进入下一阶段 WindowDecomposer/partial-write 功能原型。
- Tensor 不增加第二 active context。
- Bulk 暂不增加 active context；先补小命令 batch 测试。
- V3.3、生成 RTL、GVM 和当前 DC 输入保持不变。
