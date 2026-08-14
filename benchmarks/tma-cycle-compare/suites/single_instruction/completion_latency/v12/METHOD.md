# G2S 完成周期测试 V12（ColdThenHotV1）

## 这次测什么

每个测试点只分配一个新的 TensorMap。普通 `ld.global.cg` 可以先把 descriptor 的 128 字节放进 L2，但在第一次 TMA 前不执行 `prefetch.tensormap`，也不允许任何 TMA 使用这个地址。

同一个 kernel 内依次执行两条完全相同的命令：

1. 第一次是该 TensorMap 地址第一次进入 TMAU；
2. 等第一次真正完成后重新初始化 mbarrier；
3. 第二次立即复用完全相同的 TensorMap 地址、内容和 global base。

下一个测试点换一个新的 4 KiB 对齐地址和新的 descriptor 内容。程序记录地址、指纹、`prior_tma_use=0/1` 和 cold/hot 标志，并拒绝同一对内地址变化或不同测试点之间地址复用。

显式预取是独立控制项：第一次 TMA 前执行 `prefetch.tensormap` 并等待 1024 周期；第二次仍是同一地址的热复用。主冷启动结果不包含这条指令。

## 怎样得到周期值

NVIDIA 没有公开 TMA 内部完成时间戳。每个配置执行一轮固定、非自适应扫描，不先粗扫再细扫。640 个位置在编译时固定分成 80 个 kernel bank，每个 bank 固定 8 个位置。bank 由 CPU 在 kernel 发射前选择，不进入 `issue_begin → probe_begin` 计时；GPU 计时路径里没有 `brx.idx`。bank 内每前进一个位置，静态路径增加一条比较、一条统一分支和一条不访问 L2、shared memory 或 TensorMap 的 `pmevent`；bank N 还固定增加 `N×24` 条 PM-event。因此请求步长为三周期。GPU 可能因分支流水、双发射、指令取指和调度产生不同的实际落点间隔，最终一律使用 GPU 上记录的 `clock64`，不能把三条 PTX 直接冒充三个物理周期。

为了让 5 秒防死锁超时覆盖完整项目，`common` 和 `tensor_2d` 的测试项在运行前固定拆成 8 个 case shard；每个配置的 640 个位置始终完整地位于同一个 shard，不会先测一段再根据结果补测。每个 shard 是一个 CUDA context；其内部每个位置使用新的 TensorMap 地址和内容，每张图仍在同一 kernel 内先冷启动、真正完成，再立即热复用。不同 context 之间可能出现相同虚拟地址数值，但 TMAU 状态随 context 消失，不能构成 descriptor 复用。每个子进程超时为 5 秒。

扫描按 `639 → 0` 的固定顺序执行，每个 bank 内也是从长到短。每个 bank 的较长位置先把该 bank 的测量代码带入指令缓存，再接近完成边界；它只使用该点自己的新 TensorMap，不提前接触其他点的 TensorMap。80 个 bank 是同一轮扫描的静态代码布局，不是先跑粗扫、再根据结果决定细扫。

如果一次 `test_wait` 返回 false，cleanup 位于该次 probe 的全部时间戳之后，但必须等命令完成才能开始第二次。这保证第二次确实是同一 TensorMap 的热复用。每个位置、每条命令只有一次计时中的 `test_wait`，不存在轮询间隔。报告只把边界中点显示成一个周期值，原始表另外保存实际探针间距、半区间和非单调点数量以便审计。

## 覆盖范围

- `common`：Bulk 和安全的 Tensor G2S 容量、二维形状、rank、dtype、stride、OOB、swizzle 和合法 interleave16。
- `tensor_2d`：二维 Tensor G2S 容量关键点。
- Reduce、已知 XID 组合和已删除的 elementStride 不运行。

## 不能怎样解释

第一次结果不是“TensorMap 从显存冷读”。descriptor 已经可以在 L2；它测的是“L2 中的 descriptor 第一次被 TMAU 使用”到整个 G2S 完成。第一次减第二次可以估计首次 lookup/refill/处理造成的额外时间，但不是 NVIDIA 未公开的纯解码器计数器。
