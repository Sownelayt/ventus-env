# Ventus DMA/TMA 测试与 GVM 验证全流程

本文档面向第一次接触本项目的人，目标是说明如何从 Chisel RTL 修改开始，一路构建到
实际 `VENTUS_BACKEND=gvm` 验证，并能独立处理 DMA/TMA 相关常见问题。

文档只描述本仓库当前工作流。执行命令默认从仓库根目录
`/home/liyb/ventus-env-copilot-test` 开始。

## 1. 总览：一次 RTL 修改会经过哪些层

DMA/TMA 的端到端验证并非简单地跑一个 Scala 单元测试，而是要经过以下完整流程：

1. 修改 Chisel 源码：主要在 `gpgpu/ventus/src/`。
2. 由 Mill + CIRCT/firtool 生成 SystemVerilog。
3. Verilator 把 SystemVerilog 与 GVM C++ wrapper 编译成 `libVentusGVM-*.so`。
4. `driver` 的 auto-select 后端根据 `VENTUS_BACKEND` 选择 GVM with-cache 或 no-cache 动态库。
5. OpenCL host 程序通过 POCL 编译 `.cl` kernel，kernel 内联 DMA/TMA 自定义指令。
6. GVM 运行 Verilated DUT，同时用 Spike/GVM reference 做状态检查。
7. host 程序读回 global memory，给出 testcase 的最终 PASS/FAIL。

验证时优先看 host testcase 的最终 `OK`/`FAILED`。GVM reference 可能会打印 XREG/VREG mismatch，这些通常是噪声，只有当它导致程序非零退出或 host verdict 失败时才应视为实质失败。

## 2. 项目结构、常用子模块与关键文件

这个仓库是一个集成环境，而非单一源码项目。根仓库负责将多个子模块固定到一组能协同工作的
commit，并通过 `build-ventus.sh` 把工具链、runtime、driver、仿真器和测试统一安装到
`install/`。理解这一点很重要：一个 DMA/TMA host testcase 表面上只是执行 `./xxx.out`，
实际却会同时涉及 LLVM、libclc、POCL、driver、Spike/GVM reference、Chisel RTL 和 Verilator
动态库。

### 2.1 顶层目录地图

根目录常见内容如下：

| 路径 | 类型 | 作用 |
| --- | --- | --- |
| `README.md` / `README_zh_cn.md` | 根文档 | 环境依赖、全量构建、通用后端和回归测试说明。 |
| `Makefile` | 根入口 | 主要提供 `make init` 拉取 submodule。 |
| `.gitmodules` | submodule 清单 | 定义 `spike`、`gpgpu`、`driver`、`pocl` 等子模块来源。 |
| `build-ventus.sh` | 总构建脚本 | 按 `--build` 参数构建并安装各子模块。日常最常用。 |
| `env.sh` | 环境脚本 | 设置 `VENTUS_INSTALL_PREFIX`、`PATH`、`LD_LIBRARY_PATH`、`POCL_DEVICES`、`OCL_ICD_VENDORS`。 |
| `install/` | 安装前缀 | 所有运行时实际使用的 bin/lib/include。注意这是安装产物，不应直接修改。 |
| `llvm/` | submodule | Ventus 定制 LLVM/Clang/Lld/libclc，编译 OpenCL kernel 到 riscv32 Ventus ELF。 |
| `ocl-icd/` | submodule | OpenCL ICD loader，让 OpenCL host 程序能加载 `libpocl.so`。 |
| `pocl/` | submodule | OpenCL runtime 和 Ventus POCL device。负责 build program、分配 buffer、提交 kernel。 |
| `driver/` | submodule | Ventus driver ABI 和各后端实现：spike、rtlsim、gvm、cyclesim、ptx。 |
| `spike/` | submodule | Ventus ISA simulator，包含 DMA/TMA 指令的功能模型，也提供 GVM reference library。 |
| `gpgpu/` | submodule | Chisel RTL、GVM/rtlsim Verilator wrapper、Chisel unit tests。DMA/TMA RTL 修改主要在这里。 |
| `cyclesim/` | submodule | C++ cycle-level simulator，偏性能/周期模型验证。 |
| `systemc/` | submodule | cyclesim 的 SystemC 依赖。 |
| `sbtsim/` | submodule | SBT ELF -> PTX translator，供 PTX 后端使用。 |
| `rodinia/` | submodule | GPU-Rodinia OpenCL benchmark，用于通用回归。 |
| `OpenCL-CTS/` | submodule | OpenCL Conformance Test Suite，通常只在 Spike 上跑软件栈验证。 |
| `testcases/` | submodule | Ventus 自有 OpenCL testcase；DMA/TMA directed suite 位于 `testcases/_get_case/`。 |
| `tools/` | 根工具目录 | 用户直接执行的辅助脚本，例如 `tools/ventus-perf.py`。 |
| `docs/` / `copilot_doc/` / `gpgpu/copilot_doc/` | 文档 | 历史设计、问题分析、实验记录。以源码和当前测试为准，历史文档仅作背景参考。 |
| `regression-test.py` | 根回归脚本 | 跑 Rodinia、POCL example、部分 testcases 的回归集合。 |
| `regression-test-logs/` | 日志目录 | 根回归脚本输出。 |

当前 `.gitmodules` 中的常用 submodule：

| submodule | 上游项目 | 当前项目中的定位 |
| --- | --- | --- |
| `spike` | `ventus-gpgpu-isa-simulator` | ISA 功能模型；`VENTUS_BACKEND=spike`；DMA/TMA 指令语义基准；GVM reference。 |
| `gpgpu` | `ventus-gpgpu` | Chisel RTL；with-cache/no-cache top；Verilator/GVM 构建入口。 |
| `driver` | `ventus-driver` | POCL 到后端仿真器的统一 `vt_*` ABI。 |
| `pocl` | THU-DSP-LAB fork of POCL | OpenCL runtime；Ventus device 插件。 |
| `ocl-icd` | OCL-dev ICD loader | OpenCL 应用链接 `-lOpenCL` 后实际加载 POCL。 |
| `llvm` | THU-DSP-LAB fork of LLVM project | `install/bin/clang`、`lld`、`libclc`、Ventus target。 |
| `systemc` | Accellera SystemC | cyclesim 依赖。 |
| `cyclesim` | Ventus C++ simulator | 周期级模拟器，和 GVM/RTL directed DMA/TMA 验证不是同一个主要路径。 |
| `sbtsim` | Ventus SBT simulator | 把 Ventus ELF 翻译成 PTX，服务 `VENTUS_BACKEND=ptx/sbtsim`。 |
| `rodinia` | GPU-Rodinia | 通用应用级 benchmark。 |
| `OpenCL-CTS` | OpenCL CTS fork | OpenCL 标准一致性测试。 |
| `testcases` | Ventus OpenCL Testcase | 本项目自有 testcase，DMA/TMA 新用例主要加在这里。 |

常用 submodule 命令：

```bash
cd /home/liyb/ventus-env-copilot-test

# 首次初始化
make init
# 等价核心操作
git submodule update --init --recursive --progress

# 查看所有 submodule 当前 commit
git submodule status

# 查看某个子模块是否有未提交修改
git -C gpgpu status --short
git -C testcases status --short
```

注意：根仓库只记录 submodule 指针。如果在 `gpgpu/` 或 `testcases/` 里改文件，根仓库状态通常只显示
`m gpgpu` 或 `m testcases`。需要进入对应子模块看具体改动。

### 2.2 从 OpenCL 程序到 GVM/RTL 的实际调用链

运行一个 DMA/TMA testcase 时，实际链路是：

```text
testcases/_get_case/<case>/<case>.out
  -> libOpenCL.so (ocl-icd)
  -> libpocl.so (POCL runtime)
  -> pocl/lib/CL/devices/ventus/pocl_ventus.cc
  -> install/bin/clang 编译 .cl 到 Ventus riscv32 ELF
  -> libventus_driver.so / auto_select driver
  -> libgvm_driver.so / librtlsim_driver.so / libspike_driver.so / ...
  -> libVentusGVM.so 或 libVentusRTL.so
  -> Verilated Chisel RTL / Spike reference / physical memory model
```

这条链路里每一层负责的事情：

| 层 | 负责什么 | DMA/TMA 调试时看哪里 |
| --- | --- | --- |
| OpenCL host testcase | 构造输入、descriptor、expected model，调用 OpenCL API，判定 PASS/FAIL。 | `testcases/_get_case/<case>/<case>.c`。 |
| OpenCL kernel | 内联 `.word` 发 DMA/TMA 指令，把 shared 结果写回 global。 | `testcases/_get_case/<case>/<case>.cl`。 |
| ocl-icd | 按 `OCL_ICD_VENDORS` 找到 POCL。 | 通常不用改；环境变量错时会找不到 platform/device。 |
| POCL core | 实现 `clCreateBuffer`、`clBuildProgram`、`clEnqueueNDRangeKernel` 等 OpenCL API。 | `pocl/lib/CL/*.c`。 |
| POCL Ventus device | 编译 kernel、分配 device buffer、写 metadata、提交 kernel 给 driver。 | `pocl/lib/CL/devices/ventus/pocl_ventus.cc`。 |
| LLVM/Clang | 把 `.cl` 编译/链接成 Ventus ELF；生成 `object0.riscv`、metadata 等。 | `llvm/`、`install/bin/clang`、`pocl_ventus_post_build_program()`。 |
| driver auto_select | 解析 `VENTUS_BACKEND`，选择具体后端动态库，并切换 with-cache/no-cache symlink。 | `driver/driver/auto_select/ventus.cpp`。 |
| backend driver | 实现统一 `vt_*` ABI，把 buffer copy 和 kernel launch 转成后端调用。 | `driver/driver/gvm_device/ventus.cpp` 等。 |
| GVM/rtlsim library | Verilator 模型、物理内存、CTA 拆分、GVM probe。 | `gpgpu/sim-verilator*/`。 |
| Chisel RTL | 真正的硬件设计。 | `gpgpu/ventus/src/`。 |
| Spike/GVM reference | ISA 功能语义和 GVM 对比参考。 | `spike/riscv/insns/`、`spike/gvmref/`。 |

一个常见的困惑是工作目录问题：host testcase 通过相对路径打开 `.cl` 文件，因此必须从 testcase 所在目录运行。这并非 POCL 或 RTL 的 bug。

### 2.3 `install/` 目录的意义

`install/` 是当前项目运行时真正使用的安装前缀。`source ./env.sh` 后，工具和库优先来自这里：

| 路径 | 典型内容 |
| --- | --- |
| `install/bin/clang` | Ventus 定制 clang。 |
| `install/bin/llvm-objdump` | 反汇编 Ventus ELF。 |
| `install/bin/sbt_ptx` | PTX/SBT 后端翻译工具，若构建了 `sbtsim`。 |
| `install/lib/libOpenCL.so` | ocl-icd 安装产物。 |
| `install/lib/libpocl.so` | POCL ICD/runtime。 |
| `install/lib/pocl/libpocl-devices-ventus.so` | POCL Ventus device 插件。 |
| `install/lib/libventus_driver.so` | auto-select driver。 |
| `install/lib/libspike_driver.so` | Spike backend driver。 |
| `install/lib/librtlsim_driver.so` | RTL backend driver。 |
| `install/lib/libgvm_driver.so` | GVM backend driver。 |
| `install/lib/libcyclesim_driver.so` | cyclesim backend driver。 |
| `install/lib/libVentusRTL-withcache.so` | with-cache RTL Verilator library。 |
| `install/lib/libVentusRTL-nocache.so` | no-cache RTL Verilator library。 |
| `install/lib/libVentusGVM-withcache.so` | with-cache GVM Verilator library。 |
| `install/lib/libVentusGVM-nocache.so` | no-cache GVM Verilator library。 |
| `install/lib/libVentusGVM.so` | auto-select 运行时切换的 symlink。 |
| `install/lib/libgvmref.so` | Spike GVM reference library。 |
| `install/lib/crt0.o`、`riscv32clc.o`、`ldscripts/` | Ventus kernel 链接所需 runtime。 |
| `install/include/CL/` | OpenCL headers。 |
| `install/include/ventus_rtlsim.h` | rtlsim/GVM C API 头。 |

如果源码已经改了但运行行为没有变化，优先确认是否忘了把对应子模块重新安装到 `install/`。

### 2.4 `build-ventus.sh` 与子模块构建关系

`build-ventus.sh --build "xxx;yyy"` 会按名称调用内部函数。常见目标：

| build 目标 | 构建内容 | 什么时候需要 |
| --- | --- | --- |
| `llvm` | Ventus LLVM/Clang/Lld。 | 改 LLVM 或首次完整构建。 |
| `libclc` | Ventus libclc 和 `riscv32clc.o`。 | 改 libclc 或重新铺环境。 |
| `ocl-icd` | OpenCL ICD loader。 | 首次构建或 ICD 改动。 |
| `spike` | Spike ISA simulator 和 `libgvmref.so`。 | 改 ISA 语义、GVM reference 或首次 GVM 前置依赖。 |
| `rtlsim` / `rtl` / `gpgpu` | with-cache/no-cache RTL Verilator library。 | 只跑 `VENTUS_BACKEND=rtl/rtlsim` 时需要。 |
| `gvm` | with-cache/no-cache GVM Verilator library。 | 改 Chisel RTL 后跑 GVM 必须。 |
| `cyclesim` | C++ cycle-level simulator。 | 跑 `VENTUS_BACKEND=cyclesim`。 |
| `sbtsim` / `ptx` | SBT ELF -> PTX translator。 | 跑 `VENTUS_BACKEND=ptx/sbtsim`。 |
| `driver` | `libventus_driver.so` 和各 backend driver。 | 改 driver、切换 ABI、首次完整构建。 |
| `pocl` | POCL runtime + Ventus device。 | 改 POCL、OpenCL device 行为或首次完整构建。 |
| `rodinia` | Rodinia benchmark。 | 跑通用回归前准备。 |
| `cts` | OpenCL CTS。 | 做 OpenCL 标准一致性检查。 |
| `test-pocl` | POCL example 冒烟测试。 | 检查基本软件栈。 |

构建依赖关系可以粗略理解为：

```text
llvm + libclc + ocl-icd + spike + gpgpu/gvm
  -> driver
  -> pocl
  -> rodinia/testcases/OpenCL-CTS run
```

实际 `build-ventus.sh` 会做一些存在性检查，例如 driver 需要先有 `libVentusGVM-withcache.so`、
`libVentusGVM-nocache.so` 和 `libgvmref.so`。

### 2.5 常用子模块详解

#### `llvm/`

`llvm/` 是 Ventus 编译器工具链。日常 testcase 里的 `.cl` kernel 最终会通过
`install/bin/clang` 编译为 RISC-V 32-bit Ventus ELF。

常用入口：

| 路径 | 作用 |
| --- | --- |
| `llvm/llvm/` | LLVM core。 |
| `llvm/clang/` | Clang frontend。 |
| `llvm/lld/` | linker。 |
| `llvm/libclc/` | OpenCL C device library。 |
| `llvm/libclc/build_riscv32clc.sh` | 构建 Ventus riscv32 libclc object 的脚本。 |

DMA/TMA RTL 验证中通常不改 LLVM。只有当 `.cl` 内联 asm、kernel ABI、metadata section、
resource section 或 Ventus 指令编码在编译端需要调整时才看这里。

#### `pocl/`

`pocl/` 是 OpenCL runtime。Ventus device 插件在：

```text
pocl/lib/CL/devices/ventus/
  pocl_ventus.cc
  pocl_ventus.h
  loadelf.cpp
  loadelf.hpp
  CMakeLists.txt
```

关键点：

- `pocl/lib/CL/devices/devices.c` 在 `BUILD_VENTUS` 时注册 `ventus` device。
- `pocl/lib/CL/devices/ventus/CMakeLists.txt` 构建 `pocl-devices-ventus`，并链接 `ventus_driver`。
- `pocl_ventus_init_device_ops()` 填 POCL device ops，包括 build、run、read/write、alloc/free。
- `pocl_ventus_post_build_program()` 会调用 `VENTUS_INSTALL_PREFIX/bin/clang` 编译 kernel。
- `pocl_ventus_run()` 准备 kernel metadata、args、private memory、global/local size，然后调用 driver。

DMA/TMA testcase 中常见的 `object0.cl`、`object0.riscv`、`object0.riscv.log` 就是这一层产生的。
如果 OpenCL build 失败，先查看 host 输出的 build log，再看 POCL 的中间产物。

#### `driver/`

`driver/` 提供 POCL 调用后端的统一 ABI。核心头文件：

```text
driver/include/ventus.h
```

常见后端实现：

| 路径 | 产物 | 对应后端 |
| --- | --- | --- |
| `driver/driver/auto_select/ventus.cpp` | `libventus_driver.so` | 入口选择器。 |
| `driver/driver/spike_device/ventus.cpp` | `libspike_driver.so` | `VENTUS_BACKEND=spike/isa`。 |
| `driver/driver/rtlsim_device/ventus.cpp` | `librtlsim_driver.so` | `VENTUS_BACKEND=rtl/rtlsim/gpgpu`。 |
| `driver/driver/gvm_device/ventus.cpp` | `libgvm_driver.so` | `VENTUS_BACKEND=gvm`。 |
| `driver/driver/cyclesim_device/ventus.cpp` | `libcyclesim_driver.so` | `VENTUS_BACKEND=cyclesim`。 |
| `driver/driver/ptx_device/ventus.cpp` | `libptx_driver.so` | `VENTUS_BACKEND=ptx/sbt/sbtsim`。 |

auto-select 的实际规则：

- 未设置 `VENTUS_BACKEND` 时默认 `spike`。
- `gvm` 选择 `libgvm_driver.so`。
- `gvm-withcache` 和 `gvm` 都把 `install/lib/libVentusGVM.so` 指向
  `libVentusGVM-withcache.so`。
- `gvm-nocache` 把 `install/lib/libVentusGVM.so` 指向 `libVentusGVM-nocache.so`。
- `rtl-withcache` / `rtl-nocache` 同理切换 `libVentusRTL.so`。

GVM driver 本身并不是模拟器，它通过 `ventus_rtlsim.h` API 驱动 `libVentusGVM.so`，同时使用
Spike reference 进行初始化和状态对比。DMA/TMA directed RTL 验证主要走这个路径。

#### `spike/`

`spike/` 是功能模型和 GVM reference 的来源。DMA/TMA 指令语义集中在：

```text
spike/riscv/insns/cp_async_bulk.h
spike/riscv/insns/cp_async_copysize.h
spike/riscv/insns/cp_async_tensor.h
spike/riscv/insns/cp_async_tensor_g2s.h
spike/riscv/insns/cp_async_fence.h
spike/riscv/insns/prefetch_tensormap.h
```

Spike 语义特点：

- bulk/copysize/tensor 都是同步 copy。
- `CP_ASYNC_FENCE` 是 NOP。
- `PREFETCH_TENSORMAP` 是 NOP。
- `CP_ASYNC_TENSOR_G2S` 支持 descriptor v0、dynamic coords、OOB fill、swizzle。

因此，Spike 很适合用于排查 testcase 和 descriptor 层面的错误，但无法验证 RTL 中的异步 DMA、fence、
shared routing、cacheline 拆分等行为是否正确。

#### `gpgpu/`

`gpgpu/` 是 Chisel RTL 和 Verilator/GVM 构建的核心。

常用目录：

| 路径 | 作用 |
| --- | --- |
| `gpgpu/ventus/src/pipeline/` | pipeline、issue、decode、DMA core、warp scheduler、ALU/FPU/TC 等。 |
| `gpgpu/ventus/src/top/` | 顶层 wrapper、with-cache/no-cache top、参数导出、external mem model。 |
| `gpgpu/ventus/src/L1Cache/` | ICache、DCache、ShareMem、MSHR、bank conflict 等。 |
| `gpgpu/ventus/src/L2cache/` | L2 cache。 |
| `gpgpu/ventus/src/cta/` | CTA scheduler、resource table、CTA2warp。 |
| `gpgpu/ventus/src/mmu/` | TLB/PTW/MMU 相关。 |
| `gpgpu/ventus/src/axi/` | AXI wrapper。 |
| `gpgpu/ventus/tests/src/` | Chisel/Scala unit tests。 |
| `gpgpu/sim-verilator/` | with-cache RTL/GVM Verilator wrapper。 |
| `gpgpu/sim-verilator-nocache/` | no-cache RTL/GVM Verilator wrapper。 |
| `gpgpu/copilot_doc/` | 历史设计/调试文档。 |

DMA/TMA RTL 主要相关：

| 文件 | 为什么重要 |
| --- | --- |
| `DMA_core.scala` | 产生 L2 请求、追踪 tag、接收 L2 response、OOB fill、写 shared、产生 DMA completion。 |
| `DecodeUnit.scala` | 决定 `CP_ASYNC_TENSOR` 用 VGPR，bulk/copysize 用 scalar register。 |
| `issue.scala` | 把 DMA 指令从 issue 队列送到 DMA path。 |
| `pipe.scala` | 连接 `DMA_core`、warp scheduler、shared/cache/TLB ports。 |
| `warp_schedule.scala` | `CP_ASYNC_FENCE` 对当前 warp 的阻塞释放。 |
| `GPGPU_top.scala` | with-cache 的 DMA L2/cache/shared/TLB 接线。 |
| `GPGPU_top_nocache.scala` | no-cache 的 DMA dcache bypass adapter 和 response routing。 |
| `parameters.scala` | DMA cacheline、tag、temp mem、aligned bulk 等参数。 |

#### `testcases/`

`testcases/` 是 OpenCL application-level testcase 子模块。与 DMA/TMA 相关的测试集中在：

```text
testcases/_get_case/
  dma_test/
  copysize_test/
  tensor_dma_test/
  tma_descriptor_test/
  tma_matrix_test/
  bulk_dma_matrix_test/
  multi_warp_dma_fence_test/
  dma_shared_routing_conflict_test/
  multi_wg_dma_test/
  cases_dma_tma.csv
  run_dma_tma_rtl.sh
```

测试程序一般由：

- `Makefile`：编译 host `.c` 为 `.out`。
- `<case>.c`：OpenCL host、输入构造、expected model、验证逻辑。
- `<case>.cl`：device kernel，内联 DMA/TMA `.word` 指令。
- `object0.*`：POCL 编译运行后产生的中间文件。

#### `cyclesim/` 与 `systemc/`

`cyclesim/` 是 C++ cycle-level simulator，依赖 `systemc/`。它更侧重于周期/性能模型的验证，
不是 DMA/TMA RTL directed suite 的首选后端。只有当需要对比 `VENTUS_BACKEND=cyclesim`
的行为或调试 cyclesim 专属问题时才需要关注这里。

#### `sbtsim/` 与 PTX 后端

`sbtsim/` 提供 `sbt_ptx`，把 Ventus ELF 翻译为 PTX。`driver/driver/ptx_device/ventus.cpp`
会调用它并使用 CUDA Driver API 执行 PTX。这个后端适合某些性能/功能对照，不用于验证 Chisel
DMA/TMA RTL。

#### `rodinia/` 与 `OpenCL-CTS/`

- `rodinia/` 用于应用级回归，覆盖更真实的 OpenCL workload。
- `OpenCL-CTS/` 用于 OpenCL 标准一致性，规模很大，通常只在 Spike 上跑。

DMA/TMA 修改如果只影响 DMA core，优先跑 directed suite；如果改了通用 runtime、driver、
compiler 或 memory system，再扩展到 Rodinia/CTS。

### 2.6 Chisel/RTL

| 路径 | 用途 |
| --- | --- |
| `gpgpu/ventus/src/pipeline/DMA_core.scala` | DMA/TMA 核心实现：L2 请求、Temp_mem、shared 写入、TMA descriptor、OOB fill、swizzle、prefetch 等。 |
| `gpgpu/ventus/src/pipeline/DecodeUnit.scala` | DMA/TMA opcode 解码，决定 `CP_ASYNC_*` 使用 scalar 还是 vector operand。 |
| `gpgpu/ventus/src/pipeline/pipe.scala` | issue 到 `DMA_core`，连接 DMA cache/shared/TLB/fence 端口，接入 warp scheduler。 |
| `gpgpu/ventus/src/pipeline/warp_schedule.scala` | per-warp DMA inflight 计数与 `CP_ASYNC_FENCE` 阻塞/释放。 |
| `gpgpu/ventus/src/top/GPGPU_top.scala` | with-cache 顶层：DMA L2 cache port、DMA shared port、TLB、GVM probe。 |
| `gpgpu/ventus/src/top/GPGPU_top_nocache.scala` | no-cache 顶层：把 DMA cacheline request 转成 no-cache dcache bypass request。 |
| `gpgpu/ventus/src/top/parameters.scala` | DMA 参数：128B cacheline、4B DMA shared 写粒度、tag/inst 数量等。 |
| `gpgpu/ventus/src/GvmDutApi.scala` | DUT 与 GVM reference 之间的 probe/API。 |

### 2.7 GVM 构建入口

| 路径 | 用途 |
| --- | --- |
| `gpgpu/sim-verilator/gvm.mk` | with-cache GVM 构建：top module 是 `GPGPU_SimTop`，安装为 `libVentusGVM-withcache.so`。 |
| `gpgpu/sim-verilator-nocache/gvm.mk` | no-cache GVM 构建：top module 是 `GPGPU_top_nocache`，安装为 `libVentusGVM-nocache.so`。 |
| `build-ventus.sh` | 总构建脚本。`--build "gvm"` 会构建并安装 with-cache 和 no-cache GVM。 |
| `driver/driver/auto_select/ventus.cpp` | 解析 `VENTUS_BACKEND`，为 `libVentusGVM.so` 选择 with-cache/no-cache 实现。 |

### 2.8 DMA/TMA application-level 测试

主要都在 `testcases/_get_case/`：

| 测试 | 作用 |
| --- | --- |
| `dma_test/` | 单 workgroup，单 warp leader 发 `CP_ASYNC_BULK + CP_ASYNC_FENCE`。 |
| `copysize_test/` | `CP_ASYNC_COPYSIZE` 固定 16B 冒烟测试。 |
| `tensor_dma_test/` | `CP_ASYNC_TENSOR` 2D FP32 4x4 冒烟测试。 |
| `tma_descriptor_test/` | descriptor-addressed TMA smoke，覆盖 no-prefetch 与 `PREFETCH_TENSORMAP`。 |
| `multi_wg_dma_test/` | 多 workgroup 各自发 bulk DMA。 |
| `tma_matrix_test/` | 最重要的 TMA 矩阵测试：1D/2D/3D、subbox、elementStride、swizzle、OOB。 |
| `bulk_dma_matrix_test/` | bulk 跨 128B cacheline、shared dst offset。 |
| `multi_warp_dma_fence_test/` | 同 workgroup 多 warp、多 DMA、per-warp fence。 |
| `dma_shared_routing_conflict_test/` | shared bank conflict 与 DMA shared response routing 压力。 |

专用套件入口：

| 路径 | 用途 |
| --- | --- |
| `testcases/_get_case/cases_dma_tma.csv` | DMA/TMA suite 清单。 |
| `testcases/_get_case/run_dma_tma_rtl.sh` | 构建缺失 `.out` 并运行当前 directed DMA/TMA suite。 |
| `testcases/DMA_TMA_RTL_TEST_DESIGN.md` | 已有设计与覆盖状态记录，可配合本文档阅读。 |

### 2.9 按任务快速定位模块

遇到问题时可以先按下面的表定位，不要一上来就重建所有东西：

| 你要做的事 / 看到的现象 | 优先看哪里 | 通常需要重建什么 | 推荐先跑什么 |
| --- | --- | --- | --- |
| 新增 DMA/TMA application testcase | `testcases/_get_case/<new_test>/`、`cases_dma_tma.csv`、`run_dma_tma_rtl.sh` | 只需要 `make` 当前 testcase | 当前 testcase 的 Spike，再 GVM。 |
| 修改 TMA expected model 或 descriptor 构造 | `testcases/_get_case/tma_matrix_test/tma_matrix_test.c` | 只需要 `make` 当前 testcase | `VENTUS_BACKEND=spike ./tma_matrix_test.out <case>`。 |
| kernel asm 编译失败或指令没出现在反汇编里 | `<case>.cl`、`pocl_ventus_post_build_program()`、`install/bin/clang` | 当前 testcase；必要时重建 POCL/LLVM | 手动 `llvm-objdump`，再 Spike。 |
| OpenCL API 报错，例如 build/queue/buffer 失败 | host `.c`、`pocl/lib/CL/`、`pocl/lib/CL/devices/ventus/` | 多数只重编 testcase；POCL 改动才重建 `pocl` | Spike 单 case，查看 build log。 |
| `VENTUS_BACKEND` 没按预期选择后端 | `driver/driver/auto_select/ventus.cpp`、`install/lib/libVentus*.so` symlink | 改 driver 才重建 `driver` | 任意小 testcase，例如 `vecadd` 或 `bulk_dma_matrix_test`。 |
| Spike 与 expected 不一致 | `spike/riscv/insns/cp_async_*.h`、host expected model | 改 Spike 后重建 `spike`，GVM reference 变了则重建 `gvm` | Spike 单 case。 |
| Spike PASS，GVM host FAIL | `gpgpu/ventus/src/`，尤其 DMA/shared/cache/top/scheduler | 改 Chisel 后重建 `gvm` | 同一单 case 的 GVM with-cache；必要时 no-cache。 |
| GVM checker mismatch 但 host OK | `spike/gvmref/`、GVM probe、`GvmDutApi.scala` | 只有修 checker/probe 才重建相关模块 | 记录日志，优先确认 host verdict。 |
| DMA fence hang / timeout | `DMA_core.scala`、`pipe.scala`、`warp_schedule.scala` | `gvm` | `multi_warp_dma_fence_test` 单 case + waveform。 |
| shared bank conflict 下 DMA 数据错 | `GPGPU_top.scala`、`GPGPU_top_nocache.scala`、`L1Cache/ShareMem` | `gvm` | `dma_shared_routing_conflict_test`。 |
| with-cache 错，no-cache 对 | `GPGPU_top.scala`、L1/L2 cache、DMA L2 port | `gvm` with-cache | `bulk_dma_matrix_test` 跨线 case。 |
| no-cache 错，with-cache 对 | `GPGPU_top_nocache.scala` 的 DMA dcache adapter/route FIFO | `gvm` no-cache | `VENTUS_BACKEND=gvm-nocache` directed suite。 |
| 通用 benchmark 回归出现失败 | `regression-test.py`、`rodinia/`、`pocl/`、`driver/`、memory system | 按改动模块重建 | 对应 checklist 或单 benchmark。 |
| OpenCL 标准行为需要确认 | `OpenCL-CTS/`、`pocl/`、`llvm/` | `cts`、必要时 POCL/LLVM | Spike 上跑相关 CTS topic。 |

## 3. 环境准备

每次运行 Ventus 相关命令前都要先：

```bash
cd /home/liyb/ventus-env-copilot-test
source ./env.sh
```

`env.sh` 会设置：

| 变量 | 说明 |
| --- | --- |
| `VENTUS_INSTALL_PREFIX` | 默认是当前仓库的 `install/`。 |
| `PATH` | 把 `install/bin` 放到最前面。 |
| `LD_LIBRARY_PATH` | 把 `install/lib` 放到最前面。 |
| `POCL_DEVICES=ventus` | 让 POCL 使用 Ventus device。 |
| `OCL_ICD_VENDORS=install/lib/libpocl.so` | 让 OpenCL ICD 走本项目安装的 POCL。 |
| `POCL_ENABLE_UNINIT=1` | 当前项目需要的 POCL 行为开关。 |

常用后端：

| `VENTUS_BACKEND` | 含义 |
| --- | --- |
| unset / `spike` / `isa` | Spike ISA 模型，默认后端。 |
| `gvm` / `gvm-withcache` | GVM with-cache，默认 GVM 变体。 |
| `gvm-nocache` / `gvm-no-cache` | GVM no-cache。 |
| `rtl` / `rtlsim` / `gpgpu` | Verilator RTL 后端，不带 GVM reference。 |
| `rtl-withcache` / `rtl-nocache` | RTL with-cache/no-cache 变体。 |
| `cyclesim` | C++ cycle-level simulator。 |

如果 GVM 动态库不存在，先构建 GVM；如果 driver 或 POCL 本身改了，再额外重建 driver/POCL。

## 4. 从 Chisel 修改到 GVM 安装

### 4.1 修改前先跑基线

GVM/RTL 构建很慢，不建议无准备地直接修改。修改前先确认当前 DMA/TMA suite 的通过/失败情况：

```bash
cd /home/liyb/ventus-env-copilot-test
source ./env.sh

base=/tmp/ventus-dma-tma-baseline-$(date +%Y%m%d-%H%M%S)
mkdir -p "$base"

(
  cd testcases/_get_case
  VENTUS_BACKEND=spike ./run_dma_tma_rtl.sh "$base/spike"
) >"$base/spike.suite.log" 2>&1

(
  cd testcases/_get_case
  VENTUS_BACKEND=gvm ./run_dma_tma_rtl.sh "$base/gvm"
) >"$base/gvm.suite.log" 2>&1
```

只看摘要，不要直接打开整份 GVM 日志：

```bash
grep -E "^\[[[:space:]]*[0-9]+/[0-9]+\]|PASS|FAIL|SKIP|summary|OK|FAILED|pass:|fail:|skip:|FATAL|GVM ERROR|PC mismatch|ctx exp|ctx got" "$base/gvm.suite.log" | tail -n 120
```

如果修改目标涉及 no-cache，基线也跑：

```bash
(
  cd testcases/_get_case
  VENTUS_BACKEND=gvm-nocache ./run_dma_tma_rtl.sh "$base/gvm-nocache"
) >"$base/gvm-nocache.suite.log" 2>&1
```

### 4.2 修改 Chisel 源码

常见 DMA/TMA 修改入口：

| 要改的行为 | 首先看 |
| --- | --- |
| Bulk DMA cacheline 拆分、source tag、finish count | `DMA_core.scala` 的 `AddrCalc_l2cache`、`Temp_mem`。 |
| TMA descriptor 字段、dataType、rank、stride、subbox | `DMA_core.scala` 的 tensor setup / descriptor decode。 |
| OOB fill、swizzle、elementStride 到 shared 的落点 | `DMA_core.scala` 的 `Temp_mem`、`Addrcalc_shared`。 |
| DMA fence 卡住或 warp 不释放 | `pipe.scala` 的 `fence_end_dma` ready 路径、`warp_schedule.scala` inflight/fence wait。 |
| shared bank conflict 后 DMA response 丢失 | `GPGPU_top.scala` / `GPGPU_top_nocache.scala` shared request arbitration 和 `sourceTag` routing。 |
| no-cache 下 DMA 不通 | `GPGPU_top_nocache.scala` 的 DMA dcache adapter 与 response route FIFO。 |
| 指令 operand 类型不对 | `DecodeUnit.scala`。 |
| ISA 语义改变 | 同步改 `spike/riscv/insns/cp_async_*.h` 和 testcase expected model。 |

几个不能随便破坏的不变量：

- `dma_aligned_bulk = 4`，DMA 写 shared 的基本粒度是 4B。
- `l2cacheline = dcache_BlockWords * BytesOfWord = 32 * 4 = 128B`。
- `pipe.scala` 里 `dma_core.io.fence_end_dma.ready` 必须保持无条件 `true.B`，否则 Temp_mem reset/完成路径可能死锁。
- shared memory response 的 `sourceTag` 必须穿过 bank conflict replay，再路由到普通 pipe 或 DMA。
- TMA dataType 编码必须与 Spike 和 host test 保持一致。

### 4.3 快速编译检查

只检查 Scala/Chisel 编译：

```bash
cd /home/liyb/ventus-env-copilot-test/gpgpu
./mill -i -j 0 __.compile
```

也可以走 `gpgpu/Makefile`：

```bash
cd /home/liyb/ventus-env-copilot-test/gpgpu
make compile
```

这一步不能替代 GVM 构建，只是更快发现 Scala 语法和类型问题。

### 4.4 构建并安装 GVM

推荐一次性构建 with-cache 和 no-cache：

```bash
cd /home/liyb/ventus-env-copilot-test
source ./env.sh
bash build-ventus.sh --build "gvm"
```

如果只想构建 with-cache：

```bash
cd /home/liyb/ventus-env-copilot-test
source ./env.sh
make -C gpgpu/sim-verilator -f gvm.mk -j"$(nproc)" \
  RELEASE=1 GVM_TRACE=1 \
  GVM_REF_DIR="$PWD/install/lib" \
  PREFIX="$PWD/install" \
  install
```

如果只想构建 no-cache：

```bash
cd /home/liyb/ventus-env-copilot-test
source ./env.sh
make -C gpgpu/sim-verilator-nocache -f gvm.mk -j"$(nproc)" \
  RELEASE=1 GVM_TRACE=1 \
  GVM_REF_DIR="$PWD/install/lib" \
  PREFIX="$PWD/install" \
  install
```

构建产物：

| 文件 | 含义 |
| --- | --- |
| `install/lib/libVentusGVM-withcache.so` | with-cache GVM。 |
| `install/lib/libVentusGVM-nocache.so` | no-cache GVM。 |
| `install/lib/libVentusGVM.so` | 运行时由 auto-select driver 原子切换到对应变体。 |

`gvm.mk` 内部做的事：

1. 扫描 `gpgpu/ventus/src/**/*.scala`。
2. with-cache：运行 `./mill ventus[6.4.0].runMain top.paramToJson` 生成参数 JSON。
3. with-cache：运行 `circt.stage.ChiselMain --module top.GPGPU_SimTop` 生成 FIRRTL。
4. no-cache：运行 `circt.stage.ChiselMain --module top.GPGPU_top_nocache`。
5. `firtool --split-verilog` 输出 `sim-verilator*/verilog-out/*.sv`。
6. Verilator 编译 `.sv` 和 GVM C++ wrapper。
7. 链接 `libgvmref.so`，安装到 `install/lib`。

如果报 `Please build spike gvm reference library (libgvmref.so) for GVM!`，先构建 Spike：

```bash
cd /home/liyb/ventus-env-copilot-test
source ./env.sh
bash build-ventus.sh --build "spike"
bash build-ventus.sh --build "gvm"
```

如果怀疑 `verilog-out/` 是旧的，可以清掉对应 GVM 生成物后重建：

```bash
cd /home/liyb/ventus-env-copilot-test
source ./env.sh
make -C gpgpu/sim-verilator -f gvm.mk clean-gvm
make -C gpgpu/sim-verilator-nocache -f gvm.mk clean
bash build-ventus.sh --build "gvm"
```

### 4.5 什么时候需要重建 driver/POCL

只改 `gpgpu/ventus/src/**/*.scala`：通常只需重建 GVM。

改了 `driver/`：重建 driver：

```bash
cd /home/liyb/ventus-env-copilot-test
source ./env.sh
bash build-ventus.sh --build "driver"
```

改了 `pocl/` 或 OpenCL 编译流程：重建 POCL：

```bash
cd /home/liyb/ventus-env-copilot-test
source ./env.sh
bash build-ventus.sh --build "pocl"
```

改了 Spike ISA 模型：重建 Spike，必要时再重建 GVM 和 driver：

```bash
cd /home/liyb/ventus-env-copilot-test
source ./env.sh
bash build-ventus.sh --build "spike;gvm;driver"
```

## 5. DMA/TMA 指令和 descriptor 速查

### 5.1 指令语义与当前测试用编码

| 指令 | 语义 | 当前测试里常见编码 |
| --- | --- | --- |
| `CP_ASYNC_COPYSIZE` | 拷贝 `4 << inst[26:25]` 字节，从 global 到 shared。 | `0x04c505c2` 表示 copysize=2，即 16B。 |
| `CP_ASYNC_BULK` | 拷贝 `rs2` 字节，从 `rs1` global 到 `rd` shared。 | `0x00c515c2`，测试里把 `x10=src`、`x11=dst_shared`、`x12=size`。 |
| `CP_ASYNC_TENSOR` | VGPR descriptor 形式的 TMA，从 global tensor box 到 shared。 | `0x00c535c2`，测试里用 `v10=VRS1`、`v12=VRS2`、`v11=VRS3`。 |
| `CP_ASYNC_FENCE` | RTL 中等待当前 warp 的 DMA 完成；Spike 中是 NOP。 | `0x00004042`。 |
| `PREFETCH_TENSORMAP` | descriptor-addressed TMA 的预取 hint；Spike 中是 NOP。 | `0x0005d042`。 |
| `CP_ASYNC_TENSOR_G2S` | descriptor-addressed TMA，从 128B tensor map descriptor 读取参数。 | `0x00c5e542`。 |

Spike 对 bulk/copysize/tensor 都是同步拷贝，因此主要用于验证 testcase/model/descriptor
是否正确；GVM 才能反映真实的 RTL pipeline、fence、routing、cacheline 行为。

### 5.2 `CP_ASYNC_TENSOR` 的 96-word VGPR descriptor

`tma_matrix_test` 和 `tensor_dma_test` 使用这个形式。host/kernel 把 96 个 `uint32_t`
拆成三组，每组 32 个 word：

| 区间 | 名称 | 关键字段 |
| --- | --- | --- |
| `param_buf[0..31]` | VRS1 tensor descriptor | `[0] dataType`、`[1] rank`、`[2] globalAddress`、`[3..7] globalDim`、`[8..12] globalStrides`。 |
| `param_buf[32..63]` | VRS2 box descriptor | `[32] BoxAddress`、`[33..37] boxDim`、`[38..42] elementStrides`、`[43] interleave`、`[44] swizzle`、`[46] oobfill`。 |
| `param_buf[64..95]` | VRS3 destination | `[64] dst shared address`。 |

注意点：

- `globalStrides[d]` 是字节 stride。对 rank=N，未使用的 stride slot 应保持 0。
- dim0 的 stride 由 `datawidth * elementStride[0]` 决定，dim1 及以上用 `globalStrides[d-1]`。
- `elementStrides[d]` 是元素步长，不是字节数。
- destination 是 packed row-major，并可能受 swizzle 影响。

### 5.3 descriptor-addressed TMA 的 32-word descriptor

`tma_descriptor_test` 使用 descriptor-addressed 形式：

| word | 含义 |
| --- | --- |
| `0` | magic/version/flags，目前主要是信息字段。 |
| `1` | control：bits `[3:0] dataType`、`[7:4] rank`、`[9:8] interleave`、`[11:10] swizzle`、`[13:12] L2promotion`、`[14] oobfill`。 |
| `2` | `globalAddress`，kernel 会 patch 成运行时 src pointer。 |
| `3` | descriptor size/reserved。 |
| `4..8` | `globalDim[0..4]`。 |
| `9..13` | `byteStride[0..4]`，包含 dim0 byte stride。 |
| `14..18` | `boxDim[0..4]`。 |
| `19..23` | `elementStrides[0..4]`。 |
| `24..31` | reserved。 |

动态坐标 block `coords[0..4]` 用于决定 box 起点。`PREFETCH_TENSORMAP` 可以在
`CP_ASYNC_TENSOR_G2S` 前发出，但 Spike 不维护 cache 状态，所以只验证它不会破坏结果。

### 5.4 dataType 编码

这个表必须在 `DMA_core.scala`、Spike、host testcase 中保持一致：

| code | 类型 | 字节 |
| --- | --- | --- |
| 0 | `UINT8` | 1 |
| 1 | `UINT16` | 2 |
| 2 | `UINT32` | 4 |
| 3 | `INT8` | 1 |
| 4 | `INT16` | 2 |
| 5 | `INT32` | 4 |
| 6 | `FP32` | 4 |
| 7 | `FP16` | 2 |
| 8 | `BF16` | 2 |
| 9 | `UINT64` | 8 |
| 10 | `INT64` | 8 |
| 11 | `FP64` | 8 |

## 6. 运行 DMA/TMA 测试

### 6.1 最推荐：专用 DMA/TMA suite

先跑 Spike，再跑 GVM：

```bash
cd /home/liyb/ventus-env-copilot-test
source ./env.sh

base=/tmp/ventus-dma-tma-suite-$(date +%Y%m%d-%H%M%S)
mkdir -p "$base"

(
  cd testcases/_get_case
  VENTUS_BACKEND=spike ./run_dma_tma_rtl.sh "$base/spike"
) >"$base/spike.suite.log" 2>&1

(
  cd testcases/_get_case
  VENTUS_BACKEND=gvm ./run_dma_tma_rtl.sh "$base/gvm"
) >"$base/gvm.suite.log" 2>&1
```

no-cache：

```bash
(
  cd testcases/_get_case
  VENTUS_BACKEND=gvm-nocache ./run_dma_tma_rtl.sh "$base/gvm-nocache"
) >"$base/gvm-nocache.suite.log" 2>&1
```

注意：当前 `run_dma_tma_rtl.sh` 仅在 `VENTUS_BACKEND=gvm` 时对 `tma_matrix_test` 自动设置
`VENTUS_TMA_RUN_RTL_ONLY=1`；当 `VENTUS_BACKEND=gvm-nocache` 时脚本会清除该变量，因此
no-cache suite 中的 `tma_matrix_test` 跑的是非 RTL-directed 路径（不含 RTL-only case）。
如需跑完整的 23-case RTL-directed 路径，应单独执行 `tma_matrix_test` 并显式设置
`VENTUS_TMA_RUN_RTL_ONLY=1`。

另外，如果 `tma_descriptor_test` 在 `gvm-nocache` 下报 `UNDEFINED INSTRUCTION ...
0x00c5e542`，优先检查 `install/lib/libVentusGVM-nocache.so` 是否来自旧构建；重新构建并安装
no-cache GVM 后，该 case 应能正常执行这条 descriptor-addressed TMA 指令。

`run_dma_tma_rtl.sh` 会：

1. source `env.sh`。
2. 读取 `cases_dma_tma.csv`。
3. 只运行脚本中 `DIRECTED_CASES` 列出的 directed suite。
4. 如果 `.out` 不存在，进入对应目录执行 `make`。
5. 从 testcase 自己的目录运行可执行文件。
6. GVM 跑 `tma_matrix_test` 时默认设置 `VENTUS_TMA_RUN_RTL_ONLY=1`。
7. 每个 testcase 的日志写到传入的 log dir。

重要：这些 host 程序会按相对路径打开 sibling `.cl` 文件，所以必须从 testcase 目录运行。
如果从仓库根目录直接执行 `testcases/_get_case/tma_matrix_test/tma_matrix_test.out`，会导致
OpenCL build error `-44` 或 `<test>.cl: No such file or directory`。这是调用方式错误，不是 RTL 的 bug。

### 6.2 单个 testcase 手动运行

例如只跑 TMA matrix：

```bash
cd /home/liyb/ventus-env-copilot-test
source ./env.sh
cd testcases/_get_case/tma_matrix_test
make

VENTUS_BACKEND=spike ./tma_matrix_test.out
VENTUS_BACKEND=gvm VENTUS_TMA_RUN_RTL_ONLY=1 ./tma_matrix_test.out
VENTUS_BACKEND=gvm-nocache VENTUS_TMA_RUN_RTL_ONLY=1 ./tma_matrix_test.out
```

只跑名字匹配的 case：

```bash
VENTUS_BACKEND=gvm VENTUS_TMA_RUN_RTL_ONLY=1 ./tma_matrix_test.out FP32_2D_oob_zero_dim1
VENTUS_BACKEND=gvm ./bulk_dma_matrix_test.out bulk_32B_at_120
VENTUS_BACKEND=gvm ./multi_warp_dma_fence_test.out 2warp_cross_cacheline_each
```

其它目录也一样：

```bash
cd /home/liyb/ventus-env-copilot-test/testcases/_get_case/bulk_dma_matrix_test
source ../../../env.sh
make
VENTUS_BACKEND=spike ./bulk_dma_matrix_test.out
VENTUS_BACKEND=gvm ./bulk_dma_matrix_test.out
```

### 6.3 日志保存和摘要提取

GVM/RTL 日志会直接输出到 stdout，量可能非常大。建议所有运行都重定向：

```bash
log=/tmp/ventus-one-case-$(date +%Y%m%d-%H%M%S).log
(
  cd /home/liyb/ventus-env-copilot-test/testcases/_get_case/tma_matrix_test
  source ../../../env.sh
  VENTUS_BACKEND=gvm VENTUS_TMA_RUN_RTL_ONLY=1 ./tma_matrix_test.out FP32_2D_oob_zero_dim1
) >"$log" 2>&1
```

摘要：

```bash
grep -E "^\[[[:space:]]*[0-9]+/[0-9]+\]|PASS|FAIL|SKIP|FATAL|fatal|GVM ERROR|PC mismatch|summary|OK|FAILED|pass:|fail:|skip:|FAIL at byte|ctx exp|ctx got" "$log" | tail -n 120
```

Spike 指令日志：

```bash
cd /home/liyb/ventus-env-copilot-test/testcases/_get_case/bulk_dma_matrix_test
source ../../../env.sh
VENTUS_BACKEND=spike VENTUS_SPIKE_LOG=1 ./bulk_dma_matrix_test.out
```

`VENTUS_SPIKE_LOG=1` 会在当前 cwd 生成日志文件，通常很大，只用 `grep`/`tail`/`sed -n`
限制读取范围。

### 6.4 GVM waveform

GVM 构建时需要开启 `GVM_TRACE=1` 才能导出 FST。运行时启用 waveform：

```bash
cd /home/liyb/ventus-env-copilot-test/testcases/_get_case/tma_matrix_test
source ../../../env.sh
VENTUS_BACKEND=gvm \
VENTUS_TMA_RUN_RTL_ONLY=1 \
VENTUS_WAVEFORM=1 \
GVM_WAVEFORM_FILENAME=/tmp/tma_oob.gvm.fst \
./tma_matrix_test.out FP32_2D_oob_zero_dim1
```

限制时间窗口：

```bash
VENTUS_WAVEFORM=1 \
VENTUS_WAVEFORM_BEGIN=100000 \
VENTUS_WAVEFORM_END=200000 \
GVM_WAVEFORM_FILENAME=/tmp/window.gvm.fst \
VENTUS_BACKEND=gvm ./bulk_dma_matrix_test.out bulk_32B_at_120
```

## 7. 当前 DMA/TMA 覆盖矩阵

### 7.1 Directed suite

| 需求 | 覆盖 testcase |
| --- | --- |
| basic bulk DMA | `dma_test`、`bulk_dma_matrix_test`。 |
| fixed copysize DMA | `copysize_test`。 |
| basic tensor DMA | `tensor_dma_test`、`tma_matrix_test` baseline cases。 |
| descriptor-addressed TMA | `tma_descriptor_test` no-prefetch/prefetch。 |
| bulk 跨 128B cacheline | `bulk_dma_matrix_test`: `bulk_32B_at_120`、`bulk_8B_at_124`、`bulk_192B_aligned`。 |
| shared dst offset | `bulk_dma_matrix_test`: `bulk_64B_dst_offset`。 |
| 1D/2D/3D TMA | `tma_matrix_test`: `FP32_1D_16`、`FP32_2D_4x4_full`、`FP32_3D_2x2x2` 等。 |
| subbox | `FP32_2D_subbox_8x8_at_2_2`、`FP32_3D_subbox_6x6x6_at_1_1_1`。 |
| padded rows | `FP32_2D_padded_rows_4x4_stride64`。 |
| elementStride | `FP32_2D_estride2_cols`、`FP32_3D_estride2_dim1`、`FP32_2D_estride2_rows_cols`。 |
| swizzle | `FP32_2D_swizzle32_rows`、`FP32_2D_swizzle64_rows`、`FP32_2D_swizzle128_subbox_row1`。 |
| OOB dim0/dim1/dim2 | `FP32_2D_oob_zero_dim0`、`FP32_2D_oob_zero_dim1`、`FP32_3D_oob_zero_dim2`。 |
| OOB fill | `FP16_2D_oob_fill`、`FP16_2D_oob_subbox_dim0_dim1_fill`。 |
| OOB + elementStride | `FP32_2D_oob_estride_dim1`。 |
| 多 warp 多 DMA + fence | `multi_warp_dma_fence_test` 4 个 case。 |
| shared bank conflict + DMA routing | `dma_shared_routing_conflict_test` 2 个 case。 |
| 多 workgroup DMA | `multi_wg_dma_test`。 |

### 7.2 已知良好结果快照

case 数量会随测试扩展变化，最终以当前运行输出的 `[n/m]` 和 summary 为准。一个近期良好快照：

| 后端 | testcase | 期望摘要 |
| --- | --- | --- |
| Spike | `tma_matrix_test` | `15 pass / 0 fail / 8 skip`，skip 多为 RTL-directed OOB 或非 dim0 stride。 |
| Spike | `tma_descriptor_test` | `2 pass / 0 fail`。 |
| Spike | `bulk_dma_matrix_test` | `4 pass / 0 fail`。 |
| Spike | `multi_warp_dma_fence_test` | `4 pass / 0 fail`。 |
| Spike | `dma_shared_routing_conflict_test` | `2 pass / 0 fail`。 |
| GVM | `tma_matrix_test` + `VENTUS_TMA_RUN_RTL_ONLY=1` | `23 pass / 0 fail / 0 skip`。 |
| GVM | `tma_descriptor_test` | `2 pass / 0 fail`。 |
| GVM | `bulk_dma_matrix_test` | `4 pass / 0 fail`。 |
| GVM | `multi_warp_dma_fence_test` | `4 pass / 0 fail`。 |
| GVM | `dma_shared_routing_conflict_test` | `2 pass / 0 fail`。 |
| GVM no-cache | directed suite | `5 pass / 0 fail`；其中 `tma_descriptor_test` 为 `2 pass / 0 fail`。 |
| GVM no-cache | `tma_matrix_test` via suite | `15 pass / 0 fail / 8 skip`，因为 runner 在 no-cache 下会清除 `VENTUS_TMA_RUN_RTL_ONLY` 环境变量。 |

如果 case 数和快照不同，但当前 summary 是 `fail: 0`，通常是 suite 演进，不要直接判回归。

## 8. 常规项目回归测试

DMA/TMA directed suite 不等于仓库默认 regression。默认 regression 覆盖 Rodinia、POCL example
和部分 Ventus testcase：

```bash
cd /home/liyb/ventus-env-copilot-test
source ./env.sh

python3 ./regression-test.py
VENTUS_BACKEND=spike python3 ./regression-test.py
VENTUS_BACKEND=gvm python3 ./regression-test.py --checklist isa -j 1
```

RTL/GVM cache 变体矩阵：

```bash
source ./env.sh
python3 ./regression-test.py --matrix rtl-both
```

显式选择 backend/checklist：

```bash
source ./env.sh
python3 ./regression-test.py --matrix "gvm:rtl-with-cache,gvm-nocache:rtl-no-cache" -j 1
```

回归日志位于 `regression-test-logs/`。with-cache 类后端默认可能会重复运行，以捕获随机初始化导致的不稳定失败。

什么时候跑默认 regression：

- 只改 `DMA_core.scala`，且 directed suite 已经完整通过：通常可以先不跑全量 regression。
- 改了 scheduler、shared memory、top、driver、POCL、DecodeUnit：建议跑相关 checklist。
- 改了通用 cache/TLB/CTA 路径：必须跑更广的 regression。

OpenCL-CTS 规模很大，通常只用 Spike 验软件栈：

```bash
source ./env.sh
bash build-ventus.sh --build cts
cd OpenCL-CTS/build/test_conformance/compiler
./test_compiler
```

## 9. 如何新增或修改 DMA/TMA testcase

### 9.1 目录结构

新增单文件 host + kernel testcase 推荐：

```text
testcases/_get_case/<new_test>/
  Makefile
  <new_test>.c
  <new_test>.cl
```

Makefile 沿用现有模式：

```makefile
include ../common/make.config

CC = clang
CFLAGS = -g -O2 -std=c11 -DCL_TARGET_OPENCL_VERSION=120 -I. -I../common -Wno-deprecated-declarations
EXE = <new_test>.out
SRC = <new_test>.c

.PHONY: all clean
all: $(EXE)

$(EXE): $(SRC) <new_test>.cl ../common/ventus_opencl_test.h
	$(CC) $(CFLAGS) $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL -Wno-unused-result

clean:
	rm -f *.o *~ *.out *.linkinfo object*.cl object*.dump object*.riscv object*.riscv.log object*.vmem *.data *.metadata *.log
```

host 程序建议使用 `../common/ventus_opencl_test.h`：

- `ventus_get_default_device()` 建 context/device/queue。
- `ventus_build_program_from_source()` 从当前目录读 `.cl`。
- 失败时打印 OpenCL build log。

### 9.2 输出格式

保持短小：

```text
[1/4] case_name ...
      PASS

=== xxx summary ===
  pass: 4
  fail: 0
  skip: 0
OK
```

失败只打印首个 mismatch 和少量上下文。不要 dump 大数组，不要把 RTL 内部日志塞进 host 程序。

### 9.3 加入 DMA/TMA suite

1. 在 `testcases/_get_case/cases_dma_tma.csv` 添加：

```csv
<new_test>,<new_test>.out,,verdict
```

2. 如果要让 `run_dma_tma_rtl.sh` 运行它，也要把目录名加入脚本中的 `DIRECTED_CASES`。
3. 手动确认：

```bash
cd /home/liyb/ventus-env-copilot-test/testcases/_get_case/<new_test>
source ../../../env.sh
make
VENTUS_BACKEND=spike ./<new_test>.out
VENTUS_BACKEND=gvm ./<new_test>.out
```

### 9.4 新增 TMA matrix case

优先扩展 `testcases/_get_case/tma_matrix_test/tma_matrix_test.c` 的 `g_cases[]`，因为它已有：

- descriptor 构造：`build_descriptor()`。
- host expected replay：`compute_expected()`。
- OOB fill：整数填 0，浮点在 `oobfill=1` 时填 all-one bits。
- swizzle offset model。
- 单 case filter。
- `rtl_only` 跳过机制。

新增 case 时要检查：

- `box_total_bytes()` 不超过 `MAX_BOX_BYTES`，也不超过 `.cl` 中 `SHARED_BUF_BYTES`。
- `global_total_bytes()` 足够覆盖逻辑 tensor footprint 和物理 box walk footprint。
- `globalStrides` 未使用 slot 填 0。
- Spike 是否支持同样的语义。如果不支持，则设置 `rtl_only=1`，通过 `VENTUS_TMA_RUN_RTL_ONLY=1` 在 GVM 下运行。

### 9.5 kernel 内联 asm 经验

Bulk/copysize 这类 scalar operand 指令建议由线程 0 或 warp leader 发出：

```c
__asm__ volatile(
  "csrr %[src], 0x803\n\t"
  "lw   %[src], 4(%[src])\n\t"
  "lw   %[src], 0(%[src])\n\t"
  "mv   x10, %[src]\n\t"
  "mv   x11, %[dst]\n\t"
  "mv   x12, %[size]\n\t"
  ".word 0x00c515c2\n\t"
  ".word 0x00004042\n\t"
  : [src] "=&r"(src_addr)
  : [dst] "r"(dst_addr), [size] "r"(copy_bytes)
  : "memory", "x10", "x11", "x12"
);
```

TMA VGPR descriptor 指令必须用 vector load 把参数放进 VGPR，不能只 `mv x10` 到 SGPR：

```c
__asm__ volatile(
  "vid.v v10\n\t"
  "vsll.vi v10, v10, 2\n\t"
  "vadd.vx v10, v10, %[pb]\n\t"
  "vlw12.v v11, 256(v10)\n\t"
  "vlw12.v v12, 128(v10)\n\t"
  "vlw12.v v10, 0(v10)\n\t"
  ".word 0x00C535C2\n\t"
  ".word 0x00004042\n\t"
  :
  : [pb] "r"(param_buf_ptr)
  : "memory"
);
```

多 warp case 中，获取 warp id 建议读 CSR `0x805`，获取 workgroup id 可读 CSR `0x808`，
避免某些 vector-to-scalar 搬运导致的编译器/RTL 差异。

## 10. 失败分类与处理

### 10.1 先按 Spike/GVM 分类

| 现象 | 优先判断 |
| --- | --- |
| Spike 失败 | testcase、host expected model、descriptor 构造、kernel asm、Spike ISA 模型问题。先别改 RTL。 |
| Spike 通过，GVM host verdict 失败 | RTL/GVM/driver/runtime 实际行为差异。 |
| GVM fatal，host 未能给出 verdict | 可能是 DUT 死锁、assert、GVM reference 限制、非法指令状态。先缩小范围到单 case。 |
| GVM 报告 XREG/VREG mismatch，但 host `OK` | 多数是 checker 噪声，记录即可。 |
| `clBuildProgram` error `-44` 或找不到 `.cl` | 通常是从错误的工作目录启动了 testcase。进入 testcase 目录运行即可。 |

### 10.2 常见 GVM checker 噪声

目前已知的常见噪声：

- PC `0x80000014` 附近的 XREG mismatch。
- TMA OOB fill 场景下的 VREG mismatch。

处理原则：

1. 如果 host summary 是 `OK`、程序退出码是 0，先归为 checker noise。
2. 如果 host summary 是 `FAILED`，按 host 首个 mismatch 定位。
3. 如果 GVM fatal 直接导致非零退出，缩小到单 case，再看 fatal 前最后的 DMA/TMA 日志。

### 10.3 常见失败与修法

| 失败信号 | 常见原因 | 处理 |
| --- | --- | --- |
| `OpenCL error -44`，日志里有 `<test>.cl: No such file or directory` | 从仓库根目录运行 `.out`，host 找不到相对路径的 `.cl` 文件。 | 进入 `testcases/_get_case/<test>` 后运行。 |
| `.out` 不存在 | testcase 未构建。 | `source ../../../env.sh && make`。 |
| `libVentusGVM-withcache.so` 或 `libVentusGVM-nocache.so` 缺失 | 尚未构建 GVM。 | `bash build-ventus.sh --build "gvm"`。 |
| `libgvmref.so` 缺失 | Spike GVM reference 未构建或未安装。 | `bash build-ventus.sh --build "spike;gvm"`。 |
| `Unsupported VENTUS_BACKEND cache variant` | 后端后缀拼写不在 auto-select 支持范围。 | 用 `gvm`、`gvm-withcache`、`gvm-nocache`。 |
| `tma_descriptor_test` 在 `gvm-nocache` 下报 `UNDEFINED INSTRUCTION ... 0x00c5e542` | 常见原因是 `libVentusGVM-nocache.so` 仍是旧构建，生成的 RTL 未同步当前 DMA/TMA decode。 | 重新构建并安装 no-cache GVM：`make -C gpgpu/sim-verilator-nocache -f gvm.mk RELEASE=1 GVM_TRACE=1 GVM_REF_DIR="$PWD/install/lib" PREFIX="$PWD/install" install`，再重跑 no-cache suite。 |
| suite 首个 TMA OOB case mismatch，OOB 得到的是源 pattern | OOB validity 仅检查了 box row，未考虑完整的 tensor dim。 | 检查 `DMA_core.scala` 的 `tensor_dim0_start`、`tensor_high_dim_valid` 和 OOB fill mask。 |
| 2D/3D subbox 被填 0 | subbox 高维 offset 未正确合并到 tensor row start。 | 从 `BoxAddress - globalAddress` 中剥离 dim0 offset，保留 dim1+ 的固定 offset。 |
| multi-warp DMA fence GVM fatal | 自定义 DMA/fence asm 被运行时分支包裹，导致 reference PC 或控制流难以对应。 | 将 1-DMA 和 2-DMA 路径拆成两个 kernel，由 host 选择 kernel。 |
| shared routing conflict 的 DMA payload 变 0 | 多个静态 `__local` 对象在 LDS layout 下别名，或 sourceTag routing 丢失。 | kernel 使用单个 `scratch[]` 手动分区；RTL 侧检查 shared `sourceTag`。 |
| bulk 跨 cacheline 时只有前 128B 正确，后续错误 | L2 cacheline 拆分、tag entry、finish_cnt 或 shared offset 处理有误。 | 查看 `bulk_32B_at_120`、`bulk_192B_aligned`，对照 `DMA_core.scala` 中的 tag/mask 逻辑。 |
| GVM hang 或 timeout | DMA completion 未触发、scheduler fence wait 未释放、shared response ready 卡住。 | 单 case + waveform；优先排查 `fence_end_dma.ready`、`dma_complete`、`dma_fence_wait`。 |
| Spike 运行后终端不回显 | Spike 或子进程打乱了终端 echo 设置。 | 执行 `stty echo` 恢复。 |

### 10.4 如何缩小失败范围

1. 先用 Spike 跑同一单 case。
2. GVM 只跑单 case filter。
3. 如果 suite 脚本失败，看 `$log_dir/<test>.<backend>.log`。
4. 如果单 case 仍太长，加 waveform 时间窗口。
5. 对比附近 passing case：例如 OOB dim0 fail，就跑 full/subbox/padded rows。
6. 对 bulk 跨线问题，先跑 `bulk_8B_at_124`，再跑 `bulk_32B_at_120`，最后跑 `bulk_192B_aligned`。
7. 对 fence 问题，先跑 `2warp_1dma_each_fence`，再跑 `2warp_2dma_each_single_fence`。

## 11. 手动编译 OpenCL kernel 与反汇编

大多数时候用 POCL 自动编译即可。如果需要手动看 kernel 反汇编，可按项目规则使用
`install/` 下的工具。

示例：

```bash
cd /home/liyb/ventus-env-copilot-test
source ./env.sh

KERNEL_FUNC_NAME=tma_matrix_kernel
./install/bin/clang \
  -cl-std=CL2.0 \
  -target riscv32 \
  -mcpu=ventus-gpgpu \
  testcases/_get_case/tma_matrix_test/tma_matrix_test.cl \
  -o /tmp/tma_matrix_kernel.riscv \
  -nodefaultlibs \
  -Wl,${VENTUS_INSTALL_PREFIX}/lib/crt0.o \
  -Wl,${VENTUS_INSTALL_PREFIX}/lib/riscv32clc.o \
  -Wl,--gc-sections \
  -L${VENTUS_INSTALL_PREFIX}/lib \
  -lworkitem \
  -I${VENTUS_INSTALL_PREFIX}/include/clc \
  -O1 \
  -Wl,-T,${VENTUS_INSTALL_PREFIX}/lib/ldscripts/ventus/elf32lriscv.ld \
  -Wl,--init=${KERNEL_FUNC_NAME} \
  -w \
  -D__opencl_c_generic_address_space=1 \
  -D__opencl_c_named_address_space_builtins=1 \
  -D__OPENCL_VERSION__=200

./install/bin/llvm-objdump -d --mattr=+v,+zfinx /tmp/tma_matrix_kernel.riscv > /tmp/tma_matrix_kernel.dump
```

重点搜索：

```bash
grep -n "c2 35 c5 00\|42 40 00 00" /tmp/tma_matrix_kernel.dump
```

如果手动编译和 POCL 自动编译结果不同，以 testcase 运行时生成的 `object0.cl`、
`object0.riscv`、`object0.riscv.log` 为准。

## 12. 推荐日常工作流

### 12.1 小型 DMA/TMA RTL 修改

```bash
cd /home/liyb/ventus-env-copilot-test
source ./env.sh

# 1. Spike baseline
base=/tmp/ventus-dma-change-$(date +%Y%m%d-%H%M%S)
mkdir -p "$base"
(
  cd testcases/_get_case
  VENTUS_BACKEND=spike ./run_dma_tma_rtl.sh "$base/spike-before"
) >"$base/spike-before.log" 2>&1

# 2. 修改 gpgpu/ventus/src/**/*.scala

# 3. Chisel compile check
(cd gpgpu && ./mill -i -j 0 __.compile) >"$base/mill-compile.log" 2>&1

# 4. 重建 GVM
make -C gpgpu/sim-verilator -f gvm.mk -j"$(nproc)" RELEASE=1 GVM_TRACE=1 \
  GVM_REF_DIR="$PWD/install/lib" PREFIX="$PWD/install" install \
  >"$base/gvm-build-withcache.log" 2>&1

# 5. 跑 GVM directed suite
(
  cd testcases/_get_case
  VENTUS_BACKEND=gvm ./run_dma_tma_rtl.sh "$base/gvm-after"
) >"$base/gvm-after.log" 2>&1
```

如果修改涉及 no-cache，也执行 no-cache build/run。

### 12.2 涉及 shared/top/scheduler 的修改

这种修改的影响范围更大，建议：

1. Spike directed suite。
2. GVM with-cache directed suite。
3. GVM no-cache directed suite。
4. 相关 regression checklist。

```bash
source ./env.sh
bash build-ventus.sh --build "gvm"

base=/tmp/ventus-wide-dma-$(date +%Y%m%d-%H%M%S)
mkdir -p "$base"

(
  cd testcases/_get_case
  VENTUS_BACKEND=gvm ./run_dma_tma_rtl.sh "$base/gvm"
) >"$base/gvm.log" 2>&1

(
  cd testcases/_get_case
  VENTUS_BACKEND=gvm-nocache ./run_dma_tma_rtl.sh "$base/gvm-nocache"
) >"$base/gvm-nocache.log" 2>&1

VENTUS_BACKEND=gvm python3 regression-test.py --checklist rtl-with-cache -j 1 \
  >"$base/regression-gvm.log" 2>&1
```

### 12.3 只改 testcase/host expected model

如果只改 `testcases/_get_case/*`：

1. 不需要重建 GVM。
2. 先 `make clean && make`。
3. 先 Spike，后 GVM。
4. 如果 Spike fail，先修 testcase。

```bash
cd /home/liyb/ventus-env-copilot-test/testcases/_get_case/tma_matrix_test
source ../../../env.sh
make clean
make
VENTUS_BACKEND=spike ./tma_matrix_test.out
VENTUS_BACKEND=gvm VENTUS_TMA_RUN_RTL_ONLY=1 ./tma_matrix_test.out
```

## 13. 交付前检查清单

完成一次 DMA/TMA 修改后，至少记录：

- 修改了哪些文件，尤其是 Chisel、Spike、testcase 是否保持同步。
- Spike directed suite 的摘要和 log 目录。
- GVM with-cache directed suite 的摘要和 log 目录。
- 若相关，GVM no-cache directed suite 的摘要和 log 目录；若出现 descriptor 相关的 `UNDEFINED INSTRUCTION`，注明 no-cache GVM 库的时间戳以及是否已重建。
- 是否出现 GVM checker mismatch；如果是，需说明它导致了 host 失败还是仅为日志噪声。
- 如果 case 数与历史快照不同，说明是新增/删除 case 还是异常 skip。
- 是否需要跑更广的 regression，若没有跑，需说明原因。

建议在提交或汇报中使用这种格式：

```text
DMA/TMA verification:
  logs: /tmp/ventus-dma-change-YYYYmmdd-HHMMSS
  spike directed suite: pass, tma_matrix_test 15 pass / 0 fail / 8 skip
  gvm directed suite: pass, tma_matrix_test 23 pass / 0 fail / 0 skip
  gvm-nocache directed suite: pass, 5 pass / 0 fail
  known checker noise: PC 0x80000014 XREG mismatch only, host verdict OK
```

## 14. 快速命令速查

```bash
# 环境
cd /home/liyb/ventus-env-copilot-test
source ./env.sh

# 构建 GVM with-cache + no-cache
bash build-ventus.sh --build "gvm"

# 单独构建 with-cache GVM
make -C gpgpu/sim-verilator -f gvm.mk -j"$(nproc)" RELEASE=1 GVM_TRACE=1 \
  GVM_REF_DIR="$PWD/install/lib" PREFIX="$PWD/install" install

# 单独构建 no-cache GVM
make -C gpgpu/sim-verilator-nocache -f gvm.mk -j"$(nproc)" RELEASE=1 GVM_TRACE=1 \
  GVM_REF_DIR="$PWD/install/lib" PREFIX="$PWD/install" install

# DMA/TMA suite
base=/tmp/ventus-dma-tma-suite-$(date +%Y%m%d-%H%M%S)
mkdir -p "$base"
(cd testcases/_get_case && VENTUS_BACKEND=spike ./run_dma_tma_rtl.sh "$base/spike") >"$base/spike.log" 2>&1
(cd testcases/_get_case && VENTUS_BACKEND=gvm ./run_dma_tma_rtl.sh "$base/gvm") >"$base/gvm.log" 2>&1
# no-cache suite 期望 5/5 通过；如果 descriptor case 在 0x00c5e542 报 UNDEFINED INSTRUCTION，
# 先重建 install/lib/libVentusGVM-nocache.so。若需要完整的 tma_matrix_test RTL-directed
# 23-case 路径，需单独运行该 case 并显式设置 VENTUS_TMA_RUN_RTL_ONLY=1。
(cd testcases/_get_case && VENTUS_BACKEND=gvm-nocache ./run_dma_tma_rtl.sh "$base/gvm-nocache") >"$base/gvm-nocache.log" 2>&1

# 单 case
cd testcases/_get_case/tma_matrix_test
source ../../../env.sh
make
VENTUS_BACKEND=gvm VENTUS_TMA_RUN_RTL_ONLY=1 ./tma_matrix_test.out FP32_2D_oob_zero_dim1

# 摘要日志
grep -E "^\[[[:space:]]*[0-9]+/[0-9]+\]|PASS|FAIL|SKIP|FATAL|GVM ERROR|PC mismatch|summary|OK|FAILED|pass:|fail:|skip:|FAIL at byte|ctx exp|ctx got" "$log" | tail -n 120

# waveform
VENTUS_BACKEND=gvm VENTUS_WAVEFORM=1 GVM_WAVEFORM_FILENAME=/tmp/case.gvm.fst ./bulk_dma_matrix_test.out bulk_32B_at_120

# 默认 regression
cd /home/liyb/ventus-env-copilot-test
source ./env.sh
python3 regression-test.py --matrix rtl-both
```
