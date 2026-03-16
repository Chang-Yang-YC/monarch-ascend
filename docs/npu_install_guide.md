# Monarch NPU (Ascend) 安装指南

基于 Huawei Cloud EulerOS 2.0 + Ascend 910B 实测整理。

## 当前验证通过的环境

| 组件 | 版本 | 备注 |
|------|------|------|
| OS | Huawei Cloud EulerOS 2.0 (aarch64) | 包管理器: yum/dnf |
| NPU | Ascend 910B1 × 8 | 每卡 64GB HBM |
| CANN | 9.0.0-beta.1 | 自定义路径 `/root/hzz/cann-9.0.0-beta.1/` |
| Python | 3.11.0 | conda 环境 `monarch_ascend` |
| PyTorch | 2.7.1+cpu | torch_npu 提供 NPU 后端 |
| torch_npu | 2.7.1 | 与 CANN 9.0 配套 |
| Rust | 1.93.0-nightly | 需要 nightly（`-Zthreads`、`tracing_unstable`）|
| clang | 12.0.1 | bindgen 需要 |
| protobuf | 3.14.0 | protoc 编译器 |

---

## 安装步骤

### Step 1: CANN 安装与环境变量

CANN 按华为官方文档安装。安装完成后验证：

```bash
npu-smi info
# 应能看到所有 NPU 卡信息
```

如果 CANN 安装在非标准路径，需要 source 环境脚本：

```bash
source /path/to/cann/set_env.sh

# 例如：
source /root/hzz/cann-9.0.0-beta.1/set_env.sh
```

验证 CANN 库文件存在：

```bash
# 以下文件必须存在
ls $ASCEND_HOME/include/hixl/hixl.h       # HiXL 头文件
ls $ASCEND_HOME/lib64/libcann_hixl.so      # HiXL 动态库
ls $ASCEND_HOME/lib64/libascendcl.so       # ACL 运行时
ls $ASCEND_HOME/lib64/libhccl.so           # HCCL 集合通信
```

### Step 2: 创建 conda 环境

```bash
conda create -n monarch_ascend python=3.11 -y
conda activate monarch_ascend
```

> **⚠️ 坑 1: Python 版本必须是 3.11**
>
> PyO3 编译的 `_rust_bindings.so` 会绑定到特定 Python 版本。
> 系统 Python 可能是 3.13，如果 Rust 编译时链接了 3.13，
> 在 3.11 的 conda 环境里就会报 `undefined symbol: PyErr_SetRaisedException`
> （该符号从 Python 3.12 才有）。
>
> **确保编译和运行使用同一个 Python**。

### Step 3: 安装 PyTorch + torch_npu

```bash
pip install torch==2.7.1
pip install torch_npu==2.7.1
```

验证：

```bash
python -c "
import torch
import torch_npu
print('torch:', torch.__version__)
print('torch_npu:', torch_npu.__version__)
print('NPU available:', torch.npu.is_available())
print('NPU count:', torch.npu.device_count())
"
```

> **⚠️ 坑 2: torch 和 torch_npu 版本必须严格匹配**
>
> torch 2.7.1 只能配 torch_npu 2.7.1。版本不匹配会导致
> `RuntimeError: torch_npu is not compatible with the installed torch version`。
> torch_npu 还要与 CANN 版本配套，具体对应关系见华为文档。

### Step 4: 安装系统依赖

EulerOS 使用 yum/dnf：

```bash
# clang + llvm（hccl-sys 用 bindgen 生成 Rust 绑定，需要 libclang）
yum install clang clang-devel llvm-libs

# protobuf 编译器（Monarch 内部序列化）
yum install protobuf-compiler
```

验证：

```bash
clang --version    # 需要能找到
protoc --version   # 需要能找到
```

> **⚠️ 坑 3: 没有 apt**
>
> EulerOS / openEuler 基于 RPM，用 yum/dnf，不是 apt。

### Step 5: 安装 Rust nightly

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
rustup default nightly
```

验证：

```bash
rustc --version   # 应显示 nightly
cargo --version
```

### Step 6: 安装 Python 依赖

```bash
cd /path/to/monarch
pip install -r requirements.txt
pip install setuptools setuptools-rust
```

### Step 7: 设置环境变量并编译

```bash
conda activate monarch_ascend

# 1. source CANN 环境（每个终端都要）
source /root/hzz/cann-9.0.0-beta.1/set_env.sh

# 2. 设置 ASCEND_HOME（指向包含 include/ 和 lib64/ 的架构子目录）
export ASCEND_HOME=/root/hzz/cann-9.0.0-beta.1/aarch64-linux
```

#### 方式一：pip install（推荐）

```bash
USE_ASCEND_ENGINE=1 pip install -e .
```

#### 方式二：仅编译 Rust 后端

```bash
PYO3_PYTHON=$(which python) cargo build -p monarch_extension \
  --no-default-features \
  --features "ascend_engine,distributed_sql_telemetry,extension-module"
```

> **⚠️ 坑 4: 必须指定 `PYO3_PYTHON`**
>
> 如果不指定，PyO3 可能找到系统 Python（3.13）而非 conda 的 3.11，
> 编译出的 .so 在 conda 环境里无法加载。
>
> ```bash
> # 正确
> PYO3_PYTHON=/root/miniconda3/envs/monarch_ascend/bin/python cargo build ...
>
> # 或者（conda 环境已激活时）
> PYO3_PYTHON=$(which python) cargo build ...
> ```

> **⚠️ 坑 5: 不要用默认 features 编译**
>
> ```bash
> # ❌ 错误 — 会拉 tensor_engine → rdma-core（需要外网 git clone）
> cargo build -p monarch_extension
>
> # ❌ 错误 — 会触发 rdma-core 编译
> cargo build -p monarch_rdma --features hixl
>
> # ✅ 正确 — 只编译 Ascend 后端
> cargo build -p monarch_extension \
>   --no-default-features \
>   --features "ascend_engine,distributed_sql_telemetry,extension-module"
> ```
>
> 默认 features 包含 `tensor_engine`，它依赖 `monarch_cpp_static_libs`，
> 后者会从 GitHub 克隆 `rdma-core` 源码。在无外网环境下会报：
> `error: RPC failed; curl 16 Error in the HTTP2 framing layer`

> **⚠️ 坑 6: ASCEND_HOME 必须指向架构子目录**
>
> ```bash
> # ❌ 错误 — 这是 CANN 根目录，下面没有 include/
> export ASCEND_HOME=/root/hzz/cann-9.0.0-beta.1
>
> # ✅ 正确 — 架构子目录，下面有 include/ 和 lib64/
> export ASCEND_HOME=/root/hzz/cann-9.0.0-beta.1/aarch64-linux
> ```
>
> build.rs 会检查 `$ASCEND_HOME/include` 是否存在，路径不对会报
> `Ascend CANN installation not found`。

### Step 8: 验证安装

```bash
# 1. 基础导入
python -c "import monarch; print('monarch OK')"

# 2. Rust 绑定加载
python -c "import monarch._rust_bindings; print('rust bindings OK')"

# 3. GRPO 端到端测试（核心验收，需要至少 2 张卡）
python tests/hixl/app/test_grpo_npu.py

# 4. Ping-Pong 跨 mesh 通信
python tests/hixl/app/test_ping_pong_npu.py
```

---

## 运行时注意事项

### 每次开新终端都要做

```bash
conda activate monarch_ascend
source /root/hzz/cann-9.0.0-beta.1/set_env.sh
```

不 source CANN 环境会导致 `libascendcl.so` / `libcann_hixl.so` 找不到。

### NPU 设备隔离

HiXL 不支持同卡两个 engine 通信（底层 HCCL 通信域约束），
所以每个 mesh 必须绑定到不同物理卡：

```python
def npu_device(dev_id: int):
    def _bootstrap():
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(dev_id)
        import torch, torch_npu  # noqa
        torch.npu.set_device(0)
    return _bootstrap

mesh_a = this_host().spawn_procs(per_host={"npus": 1}, bootstrap=npu_device(0))
mesh_b = this_host().spawn_procs(per_host={"npus": 1}, bootstrap=npu_device(1))
```

### HCCS 2MB 内存对齐

默认传输模式 HCCS 要求所有 RDMA buffer 地址 2MB 对齐：

```python
from monarch._src.rdma.xdma import alloc_aligned_tensor
buf = alloc_aligned_tensor((size,), dtype=torch.float32, device="npu:0")
```

不对齐会报 `rtsIpcMemGetExportKey execution failed (error 503900)`。

### 关键环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `ASCEND_HOME` | CANN 架构子目录（编译时） | 自动检测 |
| `ASCEND_RT_VISIBLE_DEVICES` | 进程可见的物理 NPU（运行时） | 所有卡 |
| `MONARCH_HIXL_TRANSPORT` | HiXL 传输模式 | `hccs` |
| `MONARCH_NPU_DEVICE` | HiXL 使用的逻辑设备号 | `0` |
| `PYO3_PYTHON` | Rust 编译链接的 Python（编译时） | 自动检测 |

---

## HiXL 单边通信带宽实测

测试环境：Ascend 910B1，HCCS 模式（默认），NPU 5 ↔ NPU 6，TransferSync 同步传输。

测试脚本：`tests/hixl/bench_hixl_bandwidth.py`

```bash
python tests/hixl/bench_hixl_bandwidth.py 5 6
```

| 操作 | 数据量 | 带宽 (GB/s) | 延迟 (ms) |
|------|--------|------------|-----------|
| READ | 1 MB | 7.65 | 0.128 |
| WRITE | 1 MB | 7.88 | 0.124 |
| READ | 2 MB | 10.84 | 0.180 |
| WRITE | 2 MB | 11.52 | 0.170 |
| READ | 4 MB | 14.59 | 0.268 |
| WRITE | 4 MB | 14.49 | 0.270 |
| READ | 8 MB | 16.54 | 0.472 |
| WRITE | 8 MB | 16.69 | 0.468 |
| READ | 16 MB | 17.97 | 0.870 |
| WRITE | 16 MB | 17.99 | 0.869 |
| READ | 32 MB | 18.65 | 1.675 |
| WRITE | 32 MB | 18.53 | 1.687 |
| READ | 64 MB | 18.89 | 3.308 |
| WRITE | 64 MB | 18.46 | 3.385 |
| READ | 128 MB | 19.28 | 6.483 |
| WRITE | 128 MB | 19.32 | 6.469 |
| READ | 256 MB | 19.26 | 12.982 |
| WRITE | 256 MB | 19.42 | 12.870 |
| **READ** | **512 MB** | **19.45** | **25.703** |
| **WRITE** | **512 MB** | **19.48** | **25.668** |

**分析：**

- 峰值带宽约 **19.5 GB/s**，128 MB 以上趋于饱和
- READ 和 WRITE 性能对称，差异 < 1%
- 小数据延迟优秀：1 MB 仅需 ~0.13 ms
- 910B HCCS 单边理论带宽 ~28 GB/s，实测利用率约 **70%**（TransferSync 同步开销，异步流水线可更高）

---

## 踩坑速查表

| # | 现象 | 原因 | 解决 |
|---|------|------|------|
| 1 | `undefined symbol: PyErr_SetRaisedException` | Rust 链接了 Python 3.12+，运行在 3.11 | 设置 `PYO3_PYTHON=$(which python)` 重新编译 |
| 2 | `curl 16 Error in the HTTP2 framing layer` | 默认 features 从 GitHub 拉 rdma-core | 用 `--no-default-features --features ascend_engine,...` |
| 3 | `Ascend CANN installation not found` | ASCEND_HOME 路径不对 | 指向架构子目录（含 include/ 和 lib64/） |
| 4 | `rtsIpcMemGetExportKey failed (503900)` | HCCS 要求 2MB 对齐 | 使用 `alloc_aligned_tensor()` 或切 RoCE |
| 5 | `not support connect with self device` | 两个 HiXL engine 在同一张卡 | 每个 mesh 绑定不同物理卡 |
| 6 | `libascendcl.so: cannot open` | 没 source CANN 环境 | `source /path/to/cann/set_env.sh` |
| 7 | `_GLIBCXX_USE_CXX11_ABI` 链接错误 | C++ ABI 不匹配 | pip install 会自动处理；手动编译需加 `CXXFLAGS` |
| 8 | pip install 编了 GPU 版本 | 环境有 CUDA，优先选了 tensor_engine | 显式 `USE_ASCEND_ENGINE=1 USE_TENSOR_ENGINE=0 pip install -e .` |
