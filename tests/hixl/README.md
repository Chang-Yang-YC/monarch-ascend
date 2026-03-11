# HiXL Tests

Ascend NPU 单边通信 (HiXL) 集成测试集，覆盖从底层原生 API 到上层 Monarch E2E 的完整链路。

## 前置条件

- CANN 9.0+ (`source /path/to/cann/set_env.sh`)
- torch + torch_npu (版本匹配 CANN)
- conda 环境: `monarch_ascend`
- 至少 2 张 NPU 卡 (910B)

## 目录结构

```
tests/hixl/
├── e2e/          # 端到端测试 (Monarch + HIXL 完整链路)
├── unit/         # 单元测试 (聚焦单个组件)
├── native/       # 原生 C/C++ HIXL API 测试
├── debug/        # 调试/排查脚本
├── app/          # 应用级测试 (GRPO 等)
├── util/         # 工具代码 (C shim、trampoline)
├── build/        # 编译产物 (二进制、.so)
├── Makefile      # C/C++ 测试编译
└── README.md
```

## 各目录说明

### e2e/ — 端到端测试

| 文件 | 说明 |
|------|------|
| `test_hixl_bridge_minimal.py` | **核心验收用例**。两卡两 mesh，Producer(NPU0) 创建 RDMABuffer，Consumer(NPU1) 通过 write_from/read_into 跨卡读写 |
| `test_hixl_python_transfer.py` | Python ctypes 路径的完整 RDMABuffer 流程验证 |
| `test_hixl_rdma_e2e.py` | HIXL RDMA 端到端：Producer 建 buffer，Consumer 通过 HIXL 写入 |

### unit/ — 单元测试

| 文件 | 说明 |
|------|------|
| `test_npu_backend.py` | NPU 后端基础验证（环境、torch_npu、设备状态）|
| `test_hixl_rdma_minimal.py` | 最小 RDMABuffer 创建测试 |
| `test_hixl_actor_ctypes.py` | 从 Monarch actor 中直接 ctypes 调用 HIXL |
| `test_hixl_direct.py` | torch_npu vs aclrtMalloc 内存对比传输 |
| `test_hixl_rdma_manager_effect.py` | RdmaManagerActor 对 HIXL 的影响隔离 |
| `test_hixl_buffer_effect.py` | RDMABuffer 创建与 HIXL 引擎共存验证 |
| `test_hixl_acl_mem.py` | ACL 原生内存的 HIXL 传输验证 |
| `test_hixl_torch_vs_acl.py` | torch_npu vs ACL 内存类型对 HIXL 的影响 |
| `test_hixl_thread_isolation.py` | 非主线程调用 HIXL 的线程隔离测试 |
| `test_hixl_dynamic_ports.py` | 动态端口分配场景下的 HIXL 连接测试 |

### native/ — 原生 C/C++ 测试

| 文件 | 说明 |
|------|------|
| `test_hixl_connect.cpp` | HIXL Connect 各种配置组合 (HCCS/RoCE, 同卡/异卡, ip/ip:port) |
| `test_hixl_transfer.cpp` | HIXL 双进程 TransferSync (READ/WRITE) |
| `test_hixl_d2d.cpp` | HIXL CS API 的 D2D 传输测试 |
| `test_hixl_single.cpp` | 单进程 HIXL 测试 (类似官方 server_server_d2d) |
| `test_hixl_unidirectional.cpp` | 单向 vs 双向连接对 TransferSync 的影响 |
| `test_hixl_mem_type.cpp` | HUGE_ONLY vs NORMAL_ONLY 内存类型对比 |
| `test_hixl_pthread.c` | dlopen + pthread 隔离调用 HIXL |

### debug/ — 调试脚本

| 文件 | 说明 |
|------|------|
| `test_hixl_isolate_rdma.py` | 隔离 `_ensure_init_rdma_manager` vs `create_rdma_buffer_blocking` |
| `test_hixl_rdma_debug.py` | RDMA Manager 与 HIXL 交互调试 |

### app/ — 应用级测试

| 文件 | 说明 |
|------|------|
| `test_grpo_npu.py` | GRPO 训练 (Learner + Generator 双 mesh，跨卡 HIXL 权重同步) |
| `test_grpo_npu_simple.py` | 简化 GRPO (单 mesh，无 RDMA) |

### util/ — 工具代码

| 文件 | 说明 |
|------|------|
| `test_hixl_from_python.cpp` | 供 Python ctypes 调用的 HIXL 共享库 (run_server/run_client_transfer) |
| `hixl_trampoline.c` | 信号掩码重置 trampoline，解决 HIXL 修改信号处理的问题 |

## 快速运行

```bash
# 环境准备
conda activate monarch_ascend
source /root/hzz/cann-9.0.0-beta.1/set_env.sh
export HCCL_INTRA_ROCE_ENABLE=1

# 核心验收 — 两卡 HIXL 通信
python tests/hixl/e2e/test_hixl_bridge_minimal.py

# NPU 基础环境检查
python tests/hixl/unit/test_npu_backend.py

# 编译并运行 C++ 原生测试
cd tests/hixl && make && make run-connect
```

## 关键环境变量

| 变量 | 说明 |
|------|------|
| `HCCL_INTRA_ROCE_ENABLE=1` | **必须**。启用机内 RoCE 数据面 |
| `MONARCH_NPU_DEVICE` | HIXL bridge 使用的物理 NPU 设备号 |
| `MONARCH_PYTHON_HIXL_ENGINE_ID` | HIXL engine ID (ip:port) |
| `MONARCH_HIXL_LIB` | libtest_hixl.so 路径 (默认 build/libtest_hixl.so) |
| `MONARCH_HIXL_IP` | engine_id 使用的 IP (默认 127.0.0.1) |
| `MONARCH_HIXL_USE_REAL_IP` | 使用真实 IP 替代 loopback |
