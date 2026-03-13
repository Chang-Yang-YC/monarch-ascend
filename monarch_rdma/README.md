# Monarch RDMA

## Overview

Monarch RDMA is a Rust library that provides high-performance single-sided communication capabilities for the Monarch framework. It supports two backends:

- **GPU (ibverbs/rdmaxcel)**: RDMA over InfiniBand / RoCE for NVIDIA GPUs, using GPUDirect RDMA.
- **NPU (HiXL)**: Single-sided communication for Huawei Ascend NPUs, supporting HCCS (intra-supernode) and RoCE (inter-node) transports.

Both backends share a unified actor-based API (`RdmaManagerActor`) and common Python interface (`RDMABuffer` / `XDMABuffer`), enabling direct memory-to-memory transfers with minimal CPU overhead.

## Features

- **Dual-backend support**: GPU (ibverbs) and NPU (HiXL), selected at compile time via Cargo features
- **Actor-based API**: Clean, actor-based interface (`RdmaManagerActor`) for managing connections and resources
- **HCCS / RoCE transport selection** (NPU): Defaults to HCCS for intra-supernode; RoCE for cross-node, controllable via `MONARCH_HIXL_TRANSPORT`
- **Reference-counted memory registration** (NPU): Efficient per-buffer registration with automatic deregistration on release
- **Bidirectional connection coordination**: Both backends establish connections from both sides for robustness
- **Timeout propagation**: User-specified timeouts are passed through to the underlying transport library

## System Requirements

### GPU Backend

#### Hardware
- RDMA-capable NIC (e.g., Mellanox ConnectX series)
- NVIDIA GPU with CUDA support

#### Software
- **libibverbs**: RDMA verbs library
- **CUDA headers**: For GPU memory integration
- **GPUDirect RDMA**: For direct GPU memory access via RDMA

Install GPUDirect RDMA following:
https://docs.nvidia.com/networking/display/gpudirectrdmav18/installing+gpudirect+rdma

Verify installation:
```bash
lsmod | grep nvidia_peermem
```

Enable peer memory mapping in `/etc/modprobe.d/nvidia.conf`:
```
options nvidia NVreg_RegistryDwords="PeerMappingOverride=1;"
```

### NPU Backend

#### Hardware
- Huawei Ascend 910B NPU (2+ cards recommended)
- HCCS interconnect (intra-supernode) or RoCE NIC (cross-node)

#### Software
- **CANN 9.0+**: Huawei's compute architecture (`source /path/to/cann/set_env.sh`)
- **torch + torch_npu**: Version matching the installed CANN
- **libcann_hixl.so + libascendcl.so**: Provided by CANN SDK

## Building

### GPU (default)

```bash
cargo build -p monarch_extension
```

### NPU

```bash
PYO3_PYTHON=/path/to/python cargo build -p monarch_extension \
  --no-default-features \
  --features "ascend_engine,distributed_sql_telemetry,extension-module"
```

The `hixl-sys` crate's `build.rs` automatically compiles the C shim (`hixl_shim.cpp`) using the `cc` crate and links against CANN libraries.

## Architecture

```
RdmaManagerActor (shared)
├── GPU: IbvManagerActor          NPU: HixlManagerActor
│        ├─ ibv_open_device             ├─ Hixl::Initialize(engine_id)
│        ├─ QP create/connect           ├─ Hixl::Connect(peer_engine_id)
│        ├─ ibv_reg_mr / dereg          ├─ Hixl::RegisterMem / DeregisterMem
│        └─ QP put/get (WRITE/READ)     └─ Hixl::TransferSync (WRITE/READ)
├── rdma_components.rs  (RdmaRemoteBuffer — unified read/write API)
└── rdma_manager_actor.rs (shared message routing, transport_level reporting)
```

### Key Files

| Component | GPU | NPU |
|-----------|-----|-----|
| Manager Actor | `backend/ibverbs/manager_actor.rs` | `backend/hixl/manager_actor.rs` |
| FFI Bindings | `rdmaxcel-sys/src/lib.rs` | `hixl-sys/src/lib.rs` |
| C Shim | rdmaxcel C library | `hixl-sys/cpp/hixl_shim.cpp` |
| Build Script | `rdmaxcel-sys/build.rs` | `hixl-sys/build.rs` |
| Python Buffer | `python/monarch/_src/rdma/rdma.py` | `python/monarch/_src/rdma/xdma.py` |

## Environment Variables

### NPU-specific

| Variable | Description |
|----------|-------------|
| `MONARCH_HIXL_TRANSPORT` | Transport selection: `hccs` (default), `roce`, or `auto` |
| `MONARCH_NPU_DEVICE` | NPU device index for HiXL engine |
| `MONARCH_HIXL_USE_LOOPBACK` | Force engine ID to use 127.0.0.1 instead of real IP |
| `HCCL_NPU_SOCKET_PORT_RANGE` | Set to `auto` automatically in HCCS mode |

### GPU-specific

| Variable | Description |
|----------|-------------|
| `CUDA_VISIBLE_DEVICES` | Control visible GPUs |
| `MONARCH_DEBUG_RDMA` | Print device mapping info |

## NPU Memory Alignment

HCCS transport requires **2MB-aligned** device memory addresses. Use the provided helper:

```python
from monarch._src.rdma.xdma import alloc_aligned_tensor

tensor = alloc_aligned_tensor((size,), dtype=torch.float32, device="npu:0")
```

Standard `torch.zeros(..., device="npu:0")` allocations may not be 2MB-aligned. Unaligned memory will fall back to RoCE or fail with error 503900 during connect.

## License

This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
