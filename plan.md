# HIXL Backend Adaptation Plan

## 当前状态 (2026-03-11)

### 已解决的问题

| 问题 | 状态 | 说明 |
|------|------|------|
| engine_id 格式 | ✅ 已修复 | 从 `ip:0` 改为 `ip:port`，每个进程监听独立端口 |
| 连接模型 | ✅ 已修复 | 采用 server-server 模式，双向监听 |
| HIXL 实例重复初始化 | ✅ 已修复 | 统一使用全局 `PROCESS_HIXL` |
| 设备上下文同步 | ✅ 已修复 | `bridge.cpp` 从 `MONARCH_NPU_DEVICE` 环境变量读取设备 ID |

### 当前问题

**TransferSync 失败 (HIXL error 503900: HIXL_FAILED)**

```
HIXL: connected to remote engine: 192.168.0.117:16567
Exception: HIXL transfer to 192.168.0.117:16567 failed: HIXL error 503900: HIXL_FAILED
```

---

## rdmaxcel-sys vs hixl-sys 功能对比

### 架构差异

```
rdmaxcel-sys (CUDA/MLX5):              hixl-sys (Ascend NPU):
┌─────────────────────┐               ┌─────────────────────┐
│  暴露底层 RDMA 原语  │               │  高层抽象，完全封装  │
│  GPU 可直接操作      │               │  仅 Host 端操作      │
│  无锁原子计数器      │               │  无底层访问能力      │
└─────────────────────┘               └─────────────────────┘
```

### 功能对比表

| 功能 | rdmaxcel-sys | hixl-sys | 影响 |
|-----|-------------|----------|------|
| QP 创建 | `rdmaxcel_qp_create()` 细粒度控制 | `HixlCreate()` 完全封装 | 无法自定义传输参数 |
| 内存注册 | `register_segments()` 支持扫描器 | `HixlRegisterMem()` 仅 API | 无法动态发现内存 |
| WQE 操作 | `send_wqe()` **GPU 可调用** | 无 | 无法从 NPU kernel 操作 |
| Doorbell | `db_ring()` 手动控制 | 无 | 无法手动触发传输 |
| 完成轮询 | `poll_cq_with_cache()` 带缓存 | 无，同步等待 | 效率较低 |
| 原子操作 | fetch_add/load/store 完整支持 | 无 | 无法无锁并发 |
| 设备匹配 | `get_cuda_pci_address_from_ptr()` | 无 | 无法匹配物理设备 |
| 错误详情 | 21 种详细错误码 + 字符串 | 仅 `HIXL_FAILED` | **调试困难** |

---

## hixl-sys 功能增强需求

### 功能模块对比图

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                            功能模块对比                                               │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  rdmaxcel-sys (CUDA/MLX5)              hixl-sys (Ascend NPU)                         │
│  ═══════════════════════════           ═════════════════════════                     │
│                                                                                      │
│  [QP/Engine 管理]                      [Engine 管理]                                │
│  ├── create_qp() ✅                    ├── HixlCreate() ✅                          │
│  ├── rdmaxcel_qp_create() ✅           ├── HixlDestroy() ✅                         │
│  ├── rdmaxcel_qp_destroy() ✅          ├── HixlInitialize() ✅                      │
│  ├── create_mlx5dv_qp() ✅             └── HixlFinalize() ✅                        │
│  └── create_mlx5dv_cq() ✅            ❌ 缺失: QP 级别的细粒度控制                    │
│                                                                                      │
│  [连接管理]                            [连接管理]                                    │
│  ├── ibv_modify_qp() ✅                ├── HixlConnect() ✅                         │
│  ├── rdmaxcel_efa_connect() ✅         └── HixlDisconnect() ✅                      │
│  └── 连接状态查询 ✅                   ❌ 缺失: 连接状态查询                          │
│                                       ❌ 缺失: 多路径连接管理                         │
│                                                                                      │
│  [内存管理]                            [内存管理]                                    │
│  ├── register_cuda_memory() ✅         ├── HixlRegisterMem() ✅                     │
│  ├── register_segments() ✅            └── HixlDeregisterMem() ✅                   │
│  ├── deregister_segments() ✅          ❌ 缺失: 批量内存注册                          │
│  ├── rdma_get_active_segment_count() ✅ ❌ 缺失: 内存段枚举                          │
│  ├── rdma_get_all_segment_info() ✅    ❌ 缺失: 内存状态查询                          │
│  └── segment_scanner 回调 ✅           ❌ 缺失: 内存扫描器回调                        │
│                                                                                      │
│  [设备查询]                            [设备查询]                                    │
│  ├── get_cuda_pci_address_from_ptr() ✅ ❌ 缺失: NPU 地址查询                        │
│  ├── rdmaxcel_cuPointerGetAttribute() ✅ ❌ 缺失: 指针属性查询                        │
│  └── rdmaxcel_print_device_info() ✅   ❌ 缺失: 设备信息打印                         │
│                                                                                      │
│  [传输操作]                            [传输操作]                                    │
│  ├── send_wqe() [GPU callable] ✅       ├── HixlTransferSync() ✅                    │
│  ├── recv_wqe() [GPU callable] ✅       └── HixlTransferAsync() ✅                   │
│  ├── db_ring() [GPU callable] ✅        ❌ 缺失: NPU kernel 可调用接口                │
│  ├── cqe_poll() [GPU callable] ✅       ❌ 缺失: 传输进度查询                          │
│  ├── launch_* kernel 函数 ✅           ❌ 缺失: 批量传输                              │
│  └── 批量 WQE 提交 ✅                                                                 │
│                                                                                      │
│  [原子操作]                            [原子操作]                                    │
│  ├── rdmaxcel_qp_fetch_add_*() ✅       ❌ 完全缺失 (HIXL 封装了)                     │
│  ├── rdmaxcel_qp_load_*() ✅            ── 可能不需要，但缺少调试接口                  │
│  └── rdmaxcel_qp_store_*() ✅                                                         │
│                                                                                      │
│  [完成管理]                            [完成管理]                                    │
│  ├── completion_cache_t ✅              ❌ 缺失: 完成缓存                             │
│  ├── completion_cache_add/find() ✅     ❌ 缺失: 完成事件查询                          │
│  └── poll_cq_with_cache() ✅            ── TransferSync 是同步的，可能不需要          │
│                                                                                      │
│  [错误处理]                            [错误处理]                                    │
│  ├── 21 种详细错误码 ✅                 ├── 10 种错误码 ✅ (不够详细)                  │
│  └── rdmaxcel_error_string() ✅         └── HixlGetStatusString() ✅                  │
│                                         ❌ 缺失: 详细错误信息                          │
│                                         ❌ 缺失: 错误上下文                            │
│                                                                                      │
│  [传输类型]                            [传输类型]                                    │
│  ├── rdmaxcel_is_efa_dev() ✅           ❌ 缺失: HCCS/RoCE 检测                       │
│  ├── rdma_qp_type_t 枚举 ✅             ❌ 缺失: 传输类型枚举                          │
│  └── EFA 专用函数 ✅                    ❌ 缺失: 传输路径信息                          │
│                                                                                      │
│  [通知机制]                            [通知机制]                                    │
│  └── (无)                               ├── HixlSendNotify() ✅                       │
│                                         └── HixlGetNotifies() ✅                      │
│                                         ✅ HIXL 已有                                   │
│                                                                                      │
│  [设备上下文]                          [设备上下文]                                  │
│  └── (隐式 CUDA context)                ├── HixlSetAclDevice() ✅                     │
│                                         └── HixlGetAclDevice() ✅                     │
│                                         ✅ 已实现                                       │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 需要补充的功能接口

#### 1. 错误诊断增强 (优先级: P0 - 最高)

```c
// ========== 新增错误接口 ==========

/// 获取最后一次操作的详细错误信息
HixlStatus HixlGetLastError(
    HixlHandle handle,
    char* error_msg,
    size_t msg_size
);

/// 获取错误的调用栈/上下文
HixlStatus HixlGetErrorContext(
    HixlHandle handle,
    HixlErrorContext* out_context
);

typedef struct {
    HixlStatus status;           // 错误码
    char operation[64];          // 失败的操作名
    char file[256];              // 源文件
    int line;                    // 行号
    char message[512];           // 详细消息
    uint64_t timestamp_ms;       // 时间戳
} HixlErrorContext;

/// 扩展错误码枚举
typedef enum {
    // 现有错误码
    HIXL_SUCCESS           = 0,
    HIXL_PARAM_INVALID     = 103900,
    HIXL_TIMEOUT           = 103901,
    HIXL_NOT_CONNECTED     = 103902,
    HIXL_ALREADY_CONNECTED = 103903,
    HIXL_NOTIFY_FAILED     = 103904,
    HIXL_UNSUPPORTED       = 103905,
    HIXL_FAILED            = 503900,
    HIXL_RESOURCE_EXHAUSTED = 203900,
    HIXL_NOT_INITIALIZED   = 999999,
    
    // 新增细粒度错误码
    HIXL_MEM_NOT_REGISTERED    = 503901,  // 内存未注册
    HIXL_MEM_REG_FAILED        = 503902,  // 内存注册失败
    HIXL_MEM_ACCESS_DENIED     = 503903,  // 内存访问被拒绝
    HIXL_MEM_ADDR_INVALID      = 503904,  // 无效内存地址
    HIXL_MEM_SIZE_MISMATCH     = 503905,  // 内存大小不匹配
    HIXL_DEVICE_MISMATCH       = 503906,  // 设备不匹配
    HIXL_DEVICE_NOT_READY      = 503907,  // 设备未就绪
    HIXL_ACL_ERROR             = 503908,  // ACL 运行时错误
    HIXL_TRANSPORT_ERROR       = 503909,  // 传输层错误
    HIXL_HCCS_ERROR            = 503910,  // HCCS 传输错误
    HIXL_ROCE_ERROR            = 503911,  // RoCE 传输错误
    HIXL_REMOTE_ERROR          = 503912,  // 远端返回错误
    HIXL_CONNECTION_RESET      = 503913,  // 连接重置
    HIXL_BUFFER_OVERFLOW       = 503914,  // 缓冲区溢出
} HixlStatusEx;
```

#### 2. 内存管理增强 (优先级: P0)

```c
// ========== 内存查询接口 ==========

/// 内存段信息
typedef struct {
    uintptr_t addr;              // 已注册地址
    size_t size;                 // 注册大小
    HixlMemType mem_type;        // 内存类型
    uint64_t reg_time_ms;        // 注册时间
    uint32_t access_count;       // 访问次数
    HixlMemHandle handle;        // 内存句柄
} HixlMemSegmentInfo;

/// 获取已注册内存段数量
HixlStatus HixlGetRegisteredMemCount(
    HixlHandle handle,
    size_t* out_count
);

/// 枚举所有已注册内存段
HixlStatus HixlGetRegisteredMemInfo(
    HixlHandle handle,
    HixlMemSegmentInfo* info_array,
    size_t max_count,
    size_t* actual_count
);

/// 查询特定地址的内存注册状态
HixlStatus HixlQueryMemStatus(
    HixlHandle handle,
    uintptr_t addr,
    size_t size,
    HixlMemStatus* out_status
);

typedef struct {
    int is_registered;           // 是否已注册
    uintptr_t registered_base;   // 注册基地址
    size_t registered_size;      // 注册大小
    HixlMemHandle handle;        // 句柄
} HixlMemStatus;

// ========== 批量内存注册 ==========

/// 批量注册内存段
HixlStatus HixlRegisterMemBatch(
    HixlHandle handle,
    const HixlMemDesc* mem_descs,
    size_t count,
    HixlMemHandle* out_handles
);

// ========== 内存扫描器回调 ==========

/// 内存段扫描器回调类型
typedef size_t (*HixlSegmentScannerFn)(
    uintptr_t* addrs_out,
    size_t* sizes_out,
    size_t max_count
);

/// 注册内存段扫描器
void HixlRegisterSegmentScanner(HixlSegmentScannerFn scanner);
```

#### 3. 设备查询增强 (优先级: P1)

```c
// ========== 设备信息接口 ==========

/// NPU 设备信息
typedef struct {
    int device_id;               // 设备 ID
    char device_name[64];        // 设备名称
    char pci_bus_id[16];         // PCI 总线 ID
    char physical_id[32];        // 物理标识
    size_t total_memory;         // 总内存
    size_t free_memory;          // 空闲内存
    int numa_node;               // NUMA 节点
} HixlNpuDeviceInfo;

/// 获取 NPU 设备信息
HixlStatus HixlGetNpuDeviceInfo(
    int device_id,
    HixlNpuDeviceInfo* out_info
);

/// 从指针获取 NPU 设备信息
HixlStatus HixlGetNpuDeviceFromPtr(
    uintptr_t npu_ptr,
    int* out_device_id,
    HixlNpuDeviceInfo* out_info
);

/// 打印设备详细信息 (调试用)
void HixlPrintDeviceInfo(HixlHandle handle);

/// 指针属性查询
typedef enum {
    HIXL_PTR_ATTR_DEVICE_ID,     // 所属设备 ID
    HIXL_PTR_ATTR_MEMORY_TYPE,   // 内存类型
    HIXL_PTR_ATTR_IS_REGISTERED, // 是否已注册
    HIXL_PTR_ATTR_PHYSICAL_ADDR, // 物理地址
} HixlPointerAttribute;

HixlStatus HixlPointerGetAttribute(
    uintptr_t ptr,
    HixlPointerAttribute attr,
    void* out_value
);
```

#### 4. 连接管理增强 (优先级: P1)

```c
// ========== 连接状态接口 ==========

/// 连接状态
typedef enum {
    HIXL_CONN_DISCONNECTED = 0,
    HIXL_CONN_CONNECTING   = 1,
    HIXL_CONN_CONNECTED    = 2,
    HIXL_CONN_ERROR        = 3,
    HIXL_CONN_TIMEOUT      = 4,
} HixlConnectionState;

/// 连接信息
typedef struct {
    char remote_engine_id[256];  // 远端 engine ID
    HixlConnectionState state;   // 连接状态
    uint64_t connect_time_ms;    // 连接时间
    uint64_t bytes_sent;         // 已发送字节
    uint64_t bytes_recv;         // 已接收字节
    uint32_t transfer_count;     // 传输次数
    uint32_t error_count;        // 错误次数
    HixlTransportType transport; // 传输类型
} HixlConnectionInfo;

/// 获取连接状态
HixlStatus HixlGetConnectionState(
    HixlHandle handle,
    const char* remote_engine,
    HixlConnectionState* out_state
);

/// 获取连接详细信息
HixlStatus HixlGetConnectionInfo(
    HixlHandle handle,
    const char* remote_engine,
    HixlConnectionInfo* out_info
);

/// 获取所有活跃连接
HixlStatus HixlGetActiveConnections(
    HixlHandle handle,
    HixlConnectionInfo* info_array,
    size_t max_count,
    size_t* actual_count
);

/// 断开所有连接
HixlStatus HixlDisconnectAll(HixlHandle handle);
```

#### 5. 传输管理增强 (优先级: P2)

```c
// ========== 传输类型检测 ==========

typedef enum {
    HIXL_TRANSPORT_HCCS = 0,     // 芯片间直连
    HIXL_TRANSPORT_ROCE = 1,     // RDMA over Converged Ethernet
    HIXL_TRANSPORT_UNKNOWN = 2,  // 未知
} HixlTransportType;

/// 检测到远端的传输类型
HixlStatus HixlGetTransportType(
    HixlHandle handle,
    const char* remote_engine,
    HixlTransportType* out_type
);

/// 获取传输路径详细信息
typedef struct {
    HixlTransportType type;
    char local_nic[64];          // 本地网卡
    char remote_nic[64];         // 远端网卡
    uint32_t bandwidth_mbps;     // 带宽
    uint32_t latency_us;         // 延迟
} HixlTransportPathInfo;

HixlStatus HixlGetTransportPathInfo(
    HixlHandle handle,
    const char* remote_engine,
    HixlTransportPathInfo* out_info
);

// ========== 批量传输 ==========

/// 批量传输描述
typedef struct {
    uintptr_t local_addr;
    uintptr_t remote_addr;
    size_t len;
    HixlTransferOp op;           // READ 或 WRITE
} HixlTransferDesc;

/// 批量同步传输
HixlStatus HixlTransferSyncBatch(
    HixlHandle handle,
    const char* remote_engine,
    const HixlTransferDesc* descs,
    size_t count,
    int32_t timeout_ms
);

// ========== 传输进度查询 ==========

typedef struct {
    uint64_t total_bytes;
    uint64_t transferred_bytes;
    uint32_t completed_ops;
    uint32_t pending_ops;
    float progress_percent;
} HixlTransferProgress;

HixlStatus HixlGetTransferProgress(
    HixlHandle handle,
    HixlTransferReq req,
    HixlTransferProgress* out_progress
);
```

#### 6. 调试与监控接口 (优先级: P2)

```c
// ========== 调试日志 ==========

/// 日志级别
typedef enum {
    HIXL_LOG_ERROR   = 0,
    HIXL_LOG_WARN    = 1,
    HIXL_LOG_INFO    = 2,
    HIXL_LOG_DEBUG   = 3,
    HIXL_LOG_TRACE   = 4,
} HixlLogLevel;

/// 设置日志级别
void HixlSetLogLevel(HixlLogLevel level);

/// 设置日志回调
typedef void (*HixlLogCallbackFn)(
    HixlLogLevel level,
    const char* file,
    int line,
    const char* message
);
void HixlSetLogCallback(HixlLogCallbackFn callback);

// ========== 性能统计 ==========

typedef struct {
    uint64_t total_transfers;
    uint64_t total_bytes;
    uint64_t total_time_us;
    uint64_t min_latency_us;
    uint64_t max_latency_us;
    uint64_t avg_latency_us;
    uint32_t error_count;
} HixlTransferStats;

HixlStatus HixlGetTransferStats(
    HixlHandle handle,
    HixlTransferStats* out_stats
);

/// 重置统计信息
void HixlResetTransferStats(HixlHandle handle);

// ========== 诊断接口 ==========

/// 运行自诊断
HixlStatus HixlRunDiagnostics(
    HixlHandle handle,
    char* report,
    size_t report_size
);

/// 导出状态快照
HixlStatus HixlExportState(
    HixlHandle handle,
    char* json_out,
    size_t max_size
);
```

#### 7. NPU Kernel 可调用接口 (优先级: P3 - 取决于 HIXL 能力)

```c
// 如果 HIXL 底层支持，暴露给 NPU kernel 的接口

/// 检查是否支持 NPU kernel 直接操作
int HixlSupportsNpuKernelOps(void);

/// NPU 可调用的传输发起 (可能需要特定的编译器支持)
#ifdef __NPU_KERNEL__
HixlStatus HixlNpuKernelTransfer(
    HixlHandle handle,
    uintptr_t local_addr,
    uintptr_t remote_addr,
    size_t len,
    HixlTransferOp op
);
#endif
```

---

### 优先级排序

| 优先级 | 功能模块 | 具体功能 | 原因 |
|-------|---------|---------|------|
| **P0** | 错误诊断 | `HixlGetLastError`, 扩展错误码 | 当前调试完全依赖猜测 |
| **P0** | 内存查询 | `HixlQueryMemStatus`, `HixlGetRegisteredMemInfo` | 无法验证内存注册状态 |
| **P1** | 设备查询 | `HixlGetNpuDeviceFromPtr`, `HixlPointerGetAttribute` | 需要确认地址/设备匹配 |
| **P1** | 连接管理 | `HixlGetConnectionState`, `HixlGetConnectionInfo` | 需要连接状态可见性 |
| **P2** | 传输类型 | `HixlGetTransportType`, `HixlGetTransportPathInfo` | 区分 HCCS/RoCE |
| **P2** | 批量操作 | `HixlRegisterMemBatch`, `HixlTransferSyncBatch` | 性能优化 |
| **P2** | 调试监控 | `HixlSetLogLevel`, `HixlGetTransferStats` | 问题排查 |
| **P3** | 完成缓存 | 类似 rdmaxcel 的 completion cache | 性能优化，非必需 |
| **P3** | NPU Kernel | `HixlNpuKernelTransfer` | 取决于 HIXL 底层能力 |

---

### 实现建议

**第一阶段 (解决当前问题):**
- 增强 `HixlGetLastError` 返回详细错误信息
- 添加内存状态查询接口 (`HixlQueryMemStatus`)
- 扩展错误码枚举

**第二阶段 (提升可调试性):**
- 添加设备查询接口 (`HixlGetNpuDeviceFromPtr`)
- 添加连接状态接口 (`HixlGetConnectionState`)
- 添加传输类型检测 (`HixlGetTransportType`)

**第三阶段 (性能优化):**
- 批量操作接口
- 性能统计接口
- 调试日志系统

---

## 调试路径

### Step 1: 对比 ref-monarch 实现

```bash
diff monarch_rdma/src/backend/hixl/manager_actor.rs ref-monarch/monarch_rdma/src/backend/hixl/manager_actor.rs
```

同事的核心思路：
- Rust 端只存储元数据，不执行实际传输
- Python 端通过 ctypes 调用 HIXL

### Step 2: 启用 HIXL 详细日志

```bash
export HIXL_LOG_LEVEL=DEBUG
python tests/hixl/e2e/test_hixl_bridge_minimal.py
```

### Step 3: 检查内存地址

```python
print(f"tensor data_ptr: {hex(tensor.data_ptr())}")
print(f"tensor device: {tensor.device}")
```

---

## 架构决策

### 方案选择: Rust 集中方案

**目标：** 保持 HIXL 操作在 Rust 端执行，维持与 rdmaxcel 后端一致的架构。

### 备选方案: Python 集中方案 (ref-monarch)

如果 Rust 集中方案调试困难，可参考同事的实现：
- `ref-monarch/python/monarch/hixl_transfer.py` - Python 端 HIXL 操作
- `ref-monarch/monarch_rdma/src/backend/hixl/manager_actor.rs` - Rust 端退化为元数据存储

---

## 文件结构

### 核心修改文件

| 文件 | 职责 |
|------|------|
| `hixl-sys/src/bridge.cpp` | HIXL FFI 封装、设备上下文管理 |
| `hixl-sys/src/bridge.h` | C API 头文件定义 |
| `hixl-sys/src/lib.rs` | Rust FFI 绑定 |
| `monarch_rdma/src/backend/hixl/manager_actor.rs` | HIXL 后端核心逻辑 |
| `monarch_rdma/src/rdma_manager_actor.rs` | RDMA 管理器、engine_id 生成 |
| `python/monarch/mesh_controller.py` | 环境变量设置 (`MONARCH_NPU_DEVICE`) |

### 测试文件

| 文件 | 用途 |
|------|------|
| `tests/hixl/e2e/test_hixl_bridge_minimal.py` | 两 mesh 跨卡传输测试 |

---

## 构建和测试命令

```bash
# 激活环境
conda activate hixl

# 构建
USE_ASCEND_ENGINE=1 USE_TENSOR_ENGINE=0 pip install -e .

# 运行测试
python tests/hixl/e2e/test_hixl_bridge_minimal.py
```

---

## 下次工作入口

1. **首先阅读本文档** - 了解当前状态和问题

2. **运行测试确认状态**
   ```bash
   python tests/hixl/e2e/test_hixl_bridge_minimal.py
   ```

3. **如果 TransferSync 仍然失败**：
   - 对比 `ref-monarch/` 中同事的实现
   - 检查内存注册是否正确执行
   - 启用 HIXL 详细日志查找根因
   - **实施第一阶段功能增强** (错误诊断、内存查询)

4. **如果连接失败**：
   - 检查 engine_id 格式是否正确 (`ip:port`)
   - 检查端口是否被占用

---

## 参考资源

- **同事的实现**: `ref-monarch/` 目录
- **HIXL 源码参考**: `ref_hixl/src/` 目录
- **rdmaxcel-sys 参考**: `rdmaxcel-sys/src/` 目录 (CUDA 对应实现)