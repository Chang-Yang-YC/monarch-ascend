# HIXL Backend Adaptation Plan

## 本轮收敛改造方案 (2026-03-15)

### 背景结论

基于 ref_monarch 对照，当前桥接失败的主因不是端口本身，而是连接流程在多个路径重复触发（握手/Connect/Transfer 前置动作重复），导致 `HcclCommPrepare` 超时被放大。

### 改造目标

1. 收敛为单一主路径：`ensure_peer_connected(remote<-local) -> local connect -> transfer`。
2. 移除重复握手入口，避免同一轮传输触发多次 Connect。
3. 保留 ACL context restore 机制，但把日志聚焦到阶段边界，便于定位失败步骤。

### 分阶段执行

1. Phase 1（本轮立即落地）
    - 在 `monarch_rdma/src/backend/hixl/manager_actor.rs` 中移除 `submit()` 内部重复的双向握手，避免与 `rdma_components` 的握手逻辑叠加。
    - 让 `submit()` 只负责执行传输调用，不再做第二层 `ensure_peer_connected + hixl_connect_peer`。

2. Phase 2（紧随其后）
    - 统一数据面路径，只保留一处“连接前置”逻辑：
      - `MONARCH_HIXL_BIDIR_CONNECT=1` 时走 `hixl_transfer_connected` 路径。
      - `MONARCH_HIXL_BIDIR_CONNECT=0` 时走 `hixl_transfer_sync` 路径。
    - 检查并消除 transfer memory 注册策略重复（长期缓存 vs 每次临时注册）的冲突风险。

3. Phase 3（验证与回归）
    - 先跑最小链路：`tests/hixl/e2e/test_hixl_bridge_minimal.py`
    - 再跑底层连接抽样：`tests/hixl/build/test_hixl_connect --server/--client`
    - 记录每次失败所在阶段（远端回连、本端 connect、TransferSync 前后）。

### 本轮验收标准

1. 同一笔传输日志中不再出现重复 bidir handshake。
2. connect 失败时，错误集中在单一路径，便于稳定复现。
3. bridge minimal 失败模式从“随机多点失败”收敛到“固定阶段失败”。

### 执行进度 (2026-03-15)

1. 已完成（Phase 1）
    - 已移除 `monarch_rdma/src/backend/hixl/manager_actor.rs` 中 `submit()` 的重复 bidir 握手逻辑，避免与 `rdma_components` 路径叠加触发多次 connect。

2. 当前阻塞（与本轮改动无关）
    - `cargo check -p monarch_rdma --features hixl` 被仓库现有依赖问题阻塞：
      `build_utils/src/lib.rs` 调用 `pyo3_build_config::get()`，但当前特性集下该符号未启用。
    - 该错误位于 `build_utils`，不在本轮修改文件内。

3. 下一步
    - 在不依赖全仓编译通过的前提下继续推进 Phase 2（收敛 transfer 路径与内存注册策略）。
    - 完成后用 bridge 最小用例做行为回归。

## 当前状态 (2026-03-15)

### 目标路径（先保 RoCE 可运行）

1. 固化 host `ip:port` engine_id（server-server）。
2. 固化 RoCE 数据面（`HCCL_INTRA_ROCE_ENABLE=1`）。
3. Connect 失败默认立即暴露，不再“失败后继续 TransferSync”。

### 这轮已落地但尚未解决问题的改动

1. `RdmaRemoteBuffer` 的 HIXL 读写路径改为显式双向握手：
    - 先通知远端 `ensure_peer_connected(local_engine)`。
    - 再本端执行 `hixl_connect_peer(remote_engine)`。
    - 不再使用“一次性 preconnect 缓存”跳过后续握手。

2. `ensure_connected` 的兜底策略收紧。
    - `MONARCH_HIXL_ALLOW_TRANSFER_WITHOUT_CONNECT` 默认从 `true` 改为 `false`。
    - 默认行为改为“connect 失败即报错”。

### 本轮验证

1. 纯底层对照：
    - `tests/hixl/build/test_hixl_connect` 显示稳定成功窗口为 RoCE `ip:port` + 跨设备。
    - `tests/hixl/unit/test_hixl_direct.py`（ACL/torch）通过。

2. 框架最小链路（hixl 环境复测后更新）：
        - 旧结果：在 `conda activate hixl` + `USE_ASCEND_ENGINE=1 USE_TENSOR_ENGINE=0 pip install -e .` 后，
            `tests/hixl/e2e/test_hixl_bridge_minimal.py` 曾表现为两类失败：
            - `MONARCH_HIXL_BIDIR_CONNECT=0`：`Connect=503900`
            - `MONARCH_HIXL_BIDIR_CONNECT=1`：双侧 `Connect=0`，但 `TransferSync START` 后卡住并最终超时
        - 2026-03-15 复测（本轮去掉 `submit()` 重复握手后）：
            - `MONARCH_HIXL_BIDIR_CONNECT=0`：仍稳定失败于 `Connect=503900`，异常为 `HcclCommPrepare` / `Connect ACL error`。
            - `MONARCH_HIXL_BIDIR_CONNECT=1`：本轮也前移并收敛到 `Connect=503900`，未再进入 `TransferSync START`。
            - 说明失败模式已从“Connect 失败 / TransferSync 卡住”两类分叉，收敛为“传输前 Connect 阶段失败”。

3. 新增隔离实验：
        - `tests/hixl/unit/test_hixl_dynamic_ports.py` 在 `hixl` 环境通过，说明“Monarch actor + 动态端口 + ctypes 直连”本身可工作。
        - 将 transfer buffer 的 `RegisterMem` 前移到双向握手之前后，bridge 仍会失败，说明问题不只在这一处顺序。
        - 去掉 Rust HIXL 初始化时默认附带的 `AutoConnect=1` / `BufferPool=0:0` 选项后，bridge 仍会失败，说明初始化选项分叉不是唯一根因。

### 当前问题定义（更新）

不再把问题定义为“底层 connect 普遍不可用”。当前更准确的定义是：
1. 底层可用，但框架桥接路径仍未稳定。
2. 在本轮进一步收敛握手与重试后，bridge 最小链路的两侧 `Connect` 已可稳定返回 0，失败点已从 Connect 前移到 `TransferSync`。
3. 当前稳定失败画像为：`TransferSync END: 503900`，ACL recent error 指向 `Failed to invoke HcclBatchPut`，而非 `HcclCommPrepare`。
4. 动态端口 actor 直连测试在 hixl 环境仍可通过，因此问题不优先归因于端口分配本身。
5. 当前更像是“Rust backend 的传输阶段（内存注册形态/写入约束/调用顺序）”与 ctypes 直连路径之间仍有关键差异。

### 下一步（P0）

1. 围绕 `TransferSync` 阶段补充结构化日志：local/remote addr、len、op、mem_type、调用前后 ACL recent err。
2. 对照 ref_monarch 与 ctypes 直连路径，重点核对“传输阶段地址约束/注册语义”而非建连语义。
3. 在当前基线下做单变量实验，验证 `HcclBatchPut` 失败是否与地址对齐、内存注册类型或 op 方向相关。

### Rust 侧初始化项盘点（当前可注入）

按“初始化/建连阶段”划分，当前 Rust 方案可注入项包括：

1. 设备与 engine_id 相关：
    - `MONARCH_NPU_DEVICE`
    - `ASCEND_RT_VISIBLE_DEVICES`
    - `MONARCH_HIXL_IP`
2. HIXL 初始化 options 相关（会影响 `HixlInitialize`）：
    - `MONARCH_HIXL_BUFFER_POOL`（映射到 `BufferPool`）
    - `MONARCH_HIXL_AUTO_CONNECT`（映射到 `AutoConnect`）
3. 连接策略相关：
    - `MONARCH_HIXL_BIDIR_CONNECT`
    - `MONARCH_HIXL_CONNECT_RETRY`
    - `MONARCH_HIXL_CONNECT_RETRY_BACKOFF_MS`
    - `MONARCH_HIXL_CONNECT_SETTLE_MS`
    - `MONARCH_HIXL_CONNECT_TIMEOUT_MS`
    - `MONARCH_HIXL_ALLOW_TRANSFER_WITHOUT_CONNECT`
4. actor/request 边界相关：
    - `MONARCH_HIXL_REQUEST_BUFFER_TIMEOUT_MS`

问题是：同一阶段有多处开关叠加（例如 `BIDIR_CONNECT` 与 `AUTO_CONNECT` 同时存在），导致行为解释空间过大，不利于定位 `Connect=503900` 的单一根因。

### 开发阶段固定基线（先与 ref_monarch 对齐）

为了减少变量，后续开发与复测先固定为一套“最小可解释配置”：

1. 保留项（允许注入）：
    - `MONARCH_NPU_DEVICE`（设备选择）
    - `MONARCH_HIXL_IP`（必要时显式指定 host IP）
    - `MONARCH_HIXL_CONNECT_TIMEOUT_MS`（仅用于调长超时）
2. 固定项（开发基线默认，不作为实验变量）：
    - `MONARCH_HIXL_BIDIR_CONNECT=1`（统一走显式双向建连语义）
    - `MONARCH_HIXL_ALLOW_TRANSFER_WITHOUT_CONNECT=0`（connect 失败即失败）
    - `MONARCH_HIXL_AUTO_CONNECT`：不注入（等价关闭隐式自动建连分叉）
    - `MONARCH_HIXL_BUFFER_POOL=0:0`（与 ref shim 初始化保持一致）
    - `MONARCH_HIXL_CONNECT_RETRY=20`
    - `MONARCH_HIXL_CONNECT_RETRY_BACKOFF_MS=100`
    - `MONARCH_HIXL_CONNECT_SETTLE_MS=200`
    - `MONARCH_HIXL_REQUEST_BUFFER_TIMEOUT_MS=10000`
3. 传输面固定：
    - `HCCL_INTRA_ROCE_ENABLE=1`（保持当前验证窗口一致）

执行约束：

1. 若要做配置实验，必须先在本基线下复现一次，再单变量改动；禁止同时改 `BIDIR_CONNECT` 与 `AUTO_CONNECT`。
2. 在确认建连根因前，不再把 `MONARCH_HIXL_AUTO_CONNECT` 作为常规调参项。
3. 所有测试结论默认指向本基线环境，避免跨轮次配置漂移。

### 2026-03-16 本轮实施结果

1. 已完成：统一 TransferSync 前建连路径
    - `monarch_rdma/src/rdma_components.rs` 已移除 `MONARCH_HIXL_BIDIR_CONNECT` 分叉。
    - 当前 write/read 已进一步收敛为：`ensure_bidir_connected -> register_transfer_memory -> hixl_transfer_connected -> deregister_transfer_memory`。
    - 握手顺序已调整为“本端先 connect，远端回连为非严格（可配置严格模式）”。

2. 已完成：统一三套内存注册生命周期
    - `monarch_rdma/src/backend/hixl/manager_actor.rs` 中 `ProcessHixl` 改为单一地址级注册表（handle + refcount）。
    - owner buffer 注册与 transfer 临时注册都走同一套 `acquire_memory/release_memory` 机制。
    - `hixl_transfer_sync` 不再直接裸调 `register_mem/deregister_mem`，而是复用统一注册生命周期。

3. 已完成：固定初始化项基线
    - 当前 HIXL 初始化 options 固定为 `BufferPool=0:0`。
    - 不再在初始化阶段注入 `MONARCH_HIXL_AUTO_CONNECT`。
    - 新增初始化后短暂 settle 窗口（默认 500ms，可由 `MONARCH_HIXL_INIT_SETTLE_MS` 覆盖）。

4. 编译与复测结果
    - 编译：`USE_ASCEND_ENGINE=1 USE_TENSOR_ENGINE=0 pip install -e .` 通过。
    - 复测：`python tests/hixl/e2e/test_hixl_bridge_minimal.py` 仍失败，但阶段已前移到 TransferSync。
    - 关键日志：`Connect END: status=0`（双向建连成功）后，`TransferSync END: status=503900`，ACL recent error 为 `Failed to invoke HcclBatchPut`。
    - 结论：本轮重构已把故障边界从“建连阶段”推进到“传输阶段”，下一步应聚焦 TransferSync 约束与内存语义。

### ref_monarch 调用流程对照（TransferSync 前）

以下是 ref_monarch 在一次 write/read 中、进入 TransferSync 之前的实际链路：

1. 上层调用 `RdmaRemoteBuffer::write_from_local/read_into_local`。
2. 从 `RdmaRemoteBuffer` 里解析远端 `HixlBuffer`（`engine_id + remote_addr + size`）。
3. 先做本地内存注册：`register_mem_if_needed(local_addr, local_size)`（带地址级引用计数，避免重复注册/反复抖动）。
4. 执行双向建连：`ensure_connected(client, remote_mgr, remote_eid)`。
    - 先 RPC 给远端 `RdmaManagerActor::ensure_peer_connected(my_eid)`，让远端先 `do_connect(my_eid)`。
    - 再本端执行 `do_connect(remote_eid)`。
    - 本端 `connected_peers` 命中则直接跳过，避免重复 Connect。
5. 上述步骤完成后，才调用 `engine.transfer_write/transfer_read`（底层 C++ shim 对应 `TransferSync`）。

对应 ACL 上下文作用点：

1. 初始化时：`hixl_init_engine` 先 `aclrtSetDevice(dev)`，再保存 `aclrtGetCurrentContext` 到 ctx。
2. 每次 HIXL 操作前（RegisterMem / Connect / TransferSync / Cleanup）：都调用 `restore_acl_ctx -> aclrtSetCurrentContext(saved_ctx)`。
3. Connect/Transfer 失败时：立即读取 `aclGetRecentErrMsg()` 输出 ACL 侧最近错误，辅助定位 `HcclCommPrepare` 之类失败来源。

### 与当前实现的不一致点（可能导致当前问题）

围绕“TransferSync 前建连阶段”，当前实现和 ref_monarch 的关键差异如下：

1. 非 bidir 分支缺少“先远端回连”步骤。
    - 当前 `MONARCH_HIXL_BIDIR_CONNECT=0` 走 `hixl_transfer_sync`，内部只做“本端 ensure_connected(remote)”；没有显式远端 `ensure_peer_connected(my_eid)`。
    - ref_monarch 的 `ensure_connected` 固定是“远端先连回 + 本端再连出”。
    - 风险：单向建连时序更敏感，容易放大 `HcclCommPrepare` 失败窗口。

2. 内存注册生命周期模型不一致。
    - ref_monarch：`register_mem_if_needed` 地址级引用计数，长生命周期复用。
    - 当前实现：
      - 一类是 `registered_buffers`（buffer owner 注册）；
      - 一类是 `registered_transfer_buffers`（bidir 传输临时注册）；
      - 另一类是 `hixl_transfer_sync` 内“每次 register + transfer + deregister”。
    - 风险：同一地址在不同路径可能被重复注册/反复注销，导致建连前状态抖动、错误边界不稳定。

3. 建连入口仍是多路径（按开关分叉）。
    - 当前有 `ensure_bidir_connected + hixl_transfer_connected` 与 `hixl_transfer_sync` 两套前置流程。
    - ref_monarch 的核心是单一 `ensure_connected` 语义，Transfer 前建连策略一致。
    - 风险：不同模式下失败阶段不一致，难以保证复现稳定性。

4. 初始化选项和 ref_monarch 不完全一致。
    - ref_monarch shim 初始化固定带 `BufferPool=0:0`，不依赖额外 AutoConnect 分叉。
    - 当前实现允许通过环境变量注入 `MONARCH_HIXL_AUTO_CONNECT` / `MONARCH_HIXL_BUFFER_POOL`。
    - 风险：隐式 AutoConnect 可能与显式双向握手叠加，形成不可见竞态。

5. 引擎 ID 生成与兜底策略仍有不稳定点。
    - 当前 `local_ip_for_hixl` 失败兜底到 `0.0.0.0`（监听语义），并由 `pid` 派生动态端口。
    - ref_monarch 更偏向明确 host IP（并可由 `MONARCH_PYTHON_HIXL_ENGINE_ID` 固化）。
    - 风险：在多进程/重启场景下，engine_id 解析或可达性不稳，可能在 Connect 阶段触发伪失败。

6. ACL 上下文机制虽然对齐，但线程/调用边界仍需验证。
    - 两边都做了“每次调用前 restore ACL context”。
    - 但当前实现将更多状态机逻辑放在 Rust 多路径分支中（不同函数、不同锁路径）。
    - 风险：若某路径绕过或晚于预期 restore 点，Connect 阶段仍可能出现 ACL 相关错误放大。

综合判断：

1. 当前最优先不是继续改 TransferSync 本体，而是先把“建连入口收敛为单一路径”（无论 bidir 开关，语义都保持远端回连 + 本端连出）。
2. 同步收敛内存注册语义（同一地址只保留一种注册生命周期策略），避免建连前后重复 register/deregister 干扰。
3. 在此基础上再看 Connect 503900 是否仍稳定复现，若仍存在，再继续下钻 ACL/HCCL 侧根因。

### 历史说明

2026-03-12 ~ 2026-03-13 的“单向 connect 失败”排查细节仍可从 git 历史查看；但已无需参考
本文档顶部已清理为当前有效结论，避免旧阶段叙述和现状冲突。

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

### 已完成的调试步骤

#### 2026-03-12 调试记录

1. **添加调试日志到 `rdma_components.rs`**
   - 在 `write_from_local` 中打印 `self.backends` 和 `remote_hixl`
   - 确认 `HixlBuffer` 的 `engine_id` 正确传递

2. **添加调试日志到 `bridge.cpp`**
   - 在 `HixlConnect` 中添加连接参数和结果日志
   - 发现连接现在成功了

3. **关键发现**
   - engine_id 一致性：Producer `192.168.0.117:16218`，Consumer 尝试连接的也是 `192.168.0.117:16218` ✓
   - HixlBuffer 序列化正确：`{engine_id: "192.168.0.117:16218", addr: 20616935636992, size: 64}`
   - 连接成功：`HIXL: connected to remote engine: 192.168.0.117:16218`
   - 问题在 TransferSync 调用后

### 下一步调试建议

#### Step 1: 检查远端内存注册

当前流程中，远端内存注册逻辑：
```
Producer: request_buffer() -> HixlManagerActor::request_buffer() -> 注册内存
Consumer: 收到 HixlBuffer {addr, size, engine_id} -> 但这个地址在 Consumer 的 HIXL 实例中未注册！
```

**问题：** HixlBuffer 只包含地址信息，但 Consumer 端的 HIXL 实例并不知道这个内存。

**解决方案：** 参考 ref-monarch 实现，可能需要：
- 远端内存不需要在本地注册
- 或者 HIXL 的跨设备访问机制与 ibverbs 不同

#### Step 2: 对比 ref-monarch 实现

```bash
# 查看同事的实现差异
diff monarch_rdma/src/rdma_components.rs ref-monarch/monarch_rdma/src/rdma_components.rs
```

关键区别：ref-monarch 采用 Python 集中方案，所有 HIXL 操作在 Python 端。

#### Step 3: 检查 HIXL 跨设备通信要求

可能需要：
- 双向连接（Producer 也需要连接 Consumer）
- 特定的 HIXL 选项（如 `HIXL_OPTION_HCCS_MODE`）
- 确认两个 NPU 是否在同一超级节点

#### Step 4: 尝试简化测试

创建单机双进程测试，排除 Actor 框架的复杂性：
```python
# 测试两个独立进程通过 HIXL 通信
# 进程 1: 注册内存，等待连接
# 进程 2: 连接并执行 TransferSync
```

---

## 架构决策

### 方案选择: Rust 集中方案

**目标：** 保持 HIXL 操作在 Rust 端执行，维持与 rdmaxcel 后端一致的架构。

### 关键差异分析

#### rdmaxcel vs HIXL 的跨设备通信模型

```
rdmaxcel (ibverbs) 跨设备流程:
┌─────────────────────────────────────────────────────────────────────────┐
│ Producer (GPU 0)                    Consumer (GPU 1)                    │
│ ┌─────────────────┐                ┌─────────────────┐                 │
│ │ 1. 注册内存 MR   │                │ 1. 注册本地内存  │                 │
│ │ 2. 返回 MR 信息  │ ──────────────>│ 2. 收到远端地址  │                 │
│ │    (addr+rkey)  │                │ 3. 发起 RDMA Write│                │
│ │ 3. 等待         │ <────────────── │    (使用 rkey)   │                 │
│ └─────────────────┘                └─────────────────┘                 │
│                                                                      │
│ 关键：rkey (remote key) 允许远端直接访问已注册内存                      │
└─────────────────────────────────────────────────────────────────────────┘

HIXL 跨设备流程 (当前实现):
┌─────────────────────────────────────────────────────────────────────────┐
│ Producer (NPU 0)                    Consumer (NPU 1)                    │
│ ┌─────────────────┐                ┌─────────────────┐                 │
│ │ 1. HixlRegisterMem│              │ 1. HixlRegisterMem│                │
│ │ 2. 返回 HixlBuffer │ ───────────>│ 2. 收到远端地址   │                │
│ │    {addr, size,  │                │ 3. HixlConnect    │                │
│ │     engine_id}   │                │ 4. HixlTransferSync│               │
│ │ 3. HixlFinalize  │ <───────────── │    ❓ 问题点     │                 │
│ └─────────────────┘                └─────────────────┘                 │
│                                                                      │
│ 问题：HixlTransferSync 是否需要远端内存先在本地注册？                   │
│       HIXL 的跨设备访问机制是什么？                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

### 可能的问题根因

1. **远端内存未在 Consumer 端注册**
   - 当前：Consumer 直接使用 Producer 返回的地址执行 TransferSync
   - 问题：HIXL 可能需要在两端都注册内存才能执行跨设备传输

2. **HIXL 需要双向连接**
   - 当前：只有 Consumer 连接 Producer
   - 可能需要：Producer 也连接 Consumer（双向连接）

3. **跨 NPU 设备内存访问限制**
   - HIXL 可能不支持直接访问另一张 NPU 的设备内存
   - 可能需要通过 host 内存中转

4. **HIXL 初始化选项问题**
   - 当前选项：`BufferPool=0:0`, `AutoConnect=1`
   - 可能需要其他选项支持跨设备通信

### 备选方案: Python 集中方案 (ref-monarch)

如果 Rust 集中方案调试困难，可参考同事的实现：
- `ref_monarch/python/monarch/_src/rdma/hixl_transfer.py` - Python 端 HIXL 操作
- `ref-monarch/monarch_rdma/src/backend/hixl/manager_actor.rs` - Rust 端退化为元数据存储

同事方案的核心思路：
- Rust 端只存储 `engine_id` 和 `addr/size`
- 所有 HIXL 操作（init, register, connect, transfer）在 Python 端通过 ctypes 调用
- 可能更容易调试（Python 侧有更多工具支持）

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
   USE_ASCEND_ENGINE=1 USE_TENSOR_ENGINE=0 pip install -e .
   python tests/hixl/e2e/test_hixl_bridge_minimal.py
   ```

3. **如果 TransferSync 仍然失败**：
   - 检查 HIXL 是否需要在两端都注册内存
   - 尝试双向连接（Producer 也连接 Consumer）
   - 参考 `ref_hixl/src/` 中的示例代码
   - 对比 `ref-monarch/` 中同事的实现

4. **关键调试问题**：
   - HIXL TransferSync 是否需要远端内存先在本地注册？
   - HIXL 是否支持跨 NPU 设备内存直接访问？
   - 是否需要特殊的 HIXL 初始化选项？

5. **备选路径**：
   - 如果 Rust 方案继续受阻，考虑切换到 Python 集中方案（ref-monarch）
   - Python 方案可能更容易调试

---

## 下一步修改计划 (Rust 封装优先, 2026-03-13)

目标：对齐 monarch 原有后端设计（管理器 Actor + 统一 RdmaRemoteBuffer 抽象），在 Rust 体系内完成 Ascend/HIXL 的可用封装，不依赖 Python 侧接管传输。

### 阶段 1: 先把建连稳定性做实 (P0)

修改点：
1. 增强 `ensure_connected()` 的重试机制（指数退避 + 上限 + 最终错误摘要）。
2. 引入“显式双向预连接”握手：在第一次跨 mesh 传输前，由双方 `RdmaManagerActor` 分别执行一次 `connect(peer_engine_id)`。
3. engine_id 生成改为“可配置优先、自动推断兜底”：
    - 优先读取明确环境变量（如 `MONARCH_HIXL_IP` / `HIXL_BASE_PORT`）。
    - 自动推断只作为 fallback，并打印最终绑定 IP/NIC。
4. 在连接失败路径记录更多上下文：本端 engine_id、目标 engine_id、当前设备 ID、ACL device、timeout、重试次数。

主要文件：
- `monarch_rdma/src/rdma_manager_actor.rs`
- `monarch_rdma/src/backend/hixl/manager_actor.rs`
- `hixl-sys/src/bridge.cpp`

验收标准：
1. `test_hixl_bridge_minimal.py` 连续运行 20 次，Connect 失败率为 0。
2. 失败日志可直接区分：IP 不可达 / 端口未监听 / 设备上下文异常 / 远端未就绪。

### 阶段 2: 补齐 Rust 侧可观测性与诊断接口 (P0)

修改点：
1. 在 `hixl-sys` 增加 `HixlGetLastError()` 封装，并在 Rust `HixlError` 中串联最近一次错误详情。
2. 增加最小内存查询接口（先做本地缓存版）：
    - 已注册段数量
    - 按地址查询是否命中已注册区间
3. 在 `hixl_transfer_sync()` 前后打印结构化诊断信息：
    - local/remote addr 与 size
    - transfer op
    - 是否已连接
    - register/deregister 成功与否

主要文件：
- `hixl-sys/src/bridge.h`
- `hixl-sys/src/bridge.cpp`
- `hixl-sys/src/lib.rs`
- `monarch_rdma/src/backend/hixl/manager_actor.rs`

验收标准：
1. 出现 `503900` 时可输出“最近错误上下文”而不只是 `HIXL_FAILED`。
2. 能在日志中确认传输前内存注册状态。

### 阶段 3: 对齐 monarch 抽象边界 (P1)

修改点：
1. 保持 `RDMABuffer` Python API 不变，内部继续走 Rust 提交路径（不切到 Python ctypes transport）。
2. 将 HIXL 特有元数据收敛在 `RdmaBackendContext::Hixl` 与 HIXL manager，不向上层泄漏实现细节。
3. 清理调试期临时日志，保留可开关的 debug tracing（通过环境变量启用）。

主要文件：
- `monarch_rdma/src/rdma_components.rs`
- `monarch_rdma/src/backend/hixl/manager_actor.rs`
- `python/monarch/_src/rdma/rdma.py`（仅确认接口不变，不做行为迁移）

验收标准：
1. 现有调用侧无需改代码即可通过最小 e2e。
2. HIXL 细节仅在后端模块可见。

### 阶段 4: 测试矩阵与回归门禁 (P1)

测试项：
1. 单机双进程双卡：`write_from` / `read_into` 双向。
2. 同一用例在不同启动顺序下验证（Producer 先起、Consumer 先起、并发启动）。
3. 异常注入：错误 engine_id、错误设备号、超时场景。

新增测试建议：
- `tests/hixl/e2e/test_hixl_connect_retry.py`
- `tests/hixl/e2e/test_hixl_bidirectional_connect.py`
- `tests/hixl/e2e/test_hixl_engine_id_binding.py`

验收标准：
1. 关键 e2e 纳入 CI 夜间或专项流水线。
2. 出现回归时可通过测试名快速定位到连接、内存注册或传输阶段。

### 回滚与备选策略

1. 保留特性开关：`MONARCH_HIXL_BIDIR_CONNECT=0/1`、`MONARCH_HIXL_CONNECT_RETRY=N`。
2. 若阶段 1 无法稳定建连，再评估临时切换到 ref-monarch 的 Python 集中传输方案，但仅作为过渡，不作为最终架构。
3. 任一阶段回归时，优先回退到“上一阶段最后通过 e2e 的 commit”。

---

## 参考资源

- **同事的实现**: `ref_monarch/` 目录
- **HIXL 源码参考**: `ref_hixl/src/` 目录
- **rdmaxcel-sys 参考**: `rdmaxcel-sys/src/` 目录 (CUDA 对应实现)

---

## 同事 Python 方案完整对比结论与落地计划 (2026-03-13 更新)

### 对比结论摘要

1. **同事通过测试的关键不是“功能更多”，而是“状态机更保守”**
    - Python 侧在单进程内用全局锁串行化 HIXL 操作（`_lock`），并缓存 `ctx/connected_peers/registered_addrs`，降低并发时序抖动。
    - 当前 Rust 侧虽然结构更统一，但在初始化线程、注册生命周期、连接时机上更激进，容易放大 HIXL 对时序敏感的问题。

2. **ACL 上下文恢复是稳定性的硬前提**
    - ref_monarch 的 ctypes shim 在每次 Connect/Register/Transfer 前显式恢复 ACL context。
    - 我们当前 `hixl-sys/src/bridge.cpp` 也实现了相同思路，但上层初始化阶段仍需进一步保证“设备绑定 + 初始化线程”一致性。

3. **engine_id 使用 host IP 是当前正确方向**
    - 现场验证中，host IP（`192.168.*`）初始化稳定；device IP（`29.182.*`）曾触发 Initialize `503900`。
    - 因此当前默认应继续保持 host IP 策略，device IP 不应作为默认 engine_id 来源。

4. **当前主要故障已从 Initialize 前移/后移到运行期 Connect 失败**
    - 表现为单向 connect 失败（consumer->producer）而反向可能成功。
    - 说明问题更接近握手时序、连接复用、注册顺序，而不是“完全不可初始化”。

### 与当前 Rust 实现的关键差异（需收敛）

1. **全局实例管理**
    - 当前：`OnceLock<Mutex<ProcessHixl>>`，一旦初始化后很难重建。
    - 风险：错误初始化后无法平滑恢复（例如错误 engine_id、错误设备上下文）。

2. **内存注册生命周期**
    - 当前：`hixl_transfer_sync()` 每次传输都 `register_mem -> transfer -> deregister_mem`。
    - 风险：高频场景下增大失败概率，也放大 HIXL 内部状态波动。

3. **连接策略**
    - 当前：有重试与可选双向预连接（`MONARCH_HIXL_BIDIR_CONNECT`），但触发点仍在传输路径。
    - 风险：首包传输承担建连，容易把“连接未完全收敛”暴露为传输失败。

4. **IP 推断代码未完全收敛**
    - 虽然默认已改为 host IP，但 `rdma_manager_actor.rs` 中仍保留 device ip 解析辅助函数，易造成后续误用。

### 修改计划（按优先级）

#### P0: 稳定性优先（先解决运行期 connect 失败）

1. **初始化路径加固：显式设备绑定 + 专用初始化线程**
    - 在 HIXL initialize 前显式设置 ACL 设备（通过 `HixlSetAclDevice` 封装）。
    - 将初始化阶段收敛到专用 OS 线程，避免与 runtime 线程池上下文混杂。

2. **连接时机前移：首个远端 buffer 元数据阶段完成预连接**
    - 不把“首次 connect”放在读写热路径里。
    - 保持 `MONARCH_HIXL_BIDIR_CONNECT=1` 为默认，避免单向建连不对称。

3. **连接重试策略增强（默认值保守化）**
    - 提高默认重试次数（建议 20~30），保留指数退避。
    - 首次 connect 成功后增加短 settle 时间（可配置，默认 100~300ms）。

#### P1: 传输路径收敛（降低抖动）

1. **将“每次传输 register/deregister”改为“缓存注册 + 引用计数”**
    - 复用已有 buffer 级别生命周期管理，减少重复注册。
    - 在 `ReleaseBuffer` 时再统一 deregister。

2. **统一错误画像**
    - 连接失败日志统一输出：local_engine、remote_engine、acl_device、timeout、attempt、last_error。
    - 传输失败日志统一输出：local_addr、remote_addr、len、op、是否已连接、是否已注册。

#### P1: 配置与代码清理

1. **明确 host IP 为默认策略**
    - 保留 `MONARCH_HIXL_IP` 作为最高优先级覆盖。
    - 清理或注释掉不再使用的 device ip 自动推断函数，防止被误接入。

2. **完善文档和开关语义**
    - 明确 `MONARCH_HIXL_BIDIR_CONNECT`、`MONARCH_HIXL_CONNECT_RETRY`、`MONARCH_HIXL_CONNECT_TIMEOUT_MS` 的默认值和建议值。

### 验收标准（新增）

1. `tests/hixl/e2e/test_hixl_bridge_minimal.py`：连续 20 次通过，且不出现卡在 `[1/3]` 的停滞。
2. 在 `MONARCH_HIXL_BIDIR_CONNECT=1` 与 `=0` 两组模式下，失败模式应可解释且日志可定位。
3. 连接失败时必须能在单条日志中看到：
    - local/remote engine_id
    - 当前 ACL device
    - 重试次数与最终错误

### 备选方案定位（不作为主路线）

若 Rust 路线短期无法压住波动，可临时引入“Python ctypes 旁路”作为诊断开关（仅用于隔离问题边界，不作为长期架构）：
- 用于快速判断故障在 Rust 编排层，还是 HIXL 底层/环境本身。
- 最终目标仍是回归 Rust-centric 统一实现。

---

## 最小对照实验续跑结论 (2026-03-15)

### 实验目标

先验证“纯底层是否可复现/可通过”，再与 bridge 最小链路对照，判断问题是否稳定落在上层编排。

### 实验矩阵与结果

1. 纯底层 C++: `tests/hixl/build/test_hixl_connect`
    - 该二进制内置多组配置对照（HCCS/RoCE, dev0+dev0/dev0+dev1）。
    - 结果：
      - HCCS `ip-only`（dev0+dev0 / dev0+dev1）：`103900 PARAM_INVALID`
      - HCCS `ip:0`（dev0+dev1）：`503900 FAILED`
      - RoCE `ip:port`（dev0+dev0）：`103900 PARAM_INVALID`
      - RoCE `ip:port`（dev0+dev1）：`0 SUCCESS`
    - 结论：底层并非“全失败”，而是对连接模式/设备组合敏感；当前稳定成功窗口是 RoCE `ip:port` + 跨设备。

2. 纯底层 Python ctypes: `tests/hixl/unit/test_hixl_direct.py`
    - ACL 内存路径：PASS（Connect/TransferSync 均为 0）
    - torch_npu 内存路径：PASS（Connect/TransferSync 均为 0）
    - 结论：在相同机器与环境下，纯底层读写链路可稳定跑通。

3. Bridge 最小链路: `tests/hixl/e2e/test_hixl_bridge_minimal.py`
    - 在 `monarch_ascend` 环境中曾出现 PASS。
    - 但按当前约定环境 `hixl` 重新构建后，仍能稳定复现失败。
    - 结论：此前“bridge 最小链路已解决”的判断不成立，必须以 `hixl` 环境结果为准。

### 两侧连接过程（按实际代码）

1. 纯底层 `test_hixl_connect --server/--client`
    - server 进程：`Initialize(local_engine)` -> 常驻等待约 15s。
    - client 进程：`Initialize(local_engine)` -> sleep 2s -> `Connect(remote_engine, 10000)`。
    - orchestrator 模式会先 fork server，再延迟 3s fork client，避免“client 先连而远端未初始化”。

2. bridge 最小链路 `test_hixl_bridge_minimal.py`
    - Producer(NPU0)：创建张量并构建远端 handle（含 engine_id/addr/size）。
    - Consumer(NPU1)：初始化本端 HIXL、注册本端内存。
    - 双侧在传输前完成 `Connect`（日志可见两端 connect 均返回 0）。
    - 再执行 `TransferSync WRITE` 与 `TransferSync READ`，最终校验 sum 一致。

### “单独跑 --client 失败”原因定位

现象命令：
- `tests/hixl/build/test_hixl_connect --client 192.168.0.117:30002 1 192.168.0.117:30001`

直接原因：
1. 该命令只启动了 client，没有配对启动 `--server 192.168.0.117:30001 ...`。
2. `Connect()` 目标端点未就绪时，HIXL 返回失败码（常见为 `503900`）。
3. 该二进制的稳定用法是 orchestrator 自带双进程流程，或手动同时起 server/client 并保证时序。

补充：
- 因此这类失败不能直接证明“底层不可用”；它首先是“对端未就绪/调用方式不完整”导致的预期失败。

### 对当前修改计划的影响

1. 优先级调整
    - 将“底层可用性怀疑”降级，主攻“上层场景差异导致的时序/资源竞争窗口”。

2. 保留并强化的改造项
    - 连接前移与预热（把首次 connect 尽量移出热路径）。
    - 统一 connect 诊断日志（包含 local/remote engine_id、attempt、timeout、ACL device）。
    - 在失败路径区分“远端未就绪”与“参数模式不支持（ip-only/ip:0/dev 组合）”。

3. 新增回归门禁建议
    - 在 CI 或专项流水线固定加入三类最小用例：
      - `tests/hixl/build/test_hixl_connect`（模式矩阵）
      - `tests/hixl/unit/test_hixl_direct.py`（ctypes 直连）
      - `tests/hixl/e2e/test_hixl_bridge_minimal.py`（桥接路径）