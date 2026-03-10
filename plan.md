# HIXL Backend Adaptation Plan

## Overview

This document outlines the plan for debugging and fixing the HIXL (Huawei Xfer Library) backend adaptation in Monarch for Ascend NPU.

## Analyzed Issues

### Issue 1: HixlBuffer Serialization/Deserialization Design Flaw

**Severity: Critical**

**Problem:**
- `HixlBuffer` only contains `engine_id`, `addr`, `size` but lacks `HixlMemHandle`
- `HixlMemHandle` is a pointer type (`void*`) that cannot be serialized
- When a remote peer receives `HixlBuffer`, it cannot perform transfers because:
  - HIXL requires memory to be registered locally before transfer
  - From `hixl_client.cc:ClassifyTransfers()`: transfers check if memory is registered

**Current Code (`monarch_rdma/src/backend/hixl.rs`):**
```rust
pub struct HixlBuffer {
    pub engine_id: String,
    pub addr: usize,
    pub size: usize,
}
```

**Proposed Fix:**
- Option A: Register memory on demand when transfer is initiated
  - When `hixl_transfer_sync` is called, register the local memory first
  - Keep a local cache of registered memory handles
  - Deregister after transfer completes or on buffer release

- Option B: Pre-register memory during buffer exchange
  - When receiving a `HixlBuffer`, immediately register it with local HIXL instance
  - Store the handle in a thread-safe map indexed by (engine_id, addr)
  - Reuse handles for subsequent transfers

**Recommended: Option A** - simpler and matches HIXL's design pattern

---

### Issue 2: PROCESS_HIXL Double Initialization

**Severity: Critical**

**Problem:**
- Two separate HIXL instances are created:
  1. In `HixlManagerActor::ensure_initialized()`
  2. In `ensure_process_hixl()` for `PROCESS_HIXL`
- Each instance has independent memory registration tables (`mem_map_`)
- Memory registered in one instance is not accessible in the other

**Current Code (`monarch_rdma/src/backend/hixl/manager_actor.rs:102-116`):**
```rust
fn ensure_initialized(&mut self) -> Result<&hixl_sys::Hixl> {
    // Creates first HIXL instance
    let hixl = hixl_sys::Hixl::new()?;
    hixl.initialize(&self.engine_id, &[])?;
    
    // Creates second HIXL instance for PROCESS_HIXL
    let global_hixl = hixl_sys::Hixl::new()?;
    global_hixl.initialize(&xfer_engine_id, &[])?;
    // ...
}
```

**Proposed Fix:**
- Remove the duplicate HIXL instance creation
- Use a single global `PROCESS_HIXL` for all operations
- `HixlManagerActor` should only hold a reference/pointer to the global instance
- Or: Make `HixlManagerActor` the sole owner and remove `PROCESS_HIXL`

**Recommended:** Use single global `PROCESS_HIXL`, remove instance from `HixlManagerActor`

---

### Issue 3: Connection Management Missing

**Severity: Critical**

**Problem:**
- HIXL uses a client-server connection model
- Current code only calls `Connect` but never sets up a listening server
- Connection is one-directional: client connects to server, but server cannot initiate transfer to client
- From `hixl_server.cc:30-56`: Only `port > 0` triggers `HixlCSServerListen`

**Current Code (`monarch_rdma/src/backend/hixl/manager_actor.rs:177-193`):**
```rust
pub fn hixl_transfer_sync(...) -> Result<()> {
    guard.ensure_connected(remote_engine_id, timeout_ms)?;
    // But remote_engine_id might be "ip:0" which doesn't listen
}
```

**Proposed Fix:**
- Use server-server model from HIXL examples:
  - Each node listens on a port (e.g., `ip:16000`)
  - Both sides connect to each other
- Or use client-server model:
  - One side uses `ip:port` (server)
  - Other side uses `ip:0` (client-only)

**Recommended:** Server-server model for bidirectional transfers

---

### Issue 4: engine_id Format Problem

**Severity: Critical**

**Problem:**
- Current code uses `ip:0` format (e.g., `192.168.1.1:0`)
- From HIXL docs and examples: `port=0` means no listening
- Both sides using `:0` cannot establish connections

**Current Code (`monarch_rdma/src/rdma_manager_actor.rs:284-289`):**
```rust
let ip = local_ip_for_hixl();
format!("{}:0", ip)  // port=0, no server listening
```

**HIXL Example Usage (`run_example.sh`):**
```bash
# Client-server mode
"./client_server_h2d ${IP} ${IP}:16000"      # client: ip:0 -> server: ip:16000
"./client_server_h2d ${IP}:16000"            # server listens on port 16000

# Server-server mode (bidirectional)
"./server_server_d2d ${IP}:16000 ${IP}:16001"  # listens on 16000, connects to 16001
"./server_server_d2d ${IP}:16001 ${IP}:16000"  # listens on 16001, connects to 16000
```

**Proposed Fix:**
- Change engine_id format to include a listening port
- Use a base port (e.g., 16000) + rank offset
- Ensure both sides can connect to each other

**Example:**
```rust
let base_port = 16000;
let port = base_port + rank;
let engine_id = format!("{}:{}", ip, port);
```

---

### Issue 5: ACL Device Context Initialization Timing

**Severity: Medium**

**Problem:**
- ACL context needs to be set correctly for HIXL operations
- `torch_npu` may have already initialized ACL
- Thread context needs to be set for async operations

**Current Code (`hixl-sys/src/bridge.cpp:56-76`):**
```cpp
static bool ensure_acl_device() {
    aclInit(nullptr);  // May conflict with torch_npu
    aclrtSetDevice(0);
    // ...
}
```

**Proposed Fix:**
- Check if ACL is already initialized before calling `aclInit`
- Use `aclrtGetCurrentContext` to verify context exists
- Pass context to worker threads

---

### Issue 6: Memory Type Hardcoding

**Severity: Medium**

**Problem:**
- All memory is registered as `HIXL_MEM_DEVICE`
- HIXL uses memory type to select communication path:
  - `MEM_DEVICE + MEM_DEVICE` -> `UB_D2D` or `ROCE`
  - `MEM_HOST + MEM_DEVICE` -> `UB_H2D`
  - `MEM_HOST + MEM_HOST` -> `UB_H2H`
- CPU tensors will fail or use incorrect path

**Current Code (`monarch_rdma/src/backend/hixl/manager_actor.rs:251`):**
```rust
hixl_sys::HixlMemType::HIXL_MEM_DEVICE,  // Hardcoded
```

**Proposed Fix:**
- Detect memory type from tensor device
- Pass memory type through `RdmaLocalMemory` trait or add a method
- Use `HIXL_MEM_HOST` for CPU tensors

```rust
pub trait RdmaLocalMemory: Send + Sync + Debug {
    fn addr(&self) -> usize;
    fn size(&self) -> usize;
    fn is_device_memory(&self) -> bool { true }  // Add default
}
```

---

## Build Instructions

**Prerequisites:**
- Conda environment with HIXL installed
- Ascend NPU drivers and CANN toolkit

**Build Steps:**

```bash
# 1. Activate conda environment
conda activate hixl

# 2. Build and install monarch with Ascend backend
USE_ASCEND_ENGINE=1 USE_TENSOR_ENGINE=0 pip install -e .
```

**Success Criteria:** Build completes without errors

**Notes:**
- `USE_ASCEND_ENGINE=1` enables Ascend/NPU backend
- `USE_TENSOR_ENGINE=0` disables CUDA tensor engine
- After successful build, proceed to testing phases

---

## Debug and Test Plan

### Phase 1: Minimal HIXL Test

**Goal:** Verify HIXL FFI bindings work correctly

**Steps:**
1. Create a standalone test that initializes HIXL
2. Register a small device memory buffer
3. Verify `HixlInitialize`, `HixlRegisterMem` return success
4. Test on single NPU without network

**Test File:** `test_hixl_basic.rs` or `test_hixl_basic.py`

**Success Criteria:** HIXL initialization and memory registration succeed

---

### Phase 2: Single-Process Buffer Registration

**Goal:** Verify buffer registration works in Monarch context

**Steps:**
1. Start a single-process actor mesh
2. Create an NPU tensor
3. Create `RDMABuffer` from the tensor
4. Verify `HixlManagerActor.request_buffer` succeeds
5. Check logs for HIXL initialization and registration

**Test File:** Modify `test_hixl_rdma_minimal.py`

**Success Criteria:** Buffer creation returns without timeout or error

---

### Phase 3: Two-Process Connection Test

**Goal:** Verify HIXL connection between two processes

**Steps:**
1. Spawn two processes on same host
2. Each process uses different ports (16000, 16001)
3. Both register memory
4. Exchange `HixlBuffer` via actor messaging
5. Connect each other's engines
6. Verify `HixlConnect` succeeds

**Test File:** `test_hixl_connection.py`

**Success Criteria:** Bidirectional connection established

---

### Phase 4: Data Transfer Test

**Goal:** Verify actual RDMA data transfer

**Steps:**
1. Set up two connected processes
2. Process A creates tensor with known values
3. Process B creates empty tensor
4. Transfer data from A to B via HIXL
5. Verify data integrity

**Test File:** `test_hixl_transfer.py`

**Success Criteria:** Data transferred correctly

---

### Phase 5: Integration Test

**Goal:** Full Monarch RDMA functionality on NPU

**Steps:**
1. Run existing RDMA tests adapted for NPU
2. Test with distributed tensor operations
3. Performance benchmarking

---

## Implementation Priority

| Priority | Issue | Estimated Effort |
|----------|-------|------------------|
| P0 | Issue 2: Single HIXL instance | Small |
| P0 | Issue 4: engine_id format | Small |
| P0 | Issue 3: Connection model | Medium |
| P1 | Issue 1: Memory registration on transfer | Medium |
| P2 | Issue 6: Memory type detection | Small |
| P2 | Issue 5: ACL context handling | Small |

---

## Key Code Files to Modify

1. `monarch_rdma/src/backend/hixl/manager_actor.rs` - Core HIXL backend
2. `monarch_rdma/src/backend/hixl.rs` - HixlBuffer definition
3. `monarch_rdma/src/rdma_manager_actor.rs` - engine_id generation
4. `monarch_rdma/src/rdma_components.rs` - Transfer functions
5. `hixl-sys/src/bridge.cpp` - ACL initialization

---

## Test Results (2026-03-10)

### Test File: `test_grpo_npu.py`

**Test Description:**
- Two-mesh GRPO training test
- `learner_mesh`: 1 NPU (Learner + Scorer + Queues)
- `gen_mesh`: 2 NPU (Generator ×2)
- Inter-mesh weight sync via RDMA (HIXL)

**Test Output:**
```
============================================================
GRPO on NPU — Two-mesh test
  learner_mesh: 1 NPU (Learner + Scorer + Queues)
  gen_mesh:     2 NPU (Generator x2)
  Inter-mesh weight sync via RDMA (HIXL)
============================================================
[1/5] Spawning actors on learner_mesh...
[2/5] Getting weight handles and spawning generators...
```

**Error Message:**
```
Exception: failed to read into buffer: HIXL connect to 192.168.0.117:0 failed: HIXL error 503900: HIXL_FAILED
monarch._src.actor.actor_mesh.ActorError: Actor call generator.update failed.
```

### Error Analysis

**Root Cause:**
The test failed at step 2 when Generator mesh tried to read weights from Learner mesh via RDMABuffer.

**Flow Analysis from Logs:**
1. `learner_mesh` (anon_0-1AQNFdumuseg) successfully:
   - Created HIXL instance with engine_id: `192.168.0.117:0`
   - Registered memory for RDMABuffer

2. `gen_mesh` (anon_0-1uy8QirSEdz8, anon_1-1cGmu2EoHbRz) each:
   - Created separate HIXL instances
   - Used same engine_id format: `192.168.0.117:0`

3. Cross-mesh transfer failed:
   - Generator called `RDMABuffer.read_into()` to read Learner's weights
   - HIXL tried to connect to `192.168.0.117:0`
   - Connection failed with error `503900: HIXL_FAILED`

**Error Code Analysis:**
- HIXL error `503900` (`HIXL_FAILED`) indicates connection failure
- Target address `192.168.0.117:0` has no server listening (port=0)
- No process is acting as HIXL server

### Issue Verification Summary

| Issue | Status | Evidence from Test |
|-------|--------|-------------------|
| Issue 4: engine_id format | ✅ Confirmed | All instances use `ip:0` format |
| Issue 3: Connection model | ✅ Confirmed | `HIXL connect to ...:0 failed` |
| Issue 2: Double initialization | ⚠️ Partial | Multiple rdma_manager instances observed |
| Issue 1: Serialization | ⏳ Not triggered | Test failed before transfer attempt |

### Fix Priority (Updated)

Based on test results, the fix order should be:

1. **Issue 4 + Issue 3**: Fix engine_id format and connection model first
   - This is the blocking issue preventing any cross-mesh communication
   - Need to implement port assignment and server listening

2. **Issue 2**: Consolidate HIXL instances
   - May be contributing to the problem
   - Should be fixed after connection model

3. **Issue 1**: Memory registration on transfer
   - Not yet triggered in current test
   - Will need to be addressed after connection works

---

## Notes

- HIXL documentation suggests `OPTION_AUTO_CONNECT` option might simplify connection management
- HCCS (intra-supernode) path may work without explicit port configuration
- ROCE (inter-supernode) path requires proper network configuration and port assignment
