/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//
// C bridge for HIXL (Huawei Xfer Library).
//
// Wraps the C++ hixl::Hixl class into flat extern "C" functions that Rust
// can call via FFI.  The bridge is compiled against the real HIXL headers
// and linked against libcann_hixl.so at runtime (dlopen).
//
// Key design (following ref_monarch approach):
// 1. Each HixlContext stores the ACL context from initialization
// 2. Before every HIXL call, we restore the ACL context via aclrtSetCurrentContext
// 3. This enables safe multi-threaded access from Rust/Python

#include "bridge.h"
#include <dlfcn.h>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <map>
#include <vector>
#include <unistd.h>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <pthread.h>

// Helper function to get current timestamp as string
static std::string timestamp() {
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    auto t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&t), "%H:%M:%S") << "." << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

#define TS_LOG() std::cerr << "[" << timestamp() << "] [HIXL-SYS] " << std::flush

// Forward-declare hixl types only if the real headers are available.
// We dlopen libcann_hixl.so at runtime to avoid a hard link-time dependency.
// The bridge uses a factory function obtained via dlsym to create Hixl objects.

// Since hixl::Hixl is a C++ class we cannot dlsym its methods directly.
// Instead we dlopen the shim library (libhixl_c_api.so) that exports flat
// C wrappers, OR — when HIXL headers are available at compile time — we
// link directly.

// Try to include the real HIXL headers.  If the build system found them
// (build.rs adds -I for the CANN include dir) then we compile a direct bridge.
#if __has_include(<hixl/hixl.h>)
#define HIXL_HEADERS_AVAILABLE 1
#include <hixl/hixl.h>
#include <hixl/hixl_types.h>
#endif

// ACL runtime for device context setup (HIXL requires this)
#if __has_include(<acl/acl.h>)
#define ACL_AVAILABLE 1
#include <acl/acl.h>
#include <acl/acl_rt.h>
#else
#define ACL_AVAILABLE 0
#endif

#if !defined(HIXL_HEADERS_AVAILABLE)
#define HIXL_HEADERS_AVAILABLE 0
#endif

// ============================================================================
// Internal context structure (following ref_monarch approach)
// ============================================================================
// The HixlContext wraps the HIXL engine with its associated ACL context.
// This enables safe multi-threaded access by restoring the ACL context
// before each HIXL operation.

#if HIXL_HEADERS_AVAILABLE

struct HixlContext {
    hixl::Hixl* engine = nullptr;
#if ACL_AVAILABLE
    aclrtContext acl_ctx = nullptr;
#endif
    int device_id = -1;
};

#endif

#if HIXL_HEADERS_AVAILABLE

// ============================================================================
// Direct bridge — HIXL headers present, link against libcann_hixl.so
// ============================================================================

static hixl::MemType to_mem_type(HixlMemType t) {
    return (t == HIXL_MEM_DEVICE) ? hixl::MEM_DEVICE : hixl::MEM_HOST;
}

static hixl::TransferOp to_transfer_op(HixlTransferOp op) {
    return (op == HIXL_READ) ? hixl::READ : hixl::WRITE;
}

static HixlTransferStatus from_transfer_status(hixl::TransferStatus s) {
    switch (s) {
        case hixl::TransferStatus::WAITING:   return HIXL_TRANSFER_WAITING;
        case hixl::TransferStatus::COMPLETED: return HIXL_TRANSFER_COMPLETED;
        case hixl::TransferStatus::TIMEOUT:   return HIXL_TRANSFER_TIMEOUT;
        case hixl::TransferStatus::FAILED:    return HIXL_TRANSFER_FAILED;
        default:                              return HIXL_TRANSFER_FAILED;
    }
}

// ============================================================================
// ACL Context Management (following ref_monarch approach)
// ============================================================================

#if ACL_AVAILABLE
/// Restore the ACL context for the current thread.
/// This is called before every HIXL operation to ensure thread safety.
/// ACL runtime uses thread-local state, so we must restore the context
/// when calling from a different thread than the one that initialized HIXL.
static void restore_acl_context(aclrtContext ctx) {
    if (ctx != nullptr) {
        aclError ret = aclrtSetCurrentContext(ctx);
        if (ret != 0) {
            std::cerr << "[HIXL-SYS] WARNING: aclrtSetCurrentContext failed: " << ret << std::endl;
        }
    }
}
#else
static void restore_acl_context(void* ctx) {
    (void)ctx;
}
#endif

/// Get the current ACL context (for debugging)
static void* get_current_acl_context() {
#if ACL_AVAILABLE
    aclrtContext ctx = nullptr;
    aclrtGetCurrentContext(&ctx);
    return ctx;
#else
    return nullptr;
#endif
}

extern "C" {

HixlHandle HixlCreate(void) {
    try {
        auto* ctx = new HixlContext();
        ctx->engine = new hixl::Hixl();
        
#if ACL_AVAILABLE
        // Save the current ACL context at creation time.
        // This is critical for multi-threaded access — we'll restore it
        // before every HIXL call.
        aclrtGetCurrentContext(&ctx->acl_ctx);
        
        // Also record the device ID
        aclrtGetDevice(&ctx->device_id);
        
        TS_LOG() << "HixlCreate: engine created, acl_ctx=" << ctx->acl_ctx
                  << " device=" << ctx->device_id
                  << " tid=" << pthread_self() << std::endl;
#endif
        
        return static_cast<HixlHandle>(ctx);
    } catch (const std::exception &e) {
        std::cerr << "[HIXL-SYS] HixlCreate failed: " << e.what() << std::endl;
        return nullptr;
    }
}

void HixlDestroy(HixlHandle handle) {
    if (!handle) return;
    auto* ctx = static_cast<HixlContext*>(handle);
    
#if ACL_AVAILABLE
    restore_acl_context(ctx->acl_ctx);
#endif
    
    if (ctx->engine) {
        delete ctx->engine;
    }
    delete ctx;
}

HixlStatus HixlInitialize(HixlHandle handle,
                           const char *local_engine,
                           const HixlOption *options,
                           size_t num_options) {
    if (!handle || !local_engine) return HIXL_PARAM_INVALID;

    auto* ctx = static_cast<HixlContext*>(handle);
    
    // Restore ACL context before operation (ref_monarch pattern)
    restore_acl_context(ctx->acl_ctx);

    TS_LOG() << "Initialize START: engine=" << local_engine 
              << " pid=" << getpid()
              << " tid=" << pthread_self()
              << " acl_ctx=" << get_current_acl_context() << std::endl;

    std::map<hixl::AscendString, hixl::AscendString> opts;
    for (size_t i = 0; i < num_options; ++i) {
        if (options[i].key && options[i].value)
            opts[hixl::AscendString(options[i].key)] =
                hixl::AscendString(options[i].value);
    }
    
    HixlStatus status = ctx->engine->Initialize(hixl::AscendString(local_engine), opts);
    
    TS_LOG() << "Initialize END: engine=" << local_engine << " status=" << status << std::endl;
    return status;
}

void HixlFinalize(HixlHandle handle) {
    if (!handle) return;
    auto* ctx = static_cast<HixlContext*>(handle);
    
    // Restore ACL context before operation
    restore_acl_context(ctx->acl_ctx);
    
    ctx->engine->Finalize();
}

HixlStatus HixlRegisterMem(HixlHandle handle, uintptr_t addr, size_t len,
                            HixlMemType mem_type, HixlMemHandle *out) {
    if (!handle || !out) return HIXL_PARAM_INVALID;
    
    auto* ctx = static_cast<HixlContext*>(handle);
    
    // Restore ACL context before operation
    restore_acl_context(ctx->acl_ctx);
    
    // Check 2MB alignment for HCCS mode
    constexpr size_t HCCS_ALIGN = 2UL * 1024 * 1024;
    if (addr % HCCS_ALIGN != 0) {
        TS_LOG() << "WARNING: addr=" << (void*)addr << " is NOT 2MB-aligned"
                  << " (offset=" << (addr % HCCS_ALIGN) << ")"
                  << " — HCCS transfers may fail" << std::endl;
    }
    
    hixl::MemDesc mem{}; mem.addr = addr; mem.len = len;
    hixl::MemHandle mh = nullptr;
    auto s = ctx->engine->RegisterMem(mem, to_mem_type(mem_type), mh);
    *out = mh;
    
    TS_LOG() << "RegisterMem: addr=" << (void*)addr << " len=" << len
              << " status=" << s << " tid=" << pthread_self() << std::endl;
    return s;
}

HixlStatus HixlDeregisterMem(HixlHandle handle, HixlMemHandle mh) {
    if (!handle) return HIXL_PARAM_INVALID;
    
    auto* ctx = static_cast<HixlContext*>(handle);
    
    // Restore ACL context before operation
    restore_acl_context(ctx->acl_ctx);
    
    return ctx->engine->DeregisterMem(mh);
}

HixlStatus HixlConnect(HixlHandle handle, const char *remote, int32_t timeout) {
    if (!handle || !remote) return HIXL_PARAM_INVALID;
    
    auto* ctx = static_cast<HixlContext*>(handle);
    
    // Restore ACL context before operation
    restore_acl_context(ctx->acl_ctx);
    
    TS_LOG() << "Connect START: remote=" << remote
              << " timeout=" << timeout << "ms"
              << " pid=" << getpid()
              << " tid=" << pthread_self() << std::endl;
    
    HixlStatus status = ctx->engine->Connect(hixl::AscendString(remote), timeout);
    
    TS_LOG() << "Connect END: remote=" << remote << " status=" << status << std::endl;
    
    // Log ACL error if connect failed
    if (status != 0) {
#if ACL_AVAILABLE
        const char* err = aclGetRecentErrMsg();
        TS_LOG() << "Connect ACL error: " << (err ? err : "none") << std::endl;
#endif
    }
    
    return status;
}

HixlStatus HixlDisconnect(HixlHandle handle, const char *remote, int32_t timeout) {
    if (!handle || !remote) return HIXL_PARAM_INVALID;
    
    auto* ctx = static_cast<HixlContext*>(handle);
    
    // Restore ACL context before operation
    restore_acl_context(ctx->acl_ctx);
    
    return ctx->engine->Disconnect(hixl::AscendString(remote), timeout);
}

HixlStatus HixlTransferSync(HixlHandle handle, const char *remote,
                             HixlTransferOp op, const HixlTransferOpDesc *descs,
                             size_t n, int32_t timeout) {
    if (!handle || !remote) return HIXL_PARAM_INVALID;
    
    auto* ctx = static_cast<HixlContext*>(handle);
    
    // Restore ACL context before operation (CRITICAL for thread safety)
    restore_acl_context(ctx->acl_ctx);

    TS_LOG() << "TransferSync START: remote='" << remote << "'"
              << " op=" << (int)op << " n=" << n
              << " timeout=" << timeout
              << " pid=" << getpid()
              << " tid=" << pthread_self()
              << " acl_ctx=" << get_current_acl_context() << std::endl;
    for (size_t i = 0; i < n; ++i) {
        TS_LOG() << "  [" << i << "] local=" << (void*)descs[i].local_addr
                  << " remote=" << (void*)descs[i].remote_addr
                  << " len=" << descs[i].len << std::endl;
    }

    std::vector<hixl::TransferOpDesc> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i].local_addr = descs[i].local_addr;
        v[i].remote_addr = descs[i].remote_addr;
        v[i].len = descs[i].len;
    }
    
    auto s = ctx->engine->TransferSync(
        hixl::AscendString(remote), to_transfer_op(op), v, timeout);
    
    TS_LOG() << "TransferSync END: status=" << s << std::endl;
    
    // Log ACL error if transfer failed
    if (s != 0) {
#if ACL_AVAILABLE
        const char* err = aclGetRecentErrMsg();
        TS_LOG() << "TransferSync ACL error: " << (err ? err : "none") << std::endl;
#endif
    }
    
    return s;
}

HixlStatus HixlTransferAsync(HixlHandle handle, const char *remote,
                              HixlTransferOp op, const HixlTransferOpDesc *descs,
                              size_t n, HixlTransferReq *out) {
    if (!handle || !remote || !out) return HIXL_PARAM_INVALID;
    
    auto* ctx = static_cast<HixlContext*>(handle);
    
    // Restore ACL context before operation
    restore_acl_context(ctx->acl_ctx);
    
    std::vector<hixl::TransferOpDesc> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i].local_addr = descs[i].local_addr;
        v[i].remote_addr = descs[i].remote_addr;
        v[i].len = descs[i].len;
    }
    hixl::TransferArgs args{};
    hixl::TransferReq req = nullptr;
    auto s = ctx->engine->TransferAsync(
        hixl::AscendString(remote), to_transfer_op(op), v, args, req);
    *out = req;
    return s;
}

HixlStatus HixlGetTransferStatus(HixlHandle handle, HixlTransferReq req,
                                  HixlTransferStatus *out) {
    if (!handle || !out) return HIXL_PARAM_INVALID;
    
    auto* ctx = static_cast<HixlContext*>(handle);
    
    // Restore ACL context before operation
    restore_acl_context(ctx->acl_ctx);
    
    hixl::TransferStatus st;
    auto s = ctx->engine->GetTransferStatus(req, st);
    *out = from_transfer_status(st);
    return s;
}

HixlStatus HixlSendNotify(HixlHandle handle, const char *remote,
                           const char *name, const char *msg, int32_t timeout) {
    if (!handle || !remote) return HIXL_PARAM_INVALID;
    
    auto* ctx = static_cast<HixlContext*>(handle);
    
    // Restore ACL context before operation
    restore_acl_context(ctx->acl_ctx);
    
    hixl::NotifyDesc nd;
    nd.name = hixl::AscendString(name ? name : "");
    nd.notify_msg = hixl::AscendString(msg ? msg : "");
    return ctx->engine->SendNotify(hixl::AscendString(remote), nd, timeout);
}

HixlStatus HixlGetNotifies(HixlHandle handle,
                            void (*cb)(const char *, const char *, void *),
                            void *ud) {
    if (!handle || !cb) return HIXL_PARAM_INVALID;
    
    auto* ctx = static_cast<HixlContext*>(handle);
    
    // Restore ACL context before operation
    restore_acl_context(ctx->acl_ctx);
    
    std::vector<hixl::NotifyDesc> notifies;
    auto s = ctx->engine->GetNotifies(notifies);
    if (s == hixl::SUCCESS) {
        for (const auto &n : notifies)
            cb(n.name.GetString(), n.notify_msg.GetString(), ud);
    }
    return s;
}

const char *HixlGetStatusString(HixlStatus status) {
    switch (status) {
        case HIXL_SUCCESS:            return "HIXL_SUCCESS";
        case HIXL_PARAM_INVALID:      return "HIXL_PARAM_INVALID";
        case HIXL_TIMEOUT:            return "HIXL_TIMEOUT";
        case HIXL_NOT_CONNECTED:      return "HIXL_NOT_CONNECTED";
        case HIXL_ALREADY_CONNECTED:  return "HIXL_ALREADY_CONNECTED";
        case HIXL_NOTIFY_FAILED:      return "HIXL_NOTIFY_FAILED";
        case HIXL_UNSUPPORTED:        return "HIXL_UNSUPPORTED";
        case HIXL_FAILED:             return "HIXL_FAILED";
        case HIXL_RESOURCE_EXHAUSTED: return "HIXL_RESOURCE_EXHAUSTED";
        case HIXL_NOT_INITIALIZED:    return "HIXL_NOT_INITIALIZED";
        default:                      return "HIXL_UNKNOWN_ERROR";
    }
}

int32_t HixlSetAclDevice(int32_t device_id) {
#if ACL_AVAILABLE
    auto ret = aclrtSetDevice(device_id);
    if (ret != 0) {
        std::cerr << "[HIXL-SYS] HixlSetAclDevice(" << device_id << ") failed: " << ret << std::endl;
        return -2;
    }
    std::cerr << "[HIXL-SYS] HixlSetAclDevice: set device " << device_id << std::endl;
    return device_id;
#else
    return -3;
#endif
}

int32_t HixlGetAclDevice(void) {
#if ACL_AVAILABLE
    int32_t dev = -1;
    aclrtGetDevice(&dev);
    return dev;
#else
    return -3;
#endif
}

/// Get the saved ACL context from a HixlHandle (for debugging)
uintptr_t HixlGetSavedAclContext(HixlHandle handle) {
#if ACL_AVAILABLE
    if (!handle) return 0;
    auto* ctx = static_cast<HixlContext*>(handle);
    return reinterpret_cast<uintptr_t>(ctx->acl_ctx);
#else
    (void)handle;
    return 0;
#endif
}

} // extern "C"

#else // !HIXL_HEADERS_AVAILABLE

// ============================================================================
// Stub bridge — HIXL headers NOT present, all functions return NOT_INITIALIZED
// ============================================================================

extern "C" {

HixlHandle HixlCreate(void) { return nullptr; }
void HixlDestroy(HixlHandle) {}
HixlStatus HixlInitialize(HixlHandle, const char *, const HixlOption *, size_t) { return HIXL_NOT_INITIALIZED; }
void HixlFinalize(HixlHandle) {}
HixlStatus HixlRegisterMem(HixlHandle, uintptr_t, size_t, HixlMemType, HixlMemHandle *) { return HIXL_NOT_INITIALIZED; }
HixlStatus HixlDeregisterMem(HixlHandle, HixlMemHandle) { return HIXL_NOT_INITIALIZED; }
HixlStatus HixlConnect(HixlHandle, const char *, int32_t) { return HIXL_NOT_INITIALIZED; }
HixlStatus HixlDisconnect(HixlHandle, const char *, int32_t) { return HIXL_NOT_INITIALIZED; }
HixlStatus HixlTransferSync(HixlHandle, const char *, HixlTransferOp, const HixlTransferOpDesc *, size_t, int32_t) { return HIXL_NOT_INITIALIZED; }
HixlStatus HixlTransferAsync(HixlHandle, const char *, HixlTransferOp, const HixlTransferOpDesc *, size_t, HixlTransferReq *) { return HIXL_NOT_INITIALIZED; }
HixlStatus HixlGetTransferStatus(HixlHandle, HixlTransferReq, HixlTransferStatus *) { return HIXL_NOT_INITIALIZED; }
HixlStatus HixlSendNotify(HixlHandle, const char *, const char *, const char *, int32_t) { return HIXL_NOT_INITIALIZED; }
HixlStatus HixlGetNotifies(HixlHandle, void (*)(const char *, const char *, void *), void *) { return HIXL_NOT_INITIALIZED; }

const char *HixlGetStatusString(HixlStatus status) {
    switch (status) {
        case HIXL_SUCCESS:            return "HIXL_SUCCESS";
        case HIXL_PARAM_INVALID:      return "HIXL_PARAM_INVALID";
        case HIXL_TIMEOUT:            return "HIXL_TIMEOUT";
        case HIXL_NOT_CONNECTED:      return "HIXL_NOT_CONNECTED";
        case HIXL_ALREADY_CONNECTED:  return "HIXL_ALREADY_CONNECTED";
        case HIXL_NOTIFY_FAILED:      return "HIXL_NOTIFY_FAILED";
        case HIXL_UNSUPPORTED:        return "HIXL_UNSUPPORTED";
        case HIXL_FAILED:             return "HIXL_FAILED";
        case HIXL_RESOURCE_EXHAUSTED: return "HIXL_RESOURCE_EXHAUSTED";
        case HIXL_NOT_INITIALIZED:    return "HIXL_NOT_INITIALIZED";
        default:                      return "HIXL_UNKNOWN_ERROR";
    }
}

int32_t HixlSetAclDevice(int32_t) { return -3; }
int32_t HixlGetAclDevice(void) { return -3; }
uintptr_t HixlGetSavedAclContext(HixlHandle) { return 0; }

} // extern "C"

#endif // HIXL_HEADERS_AVAILABLE
