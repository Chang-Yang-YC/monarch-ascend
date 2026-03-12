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

#include "bridge.h"
#include <dlfcn.h>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <map>
#include <vector>
#include <unistd.h>

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

extern "C" {

HixlHandle HixlCreate(void) {
    try {
        return static_cast<HixlHandle>(new hixl::Hixl());
    } catch (const std::exception &e) {
        std::cerr << "[HIXL-SYS] HixlCreate failed: " << e.what() << std::endl;
        return nullptr;
    }
}

void HixlDestroy(HixlHandle handle) {
    delete static_cast<hixl::Hixl *>(handle);
}

// Ensure ACL runtime is initialized with a device context.
// HIXL requires the ACL device context to be active.
//
// This function respects the device set by torch_npu/torch.npu.set_device()
// and falls back to MONARCH_NPU_DEVICE or ASCEND_RT_VISIBLE_DEVICES env vars.
static bool ensure_acl_device() {
#if ACL_AVAILABLE
    static thread_local bool initialized = false;
    if (initialized) return true;

    // torch_npu already initialized ACL; skip aclInit to avoid state corruption.
    // We only need to ensure the correct device is set for this thread.

    // First, check if a device is already set (by torch_npu)
    int32_t current_dev = -1;
    aclError ret = aclrtGetDevice(&current_dev);
    if (ret == 0 && current_dev >= 0) {
        // Device already set by Python/torch_npu
        std::cerr << "[HIXL-SYS] ensure_acl_device: using existing device " << current_dev
                  << " (pid=" << getpid() << ")" << std::endl;
        initialized = true;
        return true;
    }

    // No device set, determine from environment variables
    int target_dev = 0;
    const char* npu_dev = getenv("MONARCH_NPU_DEVICE");
    if (npu_dev) {
        target_dev = atoi(npu_dev);
    } else {
        const char* vis = getenv("ASCEND_RT_VISIBLE_DEVICES");
        if (vis) {
            target_dev = atoi(vis);
        }
    }

    ret = aclrtSetDevice(target_dev);
    if (ret != 0) {
        std::cerr << "[HIXL-SYS] aclrtSetDevice(" << target_dev << ") failed: " << ret << std::endl;
        return false;
    }

    int32_t final_dev = -1;
    aclrtGetDevice(&final_dev);
    std::cerr << "[HIXL-SYS] ensure_acl_device: using device " << final_dev
              << " (pid=" << getpid() << ")" << std::endl;
    initialized = true;
    return true;
#else
    return true;
#endif
}

HixlStatus HixlInitialize(HixlHandle handle,
                           const char *local_engine,
                           const HixlOption *options,
                           size_t num_options) {
    if (!handle || !local_engine) return HIXL_PARAM_INVALID;

    if (!ensure_acl_device()) {
        std::cerr << "[HIXL-SYS] ACL device setup failed, HIXL Initialize may fail" << std::endl;
    }

    auto *h = static_cast<hixl::Hixl *>(handle);
    std::map<hixl::AscendString, hixl::AscendString> opts;
    for (size_t i = 0; i < num_options; ++i) {
        if (options[i].key && options[i].value)
            opts[hixl::AscendString(options[i].key)] =
                hixl::AscendString(options[i].value);
    }
    return h->Initialize(hixl::AscendString(local_engine), opts);
}

void HixlFinalize(HixlHandle handle) {
    if (handle) static_cast<hixl::Hixl *>(handle)->Finalize();
}

HixlStatus HixlRegisterMem(HixlHandle handle, uintptr_t addr, size_t len,
                            HixlMemType mem_type, HixlMemHandle *out) {
    if (!handle || !out) return HIXL_PARAM_INVALID;
    hixl::MemDesc mem{}; mem.addr = addr; mem.len = len;
    hixl::MemHandle mh = nullptr;
    auto s = static_cast<hixl::Hixl *>(handle)->RegisterMem(
        mem, to_mem_type(mem_type), mh);
    *out = mh;
    return s;
}

HixlStatus HixlDeregisterMem(HixlHandle handle, HixlMemHandle mh) {
    if (!handle) return HIXL_PARAM_INVALID;
    return static_cast<hixl::Hixl *>(handle)->DeregisterMem(mh);
}

HixlStatus HixlConnect(HixlHandle handle, const char *remote, int32_t timeout) {
    if (!handle || !remote) return HIXL_PARAM_INVALID;
    return static_cast<hixl::Hixl *>(handle)->Connect(
        hixl::AscendString(remote), timeout);
}

HixlStatus HixlDisconnect(HixlHandle handle, const char *remote, int32_t timeout) {
    if (!handle || !remote) return HIXL_PARAM_INVALID;
    return static_cast<hixl::Hixl *>(handle)->Disconnect(
        hixl::AscendString(remote), timeout);
}

HixlStatus HixlTransferSync(HixlHandle handle, const char *remote,
                             HixlTransferOp op, const HixlTransferOpDesc *descs,
                             size_t n, int32_t timeout) {
    if (!handle || !remote) return HIXL_PARAM_INVALID;
    if (!ensure_acl_device()) {
        std::cerr << "[HIXL-SYS] TransferSync: ACL device setup failed" << std::endl;
    }

    // Debug output
    std::cerr << "[HIXL-SYS] TransferSync handle=" << handle
              << " remote='" << remote << "'"
              << " op=" << (int)op << " n=" << n
              << " timeout=" << timeout << " pid=" << getpid() << std::endl;
    for (size_t i = 0; i < n; ++i) {
        std::cerr << "[HIXL-SYS]   [" << i << "] local=" << (void*)descs[i].local_addr
                  << " remote=" << (void*)descs[i].remote_addr
                  << " len=" << descs[i].len << std::endl;
    }
    std::cerr.flush();

    std::vector<hixl::TransferOpDesc> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i].local_addr = descs[i].local_addr;
        v[i].remote_addr = descs[i].remote_addr;
        v[i].len = descs[i].len;
    }
    auto s = static_cast<hixl::Hixl *>(handle)->TransferSync(
        hixl::AscendString(remote), to_transfer_op(op), v, timeout);
    
    const char* err = aclGetRecentErrMsg();
    std::cerr << "[HIXL-SYS] TransferSync result=" << s 
              << " err=" << (err ? err : "none") << std::endl;
    return s;
}

HixlStatus HixlTransferAsync(HixlHandle handle, const char *remote,
                              HixlTransferOp op, const HixlTransferOpDesc *descs,
                              size_t n, HixlTransferReq *out) {
    if (!handle || !remote || !out) return HIXL_PARAM_INVALID;
    std::vector<hixl::TransferOpDesc> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i].local_addr = descs[i].local_addr;
        v[i].remote_addr = descs[i].remote_addr;
        v[i].len = descs[i].len;
    }
    hixl::TransferArgs args{};
    hixl::TransferReq req = nullptr;
    auto s = static_cast<hixl::Hixl *>(handle)->TransferAsync(
        hixl::AscendString(remote), to_transfer_op(op), v, args, req);
    *out = req;
    return s;
}

HixlStatus HixlGetTransferStatus(HixlHandle handle, HixlTransferReq req,
                                  HixlTransferStatus *out) {
    if (!handle || !out) return HIXL_PARAM_INVALID;
    hixl::TransferStatus st;
    auto s = static_cast<hixl::Hixl *>(handle)->GetTransferStatus(req, st);
    *out = from_transfer_status(st);
    return s;
}

HixlStatus HixlSendNotify(HixlHandle handle, const char *remote,
                           const char *name, const char *msg, int32_t timeout) {
    if (!handle || !remote) return HIXL_PARAM_INVALID;
    hixl::NotifyDesc nd;
    nd.name = hixl::AscendString(name ? name : "");
    nd.notify_msg = hixl::AscendString(msg ? msg : "");
    return static_cast<hixl::Hixl *>(handle)->SendNotify(
        hixl::AscendString(remote), nd, timeout);
}

HixlStatus HixlGetNotifies(HixlHandle handle,
                            void (*cb)(const char *, const char *, void *),
                            void *ud) {
    if (!handle || !cb) return HIXL_PARAM_INVALID;
    std::vector<hixl::NotifyDesc> notifies;
    auto s = static_cast<hixl::Hixl *>(handle)->GetNotifies(notifies);
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

} // extern "C"

#endif // HIXL_HEADERS_AVAILABLE
