/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//
// C-API shim for HIXL. Compile this into libhixl_c_api.so:
//
//   g++ -std=c++17 -shared -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 \
//       -I$HIXL_HOME/include \
//       -I$ASCEND_HOME/compiler/include \
//       -o libhixl_c_api.so hixl_c_api.cpp \
//       -L$HIXL_HOME/lib -lcann_hixl -Wl,-rpath,$HIXL_HOME/lib
//
// Then place libhixl_c_api.so in LD_LIBRARY_PATH.

#include <hixl/hixl.h>
#include <hixl/hixl_types.h>
#include <map>
#include <vector>
#include <iostream>
#include <cstdint>
#include <cstddef>

// ============================================================================
// C types (must match bridge.h exactly)
// ============================================================================
typedef uint32_t HixlStatus;
typedef void *HixlHandle;
typedef void *HixlMemHandle;
typedef void *HixlTransferReq;

#define HIXL_NOT_INITIALIZED 999999U

typedef enum { HIXL_MEM_DEVICE = 0, HIXL_MEM_HOST = 1 } HixlMemType;
typedef enum { HIXL_READ = 0, HIXL_WRITE = 1 } HixlTransferOp;
typedef enum {
    HIXL_TRANSFER_WAITING = 0,
    HIXL_TRANSFER_COMPLETED = 1,
    HIXL_TRANSFER_TIMEOUT = 2,
    HIXL_TRANSFER_FAILED = 3
} HixlTransferStatus;

typedef struct { const char *key; const char *value; } HixlOption;

typedef struct {
    uintptr_t local_addr;
    uintptr_t remote_addr;
    size_t len;
} HixlTransferOpDesc;

// ============================================================================
// Helper conversions
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
// C API implementation
// ============================================================================
extern "C" {

__attribute__((visibility("default")))
HixlHandle HixlCreate(void) {
    try {
        return static_cast<HixlHandle>(new hixl::Hixl());
    } catch (const std::exception &e) {
        std::cerr << "[hixl_c_api] HixlCreate failed: " << e.what() << std::endl;
        return nullptr;
    }
}

__attribute__((visibility("default")))
void HixlDestroy(HixlHandle handle) {
    delete static_cast<hixl::Hixl *>(handle);
}

__attribute__((visibility("default")))
HixlStatus HixlInitialize(HixlHandle handle,
                           const char *local_engine,
                           const HixlOption *options,
                           size_t num_options) {
    if (!handle || !local_engine) return 103900U; // PARAM_INVALID
    auto *h = static_cast<hixl::Hixl *>(handle);
    std::map<hixl::AscendString, hixl::AscendString> opts;
    for (size_t i = 0; i < num_options; ++i) {
        if (options[i].key && options[i].value)
            opts[hixl::AscendString(options[i].key)] = hixl::AscendString(options[i].value);
    }
    return h->Initialize(hixl::AscendString(local_engine), opts);
}

__attribute__((visibility("default")))
void HixlFinalize(HixlHandle handle) {
    if (handle) static_cast<hixl::Hixl *>(handle)->Finalize();
}

__attribute__((visibility("default")))
HixlStatus HixlRegisterMem(HixlHandle handle, uintptr_t addr, size_t len,
                            HixlMemType mem_type, HixlMemHandle *out) {
    if (!handle || !out) return 103900U;
    auto *h = static_cast<hixl::Hixl *>(handle);
    hixl::MemDesc mem{}; mem.addr = addr; mem.len = len;
    hixl::MemHandle mh = nullptr;
    auto s = h->RegisterMem(mem, to_mem_type(mem_type), mh);
    *out = mh;
    return s;
}

__attribute__((visibility("default")))
HixlStatus HixlDeregisterMem(HixlHandle handle, HixlMemHandle mh) {
    if (!handle) return 103900U;
    return static_cast<hixl::Hixl *>(handle)->DeregisterMem(mh);
}

__attribute__((visibility("default")))
HixlStatus HixlConnect(HixlHandle handle, const char *remote, int32_t timeout) {
    if (!handle || !remote) return 103900U;
    return static_cast<hixl::Hixl *>(handle)->Connect(hixl::AscendString(remote), timeout);
}

__attribute__((visibility("default")))
HixlStatus HixlDisconnect(HixlHandle handle, const char *remote, int32_t timeout) {
    if (!handle || !remote) return 103900U;
    return static_cast<hixl::Hixl *>(handle)->Disconnect(hixl::AscendString(remote), timeout);
}

__attribute__((visibility("default")))
HixlStatus HixlTransferSync(HixlHandle handle, const char *remote,
                             HixlTransferOp op, const HixlTransferOpDesc *descs,
                             size_t n, int32_t timeout) {
    if (!handle || !remote) return 103900U;
    auto *h = static_cast<hixl::Hixl *>(handle);
    std::vector<hixl::TransferOpDesc> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i].local_addr = descs[i].local_addr;
        v[i].remote_addr = descs[i].remote_addr;
        v[i].len = descs[i].len;
    }
    return h->TransferSync(hixl::AscendString(remote), to_transfer_op(op), v, timeout);
}

__attribute__((visibility("default")))
HixlStatus HixlTransferAsync(HixlHandle handle, const char *remote,
                              HixlTransferOp op, const HixlTransferOpDesc *descs,
                              size_t n, HixlTransferReq *out) {
    if (!handle || !remote || !out) return 103900U;
    auto *h = static_cast<hixl::Hixl *>(handle);
    std::vector<hixl::TransferOpDesc> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i].local_addr = descs[i].local_addr;
        v[i].remote_addr = descs[i].remote_addr;
        v[i].len = descs[i].len;
    }
    hixl::TransferArgs args{};
    hixl::TransferReq req = nullptr;
    auto s = h->TransferAsync(hixl::AscendString(remote), to_transfer_op(op), v, args, req);
    *out = req;
    return s;
}

__attribute__((visibility("default")))
HixlStatus HixlGetTransferStatus(HixlHandle handle, HixlTransferReq req,
                                  HixlTransferStatus *out) {
    if (!handle || !out) return 103900U;
    hixl::TransferStatus st;
    auto s = static_cast<hixl::Hixl *>(handle)->GetTransferStatus(req, st);
    *out = from_transfer_status(st);
    return s;
}

__attribute__((visibility("default")))
HixlStatus HixlSendNotify(HixlHandle handle, const char *remote,
                           const char *name, const char *msg, int32_t timeout) {
    if (!handle || !remote) return 103900U;
    hixl::NotifyDesc nd;
    nd.name = hixl::AscendString(name ? name : "");
    nd.notify_msg = hixl::AscendString(msg ? msg : "");
    return static_cast<hixl::Hixl *>(handle)->SendNotify(
        hixl::AscendString(remote), nd, timeout);
}

__attribute__((visibility("default")))
HixlStatus HixlGetNotifies(HixlHandle handle,
                            void (*cb)(const char *, const char *, void *),
                            void *ud) {
    if (!handle || !cb) return 103900U;
    std::vector<hixl::NotifyDesc> notifies;
    auto s = static_cast<hixl::Hixl *>(handle)->GetNotifies(notifies);
    if (s == hixl::SUCCESS) {
        for (const auto &n : notifies)
            cb(n.name.GetString(), n.notify_msg.GetString(), ud);
    }
    return s;
}

} // extern "C"
