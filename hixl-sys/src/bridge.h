/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// HIXL status codes (mirrors hixl_types.h)
// ============================================================================
typedef uint32_t HixlStatus;

#define HIXL_SUCCESS           0U
#define HIXL_PARAM_INVALID     103900U
#define HIXL_TIMEOUT           103901U
#define HIXL_NOT_CONNECTED     103902U
#define HIXL_ALREADY_CONNECTED 103903U
#define HIXL_NOTIFY_FAILED     103904U
#define HIXL_UNSUPPORTED       103905U
#define HIXL_FAILED            503900U
#define HIXL_RESOURCE_EXHAUSTED 203900U
#define HIXL_NOT_INITIALIZED   999999U

// ============================================================================
// HIXL enumerations
// ============================================================================
typedef enum {
    HIXL_MEM_DEVICE = 0,
    HIXL_MEM_HOST = 1
} HixlMemType;

typedef enum {
    HIXL_READ = 0,
    HIXL_WRITE = 1
} HixlTransferOp;

typedef enum {
    HIXL_TRANSFER_WAITING = 0,
    HIXL_TRANSFER_COMPLETED = 1,
    HIXL_TRANSFER_TIMEOUT = 2,
    HIXL_TRANSFER_FAILED = 3
} HixlTransferStatus;

// ============================================================================
// HIXL structures
// ============================================================================
typedef struct {
    uintptr_t addr;
    size_t len;
    uint8_t reserved[128];
} HixlMemDesc;

typedef struct {
    uintptr_t local_addr;
    uintptr_t remote_addr;
    size_t len;
} HixlTransferOpDesc;

typedef void *HixlHandle;
typedef void *HixlMemHandle;
typedef void *HixlTransferReq;

// ============================================================================
// Option key-value pair for initialization
// ============================================================================
typedef struct {
    const char *key;
    const char *value;
} HixlOption;

// ============================================================================
// HIXL C API wrapper functions
// ============================================================================

HixlHandle HixlCreate(void);

void HixlDestroy(HixlHandle handle);

HixlStatus HixlInitialize(HixlHandle handle,
                           const char *local_engine,
                           const HixlOption *options,
                           size_t num_options);

void HixlFinalize(HixlHandle handle);

HixlStatus HixlRegisterMem(HixlHandle handle,
                            uintptr_t addr,
                            size_t len,
                            HixlMemType mem_type,
                            HixlMemHandle *out_mem_handle);

HixlStatus HixlDeregisterMem(HixlHandle handle, HixlMemHandle mem_handle);

HixlStatus HixlConnect(HixlHandle handle,
                        const char *remote_engine,
                        int32_t timeout_ms);

HixlStatus HixlDisconnect(HixlHandle handle,
                           const char *remote_engine,
                           int32_t timeout_ms);

HixlStatus HixlTransferSync(HixlHandle handle,
                             const char *remote_engine,
                             HixlTransferOp operation,
                             const HixlTransferOpDesc *op_descs,
                             size_t num_descs,
                             int32_t timeout_ms);

HixlStatus HixlTransferWrite(HixlHandle handle,
                              const char *remote_engine,
                              uintptr_t local_addr,
                              uintptr_t remote_addr,
                              size_t len,
                              int32_t timeout_ms);

HixlStatus HixlTransferRead(HixlHandle handle,
                             const char *remote_engine,
                             uintptr_t local_addr,
                             uintptr_t remote_addr,
                             size_t len,
                             int32_t timeout_ms);

HixlStatus HixlTransferAsync(HixlHandle handle,
                              const char *remote_engine,
                              HixlTransferOp operation,
                              const HixlTransferOpDesc *op_descs,
                              size_t num_descs,
                              HixlTransferReq *out_req);

HixlStatus HixlGetTransferStatus(HixlHandle handle,
                                  HixlTransferReq req,
                                  HixlTransferStatus *out_status);

HixlStatus HixlSendNotify(HixlHandle handle,
                           const char *remote_engine,
                           const char *name,
                           const char *msg,
                           int32_t timeout_ms);

HixlStatus HixlGetNotifies(HixlHandle handle,
                            void (*callback)(const char *name,
                                             const char *msg,
                                             void *user_data),
                            void *user_data);

const char *HixlGetStatusString(HixlStatus status);

// ============================================================================
// ACL Device Management (for device context synchronization)
// ============================================================================

/// Set the ACL device for the current thread.
/// Returns the device ID on success, -2 on ACL error, -3 if ACL not available.
int32_t HixlSetAclDevice(int32_t device_id);

/// Get the current ACL device for the current thread.
/// Returns -1 if no device is set, -3 if ACL not available.
int32_t HixlGetAclDevice(void);

/// Get the saved ACL context from a HixlHandle (for debugging).
/// Returns 0 if handle is null or ACL not available.
uintptr_t HixlGetSavedAclContext(HixlHandle handle);

#ifdef __cplusplus
}
#endif
