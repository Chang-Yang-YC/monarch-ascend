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
// ACL runtime types (subset needed by HCCL)
// ============================================================================
typedef void *aclrtStream;
typedef void *aclrtEvent;
typedef int aclError;

// ============================================================================
// HCCL types mirroring hccl_types.h
// ============================================================================

typedef enum {
    HCCL_SUCCESS = 0,
    HCCL_E_PARA = 1,
    HCCL_E_PTR = 2,
    HCCL_E_MEMORY = 3,
    HCCL_E_INTERNAL = 4,
    HCCL_E_NOT_SUPPORT = 5,
    HCCL_E_NOT_FOUND = 6,
    HCCL_E_UNAVAIL = 7,
    HCCL_E_SYSCALL = 8,
    HCCL_E_TIMEOUT = 9,
    HCCL_E_OPEN_FILE_FAILURE = 10,
    HCCL_E_TCP_CONNECT = 11,
    HCCL_E_ROCE_CONNECT = 12,
    HCCL_E_TCP_TRANSFER = 13,
    HCCL_E_ROCE_TRANSFER = 14,
    HCCL_E_RUNTIME = 15,
    HCCL_E_DRV = 16,
    HCCL_E_PROFILING = 17,
    HCCL_E_CCE = 18,
    HCCL_E_NETWORK = 19,
    HCCL_E_AGAIN = 20,
    HCCL_E_REMOTE = 21,
    HCCL_E_SUSPENDING = 22,
    HCCL_E_RESERVED = 23
} HcclResult;

typedef void *HcclComm;

typedef enum {
    HCCL_REDUCE_SUM = 0,
    HCCL_REDUCE_PROD = 1,
    HCCL_REDUCE_MAX = 2,
    HCCL_REDUCE_MIN = 3,
    HCCL_REDUCE_RESERVED = 255
} HcclReduceOp;

typedef enum {
    HCCL_DATA_TYPE_INT8 = 0,
    HCCL_DATA_TYPE_INT16 = 1,
    HCCL_DATA_TYPE_INT32 = 2,
    HCCL_DATA_TYPE_FP16 = 3,
    HCCL_DATA_TYPE_FP32 = 4,
    HCCL_DATA_TYPE_INT64 = 5,
    HCCL_DATA_TYPE_UINT64 = 6,
    HCCL_DATA_TYPE_UINT8 = 7,
    HCCL_DATA_TYPE_UINT16 = 8,
    HCCL_DATA_TYPE_UINT32 = 9,
    HCCL_DATA_TYPE_FP64 = 10,
    HCCL_DATA_TYPE_BFP16 = 11,
    HCCL_DATA_TYPE_RESERVED = 255
} HcclDataType;

typedef enum {
    HCCL_SEND = 0,
    HCCL_RECV = 1,
    HCCL_SEND_RECV_RESERVED
} HcclSendRecvType;

typedef struct HcclSendRecvItemDef {
    HcclSendRecvType sendRecvType;
    void *buf;
    uint64_t count;
    HcclDataType dataType;
    uint32_t remoteRank;
} HcclSendRecvItem;

#define HCCL_ROOT_INFO_BYTES 4108

typedef struct HcclRootInfoDef {
    char internal[HCCL_ROOT_INFO_BYTES];
} HcclRootInfo;

// ============================================================================
// HCCL C API wrapper functions (loaded via dlopen)
// ============================================================================

HcclResult HcclGetRootInfo(HcclRootInfo *rootInfo);
const char *HcclGetErrorString(HcclResult code);

// Communicator management
HcclResult HcclCommInitRootInfo(uint32_t nRanks, const HcclRootInfo *rootInfo,
                                uint32_t rank, HcclComm *comm);
HcclResult HcclCommInitAll(uint32_t ndev, int32_t *devices, HcclComm *comms);
HcclResult HcclCommDestroy(HcclComm comm);
HcclResult HcclGetCommAsyncError(HcclComm comm, HcclResult *asyncError);

// Sub-communicator creation
HcclResult HcclCreateSubCommConfig(HcclComm *comm, uint32_t rankNum,
                                   uint32_t *rankIds, uint64_t subCommId,
                                   uint32_t subCommRankId,
                                   void *config, HcclComm *subComm);

// Collective communication
HcclResult HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count,
                         HcclDataType dataType, HcclReduceOp op,
                         HcclComm comm, aclrtStream stream);

HcclResult HcclBroadcast(void *buf, uint64_t count, HcclDataType dataType,
                         uint32_t root, HcclComm comm, aclrtStream stream);

HcclResult HcclReduce(void *sendBuf, void *recvBuf, uint64_t count,
                      HcclDataType dataType, HcclReduceOp op,
                      uint32_t root, HcclComm comm, aclrtStream stream);

HcclResult HcclAllGather(void *sendBuf, void *recvBuf, uint64_t sendCount,
                         HcclDataType dataType, HcclComm comm,
                         aclrtStream stream);

HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount,
                             HcclDataType dataType, HcclReduceOp op,
                             HcclComm comm, aclrtStream stream);

HcclResult HcclAlltoAll(const void *sendBuf, uint64_t sendCount,
                        HcclDataType sendType, const void *recvBuf,
                        uint64_t recvCount, HcclDataType recvType,
                        HcclComm comm, aclrtStream stream);

// Point to point communication
HcclResult HcclSend(void *sendBuf, uint64_t count, HcclDataType dataType,
                    uint32_t destRank, HcclComm comm, aclrtStream stream);

HcclResult HcclRecv(void *recvBuf, uint64_t count, HcclDataType dataType,
                    uint32_t srcRank, HcclComm comm, aclrtStream stream);

HcclResult HcclBatchSendRecv(HcclSendRecvItem *sendRecvInfo, uint32_t itemNum,
                             HcclComm comm, aclrtStream stream);

// Barrier
HcclResult HcclBarrier(HcclComm comm, aclrtStream stream);

// ACL runtime helpers (loaded via dlopen from libascendcl.so)
aclError aclrtSetDevice(int32_t deviceId);
aclError aclrtCreateStream(aclrtStream *stream);
aclError aclrtDestroyStream(aclrtStream stream);
aclError aclrtSynchronizeStream(aclrtStream stream);
aclError aclrtCreateEvent(aclrtEvent *event);
aclError aclrtDestroyEvent(aclrtEvent event);
aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream);
aclError aclrtSynchronizeEvent(aclrtEvent event);
aclError aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event);

#ifdef __cplusplus
}
#endif
