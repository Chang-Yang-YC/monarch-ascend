/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "bridge.h"
#include <dlfcn.h>
#include <iostream>
#include <cstring>

// Mirrors HcclCommConfig from hccl_types.h with proper initialization
// from HcclCommConfigInit in hccl_comm.h.
static void bridge_init_hccl_comm_config(HcclCommConfig *config) {
  if (!config) return;
  memset(config, 0, sizeof(HcclCommConfig));
  // The reserved[24] field contains: size(8) + magicWord(4) + version(4) + reserved(8)
  struct ConfigHeader {
    size_t size;
    uint32_t magicWord;
    uint32_t version;
    uint64_t reserved;
  };
  auto *hdr = reinterpret_cast<ConfigHeader *>(config->reserved);
  hdr->size = sizeof(HcclCommConfig);
  hdr->magicWord = 0xf0f0f0f0;  // HCCL_COMM_CONFIG_MAGIC_WORD
  hdr->version = 9;              // HCCL_COMM_CONFIG_VERSION
  hdr->reserved = 0;
  config->hcclBufferSize = 0xffffffff;      // NOT_SET
  config->hcclDeterministic = 0xffffffff;   // NOT_SET
  config->hcclOpExpansionMode = 0;          // default
  config->hcclRdmaTrafficClass = 0xffffffff; // NOT_SET
  config->hcclRdmaServiceLevel = 0xffffffff; // NOT_SET
  config->hcclExecTimeOut = static_cast<int32_t>(0xffffffff); // NOT_SET
}

namespace hccl_sys {

struct HcclAPI {
  // Root info and error
  HcclResult (*HcclGetRootInfo_)(HcclRootInfo *);
  const char *(*HcclGetErrorString_)(HcclResult);

  // Communicator
  HcclResult (*HcclCommInitRootInfo_)(uint32_t, const HcclRootInfo *, uint32_t,
                                      HcclComm *);
  HcclResult (*HcclCommInitAll_)(uint32_t, int32_t *, HcclComm *);
  HcclResult (*HcclCommDestroy_)(HcclComm);
  HcclResult (*HcclGetCommAsyncError_)(HcclComm, HcclResult *);

  // Sub-communicator
  HcclResult (*HcclCreateSubCommConfig_)(HcclComm *, uint32_t, uint32_t *,
                                         uint64_t, uint32_t, void *,
                                         HcclComm *);

  // Collectives
  HcclResult (*HcclAllReduce_)(void *, void *, uint64_t, HcclDataType,
                               HcclReduceOp, HcclComm, aclrtStream);
  HcclResult (*HcclBroadcast_)(void *, uint64_t, HcclDataType, uint32_t,
                               HcclComm, aclrtStream);
  HcclResult (*HcclReduce_)(void *, void *, uint64_t, HcclDataType,
                            HcclReduceOp, uint32_t, HcclComm, aclrtStream);
  HcclResult (*HcclAllGather_)(void *, void *, uint64_t, HcclDataType,
                               HcclComm, aclrtStream);
  HcclResult (*HcclReduceScatter_)(void *, void *, uint64_t, HcclDataType,
                                   HcclReduceOp, HcclComm, aclrtStream);
  HcclResult (*HcclAlltoAll_)(const void *, uint64_t, HcclDataType,
                              const void *, uint64_t, HcclDataType, HcclComm,
                              aclrtStream);

  // Point to point
  HcclResult (*HcclSend_)(void *, uint64_t, HcclDataType, uint32_t, HcclComm,
                          aclrtStream);
  HcclResult (*HcclRecv_)(void *, uint64_t, HcclDataType, uint32_t, HcclComm,
                          aclrtStream);
  HcclResult (*HcclBatchSendRecv_)(HcclSendRecvItem *, uint32_t, HcclComm,
                                   aclrtStream);

  // Barrier
  HcclResult (*HcclBarrier_)(HcclComm, aclrtStream);

  HcclResult init_result_;
  static HcclAPI *get();
};

// ACL runtime function pointers
struct AclAPI {
  aclError (*aclrtSetDevice_)(int32_t);
  aclError (*aclrtCreateStream_)(aclrtStream *);
  aclError (*aclrtDestroyStream_)(aclrtStream);
  aclError (*aclrtSynchronizeStream_)(aclrtStream);
  aclError (*aclrtCreateEvent_)(aclrtEvent *);
  aclError (*aclrtDestroyEvent_)(aclrtEvent);
  aclError (*aclrtRecordEvent_)(aclrtEvent, aclrtStream);
  aclError (*aclrtSynchronizeEvent_)(aclrtEvent);
  aclError (*aclrtStreamWaitEvent_)(aclrtStream, aclrtEvent);

  bool initialized_;
  static AclAPI *get();
};

namespace {

HcclAPI create_hccl_api() {
  HcclAPI r{};
  r.init_result_ = HCCL_SUCCESS;

  void *handle = dlopen("libhccl.so", RTLD_LAZY | RTLD_NOLOAD);
  if (!handle) {
    handle = dlopen("libhccl.so", RTLD_LAZY);
  }

  if (!handle) {
    std::cerr << "[HCCL-SYS] Warning: Can't open libhccl.so: " << dlerror()
              << std::endl;
    r.init_result_ = HCCL_E_INTERNAL;
    return r;
  }

#define LOOKUP_HCCL_ENTRY(name)                                              \
  r.name##_ = reinterpret_cast<decltype(r.name##_)>(dlsym(handle, #name));   \
  if (!r.name##_) {                                                          \
    std::cerr << "[HCCL-SYS] Warning: Can't find " << #name << ": "          \
              << dlerror() << std::endl;                                     \
    r.init_result_ = HCCL_E_INTERNAL;                                        \
    return r;                                                                \
  }

  LOOKUP_HCCL_ENTRY(HcclGetRootInfo)
  LOOKUP_HCCL_ENTRY(HcclGetErrorString)
  LOOKUP_HCCL_ENTRY(HcclCommInitRootInfo)
  LOOKUP_HCCL_ENTRY(HcclCommInitAll)
  LOOKUP_HCCL_ENTRY(HcclCommDestroy)
  LOOKUP_HCCL_ENTRY(HcclGetCommAsyncError)
  LOOKUP_HCCL_ENTRY(HcclCreateSubCommConfig)
  LOOKUP_HCCL_ENTRY(HcclAllReduce)
  LOOKUP_HCCL_ENTRY(HcclBroadcast)
  LOOKUP_HCCL_ENTRY(HcclReduce)
  LOOKUP_HCCL_ENTRY(HcclAllGather)
  LOOKUP_HCCL_ENTRY(HcclReduceScatter)
  LOOKUP_HCCL_ENTRY(HcclAlltoAll)
  LOOKUP_HCCL_ENTRY(HcclSend)
  LOOKUP_HCCL_ENTRY(HcclRecv)
  LOOKUP_HCCL_ENTRY(HcclBatchSendRecv)
  LOOKUP_HCCL_ENTRY(HcclBarrier)
#undef LOOKUP_HCCL_ENTRY

  return r;
}

AclAPI create_acl_api() {
  AclAPI r{};
  r.initialized_ = false;

  void *handle = dlopen("libascendcl.so", RTLD_LAZY | RTLD_NOLOAD);
  if (!handle) {
    handle = dlopen("libascendcl.so", RTLD_LAZY);
  }

  if (!handle) {
    std::cerr << "[HCCL-SYS] Warning: Can't open libascendcl.so: " << dlerror()
              << std::endl;
    return r;
  }

#define LOOKUP_ACL_ENTRY(name)                                               \
  r.name##_ = reinterpret_cast<decltype(r.name##_)>(dlsym(handle, #name));   \
  if (!r.name##_) {                                                          \
    std::cerr << "[HCCL-SYS] Warning: Can't find " << #name << ": "          \
              << dlerror() << std::endl;                                     \
    return r;                                                                \
  }

  LOOKUP_ACL_ENTRY(aclrtSetDevice)
  LOOKUP_ACL_ENTRY(aclrtCreateStream)
  LOOKUP_ACL_ENTRY(aclrtDestroyStream)
  LOOKUP_ACL_ENTRY(aclrtSynchronizeStream)
  LOOKUP_ACL_ENTRY(aclrtCreateEvent)
  LOOKUP_ACL_ENTRY(aclrtDestroyEvent)
  LOOKUP_ACL_ENTRY(aclrtRecordEvent)
  LOOKUP_ACL_ENTRY(aclrtSynchronizeEvent)
  LOOKUP_ACL_ENTRY(aclrtStreamWaitEvent)
#undef LOOKUP_ACL_ENTRY

  r.initialized_ = true;
  return r;
}

} // namespace

HcclAPI *HcclAPI::get() {
  static HcclAPI singleton = create_hccl_api();
  return &singleton;
}

AclAPI *AclAPI::get() {
  static AclAPI singleton = create_acl_api();
  return &singleton;
}

} // namespace hccl_sys

#define GET_HCCL_API(api_ptr)                               \
  hccl_sys::HcclAPI *api_ptr = hccl_sys::HcclAPI::get();    \
  if (api_ptr->init_result_ != HCCL_SUCCESS) {              \
    return api_ptr->init_result_;                           \
  }

#define GET_HCCL_API_STR(api_ptr)                           \
  hccl_sys::HcclAPI *api_ptr = hccl_sys::HcclAPI::get();    \
  if (api_ptr->init_result_ != HCCL_SUCCESS) {              \
    return "[HCCL-SYS] HCCL library not initialized";       \
  }

#define GET_ACL_API(api_ptr)                                \
  hccl_sys::AclAPI *api_ptr = hccl_sys::AclAPI::get();      \
  if (!api_ptr->initialized_) {                             \
    return -1;                                              \
  }

extern "C" {

// ============================================================================
// HCCL wrappers
// ============================================================================

HcclResult HcclGetRootInfo(HcclRootInfo *rootInfo) {
  GET_HCCL_API(api);
  return api->HcclGetRootInfo_(rootInfo);
}

const char *HcclGetErrorString(HcclResult code) {
  GET_HCCL_API_STR(api);
  return api->HcclGetErrorString_(code);
}

HcclResult HcclCommInitRootInfo(uint32_t nRanks, const HcclRootInfo *rootInfo,
                                uint32_t rank, HcclComm *comm) {
  GET_HCCL_API(api);
  return api->HcclCommInitRootInfo_(nRanks, rootInfo, rank, comm);
}

HcclResult HcclCommInitAll(uint32_t ndev, int32_t *devices, HcclComm *comms) {
  GET_HCCL_API(api);
  return api->HcclCommInitAll_(ndev, devices, comms);
}

HcclResult HcclCommDestroy(HcclComm comm) {
  GET_HCCL_API(api);
  return api->HcclCommDestroy_(comm);
}

HcclResult HcclGetCommAsyncError(HcclComm comm, HcclResult *asyncError) {
  GET_HCCL_API(api);
  return api->HcclGetCommAsyncError_(comm, asyncError);
}

HcclResult HcclCreateSubCommConfig(HcclComm *comm, uint32_t rankNum,
                                   uint32_t *rankIds, uint64_t subCommId,
                                   uint32_t subCommRankId, void *config,
                                   HcclComm *subComm) {
  GET_HCCL_API(api);
  HcclCommConfig default_config;
  void *actual_config;
  if (config) {
    actual_config = config;
  } else {
    bridge_init_hccl_comm_config(&default_config);
    actual_config = &default_config;
  }
  return api->HcclCreateSubCommConfig_(comm, rankNum, rankIds, subCommId,
                                       subCommRankId, actual_config, subComm);
}

HcclResult HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count,
                         HcclDataType dataType, HcclReduceOp op,
                         HcclComm comm, aclrtStream stream) {
  GET_HCCL_API(api);
  return api->HcclAllReduce_(sendBuf, recvBuf, count, dataType, op, comm,
                             stream);
}

HcclResult HcclBroadcast(void *buf, uint64_t count, HcclDataType dataType,
                         uint32_t root, HcclComm comm, aclrtStream stream) {
  GET_HCCL_API(api);
  return api->HcclBroadcast_(buf, count, dataType, root, comm, stream);
}

HcclResult HcclReduce(void *sendBuf, void *recvBuf, uint64_t count,
                      HcclDataType dataType, HcclReduceOp op, uint32_t root,
                      HcclComm comm, aclrtStream stream) {
  GET_HCCL_API(api);
  return api->HcclReduce_(sendBuf, recvBuf, count, dataType, op, root, comm,
                          stream);
}

HcclResult HcclAllGather(void *sendBuf, void *recvBuf, uint64_t sendCount,
                         HcclDataType dataType, HcclComm comm,
                         aclrtStream stream) {
  GET_HCCL_API(api);
  return api->HcclAllGather_(sendBuf, recvBuf, sendCount, dataType, comm,
                             stream);
}

HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount,
                             HcclDataType dataType, HcclReduceOp op,
                             HcclComm comm, aclrtStream stream) {
  GET_HCCL_API(api);
  return api->HcclReduceScatter_(sendBuf, recvBuf, recvCount, dataType, op,
                                 comm, stream);
}

HcclResult HcclAlltoAll(const void *sendBuf, uint64_t sendCount,
                        HcclDataType sendType, const void *recvBuf,
                        uint64_t recvCount, HcclDataType recvType,
                        HcclComm comm, aclrtStream stream) {
  GET_HCCL_API(api);
  return api->HcclAlltoAll_(sendBuf, sendCount, sendType, recvBuf, recvCount,
                            recvType, comm, stream);
}

HcclResult HcclSend(void *sendBuf, uint64_t count, HcclDataType dataType,
                    uint32_t destRank, HcclComm comm, aclrtStream stream) {
  GET_HCCL_API(api);
  return api->HcclSend_(sendBuf, count, dataType, destRank, comm, stream);
}

HcclResult HcclRecv(void *recvBuf, uint64_t count, HcclDataType dataType,
                    uint32_t srcRank, HcclComm comm, aclrtStream stream) {
  GET_HCCL_API(api);
  return api->HcclRecv_(recvBuf, count, dataType, srcRank, comm, stream);
}

HcclResult HcclBatchSendRecv(HcclSendRecvItem *sendRecvInfo, uint32_t itemNum,
                             HcclComm comm, aclrtStream stream) {
  GET_HCCL_API(api);
  return api->HcclBatchSendRecv_(sendRecvInfo, itemNum, comm, stream);
}

HcclResult HcclBarrier(HcclComm comm, aclrtStream stream) {
  GET_HCCL_API(api);
  return api->HcclBarrier_(comm, stream);
}

// ============================================================================
// ACL runtime wrappers
// ============================================================================

aclError aclrtSetDevice(int32_t deviceId) {
  GET_ACL_API(api);
  return api->aclrtSetDevice_(deviceId);
}

aclError aclrtCreateStream(aclrtStream *stream) {
  GET_ACL_API(api);
  return api->aclrtCreateStream_(stream);
}

aclError aclrtDestroyStream(aclrtStream stream) {
  GET_ACL_API(api);
  return api->aclrtDestroyStream_(stream);
}

aclError aclrtSynchronizeStream(aclrtStream stream) {
  GET_ACL_API(api);
  return api->aclrtSynchronizeStream_(stream);
}

aclError aclrtCreateEvent(aclrtEvent *event) {
  GET_ACL_API(api);
  return api->aclrtCreateEvent_(event);
}

aclError aclrtDestroyEvent(aclrtEvent event) {
  GET_ACL_API(api);
  return api->aclrtDestroyEvent_(event);
}

aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream) {
  GET_ACL_API(api);
  return api->aclrtRecordEvent_(event, stream);
}

aclError aclrtSynchronizeEvent(aclrtEvent event) {
  GET_ACL_API(api);
  return api->aclrtSynchronizeEvent_(event);
}

aclError aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event) {
  GET_ACL_API(api);
  return api->aclrtStreamWaitEvent_(stream, event);
}

} // extern "C"
