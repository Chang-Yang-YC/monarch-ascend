/**
 * Shared library for direct HIXL testing from Python via ctypes.
 * Usage:
 *   lib = ctypes.CDLL("./libtest_hixl.so")
 *   ret = lib.run_server(dev=0, engine_id, addr, size)
 *   ret = lib.run_client_transfer(dev=1, local_engine, remote_engine, local_addr, local_size, remote_addr, remote_size)
 */
#include <hixl/hixl.h>
#include <hixl/hixl_types.h>
#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <thread>
#include <chrono>
#include <unistd.h>
using namespace hixl;

extern "C" {

struct HixlTestCtx {
    Hixl* engine;
    MemHandle mem_handle;
};

uintptr_t get_acl_context() {
    aclrtContext ctx = nullptr;
    aclrtGetCurrentContext(&ctx);
    fprintf(stderr, "[hixl_test] get_acl_context: %p\n", ctx);
    return reinterpret_cast<uintptr_t>(ctx);
}

void set_acl_context(uintptr_t ctx_ptr) {
    aclrtContext ctx = reinterpret_cast<aclrtContext>(ctx_ptr);
    aclError ret = aclrtSetCurrentContext(ctx);
    fprintf(stderr, "[hixl_test] set_acl_context(%p): %d\n", ctx, ret);
}

void* hixl_init_engine(int dev, const char* engine_id) {
    aclrtSetDevice(dev);
    auto* ctx = new HixlTestCtx();
    ctx->engine = new Hixl();
    std::map<AscendString, AscendString> opts;
    opts["BufferPool"] = "0:0";
    auto s = ctx->engine->Initialize(AscendString(engine_id), opts);
    fprintf(stderr, "[hixl_test] Init(%s) dev=%d: %u\n", engine_id, dev, s);
    if (s != SUCCESS) { delete ctx; return nullptr; }
    return ctx;
}

void* hixl_init_engine_with_ctx(uintptr_t acl_ctx_ptr, const char* engine_id) {
    aclrtContext acl_ctx = reinterpret_cast<aclrtContext>(acl_ctx_ptr);
    aclError ret = aclrtSetCurrentContext(acl_ctx);
    fprintf(stderr, "[hixl_test] InitWithCtx(%s) set_ctx=%p ret=%d\n", engine_id, acl_ctx, ret);
    auto* ctx = new HixlTestCtx();
    ctx->engine = new Hixl();
    std::map<AscendString, AscendString> opts;
    opts["BufferPool"] = "0:0";
    auto s = ctx->engine->Initialize(AscendString(engine_id), opts);
    fprintf(stderr, "[hixl_test] InitWithCtx(%s): %u\n", engine_id, s);
    if (s != SUCCESS) { delete ctx; return nullptr; }
    return ctx;
}

int hixl_register_mem(void* ctx_ptr, uintptr_t addr, size_t size) {
    auto* ctx = static_cast<HixlTestCtx*>(ctx_ptr);
    MemDesc d{}; d.addr = addr; d.len = size;
    auto s = ctx->engine->RegisterMem(d, MEM_DEVICE, ctx->mem_handle);
    fprintf(stderr, "[hixl_test] RegMem addr=%p size=%zu: %u\n", (void*)addr, size, s);
    return s;
}

int hixl_connect(void* ctx_ptr, const char* remote_engine_id) {
    auto* ctx = static_cast<HixlTestCtx*>(ctx_ptr);
    auto s = ctx->engine->Connect(AscendString(remote_engine_id), 10000);
    fprintf(stderr, "[hixl_test] Connect(%s): %u pid=%d\n", remote_engine_id, s, getpid());
    if (s != SUCCESS) {
        auto* msg = aclGetRecentErrMsg();
        fprintf(stderr, "[hixl_test] Connect ACL error: %s\n", msg ? msg : "none");
    }
    return s;
}

int hixl_transfer_write(void* ctx_ptr, const char* remote_engine_id,
                         uintptr_t local_addr, uintptr_t remote_addr, size_t len) {
    auto* ctx = static_cast<HixlTestCtx*>(ctx_ptr);
    TransferOpDesc td{};
    td.local_addr = local_addr;
    td.remote_addr = remote_addr;
    td.len = len;
    fprintf(stderr, "[hixl_test] TransferSync(WRITE) local=%p remote=%p len=%zu to=%s\n",
            (void*)local_addr, (void*)remote_addr, len, remote_engine_id);
    auto s = ctx->engine->TransferSync(AscendString(remote_engine_id), WRITE, {td}, 10000);
    fprintf(stderr, "[hixl_test] TransferSync result: %u\n", s);
    if (s != SUCCESS) {
        auto* msg = aclGetRecentErrMsg();
        fprintf(stderr, "[hixl_test] ACL error: %s\n", msg ? msg : "none");
    }
    return s;
}

int hixl_transfer_read(void* ctx_ptr, const char* remote_engine_id,
                        uintptr_t local_addr, uintptr_t remote_addr, size_t len) {
    auto* ctx = static_cast<HixlTestCtx*>(ctx_ptr);
    TransferOpDesc td{};
    td.local_addr = local_addr;
    td.remote_addr = remote_addr;
    td.len = len;
    fprintf(stderr, "[hixl_test] TransferSync(READ) local=%p remote=%p len=%zu from=%s\n",
            (void*)local_addr, (void*)remote_addr, len, remote_engine_id);
    auto s = ctx->engine->TransferSync(AscendString(remote_engine_id), READ, {td}, 10000);
    fprintf(stderr, "[hixl_test] TransferSync READ result: %u\n", s);
    if (s != SUCCESS) {
        auto* msg = aclGetRecentErrMsg();
        fprintf(stderr, "[hixl_test] ACL error: %s\n", msg ? msg : "none");
    }
    return s;
}

void hixl_cleanup(void* ctx_ptr) {
    auto* ctx = static_cast<HixlTestCtx*>(ctx_ptr);
    ctx->engine->DeregisterMem(ctx->mem_handle);
    ctx->engine->Finalize();
    delete ctx->engine;
    delete ctx;
}

} // extern "C"
