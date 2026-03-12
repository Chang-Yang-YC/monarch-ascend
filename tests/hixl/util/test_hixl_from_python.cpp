/**
 * Shared library providing a flat C API for HiXL (Huawei Xfer Library).
 *
 * All HiXL engine operations — including Initialize — are serialised onto
 * a single dedicated OS thread per engine.  This is necessary because:
 *   1. ACL device context is thread-local (aclrtSetDevice)
 *   2. HCCS channels may have thread-affinity
 *   3. HiXL's internal state may bind to the initializing thread
 *
 * Callers can invoke public functions from *any* thread; the implementation
 * marshals all work to the engine's worker thread and blocks until completion.
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
#include <mutex>
#include <condition_variable>
#include <functional>
#include <unistd.h>
using namespace hixl;

extern "C" {

struct HixlTestCtx {
    Hixl* engine = nullptr;
    MemHandle mem_handle;
    int device = -1;

    std::thread worker;
    std::mutex mtx;
    std::condition_variable work_cv;
    bool shutdown = false;

    std::function<void()> pending_task;
    bool task_ready = false;

    int task_result = 0;
    bool result_ready = false;
    std::condition_variable result_cv;
};

static void worker_loop(HixlTestCtx* ctx) {
    fprintf(stderr, "[hixl_test] worker_loop started: tid=%ld dev=%d\n",
            (long)pthread_self(), ctx->device);

    while (true) {
        std::unique_lock<std::mutex> lk(ctx->mtx);
        ctx->work_cv.wait(lk, [ctx] { return ctx->task_ready || ctx->shutdown; });
        if (ctx->shutdown) break;

        ctx->pending_task();
        ctx->task_ready = false;
        ctx->result_ready = true;
        ctx->result_cv.notify_one();
    }
    fprintf(stderr, "[hixl_test] worker_loop exiting: tid=%ld\n", (long)pthread_self());
}

static int run_on_worker(HixlTestCtx* ctx, std::function<int()> fn) {
    std::unique_lock<std::mutex> lk(ctx->mtx);
    ctx->task_result = -1;
    ctx->result_ready = false;
    ctx->pending_task = [&] { ctx->task_result = fn(); };
    ctx->task_ready = true;
    ctx->work_cv.notify_one();
    ctx->result_cv.wait(lk, [ctx] { return ctx->result_ready; });
    return ctx->task_result;
}

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
    auto* ctx = new HixlTestCtx();
    ctx->device = dev;

    // Start the worker thread first — it will do ALL HiXL operations
    ctx->worker = std::thread(worker_loop, ctx);

    // Run aclrtSetDevice + engine->Initialize on the worker thread
    std::string eid(engine_id);
    int rc = run_on_worker(ctx, [=]() -> int {
        aclrtSetDevice(dev);
        ctx->engine = new Hixl();
        std::map<AscendString, AscendString> opts;
        opts["BufferPool"] = "0:0";
        auto s = ctx->engine->Initialize(AscendString(eid.c_str()), opts);
        fprintf(stderr, "[hixl_test] Init(%s) dev=%d tid=%ld: %u\n",
                eid.c_str(), dev, (long)pthread_self(), s);
        return (int)s;
    });

    if (rc != 0) {
        {
            std::lock_guard<std::mutex> lk(ctx->mtx);
            ctx->shutdown = true;
        }
        ctx->work_cv.notify_one();
        ctx->worker.join();
        if (ctx->engine) { delete ctx->engine; }
        delete ctx;
        return nullptr;
    }
    return ctx;
}

void* hixl_init_engine_with_ctx(uintptr_t acl_ctx_ptr, const char* engine_id) {
    auto* ctx = new HixlTestCtx();

    ctx->worker = std::thread(worker_loop, ctx);

    aclrtContext acl_ctx = reinterpret_cast<aclrtContext>(acl_ctx_ptr);
    std::string eid(engine_id);
    int rc = run_on_worker(ctx, [=]() -> int {
        aclError ret = aclrtSetCurrentContext(acl_ctx);
        fprintf(stderr, "[hixl_test] InitWithCtx(%s) set_ctx=%p ret=%d tid=%ld\n",
                eid.c_str(), acl_ctx, ret, (long)pthread_self());
        ctx->engine = new Hixl();
        std::map<AscendString, AscendString> opts;
        opts["BufferPool"] = "0:0";
        auto s = ctx->engine->Initialize(AscendString(eid.c_str()), opts);
        fprintf(stderr, "[hixl_test] InitWithCtx(%s): %u\n", eid.c_str(), s);
        return (int)s;
    });

    if (rc != 0) {
        {
            std::lock_guard<std::mutex> lk(ctx->mtx);
            ctx->shutdown = true;
        }
        ctx->work_cv.notify_one();
        ctx->worker.join();
        if (ctx->engine) { delete ctx->engine; }
        delete ctx;
        return nullptr;
    }
    return ctx;
}

int hixl_register_mem(void* ctx_ptr, uintptr_t addr, size_t size) {
    auto* ctx = static_cast<HixlTestCtx*>(ctx_ptr);
    return run_on_worker(ctx, [=]() -> int {
        MemDesc d{}; d.addr = addr; d.len = size;
        auto s = ctx->engine->RegisterMem(d, MEM_DEVICE, ctx->mem_handle);
        fprintf(stderr, "[hixl_test] RegMem addr=%p size=%zu: %u (tid=%ld)\n",
                (void*)addr, size, s, (long)pthread_self());
        return (int)s;
    });
}

int hixl_connect(void* ctx_ptr, const char* remote_engine_id) {
    auto* ctx = static_cast<HixlTestCtx*>(ctx_ptr);
    std::string eid(remote_engine_id);
    return run_on_worker(ctx, [=]() -> int {
        auto s = ctx->engine->Connect(AscendString(eid.c_str()), 10000);
        fprintf(stderr, "[hixl_test] Connect(%s): %u pid=%d (tid=%ld)\n",
                eid.c_str(), s, getpid(), (long)pthread_self());
        if (s != SUCCESS) {
            auto* msg = aclGetRecentErrMsg();
            fprintf(stderr, "[hixl_test] Connect ACL error: %s\n", msg ? msg : "none");
        }
        return (int)s;
    });
}

int hixl_transfer_write(void* ctx_ptr, const char* remote_engine_id,
                         uintptr_t local_addr, uintptr_t remote_addr, size_t len) {
    auto* ctx = static_cast<HixlTestCtx*>(ctx_ptr);
    std::string eid(remote_engine_id);
    return run_on_worker(ctx, [=]() -> int {
        TransferOpDesc td{};
        td.local_addr = local_addr;
        td.remote_addr = remote_addr;
        td.len = len;
        fprintf(stderr, "[hixl_test] TransferSync(WRITE) local=%p remote=%p len=%zu to=%s (tid=%ld)\n",
                (void*)local_addr, (void*)remote_addr, len, eid.c_str(), (long)pthread_self());
        auto s = ctx->engine->TransferSync(AscendString(eid.c_str()), WRITE, {td}, 10000);
        fprintf(stderr, "[hixl_test] TransferSync WRITE result: %u\n", s);
        if (s != SUCCESS) {
            auto* msg = aclGetRecentErrMsg();
            fprintf(stderr, "[hixl_test] ACL error: %s\n", msg ? msg : "none");
        }
        return (int)s;
    });
}

int hixl_transfer_read(void* ctx_ptr, const char* remote_engine_id,
                        uintptr_t local_addr, uintptr_t remote_addr, size_t len) {
    auto* ctx = static_cast<HixlTestCtx*>(ctx_ptr);
    std::string eid(remote_engine_id);
    return run_on_worker(ctx, [=]() -> int {
        TransferOpDesc td{};
        td.local_addr = local_addr;
        td.remote_addr = remote_addr;
        td.len = len;
        fprintf(stderr, "[hixl_test] TransferSync(READ) local=%p remote=%p len=%zu from=%s (tid=%ld)\n",
                (void*)local_addr, (void*)remote_addr, len, eid.c_str(), (long)pthread_self());
        auto s = ctx->engine->TransferSync(AscendString(eid.c_str()), READ, {td}, 10000);
        fprintf(stderr, "[hixl_test] TransferSync READ result: %u\n", s);
        if (s != SUCCESS) {
            auto* msg = aclGetRecentErrMsg();
            fprintf(stderr, "[hixl_test] ACL error: %s\n", msg ? msg : "none");
        }
        return (int)s;
    });
}

void hixl_cleanup(void* ctx_ptr) {
    auto* ctx = static_cast<HixlTestCtx*>(ctx_ptr);
    run_on_worker(ctx, [=]() -> int {
        ctx->engine->DeregisterMem(ctx->mem_handle);
        ctx->engine->Finalize();
        fprintf(stderr, "[hixl_test] Finalize done (tid=%ld)\n", (long)pthread_self());
        return 0;
    });
    {
        std::lock_guard<std::mutex> lk(ctx->mtx);
        ctx->shutdown = true;
    }
    ctx->work_cv.notify_one();
    if (ctx->worker.joinable()) ctx->worker.join();
    delete ctx->engine;
    delete ctx;
}

} // extern "C"
