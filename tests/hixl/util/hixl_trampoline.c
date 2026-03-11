/**
 * C trampoline for HIXL calls — resets signal mask before calling.
 */
#include <stddef.h>
#include <stdint.h>
#include <dlfcn.h>
#include <stdio.h>
#include <signal.h>

typedef void* (*init_fn)(int, const char*);
typedef int (*reg_fn)(void*, size_t, size_t);
typedef int (*conn_fn)(void*, const char*);
typedef int (*xfer_fn)(void*, const char*, size_t, size_t, size_t);
typedef void (*cleanup_fn)(void*);

static void* g_lib = NULL;
static init_fn g_init = NULL;
static reg_fn g_reg = NULL;
static conn_fn g_conn = NULL;
static xfer_fn g_xfer = NULL;

static void ensure_loaded(void) {
    if (g_lib) return;
    g_lib = dlopen("/root/monarch/libtest_hixl.so", RTLD_NOW);
    if (!g_lib) {
        fprintf(stderr, "[trampoline] dlopen failed: %s\n", dlerror());
        return;
    }
    g_init = (init_fn)dlsym(g_lib, "hixl_init_engine");
    g_reg = (reg_fn)dlsym(g_lib, "hixl_register_mem");
    g_conn = (conn_fn)dlsym(g_lib, "hixl_connect");
    g_xfer = (xfer_fn)dlsym(g_lib, "hixl_transfer_write");
}

static void reset_sigmask(void) {
    sigset_t empty;
    sigemptyset(&empty);
    pthread_sigmask(SIG_SETMASK, &empty, NULL);
}

void* trampoline_init(int dev, const char* engine_id) {
    ensure_loaded();
    if (!g_init) return NULL;
    reset_sigmask();
    return g_init(dev, engine_id);
}

int trampoline_reg(void* ctx, size_t addr, size_t size) {
    if (!g_reg) return -1;
    reset_sigmask();
    return g_reg(ctx, addr, size);
}

int trampoline_connect(void* ctx, const char* remote) {
    if (!g_conn) return -1;
    reset_sigmask();
    return g_conn(ctx, remote);
}

int trampoline_transfer(void* ctx, const char* remote,
                         size_t local_addr, size_t remote_addr, size_t len) {
    if (!g_xfer) return -1;
    reset_sigmask();
    fprintf(stderr, "[trampoline] xfer: ctx=%p remote=%s local=%p remote_addr=%p len=%zu\n",
            ctx, remote, (void*)local_addr, (void*)remote_addr, len);
    return g_xfer(ctx, remote, local_addr, remote_addr, len);
}
