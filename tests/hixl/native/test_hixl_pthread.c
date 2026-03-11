/**
 * Minimal test: dlopen libtest_hixl.so + call HIXL from a pthread
 * (no Python, no Rust, no Tokio)
 *
 * Usage: compile and run from two processes (or use fork)
 *   gcc -o test_hixl_pthread test_hixl_pthread.c -ldl -lpthread -lascendcl -L/path/to/lib64
 *   HCCL_INTRA_ROCE_ENABLE=1 ./test_hixl_pthread
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <fcntl.h>

typedef void* (*init_fn)(int, const char*);
typedef int (*reg_fn)(void*, size_t, size_t);
typedef int (*conn_fn)(void*, const char*);
typedef int (*xfer_fn)(void*, const char*, size_t, size_t, size_t);
typedef void (*cleanup_fn)(void*);

/* ACL functions loaded via dlopen */
typedef int (*acl_init_fn)(const char*);
typedef int (*acl_set_dev_fn)(int);
typedef int (*acl_malloc_fn)(void**, size_t, int);
typedef int (*acl_memcpy_fn)(void*, size_t, const void*, size_t, int);
typedef int (*acl_free_fn)(void*);

struct thread_args {
    int dev;
    const char* engine_id;
    const char* peer_engine_id;
    size_t local_addr;
    size_t remote_addr;
    size_t size;
    int is_writer;
    int result;
};

static void* hixl_thread_func(void* arg) {
    struct thread_args* a = (struct thread_args*)arg;

    void* lib = dlopen("/root/monarch/libtest_hixl.so", RTLD_NOW);
    if (!lib) {
        fprintf(stderr, "[pthread] dlopen failed: %s\n", dlerror());
        a->result = -1;
        return NULL;
    }

    init_fn fn_init = (init_fn)dlsym(lib, "hixl_init_engine");
    reg_fn fn_reg = (reg_fn)dlsym(lib, "hixl_register_mem");
    conn_fn fn_conn = (conn_fn)dlsym(lib, "hixl_connect");
    xfer_fn fn_xfer = (xfer_fn)dlsym(lib, "hixl_transfer_write");
    cleanup_fn fn_clean = (cleanup_fn)dlsym(lib, "hixl_cleanup");

    if (!fn_init || !fn_reg || !fn_conn || !fn_xfer || !fn_clean) {
        fprintf(stderr, "[pthread] dlsym failed\n");
        a->result = -1;
        return NULL;
    }

    fprintf(stderr, "[pthread dev=%d] init engine %s\n", a->dev, a->engine_id);
    void* ctx = fn_init(a->dev, a->engine_id);
    if (!ctx) {
        fprintf(stderr, "[pthread dev=%d] init failed\n", a->dev);
        a->result = -2;
        return NULL;
    }

    fprintf(stderr, "[pthread dev=%d] register mem addr=%p size=%zu\n",
            a->dev, (void*)a->local_addr, a->size);
    int ret = fn_reg(ctx, a->local_addr, a->size);
    if (ret != 0) {
        fprintf(stderr, "[pthread dev=%d] register failed: %d\n", a->dev, ret);
        a->result = ret;
        return NULL;
    }

    /* Wait a bit for the peer to initialize */
    usleep(2000000);

    fprintf(stderr, "[pthread dev=%d] connect to %s\n", a->dev, a->peer_engine_id);
    int max_attempts = 20;
    for (int i = 0; i < max_attempts; i++) {
        ret = fn_conn(ctx, a->peer_engine_id);
        if (ret == 0) break;
        fprintf(stderr, "[pthread dev=%d] connect attempt %d failed: %d\n", a->dev, i+1, ret);
        usleep(500000);
    }
    if (ret != 0) {
        fprintf(stderr, "[pthread dev=%d] connect failed after %d attempts\n", a->dev, max_attempts);
        a->result = ret;
        return NULL;
    }

    if (a->is_writer) {
        /* Wait for peer to connect back */
        usleep(3000000);
        fprintf(stderr, "[pthread dev=%d] transfer write local=%p remote=%p size=%zu to=%s\n",
                a->dev, (void*)a->local_addr, (void*)a->remote_addr, a->size, a->peer_engine_id);
        ret = fn_xfer(ctx, a->peer_engine_id, a->local_addr, a->remote_addr, a->size);
        fprintf(stderr, "[pthread dev=%d] transfer result: %d\n", a->dev, ret);
        a->result = ret;
    } else {
        a->result = 0;
    }

    return NULL;
}

int main() {
    setenv("HCCL_INTRA_ROCE_ENABLE", "1", 1);

    /* Load ACL to allocate device memory */
    void* acl_lib = dlopen("libascendcl.so", RTLD_NOW | RTLD_GLOBAL);
    if (!acl_lib) {
        fprintf(stderr, "dlopen libascendcl.so failed: %s\n", dlerror());
        return 1;
    }
    acl_init_fn acl_init = (acl_init_fn)dlsym(acl_lib, "aclInit");
    acl_set_dev_fn acl_set_dev = (acl_set_dev_fn)dlsym(acl_lib, "aclrtSetDevice");
    acl_malloc_fn acl_malloc = (acl_malloc_fn)dlsym(acl_lib, "aclrtMalloc");
    acl_memcpy_fn acl_memcpy = (acl_memcpy_fn)dlsym(acl_lib, "aclrtMemcpy");
    acl_free_fn acl_free = (acl_free_fn)dlsym(acl_lib, "aclrtFree");

    if (!acl_init || !acl_set_dev || !acl_malloc || !acl_memcpy || !acl_free) {
        fprintf(stderr, "dlsym ACL functions failed\n");
        return 1;
    }

    size_t nbytes = 64;
    const char* eng_a = "127.0.0.1:19400";
    const char* eng_b = "127.0.0.1:19401";

    /* Fork: parent = dev0 (server), child = dev1 (writer) */
    pid_t pid = fork();
    if (pid < 0) { perror("fork"); return 1; }

    if (pid == 0) {
        /* Child: dev=1, writer */
        acl_init(NULL);
        acl_set_dev(1);
        void* dev_mem = NULL;
        int ret = acl_malloc(&dev_mem, nbytes, 0 /* ACL_MEM_MALLOC_HUGE_FIRST */);
        fprintf(stderr, "[child dev=1] aclrtMalloc: ret=%d addr=%p\n", ret, dev_mem);

        /* Fill with 2.0f pattern */
        float pattern[16];
        for (int i = 0; i < 16; i++) pattern[i] = 2.0f;
        acl_memcpy(dev_mem, nbytes, pattern, nbytes, 1 /* host2device */);

        struct thread_args args = {
            .dev = 1,
            .engine_id = eng_b,
            .peer_engine_id = eng_a,
            .local_addr = (size_t)dev_mem,
            .remote_addr = 0,  /* will be set from file */
            .size = nbytes,
            .is_writer = 1,
            .result = -99
        };

        /* Read remote addr from coord file */
        for (int i = 0; i < 20; i++) {
            FILE* f = fopen("/tmp/hixl_pthread_coord", "r");
            if (f) {
                char buf[64];
                if (fgets(buf, sizeof(buf), f)) {
                    args.remote_addr = (size_t)strtoull(buf, NULL, 10);
                }
                fclose(f);
                if (args.remote_addr != 0) break;
            }
            usleep(500000);
        }
        fprintf(stderr, "[child] remote_addr=%p\n", (void*)args.remote_addr);

        pthread_t t;
        pthread_create(&t, NULL, hixl_thread_func, &args);
        pthread_join(t, NULL);

        fprintf(stderr, "[child] final result: %d\n", args.result);
        acl_free(dev_mem);
        _exit(args.result);
    } else {
        /* Parent: dev=0, server */
        acl_init(NULL);
        acl_set_dev(0);
        void* dev_mem = NULL;
        int ret = acl_malloc(&dev_mem, nbytes, 0);
        fprintf(stderr, "[parent dev=0] aclrtMalloc: ret=%d addr=%p\n", ret, dev_mem);

        /* Fill with 1.0f */
        float pattern[16];
        for (int i = 0; i < 16; i++) pattern[i] = 1.0f;
        acl_memcpy(dev_mem, nbytes, pattern, nbytes, 1);

        /* Write addr to coord file */
        FILE* f = fopen("/tmp/hixl_pthread_coord", "w");
        fprintf(f, "%zu", (size_t)dev_mem);
        fclose(f);

        struct thread_args args = {
            .dev = 0,
            .engine_id = eng_a,
            .peer_engine_id = eng_b,
            .local_addr = (size_t)dev_mem,
            .remote_addr = 0,
            .size = nbytes,
            .is_writer = 0,
            .result = -99
        };

        pthread_t t;
        pthread_create(&t, NULL, hixl_thread_func, &args);
        pthread_join(t, NULL);

        fprintf(stderr, "[parent] server result: %d\n", args.result);

        int status;
        waitpid(pid, &status, 0);
        int child_exit = WEXITSTATUS(status);
        fprintf(stderr, "[parent] child exit: %d\n", child_exit);

        /* Read back and verify */
        float readback[16];
        acl_memcpy(readback, nbytes, dev_mem, nbytes, 2 /* device2host */);
        float sum = 0;
        for (int i = 0; i < 16; i++) sum += readback[i];
        fprintf(stderr, "[parent] readback sum = %.1f (expect 32.0 if write succeeded)\n", sum);

        acl_free(dev_mem);
        unlink("/tmp/hixl_pthread_coord");

        return (child_exit == 0 && args.result == 0) ? 0 : 1;
    }
}
