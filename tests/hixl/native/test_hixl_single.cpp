/**
 * Single-process HIXL test, same pattern as official server_server_d2d.
 * Usage: ./test_hixl_single <device_id> <local_engine> <remote_engine> [NORMAL|HUGE]
 */
#include <hixl/hixl.h>
#include <hixl/hixl_types.h>
#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <cstring>
#include <map>
#include <vector>
using namespace hixl;

constexpr size_t BUF_SIZE = 30;
constexpr int WAIT = 5;

const char *errmsg() {
    auto m = aclGetRecentErrMsg();
    return m ? m : "(none)";
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <dev> <local> <remote> [NORMAL|HUGE]\n", argv[0]);
        return 1;
    }
    int dev = atoi(argv[1]);
    const char* local_eng = argv[2];
    const char* remote_eng = argv[3];
    aclrtMemMallocPolicy policy = ACL_MEM_MALLOC_HUGE_ONLY;
    if (argc > 4 && strcmp(argv[4], "NORMAL") == 0)
        policy = ACL_MEM_MALLOC_NORMAL_ONLY;

    printf("[%s] dev=%d policy=%s\n", local_eng, dev, (policy == ACL_MEM_MALLOC_HUGE_ONLY) ? "HUGE" : "NORMAL");
    fflush(stdout);

    aclInit(nullptr);
    aclrtSetDevice(dev);

    Hixl eng;
    std::map<AscendString, AscendString> opts;
    opts["BufferPool"] = "0:0";
    auto s = eng.Initialize(AscendString(local_eng), opts);
    printf("[%s] Init: %u %s\n", local_eng, s, s==0?"OK":errmsg()); fflush(stdout);
    if (s != SUCCESS) return 2;

    uint8_t *buf1 = nullptr, *buf2 = nullptr;
    aclrtMalloc((void**)&buf1, BUF_SIZE, policy);
    aclrtMalloc((void**)&buf2, BUF_SIZE, policy);

    MemDesc d1{}; d1.addr = reinterpret_cast<uintptr_t>(buf1); d1.len = BUF_SIZE;
    MemDesc d2{}; d2.addr = reinterpret_cast<uintptr_t>(buf2); d2.len = BUF_SIZE;
    MemHandle h1 = nullptr, h2 = nullptr;
    s = eng.RegisterMem(d1, MEM_DEVICE, h1);
    printf("[%s] Reg1: %u\n", local_eng, s); fflush(stdout);
    s = eng.RegisterMem(d2, MEM_DEVICE, h2);
    printf("[%s] Reg2: %u\n", local_eng, s); fflush(stdout);

    printf("[%s] addr1=%p addr2=%p\n", local_eng, buf1, buf2); fflush(stdout);
    std::ofstream(local_eng) << std::hex << reinterpret_cast<uintptr_t>(buf1)
                              << " " << reinterpret_cast<uintptr_t>(buf2) << std::endl;

    std::this_thread::sleep_for(std::chrono::seconds(WAIT));

    s = eng.Connect(AscendString(remote_eng));
    printf("[%s] Connect: %u %s\n", local_eng, s, s==0?"OK":errmsg()); fflush(stdout);
    if (s != SUCCESS) { eng.Finalize(); return 3; }

    uintptr_t raddr1, raddr2;
    std::ifstream(remote_eng) >> std::hex >> raddr1 >> raddr2;
    printf("[%s] remote addrs: %p %p\n", local_eng, (void*)raddr1, (void*)raddr2); fflush(stdout);

    if (std::string(local_eng) < std::string(remote_eng)) {
        aclrtMemcpy(buf1, BUF_SIZE, local_eng, strlen(local_eng), ACL_MEMCPY_HOST_TO_DEVICE);
        TransferOpDesc td{reinterpret_cast<uintptr_t>(buf1), raddr1, strlen(local_eng)};
        s = eng.TransferSync(AscendString(remote_eng), WRITE, {td});
        printf("[%s] WRITE: %u %s\n", local_eng, s, s==0?"OK":errmsg()); fflush(stdout);
        std::this_thread::sleep_for(std::chrono::seconds(WAIT));
    } else {
        std::this_thread::sleep_for(std::chrono::seconds(WAIT));
        char val[BUF_SIZE] = {};
        aclrtMemcpy(val, BUF_SIZE, buf1, strlen(remote_eng), ACL_MEMCPY_DEVICE_TO_HOST);
        printf("[%s] peer wrote: %s\n", local_eng, val); fflush(stdout);

        TransferOpDesc td{reinterpret_cast<uintptr_t>(buf2), raddr2, strlen(remote_eng)};
        s = eng.TransferSync(AscendString(remote_eng), READ, {td});
        printf("[%s] READ: %u %s\n", local_eng, s, s==0?"OK":errmsg()); fflush(stdout);
    }

    eng.Disconnect(AscendString(remote_eng));
    std::this_thread::sleep_for(std::chrono::seconds(WAIT));
    eng.DeregisterMem(h1);
    eng.DeregisterMem(h2);
    eng.Finalize();
    aclrtFree(buf1); aclrtFree(buf2);
    aclrtResetDevice(dev); aclFinalize();
    printf("[%s] DONE\n", local_eng); fflush(stdout);
    return (s == SUCCESS) ? 0 : 4;
}
