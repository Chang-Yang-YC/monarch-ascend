/**
 * Test: server and client both connect before any transfer.
 * Mimics the official server_server_d2d pattern exactly but with NORMAL_ONLY memory.
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

constexpr size_t BUF_SIZE = 64;
const char* SRV = "192.168.0.117:18000";
const char* CLI = "192.168.0.117:18001";

int run_autoconnect(int dev, const char* local_eng, const char* remote_eng) {
    printf("[%s] dev=%d AutoConnect mode\n", local_eng, dev); fflush(stdout);
    aclInit(nullptr);
    aclrtSetDevice(dev);

    Hixl eng;
    std::map<AscendString, AscendString> opts;
    opts["BufferPool"] = "0:0";
    opts[OPTION_AUTO_CONNECT] = "1";
    auto s = eng.Initialize(AscendString(local_eng), opts);
    printf("[%s] Init: %u\n", local_eng, s); fflush(stdout);

    uint8_t *buf = nullptr;
    aclrtMalloc((void**)&buf, BUF_SIZE, ACL_MEM_MALLOC_NORMAL_ONLY);
    MemDesc d{}; d.addr = reinterpret_cast<uintptr_t>(buf); d.len = BUF_SIZE;
    MemHandle h = nullptr;
    s = eng.RegisterMem(d, MEM_DEVICE, h);
    printf("[%s] RegMem: %u addr=%p\n", local_eng, s, buf); fflush(stdout);

    std::ofstream(local_eng) << reinterpret_cast<uintptr_t>(buf);

    bool is_writer = (std::string(local_eng) > std::string(remote_eng));
    if (is_writer) {
        // Only writer connects
        std::this_thread::sleep_for(std::chrono::seconds(5));
        s = eng.Connect(AscendString(remote_eng), 10000);
        printf("[%s] Connect: %u\n", local_eng, s); fflush(stdout);
        if (s != SUCCESS) { eng.Finalize(); return 1; }

        uintptr_t remote_addr;
        std::ifstream(remote_eng) >> remote_addr;
        std::this_thread::sleep_for(std::chrono::seconds(2));

        float ones[16]; for (int i = 0; i < 16; ++i) ones[i] = 1.0f;
        aclrtMemcpy(buf, BUF_SIZE, ones, BUF_SIZE, ACL_MEMCPY_HOST_TO_DEVICE);
        TransferOpDesc td{reinterpret_cast<uintptr_t>(buf), remote_addr, BUF_SIZE};
        s = eng.TransferSync(AscendString(remote_eng), WRITE, {td}, 10000);
        printf("[%s] WRITE: %u\n", local_eng, s); fflush(stdout);
        std::this_thread::sleep_for(std::chrono::seconds(3));
    } else {
        // Server just waits — does NOT connect
        std::this_thread::sleep_for(std::chrono::seconds(12));
        uint8_t check[BUF_SIZE] = {};
        aclrtMemcpy(check, BUF_SIZE, buf, BUF_SIZE, ACL_MEMCPY_DEVICE_TO_HOST);
        float sum = 0;
        for (size_t i = 0; i < 16; ++i) sum += reinterpret_cast<float*>(check)[i];
        printf("[%s] Received sum=%.1f (expect 16)\n", local_eng, sum); fflush(stdout);
    }

    eng.Disconnect(AscendString(remote_eng));
    std::this_thread::sleep_for(std::chrono::seconds(3));
    eng.DeregisterMem(h);
    eng.Finalize();
    aclrtFree(buf);
    aclrtResetDevice(dev); aclFinalize();
    printf("[%s] DONE\n", local_eng); fflush(stdout);
    return (s == SUCCESS) ? 0 : 2;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s server|client\n", argv[0]);
        return 1;
    }
    if (strcmp(argv[1], "server") == 0) return run_autoconnect(0, SRV, CLI);
    if (strcmp(argv[1], "client") == 0) return run_autoconnect(1, CLI, SRV);
    return 1;
}
