/**
 * Quick test: ACL_MEM_MALLOC_HUGE_ONLY vs ACL_MEM_MALLOC_NORMAL_ONLY
 * for HIXL TransferSync.
 */
#include <hixl/hixl.h>
#include <hixl/hixl_types.h>
#include <acl/acl.h>
#include <acl/acl_rt.h>

#include <unistd.h>
#include <sys/wait.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <map>
#include <vector>

using namespace hixl;

static const char* SRV_ENGINE = "127.0.0.1:16000";
static const char* CLI_ENGINE = "127.0.0.1:16001";
constexpr size_t BYTES = 30;
constexpr int32_t WAIT_SEC = 5;

int run_server(int dev, aclrtMemMallocPolicy policy, const char* label) {
    aclInit(nullptr);
    aclrtSetDevice(dev);

    Hixl srv;
    std::map<AscendString, AscendString> opts;
    opts["BufferPool"] = "0:0";
    auto s = srv.Initialize(AscendString(SRV_ENGINE), opts);
    if (s != SUCCESS) { printf("[S-%s] Init fail: %u\n", label, s); return 1; }

    uint8_t *buf = nullptr;
    aclrtMalloc((void**)&buf, BYTES, policy);
    aclrtMemset(buf, BYTES, 0xAA, BYTES);

    MemDesc desc{}; desc.addr = reinterpret_cast<uintptr_t>(buf); desc.len = BYTES;
    MemHandle mh = nullptr;
    s = srv.RegisterMem(desc, MEM_DEVICE, mh);
    printf("[S-%s] RegisterMem: %u addr=%p\n", label, s, buf);
    if (s != SUCCESS) return 2;

    std::ofstream(SRV_ENGINE) << reinterpret_cast<uintptr_t>(buf);
    std::this_thread::sleep_for(std::chrono::seconds(WAIT_SEC * 3));

    char host_check[BYTES] = {};
    aclrtMemcpy(host_check, BYTES, buf, BYTES, ACL_MEMCPY_DEVICE_TO_HOST);
    printf("[S-%s] buf[0]=%u buf[1]=%u\n", label, (uint8_t)host_check[0], (uint8_t)host_check[1]);

    srv.DeregisterMem(mh);
    srv.Finalize();
    aclrtFree(buf);
    return 0;
}

int run_client(int dev, aclrtMemMallocPolicy policy, const char* label) {
    aclInit(nullptr);
    aclrtSetDevice(dev);

    Hixl cli;
    std::map<AscendString, AscendString> opts;
    opts["BufferPool"] = "0:0";
    auto s = cli.Initialize(AscendString(CLI_ENGINE), opts);
    if (s != SUCCESS) { printf("[C-%s] Init fail: %u\n", label, s); return 1; }

    std::this_thread::sleep_for(std::chrono::seconds(WAIT_SEC));
    uintptr_t remote_addr;
    std::ifstream(SRV_ENGINE) >> remote_addr;
    printf("[C-%s] remote_addr=%p\n", label, (void*)remote_addr);

    s = cli.Connect(AscendString(SRV_ENGINE));
    printf("[C-%s] Connect: %u\n", label, s);
    if (s != SUCCESS) return 2;

    uint8_t *local_buf = nullptr;
    aclrtMalloc((void**)&local_buf, BYTES, policy);
    float host_val = 42.0f;
    aclrtMemcpy(local_buf, sizeof(float), &host_val, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);

    MemDesc desc{}; desc.addr = reinterpret_cast<uintptr_t>(local_buf); desc.len = BYTES;
    MemHandle mh = nullptr;
    s = cli.RegisterMem(desc, MEM_DEVICE, mh);
    printf("[C-%s] RegisterMem: %u\n", label, s);
    if (s != SUCCESS) { aclrtFree(local_buf); return 3; }

    TransferOpDesc td{};
    td.local_addr = desc.addr;
    td.remote_addr = remote_addr;
    td.len = BYTES;

    s = cli.TransferSync(AscendString(SRV_ENGINE), WRITE, {td});
    printf("[C-%s] TransferSync(WRITE): %u %s\n", label, s, s == SUCCESS ? "OK" : "FAIL");

    cli.DeregisterMem(mh);
    cli.Disconnect(AscendString(SRV_ENGINE));
    cli.Finalize();
    aclrtFree(local_buf);
    return (s == SUCCESS) ? 0 : 4;
}

void run_test(aclrtMemMallocPolicy policy, const char* label) {
    printf("\n=== %s ===\n", label);
    unlink(SRV_ENGINE);

    pid_t srv_pid = fork();
    if (srv_pid == 0) _exit(run_server(0, policy, label));

    pid_t cli_pid = fork();
    if (cli_pid == 0) _exit(run_client(1, policy, label));

    int cli_st = 0;
    waitpid(cli_pid, &cli_st, 0);
    kill(srv_pid, SIGTERM);
    waitpid(srv_pid, nullptr, 0);
    unlink(SRV_ENGINE);

    bool ok = WIFEXITED(cli_st) && WEXITSTATUS(cli_st) == 0;
    printf(">>> %s: %s\n", label, ok ? "PASS" : "FAIL");
}

int main() {
    run_test(ACL_MEM_MALLOC_HUGE_ONLY, "HUGE_ONLY");
    run_test(ACL_MEM_MALLOC_NORMAL_ONLY, "NORMAL_ONLY");
    return 0;
}
