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
#include <vector>
#include <map>

static const char* SRV_ID = "192.168.0.117:30001";
static const char* CLI_ID = "192.168.0.117:30002";
constexpr size_t BYTES = 16 * sizeof(float);

// server: register buffer on dev 0, expose address, wait for signal
static int run_server(int write_fd, int read_fd) {
    aclInit(nullptr);
    aclrtSetDevice(0);

    hixl::Hixl srv;
    std::map<hixl::AscendString, hixl::AscendString> opts;
    auto s = srv.Initialize(hixl::AscendString(SRV_ID), opts);
    std::cout << "[S] Init: " << s << std::endl;
    if (s != 0) { uint64_t z=0; write(write_fd,&z,8); return 1; }

    void* buf = nullptr;
    aclrtMalloc(&buf, BYTES, ACL_MEM_MALLOC_NORMAL_ONLY);
    float host_src[16];
    for (int i = 0; i < 16; ++i) host_src[i] = static_cast<float>(i);
    aclrtMemcpy(buf, BYTES, host_src, BYTES, ACL_MEMCPY_HOST_TO_DEVICE);

    hixl::MemDesc mem{};
    mem.addr = reinterpret_cast<uintptr_t>(buf);
    mem.len = BYTES;
    hixl::MemHandle mh = nullptr;
    s = srv.RegisterMem(mem, hixl::MEM_DEVICE, mh);
    std::cout << "[S] RegisterMem: " << s << " addr=0x" << std::hex << mem.addr << std::dec << std::endl;
    if (s != 0) { uint64_t z=0; write(write_fd,&z,8); return 2; }

    // send addr to client
    uint64_t addr = static_cast<uint64_t>(mem.addr);
    write(write_fd, &addr, 8);

    // server also connects to client (bidirectional)
    sleep(1);
    s = srv.Connect(hixl::AscendString(CLI_ID), 15000);
    std::cout << "[S] Connect(" << CLI_ID << "): " << s << std::endl;

    // signal client that server is ready
    uint8_t ready = 1;
    write(write_fd, &ready, 1);

    // wait for "done" signal from client
    uint8_t done = 0;
    read(read_fd, &done, 1);

    // verify server buffer was written by client (WRITE test)
    float host_check[16] = {0};
    aclrtMemcpy(host_check, BYTES, buf, BYTES, ACL_MEMCPY_DEVICE_TO_HOST);
    float sum = 0;
    for (float v : host_check) sum += v;
    std::cout << "[S] Final buf sum=" << sum << std::endl;

    srv.DeregisterMem(mh);
    srv.Finalize();
    aclrtFree(buf);
    return 0;
}

// client: allocate local buffer on dev 1, connect, try READ then WRITE
static int run_client(int read_fd, int write_fd) {
    aclInit(nullptr);
    aclrtSetDevice(1);

    hixl::Hixl cli;
    std::map<hixl::AscendString, hixl::AscendString> opts;
    auto s = cli.Initialize(hixl::AscendString(CLI_ID), opts);
    std::cout << "[C] Init: " << s << std::endl;
    if (s != 0) return 1;

    // read server buffer address
    uint64_t remote_addr = 0;
    read(read_fd, &remote_addr, 8);
    if (remote_addr == 0) return 2;
    std::cout << "[C] remote_addr=0x" << std::hex << remote_addr << std::dec << std::endl;

    // connect client -> server
    sleep(2);
    s = cli.Connect(hixl::AscendString(SRV_ID), 15000);
    std::cout << "[C] Connect(" << SRV_ID << "): " << s << std::endl;
    if (s != 0) return 3;

    // wait for server "ready" signal (server has also connected to us)
    uint8_t ready = 0;
    read(read_fd, &ready, 1);
    std::cout << "[C] Server signaled ready, bidirectional connection established" << std::endl;
    sleep(1);

    // allocate local device memory
    void* local_buf = nullptr;
    aclrtMalloc(&local_buf, BYTES, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMemset(local_buf, BYTES, 0, BYTES);

    hixl::MemDesc local_mem{};
    local_mem.addr = reinterpret_cast<uintptr_t>(local_buf);
    local_mem.len = BYTES;
    hixl::MemHandle local_mh = nullptr;
    s = cli.RegisterMem(local_mem, hixl::MEM_DEVICE, local_mh);
    std::cout << "[C] RegisterMem: " << s << std::endl;
    if (s != 0) { aclrtFree(local_buf); return 4; }

    // --- TEST 1: READ from server ---
    {
        hixl::TransferOpDesc desc{};
        desc.local_addr = local_mem.addr;
        desc.remote_addr = static_cast<uintptr_t>(remote_addr);
        desc.len = BYTES;
        std::vector<hixl::TransferOpDesc> descs{desc};

        s = cli.TransferSync(hixl::AscendString(SRV_ID), hixl::READ, descs, 15000);
        std::cout << "[C] TransferSync(READ): " << s << std::endl;
        if (s == 0) {
            float host_dst[16] = {0};
            aclrtMemcpy(host_dst, BYTES, local_buf, BYTES, ACL_MEMCPY_DEVICE_TO_HOST);
            float sum = 0;
            for (float v : host_dst) sum += v;
            std::cout << "[C] READ sum=" << sum << " (expect 120)" << std::endl;
        }
    }

    // --- TEST 2: WRITE to server ---
    {
        float host_src[16];
        for (int i = 0; i < 16; ++i) host_src[i] = 100.0f + static_cast<float>(i);
        aclrtMemcpy(local_buf, BYTES, host_src, BYTES, ACL_MEMCPY_HOST_TO_DEVICE);

        hixl::TransferOpDesc desc{};
        desc.local_addr = local_mem.addr;
        desc.remote_addr = static_cast<uintptr_t>(remote_addr);
        desc.len = BYTES;
        std::vector<hixl::TransferOpDesc> descs{desc};

        s = cli.TransferSync(hixl::AscendString(SRV_ID), hixl::WRITE, descs, 15000);
        std::cout << "[C] TransferSync(WRITE): " << s << std::endl;
    }

    cli.DeregisterMem(local_mh);
    cli.Disconnect(hixl::AscendString(SRV_ID), 5000);
    cli.Finalize();
    aclrtFree(local_buf);

    // signal server done
    uint8_t done = 1;
    write(write_fd, &done, 1);

    return (s == 0) ? 0 : 5;
}

int main() {
    // pipe: srv_to_cli and cli_to_srv
    int srv2cli[2], cli2srv[2];
    pipe(srv2cli);
    pipe(cli2srv);

    pid_t srv_pid = fork();
    if (srv_pid == 0) {
        close(srv2cli[0]); close(cli2srv[1]);
        _exit(run_server(srv2cli[1], cli2srv[0]));
    }
    close(srv2cli[1]); close(cli2srv[0]);

    pid_t cli_pid = fork();
    if (cli_pid == 0) {
        _exit(run_client(srv2cli[0], cli2srv[1]));
    }

    int cli_st = 0;
    waitpid(cli_pid, &cli_st, 0);
    kill(srv_pid, SIGTERM);
    waitpid(srv_pid, nullptr, 0);

    bool ok = WIFEXITED(cli_st) && WEXITSTATUS(cli_st) == 0;
    std::cout << "\n>>> " << (ok ? "PASS" : "FAIL") << std::endl;
    return ok ? 0 : 1;
}
