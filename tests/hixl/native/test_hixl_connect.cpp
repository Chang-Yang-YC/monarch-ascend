#include <hixl/hixl.h>
#include <hixl/hixl_types.h>
#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <iostream>
#include <unistd.h>
#include <sys/wait.h>
#include <cstring>
#include <cstdlib>

// Each test in a completely fresh process pair to avoid state pollution.
struct Config {
    const char* name;
    const char* server_id;
    int server_device;
    const char* client_id;
    int client_device;
    const char* connect_to;
};

int main(int argc, char** argv) {
    // If called with arguments, run as server or client
    if (argc >= 4 && strcmp(argv[1], "--server") == 0) {
        int dev = atoi(argv[3]);
        aclInit(nullptr);
        aclrtSetDevice(dev);
        hixl::Hixl srv;
        std::map<hixl::AscendString, hixl::AscendString> opts;
        auto s = srv.Initialize(hixl::AscendString(argv[2]), opts);
        std::cout << "  [S] Init(" << argv[2] << ", dev=" << dev << "): " << s << std::endl;
        if (s != 0) return 1;
        sleep(15);
        srv.Finalize();
        return 0;
    }
    if (argc >= 5 && strcmp(argv[1], "--client") == 0) {
        int dev = atoi(argv[3]);
        aclInit(nullptr);
        aclrtSetDevice(dev);
        hixl::Hixl cli;
        std::map<hixl::AscendString, hixl::AscendString> opts;
        auto s = cli.Initialize(hixl::AscendString(argv[2]), opts);
        std::cout << "  [C] Init(" << argv[2] << ", dev=" << dev << "): " << s << std::endl;
        if (s != 0) return 1;
        // sleep briefly for server to be ready
        sleep(2);
        std::cout << "  [C] Connect(" << argv[4] << ")..." << std::endl;
        s = cli.Connect(hixl::AscendString(argv[4]), 10000);
        std::cout << "  [C] Connect: " << s;
        if (s == 0) std::cout << " (SUCCESS)";
        else if (s == 103900) std::cout << " (PARAM_INVALID)";
        else if (s == 503900) std::cout << " (FAILED)";
        else if (s == 103901) std::cout << " (TIMEOUT)";
        else if (s == 103905) std::cout << " (UNSUPPORTED)";
        else if (s == 203900) std::cout << " (RESOURCE_EXHAUSTED)";
        std::cout << std::endl;
        if (s == 0) cli.Disconnect(hixl::AscendString(argv[4]), 5000);
        cli.Finalize();
        return (s == 0) ? 0 : 1;
    }

    // === ORCHESTRATOR ===
    std::cout << "=== HIXL Connect test (clean processes) ===" << std::endl;

    Config tests[] = {
        // HCCS: ip-only, same device
        {"HCCS: ip-only, dev0+dev0",
         "192.168.0.117", 0, "192.168.0.117", 0, "192.168.0.117"},

        // HCCS: ip:0, different devices
        {"HCCS: ip:0, dev0+dev1",
         "192.168.0.117:0", 0, "192.168.0.117:0", 1, "192.168.0.117:0"},

        // RoCE: ip:port, same device
        {"RoCE: ip:port, dev0+dev0",
         "192.168.0.117:30001", 0, "192.168.0.117:30002", 0, "192.168.0.117:30001"},

        // Different devices, ip:port
        {"RoCE: ip:port, dev0+dev1",
         "192.168.0.117:30001", 0, "192.168.0.117:30002", 1, "192.168.0.117:30001"},

        // Server dev0, client dev1, ip-only
        {"HCCS: ip-only, dev0+dev1",
         "192.168.0.117", 0, "192.168.0.117", 1, "192.168.0.117"},
    };

    char exe[1024];
    readlink("/proc/self/exe", exe, sizeof(exe));

    for (const auto& t : tests) {
        std::cout << "\n--- " << t.name << " ---" << std::endl;

        pid_t srv_pid = fork();
        if (srv_pid == 0) {
            char dev[8]; snprintf(dev, sizeof(dev), "%d", t.server_device);
            execl(exe, exe, "--server", t.server_id, dev, nullptr);
            _exit(1);
        }

        sleep(3);  // let server initialize

        pid_t cli_pid = fork();
        if (cli_pid == 0) {
            char dev[8]; snprintf(dev, sizeof(dev), "%d", t.client_device);
            execl(exe, exe, "--client", t.client_id, dev, t.connect_to, nullptr);
            _exit(1);
        }

        int cli_st = 0;
        waitpid(cli_pid, &cli_st, 0);
        kill(srv_pid, SIGTERM);
        waitpid(srv_pid, nullptr, 0);

        if (WIFEXITED(cli_st) && WEXITSTATUS(cli_st) == 0) {
            std::cout << ">>> SUCCESS: " << t.name << std::endl;
        }
    }

    return 0;
}
