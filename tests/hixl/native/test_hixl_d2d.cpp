/**
 * HIXL CS D2D test over HCCS between NPU 0 and NPU 1.
 * Tries multiple endpoint configurations to find one that works.
 */
#include <cs/hixl_cs.h>
#include <hcomm/hcomm_res_defs.h>
#include <hccl/hccl_res.h>
#include <acl/acl.h>
#include <acl/acl_rt.h>

#include <unistd.h>
#include <sys/wait.h>
#include <arpa/inet.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

static constexpr size_t BYTES = 16 * sizeof(float);

struct TestCase {
    const char* name;
    CommProtocol proto;
    CommAddrType addr_type;
    bool set_ip_in_addr;
    bool zero_init;
};

static int try_server_create(const char* name, int dev,
                              CommProtocol proto, CommAddrType addr_type,
                              bool set_ip_in_addr, bool zero_init) {
    EndpointDesc ep;
    if (zero_init) {
        memset(&ep, 0, sizeof(ep));
    } else {
        EndpointDescInit(&ep, 1);
    }

    ep.protocol = proto;
    ep.commAddr.type = addr_type;
    if (set_ip_in_addr) {
        inet_pton(AF_INET, "192.168.0.117", &ep.commAddr.addr);
    } else {
        ep.commAddr.id = static_cast<uint32_t>(dev);
    }
    ep.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    ep.loc.device.devPhyId = static_cast<uint32_t>(dev);
    ep.loc.device.superDevId = 0;
    ep.loc.device.serverIdx = 0;
    ep.loc.device.superPodIdx = 0;

    HixlServerDesc srv_desc{};
    srv_desc.server_ip = "192.168.0.117";
    srv_desc.server_port = 30010;
    srv_desc.endpoint_list = &ep;
    srv_desc.endpoint_list_num = 1;

    HixlServerConfig srv_cfg{};
    HixlServerHandle srv_handle = nullptr;

    auto s = HixlCSServerCreate(&srv_desc, &srv_cfg, &srv_handle);
    std::cout << "  [" << name << "] ServerCreate: " << s << std::endl;
    if (s == HIXL_SUCCESS && srv_handle) {
        HixlCSServerDestroy(srv_handle);
    }
    return static_cast<int>(s);
}

static int try_server_no_endpoint(const char* name) {
    HixlServerDesc srv_desc{};
    srv_desc.server_ip = "192.168.0.117";
    srv_desc.server_port = 30010;
    srv_desc.endpoint_list = nullptr;
    srv_desc.endpoint_list_num = 0;

    HixlServerConfig srv_cfg{};
    HixlServerHandle srv_handle = nullptr;

    auto s = HixlCSServerCreate(&srv_desc, &srv_cfg, &srv_handle);
    std::cout << "  [" << name << "] ServerCreate(no ep): " << s << std::endl;
    if (s == HIXL_SUCCESS && srv_handle) {
        HixlCSServerDestroy(srv_handle);
    }
    return static_cast<int>(s);
}

static int try_server_null_ip(const char* name, int dev) {
    EndpointDesc ep;
    memset(&ep, 0, sizeof(ep));
    ep.protocol = COMM_PROTOCOL_HCCS;
    ep.commAddr.type = COMM_ADDR_TYPE_ID;
    ep.commAddr.id = static_cast<uint32_t>(dev);
    ep.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    ep.loc.device.devPhyId = static_cast<uint32_t>(dev);

    HixlServerDesc srv_desc{};
    srv_desc.server_ip = nullptr;
    srv_desc.server_port = 0;
    srv_desc.endpoint_list = &ep;
    srv_desc.endpoint_list_num = 1;

    HixlServerConfig srv_cfg{};
    HixlServerHandle srv_handle = nullptr;

    auto s = HixlCSServerCreate(&srv_desc, &srv_cfg, &srv_handle);
    std::cout << "  [" << name << "] ServerCreate(null ip): " << s << std::endl;
    if (s == HIXL_SUCCESS && srv_handle) {
        HixlCSServerDestroy(srv_handle);
    }
    return static_cast<int>(s);
}

int main() {
    aclInit(nullptr);
    aclrtSetDevice(0);

    TestCase tests[] = {
        {"HCCS+ID+zero",       COMM_PROTOCOL_HCCS, COMM_ADDR_TYPE_ID,   false, true},
        {"HCCS+ID+0xFF",       COMM_PROTOCOL_HCCS, COMM_ADDR_TYPE_ID,   false, false},
        {"HCCS+IPv4+zero",     COMM_PROTOCOL_HCCS, COMM_ADDR_TYPE_IP_V4, true, true},
        {"HCCS+IPv4+0xFF",     COMM_PROTOCOL_HCCS, COMM_ADDR_TYPE_IP_V4, true, false},
        {"ROCE+IPv4+zero",     COMM_PROTOCOL_ROCE, COMM_ADDR_TYPE_IP_V4, true, true},
        {"ROCE+IPv4+0xFF",     COMM_PROTOCOL_ROCE, COMM_ADDR_TYPE_IP_V4, true, false},
        {"RESERVED+ID+zero",   COMM_PROTOCOL_RESERVED, COMM_ADDR_TYPE_ID, false, true},
        {"RESERVED+IPv4+zero", COMM_PROTOCOL_RESERVED, COMM_ADDR_TYPE_IP_V4, true, true},
    };

    std::cout << "=== Endpoint parameter sweep ===" << std::endl;
    for (const auto& t : tests) {
        try_server_create(t.name, 0, t.proto, t.addr_type, t.set_ip_in_addr, t.zero_init);
    }

    std::cout << "\n=== Special cases ===" << std::endl;
    try_server_no_endpoint("no_endpoint");
    try_server_null_ip("null_ip_hccs", 0);

    return 0;
}
