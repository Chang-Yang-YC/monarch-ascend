#!/usr/bin/env python3
"""Test HiXL via ctypes ONLY from within Monarch actor processes.
No Rust RdmaBuffer involved. Verifies if the Monarch process environment
breaks HiXL."""
import os, sys, asyncio, ctypes, time

os.environ.setdefault("HCCL_INTRA_ROCE_ENABLE", "1")
os.environ["PYTHONPATH"] = os.pathsep.join(sys.path)
_hixl_lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build")
if os.path.isdir(_hixl_lib_dir):
    os.environ["LD_LIBRARY_PATH"] = _hixl_lib_dir + ":" + os.environ.get("LD_LIBRARY_PATH", "")

import torch
try:
    import torch_npu
except ImportError:
    sys.exit("torch_npu not available")

from monarch.actor import Actor, endpoint, this_host

BUF_SIZE = 4096
COORD_FILE = "/tmp/hixl_monarch_ctypes_coord"

def setup_lib():
    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "libtest_hixl.so")
    lib = ctypes.CDLL(lib_path)
    lib.hixl_init_engine.argtypes = [ctypes.c_int, ctypes.c_char_p]
    lib.hixl_init_engine.restype = ctypes.c_void_p
    lib.hixl_connect.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.hixl_connect.restype = ctypes.c_int
    lib.hixl_register_mem.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
    lib.hixl_register_mem.restype = ctypes.c_int
    lib.hixl_transfer_read.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
    lib.hixl_transfer_read.restype = ctypes.c_int
    lib.hixl_cleanup.argtypes = [ctypes.c_void_p]
    return lib


class Server(Actor):
    """Holds source data on NPU 0."""
    def __init__(self, device_id: int = 0):
        os.environ["MONARCH_NPU_DEVICE"] = str(device_id)
        torch.npu.set_device(device_id)
        self._dev = device_id
        self._data = torch.ones(BUF_SIZE // 4, dtype=torch.float32, device=f"npu:{device_id}").view(torch.uint8)
        torch.npu.synchronize()
        self._eid = f"127.0.0.1:{65000 + device_id}"
        print(f"[Server] PID={os.getpid()} dev={device_id} addr={hex(self._data.data_ptr())} eid={self._eid}", flush=True)

    @endpoint
    async def init_hixl(self) -> str:
        lib = setup_lib()
        self._ctx = lib.hixl_init_engine(self._dev, self._eid.encode())
        if not self._ctx:
            return "FAIL: init null"
        lib.hixl_register_mem(self._ctx, self._data.data_ptr(), self._data.numel())
        self._lib = lib
        # Write coord file
        with open(COORD_FILE, 'w') as f:
            f.write(f"{self._eid},{self._data.data_ptr()}")
        return f"OK eid={self._eid} addr={hex(self._data.data_ptr())}"

    @endpoint
    async def connect_to(self, remote_eid: str) -> str:
        ret = self._lib.hixl_connect(self._ctx, remote_eid.encode())
        return f"connect={ret}"

    @endpoint
    async def cleanup(self) -> None:
        self._lib.hixl_cleanup(self._ctx)


class Client(Actor):
    """Reads from Server on NPU 1."""
    def __init__(self, device_id: int = 1):
        os.environ["MONARCH_NPU_DEVICE"] = str(device_id)
        torch.npu.set_device(device_id)
        self._dev = device_id
        self._local = torch.zeros(BUF_SIZE // 4, dtype=torch.float32, device=f"npu:{device_id}").view(torch.uint8)
        torch.npu.synchronize()
        self._eid = f"127.0.0.1:{65000 + device_id}"
        print(f"[Client] PID={os.getpid()} dev={device_id} addr={hex(self._local.data_ptr())} eid={self._eid}", flush=True)

    @endpoint
    async def do_transfer(self) -> str:
        lib = setup_lib()
        ctx = lib.hixl_init_engine(self._dev, self._eid.encode())
        if not ctx:
            return "FAIL: init null"
        lib.hixl_register_mem(ctx, self._local.data_ptr(), self._local.numel())

        # Read coord file
        with open(COORD_FILE) as f:
            parts = f.read().strip().split(",")
        remote_eid = parts[0]
        remote_addr = int(parts[1])

        lib.hixl_connect(ctx, remote_eid.encode())
        time.sleep(1)

        ret = lib.hixl_transfer_read(ctx, remote_eid.encode(),
                                      self._local.data_ptr(), remote_addr, self._local.numel())
        lib.hixl_cleanup(ctx)
        if ret == 0:
            val = self._local[0].item()
            return f"READ_OK: byte0={val}"
        return f"READ_FAIL: ret={ret}"


async def main():
    print("=" * 60)
    print("Test: ctypes-only HiXL from Monarch actors")
    print("=" * 60)

    try: os.unlink(COORD_FILE)
    except: pass

    mesh1 = this_host().spawn_procs(per_host={"procs": 1})
    mesh2 = this_host().spawn_procs(per_host={"procs": 1})

    server = mesh1.spawn("server", Server, 0)
    client = mesh2.spawn("client", Client, 1)

    # Init and register on server
    init_result = await server.init_hixl.call_one()
    print(f"[main] Server init: {init_result}", flush=True)

    # Server connects to client
    client_eid = "127.0.0.1:65001"
    connect_result = await server.connect_to.call_one(client_eid)
    print(f"[main] Server connect: {connect_result}", flush=True)

    # Client does everything: init, register, connect, transfer
    transfer_result = await client.do_transfer.call_one()
    print(f"[main] Client transfer: {transfer_result}", flush=True)

    await server.cleanup.call_one()

    print("=" * 60)
    print(f"Result: {transfer_result}")
    print("=" * 60)


if __name__ == "__main__":
    from monarch._src.actor.actor_mesh import context
    context()
    asyncio.run(main())
