#!/usr/bin/env python3
"""Diagnostic: compare Rust-initialized engine vs ctypes-initialized engine."""
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
from monarch._src.rdma.xdma import XDMABuffer as RDMABuffer

BUF_SIZE = 4096

class Owner(Actor):
    def __init__(self, device_id: int = 0):
        os.environ["MONARCH_NPU_DEVICE"] = str(device_id)
        torch.npu.set_device(device_id)
        self._data = torch.ones(BUF_SIZE // 4, dtype=torch.float32, device=f"npu:{device_id}").view(torch.uint8)
        torch.npu.synchronize()
        self._buf = None
        print(f"[Owner] addr={hex(self._data.data_ptr())}", flush=True)

    @endpoint
    async def get_buffer(self) -> RDMABuffer:
        self._buf = RDMABuffer(self._data)
        return self._buf

    @endpoint
    async def get_rust_engine_info(self) -> str:
        from monarch._rust_bindings.rdma import _RdmaBuffer
        diag = _RdmaBuffer.hixl_engine_diag()
        if diag is None:
            return "none"
        return f"{diag[0]},{diag[1]}"


class Consumer(Actor):
    def __init__(self, device_id: int = 1):
        os.environ["MONARCH_NPU_DEVICE"] = str(device_id)
        torch.npu.set_device(device_id)
        self._dev = device_id
        self._local = torch.zeros(BUF_SIZE // 4, dtype=torch.float32, device=f"npu:{device_id}").view(torch.uint8)
        torch.npu.synchronize()
        print(f"[Consumer] addr={hex(self._local.data_ptr())}", flush=True)

    @endpoint
    async def test_rust_read(self, buf: RDMABuffer) -> str:
        try:
            await buf.read_into(self._local)
            return f"RUST_OK: byte0={self._local[0].item()}"
        except Exception as e:
            return f"RUST_FAIL: {e}"

    @endpoint
    async def get_rust_engine_info(self) -> str:
        from monarch._rust_bindings.rdma import _RdmaBuffer
        diag = _RdmaBuffer.hixl_engine_diag()
        if diag is None:
            return "none"
        return f"{diag[0]},{diag[1]}"

    @endpoint
    async def test_ctypes_on_rust_engine(self, remote_eid: str, remote_addr: int) -> str:
        """Use ctypes to transfer on the EXISTING Rust-managed engine."""
        from monarch._rust_bindings.rdma import _RdmaBuffer
        diag = _RdmaBuffer.hixl_engine_diag()
        if diag is None:
            return "FAIL: no engine"
        engine_ptr, my_eid = diag
        print(f"[Consumer] Using Rust engine: ptr={hex(engine_ptr)} eid={my_eid}", flush=True)

        lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "libtest_hixl.so")
        lib = ctypes.CDLL(lib_path)
        lib.hixl_register_mem.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
        lib.hixl_register_mem.restype = ctypes.c_int
        lib.hixl_connect.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.hixl_connect.restype = ctypes.c_int
        lib.hixl_transfer_read.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
        lib.hixl_transfer_read.restype = ctypes.c_int

        ctx = ctypes.c_void_p(engine_ptr)
        lib.hixl_register_mem(ctx, self._local.data_ptr(), self._local.numel())
        ret_c = lib.hixl_connect(ctx, remote_eid.encode())
        print(f"[Consumer] ctypes connect: {ret_c}", flush=True)
        time.sleep(1)
        ret_t = lib.hixl_transfer_read(ctx, remote_eid.encode(),
                                        self._local.data_ptr(), remote_addr, self._local.numel())
        print(f"[Consumer] ctypes transfer: {ret_t}", flush=True)
        if ret_t == 0:
            return f"CTYPES_ON_RUST_OK: byte0={self._local[0].item()}"
        return f"CTYPES_ON_RUST_FAIL: connect={ret_c} transfer={ret_t}"


async def main():
    print("=" * 60)
    print("Diagnostic: Rust vs ctypes on same engine")
    print("=" * 60)

    mesh1 = this_host().spawn_procs(per_host={"procs": 1})
    mesh2 = this_host().spawn_procs(per_host={"procs": 1})

    owner = mesh1.spawn("owner", Owner, 0)
    consumer = mesh2.spawn("consumer", Consumer, 1)

    buf = await owner.get_buffer.call_one()
    print("[main] Got buffer", flush=True)

    # Test 1: Rust path (expected to fail)
    rust_result = await consumer.test_rust_read.call_one(buf)
    print(f"[main] Rust read: {rust_result}", flush=True)

    # Get engine info after Rust path initialized engines
    owner_info = await owner.get_rust_engine_info.call_one()
    consumer_info = await consumer.get_rust_engine_info.call_one()
    print(f"[main] Owner engine: {owner_info}", flush=True)
    print(f"[main] Consumer engine: {consumer_info}", flush=True)

    # Test 2: ctypes on Rust-managed engine
    if "," in owner_info:
        remote_eid = owner_info.split(",")[1]
        remote_addr = int(buf._buffer.external_backend_info()[1]) if buf._buffer.external_backend_info() else 0
        ctypes_result = await consumer.test_ctypes_on_rust_engine.call_one(remote_eid, remote_addr)
    else:
        ctypes_result = "SKIP: no engine info"
    print(f"[main] ctypes on Rust engine: {ctypes_result}", flush=True)

    print("=" * 60)
    print(f"Rust:   {rust_result}")
    print(f"ctypes: {ctypes_result}")
    print("=" * 60)


if __name__ == "__main__":
    from monarch._src.actor.actor_mesh import context
    context()
    asyncio.run(main())
