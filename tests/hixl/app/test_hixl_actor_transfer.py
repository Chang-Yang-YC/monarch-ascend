#!/usr/bin/env python3
"""Minimal Monarch actor test for HiXL transfer: tests both read_into and write_from."""
import os, sys, asyncio

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
        os.environ.setdefault("HCCL_INTRA_ROCE_ENABLE", "1")
        torch.npu.set_device(device_id)
        self._dev = device_id
        self._data = torch.ones(BUF_SIZE // 4, dtype=torch.float32, device=f"npu:{device_id}").view(torch.uint8)
        torch.npu.synchronize()
        self._buf = None
        print(f"[Owner] PID={os.getpid()} dev={device_id} addr={hex(self._data.data_ptr())}", flush=True)

    @endpoint
    async def get_buffer(self) -> RDMABuffer:
        self._buf = RDMABuffer(self._data)
        return self._buf


class Consumer(Actor):
    def __init__(self, device_id: int = 1):
        os.environ["MONARCH_NPU_DEVICE"] = str(device_id)
        os.environ.setdefault("HCCL_INTRA_ROCE_ENABLE", "1")
        torch.npu.set_device(device_id)
        self._dev = device_id
        self._local = torch.zeros(BUF_SIZE // 4, dtype=torch.float32, device=f"npu:{device_id}").view(torch.uint8)
        torch.npu.synchronize()
        print(f"[Consumer] PID={os.getpid()} dev={device_id} addr={hex(self._local.data_ptr())}", flush=True)

    @endpoint
    async def test_read(self, buf: RDMABuffer) -> str:
        try:
            await buf.read_into(self._local)
            val = self._local[0].item()
            return f"READ_OK: byte0={val}"
        except Exception as e:
            return f"READ_FAIL: {e}"

    @endpoint
    async def test_write(self, buf: RDMABuffer) -> str:
        src = torch.full((BUF_SIZE // 4,), 99.0, dtype=torch.float32, device=f"npu:{self._dev}").view(torch.uint8)
        torch.npu.synchronize()
        try:
            await buf.write_from(src)
            return "WRITE_OK"
        except Exception as e:
            return f"WRITE_FAIL: {e}"


async def main():
    print("=" * 60)
    print("Test: HiXL transfer via Monarch actors (Rust path)")
    print("=" * 60)

    mesh1 = this_host().spawn_procs(per_host={"procs": 1})
    mesh2 = this_host().spawn_procs(per_host={"procs": 1})

    owner = mesh1.spawn("owner", Owner, 0)
    consumer = mesh2.spawn("consumer", Consumer, 1)

    buf = await owner.get_buffer.call_one()
    print("[main] Got buffer", flush=True)

    # Test WRITE direction first (Consumer pushes to Owner's buffer)
    print("[main] Testing WRITE...", flush=True)
    write_result = await consumer.test_write.call_one(buf)
    print(f"[main] WRITE: {write_result}", flush=True)

    # Test READ direction (Consumer pulls from Owner's buffer)
    print("[main] Testing READ...", flush=True)
    read_result = await consumer.test_read.call_one(buf)
    print(f"[main] READ: {read_result}", flush=True)

    print("=" * 60)
    print(f"WRITE: {write_result}")
    print(f"READ:  {read_result}")
    print("=" * 60)


if __name__ == "__main__":
    from monarch._src.actor.actor_mesh import context
    context()
    asyncio.run(main())
