"""
Test: HIXL transfer via Python ctypes path, through the full monarch RDMABuffer API.

Two actor meshes on different NPU devices, one creates a buffer, the other writes to it
using the Python-side HIXL transfer (bypassing Rust HIXL TransferSync).
"""

import os
import sys
import time
import logging

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(levelname)s %(message)s")

os.environ.setdefault("HCCL_INTRA_ROCE_ENABLE", "1")
os.environ.setdefault("MONARCH_HIXL_LIB", "/root/monarch/libtest_hixl.so")

import torch
import torch_npu  # noqa: F401
import monarch
from monarch._src.actor.actor_mesh import Actor
from monarch._src.actor.endpoint import endpoint
from monarch._src.actor.proc_mesh import proc_mesh_blocking
from monarch._src.rdma.rdma import RDMABuffer


def use_npu(device_id: int):
    """Bootstrap function to set up NPU device."""
    os.environ["MONARCH_NPU_DEVICE"] = str(device_id)
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(device_id)
    torch.npu.set_device(0)  # 0 within the visible set
    print(f"[PID={os.getpid()}] NPU device {device_id} initialized", flush=True)


class Producer(Actor):
    @endpoint
    def create_buffer(self) -> RDMABuffer:
        dev = int(os.environ.get("MONARCH_NPU_DEVICE", "0"))
        t = torch.ones(1024, dtype=torch.float32, device=f"npu:0")
        t.fill_(42.0)
        torch.npu.synchronize()
        print(f"[Producer PID={os.getpid()}] tensor on npu:{dev}, val=42.0, ptr={t.data_ptr():#x}", flush=True)
        buf = RDMABuffer(t)
        hixl_info = buf._buffer.hixl_info()
        print(f"[Producer] buffer hixl_info={hixl_info}, size={buf.size()}", flush=True)
        return buf


class Consumer(Actor):
    @endpoint
    def write_to_buffer(self, buf: RDMABuffer) -> str:
        dev = int(os.environ.get("MONARCH_NPU_DEVICE", "0"))
        src = torch.ones(1024, dtype=torch.float32, device=f"npu:0")
        src.fill_(99.0)
        torch.npu.synchronize()
        print(f"[Consumer PID={os.getpid()}] writing val=99.0 from npu:{dev} ptr={src.data_ptr():#x}", flush=True)
        try:
            buf.write_from(src).get(timeout=30)
            return "SUCCESS"
        except Exception as e:
            return f"FAILED: {e}"

    @endpoint
    def read_from_buffer(self, buf: RDMABuffer) -> str:
        dev = int(os.environ.get("MONARCH_NPU_DEVICE", "0"))
        dst = torch.zeros(1024, dtype=torch.float32, device=f"npu:0")
        torch.npu.synchronize()
        print(f"[Consumer PID={os.getpid()}] reading into npu:{dev} ptr={dst.data_ptr():#x}", flush=True)
        try:
            buf.read_into(dst).get(timeout=30)
            val = dst[0].item()
            return f"SUCCESS: first_val={val}"
        except Exception as e:
            return f"FAILED: {e}"


def main():
    print("=" * 60)
    print("Test: Python HIXL transfer via monarch RDMABuffer")
    print("=" * 60)

    # Create two proc meshes on different NPU devices
    mesh0 = proc_mesh_blocking(1, gpu=0)
    mesh0.run_function(use_npu, 0)
    time.sleep(1)

    mesh1 = proc_mesh_blocking(1, gpu=0)
    mesh1.run_function(use_npu, 1)
    time.sleep(1)

    # Spawn actors
    producer_mesh = mesh0.spawn("producer", Producer)
    consumer_mesh = mesh1.spawn("consumer", Consumer)

    # Test 1: Producer creates buffer, consumer reads from it
    print("\n--- Test 1: Consumer reads from Producer's buffer ---")
    buf = producer_mesh[0].create_buffer.call_one()
    result = consumer_mesh[0].read_from_buffer.call_one(buf)
    print(f"READ result: {result}")

    # Test 2: Consumer writes to Producer's buffer
    print("\n--- Test 2: Consumer writes to Producer's buffer ---")
    buf2 = producer_mesh[0].create_buffer.call_one()
    result2 = consumer_mesh[0].write_to_buffer.call_one(buf2)
    print(f"WRITE result: {result2}")

    print("\n" + "=" * 60)
    if "SUCCESS" in str(result) and "SUCCESS" in str(result2):
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)


if __name__ == "__main__":
    main()
