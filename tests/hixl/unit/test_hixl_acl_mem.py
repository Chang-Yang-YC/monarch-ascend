#!/usr/bin/env python3
"""
Test HIXL transfer with raw aclrtMalloc memory instead of torch tensors.
This isolates whether torch_npu's caching allocator causes HIXL failures.
"""
import asyncio
import os
import sys
import ctypes

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("HCCL_INTRA_ROCE_ENABLE", "1")

from monarch.actor import Actor, endpoint, this_host
from monarch.rdma import RDMABuffer


def _acl_malloc(dev_id: int, nbytes: int):
    """Allocate device memory via aclrtMalloc, return (ptr, lib)."""
    lib = ctypes.CDLL("libascendcl.so")
    lib.aclInit.restype = ctypes.c_int
    lib.aclrtSetDevice.argtypes = [ctypes.c_int]
    lib.aclrtSetDevice.restype = ctypes.c_int
    lib.aclrtMalloc.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    lib.aclrtMalloc.restype = ctypes.c_int
    lib.aclrtMemset.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_size_t]
    lib.aclrtMemset.restype = ctypes.c_int
    lib.aclrtMemcpy.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t,
        ctypes.c_void_p, ctypes.c_size_t,
        ctypes.c_int,
    ]
    lib.aclrtMemcpy.restype = ctypes.c_int

    lib.aclInit(None)
    lib.aclrtSetDevice(dev_id)

    ptr = ctypes.c_void_p()
    ACL_MEM_MALLOC_NORMAL_ONLY = 2
    ret = lib.aclrtMalloc(ctypes.byref(ptr), nbytes, ACL_MEM_MALLOC_NORMAL_ONLY)
    assert ret == 0, f"aclrtMalloc failed: {ret}"
    lib.aclrtMemset(ptr, nbytes, 0, nbytes)
    return ptr.value, lib


class AclProducer(Actor):
    def __init__(self):
        dev_id = int(os.environ.get("MONARCH_NPU_DEVICE", "0"))
        self.nbytes = 64
        self.ptr, self.lib = _acl_malloc(dev_id, self.nbytes)
        # Write known pattern: 16 floats of 1.0
        import struct
        host = struct.pack("16f", *([1.0] * 16))
        host_buf = ctypes.create_string_buffer(host)
        ACL_MEMCPY_HOST_TO_DEVICE = 1
        self.lib.aclrtMemcpy(
            ctypes.c_void_p(self.ptr), self.nbytes,
            host_buf, self.nbytes,
            ACL_MEMCPY_HOST_TO_DEVICE,
        )
        self.buf = None
        print(f"[AclProducer] dev={dev_id} ptr={hex(self.ptr)} size={self.nbytes}")

    @endpoint
    async def get_handle(self) -> RDMABuffer:
        if self.buf is None:
            from monarch._rust_bindings.monarch_rdma.rdma import RawLocalMemory
            raw = RawLocalMemory(self.ptr, self.nbytes)
            self.buf = RDMABuffer(raw)
        return self.buf

    @endpoint
    async def read_back(self) -> float:
        import struct
        host = ctypes.create_string_buffer(self.nbytes)
        ACL_MEMCPY_DEVICE_TO_HOST = 2
        self.lib.aclrtMemcpy(
            host, self.nbytes,
            ctypes.c_void_p(self.ptr), self.nbytes,
            ACL_MEMCPY_DEVICE_TO_HOST,
        )
        vals = struct.unpack("16f", host.raw)
        return sum(vals)


class AclConsumer(Actor):
    @endpoint
    async def push(self, remote: RDMABuffer) -> str:
        dev_id = int(os.environ.get("MONARCH_NPU_DEVICE", "1"))
        nbytes = 64
        ptr, lib = _acl_malloc(dev_id, nbytes)
        # Write 2.0 pattern
        import struct
        host = struct.pack("16f", *([2.0] * 16))
        host_buf = ctypes.create_string_buffer(host)
        lib.aclrtMemcpy(
            ctypes.c_void_p(ptr), nbytes,
            host_buf, nbytes,
            1,  # HOST_TO_DEVICE
        )
        print(f"[AclConsumer] dev={dev_id} ptr={hex(ptr)} writing to remote...")
        from monarch._rust_bindings.monarch_rdma.rdma import RawLocalMemory
        raw = RawLocalMemory(ptr, nbytes)
        await remote.write_from(raw, timeout=20)
        return "OK"


def use_npu(dev_id: int):
    def _bootstrap():
        os.environ["MONARCH_NPU_DEVICE"] = str(dev_id)
        os.environ["HCCL_INTRA_ROCE_ENABLE"] = "1"
    return _bootstrap


async def main():
    print("=" * 60)
    print("HIXL test with raw aclrtMalloc memory (no torch)")
    print("=" * 60)

    host = this_host()
    producer_mesh = host.spawn_procs(per_host={"gpus": 1}, bootstrap=use_npu(0))
    consumer_mesh = host.spawn_procs(per_host={"gpus": 1}, bootstrap=use_npu(1))

    producer = producer_mesh.spawn("producer", AclProducer)
    consumer = consumer_mesh.spawn("consumer", AclConsumer)

    print("[1] Get remote handle...")
    handle = await producer.get_handle.call_one()
    init_sum = await producer.read_back.call_one()
    print(f"    initial sum = {init_sum} (expect 16.0)")

    await asyncio.sleep(2)

    print("[2] write_from via HIXL...")
    result = await consumer.push.call_one(handle)
    print(f"    result = {result}")

    final_sum = await producer.read_back.call_one()
    print(f"    final sum = {final_sum} (expect 32.0)")
    print("PASS" if abs(final_sum - 32.0) < 0.1 else "FAIL")


if __name__ == "__main__":
    asyncio.run(main())
