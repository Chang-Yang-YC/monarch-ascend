"""
End-to-end test for XDMABuffer — the NPU-specific single-sided communication path.

Verifies that XDMABuffer works correctly via the transport plugin registry,
completely independent of rdma.py.

Two proc meshes on the same host, each bound to a different NPU.

Run:
    source /root/hzz/cann-9.0.0-beta.1/set_env.sh
    HCCL_INTRA_ROCE_ENABLE=1 MONARCH_TRANSPORT_BACKEND=hixl \
    python tests/hixl/e2e/test_xdma_buffer.py
"""

import os
import sys
import time
import asyncio

os.environ.setdefault("HCCL_INTRA_ROCE_ENABLE", "1")
os.environ.setdefault("MONARCH_TRANSPORT_BACKEND", "hixl")
os.environ["PYTHONPATH"] = os.pathsep.join(sys.path)

import torch
import torch_npu  # noqa: F401
from monarch.actor import Actor, endpoint, this_host
from monarch._src.rdma.xdma import XDMABuffer
from monarch._src.rdma.transport import get_transport


class Producer(Actor):
    def __init__(self, device_id: int):
        os.environ["MONARCH_NPU_DEVICE"] = str(device_id)
        os.environ.setdefault("HCCL_INTRA_ROCE_ENABLE", "1")
        os.environ.setdefault("MONARCH_TRANSPORT_BACKEND", "hixl")
        torch.npu.set_device(device_id)
        print(f"[Producer PID={os.getpid()}] bound to NPU {device_id}", flush=True)

    @endpoint
    async def create_buffer(self, fill_value: float) -> XDMABuffer:
        dev_id = int(os.environ.get("MONARCH_NPU_DEVICE", "0"))
        t = torch.ones(1024, dtype=torch.float32, device=f"npu:{dev_id}")
        t.fill_(fill_value)
        torch.npu.synchronize()
        buf = XDMABuffer(t)
        info = buf._buffer.external_backend_info()
        print(
            f"[Producer] XDMABuffer: val={fill_value}, "
            f"addr={hex(info[1]) if info else 'N/A'}, size={buf.size()}",
            flush=True,
        )
        return buf

    @endpoint
    async def connect_to_peer(self, peer_engine_id: str) -> str:
        try:
            be = get_transport()
            if be is not None:
                be.ensure_peer_connected(peer_engine_id)
                return f"CONNECTED_TO {peer_engine_id}"
            return "NO_BACKEND"
        except Exception as e:
            return f"CONNECT_FAIL: {e}"


class Consumer(Actor):
    def __init__(self, device_id: int):
        os.environ["MONARCH_NPU_DEVICE"] = str(device_id)
        os.environ.setdefault("HCCL_INTRA_ROCE_ENABLE", "1")
        os.environ.setdefault("MONARCH_TRANSPORT_BACKEND", "hixl")
        torch.npu.set_device(device_id)
        print(f"[Consumer PID={os.getpid()}] bound to NPU {device_id}", flush=True)

    @endpoint
    async def get_engine_id(self) -> str:
        be = get_transport()
        if be is None:
            return "NO_BACKEND"
        return be.get_engine_id()

    @endpoint
    async def read_from(self, buf: XDMABuffer) -> str:
        dev_id = int(os.environ.get("MONARCH_NPU_DEVICE", "0"))
        dst = torch.zeros(1024, dtype=torch.float32, device=f"npu:{dev_id}")
        torch.npu.synchronize()
        try:
            buf.read_into(dst).get(timeout=30)
            torch.npu.synchronize()
            vals = dst[:4].cpu().tolist()
            return f"READ_OK vals={vals}"
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"READ_FAIL: {e}"

    @endpoint
    async def write_to(self, buf: XDMABuffer, value: float) -> str:
        dev_id = int(os.environ.get("MONARCH_NPU_DEVICE", "0"))
        src = torch.ones(1024, dtype=torch.float32, device=f"npu:{dev_id}")
        src.fill_(value)
        torch.npu.synchronize()
        try:
            buf.write_from(src).get(timeout=30)
            return "WRITE_OK"
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"WRITE_FAIL: {e}"

    @endpoint
    async def read_and_verify(self, buf: XDMABuffer, expected: float) -> str:
        dev_id = int(os.environ.get("MONARCH_NPU_DEVICE", "0"))
        dst = torch.zeros(1024, dtype=torch.float32, device=f"npu:{dev_id}")
        torch.npu.synchronize()
        try:
            buf.read_into(dst).get(timeout=30)
            torch.npu.synchronize()
            val = dst[0].item()
            if abs(val - expected) < 0.01:
                return f"VERIFY_OK val={val}"
            return f"VERIFY_MISMATCH expected={expected} got={val}"
        except Exception as e:
            return f"VERIFY_FAIL: {e}"


async def run_tests():
    print("=" * 60)
    print("  XDMABuffer E2E Test (NPU:0 <-> NPU:1)")
    print("=" * 60)

    host = this_host()
    results = []

    mesh0 = host.spawn_procs(per_host={"procs": 1})
    mesh1 = host.spawn_procs(per_host={"procs": 1})

    producer = mesh0.spawn("producer", Producer, 0)
    consumer = mesh1.spawn("consumer", Consumer, 1)

    # Create a single shared buffer for all tests.
    # HCCS requires aligned addresses — using a single buffer avoids
    # alignment issues with the NPU memory allocator for subsequent allocs.
    print("\n--- Setup: Create buffer + bidirectional connect ---")
    buf = await producer.create_buffer.call_one(42.0)

    consumer_eid = await consumer.get_engine_id.call_one()
    print(f"  Consumer engine_id: {consumer_eid}")
    connect_result = await producer.connect_to_peer.call_one(consumer_eid)
    print(f"  Producer connect_to_peer: {connect_result}")

    # Test 1: READ — Consumer reads 42.0 from Producer
    print("\n--- Test 1: read_into (expect 42.0) ---")
    r1 = await consumer.read_from.call_one(buf)
    print(f"  Result: {r1}")
    results.append(("read", "READ_OK" in str(r1) and "42.0" in str(r1)))

    # Test 2: WRITE — Consumer overwrites with 99.0
    print("\n--- Test 2: write_from (write 99.0) ---")
    r2 = await consumer.write_to.call_one(buf, 99.0)
    print(f"  Write: {r2}")
    results.append(("write", "WRITE_OK" in str(r2)))

    # Test 3: VERIFY — Read back and confirm 99.0
    print("\n--- Test 3: verify read-after-write (expect 99.0) ---")
    r3 = await consumer.read_and_verify.call_one(buf, 99.0)
    print(f"  Verify: {r3}")
    results.append(("verify", "VERIFY_OK" in str(r3)))

    # Test 4: OVERWRITE — Consumer writes 77.0
    print("\n--- Test 4: second write (write 77.0) ---")
    r4 = await consumer.write_to.call_one(buf, 77.0)
    print(f"  Write: {r4}")
    results.append(("write2", "WRITE_OK" in str(r4)))

    # Test 5: VERIFY again
    print("\n--- Test 5: verify second write (expect 77.0) ---")
    r5 = await consumer.read_and_verify.call_one(buf, 77.0)
    print(f"  Verify: {r5}")
    results.append(("verify2", "VERIFY_OK" in str(r5)))

    # Summary
    print("\n" + "=" * 60)
    all_ok = all(ok for _, ok in results)
    for name, ok in results:
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    print("=" * 60)
    if all_ok:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    from monarch._src.actor.actor_mesh import context
    context()
    asyncio.run(run_tests())
