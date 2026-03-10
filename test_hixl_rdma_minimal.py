#!/usr/bin/env python3
"""
Minimal HIXL RDMA test — just create an RDMABuffer on NPU.
"""
import os, sys
os.environ["PYTHONUNBUFFERED"] = "1"

import asyncio
import torch
try:
    import torch_npu
except ImportError:
    print("ERROR: torch_npu not available", flush=True)
    sys.exit(1)

print(f"torch {torch.__version__}, npu available: {torch.npu.is_available()}", flush=True)

from monarch._rust_bindings.rdma import _RdmaBuffer
print(f"rdma_supported: {_RdmaBuffer.rdma_supported()}", flush=True)

from monarch.actor import Actor, endpoint, this_host
from monarch.rdma import RDMABuffer


class RdmaTestActor(Actor):
    def __init__(self):
        print("  [RdmaTestActor] init...", flush=True)

    @endpoint
    async def create_buffer(self) -> str:
        print("  [RdmaTestActor] creating tensor on npu...", flush=True)
        t = torch.randn(4, 4, device="npu")
        print(f"  [RdmaTestActor] tensor: {t.shape} on {t.device}", flush=True)

        print("  [RdmaTestActor] creating RDMABuffer...", flush=True)
        try:
            buf = RDMABuffer(t.view(torch.uint8).flatten())
            print(f"  [RdmaTestActor] RDMABuffer created: {buf}", flush=True)
            return "OK"
        except Exception as e:
            print(f"  [RdmaTestActor] RDMABuffer FAILED: {e}", flush=True)
            return f"FAIL: {e}"

    @endpoint
    async def ping(self) -> str:
        return "pong"


async def main():
    print("=" * 50, flush=True)
    print("Minimal HIXL RDMA test", flush=True)
    print("=" * 50, flush=True)

    print("[1] Creating mesh...", flush=True)
    mesh = this_host().spawn_procs(per_host={"gpus": 1})

    print("[2] Spawning actor...", flush=True)
    actor = mesh.spawn("test", RdmaTestActor)

    print("[3] Ping test...", flush=True)
    result = await actor.ping.call_one()
    print(f"  ping result: {result}", flush=True)

    print("[4] RDMA buffer creation test (30s timeout)...", flush=True)
    try:
        result = await asyncio.wait_for(
            actor.create_buffer.call_one(),
            timeout=30.0,
        )
        print(f"  result: {result}", flush=True)
    except asyncio.TimeoutError:
        print("  TIMEOUT: RDMABuffer creation hung", flush=True)
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)

    print("Test complete.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
