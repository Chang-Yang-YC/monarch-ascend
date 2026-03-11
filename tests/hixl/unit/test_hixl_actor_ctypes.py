#!/usr/bin/env python3
"""
Test: call HIXL C++ directly from within monarch actors via ctypes.
If this works, the issue is in the Rust FFI layer.
"""
import asyncio
import os
import sys
import time
import struct
import ctypes

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("HCCL_INTRA_ROCE_ENABLE", "1")

import torch
try:
    import torch_npu
except ImportError:
    print("ERROR: torch_npu not available"); sys.exit(1)

from monarch.actor import Actor, endpoint, this_host

COORD = "/tmp/hixl_actor_coord"
NBYTES = 64
SRV_ENGINE = b"127.0.0.1:19200"
CLI_ENGINE = b"127.0.0.1:19201"

def load_hixl():
    lib = ctypes.CDLL("/root/monarch/libtest_hixl.so")
    lib.hixl_init_engine.restype = ctypes.c_void_p
    lib.hixl_init_engine.argtypes = [ctypes.c_int, ctypes.c_char_p]
    lib.hixl_register_mem.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
    lib.hixl_connect.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.hixl_transfer_write.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                         ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
    lib.hixl_cleanup.argtypes = [ctypes.c_void_p]
    return lib

class Producer(Actor):
    def __init__(self):
        dev = int(os.environ.get("MONARCH_NPU_DEVICE", "0"))
        self.dev = dev
        self.tensor = torch.ones(NBYTES // 4, dtype=torch.float32, device=f"npu:{dev}")
        torch.npu.synchronize()
        self.addr = self.tensor.data_ptr()
        self.lib = load_hixl()
        self.ctx = self.lib.hixl_init_engine(dev, SRV_ENGINE)
        self.lib.hixl_register_mem(self.ctx, self.addr, NBYTES)
        with open(COORD, 'w') as f:
            f.write(str(self.addr))
        print(f"[Producer] dev={dev} addr={hex(self.addr)} engine={SRV_ENGINE}", flush=True)

    @endpoint
    async def connect_back(self) -> int:
        ret = self.lib.hixl_connect(self.ctx, CLI_ENGINE)
        print(f"[Producer] connect_back to CLI: {ret}", flush=True)
        return ret

    @endpoint
    async def read_sum(self) -> float:
        return self.tensor.sum().cpu().item()

class Consumer(Actor):
    @endpoint
    async def do_transfer(self) -> str:
        dev = int(os.environ.get("MONARCH_NPU_DEVICE", "1"))
        local = torch.full((NBYTES // 4,), 2.0, dtype=torch.float32, device=f"npu:{dev}")
        torch.npu.synchronize()
        local_addr = local.data_ptr()

        with open(COORD) as f:
            remote_addr = int(f.read().strip())

        lib = load_hixl()
        ctx = lib.hixl_init_engine(dev, CLI_ENGINE)
        lib.hixl_register_mem(ctx, local_addr, NBYTES)

        time.sleep(1)
        ret = lib.hixl_connect(ctx, SRV_ENGINE)
        print(f"[Consumer] connect: {ret}", flush=True)

        time.sleep(3)  # wait for producer to connect back

        ret = lib.hixl_transfer_write(ctx, SRV_ENGINE, local_addr, remote_addr, NBYTES)
        result = f"TransferSync: {ret} {'OK' if ret == 0 else 'FAIL'}"
        print(f"[Consumer] {result}", flush=True)
        lib.hixl_cleanup(ctx)
        return result

def use_npu(dev_id: int):
    def _bootstrap():
        os.environ["MONARCH_NPU_DEVICE"] = str(dev_id)
        os.environ["HCCL_INTRA_ROCE_ENABLE"] = "1"
        import torch
        import torch_npu
        torch.npu.set_device(dev_id)
    return _bootstrap

async def main():
    print("=" * 60)
    print("HIXL direct C++ from monarch actors (bypass Rust FFI)")
    print("=" * 60, flush=True)
    try: os.unlink(COORD)
    except: pass

    host = this_host()
    pm = host.spawn_procs(per_host={"gpus": 1}, bootstrap=use_npu(0))
    cm = host.spawn_procs(per_host={"gpus": 1}, bootstrap=use_npu(1))

    producer = pm.spawn("producer", Producer)
    consumer = cm.spawn("consumer", Consumer)

    await asyncio.sleep(3)

    # Producer connects back to consumer
    connect_ret = await producer.connect_back.call_one()
    print(f"Producer connect_back result: {connect_ret}", flush=True)

    await asyncio.sleep(2)

    result = await consumer.do_transfer.call_one()
    print(f"Transfer result: {result}", flush=True)

    final_sum = await producer.read_sum.call_one()
    print(f"Producer sum after transfer: {final_sum} (expect 32.0)", flush=True)
    print("PASS" if "OK" in result else "FAIL")

if __name__ == "__main__":
    asyncio.run(main())
