#!/usr/bin/env python3
"""
Test: run HIXL operations from a Python threading.Thread (NOT the main thread)
to determine if the HIXL failure is thread-identity related.

If this test FAILS: confirms thread-identity is the root cause.
If this test PASSES: the issue is Rust-specific, not generic threading.
"""
import asyncio
import os
import sys
import time
import threading
import ctypes

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("HCCL_INTRA_ROCE_ENABLE", "1")

import torch
try:
    import torch_npu
except ImportError:
    print("ERROR: torch_npu not available"); sys.exit(1)

from monarch.actor import Actor, endpoint, this_host

COORD = "/tmp/hixl_thread_coord"
NBYTES = 64
SRV_ENGINE = b"127.0.0.1:19300"
CLI_ENGINE = b"127.0.0.1:19301"


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

        # Run HIXL init on a SEPARATE thread (not the main Python thread)
        self.ctx = None
        self.lib = None
        init_done = threading.Event()

        def _init_on_thread():
            self.lib = load_hixl()
            self.ctx = self.lib.hixl_init_engine(dev, SRV_ENGINE)
            self.lib.hixl_register_mem(self.ctx, self.addr, NBYTES)
            init_done.set()

        t = threading.Thread(target=_init_on_thread)
        t.start()
        t.join()
        init_done.wait()

        with open(COORD, 'w') as f:
            f.write(str(self.addr))
        print(f"[Producer] dev={dev} addr={hex(self.addr)} ctx={self.ctx} (init on separate thread)", flush=True)

    @endpoint
    async def connect_back(self) -> int:
        # Also run connect on a separate thread
        result = [None]
        def _connect():
            result[0] = self.lib.hixl_connect(self.ctx, CLI_ENGINE)
        t = threading.Thread(target=_connect)
        t.start()
        t.join()
        print(f"[Producer] connect_back to CLI: {result[0]} (on separate thread)", flush=True)
        return result[0]

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

        # Run ALL HIXL operations on a separate thread
        result = [None]
        def _do_hixl():
            lib = load_hixl()
            ctx = lib.hixl_init_engine(dev, CLI_ENGINE)
            lib.hixl_register_mem(ctx, local_addr, NBYTES)
            time.sleep(1)
            ret = lib.hixl_connect(ctx, SRV_ENGINE)
            print(f"[Consumer] connect: {ret} (on separate thread)", flush=True)
            time.sleep(3)
            ret = lib.hixl_transfer_write(ctx, SRV_ENGINE, local_addr, remote_addr, NBYTES)
            result[0] = f"TransferSync: {ret} {'OK' if ret == 0 else 'FAIL'}"
            print(f"[Consumer] {result[0]} (on separate thread)", flush=True)
            lib.hixl_cleanup(ctx)

        t = threading.Thread(target=_do_hixl)
        t.start()
        t.join()
        return result[0]


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
    print("HIXL thread isolation test")
    print("  All HIXL ops run on Python threading.Thread (NOT main thread)")
    print("=" * 60, flush=True)
    try:
        os.unlink(COORD)
    except OSError:
        pass

    host = this_host()
    pm = host.spawn_procs(per_host={"gpus": 1}, bootstrap=use_npu(0))
    cm = host.spawn_procs(per_host={"gpus": 1}, bootstrap=use_npu(1))

    producer = pm.spawn("producer", Producer)
    consumer = cm.spawn("consumer", Consumer)

    await asyncio.sleep(3)
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
