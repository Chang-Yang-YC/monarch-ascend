#!/usr/bin/env python3
"""
Test: what is the minimum settling time after Connect before TransferSync works?

Tries Connect → sleep(delay) → TransferSync with decreasing delays:
  1000ms, 500ms, 200ms, 100ms, 50ms, 10ms, 0ms
"""
import ctypes
import os
import time
import multiprocessing as mp
import random

os.environ.setdefault("HCCL_INTRA_ROCE_ENABLE", "1")

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "libtest_hixl.so")
NBYTES = 64
DELAYS_MS = [1000, 500, 200, 100, 50, 10, 0]


def run_one_test(delay_ms):
    """Run a fresh server+client pair with the given post-connect delay."""
    base = 42000 + random.randint(0, 8000)
    srv_eid = f"127.0.0.1:{base}"
    cli_eid = f"127.0.0.1:{base + 1}"

    mgr = mp.Manager()
    addr_share = mgr.Value('Q', 0)
    ready = mp.Event()
    result_share = mgr.Value('i', -1)  # -1 = not run, 0 = pass, >0 = error code

    def server(ready_ev, addr_sh):
        import torch, torch_npu
        torch.npu.set_device(0)
        lib = ctypes.CDLL(LIB_PATH)
        ctx = lib.hixl_init_engine(0, srv_eid.encode())
        assert ctx

        t = torch.arange(16, dtype=torch.float32, device="npu")
        torch.npu.synchronize()
        addr = t.data_ptr()
        lib.hixl_register_mem(ctypes.c_void_p(ctx), ctypes.c_uint64(addr), ctypes.c_size_t(NBYTES))
        addr_sh.value = addr
        ready_ev.set()
        time.sleep(15)
        lib.hixl_cleanup(ctypes.c_void_p(ctx))

    def client(ready_ev, addr_sh, res_sh, delay_s):
        import torch, torch_npu
        torch.npu.set_device(1)
        lib = ctypes.CDLL(LIB_PATH)
        ctx = lib.hixl_init_engine(1, cli_eid.encode())
        assert ctx

        ready_ev.wait(timeout=10)
        remote_addr = addr_sh.value

        buf = torch.ones(16, dtype=torch.float32, device="npu")
        torch.npu.synchronize()
        addr = buf.data_ptr()
        lib.hixl_register_mem(ctypes.c_void_p(ctx), ctypes.c_uint64(addr), ctypes.c_size_t(NBYTES))

        ret = lib.hixl_connect(ctypes.c_void_p(ctx), srv_eid.encode())
        if ret != 0:
            res_sh.value = ret
            lib.hixl_cleanup(ctypes.c_void_p(ctx))
            return

        if delay_s > 0:
            time.sleep(delay_s)

        ret = lib.hixl_transfer_write(
            ctypes.c_void_p(ctx), srv_eid.encode(),
            ctypes.c_uint64(addr), ctypes.c_uint64(remote_addr), ctypes.c_size_t(NBYTES))
        res_sh.value = ret
        lib.hixl_cleanup(ctypes.c_void_p(ctx))

    delay_s = delay_ms / 1000.0
    s = mp.Process(target=server, args=(ready, addr_share))
    c = mp.Process(target=client, args=(ready, addr_share, result_share, delay_s))
    s.start()
    c.start()
    c.join(timeout=30)
    s.terminate()
    s.join(timeout=5)
    return result_share.value


if __name__ == "__main__":
    print("Connect settling time test")
    print(f"{'Delay':>8}  {'Result':>8}  Status")
    print("-" * 35)

    for delay_ms in DELAYS_MS:
        ret = run_one_test(delay_ms)
        status = "PASS" if ret == 0 else f"FAIL({ret})"
        print(f"{delay_ms:>6}ms  {ret:>8}  {status}")

    print("\nDone.")
