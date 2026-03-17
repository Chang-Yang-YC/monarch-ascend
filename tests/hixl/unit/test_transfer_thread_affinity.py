#!/usr/bin/env python3
"""
Test: does HiXL TransferSync require thread affinity?

- Init + Connect on worker thread (normal path)
- TransferSync on a DIFFERENT thread via *_direct functions

If _direct works, the single-worker serialization can be relaxed.
"""
import ctypes
import os
import time
import threading
import multiprocessing as mp
import random

os.environ.setdefault("HCCL_INTRA_ROCE_ENABLE", "1")

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "libtest_hixl.so")
_base = 41000 + random.randint(0, 9000)
SRV_EID = f"127.0.0.1:{_base}"
CLI_EID = f"127.0.0.1:{_base + 1}"
NBYTES = 64


def server_fn(ready_ev, done_ev, addr_share):
    import torch, torch_npu
    torch.npu.set_device(0)
    lib = ctypes.CDLL(LIB_PATH)

    ctx = lib.hixl_init_engine(0, SRV_EID.encode())
    assert ctx

    t = torch.arange(16, dtype=torch.float32, device="npu")
    torch.npu.synchronize()
    addr = t.data_ptr()
    assert lib.hixl_register_mem(ctypes.c_void_p(ctx), ctypes.c_uint64(addr), ctypes.c_size_t(NBYTES)) == 0

    addr_share.value = addr
    ready_ev.set()
    print(f"[S] ready: {SRV_EID} addr={hex(addr)} sum={t.sum().cpu().item()}", flush=True)

    done_ev.wait(timeout=60)
    time.sleep(2)
    torch.npu.synchronize()
    print(f"[S] final: sum={t.sum().cpu().item()}", flush=True)
    lib.hixl_cleanup(ctypes.c_void_p(ctx))


def client_fn(ready_ev, done_ev, addr_share):
    import torch, torch_npu
    torch.npu.set_device(1)
    lib = ctypes.CDLL(LIB_PATH)

    ctx = lib.hixl_init_engine(1, CLI_EID.encode())
    assert ctx
    print(f"[C] init OK: {CLI_EID}", flush=True)

    ready_ev.wait(timeout=30)
    remote_addr = addr_share.value
    time.sleep(1)

    # Buffers
    write_buf = torch.ones(16, dtype=torch.float32, device="npu")
    read_buf = torch.zeros(16, dtype=torch.float32, device="npu")
    torch.npu.synchronize()
    w_addr = write_buf.data_ptr()
    r_addr = read_buf.data_ptr()

    assert lib.hixl_register_mem(ctypes.c_void_p(ctx), ctypes.c_uint64(w_addr), ctypes.c_size_t(NBYTES)) == 0
    assert lib.hixl_register_mem(ctypes.c_void_p(ctx), ctypes.c_uint64(r_addr), ctypes.c_size_t(NBYTES)) == 0

    assert lib.hixl_connect(ctypes.c_void_p(ctx), SRV_EID.encode()) == 0
    print(f"[C] connected (all via worker thread)", flush=True)
    time.sleep(1)

    # === Test 1: WRITE via worker thread (control) ===
    ret = lib.hixl_transfer_write(
        ctypes.c_void_p(ctx), SRV_EID.encode(),
        ctypes.c_uint64(w_addr), ctypes.c_uint64(remote_addr), ctypes.c_size_t(NBYTES))
    print(f"[C] Test 1 - WRITE via worker thread: {'PASS' if ret == 0 else 'FAIL'} (ret={ret})", flush=True)
    time.sleep(1)

    # === Test 2: WRITE_DIRECT from main thread (bypasses worker) ===
    ret = lib.hixl_transfer_write_direct(
        ctypes.c_void_p(ctx), SRV_EID.encode(),
        ctypes.c_uint64(w_addr), ctypes.c_uint64(remote_addr), ctypes.c_size_t(NBYTES))
    print(f"[C] Test 2 - WRITE_DIRECT from main thread: {'PASS' if ret == 0 else 'FAIL'} (ret={ret})", flush=True)
    time.sleep(1)

    # === Test 3: READ_DIRECT from a NEW Python thread ===
    results = {}
    def do_read_on_thread():
        r = lib.hixl_transfer_read_direct(
            ctypes.c_void_p(ctx), SRV_EID.encode(),
            ctypes.c_uint64(r_addr), ctypes.c_uint64(remote_addr), ctypes.c_size_t(NBYTES))
        results["read_direct"] = r

    t = threading.Thread(target=do_read_on_thread, name="read-test-thread")
    t.start()
    t.join(timeout=30)
    torch.npu.synchronize()
    read_ret = results.get("read_direct", -999)
    read_sum = read_buf.sum().cpu().item()
    print(f"[C] Test 3 - READ_DIRECT from new thread: {'PASS' if read_ret == 0 else 'FAIL'} "
          f"(ret={read_ret}, sum={read_sum})", flush=True)

    # === Test 4: WRITE_DIRECT from yet another thread ===
    results2 = {}
    write_buf2 = torch.full((16,), 2.0, dtype=torch.float32, device="npu")
    torch.npu.synchronize()
    w2_addr = write_buf2.data_ptr()
    lib.hixl_register_mem(ctypes.c_void_p(ctx), ctypes.c_uint64(w2_addr), ctypes.c_size_t(NBYTES))

    def do_write_on_thread():
        r = lib.hixl_transfer_write_direct(
            ctypes.c_void_p(ctx), SRV_EID.encode(),
            ctypes.c_uint64(w2_addr), ctypes.c_uint64(remote_addr), ctypes.c_size_t(NBYTES))
        results2["write_direct"] = r

    t2 = threading.Thread(target=do_write_on_thread, name="write-test-thread")
    t2.start()
    t2.join(timeout=30)
    write2_ret = results2.get("write_direct", -999)
    print(f"[C] Test 4 - WRITE_DIRECT from new thread: {'PASS' if write2_ret == 0 else 'FAIL'} "
          f"(ret={write2_ret})", flush=True)

    # Summary
    print(f"\n{'='*50}", flush=True)
    all_pass = all([
        ret == 0 for ret in [
            results.get("read_direct", -1),
            results2.get("write_direct", -1),
        ]
    ])
    if all_pass:
        print("CONCLUSION: TransferSync does NOT need thread affinity!", flush=True)
        print("  -> Worker thread serialization can be removed for transfers.", flush=True)
    else:
        print("CONCLUSION: TransferSync NEEDS thread affinity.", flush=True)
        print("  -> Must keep worker thread for transfers (or use thread pool).", flush=True)
    print(f"{'='*50}", flush=True)

    done_ev.set()
    lib.hixl_cleanup(ctypes.c_void_p(ctx))


if __name__ == "__main__":
    print("HiXL TransferSync thread affinity test")
    mgr = mp.Manager()
    addr_share = mgr.Value('Q', 0)
    ready, done = mp.Event(), mp.Event()

    s = mp.Process(target=server_fn, args=(ready, done, addr_share))
    c = mp.Process(target=client_fn, args=(ready, done, addr_share))
    s.start(); c.start()
    c.join(timeout=90); s.join(timeout=10)
    print(f"exit: server={s.exitcode} client={c.exitcode}")
