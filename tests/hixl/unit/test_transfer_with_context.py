#!/usr/bin/env python3
"""
Test: if we set ACL context on a new thread, does TransferSync work?

Previous test showed Test 4 (WRITE from new thread B) failed.
HiXL developers say ACL context can be shared via aclrtSetCurrentContext.
This test verifies: set context on new threads → TransferSync should work.
"""
import ctypes
import os
import time
import threading
import multiprocessing as mp
import random

os.environ.setdefault("HCCL_INTRA_ROCE_ENABLE", "1")

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "libtest_hixl.so")
NBYTES = 64
_base = 44000 + random.randint(0, 8000)
SRV_EID = f"127.0.0.1:{_base}"
CLI_EID = f"127.0.0.1:{_base + 1}"


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
    done_ev.wait(timeout=60)
    time.sleep(3)
    torch.npu.synchronize()
    print(f"[S] final: sum={t.sum().cpu().item()}", flush=True)
    lib.hixl_cleanup(ctypes.c_void_p(ctx))


def client_fn(ready_ev, done_ev, addr_share):
    import torch, torch_npu
    torch.npu.set_device(1)
    lib = ctypes.CDLL(LIB_PATH)

    # Setup return types for get_acl_context
    lib.get_acl_context.restype = ctypes.c_uint64

    ctx = lib.hixl_init_engine(1, CLI_EID.encode())
    assert ctx
    print(f"[C] init OK", flush=True)

    ready_ev.wait(timeout=15)
    remote_addr = addr_share.value
    time.sleep(1)

    # Allocate multiple buffers
    bufs = []
    addrs = []
    for i in range(4):
        b = torch.full((16,), float(i + 1), dtype=torch.float32, device="npu")
        bufs.append(b)
    torch.npu.synchronize()
    for b in bufs:
        a = b.data_ptr()
        addrs.append(a)
        assert lib.hixl_register_mem(ctypes.c_void_p(ctx), ctypes.c_uint64(a), ctypes.c_size_t(NBYTES)) == 0

    assert lib.hixl_connect(ctypes.c_void_p(ctx), SRV_EID.encode()) == 0
    print(f"[C] connected", flush=True)

    # Get ACL context from the worker thread
    acl_ctx = lib.get_acl_context()
    print(f"[C] ACL context from worker: {hex(acl_ctx)}", flush=True)

    # === Test 1: WRITE_DIRECT from main thread WITHOUT setting context ===
    ret = lib.hixl_transfer_write_direct(
        ctypes.c_void_p(ctx), SRV_EID.encode(),
        ctypes.c_uint64(addrs[0]), ctypes.c_uint64(remote_addr), ctypes.c_size_t(NBYTES))
    print(f"[C] Test 1 - main thread, no set_context: {'PASS' if ret == 0 else 'FAIL'} ({ret})", flush=True)

    # === Test 2: WRITE_DIRECT from new thread WITHOUT setting context ===
    results = {}
    def write_no_ctx(idx):
        r = lib.hixl_transfer_write_direct(
            ctypes.c_void_p(ctx), SRV_EID.encode(),
            ctypes.c_uint64(addrs[idx]), ctypes.c_uint64(remote_addr), ctypes.c_size_t(NBYTES))
        results[f"t{idx}_no_ctx"] = r

    t2 = threading.Thread(target=write_no_ctx, args=(1,), name="no-ctx-thread")
    t2.start(); t2.join(timeout=30)
    r2 = results.get("t{}_no_ctx".format(1), -999)
    print(f"[C] Test 2 - new thread, NO context: {'PASS' if r2 == 0 else 'FAIL'} ({r2})", flush=True)

    # === Test 3: WRITE_DIRECT from new thread WITH aclrtSetCurrentContext ===
    def write_with_ctx(idx, acl_context):
        lib.set_acl_context(ctypes.c_uint64(acl_context))
        r = lib.hixl_transfer_write_direct(
            ctypes.c_void_p(ctx), SRV_EID.encode(),
            ctypes.c_uint64(addrs[idx]), ctypes.c_uint64(remote_addr), ctypes.c_size_t(NBYTES))
        results[f"t{idx}_with_ctx"] = r

    t3 = threading.Thread(target=write_with_ctx, args=(2, acl_ctx), name="with-ctx-thread")
    t3.start(); t3.join(timeout=30)
    r3 = results.get("t{}_with_ctx".format(2), -999)
    print(f"[C] Test 3 - new thread, WITH context: {'PASS' if r3 == 0 else 'FAIL'} ({r3})", flush=True)

    # === Test 4: Multiple threads WITH context, concurrently ===
    def write_concurrent(idx, acl_context, res_dict):
        lib.set_acl_context(ctypes.c_uint64(acl_context))
        r = lib.hixl_transfer_write_direct(
            ctypes.c_void_p(ctx), SRV_EID.encode(),
            ctypes.c_uint64(addrs[idx]), ctypes.c_uint64(remote_addr), ctypes.c_size_t(NBYTES))
        res_dict[f"concurrent_{idx}"] = r

    concurrent_results = {}
    threads = []
    for i in range(2, 4):
        t = threading.Thread(target=write_concurrent, args=(i, acl_ctx, concurrent_results))
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    print(f"[C] Test 4 - concurrent threads WITH context:", flush=True)
    all_ok = True
    for k, v in concurrent_results.items():
        ok = v == 0
        all_ok = all_ok and ok
        print(f"    {k}: {'PASS' if ok else 'FAIL'} ({v})", flush=True)

    # === Summary ===
    print(f"\n{'='*50}", flush=True)
    if r3 == 0:
        print("KEY FINDING: aclrtSetCurrentContext fixes thread affinity!", flush=True)
        print("  -> Worker thread serialization CAN be removed.", flush=True)
        print("  -> Just need set_context on each thread before HiXL calls.", flush=True)
    else:
        print("aclrtSetCurrentContext did NOT fix thread affinity.", flush=True)
        print("  -> Worker thread serialization must remain.", flush=True)
    if all_ok:
        print("BONUS: Concurrent transfers from multiple threads PASS!", flush=True)
    print(f"{'='*50}", flush=True)

    done_ev.set()
    lib.hixl_cleanup(ctypes.c_void_p(ctx))


if __name__ == "__main__":
    print("ACL context sharing test for HiXL TransferSync")
    mgr = mp.Manager()
    addr_share = mgr.Value('Q', 0)
    ready, done = mp.Event(), mp.Event()
    s = mp.Process(target=server_fn, args=(ready, done, addr_share))
    c = mp.Process(target=client_fn, args=(ready, done, addr_share))
    s.start(); c.start()
    c.join(timeout=90); s.join(timeout=10)
    print(f"exit: server={s.exitcode} client={c.exitcode}")
