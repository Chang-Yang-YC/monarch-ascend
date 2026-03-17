#!/usr/bin/env python3
"""
Raw HCCS test v2: RegisterMem BEFORE Connect (matching Monarch's order).
"""
import ctypes
import os
import sys
import time
import multiprocessing as mp

os.environ.pop("HCCL_INTRA_ROCE_ENABLE", None)

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "libtest_hixl.so")
MACHINE_IP = "192.168.0.117"
SRV_EID = f"{MACHINE_IP}:40011"
CLI_EID = f"{MACHINE_IP}:40012"
NBYTES = 64


def server_fn(ready_ev, done_ev, addr_share):
    os.environ.pop("HCCL_INTRA_ROCE_ENABLE", None)
    import torch, torch_npu
    dev = 0
    torch.npu.set_device(dev)

    lib = ctypes.CDLL(LIB_PATH)
    ctx = lib.hixl_init_engine(dev, SRV_EID.encode())
    assert ctx, "server init failed"
    print(f"[S] Init OK dev={dev} eid={SRV_EID}", flush=True)

    t = torch.arange(16, dtype=torch.float32, device="npu")
    torch.npu.synchronize()
    addr = t.data_ptr()
    ret = lib.hixl_register_mem(ctypes.c_void_p(ctx), ctypes.c_uint64(addr), ctypes.c_size_t(NBYTES))
    assert ret == 0, f"server RegMem failed: {ret}"
    print(f"[S] RegMem OK addr={hex(addr)} sum={t.sum().cpu().item()}", flush=True)

    addr_share.value = addr
    ready_ev.set()
    done_ev.wait(timeout=60)
    time.sleep(2)
    torch.npu.synchronize()
    print(f"[S] after: sum={t.sum().cpu().item()} vals={t.cpu().tolist()}", flush=True)
    lib.hixl_cleanup(ctypes.c_void_p(ctx))


def client_fn(ready_ev, done_ev, addr_share):
    os.environ.pop("HCCL_INTRA_ROCE_ENABLE", None)
    import torch, torch_npu
    dev = 1
    torch.npu.set_device(dev)

    lib = ctypes.CDLL(LIB_PATH)
    ctx = lib.hixl_init_engine(dev, CLI_EID.encode())
    assert ctx, "client init failed"
    print(f"[C] Init OK dev={dev} eid={CLI_EID}", flush=True)

    ready_ev.wait(timeout=30)
    remote_addr = addr_share.value
    print(f"[C] remote_addr={hex(remote_addr)}", flush=True)

    # Allocate and register ALL buffers BEFORE connect
    write_buf = torch.ones(16, dtype=torch.float32, device="npu")
    read_buf = torch.zeros(16, dtype=torch.float32, device="npu")
    torch.npu.synchronize()

    w_addr = write_buf.data_ptr()
    r_addr = read_buf.data_ptr()
    print(f"[C] write_buf addr={hex(w_addr)}, read_buf addr={hex(r_addr)}", flush=True)

    ret = lib.hixl_register_mem(ctypes.c_void_p(ctx), ctypes.c_uint64(w_addr), ctypes.c_size_t(NBYTES))
    assert ret == 0, f"RegMem write failed: {ret}"
    ret = lib.hixl_register_mem(ctypes.c_void_p(ctx), ctypes.c_uint64(r_addr), ctypes.c_size_t(NBYTES))
    assert ret == 0, f"RegMem read failed: {ret}"
    print(f"[C] RegMem both OK", flush=True)

    # NOW connect (after all RegisterMem)
    time.sleep(1)
    ret = lib.hixl_connect(ctypes.c_void_p(ctx), SRV_EID.encode())
    print(f"[C] Connect: {ret}", flush=True)
    assert ret == 0, f"Connect failed: {ret}"
    time.sleep(1)

    # WRITE
    print(f"[C] WRITE ones -> server...", flush=True)
    ret = lib.hixl_transfer_write(
        ctypes.c_void_p(ctx), SRV_EID.encode(),
        ctypes.c_uint64(w_addr), ctypes.c_uint64(remote_addr), ctypes.c_size_t(NBYTES))
    print(f"[C] WRITE: {ret}", flush=True)

    time.sleep(1)

    # READ
    print(f"[C] READ from server...", flush=True)
    ret = lib.hixl_transfer_read(
        ctypes.c_void_p(ctx), SRV_EID.encode(),
        ctypes.c_uint64(r_addr), ctypes.c_uint64(remote_addr), ctypes.c_size_t(NBYTES))
    print(f"[C] READ: {ret}", flush=True)

    torch.npu.synchronize()
    s = read_buf.sum().cpu().item()
    print(f"[C] read_buf sum={s} vals={read_buf.cpu().tolist()}", flush=True)

    if ret == 0 and abs(s - 16.0) < 1e-4:
        print("PASS: HCCS transfer works!", flush=True)
    elif ret == 0:
        print(f"FAIL: Transfer returned OK but data wrong (sum={s})", flush=True)
    else:
        print(f"FAIL: Transfer returned {ret}", flush=True)

    done_ev.set()
    lib.hixl_cleanup(ctypes.c_void_p(ctx))


if __name__ == "__main__":
    print(f"HCCL_INTRA_ROCE_ENABLE={os.environ.get('HCCL_INTRA_ROCE_ENABLE', 'NOT SET')}")
    mgr = mp.Manager()
    addr_share = mgr.Value('Q', 0)
    ready, done = mp.Event(), mp.Event()

    s = mp.Process(target=server_fn, args=(ready, done, addr_share))
    c = mp.Process(target=client_fn, args=(ready, done, addr_share))
    s.start(); c.start()
    c.join(timeout=90); s.join(timeout=10)
    print(f"exit: server={s.exitcode} client={c.exitcode}")
