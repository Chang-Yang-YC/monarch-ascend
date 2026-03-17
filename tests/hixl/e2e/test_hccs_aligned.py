#!/usr/bin/env python3
"""
HCCS test with 2MB-aligned memory.
HiXL developers confirmed: HCCS requires data_address_ptr % 2MB == 0.
Solution: allocate extra 2MB, then align the start address.
"""
import ctypes
import os
import time
import multiprocessing as mp

os.environ.pop("HCCL_INTRA_ROCE_ENABLE", None)

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "libtest_hixl.so")
MACHINE_IP = "192.168.0.117"
import random
_base_port = 40100 + random.randint(0, 9000)
SRV_EID = f"{MACHINE_IP}:{_base_port}"
CLI_EID = f"{MACHINE_IP}:{_base_port + 1}"

ALIGN = 2 * 1024 * 1024  # 2MB
NFLOATS = 16
NBYTES = NFLOATS * 4  # 64 bytes


def alloc_aligned(nfloats, device, dtype):
    """Allocate a 2MB-aligned device tensor."""
    import torch
    elem_size = torch.tensor([], dtype=dtype).element_size()
    nbytes = nfloats * elem_size
    raw = torch.empty(nbytes + ALIGN, dtype=torch.uint8, device=device)
    raw_ptr = raw.data_ptr()
    offset = (ALIGN - (raw_ptr % ALIGN)) % ALIGN
    aligned_view = raw.narrow(0, offset, nbytes).view(dtype)
    aligned_ptr = aligned_view.data_ptr()
    assert aligned_ptr % ALIGN == 0, f"not aligned: {hex(aligned_ptr)} % {ALIGN} = {aligned_ptr % ALIGN}"
    print(f"  alloc_aligned: raw={hex(raw_ptr)} offset={offset} aligned={hex(aligned_ptr)} "
          f"check={aligned_ptr % ALIGN}", flush=True)
    return aligned_view, raw  # keep raw alive to prevent GC


def server_fn(ready_ev, done_ev, addr_share):
    os.environ.pop("HCCL_INTRA_ROCE_ENABLE", None)
    import torch, torch_npu
    dev = 0
    torch.npu.set_device(dev)

    lib = ctypes.CDLL(LIB_PATH)
    ctx = lib.hixl_init_engine(dev, SRV_EID.encode())
    assert ctx, "server init failed"
    print(f"[S] Init OK dev={dev} eid={SRV_EID}", flush=True)

    t, t_raw = alloc_aligned(NFLOATS, "npu", torch.float32)
    t.copy_(torch.arange(NFLOATS, dtype=torch.float32, device="npu"))
    torch.npu.synchronize()
    addr = t.data_ptr()

    ret = lib.hixl_register_mem(ctypes.c_void_p(ctx), ctypes.c_uint64(addr), ctypes.c_size_t(NBYTES))
    assert ret == 0, f"server RegMem failed: {ret}"
    print(f"[S] RegMem OK addr={hex(addr)} aligned={addr % ALIGN == 0} sum={t.sum().cpu().item()}", flush=True)

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
    print(f"[C] remote_addr={hex(remote_addr)} aligned={remote_addr % ALIGN == 0}", flush=True)

    write_buf, w_raw = alloc_aligned(NFLOATS, "npu", torch.float32)
    write_buf.fill_(1.0)
    read_buf, r_raw = alloc_aligned(NFLOATS, "npu", torch.float32)
    read_buf.fill_(0.0)
    torch.npu.synchronize()

    w_addr = write_buf.data_ptr()
    r_addr = read_buf.data_ptr()
    print(f"[C] write={hex(w_addr)} aligned={w_addr % ALIGN == 0}", flush=True)
    print(f"[C] read ={hex(r_addr)} aligned={r_addr % ALIGN == 0}", flush=True)

    ret = lib.hixl_register_mem(ctypes.c_void_p(ctx), ctypes.c_uint64(w_addr), ctypes.c_size_t(NBYTES))
    assert ret == 0, f"RegMem write failed: {ret}"
    ret = lib.hixl_register_mem(ctypes.c_void_p(ctx), ctypes.c_uint64(r_addr), ctypes.c_size_t(NBYTES))
    assert ret == 0, f"RegMem read failed: {ret}"
    print(f"[C] RegMem both OK", flush=True)

    time.sleep(1)
    ret = lib.hixl_connect(ctypes.c_void_p(ctx), SRV_EID.encode())
    print(f"[C] Connect: {ret}", flush=True)
    if ret != 0:
        print(f"FAIL: Connect failed with {ret}", flush=True)
        done_ev.set()
        lib.hixl_cleanup(ctypes.c_void_p(ctx))
        return
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
    vals = read_buf.cpu().tolist()
    print(f"[C] read_buf sum={s} vals={vals}", flush=True)

    if ret == 0 and abs(s - 16.0) < 1e-4:
        print("PASS: HCCS with 2MB-aligned memory works!", flush=True)
    elif ret == 0:
        print(f"FAIL: Transfer OK but data wrong (sum={s}, expected 16.0)", flush=True)
    else:
        print(f"FAIL: Transfer returned {ret}", flush=True)

    done_ev.set()
    lib.hixl_cleanup(ctypes.c_void_p(ctx))


if __name__ == "__main__":
    print(f"HCCL_INTRA_ROCE_ENABLE={os.environ.get('HCCL_INTRA_ROCE_ENABLE', 'NOT SET')}")
    print(f"2MB alignment test: server={SRV_EID} client={CLI_EID}")
    mgr = mp.Manager()
    addr_share = mgr.Value('Q', 0)
    ready, done = mp.Event(), mp.Event()

    s = mp.Process(target=server_fn, args=(ready, done, addr_share))
    c = mp.Process(target=client_fn, args=(ready, done, addr_share))
    s.start(); c.start()
    c.join(timeout=90); s.join(timeout=10)
    print(f"exit: server={s.exitcode} client={c.exitcode}")
