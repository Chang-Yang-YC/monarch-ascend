#!/usr/bin/env python3
"""
Raw C-shim level HCCS test: bypass Monarch actors entirely.
Two processes, each with its own HiXL engine, test WRITE + READ via ctypes.
No HCCL_INTRA_ROCE_ENABLE set — should default to HCCS for intra-node D2D.
"""

import ctypes
import os
import sys
import time
import multiprocessing as mp

os.environ.pop("HCCL_INTRA_ROCE_ENABLE", None)

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "libtest_hixl.so")
MACHINE_IP = "192.168.0.117"
SRV_EID = f"{MACHINE_IP}:40001"
CLI_EID = f"{MACHINE_IP}:40002"
NBYTES = 64  # 16 floats


def server_proc(ready_event, done_event):
    os.environ.pop("HCCL_INTRA_ROCE_ENABLE", None)
    import torch
    import torch_npu

    dev = 0
    os.environ["MONARCH_NPU_DEVICE"] = str(dev)
    torch.npu.set_device(dev)

    lib = ctypes.CDLL(LIB_PATH)
    ctx = lib.hixl_init_engine(dev, SRV_EID.encode())
    assert ctx, "server init failed"
    print(f"[S] Init OK: {SRV_EID} dev={dev}", flush=True)

    t = torch.arange(16, dtype=torch.float32, device="npu")
    torch.npu.synchronize()
    addr = t.data_ptr()
    ret = lib.hixl_register_mem(ctypes.c_void_p(ctx), ctypes.c_uint64(addr), ctypes.c_size_t(NBYTES))
    print(f"[S] RegMem: {ret} addr={hex(addr)}", flush=True)
    assert ret == 0

    print(f"[S] tensor before: {t.cpu().tolist()}", flush=True)
    ready_event.set()

    done_event.wait(timeout=60)
    torch.npu.synchronize()
    print(f"[S] tensor after:  {t.cpu().tolist()}", flush=True)
    print(f"[S] sum after: {t.sum().cpu().item()}", flush=True)

    lib.hixl_cleanup(ctypes.c_void_p(ctx))
    print("[S] done", flush=True)


def client_proc(ready_event, done_event):
    os.environ.pop("HCCL_INTRA_ROCE_ENABLE", None)
    import torch
    import torch_npu

    dev = 1
    os.environ["MONARCH_NPU_DEVICE"] = str(dev)
    torch.npu.set_device(dev)

    lib = ctypes.CDLL(LIB_PATH)
    ctx = lib.hixl_init_engine(dev, CLI_EID.encode())
    assert ctx, "client init failed"
    print(f"[C] Init OK: {CLI_EID} dev={dev}", flush=True)

    ready_event.wait(timeout=30)
    time.sleep(1)

    # Connect to server
    ret = lib.hixl_connect(ctypes.c_void_p(ctx), SRV_EID.encode())
    print(f"[C] Connect({SRV_EID}): {ret}", flush=True)
    assert ret == 0
    time.sleep(1)

    # --- WRITE test: write ones to server ---
    write_buf = torch.ones(16, dtype=torch.float32, device="npu")
    torch.npu.synchronize()
    w_addr = write_buf.data_ptr()
    ret = lib.hixl_register_mem(ctypes.c_void_p(ctx), ctypes.c_uint64(w_addr), ctypes.c_size_t(NBYTES))
    print(f"[C] RegMem write_buf: {ret} addr={hex(w_addr)}", flush=True)
    assert ret == 0

    # Server's tensor addr — we need it for remote_addr.
    # In a real system this comes from RDMABuffer metadata; here we'll read it from shared state.
    # For simplicity, use a known address pattern. Actually let's pass it via the queue.
    print(f"[C] WRITE ones -> server...", flush=True)
    ret = lib.hixl_transfer_write(
        ctypes.c_void_p(ctx), SRV_EID.encode(),
        ctypes.c_uint64(w_addr),
        ctypes.c_uint64(0),  # placeholder — we need server addr
        ctypes.c_size_t(NBYTES),
    )
    print(f"[C] TransferSync WRITE: {ret}", flush=True)

    # --- READ test: read from server into local zeros ---
    read_buf = torch.zeros(16, dtype=torch.float32, device="npu")
    torch.npu.synchronize()
    r_addr = read_buf.data_ptr()
    ret = lib.hixl_register_mem(ctypes.c_void_p(ctx), ctypes.c_uint64(r_addr), ctypes.c_size_t(NBYTES))
    print(f"[C] RegMem read_buf: {ret} addr={hex(r_addr)}", flush=True)
    assert ret == 0

    print(f"[C] READ from server...", flush=True)
    ret = lib.hixl_transfer_read(
        ctypes.c_void_p(ctx), SRV_EID.encode(),
        ctypes.c_uint64(r_addr),
        ctypes.c_uint64(0),  # placeholder — need server addr
        ctypes.c_size_t(NBYTES),
    )
    print(f"[C] TransferSync READ: {ret}", flush=True)
    torch.npu.synchronize()
    print(f"[C] read_buf: {read_buf.cpu().tolist()}", flush=True)

    done_event.set()
    lib.hixl_cleanup(ctypes.c_void_p(ctx))
    print("[C] done", flush=True)


if __name__ == "__main__":
    print(f"HCCL_INTRA_ROCE_ENABLE={os.environ.get('HCCL_INTRA_ROCE_ENABLE', 'NOT SET')}")
    print(f"Server: {SRV_EID} (dev=0)  Client: {CLI_EID} (dev=1)")
    # Actually this test needs shared memory for the server address.
    # Let me use a simpler approach with mp.Value for the address.
    print("NOTE: This test needs the server tensor address shared to client.")
    print("Using mp.Manager for address sharing.")

    mgr = mp.Manager()
    srv_addr = mgr.Value('Q', 0)  # unsigned long long for address

    ready = mp.Event()
    done = mp.Event()

    # Rewrite with address sharing
    def server_proc_v2(ready_event, done_event, addr_share):
        os.environ.pop("HCCL_INTRA_ROCE_ENABLE", None)
        import torch
        import torch_npu

        dev = 0
        os.environ["MONARCH_NPU_DEVICE"] = str(dev)
        torch.npu.set_device(dev)

        lib = ctypes.CDLL(LIB_PATH)
        ctx = lib.hixl_init_engine(dev, SRV_EID.encode())
        assert ctx, "server init failed"
        print(f"[S] Init OK: {SRV_EID} dev={dev}", flush=True)

        t = torch.arange(16, dtype=torch.float32, device="npu")
        torch.npu.synchronize()
        addr = t.data_ptr()
        ret = lib.hixl_register_mem(ctypes.c_void_p(ctx), ctypes.c_uint64(addr), ctypes.c_size_t(NBYTES))
        assert ret == 0
        print(f"[S] RegMem: OK addr={hex(addr)}", flush=True)
        print(f"[S] tensor before: sum={t.sum().cpu().item()}", flush=True)

        addr_share.value = addr
        ready_event.set()

        done_event.wait(timeout=60)
        time.sleep(2)
        torch.npu.synchronize()
        print(f"[S] tensor after:  sum={t.sum().cpu().item()}", flush=True)
        print(f"[S] tensor after:  {t.cpu().tolist()}", flush=True)

        lib.hixl_cleanup(ctypes.c_void_p(ctx))

    def client_proc_v2(ready_event, done_event, addr_share):
        os.environ.pop("HCCL_INTRA_ROCE_ENABLE", None)
        import torch
        import torch_npu

        dev = 1
        os.environ["MONARCH_NPU_DEVICE"] = str(dev)
        torch.npu.set_device(dev)

        lib = ctypes.CDLL(LIB_PATH)
        ctx = lib.hixl_init_engine(dev, CLI_EID.encode())
        assert ctx, "client init failed"
        print(f"[C] Init OK: {CLI_EID} dev={dev}", flush=True)

        ready_event.wait(timeout=30)
        srv_tensor_addr = addr_share.value
        print(f"[C] server tensor addr: {hex(srv_tensor_addr)}", flush=True)
        time.sleep(1)

        ret = lib.hixl_connect(ctypes.c_void_p(ctx), SRV_EID.encode())
        print(f"[C] Connect: {ret}", flush=True)
        assert ret == 0, f"Connect failed: {ret}"
        time.sleep(1)

        # WRITE: ones -> server
        write_buf = torch.ones(16, dtype=torch.float32, device="npu")
        torch.npu.synchronize()
        w_addr = write_buf.data_ptr()
        ret = lib.hixl_register_mem(ctypes.c_void_p(ctx), ctypes.c_uint64(w_addr), ctypes.c_size_t(NBYTES))
        assert ret == 0

        print(f"[C] WRITE ones -> server (remote={hex(srv_tensor_addr)})...", flush=True)
        ret = lib.hixl_transfer_write(
            ctypes.c_void_p(ctx), SRV_EID.encode(),
            ctypes.c_uint64(w_addr),
            ctypes.c_uint64(srv_tensor_addr),
            ctypes.c_size_t(NBYTES),
        )
        print(f"[C] WRITE result: {ret}", flush=True)

        time.sleep(1)

        # READ: server -> local zeros
        read_buf = torch.zeros(16, dtype=torch.float32, device="npu")
        torch.npu.synchronize()
        r_addr = read_buf.data_ptr()
        ret = lib.hixl_register_mem(ctypes.c_void_p(ctx), ctypes.c_uint64(r_addr), ctypes.c_size_t(NBYTES))
        assert ret == 0

        print(f"[C] READ from server (remote={hex(srv_tensor_addr)}) -> local ({hex(r_addr)})...", flush=True)
        ret = lib.hixl_transfer_read(
            ctypes.c_void_p(ctx), SRV_EID.encode(),
            ctypes.c_uint64(r_addr),
            ctypes.c_uint64(srv_tensor_addr),
            ctypes.c_size_t(NBYTES),
        )
        print(f"[C] READ result: {ret}", flush=True)
        torch.npu.synchronize()
        vals = read_buf.cpu().tolist()
        print(f"[C] read_buf: {vals}", flush=True)
        s = read_buf.sum().cpu().item()
        print(f"[C] read_buf sum: {s}", flush=True)

        if abs(s - 16.0) < 1e-4:
            print("PASS: HCCS READ returned correct data!", flush=True)
        else:
            print(f"FAIL: expected sum=16.0, got {s}", flush=True)

        done_event.set()
        lib.hixl_cleanup(ctypes.c_void_p(ctx))

    s = mp.Process(target=server_proc_v2, args=(ready, done, srv_addr))
    c = mp.Process(target=client_proc_v2, args=(ready, done, srv_addr))
    s.start()
    c.start()
    c.join(timeout=90)
    s.join(timeout=10)
    print(f"Server exit={s.exitcode}, Client exit={c.exitcode}")
