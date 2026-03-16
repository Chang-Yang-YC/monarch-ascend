#!/usr/bin/env python3
"""
Minimal test: can two processes on the SAME NPU device communicate via HiXL (RoCE)?

Both processes bind to NPU 0 via ASCEND_RT_VISIBLE_DEVICES=0.
We force RoCE by setting HCCL_INTRA_ROCE_ENABLE=1 before HiXL init.

If this works, it proves HiXL *does* support same-device communication (over RoCE),
and the earlier "not support connect with self device" error was HCCS-specific.
"""
import ctypes
import multiprocessing
import os
import sys
import time

ALIGN_2MB = 2 * 1024 * 1024
BUF_SIZE = ALIGN_2MB

def find_lib():
    paths = [
        os.path.join(os.path.dirname(__file__), "build", "libtest_hixl.so"),
        "/root/monarch/tests/hixl/build/libtest_hixl.so",
    ]
    env = os.environ.get("MONARCH_HIXL_LIB")
    if env and os.path.isfile(env):
        return env
    for p in paths:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError("libtest_hixl.so not found")


def alloc_aligned_tensor(size_bytes: int, device: str = "npu"):
    import torch
    import torch_npu  # noqa: F401
    n_elems = (size_bytes + ALIGN_2MB + 3) // 4
    raw = torch.empty(n_elems, dtype=torch.float32, device=device)
    addr = raw.data_ptr()
    offset = (ALIGN_2MB - (addr % ALIGN_2MB)) % ALIGN_2MB
    start = offset // 4
    count = size_bytes // 4
    aligned = raw[start:start + count]
    assert aligned.data_ptr() % ALIGN_2MB == 0, f"alignment failed: {hex(aligned.data_ptr())}"
    return aligned, raw


def worker(rank, barrier, result_queue, engine_ids):
    """Each worker binds to NPU 0 and tries HiXL RoCE communication."""
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"
    os.environ["HCCL_INTRA_ROCE_ENABLE"] = "1"

    import torch
    import torch_npu  # noqa: F401
    torch.npu.set_device(0)

    lib_path = find_lib()
    lib = ctypes.CDLL(lib_path)
    lib.hixl_init_engine.restype = ctypes.c_void_p
    lib.hixl_init_engine.argtypes = [ctypes.c_int, ctypes.c_char_p]
    lib.hixl_register_mem.restype = ctypes.c_int
    lib.hixl_register_mem.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
    lib.hixl_connect.restype = ctypes.c_int
    lib.hixl_connect.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.hixl_transfer_read.restype = ctypes.c_int
    lib.hixl_transfer_read.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    ]
    lib.hixl_transfer_write.restype = ctypes.c_int
    lib.hixl_transfer_write.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    ]

    ip = os.environ.get("MONARCH_HIXL_IP", "127.0.0.1")
    port = 30000 + rank * 100 + os.getpid() % 100
    my_eid = f"{ip}:{port}"
    engine_ids[rank] = my_eid
    print(f"[Worker {rank}] PID={os.getpid()} engine_id={my_eid}", flush=True)

    ctx = lib.hixl_init_engine(0, my_eid.encode())
    if not ctx:
        result_queue.put((rank, "FAIL", "hixl_init_engine failed"))
        return

    buf, raw = alloc_aligned_tensor(BUF_SIZE)
    if rank == 0:
        buf.fill_(42.0)
    else:
        buf.fill_(0.0)

    addr = buf.data_ptr()
    ret = lib.hixl_register_mem(ctx, addr, BUF_SIZE)
    if ret != 0:
        result_queue.put((rank, "FAIL", f"register_mem failed: {ret}"))
        return

    print(f"[Worker {rank}] registered mem addr={hex(addr)} size={BUF_SIZE}", flush=True)

    barrier.wait()
    time.sleep(1)

    peer_rank = 1 - rank
    peer_eid = engine_ids[peer_rank]
    print(f"[Worker {rank}] connecting to peer {peer_eid}...", flush=True)

    for attempt in range(10):
        ret = lib.hixl_connect(ctx, peer_eid.encode())
        if ret == 0:
            break
        print(f"[Worker {rank}] connect attempt {attempt+1}/10 failed: {ret}", flush=True)
        time.sleep(1)
    else:
        result_queue.put((rank, "FAIL", f"connect to {peer_eid} failed after 10 attempts"))
        return

    print(f"[Worker {rank}] connected to {peer_eid}!", flush=True)

    barrier.wait()

    if rank == 1:
        peer_addr = addr  # same offset since both 2MB-aligned
        ret = lib.hixl_transfer_read(
            ctx, engine_ids[0].encode(),
            addr, addr, BUF_SIZE,
        )
        if ret != 0:
            result_queue.put((rank, "FAIL", f"transfer_read failed: {ret}"))
            return

        torch.npu.synchronize()
        val = buf[0].item()
        print(f"[Worker {rank}] READ from worker 0: buf[0] = {val}", flush=True)
        if abs(val - 42.0) < 0.01:
            result_queue.put((rank, "PASS", f"same-device RoCE transfer works! buf[0]={val}"))
        else:
            result_queue.put((rank, "FAIL", f"data mismatch: expected 42.0 got {val}"))
    else:
        time.sleep(3)
        result_queue.put((rank, "PASS", "writer side done"))


def main():
    print("=" * 60)
    print("Test: Same-device HiXL communication (RoCE mode)")
    print("Both processes on NPU 0, HCCL_INTRA_ROCE_ENABLE=1")
    print("=" * 60)

    mp = multiprocessing.get_context("spawn")
    barrier = mp.Barrier(2)
    result_queue = mp.Queue()
    engine_ids = mp.Manager().dict()

    workers = []
    for rank in range(2):
        p = mp.Process(target=worker, args=(rank, barrier, result_queue, engine_ids))
        p.start()
        workers.append(p)

    for p in workers:
        p.join(timeout=60)

    results = []
    while not result_queue.empty():
        results.append(result_queue.get_nowait())

    print("\n" + "=" * 60)
    for rank, status, msg in sorted(results):
        print(f"  Worker {rank}: {status} — {msg}")

    all_pass = all(s == "PASS" for _, s, _ in results)
    print("=" * 60)
    if all_pass and len(results) == 2:
        print("RESULT: HiXL DOES support same-device communication via RoCE!")
    else:
        print("RESULT: Same-device communication failed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
