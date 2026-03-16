#!/usr/bin/env python3
"""
HiXL single-sided communication bandwidth test.

Tests READ and WRITE throughput between two NPU cards using various buffer sizes.
"""
import ctypes
import multiprocessing as mp
import os
import sys
import time

ALIGN_2MB = 2 * 1024 * 1024
COORD_FILE = "/tmp/hixl_bw_coord"
WARMUP_ITERS = 5
MEASURE_ITERS = 50

SIZES = [
    (1, "1 MB", 1 * 1024 * 1024),
    (2, "2 MB", 2 * 1024 * 1024),
    (4, "4 MB", 4 * 1024 * 1024),
    (8, "8 MB", 8 * 1024 * 1024),
    (16, "16 MB", 16 * 1024 * 1024),
    (32, "32 MB", 32 * 1024 * 1024),
    (64, "64 MB", 64 * 1024 * 1024),
    (128, "128 MB", 128 * 1024 * 1024),
    (256, "256 MB", 256 * 1024 * 1024),
    (512, "512 MB", 512 * 1024 * 1024),
]


def find_lib():
    paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "tests/hixl/build/libtest_hixl.so"),
        "/root/monarch/tests/hixl/build/libtest_hixl.so",
    ]
    env = os.environ.get("MONARCH_HIXL_LIB")
    if env and os.path.isfile(env):
        return env
    for p in paths:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError("libtest_hixl.so not found")


def setup_lib():
    lib = ctypes.CDLL(find_lib())
    lib.hixl_init_engine.argtypes = [ctypes.c_int, ctypes.c_char_p]
    lib.hixl_init_engine.restype = ctypes.c_void_p
    lib.hixl_register_mem.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
    lib.hixl_register_mem.restype = ctypes.c_int
    lib.hixl_connect.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.hixl_connect.restype = ctypes.c_int
    lib.hixl_transfer_read.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    ]
    lib.hixl_transfer_read.restype = ctypes.c_int
    lib.hixl_transfer_write.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    ]
    lib.hixl_transfer_write.restype = ctypes.c_int
    lib.hixl_cleanup.argtypes = [ctypes.c_void_p]
    return lib


def alloc_aligned(size_bytes, device):
    """Allocate a 2MB-aligned NPU buffer."""
    import torch
    n_elems = (size_bytes + ALIGN_2MB + 3) // 4
    raw = torch.empty(n_elems, dtype=torch.float32, device=device)
    addr = raw.data_ptr()
    offset = (ALIGN_2MB - (addr % ALIGN_2MB)) % ALIGN_2MB
    start = offset // 4
    count = size_bytes // 4
    aligned = raw[start:start + count]
    assert aligned.data_ptr() % ALIGN_2MB == 0
    return aligned, raw


def server(dev_id, barrier, result_queue, max_size):
    """Server side: holds the data buffer, waits for client to finish."""
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(dev_id)
    import torch
    import torch_npu  # noqa: F401
    torch.npu.set_device(0)

    lib = setup_lib()
    ip = os.environ.get("MONARCH_HIXL_IP", "192.168.0.117")
    eid = f"{ip}:{40000 + dev_id}"
    ctx = lib.hixl_init_engine(0, eid.encode())
    assert ctx, "server init failed"

    buf, raw = alloc_aligned(max_size, "npu:0")
    buf.view(torch.float32).fill_(3.14)
    torch.npu.synchronize()

    ret = lib.hixl_register_mem(ctx, buf.data_ptr(), max_size)
    assert ret == 0, f"server register_mem failed: {ret}"

    with open(COORD_FILE, "w") as f:
        f.write(f"{eid}\n{buf.data_ptr()}")

    barrier.wait()  # signal ready

    for attempt in range(30):
        peer_eid = f"{ip}:{40000 + int(open(COORD_FILE + '.client').read().strip())}"
        ret = lib.hixl_connect(ctx, peer_eid.encode())
        if ret == 0:
            break
        time.sleep(1)

    barrier.wait()  # connected
    barrier.wait()  # wait for all tests done

    lib.hixl_cleanup(ctx)


def client(dev_id, server_dev_id, barrier, result_queue, max_size):
    """Client side: connects to server and runs bandwidth tests."""
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(dev_id)
    import torch
    import torch_npu  # noqa: F401
    torch.npu.set_device(0)

    lib = setup_lib()
    ip = os.environ.get("MONARCH_HIXL_IP", "192.168.0.117")
    eid = f"{ip}:{40000 + dev_id}"
    ctx = lib.hixl_init_engine(0, eid.encode())
    assert ctx, "client init failed"

    buf, raw = alloc_aligned(max_size, "npu:0")
    torch.npu.synchronize()

    ret = lib.hixl_register_mem(ctx, buf.data_ptr(), max_size)
    assert ret == 0, f"client register_mem failed: {ret}"

    with open(COORD_FILE + ".client", "w") as f:
        f.write(str(dev_id))

    barrier.wait()  # wait for server ready

    with open(COORD_FILE) as f:
        lines = f.read().strip().split("\n")
        server_eid = lines[0]
        remote_addr = int(lines[1])

    for attempt in range(30):
        ret = lib.hixl_connect(ctx, server_eid.encode())
        if ret == 0:
            break
        time.sleep(1)
    assert ret == 0, "client connect failed"

    barrier.wait()  # connected

    results = []

    for _, label, size in SIZES:
        if size > max_size:
            break

        for op_name, transfer_fn in [("READ", lib.hixl_transfer_read),
                                      ("WRITE", lib.hixl_transfer_write)]:
            # warmup
            for _ in range(WARMUP_ITERS):
                transfer_fn(ctx, server_eid.encode(),
                            buf.data_ptr(), remote_addr, size)

            # measure
            t0 = time.perf_counter()
            for _ in range(MEASURE_ITERS):
                ret = transfer_fn(ctx, server_eid.encode(),
                                  buf.data_ptr(), remote_addr, size)
                if ret != 0:
                    print(f"  {op_name} {label}: transfer failed ret={ret}", flush=True)
                    break
            t1 = time.perf_counter()

            elapsed = t1 - t0
            total_bytes = size * MEASURE_ITERS
            bw_gbps = (total_bytes / elapsed) / (1024 ** 3)
            lat_ms = (elapsed / MEASURE_ITERS) * 1000

            results.append((op_name, label, size, bw_gbps, lat_ms))
            print(f"  {op_name:5s} {label:>8s}: {bw_gbps:8.2f} GB/s  "
                  f"(latency {lat_ms:.3f} ms, {MEASURE_ITERS} iters)", flush=True)

    result_queue.put(results)
    barrier.wait()  # signal done
    lib.hixl_cleanup(ctx)


def main():
    dev_a = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    dev_b = int(sys.argv[2]) if len(sys.argv) > 2 else 6

    max_size = SIZES[-1][2]

    print("=" * 70)
    print(f"HiXL Bandwidth Test: NPU {dev_a} ↔ NPU {dev_b}")
    print(f"  Transport: HCCS (default)")
    print(f"  Buffer sizes: 1 MB → {SIZES[-1][1]}")
    print(f"  Warmup: {WARMUP_ITERS} iters, Measure: {MEASURE_ITERS} iters")
    print("=" * 70)

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(2)
    result_queue = ctx.Queue()

    for f in [COORD_FILE, COORD_FILE + ".client"]:
        try:
            os.unlink(f)
        except OSError:
            pass

    p_server = ctx.Process(target=server,
                           args=(dev_a, barrier, result_queue, max_size))
    p_client = ctx.Process(target=client,
                           args=(dev_b, dev_a, barrier, result_queue, max_size))

    p_server.start()
    time.sleep(1)
    p_client.start()

    p_client.join(timeout=300)
    p_server.join(timeout=30)

    if p_server.is_alive():
        p_server.terminate()
        p_server.join()

    if not result_queue.empty():
        results = result_queue.get()
        print("\n" + "=" * 70)
        print(f"{'Op':>5s}  {'Size':>8s}  {'BW (GB/s)':>10s}  {'Latency (ms)':>12s}")
        print("-" * 45)
        for op, label, size, bw, lat in results:
            print(f"{op:>5s}  {label:>8s}  {bw:>10.2f}  {lat:>12.3f}")
        print("=" * 70)
    else:
        print("No results collected — test may have failed.")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
