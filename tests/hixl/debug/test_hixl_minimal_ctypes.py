#!/usr/bin/env python3
"""Minimal two-process HiXL test: isolate READ vs WRITE and connection order."""
import ctypes, os, sys, time, multiprocessing as mp

os.environ["HCCL_INTRA_ROCE_ENABLE"] = "1"

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "build", "libtest_hixl.so")
BUF_SIZE = 4096
COORD = "/tmp/hixl_minimal_coord"

def setup_lib():
    lib = ctypes.CDLL(LIB_PATH)
    lib.hixl_init_engine.argtypes = [ctypes.c_int, ctypes.c_char_p]
    lib.hixl_init_engine.restype = ctypes.c_void_p
    lib.hixl_connect.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.hixl_connect.restype = ctypes.c_int
    lib.hixl_register_mem.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
    lib.hixl_register_mem.restype = ctypes.c_int
    lib.hixl_transfer_read.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
    lib.hixl_transfer_read.restype = ctypes.c_int
    lib.hixl_transfer_write.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                         ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
    lib.hixl_transfer_write.restype = ctypes.c_int
    lib.hixl_cleanup.argtypes = [ctypes.c_void_p]
    return lib

def run_data_owner(barrier, eid_self, eid_remote, connect_first):
    """Data owner (dev=0). Holds the source buffer."""
    import torch, torch_npu
    dev = 0
    torch.npu.set_device(dev)
    lib = setup_lib()
    ctx = lib.hixl_init_engine(dev, eid_self.encode())
    assert ctx, "owner init failed"
    t = torch.ones(BUF_SIZE // 4, dtype=torch.float32, device=f"npu:{dev}").view(torch.uint8)
    torch.npu.synchronize()
    lib.hixl_register_mem(ctx, t.data_ptr(), t.numel())
    with open(COORD, 'w') as f:
        f.write(str(t.data_ptr()))
    print(f"[owner] buffer={hex(t.data_ptr())}", flush=True)
    barrier.wait()  # signal ready

    if connect_first:
        time.sleep(1)
        lib.hixl_connect(ctx, eid_remote.encode())
        print("[owner] connected first", flush=True)
        barrier.wait()  # signal connected
    else:
        barrier.wait()  # wait for consumer to connect first
        time.sleep(1)
        lib.hixl_connect(ctx, eid_remote.encode())
        print("[owner] connected second", flush=True)

    barrier.wait()  # wait for transfer
    lib.hixl_cleanup(ctx)

def run_data_consumer(barrier, eid_self, eid_remote, connect_first, use_read):
    """Data consumer (dev=1). Pulls or pushes data."""
    import torch, torch_npu
    dev = 1
    torch.npu.set_device(dev)
    lib = setup_lib()
    ctx = lib.hixl_init_engine(dev, eid_self.encode())
    assert ctx, "consumer init failed"
    local = torch.zeros(BUF_SIZE // 4, dtype=torch.float32, device=f"npu:{dev}").view(torch.uint8)
    torch.npu.synchronize()
    lib.hixl_register_mem(ctx, local.data_ptr(), local.numel())
    barrier.wait()  # wait for owner ready
    with open(COORD) as f:
        remote_addr = int(f.read().strip())

    if connect_first:
        lib.hixl_connect(ctx, eid_remote.encode())
        print("[consumer] connected first", flush=True)
        barrier.wait()  # signal connected
    else:
        barrier.wait()  # wait for owner to connect first
        lib.hixl_connect(ctx, eid_remote.encode())
        print("[consumer] connected second", flush=True)

    time.sleep(3)  # let connection stabilize

    if use_read:
        ret = lib.hixl_transfer_read(ctx, eid_remote.encode(),
                                      local.data_ptr(), remote_addr, local.numel())
        op_name = "READ"
    else:
        ret = lib.hixl_transfer_write(ctx, eid_remote.encode(),
                                       local.data_ptr(), remote_addr, local.numel())
        op_name = "WRITE"

    print(f"[consumer] {op_name} result: {ret} {'OK' if ret == 0 else 'FAIL'}", flush=True)
    barrier.wait()  # signal done
    lib.hixl_cleanup(ctx)
    sys.exit(0 if ret == 0 else 1)

def run_test(label, owner_connects_first, use_read, port_base):
    print(f"\n{'='*60}", flush=True)
    print(f"Test: {label}", flush=True)
    print(f"  owner_connects_first={owner_connects_first}, op={'READ' if use_read else 'WRITE'}", flush=True)
    print(f"{'='*60}", flush=True)
    try: os.unlink(COORD)
    except: pass
    eid_owner = f"127.0.0.1:{port_base}"
    eid_consumer = f"127.0.0.1:{port_base+1}"
    barrier = mp.Barrier(2)
    p_owner = mp.Process(target=run_data_owner, args=(barrier, eid_owner, eid_consumer, owner_connects_first))
    p_consumer = mp.Process(target=run_data_consumer, args=(barrier, eid_consumer, eid_owner, not owner_connects_first, use_read))
    p_owner.start(); p_consumer.start()
    p_consumer.join(timeout=60)
    p_owner.join(timeout=10)
    if p_owner.is_alive(): p_owner.terminate(); p_owner.join()
    ok = p_consumer.exitcode == 0
    print(f">>> {label}: {'PASS' if ok else 'FAIL'}", flush=True)
    return ok

if __name__ == "__main__":
    mp.set_start_method("spawn")
    results = []
    results.append(("consumer-first + WRITE", run_test("consumer-first + WRITE", False, False, 19200)))
    results.append(("owner-first + WRITE",    run_test("owner-first + WRITE",    True,  False, 19400)))
    results.append(("consumer-first + READ",  run_test("consumer-first + READ",  False, True,  19600)))
    results.append(("owner-first + READ",     run_test("owner-first + READ",     True,  True,  19800)))
    print(f"\n{'='*60}")
    for name, ok in results:
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    print(f"{'='*60}")
