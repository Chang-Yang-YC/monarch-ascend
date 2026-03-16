#!/usr/bin/env python3
"""Test minimum delay between connect and transfer_read."""
import ctypes, os, sys, time, multiprocessing as mp

os.environ["HCCL_INTRA_ROCE_ENABLE"] = "1"

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "build", "libtest_hixl.so")
BUF_SIZE = 4096
COORD = "/tmp/hixl_timing_coord"

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
    lib.hixl_cleanup.argtypes = [ctypes.c_void_p]
    return lib

def owner(barrier, eid_self, eid_remote):
    import torch, torch_npu
    torch.npu.set_device(0)
    lib = setup_lib()
    ctx = lib.hixl_init_engine(0, eid_self.encode())
    t = torch.ones(BUF_SIZE // 4, dtype=torch.float32, device="npu:0").view(torch.uint8)
    torch.npu.synchronize()
    lib.hixl_register_mem(ctx, t.data_ptr(), t.numel())
    with open(COORD, 'w') as f:
        f.write(str(t.data_ptr()))
    barrier.wait()
    lib.hixl_connect(ctx, eid_remote.encode())
    barrier.wait()
    barrier.wait()
    lib.hixl_cleanup(ctx)

def consumer(barrier, eid_self, eid_remote, delay_s):
    import torch, torch_npu
    torch.npu.set_device(1)
    lib = setup_lib()
    ctx = lib.hixl_init_engine(1, eid_self.encode())
    local = torch.zeros(BUF_SIZE // 4, dtype=torch.float32, device="npu:1").view(torch.uint8)
    torch.npu.synchronize()
    lib.hixl_register_mem(ctx, local.data_ptr(), local.numel())
    barrier.wait()
    with open(COORD) as f:
        remote_addr = int(f.read().strip())
    barrier.wait()
    lib.hixl_connect(ctx, eid_remote.encode())
    time.sleep(delay_s)
    ret = lib.hixl_transfer_read(ctx, eid_remote.encode(),
                                  local.data_ptr(), remote_addr, local.numel())
    print(f"delay={delay_s}s: READ result={ret} {'OK' if ret == 0 else 'FAIL'}", flush=True)
    barrier.wait()
    lib.hixl_cleanup(ctx)
    sys.exit(0 if ret == 0 else 1)

if __name__ == "__main__":
    mp.set_start_method("spawn")
    for delay in [0, 0.5, 1, 2, 3]:
        try: os.unlink(COORD)
        except: pass
        port = 20000 + int(delay * 100)
        eid_o = f"127.0.0.1:{port}"
        eid_c = f"127.0.0.1:{port+1}"
        barrier = mp.Barrier(2)
        po = mp.Process(target=owner, args=(barrier, eid_o, eid_c))
        pc = mp.Process(target=consumer, args=(barrier, eid_c, eid_o, delay))
        po.start(); pc.start()
        pc.join(timeout=30); po.join(timeout=10)
        if po.is_alive(): po.terminate(); po.join()
        ok = pc.exitcode == 0
        print(f"  delay={delay}s → {'PASS' if ok else 'FAIL'}")
