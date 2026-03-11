#!/usr/bin/env python3
"""
Direct HIXL test via shared lib: compare torch_npu memory vs aclrtMalloc.
Uses multiprocessing to run server/client on different NPUs.
"""
import os, sys, time, struct, ctypes
import multiprocessing as mp

os.environ["HCCL_INTRA_ROCE_ENABLE"] = "1"

SRV = b"127.0.0.1:19100"
CLI = b"127.0.0.1:19101"
COORD = "/tmp/hixl_direct_coord"
NBYTES = 64

def load():
    lib = ctypes.CDLL("./libtest_hixl.so")
    lib.hixl_init_engine.restype = ctypes.c_void_p
    lib.hixl_init_engine.argtypes = [ctypes.c_int, ctypes.c_char_p]
    lib.hixl_register_mem.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
    lib.hixl_connect.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.hixl_transfer_write.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                         ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
    lib.hixl_cleanup.argtypes = [ctypes.c_void_p]
    return lib

def acl_malloc(dev, nbytes):
    acl = ctypes.CDLL("libascendcl.so")
    acl.aclInit(None)
    acl.aclrtSetDevice(dev)
    ptr = ctypes.c_void_p()
    ret = acl.aclrtMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes), ctypes.c_int(2))
    assert ret == 0, f"aclrtMalloc failed: {ret}"
    acl.aclrtMemset(ptr, ctypes.c_size_t(nbytes), ctypes.c_int(0), ctypes.c_size_t(nbytes))
    return ptr.value, acl

def torch_malloc(dev, nbytes):
    import torch
    import torch_npu  # noqa
    torch.npu.set_device(dev)
    t = torch.ones(nbytes // 4, dtype=torch.float32, device=f"npu:{dev}")
    torch.npu.synchronize()
    return t.data_ptr(), t

def run_server(use_torch, barrier):
    dev = 0
    if use_torch:
        addr, _keep = torch_malloc(dev, NBYTES)
        print(f"[S] torch addr={hex(addr)}", flush=True)
    else:
        addr, _acl = acl_malloc(dev, NBYTES)
        print(f"[S] acl addr={hex(addr)}", flush=True)

    lib = load()
    ctx = lib.hixl_init_engine(dev, SRV)
    assert ctx, "Server init failed"
    lib.hixl_register_mem(ctx, addr, NBYTES)

    with open(COORD, 'w') as f:
        f.write(str(addr))
    barrier.wait()

    time.sleep(3)
    lib.hixl_connect(ctx, CLI)

    time.sleep(10)

    # Read back
    acl = ctypes.CDLL("libascendcl.so")
    host = ctypes.create_string_buffer(NBYTES)
    acl.aclrtMemcpy(host, ctypes.c_size_t(NBYTES),
                     ctypes.c_void_p(addr), ctypes.c_size_t(NBYTES), 2)
    vals = struct.unpack("16f", host.raw)
    print(f"[S] sum={sum(vals):.1f}", flush=True)
    lib.hixl_cleanup(ctx)

def run_client(use_torch, barrier):
    dev = 1
    if use_torch:
        addr, _keep = torch_malloc(dev, NBYTES)
        print(f"[C] torch addr={hex(addr)}", flush=True)
    else:
        addr, _acl = acl_malloc(dev, NBYTES)
        # Write 2.0 pattern
        host = struct.pack("16f", *([2.0] * 16))
        buf = ctypes.create_string_buffer(host)
        _acl.aclrtMemcpy(ctypes.c_void_p(addr), ctypes.c_size_t(NBYTES),
                          buf, ctypes.c_size_t(NBYTES), 1)
        print(f"[C] acl addr={hex(addr)}", flush=True)

    lib = load()
    ctx = lib.hixl_init_engine(dev, CLI)
    assert ctx, "Client init failed"
    lib.hixl_register_mem(ctx, addr, NBYTES)

    barrier.wait()
    time.sleep(1)
    with open(COORD) as f:
        remote_addr = int(f.read().strip())
    print(f"[C] remote_addr={hex(remote_addr)}", flush=True)

    lib.hixl_connect(ctx, SRV)
    time.sleep(3)

    ret = lib.hixl_transfer_write(ctx, SRV, addr, remote_addr, NBYTES)
    print(f"[C] TransferSync result: {ret} {'OK' if ret == 0 else 'FAIL'}", flush=True)
    lib.hixl_cleanup(ctx)

def run_test(use_torch):
    label = "TORCH" if use_torch else "ACL"
    print(f"\n{'='*60}\nTest: {label} memory\n{'='*60}", flush=True)
    try: os.unlink(COORD)
    except: pass
    barrier = mp.Barrier(2)
    s = mp.Process(target=run_server, args=(use_torch, barrier))
    c = mp.Process(target=run_client, args=(use_torch, barrier))
    s.start(); c.start()
    c.join(timeout=30)
    s.join(timeout=5)
    if s.is_alive(): s.terminate(); s.join()
    ok = c.exitcode == 0
    print(f">>> {label}: {'PASS' if ok else 'FAIL'}", flush=True)

if __name__ == "__main__":
    mp.set_start_method("spawn")
    run_test(use_torch=False)
    run_test(use_torch=True)
