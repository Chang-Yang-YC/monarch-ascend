#!/usr/bin/env python3
"""Check if HiXL Initialize binds port 16666 (HCCL default)."""
import ctypes, os, subprocess, time

os.environ.pop("HCCL_INTRA_ROCE_ENABLE", None)
# Do NOT set HCCL_NPU_SOCKET_PORT_RANGE — we want to see the default behavior

LIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "libtest_hixl.so")

import torch, torch_npu
torch.npu.set_device(0)

ALIGN = 2 * 1024 * 1024

def show_ports(label):
    result = subprocess.run(["ss", "-tlnp"], capture_output=True, text=True)
    lines_16666 = [l for l in result.stdout.splitlines() if "16666" in l]
    lines_mine = [l for l in result.stdout.splitlines() if str(os.getpid()) in l]
    print(f"  [{label}] port 16666:", flush=True)
    for l in lines_16666:
        print(f"    {l}", flush=True)
    if not lines_16666:
        print(f"    (not bound)", flush=True)
    print(f"  [{label}] my ports (PID {os.getpid()}):", flush=True)
    for l in lines_mine:
        print(f"    {l}", flush=True)
    if not lines_mine:
        print(f"    (none)", flush=True)

show_ports("before init")

lib = ctypes.CDLL(LIB)
eid = "192.168.0.117:48888"
ctx = lib.hixl_init_engine(0, eid.encode())
print(f"Init({eid}): {'OK' if ctx else 'FAIL'}", flush=True)

show_ports("after init")

# Also allocate 2MB-aligned memory and register it
raw = torch.empty(64 + ALIGN, dtype=torch.uint8, device="npu")
offset = (ALIGN - (raw.data_ptr() % ALIGN)) % ALIGN
buf = raw.narrow(0, offset, 64)
addr = buf.data_ptr()
print(f"addr={hex(addr)} aligned={addr % ALIGN == 0}", flush=True)
ret = lib.hixl_register_mem(ctypes.c_void_p(ctx), ctypes.c_uint64(addr), ctypes.c_size_t(64))
print(f"RegMem: {ret}", flush=True)

show_ports("after regmem")

if ctx:
    lib.hixl_cleanup(ctypes.c_void_p(ctx))
    time.sleep(1)
    show_ports("after cleanup")
