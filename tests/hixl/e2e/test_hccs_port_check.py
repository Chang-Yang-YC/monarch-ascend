#!/usr/bin/env python3
"""Check which ports HiXL binds during Initialize (HCCS mode)."""
import ctypes, os, subprocess, time

os.environ.pop("HCCL_INTRA_ROCE_ENABLE", None)

LIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "libtest_hixl.so")

import torch, torch_npu
torch.npu.set_device(0)

def show_ports(label):
    pid = os.getpid()
    result = subprocess.run(
        ["ss", "-tlnp"],
        capture_output=True, text=True
    )
    my_lines = [l for l in result.stdout.splitlines() if str(pid) in l]
    print(f"  [{label}] ports bound by PID {pid}:", flush=True)
    for l in my_lines:
        print(f"    {l}", flush=True)
    if not my_lines:
        print(f"    (none)", flush=True)

print(f"PID={os.getpid()}", flush=True)
show_ports("before init")

lib = ctypes.CDLL(LIB)
eid = "192.168.0.117:48001"
ctx = lib.hixl_init_engine(0, eid.encode())
print(f"Init({eid}): ctx={'OK' if ctx else 'FAIL'}", flush=True)

show_ports("after init")

time.sleep(2)
show_ports("after 2s")

if ctx:
    lib.hixl_cleanup(ctypes.c_void_p(ctx))
    time.sleep(1)
    show_ports("after cleanup")
