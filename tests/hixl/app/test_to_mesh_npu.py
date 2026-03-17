#!/usr/bin/env python3
"""
Test to_mesh (HCCL point-to-point send/recv) on Ascend NPU.

Mirrors GPU tests from python/tests/test_controller.py and
python/tests/test_tensor_engine.py.
"""
import asyncio
import os
import sys

os.environ["PYTHONPATH"] = os.pathsep.join(sys.path)

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    print("ERROR: torch_npu not available")
    sys.exit(1)

import monarch
from monarch.actor import this_host


def _pick_two_free_npus():
    """Auto-detect two available NPU devices."""
    count = torch.npu.device_count()
    if count < 2:
        print(f"ERROR: need at least 2 NPUs, found {count}")
        sys.exit(1)
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    if visible:
        devs = [int(d.strip()) for d in visible.split(",") if d.strip()]
        if len(devs) >= 2:
            return devs[0], devs[1]
    return 0, 1


async def run_all_tests(dev_a: int, dev_b: int):
    print("=" * 60)
    print(f"to_mesh Tests on NPU (cards {dev_a}, {dev_b})")
    print("=" * 60)

    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = f"{dev_a},{dev_b}"
    mesh = this_host().spawn_procs(per_host={"npus": 2})

    # --- Test 1: slice_mesh + to_mesh (mirrors test_proc_mesh_tensor_engine) ---
    print(f"\n{'='*60}")
    print(f"  Test 1: slice_mesh + to_mesh")
    print(f"{'='*60}")

    with mesh.activate():
        f = 10 * mesh.rank_tensor("npus").npu()
        a = monarch.inspect(f, npus=0)
        b = monarch.inspect(f, npus=1)

    print(f"  rank 0 = {a} (expect 0)")
    print(f"  rank 1 = {b} (expect 10)")
    assert a == 0, f"Expected 0, got {a}"
    assert b == 10, f"Expected 10, got {b}"

    one = mesh.slice(npus=1)
    with one.activate():
        sliced_b = monarch.slice_mesh(f, npus=1).to_mesh(one)
        c = monarch.inspect(sliced_b * 10)

    print(f"  sliced rank 1 * 10 = {c} (expect 100)")
    assert c == 100, f"Expected 100, got {c}"
    print("  PASS")

    # --- Test 2: cross-slice to_mesh (npu0 -> npu1) ---
    print(f"\n{'='*60}")
    print(f"  Test 2: cross-slice to_mesh (npu0 -> npu1)")
    print(f"{'='*60}")

    npu0 = mesh.slice(npus=0)
    npu1 = mesh.slice(npus=1)

    with npu0.activate():
        x = torch.tensor([1.0, 2.0, 3.0], device="npu")
        y = x.to_mesh(npu1)

    with npu1.activate():
        val = monarch.inspect(y)

    print(f"  sent:     [1.0, 2.0, 3.0]")
    print(f"  received: {val.tolist()}")
    assert torch.allclose(val, torch.tensor([1.0, 2.0, 3.0])), f"Mismatch: {val}"
    print("  PASS")

    # --- Test 3: to_mesh + compute on destination ---
    print(f"\n{'='*60}")
    print(f"  Test 3: to_mesh + compute on destination")
    print(f"{'='*60}")

    with npu0.activate():
        p = torch.ones(2, 3, device="npu") * 5.0
        p_on_1 = p.to_mesh(npu1)

    with npu1.activate():
        q = torch.ones(2, 3, device="npu") * 3.0
        z = p_on_1 + q
        val = monarch.inspect(z)

    print(f"  5s from npu0 + 3s on npu1 = {val.flatten().tolist()[:3]}... (expect 8s)")
    expected = torch.ones(2, 3) * 8.0
    assert torch.allclose(val, expected), f"Expected 8s, got {val}"
    print("  PASS")

    # --- Test 4: bidirectional to_mesh ---
    print(f"\n{'='*60}")
    print(f"  Test 4: bidirectional to_mesh")
    print(f"{'='*60}")

    with npu0.activate():
        aa = torch.tensor([10.0, 20.0], device="npu")
        aa_on_1 = aa.to_mesh(npu1)

    with npu1.activate():
        bb = torch.tensor([30.0, 40.0], device="npu")
        bb_on_0 = bb.to_mesh(npu0)

    with npu1.activate():
        val_a = monarch.inspect(aa_on_1)
    with npu0.activate():
        val_b = monarch.inspect(bb_on_0)

    print(f"  npu0 -> npu1: sent [10, 20], got {val_a.tolist()}")
    print(f"  npu1 -> npu0: sent [30, 40], got {val_b.tolist()}")
    assert torch.allclose(val_a, torch.tensor([10.0, 20.0])), f"Mismatch: {val_a}"
    assert torch.allclose(val_b, torch.tensor([30.0, 40.0])), f"Mismatch: {val_b}"
    print("  PASS")

    print(f"\n{'='*60}")
    print("All to_mesh tests PASSED")
    print(f"{'='*60}")


async def main():
    if len(sys.argv) >= 3:
        dev_a, dev_b = int(sys.argv[1]), int(sys.argv[2])
    else:
        dev_a, dev_b = _pick_two_free_npus()
    await run_all_tests(dev_a, dev_b)


if __name__ == "__main__":
    from monarch._src.actor.actor_mesh import context
    context()
    asyncio.run(main())
