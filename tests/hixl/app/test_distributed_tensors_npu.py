#!/usr/bin/env python3
"""
Test Monarch Tensor Engine (distributed tensors) on NPU.

Adapted from docs/source/examples/distributed_tensors.py for Ascend NPU.
Device isolation uses ASCEND_RT_VISIBLE_DEVICES at parent level, mirroring
the GPU path (CUDA_VISIBLE_DEVICES). No bootstrap needed.
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

from monarch.actor import this_host


async def test_single_npu(dev_id: int):
    """Test Tensor Engine on a single NPU."""
    print(f"\n{'='*60}")
    print(f"  Part A: Single NPU Tensor Engine (card {dev_id})")
    print(f"{'='*60}")

    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(dev_id)
    mesh = this_host().spawn_procs(per_host={"npus": 1})
    print(f"  [1] mesh created: {mesh}")

    print("  [2] mesh.activate() + torch.rand...")
    try:
        with mesh.activate():
            t = torch.rand(3, 4, device="npu")
        import monarch
        val = monarch.inspect(t)
        print(f"      PASS - tensor: {val.shape}")
    except Exception as e:
        print(f"      FAIL - {e}")

    print("  [3] matmul computation...")
    try:
        with mesh.activate():
            a = torch.rand(4, 4, device="npu")
            b = torch.rand(4, 4, device="npu")
            c = a @ b
        import monarch
        val = monarch.inspect(c)
        print(f"      PASS - result: {val.shape}, sample: {val[0,:2]}")
    except Exception as e:
        print(f"      FAIL - {e}")

    return mesh


async def test_two_npus(dev_a: int, dev_b: int):
    """Test Tensor Engine with two NPUs and cross-device reduce.

    Uses ASCEND_RT_VISIBLE_DEVICES at parent level to restrict visible
    devices, then _initialize_env assigns local_rank 0->dev_a, 1->dev_b.
    Mirrors the GPU path exactly.
    """
    print(f"\n{'='*60}")
    print(f"  Part B: Two NPU Tensor Engine (cards {dev_a},{dev_b})")
    print(f"{'='*60}")

    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = f"{dev_a},{dev_b}"
    mesh = this_host().spawn_procs(per_host={"npus": 2})
    print(f"  [1] mesh created: {mesh}")

    print("  [2] mesh.activate() + torch.rand on 2 NPUs...")
    try:
        with mesh.activate():
            t = torch.rand(3, 4, device="npu")
        import monarch
        val = monarch.inspect(t)
        print(f"      PASS - tensor: {val.shape}")
    except Exception as e:
        print(f"      FAIL - {e}")
        return

    print("  [3] collective reduce(sum)...")
    try:
        with mesh.activate():
            r = torch.ones(2, 3, device="npu")
            reduced = r.reduce("npus", "sum")
        import monarch
        val = monarch.inspect(reduced)
        print(f"      PASS - result: {val}")
        print(f"      expected: all 2.0")
    except Exception as e:
        print(f"      FAIL - {e}")


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


async def main():
    if len(sys.argv) >= 3:
        dev_a, dev_b = int(sys.argv[1]), int(sys.argv[2])
    else:
        dev_a, dev_b = _pick_two_free_npus()

    print("=" * 60)
    print("Distributed Tensors on NPU (Tensor Engine path)")
    print("=" * 60)

    await test_two_npus(dev_a, dev_b)

    print(f"\n{'='*60}")
    print("All tests complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    from monarch._src.actor.actor_mesh import context
    context()
    asyncio.run(main())
