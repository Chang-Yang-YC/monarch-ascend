#!/usr/bin/env python3
"""
P0 NPU Test Suite — Collective communication primitives, multi-NPU scale,
and key upstream GPU test mirrors.

Usage:
    python test_collective_ops_npu.py --devs 1,2                 # 2-card tests
    python test_collective_ops_npu.py --devs 1,2,3,4             # 2+4 card tests
    python test_collective_ops_npu.py --devs 0,1,2,3,4,5,6,7     # all including 8-card
    python test_collective_ops_npu.py --devs 0,1,2,3,4,5,6,7 --only scale8
    python test_collective_ops_npu.py --devs 1,2 --only collective
    python test_collective_ops_npu.py --devs 1,2 --only mirror

Test groups:
    collective  — 2-card collective primitives (allreduce, allgather, etc.)
    scale       — 4-card scale-up tests
    scale8      — 8-card full-scale tests
    mirror      — upstream GPU test mirrors (2 cards)

Known HCCL limitations on Ascend 910B / CANN 9.0:
  - reduce("avg") hangs (ReduceOp.AVG not supported by HCCL)
  - reduce("product") rejected at Monarch level
"""
import argparse
import asyncio
import os
import sys
import traceback

os.environ["PYTHONPATH"] = os.pathsep.join(sys.path)

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    print("ERROR: torch_npu not available")
    sys.exit(1)

import monarch
from monarch.actor import this_host

# ---------------------------------------------------------------------------
# Test bookkeeping
# ---------------------------------------------------------------------------
_pass_count = 0
_fail_count = 0
_skip_count = 0
_fail_names = []


def _header(name: str):
    print(f"\n{'=' * 64}")
    print(f"  {name}")
    print(f"{'=' * 64}", flush=True)


def _pass(name: str, detail: str = ""):
    global _pass_count
    _pass_count += 1
    msg = f"  PASS  {name}"
    if detail:
        msg += f" — {detail}"
    print(msg, flush=True)


def _fail(name: str, err):
    global _fail_count
    _fail_count += 1
    _fail_names.append(name)
    print(f"  FAIL  {name} — {err}", flush=True)
    traceback.print_exc()


def _skip(name: str, reason: str):
    global _skip_count
    _skip_count += 1
    print(f"  SKIP  {name} — {reason}", flush=True)


def _pick_free_npus(n: int) -> list:
    count = torch.npu.device_count()
    if count < n:
        print(f"ERROR: need at least {n} NPUs, found {count}")
        sys.exit(1)
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    if visible:
        devs = [int(d.strip()) for d in visible.split(",") if d.strip()]
        if len(devs) >= n:
            return devs[:n]
    return list(range(n))


# ===================================================================
# P0-1: Collective Communication Primitives (2 NPUs)
# ===================================================================

async def test_collective_ops(devs: list):
    assert len(devs) >= 2
    dev_a, dev_b = devs[0], devs[1]
    _header(f"P0-1: Collective Ops (NPU {dev_a}, {dev_b})")

    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = f"{dev_a},{dev_b}"
    mesh = this_host().spawn_procs(per_host={"npus": 2})

    # --- allreduce(sum) ---
    try:
        with mesh.activate():
            rank = mesh.rank_tensor("npus").npu()
            x = torch.ones(3, 4, device="npu") * (rank + 1)
            y = x.reduce("npus", "sum")
            val = monarch.inspect(y)
        expected = torch.ones(3, 4) * 3.0
        assert torch.allclose(val, expected), f"Expected 3s, got {val}"
        _pass("allreduce(sum)", f"1+2={val.flatten()[0].item()}")
    except Exception as e:
        _fail("allreduce(sum)", e)

    # --- allreduce(max) ---
    try:
        with mesh.activate():
            rank = mesh.rank_tensor("npus").npu()
            x = torch.ones(4, device="npu") * (rank + 1)
            y = x.reduce("npus", "max")
            val = monarch.inspect(y)
        assert torch.allclose(val, torch.ones(4) * 2.0), f"Expected 2s, got {val}"
        _pass("allreduce(max)", f"max(1,2)={val[0].item()}")
    except Exception as e:
        _fail("allreduce(max)", e)

    # --- allreduce(min) ---
    try:
        with mesh.activate():
            rank = mesh.rank_tensor("npus").npu()
            x = torch.ones(4, device="npu") * (rank + 1)
            y = x.reduce("npus", "min")
            val = monarch.inspect(y)
        assert torch.allclose(val, torch.ones(4) * 1.0), f"Expected 1s, got {val}"
        _pass("allreduce(min)", f"min(1,2)={val[0].item()}")
    except Exception as e:
        _fail("allreduce(min)", e)

    # --- allgather(stack) ---
    try:
        with mesh.activate():
            rank = mesh.rank_tensor("npus").npu()
            x = torch.ones(3, device="npu") * (rank + 1)
            g = x.reduce("npus", "stack")
            val = monarch.inspect(g)
        expected = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        assert torch.allclose(val, expected), f"Expected [[1s],[2s]], got {val}"
        _pass("allgather(stack)", f"shape={val.shape}")
    except Exception as e:
        _fail("allgather(stack)", e)

    # --- reduce_scatter(sum) ---
    try:
        with mesh.activate():
            rank = mesh.rank_tensor("npus").npu()
            x = torch.ones(2, 6, device="npu") * (rank + 1)
            rs = x.reduce("npus", "sum", scatter=True)
            val0 = monarch.inspect(rs, npus=0)
            val1 = monarch.inspect(rs, npus=1)
        # reduce_scatter splits the result along the scatter dim; each rank
        # gets half the elements. 2*6=12 total, scattered over 2 ranks → 6 each.
        assert val0.numel() == 6, f"Expected 6 elements, got {val0.numel()}"
        assert torch.allclose(val0, torch.ones_like(val0) * 3.0), f"r0: {val0}"
        assert torch.allclose(val1, torch.ones_like(val1) * 3.0), f"r1: {val1}"
        _pass("reduce_scatter(sum)", f"shape={val0.shape}, val=3.0")
    except Exception as e:
        _fail("reduce_scatter(sum)", e)

    # --- alltoall(stack+scatter) ---
    try:
        with mesh.activate():
            rank = mesh.rank_tensor("npus").npu()
            x = torch.ones(2, 6, device="npu") * (rank + 1)
            atoa = x.reduce("npus", "stack", scatter=True)
            val0 = monarch.inspect(atoa, npus=0)
            val1 = monarch.inspect(atoa, npus=1)
        _pass("alltoall(stack+scatter)", f"r0={val0.shape}, r1={val1.shape}")
    except Exception as e:
        _fail("alltoall(stack+scatter)", e)

    # --- broadcast(to_mesh) ---
    try:
        npu0 = mesh.slice(npus=0)
        with npu0.activate():
            src = torch.tensor([42.0, 43.0, 44.0], device="npu")
            bcast = src.to_mesh(mesh)
        with mesh.activate():
            gathered = bcast.reduce("npus", "stack")
            val = monarch.inspect(gathered)
        expected = torch.tensor([[42.0, 43.0, 44.0], [42.0, 43.0, 44.0]])
        assert torch.allclose(val, expected), f"Broadcast mismatch: {val}"
        _pass("broadcast(to_mesh)", "npu0→all")
    except Exception as e:
        _fail("broadcast(to_mesh)", e)

    # --- rank-dependent reduce (mirrors test_reduce) ---
    try:
        with mesh.activate():
            rank = mesh.rank_tensor("npus").npu()
            x = 12 * rank + torch.arange(12, device="npu").reshape(3, 4).float()
            y_sum = x.reduce("npus", "sum")
            y_stack = x.reduce("npus", "stack")
            val_sum = monarch.inspect(y_sum)
            val_stack = monarch.inspect(y_stack)
        x0 = torch.arange(12).reshape(3, 4).float()
        x1 = 12 + torch.arange(12).reshape(3, 4).float()
        assert torch.allclose(val_sum, x0 + x1), "sum mismatch"
        assert torch.allclose(val_stack, torch.stack([x0, x1])), "stack mismatch"
        _pass("reduce(rank_values)", f"sum ✓, stack shape={val_stack.shape}")
    except Exception as e:
        _fail("reduce(rank_values)", e)

    # --- reduce pytree ---
    try:
        with mesh.activate():
            rank = mesh.rank_tensor("npus").npu()
            a = rank + torch.zeros(1, device="npu")
            b = rank + torch.ones(1, device="npu")
            td = {"a": a, "b": b}
            reduced = monarch.reduce(td, dims="npus", reduction="sum")
            val_a = monarch.inspect(reduced["a"])
            val_b = monarch.inspect(reduced["b"])
        assert torch.allclose(val_a, torch.tensor([1.0])), f"a: {val_a}"
        assert torch.allclose(val_b, torch.tensor([3.0])), f"b: {val_b}"
        _pass("reduce_pytree", f"a={val_a.item()}, b={val_b.item()}")
    except Exception as e:
        _fail("reduce_pytree", e)

    # --- sub-mesh compute ---
    try:
        npu1 = mesh.slice(npus=1)
        with npu1.activate():
            myrank = mesh.rank_tensor("npus").npu() + 1
            x = torch.ones(3, 4, device="npu") * myrank
            val = monarch.inspect(x)
        assert torch.allclose(val, torch.ones(3, 4) * 2.0), f"{val}"
        _pass("sub_mesh_compute", f"npu1 val={val.flatten()[0].item()}")
    except Exception as e:
        _fail("sub_mesh_compute", e)

    _skip("allreduce(avg)", "HCCL ReduceOp.AVG hangs on 910B/CANN9.0")


# ===================================================================
# P0-2: Multi-NPU Scale (4 NPUs)
# ===================================================================

async def test_multi_npu_scale(devs: list):
    assert len(devs) >= 4
    dev_str = ",".join(str(d) for d in devs[:4])
    _header(f"P0-2: Multi-NPU Scale (4 cards: {dev_str})")

    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = dev_str
    mesh = this_host().spawn_procs(per_host={"npus": 4})

    # --- 4-card allreduce(sum) ---
    try:
        with mesh.activate():
            rank = mesh.rank_tensor("npus").npu()
            x = torch.ones(3, 4, device="npu") * (rank + 1)
            y = x.reduce("npus", "sum")
            val = monarch.inspect(y)
        expected = torch.ones(3, 4) * 10.0  # 1+2+3+4
        assert torch.allclose(val, expected), f"got {val.flatten()[0]}"
        _pass("4npu_allreduce(sum)", f"1+2+3+4={val.flatten()[0].item()}")
    except Exception as e:
        _fail("4npu_allreduce(sum)", e)

    # --- 4-card allgather(stack) ---
    try:
        with mesh.activate():
            rank = mesh.rank_tensor("npus").npu()
            x = torch.ones(3, device="npu") * (rank + 1)
            g = x.reduce("npus", "stack")
            val = monarch.inspect(g)
        assert val.shape == (4, 3), f"shape {val.shape}"
        for i in range(4):
            assert torch.allclose(val[i], torch.ones(3) * (i + 1))
        _pass("4npu_allgather(stack)", f"shape={val.shape}")
    except Exception as e:
        _fail("4npu_allgather(stack)", e)

    # --- 4-card reduce_scatter ---
    try:
        with mesh.activate():
            rank = mesh.rank_tensor("npus").npu()
            # first dim must equal world_size (4) for scatter
            x = torch.ones(4, 4, device="npu") * (rank + 1)
            rs = x.reduce("npus", "sum", scatter=True)
            vals = [monarch.inspect(rs, npus=i) for i in range(4)]
        for i, v in enumerate(vals):
            assert torch.allclose(v, torch.ones_like(v) * 10.0), f"r{i}: {v}"
        _pass("4npu_reduce_scatter", f"shard={vals[0].shape}")
    except Exception as e:
        _fail("4npu_reduce_scatter", e)

    # --- 4-card broadcast ---
    try:
        npu0 = mesh.slice(npus=0)
        with npu0.activate():
            src = torch.tensor([100.0, 200.0], device="npu")
            bcast = src.to_mesh(mesh)
        with mesh.activate():
            gathered = bcast.reduce("npus", "stack")
            val = monarch.inspect(gathered)
        for i in range(4):
            assert torch.allclose(val[i], torch.tensor([100.0, 200.0]))
        _pass("4npu_broadcast", "[100,200]→all 4")
    except Exception as e:
        _fail("4npu_broadcast", e)

    # --- 4-card to_mesh (0→3) ---
    try:
        npu0 = mesh.slice(npus=0)
        npu3 = mesh.slice(npus=3)
        with npu0.activate():
            data = torch.tensor([7.0, 8.0, 9.0], device="npu")
            sent = data.to_mesh(npu3)
        with npu3.activate():
            val = monarch.inspect(sent)
        assert torch.allclose(val, torch.tensor([7.0, 8.0, 9.0]))
        _pass("4npu_to_mesh(0→3)", f"{val.tolist()}")
    except Exception as e:
        _fail("4npu_to_mesh(0→3)", e)

    # --- 4-card rank_tensor ---
    try:
        with mesh.activate():
            f = 10 * mesh.rank_tensor("npus").npu()
            vals = [monarch.inspect(f, npus=i) for i in range(4)]
        for i, v in enumerate(vals):
            assert v == i * 10, f"r{i}: {v}"
        _pass("4npu_rank_tensor", f"{vals}")
    except Exception as e:
        _fail("4npu_rank_tensor", e)


# ===================================================================
# P0-2b: Full-scale 8-NPU Tests
# ===================================================================

async def test_8npu_scale(devs: list):
    assert len(devs) >= 8
    dev_str = ",".join(str(d) for d in devs[:8])
    _header(f"P0-2b: 8-NPU Full Scale ({dev_str})")

    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = dev_str
    mesh = this_host().spawn_procs(per_host={"npus": 8})

    # --- 8-card allreduce(sum) ---
    try:
        with mesh.activate():
            rank = mesh.rank_tensor("npus").npu()
            x = torch.ones(4, 4, device="npu") * (rank + 1)
            y = x.reduce("npus", "sum")
            val = monarch.inspect(y)
        expected_sum = sum(range(1, 9))  # 1+2+...+8 = 36
        expected = torch.ones(4, 4) * expected_sum
        assert torch.allclose(val, expected), f"got {val.flatten()[0]}"
        _pass("8npu_allreduce(sum)", f"1+2+...+8={val.flatten()[0].item()}")
    except Exception as e:
        _fail("8npu_allreduce(sum)", e)

    # --- 8-card allreduce(max) ---
    try:
        with mesh.activate():
            rank = mesh.rank_tensor("npus").npu()
            x = torch.ones(4, device="npu") * (rank + 1)
            y = x.reduce("npus", "max")
            val = monarch.inspect(y)
        assert torch.allclose(val, torch.ones(4) * 8.0), f"Expected 8, got {val}"
        _pass("8npu_allreduce(max)", f"max={val[0].item()}")
    except Exception as e:
        _fail("8npu_allreduce(max)", e)

    # --- 8-card allreduce(min) ---
    try:
        with mesh.activate():
            rank = mesh.rank_tensor("npus").npu()
            x = torch.ones(4, device="npu") * (rank + 1)
            y = x.reduce("npus", "min")
            val = monarch.inspect(y)
        assert torch.allclose(val, torch.ones(4) * 1.0), f"Expected 1, got {val}"
        _pass("8npu_allreduce(min)", f"min={val[0].item()}")
    except Exception as e:
        _fail("8npu_allreduce(min)", e)

    # --- 8-card allgather(stack) ---
    try:
        with mesh.activate():
            rank = mesh.rank_tensor("npus").npu()
            x = torch.ones(4, device="npu") * (rank + 1)
            g = x.reduce("npus", "stack")
            val = monarch.inspect(g)
        assert val.shape == (8, 4), f"shape {val.shape}"
        for i in range(8):
            assert torch.allclose(val[i], torch.ones(4) * (i + 1)), f"r{i}: {val[i]}"
        _pass("8npu_allgather(stack)", f"shape={val.shape}")
    except Exception as e:
        _fail("8npu_allgather(stack)", e)

    # --- 8-card reduce_scatter(sum) ---
    try:
        with mesh.activate():
            rank = mesh.rank_tensor("npus").npu()
            x = torch.ones(8, 4, device="npu") * (rank + 1)
            rs = x.reduce("npus", "sum", scatter=True)
            vals = [monarch.inspect(rs, npus=i) for i in range(8)]
        expected_val = float(sum(range(1, 9)))  # 36.0
        for i, v in enumerate(vals):
            assert torch.allclose(v, torch.ones_like(v) * expected_val), f"r{i}: {v}"
        _pass("8npu_reduce_scatter(sum)", f"shard={vals[0].shape}, val={expected_val}")
    except Exception as e:
        _fail("8npu_reduce_scatter(sum)", e)

    # --- 8-card broadcast (npu0 → all) ---
    try:
        npu0 = mesh.slice(npus=0)
        with npu0.activate():
            src = torch.tensor([10.0, 20.0, 30.0, 40.0], device="npu")
            bcast = src.to_mesh(mesh)
        with mesh.activate():
            gathered = bcast.reduce("npus", "stack")
            val = monarch.inspect(gathered)
        assert val.shape == (8, 4), f"shape {val.shape}"
        for i in range(8):
            assert torch.allclose(val[i], torch.tensor([10.0, 20.0, 30.0, 40.0])), f"r{i}"
        _pass("8npu_broadcast", "[10,20,30,40]→all 8")
    except Exception as e:
        _fail("8npu_broadcast", e)

    # --- 8-card to_mesh: npu0 → npu7 (max hop) ---
    try:
        npu0 = mesh.slice(npus=0)
        npu7 = mesh.slice(npus=7)
        with npu0.activate():
            data = torch.tensor([1.0, 2.0, 3.0, 4.0], device="npu")
            sent = data.to_mesh(npu7)
        with npu7.activate():
            val = monarch.inspect(sent)
        assert torch.allclose(val, torch.tensor([1.0, 2.0, 3.0, 4.0]))
        _pass("8npu_to_mesh(0→7)", f"{val.tolist()}")
    except Exception as e:
        _fail("8npu_to_mesh(0→7)", e)

    # --- 8-card to_mesh: npu3 → npu5 (middle hop) ---
    try:
        npu3 = mesh.slice(npus=3)
        npu5 = mesh.slice(npus=5)
        with npu3.activate():
            data = torch.tensor([50.0, 60.0], device="npu")
            sent = data.to_mesh(npu5)
        with npu5.activate():
            val = monarch.inspect(sent)
        assert torch.allclose(val, torch.tensor([50.0, 60.0]))
        _pass("8npu_to_mesh(3→5)", f"{val.tolist()}")
    except Exception as e:
        _fail("8npu_to_mesh(3→5)", e)

    # --- 8-card rank_tensor ---
    try:
        with mesh.activate():
            f = 10 * mesh.rank_tensor("npus").npu()
            vals = [monarch.inspect(f, npus=i) for i in range(8)]
        for i, v in enumerate(vals):
            assert v == i * 10, f"r{i}: {v}"
        _pass("8npu_rank_tensor", f"{vals}")
    except Exception as e:
        _fail("8npu_rank_tensor", e)

    # --- 8-card large tensor allreduce ---
    try:
        with mesh.activate():
            big = torch.ones(512, 512, device="npu")
            reduced = big.reduce("npus", "sum")
            val = monarch.inspect(reduced)
        expected = torch.ones(512, 512) * 8.0
        assert torch.allclose(val, expected), f"max diff={torch.max(torch.abs(val - expected))}"
        mb = big.numel() * 4 / 1024 / 1024
        _pass("8npu_large_allreduce", f"512x512 ({mb:.1f} MB) sum=8.0")
    except Exception as e:
        _fail("8npu_large_allreduce", e)

    # --- 8-card sub-mesh: half-mesh reduce ---
    try:
        half_a = mesh.slice(npus=slice(0, 4))
        half_b = mesh.slice(npus=slice(4, 8))
        with half_a.activate():
            rank = mesh.rank_tensor("npus").npu()
            x = torch.ones(4, device="npu") * (rank + 1)
            y = x.reduce("npus", "sum")
            val_a = monarch.inspect(y)
        with half_b.activate():
            rank = mesh.rank_tensor("npus").npu()
            x = torch.ones(4, device="npu") * (rank + 1)
            y = x.reduce("npus", "sum")
            val_b = monarch.inspect(y)
        expected_a = float(1 + 2 + 3 + 4)
        expected_b = float(5 + 6 + 7 + 8)
        assert torch.allclose(val_a, torch.ones(4) * expected_a), f"half_a: {val_a}"
        assert torch.allclose(val_b, torch.ones(4) * expected_b), f"half_b: {val_b}"
        _pass("8npu_sub_mesh_reduce", f"half_a={expected_a}, half_b={expected_b}")
    except Exception as e:
        _fail("8npu_sub_mesh_reduce", e)


# ===================================================================
# P0-3: Upstream GPU Test Mirrors (2 NPUs)
# ===================================================================

async def test_mirror_gpu(devs: list):
    assert len(devs) >= 2
    dev_a, dev_b = devs[0], devs[1]
    _header(f"P0-3: GPU Test Mirrors (NPU {dev_a}, {dev_b})")

    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = f"{dev_a},{dev_b}"
    mesh = this_host().spawn_procs(per_host={"npus": 2})

    # --- mirror: test_proc_mesh_tensor_engine ---
    try:
        with mesh.activate():
            f = 10 * mesh.rank_tensor("npus").npu()
            a = monarch.inspect(f, npus=0)
            b = monarch.inspect(f, npus=1)
        assert a == 0 and b == 10
        one = mesh.slice(npus=1)
        with one.activate():
            sliced_b = monarch.slice_mesh(f, npus=1).to_mesh(one)
            c = monarch.inspect(sliced_b * 10)
        assert c == 100
        _pass("mirror_proc_mesh_tensor_engine", f"a={a}, b={b}, c={c}")
    except Exception as e:
        _fail("mirror_proc_mesh_tensor_engine", e)

    # --- mirror: basic tensor engine ---
    try:
        with mesh.activate():
            r = monarch.inspect(2 * torch.zeros(3, 4, device="npu"))
        assert torch.allclose(torch.zeros(3, 4), r)
        _pass("mirror_tensor_engine_basic", "2*zeros=zeros")
    except Exception as e:
        _fail("mirror_tensor_engine_basic", e)

    # --- mirror: test_broadcast_one ---
    try:
        for dim_idx in range(2):
            subset = mesh.slice(npus=dim_idx)
            with subset.activate():
                x = torch.rand(3, device="npu")
                y = x.to_mesh(mesh)
            with subset.activate():
                a = monarch.inspect(x)
            with mesh.activate():
                b = monarch.inspect(y.reduce("npus", reduction="stack"))
            assert torch.allclose(a.expand(2, -1), b, rtol=0, atol=1e-5), \
                f"bcast from npu{dim_idx} mismatch"
        _pass("mirror_broadcast_one", "both directions")
    except Exception as e:
        _fail("mirror_broadcast_one", e)

    # --- mirror: test_sub_mesh_reduce ---
    try:
        npu1 = mesh.slice(npus=1)
        with npu1.activate():
            myrank = mesh.rank_tensor("npus").npu() + 1
            x = torch.ones(3, 4, device="npu") * myrank
            val = monarch.inspect(x)
        assert torch.allclose(val, torch.ones(3, 4) * 2.0)
        _pass("mirror_sub_mesh_reduce", f"val={val.flatten()[0].item()}")
    except Exception as e:
        _fail("mirror_sub_mesh_reduce", e)

    # --- mirror: test_reduce (rank data) ---
    try:
        with mesh.activate():
            rank = mesh.rank_tensor("npus").npu()
            x = 12 * rank + torch.arange(12, device="npu").reshape(3, 4).float()
            y = x.reduce("npus", "sum")
            g = x.reduce("npus", "stack")
            val_y = monarch.inspect(y)
            val_g = monarch.inspect(g)
        x0 = torch.arange(12).reshape(3, 4).float()
        x1 = 12 + torch.arange(12).reshape(3, 4).float()
        assert torch.allclose(val_y, x0 + x1)
        assert torch.allclose(val_g, torch.stack([x0, x1]))
        _pass("mirror_reduce", f"sum ✓, stack {val_g.shape}")
    except Exception as e:
        _fail("mirror_reduce", e)

    # --- mirror: test_reduce_pytree ---
    try:
        with mesh.activate():
            rank = mesh.rank_tensor("npus").npu()
            a = rank + torch.zeros(1, device="npu")
            b = rank + torch.ones(1, device="npu")
            td = {"a": a, "b": b}
            monarch.reduce_(td, dims="npus", reduction="sum")
            reduced = monarch.reduce(td, dims="npus", reduction="sum")
            ia = monarch.inspect(td["a"])
            ib = monarch.inspect(td["b"])
            ra = monarch.inspect(reduced["a"])
            rb = monarch.inspect(reduced["b"])
        assert torch.allclose(ia, torch.tensor([1.0])), f"inplace a: {ia}"
        assert torch.allclose(ib, torch.tensor([3.0])), f"inplace b: {ib}"
        assert torch.allclose(ra, torch.tensor([2.0])), f"reduced a: {ra}"
        assert torch.allclose(rb, torch.tensor([6.0])), f"reduced b: {rb}"
        _pass("mirror_reduce_pytree", f"ia={ia.item()}, ra={ra.item()}")
    except Exception as e:
        _fail("mirror_reduce_pytree", e)

    # --- large tensor to_mesh ---
    try:
        npu0 = mesh.slice(npus=0)
        npu1 = mesh.slice(npus=1)
        with npu0.activate():
            big = torch.randn(1024, 1024, device="npu")
            big_on_1 = big.to_mesh(npu1)
        with npu0.activate():
            orig = monarch.inspect(big)
        with npu1.activate():
            recv = monarch.inspect(big_on_1)
        assert torch.allclose(orig, recv, atol=1e-6)
        mb = orig.numel() * 4 / 1024 / 1024
        _pass("large_tensor_to_mesh", f"1024x1024 ({mb:.1f} MB)")
    except Exception as e:
        _fail("large_tensor_to_mesh", e)

    # --- mirror: test_sub_mesh (controller.py) ---
    try:
        npu0 = mesh.slice(npus=0)
        npu1 = mesh.slice(npus=1)
        with npu0.activate():
            _ = torch.rand(3, 4, device="npu")
        with npu1.activate():
            _ = torch.rand(3, 4, device="npu")
        _pass("mirror_sub_mesh", "activate on different slices")
    except Exception as e:
        _fail("mirror_sub_mesh", e)

    # --- mirror: test_movement (controller.py) ---
    try:
        npu0 = mesh.slice(npus=0)
        npu1 = mesh.slice(npus=1)
        with npu0.activate():
            x = torch.rand(3, 4, device="npu")
            _ = x.to_mesh(npu1)
        with mesh.activate():
            a = torch.rand(3, 4, device="npu")
        a_sliced = a.slice_mesh(npus=0)
        _ = a_sliced.to_mesh(npu0)
        _ = a_sliced.to_mesh(npu1)
        _pass("mirror_movement", "to_mesh + slice_mesh→to_mesh")
    except Exception as e:
        _fail("mirror_movement", e)

    # --- mirror: test_to_mesh_cow (controller.py) ---
    try:
        with mesh.activate():
            t = torch.zeros((), device="npu")
            t2 = t.to_mesh(mesh)
            t.add_(1)
            val_t2 = monarch.inspect(t2)
            val_t = monarch.inspect(t)
        assert val_t2.item() == 0, f"cow: t2 should be 0, got {val_t2.item()}"
        assert val_t.item() == 1, f"cow: t should be 1, got {val_t.item()}"
        _pass("mirror_to_mesh_cow", "copy-on-write: t2=0, t=1")
    except Exception as e:
        _fail("mirror_to_mesh_cow", e)

    # --- mirror: test_to_mesh_pytree (controller.py) ---
    try:
        npu0 = mesh.slice(npus=0)
        npu1 = mesh.slice(npus=1)
        with npu0.activate():
            a = torch.zeros((1,), device="npu")
            b = torch.ones((1,), device="npu")
            tensor_dict = {"a": a, "b": b}
            moved = monarch.to_mesh(tensor_dict, npu1)
        with npu1.activate():
            moved["a"].add_(1)
            moved["b"].add_(1)
        moved_a = monarch.inspect(moved["a"])
        moved_b = monarch.inspect(moved["b"])
        assert torch.equal(moved_a, torch.tensor([1.0])), f"a: {moved_a}"
        assert torch.equal(moved_b, torch.tensor([2.0])), f"b: {moved_b}"
        _pass("mirror_to_mesh_pytree", f"a={moved_a.item()}, b={moved_b.item()}")
    except Exception as e:
        _fail("mirror_to_mesh_pytree", e)

    # --- mirror: test_slice_mesh_pytree (controller.py, adapted to 1D) ---
    try:
        with mesh.activate():
            rank = mesh.rank_tensor("npus").npu()
            a = rank + torch.zeros((1,), device="npu")
            b = rank + torch.ones((1,), device="npu")
        tensor_dict = {"a": a, "b": b}
        npu0 = mesh.slice(npus=0)
        npu1 = mesh.slice(npus=1)
        npu0_slices = monarch.slice_mesh(tensor_dict, npus=0)
        npu1_slices = monarch.slice_mesh(tensor_dict, npus=1)
        npu0_tensors = monarch.to_mesh(npu0_slices, npu0)
        npu1_tensors = monarch.to_mesh(npu1_slices, npu1)
        with npu0.activate():
            a0 = monarch.inspect(npu0_tensors["a"])
            b0 = monarch.inspect(npu0_tensors["b"])
        with npu1.activate():
            a1 = monarch.inspect(npu1_tensors["a"])
            b1 = monarch.inspect(npu1_tensors["b"])
        assert torch.equal(a0, torch.tensor([0.0])), f"a0: {a0}"
        assert torch.equal(b0, torch.tensor([1.0])), f"b0: {b0}"
        assert torch.equal(a1, torch.tensor([1.0])), f"a1: {a1}"
        assert torch.equal(b1, torch.tensor([2.0])), f"b1: {b1}"
        _pass("mirror_slice_mesh_pytree", f"a0=0,b0=1,a1=1,b1=2")
    except Exception as e:
        _fail("mirror_slice_mesh_pytree", e)

    # --- mirror: test_many (controller.py) — stress test ---
    try:
        with mesh.activate():
            x = torch.rand(3, 4, device="npu")
            for _ in range(512):
                x = x + torch.rand(3, 4, device="npu")
            val = monarch.inspect(x)
        assert val.shape == (3, 4), f"shape: {val.shape}"
        _pass("mirror_many_ops", f"512 sequential adds, shape={val.shape}")
    except Exception as e:
        _fail("mirror_many_ops", e)

    # --- mirror: test_torch_tensor (controller.py) ---
    try:
        with mesh.activate():
            t_cpu = torch.tensor([1, 2, 4])
            t_npu = torch.tensor([1, 2, 4], device="npu")
            val_cpu = monarch.inspect(t_cpu)
            val_npu = monarch.inspect(t_npu)
        assert torch.allclose(val_cpu, torch.tensor([1, 2, 4])), f"cpu: {val_cpu}"
        assert torch.allclose(val_npu, torch.tensor([1, 2, 4])), f"npu: {val_npu}"
        _pass("mirror_torch_tensor", "cpu+npu tensor creation")
    except Exception as e:
        _fail("mirror_torch_tensor", e)

    # --- mirror: test_torch_op_with_optional_tensors (controller.py) ---
    try:
        with mesh.activate():
            x = torch.rand(3, 4, device="npu")
            ln_with = torch.nn.LayerNorm(4, device="npu", bias=True, elementwise_affine=True)
            ln_none = torch.nn.LayerNorm(4, device="npu", bias=False, elementwise_affine=False)
            val1 = monarch.inspect(ln_with(x))
            val2 = monarch.inspect(ln_none(x))
        assert val1.shape == (3, 4), f"with: {val1.shape}"
        assert val2.shape == (3, 4), f"none: {val2.shape}"
        _pass("mirror_layernorm_optional", f"bias=T/F both ok")
    except Exception as e:
        _fail("mirror_layernorm_optional", e)


# ===================================================================
# Main
# ===================================================================

async def run_tests(devs: list, only: str = None):
    groups = {
        "collective": (test_collective_ops, 2),
        "scale": (test_multi_npu_scale, 4),
        "scale8": (test_8npu_scale, 8),
        "mirror": (test_mirror_gpu, 2),
    }
    to_run = [only] if only else list(groups.keys())
    for name in to_run:
        fn, min_devs = groups[name]
        if len(devs) < min_devs:
            print(f"\nSKIP {name}: needs {min_devs} NPUs, have {len(devs)}")
            continue
        await fn(devs)


async def main():
    parser = argparse.ArgumentParser(description="P0 NPU Test Suite")
    parser.add_argument("--devs", type=str, default=None,
                        help="Comma-separated NPU device IDs")
    parser.add_argument("--only", type=str, default=None,
                        choices=["collective", "scale", "scale8", "mirror"])
    args = parser.parse_args()

    if args.devs:
        devs = [int(d.strip()) for d in args.devs.split(",")]
    else:
        devs = _pick_free_npus(4)
        print(f"Auto-detected NPUs: {devs}")

    print("=" * 64)
    print(f"  P0 NPU Test Suite — devices: {devs}")
    print("=" * 64, flush=True)

    await run_tests(devs, args.only)

    print(f"\n{'=' * 64}")
    print(f"  Results: {_pass_count} passed, {_fail_count} failed, {_skip_count} skipped")
    if _fail_names:
        print(f"  Failed: {', '.join(_fail_names)}")
    print(f"{'=' * 64}", flush=True)
    sys.exit(1 if _fail_count > 0 else 0)


if __name__ == "__main__":
    from monarch._src.actor.actor_mesh import context
    context()
    asyncio.run(main())
