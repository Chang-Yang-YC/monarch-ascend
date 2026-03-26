#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Borrow feature tests on Ascend NPU.

Borrow allows safe, deterministic cross-stream tensor sharing — inspired by
Rust's borrow checker.  This script verifies:

  1. Immutable borrow blocks mutation on the original stream.
  2. Mutable borrow allows in-place ops on the borrowing stream.
  3. Data written through a mutable borrow is visible after drop.
  4. Mutable borrow blocks reads on the original stream.

Usage:
    export ASCEND_RT_VISIBLE_DEVICES=0,1
    python tests/hixl/app/test_borrow_npu.py
"""

import sys

import torch
import torch_npu  # noqa: F401

from monarch import Stream, fetch_shard
from monarch._src.job.process import ProcessJob
from monarch.common.device_mesh import no_mesh
from monarch.mesh_controller import spawn_tensor_engine


def setup_npu_mesh(n_npu: int = 1):
    job = ProcessJob({"hosts": 1})
    state = job.state(cached_path=None)
    proc_mesh = state.hosts.spawn_procs(per_host={"npu": n_npu})
    dm = spawn_tensor_engine(proc_mesh)
    dm = dm.rename(npu="npu")
    return dm, job


def test_immutable_borrow_blocks_mutation(dm):
    """Immutable borrow should prevent writes on the original stream."""
    with dm.activate():
        x = torch.rand(3, 4, device="npu")
        x.abs_()

        s = Stream("other")
        b, drop = s.borrow(x)

        try:
            x.abs_()
            return False, "mutation was allowed during immutable borrow"
        except TypeError:
            pass

        with s.activate():
            _ = b.add(b)
        drop.drop()

        x.abs_()
    return True, "immutable borrow blocks mutation correctly"


def test_mutable_borrow_lifecycle(dm):
    """Mutable borrow should allow in-place ops, then restore after drop."""
    with dm.activate():
        x = torch.rand(3, 4, device="npu")
        s = Stream("other")

        b, drop = s.borrow(x, mutable=True)
        with s.activate():
            b.abs_()
        drop.drop()

        x.abs_()
    return True, "mutable borrow + drop lifecycle OK"


def test_mutable_borrow_data_roundtrip(dm):
    """Data written via mutable borrow should be visible after drop."""
    with dm.activate():
        y = torch.ones(4, device="npu") * 42.0
        s = Stream("other")

        b, drop = s.borrow(y, mutable=True)
        with s.activate():
            b.add_(1.0)  # 42 -> 43
        drop.drop()

        result = fetch_shard(y).result()
        with no_mesh.activate():
            val = result.sum().item()

    expected = 172.0  # 43 * 4
    if abs(val - expected) > 0.1:
        return False, f"expected {expected}, got {val}"
    return True, f"fetch_shard returned {val} as expected"


def test_mutable_borrow_blocks_reads(dm):
    """Mutable borrow should block reads on the original stream."""
    with dm.activate():
        z = torch.rand(5, device="npu")
        s = Stream("other")

        _, drop = s.borrow(z, mutable=True)
        try:
            _ = z.add(z)
            return False, "read was allowed during mutable borrow"
        except TypeError:
            pass

        drop.drop()
        _ = z.add(z)
    return True, "mutable borrow blocks reads correctly"


TESTS = [
    ("Immutable borrow blocks mutation", test_immutable_borrow_blocks_mutation),
    ("Mutable borrow lifecycle", test_mutable_borrow_lifecycle),
    ("Mutable borrow data roundtrip", test_mutable_borrow_data_roundtrip),
    ("Mutable borrow blocks reads", test_mutable_borrow_blocks_reads),
]


def main():
    print("=" * 60)
    print("Borrow Feature Test on NPU")
    print("=" * 60)
    print(f"torch: {torch.__version__}")
    print(f"NPU devices: {torch.npu.device_count()}")
    print()

    dm, job = setup_npu_mesh(1)
    print(f"DeviceMesh: {dm}\n")

    passed, failed = 0, 0
    try:
        for name, fn in TESTS:
            ok, msg = fn(dm)
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {name}: {msg}")
            if ok:
                passed += 1
            else:
                failed += 1
    finally:
        dm.exit()
        job._kill()

    print()
    print(f"Results: {passed} passed, {failed} failed, {len(TESTS)} total")
    print("=" * 60)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
