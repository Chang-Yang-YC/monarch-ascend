#!/usr/bin/env python3
"""
Minimal test: HCCL collective communication between NPU 5 and NPU 6.
Uses torch.distributed with HCCL backend, bypassing Monarch.
"""
import os
import sys
import time
import multiprocessing as mp


def worker(rank, world_size, dev_a, dev_b):
    dev_id = dev_a if rank == 0 else dev_b
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(dev_id)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    import torch
    import torch_npu  # noqa: F401

    print(f"[rank {rank}] ASCEND_RT_VISIBLE_DEVICES={dev_id}, npu_count={torch.npu.device_count()}", flush=True)
    torch.npu.set_device(0)
    print(f"[rank {rank}] set_device(0) OK, current={torch.npu.current_device()}", flush=True)

    torch.distributed.init_process_group(backend="hccl", world_size=world_size, rank=rank)
    print(f"[rank {rank}] init_process_group OK", flush=True)

    t = torch.ones(4, device="npu:0") * (rank + 1)
    print(f"[rank {rank}] before all_reduce: {t}", flush=True)
    torch.distributed.all_reduce(t)
    print(f"[rank {rank}] after all_reduce: {t}", flush=True)

    torch.distributed.destroy_process_group()
    print(f"[rank {rank}] DONE", flush=True)


if __name__ == "__main__":
    dev_a, dev_b = 5, 6
    world_size = 2
    mp.set_start_method("spawn")
    procs = []
    for rank in range(world_size):
        p = mp.Process(target=worker, args=(rank, world_size, dev_a, dev_b))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    print("All processes finished. Exit codes:", [p.exitcode for p in procs])
