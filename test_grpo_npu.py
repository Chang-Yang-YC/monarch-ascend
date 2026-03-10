#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
GRPO on NPU — Two-mesh test (Learner mesh + Generator mesh)
============================================================
Adapted from docs/source/examples/grpo_actor.py for Ascend NPU.

Changes from GPU version:
  - "cuda" → "npu"
  - "gpus" → "gpus" (monarch maps this to the accelerator)
  - Replaced torch.distributions.Categorical with manual softmax+multinomial
    (Categorical may not be fully supported on NPU)
  - Removed kl_divergence import (replaced with manual computation)

Architecture:
  learner_mesh (1 NPU): Learner + Scorer + TrajectoryQueue + ReplayBuffer
  gen_mesh     (2 NPU): Generator ×2

  Mesh间通信: Generator.update() reads weights from Learner via RDMABuffer (HIXL)
"""

import os
os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")

import asyncio
import copy
import random
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

try:
    import torch_npu
except ImportError:
    print("ERROR: torch_npu not available")
    sys.exit(1)

from monarch.actor import Actor, endpoint, this_host
from monarch.rdma import RDMABuffer

# Config
G = 8  # group size for GRPO
STATE_DIM = 4
ACTION_DIM = 4
DEVICE = "npu"


@dataclass
class TrajectorySlice:
    policy_version: int
    state: torch.Tensor
    actions: torch.Tensor
    old_logps: torch.Tensor
    rewards: torch.Tensor


@dataclass
class TrainingBatch:
    states: torch.Tensor
    actions: torch.Tensor
    old_logps: torch.Tensor
    rewards: torch.Tensor
    policy_versions: List[int]


class TrajectoryQueue(Actor):
    def __init__(self):
        self.queue: asyncio.Queue[TrajectorySlice] = asyncio.Queue()

    @endpoint
    async def put(self, slice: TrajectorySlice) -> None:
        await self.queue.put(slice)

    @endpoint
    async def get(self) -> TrajectorySlice:
        return await self.queue.get()


class ReplayBuffer(Actor):
    def __init__(self):
        self.storage: List[Tuple[int, TrajectorySlice]] = []
        self.storage_event = asyncio.Event()

    @endpoint
    async def put(self, slice: TrajectorySlice) -> None:
        self.storage.append((slice.policy_version, slice))
        self.storage_event.set()

    async def _wait_for_storage(self):
        if not self.storage:
            await self.storage_event.wait()

    @endpoint
    async def sample_from(self, k: int) -> List[TrajectorySlice]:
        try:
            await asyncio.wait_for(self._wait_for_storage(), timeout=10.0)
        except asyncio.TimeoutError:
            raise RuntimeError("Timeout waiting for ReplayBuffer to be populated")

        policy_versions = [version + 1 for version, _ in self.storage]
        total = sum(policy_versions)
        probs = [v / total for v in policy_versions]
        indices = list(range(len(self.storage)))
        chosen_indices = random.choices(indices, weights=probs, k=k)
        return [self.storage[i][1] for i in chosen_indices]


class Scorer(Actor):
    def __init__(self, trajectory_queue: Any, replay_buffer: Any):
        self.trajectory_queue = trajectory_queue
        self.replay_buffer = replay_buffer
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM + 1, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
        ).to(DEVICE)
        self.running = False

    async def _score_slice(self, slice: TrajectorySlice) -> None:
        s = slice.state.to(DEVICE).unsqueeze(0).repeat(G, 1)
        a = slice.actions.to(DEVICE).float().unsqueeze(-1)
        rewards = self.net(torch.cat([s, a], dim=-1)).squeeze(-1)

        scored = TrajectorySlice(
            policy_version=slice.policy_version,
            state=slice.state,
            actions=slice.actions,
            old_logps=slice.old_logps,
            rewards=rewards,
        )
        await self.replay_buffer.put.call(scored)

    @endpoint
    async def run(self) -> None:
        if self.running:
            return
        self.running = True
        try:
            while self.running:
                try:
                    slice_ = await asyncio.wait_for(
                        self.trajectory_queue.get.call_one(),
                        timeout=10.0,
                    )
                    await self._score_slice(slice_)
                except asyncio.TimeoutError:
                    continue
        except Exception as e:
            print(f"Scorer event loop error: {e}")
        finally:
            self.running = False

    @endpoint
    async def stop_scoring(self) -> None:
        self.running = False


class Learner(Actor):
    def __init__(self, replay_buffer: Any):
        self.model = nn.Sequential(
            nn.Linear(STATE_DIM, 16), nn.Tanh(), nn.Linear(16, ACTION_DIM)
        ).to(DEVICE)
        self.ref_model = copy.deepcopy(self.model)
        for p in self.ref_model.parameters():
            p.requires_grad = False
        self.ref_model.eval()

        self.optim = optim.Adam(self.model.parameters(), lr=1e-3, eps=1e-5)
        self.eps = 0.2
        self.kl_coeff = 0.1
        self.policy_version = 0
        self.replay_buffer = replay_buffer
        self.batch_size = 2
        self.generators: Optional[Any] = None
        self._weights_handle: Dict[str, Tuple[torch.Tensor, RDMABuffer]] = {}

    @endpoint
    async def init_generators(self, generators: Any) -> None:
        self.generators = generators

    @endpoint
    async def weights_handle(self) -> Dict[str, Tuple[torch.Tensor, RDMABuffer]]:
        self._weights_handle = {
            k: (v, RDMABuffer(v.view(torch.uint8).flatten()))
            for k, v in self.model.state_dict().items()
        }
        return self._weights_handle

    def _compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        batch_size = rewards.shape[0] // G
        rewards_reshaped = rewards.view(batch_size, G)
        baselines = rewards_reshaped.mean(dim=1, keepdim=True)
        advantages = (rewards_reshaped - baselines).reshape(-1)
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    def _apply_policy_update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_logps: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.model(states)
        log_probs = torch.log_softmax(logits, dim=-1)
        new_logps = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        ratio = (new_logps - old_logps).exp()
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantages
        ppo_loss = -torch.min(unclipped, clipped).mean()

        with torch.no_grad():
            ref_logits = self.ref_model(states)
        ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        kl = (probs * (log_probs - ref_log_probs)).sum(dim=-1).mean()

        loss = ppo_loss + self.kl_coeff * kl
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optim.step()
        self.policy_version += 1
        return loss.detach()

    @endpoint
    async def step(self) -> torch.Tensor:
        if self.generators:
            await self.generators.update.call(self.policy_version)

        slices = await self.replay_buffer.sample_from.call_one(self.batch_size)
        raw_states = torch.stack([s.state for s in slices])
        actions = torch.cat([s.actions for s in slices])
        old_logps = torch.cat([s.old_logps for s in slices])
        rewards = torch.cat([s.rewards for s in slices])

        states = raw_states.repeat_interleave(G, 0).to(DEVICE)
        actions, old_logps, rewards = [
            x.to(DEVICE) for x in (actions, old_logps, rewards)
        ]
        advs = self._compute_advantages(rewards)
        return self._apply_policy_update(states, actions, old_logps, advs)


class GeneratorState:
    READY_TO_GENERATE = "READY_TO_GENERATE"
    READY_TO_UPDATE = "READY_TO_UPDATE"


class Generator(Actor):
    def __init__(self, weight_buffers, trajectory_queue):
        self.model = nn.Sequential(
            nn.Linear(STATE_DIM, 16), nn.Tanh(), nn.Linear(16, ACTION_DIM)
        ).to(DEVICE)
        self.weight_buffers = weight_buffers
        self.trajectory_queue = trajectory_queue
        self.state = GeneratorState.READY_TO_GENERATE
        self.cond = asyncio.Condition()
        self.policy_version = 0

    @endpoint
    async def generate(self, state: torch.Tensor) -> None:
        async with self.cond:
            await self.cond.wait_for(
                lambda: self.state == GeneratorState.READY_TO_GENERATE
            )

            x = state.to(DEVICE).unsqueeze(0).repeat(G, 1)
            with torch.no_grad():
                logits = self.model(x)
            probs = torch.softmax(logits, dim=-1)
            acts = torch.multinomial(probs, num_samples=1).squeeze(-1)
            logps = torch.log_softmax(logits, dim=-1).gather(
                1, acts.unsqueeze(-1)
            ).squeeze(-1)

            slice_ = TrajectorySlice(
                self.policy_version,
                state,
                acts,
                logps,
                torch.zeros(G),
            )

        await self.trajectory_queue.put.call(slice_)

        async with self.cond:
            self.state = GeneratorState.READY_TO_UPDATE
            self.cond.notify_all()

    @endpoint
    async def update(self, version: int) -> None:
        async with self.cond:
            sd = self.model.state_dict()
            for n, (_, b) in self.weight_buffers.items():
                await b.read_into(sd[n].view(torch.uint8).flatten())
            self.model.load_state_dict(sd)
            self.policy_version = version
            self.state = GeneratorState.READY_TO_GENERATE
            self.cond.notify_all()


async def main():
    print("=" * 60)
    print("GRPO on NPU — Two-mesh test")
    print("  learner_mesh: 1 NPU (Learner + Scorer + Queues)")
    print("  gen_mesh:     2 NPU (Generator x2)")
    print("  Inter-mesh weight sync via RDMA (HIXL)")
    print("=" * 60)

    learner_mesh = this_host().spawn_procs(per_host={"gpus": 1})
    gen_mesh = this_host().spawn_procs(per_host={"gpus": 2})

    print("[1/5] Spawning actors on learner_mesh...")
    traj_q = learner_mesh.spawn("traj", TrajectoryQueue)
    replay_buf = learner_mesh.spawn("rb", ReplayBuffer)
    learner = learner_mesh.spawn("learner", Learner, replay_buf)
    scorer = learner_mesh.spawn("scorer", Scorer, traj_q, replay_buf)

    print("[2/5] Getting weight handles and spawning generators...")
    wb = await learner.weights_handle.call_one()
    generators = gen_mesh.spawn("generator", Generator, wb, traj_q)
    await learner.init_generators.call(generators)

    print("[3/5] Initial generation...")
    await generators.generate.call(torch.randn(STATE_DIM))

    print("[4/5] Starting scorer event loop...")
    scorer_run_future = scorer.run.call_one()

    print("[5/5] Training loop (5 steps)...")
    for step in range(5):
        state = torch.randn(STATE_DIM)
        _, loss = await asyncio.gather(
            generators.generate.call(state),
            learner.step.call_one(),
        )
        print(f"  [Step {step:02d}] loss={loss:.4f}")

    print("Stopping scorer...")
    await scorer.stop_scoring.call_one()
    await scorer_run_future

    print("GRPO training complete!")


if __name__ == "__main__":
    asyncio.run(main())
