#!/usr/bin/env python3
"""
Simple GRPO on NPU — Single mesh, no RDMA.
Tests basic actor + NPU compute without inter-mesh communication.
"""
import os, sys
os.environ["PYTHONUNBUFFERED"] = "1"

import asyncio
import torch

try:
    import torch_npu
except ImportError:
    print("ERROR: torch_npu not available", flush=True)
    sys.exit(1)

from monarch.actor import Actor, endpoint, this_host

G = 4
STATE_DIM = 4
ACTION_DIM = 4
DEVICE = "npu"

class SimpleTrainer(Actor):
    def __init__(self):
        print(f"  [SimpleTrainer] __init__ on device {DEVICE}", flush=True)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(STATE_DIM, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, ACTION_DIM),
        ).to(DEVICE)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.step_count = 0
        print(f"  [SimpleTrainer] init done", flush=True)

    @endpoint
    async def train_step(self) -> float:
        states = torch.randn(2, STATE_DIM, device=DEVICE)
        states_expanded = states.repeat_interleave(G, 0)

        with torch.no_grad():
            logits = self.model(states_expanded)
            probs = torch.softmax(logits, dim=-1)
            actions = torch.multinomial(probs, num_samples=1).squeeze(-1)
            old_logps = torch.log_softmax(logits, dim=-1).gather(
                1, actions.unsqueeze(-1)
            ).squeeze(-1)

        rewards = -torch.abs(actions.float() - actions.float().mean())
        rewards_reshaped = rewards.view(2, G)
        baselines = rewards_reshaped.mean(dim=1, keepdim=True)
        advantages = (rewards_reshaped - baselines).reshape(-1)
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        logits = self.model(states_expanded)
        log_probs = torch.log_softmax(logits, dim=-1)
        new_logps = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        ratio = (new_logps - old_logps).exp()
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * advantages
        loss = -torch.min(unclipped, clipped).mean()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.step_count += 1
        loss_val = loss.detach().cpu().item()
        return loss_val


async def main():
    print("=" * 50, flush=True)
    print("Simple GRPO on NPU — Single mesh, no RDMA", flush=True)
    print("=" * 50, flush=True)

    print("[1] Creating proc mesh...", flush=True)
    mesh = this_host().spawn_procs(per_host={"gpus": 1})

    print("[2] Spawning trainer...", flush=True)
    trainer = mesh.spawn("trainer", SimpleTrainer)

    print("[3] Training (5 steps)...", flush=True)
    for step in range(5):
        loss = await trainer.train_step.call_one()
        print(f"  [Step {step}] loss={loss:.4f}", flush=True)

    print("Done!", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
