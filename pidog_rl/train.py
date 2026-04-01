from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch import nn

from .config import TrainingConfig
from .env import PiDogGaitEnv
from .policy import PolicyNetwork
from .utils import compute_returns, set_seed


@dataclass
class EpisodeStats:
    reward_total: float
    distance_total: float
    instability_total: float


def run_episode(
    env: PiDogGaitEnv, policy: PolicyNetwork, device: torch.device
) -> tuple[List[torch.Tensor], List[float], EpisodeStats]:
    state = env.reset()
    log_probs: List[torch.Tensor] = []
    rewards: List[float] = []
    distance_total = 0.0
    instability_total = 0.0

    done = False
    while not done:
        # State design: include gait + IMU so the policy can trade speed vs stability.
        state_tensor = torch.from_numpy(state).float().to(device)
        output = policy.sample(state_tensor)

        # Exploration vs stability: stochastic sampling supports exploration.
        action = output.action.detach().cpu().numpy()
        next_state, reward, done, info = env.step(action)

        log_probs.append(output.log_prob)
        rewards.append(reward)
        distance_total += info["distance"]
        instability_total += info["instability"]
        state = next_state

    return log_probs, rewards, EpisodeStats(
        reward_total=float(np.sum(rewards)),
        distance_total=distance_total,
        instability_total=instability_total,
    )


def train(config: TrainingConfig) -> None:
    set_seed(config.seed)
    device = torch.device("cpu")

    env = PiDogGaitEnv(
        action_scaling=config.action_scaling,
        reward_weights=config.reward_weights,
        imu_config=config.imu,
        hardware=config.hardware,
        safety=config.safety,
        max_steps=config.episode.max_steps,
    )

    policy = PolicyNetwork(env.state_dim, env.action_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)

    for episode in range(1, config.episodes + 1):
        log_probs, rewards, stats = run_episode(env, policy, device)
        returns = compute_returns(rewards, config.episode.gamma)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)

        # Policy gradient: maximize expected return via REINFORCE loss.
        loss = -torch.stack(log_probs).mul(returns_tensor).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(
            "episode",
            episode,
            "reward",
            f"{stats.reward_total:.3f}",
            "distance",
            f"{stats.distance_total:.3f}",
            "instability",
            f"{stats.instability_total:.3f}",
        )


def main() -> None:
    config = TrainingConfig()
    train(config)


if __name__ == "__main__":
    main()
