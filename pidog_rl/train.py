from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import re
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from .algorithms import Algorithm, ReinforceAlgorithm
from .config import TrainingConfig
from .env import PiDogGaitEnv
from .policy import PolicyNetwork
from .utils import set_seed

DEFAULT_OUTPUT_DIR = Path("output")


def _create_run_output_dir(base_output_dir: Path) -> Path:
    """Create a unique output directory for the current run.

    Directory name format is yy_mm_dd_x where x is the run index for that day.
    """

    base_output_dir.mkdir(parents=True, exist_ok=True)
    date_prefix = datetime.now().strftime("%y_%m_%d")
    pattern = re.compile(rf"^{re.escape(date_prefix)}_(\\d+)$")

    max_index = 0
    for entry in base_output_dir.iterdir():
        if not entry.is_dir():
            continue
        match = pattern.match(entry.name)
        if match is None:
            continue
        try:
            max_index = max(max_index, int(match.group(1)))
        except ValueError:
            continue

    for index in range(max_index + 1, max_index + 10_000):
        run_dir = base_output_dir / f"{date_prefix}_{index}"
        try:
            run_dir.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            continue
        return run_dir

    raise RuntimeError("Could not allocate a unique run output directory")


@dataclass
class EpisodeStats:
    reward_total: float
    distance_total: float
    instability_total: float


@dataclass
class TrainingHistory:
    rewards: List[float] = field(default_factory=list)
    distances: List[float] = field(default_factory=list)
    instabilities: List[float] = field(default_factory=list)

    def record(self, stats: EpisodeStats) -> None:
        self.rewards.append(stats.reward_total)
        self.distances.append(stats.distance_total)
        self.instabilities.append(stats.instability_total)


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


def save_checkpoint(
    policy: PolicyNetwork,
    algorithm: Algorithm,
    episode: int,
    history: TrainingHistory,
    path: Path,
) -> None:
    torch.save(
        {
            "episode": episode,
            "policy_state_dict": policy.state_dict(),
            "algorithm_state_dict": algorithm.state_dict(),
            "history": {
                "rewards": history.rewards,
                "distances": history.distances,
                "instabilities": history.instabilities,
            },
        },
        path,
    )


def plot_results(history: TrainingHistory, output_dir: Path) -> None:
    episodes = range(1, len(history.rewards) + 1)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(episodes, history.rewards, alpha=0.4, label="per episode")
    axes[0].plot(
        episodes, _moving_avg(history.rewards, 20), linewidth=2, label="20-ep avg"
    )
    axes[0].set_ylabel("Reward")
    axes[0].legend()
    axes[0].set_title("Training Results")

    axes[1].plot(episodes, history.distances, alpha=0.4, label="per episode")
    axes[1].plot(
        episodes, _moving_avg(history.distances, 20), linewidth=2, label="20-ep avg"
    )
    axes[1].set_ylabel("Distance")
    axes[1].legend()

    axes[2].plot(episodes, history.instabilities, alpha=0.4, label="per episode")
    axes[2].plot(
        episodes, _moving_avg(history.instabilities, 20), linewidth=2, label="20-ep avg"
    )
    axes[2].set_ylabel("Instability")
    axes[2].set_xlabel("Episode")
    axes[2].legend()

    plt.tight_layout()
    plot_name = f"training_results_{output_dir.name}.png"
    fig.savefig(output_dir / plot_name, dpi=150)
    plt.close(fig)
    print(f"saved plot to {output_dir / plot_name}")


def _moving_avg(values: List[float], window: int) -> List[float]:
    cumsum = np.cumsum(values)
    result = np.empty_like(cumsum)
    result[:window] = cumsum[:window] / np.arange(1, window + 1)
    result[window:] = (cumsum[window:] - cumsum[:-window]) / window
    return result.tolist()


def _create_algorithm(
    algorithm_name: str,
    policy: PolicyNetwork,
    learning_rate: float,
    config: TrainingConfig,
) -> Algorithm:
    """Factory function to create an algorithm instance based on config.

    Args:
        algorithm_name: Name of the algorithm to instantiate
        policy: Policy network to train
        learning_rate: Learning rate for the optimizer
        config: Training configuration

    Returns:
        Algorithm instance

    Raises:
        ValueError: If algorithm_name is not recognized
    """
    algorithm_name = algorithm_name.lower()
    if algorithm_name == "reinforce":
        return ReinforceAlgorithm(policy, learning_rate, config.episode)
    else:
        raise ValueError(
            f"Unknown algorithm: {algorithm_name}. Supported: 'reinforce'"
        )


def train(config: TrainingConfig, output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    set_seed(config.seed)
    device = torch.device("cpu")
    run_output_dir = _create_run_output_dir(output_dir)
    print(f"writing outputs to {run_output_dir}")

    env = PiDogGaitEnv(
        action_scaling=config.action_scaling,
        reward_weights=config.reward_weights,
        imu_config=config.imu,
        hardware=config.hardware,
        safety=config.safety,
        max_steps=config.episode.max_steps,
    )

    policy = PolicyNetwork(env.state_dim, env.action_dim).to(device)
    algorithm = _create_algorithm(config.algorithm, policy, config.learning_rate, config)
    history = TrainingHistory()

    checkpoint_interval = max(1, config.episodes // 5)

    for episode in range(1, config.episodes + 1):
        log_probs, rewards, stats = run_episode(env, policy, device)
        loss = algorithm.compute_loss(log_probs, rewards)
        algorithm.update(loss)

        history.record(stats)

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

        if episode % checkpoint_interval == 0:
            ckpt_path = run_output_dir / f"checkpoint_ep{episode}.pt"
            save_checkpoint(policy, algorithm, episode, history, ckpt_path)
            print(f"saved checkpoint to {ckpt_path}")

    # Save final checkpoint and plot
    final_path = run_output_dir / "checkpoint_final.pt"
    save_checkpoint(policy, algorithm, config.episodes, history, final_path)
    print(f"saved final checkpoint to {final_path}")
    plot_results(history, run_output_dir)


def main() -> None:
    config = TrainingConfig()
    train(config)


if __name__ == "__main__":
    main()
