from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from .config import TrainingConfig
from .env import PiDogGaitEnv
from .policy import PolicyNetwork


def load_policy(checkpoint_path: Path, state_dim: int, action_dim: int) -> PolicyNetwork:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    policy = PolicyNetwork(state_dim, action_dim)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
    return policy


def run_inference(
    checkpoint_path: Path, steps: int, use_hardware: bool, deterministic: bool
) -> None:
    config = TrainingConfig()
    hardware = replace(config.hardware, use_hardware=use_hardware)

    env = PiDogGaitEnv(
        action_scaling=config.action_scaling,
        reward_weights=config.reward_weights,
        imu_config=config.imu,
        hardware=hardware,
        safety=config.safety,
        max_steps=steps,
    )

    policy = load_policy(checkpoint_path, env.state_dim, env.action_dim)
    state = env.reset()

    done = False
    step = 0
    while not done:
        state_tensor = torch.from_numpy(state).float()
        with torch.no_grad():
            if deterministic:
                action, _ = policy(state_tensor)
                action = action.numpy()
            else:
                output = policy.sample(state_tensor)
                action = output.action.numpy()

        state, reward, done, info = env.step(action)
        step += 1
        print(
            "step",
            step,
            "reward",
            f"{reward:.3f}",
            "distance",
            f"{info['distance']:.3f}",
            "instability",
            f"{info['instability']:.3f}",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trained PiDog gait policy.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("output") / "26_04_04_1" / "checkpoint_final.pt",
        help="Path to a trained checkpoint.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of control steps to run.",
    )
    parser.add_argument(
        "--use-hardware",
        action="store_true",
        help="Enable PiDog hardware integration.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of using the mean.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_inference(
        checkpoint_path=args.checkpoint,
        steps=args.steps,
        use_hardware=args.use_hardware,
        deterministic=not args.stochastic,
    )


if __name__ == "__main__":
    main()
