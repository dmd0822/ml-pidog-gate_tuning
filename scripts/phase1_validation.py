from __future__ import annotations

import argparse
from typing import Iterable

import numpy as np
import torch
from torch import nn

from pidog_rl.algorithms import ReinforceAlgorithm
from pidog_rl.config import (
    ActionScaling,
    EpisodeConfig,
    HardwareConfig,
    ImuConfig,
    RewardWeights,
    SafetyLimits,
)
from pidog_rl.env import PiDogGaitEnv
from pidog_rl.utils import compute_returns, set_seed


def _assert_close(label: str, actual: float, expected: float, tol: float = 1e-6) -> None:
    if not np.isclose(actual, expected, atol=tol):
        raise AssertionError(f"{label}: expected {expected:.6f}, got {actual:.6f}")


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _validate_action_scaling() -> None:
    scaling = ActionScaling(
        stride_length=0.05,
        step_height=0.02,
        cycle_time=0.04,
        lateral_offset=0.03,
    )
    safety = SafetyLimits(
        stride_length_min=0.04,
        stride_length_max=0.12,
        step_height_min=0.015,
        step_height_max=0.05,
        cycle_time_min=0.18,
        cycle_time_max=0.42,
        lateral_offset_min=-0.03,
        lateral_offset_max=0.03,
    )
    env = PiDogGaitEnv(
        action_scaling=scaling,
        reward_weights=RewardWeights(),
        imu_config=ImuConfig(),
        hardware=HardwareConfig(use_hardware=False),
        safety=safety,
        max_steps=1,
    )
    initial = env.gait.as_array()
    action = np.array([10.0, -10.0, 10.0, -10.0], dtype=np.float32)
    env._apply_action(action)
    clipped = np.clip(action, -1.0, 1.0)

    expected_stride = np.clip(
        initial[0] + clipped[0] * scaling.stride_length,
        safety.stride_length_min,
        safety.stride_length_max,
    )
    expected_step_height = np.clip(
        initial[1] + clipped[1] * scaling.step_height,
        safety.step_height_min,
        safety.step_height_max,
    )
    expected_cycle_time = np.clip(
        initial[2] + clipped[2] * scaling.cycle_time,
        safety.cycle_time_min,
        safety.cycle_time_max,
    )
    expected_lateral = np.clip(
        initial[3] + clipped[3] * scaling.lateral_offset,
        safety.lateral_offset_min,
        safety.lateral_offset_max,
    )

    _assert_close("stride_length", env.gait.stride_length, float(expected_stride))
    _assert_close("step_height", env.gait.step_height, float(expected_step_height))
    _assert_close("cycle_time", env.gait.cycle_time, float(expected_cycle_time))
    _assert_close("lateral_offset", env.gait.lateral_offset, float(expected_lateral))


def _require_attr(obj: object, name: str) -> None:
    _assert(hasattr(obj, name), f"Expected {obj.__class__.__name__} to expose `{name}`")


def _validate_ema_baseline() -> None:
    policy = nn.Linear(1, 1, bias=False)
    config = EpisodeConfig()
    algorithm = ReinforceAlgorithm(policy, learning_rate=1e-3, config=config)

    baseline_alpha = float(config.baseline_ema_alpha)
    _assert(0.0 < baseline_alpha < 1.0, "baseline_ema_alpha should be in (0, 1)")

    rewards_1 = [1.0, 0.5, -0.2]
    log_probs_1 = [torch.tensor(0.1), torch.tensor(0.2), torch.tensor(-0.3)]
    returns_1 = compute_returns(rewards_1, config.gamma)
    expected_mean_1 = float(np.mean(returns_1))
    algorithm.compute_loss(log_probs_1, rewards_1)
    _assert_close("baseline init", float(algorithm.last_baseline), expected_mean_1)

    rewards_2 = [0.1, 0.1, 0.1]
    log_probs_2 = [torch.tensor(0.05), torch.tensor(-0.05), torch.tensor(0.02)]
    returns_2 = compute_returns(rewards_2, config.gamma)
    expected_mean_2 = float(np.mean(returns_2))
    algorithm.compute_loss(log_probs_2, rewards_2)
    expected_ema = (1.0 - baseline_alpha) * expected_mean_1 + baseline_alpha * expected_mean_2
    _assert_close("baseline ema", float(algorithm.last_baseline), expected_ema)

    state = algorithm.state_dict()
    _assert("baseline_ema" in state, "Expected baseline_ema to be checkpointed in state_dict")
    _assert_close("baseline_ema state", float(state["baseline_ema"]), expected_ema)


def _validate_grad_clipping() -> None:
    policy = nn.Linear(1, 1, bias=False)
    config = EpisodeConfig(grad_clip_norm=0.75)
    algorithm = ReinforceAlgorithm(policy, learning_rate=1e-3, config=config)

    max_grad_norm = config.grad_clip_norm
    _assert(float(max_grad_norm) > 0.0, "grad_clip_norm should be > 0")

    called = {"flag": False, "max_norm": None}
    original = torch.nn.utils.clip_grad_norm_

    def _wrapped(params: Iterable[torch.Tensor], max_norm: float, *args, **kwargs):
        called["flag"] = True
        called["max_norm"] = float(max_norm)
        return original(params, max_norm, *args, **kwargs)

    torch.nn.utils.clip_grad_norm_ = _wrapped
    try:
        log_prob = policy(torch.tensor([1.0]))
        loss = algorithm.compute_loss([log_prob.squeeze()], [1.0])
        algorithm.update(loss)
    finally:
        torch.nn.utils.clip_grad_norm_ = original

    _assert(called["flag"], "Expected torch.nn.utils.clip_grad_norm_ to be called")
    _assert_close("clip max_norm", float(called["max_norm"]), float(max_grad_norm))
    _assert(algorithm.last_grad_norm is not None, "Expected last_grad_norm to be recorded")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 validation checks")
    parser.add_argument("--check-ema-baseline", action="store_true")
    parser.add_argument("--check-grad-clip", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    set_seed(args.seed)

    _validate_action_scaling()
    if args.check_ema_baseline:
        _validate_ema_baseline()
    if args.check_grad_clip:
        _validate_grad_clipping()

    print("Phase 1 validation checks completed.")


if __name__ == "__main__":
    main()
