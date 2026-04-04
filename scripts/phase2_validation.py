from __future__ import annotations

import argparse

import numpy as np

from pidog_rl.config import (
    ActionScaling,
    HardwareConfig,
    ImuConfig,
    RewardShapingConfig,
    RewardWeights,
    SafetyLimits,
)
from pidog_rl.env import PiDogGaitEnv
from pidog_rl.utils import set_seed


def _assert_close(label: str, actual: float, expected: float, tol: float = 1e-6) -> None:
    if not np.isclose(actual, expected, atol=tol):
        raise AssertionError(f"{label}: expected {expected:.6f}, got {actual:.6f}")


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _build_env(
    reward_weights: RewardWeights | None = None,
    reward_shaping: RewardShapingConfig | None = None,
    imu_config: ImuConfig | None = None,
) -> PiDogGaitEnv:
    return PiDogGaitEnv(
        action_scaling=ActionScaling(),
        reward_weights=reward_weights or RewardWeights(),
        reward_shaping=reward_shaping or RewardShapingConfig(),
        imu_config=imu_config or ImuConfig(),
        hardware=HardwareConfig(use_hardware=False),
        safety=SafetyLimits(),
        max_steps=1,
    )


def _validate_reward_clipping() -> None:
    reward_weights = RewardWeights(forward=1.2, instability=0.5, instability_clip=1.0)
    env = _build_env(reward_weights=reward_weights)

    reward = env._compute_reward(distance=2.0, instability_raw=5.0)
    expected = reward_weights.forward * 2.0 - reward_weights.instability * reward_weights.instability_clip
    _assert_close("reward clip", reward, expected)


def _validate_distance_sanitize() -> None:
    reward_weights = RewardWeights()
    env = _build_env(reward_weights=reward_weights)

    _assert_close("invalid distance", env._sanitize_distance(reward_weights.invalid_distance), 0.0)
    _assert_close("nan distance", env._sanitize_distance(float("nan")), 0.0)
    _assert_close("inf distance", env._sanitize_distance(float("inf")), 0.0)
    _assert_close("neg inf distance", env._sanitize_distance(float("-inf")), 0.0)
    _assert_close("valid distance", env._sanitize_distance(0.45), 0.45)


def _validate_instability_sum() -> None:
    imu = np.array([-0.5, 0.2, -1.0], dtype=np.float32)
    expected = float(np.abs(imu).sum())
    _assert_close("instability sum", PiDogGaitEnv._instability(imu), expected)


def _validate_step_info_clipping() -> None:
    reward_weights = RewardWeights(forward=1.0, instability=0.5, instability_clip=0.5)
    env = _build_env(
        reward_weights=reward_weights,
        imu_config=ImuConfig(smoothing=0.0),
    )
    imu = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    def _fixed_run() -> tuple[float, np.ndarray]:
        return reward_weights.invalid_distance, imu

    env._run_robot = _fixed_run  # type: ignore[assignment]
    _, reward, done, info = env.step(np.zeros(env.action_dim, dtype=np.float32))

    _assert_close("distance sanitized", info["distance"], 0.0)
    _assert_close("instability clipped", info["instability"], reward_weights.instability_clip)

    expected_reward = reward_weights.forward * 0.0 - reward_weights.instability * reward_weights.instability_clip
    _assert_close("reward", reward, expected_reward)
    _assert(done, "Expected episode to finish at max_steps=1")


def _validate_hardware_disabled() -> None:
    env = _build_env()
    _assert(env.hardware is None, "Expected hardware to be disabled in validation env")


def _validate_reward_shaping_transform() -> None:
    shaping = RewardShapingConfig(scale=2.0, shift=1.0, clip_min=-1.0, clip_max=1.0)
    env = _build_env(reward_shaping=shaping)
    shaped = env._apply_reward_shaping(0.1)
    _assert_close("reward shaping clip", shaped, 1.0)


def _validate_reward_normalization_start() -> None:
    shaping = RewardShapingConfig(normalize=True)
    env = _build_env(reward_shaping=shaping)
    shaped = env._apply_reward_shaping(5.0)
    _assert_close("reward normalize first", shaped, 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 validation checks")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    set_seed(args.seed)

    _validate_reward_clipping()
    _validate_distance_sanitize()
    _validate_instability_sum()
    _validate_step_info_clipping()
    _validate_hardware_disabled()
    _validate_reward_shaping_transform()
    _validate_reward_normalization_start()

    print("Phase 2 validation checks completed.")


if __name__ == "__main__":
    main()
