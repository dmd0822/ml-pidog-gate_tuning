from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionScaling:
    """Scales raw policy outputs into safe parameter deltas."""

    stride_length: float = 0.01
    step_height: float = 0.005
    cycle_time: float = 0.02
    lateral_offset: float = 0.005


@dataclass(frozen=True)
class RewardWeights:
    """Balances forward progress against instability."""

    forward: float = 1.0
    instability: float = 0.35


@dataclass(frozen=True)
class EpisodeConfig:
    """Episode length and discount for return estimation."""

    max_steps: int = 50
    gamma: float = 0.98


@dataclass(frozen=True)
class ImuConfig:
    """IMU smoothing for real-world noise handling."""

    smoothing: float = 0.6


@dataclass(frozen=True)
class HardwareConfig:
    """Configuration for the SunFounder PiDog hardware integration."""

    use_hardware: bool = False
    run_duration_sec: float = 0.5
    apply_gait_method: str = "set_gait"
    run_method: str = "walk"
    imu_method: str = "get_imu"
    distance_method: str = "get_distance"
    imu_keys: tuple[str, str, str] = ("roll", "pitch", "yaw")


@dataclass(frozen=True)
class SafetyLimits:
    """Safety limits to protect servos and avoid unsafe gaits."""

    stride_length_min: float = 0.04
    stride_length_max: float = 0.12
    step_height_min: float = 0.015
    step_height_max: float = 0.05
    cycle_time_min: float = 0.18
    cycle_time_max: float = 0.42
    lateral_offset_min: float = -0.03
    lateral_offset_max: float = 0.03


@dataclass(frozen=True)
class TrainingConfig:
    """Top-level configuration bundle."""

    action_scaling: ActionScaling = ActionScaling()
    reward_weights: RewardWeights = RewardWeights()
    episode: EpisodeConfig = EpisodeConfig()
    imu: ImuConfig = ImuConfig()
    hardware: HardwareConfig = HardwareConfig()
    safety: SafetyLimits = SafetyLimits()
    seed: int = 7
    learning_rate: float = 1e-3
    episodes: int = 200
