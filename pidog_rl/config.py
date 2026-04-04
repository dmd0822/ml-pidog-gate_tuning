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
    apply_gait_method: str = "do_action"
    apply_gait_action: str | None = "forward"
    apply_gait_speed: int = 90
    stride_length_min: float = 0.04
    stride_length_max: float = 0.12
    cycle_time_min: float = 0.18
    cycle_time_max: float = 0.42
    step_height_min: float = 0.015
    step_height_max: float = 0.05
    lateral_offset_min: float = -0.03
    lateral_offset_max: float = 0.03
    speed_min: int = 60
    speed_max: int = 100
    run_duration_min: float = 0.2
    run_duration_max: float = 0.6
    body_height_min: float = 70.0
    body_height_max: float = 95.0
    lateral_offset_max_mm: float = 20.0
    stand_method: str = "do_action"
    stand_action: str | None = "stand"
    stand_speed: int = 65
    run_method: str = "wait_all_done"
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
    episodes: int = 2000
