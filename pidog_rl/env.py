from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .config import ActionScaling, HardwareConfig, ImuConfig, RewardWeights, SafetyLimits
from .pidog_hw import PidogHardware
from .utils import exp_smooth


@dataclass
class GaitParameters:
    stride_length: float
    step_height: float
    cycle_time: float
    lateral_offset: float

    def as_array(self) -> np.ndarray:
        return np.array(
            [
                self.stride_length,
                self.step_height,
                self.cycle_time,
                self.lateral_offset,
            ],
            dtype=np.float32,
        )

    def as_dict(self) -> Dict[str, float]:
        return {
            "stride_length": float(self.stride_length),
            "step_height": float(self.step_height),
            "cycle_time": float(self.cycle_time),
            "lateral_offset": float(self.lateral_offset),
        }


class PiDogGaitEnv:
    """Gym-style environment for gait tuning.

    State design: concat of gait parameters + IMU readings to let the policy see
    both the control knobs and the stability response.
    """

    def __init__(
        self,
        action_scaling: ActionScaling,
        reward_weights: RewardWeights,
        imu_config: ImuConfig,
        hardware: HardwareConfig,
        safety: SafetyLimits,
        max_steps: int,
    ) -> None:
        self.action_scaling = action_scaling
        self.reward_weights = reward_weights
        self.imu_config = imu_config
        self.hardware_config = hardware
        self.safety = safety
        self.max_steps = max_steps
        self.hardware = PidogHardware(hardware) if hardware.use_hardware else None

        self.step_count = 0
        self.gait = GaitParameters(
            stride_length=0.08,
            step_height=0.03,
            cycle_time=0.28,
            lateral_offset=0.0,
        )
        self._imu_smoothed = np.zeros(3, dtype=np.float32)

    @property
    def state_dim(self) -> int:
        return 7  # 4 gait parameters + 3 IMU channels

    @property
    def action_dim(self) -> int:
        return 4  # deltas for each gait parameter

    def reset(self) -> np.ndarray:
        self.step_count = 0
        self.gait = GaitParameters(
            stride_length=0.08,
            step_height=0.03,
            cycle_time=0.28,
            lateral_offset=0.0,
        )
        self._imu_smoothed = np.zeros(3, dtype=np.float32)
        if self.hardware is not None:
            self.hardware.ensure_standing()
        return self._get_state()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        # Action space shaping: scale and clip deltas to keep exploration safe.
        self._apply_action(action)

        # Run the robot for a short duration and read sensors.
        distance, imu = self._run_robot()
        self._imu_smoothed = exp_smooth(self._imu_smoothed, imu, self.imu_config.smoothing)

        reward = self._compute_reward(distance, self._imu_smoothed)
        self.step_count += 1
        done = self.step_count >= self.max_steps

        info = {
            "distance": float(distance),
            "instability": float(self._instability(self._imu_smoothed)),
        }
        return self._get_state(), reward, done, info

    def _get_state(self) -> np.ndarray:
        imu = self._imu_smoothed.astype(np.float32)
        return np.concatenate([self.gait.as_array(), imu], dtype=np.float32)

    def _apply_action(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != self.action_dim:
            raise ValueError(f"Expected action dim {self.action_dim}, got {action.shape[0]}")

        deltas = np.array(
            [
                action[0] * self.action_scaling.stride_length,
                action[1] * self.action_scaling.step_height,
                action[2] * self.action_scaling.cycle_time,
                action[3] * self.action_scaling.lateral_offset,
            ],
            dtype=np.float32,
        )

        # Safety constraints: keep gait parameters inside safe bounds.
        self.gait.stride_length = float(
            np.clip(
                self.gait.stride_length + deltas[0],
                self.safety.stride_length_min,
                self.safety.stride_length_max,
            )
        )
        self.gait.step_height = float(
            np.clip(
                self.gait.step_height + deltas[1],
                self.safety.step_height_min,
                self.safety.step_height_max,
            )
        )
        self.gait.cycle_time = float(
            np.clip(
                self.gait.cycle_time + deltas[2],
                self.safety.cycle_time_min,
                self.safety.cycle_time_max,
            )
        )
        self.gait.lateral_offset = float(
            np.clip(
                self.gait.lateral_offset + deltas[3],
                self.safety.lateral_offset_min,
                self.safety.lateral_offset_max,
            )
        )

    def _run_robot(self) -> Tuple[float, np.ndarray]:
        if self.hardware is not None:
            return self.hardware.run_and_measure(self.gait)
        return self._simulate_robot_run()

    def _simulate_robot_run(self) -> Tuple[float, np.ndarray]:
        """Placeholder for real robot control and IMU readings.

        Exploration vs. stability: a faster stride can improve distance but
        tends to increase IMU noise.
        """
        base_speed = self.gait.stride_length / self.gait.cycle_time
        distance = base_speed * 0.5  # short duration rollout

        # IMU response: larger stride and shorter cycle time increase wobble.
        wobble = (self.gait.stride_length * 2.0) + (0.2 / self.gait.cycle_time)
        imu_noise = np.random.normal(0.0, 0.02, size=3).astype(np.float32)
        imu = np.array(
            [
                0.3 * wobble + imu_noise[0],  # roll
                0.25 * wobble + imu_noise[1],  # pitch
                0.2 * wobble + imu_noise[2],  # yaw_rate
            ],
            dtype=np.float32,
        )
        return float(distance), imu

    def _compute_reward(self, distance: float, imu: np.ndarray) -> float:
        instability = self._instability(imu)
        # Reward engineering: forward motion minus instability penalty.
        return (
            self.reward_weights.forward * distance
            - self.reward_weights.instability * instability
        )

    @staticmethod
    def _instability(imu: np.ndarray) -> float:
        return float(np.abs(imu).sum())
