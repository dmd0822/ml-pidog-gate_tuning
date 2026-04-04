from __future__ import annotations

from typing import Any, Callable, Protocol

import numpy as np

from .config import HardwareConfig


class GaitLike(Protocol):
    def as_dict(self) -> dict[str, float]:
        ...


class PidogHardware:
    """Adapter around the official SunFounder PiDog library."""

    def __init__(self, config: HardwareConfig) -> None:
        self.config = config
        self.robot = self._create_robot()
        self._apply_gait = self._get_method(config.apply_gait_method)
        self._stand = self._get_method(config.stand_method)
        self._lie_down = self._get_method(config.lie_method)
        self._run = self._get_method(config.run_method)
        self._read_imu = self._get_method(config.imu_method)
        self._read_distance = self._get_method(config.distance_method)

    def ensure_standing(self) -> None:
        """Ensure the robot is in a standing pose before control starts."""
        if self.config.stand_action is None:
            self._stand()
            return
        self._stand(self.config.stand_action, speed=self.config.stand_speed)

    def ensure_lie_down(self) -> None:
        """Move the robot into a resting/lying pose after control ends."""
        if self.config.lie_action is None:
            self._lie_down()
            return
        self._lie_down(self.config.lie_action, speed=self.config.lie_speed)

    def run_and_measure(self, gait: GaitLike) -> tuple[float, np.ndarray]:
        # Action application: forward gait parameters to the PiDog API.
        try:
            gait_values = gait.as_dict()
            if self.config.apply_gait_action is None:
                self._apply_gait(**gait_values)
                run_duration = self.config.run_duration_sec
            else:
                speed = int(
                    round(
                        self._map_range(
                            gait_values["stride_length"],
                            self.config.stride_length_min,
                            self.config.stride_length_max,
                            self.config.speed_min,
                            self.config.speed_max,
                        )
                    )
                )
                run_duration = self._map_range(
                    gait_values["cycle_time"],
                    self.config.cycle_time_min,
                    self.config.cycle_time_max,
                    self.config.run_duration_min,
                    self.config.run_duration_max,
                )
                body_height = self._map_range(
                    gait_values["step_height"],
                    self.config.step_height_min,
                    self.config.step_height_max,
                    self.config.body_height_min,
                    self.config.body_height_max,
                )
                lateral_y = self._map_range(
                    gait_values["lateral_offset"],
                    self.config.lateral_offset_min,
                    self.config.lateral_offset_max,
                    -self.config.lateral_offset_max_mm,
                    self.config.lateral_offset_max_mm,
                )
                self.robot.set_pose(y=lateral_y, z=body_height)
                self._apply_gait(
                    self.config.apply_gait_action,
                    speed=speed,
                )
        except TypeError as exc:  # method signature mismatch
            raise TypeError(
                "PiDog gait method signature mismatch. "
                "Update HardwareConfig.apply_gait_method or adjust argument mapping."
            ) from exc

        # Run for a short duration to collect a transition.
        try:
            self._run(run_duration)
        except TypeError:
            self._run()

        distance = float(self._read_distance())
        imu_raw = self._read_imu()
        if isinstance(imu_raw, dict):
            imu = np.asarray(
                [imu_raw[key] for key in self.config.imu_keys],
                dtype=np.float32,
            )
        else:
            imu = np.asarray(list(imu_raw), dtype=np.float32)
        if imu.shape[0] < 3:
            raise ValueError("IMU reading must provide at least 3 channels.")
        return distance, imu[:3]

    @staticmethod
    def _map_range(
        value: float, in_min: float, in_max: float, out_min: float, out_max: float
    ) -> float:
        if in_max == in_min:
            return out_min
        clamped = min(max(value, in_min), in_max)
        scale = (clamped - in_min) / (in_max - in_min)
        return out_min + scale * (out_max - out_min)

    @staticmethod
    def _create_robot() -> Any:
        try:
            from pidog import Pidog
        except ImportError:
            from pidog.pidog import Pidog
        return Pidog()

    def _get_method(self, name: str) -> Callable[..., Any]:
        if not hasattr(self.robot, name):
            available = ", ".join(sorted(m for m in dir(self.robot) if not m.startswith("_")))
            raise ValueError(
                f"PiDog method '{name}' not found. Available methods: {available}"
            )
        attr = getattr(self.robot, name)
        if callable(attr):
            return attr
        return lambda: getattr(self.robot, name)
