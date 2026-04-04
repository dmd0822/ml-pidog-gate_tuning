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
        self._run = self._get_method(config.run_method)
        self._read_imu = self._get_method(config.imu_method)
        self._read_distance = self._get_method(config.distance_method)

    def ensure_standing(self) -> None:
        """Ensure the robot is in a standing pose before control starts."""
        if self.config.stand_action is None:
            self._stand()
            return
        self._stand(self.config.stand_action, speed=self.config.stand_speed)

    def run_and_measure(self, gait: GaitLike) -> tuple[float, np.ndarray]:
        # Action application: forward gait parameters to the PiDog API.
        try:
            self._apply_gait(**gait.as_dict())
        except TypeError as exc:  # method signature mismatch
            raise TypeError(
                "PiDog gait method signature mismatch. "
                "Update HardwareConfig.apply_gait_method or adjust argument mapping."
            ) from exc

        # Run for a short duration to collect a transition.
        self._run(self.config.run_duration_sec)

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
        return getattr(self.robot, name)
