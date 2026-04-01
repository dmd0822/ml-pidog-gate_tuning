from __future__ import annotations

from typing import Iterable, List

import numpy as np
import torch


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_returns(rewards: Iterable[float], gamma: float) -> List[float]:
    returns: List[float] = []
    running = 0.0
    for reward in reversed(list(rewards)):
        running = reward + gamma * running
        returns.append(running)
    returns.reverse()
    return returns


def exp_smooth(previous: np.ndarray, current: np.ndarray, smoothing: float) -> np.ndarray:
    """Exponential smoothing to damp real-world IMU noise."""
    return smoothing * previous + (1.0 - smoothing) * current
