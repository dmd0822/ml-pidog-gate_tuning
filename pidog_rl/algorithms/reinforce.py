from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch import nn

from .base import Algorithm
from ..config import EpisodeConfig
from ..utils import compute_returns


class ReinforceAlgorithm(Algorithm):
    """REINFORCE (Monte Carlo Policy Gradient) algorithm.

    Uses full episode returns with a baseline to reduce variance while keeping
    the gradient unbiased.
    """

    def __init__(
        self, policy: nn.Module, learning_rate: float, config: EpisodeConfig
    ) -> None:
        self.policy = policy
        self.config = config
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    def compute_loss(
        self,
        log_probs: List[torch.Tensor],
        rewards: List[float],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute REINFORCE loss with baseline.

        The baseline (mean return) reduces variance while keeping the gradient
        unbiased, helping the policy distinguish better-than-average actions.
        """
        device = log_probs[0].device
        returns = compute_returns(rewards, self.config.gamma)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)

        baseline = returns_tensor.mean()
        advantages = returns_tensor - baseline

        loss = -torch.stack(log_probs).mul(advantages).sum()
        return loss

    def update(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def state_dict(self) -> Dict[str, Any]:
        return {"optimizer_state_dict": self.optimizer.state_dict()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
