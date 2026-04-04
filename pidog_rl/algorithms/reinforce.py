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
        self._baseline_ema: torch.Tensor | None = None
        self.last_baseline: float | None = None
        self.last_grad_norm: float | None = None

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

        returns_mean = returns_tensor.mean()
        if self._baseline_ema is None:
            self._baseline_ema = returns_mean.detach()
        else:
            alpha = self.config.baseline_ema_alpha
            self._baseline_ema = (1.0 - alpha) * self._baseline_ema + alpha * returns_mean.detach()

        baseline = self._baseline_ema
        advantages = returns_tensor - baseline
        self.last_baseline = float(baseline.item())

        loss = -torch.stack(log_probs).mul(advantages).sum()
        return loss

    def update(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        self.last_grad_norm = None
        if self.config.grad_clip_norm and self.config.grad_clip_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.config.grad_clip_norm
            )
            self.last_grad_norm = float(grad_norm)
        self.optimizer.step()

    def state_dict(self) -> Dict[str, Any]:
        baseline_ema = None if self._baseline_ema is None else float(self._baseline_ema.item())
        return {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "baseline_ema": baseline_ema,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        baseline_ema = state_dict.get("baseline_ema")
        if baseline_ema is None:
            self._baseline_ema = None
        else:
            device = next(self.policy.parameters()).device
            self._baseline_ema = torch.tensor(baseline_ema, device=device)
