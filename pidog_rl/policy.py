from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torch.distributions import Normal


@dataclass(frozen=True)
class PolicyOutput:
    action: torch.Tensor
    log_prob: torch.Tensor


class PolicyNetwork(nn.Module):
    """Simple policy network producing continuous gait adjustments."""

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(64, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.net(state)
        mean = self.mean_head(features)
        log_std = self.log_std.expand_as(mean)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> PolicyOutput:
        mean, log_std = self(state)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return PolicyOutput(action=action, log_prob=log_prob)

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute log probability of given action under current policy."""
        mean, log_std = self(state)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        return dist.log_prob(action).sum(dim=-1)
