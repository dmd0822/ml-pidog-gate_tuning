from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch import nn

from .base import Algorithm
from ..config import EpisodeConfig, PPOConfig
from ..utils import compute_returns


class PPOAlgorithm(Algorithm):
    """Proximal Policy Optimization with clipped surrogate objective.

    PPO improves upon REINFORCE by using multiple optimization epochs on each
    batch of data, while constraining policy updates to prevent destructive
    changes via a clipped objective function.
    """

    def __init__(
        self, policy: nn.Module, learning_rate: float, episode_config: EpisodeConfig, ppo_config: PPOConfig
    ) -> None:
        self.policy = policy
        self.episode_config = episode_config
        self.ppo_config = ppo_config
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

        # Tracking for logging
        self.last_baseline: float | None = None
        self.last_grad_norm: float | None = None

        # Store episode data for multi-epoch updates
        self._states: List[torch.Tensor] = []
        self._actions: List[torch.Tensor] = []
        self._old_log_probs: List[torch.Tensor] = []
        self._advantages: torch.Tensor | None = None

    def compute_loss(
        self,
        log_probs: List[torch.Tensor],
        rewards: List[float],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute PPO clipped surrogate loss.

        PPO constrains policy updates using a clipped importance ratio to prevent
        large destructive updates. The policy is updated multiple times on the
        same batch to improve sample efficiency.
        """
        # Extract states and actions from kwargs
        states = kwargs.get("states", [])
        actions = kwargs.get("actions", [])
        
        if not states or not actions:
            raise ValueError("PPO requires 'states' and 'actions' in kwargs")

        device = log_probs[0].device

        # Compute returns and advantages
        returns = compute_returns(rewards, self.episode_config.gamma)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)

        # Use returns mean as baseline
        baseline = returns_tensor.mean()
        self.last_baseline = float(baseline.item())
        advantages = returns_tensor - baseline

        # Normalize advantages (common PPO practice)
        if self.ppo_config.normalize_advantages and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Store episode data for multi-epoch updates (detach everything to avoid graph issues)
        self._states = [s.detach() for s in states]
        self._actions = [a.detach() for a in actions]
        self._old_log_probs = [lp.detach() for lp in log_probs]
        self._advantages = advantages.detach()

        # Compute initial loss (will be recomputed in each epoch during update)
        loss = self._compute_ppo_loss()
        return loss

    def _compute_ppo_loss(self) -> torch.Tensor:
        """Compute PPO loss by recomputing log probs from current policy."""
        if self._advantages is None or not self._states or not self._actions:
            raise RuntimeError("Episode data not set; call compute_loss first")

        # Recompute log probs under current policy
        current_log_probs = []
        for state, action in zip(self._states, self._actions):
            log_prob = self.policy.log_prob(state, action)
            current_log_probs.append(log_prob)

        current_log_probs_tensor = torch.stack(current_log_probs)
        old_log_probs_tensor = torch.stack(self._old_log_probs)

        # Compute importance sampling ratio
        ratio = torch.exp(current_log_probs_tensor - old_log_probs_tensor)

        # Clipped surrogate objective
        surr1 = ratio * self._advantages
        surr2 = torch.clamp(
            ratio, 1.0 - self.ppo_config.clip_epsilon, 1.0 + self.ppo_config.clip_epsilon
        ) * self._advantages
        loss = -torch.min(surr1, surr2).sum()

        return loss

    def update(self, loss: torch.Tensor) -> None:
        """Perform multiple optimization epochs on the episode data.
        
        PPO updates the policy multiple times on the same batch of data,
        recomputing the loss each time to reflect the updated policy.
        """
        total_grad_norm = 0.0
        
        # First epoch: use the loss passed in
        self.optimizer.zero_grad()
        loss.backward()
        
        if self.episode_config.grad_clip_norm and self.episode_config.grad_clip_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.episode_config.grad_clip_norm
            )
            total_grad_norm += float(grad_norm)
        
        self.optimizer.step()
        
        # Subsequent epochs: recompute loss from scratch
        for epoch in range(1, self.ppo_config.num_epochs):
            loss = self._compute_ppo_loss()
            
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            if self.episode_config.grad_clip_norm and self.episode_config.grad_clip_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.episode_config.grad_clip_norm
                )
                total_grad_norm += float(grad_norm)

            self.optimizer.step()

        # Average grad norm across epochs for logging
        self.last_grad_norm = total_grad_norm / self.ppo_config.num_epochs if self.ppo_config.num_epochs > 0 else None

    def state_dict(self) -> Dict[str, Any]:
        """Return state for checkpointing."""
        return {
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from checkpoint."""
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
