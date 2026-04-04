from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import torch
from torch import nn


class Algorithm(ABC):
    """Base class for RL algorithms.

    Defines the interface that all algorithms must implement to be compatible
    with the training loop.
    """

    @abstractmethod
    def __init__(self, policy: nn.Module, learning_rate: float, config: Any) -> None:
        """Initialize the algorithm with a policy network and hyperparameters.

        Args:
            policy: The policy network to train
            learning_rate: Learning rate for the optimizer
            config: Algorithm-specific configuration (e.g., EpisodeConfig)
        """
        pass

    @abstractmethod
    def compute_loss(
        self,
        log_probs: List[torch.Tensor],
        rewards: List[float],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute the policy loss from episode data.

        Args:
            log_probs: Log probabilities of actions taken during the episode
            rewards: Rewards received at each step
            **kwargs: Additional algorithm-specific data (states, values, etc.)

        Returns:
            Loss tensor to backpropagate
        """
        pass

    @abstractmethod
    def update(self, loss: torch.Tensor) -> None:
        """Perform a parameter update using the computed loss.

        Args:
            loss: The loss tensor from compute_loss
        """
        pass

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """Return the algorithm state for checkpointing.

        Returns:
            Dictionary containing optimizer state and any other algorithm state
        """
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load algorithm state from a checkpoint.

        Args:
            state_dict: Dictionary containing algorithm state
        """
        pass
