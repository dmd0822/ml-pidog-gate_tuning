# Algorithms

This directory contains RL algorithm implementations for the PiDog gait tuning project.

## Available Algorithms

- **REINFORCE** (`reinforce.py`): Monte Carlo policy gradient with EMA baseline
- **PPO** (`ppo.py`): Proximal Policy Optimization with clipped surrogate objective and multi-epoch updates

## Algorithm Details

### REINFORCE
- Simple Monte Carlo policy gradient
- Uses EMA baseline for variance reduction
- Single update per episode
- Good baseline for comparison

### PPO
- Clipped surrogate objective to prevent destructive policy updates
- Multi-epoch updates on each episode (default: 4 epochs)
- Advantage normalization for stability
- Importance sampling with policy ratio clipping (default epsilon: 0.2)
- Better sample efficiency than REINFORCE

## Adding a New Algorithm

To add a new RL algorithm (e.g., A2C, SAC):

1. **Create a new file** `pidog_rl/algorithms/your_algorithm.py`

2. **Implement the `Algorithm` interface** from `base.py`:

```python
from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch import nn

from .base import Algorithm
from ..config import EpisodeConfig


class YourAlgorithm(Algorithm):
    """Your algorithm description."""

    def __init__(
        self, policy: nn.Module, learning_rate: float, config: EpisodeConfig
    ) -> None:
        self.policy = policy
        self.config = config
        # Initialize optimizer(s) and any additional networks (critic, etc.)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    def compute_loss(
        self,
        log_probs: List[torch.Tensor],
        rewards: List[float],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute the loss for your algorithm."""
        # Implement your loss computation here
        pass

    def update(self, loss: torch.Tensor) -> None:
        """Perform parameter update."""
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def state_dict(self) -> Dict[str, Any]:
        """Return state for checkpointing."""
        return {
            "optimizer_state_dict": self.optimizer.state_dict(),
            # Add any other state (e.g., critic optimizer, running stats)
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from checkpoint."""
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
```

3. **Register in `__init__.py`**:

```python
from .your_algorithm import YourAlgorithm

__all__ = ["Algorithm", "ReinforceAlgorithm", "PPOAlgorithm", "YourAlgorithm"]
```

4. **Add to the factory** in `train.py`:

```python
def _create_algorithm(...):
    algorithm_name = algorithm_name.lower()
    if algorithm_name == "reinforce":
        return ReinforceAlgorithm(policy, learning_rate, config.episode)
    elif algorithm_name == "ppo":
        return PPOAlgorithm(policy, learning_rate, config.episode, config.ppo)
    elif algorithm_name == "your_algorithm":
        return YourAlgorithm(policy, learning_rate, config.episode)
    else:
        raise ValueError(...)
```

5. **Use it** by setting `algorithm="your_algorithm"` in `TrainingConfig`

## Notes on Algorithm Interface

- **`compute_loss`**: Takes log probs and rewards from episode rollout. Use `**kwargs` for algorithm-specific data (e.g., states, actions, value estimates).
- **`update`**: Handles the optimization step. Can be called multiple times per episode or perform multiple internal epochs.
- **`state_dict` / `load_state_dict`**: Must save/load all training state (optimizers, running averages, etc.) for reproducible checkpointing.

## Actor-Critic Algorithms

For actor-critic methods, you'll need to:
1. Add a value network to your algorithm class
2. Modify `run_episode` in `train.py` to collect value predictions if needed
3. Pass values via `**kwargs` to `compute_loss`

See `base.py` for the complete interface contract.
