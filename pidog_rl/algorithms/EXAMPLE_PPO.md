# Example: Adding PPO Algorithm

This is a template/example for adding Proximal Policy Optimization (PPO) to the project.

## 1. Create `pidog_rl/algorithms/ppo.py`

```python
from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch import nn

from .base import Algorithm
from ..config import EpisodeConfig


class PPOAlgorithm(Algorithm):
    """Proximal Policy Optimization with clipped objective."""

    def __init__(
        self, policy: nn.Module, learning_rate: float, config: EpisodeConfig
    ) -> None:
        self.policy = policy
        self.config = config
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
        
        # PPO-specific hyperparameters (could be in config)
        self.clip_epsilon = 0.2
        self.num_epochs = 4
        self.minibatch_size = 64

    def compute_loss(
        self,
        log_probs: List[torch.Tensor],
        rewards: List[float],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute PPO clipped surrogate loss.
        
        For PPO, you'd typically:
        1. Compute advantages (e.g., GAE)
        2. Calculate ratio = π_new / π_old
        3. Apply clipping: min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
        """
        # This is a simplified example - real PPO needs old_log_probs
        # and would iterate over minibatches for num_epochs
        
        device = log_probs[0].device
        returns = self._compute_returns(rewards)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
        
        # Normalize advantages
        advantages = returns_tensor - returns_tensor.mean()
        advantages = advantages / (advantages.std() + 1e-8)
        
        # For full PPO, you'd need to:
        # - Store old log probs
        # - Compute importance ratio
        # - Apply clipping
        # - Add value loss and entropy bonus
        
        # Simplified version (closer to REINFORCE):
        loss = -torch.stack(log_probs).mul(advantages).mean()
        return loss

    def update(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        # PPO often uses gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.optimizer.step()

    def state_dict(self) -> Dict[str, Any]:
        return {"optimizer_state_dict": self.optimizer.state_dict()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
    
    def _compute_returns(self, rewards: List[float]) -> List[float]:
        """Compute discounted returns."""
        returns: List[float] = []
        running = 0.0
        for reward in reversed(rewards):
            running = reward + self.config.gamma * running
            returns.append(running)
        returns.reverse()
        return returns
```

## 2. Register in `pidog_rl/algorithms/__init__.py`

```python
from __future__ import annotations

from .base import Algorithm
from .reinforce import ReinforceAlgorithm
from .ppo import PPOAlgorithm  # Add this

__all__ = ["Algorithm", "ReinforceAlgorithm", "PPOAlgorithm"]  # Add to exports
```

## 3. Update factory in `pidog_rl/train.py`

```python
def _create_algorithm(
    algorithm_name: str,
    policy: PolicyNetwork,
    learning_rate: float,
    config: TrainingConfig,
) -> Algorithm:
    algorithm_name = algorithm_name.lower()
    if algorithm_name == "reinforce":
        return ReinforceAlgorithm(policy, learning_rate, config.episode)
    elif algorithm_name == "ppo":  # Add this
        return PPOAlgorithm(policy, learning_rate, config.episode)
    else:
        raise ValueError(
            f"Unknown algorithm: {algorithm_name}. Supported: 'reinforce', 'ppo'"
        )
```

## 4. Use it

In `pidog_rl/config.py`:

```python
@dataclass(frozen=True)
class TrainingConfig:
    algorithm: str = "ppo"  # Changed from "reinforce"
    # ... rest of config
```

Or at runtime:

```python
from dataclasses import replace
config = replace(TrainingConfig(), algorithm="ppo")
train(config)
```

## Notes for Full PPO Implementation

A production-quality PPO would need:

1. **Value Network**: Separate critic network for advantage estimation
2. **GAE**: Generalized Advantage Estimation instead of simple returns
3. **Minibatch Updates**: Iterate over shuffled minibatches for K epochs
4. **Old Log Probs**: Store log probs from the behavior policy for importance ratio
5. **Clipping**: Both policy ratio and value function
6. **Entropy Bonus**: Encourage exploration
7. **Modified `run_episode()`**: Collect states, values, and old log probs

Example modifications to `run_episode()`:

```python
# In train.py
def run_episode_with_values(env, policy, value_net, device):
    # ... existing code ...
    values = []
    while not done:
        state_tensor = torch.from_numpy(state).float().to(device)
        output = policy.sample(state_tensor)
        value = value_net(state_tensor)  # NEW: value prediction
        
        action = output.action.detach().cpu().numpy()
        next_state, reward, done, info = env.step(action)
        
        log_probs.append(output.log_prob)
        values.append(value)  # NEW
        rewards.append(reward)
        # ... rest ...
    
    return log_probs, values, rewards, stats  # NEW: return values
```

Then pass to algorithm:

```python
log_probs, values, rewards, stats = run_episode_with_values(env, policy, value_net, device)
loss = algorithm.compute_loss(log_probs, rewards, values=values)  # Pass via kwargs
```
