---
skill: RL Algorithm Extensibility
domain: Machine Learning
applies_to: Reinforcement Learning training loops
created: 2026-04-04
author: Donald
---

# Skill: Making RL Training Loops Algorithm-Agnostic

## Problem

RL projects often start with one algorithm (e.g., REINFORCE) hardcoded into the training loop, making it difficult to experiment with other algorithms without major refactoring.

## Solution Pattern

Use an abstract base class to define the algorithm interface, separating algorithm-specific logic from training infrastructure.

### Key Components

1. **Abstract Base Class** (`Algorithm`):
   ```python
   class Algorithm(ABC):
       @abstractmethod
       def compute_loss(self, log_probs, rewards, **kwargs) -> Tensor: ...
       @abstractmethod
       def update(self, loss: Tensor) -> None: ...
       @abstractmethod
       def state_dict(self) -> Dict[str, Any]: ...
       @abstractmethod
       def load_state_dict(self, state_dict: Dict[str, Any]) -> None: ...
   ```

2. **Concrete Implementations**: Each algorithm (REINFORCE, PPO, A2C, etc.) implements the interface

3. **Factory Function**: Creates algorithm instances based on config:
   ```python
   def _create_algorithm(name: str, policy: nn.Module, lr: float, config: Any) -> Algorithm:
       if name == "reinforce":
           return ReinforceAlgorithm(policy, lr, config)
       elif name == "ppo":
           return PPOAlgorithm(policy, lr, config)
       # ...
   ```

4. **Training Loop**: Works with any algorithm:
   ```python
   algorithm = _create_algorithm(config.algorithm, policy, config.lr, config)
   for episode in range(episodes):
       log_probs, rewards, stats = run_episode(env, policy, device)
       loss = algorithm.compute_loss(log_probs, rewards)
       algorithm.update(loss)
   ```

### Interface Design Considerations

- **`compute_loss`**: Takes episode data and returns loss tensor. Use `**kwargs` for algorithm-specific data (values, next states, etc.)
- **`update`**: Handles optimization. Can be called multiple times per episode (e.g., PPO mini-batches)
- **`state_dict` / `load_state_dict`**: Must save ALL training state (optimizers, baselines, normalizers) for reproducible checkpointing

### Extending for Actor-Critic

For algorithms needing value estimates:
1. Add value network to algorithm class
2. Modify `run_episode()` to collect value predictions
3. Pass values via `**kwargs` to `compute_loss()`

## Benefits

- Easy to experiment with different algorithms
- Clear separation of concerns
- Testable algorithm implementations
- Consistent checkpoint management

## Trade-offs

- Adds abstraction overhead
- Breaks backward compatibility with old checkpoints
- May need to extend interface for complex algorithms (e.g., off-policy with replay buffers)

## When to Apply

- When starting a new RL project (design for extensibility from the start)
- When refactoring a single-algorithm implementation
- When planning to compare multiple algorithms on the same task

## When NOT to Apply

- One-off experiments where you'll never try another algorithm
- Production systems with a proven, stable algorithm
- Very custom algorithms that don't fit a common interface
