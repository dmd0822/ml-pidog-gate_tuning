# Project Context

- **Owner:** Dave Davis
- **Project:** Reinforcement-learning gait tuning for the SunFounder PiDog
- **Stack:** Python 3.10+, PyTorch, NumPy, Matplotlib
- **Created:** 2026-04-04

## Learnings

<!-- Append new learnings below. Each entry is something lasting about the project. -->

### 2026-04-04: Refactored for Algorithm Extensibility

**Context:** REINFORCE was tightly coupled into train.py. Restructured to support multiple RL algorithms.

**Architecture:**
- Created `pidog_rl/algorithms/` package with abstract `Algorithm` base class
- Moved REINFORCE logic into `ReinforceAlgorithm` implementation
- Added algorithm factory function `_create_algorithm()` in train.py
- Updated `TrainingConfig` to include `algorithm: str = "reinforce"` field
- Changed checkpoint format: replaced `optimizer_state_dict` with `algorithm_state_dict`

**Key Files:**
- `pidog_rl/algorithms/base.py`: Abstract base class defining algorithm interface
- `pidog_rl/algorithms/reinforce.py`: REINFORCE implementation
- `pidog_rl/algorithms/README.md`: Guide for adding new algorithms
- `pidog_rl/train.py`: Refactored to use algorithm abstraction
- `pidog_rl/config.py`: Added `algorithm` field to `TrainingConfig`

**Interface Contract:**
- `compute_loss(log_probs, rewards, **kwargs)`: Compute loss from episode data
- `update(loss)`: Perform optimization step
- `state_dict()` / `load_state_dict()`: Checkpoint support

**Future Algorithms:** PPO, A2C, SAC can be added by implementing `Algorithm` and registering in the factory.

**Backward Compatibility:** Old checkpoints with `optimizer_state_dict` won't load with new code; this is acceptable as the repo is early-stage.
