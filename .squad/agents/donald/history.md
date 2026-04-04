# Project Context

- **Owner:** Dave Davis
- **Project:** Reinforcement-learning gait tuning for the SunFounder PiDog
- **Stack:** Python 3.10+, PyTorch, NumPy, Matplotlib
- **Created:** 2026-04-04

## Learnings

<!-- Append new learnings below. Each entry is something lasting about the project. -->

### 2026-04-04: Housekeeping

**Context:** Removed stale output folder from training run.

**Action:** Deleted `output/26_04_04_3/` directory and all contents to clean up intermediate artifacts.

**Key Files:**
- N/A (cleanup only)

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

### 2026-04-04: Phase 1 Stability Adjustments

**Context:** Approved Phase 1 stability improvements for REINFORCE training.

**Action:** Added EMA return baseline and gradient clipping support in `ReinforceAlgorithm`, clamped raw actions in `PiDogGaitEnv`, and expanded training logs with loss/baseline/grad norm.

**Key Files:**
- `pidog_rl/algorithms/reinforce.py`: EMA baseline + grad clip tracking
- `pidog_rl/config.py`: Episode config fields for baseline EMA + grad clip norm
- `pidog_rl/env.py`: Action clamp before scaling
- `pidog_rl/train.py`: Logged stability metrics

### 2026-04-04: Phase 1 Complete — Team Orchestration

**Context:** Phase 1 REINFORCE variance reduction work finalized across team.

**Cross-Agent Coordination:**
- **Goofy:** Validation script aligned with Donald's final implementation
- **Mickey:** Architecture docs finalized; Phase 2-4 roadmap created
- **Scribe:** Decisions merged; orchestration logs recorded

**Status:** Phase 1 foundation complete. Ready for convergence testing and Phase 2 reward tuning.
