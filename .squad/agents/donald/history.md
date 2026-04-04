# Project Context

- **Owner:** Dave Davis
- **Project:** Reinforcement-learning gait tuning for the SunFounder PiDog
- **Stack:** Python 3.10+, PyTorch, NumPy, Matplotlib
- **Created:** 2026-04-04

## Learnings

<!-- Append new learnings below. Each entry is something lasting about the project. -->

### 2026-04-04: Corrected PPO Implementation with Multi-Epoch Updates

**Context:** Previous PPO implementation was simplified to single update per episode, missing the core benefit of PPO (multi-epoch updates on same data). Fixed to implement proper PPO with state/action storage and recomputation of log probs.

**Action:** 
- Modified `run_episode()` to return `EpisodeData` containing states, actions, log_probs, rewards, and stats
- Added `PolicyNetwork.log_prob()` method to compute log probability for given state-action pairs
- Added `PPOConfig` dataclass with `clip_epsilon`, `num_epochs`, `normalize_advantages` hyperparameters
- Rewrote `PPOAlgorithm` to store episode data and perform multi-epoch updates by recomputing loss from current policy
- Default algorithm remains "reinforce" (not changed per user request)

**Architecture:**
- PPO now does true multi-epoch updates (default: 4 epochs per episode)
- Each epoch recomputes log_probs from current policy and recalculates clipped loss
- All stored tensors (states, actions, advantages, old_log_probs) are detached to avoid computational graph issues
- Training loop passes states and actions via kwargs to `compute_loss()`
- Gradient norms are averaged across epochs for logging

**Key Files:**
- `pidog_rl/train.py`: Changed `run_episode()` to return `EpisodeData`; passes states/actions to `compute_loss()`
- `pidog_rl/policy.py`: Added `log_prob()` method for recomputing action probabilities
- `pidog_rl/config.py`: Added `PPOConfig`; kept default algorithm as "reinforce"
- `pidog_rl/algorithms/ppo.py`: Complete rewrite with multi-epoch support
- `pidog_rl/algorithms/README.md`: Updated with PPO details and multi-epoch info
- `README.md`: Updated algorithm selection docs with PPOConfig explanation

**Training Results:**
- Successfully completed 2000 episodes with corrected PPO
- Outputs written to `output\26_04_04_9\`
- Final reward: ~-0.6 to -0.8 (vs ~-5 initially)
- Distance maxed at 16.265; instability ~48
- Checkpoints saved at 400, 800, 1200, 1600, 2000, final
- Phase 1 validation passed

**Stability Notes:**
- Initial multi-epoch attempt failed with "backward through graph twice" error
- Fixed by detaching all stored episode data (states, actions, advantages, old_log_probs)
- Each epoch after the first recomputes loss via `_compute_ppo_loss()` from fresh forward pass
- Gradient clipping works correctly across all epochs

### 2026-04-04: PPO Algorithm Implementation

**Context:** Implemented Proximal Policy Optimization (PPO) as an alternative to REINFORCE to improve training stability and sample efficiency.

**Action:** Created `pidog_rl/algorithms/ppo.py` following existing algorithm abstraction pattern. Registered in factory and updated config to use PPO as default.

**Architecture:**
- PPO uses clipped surrogate objective to prevent destructive policy updates
- Importance sampling ratio with clipping bounds (epsilon=0.2)
- Advantage normalization for stability
- Single update per episode (simplified from multi-epoch for compatibility with existing training loop)
- Compatible with existing checkpoint format and training infrastructure

**Key Files:**
- `pidog_rl/algorithms/ppo.py`: PPO implementation with clipped objective
- `pidog_rl/algorithms/__init__.py`: Registered PPOAlgorithm
- `pidog_rl/train.py`: Added PPO to algorithm factory
- `pidog_rl/config.py`: Changed default algorithm to "ppo"
- `pidog_rl/algorithms/README.md`: Updated available algorithms list
- `README.md`: Updated docs to reflect PPO as default

**Training Results:**
- Successfully completed 2000 episodes with PPO
- Outputs written to `output\26_04_04_5\`
- Checkpoints saved at regular intervals (400, 800, 1200, 1600, 2000, final)
- Training plot generated showing reward/distance/instability curves

**Design Decision:**
- Simplified PPO to single update per episode to work with existing training loop structure
- Future enhancement: Store states/actions to enable true multi-epoch PPO updates
- Maintains compatibility with existing infrastructure (no breaking changes)

**Stability Notes:**
- Initial implementation had backward() issues with multiple epochs on same graph
- Resolved by detaching advantages and doing single update per episode
- Gradient clipping remains available via EpisodeConfig.grad_clip_norm

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

### 2026-04-04: Phase 2 Complete — Team Orchestration

**Context:** Phase 2 reward shaping foundation finalized across team.

**Cross-Agent Coordination:**
- **Mickey:** Scoped Phase 2; established decision gates
- **Minnie:** Defined metrics framework (correlation, clipping margin, convergence, determinism)
- **Goofy:** Created phase2_validation.py aligned with implementation
- **Scribe:** Merged decisions and recorded orchestration logs

**Status:** 
- Reward shaping config + pipeline wired through training/inference
- Phase 2 validation infrastructure complete
- All decisions archived and cross-linked
- Ready for WI-1 (Reward Normalization) + WI-4 (Convergence Benchmark) execution

**Next:** Execute WI-1 + WI-4 in parallel; validate against Phase 1 baseline; gate WI-2 + WI-3 based on results.

### 2026-04-04: Phase 2 Analysis Results — Config Verification Needed

**Context:** Minnie completed Phase 2 analysis. Results show reward shaping implementation working as designed, but reward formulation insufficient to improve convergence.

**Findings:**
- Phase 2 runs -4.83 mean reward vs ORI -4.78 (−1% regression)
- Instability unchanged: 31.10 vs 31.18 (−0.2%)
- Run 3 shows signal degradation: correlation -0.262 vs R1-2 -0.337 (unusual)
- Reproducibility excellent; distance metric held

**Action Items for Donald:**
1. Verify Run 3 config matches Runs 1-2 (reason: correlation drop suggests hyperparameter sensitivity)
2. Review RewardShapingConfig penalty weights; may need 2-3x increase for next iteration
3. Consider parallel tuning experiment: stronger penalty weights vs. baseline enhancement (WI-2)

**Team Decision:** D5 merged to decisions.md; recommend proceeding to WI-2 regardless, investigate penalty weight tuning separately.
