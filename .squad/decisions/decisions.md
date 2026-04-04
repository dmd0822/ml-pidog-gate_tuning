# Team Decisions

<!-- Merged from decisions/inbox/ on 2026-04-04T17:46:06Z -->

## Active Decisions

### 2026-04-04T16:39:49Z: User directive
**By:** Dave Davis (via Copilot)  
**What:** Use the main Disney characters for the squad.  
**Why:** User request — captured for team memory

### 2026-04-04T16:48:47Z: User directive
**By:** Dave Davis (via Copilot)  
**What:** When making model changes, document the changes being made and the reason for the changes.  
**Why:** User request — captured for team memory

### 2026-04-04: Refactored RL Algorithm Architecture for Extensibility
**By:** Donald  
**Status:** Implemented  
**Decision:** Restructured project to support multiple RL algorithms through abstract base class, moving REINFORCE-specific logic out of training loop.  
**Context:** REINFORCE was hardcoded into `train.py`, making experimentation with other algorithms difficult.  
**Implementation:**
- Created `pidog_rl/algorithms/` package with abstract `Algorithm` class, `ReinforceAlgorithm` implementation, and extension guide
- Updated `train.py` with `_create_algorithm()` factory function
- Added `algorithm: str = "reinforce"` to `TrainingConfig`
- Changed checkpoints to save `algorithm_state_dict` instead of `optimizer_state_dict`

**Benefits:** Extensibility for new algorithms (PPO, A2C, SAC), separation of concerns, testability, consistent checkpoint management.  
**Trade-offs:** Adds abstraction layer; backward-incompatible checkpoint format (acceptable for early-stage project).

### 2026-04-04: Algorithm Extensibility in Architecture Docs
**By:** Mickey  
**Status:** Completed  
**Decision:** Updated ARCHITECTURE.md to document pluggable Algorithm interface and factory pattern.  
**Changes:**
- Added sections 8-9 covering `algorithms/base.py` and `algorithms/reinforce.py`
- Updated `train.py` documentation with factory function details
- New section: "Extending with New Algorithms" with step-by-step guide for A2C, PPO, SAC
- Updated file structure documentation

**Rationale:** Algorithm interface is key architectural abstraction; needs to be documented in main architecture file alongside implementation guides.

### 2026-04-04: Architecture Documentation Complete
**By:** Mickey  
**Status:** Documented  
**Decision:** Created comprehensive ARCHITECTURE.md documenting ml-pidog-gate_tuning system.  
**Rationale:** Codebase implements REINFORCE-based gait-tuning RL system with clean sim↔hardware abstraction. Documentation clarifies state/action/reward design, module responsibilities, safety constraints, configuration philosophy, and extension points.  
**Key Insights:**
- Single `PiDogGaitEnv` with pluggable hardware backend enables safe sim-to-real workflow
- All gait parameters clipped to `SafetyLimits` before application; sensor reads sanitized
- Frozen dataclasses in `config.py` serve as single source of truth
- REINFORCE + baseline suitable for short episodes; 2-layer 64-neuron policy sufficient for 4D action space

**Artifact:** ARCHITECTURE.md (~550 lines, covers all modules, design decisions, data flows, safety, extension points).

### 2026-04-04: Gait Stability & Convergence Improvement Plan
**By:** Mickey  
**Status:** Active  
**Decision:** Multi-phase plan to improve gait stability and convergence.

**Problem:**
- Slow convergence (2000 episodes with limited uptrend)
- High instability variance
- No adaptive parameter tuning

**Success Criteria:**
- 50% fewer episodes to stable policy (≤1000)
- 30% reduction in instability volatility
- Clear upward reward trend in final 500 episodes

**Phase 1 (Variance Reduction — Low Risk):**
- Gradient clipping (max_norm=1.0)
- Enhanced baseline (EMA instead of scalar mean)
- Action space clipping margin adjustment

**Phase 2 (Reward Signal Tuning — Medium Risk):**
- Adaptive instability penalty (episode-aware)
- IMU normalization via running statistics
- Distance reward shaping

**Phase 3 (Training Dynamics — Low Risk):**
- Learning rate schedule (LR *= 0.95 every 200 episodes)
- Episode length curriculum (30 → 50 → 70 steps)

**Phase 4 (Diagnostics — No Direct Impact):**
- Gradient statistics logging
- Advantage distribution logging

**Implementation Order:** Phase 1 (Week 1) → Phase 2 (Week 2) → Phase 3 (Week 3) → Phase 4 (Week 4).

**Decision Gates:** Post-Phase 1, Phase 2, Phase 3 measurements to validate approach or pivot.

## Archive

See decisions-archive.md for older decisions.
