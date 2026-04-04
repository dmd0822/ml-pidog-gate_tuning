# Project Context

- **Owner:** Dave Davis
- **Project:** Reinforcement-learning gait tuning for the SunFounder PiDog
- **Stack:** Python 3.10+, PyTorch, NumPy, Matplotlib
- **Created:** 2026-04-04

## Learnings

<!-- Append new learnings below. Each entry is something lasting about the project. -->
- Added scripts\phase1_validation.py to validate action scaling and provide optional EMA baseline/grad clipping checks. README now documents Phase 1 validation steps.
- Phase 1 validation should use action clamping to [-1, 1], EpisodeConfig.baseline_ema_alpha, and EpisodeConfig.grad_clip_norm; ReinforceAlgorithm reports baseline via last_baseline and state_dict baseline_ema.
- Added scripts\phase2_validation.py to check reward shaping (distance sanitization, instability clipping), deterministic step info, and hardware disablement safeguards.

### 2026-04-04: Phase 2 Complete — Team Orchestration

**Context:** Phase 2 deterministic validation complete across team.

**Cross-Agent Coordination:**
- **Mickey:** Scoped Phase 2; gated WI-1 + WI-4 before dependent work
- **Donald:** Wired RewardShapingConfig through training/inference
- **Minnie:** Defined metrics and validation checklist
- **Scribe:** Merged decisions and orchestration logs

**Status:**
- phase2_validation.py deterministic checks complete (shaping pipeline, distance sanitization, instability clipping, parameter bounds)
- Aligned with Donald's reward shaping implementation
- README updated with Phase 2 validation command
- Hardware mode verified disabled in validation runs

**Next:** Run phase2_validation.py as pre-training harness; verify all checks pass before executing WI-1 + WI-4; monitor for signal quality in full training runs.

### 2026-04-04: Phase 2 Analysis Results — Run 3 Determinism Investigation Needed

**Context:** Minnie completed Phase 2 analysis on 4 training runs (ORI, 26_04_04_1/2/3). Results show signal quality degradation in Run 3.

**Key Finding:** Run 3 correlation drops to -0.262 vs Runs 1-2 at -0.337 (target -0.3). This suggests potential config sensitivity or determinism issue.

**Action Items for Goofy:**
1. Run phase2_validation.py on Run 3 training logs to detect determinism issues (if any detected by validation framework)
2. Verify reward shaping pipeline consistent across all 3 Phase 2 runs
3. Check instability clipping behavior; investigate if Run 3 clipping margin differs from R1-2

**Team Recommendation:** Proceed to WI-2 regardless; investigate Run 3 separately. Consider parallel penalty weight tuning experiment for WI-1 revision.

**Decision Archived:** D5 merged to decisions.md.
