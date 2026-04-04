# Squad Decisions

## Phase 2 Decisions

### D1: Phase 2 Scope — Gait Stability & Convergence (Mickey, 2026-04-04)

**Status:** Active

Phase 1 REINFORCE baseline shows slow convergence (2000 episodes) and high variance despite EMA baseline. Scoped Phase 2 into 4 work items:

1. **Reward Normalization (P1, Highest Impact):** Normalize distance/instability into unit-variance ranges; dynamic per-episode scaling. Track running mean/std; apply clip [-10, 10] to prevent outlier spikes.

2. **Baseline Enhancement (P2, Optional):** Upgrade EMA baseline to track per-action-type baselines (separate baselines for {high-stride, high-speed, stable} action clusters). Reduce variance for refinement actions.

3. **Return Variance Clipping (P3, Quick Win):** Clip discounted returns to [5th, 95th] percentile to remove episode outliers. Log clipping rate per episode.

4. **Convergence Benchmark (P1, Gating):** Create `phase2_benchmark.py` to run Phase 1 vs. Phase 2 training side-by-side and compare convergence curves.

**Success Criteria:**
- Final 200-ep reward mean +20% vs. Phase 1
- Instability std dev −20% vs. Phase 1
- Reproducibility: Two runs within 5% (rolling avg)
- No regression on distance metric

**Decision Gates:** WI-1 + WI-4 must complete before WI-2 starts (measure P1 impact alone). WI-3 independent.

**Risk Mitigation:** Keep config flags for A/B testing; all changes preserve Phase 1 infrastructure.

### D2: Reward Shaping Hooks Implementation (Donald, 2026-04-04)

**Status:** Active

Added config-driven reward shaping pipeline inside `PiDogGaitEnv` to enable Phase 2 tuning without rewriting core reward formula. Defaults are no-op to preserve Phase 1 behavior.

**Implementation:**
- `RewardShapingConfig` in `config.py` with optional normalize, scale, shift, clip hooks
- Reward pipeline in `env.py:step()` with per-episode normalization state reset
- Wired through `train.py` and `infer.py` for consistent behavior
- Validation checks in `phase2_validation.py`

**Rationale:** Centralizes reward tuning in the env; enables quick experimentation through config without breaking API.

**Architecture Notes:**
- Pipeline is deterministic and side-effect-free
- No breaking changes to checkpoint format
- Config-driven A/B testing without code changes

### D3: Phase 2 Metrics & Validation Framework (Minnie, 2026-04-04)

**Status:** Active

Defined Phase 2 validation metrics to catch signal degradation early:

**Core Signal Diagnostics:**
1. **Reward-Instability Correlation (Primary):** Expected r < -0.3. Positive or weak correlation = penalty failure.
2. **Clipping Margin Health (Safety):** Expected < 5% episodes hit instability_clip. >10% clipped = gradient signal lost.
3. **Convergence Signals (Training Quality):** Reward variance decreasing, distance/instability trends improving or plateau (not diverging).
4. **Deterministic Safety (Integrity):** Same action → same gait parameters; all parameters within bounds.

**Validation Tools:**
- `phase2_validation.py`: Unit-level tests (determinism, clipping, bounds)
- `phase2_analysis.py`: High-level checkpoint analysis (plots, summary stats)

**Key Assumptions:**
- Stochastic IMU noise is expected (gait params deterministic)
- Negative correlation is the signal
- Clipping threshold should never be hit (indicates reward scale wrong)

### D4: Phase 2 Deterministic Validation (Goofy, 2026-04-04)

**Status:** Active

Created `scripts/phase2_validation.py` to validate reward shaping determinism and stability signals before training runs.

**Coverage:**
- Reward shaping pipeline health (normalization, scaling, clipping)
- Distance sanitization (invalid readings → 0)
- Instability clipping (raw values properly clipped)
- Gait parameter determinism (same action sequence → same parameters)
- Safety bounds enforcement (all parameters within limits)
- Hardware mode disabled in validation runs

**Documentation:** README updated with Phase 2 validation command for pre-training checklist.

## Governance

- All meaningful changes require team consensus
- Document architectural decisions here
- Keep history focused on work, decisions focused on direction
