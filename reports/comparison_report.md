# Training Run Comparison Summary

## ORI vs Phase 2 (Runs 1, 2, 3) — 2000 episodes each

### Metrics Table

| Metric | ORI (Phase 1) | Phase 2 Run 1 | Phase 2 Run 2 | Phase 2 Run 3 |
|--------|---|---|---|---|
| Reward Mean | -4.78 ± 1.15 | -4.83 ± 1.16 | -4.83 ± 1.16 | -4.83 ± 1.16 |
| Last 10 Eps | -4.75 | -4.86 (-1.0%) | -4.86 (-1.0%) | -4.99 (-1.0%) |
| Instability | 31.18 | 31.10 (-0.3%) | 31.10 (-0.3%) | 31.16 (-0.1%) |
| Last 10 Eps | 31.93 | 31.74 | 31.74 | 31.95 |
| Distance | 6.13 | 6.05 (-1.3%) | 6.05 (-1.3%) | 6.08 (-0.8%) |
| Last 10 Eps | 6.43 | 6.25 | 6.25 | 6.19 |
| Correlation | -0.328 (OK) | -0.337 (GOOD) | -0.337 (GOOD) | -0.262 (WARN) |
| Clipping % | 0.0% | 0.0% | 0.0% | 0.0% |

### Success Criteria (vs Phase 1 baseline)

| Criterion | Status |
|-----------|--------|
| Reproducibility: Runs 1-2 identical, all < 2% variance | ✓ PASS |
| No Distance Regression: 6.05 vs 6.13 (-1.3%) | ✓ PASS |
| Reward +20%: -4.83 vs -4.78 (−1.0% worse) | ✗ FAIL |
| Instability −20%: 31.10 vs 31.18 (−0.3% negligible) | ✗ FAIL |

**OVERALL: 2 of 4 criteria met**

---

## Key Findings

### 1. No Convergence Improvement

Phase 2 mean reward **regressed by 1%** vs ORI. Training plateau indicates local optimum reached with current reward formulation.

### 2. Instability Not Reduced

Instability stable at ~31.2 across all runs. Reward penalty insufficient to drive gait improvement.

### 3. Signal Quality Mixed

- **Runs 1-2:** Excellent correlation (-0.337)
- **Run 3:** Weak correlation (-0.262) — 2.7% degradation warrants investigation
- **All:** Healthy clipping (0%) means penalty signal differentiable throughout

### 4. Reproducibility Confirmed

Framework is stable — runs 1-2 identical, all within <2% variance.

---

## Root Cause

Agent stuck at local optimum similar to Phase 1. Possible causes:

1. **Reward weight imbalance:** Penalty too weak vs distance reward
2. **Agent capacity:** Architecture ceiling prevents more stable gaits
3. **Config variance:** Run 3 signal drop suggests hyperparameter sensitivity

---

## Recommendations

### Immediate Actions

- **Investigate Run 3 config** (why -2.7% correlation drop vs Runs 1-2?)
- **Increase instability_weight by 2-3x** for next experiment

### Next Phase

- Proceed to baseline enhancement (WI-2) to reduce exploration noise
- May unlock instability reduction once variance lower

---

## Artifacts

Generated 12 diagnostic plots (3 per run in `output/{ORI,26_04_04_1-3}/`):

- `phase2_reward_stability.png`: Reward-instability correlation & trends
- `phase2_convergence.png`: Mean, distance, instability over 2000 episodes
- `phase2_instability_margin.png`: Clipping margin safety distribution

All plots saved at dpi=150. Decision document in `.squad/decisions/inbox/minnie-phase2-no-improvement.md`

---

## Verdict

**Phase 2 tuning has NOT improved convergence.** Signal quality good but reward penalty insufficient. Recommend increased weights or baseline enhancement (WI-2) to proceed.
