# Project Context

- **Owner:** Dave Davis
- **Project:** Reinforcement-learning gait tuning for the SunFounder PiDog
- **Stack:** Python 3.10+, PyTorch, NumPy, Matplotlib
- **Created:** 2026-04-04

## Learnings

### Architecture & System Design
- **Core Pattern:** Single environment interface (PiDogGaitEnv) abstracts sim vs. hardware; pluggable backend via `run_robot()` method.
- **State Design:** [gait_params (4D) + imu_readings (3D)] allows policy to see both control knobs and stability response.
- **Reward Structure:** Scalar = distance − (instability_penalty); weights tunable without retraining. Instability clipped to handle outlier spikes.
- **REINFORCE + Baseline:** Simple policy gradient suitable for short episodes (50 steps); baseline reduces variance; no critic needed.
- **Policy Network:** 2-layer 64-neuron MLP → Gaussian distribution. Minimal but sufficient for 4D action space.
- **Pluggable Algorithms:** `Algorithm` ABC in `algorithms/base.py` + factory pattern in `train.py` enables algorithm swapping (REINFORCE ↔ PPO ↔ A2C) without modifying core loop.
- **Algorithm Interface:** `compute_loss()`, `update()`, `state_dict()`, `load_state_dict()` standardize checkpoint serialization and extensibility.

### Safety & Robustness
- **Gait Bounds:** All actions clipped to SafetyLimits before applying (protects servos, prevents unsafe poses).
- **Sensor Handling:** Invalid distance readings (e.g., −2.0) treated as 0; IMU smoothed with exponential filter (α=0.6) to damp noise.
- **Hardware Safety:** `ensure_standing()` before control, `ensure_lie_down()` after; short durations (0.5 sec default).

### Configuration & Reproducibility
- **Frozen Dataclasses:** All hyperparameters in TrainingConfig; encourages explicit decisions, enables serialization.
- **Run Directory Naming:** yy_mm_dd_x format with auto-incrementing index; prevents overwrite, eases archival.

### Key File Purposes
- `train.py`: REINFORCE loop with factory pattern for algorithm selection; checkpointing (every episode÷5); plotting (reward, distance, instability).
- `env.py`: Gym-style wrapper; handles gait application, sensor fusion, reward computation.
- `policy.py`: Minimal MLP + Gaussian sampler; reparameterized gradients.
- `pidog_hw.py`: Adapter pattern; maps gait params to pidog library; handles method dispatch and range remapping.
- `config.py`: Single source of truth for hyperparams, safety, hardware integration.
- `infer.py`: Deployment entry point; loads checkpoint, runs N steps (deterministic or stochastic).
- `utils.py`: Shared helpers (seed, returns, smoothing).
- `algorithms/base.py`: ABC defining algorithm contract (compute_loss, update, state_dict, load_state_dict).
- `algorithms/reinforce.py`: Concrete REINFORCE with baseline implementation.
- `algorithms/README.md`: Extension guide for adding new algorithms (PPO, A2C, SAC patterns documented).
