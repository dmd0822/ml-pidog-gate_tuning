---
title: ml-pidog-gate_tuning Architecture
description: System design for reinforcement-learning gait tuning
updated: 2026-04-04
---

# Architecture Overview

`ml-pidog-gate_tuning` is a reinforcement learning system that tunes the quadruped gait parameters of a SunFounder PiDog to maximize forward motion while minimizing IMU-measured instability. The system abstracts between simulation (default) and optional real hardware via a pluggable environment interface.

## System Diagram

```
┌─────────────────────────────────────────────────────────┐
│                     Training Loop                        │
│  (train.py: algorithm-agnostic loop)                    │
└─────────────────────────────────────────────────────────┘
         │                              │
         ▼                              ▼
┌──────────────────┐           ┌────────────────────┐
│  PolicyNetwork   │           │ PiDogGaitEnv       │
│  (policy.py)     │◄─────────►│ (env.py)           │
│                  │           │                    │
│ • 2-layer MLP    │           │ • State: [gait +   │
│ • Gaussian dist  │           │   IMU] (7D)        │
│ • Action: deltas │           │ • Action: deltas   │
│   for gaits      │           │   (4D)             │
└──────────────────┘           │ • Reward: dist -   │
                               │   instability      │
                               └────────────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    ▼                  ▼                  ▼
        ┌─────────────────────┐ ┌──────────────┐ ┌─────────────┐
        │ Simulation Path     │ │ GaitParams   │ │ HardwareCfg │
        │ (env.py: _sim...)   │ │ (env.py)     │ │ (config.py) │
        │                     │ │              │ │             │
        │ • Deterministic     │ │ 4 params:    │ │ • Speed     │
        │   physics model     │ │ - stride_len │ │   range     │
        │ • IMU: synthetic    │ │ - step_height│ │ • Safety    │
        │   wobble signal     │ │ - cycle_time │ │   limits    │
        │                     │ │ - lateral    │ │ • Method    │
        │                     │ │   offset     │ │   mapping   │
        └─────────────────────┘ └──────────────┘ └─────────────┘
                    │                                   │
                    │ (optional)                        ▼
                    │                        ┌──────────────────┐
                    │                        │ PidogHardware    │
                    │                        │ (pidog_hw.py)    │
                    │                        │                  │
                    │                        │ • Wraps pidog    │
                    │                        │   library        │
                    │                        │ • Maps params to │
                    │                        │   API calls      │
                    │                        │ • Reads IMU &    │
                    │                        │   distance       │
                    │                        └──────────────────┘
                    │                                   │
                    └───────────────┬───────────────────┘
                                    ▼
                        ┌──────────────────────┐
                        │ [Distance, IMU] ×4D  │
                        │ Sensor Readings      │
                        └──────────────────────┘
```

## Core Modules

### 1. **train.py** — Training Loop & Checkpointing
**Responsibility:** Main entry point, episode orchestration, algorithm selection, gradient updates.

**Key Classes & Functions:**
- `TrainingHistory`: Accumulates rewards, distances, instabilities per episode
- `EpisodeStats`: Summarizes episode metrics
- `run_episode()`: Executes one rollout; returns log-probs, rewards, and stats
- `save_checkpoint()`: Serializes policy weights, algorithm state, and metrics
- `plot_results()`: Generates 3-panel plot (reward, distance, instability vs. episode)
- `_create_run_output_dir()`: Allocates unique `yy_mm_dd_x` output directories
- `_create_algorithm()`: Factory function to instantiate the algorithm (REINFORCE, or extensible to others)
- `train()`: Main loop—runs episodes, applies algorithm gradient, checkpoints

**Algorithm Dispatch:**
The `_create_algorithm()` factory matches `config.algorithm` to the appropriate implementation:
```python
if algorithm_name == "reinforce":
    return ReinforceAlgorithm(...)
```
This enables swapping algorithms without modifying the training loop.

**Default Algorithm:** REINFORCE with baseline subtraction (see `algorithms/reinforce.py`).
The training loop delegates loss computation and optimizer updates to the selected algorithm.

**Output:**
- Periodic checkpoints: `checkpoint_ep{N}.pt` (5 saved per training run)
- Final checkpoint: `checkpoint_final.pt`
- Plot: `training_results_{date}_{index}.png`

---

### 2. **env.py** — Gym-Style Environment
**Responsibility:** Encapsulates gait control, sensor reading, and reward computation.

**Key Classes:**
- `GaitParameters`: Immutable dataclass holding 4 gait knobs (stride_length, step_height, cycle_time, lateral_offset)
  - `as_array()`: Returns [stride, height, cycle, offset] for neural net consumption
  - `as_dict()`: Maps to hardware API parameter names
  
- `PiDogGaitEnv`: Main environment
  - **State design:** 7D = [4 gait params] + [3 IMU channels (roll, pitch, yaw)]
    - Policy sees both the control knobs and the stability response
  - **Action design:** 4D continuous deltas, scaled by `ActionScaling` and clipped by `SafetyLimits`
  - **Reward:** `distance × weight_forward − instability × weight_instability`
    - Instability = sum of |roll| + |pitch| + |yaw|, clipped to avoid outlier spikes
    - Invalid distance readings (e.g., −2.0 from hardware) treated as 0

**Key Methods:**
- `reset()`: Initialize gait to mid-range, zero IMU, stand robot (hardware mode)
- `step(action)`: Apply action deltas, run robot, read sensors, compute reward
- `_apply_action()`: Scale and clip deltas; update gait state
- `_run_robot()`: Route to hardware or simulation
- `_simulate_robot_run()`: Synthetic rollout with deterministic physics + noise
  - Distance ∝ stride / cycle_time
  - IMU wobble ∝ stride + 1/cycle_time (speed–stability tradeoff embedded in sim)

**Sensor Handling:**
- IMU: Exponential smoothing (α=0.6) to damp real-world noise
- Distance: Sanitized for NaN/invalid values

---

### 3. **policy.py** — Stochastic Policy Network
**Responsibility:** Neural network that learns the mapping from state to action distribution.

**Architecture:**
```
Input (7D state)
  ↓
Linear(7 → 64) + ReLU
  ↓
Linear(64 → 64) + ReLU
  ↓
┌─────────────────┬────────────┐
▼                 ▼
mean head      log_std param
Linear(64 → 4)    (4D, learnable)
  ↓                 ↓
Output(mean)    std = exp(log_std)
                  ↓
             N(mean, std)
               (Gaussian)
```

**Key Classes:**
- `PolicyOutput`: Dataclass holding sampled action and log-probability
- `PolicyNetwork`: `nn.Module` wrapping the MLP + distribution head

**Key Methods:**
- `forward(state)`: Returns (mean, log_std)
- `sample(state)`: Reparameterized sample from Gaussian; returns action and log_prob

**Design Rationale:**
- 2-layer 64-neuron MLP: Simple, efficient for 4D action space
- Diagonal Gaussian: Allows independent uncertainty per action dimension
- Learnable log_std: Policy adjusts exploration as it learns

---

### 4. **config.py** — Centralized Configuration
**Responsibility:** All tunable hyperparameters and safety constraints.

**Key Dataclasses:**
- `ActionScaling`: Scales raw policy outputs (e.g., stride_length_scale=0.01)
- `RewardWeights`: Forward motion weight (1.0) vs. instability penalty (0.35)
- `EpisodeConfig`: Episode length (50 steps) and discount γ (0.98)
- `ImuConfig`: Smoothing coefficient (0.6)
- `SafetyLimits`: Hard bounds on each gait parameter (servo / stability)
  - stride_length: [0.04, 0.12]
  - step_height: [0.015, 0.05]
  - cycle_time: [0.18, 0.42]
  - lateral_offset: [−0.03, 0.03]
- `HardwareConfig`: Configures pidog library integration
  - Method names (do_action, set_pose, accData, read_distance)
  - Speed/height/offset ranges for API mapping
  - Stand/lie actions (executed before/after control)
- `TrainingConfig`: Bundles all above + learning_rate (1e-3) and episodes (2000)

**Design Philosophy:** All hyperparameters in one frozen dataclass; encourages explicit decision-making and reproducibility.

---

### 5. **pidog_hw.py** — Hardware Adapter
**Responsibility:** Bridges the environment to the SunFounder `pidog` library; handles parameter mapping and API dispatch.

**Key Classes:**
- `GaitLike` (Protocol): Any object with `.as_dict()` method (duck typing for extensibility)
- `PidogHardware`: Adapter around pidog.Pidog instance

**Key Methods:**
- `__init__()`: Lazy-loads pidog.Pidog, caches method references for speed
- `ensure_standing()` / `ensure_lie_down()`: Safe poses before/after control
- `run_and_measure(gait)`: Core integration point
  - Maps gait params to pidog API calls:
    - stride_length → speed (via linear range mapping)
    - cycle_time → run_duration_sec
    - step_height → body z-height
    - lateral_offset → body y-offset
  - Calls `do_action("forward", speed=...)` or `set_pose(...)` as configured
  - Waits for completion (via `wait_all_done()`)
  - Reads distance + IMU; extracts 3 channels (roll, pitch, yaw) by key name
  - Returns (distance: float, imu: np.ndarray[3])

**Robustness:**
- Graceful handling of optional `run_duration` argument (try-except)
- IMU dict/array polymorphism (dict with keys or array-like)
- Type checking for method availability with helpful error messages
- `_map_range()`: Linear interpolation to rescale gait ranges to pidog hardware ranges

**Design:**
- Method lookup deferred to `_get_method()` to catch misconfigurations early
- Range mapping centralizes hardware-specific calibration
- Minimal coupling to env: only `.as_dict()` interface assumed

---

### 6. **infer.py** — Inference & Deployment
**Responsibility:** Load a trained checkpoint and execute rollouts (sim or hardware).

**Key Functions:**
- `load_policy()`: Deserialize policy from checkpoint
- `run_inference()`: Execute N steps in the environment using loaded policy
  - Deterministic mode: use mean action (policy gradient only)
  - Stochastic mode: sample action (for evaluation robustness)
- `parse_args()`: CLI: `--checkpoint`, `--steps`, `--use-hardware`, `--stochastic`
- `main()`: Entry point

**Workflow:**
1. Load config + create environment (hardware=use_hardware)
2. Load policy from checkpoint
3. Rollout N steps, printing step metrics
4. If hardware: lie down safely at end

**Design:** Reuses the same `PiDogGaitEnv` class; decouples training from inference deployment.

---

### 7. **utils.py** — Utilities
**Responsibility:** Common numerical operations.

**Functions:**
- `set_seed()`: Pin numpy & torch RNG for reproducibility
- `compute_returns()`: Discounted cumulative reward (used by the REINFORCE algorithm)
  - Reverse iterate: G_t = r_t + γ × G_{t+1}
- `exp_smooth()`: Exponential moving average (damps IMU sensor noise)

---

### 8. **algorithms/base.py** — Algorithm Interface
**Responsibility:** Defines the contract for all RL algorithms.

**Key Classes:**
- `Algorithm` (ABC): Abstract base class requiring implementations of:
  - `__init__(policy, learning_rate, config)`: Initialize with policy and hyperparameters
  - `compute_loss(log_probs, rewards, **kwargs)`: Compute loss from episode trajectory
  - `update(loss)`: Perform gradient step
  - `state_dict()`: Serialize algorithm state for checkpointing
  - `load_state_dict(state_dict)`: Restore algorithm state from checkpoint

**Design Rationale:**
- Pluggable interface: enables swapping algorithms (REINFORCE ↔ PPO ↔ A2C) with minimal code changes
- Standardizes checkpointing: all algorithms export/import state via dict
- Flexible kwargs: accommodates algorithm-specific data (value estimates, replay buffer indices, etc.)

---

### 9. **algorithms/reinforce.py** — REINFORCE Implementation
**Responsibility:** Monte Carlo policy gradient with baseline.

**Key Classes:**
- `ReinforceAlgorithm`: Implements the `Algorithm` interface

**Algorithm Details:**
1. Collect full episode trajectory: (state, action, reward, log_prob)
2. Compute discounted returns: G_t = r_t + γ·G_{t+1} + ... + γ^(T-1)·r_T
3. Compute baseline (mean return) to reduce variance
4. Compute advantages: A_t = G_t − baseline
5. Loss = −Σ(log_prob_t × A_t) [negative because we maximize reward]
6. Backprop and optimizer step

**Why REINFORCE for PiDog:**
- On-policy, Monte Carlo: fits short episodes (50 steps) naturally
- Baseline reduces variance without bias
- No critic network: simpler, faster, fewer hyperparameters
- Stable convergence for continuous control with Gaussian policies

**Trade-offs vs. Alternatives:**
- vs. A2C/A3C: Adds critic (value network) for lower variance; more complex
- vs. PPO: Uses full trajectory; PPO uses minibatches + trust region; PPO better for long episodes
- vs. SAC: Entropy-regularized; for off-policy; overkill for this scale

---

## Data Flow

### Training Loop Sequence

```
Initialize policy & algorithm
├─ For episode 1 to N:
│  ├─ env.reset()
│  │  └─ state = [gait, imu=0]
│  │
│  ├─ For step 1 to max_steps:
│  │  ├─ policy.sample(state_tensor)
│  │  │  └─ return (action, log_prob)
│  │  │
│  │  ├─ env.step(action)
│  │  │  ├─ _apply_action() → update gait
│  │  │  ├─ _run_robot()
│  │  │  │  ├─ hardware.run_and_measure() or _simulate_robot_run()
│  │  │  │  └─ return (distance, imu_raw)
│  │  │  ├─ Smooth IMU
│  │  │  ├─ Compute reward = dist − instability_penalty
│  │  │  └─ return (next_state, reward, done, info)
│  │  │
│  │  └─ Accumulate log_prob, reward, distance, instability
│  │
│  ├─ Algorithm.compute_loss(log_probs, rewards)
│  ├─ Algorithm.update(loss)
│  │
│  └─ Checkpoint & plot (periodic)
```

### Inference Sequence

```
Load checkpoint
├─ Create environment (hardware config)
├─ Load policy weights
│
└─ For step 1 to N:
   ├─ policy(state) → action (mean or sampled)
   ├─ env.step(action)
   │  └─ [same as training step]
   │
   └─ Print metrics + continue or terminate if done
```

---

## Key Design Decisions

### 1. **State Representation**
- **Choice:** [gait_params (4D) + imu (3D)]
- **Rationale:** Policy needs to see both the knobs it's turning and the stability response; this couples exploration and exploitation naturally.
- **Alternative rejected:** IMU-only state would make parameter tuning invisible to the policy.

### 2. **Reward Structure**
- **Choice:** `distance × 1.0 − instability × 0.35`
- **Rationale:** Weighted scalar reward balances speed vs. stability; weight tunable without retraining.
- **Trade-off:** Instability clipping (250) prevents outlier IMU spikes from dominating the loss.
- **Hook:** Optional reward shaping (normalize/scale/shift/clip) via `RewardShapingConfig` with no-op defaults.

### 3. **Action Space**
- **Choice:** Continuous 4D deltas, scaled and clipped
- **Rationale:** Allows smooth exploration of gait space; safety limits prevent damage.
- **Alternative rejected:** Discrete actions would limit fine-tuning; unlimited continuous could overstress servo.

### 4. **Simulation vs. Hardware Abstraction**
- **Choice:** Single `PiDogGaitEnv`, pluggable hardware backend
- **Rationale:** Enables safe sim-to-real transfer; reduces code duplication; eases debugging.
- **Assumption:** Synthetic reward in sim correlates with hardware performance (not validated here).

### 5. **Policy Architecture**
- **Choice:** 2-layer 64-neuron MLP with Gaussian output
- **Rationale:** Sufficient expressiveness for 4D→4D mapping; simple to debug; fast inference.
- **Alternative rejected:** Deeper networks would overfit to sim noise; recurrent would complicate training.

### 6. **Default Learning Algorithm: REINFORCE with Baseline**
- **Choice:** Policy gradient, no critic (default algorithm)
- **Rationale:** Simple, stable, suitable for short episodes (50 steps); baseline reduces variance without bias.
- **Alternative rejected:** Actor-critic would add complexity; PPO/SAC overkill for this scale. The architecture still supports these via the `Algorithm` interface.

### 7. **Configuration Management**
- **Choice:** Frozen dataclasses bundled in TrainingConfig
- **Rationale:** Single source of truth; explicit; serializable; reproducible.
- **Alternative rejected:** YAML/JSON would lose type safety; function args too many.

---

## Safety & Robustness

### Gait Parameter Bounds
Every action is clipped to `SafetyLimits` before applying to the gait:
- Prevents damage to servos (stride, height, cycle_time bounds)
- Avoids unsafe poses (lateral_offset limits)
- Applied in `_apply_action()` before sensors are read

### Invalid Distance Handling
Hardware IMU/distance sensors may return NaN or sentinel values (e.g., −2.0):
- Treated as 0 distance in `_sanitize_distance()`
- Prevents reward collapse; encourages exploration of valid states

### IMU Noise Damping
Real IMU signals are noisy and have outlier spikes:
- Exponential smoothing (α=0.6) damps high-frequency noise
- Instability metric clipped (max 250) to prevent outliers from overwhelming the reward

### Hardware Mode Protection
- `ensure_standing()` before control starts
- `ensure_lie_down()` after inference completes
- Short run durations (0.5 sec default)
- Speed ranges limited (60–100 units)

---

## Extension Points

### 1. **Custom Reward Functions**
Modify `_compute_reward()` in env.py or `RewardWeights` in config.py:
```python
reward = distance − 0.5 * instability + bonus_for_pattern
```
For simple shaping without changing reward formula, use `RewardShapingConfig` to normalize, scale, shift, or clip.

### 2. **Different Gait Parameters**
Add fields to `GaitParameters` and `SafetyLimits`; update `state_dim` and `action_dim`.

### 3. **Alternative Hardware Backends**
Implement a new adapter class with `.run_and_measure(gait) → (distance, imu)` signature.

### 4. **Richer Sensor Data**
Add force/pressure/accelerometer channels to state; update network input size.

### 5. **Off-Policy or Model-Based Learning**
Replace `run_episode()` with replay buffer + Q-learning; reuse env and policy interfaces.

---

## Extending with New Algorithms

The system supports pluggable RL algorithms via the `Algorithm` interface. To add a new algorithm (e.g., A2C, PPO, SAC):

### Steps

1. **Create the algorithm class** in `pidog_rl/algorithms/my_algorithm.py`:
   ```python
   from .base import Algorithm
   
   class MyAlgorithm(Algorithm):
       def __init__(self, policy, learning_rate, config):
           self.policy = policy
           self.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
       
       def compute_loss(self, log_probs, rewards, **kwargs):
           # Your algorithm-specific loss computation
           returns = compute_returns(rewards, self.config.gamma)
           return your_loss_formula(log_probs, returns)
       
       def update(self, loss):
           self.optimizer.zero_grad()
           loss.backward()
           self.optimizer.step()
       
       def state_dict(self):
           return {"optimizer_state_dict": self.optimizer.state_dict()}
       
       def load_state_dict(self, state_dict):
           self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
   ```

2. **Register the algorithm** in `pidog_rl/algorithms/__init__.py`:
   ```python
   from .my_algorithm import MyAlgorithm
   __all__ = ["Algorithm", "ReinforceAlgorithm", "MyAlgorithm"]
   ```

3. **Add to the factory** in `pidog_rl/train.py` (_create_algorithm function):
   ```python
   elif algorithm_name == "my_algorithm":
       return MyAlgorithm(policy, learning_rate, config.episode)
   ```

4. **Use it** by setting `TrainingConfig(algorithm="my_algorithm")` or passing `--algorithm my_algorithm` if CLI args are added.

### Key Patterns

- **Actor-Critic (A2C, A3C):** Add a value network; pass value predictions via kwargs
- **Off-Policy (DQN, SAC):** Maintain a replay buffer; modify run_episode to sample from buffer
- **Trust-Region (PPO, TRPO):** Reuse log_probs; implement policy clipping in compute_loss

See `pidog_rl/algorithms/README.md` for detailed guidance.

---

## Running the System

### Training (Simulation)
```bash
python -m pidog_rl.train
```
Outputs checkpoints and plot to `output/yy_mm_dd_x/`.

### Inference (Simulation)
```bash
python -m pidog_rl.infer --checkpoint output/26_04_04_1/checkpoint_final.pt --steps 50
```

### Inference (Hardware)
```bash
python -m pidog_rl.infer --checkpoint output/26_04_04_1/checkpoint_final.pt --steps 10 --use-hardware
```

---

## Dependencies

- **PyTorch 2.11.0:** Neural network, autograd, gradient optimization
- **NumPy 2.4.4:** Array operations, random number generation, smoothing
- **Matplotlib 3.10.8:** Training curve plots
- **pidog (SunFounder):** Hardware control (optional; only when `use_hardware=True`)

---

## File Structure

```
pidog_rl/
├── __init__.py
├── config.py           # All config dataclasses
├── env.py              # PiDogGaitEnv, GaitParameters, sensor logic
├── policy.py           # PolicyNetwork, PolicyOutput
├── pidog_hw.py         # PidogHardware adapter
├── train.py            # Training loop, checkpointing, plotting (includes _create_algorithm factory)
├── infer.py            # Inference entry point
├── utils.py            # Shared utilities (set_seed, compute_returns, exp_smooth)
└── algorithms/
    ├── __init__.py
    ├── base.py         # Algorithm ABC
    ├── reinforce.py    # ReinforceAlgorithm implementation
    ├── README.md       # Guide for adding new algorithms
    └── EXAMPLE_PPO.md  # (Reference) Example of PPO implementation
output/
├── yy_mm_dd_0/
│   ├── checkpoint_ep500.pt
│   ├── checkpoint_final.pt
│   └── training_results_yy_mm_dd_0.png
└── ...
```

---

## Testing & Validation

**Current gaps (Future Work):**
- Unit tests for env, policy, reward computation
- Integration test for sim-to-hardware parameter mapping
- Deterministic replay (record & replay exact sensor sequences)
- Hardware-in-the-loop validation on real PiDog

---

## References

- [REINFORCE Algorithm](https://en.wikipedia.org/wiki/Policy_gradient#Williams_1992): Basic policy gradient
- [SunFounder PiDog](https://github.com/sunfounder/pidog): Hardware library
- [Gym Interface](https://github.com/openai/gym): Environment standard
