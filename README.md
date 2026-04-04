---
title: ml-pidog-gate_tuning
description: Reinforcement-learning gait tuning for the SunFounder PiDog, with an optional hardware adapter and checkpoint/plot outputs.
author: dmd0822
ms.date: 2026-04-04
ms.topic: overview
keywords:
  - reinforcement learning
  - pidog
  - robotics
  - pytorch
estimated_reading_time: 4
---

## Overview

This repository trains a small reinforcement-learning policy to adjust PiDog gait parameters for forward motion while penalizing instability (IMU wobble). By default it runs in a lightweight simulation stub, and it can optionally execute short rollouts on real PiDog hardware via the `pidog` Python package.

## What is in the repo

* [pidog_rl/train.py](pidog_rl/train.py): Training loop, algorithm factory, checkpointing, and plotting
* [pidog_rl/algorithms/](pidog_rl/algorithms/): RL algorithm implementations (PPO, REINFORCE)
* [pidog_rl/env.py](pidog_rl/env.py): Gym-style environment that applies action deltas to gait parameters, runs a rollout (sim or hardware), and computes rewards
* [pidog_rl/policy.py](pidog_rl/policy.py): Policy network that outputs a Gaussian distribution over action deltas
* [pidog_rl/pidog_hw.py](pidog_rl/pidog_hw.py): Hardware adapter around the SunFounder `pidog` API (optional)
* [pidog_rl/config.py](pidog_rl/config.py): Training, safety limits, hardware, and algorithm selection configuration
* [output/](output/): Example checkpoints produced by training

## Setup

This project expects Python 3.10+.

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
\.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Install as a package

If you want to install the project (for example on a Raspberry Pi), use pip from the repo root:

```bash
python3 -m pip install .
```

For hardware support (the `pidog` library), you must install `pidog` from SunFounder first (it is not on PyPI):

```bash
git clone --depth=1 https://github.com/sunfounder/pidog.git
python3 -m pip install ./pidog --break-system-packages
```

## Run training (simulation)

The recommended entry point is module execution so that relative imports resolve correctly. Ensure your venv is active and `hardware.use_hardware` remains `False` (default) to keep runs in the simulator.

```powershell
python -m pidog_rl.train
```

Training writes checkpoints and a plot into `output/`.

## Validation

Use the validation script to confirm stability safeguards before approving a training change:

```powershell
python scripts\phase1_validation.py
```

Additional checks for EMA baseline and gradient clipping:

```powershell
python scripts\phase1_validation.py --check-ema-baseline
python scripts\phase1_validation.py --check-grad-clip
```

To validate that reward signal quality is maintained and training is stable:

```powershell
python scripts\phase2_validation.py --seed 42
```

After a training run, analyze checkpoint quality:

```powershell
python scripts\phase2_analysis.py --checkpoint output\26_04_04_1\checkpoint_final.pt
```

For detailed instability margin analysis (detects if clipping threshold is too tight):

```powershell
python scripts\phase2_analysis.py --checkpoint output\26_04_04_1\checkpoint_final.pt --plot-instability-margin
```

### Analysis Outputs

The analysis script produces:

* `phase2_reward_stability.png`: Scatter plot showing reward-instability trade-off (should show negative correlation)
* `phase2_convergence.png`: Reward, distance, and instability trends over training (should show learning progression)
* `phase2_instability_margin.png`: Distribution of instability values relative to clipping threshold (should not exceed 5% clipped)

Key metrics to watch:
- **Correlation**: Should be < -0.3 (instability penalty is working)
- **Clipped episodes**: Should be < 5% (margin is adequate)
- **Variance trend**: Should decrease over time (exploitation > exploration)

### Algorithm Selection

The default algorithm is REINFORCE (Monte Carlo policy gradient). To use a different algorithm, modify `TrainingConfig.algorithm` in [pidog_rl/config.py](pidog_rl/config.py):

```python
@dataclass(frozen=True)
class TrainingConfig:
    algorithm: str = "reinforce"  # Options: "reinforce", "ppo"
    # ... rest of config
```

Available algorithms:
- **REINFORCE**: Monte Carlo policy gradient with EMA baseline (default)
- **PPO**: Proximal Policy Optimization with clipped objective and multi-epoch updates

PPO hyperparameters can be configured via `PPOConfig`:
```python
@dataclass(frozen=True)
class PPOConfig:
    clip_epsilon: float = 0.2        # Clipping range for importance ratio
    num_epochs: int = 4              # Number of optimization epochs per episode
    normalize_advantages: bool = True # Whether to normalize advantages
```

For instructions on adding new algorithms (A2C, SAC, etc.), see [pidog_rl/algorithms/README.md](pidog_rl/algorithms/README.md).

## Run inference (trained policy)

Use the inference script to apply a trained policy to the environment:

```powershell
python -m pidog_rl.infer --checkpoint output\26_04_04_1\checkpoint_final.pt --steps 50
```

To run on real hardware, make sure `pidog` is installed and pass `--use-hardware`:

```powershell
python -m pidog_rl.infer --checkpoint "output/26_04_04_2/checkpoint_final.pt" --steps 10 --use-hardware
```

## Raspberry Pi setup

On Raspberry Pi, you can run the setup script (recommended):

```bash
chmod +x scripts/setup_pi.sh
./scripts/setup_pi.sh
```

Or run the steps manually:

```bash
sudo apt update
sudo apt install -y git python3-pip python3-setuptools python3-smbus

cd ~/
git clone -b 2.5.x --depth=1 https://github.com/sunfounder/robot-hat.git
cd robot-hat
sudo python3 install.py

cd ~/
git clone --depth=1 https://github.com/sunfounder/vilib.git
cd vilib
sudo python3 install.py

cd ~/
git clone --depth=1 https://github.com/sunfounder/pidog.git
cd pidog
sudo rm -rf pidog.egg-info build dist
python3 -m pip install . --no-build-isolation --no-deps --break-system-packages

cd ~/
git clone https://github.com/dmd0822/ml-pidog-gate_tuning.git
cd ml-pidog-gate_tuning
python3 -m pip install -U pip
python3 -m pip install . --break-system-packages
```

Hardware mode requires the `pidog` package and access to the PiDog hardware.

> [!NOTE]
> PyTorch wheels vary by platform. If `pip install .` fails on Raspberry Pi due to torch, install a compatible CPU wheel for your Pi first, then rerun the install.

## Outputs

Training produces:

Each training run writes into a separate subfolder under `output/` named `yy_mm_dd_x`, where `x` is an auto-incrementing index for that day.

Inside each run folder:

* `checkpoint_ep*.pt`: Periodic checkpoints (episode, model weights, algorithm state, and reward history)
* `checkpoint_final.pt`: Final checkpoint
* `training_results_yy_mm_dd_x.png`: Reward, distance, and instability curves

## Hardware mode

Hardware mode is controlled by `TrainingConfig.hardware.use_hardware` in [pidog_rl/config.py](pidog_rl/config.py). When enabled, the environment uses [pidog_rl/pidog_hw.py](pidog_rl/pidog_hw.py) to call into the `pidog` library. The default mapping is:

* `stride_length` → `do_action("forward")` speed
* `cycle_time` → `run_duration_sec`
* `step_height` → `set_pose(z=...)` body height
* `lateral_offset` → `set_pose(y=...)` lateral offset

After inference completes in hardware mode, the script issues a `lie` action (configurable via `HardwareConfig.lie_*`) so the dog rests.

You can tune the mapping ranges via `HardwareConfig` (speed/body height/offset ranges). If you want a different PiDog API call, override `apply_gait_method`/`apply_gait_action`.

> [!WARNING]
> Enabling hardware mode can move the robot repeatedly during training. Keep the run duration short, ensure the robot is in a safe area, and have a quick way to cut power.

## How the reward works

Each environment step:

* Updates gait parameters using scaled, clipped action deltas (bounded by safety limits)
* Runs a short rollout and reads distance and IMU
* Computes reward as forward distance minus an instability penalty (sum of absolute IMU channels, clipped to avoid outlier spikes). Invalid distance readings (for example `-2.0` from the hardware API) are treated as zero.

You can tune trade-offs through `RewardWeights` in [pidog_rl/config.py](pidog_rl/config.py).
