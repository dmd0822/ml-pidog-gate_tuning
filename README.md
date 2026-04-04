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

* [pidog_rl/train.py](pidog_rl/train.py): Training loop (REINFORCE / policy gradient), checkpointing, and plotting
* [pidog_rl/env.py](pidog_rl/env.py): Gym-style environment that applies action deltas to gait parameters, runs a rollout (sim or hardware), and computes rewards
* [pidog_rl/policy.py](pidog_rl/policy.py): Policy network that outputs a Gaussian distribution over action deltas
* [pidog_rl/pidog_hw.py](pidog_rl/pidog_hw.py): Hardware adapter around the SunFounder `pidog` API (optional)
* [pidog_rl/config.py](pidog_rl/config.py): Training, safety limits, and hardware configuration
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

For hardware support (the `pidog` library), you must install `pidog` from SunFounder first (it is not on PyPI), then install the extra:

```bash
git clone --depth=1 https://github.com/sunfounder/pidog.git
python3 -m pip install ./pidog --break-system-packages
```

Then install the extra:

```bash
python3 -m pip install ".[hardware]"
```

## Run training (simulation)

The recommended entry point is module execution so that relative imports resolve correctly:

```powershell
python -m pidog_rl.train
```

Training writes checkpoints and a plot into `output/`.

## Run inference (trained policy)

Use the inference script to apply a trained policy to the environment:

```powershell
python -m pidog_rl.infer --checkpoint output\26_04_04_1\checkpoint_final.pt --steps 50
```

To run on real hardware, install the `pidog` extra and pass `--use-hardware`:

```powershell
python -m pidog_rl.infer --checkpoint output\26_04_04_1\checkpoint_final.pt --steps 50 --use-hardware
```

## Raspberry Pi setup

On Raspberry Pi, avoid running `pip` as root when possible. Prefer a virtual environment:

```bash
cd ~/
git clone https://github.com/dmd0822/ml-pidog-gate_tuning.git
cd ml-pidog-gate_tuning
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install .
```

Hardware mode requires the `pidog` package and access to the PiDog hardware. Install it from SunFounder first:

```bash
git clone --depth=1 https://github.com/sunfounder/pidog.git
python3 -m pip install ./pidog --break-system-packages
```

Then enable the optional extra:

```bash
python3 -m pip install ".[hardware]"
```

> [!NOTE]
> PyTorch wheels vary by platform. If `pip install .` fails on Raspberry Pi due to torch, install a compatible CPU wheel for your Pi first, then rerun the install.

## Outputs

Training produces:

Each training run writes into a separate subfolder under `output/` named `yy_mm_dd_x`, where `x` is an auto-incrementing index for that day.

Inside each run folder:

* `checkpoint_ep*.pt`: Periodic checkpoints (episode, model weights, optimizer state, and reward history)
* `checkpoint_final.pt`: Final checkpoint
* `training_results_yy_mm_dd_x.png`: Reward, distance, and instability curves

## Hardware mode

Hardware mode is controlled by `TrainingConfig.hardware.use_hardware` in [pidog_rl/config.py](pidog_rl/config.py). When enabled, the environment uses [pidog_rl/pidog_hw.py](pidog_rl/pidog_hw.py) to call into the `pidog` library.

> [!WARNING]
> Enabling hardware mode can move the robot repeatedly during training. Keep the run duration short, ensure the robot is in a safe area, and have a quick way to cut power.

## How the reward works

Each environment step:

* Updates gait parameters using scaled, clipped action deltas (bounded by safety limits)
* Runs a short rollout and reads distance and IMU
* Computes reward as forward distance minus an instability penalty (sum of absolute IMU channels)

You can tune trade-offs through `RewardWeights` in [pidog_rl/config.py](pidog_rl/config.py).
