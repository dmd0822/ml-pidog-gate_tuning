"""Phase 2 Analysis: Reward Shaping & Stability Diagnostics

This script loads checkpoints and produces plots that validate:
  1. Reward signal quality (stability vs distance trade-off is visible)
  2. Episode convergence (rewards plateau, variance drops)
  3. Gait parameter evolution (deltas are constrained, no collapse)
  4. Instability penalties work as intended (negative correlation with reward)

Usage:
  python scripts/phase2_analysis.py --checkpoint output/26_04_04_1/checkpoint_final.pt
  python scripts/phase2_analysis.py --checkpoint output/26_04_04_1/checkpoint_final.pt --plot-instability-margin
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from pidog_rl.config import TrainingConfig
from pidog_rl.env import PiDogGaitEnv
from pidog_rl.policy import PolicyNetwork
from pidog_rl.utils import set_seed


def load_checkpoint(checkpoint_path: Path) -> Tuple[dict, dict]:
    """Load a checkpoint and return history and metadata."""
    ckpt = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    return ckpt.get("history", {}), ckpt


def _moving_avg(values: list[float], window: int) -> list[float]:
    if not values:
        return values
    cumsum = np.cumsum(values)
    result = np.empty_like(cumsum)
    result[:window] = cumsum[:window] / np.arange(1, window + 1)
    result[window:] = (cumsum[window:] - cumsum[:-window]) / window
    return result.tolist()


def plot_reward_stability_tradeoff(history: dict, output_dir: Path) -> None:
    """Plot the reward vs instability relationship to validate reward shaping.
    
    A well-shaped reward function should show:
      - Negative correlation between reward and instability_raw
      - Scatter around the trend line (exploration variance is expected)
      - Gradual improvement over episodes (mean reward trends up)
    """
    rewards = history.get("rewards", [])
    instabilities_raw = history.get("instabilities", [])
    
    if not rewards or not instabilities_raw:
        print("  skipping reward-stability plot (missing data)")
        return
    
    episodes = range(1, len(rewards) + 1)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Scatter: raw reward vs raw instability
    axes[0].scatter(instabilities_raw, rewards, alpha=0.4, s=10)
    axes[0].set_ylabel("Total Reward")
    axes[0].set_xlabel("Instability (raw)")
    axes[0].set_title("Reward-Stability Trade-off (Scatter)")
    axes[0].grid(True, alpha=0.3)
    
    # Trend: compute correlation
    corr = float(np.corrcoef(rewards, instabilities_raw)[0, 1])
    axes[0].text(0.05, 0.95, f"Correlation: {corr:.3f}", 
                 transform=axes[0].transAxes, verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    # Time series: both signals smoothed
    axes[1].plot(episodes, _moving_avg(rewards, 20), linewidth=2, label="Reward (20-ep avg)", alpha=0.8)
    ax2 = axes[1].twinx()
    ax2.plot(episodes, _moving_avg(instabilities_raw, 20), linewidth=2, label="Instability (20-ep avg)", 
             alpha=0.8, color="orange")
    axes[1].set_ylabel("Reward", color="C0")
    ax2.set_ylabel("Instability", color="orange")
    axes[1].set_xlabel("Episode")
    axes[1].set_title("Reward & Instability Over Training (Smoothed)")
    axes[1].grid(True, alpha=0.3)
    
    fig.tight_layout()
    plot_name = "phase2_reward_stability.png"
    fig.savefig(output_dir / plot_name, dpi=150)
    plt.close(fig)
    print(f"  saved phase2_reward_stability.png (correlation: {corr:.3f})")


def plot_convergence_diagnostics(history: dict, output_dir: Path) -> None:
    """Plot convergence signals to confirm training stability.
    
    Expects:
      - Reward variance decreases over time (less exploration noise)
      - Distance & instability show learning progression (not random)
      - No sudden spikes or collapses
    """
    rewards = history.get("rewards", [])
    distances = history.get("distances", [])
    instabilities = history.get("instabilities", [])
    
    if not rewards or not distances or not instabilities:
        print("  skipping convergence plot (missing data)")
        return
    
    episodes = range(1, len(rewards) + 1)
    window = min(20, len(rewards) // 4)
    
    # Compute rolling variance (explores vs exploitation)
    rolling_var = []
    for i in range(len(rewards)):
        start = max(0, i - window)
        rolling_var.append(float(np.var(rewards[start:i+1])))
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    # Reward convergence with std dev band
    avg_rewards = _moving_avg(rewards, window)
    std_rewards = [
        float(np.std(rewards[max(0, i-window):i+1])) 
        for i in range(len(rewards))
    ]
    axes[0].plot(episodes, avg_rewards, linewidth=2, label="mean", color="C0")
    axes[0].fill_between(episodes, 
                         np.array(avg_rewards) - np.array(std_rewards),
                         np.array(avg_rewards) + np.array(std_rewards),
                         alpha=0.3)
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Convergence Diagnostics")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Distance trend
    axes[1].plot(episodes, distances, alpha=0.4, linewidth=0.5)
    axes[1].plot(episodes, _moving_avg(distances, window), linewidth=2, label=f"{window}-ep avg")
    axes[1].set_ylabel("Distance")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Instability trend (should decline if reward shaping is working)
    axes[2].plot(episodes, instabilities, alpha=0.4, linewidth=0.5)
    axes[2].plot(episodes, _moving_avg(instabilities, window), linewidth=2, label=f"{window}-ep avg")
    axes[2].set_ylabel("Instability")
    axes[2].set_xlabel("Episode")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    fig.tight_layout()
    plot_name = "phase2_convergence.png"
    fig.savefig(output_dir / plot_name, dpi=150)
    plt.close(fig)
    print(f"  saved phase2_convergence.png")


def plot_instability_margin(history: dict, config: TrainingConfig, output_dir: Path) -> None:
    """Verify that instability clipping margin is sufficient (not hitting ceiling).
    
    If many episodes hit instability_clip, the penalty is not differentiable for those steps,
    which means the signal is lost. This plot shows the distribution of instability values
    relative to the clipping threshold.
    """
    instabilities_raw = history.get("instabilities", [])
    if not instabilities_raw:
        print("  skipping instability margin plot (missing data)")
        return
    
    clip_threshold = config.reward_weights.instability_clip
    clipped_pct = float(np.sum(np.array(instabilities_raw) >= clip_threshold) / len(instabilities_raw) * 100)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.hist(instabilities_raw, bins=50, alpha=0.7, edgecolor="black")
    ax.axvline(clip_threshold, color="red", linestyle="--", linewidth=2, label=f"Clip threshold ({clip_threshold})")
    ax.set_xlabel("Instability (raw)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Instability Distribution (Clipped: {clipped_pct:.1f}%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    plot_name = "phase2_instability_margin.png"
    fig.savefig(output_dir / plot_name, dpi=150)
    plt.close(fig)
    print(f"  saved phase2_instability_margin.png (clipped: {clipped_pct:.1f}%)")


def print_summary_stats(history: dict, config: TrainingConfig) -> None:
    """Print human-readable summary statistics."""
    rewards = history.get("rewards", [])
    distances = history.get("distances", [])
    instabilities = history.get("instabilities", [])
    
    if not rewards:
        print("  (no data)")
        return
    
    print("\n--- Phase 2 Summary Statistics ---")
    print(f"Episodes: {len(rewards)}")
    print(f"\nReward:")
    print(f"  Mean: {float(np.mean(rewards)):.4f}")
    print(f"  Std:  {float(np.std(rewards)):.4f}")
    print(f"  Min:  {float(np.min(rewards)):.4f}, Max: {float(np.max(rewards)):.4f}")
    print(f"  Last 10 eps avg: {float(np.mean(rewards[-10:])):.4f}")
    
    if distances:
        print(f"\nDistance:")
        print(f"  Mean: {float(np.mean(distances)):.4f}")
        print(f"  Last 10 eps avg: {float(np.mean(distances[-10:])):.4f}")
    
    if instabilities:
        print(f"\nInstability (raw):")
        print(f"  Mean: {float(np.mean(instabilities)):.4f}")
        print(f"  Last 10 eps avg: {float(np.mean(instabilities[-10:])):.4f}")
        clip_threshold = config.reward_weights.instability_clip
        clipped_pct = float(np.sum(np.array(instabilities) >= clip_threshold) / len(instabilities) * 100)
        print(f"  Clipped (>= {clip_threshold}): {clipped_pct:.1f}%")
    
    # Correlation check
    if len(rewards) > 1 and len(instabilities) > 1:
        corr = float(np.corrcoef(rewards, instabilities)[0, 1])
        print(f"\nCorrelation (Reward vs Instability): {corr:.4f}")
        if corr < -0.3:
            print("  ✓ Good negative correlation (penalty is working)")
        elif corr > 0.3:
            print("  ✗ Positive correlation (instability may not be penalized)")
        else:
            print("  ~ Weak correlation (check reward weight balance)")
    
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 reward shaping analysis")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: checkpoint dir)")
    parser.add_argument("--plot-instability-margin", action="store_true", help="Also plot instability margin analysis")
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading checkpoint: {checkpoint_path}")
    history, ckpt = load_checkpoint(checkpoint_path)
    
    config = TrainingConfig()
    print_summary_stats(history, config)
    
    print("Generating plots:")
    plot_reward_stability_tradeoff(history, output_dir)
    plot_convergence_diagnostics(history, output_dir)
    if args.plot_instability_margin:
        plot_instability_margin(history, config, output_dir)
    
    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
