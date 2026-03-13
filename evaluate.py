"""
evaluate.py — Evaluate a trained DDQN flocking agent.

Usage:
    python evaluate.py --checkpoint experiments/runs/debug/checkpoints
    python evaluate.py --checkpoint experiments/runs/debug/checkpoints --animate --episodes 3
"""

import argparse
import os

import numpy as np
import yaml

from flocking.agent import load_agent
from flocking.environment import BoidEnv
from flocking.visualization import animation_plot, plot_training_curves


def make_env_params(cfg: dict) -> dict:
    params = {}
    params.update(cfg["simulation"])
    params.update(cfg["reward"])
    return params


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained DDQN flocking agent")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory")
    parser.add_argument(
        "--config",
        default=None,
        help="Config YAML (default: config_used.yaml in the run directory)",
    )
    parser.add_argument("--episodes", type=int, default=10, help="Number of eval episodes")
    parser.add_argument("--animate", action="store_true", help="Render and save animation GIF")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to save outputs (default: run directory)",
    )
    args = parser.parse_args()

    # Resolve config path: prefer explicit flag, else look for config_used.yaml in run dir
    if args.config:
        config_path = args.config
    else:
        run_dir = os.path.dirname(args.checkpoint.rstrip("/\\"))
        config_path = os.path.join(run_dir, "config_used.yaml")
        if not os.path.exists(config_path):
            config_path = "config/default.yaml"

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    output_dir = args.output_dir or os.path.dirname(args.checkpoint.rstrip("/\\"))
    os.makedirs(output_dir, exist_ok=True)

    # Build environment and load agent
    env_params = make_env_params(cfg)
    env = BoidEnv(env_params)
    agent = load_agent(args.checkpoint, env, cfg)

    # Run evaluation episodes
    print(f"Running {args.episodes} evaluation episodes...")
    rewards = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            with agent.eval_mode():
                action = agent.act(obs)
            obs, reward, done, _, _ = env.step(action)
            ep_reward += reward
        rewards.append(ep_reward)
        print(f"  Episode {ep + 1:3d}: reward = {ep_reward:.2f}")

    print(f"\nMean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Min: {np.min(rewards):.2f}  Max: {np.max(rewards):.2f}")

    # Optional: render animation
    if args.animate:
        print("\nRendering animation...")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        anim, _ = animation_plot(env_params, agent, seed=cfg["training"].get("seed"))
        gif_path = os.path.join(output_dir, "eval_animation.gif")
        anim.save(gif_path, writer="pillow", fps=20)
        print(f"Animation saved to: {gif_path}")

    # Optional: plot training curves
    scores_path = os.path.join(args.checkpoint, "scores.txt")
    if os.path.exists(scores_path):
        fig = plot_training_curves(
            scores_path,
            save_path=os.path.join(output_dir, "training_curves.svg"),
        )
        plt.close(fig)
        print(f"Training curves saved to: {os.path.join(output_dir, 'training_curves.svg')}")


if __name__ == "__main__":
    main()
