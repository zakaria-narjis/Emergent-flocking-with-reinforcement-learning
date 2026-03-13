"""
train.py — Entry point for training the DDQN flocking agent.

Usage:
    python train.py
    python train.py --override-config config/experiments/fast_debug.yaml --run-name debug
    python train.py --seed 123 --run-name repro_run
"""

import argparse
import copy
import os
import time
from datetime import datetime

import pfrl
import yaml

from flocking.agent import build_agent
from flocking.environment import BoidEnv


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def make_env_params(cfg: dict) -> dict:
    """Merge simulation and reward config sections into a single flat dict for AgentPy."""
    params = {}
    params.update(cfg["simulation"])
    params.update(cfg["reward"])
    return params


def main():
    parser = argparse.ArgumentParser(description="Train DDQN flocking agent")
    parser.add_argument(
        "--config", default="config/default.yaml", help="Base config YAML"
    )
    parser.add_argument(
        "--override-config", default=None, help="Experiment YAML to merge on top"
    )
    parser.add_argument("--run-name", default=None, help="Name for this run (default: timestamp)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    args = parser.parse_args()

    # Load and merge configs
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.override_config:
        with open(args.override_config) as f:
            override = yaml.safe_load(f)
        cfg = deep_merge(cfg, override)

    if args.seed is not None:
        cfg["training"]["seed"] = args.seed
    if args.run_name is not None:
        cfg["output"]["run_name"] = args.run_name

    # Create run directory
    run_name = cfg["output"]["run_name"] or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg["output"]["experiments_dir"], run_name)
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Save exact config used for reproducibility
    config_snapshot_path = os.path.join(run_dir, "config_used.yaml")
    with open(config_snapshot_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"Run directory: {run_dir}")
    print(f"Config snapshot saved to: {config_snapshot_path}")

    # Build environment and agent
    env_params = make_env_params(cfg)
    env = BoidEnv(env_params)

    agent = build_agent(env, cfg)

    # Train
    train_cfg = cfg["training"]
    print(f"\nStarting training: {train_cfg['total_steps']:,} steps")
    start_time = time.time()

    pfrl.experiments.train_agent_with_evaluation(
        agent,
        env,
        steps=train_cfg["total_steps"],
        eval_n_steps=None,
        eval_n_episodes=train_cfg["eval_n_episodes"],
        eval_interval=train_cfg["eval_interval"],
        outdir=checkpoints_dir,
    )

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Checkpoints saved to: {checkpoints_dir}")


if __name__ == "__main__":
    main()
