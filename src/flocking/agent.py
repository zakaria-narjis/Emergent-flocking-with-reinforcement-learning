import numpy as np
import torch
import pfrl
from pfrl.agents import DoubleDQN
from pfrl.replay_buffers import PrioritizedReplayBuffer
from pfrl.policies import LinearDecayEpsilonGreedy

from flocking.models import QFunction


def build_agent(env, cfg: dict) -> DoubleDQN:
    """
    Construct a fully configured PFRL DoubleDQN agent from the config dict.

    Args:
        env: A BoidEnv (or any gym.Env) instance
        cfg: Full config dict (with keys: agent, exploration, training)

    Returns:
        Configured DoubleDQN agent (not yet trained)
    """
    agent_cfg = cfg["agent"]
    explore_cfg = cfg["exploration"]
    train_cfg = cfg["training"]

    q_func = QFunction()
    optimizer = torch.optim.Adam(q_func.parameters(), eps=agent_cfg["optimizer_eps"])

    explorer = LinearDecayEpsilonGreedy(
        start_epsilon=explore_cfg["start_epsilon"],
        end_epsilon=explore_cfg["end_epsilon"],
        decay_steps=explore_cfg["decay_steps"],
        random_action_func=env.action_space.sample,
    )

    replay_buffer = PrioritizedReplayBuffer(capacity=agent_cfg["replay_buffer_capacity"])

    phi = lambda x: x.astype(np.float32, copy=False)

    gpu = 0 if train_cfg.get("device", "cpu") == "cuda" else -1

    agent = DoubleDQN(
        q_func,
        optimizer,
        replay_buffer,
        gamma=agent_cfg["gamma"],
        explorer=explorer,
        replay_start_size=agent_cfg["replay_start_size"],
        update_interval=agent_cfg["update_interval"],
        target_update_interval=agent_cfg["target_update_interval"],
        phi=phi,
        gpu=gpu,
    )
    return agent


def load_agent(checkpoint_path: str, env, cfg: dict) -> DoubleDQN:
    """
    Load a trained agent from a checkpoint directory.

    Args:
        checkpoint_path: Path to the directory saved by pfrl (contains model.pt etc.)
        env: A BoidEnv instance
        cfg: Config dict

    Returns:
        DoubleDQN agent with loaded weights
    """
    agent = build_agent(env, cfg)
    agent.load(checkpoint_path)
    return agent
