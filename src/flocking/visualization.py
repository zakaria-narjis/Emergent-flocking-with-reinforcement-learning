import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

from flocking.simulation import BoidsModel
from flocking.utils import flatten_state


def animation_plot_single(model: BoidsModel, ax: plt.Axes):
    """
    Draw one animation frame: scatter boids, highlight RL agent, show radii and border.
    """
    ax.cla()
    size = model.p.size

    # All boid positions
    boid_positions = np.array([model.space.positions[b] for b in model.boids])
    ax.scatter(boid_positions[:, 0], boid_positions[:, 1], s=10, color="black", zorder=3)

    # RL agent boid
    agent_boid = model.agent_boids[0]
    agent_pos = model.space.positions[agent_boid]
    ax.scatter([agent_pos[0]], [agent_pos[1]], s=50, color="blue", zorder=4, label="Agent")

    # Separation radius (inner)
    inner_circle = plt.Circle(
        agent_pos, model.p.inner_radius, color="red", fill=False, linewidth=1, linestyle="--"
    )
    # Cohesion/alignment radius (outer)
    outer_circle = plt.Circle(
        agent_pos, model.p.outer_radius, color="green", fill=False, linewidth=1, linestyle="--"
    )
    ax.add_patch(inner_circle)
    ax.add_patch(outer_circle)

    # Border avoidance zone
    bd = model.p.border_distance
    border_rect = patches.Rectangle(
        (bd, bd), size - 2 * bd, size - 2 * bd,
        linewidth=1, edgecolor="red", facecolor="none", linestyle=":"
    )
    ax.add_patch(border_rect)

    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect("equal")
    ax.set_title(f"Step {model._step_count}")


def animate(model: BoidsModel, agent, fig, ax, steps: int = None, seed: int = None, skip: int = 0):
    """
    Generate a FuncAnimation by running the trained agent for `steps` ticks.

    Args:
        model:  BoidsModel instance
        agent:  Trained PFRL DoubleDQN agent
        fig:    Matplotlib figure
        ax:     Matplotlib axes
        steps:  Number of simulation steps (defaults to model.p.steps)
        seed:   Random seed for reset
        skip:   Number of initial frames to skip before recording

    Returns:
        (animation, frames_list)
    """
    n_steps = steps or model.p.steps
    frames = []

    # Reset environment
    states = model.init(steps=n_steps, seed=seed)
    obs = flatten_state(states)

    def update(frame_idx):
        nonlocal obs
        with agent.eval_mode():
            action = agent.act(obs)
        next_states, done, reward = model.one_step(action)
        obs = flatten_state(next_states)

        if frame_idx >= skip:
            animation_plot_single(model, ax)
            frames.append(np.array(fig.canvas.renderer.tostring_rgb()))

    anim = FuncAnimation(fig, update, frames=n_steps, interval=50, repeat=False)
    return anim, frames


def animation_plot(params: dict, agent, steps: int = None, seed: int = None):
    """
    Convenience wrapper: creates a figure, runs animation, returns HTML5 video.

    Args:
        params: Flat simulation parameter dict
        agent:  Trained PFRL DoubleDQN agent
        steps:  Episode length override
        seed:   Random seed

    Returns:
        (FuncAnimation, BoidsModel)
    """
    model = BoidsModel(params)
    fig, ax = plt.subplots(figsize=(6, 6))

    anim, _ = animate(model, agent, fig, ax, steps=steps, seed=seed)
    return anim, model


def plot_training_curves(scores_path: str, save_path: str = None):
    """
    Plot training evaluation scores from the pfrl scores.txt file.

    Args:
        scores_path: Path to scores.txt (tab-separated: steps, mean_reward, ...)
        save_path:   If given, save figure to this path
    """
    import pandas as pd

    df = pd.read_csv(scores_path, sep="\t")
    # pfrl columns: "steps", "mean", "median", "stdev", "max", "min", "elapsed"
    steps_col = df.columns[0]
    mean_col = "mean" if "mean" in df.columns else df.columns[1]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df[steps_col], df[mean_col], label="Mean reward")
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Reward")
    ax.set_title("Training curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
    return fig
