import numpy as np
import gymnasium as gym
from gymnasium import spaces

from flocking.simulation import BoidsModel
from flocking.utils import flatten_state


class BoidEnv(gym.Env):
    """
    Gymnasium environment wrapping the AgentPy boids simulation.

    Observation space (11D, float32):
        [position(2), velocity(2), cohesion(2), separation(2), alignment(2), n_neighbors(1)]

    Action space:
        Discrete(9) — indices into DISCRETE_ACTIONS (3x3 velocity adjustment grid)

    Args:
        params: Flat parameter dict (merged simulation + reward sections from config)
    """

    metadata = {"render_modes": []}

    def __init__(self, params: dict):
        super().__init__()
        self.params = params
        self.model = BoidsModel(params)

        size = params["size"]
        population = params["population"]

        # Observation: pos(2) + vel(2) + cohesion(2) + sep(2) + align(2) + n_neighbors(1) = 11
        low = np.array(
            [0.0, 0.0,           # position
             -1.0, -1.0,         # velocity (normalized)
             -size, -size,       # cohesion vector
             -size, -size,       # separation vector
             -size, -size,       # alignment vector
             0.0],               # n_neighbors
            dtype=np.float32,
        )
        high = np.array(
            [size, size,
             1.0, 1.0,
             size, size,
             size, size,
             size, size,
             float(population)],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(9)

        self._initialized = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        states = self.model.init(seed=seed)
        self._initialized = True
        obs = flatten_state(states)
        return obs, {}

    def step(self, action: int):
        assert self._initialized, "Call reset() before step()"
        next_states, done, reward = self.model.one_step(action)
        obs = flatten_state(next_states)
        return obs, reward, done, False, {}
