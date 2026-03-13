import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def flatten_state(states: list) -> np.ndarray:
    """
    Flatten the agent state list into a 1D float32 array for the neural network.

    State list structure:
        [position(2D), velocity(2D), cohesion(2D), separation(2D), alignment(2D),
         n_neighbors(scalar), border_v(2D)]

    The border vector (states[-1]) is excluded from the observation.
    n_neighbors (states[-2]) is a scalar appended separately.
    Result shape: (11,)
    """
    vectors = np.concatenate(states[:-2]).flatten()   # pos + vel + cohesion + sep + align = 10
    n_neighbors = states[-2]                           # scalar
    return np.float32(np.append(vectors, n_neighbors))
