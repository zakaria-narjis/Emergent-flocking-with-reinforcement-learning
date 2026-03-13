import numpy as np
import agentpy as ap
from itertools import product

from flocking.utils import normalize

# 9 discrete actions: all combinations of {0,1,2} x {0,1,2}
# Mapped to velocity deltas by subtracting 1: (0→-1, 1→0, 2→+1)
DISCRETE_ACTIONS = list(product(range(3), repeat=2))


class Boid(ap.Agent):
    """Standard Reynolds boid following the four flocking rules."""

    def setup(self):
        self.velocity = normalize(
            np.array([np.random.uniform(-1, 1) for _ in range(self.p.ndim)])
        )

    def setup_pos(self, space):
        self.space = space
        self.neighbors = space.neighbors
        self.pos = space.positions[self]

    def update_velocity(self):
        pos = self.pos
        ndim = self.p.ndim

        # Neighbors within outer radius (cohesion + alignment)
        outer_neighbors, outer_positions = self.neighbors(
            self, distance=self.p.outer_radius
        )
        n_outer = len(outer_neighbors)

        # Neighbors within inner radius (separation)
        inner_neighbors, inner_positions = self.neighbors(
            self, distance=self.p.inner_radius
        )

        # Rule 1 – Cohesion: steer toward center of mass of outer neighbors
        if n_outer > 0:
            center_of_mass = np.mean(outer_positions, axis=0)
            v1 = (center_of_mass - pos) * self.p.cohesion_strength
        else:
            v1 = np.zeros(ndim)

        # Rule 2 – Separation: steer away from inner neighbors
        if len(inner_neighbors) > 0:
            v2 = (pos - np.mean(inner_positions, axis=0)) * self.p.separation_strength
        else:
            v2 = np.zeros(ndim)

        # Rule 3 – Alignment: match velocity of outer neighbors
        if n_outer > 0:
            avg_vel = np.mean([b.velocity for b in outer_neighbors], axis=0)
            v3 = (avg_vel - self.velocity) * self.p.alignment_strength
        else:
            v3 = np.zeros(ndim)

        # Rule 4 – Border avoidance
        v4 = np.zeros(ndim)
        for i in range(ndim):
            if pos[i] < self.p.border_distance:
                v4[i] += self.p.border_strength
            elif pos[i] > self.p.size - self.p.border_distance:
                v4[i] -= self.p.border_strength

        self.velocity = normalize(self.velocity + v1 + v2 + v3 + v4)

    def update_position(self):
        self.space.move_by(self, self.velocity)


class AgentBoid(ap.Agent):
    """RL-controlled boid. Learns to flock via DDQN."""

    def setup(self):
        self.velocity = normalize(
            np.array([np.random.uniform(-1, 1) for _ in range(self.p.ndim)])
        )
        self.possible_actions = DISCRETE_ACTIONS
        self.current_states = None

    def setup_pos(self, space):
        self.space = space
        self.neighbors = space.neighbors
        self.pos = space.positions[self]

    def get_vectors(self):
        """
        Compute flocking force vectors for the current state.

        Returns:
            (position, velocity, cohesion_v, separation_v, alignment_v, n_neighbors, border_v)
        """
        pos = self.pos
        ndim = self.p.ndim

        outer_neighbors, outer_positions = self.neighbors(
            self, distance=self.p.outer_radius
        )
        n_outer = len(outer_neighbors)

        inner_neighbors, inner_positions = self.neighbors(
            self, distance=self.p.inner_radius
        )

        # Cohesion
        if n_outer > 0:
            center_of_mass = np.mean(outer_positions, axis=0)
            cohesion_v = (center_of_mass - pos) * self.p.cohesion_strength
        else:
            cohesion_v = np.zeros(ndim)

        # Separation
        if len(inner_neighbors) > 0:
            separation_v = (pos - np.mean(inner_positions, axis=0)) * self.p.separation_strength
        else:
            separation_v = np.zeros(ndim)

        # Alignment
        if n_outer > 0:
            avg_vel = np.mean([b.velocity for b in outer_neighbors], axis=0)
            alignment_v = (avg_vel - self.velocity) * self.p.alignment_strength
        else:
            alignment_v = np.zeros(ndim)

        # Border avoidance
        border_v = np.zeros(ndim)
        for i in range(ndim):
            if pos[i] < self.p.border_distance:
                border_v[i] += self.p.border_strength
            elif pos[i] > self.p.size - self.p.border_distance:
                border_v[i] -= self.p.border_strength

        return pos, self.velocity, cohesion_v, separation_v, alignment_v, n_outer, border_v

    def get_states(self):
        """Return state list: [pos, vel, cohesion, separation, alignment, n_neighbors, border_v]."""
        pos, vel, cohesion_v, separation_v, alignment_v, n_neighbors, border_v = self.get_vectors()
        return [pos, vel, cohesion_v, separation_v, alignment_v, n_neighbors, border_v]

    def update_states(self):
        self.current_states = self.get_states()

    def get_perfect_ns_vectors(self):
        """
        Simulate one step of pure Reynolds rules from current state.
        Used as the ideal reference for reward computation.
        """
        pos, vel, cohesion_v, separation_v, alignment_v, n_neighbors, border_v = self.get_vectors()
        perfect_velocity = normalize(vel + cohesion_v + separation_v + alignment_v + border_v)
        perfect_next_pos = pos + perfect_velocity

        # Return the flocking vectors at current state (reference for reward)
        return [pos, vel, cohesion_v, separation_v, alignment_v, n_neighbors, border_v]

    def take_action(self, action: int):
        """
        Execute a discrete action, update position, and return reward.

        Args:
            action: int in [0, 8], index into DISCRETE_ACTIONS

        Returns:
            reward (float)
        """
        states = self.get_states()
        perfect_ns = self.get_perfect_ns_vectors()

        # Convert action index to velocity delta: (0,1,2) → (-1,0,+1)
        row, col = self.possible_actions[action]
        delta = np.array([row - 1, col - 1], dtype=float)
        self.velocity = normalize(self.velocity + delta)
        self.space.move_by(self, self.velocity)

        next_states = self.get_states()
        reward = self.get_reward(states, perfect_ns, next_states)
        self.current_states = next_states
        return reward

    def get_reward(self, states, next_perfect_states, next_states):
        """
        Dual-component reward:
        - Flocking error: MSE between ideal Reynolds vectors and actual next vectors
        - Grouping error: penalty for having no neighbors (isolation)
        """
        # Ideal flocking vectors from current state
        reynolds_s = np.concatenate([
            next_perfect_states[2],  # cohesion
            next_perfect_states[3],  # separation
            next_perfect_states[4],  # alignment
        ])
        # Actual vectors after taking action
        reynolds_ns = np.concatenate([
            next_states[2],
            next_states[3],
            next_states[4],
        ])

        flocking_error = -float(np.mean((reynolds_s - reynolds_ns) ** 2))

        n_neighbors = next_states[5]
        if n_neighbors == 0:
            center = np.array([self.p.size / 2] * self.p.ndim)
            dist_from_center = np.linalg.norm(next_states[0] - center)
            grouping_error = -(self.p.grouping_penalty_base + dist_from_center)
        else:
            grouping_error = 0.0

        reward = (
            flocking_error * self.p.flocking_error_weight
            + grouping_error * self.p.grouping_error_weight
        )
        return reward

    def update_position(self):
        self.space.move_by(self, self.velocity)
