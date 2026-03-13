import agentpy as ap

from flocking.boids import Boid, AgentBoid


class BoidsModel(ap.Model):
    """
    AgentPy simulation model containing regular Boids and one RL-controlled AgentBoid.

    Expected parameters (flat dict, typically merged from config simulation + reward sections):
        size, ndim, population, agents_population, steps,
        inner_radius, outer_radius, border_distance,
        cohesion_strength, separation_strength, alignment_strength, border_strength,
        flocking_error_weight, grouping_error_weight, grouping_penalty_base
    """

    def setup(self):
        self.space = ap.Space(self, shape=[self.p.size] * self.p.ndim)

        # Regular boids
        self.boids = ap.AgentList(self, self.p.population, Boid)
        self.space.add_agents(self.boids, random=True)
        self.boids.setup_pos(self.space)

        # RL-controlled agent boids
        self.agent_boids = ap.AgentList(self, self.p.agents_population, AgentBoid)
        self.space.add_agents(self.agent_boids, random=True)
        self.agent_boids.setup_pos(self.space)

        self._step_count = 0

    def step(self):
        """Advance all regular boids by one tick (used during setup/init)."""
        self.boids.update_velocity()
        self.boids.update_position()

    def init(self, steps: int = None, seed: int = None):
        """
        Reset simulation and return the initial observation of the first agent boid.

        Args:
            steps: Episode length (overrides p.steps if given)
            seed:  Random seed

        Returns:
            Initial state list for the agent boid
        """
        params = dict(self.p)
        if steps is not None:
            params["steps"] = steps

        self._sim_steps = params.get("steps", self.p.steps)
        self._step_count = 0

        self.setup()
        initial_states = self.agent_boids[0].get_states()
        self.agent_boids[0].current_states = initial_states
        return initial_states

    def one_step(self, action: int, display: bool = False):
        """
        Advance the simulation by one tick with the given RL action.

        Returns:
            (next_states, done, reward)
        """
        # Move regular boids
        self.boids.update_velocity()
        self.boids.update_position()

        # Move RL agent and get reward
        reward = self.agent_boids[0].take_action(action)
        next_states = self.agent_boids[0].current_states

        self._step_count += 1
        done = self._step_count >= self._sim_steps

        return next_states, done, reward
