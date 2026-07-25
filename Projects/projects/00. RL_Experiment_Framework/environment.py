import random
from typing import Any, Optional


class LineWorldEnv:
    """A one-dimensional reinforcement-learning environment.

    Actions:
        0: move left
        1: move right

    Rewards:
        -1 for reaching the left terminal state
        +1 for reaching the right terminal state
         0 otherwise
    """

    LEFT = 0
    RIGHT = 1

    def __init__(
        self,
        length: int,
        max_timesteps: int,
        seed: Optional[int] = None,
    ) -> None:
        self._validate_configuration(length, max_timesteps)

        self.length = length
        self.max_timesteps = max_timesteps
        self.actions = frozenset({self.LEFT, self.RIGHT})

        # The environment owns its random-number generator.
        self._rng = random.Random(seed)

        # Use an immutable reward structure so external code cannot modify it.
        self.rewards = (
            -1.0,
            *([0.0] * (self.length - 2)),
            1.0,
        )

        self.agent_position = 0
        self.timestep = 0
        self.terminated = False
        self.truncated = False

        self.reset()

    @staticmethod
    def _validate_configuration(
        length: int,
        max_timesteps: int,
    ) -> None:
        if not isinstance(length, int) or isinstance(length, bool):
            raise TypeError("length must be an integer.")

        if length < 3:
            raise ValueError(
                "length must be at least 3 so that the environment "
                "has two terminal states and one non-terminal state."
            )

        if (
            not isinstance(max_timesteps, int)
            or isinstance(max_timesteps, bool)
        ):
            raise TypeError("max_timesteps must be an integer.")

        if max_timesteps <= 0:
            raise ValueError("max_timesteps must be greater than zero.")

    def _get_info(self) -> dict[str, Any]:
        return {
            "length": self.length,
            "rewards": self.rewards,
            "agent_position": self.agent_position,
            "timestep": self.timestep,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }

    def reset(
        self,
        seed: Optional[int] = None,
    ) -> tuple[int, dict[str, Any]]:
        """Start a new episode.

        Supplying a seed reproduces the same initial random state.
        Omitting it continues the environment's existing random sequence.
        """

        if seed is not None:
            self._rng.seed(seed)

        self.terminated = False
        self.truncated = False
        self.timestep = 0

        # randrange excludes the upper endpoint, so terminal states
        # 0 and length - 1 cannot be selected.
        self.agent_position = self._rng.randrange(1, self.length - 1)

        return self.agent_position, self._get_info()

    def step(
        self,
        action: int,
    ) -> tuple[int, float, bool, bool, dict[str, Any]]:
        """Execute one action in the environment."""

        if self.terminated or self.truncated:
            raise RuntimeError(
                "The episode has ended. Call reset() before stepping again."
            )

        # Using type(action) prevents True and False from being accepted
        # as integer actions 1 and 0.
        if type(action) is not int or action not in self.actions:
            raise ValueError(
                f"Undefined action {action!r}. "
                f"Valid actions are {sorted(self.actions)}."
            )

        if action == self.LEFT:
            self.agent_position -= 1
        else:
            self.agent_position += 1

        self.timestep += 1

        # Natural termination occurs only at an endpoint.
        self.terminated = self.agent_position in {
            0,
            self.length - 1,
        }

        # A natural terminal transition takes priority over the time limit.
        self.truncated = (
            self.timestep >= self.max_timesteps
            and not self.terminated
        )

        reward = self.rewards[self.agent_position]

        return (
            self.agent_position,
            reward,
            self.terminated,
            self.truncated,
            self._get_info(),
        )

    def render(self) -> None:
        """Display a simple text representation of the environment."""

        cells = ["."] * self.length
        cells[0] = "L"
        cells[-1] = "G"

        if not self.terminated:
            cells[self.agent_position] = "A"

        print(" ".join(cells))