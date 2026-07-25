import pytest

from environment import LineWorldEnv


def test_reaching_right_terminal():
    env = LineWorldEnv(length=5, max_timesteps=10, seed=42)
    env.agent_position = 2

    state, reward, terminated, truncated, _ = env.step(
        LineWorldEnv.RIGHT
    )

    assert state == 3
    assert reward == 0.0
    assert terminated is False
    assert truncated is False

    state, reward, terminated, truncated, _ = env.step(
        LineWorldEnv.RIGHT
    )

    assert state == 4
    assert reward == 1.0
    assert terminated is True
    assert truncated is False


def test_reaching_left_terminal():
    env = LineWorldEnv(length=5, max_timesteps=10, seed=42)
    env.agent_position = 2

    env.step(LineWorldEnv.LEFT)

    state, reward, terminated, truncated, _ = env.step(
        LineWorldEnv.LEFT
    )

    assert state == 0
    assert reward == -1.0
    assert terminated is True
    assert truncated is False


def test_episode_is_truncated_at_time_limit():
    env = LineWorldEnv(length=7, max_timesteps=2, seed=42)
    env.agent_position = 3

    env.step(LineWorldEnv.RIGHT)

    state, reward, terminated, truncated, info = env.step(
        LineWorldEnv.LEFT
    )

    assert state == 3
    assert reward == 0.0
    assert terminated is False
    assert truncated is True
    assert info["timestep"] == 2


def test_cannot_step_after_episode_ends():
    env = LineWorldEnv(length=3, max_timesteps=10, seed=42)

    env.step(LineWorldEnv.RIGHT)

    with pytest.raises(RuntimeError):
        env.step(LineWorldEnv.LEFT)


def test_invalid_action_is_rejected():
    env = LineWorldEnv(length=5, max_timesteps=10, seed=42)

    with pytest.raises(ValueError):
        env.step(5)


def test_reset_never_starts_at_terminal_state():
    env = LineWorldEnv(length=7, max_timesteps=10, seed=42)

    for _ in range(100):
        state, info = env.reset()

        assert 1 <= state <= env.length - 2
        assert info["terminated"] is False
        assert info["truncated"] is False
        assert info["timestep"] == 0


def test_reset_with_same_seed_is_reproducible():
    env = LineWorldEnv(length=10, max_timesteps=10)

    state_1, _ = env.reset(seed=123)
    state_2, _ = env.reset(seed=123)

    assert state_1 == state_2


def main():
    """Run the tests when this file is executed directly."""

    test_reaching_right_terminal()
    print("Passed: reaching right terminal")

    test_reaching_left_terminal()
    print("Passed: reaching left terminal")

    test_episode_is_truncated_at_time_limit()
    print("Passed: time-limit truncation")

    test_cannot_step_after_episode_ends()
    print("Passed: post-episode step rejection")

    test_invalid_action_is_rejected()
    print("Passed: invalid action rejection")

    test_reset_never_starts_at_terminal_state()
    print("Passed: valid reset positions")

    test_reset_with_same_seed_is_reproducible()
    print("Passed: seed reproducibility")

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()