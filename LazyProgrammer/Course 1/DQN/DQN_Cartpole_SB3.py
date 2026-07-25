#!/usr/bin/env python3
"""CartPole DQN using Stable-Baselines3 with explicit replay and epsilon-greedy.

Unlike a minimal SB3 example, this script deliberately exposes the two parts
that are normally hidden inside SB3's DQN implementation:

1. The replay buffer is passed explicitly via ``replay_buffer_class``.
2. The training action-selection rule is implemented explicitly in
   ``ExplicitEpsilonGreedyDQN._sample_action``:

       - before ``learning_starts``: choose a uniformly random action;
       - afterward: choose a random action with probability epsilon;
       - otherwise: choose the greedy argmax-Q action.

Install:
    python -m pip install "stable-baselines3[extra]>=2.3,<3" \
        "gymnasium[classic-control]>=0.29" matplotlib

Run:
    python DQN_Cartpole_SB3_explicit.py

Useful options:
    python DQN_Cartpole_SB3_explicit.py --timesteps 150000 --seed 7
    python DQN_Cartpole_SB3_explicit.py --record-video --show-plot
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import gymnasium as gym
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn
    from stable_baselines3 import DQN
    from stable_baselines3.common.buffers import ReplayBuffer
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError as exc:
    raise SystemExit(
        "Missing dependencies. Install them with:\n"
        '  python -m pip install "stable-baselines3[extra]>=2.3,<3" '
        '"gymnasium[classic-control]>=0.29" matplotlib\n'
        f"Original import error: {exc}"
    ) from exc


@dataclass(frozen=True)
class Config:
    env_id: str = "CartPole-v1"
    total_timesteps: int = 100_000

    # Conservative CartPole settings: frequent, small updates are easier to
    # inspect and are less seed-sensitive than large burst updates.
    learning_rate: float = 1.0e-3
    buffer_size: int = 50_000
    learning_starts: int = 1_000
    batch_size: int = 64
    gamma: float = 0.99
    tau: float = 1.0
    train_freq: int = 1
    gradient_steps: int = 1
    target_update_interval: int = 500
    max_grad_norm: float = 10.0

    # Explicit epsilon-greedy schedule.
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.02
    exploration_fraction: float = 0.30

    hidden_layers: tuple[int, ...] = (128,)
    seed: int = 42

    diagnostics_freq: int = 1_000
    eval_freq: int = 5_000
    eval_episodes: int = 20
    final_eval_episodes: int = 100
    solved_reward: float = 475.0
    output_dir: Path = Path("dqn_cartpole_sb3_explicit_output")


class ExplicitReplayBuffer(ReplayBuffer):
    """Named replay buffer used directly by the DQN model."""

    def diagnostics(self) -> dict[str, int | float | bool]:
        current_size = int(self.size())
        return {
            "size": current_size,
            "capacity": int(self.buffer_size),
            "fill_fraction": current_size / float(self.buffer_size),
            "position": int(self.pos),
            "full": bool(self.full),
        }


class ExplicitEpsilonGreedyDQN(DQN):
    """SB3 DQN with visible training-time epsilon-greedy action selection.

    SB3 normally performs this logic internally. Overriding ``_sample_action``
    makes the exact behavior visible and testable while retaining SB3's Q-network,
    target network, optimizer, replay sampling, TD target, and logging.
    """

    warmup_random_actions: int
    epsilon_random_actions: int
    greedy_actions: int

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.warmup_random_actions = 0
        self.epsilon_random_actions = 0
        self.greedy_actions = 0

    def _on_step(self) -> None:
        """Update the target network and an explicit linear epsilon schedule."""

        # Keep SB3's target-network update and internal bookkeeping.
        super()._on_step()

        # Explicit linear annealing:
        # epsilon(t) = eps_start + progress * (eps_end - eps_start)
        # until exploration_fraction * total_timesteps, then eps_end.
        decay_steps = max(1, int(self.exploration_fraction * self._total_timesteps))
        progress = min(float(self.num_timesteps) / float(decay_steps), 1.0)
        self.exploration_rate = (
            self.exploration_initial_eps
            + progress
            * (self.exploration_final_eps - self.exploration_initial_eps)
        )
        self.logger.record("rollout/exploration_rate", self.exploration_rate)

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Any = None,
        n_envs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Choose actions using an explicit warm-up and epsilon-greedy rule."""

        if action_noise is not None:
            raise ValueError("DQN uses epsilon-greedy, not continuous action noise.")

        # Phase 1: fill the replay buffer with uniformly random transitions.
        if self.num_timesteps < learning_starts:
            actions = np.asarray(
                [self.action_space.sample() for _ in range(n_envs)],
                dtype=np.int64,
            )
            self.warmup_random_actions += n_envs
            return actions, actions.copy()

        # Phase 2: explicit epsilon-greedy exploration.
        if np.random.random() < float(self.exploration_rate):
            actions = np.asarray(
                [self.action_space.sample() for _ in range(n_envs)],
                dtype=np.int64,
            )
            self.epsilon_random_actions += n_envs
            return actions, actions.copy()

        # Greedy branch: argmax_a Q(s, a). DQNPolicy.predict is greedy.
        assert self._last_obs is not None
        actions, _ = self.policy.predict(
            self._last_obs,
            deterministic=True,
        )
        actions = np.asarray(actions, dtype=np.int64).reshape(n_envs)
        self.greedy_actions += n_envs
        return actions, actions.copy()


class TrainingDiagnosticsCallback(BaseCallback):
    """Record returns and verify that learning/exploration are active."""

    def __init__(self, print_freq: int) -> None:
        super().__init__(verbose=0)
        if print_freq <= 0:
            raise ValueError("print_freq must be positive")
        self.print_freq = print_freq
        self.episode_steps: list[int] = []
        self.episode_returns: list[float] = []
        self.epsilon_steps: list[int] = []
        self.epsilon_values: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode is not None:
                self.episode_steps.append(int(self.num_timesteps))
                self.episode_returns.append(float(episode["r"]))

        if self.num_timesteps % self.print_freq != 0:
            return True

        model = self.model
        if not isinstance(model, ExplicitEpsilonGreedyDQN):
            raise TypeError("Expected ExplicitEpsilonGreedyDQN")
        if not isinstance(model.replay_buffer, ExplicitReplayBuffer):
            raise TypeError("Expected ExplicitReplayBuffer")

        epsilon = float(model.exploration_rate)
        self.epsilon_steps.append(int(self.num_timesteps))
        self.epsilon_values.append(epsilon)

        buffer_stats = model.replay_buffer.diagnostics()
        recent_returns = self.episode_returns[-20:]
        mean_return = float(np.mean(recent_returns)) if recent_returns else float("nan")
        updates = int(getattr(model, "_n_updates", 0))

        print(
            f"step={self.num_timesteps:6d} | "
            f"return20={mean_return:7.2f} | "
            f"epsilon={epsilon:5.3f} | "
            f"buffer={buffer_stats['size']:5d}/{buffer_stats['capacity']} | "
            f"updates={updates:6d} | "
            f"actions(warmup/random/greedy)="
            f"{model.warmup_random_actions}/"
            f"{model.epsilon_random_actions}/"
            f"{model.greedy_actions}"
        )
        return True


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(config: Config, train_env: Any) -> ExplicitEpsilonGreedyDQN:
    policy_kwargs = {
        "net_arch": list(config.hidden_layers),
        "activation_fn": nn.ReLU,
    }

    model = ExplicitEpsilonGreedyDQN(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=config.learning_rate,
        buffer_size=config.buffer_size,
        learning_starts=config.learning_starts,
        batch_size=config.batch_size,
        tau=config.tau,
        gamma=config.gamma,
        train_freq=(config.train_freq, "step"),
        gradient_steps=config.gradient_steps,
        replay_buffer_class=ExplicitReplayBuffer,
        replay_buffer_kwargs={"handle_timeout_termination": True},
        optimize_memory_usage=False,
        target_update_interval=config.target_update_interval,
        exploration_fraction=config.exploration_fraction,
        exploration_initial_eps=config.exploration_initial_eps,
        exploration_final_eps=config.exploration_final_eps,
        max_grad_norm=config.max_grad_norm,
        policy_kwargs=policy_kwargs,
        seed=config.seed,
        device="auto",
        verbose=1,
    )

    if not isinstance(model.replay_buffer, ExplicitReplayBuffer):
        raise RuntimeError("ExplicitReplayBuffer was not constructed.")

    return model


def plot_training(
    config: Config,
    diagnostics: TrainingDiagnosticsCallback,
    show_plot: bool,
) -> None:
    plot_dir = config.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    if diagnostics.episode_returns:
        returns = np.asarray(diagnostics.episode_returns, dtype=np.float64)
        window = min(20, len(returns))
        moving_average = np.convolve(
            returns,
            np.ones(window, dtype=np.float64) / window,
            mode="valid",
        )

        plt.figure(figsize=(9, 5))
        plt.plot(
            diagnostics.episode_steps,
            diagnostics.episode_returns,
            alpha=0.25,
            label="episode return",
        )
        plt.plot(
            diagnostics.episode_steps[window - 1 :],
            moving_average,
            label=f"{window}-episode moving average",
        )
        plt.axhline(
            config.solved_reward,
            linestyle="--",
            label=f"target ({config.solved_reward:.0f})",
        )
        plt.xlabel("Environment timestep")
        plt.ylabel("Episode return")
        plt.title("CartPole DQN training return")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "training_returns.png", dpi=160)
        if show_plot:
            plt.show()
        plt.close()

    if diagnostics.epsilon_values:
        plt.figure(figsize=(9, 5))
        plt.plot(diagnostics.epsilon_steps, diagnostics.epsilon_values)
        plt.xlabel("Environment timestep")
        plt.ylabel("Epsilon")
        plt.title("Explicit epsilon-greedy schedule")
        plt.tight_layout()
        plt.savefig(plot_dir / "epsilon_schedule.png", dpi=160)
        if show_plot:
            plt.show()
        plt.close()


def record_video(model: DQN, config: Config) -> None:
    video_dir = config.output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(config.env_id, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(video_dir),
        episode_trigger=lambda episode_id: episode_id == 0,
        name_prefix="dqn-cartpole",
    )

    observation, _ = env.reset(seed=config.seed + 30_000)
    terminated = False
    truncated = False
    while not (terminated or truncated):
        # Evaluation must be greedy: epsilon is intentionally disabled here.
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, _ = env.step(
            int(np.asarray(action).item())
        )
    env.close()
    print(f"Video saved under: {video_dir.resolve()}")


def train(config: Config, show_plot: bool, record: bool) -> None:
    seed_everything(config.seed)

    model_dir = config.output_dir / "models"
    replay_dir = config.output_dir / "replay_buffer"
    eval_dir = config.output_dir / "evaluations"
    monitor_dir = config.output_dir / "monitor"
    best_dir = config.output_dir / "best_model"
    for directory in (model_dir, replay_dir, eval_dir, monitor_dir, best_dir):
        directory.mkdir(parents=True, exist_ok=True)

    train_env = make_vec_env(
        config.env_id,
        n_envs=1,
        seed=config.seed,
        monitor_dir=str(monitor_dir),
    )
    eval_env = make_vec_env(
        config.env_id,
        n_envs=1,
        seed=config.seed + 10_000,
    )

    model = build_model(config, train_env)
    diagnostics = TrainingDiagnosticsCallback(config.diagnostics_freq)
    evaluation = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(eval_dir),
        eval_freq=config.eval_freq,
        n_eval_episodes=config.eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )

    print("\nTraining behavior")
    print(f"  warm-up random actions: first {config.learning_starts} steps")
    print(
        "  epsilon schedule: "
        f"{config.exploration_initial_eps:.2f} -> "
        f"{config.exploration_final_eps:.2f} during the first "
        f"{config.exploration_fraction:.0%} of training"
    )
    print(f"  explicit replay buffer: {type(model.replay_buffer).__name__}")
    print(f"  replay capacity: {config.buffer_size}\n")

    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=CallbackList([diagnostics, evaluation]),
            log_interval=10,
            progress_bar=False,
        )

        if int(getattr(model, "_n_updates", 0)) <= 0:
            raise RuntimeError(
                "No gradient updates occurred. Increase --timesteps above "
                f"learning_starts={config.learning_starts}."
            )

        final_model_base = model_dir / "dqn_cartpole_final"
        replay_path = replay_dir / "dqn_cartpole_replay_buffer.pkl"
        model.save(final_model_base)
        model.save_replay_buffer(replay_path)

        best_model_path = best_dir / "best_model.zip"
        if best_model_path.exists():
            evaluation_model = ExplicitEpsilonGreedyDQN.load(
                best_model_path,
                env=eval_env,
            )
            selected_path = best_model_path
        else:
            evaluation_model = model
            selected_path = final_model_base.with_suffix(".zip")

        mean_reward, std_reward = evaluate_policy(
            evaluation_model,
            eval_env,
            n_eval_episodes=config.final_eval_episodes,
            deterministic=True,
            warn=True,
        )

        print("\nTraining complete")
        print(f"  selected model: {selected_path.resolve()}")
        print(f"  replay buffer: {replay_path.resolve()}")
        print(f"  gradient updates: {int(model._n_updates)}")
        print(
            "  action counts: "
            f"warm-up={model.warmup_random_actions}, "
            f"epsilon-random={model.epsilon_random_actions}, "
            f"greedy={model.greedy_actions}"
        )
        print(
            f"  deterministic evaluation ({config.final_eval_episodes} episodes): "
            f"{mean_reward:.2f} +/- {std_reward:.2f}"
        )

        if mean_reward >= config.solved_reward:
            print("  status: solved")
        else:
            print(
                "  status: below target; run with --timesteps 150000 or try "
                "another --seed and inspect the printed return20 curve."
            )

        plot_training(config, diagnostics, show_plot)
        if record:
            record_video(evaluation_model, config)
    finally:
        train_env.close()
        eval_env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stable-Baselines3 DQN for CartPole with explicit replay buffer "
            "and explicit epsilon-greedy training actions."
        )
    )
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dqn_cartpole_sb3_explicit_output"),
    )
    parser.add_argument("--show-plot", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.timesteps <= 1_000:
        raise SystemExit("--timesteps must be greater than 1000")

    config = Config(
        total_timesteps=args.timesteps,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    train(config, show_plot=args.show_plot, record=args.record_video)


if __name__ == "__main__":
    main()
