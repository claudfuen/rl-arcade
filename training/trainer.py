"""
Training loop for PPO agents.

This module orchestrates the full training process:
1. Collect experience from environments
2. Update the policy using PPO
3. Log metrics and save checkpoints
4. Repeat until convergence

The trainer supports real-time visualization of both the game
and training metrics.
"""

import os
import time
import numpy as np
from typing import Dict, List, Optional
from collections import deque
from torch.utils.tensorboard import SummaryWriter

from agents import PPO
from environments import make_vec_env, make_env
from config import PPOConfig, TrainingConfig, EnvConfig
from .callbacks import BaseCallback

# Optional session import (for backwards compatibility)
try:
    from sessions import Session
except ImportError:
    Session = None


class Trainer:
    """
    Training loop manager for PPO with real-time visualization.

    Handles:
    - Environment setup
    - Training loop execution
    - Metric tracking and logging
    - Checkpoint management
    - Real-time game rendering
    - Live training dashboard
    """

    def __init__(
        self,
        env_config: EnvConfig = None,
        ppo_config: PPOConfig = None,
        training_config: TrainingConfig = None,
        callbacks: List[BaseCallback] = None,
        show_dashboard: bool = False,
        session: "Session" = None,
    ):
        """
        Initialize trainer.

        Args:
            env_config: Environment configuration
            ppo_config: PPO algorithm configuration
            training_config: Training loop configuration
            callbacks: List of callback objects
            show_dashboard: Whether to show live matplotlib dashboard
            session: Optional Session for tracking and resume support
        """
        self.env_config = env_config or EnvConfig()
        self.ppo_config = ppo_config or PPOConfig()
        self.training_config = training_config or TrainingConfig()
        self.callbacks = callbacks or []
        self.show_dashboard = show_dashboard
        self.session = session

        # If session provided, override directories to use session paths
        if session is not None:
            self.training_config.checkpoint_dir = session.checkpoint_dir
            self.training_config.log_dir = session.log_dir

        # Create directories
        os.makedirs(self.training_config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.training_config.log_dir, exist_ok=True)

        # Create vectorized environment for training
        print(f"Creating {self.env_config.n_envs} parallel environments...")
        self.env = make_vec_env(
            env_name=self.env_config.env_name,
            n_envs=self.env_config.n_envs,
            frame_skip=self.env_config.frame_skip,
            frame_stack=self.env_config.frame_stack,
            frame_height=self.env_config.frame_height,
            frame_width=self.env_config.frame_width,
        )

        # Create render environment if visualization enabled
        self.render_env = None
        self.demo_env = None
        if self.training_config.render_training:
            print("Creating render environment...")
            self.render_env = self._create_render_env()
        # Create demo env if demos enabled OR if dashboard is shown (user can enable via slider)
        if self.training_config.demo_every > 0 or show_dashboard:
            if self.training_config.demo_every > 0:
                print(f"Demo mode: will play a game every {self.training_config.demo_every} updates")
            self.demo_env = self._create_demo_env()

        # Create PPO agent
        print(f"Creating PPO agent on {self.training_config.device}...")
        self.agent = PPO(
            env=self.env,
            config=self.ppo_config,
            device=self.training_config.device,
        )

        # Setup TensorBoard
        self.writer = SummaryWriter(self.training_config.log_dir)

        # Episode tracking
        self.episode_rewards = deque(maxlen=100)  # Rolling window
        self.episode_lengths = deque(maxlen=100)
        self.current_episode_rewards = np.zeros(self.env_config.n_envs)
        self.current_episode_lengths = np.zeros(self.env_config.n_envs)
        self.total_episodes = 0

        # Training metrics for dashboard
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []

        # Render frame counter
        self.render_frame_count = 0

        # Dashboard
        self.dashboard = None
        if show_dashboard:
            from visualization import TrainingDashboard
            self.dashboard = TrainingDashboard(
                demo_every=self.training_config.demo_every
            )

        # Timing
        self.start_time = None

    @classmethod
    def from_session(
        cls,
        session: "Session",
        show_dashboard: bool = False,
        callbacks: List[BaseCallback] = None,
    ) -> "Trainer":
        """
        Create a Trainer from a saved session, ready to resume training.

        Args:
            session: Session to resume
            show_dashboard: Whether to show live training dashboard
            callbacks: Additional callbacks (SessionCheckpointCallback added automatically)

        Returns:
            Trainer instance with agent state restored from session checkpoint
        """
        from .callbacks import SessionCheckpointCallback

        # Get configs from session
        env_config = session.get_env_config()
        ppo_config = session.get_ppo_config()
        training_config = session.get_training_config()

        # Setup callbacks - always include session checkpoint callback
        callbacks = callbacks or []
        callbacks.insert(0, SessionCheckpointCallback(session))

        # Create trainer
        trainer = cls(
            env_config=env_config,
            ppo_config=ppo_config,
            training_config=training_config,
            callbacks=callbacks,
            show_dashboard=show_dashboard,
            session=session,
        )

        # Load checkpoint if exists
        latest_path = session.get_checkpoint_path("latest")
        if os.path.exists(latest_path):
            print(f"Resuming from checkpoint: {latest_path}")
            trainer.agent.load(latest_path, restore_rng=True)
            print(f"  Timesteps: {trainer.agent.num_timesteps:,}")

        return trainer

    def _create_render_env(self):
        """Create an environment for visualization."""
        import gymnasium as gym
        from config import ENV_IDS

        env_name = self.env_config.env_name

        if env_name == "mario":
            # Mario uses nes-py rendering
            return make_env(env_name)
        else:
            # Atari with rgb_array mode (we'll display with cv2 for speed control)
            from environments.wrappers import (
                GrayscaleWrapper,
                ResizeWrapper,
                FrameStackWrapper,
                NormalizeWrapper,
                MaxAndSkipWrapper,
            )

            base_env = gym.make(ENV_IDS[env_name], render_mode="human")
            base_env = MaxAndSkipWrapper(base_env, skip=4)
            base_env = GrayscaleWrapper(base_env)
            base_env = ResizeWrapper(base_env)
            base_env = FrameStackWrapper(base_env)
            return NormalizeWrapper(base_env)

    def _create_demo_env(self):
        """Create environment for demo playback with rendering."""
        import gymnasium as gym
        from config import ENV_IDS

        env_name = self.env_config.env_name

        if env_name == "mario":
            return make_env(env_name)
        else:
            from environments.wrappers import (
                GrayscaleWrapper,
                ResizeWrapper,
                FrameStackWrapper,
                NormalizeWrapper,
                MaxAndSkipWrapper,
            )
            # Use human render mode for demo
            base_env = gym.make(ENV_IDS[env_name], render_mode="human")
            base_env = MaxAndSkipWrapper(base_env, skip=4)
            base_env = GrayscaleWrapper(base_env)
            base_env = ResizeWrapper(base_env)
            base_env = FrameStackWrapper(base_env)
            return NormalizeWrapper(base_env)

    def _play_demo_episode(self, update_num: int):
        """Play one episode to show current agent performance."""
        import torch

        print(f"\nDemo episode (update {update_num})...")

        # Notify dashboard that demo started
        if self.dashboard:
            self.dashboard.demo_started()

        obs, _ = self.demo_env.reset()
        done = False
        total_reward = 0
        steps = 0
        max_steps = 2000  # Limit demo length
        skipped = False

        self.agent.network.eval()

        while not done and steps < max_steps:
            # Check if user wants to skip
            if self.dashboard and self.dashboard.should_skip_demo():
                skipped = True
                break

            with torch.no_grad():
                obs_tensor = torch.tensor(obs[None], dtype=torch.float32, device=self.agent.device)
                action_logits, _ = self.agent.network(obs_tensor)
                action = action_logits.argmax(dim=-1).item()

            obs, reward, terminated, truncated, info = self.demo_env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

            # Mario needs explicit render
            if self.env_config.env_name == "mario":
                self.demo_env.render()

        # Notify dashboard that demo ended
        if self.dashboard:
            self.dashboard.demo_ended()

        if skipped:
            print(f"   Demo skipped after {steps} steps (reward: {total_reward:.0f})")
        else:
            print(f"   Demo reward: {total_reward:.0f} | Steps: {steps}")

    def train(self, total_timesteps: Optional[int] = None) -> Dict:
        """
        Run the training loop.

        Args:
            total_timesteps: Override total timesteps from config

        Returns:
            Final training metrics
        """
        total_timesteps = total_timesteps or self.training_config.total_timesteps

        print(f"\n{'='*60}")
        print(f"Starting training: {self.env_config.env_name}")
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"Parallel environments: {self.env_config.n_envs}")
        print(f"Steps per update: {self.ppo_config.n_steps * self.env_config.n_envs}")
        print(f"{'='*60}\n")

        # Initialize
        obs = self.env.reset()
        self.start_time = time.time()
        num_updates = 0

        # Initialize render environment
        render_obs = None
        if self.render_env:
            render_obs, _ = self.render_env.reset()

        # Notify callbacks
        for callback in self.callbacks:
            callback.on_training_start(self)

        while self.agent.num_timesteps < total_timesteps:
            # Collect rollout (and render if enabled)
            obs, render_obs = self._collect_rollout_with_tracking(obs, render_obs)

            # Update policy
            train_metrics = self.agent.train()
            num_updates += 1

            # Store metrics for dashboard
            self.policy_losses.append(train_metrics['policy_loss'])
            self.value_losses.append(train_metrics['value_loss'])
            self.entropies.append(-train_metrics['entropy_loss'])

            # Update dashboard
            if self.dashboard:
                self.dashboard.update(
                    policy_loss=train_metrics['policy_loss'],
                    value_loss=train_metrics['value_loss'],
                    entropy=-train_metrics['entropy_loss'],
                    timestep=self.agent.num_timesteps,
                )

            # Log metrics
            if num_updates % self.training_config.log_interval == 0:
                self._log_training_progress(train_metrics, num_updates)

            # Save checkpoint
            if num_updates % self.training_config.save_interval == 0:
                self._save_checkpoint(num_updates)

            # Play demo episode (use dashboard value if available, else config)
            demo_every = self.dashboard.demo_every if self.dashboard else self.training_config.demo_every
            if self.demo_env and demo_every > 0 and num_updates % demo_every == 0:
                self._play_demo_episode(num_updates)

            # Run callbacks
            for callback in self.callbacks:
                callback.on_step(self)

        # Final save
        self._save_checkpoint(num_updates, final=True)

        # Notify callbacks
        for callback in self.callbacks:
            callback.on_training_end(self)

        # Save dashboard
        if self.dashboard:
            self.dashboard.save(os.path.join(self.training_config.log_dir, "training_plot.png"))
            self.dashboard.close()

        # Cleanup
        self.writer.close()
        self.env.close()
        if self.render_env:
            self.render_env.close()
        if self.demo_env:
            self.demo_env.close()

        return self._get_final_metrics()

    def _collect_rollout_with_tracking(
        self,
        obs: np.ndarray,
        render_obs: Optional[np.ndarray] = None,
    ) -> tuple:
        """
        Collect rollout while tracking episode statistics and rendering.

        Args:
            obs: Current observations from training envs
            render_obs: Current observation from render env

        Returns:
            Tuple of (final obs, final render obs)
        """
        self.agent.network.eval()
        import torch

        for step in range(self.ppo_config.n_steps):
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.agent.device)
                action, log_prob, _, value = self.agent.network.get_action_and_value(
                    obs_tensor
                )

            # Convert to numpy
            action_np = action.cpu().numpy()
            log_prob_np = log_prob.cpu().numpy()
            value_np = value.cpu().numpy()

            # Step training environment
            next_obs, reward, done, info = self.env.step(action_np)

            # Step render environment (using first env's action)
            if self.render_env and render_obs is not None:
                render_obs = self._step_render_env(render_obs, action_np[0])

            # Track episode progress
            self.current_episode_rewards += reward
            self.current_episode_lengths += 1

            # Handle episode endings
            for i, d in enumerate(done):
                if d:
                    ep_reward = self.current_episode_rewards[i]
                    ep_length = self.current_episode_lengths[i]

                    self.episode_rewards.append(ep_reward)
                    self.episode_lengths.append(ep_length)
                    self.total_episodes += 1

                    # Log to TensorBoard
                    self.writer.add_scalar(
                        "episode/reward",
                        ep_reward,
                        self.total_episodes,
                    )
                    self.writer.add_scalar(
                        "episode/length",
                        ep_length,
                        self.total_episodes,
                    )

                    # Update dashboard with episode info
                    if self.dashboard:
                        self.dashboard.update(
                            reward=ep_reward,
                            length=ep_length,
                        )

                    # Reset tracking for this environment
                    self.current_episode_rewards[i] = 0
                    self.current_episode_lengths[i] = 0

            # Store transition
            self.agent.buffer.add(
                obs=obs,
                action=action_np,
                reward=reward,
                done=done,
                value=value_np,
                log_prob=log_prob_np,
            )

            obs = next_obs
            self.agent.num_timesteps += self.env_config.n_envs

        # Compute value of final state for GAE
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.agent.device)
            last_value = self.agent.network.get_value(obs_tensor).cpu().numpy()

        # Compute returns and advantages
        self.agent.buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=self.ppo_config.gamma,
            gae_lambda=self.ppo_config.gae_lambda,
        )

        return obs, render_obs

    def _step_render_env(self, obs: np.ndarray, action: int) -> np.ndarray:
        """Step the render environment and display."""
        import torch

        # Get action for render env (use same network with render env's observation)
        with torch.no_grad():
            obs_tensor = torch.tensor(obs[None], dtype=torch.float32, device=self.agent.device)
            action_logits, _ = self.agent.network(obs_tensor)
            action = action_logits.argmax(dim=-1).item()

        # Step
        next_obs, reward, terminated, truncated, info = self.render_env.step(action)
        self.render_frame_count += 1

        # Only render every Nth frame for speed
        should_render = (self.render_frame_count % self.training_config.render_every) == 0

        # Check if Mario (needs explicit render call)
        if self.env_config.env_name == "mario" and should_render:
            self.render_env.render()

        # Reset if done
        if terminated or truncated:
            next_obs, _ = self.render_env.reset()

        return next_obs

    def _log_training_progress(
        self,
        train_metrics: Dict[str, float],
        num_updates: int,
    ):
        """Log training metrics to console and TensorBoard."""
        elapsed = time.time() - self.start_time
        fps = self.agent.num_timesteps / elapsed

        # Console output
        print(f"\n--- Update {num_updates} ---")
        print(f"Timesteps: {self.agent.num_timesteps:,}")
        print(f"Episodes: {self.total_episodes}")
        print(f"FPS: {fps:.0f}")

        if len(self.episode_rewards) > 0:
            print(f"Mean reward (100 ep): {np.mean(self.episode_rewards):.2f}")
            print(f"Mean length (100 ep): {np.mean(self.episode_lengths):.0f}")

        print(f"Policy loss: {train_metrics['policy_loss']:.4f}")
        print(f"Value loss: {train_metrics['value_loss']:.4f}")
        print(f"Entropy: {-train_metrics['entropy_loss']:.4f}")
        print(f"Clip fraction: {train_metrics['clip_fraction']:.3f}")

        # TensorBoard
        self.writer.add_scalar(
            "train/policy_loss",
            train_metrics["policy_loss"],
            self.agent.num_timesteps,
        )
        self.writer.add_scalar(
            "train/value_loss",
            train_metrics["value_loss"],
            self.agent.num_timesteps,
        )
        self.writer.add_scalar(
            "train/entropy",
            -train_metrics["entropy_loss"],
            self.agent.num_timesteps,
        )
        self.writer.add_scalar(
            "train/clip_fraction",
            train_metrics["clip_fraction"],
            self.agent.num_timesteps,
        )
        self.writer.add_scalar(
            "performance/fps",
            fps,
            self.agent.num_timesteps,
        )

        if len(self.episode_rewards) > 0:
            self.writer.add_scalar(
                "performance/mean_reward",
                np.mean(self.episode_rewards),
                self.agent.num_timesteps,
            )
            self.writer.add_scalar(
                "performance/mean_length",
                np.mean(self.episode_lengths),
                self.agent.num_timesteps,
            )

    def _save_checkpoint(self, num_updates: int, final: bool = False):
        """Save model checkpoint."""
        if final:
            path = os.path.join(
                self.training_config.checkpoint_dir,
                "final_model.pt",
            )
        else:
            path = os.path.join(
                self.training_config.checkpoint_dir,
                f"checkpoint_{num_updates}.pt",
            )

        # Save with session metadata if available
        self.agent.save(path, session=self.session)
        print(f"Checkpoint saved: {path}")

        # Update session progress if available
        if self.session is not None:
            mean_reward = np.mean(self.episode_rewards) if self.episode_rewards else None
            self.session.update_progress(
                timesteps=self.agent.num_timesteps,
                episodes=self.total_episodes,
                updates=num_updates,
                best_reward=mean_reward,
            )

            # Mark completed if final
            if final:
                self.session.mark_completed()

    def _get_final_metrics(self) -> Dict:
        """Return final training metrics."""
        return {
            "total_timesteps": self.agent.num_timesteps,
            "total_episodes": self.total_episodes,
            "mean_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0,
            "mean_length": np.mean(self.episode_lengths) if self.episode_lengths else 0,
            "training_time": time.time() - self.start_time,
        }
