"""
Training callbacks for monitoring and checkpointing.

Callbacks allow you to inject custom behavior at various points
in the training loop without modifying the core training code.
"""

import os
import numpy as np
from typing import TYPE_CHECKING, Optional
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from .trainer import Trainer
    from sessions import Session


class BaseCallback(ABC):
    """
    Base class for training callbacks.

    Implement the methods you need:
    - on_training_start: Called once at the beginning
    - on_step: Called after each rollout+update
    - on_training_end: Called once at the end
    """

    @abstractmethod
    def on_training_start(self, trainer: "Trainer"):
        """Called when training starts."""
        pass

    @abstractmethod
    def on_step(self, trainer: "Trainer"):
        """Called after each training update."""
        pass

    @abstractmethod
    def on_training_end(self, trainer: "Trainer"):
        """Called when training ends."""
        pass


class CheckpointCallback(BaseCallback):
    """
    Save the best model based on mean reward.

    Keeps track of the best performance seen so far and saves
    a separate "best_model.pt" checkpoint.
    """

    def __init__(self, save_path: str = "checkpoints/best_model.pt"):
        self.save_path = save_path
        self.best_mean_reward = -float("inf")

    def on_training_start(self, trainer: "Trainer"):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

    def on_step(self, trainer: "Trainer"):
        if len(trainer.episode_rewards) == 0:
            return

        mean_reward = np.mean(trainer.episode_rewards)
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            trainer.agent.save(self.save_path)
            print(f"New best reward: {mean_reward:.2f} - Model saved!")

    def on_training_end(self, trainer: "Trainer"):
        print(f"Best mean reward achieved: {self.best_mean_reward:.2f}")


class EpisodeLoggerCallback(BaseCallback):
    """
    Log detailed episode information.

    Useful for debugging and understanding agent behavior.
    """

    def __init__(self, log_interval: int = 10):
        self.log_interval = log_interval
        self.last_logged_episode = 0

    def on_training_start(self, trainer: "Trainer"):
        pass

    def on_step(self, trainer: "Trainer"):
        # Log every N episodes
        if trainer.total_episodes - self.last_logged_episode >= self.log_interval:
            self.last_logged_episode = trainer.total_episodes

            if len(trainer.episode_rewards) > 0:
                recent_rewards = list(trainer.episode_rewards)[-self.log_interval:]
                print(f"\nRecent {len(recent_rewards)} episodes:")
                print(f"  Rewards: {[f'{r:.1f}' for r in recent_rewards]}")
                print(f"  Min: {min(recent_rewards):.1f}")
                print(f"  Max: {max(recent_rewards):.1f}")
                print(f"  Mean: {np.mean(recent_rewards):.1f}")

    def on_training_end(self, trainer: "Trainer"):
        pass


class VideoRecorderCallback(BaseCallback):
    """
    Record videos of the agent playing.

    Note: This is a simplified version. Full video recording would
    require additional setup with gymnasium's RecordVideo wrapper.
    """

    def __init__(
        self,
        video_dir: str = "videos",
        record_interval: int = 100,
    ):
        self.video_dir = video_dir
        self.record_interval = record_interval
        self.last_recorded_episode = 0

    def on_training_start(self, trainer: "Trainer"):
        os.makedirs(self.video_dir, exist_ok=True)
        print(f"Video recording enabled. Videos will be saved to {self.video_dir}/")

    def on_step(self, trainer: "Trainer"):
        # This is a placeholder - full video recording would need
        # additional implementation with gymnasium's video wrapper
        if trainer.total_episodes - self.last_recorded_episode >= self.record_interval:
            self.last_recorded_episode = trainer.total_episodes
            print(f"[Video] Would record episode {trainer.total_episodes}")

    def on_training_end(self, trainer: "Trainer"):
        pass


class EarlyStoppingCallback(BaseCallback):
    """
    Stop training when a reward threshold is reached.

    Useful for quickly testing if an implementation works.
    """

    def __init__(
        self,
        reward_threshold: float,
        min_episodes: int = 100,
    ):
        self.reward_threshold = reward_threshold
        self.min_episodes = min_episodes
        self.should_stop = False

    def on_training_start(self, trainer: "Trainer"):
        print(f"Early stopping enabled at reward threshold: {self.reward_threshold}")

    def on_step(self, trainer: "Trainer"):
        if (
            trainer.total_episodes >= self.min_episodes
            and len(trainer.episode_rewards) >= self.min_episodes
        ):
            mean_reward = np.mean(trainer.episode_rewards)
            if mean_reward >= self.reward_threshold:
                print(f"\nEarly stopping: Reached reward {mean_reward:.2f}")
                self.should_stop = True

    def on_training_end(self, trainer: "Trainer"):
        pass


class SessionCheckpointCallback(BaseCallback):
    """
    Session-aware checkpointing callback.

    Saves checkpoints with full session metadata and maintains
    a 'latest.pt' symlink for easy resume.
    """

    def __init__(
        self,
        session: "Session",
        save_interval: int = 50,
    ):
        """
        Initialize session checkpoint callback.

        Args:
            session: Session to save checkpoints for
            save_interval: Save checkpoint every N updates
        """
        self.session = session
        self.save_interval = save_interval
        self.best_mean_reward = float("-inf")
        self.num_updates = 0

    def on_training_start(self, trainer: "Trainer"):
        os.makedirs(self.session.checkpoint_dir, exist_ok=True)
        print(f"Session checkpoints: {self.session.checkpoint_dir}")

    def on_step(self, trainer: "Trainer"):
        self.num_updates += 1

        # Save periodic checkpoint
        if self.num_updates % self.save_interval == 0:
            self._save_checkpoint(trainer, f"checkpoint_{self.num_updates}")

        # Always update latest
        self._save_checkpoint(trainer, "latest")

        # Track best model
        if len(trainer.episode_rewards) > 0:
            mean_reward = np.mean(trainer.episode_rewards)
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self._save_checkpoint(trainer, "best")
                print(f"New best reward: {mean_reward:.2f}")

        # Update session progress
        self.session.update_progress(
            timesteps=trainer.agent.num_timesteps,
            episodes=trainer.total_episodes,
            updates=self.num_updates,
            best_reward=self.best_mean_reward if self.best_mean_reward > float("-inf") else None,
        )

    def on_training_end(self, trainer: "Trainer"):
        # Final save
        self._save_checkpoint(trainer, "final")
        self.session.mark_completed()
        print(f"Session completed: {self.session.session_id}")

    def _save_checkpoint(self, trainer: "Trainer", name: str):
        """Save a checkpoint with session metadata and dashboard state."""
        path = self.session.get_checkpoint_path(name)
        trainer.agent.save(path, session=self.session, full_state=True)

        # Save dashboard state for graph restoration on resume
        if trainer.dashboard is not None:
            self.session.save_dashboard_state(trainer.dashboard.export_state())
