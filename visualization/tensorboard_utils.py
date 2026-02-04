"""
TensorBoard logging utilities.

TensorBoard provides powerful visualization for ML experiments:
- Scalar plots (rewards, losses over time)
- Histograms (weight distributions)
- Images (game frames, attention maps)
- Videos (agent playing)
- Custom layouts

To view logs:
    tensorboard --logdir logs/
Then open http://localhost:6006 in your browser.
"""

import os
import numpy as np
import torch
from typing import Dict, Optional, Union
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    """
    Wrapper for TensorBoard logging with RL-specific utilities.

    Provides convenient methods for logging:
    - Training metrics (losses, learning rates)
    - Episode statistics (rewards, lengths)
    - Network diagnostics (gradients, weights)
    - Hyperparameters
    """

    def __init__(
        self,
        log_dir: str = "logs",
        experiment_name: Optional[str] = None,
        comment: str = "",
    ):
        """
        Initialize TensorBoard logger.

        Args:
            log_dir: Base directory for logs
            experiment_name: Name for this run (auto-generated if None)
            comment: Additional comment appended to run name
        """
        if experiment_name:
            full_path = os.path.join(log_dir, experiment_name)
        else:
            full_path = log_dir

        self.writer = SummaryWriter(log_dir=full_path, comment=comment)
        self.log_dir = full_path
        print(f"TensorBoard logs: {full_path}")
        print("View with: tensorboard --logdir {log_dir}")

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """Log multiple scalars under a common tag."""
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_training_metrics(
        self,
        step: int,
        policy_loss: float,
        value_loss: float,
        entropy: float,
        clip_fraction: float,
        learning_rate: Optional[float] = None,
    ):
        """
        Log standard PPO training metrics.

        Args:
            step: Global timestep
            policy_loss: Policy gradient loss
            value_loss: Value function loss
            entropy: Policy entropy
            clip_fraction: Fraction of clipped ratios
            learning_rate: Current learning rate
        """
        self.writer.add_scalar("train/policy_loss", policy_loss, step)
        self.writer.add_scalar("train/value_loss", value_loss, step)
        self.writer.add_scalar("train/entropy", entropy, step)
        self.writer.add_scalar("train/clip_fraction", clip_fraction, step)

        if learning_rate is not None:
            self.writer.add_scalar("train/learning_rate", learning_rate, step)

    def log_episode(
        self,
        episode: int,
        reward: float,
        length: int,
        timestep: Optional[int] = None,
    ):
        """
        Log episode statistics.

        Args:
            episode: Episode number
            reward: Total episode reward
            length: Episode length in steps
            timestep: Global timestep (optional)
        """
        self.writer.add_scalar("episode/reward", reward, episode)
        self.writer.add_scalar("episode/length", length, episode)

        if timestep is not None:
            self.writer.add_scalar("episode/reward_vs_timestep", reward, timestep)

    def log_episode_batch(
        self,
        step: int,
        rewards: list,
        lengths: list,
    ):
        """
        Log statistics over a batch of episodes.

        Args:
            step: Global timestep
            rewards: List of episode rewards
            lengths: List of episode lengths
        """
        if len(rewards) == 0:
            return

        rewards = np.array(rewards)
        lengths = np.array(lengths)

        self.writer.add_scalar("performance/mean_reward", rewards.mean(), step)
        self.writer.add_scalar("performance/std_reward", rewards.std(), step)
        self.writer.add_scalar("performance/min_reward", rewards.min(), step)
        self.writer.add_scalar("performance/max_reward", rewards.max(), step)
        self.writer.add_scalar("performance/mean_length", lengths.mean(), step)

    def log_network_diagnostics(
        self,
        model: torch.nn.Module,
        step: int,
        log_weights: bool = True,
        log_gradients: bool = True,
    ):
        """
        Log network weight and gradient statistics.

        Useful for debugging training issues:
        - Dead ReLUs (many zero weights)
        - Vanishing/exploding gradients
        - Layer imbalances

        Args:
            model: PyTorch model
            step: Global timestep
            log_weights: Whether to log weight histograms
            log_gradients: Whether to log gradient histograms
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                if log_weights:
                    self.writer.add_histogram(f"weights/{name}", param.data, step)

                if log_gradients and param.grad is not None:
                    self.writer.add_histogram(f"gradients/{name}", param.grad, step)

                    # Also log gradient norms
                    grad_norm = param.grad.norm().item()
                    self.writer.add_scalar(f"grad_norms/{name}", grad_norm, step)

    def log_game_frame(
        self,
        tag: str,
        frame: np.ndarray,
        step: int,
    ):
        """
        Log a game frame as an image.

        Args:
            tag: Image tag
            frame: Frame array (H, W) or (H, W, C)
            step: Global step
        """
        if frame.ndim == 2:
            # Grayscale - add channel dimension
            frame = frame[np.newaxis, :, :]  # (1, H, W)
        elif frame.ndim == 3:
            # Move channels to first dimension for TensorBoard
            frame = np.transpose(frame, (2, 0, 1))  # (C, H, W)

        self.writer.add_image(tag, frame, step)

    def log_hyperparameters(
        self,
        hparams: Dict[str, Union[float, int, str, bool]],
        metrics: Optional[Dict[str, float]] = None,
    ):
        """
        Log hyperparameters for this experiment.

        TensorBoard's HParams plugin allows comparing different
        hyperparameter configurations across runs.

        Args:
            hparams: Dictionary of hyperparameter names and values
            metrics: Final metrics to associate with these hyperparameters
        """
        if metrics is None:
            metrics = {}

        self.writer.add_hparams(hparams, metrics)

    def log_text(self, tag: str, text: str, step: int):
        """Log text (useful for logging hyperparameters as readable text)."""
        self.writer.add_text(tag, text, step)

    def flush(self):
        """Flush pending events to disk."""
        self.writer.flush()

    def close(self):
        """Close the writer."""
        self.writer.close()


def log_config_to_tensorboard(
    writer: SummaryWriter,
    ppo_config,
    env_config,
    training_config,
):
    """
    Log configuration as text to TensorBoard.

    Creates a readable summary of all hyperparameters.
    """
    config_text = f"""
## Environment
- Name: {env_config.env_name}
- Frame skip: {env_config.frame_skip}
- Frame stack: {env_config.frame_stack}
- Parallel envs: {env_config.n_envs}

## PPO
- Learning rate: {ppo_config.learning_rate}
- Gamma: {ppo_config.gamma}
- GAE lambda: {ppo_config.gae_lambda}
- Clip range: {ppo_config.clip_range}
- N steps: {ppo_config.n_steps}
- N epochs: {ppo_config.n_epochs}
- Batch size: {ppo_config.batch_size}
- VF coefficient: {ppo_config.vf_coef}
- Entropy coefficient: {ppo_config.ent_coef}

## Training
- Total timesteps: {training_config.total_timesteps}
- Device: {training_config.device}
"""
    writer.add_text("config/hyperparameters", config_text, 0)
