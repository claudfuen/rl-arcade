"""
Configuration and hyperparameters for RL training.

This module centralizes all tunable parameters, making it easy to
experiment with different settings without modifying code.
"""

from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class PPOConfig:
    """PPO algorithm hyperparameters."""

    # Learning
    learning_rate: float = 2.5e-4
    gamma: float = 0.99  # Discount factor for future rewards
    gae_lambda: float = 0.95  # GAE parameter for advantage estimation

    # PPO-specific
    clip_range: float = 0.2  # Clipping parameter for policy updates
    clip_range_vf: Optional[float] = None  # Value function clip (None = no clip)

    # Training batch sizes
    n_steps: int = 128  # Steps to collect before each update
    n_epochs: int = 4  # Number of PPO update epochs per rollout
    batch_size: int = 256  # Mini-batch size for updates

    # Loss coefficients
    vf_coef: float = 0.5  # Value function loss coefficient
    ent_coef: float = 0.02  # Entropy bonus coefficient (encourages exploration)
    max_grad_norm: float = 0.5  # Gradient clipping threshold


@dataclass
class NetworkConfig:
    """Neural network architecture settings."""

    # CNN feature extractor (Nature DQN architecture)
    conv_channels: tuple = (32, 64, 64)
    conv_kernels: tuple = (8, 4, 3)
    conv_strides: tuple = (4, 2, 1)

    # Fully connected layers
    hidden_size: int = 512

    # Input dimensions (after preprocessing)
    frame_stack: int = 4
    frame_height: int = 84
    frame_width: int = 84


@dataclass
class TrainingConfig:
    """Training loop configuration."""

    # Duration
    total_timesteps: int = 1_000_000

    # Logging
    log_interval: int = 10  # Log every N updates
    save_interval: int = 50  # Save checkpoint every N updates
    video_interval: int = 100  # Record video every N episodes

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # Device
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    # Rendering
    render_training: bool = False
    render_every: int = 1  # Render every Nth frame (higher = faster)
    demo_every: int = 0  # Play a demo episode every N updates (0 = disabled)


@dataclass
class EnvConfig:
    """Environment configuration."""

    # Environment selection
    env_name: str = "breakout"  # Options: breakout, pong, mario

    # Preprocessing
    frame_skip: int = 4  # Number of frames to skip (action repeat)
    frame_stack: int = 4  # Number of frames to stack
    frame_height: int = 84
    frame_width: int = 84

    # Parallel environments
    n_envs: int = 8  # Number of parallel environments

    # Episode limits
    max_episode_steps: Optional[int] = None  # None = use default


# Environment name mappings to actual gym IDs
ENV_IDS = {
    "breakout": "ALE/Breakout-v5",
    "pong": "ALE/Pong-v5",
    "spaceinvaders": "ALE/SpaceInvaders-v5",
    "mario": "SuperMarioBros-v0",
    "pokemon": "pokemon_red",
    "sonic": "SonicTheHedgehog-Genesis",
}


def get_default_config():
    """Get default configuration for all components."""
    return {
        "ppo": PPOConfig(),
        "network": NetworkConfig(),
        "training": TrainingConfig(),
        "env": EnvConfig(),
    }
