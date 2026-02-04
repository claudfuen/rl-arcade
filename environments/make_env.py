"""
Environment factory functions.

Creates preprocessed environments ready for RL training.
Supports Atari games (via ale-py) and Super Mario Bros.
"""

import gymnasium as gym
import numpy as np
from typing import Callable, Optional, List
from functools import partial

# Register Atari environments
import ale_py
gym.register_envs(ale_py)

from .wrappers import (
    GrayscaleWrapper,
    ResizeWrapper,
    FrameStackWrapper,
    NormalizeWrapper,
    MaxAndSkipWrapper,
    EpisodicLifeWrapper,
    FireResetWrapper,
    ClipRewardWrapper,
)
from config import ENV_IDS, EnvConfig


def make_atari_env(
    env_id: str,
    frame_skip: int = 4,
    frame_stack: int = 4,
    frame_height: int = 84,
    frame_width: int = 84,
    clip_rewards: bool = True,
    episodic_life: bool = True,
) -> gym.Env:
    """
    Create a preprocessed Atari environment.

    Preprocessing pipeline:
    1. MaxAndSkip: Skip frames and handle flickering
    2. EpisodicLife: Treat life loss as episode end
    3. FireReset: Auto-press FIRE at start
    4. Grayscale: RGB → single channel
    5. Resize: Scale to 84x84
    6. FrameStack: Stack 4 frames for motion
    7. Normalize: Scale pixels to [0, 1]
    8. ClipReward: Normalize rewards to {-1, 0, +1}

    Args:
        env_id: Gymnasium environment ID (e.g., "ALE/Breakout-v5")
        frame_skip: Number of frames to skip per action
        frame_stack: Number of frames to stack
        frame_height: Height to resize frames to
        frame_width: Width to resize frames to
        clip_rewards: Whether to clip rewards
        episodic_life: Whether to treat life loss as episode end

    Returns:
        Preprocessed gymnasium environment
    """
    # Create base environment
    env = gym.make(env_id, render_mode=None)

    # Apply preprocessing wrappers in order
    env = MaxAndSkipWrapper(env, skip=frame_skip)

    if episodic_life:
        env = EpisodicLifeWrapper(env)

    # Check if FIRE action is needed
    action_meanings = env.unwrapped.get_action_meanings()
    if len(action_meanings) >= 3 and action_meanings[1] == "FIRE":
        env = FireResetWrapper(env)

    # Visual preprocessing
    env = GrayscaleWrapper(env)
    env = ResizeWrapper(env, height=frame_height, width=frame_width)
    env = FrameStackWrapper(env, n_frames=frame_stack)
    env = NormalizeWrapper(env)

    # Reward preprocessing
    if clip_rewards:
        env = ClipRewardWrapper(env)

    return env


def make_mario_env(
    frame_skip: int = 4,
    frame_stack: int = 4,
    frame_height: int = 84,
    frame_width: int = 84,
) -> gym.Env:
    """
    Create a preprocessed Super Mario Bros environment.

    Mario-specific notes:
    - Uses simplified action space (RIGHT_ONLY or SIMPLE)
    - Reward is based on x-position progress
    - No flickering issues like Atari

    Args:
        frame_skip: Number of frames to skip per action
        frame_stack: Number of frames to stack
        frame_height: Height to resize frames to
        frame_width: Width to resize frames to

    Returns:
        Preprocessed Mario environment
    """
    from nes_py.wrappers import JoypadSpace
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

    # Create Mario environment
    env = gym_super_mario_bros.make("SuperMarioBros-v0")

    # Simplify action space (7 actions instead of 256)
    # SIMPLE_MOVEMENT = [['NOOP'], ['right'], ['right', 'A'], ['right', 'B'],
    #                   ['right', 'A', 'B'], ['A'], ['left']]
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    # Wrap in gymnasium compatibility layer
    env = GymCompatibilityWrapper(env)

    # Apply preprocessing (no MaxAndSkip as nes-py handles frame timing)
    env = GrayscaleWrapper(env)
    env = ResizeWrapper(env, height=frame_height, width=frame_width)
    env = FrameStackWrapper(env, n_frames=frame_stack)
    env = NormalizeWrapper(env)

    return env


class GymCompatibilityWrapper(gym.Wrapper):
    """
    Wrapper to make old gym environments compatible with gymnasium API.

    gym_super_mario_bros uses the old gym API (returns 4 values from step).
    This wrapper converts to the new gymnasium API (returns 5 values).
    """

    def __init__(self, env):
        # Don't call super().__init__ as old gym env isn't gymnasium compatible
        self.env = env
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=env.observation_space.shape,
            dtype=np.uint8,
        )
        self.action_space = gym.spaces.Discrete(env.action_space.n)
        # Required attributes for gymnasium compatibility
        self.unwrapped = env
        self.metadata = getattr(env, 'metadata', {})
        self.spec = getattr(env, 'spec', None)

    def reset(self, **kwargs):
        """Reset with gymnasium-style return."""
        # Old gym reset() doesn't accept kwargs
        obs = self.env.reset()
        return obs, {}

    def step(self, action):
        """Step with gymnasium-style return (5 values)."""
        obs, reward, done, info = self.env.step(action)
        # In gymnasium, 'done' is split into 'terminated' and 'truncated'
        terminated = done
        truncated = False
        return obs, float(reward), terminated, truncated, info

    def render(self, mode=None):
        """Render the environment."""
        return self.env.render()

    def close(self):
        return self.env.close()


def make_env(
    env_name: str = "breakout",
    frame_skip: int = 4,
    frame_stack: int = 4,
    frame_height: int = 84,
    frame_width: int = 84,
    **kwargs,
) -> gym.Env:
    """
    Factory function to create any supported environment.

    Supported environments:
    - "breakout": Atari Breakout (good for beginners)
    - "pong": Atari Pong (quick to train)
    - "spaceinvaders": Atari Space Invaders
    - "mario": Super Mario Bros (more complex)
    - "pokemon": Pokemon Red/Blue (requires ROM)
    - "sonic": Sonic the Hedgehog (requires ROM)

    Args:
        env_name: Short name of the environment
        frame_skip: Frames to skip per action
        frame_stack: Frames to stack for observation
        frame_height: Observation height
        frame_width: Observation width
        **kwargs: Additional arguments for specific environments

    Returns:
        Preprocessed environment ready for training
    """
    env_name = env_name.lower()

    if env_name == "mario":
        return make_mario_env(
            frame_skip=frame_skip,
            frame_stack=frame_stack,
            frame_height=frame_height,
            frame_width=frame_width,
        )
    elif env_name == "pokemon":
        from .pokemon import make_pokemon_env
        return make_pokemon_env(
            frame_stack=frame_stack,
            frame_height=frame_height,
            frame_width=frame_width,
            **kwargs,
        )
    elif env_name == "sonic":
        from .sonic import make_sonic_env
        return make_sonic_env(
            frame_stack=frame_stack,
            frame_height=frame_height,
            frame_width=frame_width,
            **kwargs,
        )
    elif env_name in ["breakout", "pong", "spaceinvaders"]:
        return make_atari_env(
            env_id=ENV_IDS[env_name],
            frame_skip=frame_skip,
            frame_stack=frame_stack,
            frame_height=frame_height,
            frame_width=frame_width,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown environment: {env_name}. "
            f"Supported: breakout, pong, spaceinvaders, mario, pokemon, sonic"
        )


class VecEnv:
    """
    Vectorized environment that runs multiple environments in parallel.

    Why vectorized environments?
    - Collect more experience per unit time
    - Better GPU utilization with batched operations
    - More diverse experience for training
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        """
        Create vectorized environment from list of env creation functions.

        Args:
            env_fns: List of functions that create environments
        """
        self.envs = [fn() for fn in env_fns]
        self.n_envs = len(self.envs)

        # Get spaces from first environment
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self) -> np.ndarray:
        """Reset all environments and return stacked observations."""
        observations = []
        for env in self.envs:
            obs, _ = env.reset()
            observations.append(obs)
        return np.stack(observations)

    def step(self, actions: np.ndarray):
        """
        Step all environments with given actions.

        Args:
            actions: Array of actions, one per environment

        Returns:
            observations: Stacked observations (n_envs, *obs_shape)
            rewards: Array of rewards (n_envs,)
            dones: Array of done flags (n_envs,)
            infos: List of info dicts
        """
        observations = []
        rewards = []
        dones = []
        infos = []

        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Auto-reset on done
            if done:
                info["terminal_observation"] = obs
                obs, _ = env.reset()

            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        return (
            np.stack(observations),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            infos,
        )

    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()


def make_vec_env(
    env_name: str = "breakout",
    n_envs: int = 8,
    **kwargs,
) -> VecEnv:
    """
    Create a vectorized environment with multiple parallel instances.

    Args:
        env_name: Name of the environment
        n_envs: Number of parallel environments
        **kwargs: Arguments passed to make_env

    Returns:
        Vectorized environment
    """
    env_fns = [partial(make_env, env_name=env_name, **kwargs) for _ in range(n_envs)]
    return VecEnv(env_fns)
