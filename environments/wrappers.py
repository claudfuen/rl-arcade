"""
Environment wrappers for preprocessing game frames.

These wrappers transform raw game frames into a format suitable for
neural network training:
1. Convert RGB to grayscale (reduce computation)
2. Resize to 84x84 (standard for game RL)
3. Stack multiple frames (capture motion information)
4. Normalize pixel values (better gradient flow)

Key Concept - Frame Stacking:
    A single frame doesn't show velocity or direction of moving objects.
    By stacking 4 consecutive frames, the network can perceive motion
    (e.g., which direction the ball is moving in Pong).
"""

import gymnasium as gym
import numpy as np
import cv2
from collections import deque
from typing import Tuple, Optional


class GrayscaleWrapper(gym.ObservationWrapper):
    """
    Convert RGB observations to grayscale.

    Why grayscale?
    - Reduces input from 3 channels to 1 (3x less computation)
    - Color rarely contains useful information for game playing
    - Makes the network focus on shapes and movement
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Update observation space to single channel
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(old_shape[0], old_shape[1], 1),
            dtype=np.uint8,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Convert RGB observation to grayscale."""
        # cv2.cvtColor expects (H, W, C) format
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return gray[:, :, np.newaxis]  # Add channel dimension back


class ResizeWrapper(gym.ObservationWrapper):
    """
    Resize observations to a fixed size (default 84x84).

    Why 84x84?
    - Standard size used in DQN and most game RL papers
    - Small enough for efficient processing
    - Large enough to retain important visual details
    """

    def __init__(self, env: gym.Env, height: int = 84, width: int = 84):
        super().__init__(env)
        self.height = height
        self.width = width

        # Get number of channels from current observation space
        old_shape = self.observation_space.shape
        n_channels = old_shape[2] if len(old_shape) == 3 else 1

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(height, width, n_channels),
            dtype=np.uint8,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Resize observation to target dimensions."""
        # cv2.resize expects (H, W) or (H, W, C)
        resized = cv2.resize(
            obs,
            (self.width, self.height),
            interpolation=cv2.INTER_AREA,
        )
        # Ensure 3D shape
        if len(resized.shape) == 2:
            resized = resized[:, :, np.newaxis]
        return resized


class FrameStackWrapper(gym.Wrapper):
    """
    Stack multiple consecutive frames as observation.

    Key Concept - Temporal Information:
        Neural networks see static images. By stacking frames,
        we give the network a "short-term memory" to perceive:
        - Object velocity (position change between frames)
        - Direction of movement
        - Acceleration patterns

    The observation becomes shape (H, W, n_frames) instead of (H, W, 1).
    """

    def __init__(self, env: gym.Env, n_frames: int = 4):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)

        # Update observation space for stacked frames
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(old_shape[0], old_shape[1], n_frames),
            dtype=np.uint8,
        )

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        """Reset environment and initialize frame stack."""
        obs, info = self.env.reset(**kwargs)

        # Fill the frame stack with the initial observation
        for _ in range(self.n_frames):
            self.frames.append(obs)

        return self._get_observation(), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute action and update frame stack."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Stack frames along the channel dimension."""
        # Each frame is (H, W, 1), we want (H, W, n_frames)
        return np.concatenate(list(self.frames), axis=2)


class NormalizeWrapper(gym.ObservationWrapper):
    """
    Normalize pixel values from [0, 255] to [0, 1].

    Why normalize?
    - Neural networks train better with small input values
    - Prevents large gradients from pixel values
    - Consistent scale across different games
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=self.observation_space.shape,
            dtype=np.float32,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation to [0, 1] range."""
        return obs.astype(np.float32) / 255.0


class MaxAndSkipWrapper(gym.Wrapper):
    """
    Return max pixel values over last N frames and repeat action N times.

    Why frame skipping?
    - Games run at 60 FPS, but decisions don't need to be that frequent
    - Reduces computation by 4x (with skip=4)
    - Human reaction time is ~200ms anyway

    Why max over frames?
    - Some Atari games have flickering sprites (only visible every other frame)
    - Taking max ensures we don't miss any objects
    """

    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self.skip = skip
        # Buffer to store last 2 frames for max operation
        self._obs_buffer = np.zeros(
            (2,) + env.observation_space.shape,
            dtype=np.uint8,
        )

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Repeat action for `skip` frames and return max of last 2."""
        total_reward = 0.0
        terminated = truncated = False
        last_obs = None

        for i in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            last_obs = obs

            # Store last 2 frames for max operation
            if i >= self.skip - 2:
                self._obs_buffer[i - (self.skip - 2)] = obs

            if terminated or truncated:
                # If episode ends early, fill buffer with last observation
                self._obs_buffer[0] = obs
                self._obs_buffer[1] = obs
                break

        # Take max over last 2 frames (handles flickering sprites)
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info


class EpisodicLifeWrapper(gym.Wrapper):
    """
    Treat loss of life as end of episode (but only reset on true game over).

    Why?
    - Gives more frequent learning signals
    - Agent learns consequences of dying faster
    - Standard practice for Atari games
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        """Reset only on true game over."""
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # No-op step to advance from terminal/lost life state
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Check for life loss and treat as episode end."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated

        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            # Lost a life, treat as episode end for training
            terminated = True
        self.lives = lives

        return obs, reward, terminated, truncated, info


class FireResetWrapper(gym.Wrapper):
    """
    Press FIRE button at the start of episode.

    Some Atari games (like Breakout) require pressing FIRE to start.
    This wrapper handles that automatically.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        """Reset and press FIRE to start the game."""
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)  # FIRE
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class ClipRewardWrapper(gym.RewardWrapper):
    """
    Clip rewards to {-1, 0, +1}.

    Why clip rewards?
    - Different games have vastly different reward scales
    - Clipping normalizes learning across games
    - Prevents large reward spikes from destabilizing training
    """

    def reward(self, reward: float) -> float:
        """Clip reward to {-1, 0, +1}."""
        return np.sign(reward)
