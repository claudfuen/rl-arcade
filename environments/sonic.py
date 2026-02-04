"""
Sonic the Hedgehog environment using gym-retro.

Train an RL agent to play Sonic on Sega Genesis!

NOTE: You need to import the ROM after installing gym-retro:
      python -m retro.import /path/to/your/roms/
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple


class SonicEnv(gym.Env):
    """
    Sonic the Hedgehog environment using gym-retro.

    Observations:
        - Game screen (224x320 RGB, resized to 84x84)

    Actions (simplified):
        - 0: No action
        - 1: Left
        - 2: Right
        - 3: Jump
        - 4: Right + Jump
        - 5: Left + Jump
        - 6: Down (roll)
        - 7: Right + Down (spin dash)

    Rewards:
        - Moving right (progress in level)
        - Collecting rings
        - Completing level (big bonus)
        - Penalty for dying
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    # Simplified action mapping
    # retro uses multi-binary: [B, A, MODE, START, UP, DOWN, LEFT, RIGHT, C, Y, X, Z]
    ACTIONS = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0: No action
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 1: Left
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 2: Right
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3: Jump (B)
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 4: Right + Jump
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 5: Left + Jump
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 6: Down (roll)
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],  # 7: Right + Down
    ]

    def __init__(
        self,
        game: str = "SonicTheHedgehog-Genesis",
        state: str = "GreenHillZone.Act1",
        render_mode: Optional[str] = None,
    ):
        """
        Initialize Sonic environment.

        Args:
            game: Retro game name
            state: Starting level state
            render_mode: "human" or "rgb_array"
        """
        super().__init__()

        self.game = game
        self.state = state
        self.render_mode = render_mode

        # Will be initialized in reset()
        self.retro_env = None

        # Action and observation spaces
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(224, 320, 3),
            dtype=np.uint8,
        )

        # Tracking for rewards
        self.prev_x = 0

    def _init_retro(self):
        """Initialize retro environment."""
        try:
            import retro
        except ImportError:
            raise ImportError(
                "gym-retro not installed. Install with: pip install gym-retro\n"
                "Then import your ROMs with: python -m retro.import /path/to/roms/"
            )

        self.retro_env = retro.make(
            game=self.game,
            state=self.state,
            use_restricted_actions=retro.Actions.ALL,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Reset environment."""
        super().reset(seed=seed)

        if self.retro_env is None:
            self._init_retro()

        obs = self.retro_env.reset()
        self.prev_x = 0

        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute action."""
        # Convert discrete action to multi-binary
        retro_action = self.ACTIONS[action]

        obs, reward, done, info = self.retro_env.step(retro_action)

        # Additional reward shaping based on x position
        current_x = info.get("x", 0)
        x_reward = (current_x - self.prev_x) * 0.1
        self.prev_x = current_x

        # Combine rewards
        total_reward = reward + x_reward

        return obs, total_reward, done, False, info

    def render(self):
        """Render the game."""
        if self.retro_env is not None:
            return self.retro_env.render()

    def close(self):
        """Clean up resources."""
        if self.retro_env is not None:
            self.retro_env.close()
            self.retro_env = None


def make_sonic_env(
    game: str = "SonicTheHedgehog-Genesis",
    state: str = "GreenHillZone.Act1",
    frame_stack: int = 4,
    frame_height: int = 84,
    frame_width: int = 84,
    render_mode: Optional[str] = None,
) -> gym.Env:
    """
    Create a preprocessed Sonic environment.

    Args:
        game: Retro game name
        state: Starting level state
        frame_stack: Number of frames to stack
        frame_height: Height to resize frames to
        frame_width: Width to resize frames to
        render_mode: "human" or "rgb_array"

    Returns:
        Preprocessed Sonic environment
    """
    from .wrappers import (
        GrayscaleWrapper,
        ResizeWrapper,
        FrameStackWrapper,
        NormalizeWrapper,
    )

    # Create base environment
    env = SonicEnv(game=game, state=state, render_mode=render_mode)

    # Apply preprocessing
    env = GrayscaleWrapper(env)
    env = ResizeWrapper(env, height=frame_height, width=frame_width)
    env = FrameStackWrapper(env, n_frames=frame_stack)
    env = NormalizeWrapper(env)

    return env
