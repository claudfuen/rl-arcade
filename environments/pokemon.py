"""
Pokemon Red/Blue environment using PyBoy emulator.

This environment lets you train an RL agent to play Pokemon!
The agent learns to explore, battle, catch Pokemon, and collect badges.

NOTE: You must provide your own ROM file (pokemon_red.gb or pokemon_blue.gb)
      Place it in the 'roms/' directory.
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any


class PokemonRedEnv(gym.Env):
    """
    Pokemon Red/Blue environment for reinforcement learning.

    Observations:
        - Game screen (144x160 RGB or grayscale)

    Actions:
        - 0: No action
        - 1: A button
        - 2: B button
        - 3: Start
        - 4: Up
        - 5: Down
        - 6: Left
        - 7: Right

    Rewards are based on:
        - Gaining experience points
        - Leveling up Pokemon
        - Catching new Pokemon
        - Getting badges
        - Healing at Pokemon Center
        - Exploring new map locations
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    # Memory addresses for Pokemon Red (US version)
    # These are used to read game state for reward shaping
    MEMORY_ADDRS = {
        "player_x": 0xD362,
        "player_y": 0xD361,
        "map_id": 0xD35E,
        "badges": 0xD356,
        "money": (0xD347, 0xD348, 0xD349),  # BCD encoded
        "party_count": 0xD163,
        "party_level_1": 0xD18C,
        "party_level_2": 0xD1B8,
        "party_level_3": 0xD1E4,
        "party_level_4": 0xD210,
        "party_level_5": 0xD23C,
        "party_level_6": 0xD268,
        "party_hp_1": (0xD16C, 0xD16D),
        "pokedex_owned": 0xD2F7,  # Number of Pokemon owned
        "battle_type": 0xD057,  # 0 = no battle, 1 = wild, 2 = trainer
    }

    # Button mappings for PyBoy
    ACTIONS = [
        [],           # 0: No action
        ["a"],        # 1: A
        ["b"],        # 2: B
        ["start"],    # 3: Start
        ["up"],       # 4: Up
        ["down"],     # 5: Down
        ["left"],     # 6: Left
        ["right"],    # 7: Right
    ]

    def __init__(
        self,
        rom_path: str = "roms/pokemon_red.gb",
        render_mode: Optional[str] = None,
        headless: bool = True,
        save_state: Optional[str] = None,
        max_steps: int = 10000,
        frame_skip: int = 24,  # Pokemon runs at 60fps, skip frames for speed
    ):
        """
        Initialize Pokemon environment.

        Args:
            rom_path: Path to Pokemon ROM file
            render_mode: "human" for window, "rgb_array" for pixel data
            headless: Run without display window
            save_state: Path to save state to start from
            max_steps: Maximum steps per episode
            frame_skip: Number of frames to skip per action
        """
        super().__init__()

        self.rom_path = rom_path
        self.render_mode = render_mode
        self.headless = headless and render_mode != "human"
        self.save_state = save_state
        self.max_steps = max_steps
        self.frame_skip = frame_skip

        # Will be initialized in reset()
        self.pyboy = None
        self.screen = None

        # Action and observation spaces
        self.action_space = spaces.Discrete(8)

        # Game Boy screen is 160x144
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(144, 160, 3),
            dtype=np.uint8,
        )

        # State tracking for rewards
        self.prev_state = {}
        self.visited_locations = set()
        self.step_count = 0

    def _init_pyboy(self):
        """Initialize PyBoy emulator."""
        try:
            from pyboy import PyBoy
        except ImportError:
            raise ImportError(
                "PyBoy not installed. Install with: pip install pyboy"
            )

        if not os.path.exists(self.rom_path):
            raise FileNotFoundError(
                f"ROM not found: {self.rom_path}\n"
                f"Please place your Pokemon ROM in the 'roms/' directory.\n"
                f"Expected: roms/pokemon_red.gb or roms/pokemon_blue.gb"
            )

        # Initialize PyBoy
        window_type = "null" if self.headless else "SDL2"
        self.pyboy = PyBoy(
            self.rom_path,
            window=window_type,
        )

        # Disable sound for speed
        self.pyboy.set_emulation_speed(0)  # Unlimited speed

    def _read_memory(self, addr: int) -> int:
        """Read a byte from game memory."""
        return self.pyboy.memory[addr]

    def _read_memory_word(self, addrs: Tuple[int, int]) -> int:
        """Read a 16-bit word from two memory addresses."""
        return self._read_memory(addrs[0]) + (self._read_memory(addrs[1]) << 8)

    def _get_game_state(self) -> Dict[str, Any]:
        """Read current game state from memory."""
        state = {
            "player_x": self._read_memory(self.MEMORY_ADDRS["player_x"]),
            "player_y": self._read_memory(self.MEMORY_ADDRS["player_y"]),
            "map_id": self._read_memory(self.MEMORY_ADDRS["map_id"]),
            "badges": bin(self._read_memory(self.MEMORY_ADDRS["badges"])).count("1"),
            "party_count": self._read_memory(self.MEMORY_ADDRS["party_count"]),
            "pokedex_owned": self._read_memory(self.MEMORY_ADDRS["pokedex_owned"]),
        }

        # Sum party levels
        total_levels = 0
        for i in range(1, 7):
            level = self._read_memory(self.MEMORY_ADDRS[f"party_level_{i}"])
            if level > 0 and level <= 100:
                total_levels += level
        state["total_levels"] = total_levels

        return state

    def _calculate_reward(self, state: Dict[str, Any]) -> float:
        """Calculate reward based on game state changes."""
        reward = 0.0

        if not self.prev_state:
            self.prev_state = state
            return 0.0

        # Reward for leveling up (big reward)
        level_diff = state["total_levels"] - self.prev_state.get("total_levels", 0)
        if level_diff > 0:
            reward += level_diff * 10.0

        # Reward for catching Pokemon (big reward)
        pokemon_diff = state["pokedex_owned"] - self.prev_state.get("pokedex_owned", 0)
        if pokemon_diff > 0:
            reward += pokemon_diff * 20.0

        # Reward for badges (huge reward)
        badge_diff = state["badges"] - self.prev_state.get("badges", 0)
        if badge_diff > 0:
            reward += badge_diff * 100.0

        # Reward for exploring new locations
        location = (state["map_id"], state["player_x"], state["player_y"])
        if location not in self.visited_locations:
            self.visited_locations.add(location)
            reward += 0.1  # Small exploration bonus

        # Small penalty for time (encourages efficiency)
        reward -= 0.001

        self.prev_state = state
        return reward

    def _get_observation(self) -> np.ndarray:
        """Get current screen as observation."""
        # Get screen as numpy array
        screen = self.pyboy.screen.ndarray
        return screen.copy()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Reset the environment."""
        super().reset(seed=seed)

        # Initialize emulator if needed
        if self.pyboy is None:
            self._init_pyboy()

        # Load save state if provided, otherwise reset
        if self.save_state and os.path.exists(self.save_state):
            with open(self.save_state, "rb") as f:
                self.pyboy.load_state(f)
        else:
            # Skip intro sequence (run for a bit to get past Nintendo logo)
            for _ in range(500):
                self.pyboy.tick()

        # Reset state tracking
        self.prev_state = {}
        self.visited_locations = set()
        self.step_count = 0

        obs = self._get_observation()
        info = {"game_state": self._get_game_state()}

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute action in the environment."""
        self.step_count += 1

        # Press button
        buttons = self.ACTIONS[action]
        for button in buttons:
            self.pyboy.button(button)

        # Run emulator for frame_skip frames
        for _ in range(self.frame_skip):
            self.pyboy.tick()

        # Release button
        for button in buttons:
            self.pyboy.button_release(button)

        # Get observation and state
        obs = self._get_observation()
        state = self._get_game_state()
        reward = self._calculate_reward(state)

        # Check if episode should end
        terminated = False
        truncated = self.step_count >= self.max_steps

        info = {"game_state": state}

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the game."""
        if self.render_mode == "rgb_array":
            return self._get_observation()
        elif self.render_mode == "human":
            # PyBoy handles rendering when not headless
            pass

    def close(self):
        """Clean up resources."""
        if self.pyboy is not None:
            self.pyboy.stop()
            self.pyboy = None


def make_pokemon_env(
    rom_path: str = "roms/pokemon_red.gb",
    frame_stack: int = 4,
    frame_height: int = 84,
    frame_width: int = 84,
    render_mode: Optional[str] = None,
) -> gym.Env:
    """
    Create a preprocessed Pokemon environment.

    Args:
        rom_path: Path to ROM file
        frame_stack: Number of frames to stack
        frame_height: Height to resize frames to
        frame_width: Width to resize frames to
        render_mode: "human" or "rgb_array"

    Returns:
        Preprocessed Pokemon environment
    """
    from .wrappers import (
        GrayscaleWrapper,
        ResizeWrapper,
        FrameStackWrapper,
        NormalizeWrapper,
    )

    # Create base environment
    env = PokemonRedEnv(
        rom_path=rom_path,
        render_mode=render_mode,
        headless=render_mode != "human",
    )

    # Apply preprocessing
    env = GrayscaleWrapper(env)
    env = ResizeWrapper(env, height=frame_height, width=frame_width)
    env = FrameStackWrapper(env, n_frames=frame_stack)
    env = NormalizeWrapper(env)

    return env
