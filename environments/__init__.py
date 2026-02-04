"""
Environment module for RL training.

Provides preprocessed game environments ready for neural network training.

Supported games:
    - Atari: pong, breakout, spaceinvaders
    - Nintendo: mario
    - Game Boy: pokemon (requires ROM)
    - Sega: sonic (requires ROM)
"""

from .wrappers import (
    GrayscaleWrapper,
    ResizeWrapper,
    FrameStackWrapper,
    NormalizeWrapper,
    MaxAndSkipWrapper,
)
from .make_env import make_env, make_vec_env

__all__ = [
    "GrayscaleWrapper",
    "ResizeWrapper",
    "FrameStackWrapper",
    "NormalizeWrapper",
    "MaxAndSkipWrapper",
    "make_env",
    "make_vec_env",
]
