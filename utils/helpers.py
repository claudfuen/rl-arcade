"""
Miscellaneous utility functions for RL training.
"""

import random
import numpy as np
import torch
from typing import Callable, Optional


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.

    Note: Full reproducibility in RL is difficult due to:
    - Asynchronous environment execution
    - GPU non-determinism
    - Different library versions

    But setting seeds helps reduce variance between runs.

    Args:
        seed: Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Additional settings for reproducibility (may hurt performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device: Optional[str] = None) -> str:
    """
    Get the best available device.

    Args:
        device: Preferred device ("cpu", "cuda", "mps", or None for auto)

    Returns:
        Device string
    """
    if device is not None:
        return device

    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Create a function that returns a linearly decreasing value.

    Used for learning rate scheduling:
    - Start with initial_value
    - Linearly decrease to 0 as progress goes from 0 to 1

    Args:
        initial_value: Starting value

    Returns:
        Function that takes progress (0-1) and returns scheduled value

    Example:
        lr_schedule = linear_schedule(2.5e-4)
        current_lr = lr_schedule(0.5)  # Returns 1.25e-4 (halfway)
    """

    def schedule(progress_remaining: float) -> float:
        """
        Args:
            progress_remaining: Fraction of training remaining (1 -> 0)
        """
        return progress_remaining * initial_value

    return schedule


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Calculate explained variance.

    Explained variance is a measure of how well the value function
    predictions match the actual returns:
    - 1.0 = perfect predictions
    - 0.0 = predictions no better than mean
    - < 0 = predictions worse than mean

    This is useful for diagnosing value function training:
    - Low explained variance = value function isn't learning
    - High explained variance = good advantage estimates

    Args:
        y_pred: Predicted values
        y_true: True values

    Returns:
        Explained variance score
    """
    var_y = np.var(y_true)
    if var_y == 0:
        return float("nan")
    return 1 - np.var(y_true - y_pred) / var_y


def smooth_rewards(rewards: list, window: int = 100) -> np.ndarray:
    """
    Compute smoothed rewards using exponential moving average.

    Args:
        rewards: List of episode rewards
        window: Smoothing window size

    Returns:
        Smoothed reward array
    """
    if len(rewards) == 0:
        return np.array([])

    smoothed = np.zeros(len(rewards))
    smoothed[0] = rewards[0]

    alpha = 2 / (window + 1)
    for i in range(1, len(rewards)):
        smoothed[i] = alpha * rewards[i] + (1 - alpha) * smoothed[i - 1]

    return smoothed


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def format_number(n: int) -> str:
    """Format large numbers with K/M suffixes."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    else:
        return str(n)


class RunningMeanStd:
    """
    Compute running mean and standard deviation.

    Used for observation normalization during training.
    Updates statistics incrementally without storing all data.
    """

    def __init__(self, shape: tuple = ()):
        """
        Args:
            shape: Shape of the data to track
        """
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4  # Small value to avoid division by zero

    def update(self, x: np.ndarray):
        """
        Update statistics with new batch of data.

        Uses Welford's online algorithm for numerical stability.

        Args:
            x: Batch of observations
        """
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_count: int,
    ):
        """Update from precomputed moments."""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        """
        Normalize observations using running statistics.

        Args:
            x: Observations to normalize
            clip: Clip normalized values to [-clip, clip]

        Returns:
            Normalized observations
        """
        normalized = (x - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(normalized, -clip, clip)
