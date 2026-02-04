"""
Real-time training dashboard using matplotlib.

Provides live visualization of training progress including:
- Reward curves
- Loss plots
- Episode statistics
- Live game preview (optional)
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import numpy as np
from collections import deque
from typing import Optional, Deque
import time


class TrainingDashboard:
    """
    Real-time matplotlib dashboard for monitoring training.

    Creates a figure with multiple subplots:
    - Reward over episodes (with rolling average)
    - Policy and value losses
    - Episode lengths
    - Entropy over time

    Usage:
        dashboard = TrainingDashboard()
        for episode in training_loop:
            dashboard.update(reward=score, length=steps, losses=loss_dict)
        dashboard.close()
    """

    def __init__(
        self,
        window_size: int = 100,
        update_interval: float = 0.5,
        figsize: tuple = (14, 10),
        demo_every: int = 25,
        total_timesteps: int = 1_000_000,
    ):
        """
        Initialize the dashboard.

        Args:
            window_size: Rolling window for averaging
            update_interval: Minimum seconds between display updates
            figsize: Figure size (width, height) in inches
            demo_every: Initial demo interval (0=disabled)
            total_timesteps: Initial training length target
        """
        self.window_size = window_size
        self.update_interval = update_interval
        self.last_update_time = 0

        # Data storage
        self.rewards: Deque[float] = deque(maxlen=10000)
        self.lengths: Deque[float] = deque(maxlen=10000)
        self.policy_losses: Deque[float] = deque(maxlen=10000)
        self.value_losses: Deque[float] = deque(maxlen=10000)
        self.entropies: Deque[float] = deque(maxlen=10000)
        self.timesteps: Deque[int] = deque(maxlen=10000)

        self.current_timestep = 0

        # Demo control
        self._skip_demo_requested = False
        self._demo_active = False
        self._demo_every = demo_every

        # Training length control
        self._total_timesteps = total_timesteps

        # Setup matplotlib
        plt.ion()  # Interactive mode
        self.fig, self.axes = plt.subplots(2, 2, figsize=figsize)
        self.fig.suptitle("PPO Training Dashboard", fontsize=14, fontweight="bold")

        # Initialize plots
        self._setup_plots()

        # Add controls
        self._setup_controls()

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)  # Make room for controls
        plt.show(block=False)
        plt.pause(0.1)

    def _setup_plots(self):
        """Initialize empty plots with labels and styling."""
        # Reward plot (top left)
        self.ax_reward = self.axes[0, 0]
        self.ax_reward.set_title("Episode Reward")
        self.ax_reward.set_xlabel("Episode")
        self.ax_reward.set_ylabel("Reward")
        self.ax_reward.grid(True, alpha=0.3)
        (self.line_reward,) = self.ax_reward.plot([], [], "b-", alpha=0.3, label="Raw")
        (self.line_reward_avg,) = self.ax_reward.plot(
            [], [], "b-", linewidth=2, label=f"Rolling Avg ({self.window_size})"
        )
        self.ax_reward.legend(loc="upper left")

        # Loss plot (top right)
        self.ax_loss = self.axes[0, 1]
        self.ax_loss.set_title("Training Losses")
        self.ax_loss.set_xlabel("Update")
        self.ax_loss.set_ylabel("Loss")
        self.ax_loss.grid(True, alpha=0.3)
        (self.line_policy,) = self.ax_loss.plot(
            [], [], "r-", linewidth=1.5, label="Policy"
        )
        (self.line_value,) = self.ax_loss.plot(
            [], [], "g-", linewidth=1.5, label="Value"
        )
        self.ax_loss.legend(loc="upper right")

        # Episode length plot (bottom left)
        self.ax_length = self.axes[1, 0]
        self.ax_length.set_title("Episode Length")
        self.ax_length.set_xlabel("Episode")
        self.ax_length.set_ylabel("Steps")
        self.ax_length.grid(True, alpha=0.3)
        (self.line_length,) = self.ax_length.plot([], [], "g-", alpha=0.3)
        (self.line_length_avg,) = self.ax_length.plot(
            [], [], "g-", linewidth=2, label=f"Rolling Avg"
        )
        self.ax_length.legend(loc="upper left")

        # Entropy plot (bottom right)
        self.ax_entropy = self.axes[1, 1]
        self.ax_entropy.set_title("Policy Entropy")
        self.ax_entropy.set_xlabel("Update")
        self.ax_entropy.set_ylabel("Entropy")
        self.ax_entropy.grid(True, alpha=0.3)
        (self.line_entropy,) = self.ax_entropy.plot([], [], "m-", linewidth=1.5)

    def _setup_controls(self):
        """Setup dashboard controls (sliders and skip button)."""
        # Demo interval slider (left side)
        self.demo_slider_ax = self.fig.add_axes([0.08, 0.02, 0.25, 0.025])
        self.demo_slider = Slider(
            self.demo_slider_ax,
            "",
            valmin=0,
            valmax=100,
            valinit=self._demo_every,
            valstep=5,
            color="lightblue",
        )
        self.demo_slider.on_changed(self._on_demo_slider_changed)
        self._update_demo_label()

        # Training length slider (center) - in thousands
        initial_k = self._total_timesteps // 1000
        self.length_slider_ax = self.fig.add_axes([0.38, 0.02, 0.25, 0.025])
        self.length_slider = Slider(
            self.length_slider_ax,
            "",
            valmin=50,
            valmax=5000,
            valinit=initial_k,
            valstep=50,
            color="lightgreen",
        )
        self.length_slider.on_changed(self._on_length_slider_changed)
        self._update_length_label()

        # Skip demo button (right side, initially hidden)
        self.skip_button_ax = self.fig.add_axes([0.68, 0.02, 0.12, 0.025])
        self.skip_button = Button(
            self.skip_button_ax,
            "Skip Demo",
            color="lightsalmon",
            hovercolor="salmon"
        )
        self.skip_button.on_clicked(self._on_skip_clicked)
        self.skip_button_ax.set_visible(False)

    def _on_demo_slider_changed(self, val):
        """Handle demo interval slider change."""
        self._demo_every = int(val)
        self._update_demo_label()

    def _on_length_slider_changed(self, val):
        """Handle training length slider change."""
        self._total_timesteps = int(val) * 1000
        self._update_length_label()

    def _update_demo_label(self):
        """Update demo slider label."""
        if self._demo_every == 0:
            self.demo_slider_ax.set_title("Demo: OFF", fontsize=9, loc="left")
        else:
            self.demo_slider_ax.set_title(f"Demo every {self._demo_every}", fontsize=9, loc="left")

    def _update_length_label(self):
        """Update training length slider label."""
        k = self._total_timesteps // 1000
        self.length_slider_ax.set_title(f"Train: {k}K steps", fontsize=9, loc="left")

    def _on_skip_clicked(self, event):
        """Handle skip button click."""
        self._skip_demo_requested = True

    @property
    def demo_every(self) -> int:
        """Get current demo interval setting."""
        return self._demo_every

    @demo_every.setter
    def demo_every(self, value: int):
        """Set demo interval and update slider."""
        self._demo_every = max(0, min(100, value))
        self.demo_slider.set_val(self._demo_every)

    @property
    def total_timesteps(self) -> int:
        """Get current training length target."""
        return self._total_timesteps

    @total_timesteps.setter
    def total_timesteps(self, value: int):
        """Set training length and update slider."""
        self._total_timesteps = max(50000, min(5_000_000, value))
        self.length_slider.set_val(self._total_timesteps // 1000)

    def demo_started(self):
        """Call when a demo episode starts."""
        self._demo_active = True
        self._skip_demo_requested = False
        self.skip_button_ax.set_visible(True)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def demo_ended(self):
        """Call when a demo episode ends."""
        self._demo_active = False
        self._skip_demo_requested = False
        self.skip_button_ax.set_visible(False)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def should_skip_demo(self) -> bool:
        """Check if user requested to skip the current demo."""
        # Process any pending events to catch button clicks
        if self._demo_active:
            self.fig.canvas.flush_events()
        return self._skip_demo_requested

    def update(
        self,
        reward: Optional[float] = None,
        length: Optional[float] = None,
        policy_loss: Optional[float] = None,
        value_loss: Optional[float] = None,
        entropy: Optional[float] = None,
        timestep: Optional[int] = None,
    ):
        """
        Update dashboard with new data.

        Args:
            reward: Episode reward (if episode ended)
            length: Episode length (if episode ended)
            policy_loss: Policy loss from training update
            value_loss: Value loss from training update
            entropy: Entropy from training update
            timestep: Current timestep
        """
        # Store data
        if reward is not None:
            self.rewards.append(reward)
        if length is not None:
            self.lengths.append(length)
        if policy_loss is not None:
            self.policy_losses.append(policy_loss)
        if value_loss is not None:
            self.value_losses.append(value_loss)
        if entropy is not None:
            self.entropies.append(entropy)
        if timestep is not None:
            self.current_timestep = timestep
            self.timesteps.append(timestep)

        # Throttle display updates
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return
        self.last_update_time = current_time

        self._refresh_display()

    def _refresh_display(self):
        """Redraw all plots with current data."""
        # Update reward plot
        if len(self.rewards) > 0:
            episodes = np.arange(len(self.rewards))
            rewards = np.array(self.rewards)

            self.line_reward.set_data(episodes, rewards)

            # Rolling average
            avg_rewards = self._rolling_average(rewards, self.window_size)
            self.line_reward_avg.set_data(episodes, avg_rewards)

            self.ax_reward.relim()
            self.ax_reward.autoscale_view()

        # Update loss plot
        if len(self.policy_losses) > 0:
            updates = np.arange(len(self.policy_losses))

            self.line_policy.set_data(updates, np.array(self.policy_losses))
            self.line_value.set_data(updates, np.array(self.value_losses))

            self.ax_loss.relim()
            self.ax_loss.autoscale_view()

        # Update length plot
        if len(self.lengths) > 0:
            episodes = np.arange(len(self.lengths))
            lengths = np.array(self.lengths)

            self.line_length.set_data(episodes, lengths)

            avg_lengths = self._rolling_average(lengths, self.window_size)
            self.line_length_avg.set_data(episodes, avg_lengths)

            self.ax_length.relim()
            self.ax_length.autoscale_view()

        # Update entropy plot
        if len(self.entropies) > 0:
            updates = np.arange(len(self.entropies))
            self.line_entropy.set_data(updates, np.array(self.entropies))

            self.ax_entropy.relim()
            self.ax_entropy.autoscale_view()

        # Update title with stats
        if len(self.rewards) > 0:
            recent_avg = np.mean(list(self.rewards)[-self.window_size :])
            self.fig.suptitle(
                f"PPO Training Dashboard | "
                f"Episodes: {len(self.rewards)} | "
                f"Timesteps: {self.current_timestep:,} | "
                f"Avg Reward: {recent_avg:.1f}",
                fontsize=12,
            )

        # Redraw
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    @staticmethod
    def _rolling_average(data: np.ndarray, window: int) -> np.ndarray:
        """Compute rolling average with same output size as input."""
        if len(data) < window:
            window = len(data)

        # Cumulative sum for efficient rolling average
        cumsum = np.cumsum(np.insert(data, 0, 0))
        rolling = (cumsum[window:] - cumsum[:-window]) / window

        # Pad beginning to match size
        pad = data[: len(data) - len(rolling)].cumsum() / np.arange(
            1, len(data) - len(rolling) + 1
        )
        return np.concatenate([pad, rolling])

    def save(self, filepath: str):
        """Save the current figure to a file."""
        self.fig.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"Dashboard saved to {filepath}")

    def export_state(self) -> dict:
        """Export dashboard data for persistence (e.g., when saving session)."""
        return {
            "rewards": list(self.rewards),
            "lengths": list(self.lengths),
            "policy_losses": list(self.policy_losses),
            "value_losses": list(self.value_losses),
            "entropies": list(self.entropies),
            "timesteps": list(self.timesteps),
            "current_timestep": self.current_timestep,
            "demo_every": self._demo_every,
            "total_timesteps": self._total_timesteps,
        }

    def import_state(self, state: dict):
        """Restore dashboard data from saved state."""
        if state is None:
            return

        # Restore data
        self.rewards.extend(state.get("rewards", []))
        self.lengths.extend(state.get("lengths", []))
        self.policy_losses.extend(state.get("policy_losses", []))
        self.value_losses.extend(state.get("value_losses", []))
        self.entropies.extend(state.get("entropies", []))
        self.timesteps.extend(state.get("timesteps", []))
        self.current_timestep = state.get("current_timestep", 0)

        # Restore settings
        if "demo_every" in state:
            self.demo_every = state["demo_every"]
        if "total_timesteps" in state:
            self.total_timesteps = state["total_timesteps"]

        # Refresh display to show restored data
        self._refresh_display()

    def close(self):
        """Close the dashboard."""
        plt.ioff()
        plt.close(self.fig)


class LiveGameViewer:
    """
    Simple live viewer for watching the agent play.

    Displays game frames in a matplotlib window.
    """

    def __init__(self, figsize: tuple = (6, 6)):
        """Initialize the viewer."""
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.axis("off")
        self.ax.set_title("Agent Playing")
        self.img = None
        plt.show(block=False)

    def show_frame(self, frame: np.ndarray):
        """
        Display a game frame.

        Args:
            frame: Game frame (H, W) or (H, W, C)
        """
        if self.img is None:
            # First frame
            if frame.ndim == 2:
                self.img = self.ax.imshow(frame, cmap="gray")
            else:
                self.img = self.ax.imshow(frame)
        else:
            # Update existing
            self.img.set_data(frame)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self):
        """Close the viewer."""
        plt.ioff()
        plt.close(self.fig)
