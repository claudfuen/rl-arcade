"""
Session management for training runs.

This module provides a robust system for tracking, persisting, and resuming
training sessions. Each session captures:
- Game and configuration used
- Training progress (timesteps, episodes, rewards)
- RNG states for exact reproducibility
- Checkpoints organized by session

Directory structure:
    sessions/
    ├── session_index.json          # Quick lookup of all sessions
    └── <session_id>/
        ├── session.json            # Full metadata
        ├── checkpoints/
        │   ├── checkpoint_50.pt
        │   └── latest.pt
        └── logs/
"""

import os
import json
import random
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any
import numpy as np
import torch

from config import (
    PPOConfig, TrainingConfig, EnvConfig,
    config_to_dict, config_from_dict,
)


# Default sessions directory
SESSIONS_DIR = "sessions"


def capture_rng_states() -> Dict[str, Any]:
    """Capture current RNG states for all sources of randomness."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().tolist(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_states(states: Dict[str, Any]) -> None:
    """Restore RNG states from a captured dictionary."""
    if states is None:
        return

    if "python" in states and states["python"] is not None:
        random.setstate(states["python"])

    if "numpy" in states and states["numpy"] is not None:
        # numpy state is a tuple, may need conversion from list
        np_state = states["numpy"]
        if isinstance(np_state, list):
            np_state = tuple(
                np.array(x) if isinstance(x, list) else x
                for x in np_state
            )
        np.random.set_state(np_state)

    if "torch" in states and states["torch"] is not None:
        torch_state = torch.tensor(states["torch"], dtype=torch.uint8)
        torch.set_rng_state(torch_state)

    if "torch_cuda" in states and states["torch_cuda"] is not None:
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(states["torch_cuda"])


@dataclass
class SessionProgress:
    """Training progress tracking."""
    timesteps: int = 0
    episodes: int = 0
    updates: int = 0
    best_reward: float = float("-inf")


@dataclass
class Session:
    """
    Represents a training session with full state for resume capability.

    A session captures everything needed to:
    1. Understand what was trained (game, configs)
    2. Resume training exactly where it left off
    3. Track progress and best results
    """

    session_id: str
    game: str
    status: str = "in_progress"  # "in_progress" | "completed"

    # Configurations (stored as dicts for JSON serialization)
    env_config: Dict = field(default_factory=dict)
    ppo_config: Dict = field(default_factory=dict)
    training_config: Dict = field(default_factory=dict)

    # Progress tracking
    progress: Dict = field(default_factory=lambda: asdict(SessionProgress()))

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Session directory paths (computed, not stored)
    _base_dir: str = field(default=SESSIONS_DIR, repr=False)

    @property
    def session_dir(self) -> str:
        """Path to session directory."""
        return os.path.join(self._base_dir, self.session_id)

    @property
    def checkpoint_dir(self) -> str:
        """Path to checkpoints directory."""
        return os.path.join(self.session_dir, "checkpoints")

    @property
    def log_dir(self) -> str:
        """Path to logs directory."""
        return os.path.join(self.session_dir, "logs")

    @classmethod
    def create_new(
        cls,
        game: str,
        env_config: EnvConfig,
        ppo_config: PPOConfig,
        training_config: TrainingConfig,
        base_dir: str = SESSIONS_DIR,
    ) -> "Session":
        """Create a new session with generated ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"{game}_{timestamp}"

        session = cls(
            session_id=session_id,
            game=game,
            env_config=config_to_dict(env_config),
            ppo_config=config_to_dict(ppo_config),
            training_config=config_to_dict(training_config),
            _base_dir=base_dir,
        )

        # Create directories
        os.makedirs(session.checkpoint_dir, exist_ok=True)
        os.makedirs(session.log_dir, exist_ok=True)

        # Save initial session file
        session.save()

        return session

    def save(self) -> None:
        """Save session metadata to session.json."""
        self.updated_at = datetime.now().isoformat()

        session_file = os.path.join(self.session_dir, "session.json")
        os.makedirs(self.session_dir, exist_ok=True)

        data = {
            "session_id": self.session_id,
            "game": self.game,
            "status": self.status,
            "env_config": self.env_config,
            "ppo_config": self.ppo_config,
            "training_config": self.training_config,
            "progress": self.progress,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

        with open(session_file, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, session_id: str, base_dir: str = SESSIONS_DIR) -> "Session":
        """Load session from disk."""
        session_file = os.path.join(base_dir, session_id, "session.json")

        if not os.path.exists(session_file):
            raise FileNotFoundError(f"Session not found: {session_id}")

        with open(session_file, "r") as f:
            data = json.load(f)

        return cls(
            session_id=data["session_id"],
            game=data["game"],
            status=data.get("status", "in_progress"),
            env_config=data.get("env_config", {}),
            ppo_config=data.get("ppo_config", {}),
            training_config=data.get("training_config", {}),
            progress=data.get("progress", asdict(SessionProgress())),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            _base_dir=base_dir,
        )

    def get_checkpoint_path(self, name: str = "latest") -> str:
        """Get path to a checkpoint file."""
        if not name.endswith(".pt"):
            name = f"{name}.pt"
        return os.path.join(self.checkpoint_dir, name)

    def update_progress(
        self,
        timesteps: int = None,
        episodes: int = None,
        updates: int = None,
        best_reward: float = None,
    ) -> None:
        """Update progress and save."""
        if timesteps is not None:
            self.progress["timesteps"] = timesteps
        if episodes is not None:
            self.progress["episodes"] = episodes
        if updates is not None:
            self.progress["updates"] = updates
        if best_reward is not None:
            if best_reward > self.progress.get("best_reward", float("-inf")):
                self.progress["best_reward"] = best_reward
        self.save()

    def mark_completed(self) -> None:
        """Mark session as completed."""
        self.status = "completed"
        self.save()

    def get_env_config(self) -> EnvConfig:
        """Get EnvConfig object from stored dict."""
        return config_from_dict(EnvConfig, self.env_config)

    def get_ppo_config(self) -> PPOConfig:
        """Get PPOConfig object from stored dict."""
        return config_from_dict(PPOConfig, self.ppo_config)

    def get_training_config(self) -> TrainingConfig:
        """Get TrainingConfig object from stored dict."""
        config = config_from_dict(TrainingConfig, self.training_config)
        # Override paths to use session directories
        config.checkpoint_dir = self.checkpoint_dir
        config.log_dir = self.log_dir
        return config

    def progress_percent(self) -> float:
        """Calculate training progress percentage."""
        total = self.training_config.get("total_timesteps", 1_000_000)
        current = self.progress.get("timesteps", 0)
        return min(100.0, (current / total) * 100)

    def save_dashboard_state(self, state: dict) -> None:
        """Save dashboard state to session directory."""
        dashboard_file = os.path.join(self.session_dir, "dashboard_state.json")
        with open(dashboard_file, "w") as f:
            json.dump(state, f)
        num_episodes = len(state.get("rewards", []))
        print(f"  Dashboard state saved: {num_episodes} episodes -> {dashboard_file}")

    def load_dashboard_state(self) -> dict:
        """Load dashboard state from session directory."""
        dashboard_file = os.path.join(self.session_dir, "dashboard_state.json")
        if os.path.exists(dashboard_file):
            with open(dashboard_file, "r") as f:
                return json.load(f)
        print(f"  No dashboard state at: {dashboard_file}")
        return None


class SessionManager:
    """
    Manager for discovering, creating, and resuming sessions.
    """

    def __init__(self, base_dir: str = SESSIONS_DIR):
        self.base_dir = base_dir
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        """Ensure sessions directory exists."""
        os.makedirs(self.base_dir, exist_ok=True)

    def list_sessions(self, status_filter: str = None) -> List[Session]:
        """
        List all sessions, optionally filtered by status.

        Args:
            status_filter: "in_progress", "completed", or None for all

        Returns:
            List of Session objects, sorted by updated_at descending
        """
        sessions = []

        if not os.path.exists(self.base_dir):
            return sessions

        for entry in os.listdir(self.base_dir):
            session_dir = os.path.join(self.base_dir, entry)
            session_file = os.path.join(session_dir, "session.json")

            if os.path.isdir(session_dir) and os.path.exists(session_file):
                try:
                    session = Session.load(entry, self.base_dir)
                    if status_filter is None or session.status == status_filter:
                        sessions.append(session)
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Could not load session {entry}: {e}")

        # Sort by updated_at, most recent first
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions

    def get_incomplete_sessions(self) -> List[Session]:
        """Get all in-progress sessions."""
        return self.list_sessions(status_filter="in_progress")

    def create_session(
        self,
        game: str,
        env_config: EnvConfig,
        ppo_config: PPOConfig,
        training_config: TrainingConfig,
    ) -> Session:
        """Create a new training session."""
        return Session.create_new(
            game=game,
            env_config=env_config,
            ppo_config=ppo_config,
            training_config=training_config,
            base_dir=self.base_dir,
        )

    def load_session(self, session_id: str) -> Session:
        """Load a session by ID."""
        return Session.load(session_id, self.base_dir)

    def migrate_legacy_checkpoint(
        self,
        checkpoint_path: str,
        game: str,
        env_config: EnvConfig = None,
        ppo_config: PPOConfig = None,
        training_config: TrainingConfig = None,
    ) -> Session:
        """
        Create a session from a legacy checkpoint file.

        This allows resuming training from old-format checkpoints
        that don't have session metadata.

        Args:
            checkpoint_path: Path to legacy checkpoint file
            game: Game name (required, as legacy checkpoints don't store this)
            env_config: Override env config (uses defaults if None)
            ppo_config: Override ppo config (uses defaults if None)
            training_config: Override training config (uses defaults if None)

        Returns:
            New Session with checkpoint copied in
        """
        import shutil

        # Load checkpoint to get timesteps
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        timesteps = checkpoint.get("num_timesteps", 0)

        # Use defaults if not provided
        env_config = env_config or EnvConfig(env_name=game)
        ppo_config = ppo_config or PPOConfig()
        training_config = training_config or TrainingConfig()

        # Create session
        session = self.create_session(
            game=game,
            env_config=env_config,
            ppo_config=ppo_config,
            training_config=training_config,
        )

        # Copy checkpoint to session
        latest_path = session.get_checkpoint_path("latest")
        shutil.copy2(checkpoint_path, latest_path)

        # Update progress
        session.update_progress(timesteps=timesteps)

        print(f"Migrated legacy checkpoint to session: {session.session_id}")
        print(f"  Timesteps: {timesteps:,}")

        return session

    def get_session_by_prefix(self, prefix: str) -> Optional[Session]:
        """Find a session by ID prefix (for CLI convenience)."""
        sessions = self.list_sessions()
        matches = [s for s in sessions if s.session_id.startswith(prefix)]

        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            print(f"Multiple sessions match '{prefix}':")
            for s in matches:
                print(f"  - {s.session_id}")
            return None
        else:
            print(f"No session found matching '{prefix}'")
            return None
