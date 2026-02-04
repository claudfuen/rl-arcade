#!/usr/bin/env python3
"""
Reinforcement Learning Game Agent - Entry Point

Train neural networks to play classic games using PPO!

Usage:
    # Train with real-time game visualization
    python3 main.py train --env pong --render

    # Train with live training graphs
    python3 main.py train --env pong --dashboard

    # Train with BOTH game view and graphs
    python3 main.py train --env pong --render --dashboard

    # Quick demo (watch agent learn Pong)
    python3 main.py demo

    # Watch a trained agent play
    python3 main.py play --env breakout --checkpoint checkpoints/best_model.pt

Supported environments:
    - pong: Atari Pong (easiest, quick to train)
    - breakout: Atari Breakout (good starting point)
    - spaceinvaders: Atari Space Invaders
    - mario: Super Mario Bros (challenging)
    - pokemon: Pokemon Red/Blue (requires ROM in roms/)
    - sonic: Sonic the Hedgehog (requires gym-retro setup)
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PPOConfig, TrainingConfig, EnvConfig, ENV_IDS
from utils import set_seed, get_device
from sessions import Session, SessionManager


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train RL agents to play classic games",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Demo command (quick start)
    demo_parser = subparsers.add_parser(
        "demo", help="Quick demo - watch an agent learn Pong with live visuals"
    )
    demo_parser.add_argument(
        "--env",
        type=str,
        default="pong",
        choices=["pong", "breakout", "spaceinvaders", "mario", "pokemon", "sonic"],
        help="Environment (default: pong)",
    )

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new agent")
    train_parser.add_argument(
        "--env",
        type=str,
        default="breakout",
        choices=["pong", "breakout", "spaceinvaders", "mario", "pokemon", "sonic"],
        help="Environment to train on (default: breakout)",
    )
    train_parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="Total training timesteps (default: 1M)",
    )
    train_parser.add_argument(
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel environments (default: 8)",
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        default=2.5e-4,
        help="Learning rate (default: 2.5e-4)",
    )
    train_parser.add_argument(
        "--entropy",
        type=float,
        default=0.02,
        help="Entropy coefficient for exploration (default: 0.02, higher=more exploration)",
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    train_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on (cpu, cuda, mps). Auto-detected if not specified.",
    )
    train_parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for TensorBoard logs (default: logs/)",
    )
    train_parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for model checkpoints (default: checkpoints/)",
    )
    train_parser.add_argument(
        "--render",
        action="store_true",
        help="[SLOW] Show game continuously during training",
    )
    train_parser.add_argument(
        "--demo-every",
        type=int,
        default=25,
        help="Play a demo game every N updates (default: 25, 0=disabled)",
    )
    train_parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Show live training graphs (rewards, losses)",
    )
    train_parser.add_argument(
        "--render-every",
        type=int,
        default=1,
        help="For --render mode: render every Nth frame (default: 1)",
    )

    # Play command
    play_parser = subparsers.add_parser("play", help="Watch a trained agent play")
    play_parser.add_argument(
        "--env",
        type=str,
        default="breakout",
        choices=["pong", "breakout", "spaceinvaders", "mario", "pokemon", "sonic"],
        help="Environment to play",
    )
    play_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    play_parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to play (default: 5)",
    )
    play_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on",
    )

    # Resume command
    resume_parser = subparsers.add_parser(
        "resume", help="Resume interrupted training session"
    )
    resume_parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Session ID (or prefix) to resume",
    )
    resume_parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Legacy checkpoint to migrate (requires --game)",
    )
    resume_parser.add_argument(
        "--game",
        type=str,
        choices=["pong", "breakout", "spaceinvaders", "mario", "pokemon", "sonic"],
        help="Game for legacy checkpoint migration",
    )
    resume_parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Show live training graphs",
    )
    resume_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on",
    )

    # Sessions command
    sessions_parser = subparsers.add_parser(
        "sessions", help="List training sessions"
    )
    sessions_parser.add_argument(
        "--status",
        type=str,
        choices=["in_progress", "completed", "all"],
        default="all",
        help="Filter by status (default: all)",
    )

    return parser.parse_args()


def demo(args):
    """Quick demo with periodic game visualization."""
    print("\n" + "=" * 60)
    print("  🎮 RL Demo - Watch an Agent Learn!")
    print("=" * 60)
    print("\nThis will:")
    print("  • Train fast in the background")
    print("  • Show a demo game every 25 updates")
    print("  • Display live training graphs")
    print("\nPress Ctrl+C to stop at any time.\n")

    # Run training with visualization
    class DemoArgs:
        env = args.env
        timesteps = 200_000
        n_envs = 8
        lr = 2.5e-4
        entropy = 0.02
        seed = None
        device = None
        log_dir = "logs"
        checkpoint_dir = "checkpoints"
        render = False  # Don't render continuously
        demo_every = 25  # Show demo game every 25 updates
        render_every = 1
        dashboard = True

    train(DemoArgs())


def train(args):
    """Run training with optional visualization."""
    from training import Trainer, CheckpointCallback, EpisodeLoggerCallback, SessionCheckpointCallback

    print("\n" + "=" * 60)
    print("  RL Game Agent Training")
    print("=" * 60)

    # Set seed if provided
    if args.seed is not None:
        print(f"Setting random seed: {args.seed}")
        set_seed(args.seed)

    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")

    if args.render:
        print("Real-time game rendering: ENABLED")
    if args.dashboard:
        print("Live training dashboard: ENABLED")

    # Create configurations
    ppo_config = PPOConfig(
        learning_rate=args.lr,
        ent_coef=getattr(args, 'entropy', 0.02),
    )

    env_config = EnvConfig(
        env_name=args.env,
        n_envs=args.n_envs,
    )

    training_config = TrainingConfig(
        total_timesteps=args.timesteps,
        device=device,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        render_training=args.render,
        render_every=getattr(args, 'render_every', 1),
        demo_every=getattr(args, 'demo_every', 0),
    )

    # Create session for tracking
    manager = SessionManager()
    session = manager.create_session(
        game=args.env,
        env_config=env_config,
        ppo_config=ppo_config,
        training_config=training_config,
    )
    print(f"Session created: {session.session_id}")

    # Create callbacks - session callback handles checkpointing
    callbacks = [
        SessionCheckpointCallback(session),
        EpisodeLoggerCallback(log_interval=20),
    ]

    # Create trainer and run
    trainer = Trainer(
        env_config=env_config,
        ppo_config=ppo_config,
        training_config=training_config,
        callbacks=callbacks,
        show_dashboard=getattr(args, 'dashboard', False),
        session=session,
    )

    try:
        metrics = trainer.train()

        print("\n" + "=" * 60)
        print("  Training Complete!")
        print("=" * 60)
        print(f"Session: {session.session_id}")
        print(f"Total timesteps: {metrics['total_timesteps']:,}")
        print(f"Total episodes: {metrics['total_episodes']}")
        print(f"Final mean reward: {metrics['mean_reward']:.2f}")
        print(f"Training time: {metrics['training_time']:.1f}s")
        print(f"\nCheckpoints saved to: {session.checkpoint_dir}")
        print(f"TensorBoard logs: {session.log_dir}")
        print("\nView training progress with:")
        print(f"  tensorboard --logdir {session.log_dir}")
        print(f"\nTo resume this session:")
        print(f"  python main.py resume --session {session.session_id}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print(f"Session saved: {session.session_id}")
        print(f"Resume with: python main.py resume --session {session.session_id}")


def play(args):
    """Watch a trained agent play."""
    import torch
    import time
    from environments import make_env
    from agents import ActorCritic

    print("\n" + "=" * 60)
    print("  Watching Trained Agent Play")
    print("=" * 60)

    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return

    device = get_device(args.device)
    print(f"Using device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")

    # Create environment with render mode
    import gymnasium as gym

    if args.env == "mario":
        # Mario doesn't support render_mode in make, need to call render()
        env = make_env(args.env)
        has_render = True
    else:
        # For Atari, we need to recreate with render_mode
        from environments.wrappers import (
            GrayscaleWrapper,
            ResizeWrapper,
            FrameStackWrapper,
            NormalizeWrapper,
            MaxAndSkipWrapper,
        )

        base_env = gym.make(ENV_IDS[args.env], render_mode="human")
        base_env = MaxAndSkipWrapper(base_env, skip=4)
        base_env = GrayscaleWrapper(base_env)
        base_env = ResizeWrapper(base_env)
        base_env = FrameStackWrapper(base_env)
        env = NormalizeWrapper(base_env)
        has_render = False  # render_mode="human" auto-renders

    # Load model
    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape
    n_channels = obs_shape[-1]

    model = ActorCritic(n_actions=n_actions, n_input_channels=n_channels).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["network_state_dict"])
    model.eval()

    print(f"\nPlaying {args.episodes} episodes...\n")

    total_rewards = []

    for episode in range(args.episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0

        while not done:
            # Get action from model
            with torch.no_grad():
                obs_tensor = torch.tensor(obs[None], dtype=torch.float32, device=device)
                action_logits, _ = model(obs_tensor)
                action = action_logits.argmax(dim=-1).item()

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1

            if has_render:
                env.render()
                time.sleep(0.02)  # Slow down for visibility

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.0f}, Steps = {steps}")

    env.close()

    print(f"\n{'=' * 40}")
    print(f"Average reward: {sum(total_rewards) / len(total_rewards):.2f}")
    print(f"Best reward: {max(total_rewards):.0f}")


def resume(args):
    """Resume an interrupted training session."""
    from training import Trainer, EpisodeLoggerCallback

    print("\n" + "=" * 60)
    print("  Resume Training Session")
    print("=" * 60)

    manager = SessionManager()

    # Handle legacy checkpoint migration
    if args.checkpoint:
        if not args.game:
            print("Error: --game is required when migrating legacy checkpoint")
            sys.exit(1)

        session = manager.migrate_legacy_checkpoint(
            checkpoint_path=args.checkpoint,
            game=args.game,
        )
    # Resume by session ID
    elif args.session:
        session = manager.get_session_by_prefix(args.session)
        if session is None:
            sys.exit(1)
    # Interactive selection
    else:
        sessions = manager.get_incomplete_sessions()
        if not sessions:
            print("No incomplete sessions found.")
            print("\nStart a new training session with:")
            print("  python main.py train --env pong")
            sys.exit(0)

        print("\nIncomplete sessions:\n")
        for i, s in enumerate(sessions, 1):
            progress = s.progress_percent()
            best = s.progress.get("best_reward", "N/A")
            if isinstance(best, float) and best != float("-inf"):
                best = f"{best:.1f}"
            else:
                best = "N/A"
            print(f"  {i}. {s.session_id}")
            print(f"     Game: {s.game} | Progress: {progress:.1f}% | Best reward: {best}")

        while True:
            try:
                choice = input(f"\nSelect session (1-{len(sessions)}): ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(sessions):
                    session = sessions[idx]
                    break
            except ValueError:
                pass
            print("Invalid choice.")

    print(f"\nResuming session: {session.session_id}")
    print(f"Game: {session.game}")
    print(f"Progress: {session.progress_percent():.1f}%")

    # Get device
    device = get_device(args.device)

    # Update training config device
    training_config = session.get_training_config()
    training_config.device = device

    # Create trainer from session
    callbacks = [EpisodeLoggerCallback(log_interval=20)]
    trainer = Trainer.from_session(
        session=session,
        show_dashboard=args.dashboard,
        callbacks=callbacks,
    )

    try:
        metrics = trainer.train()

        print("\n" + "=" * 60)
        print("  Training Complete!")
        print("=" * 60)
        print(f"Session: {session.session_id}")
        print(f"Total timesteps: {metrics['total_timesteps']:,}")
        print(f"Final mean reward: {metrics['mean_reward']:.2f}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted.")
        print(f"Session saved: {session.session_id}")
        print(f"Resume with: python main.py resume --session {session.session_id}")


def sessions(args):
    """List training sessions."""
    print("\n" + "=" * 60)
    print("  Training Sessions")
    print("=" * 60)

    manager = SessionManager()

    status_filter = None if args.status == "all" else args.status
    all_sessions = manager.list_sessions(status_filter=status_filter)

    if not all_sessions:
        print("\nNo sessions found.")
        print("\nStart a new training session with:")
        print("  python main.py train --env pong")
        return

    print(f"\n{'ID':<35} {'Game':<12} {'Status':<12} {'Progress':<10} {'Best':<10}")
    print("-" * 80)

    for s in all_sessions:
        progress = f"{s.progress_percent():.1f}%"
        best = s.progress.get("best_reward", float("-inf"))
        if isinstance(best, float) and best != float("-inf"):
            best_str = f"{best:.1f}"
        else:
            best_str = "N/A"

        status_icon = "..." if s.status == "in_progress" else "done"
        print(f"{s.session_id:<35} {s.game:<12} {status_icon:<12} {progress:<10} {best_str:<10}")

    print()
    incomplete = [s for s in all_sessions if s.status == "in_progress"]
    if incomplete:
        print(f"Resume incomplete sessions with:")
        print(f"  python main.py resume --session <session_id>")


def main():
    """Main entry point."""
    args = parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "play":
        play(args)
    elif args.command == "demo":
        demo(args)
    elif args.command == "resume":
        resume(args)
    elif args.command == "sessions":
        sessions(args)
    else:
        print("Please specify a command: train, play, demo, resume, or sessions")
        print("\nQuick start:")
        print("  python3 main.py demo          # Watch agent learn with live visuals")
        print("  python3 main.py train --help  # See all training options")
        print("  python3 main.py sessions      # List training sessions")
        print("  python3 main.py resume        # Resume interrupted training")
        sys.exit(1)


if __name__ == "__main__":
    main()
