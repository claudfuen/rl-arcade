#!/usr/bin/env python3
"""
Interactive CLI for RL Game Agent Training.

Just run: python3 cli.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def clear_screen():
    os.system('clear' if os.name != 'nt' else 'cls')


def print_header():
    print("\n" + "=" * 50)
    print("  🎮 RL Game Agent Trainer")
    print("=" * 50)


def get_choice(prompt, options):
    """Get user choice from numbered options."""
    print(f"\n{prompt}\n")
    for i, (key, label) in enumerate(options.items(), 1):
        print(f"  {i}. {label}")

    while True:
        try:
            choice = input("\nEnter number: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return list(options.keys())[idx]
            print("Invalid choice, try again.")
        except ValueError:
            print("Please enter a number.")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            sys.exit(0)


def get_number(prompt, default, min_val=0, max_val=None):
    """Get a number from user with default."""
    while True:
        try:
            value = input(f"{prompt} [{default}]: ").strip()
            if not value:
                return default
            value = int(value)
            if value < min_val:
                print(f"Must be at least {min_val}")
                continue
            if max_val and value > max_val:
                print(f"Must be at most {max_val}")
                continue
            return value
        except ValueError:
            print("Please enter a number.")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            sys.exit(0)


def confirm(prompt, default=True):
    """Get yes/no confirmation."""
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        response = input(f"{prompt} {suffix}: ").strip().lower()
        if not response:
            return default
        return response in ('y', 'yes')
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)


def main_menu():
    """Main interactive menu."""
    clear_screen()
    print_header()

    action = get_choice("What would you like to do?", {
        "train": "🏋️  Train a new agent",
        "play": "🎬  Watch a trained agent play",
        "quick": "⚡  Quick demo (recommended for first time)",
    })

    if action == "quick":
        return quick_demo()
    elif action == "train":
        return train_menu()
    elif action == "play":
        return play_menu()


def quick_demo():
    """Quick demo with good defaults."""
    clear_screen()
    print_header()
    print("\n⚡ Quick Demo Mode\n")

    env = get_choice("Pick a game:", {
        "pong": "🏓 Pong (easiest, learns fastest)",
        "breakout": "🧱 Breakout (medium difficulty)",
        "spaceinvaders": "👾 Space Invaders (harder)",
        "mario": "🍄 Super Mario Bros (hardest)",
    })

    print(f"\n🚀 Starting {env.title()} training with demo every 25 updates...")
    print("   Watch the agent improve over time!")
    print("   Press Ctrl+C to stop.\n")

    run_training(
        env=env,
        timesteps=300_000,
        n_envs=8,
        demo_every=25,
        dashboard=True,
    )


def train_menu():
    """Training configuration menu."""
    clear_screen()
    print_header()
    print("\n🏋️  Training Configuration\n")

    # Game selection
    env = get_choice("Pick a game:", {
        "pong": "🏓 Pong (easiest)",
        "breakout": "🧱 Breakout",
        "spaceinvaders": "👾 Space Invaders",
        "mario": "🍄 Super Mario Bros",
    })

    # Duration
    print("\n📊 Training Duration")
    print("   Rough guide: 100k=quick test, 500k=decent, 1M+=good results")
    timesteps = get_number("Timesteps (in thousands)", 500, min_val=10, max_val=10000) * 1000

    # Visualization
    print("\n👁️  Visualization")
    dashboard = confirm("Show live training graphs?", default=True)
    demo_every = get_number("Show demo game every N updates (0=never)", 25, min_val=0)

    # Performance
    print("\n⚡ Performance")
    n_envs = get_number("Parallel environments (more=faster, uses more RAM)", 8, min_val=1, max_val=32)

    # Exploration
    print("\n🎲 Exploration")
    print("   Higher = more random exploration (good for learning)")
    entropy = get_number("Entropy coefficient (1-10, default 2)", 2, min_val=1, max_val=10) / 100

    # Confirm
    clear_screen()
    print_header()
    print("\n📋 Training Summary:")
    print(f"   Game:        {env}")
    print(f"   Timesteps:   {timesteps:,}")
    print(f"   Envs:        {n_envs}")
    print(f"   Dashboard:   {'Yes' if dashboard else 'No'}")
    print(f"   Demo every:  {demo_every if demo_every > 0 else 'Disabled'}")
    print(f"   Entropy:     {entropy}")

    if not confirm("\nStart training?", default=True):
        print("Cancelled.")
        return

    run_training(
        env=env,
        timesteps=timesteps,
        n_envs=n_envs,
        demo_every=demo_every,
        dashboard=dashboard,
        entropy=entropy,
    )


def play_menu():
    """Play a trained agent."""
    clear_screen()
    print_header()
    print("\n🎬 Watch Trained Agent\n")

    # Check for checkpoints
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        print("❌ No checkpoints found! Train an agent first.")
        input("\nPress Enter to continue...")
        return main_menu()

    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    if not checkpoints:
        print("❌ No checkpoints found! Train an agent first.")
        input("\nPress Enter to continue...")
        return main_menu()

    print("Available checkpoints:")
    for i, cp in enumerate(checkpoints, 1):
        print(f"  {i}. {cp}")

    while True:
        try:
            choice = input(f"\nSelect checkpoint (1-{len(checkpoints)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(checkpoints):
                checkpoint = os.path.join(checkpoint_dir, checkpoints[idx])
                break
        except ValueError:
            pass
        print("Invalid choice.")

    env = get_choice("Which game was this trained on?", {
        "pong": "🏓 Pong",
        "breakout": "🧱 Breakout",
        "spaceinvaders": "👾 Space Invaders",
        "mario": "🍄 Super Mario Bros",
    })

    episodes = get_number("How many episodes to play?", 5, min_val=1, max_val=100)

    run_play(env=env, checkpoint=checkpoint, episodes=episodes)


def run_training(env, timesteps, n_envs=8, demo_every=25, dashboard=True, entropy=0.02):
    """Run training with given parameters."""
    from config import PPOConfig, TrainingConfig, EnvConfig
    from training import Trainer, CheckpointCallback, EpisodeLoggerCallback
    from utils import get_device

    device = get_device()
    print(f"\n🖥️  Using device: {device}\n")

    ppo_config = PPOConfig(ent_coef=entropy)
    env_config = EnvConfig(env_name=env, n_envs=n_envs)
    training_config = TrainingConfig(
        total_timesteps=timesteps,
        device=device,
        demo_every=demo_every,
    )

    callbacks = [
        CheckpointCallback(save_path="checkpoints/best_model.pt"),
        EpisodeLoggerCallback(log_interval=20),
    ]

    trainer = Trainer(
        env_config=env_config,
        ppo_config=ppo_config,
        training_config=training_config,
        callbacks=callbacks,
        show_dashboard=dashboard,
    )

    try:
        metrics = trainer.train()
        print("\n" + "=" * 50)
        print("  ✅ Training Complete!")
        print("=" * 50)
        print(f"  Episodes:    {metrics['total_episodes']}")
        print(f"  Mean reward: {metrics['mean_reward']:.1f}")
        print(f"  Time:        {metrics['training_time']:.0f}s")
        print("\n  Checkpoint saved to: checkpoints/")
    except KeyboardInterrupt:
        print("\n\n⏹️  Training stopped.")

    input("\nPress Enter to continue...")


def run_play(env, checkpoint, episodes):
    """Run agent playback."""
    import torch
    import time
    from environments import make_env
    from agents import ActorCritic
    from utils import get_device
    from config import ENV_IDS
    import gymnasium as gym

    device = get_device()
    print(f"\n🖥️  Using device: {device}")
    print(f"📂 Loading: {checkpoint}\n")

    # Create environment
    if env == "mario":
        play_env = make_env(env)
        has_render = True
    else:
        from environments.wrappers import (
            GrayscaleWrapper, ResizeWrapper,
            FrameStackWrapper, NormalizeWrapper, MaxAndSkipWrapper,
        )
        base_env = gym.make(ENV_IDS[env], render_mode="human")
        base_env = MaxAndSkipWrapper(base_env, skip=4)
        base_env = GrayscaleWrapper(base_env)
        base_env = ResizeWrapper(base_env)
        base_env = FrameStackWrapper(base_env)
        play_env = NormalizeWrapper(base_env)
        has_render = False

    # Load model
    n_actions = play_env.action_space.n
    obs_shape = play_env.observation_space.shape
    model = ActorCritic(n_actions=n_actions, n_input_channels=obs_shape[-1]).to(device)

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["network_state_dict"])
    model.eval()

    total_rewards = []

    for ep in range(episodes):
        obs, _ = play_env.reset()
        episode_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs[None], dtype=torch.float32, device=device)
                action_logits, _ = model(obs_tensor)
                action = action_logits.argmax(dim=-1).item()

            obs, reward, terminated, truncated, info = play_env.step(action)
            done = terminated or truncated
            episode_reward += reward

            if has_render:
                play_env.render()
                time.sleep(0.02)

        total_rewards.append(episode_reward)
        print(f"Episode {ep + 1}: {episode_reward:.0f}")

    play_env.close()

    print(f"\n📊 Average: {sum(total_rewards)/len(total_rewards):.1f}")
    print(f"🏆 Best:    {max(total_rewards):.0f}")

    input("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        while True:
            main_menu()
    except KeyboardInterrupt:
        print("\n\nGoodbye! 👋")
