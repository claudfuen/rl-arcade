# 🎮 RL Arcade

**Train AI agents to play classic video games using reinforcement learning.**

Watch neural networks learn to play Pong, Breakout, Super Mario Bros, and more — from scratch, with no human knowledge.

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/pytorch-2.0+-orange.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
</p>

---

## ✨ Features

- 🕹️ **Multiple Games** — Pong, Breakout, Space Invaders, Super Mario Bros
- 📊 **Live Dashboard** — Watch training metrics update in real-time
- 🎬 **Demo Mode** — See the agent play periodically during training
- 🧠 **Clean PPO Implementation** — Well-documented, educational codebase
- ⚡ **Fast Training** — Vectorized environments for parallel data collection
- 🎯 **Interactive CLI** — No flags to memorize, just answer prompts

---

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/claudfuen/rl-arcade.git
cd rl-arcade
pip install -r requirements.txt
```

### 2. Run

**Interactive mode (recommended):**
```bash
python3 cli.py
```

**Or quick demo:**
```bash
python3 main.py demo --env pong
```

That's it! You'll see a Pong agent go from random flailing to actually winning.

---

## 🎯 Supported Games

| Game | Difficulty | Training Time | Description |
|------|------------|---------------|-------------|
| 🏓 **Pong** | Easy | ~10 min | Classic paddle game. Great for beginners. |
| 🧱 **Breakout** | Medium | ~30 min | Break bricks with a ball. Satisfying to watch. |
| 👾 **Space Invaders** | Medium | ~45 min | Shoot descending aliens. |
| 🍄 **Super Mario Bros** | Hard | ~2 hours | Navigate World 1-1. The ultimate test. |

*Training times are rough estimates for visible improvement on an M1 Mac.*

---

## 📖 Usage

### Interactive CLI

Just run and follow the prompts:

```bash
python3 cli.py
```

```
==================================================
  🎮 RL Game Agent Trainer
==================================================

What would you like to do?

  1. 🏋️  Train a new agent
  2. 🎬  Watch a trained agent play
  3. ⚡  Quick demo (recommended for first time)

Enter number:
```

### Command Line

```bash
# Train with live dashboard and periodic demos
python3 main.py train --env pong --timesteps 200000 --dashboard

# Watch a trained agent
python3 main.py play --env pong --checkpoint checkpoints/best_model.pt

# Quick demo
python3 main.py demo --env breakout
```

### Key Options

| Flag | Description |
|------|-------------|
| `--env` | Game: `pong`, `breakout`, `spaceinvaders`, `mario` |
| `--timesteps` | Training duration (default: 1M) |
| `--dashboard` | Show live training graphs |
| `--demo-every` | Play demo game every N updates (default: 25) |
| `--n-envs` | Parallel environments (default: 8, more = faster) |
| `--entropy` | Exploration coefficient (default: 0.02) |

---

## 🧠 How It Works

This project uses **Proximal Policy Optimization (PPO)**, a state-of-the-art reinforcement learning algorithm.

### The Learning Loop

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   1. Agent sees game screen (84x84 grayscale pixels)   │
│                          ↓                              │
│   2. Neural network outputs action probabilities        │
│                          ↓                              │
│   3. Agent takes action, receives reward                │
│                          ↓                              │
│   4. PPO updates network to increase good actions       │
│                          ↓                              │
│   5. Repeat millions of times                          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Architecture

```
Game Frame (210x160 RGB)
         ↓
   Preprocessing
   • Grayscale
   • Resize to 84x84
   • Stack 4 frames (for motion)
   • Normalize to [0,1]
         ↓
┌─────────────────────┐
│   CNN Backbone      │
│   ┌───────────────┐ │
│   │ Conv 32x8x8   │ │
│   │ Conv 64x4x4   │ │
│   │ Conv 64x3x3   │ │
│   │ FC 512        │ │
│   └───────────────┘ │
└─────────┬───────────┘
          │
    ┌─────┴─────┐
    ↓           ↓
┌───────┐  ┌────────┐
│Policy │  │ Value  │
│ Head  │  │  Head  │
└───┬───┘  └───┬────┘
    ↓          ↓
 Action    State Value
 Probs     Estimate
```

### Key Concepts

| Concept | What It Does |
|---------|--------------|
| **Policy Gradient** | Learn by increasing probability of rewarded actions |
| **Value Function** | Estimate "how good" each state is |
| **Advantage** | How much better an action was than expected |
| **Clipping** | Prevent destructively large policy updates |
| **Entropy Bonus** | Encourage exploration |
| **Frame Stacking** | Give network temporal information |

---

## 📁 Project Structure

```
rl-arcade/
├── cli.py                 # Interactive CLI (start here!)
├── main.py               # Command-line entry point
├── config.py             # Hyperparameters
├── requirements.txt
│
├── agents/
│   ├── networks.py       # CNN actor-critic architecture
│   └── ppo.py           # PPO algorithm implementation
│
├── environments/
│   ├── wrappers.py      # Frame preprocessing
│   └── make_env.py      # Environment factory
│
├── training/
│   ├── trainer.py       # Training loop
│   └── callbacks.py     # Checkpointing, logging
│
└── visualization/
    ├── dashboard.py     # Live matplotlib plots
    └── tensorboard_utils.py
```

---

## 📈 Training Tips

### General

- **Start with Pong** — It's the fastest to train and great for verifying your setup
- **Use the dashboard** — Watch entropy decrease and rewards increase over time
- **More environments = faster** — `--n-envs 16` collects data faster (if you have RAM)

### If the agent isn't learning:

1. **Increase entropy** — `--entropy 0.05` encourages more exploration
2. **Train longer** — Some games need 500k+ steps
3. **Check the dashboard** — Is entropy collapsing? Are losses stable?

### Typical learning progression:

```
Steps 0-10k:      Random behavior, negative rewards
Steps 10k-50k:    Agent discovers some actions matter
Steps 50k-200k:   Basic strategy emerges (tracks ball, etc.)
Steps 200k+:      Refinement, higher scores
```

---

## 🛠️ Development

### Running TensorBoard

```bash
tensorboard --logdir logs
# Open http://localhost:6006
```

### Code Style

```bash
pip install black isort
black .
isort .
```

---

## 🗺️ Roadmap

- [x] Atari games (Pong, Breakout, Space Invaders)
- [x] Super Mario Bros
- [x] Interactive CLI
- [x] Live training dashboard
- [ ] Pokemon Red/Blue (Game Boy) 🔜
- [ ] Sonic the Hedgehog
- [ ] Save/load training progress
- [ ] Hyperparameter tuning guide
- [ ] Pre-trained model zoo

---

## 📚 Learn More

This codebase is designed to be educational. Key files to read:

1. **`agents/ppo.py`** — The PPO algorithm with detailed comments
2. **`agents/networks.py`** — Neural network architecture
3. **`environments/wrappers.py`** — Frame preprocessing explained

### Recommended Reading

- [Spinning Up in Deep RL](https://spinningup.openai.com/) — OpenAI's RL tutorial
- [PPO Paper](https://arxiv.org/abs/1707.06347) — Original PPO paper
- [Deep RL Bootcamp](https://sites.google.com/view/deep-rl-bootcamp/lectures) — Berkeley lectures

---

## 🤝 Contributing

Contributions welcome! Some ideas:

- Add new games (Tetris, Pac-Man, etc.)
- Improve training speed
- Add more visualization options
- Write tutorials

---

## 📄 License

MIT License — use this however you want!

---

<p align="center">
  <b>Built for learning. Have fun! 🎮</b>
</p>
