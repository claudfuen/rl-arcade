"""
Proximal Policy Optimization (PPO) algorithm implementation.

PPO is a policy gradient method that has become the go-to algorithm
for many RL applications due to its:
- Stability: Clipping prevents catastrophically large updates
- Simplicity: Relatively easy to implement and tune
- Sample efficiency: Reuses data multiple times per update

Key Concepts:
1. Policy Gradient: Learn by increasing probability of good actions
2. Advantage: How much better an action is than expected
3. Clipping: Limit how much the policy can change per update
4. GAE: Smart way to estimate advantages with less variance
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Dict, Generator, NamedTuple
from dataclasses import dataclass

from .networks import ActorCritic
from config import PPOConfig


class RolloutSample(NamedTuple):
    """A single sample from the rollout buffer."""

    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    old_values: torch.Tensor


class RolloutBuffer:
    """
    Buffer to store and process trajectories for PPO.

    Key Concept - Experience Collection:
        We collect a batch of experience (n_steps * n_envs transitions)
        before each training update. This provides:
        - Diverse experiences from multiple environments
        - Enough data for multiple mini-batch updates
        - Stable gradient estimates

    Key Concept - Generalized Advantage Estimation (GAE):
        GAE balances bias vs variance in advantage estimation:
        - Low lambda (0): Uses just one-step TD error (low variance, high bias)
        - High lambda (1): Uses full Monte Carlo return (high variance, low bias)
        - Lambda=0.95 is a good middle ground

        GAE formula:
        A_t = sum_{l=0}^{inf} (gamma * lambda)^l * delta_{t+l}
        where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    """

    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        obs_shape: tuple,
        device: str = "cpu",
    ):
        """
        Initialize the rollout buffer.

        Args:
            n_steps: Number of steps to collect per environment
            n_envs: Number of parallel environments
            obs_shape: Shape of observations
            device: Device to store tensors on
        """
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.obs_shape = obs_shape
        self.device = device

        # Preallocate buffers
        self.observations = np.zeros(
            (n_steps, n_envs) + obs_shape, dtype=np.float32
        )
        self.actions = np.zeros((n_steps, n_envs), dtype=np.int64)
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.log_probs = np.zeros((n_steps, n_envs), dtype=np.float32)

        # Computed during finalization
        self.advantages = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.returns = np.zeros((n_steps, n_envs), dtype=np.float32)

        self.ptr = 0  # Current position in buffer

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        value: np.ndarray,
        log_prob: np.ndarray,
    ):
        """
        Add a transition to the buffer.

        Args:
            obs: Observations (n_envs, *obs_shape)
            action: Actions taken (n_envs,)
            reward: Rewards received (n_envs,)
            done: Episode done flags (n_envs,)
            value: Value estimates (n_envs,)
            log_prob: Log probabilities of actions (n_envs,)
        """
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.ptr += 1

    def compute_returns_and_advantages(
        self,
        last_value: np.ndarray,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """
        Compute returns and advantages using GAE.

        This is the heart of credit assignment in PPO:
        1. Work backwards from the final state
        2. At each step, compute TD error: delta = r + gamma*V(s') - V(s)
        3. Accumulate advantages using GAE formula

        Args:
            last_value: Value estimate for the final state
            gamma: Discount factor (how much to value future rewards)
            gae_lambda: GAE parameter (bias-variance tradeoff)
        """
        last_gae = 0

        # Work backwards through the trajectory
        for step in reversed(range(self.n_steps)):
            if step == self.n_steps - 1:
                # For the last step, use the provided last_value
                next_non_terminal = 1.0 - self.dones[step]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = self.values[step + 1]

            # TD error: how much better/worse was this transition than expected
            delta = (
                self.rewards[step]
                + gamma * next_value * next_non_terminal
                - self.values[step]
            )

            # GAE advantage: accumulate TD errors with decay
            self.advantages[step] = last_gae = (
                delta + gamma * gae_lambda * next_non_terminal * last_gae
            )

        # Returns = advantages + values (for value function loss)
        self.returns = self.advantages + self.values

    def get_samples(
        self,
        batch_size: int,
    ) -> Generator[RolloutSample, None, None]:
        """
        Generate mini-batches for training.

        Shuffles all data and yields batches of specified size.
        Each sample contains:
        - observations, actions, old_log_probs (from rollout)
        - advantages, returns (computed)
        - old_values (for value clipping)

        Args:
            batch_size: Size of each mini-batch

        Yields:
            RolloutSample containing tensors for training
        """
        # Flatten time and environment dimensions
        total_size = self.n_steps * self.n_envs
        indices = np.random.permutation(total_size)

        # Flatten arrays for easier indexing
        obs_flat = self.observations.reshape((total_size,) + self.obs_shape)
        actions_flat = self.actions.reshape(total_size)
        log_probs_flat = self.log_probs.reshape(total_size)
        advantages_flat = self.advantages.reshape(total_size)
        returns_flat = self.returns.reshape(total_size)
        values_flat = self.values.reshape(total_size)

        # Normalize advantages (important for stable training)
        advantages_flat = (advantages_flat - advantages_flat.mean()) / (
            advantages_flat.std() + 1e-8
        )

        # Yield batches
        for start in range(0, total_size, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            yield RolloutSample(
                observations=torch.tensor(
                    obs_flat[batch_indices], dtype=torch.float32, device=self.device
                ),
                actions=torch.tensor(
                    actions_flat[batch_indices], dtype=torch.long, device=self.device
                ),
                old_log_probs=torch.tensor(
                    log_probs_flat[batch_indices], dtype=torch.float32, device=self.device
                ),
                advantages=torch.tensor(
                    advantages_flat[batch_indices], dtype=torch.float32, device=self.device
                ),
                returns=torch.tensor(
                    returns_flat[batch_indices], dtype=torch.float32, device=self.device
                ),
                old_values=torch.tensor(
                    values_flat[batch_indices], dtype=torch.float32, device=self.device
                ),
            )

    def reset(self):
        """Reset buffer for next rollout."""
        self.ptr = 0


class PPO:
    """
    Proximal Policy Optimization algorithm.

    Key Concept - Clipped Surrogate Objective:
        The core idea of PPO is to limit how much the policy can change
        in a single update. This prevents catastrophically large updates
        that could destabilize training.

        L^CLIP = min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)

        where r_t = pi_new(a|s) / pi_old(a|s) is the probability ratio

        - If advantage is positive and ratio > 1+eps: clipped (don't increase more)
        - If advantage is negative and ratio < 1-eps: clipped (don't decrease more)
        - This keeps the new policy "close" to the old policy

    Key Concept - Value Loss:
        The value function is trained to predict expected returns.
        L^VF = (V(s) - R_t)^2

        Optionally, we can also clip the value function to prevent
        large updates (helps with stability but can hurt performance).

    Key Concept - Entropy Bonus:
        Adding entropy to the loss encourages exploration:
        - High entropy = spread out probabilities = more exploration
        - As training progresses, entropy naturally decreases
        - We subtract entropy from loss (to maximize it)
    """

    def __init__(
        self,
        env,
        config: PPOConfig = None,
        device: str = "cpu",
    ):
        """
        Initialize PPO algorithm.

        Args:
            env: Vectorized environment (VecEnv)
            config: PPO hyperparameters
            device: Device to run on
        """
        self.env = env
        self.config = config or PPOConfig()
        self.device = device

        # Get environment dimensions
        self.n_envs = env.n_envs
        self.n_actions = env.action_space.n
        self.obs_shape = env.observation_space.shape

        # Create actor-critic network
        n_input_channels = self.obs_shape[-1]  # Channels last format
        self.network = ActorCritic(
            n_actions=self.n_actions,
            n_input_channels=n_input_channels,
        ).to(device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate,
            eps=1e-5,  # Slightly higher eps for stability
        )

        # Create rollout buffer
        self.buffer = RolloutBuffer(
            n_steps=self.config.n_steps,
            n_envs=self.n_envs,
            obs_shape=self.obs_shape,
            device=device,
        )

        # Training state
        self.num_timesteps = 0

    def collect_rollout(self, obs: np.ndarray) -> np.ndarray:
        """
        Collect a rollout of experience.

        Runs the current policy for n_steps in each environment,
        storing transitions in the rollout buffer.

        Args:
            obs: Current observations (n_envs, *obs_shape)

        Returns:
            obs: Final observations after rollout
        """
        self.network.eval()

        for step in range(self.config.n_steps):
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                action, log_prob, _, value = self.network.get_action_and_value(
                    obs_tensor
                )

            # Convert to numpy
            action_np = action.cpu().numpy()
            log_prob_np = log_prob.cpu().numpy()
            value_np = value.cpu().numpy()

            # Step environment
            next_obs, reward, done, info = self.env.step(action_np)

            # Store transition
            self.buffer.add(
                obs=obs,
                action=action_np,
                reward=reward,
                done=done,
                value=value_np,
                log_prob=log_prob_np,
            )

            obs = next_obs
            self.num_timesteps += self.n_envs

        # Compute value of final state for GAE
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
            last_value = self.network.get_value(obs_tensor).cpu().numpy()

        # Compute returns and advantages
        self.buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )

        return obs

    def train(self) -> Dict[str, float]:
        """
        Perform PPO update using collected rollout.

        Returns:
            Dictionary of training metrics
        """
        self.network.train()

        # Metrics accumulators
        pg_losses = []
        value_losses = []
        entropy_losses = []
        clip_fractions = []

        # Multiple epochs over the data
        for epoch in range(self.config.n_epochs):
            for batch in self.buffer.get_samples(self.config.batch_size):
                # Get current policy outputs
                _, log_prob, entropy, value = self.network.get_action_and_value(
                    batch.observations, batch.actions
                )

                # Compute probability ratio
                # r_t = pi_new(a|s) / pi_old(a|s) = exp(log_pi_new - log_pi_old)
                ratio = torch.exp(log_prob - batch.old_log_probs)

                # Clipped surrogate objective
                # This is the key PPO innovation
                pg_loss1 = -batch.advantages * ratio
                pg_loss2 = -batch.advantages * torch.clamp(
                    ratio,
                    1 - self.config.clip_range,
                    1 + self.config.clip_range,
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (optionally clipped)
                if self.config.clip_range_vf is not None:
                    # Clipped value loss
                    value_clipped = batch.old_values + torch.clamp(
                        value - batch.old_values,
                        -self.config.clip_range_vf,
                        self.config.clip_range_vf,
                    )
                    value_loss1 = (value - batch.returns) ** 2
                    value_loss2 = (value_clipped - batch.returns) ** 2
                    value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
                else:
                    # Standard MSE value loss
                    value_loss = 0.5 * ((value - batch.returns) ** 2).mean()

                # Entropy loss (negative because we want to maximize entropy)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    pg_loss
                    + self.config.vf_coef * value_loss
                    + self.config.ent_coef * entropy_loss
                )

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping (prevents exploding gradients)
                nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    self.config.max_grad_norm,
                )

                self.optimizer.step()

                # Track metrics
                pg_losses.append(pg_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())

                # Track clip fraction (how often clipping was active)
                with torch.no_grad():
                    clip_fraction = (
                        (torch.abs(ratio - 1) > self.config.clip_range)
                        .float()
                        .mean()
                        .item()
                    )
                    clip_fractions.append(clip_fraction)

        # Reset buffer for next rollout
        self.buffer.reset()

        return {
            "policy_loss": np.mean(pg_losses),
            "value_loss": np.mean(value_losses),
            "entropy_loss": np.mean(entropy_losses),
            "clip_fraction": np.mean(clip_fractions),
        }

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """
        Predict action for given observation.

        Args:
            obs: Observation (batch or single)
            deterministic: If True, return mode of distribution

        Returns:
            Predicted actions
        """
        self.network.eval()

        with torch.no_grad():
            # Add batch dimension if needed
            single_obs = obs.ndim == len(self.obs_shape)
            if single_obs:
                obs = obs[np.newaxis]

            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
            action_logits, _ = self.network(obs_tensor)

            if deterministic:
                actions = action_logits.argmax(dim=-1)
            else:
                dist = torch.distributions.Categorical(logits=action_logits)
                actions = dist.sample()

            actions = actions.cpu().numpy()

            if single_obs:
                actions = actions[0]

            return actions

    def save(self, path: str, session=None, full_state: bool = True):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            session: Optional Session object for metadata
            full_state: If True, include RNG states for exact reproducibility
        """
        from sessions import capture_rng_states
        from config import config_to_dict

        checkpoint = {
            # Core state (always included, backwards compatible)
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "num_timesteps": self.num_timesteps,
            # Version marker for format detection
            "version": "2.0",
        }

        # Add session metadata if available
        if session is not None:
            checkpoint["session_id"] = session.session_id
            checkpoint["game"] = session.game
            checkpoint["env_config"] = session.env_config
            checkpoint["ppo_config"] = session.ppo_config
            checkpoint["training_config"] = session.training_config

        # Add RNG states for exact reproducibility
        if full_state:
            checkpoint["rng_states"] = capture_rng_states()

        torch.save(checkpoint, path)

    def load(self, path: str, restore_rng: bool = False) -> dict:
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint file
            restore_rng: If True and checkpoint has RNG states, restore them

        Returns:
            Full checkpoint dictionary (caller can use metadata)
        """
        from sessions import restore_rng_states

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Load core state
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.num_timesteps = checkpoint.get("num_timesteps", 0)

        # Restore RNG states if requested and available
        if restore_rng and "rng_states" in checkpoint:
            restore_rng_states(checkpoint["rng_states"])

        return checkpoint
