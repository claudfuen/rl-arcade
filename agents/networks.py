"""
Neural network architectures for game-playing agents.

This module implements the Nature DQN architecture, which has become
the standard CNN architecture for processing game frames in RL.

Key Concept - Actor-Critic Architecture:
    The network has two "heads" sharing a common feature extractor:
    - Actor (Policy): Outputs action probabilities
    - Critic (Value): Estimates expected future reward

    Why share features?
    - Both tasks need to understand the game state
    - Sharing reduces parameters and improves efficiency
    - Features learned for value help policy and vice versa
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Tuple


class CNNFeatureExtractor(nn.Module):
    """
    Convolutional feature extractor (Nature DQN architecture).

    Architecture:
        Conv1: 32 filters, 8x8 kernel, stride 4 → ReLU
        Conv2: 64 filters, 4x4 kernel, stride 2 → ReLU
        Conv3: 64 filters, 3x3 kernel, stride 1 → ReLU
        Flatten → Linear(3136, 512) → ReLU

    Why this architecture?
    - Proven effective across many Atari games
    - Progressively reduces spatial dimensions while increasing channels
    - Final layer captures high-level game features

    Input: (batch, 4, 84, 84) - 4 stacked grayscale frames
    Output: (batch, 512) - feature vector
    """

    def __init__(
        self,
        n_input_channels: int = 4,
        feature_dim: int = 512,
    ):
        """
        Initialize the CNN feature extractor.

        Args:
            n_input_channels: Number of input channels (stacked frames)
            feature_dim: Size of the output feature vector
        """
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate the size after convolutions
        # Input: 84x84 → conv1(8,4): 20x20 → conv2(4,2): 9x9 → conv3(3,1): 7x7
        conv_output_size = 64 * 7 * 7  # 3136

        # Fully connected layer
        self.fc = nn.Linear(conv_output_size, feature_dim)

        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from observation.

        Args:
            x: Observation tensor (batch, height, width, channels)
               Note: Input is (H, W, C) but PyTorch needs (C, H, W)

        Returns:
            Feature vector (batch, feature_dim)
        """
        # Convert from (B, H, W, C) to (B, C, H, W) for PyTorch
        if x.dim() == 4 and x.shape[-1] in [1, 4]:  # Channels last format
            x = x.permute(0, 3, 1, 2)

        # Convolutional layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten and fully connected
        x = x.reshape(x.size(0), -1)  # Flatten
        x = F.relu(self.fc(x))

        return x


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.

    Architecture:
        Shared CNN Feature Extractor
                    ↓
            Feature Vector (512)
                ↙       ↘
        Policy Head    Value Head
        (Actor)        (Critic)
             ↓              ↓
        Action Probs   State Value

    Key Concept - Policy Gradient:
        The actor outputs a probability distribution over actions.
        We sample from this distribution to select actions.
        The gradient of log-probability guides learning:
        - Increase probability of actions that led to high rewards
        - Decrease probability of actions that led to low rewards

    Key Concept - Value Function:
        The critic estimates how good the current state is.
        This "baseline" reduces variance in policy gradient:
        - Instead of using raw rewards, we use "advantage"
        - Advantage = Actual reward - Expected reward (value)
        - Positive advantage → action was better than expected
        - Negative advantage → action was worse than expected
    """

    def __init__(
        self,
        n_actions: int,
        n_input_channels: int = 4,
        feature_dim: int = 512,
    ):
        """
        Initialize Actor-Critic network.

        Args:
            n_actions: Number of possible actions in the environment
            n_input_channels: Number of input channels (stacked frames)
            feature_dim: Size of the shared feature vector
        """
        super().__init__()

        # Shared feature extractor
        self.features = CNNFeatureExtractor(
            n_input_channels=n_input_channels,
            feature_dim=feature_dim,
        )

        # Policy head (actor)
        # Outputs logits for each action
        self.policy_head = nn.Linear(feature_dim, n_actions)

        # Value head (critic)
        # Outputs single scalar value estimate
        self.value_head = nn.Linear(feature_dim, 1)

        # Initialize weights (important for stable training)
        self._init_weights()

    def _init_weights(self):
        """
        Initialize network weights.

        Using orthogonal initialization with different scales:
        - Conv and hidden layers: scale=sqrt(2) (ReLU layers)
        - Policy head: scale=0.01 (small initial actions)
        - Value head: scale=1 (standard output)
        """
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # Smaller initialization for policy (more uniform initial distribution)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.constant_(self.policy_head.bias, 0)

        # Standard initialization for value
        nn.init.orthogonal_(self.value_head.weight, gain=1)
        nn.init.constant_(self.value_head.bias, 0)

    def forward(
        self,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            obs: Observation tensor (batch, height, width, channels)

        Returns:
            action_logits: Logits for each action (batch, n_actions)
            values: State value estimates (batch, 1)
        """
        features = self.features(obs)
        action_logits = self.policy_head(features)
        values = self.value_head(features)
        return action_logits, values

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, entropy, and value.

        This method is used during:
        1. Action selection (action=None): Sample from policy
        2. Training (action=given): Compute log prob of taken action

        Args:
            obs: Observation tensor
            action: Optional action tensor (for computing log prob)

        Returns:
            action: Selected or given action
            log_prob: Log probability of the action
            entropy: Entropy of the action distribution
            value: State value estimate
        """
        action_logits, values = self.forward(obs)

        # Create categorical distribution from logits
        dist = Categorical(logits=action_logits)

        # Sample action if not provided
        if action is None:
            action = dist.sample()

        # Compute log probability and entropy
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, values.squeeze(-1)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get only the value estimate (used for GAE computation).

        Args:
            obs: Observation tensor

        Returns:
            values: State value estimates
        """
        features = self.features(obs)
        return self.value_head(features).squeeze(-1)
