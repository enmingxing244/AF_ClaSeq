"""
PPO Agent for MSA subset selection.

This module implements the Proximal Policy Optimization agent
for learning sequence selection policies.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import logging

from af_claseq.msa_rl.models import ActorCriticNetwork, create_actor_critic_model

logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    """Configuration for PPO agent."""
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE parameter
    clip_ratio: float = 0.2  # PPO clip ratio
    value_clip_ratio: float = 0.2  # Value function clip ratio
    entropy_coef: float = 0.01  # Entropy regularization
    value_coef: float = 0.5  # Value function loss coefficient
    max_grad_norm: float = 0.5  # Gradient clipping
    ppo_epochs: int = 4  # Number of PPO epochs per update
    mini_batch_size: int = 64  # Mini-batch size for PPO updates
    buffer_size: int = 2048  # Experience buffer size
    update_frequency: int = 2048  # Update frequency in steps


class ExperienceBuffer:
    """Buffer for storing and managing RL experiences."""
    
    def __init__(self, buffer_size: int, state_dim: int, sequence_feature_dim: int, max_sequences: int):
        """
        Initialize experience buffer.
        
        Args:
            buffer_size: Maximum number of experiences to store
            state_dim: Dimension of state space
            sequence_feature_dim: Dimension of sequence features
            max_sequences: Maximum number of sequences in action space
        """
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.sequence_feature_dim = sequence_feature_dim
        self.max_sequences = max_sequences
        
        # Initialize buffers
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.available_sequences = np.zeros((buffer_size, max_sequences, sequence_feature_dim), dtype=np.float32)
        self.action_masks = np.zeros((buffer_size, max_sequences), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.bool_)
        
        self.ptr = 0
        self.size = 0
        
    def store(
        self,
        state: np.ndarray,
        available_sequences: np.ndarray,
        action_mask: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """Store a single experience."""
        idx = self.ptr % self.buffer_size
        
        self.states[idx] = state
        
        # Handle variable sequence lengths by padding
        n_seqs = available_sequences.shape[0]
        self.available_sequences[idx, :n_seqs] = available_sequences
        self.available_sequences[idx, n_seqs:] = 0  # Pad with zeros
        
        self.action_masks[idx, :len(action_mask)] = action_mask
        self.action_masks[idx, len(action_mask):] = 0  # Pad with zeros
        
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.values[idx] = value
        self.log_probs[idx] = log_prob
        self.dones[idx] = done
        
        self.ptr += 1
        self.size = min(self.size + 1, self.buffer_size)
    
    def get_batch(self, batch_indices: np.ndarray) -> Tuple[torch.Tensor, ...]:
        """Get a batch of experiences."""
        return (
            torch.FloatTensor(self.states[batch_indices]),
            torch.FloatTensor(self.available_sequences[batch_indices]),
            torch.FloatTensor(self.action_masks[batch_indices]),
            torch.LongTensor(self.actions[batch_indices]),
            torch.FloatTensor(self.rewards[batch_indices]),
            torch.FloatTensor(self.values[batch_indices]),
            torch.FloatTensor(self.log_probs[batch_indices]),
            torch.BoolTensor(self.dones[batch_indices])
        )
    
    def compute_advantages(self, gamma: float, gae_lambda: float, last_value: float = 0.0) -> np.ndarray:
        """Compute GAE advantages."""
        advantages = np.zeros_like(self.rewards[:self.size])
        last_gae_lam = 0
        
        for step in reversed(range(self.size)):
            if step == self.size - 1:
                next_non_terminal = 1.0 - float(self.dones[step])
                next_value = last_value
            else:
                next_non_terminal = 1.0 - float(self.dones[step])
                next_value = self.values[step + 1]
            
            delta = self.rewards[step] + gamma * next_value * next_non_terminal - self.values[step]
            advantages[step] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        
        return advantages
    
    def clear(self):
        """Clear the buffer."""
        self.ptr = 0
        self.size = 0


class PPOAgent:
    """
    Proximal Policy Optimization agent for MSA subset selection.
    """
    
    def __init__(
        self,
        state_dim: int,
        sequence_feature_dim: int,
        max_sequences: int,
        config: PPOConfig,
        device: torch.device = torch.device("cpu")
    ):
        """
        Initialize PPO agent.
        
        Args:
            state_dim: Dimension of state space
            sequence_feature_dim: Dimension of sequence features
            max_sequences: Maximum number of sequences in action space
            config: PPO configuration
            device: Device to run on
        """
        self.state_dim = state_dim
        self.sequence_feature_dim = sequence_feature_dim
        self.max_sequences = max_sequences
        self.config = config
        self.device = device
        
        # Create actor-critic model
        self.model = create_actor_critic_model(
            state_dim=state_dim,
            sequence_feature_dim=sequence_feature_dim,
            device=device
        )
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        
        # Experience buffer
        self.buffer = ExperienceBuffer(
            buffer_size=config.buffer_size,
            state_dim=state_dim,
            sequence_feature_dim=sequence_feature_dim,
            max_sequences=max_sequences
        )
        
        # Training statistics
        self.total_steps = 0
        self.total_episodes = 0
        self.episode_rewards = deque(maxlen=100)
        self.training_metrics = {}
        
    def select_action(
        self,
        state: np.ndarray,
        available_sequences: np.ndarray,
        action_mask: np.ndarray,
        training: bool = True
    ) -> Tuple[int, float, float]:
        """
        Select action using current policy.
        
        Args:
            state: Current state
            available_sequences: Available sequence features
            action_mask: Mask for valid actions
            training: Whether in training mode
            
        Returns:
            action, log_prob, value
        """
        with torch.no_grad():
            # Convert to tensors
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Pad sequences to max_sequences
            padded_sequences = np.zeros((self.max_sequences, self.sequence_feature_dim), dtype=np.float32)
            n_seqs = min(len(available_sequences), self.max_sequences)
            padded_sequences[:n_seqs] = available_sequences[:n_seqs]
            seq_tensor = torch.FloatTensor(padded_sequences).unsqueeze(0).to(self.device)
            
            # Pad action mask
            padded_mask = np.zeros(self.max_sequences, dtype=np.float32)
            padded_mask[:len(action_mask)] = action_mask
            mask_tensor = torch.FloatTensor(padded_mask).unsqueeze(0).to(self.device)
            
            # Get action and value
            action, log_prob, entropy, value = self.model.get_action_and_value(
                state_tensor, seq_tensor, mask_tensor
            )
            
            return action.item(), log_prob.item(), value.item()
    
    def store_experience(
        self,
        state: np.ndarray,
        available_sequences: np.ndarray,
        action_mask: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """Store experience in buffer."""
        self.buffer.store(
            state, available_sequences, action_mask, action, reward, value, log_prob, done
        )
        self.total_steps += 1
        
        if done:
            self.total_episodes += 1
    
    def update(self, last_value: float = 0.0) -> Dict[str, float]:
        """
        Update the policy using PPO.
        
        Args:
            last_value: Value of the last state (for GAE computation)
            
        Returns:
            Training metrics
        """
        if self.buffer.size < self.config.mini_batch_size:
            return {}
        
        # Compute advantages
        advantages = self.buffer.compute_advantages(
            self.config.gamma, self.config.gae_lambda, last_value
        )
        returns = advantages + self.buffer.values[:self.buffer.size]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training metrics
        metrics = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy_loss': 0.0,
            'total_loss': 0.0,
            'kl_divergence': 0.0,
            'clip_fraction': 0.0
        }
        
        # PPO training epochs
        for epoch in range(self.config.ppo_epochs):
            # Create mini-batches
            indices = np.random.permutation(self.buffer.size)
            
            for start in range(0, self.buffer.size, self.config.mini_batch_size):
                end = start + self.config.mini_batch_size
                batch_indices = indices[start:end]
                
                if len(batch_indices) < self.config.mini_batch_size:
                    continue
                
                # Get batch data
                batch_data = self.buffer.get_batch(batch_indices)
                batch_metrics = self._update_batch(batch_data, advantages[batch_indices], returns[batch_indices])
                
                # Accumulate metrics
                for key, value in batch_metrics.items():
                    metrics[key] += value
        
        # Average metrics
        num_batches = (self.buffer.size // self.config.mini_batch_size) * self.config.ppo_epochs
        if num_batches > 0:
            for key in metrics:
                metrics[key] /= num_batches
        
        # Clear buffer
        self.buffer.clear()
        
        # Store metrics
        self.training_metrics.update(metrics)
        
        return metrics
    
    def _update_batch(
        self,
        batch_data: Tuple[torch.Tensor, ...],
        advantages: np.ndarray,
        returns: np.ndarray
    ) -> Dict[str, float]:
        """Update on a single batch."""
        (states, available_seqs, action_masks, actions, old_rewards, 
         old_values, old_log_probs, dones) = batch_data
        
        # Move to device
        states = states.to(self.device)
        available_seqs = available_seqs.to(self.device)
        action_masks = action_masks.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        old_values = old_values.to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Forward pass
        _, new_log_probs, entropy, new_values = self.model.get_action_and_value(
            states, available_seqs, action_masks, actions
        )
        
        # PPO policy loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value function loss
        if self.config.value_clip_ratio > 0:
            value_pred_clipped = old_values + torch.clamp(
                new_values.squeeze() - old_values, 
                -self.config.value_clip_ratio, 
                self.config.value_clip_ratio
            )
            value_loss_clipped = (value_pred_clipped - returns).pow(2)
            value_loss_unclipped = (new_values.squeeze() - returns).pow(2)
            value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
        else:
            value_loss = 0.5 * (new_values.squeeze() - returns).pow(2).mean()
        
        # Entropy loss
        entropy_loss = -entropy.mean()
        
        # Total loss
        total_loss = (
            policy_loss + 
            self.config.value_coef * value_loss + 
            self.config.entropy_coef * entropy_loss
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        
        # Calculate additional metrics
        with torch.no_grad():
            kl_div = (old_log_probs - new_log_probs).mean()
            clip_fraction = ((ratio - 1.0).abs() > self.config.clip_ratio).float().mean()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item(),
            'kl_divergence': kl_div.item(),
            'clip_fraction': clip_fraction.item()
        }
    
    def should_update(self) -> bool:
        """Check if it's time to update the policy."""
        return self.buffer.size >= self.config.update_frequency
    
    def save_checkpoint(self, filepath: str):
        """Save agent checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'training_metrics': self.training_metrics
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_steps = checkpoint.get('total_steps', 0)
        self.total_episodes = checkpoint.get('total_episodes', 0)
        self.training_metrics = checkpoint.get('training_metrics', {})
        logger.info(f"Loaded checkpoint from {filepath}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'average_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'recent_metrics': self.training_metrics
        }
    
    def set_eval_mode(self):
        """Set model to evaluation mode."""
        self.model.eval()
    
    def set_train_mode(self):
        """Set model to training mode."""
        self.model.train()


def create_ppo_agent(
    state_dim: int,
    sequence_feature_dim: int,
    max_sequences: int,
    config: Optional[PPOConfig] = None,
    device: Optional[torch.device] = None
) -> PPOAgent:
    """
    Factory function to create PPO agent.
    
    Args:
        state_dim: Dimension of state space
        sequence_feature_dim: Dimension of sequence features
        max_sequences: Maximum number of sequences
        config: PPO configuration (uses default if None)
        device: Device to run on (uses CPU if None)
        
    Returns:
        Initialized PPO agent
    """
    if config is None:
        config = PPOConfig()
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return PPOAgent(
        state_dim=state_dim,
        sequence_feature_dim=sequence_feature_dim,
        max_sequences=max_sequences,
        config=config,
        device=device
    )