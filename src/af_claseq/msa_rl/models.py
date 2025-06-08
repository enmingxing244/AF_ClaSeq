"""
Neural network models for MSA subset selection RL.

This module contains the policy and value networks for the PPO agent,
designed to handle variable-length action spaces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any
import math


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for sequence selection."""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, key.size(1), self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, value.size(1), self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads and linear transformation
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        output = self.w_o(attention_output)
        
        # Residual connection and layer norm
        return self.layer_norm(output + query)
    
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        return torch.matmul(attention_weights, V)


class SequenceEncoder(nn.Module):
    """Encoder for individual sequence features."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class PolicyNetwork(nn.Module):
    """
    Policy network for sequence selection using attention mechanism.
    Handles variable-length action spaces by attending over available sequences.
    """
    
    def __init__(
        self,
        state_dim: int,
        sequence_feature_dim: int,
        hidden_dim: int = 256,
        embedding_dim: int = 128,
        n_attention_heads: int = 8,
        n_layers: int = 2
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.sequence_feature_dim = sequence_feature_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Sequence encoder
        self.sequence_encoder = SequenceEncoder(
            sequence_feature_dim, hidden_dim, embedding_dim
        )
        
        # Attention layers
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(embedding_dim, n_attention_heads)
            for _ in range(n_layers)
        ])
        
        # Final scoring layer
        self.action_scorer = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(
        self,
        state: torch.Tensor,
        available_sequences: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of policy network.
        
        Args:
            state: Current state representation [batch_size, state_dim]
            available_sequences: Available sequence features [batch_size, n_sequences, seq_feature_dim]
            action_mask: Mask for valid actions [batch_size, n_sequences]
            
        Returns:
            Action probabilities [batch_size, n_sequences]
        """
        batch_size = state.size(0)
        n_sequences = available_sequences.size(1)
        
        # Encode state
        state_embedding = self.state_encoder(state)  # [batch_size, embedding_dim]
        
        # Encode sequences
        seq_embeddings = self.sequence_encoder(available_sequences)  # [batch_size, n_seq, embedding_dim]
        
        # Add state context to sequence embeddings
        state_expanded = state_embedding.unsqueeze(1).expand(-1, n_sequences, -1)
        
        # Apply attention layers
        attended_seqs = seq_embeddings
        for attention_layer in self.attention_layers:
            attended_seqs = attention_layer(attended_seqs, attended_seqs, attended_seqs)
        
        # Combine state and sequence representations
        combined = torch.cat([attended_seqs, state_expanded], dim=-1)
        
        # Score each sequence
        action_scores = self.action_scorer(combined).squeeze(-1)  # [batch_size, n_sequences]
        
        # Apply mask if provided
        if action_mask is not None:
            action_scores = action_scores.masked_fill(action_mask == 0, -1e9)
        
        # Convert to probabilities
        action_probs = F.softmax(action_scores, dim=-1)
        
        return action_probs
    
    def get_action_distribution(
        self,
        state: torch.Tensor,
        available_sequences: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None
    ) -> torch.distributions.Categorical:
        """Get action distribution for sampling."""
        action_probs = self.forward(state, available_sequences, action_mask)
        return torch.distributions.Categorical(action_probs)


class ValueNetwork(nn.Module):
    """Value network for state value estimation."""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of value network.
        
        Args:
            state: State representation [batch_size, state_dim]
            
        Returns:
            State values [batch_size, 1]
        """
        return self.value_net(state)


class ActorCriticNetwork(nn.Module):
    """Combined actor-critic network for PPO."""
    
    def __init__(
        self,
        state_dim: int,
        sequence_feature_dim: int,
        hidden_dim: int = 256,
        embedding_dim: int = 128,
        n_attention_heads: int = 8,
        n_attention_layers: int = 2,
        shared_layers: bool = True
    ):
        super().__init__()
        
        self.shared_layers = shared_layers
        
        # Shared feature extractor (optional)
        if shared_layers:
            self.shared_encoder = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
            policy_state_dim = hidden_dim
            value_state_dim = hidden_dim
        else:
            self.shared_encoder = None
            policy_state_dim = state_dim
            value_state_dim = state_dim
        
        # Policy network (actor)
        self.policy_net = PolicyNetwork(
            state_dim=policy_state_dim,
            sequence_feature_dim=sequence_feature_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            n_attention_heads=n_attention_heads,
            n_layers=n_attention_layers
        )
        
        # Value network (critic)
        self.value_net = ValueNetwork(
            state_dim=value_state_dim,
            hidden_dim=hidden_dim
        )
        
    def forward(
        self,
        state: torch.Tensor,
        available_sequences: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both action probabilities and state values.
        
        Args:
            state: Current state representation [batch_size, state_dim]
            available_sequences: Available sequence features [batch_size, n_sequences, seq_feature_dim]
            action_mask: Mask for valid actions [batch_size, n_sequences]
            
        Returns:
            action_probs: Action probabilities [batch_size, n_sequences]
            state_values: State values [batch_size, 1]
        """
        # Apply shared encoder if using shared layers
        if self.shared_layers:
            shared_features = self.shared_encoder(state)
            policy_state = shared_features
            value_state = shared_features
        else:
            policy_state = state
            value_state = state
        
        # Get action probabilities from policy network
        action_probs = self.policy_net(policy_state, available_sequences, action_mask)
        
        # Get state values from value network
        state_values = self.value_net(value_state)
        
        return action_probs, state_values
    
    def get_action_and_value(
        self,
        state: torch.Tensor,
        available_sequences: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, entropy, and value for PPO training.
        
        Args:
            state: Current state representation
            available_sequences: Available sequence features
            action_mask: Mask for valid actions
            action: Specific action to evaluate (if None, sample from policy)
            
        Returns:
            action: Selected or provided action
            log_prob: Log probability of the action
            entropy: Policy entropy
            value: State value
        """
        action_probs, value = self.forward(state, available_sequences, action_mask)
        dist = torch.distributions.Categorical(action_probs)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value


def create_actor_critic_model(
    state_dim: int,
    sequence_feature_dim: int,
    device: torch.device,
    **model_kwargs
) -> ActorCriticNetwork:
    """
    Create and initialize actor-critic model.
    
    Args:
        state_dim: Dimension of state space
        sequence_feature_dim: Dimension of sequence features
        device: Device to place model on
        **model_kwargs: Additional model arguments
        
    Returns:
        Initialized actor-critic model
    """
    model = ActorCriticNetwork(
        state_dim=state_dim,
        sequence_feature_dim=sequence_feature_dim,
        **model_kwargs
    )
    
    # Initialize weights
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0)
    
    return model.to(device)


class SequenceSelectionHead(nn.Module):
    """
    Alternative simpler head for sequence selection.
    Uses direct attention between state and sequences.
    """
    
    def __init__(self, state_dim: int, sequence_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.seq_proj = nn.Linear(sequence_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, 1)
        
    def forward(self, state: torch.Tensor, sequences: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            state: [batch_size, state_dim]
            sequences: [batch_size, n_sequences, sequence_dim]
            mask: [batch_size, n_sequences]
        
        Returns:
            action_logits: [batch_size, n_sequences]
        """
        batch_size, n_sequences, _ = sequences.shape
        
        # Project state and sequences to same dimension
        state_emb = self.state_proj(state).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        seq_emb = self.seq_proj(sequences)  # [batch_size, n_sequences, hidden_dim]
        
        # Apply attention with state as query, sequences as key/value
        attn_mask = None
        if mask is not None:
            attn_mask = mask.unsqueeze(1)  # [batch_size, 1, n_sequences]
        
        attended, _ = self.attention(state_emb, seq_emb, seq_emb, key_padding_mask=attn_mask)
        
        # Project to action scores
        action_logits = self.output_proj(attended).squeeze(-1).squeeze(1)
        
        # Apply mask
        if mask is not None:
            action_logits = action_logits.masked_fill(mask == 0, -1e9)
        
        return action_logits