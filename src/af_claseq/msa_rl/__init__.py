"""
MSA RL Module for AF-ClaSeq.

This module provides reinforcement learning capabilities for intelligent
MSA subset selection to predict rare protein conformational states.
"""

from af_claseq.msa_rl.environment import MSASubsetEnvironment, RewardConfig, create_reward_config
from af_claseq.msa_rl.agent import PPOAgent, PPOConfig, create_ppo_agent
from af_claseq.msa_rl.trainer import MSATrainer, TrainingConfig, train_msa_rl_agent
from af_claseq.msa_rl.sequence_encoder import SequenceEncoder, create_sequence_features
from af_claseq.msa_rl.models import ActorCriticNetwork, create_actor_critic_model
from af_claseq.msa_rl.utils import (
    evaluate_msa_subset,
    generate_multiple_subsets,
    analyze_learned_policy,
    create_inference_config,
    validate_msa_file,
    setup_logging,
    get_device_info
)

__version__ = "1.0.0"
__author__ = "AF-ClaSeq Team"

__all__ = [
    # Environment
    "MSASubsetEnvironment",
    "RewardConfig", 
    "create_reward_config",
    
    # Agent
    "PPOAgent",
    "PPOConfig",
    "create_ppo_agent",
    
    # Training
    "MSATrainer",
    "TrainingConfig",
    "train_msa_rl_agent",
    
    # Models
    "ActorCriticNetwork",
    "create_actor_critic_model",
    
    # Sequence encoding
    "SequenceEncoder",
    "create_sequence_features",
    
    # Utilities
    "evaluate_msa_subset",
    "generate_multiple_subsets", 
    "analyze_learned_policy",
    "create_inference_config",
    "validate_msa_file",
    "setup_logging",
    "get_device_info"
]