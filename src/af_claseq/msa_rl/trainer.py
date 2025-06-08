"""
Training loop for MSA subset selection RL agent.

This module coordinates training between the environment and agent,
including logging, evaluation, and model checkpointing.
"""

import os
import time
import json
import wandb
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import logging
from collections import deque

from af_claseq.msa_rl.environment import MSASubsetEnvironment, RewardConfig
from af_claseq.msa_rl.agent import PPOAgent, PPOConfig, create_ppo_agent
from af_claseq.msa_rl.sequence_encoder import SequenceEncoder
from af_claseq.utils.slurm_utils import SlurmJobSubmitter

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Training parameters
    total_episodes: int = 1000
    max_steps_per_episode: int = 20
    eval_frequency: int = 50
    save_frequency: int = 100
    
    # Environment parameters
    target_subset_size: int = 10
    max_pool_size: int = 1000
    diversity_bonus: float = 0.1
    
    # Logging
    log_frequency: int = 10
    wandb_project: Optional[str] = "msa_rl"
    wandb_run_name: Optional[str] = None
    
    # Output directories
    output_dir: str = "./msa_rl_results"
    checkpoint_dir: str = "./msa_rl_checkpoints"
    
    # Early stopping
    early_stopping_patience: int = 200
    early_stopping_threshold: float = 0.8  # Target reward threshold


class MSATrainer:
    """
    Main trainer class for MSA subset selection RL.
    """
    
    def __init__(
        self,
        msa_file: str,
        query_sequence: str,
        reward_config: RewardConfig,
        slurm_submitter: SlurmJobSubmitter,
        training_config: TrainingConfig,
        ppo_config: Optional[PPOConfig] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            msa_file: Path to MSA file
            query_sequence: Query sequence string
            reward_config: Reward configuration
            slurm_submitter: SLURM job submitter
            training_config: Training configuration
            ppo_config: PPO agent configuration
            device: Device to run on
        """
        self.training_config = training_config
        self.reward_config = reward_config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create output directories
        self.output_dir = Path(training_config.output_dir)
        self.checkpoint_dir = Path(training_config.checkpoint_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize environment
        logger.info("Initializing training environment...")
        self.env = MSASubsetEnvironment(
            msa_file=msa_file,
            query_sequence=query_sequence,
            reward_config=reward_config,
            slurm_submitter=slurm_submitter,
            target_subset_size=training_config.target_subset_size,
            max_pool_size=training_config.max_pool_size,
            diversity_bonus=training_config.diversity_bonus
        )
        
        # Initialize agent
        logger.info("Initializing PPO agent...")
        if ppo_config is None:
            ppo_config = PPOConfig()
        
        self.agent = create_ppo_agent(
            state_dim=self.env.get_state_dim(),
            sequence_feature_dim=self.env.encoder.get_feature_dim(),
            max_sequences=self.env.max_pool_size,
            config=ppo_config,
            device=self.device
        )
        
        # Training state
        self.episode_count = 0
        self.total_steps = 0
        self.best_reward = -float('inf')
        self.episodes_since_improvement = 0
        
        # Statistics tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.success_episodes = deque(maxlen=100)  # Episodes reaching target reward
        self.training_metrics = {}
        
        # Initialize logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Set up experiment logging."""
        # Save configurations
        config_file = self.output_dir / "training_config.json"
        with open(config_file, 'w') as f:
            json.dump(asdict(self.training_config), f, indent=2)
        
        reward_config_file = self.output_dir / "reward_config.json"
        with open(reward_config_file, 'w') as f:
            json.dump(asdict(self.reward_config), f, indent=2)
        
        # Initialize wandb if specified
        if self.training_config.wandb_project:
            wandb.init(
                project=self.training_config.wandb_project,
                name=self.training_config.wandb_run_name,
                config={
                    **asdict(self.training_config),
                    **asdict(self.reward_config),
                    **asdict(self.agent.config),
                    'state_dim': self.env.get_state_dim(),
                    'sequence_feature_dim': self.env.encoder.get_feature_dim(),
                    'device': str(self.device)
                }
            )
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop.
        
        Returns:
            Training results summary
        """
        logger.info(f"Starting training for {self.training_config.total_episodes} episodes")
        logger.info(f"Target subset size: {self.training_config.target_subset_size}")
        logger.info(f"Device: {self.device}")
        
        start_time = time.time()
        
        try:
            for episode in range(self.training_config.total_episodes):
                self.episode_count = episode + 1
                
                # Run episode
                episode_result = self._run_episode()
                
                # Update statistics
                self._update_statistics(episode_result)
                
                # Update agent if buffer is full
                if self.agent.should_update():
                    last_value = episode_result.get('last_value', 0.0)
                    training_metrics = self.agent.update(last_value)
                    self.training_metrics.update(training_metrics)
                
                # Periodic evaluation
                if episode % self.training_config.eval_frequency == 0:
                    eval_results = self._evaluate()
                    logger.info(f"Episode {episode}: Evaluation reward = {eval_results['mean_reward']:.3f}")
                
                # Logging
                if episode % self.training_config.log_frequency == 0:
                    self._log_progress(episode)
                
                # Save checkpoint
                if episode % self.training_config.save_frequency == 0:
                    self._save_checkpoint(episode)
                
                # Early stopping check
                if self._check_early_stopping():
                    logger.info(f"Early stopping at episode {episode}")
                    break
                    
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
        finally:
            # Final cleanup
            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Save final model
            self._save_checkpoint("final")
            
            # Close environment
            self.env.close()
        
        return self._get_training_summary()
    
    def _run_episode(self) -> Dict[str, Any]:
        """
        Run a single training episode.
        
        Returns:
            Episode results
        """
        state = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        last_value = 0.0
        
        for step in range(self.training_config.max_steps_per_episode):
            # Get available sequences and action mask
            available_sequences = np.array([
                self.env.sequence_features[header] 
                for header in self.env.remaining_headers
            ])
            action_mask = np.ones(len(self.env.remaining_headers), dtype=np.float32)
            
            # Select action
            action, log_prob, value = self.agent.select_action(
                state, available_sequences, action_mask, training=True
            )
            
            # Take step in environment
            next_state, reward, done, info = self.env.step(action)
            
            # Store experience
            self.agent.store_experience(
                state, available_sequences, action_mask, action, 
                reward, value, log_prob, done
            )
            
            # Update tracking variables
            episode_reward += reward
            episode_length += 1
            last_value = value
            state = next_state
            self.total_steps += 1
            
            if done:
                break
        
        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'success': episode_reward > self.training_config.early_stopping_threshold,
            'last_value': last_value,
            'info': info
        }
    
    def _evaluate(self, num_episodes: int = 5) -> Dict[str, float]:
        """
        Evaluate current policy.
        
        Args:
            num_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation metrics
        """
        self.agent.set_eval_mode()
        
        eval_rewards = []
        eval_lengths = []
        eval_successes = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0.0
            episode_length = 0
            
            for step in range(self.training_config.max_steps_per_episode):
                available_sequences = np.array([
                    self.env.sequence_features[header] 
                    for header in self.env.remaining_headers
                ])
                action_mask = np.ones(len(self.env.remaining_headers), dtype=np.float32)
                
                # Select action (no exploration)
                action, _, _ = self.agent.select_action(
                    state, available_sequences, action_mask, training=False
                )
                
                next_state, reward, done, _ = self.env.step(action)
                
                episode_reward += reward
                episode_length += 1
                state = next_state
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
            eval_successes.append(episode_reward > self.training_config.early_stopping_threshold)
        
        self.agent.set_train_mode()
        
        return {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_length': np.mean(eval_lengths),
            'success_rate': np.mean(eval_successes)
        }
    
    def _update_statistics(self, episode_result: Dict[str, Any]):
        """Update training statistics."""
        self.episode_rewards.append(episode_result['episode_reward'])
        self.episode_lengths.append(episode_result['episode_length'])
        self.success_episodes.append(episode_result['success'])
        
        # Track best reward
        if episode_result['episode_reward'] > self.best_reward:
            self.best_reward = episode_result['episode_reward']
            self.episodes_since_improvement = 0
        else:
            self.episodes_since_improvement += 1
    
    def _log_progress(self, episode: int):
        """Log training progress."""
        if not self.episode_rewards:
            return
        
        avg_reward = np.mean(self.episode_rewards)
        avg_length = np.mean(self.episode_lengths)
        success_rate = np.mean(self.success_episodes)
        
        log_msg = (
            f"Episode {episode}: "
            f"Avg Reward = {avg_reward:.3f}, "
            f"Avg Length = {avg_length:.1f}, "
            f"Success Rate = {success_rate:.2f}, "
            f"Best Reward = {self.best_reward:.3f}"
        )
        logger.info(log_msg)
        
        # Wandb logging
        if self.training_config.wandb_project:
            log_dict = {
                'episode': episode,
                'avg_reward': avg_reward,
                'avg_length': avg_length,
                'success_rate': success_rate,
                'best_reward': self.best_reward,
                'total_steps': self.total_steps
            }
            
            # Add training metrics if available
            if self.training_metrics:
                for key, value in self.training_metrics.items():
                    log_dict[f'training/{key}'] = value
            
            wandb.log(log_dict)
    
    def _save_checkpoint(self, episode: Any):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_episode_{episode}.pt"
        self.agent.save_checkpoint(str(checkpoint_path))
        
        # Save additional training state
        training_state = {
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'best_reward': self.best_reward,
            'episodes_since_improvement': self.episodes_since_improvement,
            'episode_rewards': list(self.episode_rewards),
            'episode_lengths': list(self.episode_lengths),
            'success_episodes': list(self.success_episodes)
        }
        
        state_path = self.checkpoint_dir / f"training_state_episode_{episode}.json"
        with open(state_path, 'w') as f:
            json.dump(training_state, f, indent=2)
    
    def _check_early_stopping(self) -> bool:
        """Check if early stopping criteria are met."""
        if self.episodes_since_improvement >= self.training_config.early_stopping_patience:
            return True
        
        # Check if we've achieved consistent success
        if len(self.success_episodes) >= 50:
            recent_success_rate = np.mean(list(self.success_episodes)[-50:])
            if recent_success_rate >= 0.8:  # 80% success rate
                return True
        
        return False
    
    def _get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training results."""
        return {
            'total_episodes': self.episode_count,
            'total_steps': self.total_steps,
            'best_reward': self.best_reward,
            'final_avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'final_success_rate': np.mean(self.success_episodes) if self.success_episodes else 0.0,
            'training_completed': True
        }
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        # Load agent
        self.agent.load_checkpoint(checkpoint_path)
        
        # Load training state if available
        state_path = checkpoint_path.replace('.pt', '_training_state.json')
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                training_state = json.load(f)
            
            self.episode_count = training_state.get('episode_count', 0)
            self.total_steps = training_state.get('total_steps', 0)
            self.best_reward = training_state.get('best_reward', -float('inf'))
            self.episodes_since_improvement = training_state.get('episodes_since_improvement', 0)
            
            # Restore statistics
            if 'episode_rewards' in training_state:
                self.episode_rewards.extend(training_state['episode_rewards'])
            if 'episode_lengths' in training_state:
                self.episode_lengths.extend(training_state['episode_lengths'])
            if 'success_episodes' in training_state:
                self.success_episodes.extend(training_state['success_episodes'])


def train_msa_rl_agent(
    msa_file: str,
    query_sequence: str,
    rare_state_pdb: str,
    slurm_submitter: SlurmJobSubmitter,
    output_dir: str = "./msa_rl_results",
    **kwargs
) -> Dict[str, Any]:
    """
    High-level function to train MSA RL agent.
    
    Args:
        msa_file: Path to MSA file
        query_sequence: Query sequence string
        rare_state_pdb: Path to rare state reference PDB
        slurm_submitter: SLURM job submitter instance
        output_dir: Output directory for results
        **kwargs: Additional configuration parameters
        
    Returns:
        Training results
    """
    # Create configurations
    reward_config = RewardConfig(rare_state_pdb=rare_state_pdb)
    training_config = TrainingConfig(output_dir=output_dir)
    
    # Update configs with kwargs
    for key, value in kwargs.items():
        if hasattr(reward_config, key):
            setattr(reward_config, key, value)
        elif hasattr(training_config, key):
            setattr(training_config, key, value)
    
    # Create trainer
    trainer = MSATrainer(
        msa_file=msa_file,
        query_sequence=query_sequence,
        reward_config=reward_config,
        slurm_submitter=slurm_submitter,
        training_config=training_config
    )
    
    # Train
    results = trainer.train()
    
    return results