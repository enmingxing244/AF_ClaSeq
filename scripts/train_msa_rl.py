#!/usr/bin/env python3
"""
MSA RL Training Script

This script trains a reinforcement learning agent to select informative
MSA subsets for predicting rare protein conformational states.

Usage:
    python train_msa_rl.py --config config.yaml
    
Example config.yaml:
    msa_file: "/path/to/input.a3m"
    query_sequence: "MGKV..."  # or query_pdb: "/path/to/query.pdb"
    rare_state_pdb: "/path/to/rare_state.pdb"
    common_state_pdb: "/path/to/common_state.pdb"  # optional
    
    # SLURM configuration
    slurm:
        conda_env_path: "/path/to/conda/env"
        slurm_account: "your_account"
        slurm_partition: "gpu"
        slurm_time: "04:00:00"
        
    # Training configuration
    training:
        total_episodes: 1000
        target_subset_size: 10
        output_dir: "./msa_rl_results"
        wandb_project: "msa_rl_experiment"
        
    # PPO configuration
    ppo:
        learning_rate: 3e-4
        gamma: 0.99
        clip_ratio: 0.2
"""

import argparse
import yaml
import sys
import os
import logging
from pathlib import Path

from af_claseq.msa_rl import (
    train_msa_rl_agent,
    RewardConfig,
    TrainingConfig,
    PPOConfig,
    setup_logging,
    validate_msa_file,
    get_device_info
)
from af_claseq.utils.slurm_utils import SlurmJobSubmitter
from af_claseq.utils.sequence_processing import get_protein_sequence

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def validate_config(config: dict) -> bool:
    """Validate configuration parameters."""
    required_fields = ['msa_file', 'rare_state_pdb', 'slurm']
    
    for field in required_fields:
        if field not in config:
            logger.error(f"Required field '{field}' missing from config")
            return False
    
    # Check file existence
    if not os.path.exists(config['msa_file']):
        logger.error(f"MSA file not found: {config['msa_file']}")
        return False
        
    if not os.path.exists(config['rare_state_pdb']):
        logger.error(f"Rare state PDB not found: {config['rare_state_pdb']}")
        return False
    
    # Validate MSA file
    if not validate_msa_file(config['msa_file']):
        return False
    
    return True


def create_slurm_submitter(slurm_config: dict) -> SlurmJobSubmitter:
    """Create SLURM job submitter from config."""
    return SlurmJobSubmitter(
        conda_env_path=slurm_config['conda_env_path'],
        slurm_account=slurm_config['slurm_account'],
        slurm_output=slurm_config.get('slurm_output', '/dev/null'),
        slurm_error=slurm_config.get('slurm_error', '/dev/null'),
        slurm_nodes=slurm_config.get('slurm_nodes', 1),
        slurm_gpus_per_task=slurm_config.get('slurm_gpus_per_task', 1),
        slurm_tasks=slurm_config.get('slurm_tasks', 1),
        slurm_cpus_per_task=slurm_config.get('slurm_cpus_per_task', 4),
        slurm_time=slurm_config.get('slurm_time', '04:00:00'),
        slurm_partition=slurm_config.get('slurm_partition', 'gpu'),
        check_interval=60,
        job_name_prefix="msa_rl",
        prediction_num_model=1,
        prediction_num_seed=1
    )


def get_query_sequence(config: dict) -> str:
    """Get query sequence from config."""
    if 'query_sequence' in config:
        return config['query_sequence']
    elif 'query_pdb' in config:
        if not os.path.exists(config['query_pdb']):
            raise FileNotFoundError(f"Query PDB not found: {config['query_pdb']}")
        return get_protein_sequence(config['query_pdb'])
    else:
        # Try to extract from rare state PDB
        logger.warning("No query sequence specified, extracting from rare state PDB")
        return get_protein_sequence(config['rare_state_pdb'])


def main():
    parser = argparse.ArgumentParser(description="Train MSA RL Agent")
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    parser.add_argument(
        '--resume',
        type=str,
        help='Path to checkpoint to resume training from'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        
        # Validate configuration
        if not validate_config(config):
            logger.error("Configuration validation failed")
            sys.exit(1)
        
        # Print device information
        device_info = get_device_info()
        logger.info(f"Device info: {device_info}")
        
        # Get query sequence
        query_sequence = get_query_sequence(config)
        logger.info(f"Query sequence length: {len(query_sequence)}")
        
        # Create SLURM submitter
        slurm_submitter = create_slurm_submitter(config['slurm'])
        
        # Create reward configuration
        reward_config = RewardConfig(
            rare_state_pdb=config['rare_state_pdb'],
            common_state_pdb=config.get('common_state_pdb'),
            **config.get('reward', {})
        )
        
        # Create training configuration
        training_config = TrainingConfig(
            **config.get('training', {})
        )
        
        # Create PPO configuration
        ppo_config = PPOConfig(
            **config.get('ppo', {})
        )
        
        logger.info("Starting MSA RL training...")
        logger.info(f"MSA file: {config['msa_file']}")
        logger.info(f"Rare state PDB: {config['rare_state_pdb']}")
        logger.info(f"Target subset size: {training_config.target_subset_size}")
        logger.info(f"Total episodes: {training_config.total_episodes}")
        logger.info(f"Output directory: {training_config.output_dir}")
        
        # Train the agent
        results = train_msa_rl_agent(
            msa_file=config['msa_file'],
            query_sequence=query_sequence,
            rare_state_pdb=config['rare_state_pdb'],
            slurm_submitter=slurm_submitter,
            output_dir=training_config.output_dir,
            # Pass all configuration parameters
            **vars(training_config),
            **vars(ppo_config),
            **vars(reward_config)
        )
        
        # Print results
        logger.info("Training completed successfully!")
        logger.info(f"Total episodes: {results['total_episodes']}")
        logger.info(f"Best reward achieved: {results['best_reward']:.3f}")
        logger.info(f"Final average reward: {results['final_avg_reward']:.3f}")
        logger.info(f"Final success rate: {results['final_success_rate']:.2f}")
        
        # Save final results
        results_file = Path(training_config.output_dir) / "final_results.yaml"
        with open(results_file, 'w') as f:
            yaml.dump(results, f, indent=2)
        logger.info(f"Results saved to {results_file}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()