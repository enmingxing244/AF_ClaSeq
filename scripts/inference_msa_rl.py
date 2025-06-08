#!/usr/bin/env python3
"""
MSA RL Inference Script

This script uses a trained RL agent to generate informative MSA subsets
for predicting rare protein conformational states.

Usage:
    python inference_msa_rl.py --model checkpoint.pt --msa input.a3m --rare-state rare.pdb --output results/
    
The script will:
1. Load the trained model
2. Generate multiple diverse MSA subsets
3. Evaluate each subset using AF2 prediction
4. Save the best subsets and analysis results
"""

import argparse
import yaml
import sys
import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging

from af_claseq.msa_rl import (
    PPOAgent,
    MSASubsetEnvironment,
    RewardConfig,
    generate_multiple_subsets,
    evaluate_msa_subset,
    analyze_learned_policy,
    setup_logging,
    validate_msa_file
)
from af_claseq.utils.slurm_utils import SlurmJobSubmitter
from af_claseq.utils.sequence_processing import get_protein_sequence, write_a3m
from af_claseq.utils.structure_analysis import StructureAnalyzer

logger = logging.getLogger(__name__)


def load_trained_agent(
    checkpoint_path: str,
    state_dim: int,
    sequence_feature_dim: int,
    max_sequences: int,
    device: torch.device
) -> PPOAgent:
    """Load trained agent from checkpoint."""
    from af_claseq.msa_rl.agent import create_ppo_agent, PPOConfig
    
    # Create agent with default config (will be overridden by checkpoint)
    agent = create_ppo_agent(
        state_dim=state_dim,
        sequence_feature_dim=sequence_feature_dim,
        max_sequences=max_sequences,
        config=PPOConfig(),
        device=device
    )
    
    # Load checkpoint
    agent.load_checkpoint(checkpoint_path)
    agent.set_eval_mode()
    
    logger.info(f"Loaded trained agent from {checkpoint_path}")
    return agent


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
        slurm_time=slurm_config.get('slurm_time', '02:00:00'),
        slurm_partition=slurm_config.get('slurm_partition', 'gpu'),
        check_interval=60,
        job_name_prefix="msa_rl_inference",
        prediction_num_model=1,
        prediction_num_seed=1
    )


def run_inference(
    model_path: str,
    msa_file: str,
    rare_state_pdb: str,
    slurm_config: dict,
    output_dir: str,
    num_subsets: int = 10,
    temperature: float = 1.0,
    target_subset_size: int = 10,
    evaluate_subsets: bool = True
) -> Dict:
    """
    Run inference with trained model.
    
    Args:
        model_path: Path to trained model checkpoint
        msa_file: Path to input MSA file
        rare_state_pdb: Path to rare state reference PDB
        slurm_config: SLURM configuration
        output_dir: Output directory
        num_subsets: Number of subsets to generate
        temperature: Sampling temperature
        target_subset_size: Size of generated subsets
        evaluate_subsets: Whether to evaluate subsets with AF2
        
    Returns:
        Results dictionary
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get query sequence
    query_sequence = get_protein_sequence(rare_state_pdb)
    logger.info(f"Query sequence length: {len(query_sequence)}")
    
    # Create reward configuration
    reward_config = RewardConfig(rare_state_pdb=rare_state_pdb)
    
    # Create SLURM submitter
    slurm_submitter = create_slurm_submitter(slurm_config)
    
    # Create environment
    logger.info("Creating environment...")
    env = MSASubsetEnvironment(
        msa_file=msa_file,
        query_sequence=query_sequence,
        reward_config=reward_config,
        slurm_submitter=slurm_submitter,
        target_subset_size=target_subset_size,
        temp_dir=str(output_path / "temp")
    )
    
    # Load trained agent
    logger.info("Loading trained agent...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = load_trained_agent(
        checkpoint_path=model_path,
        state_dim=env.get_state_dim(),
        sequence_feature_dim=env.encoder.get_feature_dim(),
        max_sequences=env.max_pool_size,
        device=device
    )
    
    # Generate subsets
    logger.info(f"Generating {num_subsets} MSA subsets...")
    subsets_with_scores = generate_multiple_subsets(
        agent=agent,
        env=env,
        num_subsets=num_subsets,
        temperature=temperature
    )
    
    results = {
        'generated_subsets': [],
        'evaluations': [],
        'best_subset_idx': 0,
        'analysis': {}
    }
    
    # Save generated subsets
    subsets_dir = output_path / "generated_subsets"
    subsets_dir.mkdir(exist_ok=True)
    
    for i, (subset, predicted_reward) in enumerate(subsets_with_scores):
        subset_info = {
            'index': i,
            'predicted_reward': predicted_reward,
            'subset_size': len(subset),
            'sequence_headers': list(subset.keys()),
            'file_path': str(subsets_dir / f"subset_{i:02d}.a3m")
        }
        
        # Save A3M file
        write_a3m(subset, subset_info['file_path'], rare_state_pdb)
        
        results['generated_subsets'].append(subset_info)
        logger.info(f"Subset {i}: {len(subset)} sequences, predicted reward: {predicted_reward:.3f}")
    
    # Evaluate subsets if requested
    if evaluate_subsets:
        logger.info("Evaluating subsets with AF2 predictions...")
        
        for i, (subset, predicted_reward) in enumerate(subsets_with_scores):
            logger.info(f"Evaluating subset {i}/{len(subsets_with_scores)}")
            
            try:
                evaluation = evaluate_msa_subset(
                    msa_subset=subset,
                    query_sequence=query_sequence,
                    reward_config=reward_config,
                    slurm_submitter=slurm_submitter,
                    temp_dir=str(output_path / f"eval_temp_{i}")
                )
                
                evaluation['predicted_reward'] = predicted_reward
                evaluation['subset_index'] = i
                results['evaluations'].append(evaluation)
                
                logger.info(f"Subset {i} evaluation: pLDDT={evaluation['plddt']:.1f}, "
                          f"RMSD={evaluation['rmsd']:.2f}, TM-score={evaluation['tm_score']:.3f}, "
                          f"Reward={evaluation['reward']:.3f}")
                
            except Exception as e:
                logger.error(f"Failed to evaluate subset {i}: {e}")
                evaluation = {
                    'subset_index': i,
                    'predicted_reward': predicted_reward,
                    'plddt': 0.0,
                    'rmsd': float('inf'),
                    'tm_score': 0.0,
                    'reward': 0.0,
                    'error': str(e)
                }
                results['evaluations'].append(evaluation)
        
        # Find best subset
        if results['evaluations']:
            best_idx = max(range(len(results['evaluations'])), 
                          key=lambda i: results['evaluations'][i]['reward'])
            results['best_subset_idx'] = best_idx
            
            logger.info(f"Best subset: {best_idx} (reward: {results['evaluations'][best_idx]['reward']:.3f})")
    
    # Analyze learned policy
    logger.info("Analyzing learned policy...")
    try:
        analysis = analyze_learned_policy(
            agent=agent,
            env=env,
            output_dir=str(output_path / "analysis")
        )
        results['analysis'] = analysis
    except Exception as e:
        logger.error(f"Failed to analyze policy: {e}")
    
    # Save results
    results_file = output_path / "inference_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Inference results saved to {results_file}")
    
    # Create summary report
    _create_summary_report(results, output_path)
    
    # Clean up
    env.close()
    
    return results


def _create_summary_report(results: Dict, output_path: Path):
    """Create a human-readable summary report."""
    report_file = output_path / "summary_report.txt"
    
    with open(report_file, 'w') as f:
        f.write("MSA RL Inference Summary Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Generated {len(results['generated_subsets'])} MSA subsets\n\n")
        
        if results['evaluations']:
            f.write("Evaluation Results:\n")
            f.write("-" * 20 + "\n")
            
            evaluations = sorted(results['evaluations'], 
                               key=lambda x: x['reward'], reverse=True)
            
            f.write("Rank | Subset | pLDDT | RMSD  | TM-score | Reward | Predicted\n")
            f.write("-----|--------|-------|-------|----------|--------|-----------\n")
            
            for rank, eval_result in enumerate(evaluations[:10], 1):
                f.write(f"{rank:4d} | {eval_result['subset_index']:6d} | "
                       f"{eval_result['plddt']:5.1f} | "
                       f"{eval_result['rmsd']:5.2f} | "
                       f"{eval_result['tm_score']:8.3f} | "
                       f"{eval_result['reward']:6.3f} | "
                       f"{eval_result['predicted_reward']:9.3f}\n")
            
            f.write(f"\nBest subset: {results['best_subset_idx']} "
                   f"(reward: {evaluations[0]['reward']:.3f})\n")
        
        if 'analysis' in results and 'sequence_importance' in results['analysis']:
            f.write("\nMost Selected Sequences:\n")
            f.write("-" * 25 + "\n")
            
            most_selected = results['analysis']['sequence_importance']['most_selected']
            for header, count in most_selected[:10]:
                f.write(f"{header}: {count} times\n")
    
    logger.info(f"Summary report saved to {report_file}")


def main():
    parser = argparse.ArgumentParser(description="MSA RL Inference")
    parser.add_argument(
        '--model', 
        type=str, 
        required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--msa',
        type=str,
        required=True,
        help='Path to input MSA file'
    )
    parser.add_argument(
        '--rare-state',
        type=str,
        required=True,
        help='Path to rare state reference PDB'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory'
    )
    parser.add_argument(
        '--slurm-config',
        type=str,
        required=True,
        help='Path to SLURM configuration YAML file'
    )
    parser.add_argument(
        '--num-subsets',
        type=int,
        default=10,
        help='Number of subsets to generate'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Sampling temperature (higher = more diverse)'
    )
    parser.add_argument(
        '--subset-size',
        type=int,
        default=10,
        help='Target subset size'
    )
    parser.add_argument(
        '--no-evaluate',
        action='store_true',
        help='Skip AF2 evaluation of generated subsets'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    
    try:
        # Validate inputs
        if not os.path.exists(args.model):
            logger.error(f"Model checkpoint not found: {args.model}")
            sys.exit(1)
            
        if not os.path.exists(args.msa):
            logger.error(f"MSA file not found: {args.msa}")
            sys.exit(1)
            
        if not os.path.exists(args.rare_state):
            logger.error(f"Rare state PDB not found: {args.rare_state}")
            sys.exit(1)
            
        if not os.path.exists(args.slurm_config):
            logger.error(f"SLURM config not found: {args.slurm_config}")
            sys.exit(1)
        
        # Validate MSA file
        if not validate_msa_file(args.msa):
            logger.error("MSA file validation failed")
            sys.exit(1)
        
        # Load SLURM configuration
        with open(args.slurm_config, 'r') as f:
            slurm_config = yaml.safe_load(f)['slurm']
        
        # Run inference
        logger.info("Starting MSA RL inference...")
        results = run_inference(
            model_path=args.model,
            msa_file=args.msa,
            rare_state_pdb=args.rare_state,
            slurm_config=slurm_config,
            output_dir=args.output,
            num_subsets=args.num_subsets,
            temperature=args.temperature,
            target_subset_size=args.subset_size,
            evaluate_subsets=not args.no_evaluate
        )
        
        logger.info("Inference completed successfully!")
        if results['evaluations']:
            best_reward = max(eval_result['reward'] for eval_result in results['evaluations'])
            logger.info(f"Best subset achieved reward: {best_reward:.3f}")
        
    except KeyboardInterrupt:
        logger.info("Inference interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()