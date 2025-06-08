"""
Utility functions for MSA RL training and inference.
"""

import os
import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

from af_claseq.msa_rl.environment import MSASubsetEnvironment, RewardConfig
from af_claseq.msa_rl.agent import PPOAgent, create_ppo_agent
from af_claseq.utils.sequence_processing import read_a3m_to_dict, write_a3m, get_protein_sequence
from af_claseq.utils.structure_analysis import StructureAnalyzer

logger = logging.getLogger(__name__)


def evaluate_msa_subset(
    msa_subset: Dict[str, str],
    query_sequence: str,
    reward_config: RewardConfig,
    slurm_submitter,
    temp_dir: str = "/tmp/msa_eval"
) -> Dict[str, float]:
    """
    Evaluate an MSA subset by running AF2 prediction and calculating metrics.
    
    Args:
        msa_subset: Dictionary of header -> sequence for the subset
        query_sequence: Query sequence
        reward_config: Reward configuration
        slurm_submitter: SLURM job submitter
        temp_dir: Temporary directory for prediction
        
    Returns:
        Dictionary of evaluation metrics
    """
    import tempfile
    import shutil
    
    # Create temporary directory
    temp_path = Path(temp_dir)
    temp_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create A3M file
        a3m_file = temp_path / "subset.a3m"
        write_a3m(msa_subset, str(a3m_file), reward_config.rare_state_pdb)
        
        # Run AF2 prediction
        job_id = f"eval_{np.random.randint(0, 999999)}"
        slurm_submitter.process_folder(str(temp_path), job_id)
        
        # Find PDB file
        pdb_files = list(temp_path.glob("*.pdb"))
        if not pdb_files:
            logger.warning("No PDB file generated")
            return {'plddt': 0.0, 'rmsd': float('inf'), 'tm_score': 0.0, 'reward': 0.0}
        
        pdb_file = pdb_files[0]
        
        # Calculate metrics
        analyzer = StructureAnalyzer()
        
        # pLDDT
        full_indices = list(range(1, len(query_sequence) + 1))
        plddt = analyzer.plddt_process(str(pdb_file), full_indices)
        
        # RMSD to rare state
        rmsd = analyzer.calculate_ca_rmsd(
            reference_pdb=reward_config.rare_state_pdb,
            target_pdb=str(pdb_file),
            superposition_indices=full_indices,
            rmsd_indices=full_indices
        )
        
        # TM-score to rare state
        tm_score = analyzer.calculate_tm_score(
            target_pdb=str(pdb_file),
            reference_pdb=reward_config.rare_state_pdb
        )
        
        # Calculate reward
        plddt_reward = max(0, (plddt - reward_config.plddt_threshold) / (100.0 - reward_config.plddt_threshold))
        rmsd_reward = max(0, (reward_config.max_rmsd - rmsd) / reward_config.max_rmsd)
        tm_score_reward = tm_score
        
        total_reward = (
            reward_config.plddt_weight * plddt_reward +
            reward_config.rmsd_weight * rmsd_reward +
            reward_config.tm_score_weight * tm_score_reward
        )
        
        return {
            'plddt': plddt or 0.0,
            'rmsd': rmsd if not np.isnan(rmsd) else float('inf'),
            'tm_score': tm_score if not np.isnan(tm_score) else 0.0,
            'reward': max(0.0, total_reward),
            'subset_size': len(msa_subset)
        }
        
    finally:
        # Cleanup
        if temp_path.exists():
            shutil.rmtree(temp_path, ignore_errors=True)


def generate_multiple_subsets(
    agent: PPOAgent,
    env: MSASubsetEnvironment,
    num_subsets: int = 5,
    temperature: float = 1.0
) -> List[Tuple[Dict[str, str], float]]:
    """
    Generate multiple diverse MSA subsets using the trained agent.
    
    Args:
        agent: Trained PPO agent
        env: MSA environment
        num_subsets: Number of subsets to generate
        temperature: Temperature for action sampling (higher = more diverse)
        
    Returns:
        List of (subset, predicted_reward) tuples
    """
    agent.set_eval_mode()
    subsets = []
    
    for i in range(num_subsets):
        state = env.reset()
        current_subset = {}
        predicted_rewards = []
        
        for step in range(env.target_subset_size):
            available_sequences = np.array([
                env.sequence_features[header] 
                for header in env.remaining_headers
            ])
            action_mask = np.ones(len(env.remaining_headers), dtype=np.float32)
            
            # Get action probabilities
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                
                # Pad sequences
                padded_sequences = np.zeros((agent.max_sequences, agent.sequence_feature_dim), dtype=np.float32)
                n_seqs = min(len(available_sequences), agent.max_sequences)
                padded_sequences[:n_seqs] = available_sequences[:n_seqs]
                seq_tensor = torch.FloatTensor(padded_sequences).unsqueeze(0).to(agent.device)
                
                # Pad mask
                padded_mask = np.zeros(agent.max_sequences, dtype=np.float32)
                padded_mask[:len(action_mask)] = action_mask
                mask_tensor = torch.FloatTensor(padded_mask).unsqueeze(0).to(agent.device)
                
                # Get action probabilities and value
                action_probs, value = agent.model(state_tensor, seq_tensor, mask_tensor)
                
                # Apply temperature
                if temperature != 1.0:
                    action_probs = torch.softmax(torch.log(action_probs + 1e-8) / temperature, dim=-1)
                
                # Sample action
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
                
                predicted_rewards.append(value.item())
            
            # Take action
            next_state, _, done, _ = env.step(action.item())
            current_subset.update(env.current_subset)
            state = next_state
            
            if done:
                break
        
        avg_predicted_reward = np.mean(predicted_rewards)
        subsets.append((current_subset.copy(), avg_predicted_reward))
    
    # Sort by predicted reward
    subsets.sort(key=lambda x: x[1], reverse=True)
    
    return subsets


def analyze_learned_policy(
    agent: PPOAgent,
    env: MSASubsetEnvironment,
    output_dir: str
) -> Dict[str, Any]:
    """
    Analyze what the trained agent has learned.
    
    Args:
        agent: Trained agent
        env: Environment
        output_dir: Directory to save analysis results
        
    Returns:
        Analysis results
    """
    agent.set_eval_mode()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    analysis = {
        'sequence_importance': {},
        'selection_patterns': {},
        'diversity_metrics': {}
    }
    
    # Analyze sequence importance
    logger.info("Analyzing sequence importance...")
    sequence_selection_counts = {}
    
    # Run multiple episodes and track selections
    for episode in range(50):
        state = env.reset()
        selected_sequences = []
        
        for step in range(env.target_subset_size):
            available_sequences = np.array([
                env.sequence_features[header] 
                for header in env.remaining_headers
            ])
            action_mask = np.ones(len(env.remaining_headers), dtype=np.float32)
            
            action, _, _ = agent.select_action(state, available_sequences, action_mask, training=False)
            selected_header = env.remaining_headers[action]
            selected_sequences.append(selected_header)
            
            # Count selection
            if selected_header not in sequence_selection_counts:
                sequence_selection_counts[selected_header] = 0
            sequence_selection_counts[selected_header] += 1
            
            state, _, done, _ = env.step(action)
            if done:
                break
    
    # Sort sequences by selection frequency
    sorted_sequences = sorted(sequence_selection_counts.items(), key=lambda x: x[1], reverse=True)
    analysis['sequence_importance'] = {
        'most_selected': sorted_sequences[:20],  # Top 20
        'selection_counts': sequence_selection_counts
    }
    
    # Analyze selection patterns
    logger.info("Analyzing selection patterns...")
    
    # Feature importance analysis
    selected_features = []
    for header, count in sorted_sequences[:50]:  # Top 50 selected sequences
        if header in env.sequence_features:
            selected_features.append(env.sequence_features[header])
    
    if selected_features:
        selected_features = np.array(selected_features)
        all_features = np.array(list(env.sequence_features.values()))
        
        analysis['selection_patterns'] = {
            'selected_mean': np.mean(selected_features, axis=0).tolist(),
            'selected_std': np.std(selected_features, axis=0).tolist(),
            'population_mean': np.mean(all_features, axis=0).tolist(),
            'population_std': np.std(all_features, axis=0).tolist(),
            'feature_names': [
                'aa_composition', 'identity', 'coverage', 'properties', 'gaps', 'length'
            ]
        }
    
    # Save analysis
    with open(output_path / "policy_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)
    
    logger.info(f"Policy analysis saved to {output_path}")
    return analysis


def create_inference_config(
    model_checkpoint: str,
    msa_file: str,
    query_sequence: str,
    rare_state_pdb: str,
    output_dir: str = "./inference_results"
) -> Dict[str, Any]:
    """
    Create configuration for inference runs.
    
    Args:
        model_checkpoint: Path to trained model checkpoint
        msa_file: Path to MSA file
        query_sequence: Query sequence
        rare_state_pdb: Path to rare state reference
        output_dir: Output directory
        
    Returns:
        Inference configuration
    """
    return {
        'model_checkpoint': model_checkpoint,
        'msa_file': msa_file,
        'query_sequence': query_sequence,
        'reward_config': {
            'rare_state_pdb': rare_state_pdb,
            'plddt_weight': 0.3,
            'rmsd_weight': 0.4,
            'tm_score_weight': 0.3,
            'plddt_threshold': 70.0,
            'max_rmsd': 5.0
        },
        'inference_params': {
            'num_subsets': 10,
            'temperature': 1.0,
            'target_subset_size': 10
        },
        'output_dir': output_dir
    }


def validate_msa_file(msa_file: str) -> bool:
    """
    Validate MSA file format and content.
    
    Args:
        msa_file: Path to MSA file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        sequences = read_a3m_to_dict(msa_file)
        
        if len(sequences) < 10:
            logger.error(f"MSA file has too few sequences: {len(sequences)}")
            return False
        
        # Check sequence lengths
        seq_lengths = [len(seq) for seq in sequences.values()]
        if len(set(seq_lengths)) > 1:
            logger.warning("MSA sequences have different lengths - this may cause issues")
        
        logger.info(f"MSA file validated: {len(sequences)} sequences, length {seq_lengths[0]}")
        return True
        
    except Exception as e:
        logger.error(f"Error validating MSA file: {e}")
        return False


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
    """
    level = getattr(logging, log_level.upper())
    
    # Configure logging
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    # Set specific logger levels
    logging.getLogger('af_claseq.msa_rl').setLevel(level)


def get_device_info() -> Dict[str, Any]:
    """Get information about available devices."""
    device_info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None
    }
    
    if torch.cuda.is_available():
        device_info['device_name'] = torch.cuda.get_device_name()
        device_info['memory_allocated'] = torch.cuda.memory_allocated()
        device_info['memory_reserved'] = torch.cuda.memory_reserved()
    
    return device_info