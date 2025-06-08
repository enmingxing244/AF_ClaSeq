"""
Reinforcement Learning Environment for MSA Subset Selection.

This module implements the RL environment for learning to select
informative MSA subsets for rare conformation prediction.
"""

import os
import tempfile
import shutil
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
import random
import json
from dataclasses import dataclass

from af_claseq.utils.sequence_processing import read_a3m_to_dict, write_a3m, get_protein_sequence
from af_claseq.utils.structure_analysis import StructureAnalyzer
from af_claseq.utils.slurm_utils import SlurmJobSubmitter
from af_claseq.msa_rl.sequence_encoder import SequenceEncoder

logger = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    """Configuration for reward calculation."""
    rare_state_pdb: str  # Reference PDB for rare state
    common_state_pdb: Optional[str] = None  # Reference PDB for common state
    plddt_weight: float = 0.3
    rmsd_weight: float = 0.4
    tm_score_weight: float = 0.3
    plddt_threshold: float = 70.0
    max_rmsd: float = 5.0  # RMSD values above this get zero reward


class MSASubsetEnvironment:
    """
    RL Environment for MSA subset selection.
    
    State: Current subset composition + remaining sequence pool features
    Action: Select next sequence to add to subset
    Reward: Structural quality of AF2 prediction (pLDDT + RMSD to rare state)
    """
    
    def __init__(
        self,
        msa_file: str,
        query_sequence: str,
        reward_config: RewardConfig,
        slurm_submitter: SlurmJobSubmitter,
        target_subset_size: int = 10,
        max_pool_size: int = 1000,
        temp_dir: str = "/tmp/msa_rl",
        embedding_dim: int = 64,
        diversity_bonus: float = 0.1
    ):
        """
        Initialize the MSA subset selection environment.
        
        Args:
            msa_file: Path to input MSA file
            query_sequence: Query protein sequence
            reward_config: Configuration for reward calculation
            slurm_submitter: SLURM job submitter instance
            target_subset_size: Target size of MSA subset
            max_pool_size: Maximum number of sequences to consider
            temp_dir: Temporary directory for AF2 predictions
            embedding_dim: Dimension of sequence embeddings
            diversity_bonus: Bonus for sequence diversity
        """
        self.msa_file = msa_file
        self.query_sequence = query_sequence
        self.reward_config = reward_config
        self.slurm_submitter = slurm_submitter
        self.target_subset_size = target_subset_size
        self.max_pool_size = max_pool_size
        self.temp_dir = Path(temp_dir)
        self.embedding_dim = embedding_dim
        self.diversity_bonus = diversity_bonus
        
        # Create temp directory
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize sequence encoder
        self.encoder = SequenceEncoder(query_sequence, embedding_dim)
        
        # Initialize structure analyzer
        self.structure_analyzer = StructureAnalyzer()
        
        # Load and prepare MSA
        self._load_msa()
        
        # Episode state
        self.reset()
        
    def _load_msa(self):
        """Load and prepare MSA sequences."""
        logger.info(f"Loading MSA from {self.msa_file}")
        
        # Read MSA
        all_sequences = read_a3m_to_dict(self.msa_file)
        logger.info(f"Loaded {len(all_sequences)} sequences")
        
        # Filter sequences (coverage, etc.)
        from af_claseq.utils.sequence_processing import filter_a3m_by_coverage
        filtered_sequences = filter_a3m_by_coverage(all_sequences, coverage_threshold=0.7)
        logger.info(f"After coverage filtering: {len(filtered_sequences)} sequences")
        
        # Limit pool size for computational efficiency
        if len(filtered_sequences) > self.max_pool_size:
            headers = list(filtered_sequences.keys())
            random.shuffle(headers)
            selected_headers = headers[:self.max_pool_size]
            filtered_sequences = {h: filtered_sequences[h] for h in selected_headers}
            logger.info(f"Limited to {len(filtered_sequences)} sequences")
        
        self.all_sequences = filtered_sequences
        self.sequence_headers = list(filtered_sequences.keys())
        
        # Pre-compute sequence features
        self.sequence_features = {}
        for header, sequence in self.all_sequences.items():
            self.sequence_features[header] = self.encoder.encode_sequence(sequence, header)
        
        logger.info("Pre-computed sequence features")
        
    def reset(self) -> np.ndarray:
        """
        Reset the environment for a new episode.
        
        Returns:
            Initial state observation
        """
        # Reset episode state
        self.current_subset = {}
        self.remaining_headers = self.sequence_headers.copy()
        self.step_count = 0
        self.episode_rewards = []
        
        # Create unique episode directory
        self.episode_dir = self.temp_dir / f"episode_{random.randint(0, 999999)}"
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """
        Get current state representation.
        
        Returns:
            State vector combining subset features and pool statistics
        """
        # Current subset summary
        if self.current_subset:
            subset_summary = self.encoder.encode_subset_summary(self.current_subset)
        else:
            subset_summary = np.zeros(self.embedding_dim, dtype=np.float32)
        
        # Remaining pool statistics
        if self.remaining_headers:
            remaining_features = np.array([
                self.sequence_features[h] for h in self.remaining_headers
            ])
            pool_stats = np.concatenate([
                np.mean(remaining_features, axis=0),
                np.std(remaining_features, axis=0),
                [len(self.remaining_headers) / self.max_pool_size]  # Normalized count
            ])
        else:
            pool_stats = np.zeros(self.encoder.get_feature_dim() * 2 + 1, dtype=np.float32)
        
        # Progress indicator
        progress = self.step_count / self.target_subset_size
        
        # Combine all state components
        state = np.concatenate([
            subset_summary,
            pool_stats,
            [progress]
        ])
        
        return state.astype(np.float32)
    
    def get_state_dim(self) -> int:
        """Get the dimension of the state space."""
        return self.embedding_dim + self.encoder.get_feature_dim() * 2 + 1 + 1
    
    def get_action_dim(self) -> int:
        """Get the dimension of the action space (number of remaining sequences)."""
        return len(self.remaining_headers)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Index of sequence to add (in remaining_headers)
            
        Returns:
            next_state, reward, done, info
        """
        if action >= len(self.remaining_headers):
            raise ValueError(f"Invalid action {action}, only {len(self.remaining_headers)} sequences available")
        
        # Add selected sequence to subset
        selected_header = self.remaining_headers[action]
        selected_sequence = self.all_sequences[selected_header]
        self.current_subset[selected_header] = selected_sequence
        self.remaining_headers.pop(action)
        self.step_count += 1
        
        # Check if episode is done
        done = (self.step_count >= self.target_subset_size)
        
        # Calculate reward
        if done:
            reward = self._calculate_final_reward()
        else:
            reward = self._calculate_intermediate_reward()
        
        self.episode_rewards.append(reward)
        
        # Get next state
        next_state = self._get_state()
        
        # Info dictionary
        info = {
            'step_count': self.step_count,
            'subset_size': len(self.current_subset),
            'selected_header': selected_header,
            'episode_reward': sum(self.episode_rewards),
            'done': done
        }
        
        return next_state, reward, done, info
    
    def _calculate_intermediate_reward(self) -> float:
        """Calculate intermediate reward for non-terminal states."""
        # Small diversity bonus for adding diverse sequences
        if len(self.current_subset) >= 2:
            sequences = list(self.current_subset.values())
            last_seq = sequences[-1]
            
            # Calculate average identity with existing sequences
            identities = []
            for seq in sequences[:-1]:
                identity = self.encoder._calculate_sequence_identity(last_seq, seq)
                identities.append(identity)
            
            avg_identity = np.mean(identities) if identities else 0.0
            diversity_reward = self.diversity_bonus * (1.0 - avg_identity)
            
            return diversity_reward
        
        return 0.0
    
    def _calculate_final_reward(self) -> float:
        """
        Calculate final reward by running AF2 prediction and evaluating structure.
        
        Returns:
            Reward value based on structural quality
        """
        try:
            # Create A3M file for current subset
            subset_dir = self.episode_dir / "prediction"
            subset_dir.mkdir(exist_ok=True)
            
            a3m_file = subset_dir / "subset.a3m"
            write_a3m(self.current_subset, str(a3m_file), self.reward_config.rare_state_pdb)
            
            # Submit AF2 prediction job
            job_id = f"rl_eval_{random.randint(0, 999999)}"
            logger.info(f"Submitting AF2 job for subset evaluation: {job_id}")
            
            # Process folder (this will wait for completion)
            self.slurm_submitter.process_folder(str(subset_dir), job_id)
            
            # Find generated PDB file
            pdb_files = list(subset_dir.glob("*.pdb"))
            if not pdb_files:
                logger.warning("No PDB file generated, returning zero reward")
                return 0.0
            
            pdb_file = pdb_files[0]  # Take first PDB file
            logger.info(f"Evaluating structure: {pdb_file}")
            
            # Calculate structural metrics
            reward = self._evaluate_structure(str(pdb_file))
            
            # Clean up
            shutil.rmtree(subset_dir, ignore_errors=True)
            
            return reward
            
        except Exception as e:
            logger.error(f"Error calculating final reward: {e}")
            return 0.0
    
    def _evaluate_structure(self, pdb_file: str) -> float:
        """
        Evaluate predicted structure and return reward.
        
        Args:
            pdb_file: Path to predicted PDB file
            
        Returns:
            Reward value
        """
        try:
            # Calculate pLDDT
            full_indices = list(range(1, len(self.query_sequence) + 1))
            plddt = self.structure_analyzer.plddt_process(pdb_file, full_indices)
            
            if plddt is None or plddt < self.reward_config.plddt_threshold:
                logger.warning(f"Low pLDDT ({plddt}), returning zero reward")
                return 0.0
            
            # Calculate RMSD to rare state
            rmsd = self.structure_analyzer.calculate_ca_rmsd(
                reference_pdb=self.reward_config.rare_state_pdb,
                target_pdb=pdb_file,
                superposition_indices=full_indices,
                rmsd_indices=full_indices
            )
            
            if np.isnan(rmsd) or rmsd > self.reward_config.max_rmsd:
                logger.warning(f"Invalid or high RMSD ({rmsd}), returning zero reward")
                return 0.0
            
            # Calculate TM-score to rare state
            tm_score = self.structure_analyzer.calculate_tm_score(
                target_pdb=pdb_file,
                reference_pdb=self.reward_config.rare_state_pdb
            )
            
            # Normalize metrics for reward calculation
            plddt_reward = (plddt - self.reward_config.plddt_threshold) / (100.0 - self.reward_config.plddt_threshold)
            rmsd_reward = max(0.0, (self.reward_config.max_rmsd - rmsd) / self.reward_config.max_rmsd)
            tm_score_reward = tm_score  # TM-score is already normalized [0,1]
            
            # Combine rewards
            total_reward = (
                self.reward_config.plddt_weight * plddt_reward +
                self.reward_config.rmsd_weight * rmsd_reward +
                self.reward_config.tm_score_weight * tm_score_reward
            )
            
            # Add bonus for exceptional performance
            if plddt > 80 and rmsd < 2.0 and tm_score > 0.7:
                total_reward += 0.5  # Exceptional bonus
            
            logger.info(f"Structure evaluation - pLDDT: {plddt:.2f}, RMSD: {rmsd:.2f}, "
                       f"TM-score: {tm_score:.3f}, Total reward: {total_reward:.3f}")
            
            return max(0.0, total_reward)  # Ensure non-negative reward
            
        except Exception as e:
            logger.error(f"Error evaluating structure: {e}")
            return 0.0
    
    def get_available_actions(self) -> List[int]:
        """Get list of available action indices."""
        return list(range(len(self.remaining_headers)))
    
    def get_action_mask(self) -> np.ndarray:
        """Get binary mask for valid actions."""
        mask = np.zeros(self.max_pool_size, dtype=np.float32)
        mask[:len(self.remaining_headers)] = 1.0
        return mask
    
    def render(self, mode: str = "human"):
        """Render the current state (for debugging)."""
        print(f"Episode Step: {self.step_count}/{self.target_subset_size}")
        print(f"Current subset size: {len(self.current_subset)}")
        print(f"Remaining sequences: {len(self.remaining_headers)}")
        if self.episode_rewards:
            print(f"Last reward: {self.episode_rewards[-1]:.3f}")
            print(f"Episode total: {sum(self.episode_rewards):.3f}")
    
    def close(self):
        """Clean up environment resources."""
        if self.episode_dir.exists():
            shutil.rmtree(self.episode_dir, ignore_errors=True)


class ParallelMSAEnvironment:
    """
    Wrapper for running multiple MSA environments in parallel.
    Useful for collecting diverse training experiences.
    """
    
    def __init__(
        self,
        msa_files: List[str],
        query_sequences: List[str],
        reward_configs: List[RewardConfig],
        slurm_submitter: SlurmJobSubmitter,
        **env_kwargs
    ):
        """
        Initialize parallel environments.
        
        Args:
            msa_files: List of MSA file paths
            query_sequences: List of query sequences
            reward_configs: List of reward configurations
            slurm_submitter: SLURM job submitter instance
            **env_kwargs: Additional environment arguments
        """
        assert len(msa_files) == len(query_sequences) == len(reward_configs)
        
        self.environments = []
        for msa_file, query_seq, reward_config in zip(msa_files, query_sequences, reward_configs):
            env = MSASubsetEnvironment(
                msa_file=msa_file,
                query_sequence=query_seq,
                reward_config=reward_config,
                slurm_submitter=slurm_submitter,
                **env_kwargs
            )
            self.environments.append(env)
    
    def reset(self) -> List[np.ndarray]:
        """Reset all environments."""
        return [env.reset() for env in self.environments]
    
    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], List[bool], List[Dict]]:
        """Step all environments."""
        results = [env.step(action) for env, action in zip(self.environments, actions)]
        states, rewards, dones, infos = zip(*results)
        return list(states), list(rewards), list(dones), list(infos)
    
    def get_state_dim(self) -> int:
        """Get state dimension (assumed same for all environments)."""
        return self.environments[0].get_state_dim()
    
    def get_max_action_dim(self) -> int:
        """Get maximum action dimension across environments."""
        return max(env.get_action_dim() for env in self.environments)
    
    def close(self):
        """Close all environments."""
        for env in self.environments:
            env.close()


def create_reward_config(rare_state_pdb: str, **kwargs) -> RewardConfig:
    """Helper function to create reward configuration."""
    return RewardConfig(rare_state_pdb=rare_state_pdb, **kwargs)