"""
M-fold sampling module for AF-ClaSeq pipeline.

This module provides classes to:
1. Run M-fold sampling of sequences
2. Plot distributions of metrics in 1D and 2D
3. Analyze sampling results
"""

import argparse
import logging
import os
import random
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hex2color
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from af_claseq.utils.sequence_processing import read_a3m_to_dict, write_a3m, filter_a3m_by_coverage
from af_claseq.utils.slurm_utils import SlurmJobSubmitter
from af_claseq.utils.logging_utils import get_logger


class MFoldSampler:
    """
    Main class for running M-fold sampling.
    
    M-fold sampling creates multiple groupings of sequences to 
    explore the protein's conformational landscape more effectively.
    """
    
    def __init__(
        self,
        input_a3m: str,
        default_pdb: str,
        m_fold_sampling_base_dir: str,
        group_size: int = 10,
        coverage_threshold: float = 0.0,
        random_select_num_seqs: Optional[int] = None,
        slurm_submitter: Optional[SlurmJobSubmitter] = None,
        random_seed: int = 42,
        max_workers: int = 64,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the M-fold sampler.
        
        Args:
            input_a3m: Path to input A3M file
            default_pdb: Path to reference PDB file
            m_fold_sampling_base_dir: Base directory for output
            group_size: Size of each sequence group
            coverage_threshold: Minimum required sequence coverage
            random_select_num_seqs: Number of sequences to randomly select
            slurm_submitter: Optional SlurmJobSubmitter instance
            random_seed: Random seed for reproducibility
            max_workers: Maximum number of concurrent workers
            logger: Optional logger instance
        """
        self.input_a3m = input_a3m
        self.default_pdb = default_pdb
        self.m_fold_sampling_base_dir = m_fold_sampling_base_dir
        self.group_size = group_size
        self.coverage_threshold = coverage_threshold
        self.random_select_num_seqs = random_select_num_seqs
        self.slurm_submitter = slurm_submitter
        self.max_workers = max_workers
        
        # Set up directories
        self.init_dir = os.path.join(self.m_fold_sampling_base_dir, "02_init_random_split")
        self.sampling_base_dir = os.path.join(self.m_fold_sampling_base_dir, "02_sampling")
        os.makedirs(self.init_dir, exist_ok=True)
        os.makedirs(self.sampling_base_dir, exist_ok=True)
        
        # Set up logger
        self.logger = logger or get_logger("m_fold_sampling")
        
        # Set random seed
        random.seed(random_seed)
    
    def run(self) -> bool:
        """
        Run the M-fold sampling process.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Apply coverage filter if needed
            filtered_a3m = self._apply_coverage_filter()
            
            # Run initial random split
            num_groups = self._run_initial_split(filtered_a3m)
            
            # Create sampling splits
            self._create_sampling_splits(num_groups)
            
            # Submit SLURM jobs if available
            if self.slurm_submitter:
                self._submit_jobs()
            else:
                self.logger.info("No SLURM submitter provided. Skipping job submission.")
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error in M-fold sampling: {str(e)}", exc_info=True)
            return False
    
    def _apply_coverage_filter(self) -> str:
        """
        Apply coverage filtering to input sequences if needed.
        
        Returns:
            Path to filtered or original A3M file
        """
        if self.coverage_threshold <= 0:
            self.logger.info("Coverage threshold is not set. Using sequences as they are.")
            return self.input_a3m
        
        self.logger.info(f"Applying coverage filter with threshold {self.coverage_threshold}")
        sequences = read_a3m_to_dict(self.input_a3m)
        filtered_sequences = filter_a3m_by_coverage(sequences, self.coverage_threshold)
        
        self.logger.info(f"Filtered sequences from {len(sequences)} to {len(filtered_sequences)}")
        
        if not filtered_sequences:
            raise ValueError("No sequences remained after filtering")
        
        filtered_a3m_path = os.path.join(self.m_fold_sampling_base_dir, "filtered_sequences.a3m")
        write_a3m(filtered_sequences, filtered_a3m_path, reference_pdb=self.default_pdb)
        
        return filtered_a3m_path
    
    def _run_initial_split(self, input_a3m: str) -> int:
        """
        Perform initial random split of sequences.
        
        Args:
            input_a3m: Path to input A3M file
            
        Returns:
            Number of groups created
        """
        self.logger.info("Performing initial random split")
        num_groups = self._initial_random_split(
            input_a3m=input_a3m,
            output_dir=self.init_dir,
            reference_pdb=self.default_pdb,
            group_size=self.group_size,
            random_select_num_seqs=self.random_select_num_seqs
        )
        self.logger.info(f"Created {num_groups} initial groups")
        return num_groups
    
    def _create_sampling_splits(self, num_groups: int) -> None:
        """
        Create sampling splits by leaving one group out at a time.
        
        Args:
            num_groups: Number of groups from initial split
        """
        self.logger.info("Creating sampling splits")
        self._create_sampling_splits_impl(
            init_dir=self.init_dir,
            output_base_dir=self.sampling_base_dir,
            reference_pdb=self.default_pdb,
            group_size=self.group_size
        )
        self.logger.info(f"Created {num_groups-1} sampling splits")
    
    def _submit_jobs(self) -> None:
        """Submit SLURM jobs for each sampling directory"""
        sampling_dirs = [d for d in os.listdir(self.sampling_base_dir) 
                        if d.startswith('sampling_')]
        sampling_paths = [os.path.join(self.sampling_base_dir, d) for d in sampling_dirs]
        job_ids = [f"sample_{i+1}" for i in range(len(sampling_dirs))]
        
        self.logger.info(f"Submitting {len(sampling_dirs)} SLURM jobs")
        self.slurm_submitter.process_folders_concurrently(
            folders=sampling_paths,
            job_ids=job_ids,
            max_workers=self.max_workers
        )
        self.logger.info("Submitted all SLURM jobs")
    
    @staticmethod
    def _split_into_groups(
        sequences: Dict[str, str], 
        group_size: int
    ) -> List[Dict[str, str]]:
        """
        Split sequences into groups of specified size.
        
        Args:
            sequences: Dictionary mapping sequence headers to sequences
            group_size: Size of each group
            
        Returns:
            List of dictionaries, where each dictionary contains a group of sequences
        """
        groups = []
        headers = list(sequences.keys())
        
        for i in range(0, len(headers), group_size):
            group = {}
            group_headers = headers[i:i + group_size]
            
            if len(group_headers) < group_size and groups:
                groups[-1].update({h: sequences[h] for h in group_headers})
            else:
                group = {h: sequences[h] for h in group_headers}
                groups.append(group)
                
        return groups
    
    @staticmethod
    def _initial_random_split(
        input_a3m: str, 
        output_dir: str,
        reference_pdb: str,
        group_size: int,
        random_select_num_seqs: Optional[int] = None
    ) -> int:
        """
        Perform initial random split of sequences.
        
        Args:
            input_a3m: Path to input A3M file
            output_dir: Directory to write output files
            reference_pdb: Path to reference PDB file
            group_size: Size of each sequence group
            random_select_num_seqs: Number of sequences to randomly select
            
        Returns:
            Number of groups created
        """
        os.makedirs(output_dir, exist_ok=True)
        sequences = read_a3m_to_dict(input_a3m)
        
        headers = list(sequences.keys())
        random.shuffle(headers)
        
        if random_select_num_seqs is not None:
            headers = headers[:random_select_num_seqs]
        
        shuffled_sequences = {h: sequences[h] for h in headers}
        
        groups = MFoldSampler._split_into_groups(shuffled_sequences, group_size)
        
        for i, group in enumerate(groups, 1):
            output_file = os.path.join(output_dir, f'group_{i}.a3m')
            write_a3m(group, output_file, reference_pdb=reference_pdb)
        
        return len(groups)
    
    @staticmethod
    def _create_sampling_splits_impl(
        init_dir: str, 
        output_base_dir: str, 
        reference_pdb: str,
        group_size: int
    ) -> None:
        """
        Create sampling splits by exhaustively leaving one group out at a time.
        
        Args:
            init_dir: Directory containing initial groups
            output_base_dir: Base directory for output sampling splits
            reference_pdb: Path to reference PDB file
            group_size: Size of each sequence group
        """
        groups = [f for f in os.listdir(init_dir) if f.endswith('.a3m')]
        num_groups = len(groups)
        
        for i in range(num_groups):
            sample_dir = os.path.join(output_base_dir, f'sampling_{i+1}')
            os.makedirs(sample_dir, exist_ok=True)
            
            all_sequences = {}
            for j, group in enumerate(groups):
                if j != i:
                    group_path = os.path.join(init_dir, group)
                    group_sequences = read_a3m_to_dict(group_path)
                    all_sequences.update(group_sequences)
            
            headers = list(all_sequences.keys())
            random.shuffle(headers)
            shuffled_sequences = {h: all_sequences[h] for h in headers}
            new_groups = MFoldSampler._split_into_groups(shuffled_sequences, group_size)
            
            for j, group in enumerate(new_groups, 1):
                output_file = os.path.join(sample_dir, f'group_{j}.a3m')
                write_a3m(group, output_file, reference_pdb=reference_pdb)
