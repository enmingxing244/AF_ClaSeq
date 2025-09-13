"""
Shuffle management for the divide-and-conquer workflow.
Handles random shuffling and grouping of sequences within clades.
"""

import os
import random
import math
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import OrderedDict
import logging

from .utils import (
    read_a3m, write_a3m, create_directory, count_sequences_in_a3m, 
    validate_file_exists, get_file_stem, WorkflowError
)


class ShuffleManager:
    """
    Manages random shuffling and grouping of sequences within clades.
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger, 
                 query_header: str, query_sequence: str):
        """
        Initialize ShuffleManager.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
            query_header: Processed query sequence header
            query_sequence: Query sequence content
        """
        self.config = config
        self.logger = logger
        self.shuffle_config = config['shuffling']
        self.query_header = query_header
        self.query_sequence = query_sequence
        
        # Extract shuffling parameters
        self.num_shuffles = self.shuffle_config.get('num_shuffles', 10)
        self.group_size = self.shuffle_config.get('group_size', 50)
        self.random_seed = self.shuffle_config.get('random_seed', None)
        
        if self.random_seed:
            random.seed(self.random_seed)
            self.logger.info(f"Random seed set to: {self.random_seed}")
        
        self.logger.info(f"Shuffle parameters:")
        self.logger.info(f"  Number of shuffles: {self.num_shuffles}")
        self.logger.info(f"  Group size: {self.group_size}")
        self.logger.info(f"  Query header: {self.query_header}")
    
    def shuffle_and_split_clade(self, clade_dir: str) -> List[str]:
        """
        Shuffle and split sequences in a single clade into groups.
        
        Args:
            clade_dir: Path to clade directory
            
        Returns:
            List of shuffle directory paths
        """
        clade_name = os.path.basename(clade_dir)
        self.logger.info(f"Processing clade: {clade_name}")
        
        # Find the clade A3M file
        clade_a3m_file = os.path.join(clade_dir, f"{clade_name}.a3m")
        validate_file_exists(clade_a3m_file, f"Clade A3M file for {clade_name}")
        
        # Read sequences from clade
        sequences = read_a3m(clade_a3m_file)
        
        if not sequences:
            self.logger.warning(f"No sequences found in {clade_a3m_file}")
            return []
        
        # Remove query sequence if it exists in the clade (we'll add it back later)
        clade_sequences = OrderedDict()
        query_found_in_clade = False
        
        for header, sequence in sequences.items():
            if sequence == self.query_sequence:
                query_found_in_clade = True
                self.logger.debug(f"Query sequence found in clade {clade_name}")
            else:
                clade_sequences[header] = sequence
        
        sequence_count = len(clade_sequences)
        self.logger.info(f"  Sequences in clade (excluding query): {sequence_count}")
        
        if sequence_count == 0:
            self.logger.warning(f"No non-query sequences in clade {clade_name}")
            return []
        
        # Calculate number of groups
        num_groups = math.ceil(sequence_count / self.group_size)
        actual_group_size = math.ceil(sequence_count / num_groups)
        
        self.logger.info(f"  Will create {num_groups} groups of ~{actual_group_size} sequences each")
        
        # Create shuffle directories
        shuffle_dirs = []
        
        for shuffle_num in range(1, self.num_shuffles + 1):
            shuffle_dir = os.path.join(clade_dir, f"shuffle_{shuffle_num:02d}")
            create_directory(shuffle_dir)
            shuffle_dirs.append(shuffle_dir)
            
            # Shuffle sequences
            seq_items = list(clade_sequences.items())
            random.shuffle(seq_items)
            
            # Split into groups
            for group_num in range(1, num_groups + 1):
                start_idx = (group_num - 1) * actual_group_size
                end_idx = min(start_idx + actual_group_size, len(seq_items))
                
                if start_idx >= len(seq_items):
                    break
                
                group_sequences = seq_items[start_idx:end_idx]
                
                # Create group file with query sequence first
                group_file = os.path.join(shuffle_dir, f"group_{group_num:03d}.a3m")
                
                # Build final sequences with query first
                final_sequences = OrderedDict()
                final_sequences[self.query_header] = self.query_sequence
                
                for header, sequence in group_sequences:
                    final_sequences[header] = sequence
                
                write_a3m(final_sequences, group_file)
                
                self.logger.debug(f"    Created {group_file} with {len(final_sequences)} sequences")
            
            self.logger.info(f"  Shuffle {shuffle_num}: Created {num_groups} groups in {shuffle_dir}")
        
        return shuffle_dirs
    
    def process_all_clades(self, clade_dirs: List[str]) -> List[str]:
        """
        Process all clades for shuffling and grouping.
        
        Args:
            clade_dirs: List of clade directory paths
            
        Returns:
            List of all shuffle directory paths
        """
        self.logger.info("=" * 50)
        self.logger.info("SHUFFLE PROCESSING STARTED")
        self.logger.info("=" * 50)
        
        all_shuffle_dirs = []
        
        for i, clade_dir in enumerate(clade_dirs, 1):
            self.logger.info(f"Processing clade {i}/{len(clade_dirs)}: {os.path.basename(clade_dir)}")
            
            try:
                shuffle_dirs = self.shuffle_and_split_clade(clade_dir)
                all_shuffle_dirs.extend(shuffle_dirs)
                
            except Exception as e:
                self.logger.error(f"Error processing clade {clade_dir}: {e}")
                continue
        
        self.logger.info("=" * 50)
        self.logger.info("SHUFFLE PROCESSING COMPLETED")
        self.logger.info(f"Total shuffle directories created: {len(all_shuffle_dirs)}")
        self.logger.info("=" * 50)
        
        # Print summary
        self._print_shuffle_summary(clade_dirs, all_shuffle_dirs)
        
        return all_shuffle_dirs
    
    def _print_shuffle_summary(self, clade_dirs: List[str], shuffle_dirs: List[str]) -> None:
        """
        Print summary of shuffle processing.
        
        Args:
            clade_dirs: List of clade directories
            shuffle_dirs: List of shuffle directories
        """
        self.logger.info("Clade Shuffle and Split Summary")
        self.logger.info("=" * 50)
        self.logger.info(f"Clades directory: {os.path.dirname(clade_dirs[0]) if clade_dirs else 'N/A'}")
        self.logger.info(f"Number of shuffles: {self.num_shuffles}")
        self.logger.info(f"Group size: {self.group_size}")
        
        if self.random_seed is not None:
            self.logger.info(f"Random seed: {self.random_seed}")
        
        self.logger.info("\nProcessing complete. Check each clade subdirectory for results.")
        self.logger.info("\nDirectory structure:")
        self.logger.info("  clades/")
        
        for clade_dir in clade_dirs:
            clade_name = os.path.basename(clade_dir)
            self.logger.info(f"    {clade_name}/")
            
            # Count groups in first shuffle to show structure
            first_shuffle = os.path.join(clade_dir, "shuffle_01")
            if os.path.exists(first_shuffle):
                group_files = [f for f in os.listdir(first_shuffle) if f.startswith('group_') and f.endswith('.a3m')]
                group_count = len(group_files)
                
                for shuffle_num in range(1, min(4, self.num_shuffles + 1)):  # Show first 3 shuffles
                    self.logger.info(f"        shuffle_{shuffle_num:02d}/")
                    if shuffle_num == 1:
                        for group_num in range(1, min(4, group_count + 1)):  # Show first 3 groups
                            self.logger.info(f"          group_{group_num:03d}.a3m")
                        if group_count > 3:
                            self.logger.info(f"          ... ({group_count - 3} more groups)")
                    else:
                        self.logger.info(f"          group_001.a3m")
                        self.logger.info(f"          group_002.a3m")
                        self.logger.info("          ...")
                
                if self.num_shuffles > 3:
                    self.logger.info(f"        ... (shuffle_04 to shuffle_{self.num_shuffles:02d})")
    
    def validate_shuffle_results(self, shuffle_dirs: List[str]) -> bool:
        """
        Validate that shuffle results are correct.
        
        Args:
            shuffle_dirs: List of shuffle directories to validate
            
        Returns:
            True if validation passes
        """
        self.logger.info("Validating shuffle results...")
        
        validation_errors = []
        total_groups = 0
        
        for shuffle_dir in shuffle_dirs:
            if not os.path.exists(shuffle_dir):
                validation_errors.append(f"Shuffle directory not found: {shuffle_dir}")
                continue
            
            # Find group files
            group_files = [f for f in os.listdir(shuffle_dir) 
                          if f.startswith('group_') and f.endswith('.a3m')]
            
            if not group_files:
                validation_errors.append(f"No group files found in {shuffle_dir}")
                continue
            
            total_groups += len(group_files)
            
            # Validate each group file
            for group_file in group_files:
                group_path = os.path.join(shuffle_dir, group_file)
                
                try:
                    sequences = read_a3m(group_path)
                    
                    if not sequences:
                        validation_errors.append(f"Empty group file: {group_path}")
                        continue
                    
                    # Check if query sequence is first
                    first_header = next(iter(sequences.keys()))
                    first_sequence = next(iter(sequences.values()))
                    
                    if first_sequence != self.query_sequence:
                        validation_errors.append(
                            f"Query sequence not first in {group_path}"
                        )
                    
                except Exception as e:
                    validation_errors.append(f"Error reading {group_path}: {e}")
        
        # Report validation results
        if validation_errors:
            self.logger.error(f"Validation failed with {len(validation_errors)} errors:")
            for error in validation_errors[:10]:  # Show first 10 errors
                self.logger.error(f"  - {error}")
            if len(validation_errors) > 10:
                self.logger.error(f"  ... and {len(validation_errors) - 10} more errors")
            return False
        else:
            self.logger.info(f"Validation passed: {len(shuffle_dirs)} shuffle directories, "
                           f"{total_groups} total group files")
            return True