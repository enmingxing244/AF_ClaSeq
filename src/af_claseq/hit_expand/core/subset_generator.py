#!/usr/bin/env python3
"""
High-quality MSA subset generator for creating random subsets with query sequence.
Based on generate_abl1_subsets.py but with enterprise-grade code quality.
"""

import os
import random
import logging
from pathlib import Path
from typing import List, Tuple, Union, Optional, Dict
from dataclasses import dataclass

from af_claseq.hit_expand.core.sequence_io import parse_sequence_file, write_a3m_file

logger = logging.getLogger(__name__)


# Import from central config location
from ..config.settings import SubsetConfig
    

class SubsetGenerationError(Exception):
    """Raised when subset generation fails."""
    pass


class MSASubsetGenerator:
    """Generates random subsets from MSA for structure prediction experiments."""
    
    def __init__(self, config: SubsetConfig):
        """
        Initialize subset generator.
        
        Args:
            config: Subset generation configuration
        """
        self.config = config
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.config.num_subsets <= 0:
            raise ValueError("Number of subsets must be positive")
        
        if self.config.num_random_sequences <= 0:
            raise ValueError("Number of random sequences must be positive")
        
        if not self.config.output_prefix:
            raise ValueError("Output prefix cannot be empty")
    
    def generate_subsets(self, input_file: Union[str, Path], 
                        output_dir: Union[str, Path]) -> List[Path]:
        """
        Generate random subsets from input MSA file.
        
        Args:
            input_file: Path to input A3M file
            output_dir: Directory to save subset files
            
        Returns:
            List of paths to generated subset files
            
        Raises:
            SubsetGenerationError: If generation fails
            FileNotFoundError: If input file doesn't exist
        """
        input_file = Path(input_file)
        output_dir = Path(output_dir)
        
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        logger.info(f"Generating subsets from {input_file}")
        
        try:
            # Parse input file (supports both FASTA and A3M)
            sequences, headers = parse_sequence_file(input_file)
            
            if len(sequences) < self.config.num_random_sequences + 1:
                raise SubsetGenerationError(
                    f"Not enough sequences in {input_file}. "
                    f"Need at least {self.config.num_random_sequences + 1}, "
                    f"found {len(sequences)}"
                )
            
            # Extract query and other sequences
            query_sequence, query_header, other_sequences, other_headers = (
                self._extract_query_and_others(sequences, headers)
            )
            
            logger.info(f"Query sequence: {query_header}")
            logger.info(f"Available sequences for sampling: {len(other_sequences)}")
            
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Set random seed for reproducibility
            random.seed(self.config.random_seed)
            
            # Generate subsets
            subset_files = self._generate_subset_files(
                query_sequence, query_header, other_sequences, other_headers, output_dir
            )
            
            logger.info(f"Generated {len(subset_files)} subset files in {output_dir}")
            return subset_files
            
        except Exception as e:
            logger.error(f"Subset generation failed: {e}")
            raise SubsetGenerationError(f"Failed to generate subsets: {e}")
    
    def _extract_query_and_others(self, sequences: List[str], 
                                 headers: List[str]) -> Tuple[str, str, List[str], List[str]]:
        """
        Extract query sequence (first) and other sequences.
        
        Args:
            sequences: List of all sequences
            headers: List of all headers
            
        Returns:
            Tuple of (query_seq, query_header, other_seqs, other_headers)
        """
        if not sequences:
            raise SubsetGenerationError("No sequences provided")
        
        # Query is always the first sequence
        query_sequence = sequences[0]
        query_header = headers[0]
        
        # Other sequences for random sampling
        other_sequences = sequences[1:]
        other_headers = headers[1:]
        
        return query_sequence, query_header, other_sequences, other_headers
    
    def _generate_subset_files(self, query_sequence: str, query_header: str,
                              other_sequences: List[str], other_headers: List[str],
                              output_dir: Path) -> List[Path]:
        """Generate individual subset files."""
        subset_files = []
        
        for i in range(self.config.num_subsets):
            # Generate subset filename
            subset_filename = f"{self.config.output_prefix}_{i+1:04d}.a3m"
            subset_path = output_dir / subset_filename
            
            # Randomly select sequences
            try:
                selected_sequences, selected_headers = self._select_random_sequences(
                    other_sequences, other_headers
                )
            except ValueError as e:
                logger.error(f"Failed to select sequences for subset {i+1}: {e}")
                continue
            
            
            subset_sequences = [query_sequence] + selected_sequences
            subset_headers = [query_header] + selected_headers
            
            
            # Write subset file
            try:
                write_a3m_file(subset_sequences, subset_headers, subset_path)
                subset_files.append(subset_path)
            except Exception as e:
                logger.error(f"Failed to write subset {i+1}: {e}")
                continue
            
            # Progress logging
            if (i + 1) % 100 == 0:
                logger.info(f"Generated {i + 1} subsets...")
        
        return subset_files
    
    def _select_random_sequences(self, sequences: List[str], 
                                headers: List[str]) -> Tuple[List[str], List[str]]:
        """
        Randomly select sequences from the pool.
        
        Args:
            sequences: Pool of sequences to select from
            headers: Corresponding headers
            
        Returns:
            Tuple of (selected_sequences, selected_headers)
            
        Raises:
            ValueError: If not enough sequences available
        """
        if len(sequences) < self.config.num_random_sequences:
            raise ValueError(
                f"Not enough sequences to select {self.config.num_random_sequences} "
                f"from {len(sequences)} available"
            )
        
        # Randomly select indices
        selected_indices = random.sample(
            range(len(sequences)), 
            self.config.num_random_sequences
        )
        
        # Extract selected sequences and headers
        selected_sequences = [sequences[i] for i in selected_indices]
        selected_headers = [headers[i] for i in selected_indices]
        
        return selected_sequences, selected_headers


class BatchOrganizer:
    """Organizes subset files into batches for parallel processing."""
    
    def __init__(self, num_batches: int = 50):
        """
        Initialize batch organizer.
        
        Args:
            num_batches: Number of batches to create
        """
        if num_batches <= 0:
            raise ValueError("Number of batches must be positive")
        
        self.num_batches = num_batches
    
    def organize_batches(self, subset_dir: Union[str, Path], 
                        batch_dir: Union[str, Path]) -> Dict[str, List[Path]]:
        """
        Organize subset files into batches.
        
        Args:
            subset_dir: Directory containing subset files
            batch_dir: Directory to create batch subdirectories
            
        Returns:
            Dictionary mapping batch names to lists of file paths
            
        Raises:
            FileNotFoundError: If subset directory doesn't exist
        """
        subset_dir = Path(subset_dir)
        batch_dir = Path(batch_dir)
        
        if not subset_dir.exists():
            raise FileNotFoundError(f"Subset directory not found: {subset_dir}")
        
        # Find all A3M files
        subset_files = sorted(list(subset_dir.glob("*.a3m")))
        total_files = len(subset_files)
        
        if total_files == 0:
            logger.warning(f"No A3M files found in {subset_dir}")
            return {}
        
        logger.info(f"Organizing {total_files} files into {self.num_batches} batches")
        
        # Calculate files per batch
        files_per_batch = (total_files + self.num_batches - 1) // self.num_batches
        
        # Create batch directory
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        # Organize files into batches
        batch_files = {}
        
        for batch_idx in range(self.num_batches):
            batch_name = f"batch_{batch_idx+1:02d}"
            batch_path = batch_dir / batch_name
            batch_path.mkdir(parents=True, exist_ok=True)
            
            # Calculate file range for this batch
            start_idx = batch_idx * files_per_batch
            end_idx = min(start_idx + files_per_batch, total_files)
            
            if start_idx >= total_files:
                break
            
            batch_file_list = subset_files[start_idx:end_idx]
            batch_files[batch_name] = []
            
            logger.info(f"Creating {batch_name} with {len(batch_file_list)} files")
            
            # Copy files to batch directory
            for src_file in batch_file_list:
                dst_file = batch_path / src_file.name
                
                try:
                    # Copy file content
                    dst_file.write_text(src_file.read_text())
                    batch_files[batch_name].append(dst_file)
                except Exception as e:
                    logger.error(f"Failed to copy {src_file} to {dst_file}: {e}")
        
        # Log batch summary
        logger.info("Batch organization summary:")
        for batch_name, files in batch_files.items():
            logger.info(f"  {batch_name}: {len(files)} files")
        
        return batch_files


def generate_msa_subsets(input_file: Union[str, Path], 
                        output_dir: Union[str, Path],
                        num_subsets: int = 2000,
                        num_random_sequences: int = 8,
                        random_seed: int = 42) -> List[Path]:
    """
    Convenience function to generate MSA subsets.
    
    Args:
        input_file: Path to input A3M file
        output_dir: Directory to save subset files
        num_subsets: Number of subsets to generate
        num_random_sequences: Number of random sequences per subset
        random_seed: Random seed for reproducibility
        
    Returns:
        List of paths to generated subset files
    """
    config = SubsetConfig(
        num_subsets=num_subsets,
        num_random_sequences=num_random_sequences,
        random_seed=random_seed
    )
    
    generator = MSASubsetGenerator(config)
    return generator.generate_subsets(input_file, output_dir)


def organize_subsets_into_batches(subset_dir: Union[str, Path], 
                                 batch_dir: Union[str, Path],
                                 num_batches: int = 50) -> Dict[str, List[Path]]:
    """
    Convenience function to organize subsets into batches.
    
    Args:
        subset_dir: Directory containing subset files
        batch_dir: Directory to create batch subdirectories
        num_batches: Number of batches to create
        
    Returns:
        Dictionary mapping batch names to lists of file paths
    """
    organizer = BatchOrganizer(num_batches)
    return organizer.organize_batches(subset_dir, batch_dir)