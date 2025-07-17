#!/usr/bin/env python3
"""
Subset generator for hit expand pipeline.

This module generates random subsets of sequences for structure prediction
and organizes them into batches for parallel processing.
"""

import os
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import json
from tqdm import tqdm

from af_claseq.utils.logging_utils import get_logger
from af_claseq.utils.sequence_processing import A3MParser, SequenceFormatError

logger = get_logger(__name__)


@dataclass
class SubsetConfig:
    """Configuration for subset generation."""
    num_subsets: int = 2000
    num_random_sequences: int = 8
    num_batches: int = 80
    batch_prefix: str = "batch"
    output_prefix: str = "subset"
    ensure_query_first: bool = True
    random_seed: Optional[int] = 42
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.num_subsets < 1:
            raise ValueError("Number of subsets must be positive")
        if self.num_random_sequences < 1:
            raise ValueError("Number of random sequences must be positive")
        if self.num_batches < 1:
            raise ValueError("Number of batches must be positive")


class SubsetGeneratorError(Exception):
    """Raised when subset generation fails."""
    pass


class SubsetGenerator:
    """
    Generates random subsets of sequences for structure prediction.
    
    This class takes an expanded MSA and generates random subsets for
    structure prediction, organizing them into batches for parallel processing.
    """
    
    def __init__(self, config: SubsetConfig):
        """
        Initialize subset generator.
        
        Args:
            config: Subset generation configuration
        """
        self.config = config
        
        # Set random seed for reproducibility
        if config.random_seed is not None:
            random.seed(config.random_seed)
            logger.info(f"Random seed set to {config.random_seed}")
        
        logger.info("Subset generator initialized")
    
    def generate_subsets(self,
                        expanded_msa: Path,
                        output_dir: Path) -> Dict[str, any]:
        """
        Generate random subsets from expanded MSA.
        
        Args:
            expanded_msa: Path to expanded MSA file
            output_dir: Output directory for subsets
            
        Returns:
            Dictionary with generation results and statistics
            
        Raises:
            SubsetGeneratorError: If generation fails
        """
        try:
            # Parse expanded MSA
            logger.info(f"Loading expanded MSA from {expanded_msa}")
            parser = A3MParser(strict_validation=False)
            sequences = parser.parse_file(expanded_msa)
            
            if len(sequences) == 0:
                raise SubsetGeneratorError("No sequences found in expanded MSA")
            
            # Get query sequence (first sequence)
            query_header, query_sequence = parser.get_query_sequence(sequences)
            logger.info(f"Query sequence: {query_header}")
            
            # Get non-query sequences for sampling
            non_query_sequences = {h: s for h, s in sequences.items() if h != query_header}
            
            if len(non_query_sequences) < self.config.num_random_sequences:
                logger.warning(f"Only {len(non_query_sequences)} non-query sequences available, "
                             f"but {self.config.num_random_sequences} requested per subset")
                # Adjust to available sequences
                actual_num_seqs = len(non_query_sequences)
            else:
                actual_num_seqs = self.config.num_random_sequences
            
            # Create output directory
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate subsets
            subset_paths = self._generate_random_subsets(
                query_header=query_header,
                query_sequence=query_sequence,
                non_query_sequences=non_query_sequences,
                output_dir=output_dir,
                num_sequences_per_subset=actual_num_seqs
            )
            
            # Organize subsets into batches
            batch_info = self._organize_into_batches(subset_paths, output_dir)
            
            # Generate statistics
            stats = {
                "total_sequences": len(sequences),
                "query_sequence": query_header,
                "non_query_sequences": len(non_query_sequences),
                "subsets_generated": len(subset_paths),
                "sequences_per_subset": actual_num_seqs,
                "batches_created": len(batch_info),
                "subsets_per_batch": [len(batch) for batch in batch_info.values()],
                "config": {
                    "num_subsets": self.config.num_subsets,
                    "num_random_sequences": self.config.num_random_sequences,
                    "num_batches": self.config.num_batches,
                    "random_seed": self.config.random_seed
                }
            }
            
            # Save generation metadata
            metadata_file = output_dir / "subset_generation_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"Generated {len(subset_paths)} subsets in {len(batch_info)} batches")
            
            return {
                "subset_paths": subset_paths,
                "batch_info": batch_info,
                "statistics": stats,
                "metadata_file": str(metadata_file)
            }
            
        except Exception as e:
            logger.error(f"Subset generation failed: {e}")
            raise SubsetGeneratorError(f"Subset generation failed: {e}")

    def generate_subsets_with_query(self,
                                   expanded_msa: Path,
                                   query_header: str,
                                   query_sequence: str,
                                   output_dir: Path) -> Dict[str, any]:
        """
        Generate random subsets with a specific query sequence always included first.
        
        Args:
            expanded_msa: Path to expanded MSA file
            query_header: Query sequence header
            query_sequence: Query sequence content
            output_dir: Output directory for subsets
            
        Returns:
            Dictionary with generation results and statistics
            
        Raises:
            SubsetGeneratorError: If generation fails
        """
        try:
            # Parse expanded MSA
            logger.info(f"Loading expanded MSA from {expanded_msa}")
            parser = A3MParser(strict_validation=False)
            sequences = parser.parse_file(expanded_msa)
            
            if len(sequences) == 0:
                raise SubsetGeneratorError("No sequences found in expanded MSA")
            
            logger.info(f"Using provided query sequence: {query_header}")
            
            # Use all sequences as potential sampling pool (representative sequences)
            sampling_sequences = sequences
            
            if len(sampling_sequences) < self.config.num_random_sequences:
                logger.warning(f"Only {len(sampling_sequences)} sequences available, "
                             f"but {self.config.num_random_sequences} requested per subset")
                # Adjust to available sequences
                actual_num_seqs = len(sampling_sequences)
            else:
                actual_num_seqs = self.config.num_random_sequences
            
            # Create output directory
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate subsets with the specific query sequence
            subset_paths = self._generate_random_subsets(
                query_header=query_header,
                query_sequence=query_sequence,
                non_query_sequences=sampling_sequences,  # Use all representative sequences for sampling
                output_dir=output_dir,
                num_sequences_per_subset=actual_num_seqs
            )
            
            # Organize subsets into batches
            batch_info = self._organize_into_batches(subset_paths, output_dir)
            
            # Generate statistics
            stats = {
                "total_sequences": len(sequences),
                "query_sequence": query_header,
                "sampling_sequences": len(sampling_sequences),
                "subsets_generated": len(subset_paths),
                "sequences_per_subset": actual_num_seqs + 1,  # +1 for query sequence
                "batches_created": len(batch_info),
                "subsets_per_batch": [len(batch) for batch in batch_info.values()],
                "config": {
                    "num_subsets": self.config.num_subsets,
                    "num_random_sequences": self.config.num_random_sequences,
                    "num_batches": self.config.num_batches,
                    "random_seed": self.config.random_seed
                }
            }
            
            # Save generation metadata
            metadata_file = output_dir / "subset_generation_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"Generated {len(subset_paths)} subsets with query sequence in {len(batch_info)} batches")
            
            return {
                "subset_paths": subset_paths,
                "batch_info": batch_info,
                "statistics": stats,
                "metadata_file": str(metadata_file)
            }
            
        except Exception as e:
            logger.error(f"Subset generation with query failed: {e}")
            raise SubsetGeneratorError(f"Subset generation with query failed: {e}")
    
    def _generate_random_subsets(self,
                                query_header: str,
                                query_sequence: str,
                                non_query_sequences: Dict[str, str],
                                output_dir: Path,
                                num_sequences_per_subset: int) -> List[Path]:
        """
        Generate random subsets of sequences.
        
        Args:
            query_header: Query sequence header
            query_sequence: Query sequence
            non_query_sequences: Non-query sequences to sample from
            output_dir: Output directory
            num_sequences_per_subset: Number of sequences per subset
            
        Returns:
            List of subset file paths
        """
        subset_paths = []
        non_query_list = list(non_query_sequences.items())
        
        logger.info(f"Generating {self.config.num_subsets} subsets with {num_sequences_per_subset} sequences each")
        
        # Temporarily suppress INFO logging from sequence_processing module
        seq_processing_logger = logging.getLogger('af_claseq.sequence_processing')
        original_level = seq_processing_logger.level
        seq_processing_logger.setLevel(logging.WARNING)
        
        # Create progress bar
        pbar = tqdm(range(self.config.num_subsets), desc="Generating subsets", unit="subset")
        
        for i in pbar:
            subset_id = f"{self.config.output_prefix}_{i+1:06d}"
            subset_file = output_dir / f"{subset_id}.a3m"
            
            # Update progress bar description
            pbar.set_postfix({"current": subset_id})
            
            # Sample random sequences
            if len(non_query_list) >= num_sequences_per_subset:
                selected_sequences = random.sample(non_query_list, num_sequences_per_subset)
            else:
                # If not enough sequences, use all with replacement
                selected_sequences = random.choices(non_query_list, k=num_sequences_per_subset)
            
            # Create subset MSA
            subset_sequences = {}
            
            # Add query sequence first if configured
            if self.config.ensure_query_first:
                subset_sequences[query_header] = query_sequence
            
            # Add selected sequences
            for seq_header, seq_sequence in selected_sequences:
                subset_sequences[seq_header] = seq_sequence
            
            # Write subset to file
            parser = A3MParser(strict_validation=False)
            parser.write_sequences(subset_sequences, subset_file, ensure_query_first=True)
            
            subset_paths.append(subset_file)
        
        # Close progress bar
        pbar.close()
        
        # Restore original logging level
        seq_processing_logger.setLevel(original_level)
        
        logger.info(f"Generated {len(subset_paths)} subset files")
        return subset_paths
    
    def _organize_into_batches(self,
                              subset_paths: List[Path],
                              output_dir: Path) -> Dict[str, List[Path]]:
        """
        Organize subsets into batches for parallel processing.
        
        Args:
            subset_paths: List of subset file paths
            output_dir: Output directory
            
        Returns:
            Dictionary mapping batch names to subset paths
        """
        batch_info = {}
        
        # Calculate subsets per batch
        subsets_per_batch = len(subset_paths) // self.config.num_batches
        remainder = len(subset_paths) % self.config.num_batches
        
        logger.info(f"Organizing {len(subset_paths)} subsets into {self.config.num_batches} batches")
        logger.info(f"Base subsets per batch: {subsets_per_batch}, remainder: {remainder}")
        
        start_idx = 0
        moved_files = []  # Track files that have been moved
        
        for batch_num in range(self.config.num_batches):
            batch_name = f"{self.config.batch_prefix}_{batch_num+1:03d}"
            
            # Calculate end index for this batch
            batch_size = subsets_per_batch + (1 if batch_num < remainder else 0)
            end_idx = start_idx + batch_size
            
            # Get subsets for this batch
            batch_subsets = subset_paths[start_idx:end_idx]
            
            # Create batch directory and move subsets
            batch_dir = output_dir / batch_name
            batch_dir.mkdir(exist_ok=True)
            
            # Move subset files to batch directory
            batch_moved_paths = []
            for subset_path in batch_subsets:
                dest_path = batch_dir / subset_path.name
                if not dest_path.exists():
                    try:
                        # Move file to batch directory
                        import shutil
                        shutil.move(str(subset_path), str(dest_path))
                        batch_moved_paths.append(dest_path)
                        moved_files.append(subset_path)
                        logger.debug(f"Moved {subset_path.name} to {batch_name}")
                    except Exception as e:
                        logger.warning(f"Failed to move {subset_path}: {e}")
                        # Fallback to copying if move fails
                        shutil.copy2(subset_path, dest_path)
                        batch_moved_paths.append(dest_path)
                else:
                    batch_moved_paths.append(dest_path)
            
            # Update batch info with new paths
            batch_info[batch_name] = batch_moved_paths
            
            start_idx = end_idx
            logger.debug(f"Batch {batch_name}: {len(batch_moved_paths)} subsets")
        
        logger.info(f"Created {len(batch_info)} batches")
        logger.info(f"Moved {len(moved_files)} subset files into batch directories")
        logger.info("A3M files outside batch folders have been cleaned up for neater file structure")
        
        return batch_info
    
    
    def validate_subsets(self, subset_paths: List[Path]) -> Dict[str, any]:
        """
        Validate generated subsets.
        
        Args:
            subset_paths: List of subset file paths
            
        Returns:
            Validation results
        """
        validation_results = {
            "total_subsets": len(subset_paths),
            "valid_subsets": 0,
            "invalid_subsets": 0,
            "errors": []
        }
        
        parser = A3MParser(strict_validation=False)
        
        for subset_path in subset_paths:
            try:
                sequences = parser.parse_file(subset_path)
                
                if len(sequences) > 0:
                    validation_results["valid_subsets"] += 1
                else:
                    validation_results["invalid_subsets"] += 1
                    validation_results["errors"].append(f"Empty subset: {subset_path}")
                    
            except Exception as e:
                validation_results["invalid_subsets"] += 1
                validation_results["errors"].append(f"Parse error in {subset_path}: {e}")
        
        validation_results["success_rate"] = (
            validation_results["valid_subsets"] / validation_results["total_subsets"]
            if validation_results["total_subsets"] > 0 else 0.0
        )
        
        logger.info(f"Subset validation: {validation_results['valid_subsets']}/{validation_results['total_subsets']} valid "
                   f"({validation_results['success_rate']:.1%} success rate)")
        
        return validation_results
    
    def get_subset_statistics(self, subset_paths: List[Path]) -> Dict[str, any]:
        """
        Generate detailed statistics about generated subsets.
        
        Args:
            subset_paths: List of subset file paths
            
        Returns:
            Dictionary with subset statistics
        """
        stats = {
            "total_subsets": len(subset_paths),
            "sequence_counts": [],
            "file_sizes": [],
            "unique_sequences": set(),
            "most_common_sequences": {}
        }
        
        parser = A3MParser(strict_validation=False)
        sequence_counter = {}
        
        for subset_path in subset_paths:
            try:
                # Get file size
                stats["file_sizes"].append(subset_path.stat().st_size)
                
                # Parse sequences
                sequences = parser.parse_file(subset_path)
                stats["sequence_counts"].append(len(sequences))
                
                # Track unique sequences
                for header, sequence in sequences.items():
                    stats["unique_sequences"].add(sequence)
                    
                    # Count sequence occurrences
                    if sequence in sequence_counter:
                        sequence_counter[sequence] += 1
                    else:
                        sequence_counter[sequence] = 1
                        
            except Exception as e:
                logger.warning(f"Error processing {subset_path} for statistics: {e}")
        
        # Calculate statistics
        if stats["sequence_counts"]:
            stats["sequence_count_stats"] = {
                "min": min(stats["sequence_counts"]),
                "max": max(stats["sequence_counts"]),
                "mean": sum(stats["sequence_counts"]) / len(stats["sequence_counts"]),
                "median": sorted(stats["sequence_counts"])[len(stats["sequence_counts"]) // 2]
            }
        
        if stats["file_sizes"]:
            stats["file_size_stats"] = {
                "min_bytes": min(stats["file_sizes"]),
                "max_bytes": max(stats["file_sizes"]),
                "mean_bytes": sum(stats["file_sizes"]) / len(stats["file_sizes"]),
                "total_bytes": sum(stats["file_sizes"])
            }
        
        # Most common sequences (top 10)
        if sequence_counter:
            most_common = sorted(sequence_counter.items(), key=lambda x: x[1], reverse=True)[:10]
            stats["most_common_sequences"] = {
                f"sequence_{i+1}": {"sequence": seq[:50] + "...", "count": count}
                for i, (seq, count) in enumerate(most_common)
            }
        
        stats["unique_sequence_count"] = len(stats["unique_sequences"])
        # Don't store the actual unique sequences to save memory
        del stats["unique_sequences"]
        
        return stats