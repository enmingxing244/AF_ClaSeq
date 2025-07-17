#!/usr/bin/env python3
"""
A3M Sequence Blending Script

This script allows you to blend sequences from multiple A3M files with fixed ratios.
It randomly extracts a specified number of sequences from each input A3M file,
reads their aligned sequences, and combines them into a new output A3M file.

Usage:
    python blend_a3m_sequences.py file1.a3m file2.a3m file3.a3m --num-sequences 100 50 25 --output blended.a3m

The number of sequences should match the order of input files.
"""

import argparse
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from af_claseq.utils.sequence_processing import A3MParser
from af_claseq.utils.logging_utils import get_logger


class A3MBlender:
    """A3M sequence blending utility."""
    
    def __init__(self, input_files: List[str], num_sequences: List[int], 
                 output_file: str, random_seed: int = 42, 
                 ensure_query_first: bool = True, logger: Optional[logging.Logger] = None):
        """
        Initialize A3M blender.
        
        Args:
            input_files: List of input A3M file paths
            num_sequences: List of sequence counts corresponding to each input file
            output_file: Output A3M file path
            random_seed: Random seed for reproducibility
            ensure_query_first: Whether to ensure query sequence is first
            logger: Optional logger instance
        """
        self.input_files = [Path(f) for f in input_files]
        self.num_sequences = num_sequences
        self.output_file = Path(output_file)
        self.random_seed = random_seed
        self.ensure_query_first = ensure_query_first
        self.logger = logger or get_logger(__name__)
        self.parser = A3MParser(strict_validation=False)
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        self.logger.info(f"Set random seed to {random_seed}")
        
        # Validate inputs
        if len(self.input_files) != len(self.num_sequences):
            raise ValueError(f"Number of input files ({len(self.input_files)}) must match number of sequence counts ({len(self.num_sequences)})")
    
    def blend_sequences(self) -> Path:
        """
        Blend sequences from multiple A3M files according to specifications.
        
        Returns:
            Path to the output blended A3M file
        """
        self.logger.info(f"Blending sequences from {len(self.input_files)} A3M files")
        
        blended_sequences = {}
        query_sequence = None
        query_header = None
        total_extracted = 0
        
        # Process each input file
        for i, (file_path, num_seqs) in enumerate(zip(self.input_files, self.num_sequences)):
            if not file_path.exists():
                self.logger.error(f"Input file not found: {file_path}")
                continue
            
            if num_seqs <= 0:
                self.logger.warning(f"Skipping {file_path}: num_sequences is {num_seqs}")
                continue
            
            self.logger.info(f"Processing {file_path}: extracting {num_seqs} sequences")
            
            # Parse sequences from file
            try:
                all_sequences = self.parser.parse_file(file_path)
                self.logger.debug(f"Loaded {len(all_sequences)} sequences from {file_path}")
                
                if not all_sequences:
                    self.logger.warning(f"No sequences found in {file_path}")
                    continue
                
                # Handle query sequence (first file determines query)
                if i == 0 and self.ensure_query_first:
                    # Use first sequence as query
                    seq_list = list(all_sequences.items())
                    query_header, query_sequence = seq_list[0]
                    available_sequences = seq_list[1:]  # Exclude query
                    self.logger.info(f"Using query sequence from first file: {query_header}")
                else:
                    available_sequences = list(all_sequences.items())
                
                # Sample sequences
                if len(available_sequences) >= num_seqs:
                    selected_sequences = random.sample(available_sequences, num_seqs)
                else:
                    self.logger.warning(f"Requested {num_seqs} sequences but only {len(available_sequences)} available in {file_path}")
                    selected_sequences = available_sequences  # Use all available
                
                # Add to blended collection with source prefix
                file_prefix = f"src{i+1}"
                for j, (header, sequence) in enumerate(selected_sequences):
                    # Create unique header to avoid conflicts
                    new_header = f">{file_prefix}_{j+1:06d}_{header.lstrip('>')}"
                    blended_sequences[new_header] = sequence
                
                extracted_count = len(selected_sequences)
                total_extracted += extracted_count
                self.logger.info(f"Extracted {extracted_count} sequences from {file_path}")
                
            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")
                continue
        
        if not blended_sequences:
            raise ValueError("No sequences were successfully extracted from any input file")
        
        # Prepare final sequence collection
        final_sequences = {}
        
        # Add query sequence first if specified
        if query_sequence and self.ensure_query_first:
            final_sequences[query_header] = query_sequence
            total_sequences = len(blended_sequences) + 1
        else:
            total_sequences = len(blended_sequences)
        
        # Add blended sequences in random order
        sequence_items = list(blended_sequences.items())
        random.shuffle(sequence_items)
        final_sequences.update(sequence_items)
        
        # Write output file
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.parser.write_sequences(final_sequences, self.output_file, ensure_query_first=self.ensure_query_first)
        
        self.logger.info(f"Successfully created blended A3M file: {self.output_file}")
        self.logger.info(f"Total sequences in output: {total_sequences}")
        self.logger.info(f"Query sequence included: {query_sequence is not None and self.ensure_query_first}")
        
        # Print summary
        self._print_blend_summary(total_extracted, total_sequences)
        
        return self.output_file
    
    def _print_blend_summary(self, total_extracted: int, total_sequences: int):
        """Print a summary of the blending operation."""
        
        print("\n" + "="*60)
        print("A3M SEQUENCE BLENDING SUMMARY")
        print("="*60)
        
        print(f"Output file: {self.output_file}")
        print(f"Total sequences: {total_sequences}")
        print(f"Random seed: {self.random_seed}")
        print()
        
        print("Source breakdown:")
        for i, (file_path, num_seqs) in enumerate(zip(self.input_files, self.num_sequences)):
            print(f"  Source {i+1}: {file_path.name}")
            print(f"    Requested: {num_seqs} sequences")
        
        print(f"\nTotal extracted: {total_extracted} sequences")
        print("="*60)


def main():
    """Main function."""
    
    parser = argparse.ArgumentParser(
        description="Blend sequences from multiple A3M files with fixed ratios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Blend sequences from 3 files with 100, 50, 25 sequences respectively
    python blend_a3m_sequences.py file1.a3m file2.a3m file3.a3m --num-sequences 100 50 25 --output blended.a3m
    
    # Blend with custom random seed
    python blend_a3m_sequences.py file1.a3m file2.a3m --num-sequences 200 100 --output result.a3m --random-seed 123
    
    # Blend without ensuring query sequence first
    python blend_a3m_sequences.py file1.a3m file2.a3m --num-sequences 50 50 --output result.a3m --no-query-first
        """
    )
    
    parser.add_argument(
        'input_files',
        nargs='+',
        help='Input A3M files to blend'
    )
    
    parser.add_argument(
        '--num-sequences', '-n',
        nargs='+',
        type=int,
        required=True,
        help='Number of sequences to extract from each input file (in same order as input files)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output A3M file path'
    )
    
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--no-query-first',
        action='store_true',
        help='Do not ensure query sequence is first in output'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = get_logger(__name__)
    
    # Validate arguments
    if len(args.input_files) != len(args.num_sequences):
        logger.error(f"Number of input files ({len(args.input_files)}) must match number of sequence counts ({len(args.num_sequences)})")
        sys.exit(1)
    
    # Check if all input files exist
    for file_path in args.input_files:
        if not Path(file_path).exists():
            logger.error(f"Input file not found: {file_path}")
            sys.exit(1)
    
    try:
        # Create blender and run
        blender = A3MBlender(
            input_files=args.input_files,
            num_sequences=args.num_sequences,
            output_file=args.output,
            random_seed=args.random_seed,
            ensure_query_first=not args.no_query_first,
            logger=logger
        )
        
        output_file = blender.blend_sequences()
        
        logger.info(f"A3M blending completed successfully: {output_file}")
        
    except Exception as e:
        logger.error(f"A3M blending failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()