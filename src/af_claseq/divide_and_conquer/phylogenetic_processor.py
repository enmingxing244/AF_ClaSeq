"""
Phylogenetic processing for the divide-and-conquer workflow.
Handles FastTree execution and phylogenetic clade-based sequence splitting.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import OrderedDict
import logging

from af_claseq.divide_and_conquer.nwk_parse import split_alignment_by_clades

from af_claseq.divide_and_conquer.utils import (
    process_sequences_with_header_conflicts,
    validate_file_exists
)
from af_claseq.utils.sequence_processing import write_a3m
from af_claseq.utils.exceptions import WorkflowError
from af_claseq.utils.sequence_processing import read_a3m_to_dict


class PhylogeneticProcessor:
    """
    Handles phylogenetic tree construction and clade-based sequence splitting.
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize PhylogeneticProcessor.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.fasttree_binary = config['input']['fasttree_binary']
        self.clade_config = config['clade_splitting']
        # Use working_dir directly as specified in config
        self.working_dir = config.get('output', {}).get('working_dir', '.')

        # Create working directory
        os.makedirs(self.working_dir, exist_ok=True)
        
        # Validate FastTree binary exists
        validate_file_exists(self.fasttree_binary, "FastTree binary")
        self.logger.info(f"Using FastTree binary: {self.fasttree_binary}")
        self.logger.info(f"Working directory: {self.working_dir}")
    
    def preprocess_a3m(self, a3m_file: str) -> Tuple[str, str, str]:
        """
        Preprocess A3M file to handle header conflicts and remove duplicates.
        
        Args:
            a3m_file: Path to input A3M file
            
        Returns:
            Tuple of (preprocessed_file_path, query_header, query_sequence)
        """
        self.logger.info(f"Preprocessing A3M file: {a3m_file}")
        validate_file_exists(a3m_file, "Input A3M file")
        
        # Read original sequences
        sequences = read_a3m_to_dict(a3m_file)
        self.logger.info(f"Read {len(sequences)} sequences from {a3m_file}")
        
        if not sequences:
            raise WorkflowError(f"No sequences found in {a3m_file}")
        
        # Extract query sequence (first sequence)
        sequence_items = list(sequences.items())
        query_header, original_query_sequence = sequence_items[0]
        self.logger.info(f"Query sequence header: {query_header}")
        self.logger.info(f"Original query sequence length: {len(original_query_sequence)}")
        
        # Remove lowercase letters from sequences (insertions relative to reference)
        self.logger.info("Removing lowercase letters (insertions) from sequences...")
        cleaned_sequences = {}
        for header, sequence in sequences.items():
            # Remove lowercase letters, keeping only uppercase letters and gaps
            cleaned_seq = ''.join(c for c in sequence if c.isupper() or c == '-')
            cleaned_sequences[header] = cleaned_seq
        
        # Log cleaning statistics
        original_lengths = [len(seq) for seq in sequences.values()]
        cleaned_lengths = [len(seq) for seq in cleaned_sequences.values()]
        if original_lengths and cleaned_lengths:
            avg_original = sum(original_lengths) / len(original_lengths)
            avg_cleaned = sum(cleaned_lengths) / len(cleaned_lengths)
            self.logger.info(f"Sequence cleaning: avg length {avg_original:.1f} -> {avg_cleaned:.1f}")
        
        # Process sequences for header conflicts and deduplication
        processed_sequences = process_sequences_with_header_conflicts(cleaned_sequences, self.logger)
        
        # Ensure query sequence is first with its processed header
        final_sequences = OrderedDict()
        query_found = False
        processed_query_header = None
        
        # Get the cleaned query sequence  
        cleaned_query_sequence = ''.join(c for c in original_query_sequence if c.isupper() or c == '-')
        self.logger.info(f"Cleaned query sequence length: {len(cleaned_query_sequence)}")
        
        # Find the query sequence in processed sequences
        for header, sequence in processed_sequences.items():
            if sequence == cleaned_query_sequence and not query_found:
                processed_query_header = header
                final_sequences[header] = sequence
                query_found = True
                break
        
        # Add remaining sequences
        for header, sequence in processed_sequences.items():
            if header != processed_query_header:
                final_sequences[header] = sequence
        
        if not query_found:
            raise WorkflowError("Query sequence lost during preprocessing")
        
        # Write preprocessed file in working directory
        file_stem = Path(a3m_file).stem
        preprocessed_file = os.path.join(self.working_dir, f"{file_stem}_preprocessed.a3m")
        write_a3m(final_sequences, preprocessed_file)
        
        self.logger.info(f"Preprocessed file saved: {preprocessed_file}")
        self.logger.info(f"Final sequence count: {len(final_sequences)}")
        self.logger.info(f"Processed query header: {processed_query_header}")
        
        return preprocessed_file, processed_query_header, cleaned_query_sequence
    
    def run_fasttree(self, a3m_file: str) -> str:
        """
        Run FastTree to generate phylogenetic tree.
        
        Args:
            a3m_file: Path to preprocessed A3M file
            
        Returns:
            Path to generated tree file
        """
        self.logger.info("Running FastTree for phylogenetic tree construction...")
        
        validate_file_exists(a3m_file, "Preprocessed A3M file")
        
        # Prepare output file in working directory
        file_stem = Path(a3m_file).stem
        tree_file = os.path.join(self.working_dir, f"{file_stem}.nwk")
        
        # FastTree command
        cmd = [self.fasttree_binary, a3m_file]
        
        self.logger.info(f"FastTree command: {' '.join(cmd)} > {tree_file}")
        
        try:
            # Run FastTree and capture output
            with open(tree_file, 'w') as output_file:
                result = subprocess.run(
                    cmd,
                    stdout=output_file,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True
                )
            
            # Check if tree file was created and is not empty
            if not os.path.exists(tree_file) or os.path.getsize(tree_file) == 0:
                raise WorkflowError(f"FastTree failed to generate tree file: {tree_file}")
            
            self.logger.info(f"FastTree completed successfully")
            self.logger.info(f"Tree file generated: {tree_file}")
            
            # Log any warnings from FastTree stderr
            if result.stderr:
                self.logger.warning(f"FastTree stderr: {result.stderr}")
            
            return tree_file
            
        except subprocess.CalledProcessError as e:
            error_msg = f"FastTree execution failed: {e}"
            if e.stderr:
                error_msg += f"\nStderr: {e.stderr}"
            self.logger.error(error_msg)
            raise WorkflowError(error_msg)
        
        except Exception as e:
            error_msg = f"Unexpected error running FastTree: {e}"
            self.logger.error(error_msg)
            raise WorkflowError(error_msg)
    
    def split_by_clades(self, tree_file: str, a3m_file: str) -> List[str]:
        """
        Split alignment by phylogenetic clades using existing nwk_parse functionality.
        
        Args:
            tree_file: Path to phylogenetic tree file
            a3m_file: Path to preprocessed A3M file
            
        Returns:
            List of clade directory paths
        """
        self.logger.info("Splitting alignment by phylogenetic clades...")
        
        validate_file_exists(tree_file, "Tree file")
        validate_file_exists(a3m_file, "Preprocessed A3M file")
        
        # Prepare output directory - create clades directory in working directory
        output_dir = os.path.join(self.working_dir, "clades")
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract configuration parameters
        min_clade_size = self.clade_config.get('min_clade_size', 10)
        max_clade_size = self.clade_config.get('max_clade_size', 100)

        self.logger.info(f"Distance-guided clade splitting parameters:")
        self.logger.info(f"  Min clade size: {min_clade_size}")
        self.logger.info(f"  Max clade size: {max_clade_size}")
        self.logger.info(f"  Output directory: {output_dir}")
        
        try:
            # Use distance-guided phylogenetic clade splitting
            split_alignment_by_clades(
                tree_file=tree_file,
                a3m_file=a3m_file,
                output_dir=output_dir,
                min_clade_size=min_clade_size,
                max_clade_size=max_clade_size,
                verbose=True
            )
            
            # Get list of created clade directories
            clade_dirs = []
            if os.path.exists(output_dir):
                for item in os.listdir(output_dir):
                    item_path = os.path.join(output_dir, item)
                    # Process both clade_*.a3m files AND unclustered.a3m
                    if os.path.isfile(item_path) and item.endswith('.a3m') and (item.startswith('clade_') or item == 'unclustered.a3m'):
                        # Create directory for this clade/unclustered file
                        clade_name = Path(item).stem
                        if item == 'unclustered.a3m':
                            clade_name = 'unclustered'  # Special name for unclustered sequences

                        clade_dir = os.path.join(output_dir, clade_name)
                        os.makedirs(clade_dir, exist_ok=True)
                        
                        # Move the a3m file into the clade directory
                        clade_a3m_file = os.path.join(clade_dir, f"{clade_name}.a3m")
                        os.rename(item_path, clade_a3m_file)
                        
                        clade_dirs.append(clade_dir)
            
            clade_dirs.sort()  # Sort for consistent ordering
            
            if not clade_dirs:
                raise WorkflowError("No clades were generated from tree splitting")
            
            # Count regular clades vs unclustered
            regular_clades = [d for d in clade_dirs if not os.path.basename(d) == 'unclustered']
            unclustered_dirs = [d for d in clade_dirs if os.path.basename(d) == 'unclustered']
            
            self.logger.info(f"Successfully created {len(regular_clades)} clades + {len(unclustered_dirs)} unclustered group:")
            
            # Log regular clades first
            for i, clade_dir in enumerate(regular_clades, 1):
                clade_a3m = os.path.join(clade_dir, f"{os.path.basename(clade_dir)}.a3m")
                if os.path.exists(clade_a3m):
                    with open(clade_a3m, 'r') as f:
                        seq_count = sum(1 for line in f if line.startswith('>'))
                    self.logger.info(f"  {i}. {clade_dir}: {seq_count} sequences")
                else:
                    self.logger.warning(f"  {i}. {clade_dir}: A3M file not found")
            
            # Log unclustered separately if it exists
            for clade_dir in unclustered_dirs:
                clade_a3m = os.path.join(clade_dir, f"{os.path.basename(clade_dir)}.a3m")
                if os.path.exists(clade_a3m):
                    with open(clade_a3m, 'r') as f:
                        seq_count = sum(1 for line in f if line.startswith('>'))
                    self.logger.info(f"  UNCLUSTERED: {clade_dir}: {seq_count} sequences")
                else:
                    self.logger.warning(f"  UNCLUSTERED: {clade_dir}: A3M file not found")
            
            return clade_dirs
            
        except Exception as e:
            error_msg = f"Clade splitting failed: {e}"
            self.logger.error(error_msg)
            raise WorkflowError(error_msg)
    
    def process_complete(self, a3m_file: str) -> Tuple[List[str], str, str]:
        """
        Complete phylogenetic processing workflow.
        
        Args:
            a3m_file: Path to input A3M file
            
        Returns:
            Tuple of (clade_directories, query_header, query_sequence)
        """
        self.logger.info("=" * 50)
        self.logger.info("PHYLOGENETIC PROCESSING STARTED")
        self.logger.info("=" * 50)
        
        try:
            # Step 1: Preprocess A3M file
            preprocessed_file, processed_query_header, cleaned_query_sequence = self.preprocess_a3m(a3m_file)
            
            # Step 2: Run FastTree
            tree_file = self.run_fasttree(preprocessed_file)
            
            # Step 3: Split by clades
            clade_dirs = self.split_by_clades(tree_file, preprocessed_file)
            
            self.logger.info("=" * 50)
            self.logger.info("PHYLOGENETIC PROCESSING COMPLETED")
            
            regular_clades = [d for d in clade_dirs if not os.path.basename(d) == 'unclustered']
            unclustered_dirs = [d for d in clade_dirs if os.path.basename(d) == 'unclustered']
            
            if unclustered_dirs:
                self.logger.info(f"Generated {len(regular_clades)} clades + 1 unclustered group (total: {len(clade_dirs)} groups)")
            else:
                self.logger.info(f"Generated {len(clade_dirs)} clades (no unclustered sequences)")
            self.logger.info("=" * 50)
            
            return clade_dirs, processed_query_header, cleaned_query_sequence
            
        except Exception as e:
            self.logger.error("=" * 50)
            self.logger.error("PHYLOGENETIC PROCESSING FAILED")
            self.logger.error(f"Error: {e}")
            self.logger.error("=" * 50)
            raise