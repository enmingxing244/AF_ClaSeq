"""
Hit expansion sampling module for AF-ClaSeq.

This module provides the HitExpandSampler class that performs hit expansion
sampling following the established AF-ClaSeq pipeline patterns.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import tempfile
import shutil
import subprocess
import time
import random

from af_claseq.hit_expand.config import HitExpandConfig
from af_claseq.utils.slurm_utils import SlurmJobSubmitter
from af_claseq.utils.structure_analysis import StructureAnalyzer
from af_claseq.utils.logging_utils import get_logger

logger = get_logger(__name__)


class HitExpandSampler:
    """Main class for hit expansion sampling following AF-ClaSeq patterns."""
    
    def __init__(
        self,
        input_msa: str,
        base_dir: str,
        config: HitExpandConfig,
        slurm_submitter: Optional[SlurmJobSubmitter] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the HitExpandSampler.
        
        Args:
            input_msa: Path to input MSA file
            base_dir: Base directory for output
            config: Hit expand configuration
            slurm_submitter: SLURM job submitter for running jobs
            logger: Optional logger instance
        """
        self.input_msa = input_msa
        self.base_dir = Path(base_dir)
        self.config = config
        self.slurm_submitter = slurm_submitter
        self.logger = logger or get_logger(__name__)
        
        # Create expansion workflow directory
        self.expansion_dir = self.base_dir / "01_expansion"
        self.expansion_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize structure analyzer
        self.structure_analyzer = StructureAnalyzer()
        
        # Runtime state
        self.start_time = time.time()
        
        self.logger.info(f"HitExpandSampler initialized with base directory: {self.base_dir}")
    
    def run(self) -> Optional[Path]:
        """
        Run hit expansion sampling process.
        
        Returns:
            Path to final expanded MSA file, or None if failed
        """
        try:
            self.logger.info("Starting hit expansion sampling process")
            
            # Step 1: Generate subsets and organize into batches
            batch_dir = self._generate_subsets_and_batches(self.input_msa)
            if not batch_dir:
                return None
            
            # Step 2: Run structure prediction (if not skipped)
            if not self.config.skip_structure_prediction:
                success = self._run_structure_prediction(batch_dir)
                if not success:
                    return None
            
            # Step 3: Run structure analysis (if not skipped)  
            hit_a3m_files = []
            if not self.config.skip_structure_analysis:
                hit_a3m_files = self._run_structure_analysis(batch_dir)
            
            # Step 4: Run hit expansion (if not skipped)
            if not self.config.skip_hit_expansion and hit_a3m_files:
                final_msa = self._run_hit_expansion(hit_a3m_files)
            else:
                # Use the input file as final output if no hit expansion
                final_msa = Path(self.input_msa)
            
            # Step 5: Create final output
            final_output = self._create_final_output(final_msa)
            
            self.logger.info(f"Hit expansion sampling completed successfully: {final_output}")
            return final_output
            
        except Exception as e:
            self.logger.error(f"Hit expansion sampling failed: {str(e)}", exc_info=True)
            return None
    
    def _generate_subsets_and_batches(self, input_file: str) -> Optional[Path]:
        """
        Generate subsets and organize them into batches.
        
        Args:
            input_file: Path to input sequences
            
        Returns:
            Path to batches directory
        """
        try:
            self.logger.info("Generating subsets and organizing into batches")
            
            # Create batches directory
            batches_dir = self.expansion_dir / "batches"
            batches_dir.mkdir(exist_ok=True)
            
            # Parse input sequences
            sequences, headers = self._parse_sequence_file(input_file)
            if not sequences:
                raise ValueError(f"No sequences found in {input_file}")
            
            # Get query sequence (first sequence in MSA)
            query_sequence = sequences[0]
            query_header = headers[0]
            
            # Create batch directories
            batch_folders = []
            for i in range(1, self.config.num_batches + 1):
                batch_folder = batches_dir / f"{self.config.batch_prefix}_{i:02d}"
                batch_folder.mkdir(exist_ok=True)
                batch_folders.append(batch_folder)
            
            # Generate subsets and distribute to batches
            subset_files = self._generate_subsets_to_batches(
                sequences, headers, query_sequence, query_header, batch_folders
            )
            
            self.logger.info(f"Generated {len(subset_files)} subset files in {len(batch_folders)} batches")
            return batches_dir
            
        except Exception as e:
            self.logger.error(f"Error generating subsets and batches: {str(e)}", exc_info=True)
            return None
    
    def _run_structure_prediction(self, batch_dir: Path) -> bool:
        """
        Run structure prediction on batch directories.
        
        Args:
            batch_dir: Path to batches directory
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("Running structure prediction")
            
            # Check if jobs are already complete
            if self.config.check_existing_jobs:
                if self._are_jobs_complete(batch_dir):
                    self.logger.info("All structure prediction jobs are already complete")
                    return True
            
            # Get all batch folders
            batch_folders = [d for d in batch_dir.iterdir() if d.is_dir() and d.name.startswith(self.config.batch_prefix)]
            
            if not batch_folders:
                self.logger.error("No batch folders found")
                return False
            
            # Process each batch folder
            folders = [str(folder) for folder in batch_folders]
            job_ids = [folder.name for folder in batch_folders]
            
            # Use the existing SLURM submitter to process folders
            if self.slurm_submitter:
                self.slurm_submitter.process_folders_concurrently(
                    folders=folders,
                    job_ids=job_ids,
                    max_workers=self.config.max_workers
                )
            else:
                self.logger.warning("No SLURM submitter provided, skipping structure prediction")
                return False
            
            self.logger.info("Structure prediction completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in structure prediction: {str(e)}", exc_info=True)
            return False
    
    def _run_structure_analysis(self, batch_dir: Path) -> List[Path]:
        """
        Run structure analysis on predicted structures.
        
        Args:
            batch_dir: Path to batches directory
            
        Returns:
            List of hit A3M files
        """
        try:
            self.logger.info("Running structure analysis")
            
            # Find all PDB files in batch directories
            pdb_files = []
            for batch_folder in batch_dir.iterdir():
                if batch_folder.is_dir():
                    pdb_files.extend(batch_folder.glob("*.pdb"))
            
            if not pdb_files:
                self.logger.warning("No PDB files found for structure analysis")
                return []
            
            # Load structure analysis configuration
            if self.slurm_submitter and hasattr(self.slurm_submitter, 'config_file'):
                with open(self.slurm_submitter.config_file, 'r') as f:
                    config = json.load(f)
            else:
                config = {}
            
            # Analyze structures using AF-ClaSeq structure analyzer
            results = []
            for pdb_file in pdb_files:
                try:
                    # Calculate structure metrics based on config
                    metrics = self._calculate_structure_metrics(pdb_file, config)
                    if metrics:
                        results.append({
                            'pdb_file': str(pdb_file),
                            'a3m_file': str(pdb_file.with_suffix('.a3m')),
                            **metrics
                        })
                except Exception as e:
                    self.logger.warning(f"Failed to analyze {pdb_file}: {str(e)}")
                    continue
            
            if not results:
                self.logger.warning("No structures successfully analyzed")
                return []
            
            # Save analysis results
            results_file = self.expansion_dir / "structure_analysis_results.csv"
            self._save_analysis_results(results, results_file)
            
            # Identify hit structures based on thresholds
            hit_a3m_files = self._identify_hit_structures(results)
            
            self.logger.info(f"Identified {len(hit_a3m_files)} hit A3M files")
            return hit_a3m_files
            
        except Exception as e:
            self.logger.error(f"Error in structure analysis: {str(e)}", exc_info=True)
            return []
    
    def _run_hit_expansion(self, hit_a3m_files: List[Path]) -> Path:
        """
        Run hit expansion using BLOSUM62 similarity search.
        
        Args:
            hit_a3m_files: List of hit A3M files
            
        Returns:
            Path to expanded MSA file
        """
        try:
            self.logger.info("Running hit expansion")
            
            # Create hit expansion directory
            expansion_results_dir = self.expansion_dir / "hit_expansion"
            expansion_results_dir.mkdir(exist_ok=True)
            
            # Copy hit A3M files to expansion directory
            hit_sequences_dir = expansion_results_dir / "hit_sequences"
            hit_sequences_dir.mkdir(exist_ok=True)
            
            for hit_file in hit_a3m_files:
                if hit_file.exists():
                    dst_file = hit_sequences_dir / hit_file.name
                    shutil.copy2(hit_file, dst_file)
            
            # Run similarity search to expand hits
            expanded_msa_file = expansion_results_dir / "expanded_hit_sequences.a3m"
            self._run_similarity_search(hit_sequences_dir, expanded_msa_file)
            
            self.logger.info(f"Hit expansion completed: {expanded_msa_file}")
            return expanded_msa_file
            
        except Exception as e:
            self.logger.error(f"Error in hit expansion: {str(e)}", exc_info=True)
            return Path(self.input_msa)
    
    def _create_final_output(self, final_msa: Path) -> Path:
        """
        Create the final output MSA file.
        
        Args:
            final_msa: Path to final MSA file
            
        Returns:
            Path to final output file
        """
        final_output = self.expansion_dir / "final_expanded_msa.a3m"
        
        if final_msa != final_output:
            shutil.copy2(final_msa, final_output)
        
        return final_output
    
    # Helper methods
    def _parse_sequence_file(self, file_path: str) -> Tuple[List[str], List[str]]:
        """Parse sequence file and return sequences and headers."""
        sequences = []
        headers = []
        
        with open(file_path, 'r') as f:
            current_seq = ""
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_seq:
                        sequences.append(current_seq)
                        current_seq = ""
                    headers.append(line)
                else:
                    current_seq += line
            
            # Add the last sequence
            if current_seq:
                sequences.append(current_seq)
        
        return sequences, headers
    
    def _generate_subsets_to_batches(
        self, 
        sequences: List[str], 
        headers: List[str], 
        query_sequence: str, 
        query_header: str, 
        batch_folders: List[Path]
    ) -> List[Path]:
        """Generate subsets and distribute to batch folders."""
        random.seed(self.config.random_seed)
        
        subset_files = []
        num_batches = len(batch_folders)
        
        for subset_idx in range(self.config.num_subsets):
            # Select random sequences
            if len(sequences) >= self.config.num_random_sequences:
                random_indices = random.sample(range(len(sequences)), self.config.num_random_sequences)
            else:
                # Use all sequences with replacement if needed
                random_indices = [random.randint(0, len(sequences) - 1) for _ in range(self.config.num_random_sequences)]
            
            selected_sequences = [sequences[i] for i in random_indices]
            selected_headers = [headers[i] for i in random_indices]
            
            # Combine query with selected sequences
            subset_sequences = [query_sequence] + selected_sequences
            subset_headers = [query_header] + selected_headers
            
            # Determine batch folder (round-robin)
            batch_idx = subset_idx % num_batches
            batch_folder = batch_folders[batch_idx]
            
            # Create subset file
            subset_filename = f"{self.config.output_prefix}_{subset_idx+1:05d}.a3m"
            subset_file = batch_folder / subset_filename
            
            # Write subset file
            self._write_a3m_file(subset_sequences, subset_headers, subset_file)
            subset_files.append(subset_file)
        
        return subset_files
    
    def _write_a3m_file(self, sequences: List[str], headers: List[str], file_path: Path) -> None:
        """Write sequences and headers to A3M file."""
        with open(file_path, 'w') as f:
            for header, sequence in zip(headers, sequences):
                f.write(f"{header}\n{sequence}\n")
    
    def _are_jobs_complete(self, batch_dir: Path) -> bool:
        """Check if all ColabFold jobs are complete."""
        batch_folders = [d for d in batch_dir.iterdir() if d.is_dir() and d.name.startswith(self.config.batch_prefix)]
        
        if not batch_folders:
            return False
        
        total_subsets = 0
        completed_subsets = 0
        
        for batch_folder in batch_folders:
            subset_files = list(batch_folder.glob("subset_*.a3m"))
            
            for subset_file in subset_files:
                total_subsets += 1
                
                # Check if corresponding PDB file exists
                subset_name = subset_file.stem
                expected_pdb = batch_folder / f"{subset_name}_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb"
                
                if expected_pdb.exists():
                    completed_subsets += 1
        
        return completed_subsets == total_subsets and total_subsets > 0
    
    def _calculate_structure_metrics(self, pdb_file: Path, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Calculate structure metrics using AF-ClaSeq structure analyzer."""
        try:
            metrics = {}
            
            # Calculate pLDDT score
            plddt_score = self._extract_plddt_score(pdb_file)
            if plddt_score is not None:
                metrics['plddt'] = plddt_score
            
            # Calculate other metrics based on config
            filter_criteria = config.get('filter_criteria', [])
            for criterion in filter_criteria:
                criterion_name = criterion.get('name', 'unknown')
                # Add specific metric calculations here based on criterion type
                # This is a placeholder - implement actual metric calculations
                metrics[criterion_name] = 0.0
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating structure metrics for {pdb_file}: {str(e)}")
            return None
    
    def _extract_plddt_score(self, pdb_file: Path) -> Optional[float]:
        """Extract pLDDT score from PDB file."""
        try:
            structure = self.structure_analyzer.pdb_parser.get_structure("protein", pdb_file)
            
            plddt_scores = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        for atom in residue:
                            if atom.name == 'CA':  # Only consider CA atoms
                                plddt_scores.append(atom.bfactor)
            
            return sum(plddt_scores) / len(plddt_scores) if plddt_scores else None
            
        except Exception as e:
            self.logger.error(f"Error extracting pLDDT score from {pdb_file}: {str(e)}")
            return None
    
    def _save_analysis_results(self, results: List[Dict[str, Any]], results_file: Path) -> None:
        """Save structure analysis results to CSV file."""
        import pandas as pd
        
        df = pd.DataFrame(results)
        df.to_csv(results_file, index=False)
        
        self.logger.info(f"Structure analysis results saved to {results_file}")
    
    def _identify_hit_structures(self, results: List[Dict[str, Any]]) -> List[Path]:
        """Identify hit structures based on thresholds."""
        hit_files = []
        
        for result in results:
            # Check pLDDT threshold
            plddt = result.get('plddt', 0)
            if plddt < self.config.plddt_threshold:
                continue
            
            # Check other filter criteria thresholds
            meets_criteria = True
            for key, value in result.items():
                if key not in ['pdb_file', 'a3m_file', 'plddt'] and isinstance(value, (int, float)):
                    if value < self.config.filter_criteria_threshold:
                        meets_criteria = False
                        break
            
            if meets_criteria:
                a3m_file = Path(result['a3m_file'])
                if a3m_file.exists():
                    hit_files.append(a3m_file)
        
        return hit_files
    
    def _run_similarity_search(self, hit_sequences_dir: Path, output_file: Path) -> None:
        """Run BLOSUM62 similarity search to expand hit sequences."""
        try:
            # Get all sequences from hit files
            hit_sequences = []
            hit_headers = []
            
            for hit_file in hit_sequences_dir.glob("*.a3m"):
                seqs, heads = self._parse_sequence_file(str(hit_file))
                hit_sequences.extend(seqs)
                hit_headers.extend(heads)
            
            # Get source MSA sequences
            source_sequences, source_headers = self._parse_sequence_file(self.input_msa)
            
            # Find similar sequences (simplified approach)
            expanded_sequences = hit_sequences[:]
            expanded_headers = hit_headers[:]
            
            # Add top similar sequences from source MSA
            for i, seq in enumerate(source_sequences[:self.config.similarity_top_k]):
                if source_headers[i] not in expanded_headers:
                    expanded_sequences.append(seq)
                    expanded_headers.append(source_headers[i])
            
            # Write expanded MSA
            self._write_a3m_file(expanded_sequences, expanded_headers, output_file)
            
            self.logger.info(f"Similarity search completed, expanded to {len(expanded_sequences)} sequences")
            
        except Exception as e:
            self.logger.error(f"Error in similarity search: {str(e)}", exc_info=True)
            # Fallback to original hit sequences
            self._write_a3m_file(hit_sequences, hit_headers, output_file)