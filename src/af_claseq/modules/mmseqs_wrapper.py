#!/usr/bin/env python3
"""
MMseqs2 wrapper for sequence clustering operations.

This module provides a clean interface to MMseqs2 for clustering sequences
in the hit expand pipeline.
"""

import os
import subprocess
import tempfile
import shutil
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from af_claseq.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class MMseqsConfig:
    """Configuration for MMseqs2 operations."""
    bin_path: str = "/fs/ess/PAA0203/xing244/packages/mmseqs/bin/mmseqs"
    coverage: float = 0.8
    min_seq_id: float = 0.7
    cov_mode: int = 0
    cluster_mode: int = 0
    threads: int = 8
    tmp_dir: str = "/tmp"
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not (0.0 <= self.coverage <= 1.0):
            raise ValueError("Coverage must be between 0.0 and 1.0")
        if not (0.0 <= self.min_seq_id <= 1.0):
            raise ValueError("Minimum sequence identity must be between 0.0 and 1.0")
        if self.threads < 1:
            raise ValueError("Number of threads must be positive")


class MMseqsError(Exception):
    """Raised when MMseqs2 operation fails."""
    pass


class MMseqsWrapper:
    """
    High-level wrapper for MMseqs2 sequence clustering operations.
    
    Provides clean, validated interface to MMseqs2 with proper error handling
    and temporary directory management.
    """
    
    def __init__(self, config: MMseqsConfig):
        """
        Initialize MMseqs2 wrapper.
        
        Args:
            config: MMseqs2 configuration
            
        Raises:
            MMseqsError: If MMseqs2 binary is not found or invalid
        """
        self.config = config
        
        # Validate MMseqs2 binary
        if not Path(config.bin_path).exists():
            raise MMseqsError(f"MMseqs2 binary not found: {config.bin_path}")
        
        # Test MMseqs2 executable
        try:
            result = subprocess.run(
                [config.bin_path, "version"],
                capture_output=True, text=True, check=True
            )
            logger.info(f"MMseqs2 version: {result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            raise MMseqsError(f"MMseqs2 binary test failed: {e}")
        except FileNotFoundError:
            raise MMseqsError(f"MMseqs2 binary not executable: {config.bin_path}")
        
        logger.info("MMseqs2 wrapper initialized successfully")
    
    def cluster_sequences(self, 
                         input_fasta: Path,
                         output_dir: Path,
                         prefix: str = "cluster") -> Dict[str, Any]:
        """
        Cluster sequences using MMseqs2.
        
        Args:
            input_fasta: Path to input FASTA file
            output_dir: Output directory for results
            prefix: Prefix for output files
            
        Returns:
            Dictionary with clustering results and statistics
            
        Raises:
            MMseqsError: If clustering fails
        """
        input_fasta = Path(input_fasta)
        output_dir = Path(output_dir)
        
        # Validate inputs
        if not input_fasta.exists():
            raise MMseqsError(f"Input FASTA file not found: {input_fasta}")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create temporary directory for MMseqs2 operations
        with tempfile.TemporaryDirectory(dir=self.config.tmp_dir, prefix="mmseqs_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            try:
                # Run clustering workflow
                result = self._run_clustering_workflow(
                    input_fasta=input_fasta,
                    output_dir=output_dir,
                    tmp_dir=tmp_path,
                    prefix=prefix
                )
                
                logger.info(f"MMseqs2 clustering completed successfully")
                return result
                
            except Exception as e:
                logger.error(f"MMseqs2 clustering failed: {e}")
                raise MMseqsError(f"Clustering failed: {e}")
    
    def _run_clustering_workflow(self,
                                input_fasta: Path,
                                output_dir: Path,
                                tmp_dir: Path,
                                prefix: str) -> Dict[str, Any]:
        """
        Run the complete MMseqs2 clustering workflow.
        
        Args:
            input_fasta: Input FASTA file
            output_dir: Output directory
            tmp_dir: Temporary directory
            prefix: Output prefix
            
        Returns:
            Dictionary with results and statistics
        """
        # Define file paths
        db_path = tmp_dir / f"{prefix}_db"
        cluster_db_path = tmp_dir / f"{prefix}_cluster"
        rep_seq_path = output_dir / f"{prefix}_representatives.fasta"
        cluster_tsv_path = output_dir / f"{prefix}_clusters.tsv"
        
        # Step 1: Create sequence database
        logger.info("Creating MMseqs2 database...")
        self._run_mmseqs_command([
            "createdb", str(input_fasta), str(db_path)
        ])
        
        # Step 2: Cluster sequences
        logger.info("Clustering sequences...")
        cluster_cmd = [
            "cluster", str(db_path), str(cluster_db_path), str(tmp_dir),
            "--cov-mode", str(self.config.cov_mode),
            "--cluster-mode", str(self.config.cluster_mode),
            "-c", str(self.config.coverage),
            "--min-seq-id", str(self.config.min_seq_id),
            "--threads", str(self.config.threads)
        ]
        self._run_mmseqs_command(cluster_cmd)
        
        # Step 3: Extract representative sequences
        logger.info("Extracting representative sequences...")
        self._run_mmseqs_command([
            "result2repseq", str(db_path), str(cluster_db_path), str(rep_seq_path)
        ])
        
        # Step 4: Create cluster TSV file
        logger.info("Creating cluster membership file...")
        self._run_mmseqs_command([
            "createtsv", str(db_path), str(db_path), str(cluster_db_path), str(cluster_tsv_path)
        ])
        
        # Parse results and generate statistics
        stats = self._parse_clustering_results(
            input_fasta=input_fasta,
            rep_seq_path=rep_seq_path,
            cluster_tsv_path=cluster_tsv_path
        )
        
        return {
            "representative_sequences": str(rep_seq_path),
            "cluster_membership": str(cluster_tsv_path),
            "statistics": stats
        }
    
    def _run_mmseqs_command(self, cmd_args: List[str]) -> subprocess.CompletedProcess:
        """
        Run MMseqs2 command with error handling.
        
        Args:
            cmd_args: Command arguments (excluding binary path)
            
        Returns:
            Completed process result
            
        Raises:
            MMseqsError: If command fails
        """
        full_cmd = [self.config.bin_path] + cmd_args
        
        try:
            result = subprocess.run(
                full_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            if result.stderr:
                logger.debug(f"MMseqs2 stderr: {result.stderr}")
            
            return result
            
        except subprocess.CalledProcessError as e:
            error_msg = f"MMseqs2 command failed: {' '.join(full_cmd)}\n"
            error_msg += f"Return code: {e.returncode}\n"
            error_msg += f"Stdout: {e.stdout}\n"
            error_msg += f"Stderr: {e.stderr}"
            raise MMseqsError(error_msg)
    
    def _parse_clustering_results(self,
                                 input_fasta: Path,
                                 rep_seq_path: Path,
                                 cluster_tsv_path: Path) -> Dict[str, Any]:
        """
        Parse clustering results and generate statistics.
        
        Args:
            input_fasta: Original input file
            rep_seq_path: Representative sequences file
            cluster_tsv_path: Cluster membership file
            
        Returns:
            Dictionary with clustering statistics
        """
        try:
            # Count input sequences
            input_count = self._count_sequences_in_fasta(input_fasta)
            
            # Count representative sequences
            rep_count = self._count_sequences_in_fasta(rep_seq_path)
            
            # Parse cluster information
            cluster_info = self._parse_cluster_membership(cluster_tsv_path)
            
            # Calculate statistics
            stats = {
                "input_sequences": input_count,
                "representative_sequences": rep_count,
                "num_clusters": len(cluster_info),
                "reduction_ratio": rep_count / input_count if input_count > 0 else 0.0,
                "cluster_sizes": [len(members) for members in cluster_info.values()],
                "config": {
                    "coverage": self.config.coverage,
                    "min_seq_id": self.config.min_seq_id,
                    "cov_mode": self.config.cov_mode,
                    "cluster_mode": self.config.cluster_mode
                }
            }
            
            # Add size distribution statistics
            if stats["cluster_sizes"]:
                stats["cluster_size_stats"] = {
                    "min": min(stats["cluster_sizes"]),
                    "max": max(stats["cluster_sizes"]),
                    "mean": sum(stats["cluster_sizes"]) / len(stats["cluster_sizes"]),
                    "median": sorted(stats["cluster_sizes"])[len(stats["cluster_sizes"]) // 2]
                }
            
            logger.info(f"Clustering statistics: {input_count} -> {rep_count} sequences "
                       f"({stats['reduction_ratio']:.2%} reduction)")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error parsing clustering results: {e}")
            return {"error": str(e)}
    
    def _count_sequences_in_fasta(self, fasta_path: Path) -> int:
        """Count sequences in FASTA file."""
        try:
            count = 0
            with open(fasta_path, 'r') as f:
                for line in f:
                    if line.startswith('>'):
                        count += 1
            return count
        except Exception as e:
            logger.error(f"Error counting sequences in {fasta_path}: {e}")
            return 0
    
    def _parse_cluster_membership(self, cluster_tsv_path: Path) -> Dict[str, List[str]]:
        """
        Parse cluster membership file.
        
        Args:
            cluster_tsv_path: Path to cluster TSV file
            
        Returns:
            Dictionary mapping representative IDs to member lists
        """
        cluster_info = {}
        
        try:
            with open(cluster_tsv_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        rep_id = parts[0]
                        member_id = parts[1]
                        
                        if rep_id not in cluster_info:
                            cluster_info[rep_id] = []
                        cluster_info[rep_id].append(member_id)
            
        except Exception as e:
            logger.error(f"Error parsing cluster membership file: {e}")
        
        return cluster_info
    
    def convert_a3m_to_fasta(self, a3m_path: Path, fasta_path: Path) -> None:
        """
        Convert A3M file to FASTA format for MMseqs2.
        
        Args:
            a3m_path: Input A3M file
            fasta_path: Output FASTA file
            
        Raises:
            MMseqsError: If conversion fails
        """
        try:
            fasta_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(a3m_path, 'r') as infile, open(fasta_path, 'w') as outfile:
                for line in infile:
                    line = line.strip()
                    if not line:
                        continue
                    
                    if line.startswith('>'):
                        outfile.write(f"{line}\n")
                    else:
                        # Remove lowercase letters (insertions) and gaps for clustering
                        clean_seq = ''.join(char for char in line if char.isupper() and char != '-')
                        if clean_seq:  # Only write non-empty sequences
                            outfile.write(f"{clean_seq}\n")
            
            logger.info(f"Converted A3M to FASTA: {a3m_path} -> {fasta_path}")
            
        except Exception as e:
            raise MMseqsError(f"A3M to FASTA conversion failed: {e}")
    
    def get_cluster_representatives(self, cluster_results: Dict[str, Any]) -> List[str]:
        """
        Get list of representative sequence IDs from clustering results.
        
        Args:
            cluster_results: Results from cluster_sequences()
            
        Returns:
            List of representative sequence IDs
        """
        rep_seq_path = Path(cluster_results["representative_sequences"])
        
        rep_ids = []
        try:
            with open(rep_seq_path, 'r') as f:
                for line in f:
                    if line.startswith('>'):
                        # Extract ID (remove '>' and any description)
                        seq_id = line[1:].split()[0]
                        rep_ids.append(seq_id)
        except Exception as e:
            logger.error(f"Error reading representative sequences: {e}")
        
        return rep_ids