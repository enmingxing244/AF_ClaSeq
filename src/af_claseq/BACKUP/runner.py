"""
Main runner class for hit expand pipeline integration.

This module provides the HitExpandRunner class that integrates MSA pipeline
functionality into the AF-ClaSeq pipeline, replacing the iterative enrichment step.
"""

import os
import shutil
import subprocess
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import time
import json

from af_claseq.hit_expand.config import HitExpandConfig
from af_claseq.hit_expand.hit_expansion import HitExpandSampler
from af_claseq.hit_expand.analyzer import HitExpandAnalyzer
from af_claseq.hit_expand.output_manager import HitExpandOutputManager
from af_claseq.hit_expand.plotting import HitExpandPlotter
from af_claseq.utils.slurm_utils import SlurmJobSubmitter
from af_claseq.utils.structure_analysis import StructureAnalyzer
from af_claseq.utils.logging_utils import get_logger

logger = get_logger(__name__)


class HitExpandRunner:
    """Main runner class for hit expand stage integration."""
    
    def __init__(
        self,
        config: HitExpandConfig,
        slurm_submitter: SlurmJobSubmitter,
        base_dir: Path,
        config_file: str,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the HitExpandRunner.
        
        Args:
            config: Hit expand configuration
            slurm_submitter: SLURM job submitter for running jobs
            base_dir: Base directory for output
            config_file: Path to AF-ClaSeq config JSON file
            logger: Optional logger instance
        """
        self.config = config
        self.slurm_submitter = slurm_submitter
        self.base_dir = Path(base_dir)
        self.config_file = config_file
        self.logger = logger or get_logger(__name__)
        
        # Initialize components
        self.output_manager = HitExpandOutputManager(self.base_dir, self.logger)
        self.plotter = HitExpandPlotter(self.base_dir, self.logger)
        self.analyzer = HitExpandAnalyzer(self.config_file, self.logger)
        
        # Create output directories
        self.output_manager.create_directories()
        
        # Runtime state
        self.start_time = time.time()
        self.current_step = "initialized"
        self.completed_steps = []
        self.failed_steps = []
        
        self.logger.info(f"HitExpandRunner initialized with base directory: {self.base_dir}")
    
    def run(self) -> Optional[Path]:
        """
        Run the complete hit expand stage.
        
        Returns:
            Path to final output MSA file, or None if failed
        """
        try:
            self.start_time = time.time()
            self.logger.info("=== STARTING HIT EXPAND STAGE ===")
            
            # Step 1: Clustering step (if not skipped)
            if not self.config.skip_clustering:
                clustered_msa = self._run_clustering_step()
                if not clustered_msa:
                    self.logger.error("Clustering step failed")
                    return None
            else:
                self.logger.info("Skipping clustering step - using input MSA directly")
                clustered_msa = Path(self.config.input_msa)
            
            # Step 2: Hit expansion workflow
            expanded_msa = self._run_expansion_workflow(clustered_msa)
            if not expanded_msa:
                self.logger.error("Hit expansion workflow failed")
                return None
            
            # Step 3: Analysis step (if not skipped)
            analysis_results = None
            if not self.config.skip_structure_analysis:
                analysis_results = self._run_analysis_step(expanded_msa)
            
            # Step 4: Generate plots
            self._generate_plots(expanded_msa, analysis_results)
            
            # Step 5: Create final output
            final_output = self._finalize_output(expanded_msa)
            
            total_time = time.time() - self.start_time
            self.logger.info(f"=== HIT EXPAND STAGE COMPLETED in {total_time:.1f}s ===")
            self.logger.info(f"Final output: {final_output}")
            
            return final_output
            
        except Exception as e:
            self.logger.error(f"Hit expand stage failed: {str(e)}", exc_info=True)
            self.failed_steps.append(self.current_step)
            return None
    
    def _run_clustering_step(self) -> Optional[Path]:
        """
        Run MMseqs2 clustering on the input MSA.
        
        Returns:
            Path to clustered representative sequences file
        """
        self.current_step = "clustering"
        self.logger.info("Step 1: Running clustering step")
        
        try:
            # Setup paths
            input_msa = Path(self.config.input_msa)
            if not input_msa.exists():
                raise FileNotFoundError(f"Input MSA file not found: {input_msa}")
            
            clustering_dir = self.base_dir / "00_clustering"
            clustering_dir.mkdir(exist_ok=True)
            
            # Run MMseqs2 clustering
            cluster_prefix = clustering_dir / "cluster_result"
            tmp_dir = clustering_dir / "tmp"
            tmp_dir.mkdir(exist_ok=True)
            
            mmseqs_cmd = [
                self.config.mmseqs_bin,
                "easy-cluster",
                str(input_msa),
                str(cluster_prefix),
                str(tmp_dir),
                "-c", str(self.config.mmseqs_coverage),
                "--min-seq-id", str(self.config.mmseqs_min_seq_id),
                "--cov-mode", str(self.config.mmseqs_cov_mode),
                "--cluster-mode", str(self.config.mmseqs_cluster_mode),
                "--threads", str(self.config.mmseqs_threads),
                "-v", "3"
            ]
            
            self.logger.info(f"Running MMseqs2 command: {' '.join(mmseqs_cmd)}")
            
            result = subprocess.run(
                mmseqs_cmd,
                capture_output=True,
                text=True,
                check=True,
                cwd=clustering_dir
            )
            
            self.logger.info("MMseqs2 clustering completed successfully")
            
            # Expected output file
            rep_seq_file = Path(f"{str(cluster_prefix)}_rep_seq.fasta")
            if not rep_seq_file.exists():
                raise FileNotFoundError(f"MMseqs2 issue - Expected output file not found: {rep_seq_file}")
            
            # Convert back to A3M format
            clustered_a3m = clustering_dir / "clustered_representatives.a3m"
            self._convert_fasta_to_a3m(rep_seq_file, clustered_a3m)
            
            self.completed_steps.append("clustering")
            return clustered_a3m
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"MMseqs2 clustering failed: {e}")
            self.logger.error(f"MMseqs2 stderr: {e.stderr}")
            return None
        except Exception as e:
            self.logger.error(f"Error in MMseqs2 clustering: {str(e)}", exc_info=True)
            return None
    
    def _run_expansion_workflow(self, clustered_msa: Path) -> Optional[Path]:
        """
        Run the hit expansion workflow on clustered sequences.
        
        Args:
            clustered_msa: Path to clustered representative sequences
            
        Returns:
            Path to final expanded MSA output
        """
        self.current_step = "expansion_workflow"
        self.logger.info("Step 2: Running hit expansion workflow")
        
        try:
            # Create hit expand sampler
            sampler = HitExpandSampler(
                input_msa=str(clustered_msa),
                base_dir=str(self.base_dir),
                config=self.config,
                slurm_submitter=self.slurm_submitter,
                logger=self.logger
            )
            
            # Run expansion workflow
            expanded_msa = sampler.run()
            
            if expanded_msa:
                self.completed_steps.append("expansion_workflow")
                return expanded_msa
            else:
                self.logger.error("Hit expansion workflow returned no output")
                return None
                
        except Exception as e:
            self.logger.error(f"Error in hit expansion workflow: {str(e)}", exc_info=True)
            return None
    
    def _run_analysis_step(self, expanded_msa: Path) -> Optional[Path]:
        """
        Run structure analysis step.
        
        Args:
            expanded_msa: Path to expanded MSA
            
        Returns:
            Path to analysis results file
        """
        self.current_step = "structure_analysis"
        self.logger.info("Step 3: Running structure analysis")
        
        try:
            # Find structure prediction results
            structures_dir = self.base_dir / "01_expansion" / "batches"
            if not structures_dir.exists():
                self.logger.warning("No structures directory found for analysis")
                return None
            
            # Analyze structures
            df_results = self.analyzer.analyze_structures(
                structures_dir=structures_dir,
                plddt_threshold=self.config.plddt_threshold
            )
            
            if df_results.empty:
                self.logger.warning("No structures analyzed")
                return None
            
            # Apply filtering criteria
            df_filtered = self.analyzer.filter_by_criteria(
                df=df_results,
                filter_threshold=self.config.filter_criteria_threshold
            )
            
            # Save analysis results
            analysis_dir = self.base_dir / "02_analysis"
            analysis_dir.mkdir(exist_ok=True)
            
            results_file = analysis_dir / "structure_analysis_results.csv"
            self.analyzer.save_results(df_filtered, results_file)
            
            self.completed_steps.append("structure_analysis")
            return results_file
            
        except Exception as e:
            self.logger.error(f"Error in structure analysis: {str(e)}", exc_info=True)
            return None
    
    def _generate_plots(self, expanded_msa: Path, analysis_results: Optional[Path] = None) -> None:
        """
        Generate plots for the hit expand results.
        
        Args:
            expanded_msa: Path to expanded MSA output
            analysis_results: Optional path to analysis results
        """
        self.current_step = "plot_generation"
        self.logger.info("Step 4: Generating plots")
        
        try:
            # Create plots directory
            plots_dir = self.base_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            # Generate plots using the plotter
            self.plotter.create_hit_expand_plots(
                msa_output=expanded_msa,
                config_file=self.config_file,
                plots_dir=plots_dir
            )
            
            self.completed_steps.append("plot_generation")
            
        except Exception as e:
            self.logger.error(f"Error generating plots: {str(e)}", exc_info=True)
    
    def _finalize_output(self, expanded_msa: Path) -> Path:
        """
        Finalize the hit expand output and create summary.
        
        Args:
            expanded_msa: Path to expanded MSA output
            
        Returns:
            Path to final output file
        """
        self.current_step = "finalization"
        self.logger.info("Step 5: Finalizing output")
        
        try:
            # Create final output file
            final_output = self.base_dir / "hit_expand_final_msa.a3m"
            
            # Copy expanded MSA to final location
            if expanded_msa != final_output:
                shutil.copy2(expanded_msa, final_output)
            
            # Create pipeline summary
            summary = self._create_pipeline_summary()
            summary_file = self.base_dir / "hit_expand_summary.json"
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            self.completed_steps.append("finalization")
            return final_output
            
        except Exception as e:
            self.logger.error(f"Error finalizing output: {str(e)}", exc_info=True)
            return expanded_msa
    
    def _create_pipeline_summary(self) -> Dict[str, Any]:
        """Create hit expand stage execution summary."""
        total_time = time.time() - self.start_time
        
        return {
            "stage_name": "hit_expand",
            "execution_time_seconds": total_time,
            "completed_steps": self.completed_steps,
            "failed_steps": self.failed_steps,
            "configuration": {
                "mmseqs_coverage": self.config.mmseqs_coverage,
                "mmseqs_min_seq_id": self.config.mmseqs_min_seq_id,
                "num_subsets": self.config.num_subsets,
                "num_batches": self.config.num_batches,
                "plddt_threshold": self.config.plddt_threshold,
                "similarity_threshold": self.config.similarity_threshold,
            },
            "output_files": {
                "final_msa": "hit_expand_final_msa.a3m",
                "clustering_results": "00_clustering/",
                "expansion_output": "01_expansion/",
                "analysis_results": "02_analysis/",
                "plots": "plots/",
                "summary": "hit_expand_summary.json"
            }
        }
    
    
    def _convert_fasta_to_a3m(self, fasta_file: Path, a3m_file: Path) -> None:
        """Convert FASTA format to A3M format."""
        # For simplicity, this is a basic conversion
        # A more sophisticated conversion might be needed for full A3M compliance
        shutil.copy2(fasta_file, a3m_file)
    
    def setup_logging(self) -> None:
        """Setup logging for the hit expand runner."""
        log_file = self.base_dir / "hit_expand.log"
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
        
        self.logger.info("Hit expand logging initialized")
    
    def log_parameters(self) -> None:
        """Log the configuration parameters."""
        self.logger.info("=== HIT EXPAND CONFIGURATION ===")
        self.logger.info(f"Input MSA: {self.config.input_msa}")
        self.logger.info(f"Base directory: {self.base_dir}")
        self.logger.info(f"MMseqs2 binary: {self.config.mmseqs_bin}")
        self.logger.info(f"MMseqs2 coverage: {self.config.mmseqs_coverage}")
        self.logger.info(f"MMseqs2 min seq ID: {self.config.mmseqs_min_seq_id}")
        self.logger.info(f"Number of subsets: {self.config.num_subsets}")
        self.logger.info(f"Number of batches: {self.config.num_batches}")
        self.logger.info(f"pLDDT threshold: {self.config.plddt_threshold}")
        self.logger.info(f"Similarity threshold: {self.config.similarity_threshold}")
        self.logger.info(f"Skip clustering: {self.config.skip_clustering}")
        self.logger.info(f"Skip structure prediction: {self.config.skip_structure_prediction}")
        self.logger.info(f"Skip structure analysis: {self.config.skip_structure_analysis}")
        self.logger.info(f"Skip hit expansion: {self.config.skip_hit_expansion}")
        self.logger.info("=== END CONFIGURATION ===")