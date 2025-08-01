#!/usr/bin/env python3
"""
Hit Expand Pipeline Orchestrator.

This module orchestrates the complete hit expand workflow:
1. Sequence clustering with MMseqs2
2. Subset generation for structure prediction
3. Structure prediction job submission and monitoring
4. Structure analysis
5. Similarity search with BLOSUM62 to expand good sequences
6. Expanded subset generation from final MSA
"""

import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
import shutil
import pandas as pd
from tqdm import tqdm

from af_claseq.utils.logging_utils import get_logger
from af_claseq.utils.slurm_utils import SlurmJobSubmitter
from af_claseq.utils.structure_analysis import StructureAnalyzer
from af_claseq.utils.sequence_processing import A3MParser, validate_a3m_file
from af_claseq.utils.plotting_manager import (
    plot_1d_distribution, 
    create_2d_scatter_plot,
    create_joint_plot
)

from af_claseq.modules.mmseqs_wrapper import MMseqsWrapper, MMseqsConfig
from af_claseq.modules.similarity_search import BLOSUM62SimilaritySearch, SimilaritySearchConfig
from af_claseq.modules.cluster_based_expansion import ClusterBasedExpansion, ClusterExpansionError
from af_claseq.modules.subset_generator import SubsetGenerator, SubsetConfig

from af_claseq.pipeline.config import HitExpandConfig

logger = get_logger(__name__)


class HitExpandError(Exception):
    """Raised when hit expand pipeline fails."""
    pass


class HitExpandRunner:
    """
    Main orchestrator for the hit expand pipeline.
    
    Coordinates all components of the hit expand workflow to transform
    an input MSA into expanded sequences ready for structure prediction.
    """
    
    def __init__(self,
                 config: HitExpandConfig,
                 slurm_submitter: SlurmJobSubmitter,
                 base_dir: Path,
                 config_file: str,
                 general_config: Optional[Any] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize hit expand runner.
        
        Args:
            config: Hit expand configuration
            slurm_submitter: SLURM job submitter instance
            base_dir: Base output directory
            config_file: Path to JSON config file for structure analysis
            general_config: General configuration for explicit metric selection
            logger: Optional logger instance
        """
        self.config = config
        self.slurm_submitter = slurm_submitter
        self.base_dir = Path(base_dir)
        self.config_file = config_file
        self.general_config = general_config
        self.logger = logger or get_logger(__name__)
        
        # Create base directory
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._init_components()
        
        # Track workflow state
        self.workflow_state = {
            "clustering_completed": False,
            "similarity_search_completed": False,
            "subset_generation_completed": False,
            "structure_prediction_completed": False,
            "structure_analysis_completed": False,
            "expanded_subset_generation_completed": False
        }
        
        self.logger.info(f"Hit expand runner initialized for {base_dir}")
    
    def _init_components(self):
        """Initialize pipeline components."""
        # MMseqs2 wrapper
        mmseqs_config = MMseqsConfig(
            bin_path=self.config.mmseqs_bin,
            coverage=self.config.mmseqs_coverage,
            min_seq_id=self.config.mmseqs_min_seq_id,
            cov_mode=self.config.mmseqs_cov_mode,
            cluster_mode=self.config.mmseqs_cluster_mode,
            threads=self.config.mmseqs_threads,
            tmp_dir=self.config.mmseqs_tmp_dir
        )
        self.mmseqs_wrapper = MMseqsWrapper(mmseqs_config)
        
        # Similarity search
        similarity_config = SimilaritySearchConfig(
            top_k=self.config.similarity_top_k,
            similarity_threshold=self.config.similarity_threshold,
            exclude_query_headers=self.config.exclude_query_headers
        )
        self.similarity_search = BLOSUM62SimilaritySearch(similarity_config)
        
        # Subset generator
        subset_config = SubsetConfig(
            num_subsets=self.config.num_subsets,
            num_random_sequences=self.config.num_random_sequences,
            num_batches=self.config.num_batches,
            batch_prefix=self.config.batch_prefix,
            output_prefix=self.config.output_prefix,
            random_seed=self.config.random_seed
        )
        self.subset_generator = SubsetGenerator(subset_config)
        
        # Structure analyzer
        self.structure_analyzer = StructureAnalyzer()
        
        # A3M parser for sequence processing
        from af_claseq.utils.sequence_processing import A3MParser
        self.parser = A3MParser(strict_validation=False)
        
        self.logger.info("All pipeline components initialized")
    
    def run(self) -> Optional[Path]:
        """
        Run the complete multi-round hit expand pipeline.
        
        Returns:
            Path to final MSA file if successful, None otherwise
        """
        try:
            self.logger.info("=== STARTING MULTI-ROUND HIT EXPAND PIPELINE ===")
            self.logger.info(f"Running {self.config.rounds} rounds of hit expansion")
            
            # Validate input MSA
            input_msa = Path(self.config.input_msa)
            if not validate_a3m_file(input_msa, strict=False):
                raise HitExpandError(f"Invalid input MSA: {input_msa}")
            
            # Stage 1: Clustering (done once, shared across all rounds)
            clustering_dir = self.base_dir / "01_clustering"
            if not self.config.skip_clustering:
                if self._check_done_file(clustering_dir, "01_clustering"):
                    self.logger.info("=== STAGE 1: CLUSTERING ALREADY COMPLETED - SKIPPING ===")
                    representative_sequences = self._load_clustered_sequences(clustering_dir)
                else:
                    representative_sequences = self._run_clustering_stage(input_msa)
            else:
                self.logger.info("Skipping clustering stage")
                parser = A3MParser(strict_validation=False)
                representative_sequences = parser.parse_file(input_msa)
            
            # Run multiple rounds
            final_msa_path = None
            for round_num in range(1, self.config.rounds + 1):
                self.logger.info(f"=== STARTING HIT EXPAND ROUND {round_num}/{self.config.rounds} ===")
                
                round_dir = self.base_dir / f"round_{round_num}"
                round_dir.mkdir(exist_ok=True)
                
                # Determine input for this round
                if round_num == 1:
                    # Round 1: Use clustered sequences
                    round_input_sequences = representative_sequences
                    round_input_type = "clustered"
                else:
                    # Round 2+: Use previous round's final MSA
                    prev_round_dir = self.base_dir / f"round_{round_num - 1}"
                    prev_final_msa = prev_round_dir / f"hit_expand_final_msa_round_{round_num - 1}.a3m"
                    
                    if not prev_final_msa.exists():
                        self.logger.warning(f"No final MSA found from round {round_num - 1}: {prev_final_msa}. Ending at round {round_num - 1}")
                        break
                    
                    # Load sequences from previous round's final MSA
                    from af_claseq.utils.sequence_processing import A3MParser
                    parser = A3MParser(strict_validation=False)
                    round_input_sequences = parser.parse_file(prev_final_msa)
                    round_input_type = "final_msa"
                
                # Run the round
                round_msa = self._run_single_round(
                    round_num=round_num,
                    round_dir=round_dir,
                    input_sequences=round_input_sequences,
                    input_type=round_input_type
                )
                
                if round_msa:
                    final_msa_path = round_msa
                    
                    # Check if we found new sequences in this round
                    if round_num > 1:
                        # Inline _check_new_sequences_found
                        search_dir = round_dir / "05_similarity_search"
                        new_sequences_found = False
                        if search_dir.exists():
                            a3m_files = list(search_dir.glob("*.a3m"))
                            if a3m_files:
                                total_sequences = 0
                                for a3m_file in a3m_files:
                                    try:
                                        parser = A3MParser(strict_validation=False)
                                        sequences = parser.parse_file(a3m_file)
                                        total_sequences += len(sequences)
                                    except Exception as e:
                                        self.logger.debug(f"Failed to parse {a3m_file}: {e}")
                                new_sequences_found = total_sequences > 0
                        
                        if not new_sequences_found:
                            self.logger.info(f"No new sequences found in round {round_num}. Stopping early.")
                            break
                else:
                    self.logger.error(f"Round {round_num} failed")
                    break
            
            # Run final structure analysis on the last round's expanded subsets
            if final_msa_path:
                # Determine the final round number (the round that actually completed)
                final_round_num = self.config.rounds
                for r in range(self.config.rounds, 0, -1):
                    round_dir = self.base_dir / f"round_{r}"
                    if (round_dir / "06_expanded_subsets").exists():
                        final_round_num = r
                        break
                
                self.logger.info(f"Running final structure analysis on round {final_round_num}'s expanded subsets")
                self._run_final_structure_analysis(final_round_num)
            
            self.logger.info(f"=== MULTI-ROUND HIT EXPAND PIPELINE COMPLETED ===")
            self.logger.info(f"Final MSA: {final_msa_path}")
            
            return final_msa_path
            
        except Exception as e:
            self.logger.error(f"Multi-round hit expand pipeline failed: {e}")
            raise HitExpandError(f"Pipeline failed: {e}")
    
    def _run_clustering_stage(self, input_msa: Path) -> Dict[str, str]:
        """Run MMseqs2 clustering stage."""
        self.logger.info("=== STAGE 1: SEQUENCE CLUSTERING ===")
        
        clustering_dir = self.base_dir / "01_clustering"
        clustering_dir.mkdir(exist_ok=True)
        
        # Run clustering directly on A3M file (MMseqs2 easy-cluster handles A3M)
        cluster_results = self.mmseqs_wrapper.cluster_sequences(
            input_file=input_msa,
            output_dir=clustering_dir,
            prefix="clustered"
        )
        
        # Load representative sequences from the _rep_seq.fasta output
        rep_fasta = Path(cluster_results["representative_sequences"])
        clustered_sequences = {}
        
        with open(rep_fasta, 'r') as f:
            current_header = None
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    current_header = line
                elif current_header and line:
                    clustered_sequences[current_header] = line
                    current_header = None
        
        self.workflow_state["clustering_completed"] = True
        self.logger.info(f"Clustering completed: {len(clustered_sequences)} representative sequences")
        
        # Create DONE file to mark completion
        self._create_done_file(clustering_dir, "01_clustering")
        
        return clustered_sequences
    
    
    
    
    def _run_expanded_subset_plotting(self, expanded_subsets_dir: Path, subset_results: Dict[str, Any]):
        """Run plotting for expanded subset structure analysis results."""
        self.logger.info("=== EXPANDED SUBSET PLOTTING ===")
        
        # Create plots directory within expanded subsets
        plots_dir = expanded_subsets_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Find all PDB files in the expanded subsets directory
        pdb_files = list(expanded_subsets_dir.rglob("*.pdb"))
        
        if not pdb_files:
            self.logger.warning("No PDB files found for expanded subset plotting")
            return
        
        self.logger.info(f"Found {len(pdb_files)} PDB files for analysis")
        
        # Load filter configuration for metrics to calculate
        filter_config_path = Path(self.config_file)
        if filter_config_path.exists():
            with open(filter_config_path, 'r') as f:
                filter_config = json.load(f)
        else:
            self.logger.warning(f"Filter config not found: {filter_config_path}")
            return
        
        # Run structure analysis on all PDB files
        analysis_results = {}
        
        # Get analysis parameters (same as main structure analysis)
       
        all_filter_criteria = filter_config.get("filter_criteria", [])
        composite_metrics = filter_config.get("composite_metrics", [])
        
        # Temporarily suppress INFO logging from sequence_processing module
        seq_processing_logger = logging.getLogger('af_claseq.sequence_processing')
        original_seq_level = seq_processing_logger.level
        seq_processing_logger.setLevel(logging.WARNING)
        
        # Create progress bar for structure analysis
        pbar = tqdm(pdb_files, desc="Analyzing expanded structures", unit="structure")
        
        for pdb_file in pbar:
            pbar.set_postfix({"current": pdb_file.name})
            
            try:
                # Run structure analysis using the same method as main analysis
                metrics = self.structure_analyzer.process_single_pdb(
                    pdb_path=str(pdb_file),
                    filter_criteria=all_filter_criteria,
                    basics=filter_config.get("basics", {}),
                    plddt_threshold=0.0, # plddt threshold will not be applied in the plotting steps
                    composite_metrics=composite_metrics
                )
                
                if metrics and "error" not in metrics:
                    analysis_results[str(pdb_file)] = metrics
                    
            except Exception as e:
                self.logger.debug(f"Error analyzing {pdb_file}: {e}")
        
        # Close progress bar
        pbar.close()
        
        # Restore original logging level
        seq_processing_logger.setLevel(original_seq_level)
        
        self.logger.info(f"Successfully analyzed {len(analysis_results)} expanded structures")
        
        # Save results as CSV and create plots (no filtering needed)
        all_filter_criteria = filter_config.get("filter_criteria", [])
        csv_file, plot_files = self._save_results_and_create_plots(
            analysis_results, 
            analysis_results,  # Use same results for both (no filtering)
            expanded_subsets_dir, 
            all_filter_criteria
        )
        
        self.logger.info(f"Saved expanded subset analysis to: {csv_file}")
        self.logger.info(f"Created {len(plot_files)} plots in {plots_dir}")
    
    
    def _run_generic_structure_prediction(self, 
                                         subset_results: Dict[str, Any],
                                         base_dir: Path,
                                         job_prefix: str = "hit_expand",
                                         stage_name: str = "structure prediction") -> Dict[str, Any]:
        """Generic structure prediction logic that can be reused for different stages."""
        # Create job specifications for batch directories
        job_specs = []
        for batch_name, subset_paths in subset_results["batch_info"].items():
            batch_dir = base_dir / batch_name
            job_spec = {
                "name": f"{job_prefix}_{batch_name}",
                "task_dir": str(batch_dir),
                "gres": "gpu:1",
                "num_subsets": len(subset_paths)
            }
            job_specs.append(job_spec)
        
        self.logger.info(f"Created {len(job_specs)} {stage_name} job specifications")
        
        # Submit and monitor jobs using the generic function
        submitted_jobs = self._submit_and_monitor_structure_jobs(
            job_specs,
            job_prefix=job_prefix,
            stage_name=stage_name
        )
        
        return {
            "job_specs": job_specs,
            "submitted_jobs": submitted_jobs,
            "prediction_dir": base_dir
        }
    
    def _submit_and_monitor_structure_jobs(self, 
                                          job_specs: List[Dict[str, Any]], 
                                          job_prefix: str = "hit_expand",
                                          stage_name: str = "structure prediction") -> List[str]:
        """Generic function to submit and monitor structure prediction jobs."""
        submitted_jobs = []
        
        # Temporarily suppress INFO logging from slurm_utils module
        slurm_logger = logging.getLogger('af_claseq.slurm_utils')
        original_slurm_level = slurm_logger.level
        slurm_logger.setLevel(logging.WARNING)
        
        # Create progress bar for job submission
        pbar = tqdm(job_specs, desc=f"Submitting {stage_name} jobs", unit="batch")
        
        for job_spec in pbar:
            batch_dir = job_spec["task_dir"]
            job_name = job_spec["name"]
            
            # Update progress bar description
            pbar.set_postfix({"current": job_name})
            
            # Check for existing results if requested
            if self.config.check_existing_jobs:
                task_dir = Path(batch_dir)
                existing_pdbs = list(task_dir.rglob("*.pdb"))
                if existing_pdbs:
                    self.logger.info(f"Existing results found in {task_dir}, skipping job submission")
                    continue
            
            # Extract batch identifier from job name for job_id
            batch_id = job_name.replace(f"{job_prefix}_", "")
            
            # Submit using the existing submit_job method
            job_id = self.slurm_submitter.submit_job(
                task_dir=batch_dir,
                job_id=batch_id
            )
            
            if job_id:
                submitted_jobs.append(job_id)
            else:
                self.logger.error(f"Failed to submit job for {batch_dir}")
        
        # Close progress bar
        pbar.close()
        
        # Restore original logging level
        slurm_logger.setLevel(original_slurm_level)
        
        self.logger.info(f"Submitted {len(submitted_jobs)}/{len(job_specs)} {stage_name} jobs")
        
        # Monitor jobs if requested
        if self.config.monitor_jobs and submitted_jobs:
            self.logger.info(f"Monitoring {len(submitted_jobs)} {stage_name} jobs")
            job_states = self.slurm_submitter.monitor_jobs(
                job_ids=submitted_jobs,
                check_interval=self.config.job_check_interval,
                timeout=self.config.job_timeout
            )
            
            # Log job completion statistics
            completed_jobs = sum(1 for state in job_states.values() if state.value == "COMPLETED")
            self.logger.info(f"{stage_name} completed: {completed_jobs}/{len(submitted_jobs)} jobs successful")
        
        return submitted_jobs
    
    
    
    def _extract_good_sequences(self, 
                               analysis_results: Dict[str, Any], 
                               subset_results: Dict[str, Any]) -> Dict[str, str]:
        """
        Extract sequences from subsets that correspond to good structures.
        
        Args:
            analysis_results: Results from structure analysis
            subset_results: Results from subset generation
            
        Returns:
            Dictionary of good sequences
        """
        self.logger.info("=== EXTRACTING GOOD SEQUENCES ===")
        
        filtered_results = analysis_results["filtered_results"]
        good_sequences = {}
        
        if not filtered_results:
            self.logger.warning("No structures passed filtering criteria - using representative sequences")
            return subset_results.get("representative_sequences", {})
        
        # Extract sequences only from subsets that produced good structures
        parser = A3MParser(strict_validation=False)
        
        for pdb_path_str in filtered_results.keys():
            try:
                pdb_path = Path(pdb_path_str)
                
                # Construct A3M file path from PDB file path
                # PDB files are named like: subset_000001_unrelaxed_rank_001_...
                # A3M files are named like: subset_000001.a3m and stored in batch folder
                pdb_name = pdb_path.name
                a3m_name = pdb_name.split('_unrelaxed')[0] + '.a3m'
                
                # A3M file is in the same directory as the PDB file (batch folder)
                a3m_path = pdb_path.parent / a3m_name
                
                if a3m_path.exists():
                    # Parse sequences from this A3M file
                    sequences = parser.parse_file(a3m_path)
                    
                    # Add all sequences from this subset to good sequences (excluding query sequences)
                    for header, sequence in sequences.items():
                        if header not in good_sequences:  # Avoid duplicates
                            # Check if this is a query sequence and exclude it
                            header_clean = header.lower().strip()
                            is_query = False
                            
                            # Check for common query indicators
                            query_indicators = ['query', 'target', 'template', 'reference']
                            if any(indicator in header_clean for indicator in query_indicators):
                                is_query = True
                            
                            # Check for sequence ID patterns that might indicate query
                            if header_clean.startswith('>101') or header_clean.startswith('101') or header_clean.startswith('>query'):
                                is_query = True
                            
                            if not is_query:
                                good_sequences[header] = sequence
                            else:
                                self.logger.debug(f"Excluded query sequence: {header}")
                    
                    self.logger.debug(f"Added {len(sequences)} sequences from {a3m_name}")
                else:
                    self.logger.warning(f"A3M file not found for structure {pdb_path}: {a3m_path}")
                    
            except Exception as e:
                self.logger.warning(f"Error processing structure path {pdb_path_str}: {e}")
        
        if not good_sequences:
            self.logger.warning("No good sequences extracted - using representative sequences")
            good_sequences = subset_results.get("representative_sequences", {})
        
        self.logger.info(f"Extracted {len(good_sequences)} good sequences from {len(filtered_results)} good structures")
        
        return good_sequences
    
    def _save_results_and_create_plots(self,
                                      analysis_results: Dict[str, Any],
                                      filtered_results: Dict[str, Any],
                                      output_dir: Path,
                                      filter_criteria: List[Dict[str, Any]]) -> tuple:
        """
        Save analysis results to CSV and create plots.
        
        Args:
            analysis_results: All structure analysis results
            filtered_results: Filtered analysis results
            output_dir: Directory to save outputs
            filter_criteria: List of filter criteria used
            
        Returns:
            Tuple of (csv_file_path, list_of_plot_paths)
        """
        try:
            # Convert results to DataFrame
            all_data = []
            for pdb_path, metrics in analysis_results.items():
                if "error" not in metrics:
                    # Keep full path in PDB column for easier filtering
                    row = {"PDB": pdb_path}
                    # Add metrics but skip the duplicate PDB field
                    for key, value in metrics.items():
                        if key != "PDB":  # Skip duplicate PDB field from metrics
                            row[key] = value
                    all_data.append(row)
            
            if not all_data:
                self.logger.warning("No valid data for CSV/plotting")
                return None, []
            
            df = pd.DataFrame(all_data)
            
            
            # Save to CSV
            csv_file = output_dir / "structure_analysis_results.csv"
            df.to_csv(csv_file, index=False)
            self.logger.info(f"Saved analysis results to CSV: {csv_file}")
            
            # Create plots directory
            plots_dir = output_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            plot_files = []
            
            # Get selected metrics using explicit metric selection system
            if self.general_config:
                from af_claseq.pipeline.config import get_selected_metrics
                selected_metrics = get_selected_metrics(self.general_config)
                self.logger.info(f"Using explicitly selected metrics: {selected_metrics}")
            else:
                # Fall back to extracting metric names from filter criteria
                selected_metrics = [criterion.get('name') for criterion in filter_criteria if criterion.get('name')]
                self.logger.info(f"Using filter criteria metrics (fallback): {selected_metrics}")
            
            # Check which metrics are actually available in the DataFrame
            available_metrics = [m for m in selected_metrics if m in df.columns]
            missing_metrics = [m for m in selected_metrics if m not in df.columns]
            
            if missing_metrics:
                self.logger.warning(f"Metrics missing from DataFrame: {missing_metrics}")
            self.logger.info(f"Available metrics for plotting: {available_metrics}")
            
            # Extract plotting configuration from config
            plotting_config = {
                'scatter_plot_metric1_min': self.config.scatter_plot_metric1_min,
                'scatter_plot_metric1_max': self.config.scatter_plot_metric1_max,
                'scatter_plot_metric2_min': self.config.scatter_plot_metric2_min,
                'scatter_plot_metric2_max': self.config.scatter_plot_metric2_max,
                'scatter_plot_metric1_ticks': self.config.scatter_plot_metric1_ticks,
                'scatter_plot_metric2_ticks': self.config.scatter_plot_metric2_ticks,
                'plddt_plot_min': self.config.plddt_plot_min,
                'plddt_plot_max': self.config.plddt_plot_max,
                'plddt_plot_ticks': self.config.plddt_plot_ticks
            }
            
            # Use available metrics instead of expected metrics
            metric_names = available_metrics
            
            # Create 1D distribution plots for each metric
            for metric_name in metric_names:
                if metric_name in df.columns:
                    try:
                        plot_path = plot_1d_distribution(
                            results_df=df,
                            metric_name=metric_name,
                            output_dir=str(plots_dir),
                            n_plot_bins=20,
                            logger=self.logger
                        )
                        if plot_path:
                            plot_files.append(plot_path)
                    except Exception as e:
                        self.logger.warning(f"Failed to create 1D plot for {metric_name}: {e}")
            
            # Also create 1D plots for pLDDT metrics
            for plddt_metric in ['plddt', 'local_plddt']:
                if plddt_metric in df.columns:
                    try:
                        plot_path = plot_1d_distribution(
                            results_df=df,
                            metric_name=plddt_metric,
                            output_dir=str(plots_dir),
                            n_plot_bins=20,
                            logger=self.logger
                        )
                        if plot_path:
                            plot_files.append(plot_path)
                    except Exception as e:
                        self.logger.warning(f"Failed to create 1D plot for {plddt_metric}: {e}")
            
            # Create 2D scatter plots for metric combinations
            if len(metric_names) >= 2 and all(m in df.columns for m in metric_names[:2]):
                try:
                    # Determine threshold values for dashed lines (ONLY filter_criteria, NOT pLDDT)
                    threshold_x = None
                    threshold_y = None
                    
                    if metric_names[0] == self.config.filter_criteria:
                        threshold_x = self.config.filter_criteria_threshold
                    
                    if metric_names[1] == self.config.filter_criteria:
                        threshold_y = self.config.filter_criteria_threshold
                    
                    # Create scatter plot colored by pLDDT using the standard plotting style
                    plot_path = create_2d_scatter_plot(
                        results_df=df,
                        metric_name1=metric_names[0],
                        metric_name2=metric_names[1],
                        output_dir=str(plots_dir),
                        color_metric='plddt',
                        title=None,
                        x_min=plotting_config['scatter_plot_metric1_min'],
                        x_max=plotting_config['scatter_plot_metric1_max'],
                        y_min=plotting_config['scatter_plot_metric2_min'],
                        y_max=plotting_config['scatter_plot_metric2_max'],
                        x_ticks=plotting_config['scatter_plot_metric1_ticks'],
                        y_ticks=plotting_config['scatter_plot_metric2_ticks'],
                        threshold_x=threshold_x,
                        threshold_y=threshold_y,
                        logger=self.logger
                    )
                    if plot_path:
                        plot_files.append(plot_path)
                    
                    # Create joint plot with marginal distributions
                    plot_path = create_joint_plot(
                        results_df=df,
                        metric_name1=metric_names[0],
                        metric_name2=metric_names[1],
                        output_dir=str(plots_dir),
                        color_metric='plddt',
                        x_min=plotting_config['scatter_plot_metric1_min'],
                        x_max=plotting_config['scatter_plot_metric1_max'],
                        y_min=plotting_config['scatter_plot_metric2_min'],
                        y_max=plotting_config['scatter_plot_metric2_max'],
                        logger=self.logger
                    )
                    if plot_path:
                        plot_files.append(plot_path)
                        
                    # Create local_plddt colored version if available
                    if 'local_plddt' in df.columns:
                        plot_path = create_2d_scatter_plot(
                            results_df=df,
                            metric_name1=metric_names[0],
                            metric_name2=metric_names[1],
                            output_dir=str(plots_dir),
                            color_metric='local_plddt',
                            title=None,
                            x_min=plotting_config['scatter_plot_metric1_min'],
                            x_max=plotting_config['scatter_plot_metric1_max'],
                            y_min=plotting_config['scatter_plot_metric2_min'],
                            y_max=plotting_config['scatter_plot_metric2_max'],
                            x_ticks=plotting_config['scatter_plot_metric1_ticks'],
                            y_ticks=plotting_config['scatter_plot_metric2_ticks'],
                            threshold_x=threshold_x,
                            threshold_y=threshold_y,
                            logger=self.logger
                        )
                        if plot_path:
                            plot_files.append(plot_path)
                            
                except Exception as e:
                    self.logger.warning(f"Failed to create 2D plots: {e}")
            
            # Create pLDDT vs first metric scatter plot if available
            if metric_names and metric_names[0] in df.columns:
                try:
                    # Determine threshold values for dashed lines (ONLY filter_criteria, NOT pLDDT)
                    threshold_x = None  # plddt is x-axis, never show pLDDT threshold line
                    threshold_y = None
                    
                    # Only check y-axis (metric_names[0]) for filter_criteria threshold
                    if metric_names[0] == self.config.filter_criteria:
                        threshold_y = self.config.filter_criteria_threshold
                    
                    plot_path = create_2d_scatter_plot(
                        results_df=df,
                        metric_name1='plddt',
                        metric_name2=metric_names[0],
                        output_dir=str(plots_dir),
                        color_metric='plddt',
                        title=None,
                        x_min=plotting_config['plddt_plot_min'],
                        x_max=plotting_config['plddt_plot_max'],
                        y_min=plotting_config['scatter_plot_metric1_min'],
                        y_max=plotting_config['scatter_plot_metric1_max'],
                        x_ticks=plotting_config['plddt_plot_ticks'],
                        y_ticks=plotting_config['scatter_plot_metric1_ticks'],
                        threshold_x=threshold_x,
                        threshold_y=threshold_y,
                        logger=self.logger
                    )
                    if plot_path:
                        plot_files.append(plot_path)
                except Exception as e:
                    self.logger.warning(f"Failed to create pLDDT scatter plot: {e}")
            
            # Add filtered results summary
            if filtered_results:
              
                # Extract PDB filenames from filtered_results keys (which are full paths)
                filtered_pdb_names = [pdb_path for pdb_path in filtered_results.keys()]
                self.logger.debug(f"Looking for PDB names: {filtered_pdb_names}")
                self.logger.debug(f"Available PDB names in df: {df['PDB'].tolist()}")
                
                filtered_df = df[df['PDB'].isin(filtered_pdb_names)]
                self.logger.info(f"Filtered DataFrame contains {len(filtered_df)} structures")
                
                if len(filtered_df) > 0:
                    print(filtered_df)
                    # breakpoint()
                    filtered_csv = output_dir / "filtered_structures.csv"
                    filtered_df.to_csv(filtered_csv, index=False)
                    self.logger.info(f"Saved {len(filtered_df)} filtered structures to CSV: {filtered_csv}")
                else:
                    self.logger.warning("No structures found in filtered DataFrame - check PDB name matching")
            
            self.logger.info(f"Created {len(plot_files)} plots in {plots_dir}")
            
            return csv_file, plot_files
            
        except Exception as e:
            self.logger.error(f"Error saving results and creating plots: {e}")
            return None, []
    
    def _filter_structures(self, 
                          analysis_results: Dict[str, Any],
                          filter_config: Dict[str, Any]) -> Dict[str, Any]:
        """Filter structures based on analysis criteria using hit_expand configuration."""
        filtered_results = {}
        
        # Get hit_expand specific thresholds from loaded configuration
        hit_expand_plddt_threshold = self.config.plddt_threshold
        filter_criteria_threshold = self.config.filter_criteria_threshold
        filter_criteria_name = self.config.filter_criteria
        
        self.logger.info(f"Filtering with pLDDT >= {hit_expand_plddt_threshold}")
        self.logger.info(f"Filtering with {filter_criteria_name} threshold: {filter_criteria_threshold}")
        
        # Get the selected filter criteria configuration (same logic as analysis stage)
        all_filter_criteria = filter_config.get("filter_criteria", [])
        composite_metrics = filter_config.get("composite_metrics", [])
        selected_filter_criteria = []
        
        # Check if we should use composite metrics
        use_composite_metrics = False
        if self.general_config and hasattr(self.general_config, 'use_composite_metrics'):
            use_composite_metrics = self.general_config.use_composite_metrics
            
        if filter_criteria_name:
            # Look in the appropriate section based on use_composite_metrics flag
            if use_composite_metrics:
                # Look in composite_metrics section
                for metric in composite_metrics:
                    if metric.get("name") == filter_criteria_name:
                        selected_filter_criteria.append(metric)
                        self.logger.info(f"Using composite metric '{filter_criteria_name}' for filtering")
            else:
                # Look in filter_criteria section
                for criterion in all_filter_criteria:
                    if criterion.get("name") == filter_criteria_name:
                        selected_filter_criteria.append(criterion)
        else:
            selected_filter_criteria = all_filter_criteria
        
        self.logger.info(f"Filter criteria for filtering stage: {len(selected_filter_criteria)} criteria")
        if not selected_filter_criteria:
            self.logger.warning("NO CRITERIA-BASED FILTERING - only pLDDT filtering will be applied!")
        
        for pdb_file, metrics in analysis_results.items():
            if "error" in metrics:
                continue
            
            # Check pLDDT threshold (already filtered in process_single_pdb, but double-check)
            plddt_score = metrics.get("plddt", 0.0)
            if plddt_score < hit_expand_plddt_threshold:
                self.logger.debug(f"Structure {pdb_file} filtered out: pLDDT {plddt_score} < {hit_expand_plddt_threshold}")
                continue
            
            # Check filter criteria against threshold
            passes_filter = True
            for criterion in selected_filter_criteria:
                criterion_name = criterion.get("name")
                criterion_type = criterion.get("type", "")
                criterion_method = criterion.get("method", "below")  # default filtering method
                
                if criterion_name in metrics:
                    criterion_value = metrics[criterion_name]
                    
                    # Apply threshold based on criterion type and method
                    if criterion_method == "above":
                        # For metrics where higher is better (e.g., TM-score)
                        if criterion_value < filter_criteria_threshold:
                            self.logger.debug(f"Structure {pdb_file} filtered out: {criterion_name} {criterion_value} < {filter_criteria_threshold}")
                            passes_filter = False
                            break
                        else:
                            self.logger.debug(f"Structure {pdb_file} passed: {criterion_name} {criterion_value} >= {filter_criteria_threshold}")
                    else:  # method == "below" (default)
                        # For metrics where lower is better (e.g., RMSD)
                        if criterion_value > filter_criteria_threshold:
                            self.logger.debug(f"Structure {pdb_file} filtered out: {criterion_name} {criterion_value} > {filter_criteria_threshold}")
                            passes_filter = False
                            break
                        else:
                            self.logger.debug(f"Structure {pdb_file} passed: {criterion_name} {criterion_value} <= {filter_criteria_threshold}")
                else:
                    # CRITICAL: This criterion was not calculated!
                    self.logger.warning(f"Structure {pdb_file}: Missing criterion '{criterion_name}' - STRUCTURE PASSES BY DEFAULT!")
                    self.logger.warning(f"Available metrics: {list(metrics.keys())}")
                    # Could choose to fail here instead: passes_filter = False
            
            if passes_filter:
                self.logger.debug(f"Structure {pdb_file} passed all filters")
                filtered_results[pdb_file] = metrics
        
        self.logger.info(f"Structures passed filtering: {len(filtered_results)}/{len(analysis_results)}")
        return filtered_results
    
    def _passes_filter_criteria(self, metrics: Dict[str, Any]) -> bool:
        """
        Check if a structure passes the filter criteria.
        
        This method uses the same logic as _filter_structures but for individual structures.
        
        Args:
            metrics: Structure analysis metrics
            
        Returns:
            True if structure passes all criteria, False otherwise
        """
        # Check pLDDT threshold
        plddt_score = metrics.get("plddt", 0.0)
        if plddt_score < self.config.plddt_threshold:
            return False
        
        # Check filter criteria threshold
        if self.config.filter_criteria != "default":
            criterion_value = metrics.get(self.config.filter_criteria)
            if criterion_value is None:
                # Missing criterion - by default, structure passes (matches main filtering logic)
                self.logger.debug(f"Missing criterion '{self.config.filter_criteria}' - structure passes by default")
                return True
            
            # Apply threshold based on criterion type
            # For RMSD (method="below"), lower values are better
            if criterion_value > self.config.filter_criteria_threshold:
                return False
        
        return True
    
    
    def _load_existing_subset_results(self, subsets_dir: Path) -> Dict[str, Any]:
        """
        Load existing subset results from subsets directory.
        
        Args:
            subsets_dir: Path to subsets directory
            
        Returns:
            Dictionary with subset results compatible with pipeline
        """
        self.logger.info("Loading existing subset results...")
        
        # Find all A3M files in subsets directory
        subset_paths = list(subsets_dir.rglob("*.a3m"))
        
        # Find batch directories
        batch_dirs = [d for d in subsets_dir.iterdir() if d.is_dir() and d.name.startswith("batch_")]
        batch_info = {}
        
        for batch_dir in batch_dirs:
            batch_name = batch_dir.name
            batch_a3m_files = list(batch_dir.glob("*.a3m"))
            batch_info[batch_name] = batch_a3m_files
        
        # Try to load metadata if it exists
        metadata_file = subsets_dir / "subset_generation_metadata.json"
        statistics = {}
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    statistics = json.load(f)
                self.logger.info(f"Loaded existing metadata from {metadata_file}")
            except Exception as e:
                self.logger.warning(f"Could not load metadata: {e}")
        
        subset_results = {
            "subset_paths": subset_paths,
            "batch_info": batch_info,
            "statistics": statistics,
            "representative_sequences": {},  # Will be loaded if needed
            "validation": {"total_subsets": len(subset_paths), "valid_subsets": len(subset_paths)}
        }
        
        self.logger.info(f"Loaded {len(subset_paths)} subset files in {len(batch_info)} batches")
        return subset_results
    
    def _run_analysis_only_workflow(self, subset_results: Dict[str, Any]) -> Optional[Path]:
        """
        Run analysis-only workflow for existing subset results.
        
        Args:
            subset_results: Existing subset results
            
        Returns:
            Path to final MSA file if successful, None otherwise
        """
        self.logger.info("=== RUNNING ANALYSIS-ONLY WORKFLOW ===")
        
        # Use subsets directory as prediction directory (since we use it directly now)
        subsets_dir = self.base_dir / "02_subsets"
        prediction_results = {
            "job_specs": [],
            "submitted_jobs": [],
            "prediction_dir": subsets_dir
        }
        
        # Stage 4: Structure analysis and filtering (if not skipped)
        if not self.config.skip_structure_analysis:
            analysis_results = self._run_structure_analysis_stage(prediction_results)
            good_sequences = self._extract_good_sequences(analysis_results, subset_results)
            
            # Check if any structures passed filtering
            filtered_results = analysis_results.get("filtered_results", {})
            if not filtered_results:
                self.logger.warning("=== NO STRUCTURES PASSED FILTERING CRITERIA ===")
                self.logger.warning("Aborting similarity search - no good sequences to expand")
                self.logger.info("Saving sequences from first available subset as final result")
                
                # Use sequences from first available subset as fallback
                if subset_results["subset_paths"]:
                    parser = A3MParser(strict_validation=False)
                    fallback_sequences = parser.parse_file(subset_results["subset_paths"][0])
                    final_msa = self.base_dir / "hit_expand_final_msa_round_1.a3m"
                    parser.write_sequences(fallback_sequences, final_msa)
                else:
                    # Create empty result if no subsets available
                    final_msa = self.base_dir / "hit_expand_final_msa_round_1.a3m"
                    parser = A3MParser(strict_validation=False)
                    parser.write_sequences({}, final_msa)
                
                self.logger.info(f"=== ANALYSIS-ONLY WORKFLOW COMPLETED (NO EXPANSION) ===")
                return final_msa
                
        else:
            self.logger.info("Skipping structure analysis stage")
            # Use representative sequences if available, otherwise parse from first subset
            good_sequences = subset_results.get("representative_sequences", {})
            if not good_sequences:
                # Parse sequences from first available subset as fallback
                if subset_results["subset_paths"]:
                    parser = A3MParser(strict_validation=False)
                    good_sequences = parser.parse_file(subset_results["subset_paths"][0])
        
        # Stage 5: Similarity search to expand good sequences (if not skipped)
        if not self.config.skip_hit_expansion:
            # Need original input MSA for similarity search
            input_msa = Path(self.config.input_msa)
            if input_msa.exists():
                search_dir = self.base_dir / "05_similarity_search"
                final_msa = self._run_stage_with_done_check(
                    stage_dir=search_dir,
                    stage_name="STAGE 5: SIMILARITY SEARCH & EXPANSION",
                    stage_id="05_similarity_search",
                    stage_function=self._run_similarity_search,
                    good_sequences=good_sequences,
                    output_dir=search_dir,
                    source_msa=input_msa,
                    round_num=1,
                    return_expanded_msa=True
                )
            else:
                self.logger.warning(f"Input MSA not found: {input_msa}, saving good sequences directly")
                final_msa = self.base_dir / "hit_expand_final_msa_round_1.a3m"
                parser = A3MParser(strict_validation=False)
                parser.write_sequences(good_sequences, final_msa)
        else:
            self.logger.info("Skipping hit expansion stage")
            final_msa = self.base_dir / "hit_expand_final_msa_round_1.a3m"
            parser = A3MParser(strict_validation=False)
            parser.write_sequences(good_sequences, final_msa)
        
        # Stage 6: Generate expanded subsets from final MSA
        expanded_subsets_dir = self.base_dir / "06_expanded_subsets"
        
        # Check if existing results exist
        existing_pdbs = list(expanded_subsets_dir.rglob("*.pdb")) if expanded_subsets_dir.exists() else []
        if existing_pdbs:
            self.logger.info(f"Found {len(existing_pdbs)} existing structure prediction results in {expanded_subsets_dir}")
            self.logger.info("Skipping subset generation and structure prediction - proceeding directly to plotting")
            
            # Load existing subset results for plotting
            expanded_subset_results = self._load_existing_subset_results(expanded_subsets_dir)
            
            # Run plotting directly on existing results
            self._run_expanded_subset_plotting(expanded_subsets_dir, expanded_subset_results)
        else:
            # Run subset generation and structure prediction - inlined _run_expanded_subset_workflow
            def expanded_subset_workflow_inline(final_msa: Path, expanded_subsets_dir: Path) -> Dict[str, Any]:
                # Generate subsets using unified function
                subset_results = self._run_subset_generation(
                    expanded_msa=final_msa,
                    output_dir=expanded_subsets_dir,
                    stage_name="EXPANDED SUBSET GENERATION",
                    include_query=True
                )
                
                self.logger.info(f"Expanded subset generation completed: {len(subset_results['subset_paths'])} subsets")
                
                # Run structure prediction on expanded subsets
                prediction_results = self._run_generic_structure_prediction(
                    subset_results=subset_results,
                    base_dir=expanded_subsets_dir,
                    job_prefix="expanded",
                    stage_name="expanded structure prediction"
                )
                
                # Run plotting on the prediction results
                self._run_expanded_subset_plotting(expanded_subsets_dir, subset_results)
                
                self.workflow_state["expanded_subset_generation_completed"] = True
                
                return subset_results
            
            expanded_subset_results = self._run_stage_with_done_check(
                stage_dir=expanded_subsets_dir,
                stage_name="STAGE 6: EXPANDED SUBSET GENERATION",
                stage_id="06_expanded_subsets",
                stage_function=expanded_subset_workflow_inline,
                final_msa=final_msa,
                expanded_subsets_dir=expanded_subsets_dir
            )
        
        self.logger.info(f"=== ANALYSIS-ONLY WORKFLOW COMPLETED ===")
        self.logger.info(f"Final MSA: {final_msa}")
        self.logger.info(f"Expanded subsets: {len(expanded_subset_results['subset_paths'])} subsets in {len(expanded_subset_results['batch_info'])} batches")
        
        return final_msa

    def _create_done_file(self, stage_dir: Path, stage_name: str):
        """Create a DONE file to mark stage completion."""
        done_file = stage_dir / f"{stage_name}.DONE"
        with open(done_file, 'w') as f:
            f.write(f"Stage {stage_name} completed successfully\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.logger.info(f"Created DONE file: {done_file}")
    
    def _check_done_file(self, stage_dir: Path, stage_name: str) -> bool:
        """Check if a DONE file exists for the stage."""
        done_file = stage_dir / f"{stage_name}.DONE"
        return done_file.exists()
    
    def _remove_done_file(self, stage_dir: Path, stage_name: str):
        """Remove a DONE file (for cleanup or re-runs)."""
        done_file = stage_dir / f"{stage_name}.DONE"
        if done_file.exists():
            done_file.unlink()
            self.logger.info(f"Removed DONE file: {done_file}")

    def _run_stage_with_done_check(self, 
                                  stage_dir: Path,
                                  stage_name: str,
                                  stage_id: str,
                                  stage_function: callable,
                                  skip_if_done: bool = True,
                                  **kwargs) -> Any:
        """
        Generic stage runner that handles common patterns:
        - Stage logging
        - Directory creation
        - DONE file checking/creation
        - Error handling
        
        Args:
            stage_dir: Directory for this stage
            stage_name: Human-readable stage name for logging
            stage_id: Identifier for DONE file
            stage_function: The actual function to run
            skip_if_done: Whether to skip if DONE file exists
            **kwargs: Arguments to pass to stage_function
            
        Returns:
            Result from stage_function
        """
        # Check if stage already completed
        if skip_if_done and self._check_done_file(stage_dir, stage_id):
            self.logger.info(f"=== {stage_name} ALREADY COMPLETED - SKIPPING ===")
            # Return a default result based on common patterns
            if "load_existing" in kwargs:
                return kwargs["load_existing"](stage_dir)
            return {"skipped": True, "stage_dir": stage_dir}
        
        # Log stage start
        self.logger.info(f"=== {stage_name} ===")
        
        # Create stage directory
        stage_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Run the actual stage function
            result = stage_function(**kwargs)
            
            # Create DONE file on success
            self._create_done_file(stage_dir, stage_id)
            
            return result
            
        except Exception as e:
            self.logger.error(f"{stage_name} failed: {e}")
            raise

    def _run_subset_generation(self, 
                              sequences: Optional[Dict[str, str]] = None,
                              expanded_msa: Optional[Path] = None,
                              output_dir: Path = None,
                              stage_name: str = "SUBSET GENERATION",
                              stage_id: str = "subset_generation",
                              config_override: Optional[SubsetConfig] = None,
                              include_query: bool = True,
                              prefix_override: Optional[str] = None) -> Dict[str, Any]:
        """
        Unified subset generation function that replaces:
        - _run_subset_generation_stage_for_round
        - _run_expanded_subset_generation_stage
        - _run_expanded_subset_generation_stage_for_round
        
        Args:
            sequences: Dict of sequences to use (mutually exclusive with expanded_msa)
            expanded_msa: Path to MSA file (mutually exclusive with sequences)
            output_dir: Output directory for subsets
            stage_name: Human-readable stage name for logging
            stage_id: Stage identifier for DONE files
            config_override: Optional custom subset configuration
            include_query: Whether to ensure query sequence is included
            prefix_override: Override the output prefix (e.g., "expanded_subset")
            
        Returns:
            Dict with subset generation results
        """
        if sequences is None and expanded_msa is None:
            raise ValueError("Either sequences or expanded_msa must be provided")
        
        if sequences is not None and expanded_msa is not None:
            raise ValueError("Only one of sequences or expanded_msa should be provided")
        
        # Parse sequences if MSA file provided
        if expanded_msa:
            parser = A3MParser(strict_validation=False)
            sequences = parser.parse_file(expanded_msa)
            self.logger.info(f"Parsed {len(sequences)} sequences from MSA file")
        
        # Get query sequence if needed
        query_header = None
        query_sequence = None
        if include_query:
            input_msa = Path(self.config.input_msa)
            parser = A3MParser(strict_validation=False)
            source_sequences = parser.parse_file(input_msa)
            query_header, query_sequence = parser.get_query_sequence(source_sequences)
            self.logger.info(f"Source query sequence: {query_header}")
            
            # Ensure query is first
            sequences_with_query = {query_header: query_sequence}
            sequences_with_query.update(sequences)
            sequences = sequences_with_query
        
        # Use custom config or create one with overrides
        if config_override:
            subset_config = config_override
        else:
            # Create config based on current subset generator config
            subset_config = SubsetConfig(
                num_subsets=self.config.num_subsets,
                num_random_sequences=self.config.num_random_sequences,
                num_batches=self.config.num_batches,
                batch_prefix=prefix_override.replace("_subset", "_batch") if prefix_override else self.config.batch_prefix,
                output_prefix=prefix_override or self.config.output_prefix,
                random_seed=self.config.random_seed
            )
        
        # Create temporary generator if using custom config
        if config_override or prefix_override:
            subset_generator = SubsetGenerator(subset_config)
        else:
            subset_generator = self.subset_generator
        
        # Generate subsets
        subset_results = subset_generator.generate_subsets(
            sequences=sequences,
            output_dir=output_dir
        )
        
        # Validate subsets
        validation_results = subset_generator.validate_subsets(
            subset_results["subset_paths"]
        )
        
        # Add additional metadata
        subset_results["validation"] = validation_results
        subset_results["representative_sequences"] = sequences
        if query_header:
            subset_results["query_header"] = query_header
            subset_results["query_sequence"] = query_sequence
        
        self.logger.info(f"{stage_name} complete: {len(subset_results['subset_paths'])} subsets created")
        
        return subset_results



    def _run_structure_analysis(self,
                               prediction_results: Dict[str, Any],
                               output_dir: Path,
                               stage_name: str = "STRUCTURE ANALYSIS",
                               stage_id: str = "structure_analysis",
                               filter_results: bool = True,
                               save_plots: bool = True) -> Dict[str, Any]:
        """
        Unified structure analysis function that replaces:
        - _run_structure_analysis_stage_for_round
        - Parts of _run_expanded_subset_plotting
        
        Args:
            prediction_results: Results from structure prediction
            output_dir: Directory to save analysis results
            stage_name: Human-readable stage name
            stage_id: Stage identifier
            filter_results: Whether to apply filtering criteria
            save_plots: Whether to generate and save plots
            
        Returns:
            Dict with analysis results and filtered results
        """
        # Collect all PDB files from prediction results
        prediction_dir = prediction_results["prediction_dir"]
        pdb_files = list(prediction_dir.rglob("*.pdb"))
        
        if not pdb_files:
            self.logger.warning("No PDB files found for structure analysis")
            return {"pdb_files": [], "analysis_results": {}, "filtered_results": {}}
        
        self.logger.info(f"Found {len(pdb_files)} PDB files for analysis")
        
        # Load filter configuration
        with open(self.config_file, 'r') as f:
            filter_config = json.load(f)
        
        
        all_filter_criteria = filter_config.get("filter_criteria", [])
        composite_metrics = filter_config.get("composite_metrics", [])
        
        # Analyze structures in parallel
        analysis_results = self.structure_analyzer.process_pdbs_parallel(
            pdb_files=pdb_files,
            filter_criteria=all_filter_criteria,
            basics=filter_config.get("basics", {}),
            plddt_threshold=0.0, # in the structure analysis steps, all structure will considered and will not apply threshold
            n_jobs=-1,  # Use all available CPU cores
            composite_metrics=composite_metrics
        )
        
        # Apply filtering if requested
        if filter_results:
            filtered_results = self._filter_structures(analysis_results, filter_config)
        else:
            # No filtering - all results pass
            filtered_results = analysis_results
        
        # Save results and create plots if requested
        csv_file = None
        plot_files = []
        if save_plots:
            csv_file, plot_files = self._save_results_and_create_plots(
                analysis_results, filtered_results, output_dir, all_filter_criteria
            )
        
        self.logger.info(f"Analysis complete. {len(filtered_results)} structures passed filtering")
        
        return {
            "pdb_files": pdb_files,
            "analysis_results": analysis_results,
            "filtered_results": filtered_results,
            "csv_file": csv_file,
            "plot_files": plot_files
        }

    def _run_similarity_search(self,
                              good_sequences: Dict[str, str],
                              output_dir: Path,
                              source_msa: Path,
                              stage_name: str = "SIMILARITY SEARCH",
                              stage_id: str = "similarity_search",
                              round_num: int = 1,
                              return_expanded_msa: bool = True) -> Union[Path, Dict[str, str]]:
        """
        Unified expansion function supporting both BLOSUM62 and MMseqs2 cluster-based methods.
        
        Args:
            good_sequences: Sequences to search for similar ones
            output_dir: Output directory for results
            source_msa: Source MSA to search against
            stage_name: Human-readable stage name
            stage_id: Stage identifier
            round_num: Current round number
            return_expanded_msa: If True, return path to MSA; if False, return sequences dict
            
        Returns:
            Either Path to expanded MSA or Dict of newly found sequences
        """
        self.logger.info(f"Running expansion using method: {self.config.expansion_method}")
        
        expanded_sequences = {}
        
        # Choose expansion method based on configuration
        if self.config.expansion_method == "mmseqs_result":
            # Use MMseqs2 cluster-based expansion
            cluster_file = self.base_dir / "01_clustering" / "clustered_cluster.tsv"
            
            if not cluster_file.exists():
                self.logger.warning(f"Cluster file not found: {cluster_file}")
                self.logger.warning("Falling back to BLOSUM62 similarity search")
                # Fallback to BLOSUM62
                expanded_msa_path = self._run_blosum62_expansion(
                    good_sequences, output_dir, source_msa
                )
                expanded_sequences = self._load_sequences_from_msa(expanded_msa_path)
            else:
                try:
                    expander = ClusterBasedExpansion(
                        cluster_file, 
                        source_msa, 
                        max_sequences_per_cluster=self.config.max_sequences_per_cluster
                    )
                    expanded_sequences = expander.expand_by_clusters(
                        good_sequences,
                        output_dir=output_dir
                    )
                    
                    # Get statistics and log them
                    stats = expander.get_expansion_statistics()
                    self.logger.info(f"Cluster expansion stats: {stats}")
                    
                except ClusterExpansionError as e:
                    self.logger.error(f"Cluster expansion failed: {e}")
                    self.logger.info("Falling back to BLOSUM62 similarity search")
                    # Fallback to BLOSUM62
                    expanded_msa_path = self._run_blosum62_expansion(
                        good_sequences, output_dir, source_msa
                    )
                    expanded_sequences = self._load_sequences_from_msa(expanded_msa_path)
                    
        elif self.config.expansion_method == "BLOSUM62":
            # Use existing BLOSUM62 similarity search
            expanded_msa_path = self._run_blosum62_expansion(
                good_sequences, output_dir, source_msa
            )
            expanded_sequences = self._load_sequences_from_msa(expanded_msa_path)
            
        else:
            raise ValueError(f"Unknown expansion method: {self.config.expansion_method}")
        
        # Always save expanded sequences to MSA file (regardless of return_expanded_msa)
        expanded_msa_path = output_dir / "expanded_sequences.a3m"
        if expanded_sequences:
            parser = A3MParser(strict_validation=False)
            parser.write_sequences(expanded_sequences, expanded_msa_path)
            self.logger.info(f"Saved {len(expanded_sequences)} expanded sequences to {expanded_msa_path}")
        
        # Return sequences directly if not returning MSA
        if not return_expanded_msa:
            self.logger.info(f"Round {round_num}: Found {len(expanded_sequences)} sequences")
            return expanded_sequences
        
        # Return based on what's requested
        if return_expanded_msa:
            # Copy to final location if it's round 1
            if round_num == 1:
                final_msa_path = self.base_dir / "hit_expand_final_msa_round_1.a3m"
                if expanded_msa_path.exists():
                    shutil.copy2(expanded_msa_path, final_msa_path)
                    self.logger.info(f"Expansion completed: {final_msa_path}")
                    return final_msa_path
                else:
                    self.logger.error("Expanded MSA file was not created")
                    return None
            else:
                return expanded_msa_path
        else:
            # Return sequences dictionary
            return expanded_sequences
    
    def _run_blosum62_expansion(self,
                               good_sequences: Dict[str, str],
                               output_dir: Path,
                               source_msa: Path) -> Path:
        """
        Run BLOSUM62-based similarity search expansion.
        
        Args:
            good_sequences: Sequences to expand
            output_dir: Output directory
            source_msa: Source MSA file
            
        Returns:
            Path to expanded MSA file
        """
        return self.similarity_search.search_and_expand(
            representative_sequences=good_sequences,
            source_msa=source_msa,
            output_dir=output_dir
        )
    
    def _load_sequences_from_msa(self, msa_path: Path) -> Dict[str, str]:
        """
        Load sequences from an MSA file.
        
        Args:
            msa_path: Path to MSA file
            
        Returns:
            Dictionary of sequences
        """
        if not msa_path or not msa_path.exists():
            self.logger.warning(f"MSA file not found: {msa_path}")
            return {}
            
        parser = A3MParser(strict_validation=False)
        return parser.parse_file(msa_path)

    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status and statistics."""
        return {
            "workflow_state": self.workflow_state,
            "base_dir": str(self.base_dir),
            "config": {
                "input_msa": self.config.input_msa,
                "num_subsets": self.config.num_subsets,
                "num_batches": self.config.num_batches,
                "similarity_threshold": self.config.similarity_threshold,
                "plddt_threshold": self.config.plddt_threshold
            }
        }
    
    def setup_logging(self):
        """Set up logging for the hit expand pipeline."""
        log_dir = self.base_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / "hit_expand_pipeline.log"
        
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
        
        self.logger.info(f"Logging configured for hit expand pipeline: {log_file}")
    
    def log_parameters(self):
        """Log all pipeline parameters."""
        self.logger.info("=== HIT EXPAND PIPELINE PARAMETERS ===")
        self.logger.info(f"Input MSA: {self.config.input_msa}")
        self.logger.info(f"Base directory: {self.base_dir}")
        self.logger.info(f"Config file: {self.config_file}")
        self.logger.info(f"Number of subsets: {self.config.num_subsets}")
        self.logger.info(f"Sequences per subset: {self.config.num_random_sequences}")
        self.logger.info(f"Number of batches: {self.config.num_batches}")
        self.logger.info(f"Similarity threshold: {self.config.similarity_threshold}")
        self.logger.info(f"pLDDT threshold: {self.config.plddt_threshold}")
        self.logger.info(f"Filter criteria threshold: {self.config.filter_criteria_threshold}")
        self.logger.info(f"Random seed: {self.config.random_seed}")
        self.logger.info(f"Number of rounds: {self.config.rounds}")
        self.logger.info(f"Cumulative expansion: {self.config.cumulative_expansion}")
        self.logger.info("========================================")
    
    # Multi-round helper methods
    def _load_clustered_sequences(self, clustering_dir: Path) -> Dict[str, str]:
        """Load representative sequences from clustering results."""
        rep_fasta = clustering_dir / "clustered_rep_seq.fasta"
        representative_sequences = {}
        
        if rep_fasta.exists():
            with open(rep_fasta, 'r') as f:
                current_header = None
                for line in f:
                    line = line.strip()
                    if line.startswith('>'):
                        current_header = line
                    elif current_header and line:
                        representative_sequences[current_header] = line
                        current_header = None
        else:
            # Fallback: parse input MSA directly
            parser = A3MParser(strict_validation=False)
            representative_sequences = parser.parse_file(Path(self.config.input_msa))
        
        self.logger.info(f"Loaded {len(representative_sequences)} representative sequences")
        return representative_sequences
    
    def _load_expanded_subsets(self, expanded_subsets_dir: Path) -> Dict[str, str]:
        """Load all sequences from expanded subset A3M files."""
        all_sequences = {}
        a3m_files = list(expanded_subsets_dir.glob("*.a3m"))
        
        with tqdm(total=len(a3m_files), desc="Loading expanded subsets") as pbar:
            for a3m_file in a3m_files:
                from af_claseq.utils.sequence_processing import A3MParser
                parser = A3MParser(strict_validation=False)
                sequences = parser.parse_file(a3m_file)
                all_sequences.update(sequences)
                pbar.update(1)
        
        self.logger.info(f"Loaded {len(all_sequences)} sequences from {len(a3m_files)} expanded subset files")
        return all_sequences
    
    def _load_expanded_subset_prediction_results(self, expanded_subsets_dir: Path) -> Dict[str, Any]:
        """Load structure prediction results from previous round's expanded subsets."""
        if not expanded_subsets_dir.exists():
            self.logger.error(f"Previous round's expanded subsets directory not found: {expanded_subsets_dir}")
            return {"job_specs": [], "submitted_jobs": [], "prediction_dir": expanded_subsets_dir}
        
        # Find all PDB files in the expanded subsets directory
        pdb_files = list(expanded_subsets_dir.glob("**/*.pdb"))
        
        if not pdb_files:
            self.logger.warning(f"No PDB files found in expanded subsets directory: {expanded_subsets_dir}")
            return {"job_specs": [], "submitted_jobs": [], "prediction_dir": expanded_subsets_dir}
        
        self.logger.info(f"Found {len(pdb_files)} PDB files in previous round's expanded subsets")
        
        # Create prediction results structure similar to what _run_generic_structure_prediction returns
        prediction_results = {
            "job_specs": [],
            "submitted_jobs": [],
            "prediction_dir": expanded_subsets_dir,
            "pdb_files": pdb_files,
            "structure_count": len(pdb_files)
        }
        
        return prediction_results
    
    def _extract_good_sequences_from_previous_round(self, analysis_results: Dict[str, Any], 
                                                   input_sequences: Dict[str, str], 
                                                   prev_expanded_dir: Path) -> Dict[str, str]:
        """Extract good sequences from analysis results for round 2+."""
        filtered_results = analysis_results.get("filtered_results", {})
        
        if not filtered_results:
            self.logger.warning("No structures passed filtering - using all input sequences")
            return input_sequences
        
        # For round 2+, we need to map the filtered PDB results back to sequence headers
        # This is a simplified approach - use all input sequences if any structures passed
        self.logger.info(f"Round 2+ structure analysis: {len(filtered_results)} structures passed filtering")
        self.logger.info("Using all input sequences for similarity search")
        
        return input_sequences
    
    def _run_final_structure_analysis(self, final_round_num: int) -> bool:
        """Run final structure analysis on the last round's 06_expanded_subsets for evaluation."""
        self.logger.info("=== RUNNING FINAL STRUCTURE ANALYSIS ===")
        
        # Get the final round's expanded subsets directory
        final_round_dir = self.base_dir / f"round_{final_round_num}"
        final_expanded_dir = final_round_dir / "06_expanded_subsets"
        
        if not final_expanded_dir.exists():
            self.logger.warning(f"Final round's expanded subsets not found: {final_expanded_dir}")
            return False
        
        # Load prediction results from final expanded subsets
        prediction_results = self._load_expanded_subset_prediction_results(final_expanded_dir)
        
        if not prediction_results.get("pdb_files"):
            self.logger.warning("No structures found in final expanded subsets")
            return False
        
        # Create final analysis directory within the final round folder
        final_analysis_dir = final_round_dir / "final_structure_analysis"
        final_analysis_dir.mkdir(exist_ok=True)
        
        # Run final structure analysis
        try:
            self.logger.info(f"Analyzing {len(prediction_results['pdb_files'])} structures in final expanded subsets")
            final_analysis_results = self._run_structure_analysis(
                prediction_results=prediction_results,
                output_dir=final_analysis_dir,
                stage_name="FINAL STRUCTURE ANALYSIS",
                stage_id="final_structure_analysis",
                filter_results=True,
                save_plots=True
            )
            
            # Log final results
            final_filtered = final_analysis_results.get("filtered_results", {})
            total_structures = len(prediction_results['pdb_files'])
            filtered_count = len(final_filtered)
            
            self.logger.info("=== FINAL EVALUATION RESULTS ===")
            self.logger.info(f"Total structures in final expanded subsets: {total_structures}")
            self.logger.info(f"Structures passing final filters: {filtered_count}")
            if total_structures > 0:
                success_rate = (filtered_count / total_structures) * 100
                self.logger.info(f"Final success rate: {success_rate:.1f}%")
            
            # Create done file for final analysis
            self._create_done_file(final_analysis_dir, "final_structure_analysis")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Final structure analysis failed: {e}")
            return False
    
    def _run_single_round(self, round_num: int, round_dir: Path, 
                         input_sequences: Dict[str, str], input_type: str) -> Optional[Path]:
        """Run a single round of hit expand."""
        self.logger.info(f"Running round {round_num} with {len(input_sequences)} input sequences ({input_type})")
        
        try:
            if round_num == 1:
                # ROUND 1: Full pipeline
                # Stage 2: Generate subsets from clustered sequences
                subsets_dir = round_dir / "02_subsets"
                if self._check_done_file(subsets_dir, "02_subsets"):
                    self.logger.info("=== STAGE 2: SUBSET GENERATION ALREADY COMPLETED - SKIPPING ===")
                    subset_results = self._load_existing_subset_results(subsets_dir)
                else:
                    subset_results = self._run_stage_with_done_check(
                        stage_dir=subsets_dir,
                        stage_name="STAGE 2: SUBSET GENERATION",
                        stage_id="02_subsets",
                        stage_function=self._run_subset_generation,
                        sequences=input_sequences,
                        output_dir=subsets_dir,
                        include_query=True
                    )
                
                # Stage 3: Structure prediction
                prediction_dir = subsets_dir  # Using subsets directory directly
                if not self.config.skip_structure_prediction:
                    if self._check_done_file(prediction_dir, "03_structure_prediction"):
                        self.logger.info("=== STAGE 3: STRUCTURE PREDICTION ALREADY COMPLETED - SKIPPING ===")
                        prediction_results = {
                            "job_specs": [],
                            "submitted_jobs": [],
                            "prediction_dir": prediction_dir
                        }
                    else:
                        prediction_results = self._run_generic_structure_prediction(
                            subset_results=subset_results,
                            base_dir=prediction_dir,
                            job_prefix="hit_expand",
                            stage_name=f"structure prediction round {round_num}"
                        )
                else:
                    self.logger.info("Skipping structure prediction stage")
                    prediction_results = None
                
                # Stage 4: Structure analysis
                analysis_dir = round_dir / "04_structure_analysis"
                if not self.config.skip_structure_analysis and prediction_results:
                    if self._check_done_file(analysis_dir, "04_structure_analysis"):
                        self.logger.info("=== STAGE 4: STRUCTURE ANALYSIS ALREADY COMPLETED - SKIPPING ===")
                        analysis_results = {"analysis_results": {}, "filtered_results": {}}
                        good_sequences = subset_results.get("representative_sequences", {})
                    else:
                        analysis_results = self._run_stage_with_done_check(
                            stage_dir=analysis_dir,
                            stage_name="STAGE 4: STRUCTURE ANALYSIS",
                            stage_id="04_structure_analysis",
                            stage_function=self._run_structure_analysis,
                            prediction_results=prediction_results,
                            output_dir=analysis_dir,
                            filter_results=True,
                            save_plots=True
                        )
                        good_sequences = self._extract_good_sequences(analysis_results, subset_results)
                else:
                    self.logger.info("Skipping structure analysis stage")
                    good_sequences = input_sequences
                
            else:
                # ROUND 2+: Use previous round's 06_expanded_subsets (no stages 2&3)
                prev_round_dir = self.base_dir / f"round_{round_num - 1}"
                prev_expanded_dir = prev_round_dir / "06_expanded_subsets"
                
                self.logger.info(f"=== ROUND {round_num}: USING PREVIOUS ROUND'S EXPANDED SUBSETS ===")
                self.logger.info(f"Previous expanded subsets: {prev_expanded_dir}")
                
                # Skip stages 2&3 - Load structure prediction results from previous round's 06_expanded_subsets
                prediction_results = self._load_expanded_subset_prediction_results(prev_expanded_dir)
                
                if not prediction_results.get("pdb_files"):
                    self.logger.error(f"No structures found in previous round's expanded subsets: {prev_expanded_dir}")
                    return None
                
                # Stage 4: Structure analysis on previous round's expanded subsets
                analysis_dir = round_dir / "04_structure_analysis"
                if not self.config.skip_structure_analysis:
                    if self._check_done_file(analysis_dir, "04_structure_analysis"):
                        self.logger.info("=== STAGE 4: STRUCTURE ANALYSIS ALREADY COMPLETED - SKIPPING ===")
                        analysis_results = {"analysis_results": {}, "filtered_results": {}}
                        # For round 2+, good_sequences come from the input_sequences (previous round's final MSA)
                        good_sequences = input_sequences
                    else:
                        self.logger.info(f"=== STAGE 4: ANALYZING PREVIOUS ROUND'S EXPANDED SUBSETS ===")
                        analysis_results = self._run_stage_with_done_check(
                            stage_dir=analysis_dir,
                            stage_name=f"STAGE 4: STRUCTURE ANALYSIS (Round {round_num})",
                            stage_id="04_structure_analysis",
                            stage_function=self._run_structure_analysis,
                            prediction_results=prediction_results,
                            output_dir=analysis_dir,
                            filter_results=True,
                            save_plots=True
                        )
                        # Extract good sequences based on analysis results
                        # For round 2+, we need to map structure analysis back to sequences
                        good_sequences = self._extract_good_sequences_from_previous_round(
                            analysis_results, input_sequences, prev_expanded_dir
                        )
                else:
                    self.logger.info("Skipping structure analysis stage")
                    good_sequences = input_sequences
            
            # Check if any structures passed filtering
            filtered_results = analysis_results.get("filtered_results", {})
            if not filtered_results:
                self.logger.warning(f"=== ROUND {round_num}: NO STRUCTURES PASSED FILTERING CRITERIA ===")
                if round_num == 1:
                    self.logger.warning("Aborting similarity search - no good sequences to expand")
                    self.logger.info("Saving representative sequences as final result")
                    final_msa = round_dir / f"hit_expand_final_msa_round_{round_num}.a3m"
                    from af_claseq.utils.sequence_processing import A3MParser
                    parser = A3MParser(strict_validation=False)
                    parser.write_sequences(input_sequences, final_msa)
                    return final_msa
                else:
                    self.logger.info("No new hits found in this round")
                    return None
            
            # Stage 5: Similarity search against ORIGINAL source
            if not self.config.skip_hit_expansion:
                search_dir = round_dir / "05_similarity_search"
                
                # Search for similar sequences (no exclusions - each round works independently)
                newly_found_sequences = self._run_stage_with_done_check(
                    stage_dir=search_dir,
                    stage_name=f"STAGE 5: SIMILARITY SEARCH (Round {round_num})",
                    stage_id="05_similarity_search",
                    stage_function=self._run_similarity_search,
                    good_sequences=good_sequences,
                    output_dir=search_dir,
                    source_msa=Path(self.config.input_msa),
                    round_num=round_num,
                    return_expanded_msa=False
                )
                
                if not newly_found_sequences:
                    self.logger.info(f"No new sequences found in round {round_num}")
                    return None
                
                # Stage 6: Generate expanded subsets and predict ONLY new sequences
                expanded_subsets_dir = round_dir / "06_expanded_subsets"
                if self._check_done_file(expanded_subsets_dir, "06_expanded_subsets"):
                    self.logger.info("=== STAGE 6: EXPANDED SUBSET GENERATION ALREADY COMPLETED - SKIPPING ===")
                else:
                    if round_num > 1:
                        # Copy good structures from previous round
                        self._copy_good_structures_from_previous_round(
                            prev_expanded_dir, expanded_subsets_dir, good_sequences
                        )
                    
                    # Generate subsets and predict only new sequences
                    start_index = self._get_next_subset_index(expanded_subsets_dir)
                    
                    # Inline _run_expanded_subset_for_new_sequences
                    # Create subset configuration for expanded sequences
                    expanded_subset_config = SubsetConfig(
                        num_subsets=self.config.num_subsets,
                        num_random_sequences=self.config.num_random_sequences,
                        num_batches=self.config.num_batches,
                        batch_prefix="expanded_batch",
                        output_prefix="expanded_subset",
                        random_seed=self.config.random_seed
                    )
                    
                    # Generate subsets using unified function
                    subset_results = self._run_subset_generation(
                        sequences=newly_found_sequences,
                        output_dir=expanded_subsets_dir,
                        stage_name="NEW SEQUENCES SUBSET GENERATION",
                        config_override=expanded_subset_config,
                        include_query=True
                    )
                    
                    # Create job specifications for the new subsets
                    job_specs = []
                    for batch_name, subset_paths_in_batch in subset_results["batch_info"].items():
                        if subset_paths_in_batch:
                            batch_dir = Path(subset_paths_in_batch[0]).parent
                            job_spec = {
                                "name": f"expanded_{batch_name}",
                                "task_dir": str(batch_dir),
                                "job_id": batch_name,
                                "gres": "gpu:1",
                                "num_subsets": len(subset_paths_in_batch)
                            }
                            job_specs.append(job_spec)
                    
                    # Submit ColabFold jobs for the new subsets
                    if job_specs:
                        self._submit_and_monitor_structure_jobs(
                            job_specs=job_specs,
                            job_prefix="expanded",
                            stage_name="expanded subset structure prediction"
                        )
                
                # Create final MSA for this round by copying expanded_sequences.a3m
                final_msa_name = f"hit_expand_final_msa_round_{round_num}.a3m"
                final_msa = round_dir / final_msa_name
                
                # Copy expanded_sequences.a3m from similarity search
                expanded_sequences_file = search_dir / "expanded_sequences.a3m"
                if expanded_sequences_file.exists():
                    shutil.copy2(expanded_sequences_file, final_msa)
                    self.logger.info(f"Round {round_num} final MSA created by copying expanded sequences")
                else:
                    self.logger.error(f"expanded_sequences.a3m not found: {expanded_sequences_file}")
                    return None
                
                return final_msa
            
            else:
                self.logger.info("Skipping hit expansion stage")
                final_msa = round_dir / f"hit_expand_final_msa_round_{round_num}.a3m"
                from af_claseq.utils.sequence_processing import A3MParser
                parser = A3MParser(strict_validation=False)
                parser.write_sequences(good_sequences, final_msa)
                return final_msa
        
        except Exception as e:
            self.logger.error(f"Round {round_num} failed: {e}")
            return None
    
    
    def _get_all_previous_sequences(self, current_round: int) -> set:
        """Get all sequences found in previous rounds to avoid duplicates."""
        all_sequences = set()
        
        # Collect from all previous rounds
        for round_num in range(1, current_round):
            round_dir = self.base_dir / f"round_{round_num}"
            
            # Check both similarity search results and expanded subsets
            for subdir in ["05_similarity_search", "06_expanded_subsets"]:
                search_dir = round_dir / subdir
                if search_dir.exists():
                    for a3m_file in search_dir.glob("*.a3m"):
                        try:
                            from af_claseq.utils.sequence_processing import A3MParser
                            parser = A3MParser(strict_validation=False)
                            sequences = parser.parse_file(a3m_file)
                            all_sequences.update(sequences.keys())
                        except Exception as e:
                            self.logger.debug(f"Failed to parse {a3m_file}: {e}")
        
        self.logger.info(f"Found {len(all_sequences)} sequences from previous {current_round - 1} rounds")
        return all_sequences
    
    def _copy_good_structures_from_previous_round(self, prev_expanded_dir: Path, 
                                                 new_expanded_dir: Path, 
                                                 good_sequences: Dict[str, str]):
        """Copy structures that passed filters to new round directory."""
        import shutil
        
        new_expanded_dir.mkdir(parents=True, exist_ok=True)
        
        # Get subset names from good sequences
        subset_names = set()
        for header in good_sequences.keys():
            # Extract subset name from header
            if "_subset_" in header:
                subset_name = header.split("_subset_")[1].split("_")[0]
                subset_names.add(f"subset_{subset_name}")
        
        copied_count = 0
        # Copy files for each subset
        for subset_name in subset_names:
            pattern = f"*{subset_name}*"
            for file in prev_expanded_dir.glob(pattern):
                try:
                    dest = new_expanded_dir / file.name
                    shutil.copy2(file, dest)
                    copied_count += 1
                except Exception as e:
                    self.logger.debug(f"Failed to copy {file}: {e}")
        
        self.logger.info(f"Copied {copied_count} structure files from previous round")
    
    def _get_next_subset_index(self, expanded_subsets_dir: Path) -> int:
        """Get the next subset index to avoid overwriting existing files."""
        if not expanded_subsets_dir.exists():
            return 0
        
        existing_files = list(expanded_subsets_dir.glob("subset_*.a3m"))
        if not existing_files:
            return 0
        
        # Extract indices and find max
        indices = []
        for f in existing_files:
            try:
                idx = int(f.stem.split('_')[1])
                indices.append(idx)
            except:
                pass
        
        return max(indices) + 1 if indices else 0

