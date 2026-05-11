#!/usr/bin/env python3
"""
AF-ClaSeq Pipeline

A comprehensive pipeline for protein structure prediction and analysis
using AlphaFold and sequence-based sampling approaches.
"""

import os
import sys
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

# Import modules from AF-ClaSeq
from af_claseq.utils.slurm_utils import SlurmJobSubmitter
from af_claseq.utils.structure_analysis import StructureAnalyzer
from af_claseq.utils.sequence_processing import filter_a3m_by_coverage, write_a3m
from af_claseq.m_fold_sampling_voting.m_fold_sampling import MFoldSampler
from af_claseq.m_fold_sampling_voting.sequence_voting import SequenceVotingRunner, SequenceVotingPlotter
from af_claseq.m_fold_sampling_voting.sequence_recompile import SequenceRecompiler
from af_claseq.m_fold_sampling_voting.pure_seq_pred import PureSequenceAF2Prediction
from af_claseq.m_fold_sampling_voting.pure_seq_plot import PureSequencePlotter, create_pure_seq_plot_config_from_dict
from af_claseq.utils.logging_utils import setup_logger 

from af_claseq.m_fold_sampling_voting.config import load_pipeline_config


class AFClaSeqPipeline:
    """Main pipeline class for AF-ClaSeq"""
    
    def __init__(self, yaml_input: str):
        """Initialize the pipeline with YAML configuration file"""
        self.config = load_pipeline_config(yaml_input)
        self.logger = self._setup_logging()
        self.logger.info(f"Pipeline initialized with config from {yaml_input}")
        
        # Create output directories
        self._create_directories()
        
        # Initialize core components
        self.structure_analyzer = StructureAnalyzer()
        self.slurm_submitter = self._init_slurm_submitter()
        
        # Load filter configuration
        self.filter_config = self._load_filter_config()
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration"""
        # Use base_dir directly as specified in config
        base_dir = Path(self.config.general.base_dir)
        base_dir.mkdir(exist_ok=True, parents=True)

        log_dir = base_dir / "logs"
        log_dir.mkdir(exist_ok=True, parents=True)
        
        log_file = log_dir / "af_claseq_pipeline.log"
        
        # Set up the root logger for the whole package
        return setup_logger(
            name="af_claseq",  # Root logger for the package
            log_file=log_file,
            level=logging.INFO,
            propagate=False,  # Root logger doesn't propagate
            add_console_handler=True
        )
    
    def _load_filter_config(self) -> Dict[str, Any]:
        """Load filter configuration from JSON file"""
        with open(self.config.general.config_file, 'r') as f:
            return json.load(f)
    
    def _init_slurm_submitter(self) -> SlurmJobSubmitter:
        """Initialize SLURM job submitter with configuration parameters"""
        return SlurmJobSubmitter(
            conda_env_path=self.config.slurm.conda_env_path,
            slurm_account=self.config.slurm.slurm_account,
            slurm_output=self.config.slurm.slurm_output,
            slurm_error=self.config.slurm.slurm_error,
            slurm_nodes=self.config.slurm.slurm_nodes,
            slurm_gpus_per_task=self.config.slurm.slurm_gpus_per_task,
            slurm_tasks=self.config.slurm.slurm_tasks,
            slurm_cpus_per_task=self.config.slurm.slurm_cpus_per_task,
            slurm_time=self.config.slurm.slurm_time,
            slurm_partition=self.config.slurm.slurm_partition,
            check_interval=self.config.pipeline_control.check_interval,
            job_name_prefix=self.config.general.protein_name,
            num_models=self.config.general.num_models,
            random_seed=self.config.general.random_seed
        )
    
    def _create_directories(self) -> None:
        """Create necessary output directories"""
        base_dir = Path(self.config.general.base_dir)
        base_dir.mkdir(exist_ok=True, parents=True)
        
        # Create stage directories
        stages = [
            "01_m_fold_sampling",
            "02_voting",
            "03_recompile",
            "04_plots"
        ]
        
        for stage in stages:
            (base_dir / stage).mkdir(exist_ok=True)
    
    def print_welcome(self) -> None:
        """Print welcome message with pipeline information"""
        print("\n" + "="*80)
        print(" "*30 + "AF-ClaSeq Pipeline")
        print(" "*20 + "Protein Structure Prediction and Analysis")
        print("="*80)
        print(f"Protein: {self.config.general.protein_name}")
        print(f"Base directory: {self.config.general.base_dir}")
        print(f"Configuration file: {self.config.general.config_file}")
        print("="*80 + "\n")

    def _filter_metrics_to_process(self, selected_metrics: List[str]) -> List[str]:
        """
        Filter selected metrics based on metrics_to_process configuration.

        Args:
            selected_metrics: All available metrics from config

        Returns:
            Filtered list of metrics to actually process
        """
        metrics_to_process = self.config.recompile_predict.metrics_to_process

        # Backward compatibility: If not specified, empty, or None → process all
        if not metrics_to_process or len(metrics_to_process) == 0:
            self.logger.info(f"No metric filter specified, processing all metrics: {selected_metrics}")
            return selected_metrics

        # Filter metrics
        filtered_metrics = [m for m in selected_metrics if m in metrics_to_process]

        # Validation
        if len(filtered_metrics) == 0:
            self.logger.warning(
                f"None of the specified metrics {metrics_to_process} "
                f"were found in available metrics {selected_metrics}"
            )
            self.logger.warning("Falling back to processing all metrics")
            return selected_metrics

        # Log filtering results
        self.logger.info(f"Metric filter applied: {len(selected_metrics)} available → {len(filtered_metrics)} to process")
        self.logger.info(f"Processing metrics: {filtered_metrics}")

        if len(filtered_metrics) < len(selected_metrics):
            skipped = [m for m in selected_metrics if m not in filtered_metrics]
            self.logger.info(f"Skipping metrics: {skipped}")

        return filtered_metrics

    def _get_metric_colors(self, criterion_name: str) -> List[str]:
        """
        Get color configuration for a specific metric by name.

        Args:
            criterion_name: Name of the metric (e.g., "bound_rmsd")

        Returns:
            List of [initial_color, end_color]
        """
        if criterion_name == self.config.recompile_predict.metric_name_1:
            return self.config.general.metric1_color
        elif criterion_name == self.config.recompile_predict.metric_name_2:
            return self.config.general.metric2_color
        else:
            # Fallback to metric1_color
            self.logger.warning(
                f"No explicit color mapping for '{criterion_name}', "
                f"using metric1_color as fallback"
            )
            return self.config.general.metric1_color
    
    
    def run_m_fold_sampling(self) -> bool:
        """
        Stage 01_RUN: Run M-fold sampling with ultra-parallel job submission
        """
        self.logger.info("=== STAGE 01_RUN: M-FOLD SAMPLING (ULTRA-PARALLEL) ===")

        try:
            input_a3m = self.config.general.source_a3m
            if not Path(input_a3m).exists():
                self.logger.error(f"Source A3M not found: {input_a3m}")
                return False

            # Get number of rounds from configuration
            num_rounds = self.config.m_fold_sampling.rounds
            self.logger.info(f"Preparing {num_rounds} rounds for ultra-parallel job submission")

            # ===== PHASE 1: PREPARE ALL ROUNDS (NO JOB SUBMISSION) =====
            self.logger.info("Phase 1: Preparing all rounds (creating directories and A3M files)")

            all_preparation_successful = True

            for round_num in range(1, num_rounds + 1):
                self.logger.info(f"Preparing round {round_num}/{num_rounds}")

                # Create round-specific directory
                round_base_dir = Path(self.config.general.base_dir) / "01_m_fold_sampling" / f"round_{round_num}"
                round_base_dir.mkdir(exist_ok=True, parents=True)

                # Create sampler instance WITHOUT slurm_submitter (no job submission)
                sampler = MFoldSampler(
                    input_a3m=input_a3m,
                    m_fold_sampling_base_dir=str(round_base_dir),
                    group_size=self.config.m_fold_sampling.m_fold_group_size,
                    coverage_threshold=self.config.general.coverage_threshold,
                    random_select_num_seqs=self.config.m_fold_sampling.m_fold_random_select,
                    slurm_submitter=None,  # KEY: No job submission during preparation
                    random_seed=self.config.general.random_seed + round_num,  # Different seed per round
                    max_workers=self.config.slurm.max_workers,
                    logger=self.logger,
                    default_pdb=self.config.general.default_pdb
                )

                # Run preparation only (no job submission)
                prep_success = sampler.run()

                if not prep_success:
                    self.logger.error(f"Error preparing round {round_num}")
                    all_preparation_successful = False

            if not all_preparation_successful:
                self.logger.error("Failed to prepare some rounds")
                return False

            # ===== PHASE 2: COLLECT ALL SAMPLING DIRECTORIES FROM ALL ROUNDS =====
            self.logger.info("Phase 2: Collecting all sampling directories from all rounds")

            all_job_folders = []
            all_job_ids = []

            # Truncate protein name to keep job names concise (max 8 chars for protein part)
            protein_name = self.config.general.protein_name
            protein_prefix = protein_name[:8] if len(protein_name) > 8 else protein_name

            for round_num in range(1, num_rounds + 1):
                round_base_dir = Path(self.config.general.base_dir) / "01_m_fold_sampling" / f"round_{round_num}"
                sampling_dir = round_base_dir / "01_sampling"

                if not sampling_dir.exists():
                    self.logger.warning(f"Sampling directory not found for round {round_num}: {sampling_dir}")
                    continue

                # Get all sampling_XX directories in this round (sorted for consistent ordering)
                sampling_subdirs = sorted([d for d in os.listdir(sampling_dir) if d.startswith('sampling_')])

                for sampling_subdir in sampling_subdirs:
                    sampling_path = sampling_dir / sampling_subdir
                    all_job_folders.append(str(sampling_path))

                    # Create unique job ID: <protein>_r<round>_s<sampling>
                    # Example: KaiB_r1_s1, KaiB_r2_s3, Protein_r1_s10
                    sampling_num = sampling_subdir.replace('sampling_', '')
                    job_id = f"{protein_prefix}_r{round_num}_s{sampling_num}"
                    all_job_ids.append(job_id)

            if not all_job_folders:
                self.logger.error("No sampling directories found across any rounds")
                return False

            self.logger.info(f"Collected {len(all_job_folders)} sampling directories across {num_rounds} rounds")
            self.logger.info(f"Average sampling directories per round: {len(all_job_folders) / num_rounds:.1f}")

            # ===== PHASE 3: SUBMIT ALL JOBS AT ONCE (LIMITED BY MAX_WORKERS) =====
            max_workers = self.config.slurm.max_workers
            self.logger.info(f"Phase 3: Submitting ALL {len(all_job_folders)} jobs to SLURM")
            self.logger.info(f"Concurrent worker limit: {max_workers} jobs at a time")

            if len(all_job_folders) > max_workers:
                batches = (len(all_job_folders) + max_workers - 1) // max_workers
                self.logger.info(f"Jobs will be processed in approximately {batches} batches")

            self.slurm_submitter.process_folders_concurrently(
                folders=all_job_folders,
                job_ids=all_job_ids,
                max_workers=max_workers
            )

            self.logger.info(f"✓ Completed all {num_rounds} rounds of M-fold sampling with ultra-parallel submission")
            self.logger.info(f"✓ Total jobs processed: {len(all_job_folders)}")
            return True

        except Exception as e:
            self.logger.error(f"Error in M-fold sampling: {str(e)}", exc_info=True)
            return False
        
    def plot_m_fold_sampling(self) -> bool:
        """
        Stage 01_ANALYSIS: Analyze results of M-fold sampling
        """
        self.logger.info("=== STAGE 01_ANALYSIS: M-FOLD SAMPLING ANALYSIS ===")

        try:
            # Import plotting functions directly
            from af_claseq.utils.plotting_manager import (
                plot_m_fold_sampling_1d,
                plot_m_fold_sampling_2d,
                load_results_df,
            )

            # Setup directories
            base_dir = Path(self.config.general.base_dir)
            output_dir = base_dir / "01_m_fold_sampling/plot"
            csv_dir = base_dir / "01_m_fold_sampling/csv"
            
            # Create output directories
            output_dir.mkdir(parents=True, exist_ok=True)
            csv_dir.mkdir(parents=True, exist_ok=True)

            # Load filter criteria from config file
            with open(self.config.general.config_file, 'r') as f:
                config = json.load(f)

            filter_criteria = config.get('filter_criteria', [])
            
            # Get selected metrics using the new explicit metric selection system  
            from af_claseq.m_fold_sampling_voting.config import get_selected_metrics
            selected_metrics = get_selected_metrics(self.config.general)
            num_criteria = len(selected_metrics)
            
            # Get number of rounds from configuration
            num_rounds = self.config.m_fold_sampling.rounds
            self.logger.info(f"Analyzing data from {num_rounds} rounds of M-fold sampling")
            
            # Gather all results directories from all rounds
            all_results_dirs = []
            for round_num in range(1, num_rounds + 1):
                round_dir = base_dir / "01_m_fold_sampling" / f"round_{round_num}"
                if round_dir.exists():
                    all_results_dirs.append(round_dir)
                    self.logger.info(f"Including round {round_num} in analysis")
            
            if not all_results_dirs:
                self.logger.error("No M-fold sampling round directories found")
                return False
            
            if num_criteria == 1:
                # Only one criterion - use 1D plots
                self.logger.info("One filter criterion detected - generating 1D plots")
                # Get metric name using the new explicit metric selection system
                from af_claseq.m_fold_sampling_voting.config import get_selected_metrics
                metric_names = get_selected_metrics(self.config.general)
                metric_name = metric_names[0] if metric_names else filter_criteria[0].get('name', 'criterion_0')
                
                # Create metric colors dictionary
                metric_colors = {
                    metric_name: self.config.general.metric1_color
                }

                from af_claseq.m_fold_sampling_voting.config import get_metric_bin_config
                mbc = get_metric_bin_config(self.config.general, metric_name)
                n_bins = mbc.compute_num_bins() if mbc is not None and mbc.bin_width is not None else 30
                x_label = self.config.general.metric1_label or metric_name

                plot_m_fold_sampling_1d(
                    results_dir=all_results_dirs,
                    metric_name=metric_name,
                    output_dir=output_dir,
                    csv_dir=csv_dir,
                    config_file=self.config.general.config_file,
                    initial_color=self.config.general.metric1_color[0],
                    end_color=self.config.general.metric1_color[1],
                    x_min=self.config.m_fold_sampling.m_fold_metric1_min,
                    x_max=self.config.m_fold_sampling.m_fold_metric1_max,
                    y_min=self.config.m_fold_sampling.m_fold_count_min,
                    y_max=self.config.m_fold_sampling.m_fold_count_max,
                    x_ticks=self.config.m_fold_sampling.m_fold_metric1_ticks,
                    log_scale=self.config.m_fold_sampling.m_fold_log_scale,
                    n_plot_bins=n_bins,
                    gradient_ascending=self.config.m_fold_sampling.m_fold_gradient_ascending,
                    linear_gradient=self.config.m_fold_sampling.m_fold_linear_gradient,
                    plddt_threshold=self.config.m_fold_sampling.m_fold_plddt_threshold,
                    figsize=self.config.m_fold_sampling.m_fold_figsize,
                    show_bin_lines=self.config.m_fold_sampling.m_fold_show_bin_lines,
                    logger=self.logger,
                    metric_colors=metric_colors,
                    x_label=x_label,
                )

            elif num_criteria == 2:
                self.logger.info("Two filter criteria detected - generating 1D plots for each criterion and 2D plots")

                from af_claseq.m_fold_sampling_voting.config import get_selected_metrics
                metric_names = get_selected_metrics(self.config.general)

                metric_colors = {
                    metric_names[0]: self.config.general.metric1_color,
                    metric_names[1]: self.config.general.metric2_color
                }

                self.logger.info(f"Computing metrics for {metric_names} in a single pass")
                combined_df = load_results_df(
                    results_dir=all_results_dirs,
                    metric_names=metric_names,
                    csv_dir=str(csv_dir),
                    config_file=self.config.general.config_file,
                    plddt_threshold=self.config.m_fold_sampling.m_fold_plddt_threshold,
                    logger=self.logger,
                )

                from af_claseq.m_fold_sampling_voting.config import get_metric_bin_config
                for i, metric_name in enumerate(metric_names):
                    self.logger.info(f"Generating 1D plot for {metric_name}")
                    if i == 0:
                        metric_min = self.config.m_fold_sampling.m_fold_metric1_min
                        metric_max = self.config.m_fold_sampling.m_fold_metric1_max
                        metric_ticks = self.config.m_fold_sampling.m_fold_metric1_ticks
                        colors = self.config.general.metric1_color
                        x_label = self.config.general.metric1_label or metric_name
                    else:
                        metric_min = self.config.m_fold_sampling.m_fold_metric2_min
                        metric_max = self.config.m_fold_sampling.m_fold_metric2_max
                        metric_ticks = self.config.m_fold_sampling.m_fold_metric2_ticks
                        colors = self.config.general.metric2_color
                        x_label = self.config.general.metric2_label or metric_name

                    mbc = get_metric_bin_config(self.config.general, metric_name)
                    n_bins = mbc.compute_num_bins() if mbc is not None and mbc.bin_width is not None else 30

                    plot_m_fold_sampling_1d(
                        results_dir=all_results_dirs,
                        metric_name=metric_name,
                        output_dir=output_dir,
                        csv_dir=csv_dir,
                        config_file=self.config.general.config_file,
                        initial_color=colors[0],
                        end_color=colors[1],
                        x_min=metric_min,
                        x_max=metric_max,
                        y_min=self.config.m_fold_sampling.m_fold_count_min,
                        y_max=self.config.m_fold_sampling.m_fold_count_max,
                        x_ticks=metric_ticks,
                        log_scale=self.config.m_fold_sampling.m_fold_log_scale,
                        n_plot_bins=n_bins,
                        gradient_ascending=self.config.m_fold_sampling.m_fold_gradient_ascending,
                        linear_gradient=self.config.m_fold_sampling.m_fold_linear_gradient,
                        plddt_threshold=self.config.m_fold_sampling.m_fold_plddt_threshold,
                        figsize=self.config.m_fold_sampling.m_fold_figsize,
                        show_bin_lines=self.config.m_fold_sampling.m_fold_show_bin_lines,
                        logger=self.logger,
                        metric_colors=metric_colors,
                        x_label=x_label,
                        results_df=combined_df,
                    )

                self.logger.info(f"Generating 2D plot for combined criteria: {metric_names[0]} vs {metric_names[1]}")
                plot_m_fold_sampling_2d(
                    results_dir=all_results_dirs,
                    metric_name1=metric_names[0],
                    metric_name2=metric_names[1],
                    output_dir=output_dir,
                    csv_dir=csv_dir,
                    config_file=self.config.general.config_file,
                    x_min=self.config.m_fold_sampling.m_fold_metric1_min,
                    x_max=self.config.m_fold_sampling.m_fold_metric1_max,
                    y_min=self.config.m_fold_sampling.m_fold_metric2_min,
                    y_max=self.config.m_fold_sampling.m_fold_metric2_max,
                    x_ticks=self.config.m_fold_sampling.m_fold_metric1_ticks,
                    y_ticks=self.config.m_fold_sampling.m_fold_metric2_ticks,
                    plddt_threshold=self.config.m_fold_sampling.m_fold_plddt_threshold,
                    logger=self.logger,
                    x_label=self.config.general.metric1_label,
                    y_label=self.config.general.metric2_label,
                    results_df=combined_df,
                )

            elif num_criteria > 2:
                # More than two criteria - not supported
                self.logger.error(f"Found {num_criteria} filter criteria. Cannot plot more than 2 dimensions.")
                self.logger.error("Please modify your config.json to include only 1 or 2 criteria for visualization.")
                return False

            else:
                # No criteria found
                self.logger.error("No filter criteria found in config file")
                return False

            self.logger.info("Completed M-fold sampling analysis and plotting successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error in M-fold sampling analysis: {str(e)}", exc_info=True)
            return False
    def run_sequence_voting(self) -> bool:
        """
        Stage 02: Run sequence voting analysis
        """
        self.logger.info("=== STAGE 02: SEQUENCE VOTING ===")

        try:
            # Load filter criteria from config file
            with open(self.config.general.config_file, 'r') as f:
                filter_config = json.load(f)

            filter_criteria = filter_config.get('filter_criteria', [])
            if not filter_criteria:
                self.logger.error("No filter criteria found in config file")
                return False

            # Create base output directory
            base_dir = Path(self.config.general.base_dir)
            voting_dir = base_dir / "02_voting"
            voting_dir.mkdir(exist_ok=True)
            
            # Resolve symlinks so PDB paths match precomputed CSVs
            m_fold_sampling_dir = Path(os.path.realpath(base_dir / "01_m_fold_sampling"))
            
            results_files = []
            
            # Get selected metrics using the new explicit metric selection system  
            from af_claseq.m_fold_sampling_voting.config import get_selected_metrics
            selected_metrics = get_selected_metrics(self.config.general)
            
            if not selected_metrics:
                # Fall back to all filter criteria if no explicit selection
                selected_metrics = [criterion.get('name') for criterion in filter_criteria if criterion.get('name')]
            
            # Process each selected metric separately
            for i, criterion_name in enumerate(selected_metrics):
                if not criterion_name:
                    self.logger.warning("Invalid metric name found, skipping")
                    continue
                
                self.logger.info(f"Processing selected metric: {criterion_name}")

                # Create criterion-specific output directory
                criterion_output_dir = voting_dir / criterion_name
                criterion_output_dir.mkdir(exist_ok=True)

                # Resolve per-metric bin config (unit-based binning)
                from af_claseq.m_fold_sampling_voting.config import get_metric_bin_config
                metric_bin_cfg = get_metric_bin_config(self.config.general, criterion_name)

                # Create voting runner instance for this criterion
                voting_runner = SequenceVotingRunner(
                    sampling_dir=m_fold_sampling_dir,
                    source_msa=self.config.general.source_a3m,
                    config_path=self.config.general.config_file,
                    output_dir=criterion_output_dir,
                    num_bins=30,
                    max_workers=self.config.slurm.max_workers,
                    vote_threshold=self.config.sequence_voting.vote_threshold,
                    min_value=self.config.sequence_voting.vote_min_value,
                    max_value=self.config.sequence_voting.vote_max_value,
                    use_focused_bins=self.config.sequence_voting.use_focused_bins,
                    precomputed_metrics=str(os.path.realpath(base_dir / "01_m_fold_sampling/csv")),
                    plddt_threshold=self.config.m_fold_sampling.m_fold_plddt_threshold,
                    filter_criterion=criterion_name,
                    metric_bin_cfg=metric_bin_cfg,
                )

                # Run voting analysis for this criterion
                results_file = voting_runner.run()

                if results_file:
                    results_files.append((criterion_name, results_file))

                    # Determine total bin count from config (not from CSV max, which is only the highest occupied bin)
                    if metric_bin_cfg is not None and metric_bin_cfg.bin_width is not None:
                        actual_num_bins = metric_bin_cfg.compute_num_bins()
                    else:
                        actual_num_bins = 30

                    # Create plotter for visualization (x-axis = bin index)
                    colors = self.config.general.metric1_color if i == 0 else self.config.general.metric2_color

                    plotter = SequenceVotingPlotter(
                        results_file=results_file,
                        output_dir=criterion_output_dir,
                        initial_color=colors[0],
                        end_color=colors[1],
                        figsize=self.config.sequence_voting.vote_figsize,
                        y_min=self.config.sequence_voting.vote_y_min,
                        y_max=self.config.sequence_voting.vote_y_max,
                        x_ticks=self.config.sequence_voting.vote_x_ticks,
                        num_bins=actual_num_bins,
                    )

                    # Plot voting distributions
                    plotter.plot()
                else:
                    self.logger.error(f"Sequence voting failed to produce results for criterion: {criterion_name}")

            if results_files:
                self.logger.info(f"Completed sequence voting successfully for {len(results_files)} criteria")
                return True
            else:
                self.logger.error("No sequence voting results were produced")
                return False

        except Exception as e:
            self.logger.error(f"Error in sequence voting: {str(e)}", exc_info=True)
            return False
    
    def run_recompile_and_predict(self) -> bool:
        """
        Stage 03: Recompile sequences and run structure prediction
        """
        self.logger.info("=== STAGE 03: SEQUENCE RECOMPILATION & PREDICTION ===")
        
        try:
            # Load filter criteria from config file
            with open(self.config.general.config_file, 'r') as f:
                filter_config = json.load(f)
            
            filter_criteria = filter_config.get('filter_criteria', [])
            if not filter_criteria:
                self.logger.error("No filter criteria found in config file")
                return False
            
            # Create base output directory
            base_dir = Path(self.config.general.base_dir)
            base_output_dir = base_dir / "03_recompile"
            base_output_dir.mkdir(exist_ok=True)
            
            # Determine input MSA
            source_msa = self.config.general.source_a3m
            
            all_successful = True
            
            # Get selected metrics using the new explicit metric selection system
            from af_claseq.m_fold_sampling_voting.config import get_selected_metrics
            all_available_metrics = get_selected_metrics(self.config.general)

            if not all_available_metrics:
                # Fall back to all filter criteria if no explicit selection
                all_available_metrics = [criterion.get('name') for criterion in filter_criteria if criterion.get('name')]

            # Apply metric filtering based on metrics_to_process
            selected_metrics = self._filter_metrics_to_process(all_available_metrics)

            # Process each selected metric separately
            for criterion_name in selected_metrics:
                if not criterion_name:
                    self.logger.warning("Invalid metric name found, skipping")
                    continue

                self.logger.info(f"Processing recompilation and prediction for: {criterion_name}")

                # Create criterion-specific output directory
                criterion_output_dir = base_output_dir / criterion_name
                criterion_output_dir.mkdir(exist_ok=True)

                # Get voting results for this criterion
                voting_results = base_dir / f"02_voting/{criterion_name}/voting_results.csv"
                raw_votes_json = base_dir / f"02_voting/{criterion_name}/raw_sequence_votes.json"

                if not voting_results.exists():
                    self.logger.error(f"Voting results not found for criterion: {criterion_name}")
                    all_successful = False
                    continue

                # Determine bin numbers by matching metric name
                bin_numbers = None
                if criterion_name == self.config.recompile_predict.metric_name_1:
                    bin_numbers = self.config.recompile_predict.bin_numbers_1
                elif criterion_name == self.config.recompile_predict.metric_name_2:
                    bin_numbers = self.config.recompile_predict.bin_numbers_2
                else:
                    # Fallback to bin_numbers_1
                    self.logger.warning(f"No bin numbers specified for '{criterion_name}', using bin_numbers_1")
                    bin_numbers = self.config.recompile_predict.bin_numbers_1

                # If bin_numbers not specified, report error and skip this criterion
                if not bin_numbers:
                    self.logger.error(f"No bin numbers specified for criterion: {criterion_name}. Please specify bin_numbers_1 in the recompile_predict section of your configuration file.")
                    all_successful = False
                    continue

                # Determine total bin count from config
                from af_claseq.m_fold_sampling_voting.config import get_metric_bin_config
                mbc = get_metric_bin_config(self.config.general, criterion_name)
                if mbc is not None and mbc.bin_width is not None:
                    actual_num_bins = mbc.compute_num_bins()
                else:
                    actual_num_bins = 30

                # Get metric colors by name (not index)
                colors = self._get_metric_colors(criterion_name)

                recompiler = SequenceRecompiler(
                    output_dir=criterion_output_dir,
                    source_msa=source_msa,
                    voting_results=voting_results,
                    bin_numbers=bin_numbers,
                    num_total_bins=actual_num_bins,
                    initial_color=colors[0],
                    combine_bins=self.config.recompile_predict.combine_bins,
                    raw_votes_json=raw_votes_json if raw_votes_json.exists() else None,
                    logger=self.logger,
                    default_pdb=self.config.general.default_pdb
                )
                
                # Recompile sequences
                recompiler.recompile_sequences()
                
                # Create prediction configuration
                prediction_config = {
                    'pure_seq_pred_base_dir': criterion_output_dir,
                    'bin_numbers': bin_numbers,
                    'combine_bins': self.config.recompile_predict.combine_bins,
                    'conda_env_path': self.config.slurm.conda_env_path,
                    'slurm_account': self.config.slurm.slurm_account,
                    'slurm_output': self.config.slurm.slurm_output,
                    'slurm_error': self.config.slurm.slurm_error,
                    'slurm_nodes': self.config.slurm.slurm_nodes,
                    'slurm_gpus_per_task': self.config.slurm.slurm_gpus_per_task,
                    'slurm_tasks': self.config.slurm.slurm_tasks,
                    'slurm_cpus_per_task': self.config.slurm.slurm_cpus_per_task,
                    'slurm_time': self.config.slurm.slurm_time,
                    'slurm_partition': self.config.slurm.slurm_partition,
                    'prediction_num_model': self.config.recompile_predict.prediction_num_model,
                    'prediction_num_seed': self.config.recompile_predict.prediction_num_seed,
                    'check_interval': self.config.pipeline_control.check_interval,
                    'max_workers': self.config.slurm.max_workers,
                    'job_name_prefix': f"{self.config.general.protein_name}_{criterion_name}"
                }
                
                # Create and run predictor
                predictor = PureSequenceAF2Prediction(
                    config=prediction_config,
                    logger=self.logger
                )
                
                result = predictor.run()
                
                if not result:
                    self.logger.error(f"Error in prediction process for criterion: {criterion_name}")
                    all_successful = False
            
            if all_successful:
                self.logger.info("Completed recompilation and prediction successfully for all criteria")
                return True
            else:
                self.logger.warning("Recompilation and prediction completed with some errors")
                return False
            
        except Exception as e:
            self.logger.error(f"Error in recompilation and prediction: {str(e)}", exc_info=True)
            return False
        
    def run_pure_sequence_plotting(self) -> bool:
        """
        Stage 04: Plot and analyze prediction results
        """
        self.logger.info("=== STAGE 04: PURE SEQUENCE PREDICTION PLOTTING ===")
        
        try:
            # Load filter criteria from config file
            with open(self.config.general.config_file, 'r') as f:
                filter_config = json.load(f)
            
            filter_criteria = filter_config.get('filter_criteria', [])
            if not filter_criteria:
                self.logger.error("No filter criteria found in config file")
                return False
            
            # Create base output directory
            base_dir = Path(self.config.general.base_dir)
            base_output_dir = os.path.join(self.config.general.base_dir, "04_plots")
            os.makedirs(base_output_dir, exist_ok=True)

            all_successful = True

            # Get selected metrics using the new explicit metric selection system
            from af_claseq.m_fold_sampling_voting.config import get_selected_metrics
            all_available_metrics = get_selected_metrics(self.config.general)

            if not all_available_metrics:
                # Fall back to all filter criteria if no explicit selection
                all_available_metrics = [criterion.get('name') for criterion in filter_criteria if criterion.get('name')]

            # Apply metric filtering based on metrics_to_process
            selected_metrics = self._filter_metrics_to_process(all_available_metrics)

            # Process each selected metric separately
            for criterion_name in selected_metrics:
                if not criterion_name:
                    self.logger.warning("Invalid metric name found, skipping")
                    continue
                
                self.logger.info(f"Processing plots for selected metric: {criterion_name}")
                
                # Create criterion-specific output directory
                criterion_output_dir = os.path.join(base_output_dir, criterion_name)
                os.makedirs(criterion_output_dir, exist_ok=True)
                
                # Get the recompile directory for this criterion as the base dir for plotting
                recompile_dir = os.path.join(self.config.general.base_dir, f"03_recompile/{criterion_name}")
                
                if not os.path.exists(recompile_dir):
                    self.logger.error(f"Recompile directory not found for criterion: {recompile_dir}")
                    all_successful = False
                    continue
                
                # Create plotting configuration
                # Get metric colors by name (not index)
                colors = self._get_metric_colors(criterion_name)

                plot_config = {
                    'base_dir': recompile_dir,
                    'output_dir': criterion_output_dir,
                    'config_file': self.config.general.config_file,
                    'color_prediction': colors[0],
                    'color_control': colors[1],
                    'metric1_min': self.config.pure_sequence_plotting.metric1_min,
                    'metric1_max': self.config.pure_sequence_plotting.metric1_max,
                    'metric2_min': self.config.pure_sequence_plotting.metric2_min,
                    'metric2_max': self.config.pure_sequence_plotting.metric2_max,
                    'metric1_ticks': self.config.pure_sequence_plotting.metric1_ticks,
                    'metric2_ticks': self.config.pure_sequence_plotting.metric2_ticks,
                    'plddt_threshold': self.config.pure_sequence_plotting.plddt_threshold,
                    'figsize': self.config.pure_sequence_plotting.figsize,
                    'dpi': self.config.pure_sequence_plotting.dpi,
                    'max_workers': self.config.slurm.max_workers,
                    'metric1_label': self.config.general.metric1_label,
                    'metric2_label': self.config.general.metric2_label,
                }
                
                # Create and run plotter
                plotter = PureSequencePlotter(
                    config=create_pure_seq_plot_config_from_dict(plot_config),
                    logger=self.logger
                )
                
                result = plotter.run()
                
                if not result:
                    self.logger.error(f"Error in plotting process for criterion: {criterion_name}")
                    all_successful = False
            
            if all_successful:
                self.logger.info("Completed pure sequence plotting successfully for all criteria")
                return True
            else:
                self.logger.warning("Pure sequence plotting completed with some errors")
                return False
            
        except Exception as e:
            self.logger.error(f"Error in pure sequence plotting: {str(e)}", exc_info=True)
            return False
        
    def run(self) -> bool:
        """Run the pipeline with selected stages. Returns True on success."""
        self.print_welcome()

        self.logger.info("=== AF-ClaSeq PIPELINE STARTED ===")
        self.logger.info(f"Configuration loaded from YAML file")

        stages_to_run = self.config.pipeline_control.stages

        try:

            # Stage 01: M-fold Sampling
            if "01_M_FOLD_SAMPLING_RUN" in stages_to_run:
                if not self.run_m_fold_sampling():
                    self.logger.error("Stopping pipeline due to failure in stage 01_M_FOLD_SAMPLING_RUN")
                    return False

            if "01_M_FOLD_SAMPLING_PLOT" in stages_to_run:
                if not self.plot_m_fold_sampling():
                    self.logger.error("Stopping pipeline due to failure in stage 01_M_FOLD_SAMPLING_PLOT")
                    return False

            # Stage 02: Sequence Voting
            if "02_VOTING_RUN" in stages_to_run:
                if not self.run_sequence_voting():
                    self.logger.error("Stopping pipeline due to failure in stage 02_VOTING_RUN")
                    return False

            # Stage 03: Recompilation & Prediction
            if "03_RECOMPILE_PREDICT_RUN" in stages_to_run:
                if not self.run_recompile_and_predict():
                    self.logger.error("Stopping pipeline due to failure in stage 03_RECOMPILE_PREDICT_RUN")
                    return False

            # Stage 04: Pure Sequence Plotting
            if "04_PURE_SEQ_PLOT_RUN" in stages_to_run:
                if not self.run_pure_sequence_plotting():
                    self.logger.error("Stopping pipeline due to failure in stage 04_PURE_SEQ_PLOT_RUN")
                    return False

            self.logger.info("=== AF-ClaSEQ PIPELINE COMPLETED SUCCESSFULLY ===")
            self.logger.info("All requested stages executed without errors")
            self.logger.info("Results are ready for analysis in the output directories")
            return True

        except Exception as e:
            self.logger.error(f"Unhandled error in pipeline: {str(e)}", exc_info=True)
            return False


def main():
    """Main entry point for the pipeline"""
    if len(sys.argv) != 2:
        print("Usage: python run_m_fold_sampling_voting.py <config.yaml>")
        sys.exit(1)
    
    yaml_input = sys.argv[1]
    
    if not os.path.exists(yaml_input):
        print(f"Error: Config file not found: {yaml_input}")
        sys.exit(1)
    
    # Initialize and run the pipeline
    pipeline = AFClaSeqPipeline(yaml_input)
    success = pipeline.run()
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
