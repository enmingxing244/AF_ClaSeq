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
from pathlib import Path
from typing import Dict, Any

# Import modules from AF-ClaSeq
from af_claseq.utils.slurm_utils import SlurmJobSubmitter
from af_claseq.utils.structure_analysis import StructureAnalyzer
from af_claseq.utils.sequence_processing import A3MParser, filter_a3m_by_coverage, write_a3m
from af_claseq.pipeline.init_bootstrapping import InitBootstrappingRunner
from af_claseq.pipeline.hit_expand import HitExpandRunner
from af_claseq.pipeline.m_fold_sampling import MFoldSampler
from af_claseq.pipeline.sequence_voting import SequenceVotingRunner, SequenceVotingPlotter
from af_claseq.pipeline.sequence_recompile import SequenceRecompiler
from af_claseq.pipeline.pure_seq_pred import PureSequenceAF2Prediction
from af_claseq.pipeline.pure_seq_plot import PureSequencePlotter, create_pure_seq_plot_config_from_dict
from af_claseq.utils.logging_utils import setup_logger 

from af_claseq.pipeline.config import load_pipeline_config


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
        base_dir = Path(self.config.general.base_dir)
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
            "00_init_bootstrapping",
            "01_hit_expand",
            "02_m_fold_sampling",
            "03_voting",
            "04_recompile",
            "05_plots"
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
    
    def run_init_bootstrapping(self) -> bool:
        """
        Stage 00: Run init bootstrapping for quick preview
        """
        self.logger.info("=== STAGE 00: INIT BOOTSTRAPPING ===")
        
        try:
            # Use init_bootstrapping config
            init_config = self.config.init_bootstrapping
            
            # Determine input MSA (use general.source_a3m)
            input_msa = Path(self.config.general.source_a3m)
            if not input_msa.exists():
                self.logger.error(f"Input MSA not found: {input_msa}")
                return False
            
            # Create runner instance
            runner = InitBootstrappingRunner(
                config=init_config,
                slurm_submitter=self.slurm_submitter,
                base_dir=Path(self.config.general.base_dir) / "00_init_bootstrapping",
                input_msa=input_msa,
                logger=self.logger
            )
            
            # Run the init bootstrapping process
            success = runner.run()
            
            if success:
                self.logger.info("Completed init bootstrapping successfully")
                self.logger.info("Review results before running full pipeline")
                return True
            else:
                self.logger.error("Init bootstrapping failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error in init bootstrapping: {str(e)}", exc_info=True)
            return False
    
    def run_hit_expand(self) -> bool:
        """
        Stage 01_RUN: Run hit expand pipeline
        """
        self.logger.info("=== STAGE 01_RUN: HIT EXPAND ===")
        
        try:
            # Use hit_expand config directly (it's already a HitExpandConfig object)
            hit_expand_config = self.config.hit_expand
            
            # Set input_msa if not provided (use general.source_a3m as fallback)
            if not hit_expand_config.input_msa:
                hit_expand_config.input_msa = self.config.general.source_a3m
                
            # Create runner instance
            runner = HitExpandRunner(
                config=hit_expand_config,
                slurm_submitter=self.slurm_submitter,
                base_dir=Path(self.config.general.base_dir) / "01_hit_expand",
                config_file=self.config.general.config_file,
                logger=self.logger
            )
            
            # Set up logging and log parameters
            runner.setup_logging()
            runner.log_parameters()
            
            # Run the hit expand process
            final_msa = runner.run()
            
            if final_msa is None:
                self.logger.error("Hit expand failed to produce output")
                return False
                
            self.logger.info(f"Completed hit expand successfully. Final output: {final_msa}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in hit expand: {str(e)}", exc_info=True)
            return False
    
    def analyze_hit_expand(self) -> bool:
        """
        Stage 01_ANALYSIS: Analyze results of hit expand
        """
        self.logger.info("=== STAGE 01_ANALYSIS: HIT EXPAND ANALYSIS ===")
        try:
            # Use existing plotting utilities from af_claseq.utils.plotting_manager
            from af_claseq.utils.plotting_manager import (
                create_2d_scatter_plot,
                load_results_df
            )
            
            # Find final MSA output and structure analysis results
            hit_expand_dir = Path(self.config.general.base_dir) / "01_hit_expand"
            final_msa = hit_expand_dir / "hit_expand_final_msa.a3m"
            
            # Check for alternative MSA locations
            if not final_msa.exists():
                alternative_locations = [
                    hit_expand_dir / "02_similarity_search" / "expanded_sequences.a3m",
                    hit_expand_dir / "05_structure_analysis" / "filtered_sequences.a3m",
                    hit_expand_dir / "expanded_sequences.a3m"
                ]
                
                for alt_location in alternative_locations:
                    if alt_location.exists():
                        final_msa = alt_location
                        self.logger.info(f"Found MSA at alternative location: {alt_location}")
                        break
            
            # Look for structure analysis results
            analysis_results_file = hit_expand_dir / "05_structure_analysis" / "structure_analysis_results.json"
            
            if analysis_results_file.exists():
                # Load and visualize structure analysis results
                with open(analysis_results_file, 'r') as f:
                    analysis_data = json.load(f)
                
                plots_dir = hit_expand_dir / "plots"
                plots_dir.mkdir(exist_ok=True)
                
                # Create basic analysis plots using existing utilities
                self.logger.info("Creating hit expand analysis visualizations")
                
                # Simple summary of results
                all_results = analysis_data.get("all_results", {})
                filtered_results = analysis_data.get("filtered_results", {})
                
                self.logger.info(f"Structure analysis summary:")
                self.logger.info(f"  Total structures analyzed: {len(all_results)}")
                self.logger.info(f"  Structures passing filters: {len(filtered_results)}")
                
                if len(filtered_results) > 0:
                    success_rate = len(filtered_results) / len(all_results) * 100
                    self.logger.info(f"  Success rate: {success_rate:.1f}%")
                
                saved_plots = {"summary": "Analysis completed"}
                
            elif final_msa.exists():
                # Just report on MSA if no structure analysis available
                from af_claseq.utils.sequence_processing import count_sequences_in_a3m
                seq_count = count_sequences_in_a3m(str(final_msa))
                
                self.logger.info(f"Hit expand MSA summary:")
                self.logger.info(f"  Final MSA: {final_msa}")
                self.logger.info(f"  Total sequences: {seq_count}")
                
                saved_plots = {"msa_summary": f"{seq_count} sequences"}
                
            else:
                self.logger.warning("No final MSA file found for analysis")
                self.logger.info(f"Searched in: {hit_expand_dir}")
                saved_plots = {}
            
            # Log results
            if saved_plots:
                self.logger.info(f"Hit expand analysis completed: {saved_plots}")
            
            self.logger.info("Completed hit expand analysis successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in hit expand analysis: {str(e)}", exc_info=True)
            return False
    
    
    def run_m_fold_sampling(self) -> bool:
        """
        Stage 02_RUN: Run M-fold sampling
        """
        self.logger.info("=== STAGE 02_RUN: M-FOLD SAMPLING ===")
        
        try:
            # Determine input A3M file
            input_a3m = self.config.m_fold_sampling.m_fold_samp_input_a3m
            if not Path(input_a3m).exists():
                # First try hit_expand output
                hit_expand_output = Path(self.config.general.base_dir) / "01_hit_expand/hit_expand_final_msa.a3m"
                if hit_expand_output.exists():
                    input_a3m = str(hit_expand_output)
                    self.logger.info(f"Using hit expand output as input: {input_a3m}")
            
            # Get number of rounds from configuration
            num_rounds = self.config.m_fold_sampling.rounds
            self.logger.info(f"Running M-fold sampling for {num_rounds} rounds")
            
            all_successful = True
            
            # Run M-fold sampling for each round
            for round_num in range(1, num_rounds + 1):
                self.logger.info(f"Starting M-fold sampling round {round_num}/{num_rounds}")
                
                # Create round-specific directory
                round_base_dir = Path(self.config.general.base_dir) / "02_m_fold_sampling" / f"round_{round_num}"
                round_base_dir.mkdir(exist_ok=True, parents=True)
                
                # Create sampler instance for this round
                sampler = MFoldSampler(
                    input_a3m=input_a3m,
                    default_pdb=self.config.general.default_pdb,
                    m_fold_sampling_base_dir=str(round_base_dir),
                    group_size=self.config.m_fold_sampling.m_fold_group_size,
                    coverage_threshold=self.config.general.coverage_threshold,
                    random_select_num_seqs=self.config.m_fold_sampling.m_fold_random_select,
                    slurm_submitter=self.slurm_submitter,
                    random_seed=self.config.general.random_seed + round_num,  # Use different seed for each round
                    max_workers=self.config.slurm.max_workers,
                    logger=self.logger
                )
                
                # Run the sampling process for this round
                round_success = sampler.run()
                
                if not round_success:
                    self.logger.error(f"Error in M-fold sampling round {round_num}")
                    all_successful = False
            
            if all_successful:
                self.logger.info(f"Completed all {num_rounds} rounds of M-fold sampling successfully")
            else:
                self.logger.warning(f"Completed M-fold sampling with some errors")
                
            return all_successful
            
        except Exception as e:
            self.logger.error(f"Error in M-fold sampling: {str(e)}", exc_info=True)
            return False
        
    def plot_m_fold_sampling(self) -> bool:
        """
        Stage 02_ANALYSIS: Analyze results of M-fold sampling
        """
        self.logger.info("=== STAGE 02_ANALYSIS: M-FOLD SAMPLING ANALYSIS ===")

        try:
            # Import plotting functions directly
            from af_claseq.utils.plotting_manager import (
                plot_m_fold_sampling_1d,
                plot_m_fold_sampling_2d
            )

            # Setup directories
            base_dir = Path(self.config.general.base_dir)
            output_dir = base_dir / "02_m_fold_sampling/plot"
            csv_dir = base_dir / "02_m_fold_sampling/csv"
            
            # Create output directories
            output_dir.mkdir(parents=True, exist_ok=True)
            csv_dir.mkdir(parents=True, exist_ok=True)

            # Load filter criteria from config file
            with open(self.config.general.config_file, 'r') as f:
                config = json.load(f)

            filter_criteria = config.get('filter_criteria', [])
            num_criteria = len(filter_criteria)
            
            # Get number of rounds from configuration
            num_rounds = self.config.m_fold_sampling.rounds
            self.logger.info(f"Analyzing data from {num_rounds} rounds of M-fold sampling")
            
            # Gather all results directories from all rounds
            all_results_dirs = []
            for round_num in range(1, num_rounds + 1):
                round_dir = base_dir / "02_m_fold_sampling" / f"round_{round_num}"
                if round_dir.exists():
                    all_results_dirs.append(round_dir)
                    self.logger.info(f"Including round {round_num} in analysis")
            
            if not all_results_dirs:
                self.logger.error("No M-fold sampling round directories found")
                return False
            
            if num_criteria == 1:
                # Only one criterion - use 1D plots
                self.logger.info("One filter criterion detected - generating 1D plots")
                metric_name = filter_criteria[0].get('name', 'criterion_0')
                
                # Create metric colors dictionary
                metric_colors = {
                    metric_name: self.config.general.metric1_color
                }

                plot_m_fold_sampling_1d(
                    results_dir=all_results_dirs,  # Pass list of all round directories
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
                    n_plot_bins=self.config.general.num_bins,
                    gradient_ascending=self.config.m_fold_sampling.m_fold_gradient_ascending,
                    linear_gradient=self.config.m_fold_sampling.m_fold_linear_gradient,
                    plddt_threshold=self.config.m_fold_sampling.m_fold_plddt_threshold,
                    figsize=self.config.m_fold_sampling.m_fold_figsize,
                    show_bin_lines=self.config.m_fold_sampling.m_fold_show_bin_lines,
                    logger=self.logger,
                    metric_colors=metric_colors
                )

            elif num_criteria == 2:
                # Two criteria - generate both 1D plots for each criterion and 2D plots
                self.logger.info("Two filter criteria detected - generating 1D plots for each criterion and 2D plots")

                # Get metric names
                metric_names = [criterion.get('name', f'criterion_{i}') for i, criterion in enumerate(filter_criteria)]
                
                # Create metric colors dictionary mapping metric names to their colors
                metric_colors = {
                    metric_names[0]: self.config.general.metric1_color,
                    metric_names[1]: self.config.general.metric2_color
                }

                # Generate 1D plots for each criterion
                for i, metric_name in enumerate(metric_names):
                    self.logger.info(f"Generating 1D plot for {metric_name}")
                    # Use metric1 parameters for first criterion, metric2 for second
                    if i == 0:
                        metric_min = self.config.m_fold_sampling.m_fold_metric1_min
                        metric_max = self.config.m_fold_sampling.m_fold_metric1_max
                        metric_ticks = self.config.m_fold_sampling.m_fold_metric1_ticks
                        colors = self.config.general.metric1_color
                    else:
                        metric_min = self.config.m_fold_sampling.m_fold_metric2_min
                        metric_max = self.config.m_fold_sampling.m_fold_metric2_max
                        metric_ticks = self.config.m_fold_sampling.m_fold_metric2_ticks
                        colors = self.config.general.metric2_color

                    plot_m_fold_sampling_1d(
                        results_dir=all_results_dirs,  # Pass list of all round directories
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
                        n_plot_bins=self.config.general.num_bins,
                        gradient_ascending=self.config.m_fold_sampling.m_fold_gradient_ascending,
                        linear_gradient=self.config.m_fold_sampling.m_fold_linear_gradient,
                        plddt_threshold=self.config.m_fold_sampling.m_fold_plddt_threshold,
                        figsize=self.config.m_fold_sampling.m_fold_figsize,
                        show_bin_lines=self.config.m_fold_sampling.m_fold_show_bin_lines,
                        logger=self.logger,
                        metric_colors=metric_colors
                    )

                # Generate 2D plot
                self.logger.info(f"Generating 2D plot for combined criteria: {metric_names[0]} vs {metric_names[1]}")
                plot_m_fold_sampling_2d(
                    results_dir=all_results_dirs,  # Pass list of all round directories
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
                    logger=self.logger
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
        Stage 03: Run sequence voting analysis
        """
        self.logger.info("=== STAGE 03: SEQUENCE VOTING ===")

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
            voting_dir = base_dir / "03_voting"
            voting_dir.mkdir(exist_ok=True)
            
            # Use the correct path for m-fold sampling directory
            m_fold_sampling_dir = base_dir / "02_m_fold_sampling"
            
            results_files = []

            # Process each filter criterion separately
            for i, criterion in enumerate(filter_criteria):
                criterion_name = criterion.get('name')
                if not criterion_name:
                    self.logger.warning("Filter criterion without name found, skipping")
                    continue
                
                self.logger.info(f"Processing filter criterion: {criterion_name}")

                # Create criterion-specific output directory
                criterion_output_dir = voting_dir / criterion_name
                criterion_output_dir.mkdir(exist_ok=True)

                # Create voting runner instance for this criterion
                voting_runner = SequenceVotingRunner(
                    sampling_dir=m_fold_sampling_dir,  # Updated to use the base m-fold directory
                    source_msa=self.config.general.source_a3m,
                    config_path=self.config.general.config_file,
                    output_dir=criterion_output_dir,
                    num_bins=self.config.general.num_bins,
                    max_workers=self.config.slurm.max_workers,
                    vote_threshold=self.config.sequence_voting.vote_threshold,
                    min_value=self.config.sequence_voting.vote_min_value,
                    max_value=self.config.sequence_voting.vote_max_value,
                    use_focused_bins=self.config.sequence_voting.use_focused_bins,
                    precomputed_metrics=base_dir / "02_m_fold_sampling/csv",
                    plddt_threshold=self.config.m_fold_sampling.m_fold_plddt_threshold,
                    filter_criterion=criterion_name
                )

                # Run voting analysis for this criterion
                results_file = voting_runner.run()
                
                if results_file:
                    results_files.append((criterion_name, results_file))

                    # Create plotter for visualization
                    # Determine which metric colors to use based on criterion index
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
                        num_bins=self.config.general.num_bins,
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
        Stage 04: Recompile sequences and run structure prediction
        """
        self.logger.info("=== STAGE 04: SEQUENCE RECOMPILATION & PREDICTION ===")
        
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
            base_output_dir = base_dir / "04_recompile"
            base_output_dir.mkdir(exist_ok=True)
            
            # Determine input MSA
            source_msa = self.config.general.source_a3m
            # First try hit_expand output
            if (base_dir / "01_hit_expand/hit_expand_final_msa.a3m").exists():
                source_msa = str(base_dir / "01_hit_expand/hit_expand_final_msa.a3m")
            
            all_successful = True
            
            # Process each filter criterion separately
            for i, criterion in enumerate(filter_criteria):
                criterion_name = criterion.get('name')
                if not criterion_name:
                    self.logger.warning("Filter criterion without name found, skipping")
                    continue
                
                self.logger.info(f"Processing recompilation and prediction for criterion: {criterion_name}")
                
                # Create criterion-specific output directory
                criterion_output_dir = base_output_dir / criterion_name
                criterion_output_dir.mkdir(exist_ok=True)
                
                # Get voting results for this criterion
                voting_results = base_dir / f"03_voting/{criterion_name}/voting_results.csv"
                raw_votes_json = base_dir / f"03_voting/{criterion_name}/raw_sequence_votes.json"
                
                if not voting_results.exists():
                    self.logger.error(f"Voting results not found for criterion: {criterion_name}")
                    all_successful = False
                    continue
                
                # Determine bin numbers for this criterion
                bin_numbers = None
                if len(filter_criteria) == 1:
                    # Single criterion case - use the default bin_numbers
                    bin_numbers = self.config.recompile_predict.bin_numbers_1
                else:
                    # Multiple criteria case - match criterion name with metric names
                    if criterion_name == self.config.recompile_predict.metric_name_1:
                        bin_numbers = self.config.recompile_predict.bin_numbers_1
                    elif criterion_name == self.config.recompile_predict.metric_name_2:
                        bin_numbers = self.config.recompile_predict.bin_numbers_2
                    else:
                        # Fall back to default bin_numbers if no match found
                        bin_numbers = self.config.recompile_predict.bin_numbers_1
                
                # If bin_numbers not specified, report error and skip this criterion
                if not bin_numbers:
                    self.logger.error(f"No bin numbers specified for criterion: {criterion_name}. Please specify bin_numbers_1 in the recompile_predict section of your configuration file.")
                    all_successful = False
                    continue
                
                # Create sequence recompiler for this criterion
                # Determine which metric colors to use based on criterion index
                colors = self.config.general.metric1_color if i == 0 else self.config.general.metric2_color
                
                recompiler = SequenceRecompiler(
                    output_dir=criterion_output_dir,
                    source_msa=source_msa,
                    default_pdb=self.config.general.default_pdb,
                    voting_results=voting_results,
                    bin_numbers=bin_numbers,
                    num_total_bins=self.config.general.num_bins,
                    initial_color=colors[0],
                    combine_bins=self.config.recompile_predict.combine_bins,
                    raw_votes_json=raw_votes_json if raw_votes_json.exists() else None,
                    logger=self.logger
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
        Stage 05: Plot and analyze prediction results
        """
        self.logger.info("=== STAGE 05: PURE SEQUENCE PREDICTION PLOTTING ===")
        
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
            base_output_dir = os.path.join(self.config.general.base_dir, "05_plots")
            os.makedirs(base_output_dir, exist_ok=True)
            
            all_successful = True
            
            # Process each filter criterion separately
            for i, criterion in enumerate(filter_criteria):
                criterion_name = criterion.get('name')
                if not criterion_name:
                    self.logger.warning("Filter criterion without name found, skipping")
                    continue
                
                self.logger.info(f"Processing plots for criterion: {criterion_name}")
                
                # Create criterion-specific output directory
                criterion_output_dir = os.path.join(base_output_dir, criterion_name)
                os.makedirs(criterion_output_dir, exist_ok=True)
                
                # Get the recompile directory for this criterion as the base dir for plotting
                recompile_dir = os.path.join(self.config.general.base_dir, f"04_recompile/{criterion_name}")
                
                if not os.path.exists(recompile_dir):
                    self.logger.error(f"Recompile directory not found for criterion: {recompile_dir}")
                    all_successful = False
                    continue
                
                # Create plotting configuration
                # Use metric1_color for prediction and metric2_color for control (or vice versa based on design)
                plot_config = {
                    'base_dir': recompile_dir,
                    'output_dir': criterion_output_dir,
                    'config_file': self.config.general.config_file,
                    'color_prediction': self.config.general.metric1_color[0] if i == 0 else self.config.general.metric2_color[0],
                    'color_control': self.config.general.metric1_color[1] if i == 0 else self.config.general.metric2_color[1],
                    'metric1_min': self.config.pure_sequence_plotting.metric1_min,
                    'metric1_max': self.config.pure_sequence_plotting.metric1_max,
                    'metric2_min': self.config.pure_sequence_plotting.metric2_min,
                    'metric2_max': self.config.pure_sequence_plotting.metric2_max,
                    'metric1_ticks': self.config.pure_sequence_plotting.metric1_ticks,
                    'metric2_ticks': self.config.pure_sequence_plotting.metric2_ticks,
                    'plddt_threshold': self.config.pure_sequence_plotting.plddt_threshold,
                    'figsize': self.config.pure_sequence_plotting.figsize,
                    'dpi': self.config.pure_sequence_plotting.dpi,
                    'max_workers': self.config.slurm.max_workers
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
        
    def run(self) -> None:
        """Run the pipeline with selected stages"""
        self.print_welcome()
        
        self.logger.info("=== AF-ClaSeq PIPELINE STARTED ===")
        self.logger.info(f"Configuration loaded from YAML file")
        
        stages_to_run = self.config.pipeline_control.stages
        pipeline_success = True
        
        try:
            # Stage 00: Init Bootstrapping (optional quick preview)
            if "00_INIT_BOOTSTRAPPING" in stages_to_run:
                if not self.run_init_bootstrapping():
                    self.logger.error("Stopping pipeline due to failure in stage 00_INIT_BOOTSTRAPPING")
                    pipeline_success = False
                    return
            
            # Stage 01: Hit Expand (replaces Iterative Shuffling)
            if "01_HIT_EXPAND_RUN" in stages_to_run:
                if not self.run_hit_expand():
                    self.logger.error("Stopping pipeline due to failure in stage 01_HIT_EXPAND_RUN")
                    pipeline_success = False
                    return
            
            if "01_HIT_EXPAND_ANALYSIS" in stages_to_run:
                if not self.analyze_hit_expand():
                    self.logger.error("Stopping pipeline due to failure in stage 01_HIT_EXPAND_ANALYSIS")
                    pipeline_success = False
                    return
            
            
            # Stage 02: M-fold Sampling
            if "02_M_FOLD_SAMPLING_RUN" in stages_to_run:
                if not self.run_m_fold_sampling():
                    self.logger.error("Stopping pipeline due to failure in stage 02_M_FOLD_SAMPLING_RUN")
                    pipeline_success = False
                    return
            
            if "02_M_FOLD_SAMPLING_PLOT" in stages_to_run:
                if not self.plot_m_fold_sampling():
                    self.logger.error("Stopping pipeline due to failure in stage 02_M_FOLD_SAMPLING_PLOT")
                    pipeline_success = False
                    return
            
            # Stage 03: Sequence Voting
            if "03_VOTING_RUN" in stages_to_run:
                if not self.run_sequence_voting():
                    self.logger.error("Stopping pipeline due to failure in stage 03_VOTING_RUN")
                    pipeline_success = False
                    return
            
            # Stage 04: Recompilation & Prediction
            if "04_RECOMPILE_PREDICT_RUN" in stages_to_run:
                if not self.run_recompile_and_predict():
                    self.logger.error("Stopping pipeline due to failure in stage 04_RECOMPILE_PREDICT_RUN")
                    pipeline_success = False
                    return
            
            # Stage 05: Pure Sequence Plotting
            if "05_PURE_SEQ_PLOT_RUN" in stages_to_run:
                if not self.run_pure_sequence_plotting():
                    self.logger.error("Stopping pipeline due to failure in stage 05_PURE_SEQ_PLOT_RUN")
                    pipeline_success = False
                    return
            
            if pipeline_success:
                self.logger.info("=== AF-ClaSEQ PIPELINE COMPLETED SUCCESSFULLY ===")
                self.logger.info("All requested stages executed without errors")
                self.logger.info("Results are ready for analysis in the output directories")
            
        except Exception as e:
            self.logger.error(f"Unhandled error in pipeline: {str(e)}", exc_info=True)
            sys.exit(1)


def main():
    """Main entry point for the pipeline"""
    if len(sys.argv) != 2:
        print("Usage: python run_af_claseq_pipeline.py <config.yaml>")
        sys.exit(1)
    
    yaml_input = sys.argv[1]
    
    if not os.path.exists(yaml_input):
        print(f"Error: Config file not found: {yaml_input}")
        sys.exit(1)
    
    # Initialize and run the pipeline
    pipeline = AFClaSeqPipeline(yaml_input)
    pipeline.run()


if __name__ == "__main__":
    main()
