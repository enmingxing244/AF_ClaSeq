#!/usr/bin/env python3
"""
AF-ClaSeq Pipeline

A comprehensive pipeline for protein structure prediction and analysis
using AlphaFold and sequence-based sampling approaches.
"""

import os
import sys
import logging
import yaml
import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple, Union

# Import modules from AF-ClaSeq
from af_claseq.utils.slurm_utils import SlurmJobSubmitter
from af_claseq.utils.structure_analysis import StructureAnalyzer
from af_claseq.utils.sequence_processing import (
    read_a3m_to_dict, filter_a3m_by_coverage
)

# Import pipeline components
from af_claseq.pipeline.iter_shuf_enrich import IterShufEnrichRunner, IterShufEnrichPlotter, IterShufEnrichCombiner
from af_claseq.pipeline.m_fold_sampling import MFoldSampler, MFoldPlotter
from af_claseq.pipeline.sequence_voting import SequenceVotingRunner, SequenceVotingPlotter
from af_claseq.pipeline.sequence_recompile import SequenceRecompiler
from af_claseq.pipeline.pure_seq_pred import PureSequenceAF2Prediction, create_af2_prediction_config_from_dict
from af_claseq.pipeline.pure_seq_plot import PureSequencePlotter, create_pure_seq_plot_config_from_dict

# Import configuration classes
from af_claseq.pipeline.config import (
    PipelineConfig, load_pipeline_config
)


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
        log_dir = os.path.join(self.config.general.base_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, "af_claseq_pipeline.log")
        logger = logging.getLogger("af_claseq")
        
        if not logger.handlers:  # Only set up handlers if they don't exist
            logger.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
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
        base_dir = self.config.general.base_dir
        os.makedirs(base_dir, exist_ok=True)
        
        # Create stage directories
        stages = [
            "01_iterative_shuffling",
            "02_m_fold_sampling",
            "03_voting",
            "04_recompile",
        ]
        
        for stage in stages:
            os.makedirs(os.path.join(base_dir, stage), exist_ok=True)
    
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
    
    def run_iterative_shuffling(self) -> bool:
        """
        Stage 01_RUN: Run iterative shuffling of sequences
        """
        self.logger.info("=== STAGE 01_RUN: ITERATIVE SHUFFLING ===")
        
        try:
            # Create runner instance with configuration
            runner = IterShufEnrichRunner(
                iter_shuf_input_a3m=self.config.iterative_shuffling.iter_shuf_input_a3m,
                default_pdb=self.config.general.default_pdb,
                iter_shuf_enrich_base_dir=os.path.join(self.config.general.base_dir, "01_iterative_shuffling"),
                config_file=self.config.general.config_file,
                slurm_submitter=self.slurm_submitter,
                seq_num_per_shuffle=self.config.iterative_shuffling.seq_num_per_shuffle,
                num_shuffles=self.config.iterative_shuffling.num_shuffles,
                coverage_threshold=self.config.general.coverage_threshold,
                num_iterations=self.config.iterative_shuffling.num_iterations,
                quantile=self.config.iterative_shuffling.quantile,
                plddt_threshold=self.config.iterative_shuffling.plddt_threshold,
                resume_from_iter=self.config.iterative_shuffling.resume_from_iter,
                max_workers=self.config.slurm.max_workers,
                check_interval=self.config.pipeline_control.check_interval,
                random_seed=self.config.general.random_seed
            )
            
            # Set up logging and log parameters
            runner.setup_logging()
            runner.log_parameters()
            
            # Run the iterative shuffling process
            final_a3m = runner.run()
            
            if final_a3m is None:
                self.logger.error("Iterative shuffling failed to produce output")
                return False
                
            self.logger.info(f"Completed iterative shuffling successfully. Final output: {final_a3m}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in iterative shuffling: {str(e)}", exc_info=True)
            return False
    
    def analyze_iterative_shuffling(self) -> bool:
        """
        Stage 01_ANALYSIS: Analyze results of iterative shuffling
        """
        self.logger.info("=== STAGE 01_ANALYSIS: ITERATIVE SHUFFLING ANALYSIS ===")
        try:
            # Create plotter instance
            plotter = IterShufEnrichPlotter(
                iter_shuf_enrich_base_dir=self.config.general.base_dir,
                config_path=self.config.general.config_file,
                num_cols=self.config.iterative_shuffling.iter_shuf_plot_num_cols,
                x_min=self.config.iterative_shuffling.iter_shuf_plot_x_min,
                x_max=self.config.iterative_shuffling.iter_shuf_plot_x_max,
                y_min=self.config.iterative_shuffling.iter_shuf_plot_y_min,
                y_max=self.config.iterative_shuffling.iter_shuf_plot_y_max,
                xticks=self.config.iterative_shuffling.iter_shuf_plot_xticks,
                bin_step=self.config.iterative_shuffling.iter_shuf_plot_bin_step
            )
            
            # Plot metric distributions across iterations
            plotter.analyze_and_plot()
            
            # Combine filtered sequences
            combiner = IterShufEnrichCombiner(
                iter_shuf_enrich_base_dir=self.config.general.base_dir,
                config_path=self.config.general.config_file,
                default_pdb=self.config.general.default_pdb,
                combine_threshold=self.config.iterative_shuffling.iter_shuf_combine_threshold,
                max_workers=self.config.slurm.max_workers
            )
            
            combined_a3m = combiner.combine()
            
            if combined_a3m is None:
                self.logger.error("Failed to combine sequences from iterative shuffling")
                return False
                
            self.logger.info(f"Completed iterative shuffling analysis successfully. Combined output: {combined_a3m}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in iterative shuffling analysis: {str(e)}", exc_info=True)
            return False
    
    def run_m_fold_sampling(self) -> bool:
        """
        Stage 02_RUN: Run M-fold sampling
        """
        self.logger.info("=== STAGE 02_RUN: M-FOLD SAMPLING ===")
        
        try:
            # Determine input A3M file
            input_a3m = self.config.m_fold_sampling.m_fold_samp_input_a3m
            if not os.path.exists(input_a3m):
                # Use output from iterative shuffling if available
                iter_shuf_output = os.path.join(
                    self.config.general.base_dir,
                    "01_iterative_shuffling/gathered_seq_after_iter_shuffling.a3m"
                )
                if os.path.exists(iter_shuf_output):
                    input_a3m = iter_shuf_output
                    self.logger.info(f"Using iterative shuffling output as input: {input_a3m}")
            
            # Create sampler instance
            sampler = MFoldSampler(
                input_a3m=input_a3m,
                default_pdb=self.config.general.default_pdb,
                m_fold_sampling_base_dir=os.path.join(self.config.general.base_dir, "02_m_fold_sampling"),
                group_size=self.config.m_fold_sampling.m_fold_group_size,
                coverage_threshold=self.config.general.coverage_threshold,
                random_select_num_seqs=self.config.m_fold_sampling.m_fold_random_select,
                slurm_submitter=self.slurm_submitter,
                random_seed=self.config.general.random_seed,
                max_workers=self.config.slurm.max_workers,
                logger=self.logger
            )
            
            # Run the sampling process
            sampler.run()
            self.logger.info("Completed M-fold sampling successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in M-fold sampling: {str(e)}", exc_info=True)
            return False
        
    def plot_m_fold_sampling(self) -> bool:
        """
        Stage 02_ANALYSIS: Analyze results of M-fold sampling
        """
        self.logger.info("=== STAGE 02_ANALYSIS: M-FOLD SAMPLING ANALYSIS ===")
        
        try:
            # Create plotter instance
            plotter = MFoldPlotter(
                results_dir=os.path.join(self.config.general.base_dir, "02_m_fold_sampling"),
                config_file=self.config.general.config_file,
                output_dir=os.path.join(self.config.general.base_dir, "02_m_fold_sampling/plot"),
                csv_dir=os.path.join(self.config.general.base_dir, "02_m_fold_sampling/csv"),
                logger=self.logger
            )
            
            # Load filter criteria from config file to determine plot type
            with open(self.config.general.config_file, 'r') as f:
                config = json.load(f)
            
            # Get number of filter criteria
            num_criteria = len(config.get('filter_criteria', []))
            
            if num_criteria == 1:
                # Only one criterion - use 1D plots
                self.logger.info("One filter criterion detected - generating 1D plots")
                plotter.plot_1d(
                    initial_color=self.config.general.plot_initial_color,
                    end_color=self.config.general.plot_end_color,
                    x_min=self.config.m_fold_sampling.m_fold_x_min,
                    x_max=self.config.m_fold_sampling.m_fold_x_max,
                    y_min=self.config.m_fold_sampling.m_fold_y_min,
                    y_max=self.config.m_fold_sampling.m_fold_y_max,
                    plddt_threshold=self.config.m_fold_sampling.m_fold_plddt_threshold
                )
            elif num_criteria == 2:
                # Two criteria - use 2D plots
                self.logger.info("Two filter criteria detected - generating 2D plots")
                plotter.plot_2d(
                    x_min=self.config.m_fold_sampling.m_fold_x_min,
                    x_max=self.config.m_fold_sampling.m_fold_x_max,
                    y_min=self.config.m_fold_sampling.m_fold_y_min,
                    y_max=self.config.m_fold_sampling.m_fold_y_max,
                    plddt_threshold=self.config.m_fold_sampling.m_fold_plddt_threshold
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
            base_output_dir = os.path.join(self.config.general.base_dir, "03_voting")
            os.makedirs(base_output_dir, exist_ok=True)
            
            results_files = []
            
            # Process each filter criterion separately
            for criterion in filter_criteria:
                criterion_name = criterion.get('name')
                if not criterion_name:
                    self.logger.warning("Filter criterion without name found, skipping")
                    continue
                
                self.logger.info(f"Processing filter criterion: {criterion_name}")
                
                # Create criterion-specific output directory
                criterion_output_dir = os.path.join(base_output_dir, criterion_name)
                os.makedirs(criterion_output_dir, exist_ok=True)
                
                # Create voting runner instance for this criterion
                voting_runner = SequenceVotingRunner(
                    sampling_dir=os.path.join(self.config.general.base_dir, "02_m_fold_sampling"),
                    source_msa=self.config.general.source_a3m,
                    config_path=self.config.general.config_file,
                    output_dir=criterion_output_dir,
                    num_bins=self.config.general.num_bins,
                    max_workers=self.config.slurm.max_workers,
                    vote_threshold=self.config.sequence_voting.vote_threshold,
                    min_value=self.config.sequence_voting.vote_min_value,
                    max_value=self.config.sequence_voting.vote_max_value,
                    use_focused_bins=self.config.sequence_voting.use_focused_bins,
                    precomputed_metrics=self.config.sequence_voting.precomputed_metrics if hasattr(self.config.sequence_voting, 'precomputed_metrics') else None,
                    plddt_threshold=self.config.m_fold_sampling.m_fold_plddt_threshold,
                    hierarchical_sampling=self.config.sequence_voting.vote_hierarchical_sampling,
                    filter_criterion=criterion_name
                )
                
                # Set up logging for voting runner
                voting_runner.setup_logging()
                
                # Run voting analysis for this criterion
                results_file = voting_runner.run()
                
                if results_file:
                    results_files.append((criterion_name, results_file))
                    
                    # Create plotter for visualization
                    plotter = SequenceVotingPlotter(
                        results_file=results_file,
                        output_dir=criterion_output_dir,
                        initial_color=self.config.general.plot_initial_color,
                        end_color=self.config.general.plot_end_color if hasattr(self.config.general, 'plot_end_color') else None,
                        figsize=self.config.sequence_voting.vote_figsize,
                        y_min=self.config.sequence_voting.vote_y_min,
                        y_max=self.config.sequence_voting.vote_y_max,
                        x_ticks=self.config.sequence_voting.vote_x_ticks
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
            base_output_dir = os.path.join(self.config.general.base_dir, "04_recompile")
            os.makedirs(base_output_dir, exist_ok=True)
            
            # Determine input MSA
            source_msa = self.config.general.source_a3m
            if os.path.exists(os.path.join(self.config.general.base_dir, "01_iterative_shuffling/gathered_seq_after_iter_shuffling.a3m")):
                source_msa = os.path.join(self.config.general.base_dir, "01_iterative_shuffling/gathered_seq_after_iter_shuffling.a3m")
            
            # If bin_numbers not specified, report error and stop pipeline
            bin_numbers = self.config.recompile_predict.bin_numbers
            if not bin_numbers:
                self.logger.error("Pipeline stopped: No bin numbers specified in configuration. Please specify bin_numbers in the recompile_predict section of your configuration file.")
                return False
            
            all_successful = True
            
            # Process each filter criterion separately
            for criterion in filter_criteria:
                criterion_name = criterion.get('name')
                if not criterion_name:
                    self.logger.warning("Filter criterion without name found, skipping")
                    continue
                
                self.logger.info(f"Processing recompilation and prediction for criterion: {criterion_name}")
                
                # Create criterion-specific output directory
                criterion_output_dir = os.path.join(base_output_dir, criterion_name)
                os.makedirs(criterion_output_dir, exist_ok=True)
                
                # Get voting results for this criterion
                voting_results = os.path.join(self.config.general.base_dir, f"03_voting/{criterion_name}/03_voting_results.csv")
                raw_votes_json = os.path.join(self.config.general.base_dir, f"03_voting/{criterion_name}/raw_sequence_votes.json")
                
                if not os.path.exists(voting_results):
                    self.logger.error(f"Voting results not found for criterion: {criterion_name}")
                    all_successful = False
                    continue
                
                # Create sequence recompiler for this criterion
                recompiler = SequenceRecompiler(
                    output_dir=criterion_output_dir,
                    source_msa=source_msa,
                    default_pdb=self.config.general.default_pdb,
                    voting_results=voting_results,
                    bin_numbers=bin_numbers,
                    num_total_bins=self.config.general.num_bins,
                    initial_color=self.config.general.plot_initial_color,
                    combine_bins=self.config.recompile_predict.combine_bins,
                    raw_votes_json=raw_votes_json if os.path.exists(raw_votes_json) else None,
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
                af2_config = create_af2_prediction_config_from_dict(prediction_config)
                predictor = PureSequenceAF2Prediction(
                    config=af2_config,
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
            base_output_dir = os.path.join(self.config.general.base_dir, "05_plots")
            os.makedirs(base_output_dir, exist_ok=True)
            
            all_successful = True
            
            # Process each filter criterion separately
            for criterion in filter_criteria:
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
                plot_config = {
                    'base_dir': recompile_dir,
                    'output_dir': criterion_output_dir,
                    'config_file': self.config.general.config_file,
                    'color_prediction': self.config.general.plot_initial_color,
                    'color_control': self.config.general.plot_end_color if hasattr(self.config.general, 'plot_end_color') else None,
                    'x_min': getattr(self.config.pure_sequence_plotting, 'x_min', None) if hasattr(self.config, 'pure_sequence_plotting') else None,
                    'x_max': getattr(self.config.pure_sequence_plotting, 'x_max', None) if hasattr(self.config, 'pure_sequence_plotting') else None,
                    'y_min': getattr(self.config.pure_sequence_plotting, 'y_min', None) if hasattr(self.config, 'pure_sequence_plotting') else None,
                    'y_max': getattr(self.config.pure_sequence_plotting, 'y_max', None) if hasattr(self.config, 'pure_sequence_plotting') else None,
                    'x_ticks': getattr(self.config.pure_sequence_plotting, 'x_ticks', None) if hasattr(self.config, 'pure_sequence_plotting') else None,
                    'y_ticks': getattr(self.config.pure_sequence_plotting, 'y_ticks', None) if hasattr(self.config, 'pure_sequence_plotting') else None,
                    'plddt_threshold': getattr(self.config.pure_sequence_plotting, 'plddt_threshold', 70.0) if hasattr(self.config, 'pure_sequence_plotting') else 70.0,
                    'figsize': getattr(self.config.pure_sequence_plotting, 'figsize', (15, 7)) if hasattr(self.config, 'pure_sequence_plotting') else (15, 7),
                    'dpi': getattr(self.config.pure_sequence_plotting, 'dpi', 300) if hasattr(self.config, 'pure_sequence_plotting') else 300,
                    'max_workers': self.config.slurm.max_workers
                }
                
                # Create and run plotter
                plot_config_obj = create_pure_seq_plot_config_from_dict(plot_config)
                plotter = PureSequencePlotter(
                    config=plot_config_obj,
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
        
        try:
            # Stage 01: Iterative Shuffling
            if "01_RUN" in stages_to_run:
                if not self.run_iterative_shuffling():
                    self.logger.error("Stopping pipeline due to failure in stage 01_RUN")
                    return
            
            if "01_ANALYSIS" in stages_to_run:
                if not self.analyze_iterative_shuffling():
                    self.logger.error("Stopping pipeline due to failure in stage 01_ANALYSIS")
                    return
            
            # Stage 02: M-fold Sampling
            if "02_RUN" in stages_to_run:
                if not self.run_m_fold_sampling():
                    self.logger.error("Stopping pipeline due to failure in stage 02_RUN")
                    return
            
            if "02_PLOT" in stages_to_run:
                if not self.plot_m_fold_sampling():
                    self.logger.error("Stopping pipeline due to failure in stage 02_PLOT")
                    return
            
            # Stage 03: Sequence Voting
            if "03" in stages_to_run:
                if not self.run_sequence_voting():
                    self.logger.error("Stopping pipeline due to failure in stage 03")
                    return
            
            # Stage 04: Recompilation & Prediction
            if "04" in stages_to_run:
                if not self.run_recompile_and_predict():
                    self.logger.error("Stopping pipeline due to failure in stage 04")
                    return
            
            # Stage 05: Pure Sequence Plotting
            if "05" in stages_to_run:
                if not self.run_pure_sequence_plotting():
                    self.logger.error("Stopping pipeline due to failure in stage 05")
                    return
            
            self.logger.info("=== AF-ClaSEQ PIPELINE COMPLETED SUCCESSFULLY ===")
            
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