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
from af_claseq.pipeline.m_fold_sampling import MFoldSampler, MFoldAnalyzer
from af_claseq.pipeline.sequence_voting import SequenceVotingRunner, SequenceVotingPlotter
from af_claseq.pipeline.sequence_recompile import SequenceRecompiler
from af_claseq.pipeline.pure_seq_pred import PureSequenceAF2Prediction, create_af2_prediction_config_from_dict

# Import configuration classes
from af_claseq.pipeline.config import (
    PipelineConfig, load_pipeline_config
)


class AFClaSeqPipeline:
    """Main pipeline class for AF-ClaSeq"""
    
    def __init__(self, config_path: str):
        """Initialize the pipeline with YAML configuration file"""
        self.config = load_pipeline_config(config_path)
        self.logger = self._setup_logging()
        self.logger.info(f"Pipeline initialized with config from {config_path}")
        
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
                base_dir=os.path.join(self.config.general.base_dir, "01_iterative_shuffling"),
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
            
            # Run the iterative shuffling process
            runner.run()
            self.logger.info("Completed iterative shuffling successfully")
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
                base_dir=self.config.general.base_dir,
                config_path=self.config.general.config_file,
                plot_num_cols=self.config.iterative_shuffling.iter_shuf_plot_num_cols,
                plot_x_min=self.config.iterative_shuffling.iter_shuf_plot_x_min,
                plot_x_max=self.config.iterative_shuffling.iter_shuf_plot_x_max,
                plot_y_min=self.config.iterative_shuffling.iter_shuf_plot_y_min,
                plot_y_max=self.config.iterative_shuffling.iter_shuf_plot_y_max,
                plot_xticks=self.config.iterative_shuffling.iter_shuf_plot_xticks,
                plot_bin_step=self.config.iterative_shuffling.iter_shuf_plot_bin_step
            )
            
            # Plot metric distributions across iterations
            plotter.analyze_and_plot()
            
            # Combine filtered sequences
            combiner = IterShufEnrichCombiner(
                base_dir=self.config.general.base_dir,
                config_path=self.config.general.config_file,
                default_pdb=self.config.general.default_pdb,
                combine_threshold=self.config.iterative_shuffling.iter_shuf_combine_threshold,
                max_workers=self.config.slurm.max_workers
            )
            
            combiner.combine()
            
            self.logger.info("Completed iterative shuffling analysis successfully")
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
                base_dir=os.path.join(self.config.general.base_dir, "02_m_fold_sampling"),
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
    def analyze_m_fold_sampling(self) -> bool:
        """
        Stage 02_ANALYSIS: Analyze results of M-fold sampling
        """
        self.logger.info("=== STAGE 02_ANALYSIS: M-FOLD SAMPLING ANALYSIS ===")
        
        try:
            # Create analyzer instance
            analyzer = MFoldAnalyzer(
                results_dir=os.path.join(self.config.general.base_dir, "02_m_fold_sampling"),
                config_file=self.config.general.config_file,
                output_dir=os.path.join(self.config.general.base_dir, "02_m_fold_sampling/analysis"),
                logger=self.logger
            )
            
            # Extract metrics and find best structures
            analyzer.extract_metrics(
                plddt_threshold=self.config.m_fold_sampling.m_fold_plddt_threshold,
                max_workers=self.config.slurm.max_workers
            )
            
            # Find best structures by primary metric
            metric_name = self.filter_config['filter_criteria'][0]['name']
            metric_type = 'minimize' if self.filter_config['filter_criteria'][0].get('method') == 'below' else 'maximize'
            analyzer.find_best_structures(
                metric_name=metric_name,
                metric_type=metric_type,
                n_best=self.config.m_fold_sampling.m_fold_n_best,
                plddt_threshold=self.config.m_fold_sampling.m_fold_plddt_threshold
            )
            
            self.logger.info("Completed M-fold sampling analysis successfully")
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
            # Create voting runner instance
            voting_runner = SequenceVotingRunner(
                sampling_dir=os.path.join(self.config.general.base_dir, "02_m_fold_sampling"),
                source_msa=self.config.general.source_a3m,
                config_path=self.config.general.config_file,
                output_dir=os.path.join(self.config.general.base_dir, "03_voting"),
                num_bins=self.config.general.num_bins,
                max_workers=self.config.slurm.max_workers,
                vote_threshold=self.config.sequence_voting.vote_threshold,
                min_value=self.config.sequence_voting.vote_min_value,
                max_value=self.config.sequence_voting.vote_max_value,
                use_focused_bins=self.config.sequence_voting.use_focused_bins,
                plddt_threshold=self.config.m_fold_sampling.m_fold_plddt_threshold,
                hierarchical_sampling=self.config.sequence_voting.vote_hierarchical_sampling
            )
            
            # Run voting analysis
            results_file = voting_runner.run()
            
            if results_file:
                # Create plotter for visualization
                plotter = SequenceVotingPlotter(
                    results_file=results_file,
                    output_dir=os.path.join(self.config.general.base_dir, "03_voting"),
                    initial_color=self.config.general.plot_initial_color,
                    figsize=self.config.sequence_voting.vote_figsize,
                    y_min=self.config.sequence_voting.vote_y_min,
                    y_max=self.config.sequence_voting.vote_y_max,
                    x_ticks=self.config.sequence_voting.vote_x_ticks
                )
                
                # Plot voting distributions
                plotter.plot()
                
                self.logger.info("Completed sequence voting successfully")
                return True
            else:
                self.logger.error("Sequence voting failed to produce results")
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
            # Create output directory
            output_dir = os.path.join(self.config.general.base_dir, "04_recompile")
            os.makedirs(output_dir, exist_ok=True)
            
            # Determine input MSA and voting results
            source_msa = self.config.general.source_a3m
            if os.path.exists(os.path.join(self.config.general.base_dir, "01_iterative_shuffling/gathered_seq_after_iter_shuffling.a3m")):
                source_msa = os.path.join(self.config.general.base_dir, "01_iterative_shuffling/gathered_seq_after_iter_shuffling.a3m")
            
            voting_results = os.path.join(self.config.general.base_dir, "03_voting/03_voting_results.csv")
            raw_votes_json = os.path.join(self.config.general.base_dir, "03_voting/raw_sequence_votes.json")
            
            # If bin_numbers not specified, use default range
            bin_numbers = self.config.recompile_predict.bin_numbers
            if not bin_numbers:
                self.logger.warning("No bin numbers specified, using default range (5-10)")
                bin_numbers = list(range(5, 11))
            
            # Create sequence recompiler
            recompiler = SequenceRecompiler(
                output_dir=output_dir,
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
                'base_dir': output_dir,
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
                'job_name_prefix': self.config.general.protein_name
            }
            
            # Create and run predictor
            af2_config = create_af2_prediction_config_from_dict(prediction_config)
            predictor = PureSequenceAF2Prediction(
                config=af2_config,
                logger=self.logger
            )
            
            result = predictor.run()
            
            if result:
                self.logger.info("Completed recompilation and prediction successfully")
                return True
            else:
                self.logger.error("Error in prediction process")
                return False
            
        except Exception as e:
            self.logger.error(f"Error in recompilation and prediction: {str(e)}", exc_info=True)
            return False
    
    def run(self) -> None:
        """Run the pipeline with selected stages"""
        self.print_welcome()
        
        self.logger.info("=== AF-ClaSEQ PIPELINE STARTED ===")
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
            
            if "02_ANALYSIS" in stages_to_run:
                if not self.analyze_m_fold_sampling():
                    self.logger.error("Stopping pipeline due to failure in stage 02_ANALYSIS")
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
            
            self.logger.info("=== AF-ClaSEQ PIPELINE COMPLETED SUCCESSFULLY ===")
            
        except Exception as e:
            self.logger.error(f"Unhandled error in pipeline: {str(e)}", exc_info=True)
            sys.exit(1)


def main():
    """Main entry point for the pipeline"""
    if len(sys.argv) != 2:
        print("Usage: python run_af_claseq_pipeline.py <config.yaml>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    # Initialize and run the pipeline
    pipeline = AFClaSeqPipeline(config_path)
    pipeline.run()


if __name__ == "__main__":
    main()