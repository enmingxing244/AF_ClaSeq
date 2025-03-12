"""
M-fold sampling module for AF-ClaSeq pipeline.

This module provides classes to:
1. Run M-fold sampling of sequences
2. Plot distributions of metrics in 1D and 2D
3. Analyze sampling results
"""

import argparse
import logging
import os
import random
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hex2color
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from af_claseq.utils.sequence_processing import read_a3m_to_dict, write_a3m, filter_a3m_by_coverage
from af_claseq.utils.structure_analysis import get_result_df, load_filter_modes, apply_filters
from af_claseq.utils.slurm_utils import SlurmJobSubmitter


class MFoldSampler:
    """
    Main class for running M-fold sampling.
    
    M-fold sampling creates multiple groupings of sequences to 
    explore the protein's conformational landscape more effectively.
    """
    
    def __init__(
        self,
        input_a3m: str,
        default_pdb: str,
        base_dir: str,
        group_size: int = 10,
        coverage_threshold: float = 0.0,
        random_select_num_seqs: Optional[int] = None,
        slurm_submitter: Optional[SlurmJobSubmitter] = None,
        random_seed: int = 42,
        max_workers: int = 64,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the M-fold sampler.
        
        Args:
            input_a3m: Path to input A3M file
            default_pdb: Path to reference PDB file
            base_dir: Base directory for output
            group_size: Size of each sequence group
            coverage_threshold: Minimum required sequence coverage
            random_select_num_seqs: Number of sequences to randomly select
            slurm_submitter: Optional SlurmJobSubmitter instance
            random_seed: Random seed for reproducibility
            max_workers: Maximum number of concurrent workers
            logger: Optional logger instance
        """
        self.input_a3m = input_a3m
        self.default_pdb = default_pdb
        self.base_dir = base_dir
        self.group_size = group_size
        self.coverage_threshold = coverage_threshold
        self.random_select_num_seqs = random_select_num_seqs
        self.slurm_submitter = slurm_submitter
        self.max_workers = max_workers
        
        # Set up directories
        self.init_dir = os.path.join(self.base_dir, "02_init_random_split")
        self.sampling_base_dir = os.path.join(self.base_dir, "02_sampling")
        os.makedirs(self.init_dir, exist_ok=True)
        os.makedirs(self.sampling_base_dir, exist_ok=True)
        
        # Set up logger
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            if not self.logger.handlers:
                self._setup_logging()
        
        # Set random seed
        random.seed(random_seed)
    
    def _setup_logging(self):
        """Set up logging configuration"""
        log_file = os.path.join(self.base_dir, "m_fold_sampling.log")
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.addHandler(logging.StreamHandler())
        self.logger.setLevel(logging.INFO)
    
    def run(self) -> bool:
        """
        Run the M-fold sampling process.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Apply coverage filter if needed
            filtered_a3m = self._apply_coverage_filter()
            
            # Run initial random split
            num_groups = self._run_initial_split(filtered_a3m)
            
            # Create sampling splits
            self._create_sampling_splits(num_groups)
            
            # Submit SLURM jobs if available
            if self.slurm_submitter:
                self._submit_jobs()
            else:
                self.logger.info("No SLURM submitter provided. Skipping job submission.")
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error in M-fold sampling: {str(e)}", exc_info=True)
            return False
    
    def _apply_coverage_filter(self) -> str:
        """
        Apply coverage filtering to input sequences if needed.
        
        Returns:
            Path to filtered or original A3M file
        """
        if self.coverage_threshold <= 0:
            self.logger.info("Coverage threshold is not set. Using sequences as they are.")
            return self.input_a3m
        
        self.logger.info(f"Applying coverage filter with threshold {self.coverage_threshold}")
        sequences = read_a3m_to_dict(self.input_a3m)
        filtered_sequences = filter_a3m_by_coverage(sequences, self.coverage_threshold)
        
        self.logger.info(f"Filtered sequences from {len(sequences)} to {len(filtered_sequences)}")
        
        if not filtered_sequences:
            raise ValueError("No sequences remained after filtering")
        
        filtered_a3m_path = os.path.join(self.base_dir, "filtered_sequences.a3m")
        write_a3m(filtered_sequences, filtered_a3m_path, reference_pdb=self.default_pdb)
        
        return filtered_a3m_path
    
    def _run_initial_split(self, input_a3m: str) -> int:
        """
        Perform initial random split of sequences.
        
        Args:
            input_a3m: Path to input A3M file
            
        Returns:
            Number of groups created
        """
        self.logger.info("Performing initial random split")
        num_groups = self._initial_random_split(
            input_a3m=input_a3m,
            output_dir=self.init_dir,
            reference_pdb=self.default_pdb,
            group_size=self.group_size,
            random_select_num_seqs=self.random_select_num_seqs
        )
        self.logger.info(f"Created {num_groups} initial groups")
        return num_groups
    
    def _create_sampling_splits(self, num_groups: int) -> None:
        """
        Create sampling splits by leaving one group out at a time.
        
        Args:
            num_groups: Number of groups from initial split
        """
        self.logger.info("Creating sampling splits")
        self._create_sampling_splits_impl(
            init_dir=self.init_dir,
            output_base_dir=self.sampling_base_dir,
            reference_pdb=self.default_pdb,
            group_size=self.group_size
        )
        self.logger.info(f"Created {num_groups-1} sampling splits")
    
    def _submit_jobs(self) -> None:
        """Submit SLURM jobs for each sampling directory"""
        sampling_dirs = [d for d in os.listdir(self.sampling_base_dir) 
                        if d.startswith('sampling_')]
        sampling_paths = [os.path.join(self.sampling_base_dir, d) for d in sampling_dirs]
        job_ids = [f"sample_{i+1}" for i in range(len(sampling_dirs))]
        
        self.logger.info(f"Submitting {len(sampling_dirs)} SLURM jobs")
        self.slurm_submitter.process_folders_concurrently(
            folders=sampling_paths,
            job_ids=job_ids,
            max_workers=self.max_workers
        )
        self.logger.info("Submitted all SLURM jobs")
    
    @staticmethod
    def _split_into_groups(
        sequences: Dict[str, str], 
        group_size: int
    ) -> List[Dict[str, str]]:
        """
        Split sequences into groups of specified size.
        
        Args:
            sequences: Dictionary mapping sequence headers to sequences
            group_size: Size of each group
            
        Returns:
            List of dictionaries, where each dictionary contains a group of sequences
        """
        groups = []
        headers = list(sequences.keys())
        
        for i in range(0, len(headers), group_size):
            group = {}
            group_headers = headers[i:i + group_size]
            
            if len(group_headers) < group_size and groups:
                groups[-1].update({h: sequences[h] for h in group_headers})
            else:
                group = {h: sequences[h] for h in group_headers}
                groups.append(group)
                
        return groups
    
    @staticmethod
    def _initial_random_split(
        input_a3m: str, 
        output_dir: str,
        reference_pdb: str,
        group_size: int,
        random_select_num_seqs: Optional[int] = None
    ) -> int:
        """
        Perform initial random split of sequences.
        
        Args:
            input_a3m: Path to input A3M file
            output_dir: Directory to write output files
            reference_pdb: Path to reference PDB file
            group_size: Size of each sequence group
            random_select_num_seqs: Number of sequences to randomly select
            
        Returns:
            Number of groups created
        """
        os.makedirs(output_dir, exist_ok=True)
        sequences = read_a3m_to_dict(input_a3m)
        
        headers = list(sequences.keys())
        random.shuffle(headers)
        
        if random_select_num_seqs is not None:
            headers = headers[:random_select_num_seqs]
        
        shuffled_sequences = {h: sequences[h] for h in headers}
        
        groups = MFoldSampler._split_into_groups(shuffled_sequences, group_size)
        
        for i, group in enumerate(groups, 1):
            output_file = os.path.join(output_dir, f'group_{i}.a3m')
            write_a3m(group, output_file, reference_pdb=reference_pdb)
        
        return len(groups)
    
    @staticmethod
    def _create_sampling_splits_impl(
        init_dir: str, 
        output_base_dir: str, 
        reference_pdb: str,
        group_size: int
    ) -> None:
        """
        Create sampling splits by exhaustively leaving one group out at a time.
        
        Args:
            init_dir: Directory containing initial groups
            output_base_dir: Base directory for output sampling splits
            reference_pdb: Path to reference PDB file
            group_size: Size of each sequence group
        """
        groups = [f for f in os.listdir(init_dir) if f.endswith('.a3m')]
        num_groups = len(groups)
        
        for i in range(num_groups):
            sample_dir = os.path.join(output_base_dir, f'sampling_{i+1}')
            os.makedirs(sample_dir, exist_ok=True)
            
            all_sequences = {}
            for j, group in enumerate(groups):
                if j != i:
                    group_path = os.path.join(init_dir, group)
                    group_sequences = read_a3m_to_dict(group_path)
                    all_sequences.update(group_sequences)
            
            headers = list(all_sequences.keys())
            random.shuffle(headers)
            shuffled_sequences = {h: all_sequences[h] for h in headers}
            new_groups = MFoldSampler._split_into_groups(shuffled_sequences, group_size)
            
            for j, group in enumerate(new_groups, 1):
                output_file = os.path.join(sample_dir, f'group_{j}.a3m')
                write_a3m(group, output_file, reference_pdb=reference_pdb)


class MFoldPlotter:
    """
    Class for generating plots from M-fold sampling results.
    
    This class provides methods for creating 1D and 2D visualizations
    of metrics from the M-fold sampling process.
    """
    
    def __init__(
        self,
        results_dir: str,
        config_file: str,
        output_dir: str,
        csv_dir: str,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the plotter.
        
        Args:
            results_dir: Directory containing M-fold sampling results
            config_file: Path to config file with filter criteria
            output_dir: Directory for saving plot outputs
            csv_dir: Directory for saving CSV results
            logger: Optional logger instance
        """
        self.results_dir = results_dir
        self.config_file = config_file
        self.output_dir = output_dir
        self.csv_dir = csv_dir
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)
        
        # Set up logger
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
    
    def plot_1d(
        self,
        initial_color: Union[str, Tuple[float, float, float]] = '#87CEEB',
        end_color: Union[str, Tuple[float, float, float]] = '#FFFFFF',
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        log_scale: bool = False,
        n_plot_bins: int = 50,
        gradient_ascending: bool = False,
        linear_gradient: bool = False,
        plddt_threshold: float = 0,
        figsize: Tuple[float, float] = (10, 5),
        show_bin_lines: bool = False,
        x_ticks: Optional[List[float]] = None
    ) -> None:
        """
        Create 1D plots for M-fold sampling metrics.
        
        Args:
            initial_color: Initial color for gradient
            end_color: End color for gradient
            x_min: Minimum x-axis value
            x_max: Maximum x-axis value
            y_min: Minimum y-axis value
            y_max: Maximum y-axis value
            log_scale: Whether to use log scale for y-axis
            n_plot_bins: Number of bins for histogram
            gradient_ascending: Whether to use ascending gradient
            linear_gradient: Whether to use linear gradient
            plddt_threshold: pLDDT threshold for filtering structures
            figsize: Figure size in inches
            show_bin_lines: Whether to show bin lines
            x_ticks: List of x-axis tick positions
        """
        try:
            self.logger.info("Generating 1D plots")
            
            # Process colors if needed
            if isinstance(initial_color, str):
                initial_color = hex2color(initial_color)
            if isinstance(end_color, str):
                end_color = hex2color(end_color)
            
            # Import here to avoid circular imports
            from af_claseq.utils.plotting_manager import plot_m_fold_sampling_1d
            
            plot_m_fold_sampling_1d(
                results_dir=self.results_dir,
                config_file=self.config_file,
                output_dir=self.output_dir,
                csv_dir=self.csv_dir,
                gradient_ascending=gradient_ascending,
                initial_color=initial_color,
                end_color=end_color,
                x_min=x_min,
                x_max=x_max,
                log_scale=log_scale,
                n_plot_bins=n_plot_bins,
                linear_gradient=linear_gradient,
                plddt_threshold=plddt_threshold,
                figsize=figsize,
                show_bin_lines=show_bin_lines,
                y_min=y_min,
                y_max=y_max,
                x_ticks=x_ticks
            )
            
            self.logger.info("1D plots generated successfully")
            
        except Exception as e:
            self.logger.error(f"Error generating 1D plots: {str(e)}", exc_info=True)
    
    def plot_2d(
        self,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        plddt_threshold: float = 0,
        x_ticks: Optional[List[float]] = None,
        y_ticks: Optional[List[float]] = None
    ) -> None:
        """
        Create 2D plots for M-fold sampling metrics.
        
        Args:
            x_min: Minimum x-axis value
            x_max: Maximum x-axis value
            y_min: Minimum y-axis value
            y_max: Maximum y-axis value
            plddt_threshold: pLDDT threshold for filtering structures
            x_ticks: List of x-axis tick positions
            y_ticks: List of y-axis tick positions
        """
        try:
            self.logger.info("Generating 2D plots")
            
            # Import here to avoid circular imports
            from af_claseq.utils.plotting_manager import (
                plot_m_fold_sampling_2d, 
                plot_m_fold_sampling_2d_joint
            )
            
            # Regular 2D plots
            plot_m_fold_sampling_2d(
                results_dir=self.results_dir,
                config_file=self.config_file,
                output_dir=self.output_dir,
                csv_dir=self.csv_dir,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                plddt_threshold=plddt_threshold,
                x_ticks=x_ticks,
                y_ticks=y_ticks
            )
            
            # Joint 2D plots
            plot_m_fold_sampling_2d_joint(
                results_dir=self.results_dir,
                config_file=self.config_file,
                output_dir=self.output_dir,
                csv_dir=self.csv_dir,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                plddt_threshold=plddt_threshold
            )
            
            self.logger.info("2D plots generated successfully")
            
        except Exception as e:
            self.logger.error(f"Error generating 2D plots: {str(e)}", exc_info=True)


class MFoldAnalyzer:
    """
    Class for analyzing results from M-fold sampling.
    
    This class provides methods for extracting, processing, and analyzing
    data from the M-fold sampling process.
    """
    
    def __init__(
        self,
        results_dir: str,
        config_file: str,
        output_dir: str,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the analyzer.
        
        Args:
            results_dir: Directory containing M-fold sampling results
            config_file: Path to config file with filter criteria
            output_dir: Directory for saving analysis outputs
            logger: Optional logger instance
        """
        self.results_dir = results_dir
        self.config_file = config_file
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up logger
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
        
        # Load filter config
        self.filter_modes = load_filter_modes(self.config_file)
    
    def extract_metrics(
        self, 
        plddt_threshold: float = 0,
        max_workers: int = 16
    ) -> pd.DataFrame:
        """
        Extract metrics from all sampling results.
        
        Args:
            plddt_threshold: pLDDT threshold for filtering structures
            max_workers: Maximum number of concurrent workers
            
        Returns:
            DataFrame containing metrics for all structures
        """
        try:
            self.logger.info("Extracting metrics from sampling results")
            
            # Get all sampling directories
            sampling_dirs = [d for d in os.listdir(self.results_dir) 
                            if d.startswith('sampling_')]
            sampling_paths = [os.path.join(self.results_dir, d) for d in sampling_dirs]
            
            if not sampling_dirs:
                self.logger.error("No sampling directories found")
                return pd.DataFrame()
            
            # Extract data from each directory in parallel
            all_metrics = []
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_dir = {
                    executor.submit(self._extract_from_directory, path, plddt_threshold): 
                    os.path.basename(path) 
                    for path in sampling_paths
                }
                
                for future in tqdm(as_completed(future_to_dir), 
                                  total=len(future_to_dir),
                                  desc="Processing directories"):
                    dir_name = future_to_dir[future]
                    try:
                        metrics = future.result()
                        if metrics is not None and not metrics.empty:
                            # Add sampling directory information
                            metrics['sampling_dir'] = dir_name
                            all_metrics.append(metrics)
                    except Exception as e:
                        self.logger.error(f"Error processing {dir_name}: {str(e)}")
            
            if not all_metrics:
                self.logger.warning("No metrics extracted from any sampling directory")
                return pd.DataFrame()
            
            # Combine all metrics
            combined_df = pd.concat(all_metrics, ignore_index=True)
            
            # Save to CSV
            csv_path = os.path.join(self.output_dir, "all_metrics.csv")
            combined_df.to_csv(csv_path, index=False)
            self.logger.info(f"Saved all metrics to {csv_path}")
            
            return combined_df
            
        except Exception as e:
            self.logger.error(f"Error extracting metrics: {str(e)}", exc_info=True)
            return pd.DataFrame()
    
    def _extract_from_directory(
        self, 
        directory: str,
        plddt_threshold: float
    ) -> Optional[pd.DataFrame]:
        """
        Extract metrics from a single sampling directory.
        
        Args:
            directory: Path to sampling directory
            plddt_threshold: pLDDT threshold for filtering structures
            
        Returns:
            DataFrame with metrics or None if processing fails
        """
        try:
            results_df = get_result_df(
                directory,
                self.filter_modes['filter_criteria'],
                self.filter_modes['basics']
            )
            
            if results_df.empty:
                return None
            
            # Apply pLDDT filter if threshold > 0
            if plddt_threshold > 0:
                results_df = results_df[results_df['plddt'] >= plddt_threshold]
                
            return results_df
            
        except Exception as e:
            self.logger.error(f"Error extracting from {directory}: {str(e)}")
            return None
    
    def find_best_structures(
        self, 
        metric_name: str,
        metric_type: str = 'minimize',
        n_best: int = 10,
        plddt_threshold: float = 0
    ) -> pd.DataFrame:
        """
        Find the best structures according to a specific metric.
        
        Args:
            metric_name: Name of the metric to use
            metric_type: 'minimize' for metrics like RMSD or 'maximize' for metrics like TM-score
            n_best: Number of best structures to return
            plddt_threshold: pLDDT threshold for filtering structures
            
        Returns:
            DataFrame containing the best structures
        """
        try:
            self.logger.info(f"Finding best structures by {metric_name}")
            
            # Get all metrics
            all_metrics = self.extract_metrics(plddt_threshold=plddt_threshold)
            
            if all_metrics.empty:
                return pd.DataFrame()
            
            # Check if metric exists
            if metric_name not in all_metrics.columns:
                self.logger.error(f"Metric {metric_name} not found in results")
                return pd.DataFrame()
            
            # Sort by metric (ascending for 'minimize', descending for 'maximize')
            ascending = (metric_type == 'minimize')
            best_df = all_metrics.sort_values(by=metric_name, ascending=ascending).head(n_best)
            
            # Save to CSV
            csv_path = os.path.join(self.output_dir, f"best_{metric_name}_{n_best}.csv")
            best_df.to_csv(csv_path, index=False)
            self.logger.info(f"Saved best structures to {csv_path}")
            
            return best_df
            
        except Exception as e:
            self.logger.error(f"Error finding best structures: {str(e)}", exc_info=True)
            return pd.DataFrame()