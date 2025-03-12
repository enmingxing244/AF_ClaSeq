"""
Iterative shuffling enrichment module for AF-ClaSeq pipeline.

This module provides classes to:
1. Run iterative shuffling of sequences
2. Plot metric distributions across iterations
3. Combine filtered sequences for downstream analysis
"""

import argparse
import logging
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import Dict, Any, List, Optional, Tuple, Union

from af_claseq.utils.slurm_utils import SlurmJobSubmitter
from af_claseq.utils.sequence_processing import (
    read_a3m_to_dict,
    filter_a3m_by_coverage,
    write_a3m,
    process_all_sequences,
    collect_a3m_files,
    concatenate_a3m_content
)
from af_claseq.utils.structure_analysis import (
    get_result_df,
    apply_filters,
    load_filter_modes
)


class IterShufEnrichRunner:
    """
    Runner class for iterative shuffling enrichment.
    
    This class handles the process of running multiple iterations of 
    sequence shuffling to enrich for sequences that lead to desired
    conformational states.
    """
    
    def __init__(
        self,
        iter_shuf_input_a3m: str,
        default_pdb: str,
        base_dir: str,
        config_file: str,
        slurm_submitter: Optional[SlurmJobSubmitter] = None,
        seq_num_per_shuffle: int = 10,
        num_shuffles: int = 10,
        coverage_threshold: float = 0.8,
        num_iterations: int = 3,
        quantile: float = 0.6,
        plddt_threshold: int = 75,
        resume_from_iter: Optional[int] = None,
        max_workers: int = 64,
        check_interval: int = 60,
        random_seed: int = 42
    ):
        """
        Initialize the runner with configuration parameters.
        
        Args:
            iter_shuf_input_a3m: Path to input A3M file
            default_pdb: Path to reference PDB file
            base_dir: Base directory for output
            config_file: Path to config file with filter criteria
            slurm_submitter: SlurmJobSubmitter instance (will create one if None)
            seq_num_per_shuffle: Size of each sequence group during shuffling
            num_shuffles: Number of shuffles
            coverage_threshold: Minimum required sequence coverage
            num_iterations: Number of iterations to run
            quantile: Quantile threshold for filtering
            plddt_threshold: pLDDT threshold for filtering
            resume_from_iter: Resume from a specific iteration number
            max_workers: Maximum number of concurrent workers
            check_interval: Interval to check job status in seconds
            random_seed: Random seed for reproducibility
        """
        self.iter_shuf_input_a3m = iter_shuf_input_a3m
        self.default_pdb = default_pdb
        self.base_dir = base_dir
        self.config_file = config_file
        self.seq_num_per_shuffle = seq_num_per_shuffle
        self.num_shuffles = num_shuffles
        self.coverage_threshold = coverage_threshold
        self.num_iterations = num_iterations
        self.quantile = quantile
        self.plddt_threshold = plddt_threshold
        self.resume_from_iter = resume_from_iter
        self.max_workers = max_workers
        self.check_interval = check_interval
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        
        # Initialize output directories
        self.iter_shuffle_base = os.path.join(self.base_dir, "01_iterative_shuffling")
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.iter_shuffle_base, exist_ok=True)
        
        # Set up logger
        self.logger = logging.getLogger(__name__)
        
        # Load filter config
        self.config = load_filter_modes(self.config_file)
        
        # Set up SLURM submitter if not provided
        self.slurm_submitter = slurm_submitter
    
    def setup_logging(self, log_file: Optional[str] = None) -> None:
        """
        Set up logging configuration.
        
        Args:
            log_file: Path to log file (default is in iterative shuffling directory)
        """
        if log_file is None:
            log_file = os.path.join(self.iter_shuffle_base, "01_iterative_shuffling.log")
            
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def log_parameters(self) -> None:
        """Log all input parameters"""
        self.logger.info("=== Iterative Shuffling Enrichment Parameters ===")
        for param, value in self.__dict__.items():
            if param != 'config' and param != 'slurm_submitter' and param != 'logger':
                self.logger.info(f"{param}: {value}")
        self.logger.info("=======================================")
    
    def run(self) -> Optional[str]:
        """
        Run the iterative shuffling process.
        
        Returns:
            Path to final output A3M file or None if process fails
        """
        self.logger.info("Starting iterative shuffling process")
        
        try:
            # Initial sequence filter by coverage
            if not self.resume_from_iter:
                sequences = read_a3m_to_dict(self.iter_shuf_input_a3m)
                filtered_sequences = filter_a3m_by_coverage(sequences, self.coverage_threshold)
                self.logger.info(f"Filtered sequences from {len(sequences)} to {len(filtered_sequences)} "
                                f"based on coverage threshold {self.coverage_threshold}")
                
                if not filtered_sequences:
                    raise ValueError("No sequences remained after filtering")
                
                filtered_a3m_path = os.path.join(self.iter_shuffle_base, "filtered_sequences.a3m")
                write_a3m(filtered_sequences, filtered_a3m_path, reference_pdb=self.default_pdb)
                
                # Start iteration
                current_input = filtered_a3m_path
                start_iter = 1
                
            elif self.resume_from_iter:
                start_iter = self.resume_from_iter
                prev_iter_a3m = os.path.join(
                    self.iter_shuffle_base, 
                    f'Iteration_{self.resume_from_iter - 1}', 
                    f'combined_filtered_iteration_{self.resume_from_iter - 1}.a3m'
                )
                if os.path.exists(prev_iter_a3m):
                    current_input = prev_iter_a3m
                    self.logger.info(f"Resuming from iteration {self.resume_from_iter}")
                else:
                    raise ValueError(f"Cannot resume - file not found: {prev_iter_a3m}")
            
            # Process each iteration
            for iteration in range(start_iter, self.num_iterations + 1):
                self.logger.info(f"Starting iteration {iteration}")
                current_input = self._process_iteration(
                    iteration=iteration,
                    input_a3m=current_input
                )
                
                if current_input is None:
                    self.logger.error(f"Iteration {iteration} failed")
                    return None
            
            self.logger.info("All iterations completed successfully")
            return current_input
            
        except Exception as e:
            self.logger.error(f"An error occurred in iterative shuffling: {str(e)}", exc_info=True)
            return None
    
    def _process_iteration(self, iteration: int, input_a3m: str) -> Optional[str]:
        """
        Process a single iteration of shuffling.
        
        Args:
            iteration: Iteration number
            input_a3m: Path to input A3M file for this iteration
            
        Returns:
            Path to output A3M file containing filtered sequences or None if process fails
        """
        iter_dir = os.path.join(self.iter_shuffle_base, f'Iteration_{iteration}')
        os.makedirs(iter_dir, exist_ok=True)
        self.logger.info(f'Processing iteration {iteration}...')
        
        try:
            # Process sequences with shuffling
            process_all_sequences(
                dir_path=iter_dir,
                file_path=input_a3m,
                num_shuffles=self.num_shuffles,
                seq_num_per_shuffle=self.seq_num_per_shuffle,
                reference_pdb=self.default_pdb
            )
            
            # Get job name prefix from base directory path
            base_path_parts = self.base_dir.split(os.sep)
            try:
                results_idx = base_path_parts.index('results')
                job_name_prefix = base_path_parts[results_idx + 1] if results_idx + 1 < len(base_path_parts) else "fold"
            except ValueError:
                job_name_prefix = "fold"
            
            # Get shuffle directories and submit jobs
            shuffle_dirs = [d for d in os.listdir(iter_dir) if d.startswith('shuffle_')]
            job_folders = [f"shuffle_{i+1}" for i, _ in enumerate(shuffle_dirs)]
            all_paths = [os.path.join(iter_dir, d) for d in shuffle_dirs]
            
            # Submit SLURM jobs
            self.slurm_submitter.process_folders_concurrently(
                folders=all_paths,
                job_ids=job_folders,
                max_workers=self.max_workers
            )
            
            # Get results and apply filters
            result_df = get_result_df(
                parent_dir=iter_dir,
                filter_criteria=self.config['filter_criteria'],
                basics=self.config['basics']
            )
            
            result_df_filtered = result_df[result_df['plddt'] > self.plddt_threshold]
            filtered_df = apply_filters(
                df_threshold=result_df_filtered,
                df_operate=result_df_filtered,
                filter_criteria=self.config['filter_criteria'],
                quantile=self.quantile
            )
            
            # Save filtered DataFrame to CSV
            filtered_df_path = os.path.join(iter_dir, f'filtered_results_iteration_{iteration}.csv')
            filtered_df.to_csv(filtered_df_path, index=False)
            self.logger.info(f'Saved filtered results to {filtered_df_path}')
            
            # Collect and concatenate filtered sequences
            a3m_files = collect_a3m_files([filtered_df])
            concatenated_a3m_path = os.path.join(iter_dir, f'combined_filtered_iteration_{iteration}.a3m')
            concatenate_a3m_content(a3m_files, self.default_pdb, concatenated_a3m_path)
            
            return concatenated_a3m_path
            
        except Exception as e:
            self.logger.error(f"Error in iteration {iteration}: {str(e)}", exc_info=True)
            return None


class IterShufEnrichPlotter:
    """
    Class for creating metric distribution plots across iterations.
    
    This class handles the calculation of metrics (RMSD or TM-score) and
    creates distribution plots for each iteration to visualize changes
    in the metric distribution over iterations.
    """
    
    def __init__(
        self,
        base_dir: str,
        config_path: str,
        plot_num_cols: int = 5,
        plot_x_min: float = 0,
        plot_x_max: float = 20,
        plot_y_min: float = 0.8,
        plot_y_max: float = 10000,
        plot_xticks: Optional[List[float]] = None,
        plot_bin_step: float = 0.2
    ):
        """
        Initialize the plotter with configuration parameters.
        
        Args:
            base_dir: Base directory for output
            config_path: Path to config file with filter criteria
            plot_num_cols: Number of columns in plot grid
            plot_x_min: Minimum x-axis value
            plot_x_max: Maximum x-axis value
            plot_y_min: Minimum y-axis value
            plot_y_max: Maximum y-axis value
            plot_xticks: List of x-axis tick positions
            plot_bin_step: Step size for binning
        """
        self.base_dir = base_dir
        self.config_path = config_path
        self.plot_num_cols = plot_num_cols
        self.plot_x_min = plot_x_min
        self.plot_x_max = plot_x_max
        self.plot_y_min = plot_y_min
        self.plot_y_max = plot_y_max
        self.plot_xticks = plot_xticks
        self.plot_bin_step = plot_bin_step
        
        # Set parent directory for iterative shuffling
        self.parent_dir = os.path.join(self.base_dir, '01_iterative_shuffling')
        
        # Create analysis directories
        self.analysis_dir = os.path.join(self.parent_dir, 'analysis')
        self.plot_dir = os.path.join(self.analysis_dir, 'plot')
        os.makedirs(self.analysis_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        
        # Set up logger
        self.logger = logging.getLogger(__name__)
        
        # Configure matplotlib for publication-quality plots
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['font.size'] = 24
        plt.rcParams['axes.labelsize'] = 24
        plt.rcParams['axes.titlesize'] = 24
        plt.rcParams['xtick.labelsize'] = 24
        plt.rcParams['ytick.labelsize'] = 24
    
    def analyze_and_plot(self) -> Optional[str]:
        """
        Run the metrics analysis and plotting.
        
        Returns:
            Path to the CSV file with metric values or None if process fails
        """
        try:
            # Get list of iteration directories
            iteration_dirs = [d for d in os.listdir(self.parent_dir) if d.startswith('Iteration_')]
            iteration_dirs.sort(key=lambda x: int(x.split('_')[1]))
            
            if not iteration_dirs:
                self.logger.error("No iterations found. Run iterative shuffling first.")
                return None
            
            # Load filter modes to determine metric type
            filter_modes = load_filter_modes(self.config_path)
            metric_type = filter_modes['filter_criteria'][0]['type']
            metric_name = filter_modes['filter_criteria'][0]['name']
            
            # Check if values CSV already exists
            csv_path = os.path.join(self.plot_dir, f'{metric_name}_values.csv')
            if os.path.exists(csv_path):
                self.logger.info(f"Loading pre-calculated {metric_type} values from {csv_path}")
                combined_df = pd.read_csv(csv_path)
            else:
                self.logger.info(f"Calculating {metric_type} values for all iterations")
                combined_df, _ = self._calculate_metric_values(iteration_dirs, filter_modes)
                
                # Save to CSV
                csv_path = os.path.join(self.plot_dir, f'{metric_name}_values.csv')
                combined_df.to_csv(csv_path, index=False)
                self.logger.info(f"Saved metrics to {csv_path}")
            
            # Adjust plot parameters based on metric type
            if metric_type == 'tmscore':
                self.plot_x_min = 0
                self.plot_x_max = 1
                if self.plot_xticks is None:
                    self.plot_xticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                self.plot_bin_step = 0.02
            
            # Create distribution plots
            self._create_distribution_plot(combined_df, metric_name, metric_type)
            
            return csv_path
            
        except Exception as e:
            self.logger.error(f"An error occurred in metrics analysis: {str(e)}", exc_info=True)
            return None
    
    def _calculate_metric_values(
        self, 
        iteration_dirs: List[str],
        filter_modes: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, str]:
        """
        Calculate metric values for all iterations.
        
        Args:
            iteration_dirs: List of iteration directory names
            filter_modes: Filter criteria configuration
            
        Returns:
            DataFrame with metric values and metric type
        """
        all_data = []
        
        metric_type = filter_modes['filter_criteria'][0]['type']
        metric_name = filter_modes['filter_criteria'][0]['name']
        
        for iteration_dir in tqdm(iteration_dirs, desc="Processing iterations"):
            iteration_path = os.path.join(self.parent_dir, iteration_dir)
            
            # Get results dataframe for this iteration
            results_df = get_result_df(
                iteration_path,
                filter_modes['filter_criteria'],
                filter_modes['basics']
            )
            
            # Extract values from results
            metric_values = results_df[metric_name].dropna()
            plddt_values = results_df['plddt'].dropna()
            pdb_paths = results_df['PDB']
            seq_counts = results_df['seq_count']
            
            if not metric_values.empty:
                iteration_data = pd.DataFrame({
                    'iteration': iteration_dir.split('_')[1],
                    'PDB': pdb_paths,
                    'seq_count': seq_counts,
                    'plddt': plddt_values,
                    metric_name: metric_values
                })
                all_data.append(iteration_data)
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        return combined_df, metric_type
    
    def _create_distribution_plot(
        self, 
        df: pd.DataFrame, 
        metric_name: str, 
        metric_type: str
    ) -> None:
        """
        Create distribution plots for each iteration.
        
        Args:
            df: DataFrame with metric values
            metric_name: Name of the metric
            metric_type: Type of metric ('rmsd' or 'tmscore')
        """
        iterations = sorted(df['iteration'].unique())
        
        n_iterations = len(iterations)
        n_cols = min(self.plot_num_cols, n_iterations)
        n_rows = (n_iterations + n_cols - 1) // n_cols
        
        # Create figure with subplots
        fig, axes = plt.subplots(
            n_rows, 
            n_cols, 
            figsize=(5*n_cols, 4*n_rows), 
            sharex=True, 
            sharey=True
        )
        
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(-1, 1) if n_cols == 1 else axes.reshape(1, -1)
        
        bins = np.arange(
            self.plot_x_min, 
            self.plot_x_max + self.plot_bin_step, 
            self.plot_bin_step
        )
        
        # Plot each iteration
        for idx, iteration in enumerate(iterations):
            row = idx // n_cols
            col = idx % n_cols
            
            iteration_data = df[df['iteration'] == iteration][metric_name]
            
            quantile = 0.2 if metric_type == 'rmsd' else 0.8
            quantile_value = np.quantile(iteration_data, quantile)
            axes[row, col].hist(iteration_data, bins=bins, color='skyblue', edgecolor=None, alpha=0.7)
            
            unit = 'Å' if metric_type == 'rmsd' else ''
            axes[row, col].axvline(
                x=quantile_value, 
                color='red', 
                linestyle='--', 
                alpha=0.7,
                label=f'{quantile_value:.3f}{unit}'
            )
            
            axes[row, col].set_title(f'Iteration {iteration}')
            axes[row, col].legend(fontsize=16, loc='upper right')
            
            # Apply axis settings
            axes[row, col].set_xlim(self.plot_x_min, self.plot_x_max)
            if self.plot_xticks:
                axes[row, col].set_xticks(self.plot_xticks)
                # Format x-axis tick labels to show exact values given
                axes[row, col].xaxis.set_major_formatter(
                    ticker.FuncFormatter(lambda x, p: str(int(x)) if x.is_integer() else f'{x:.1f}')
                )
            
            axes[row, col].set_yscale('log')
            axes[row, col].set_ylim(self.plot_y_min, self.plot_y_max)
            axes[row, col].yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
            axes[row, col].yaxis.set_minor_locator(
                ticker.LogLocator(base=10.0, subs=list(np.arange(2, 10) * 0.1), numticks=10)
            )
            axes[row, col].tick_params(axis='y', which='major', length=8, width=1)
            axes[row, col].tick_params(axis='y', which='minor', length=5, width=1)
            axes[row, col].tick_params(axis='x', which='major', length=5, width=1)
        
        # Add single x and y labels for entire figure
        x_label = f'RMSD to {metric_name.split("_")[0]} (Å)' if metric_type == 'rmsd' else f'TM-score to {metric_name.split("_")[0]}'
        fig.text(0.5, 0.02, x_label, ha='center', va='center')
        fig.text(0.02, 0.5, 'Counts of predicted structures', ha='center', va='center', rotation='vertical')
        
        # Remove empty subplots
        for idx in range(len(iterations), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            fig.delaxes(axes[row, col])
        
        plt.subplots_adjust(wspace=0.14, hspace=0.2, left=0.1, bottom=0.1)
        
        # Save plot
        output_path = os.path.join(self.plot_dir, f'{metric_name}_distribution_by_iteration.png')
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        
        plt.close()
        self.logger.info(f"{metric_type.upper()} distribution plots saved to {output_path}")


class IterShufEnrichCombiner:
    """
    Class for combining sequences from A3M files based on metric filtering.
    
    This class processes PDB files to extract A3M files for sequences that
    pass filtering criteria, then combines them into a single A3M file for
    downstream analysis.
    """
    
    def __init__(
        self,
        base_dir: str,
        config_path: str,
        default_pdb: str,
        combine_threshold: float,
        max_workers: int = 32
    ):
        """
        Initialize the combiner with configuration parameters.
        
        Args:
            base_dir: Base directory for output
            config_path: Path to config file with filter criteria
            default_pdb: Path to reference PDB file
            combine_threshold: Threshold for filtering (TM-score > threshold or RMSD < threshold)
            max_workers: Maximum number of concurrent workers
        """
        self.base_dir = base_dir
        self.config_path = config_path
        self.default_pdb = default_pdb
        self.combine_threshold = combine_threshold
        self.max_workers = max_workers
        
        # Set parent directory for iterative shuffling
        self.parent_dir = os.path.join(self.base_dir, '01_iterative_shuffling')
        
        # Create output directories
        self.a3m_combine_dir = os.path.join(self.parent_dir, 'analysis/a3m_combine')
        os.makedirs(self.a3m_combine_dir, exist_ok=True)
        
        # Set up logger
        self.logger = logging.getLogger(__name__)
    
    def combine(self) -> Optional[str]:
        """
        Run the sequence combination process.
        
        Returns:
            Path to combined A3M file or None if process fails
        """
        try:
            # Load config file to get filter criteria
            with open(self.config_path) as f:
                config = json.load(f)
            
            filter_name = config['filter_criteria'][0]['name']
            metric_type = config['filter_criteria'][0]['type']
            
            # Read metric values from CSV into a dictionary for faster lookup
            csv_path = os.path.join(self.parent_dir, f'analysis/plot/{filter_name}_values.csv')
            if not os.path.exists(csv_path):
                self.logger.error(f"Metric values CSV not found: {csv_path}. Run metrics analysis first.")
                return None
                
            self.logger.info(f"Reading metric values from {csv_path}")
            self.logger.info(f"Using metric: {filter_name} with threshold {self.combine_threshold}")
            
            metric_df = pd.read_csv(csv_path)
            metric_dict = pd.Series(metric_df[f'{filter_name}'].values, index=metric_df['PDB']).to_dict()
            
            # Get all PDB files
            pdb_files = self._get_pdb_files()
            
            # Process PDBs in parallel and get filtered A3M files
            a3m_files = self._parallel_process_pdbs(
                pdb_files=pdb_files,
                metric_dict=metric_dict,
                metric_type=metric_type
            )
            
            self.logger.info(f"Number of filtered PDBs/A3M files: {len(a3m_files)}")
            
            # Combine sequences from all A3M files
            output_file = os.path.join(self.a3m_combine_dir, 'gathered_seq_after_iter_shuffling.a3m')
            
            # Concatenate A3M content
            concatenate_a3m_content(a3m_files, self.default_pdb, output_file)
            
            # Create a symlink in the parent directory for easy access
            combined_a3m_path = os.path.join(self.parent_dir, "gathered_seq_after_iter_shuffling.a3m")
            if not os.path.exists(combined_a3m_path):
                os.symlink(output_file, combined_a3m_path)
            
            self.logger.info(f"Combined sequences saved to {output_file}")
            return combined_a3m_path
            
        except Exception as e:
            self.logger.error(f"An error occurred in sequence combination: {str(e)}", exc_info=True)
            return None
    
    def _get_pdb_files(self) -> List[str]:
        """
        Recursively find all PDB files in directory.
        
        Returns:
            List of PDB file paths
        """
        pdb_files = [
            os.path.join(root, file)
            for root, dirs, files in os.walk(self.parent_dir)
            for file in files
            if file.endswith('.pdb')
        ]
        return pdb_files
    
    def _parallel_process_pdbs(
        self,
        pdb_files: List[str],
        metric_dict: Dict[str, float],
        metric_type: str
    ) -> List[str]:
        """
        Process PDB files in parallel using threads.
        
        Args:
            pdb_files: List of PDB file paths
            metric_dict: Dictionary mapping PDB paths to metric values
            metric_type: Type of metric ('rmsd' or 'tmscore')
            
        Returns:
            List of A3M file paths that meet filtering criteria
        """
        a3m_files = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._process_pdb, 
                    pdb, 
                    metric_dict, 
                    self.combine_threshold, 
                    metric_type
                ): pdb for pdb in pdb_files
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing PDBs"):
                try:
                    a3m_path = future.result()
                    if a3m_path:
                        a3m_files.append(a3m_path)
                except Exception as e:
                    pdb = futures[future]
                    self.logger.error(f"Error processing PDB {pdb}: {e}")
        
        return a3m_files
    
    @staticmethod
    def _process_pdb(
        pdb: str,
        metric_dict: Dict[str, float],
        threshold: float,
        metric_type: str
    ) -> Optional[str]:
        """
        Process a single PDB file and return A3M path if it meets criteria.
        
        Args:
            pdb: PDB file path
            metric_dict: Dictionary mapping PDB paths to metric values
            threshold: Threshold for filtering
            metric_type: Type of metric ('rmsd' or 'tmscore')
            
        Returns:
            A3M file path if PDB meets criteria, None otherwise
        """
        value = metric_dict.get(pdb)
        if value is not None:
            if metric_type == 'tmscore' and value > threshold:
                return IterShufEnrichCombiner._get_a3m_path_from_pdb(pdb)
            elif metric_type == 'rmsd' and value < threshold:
                return IterShufEnrichCombiner._get_a3m_path_from_pdb(pdb)
        return None
    
    @staticmethod
    def _get_a3m_path_from_pdb(pdb_path: str) -> str:
        """
        Convert PDB path to corresponding A3M path.
        
        Args:
            pdb_path: Path to PDB file
            
        Returns:
            Path to corresponding A3M file
        """
        # Remove _unrelaxed and everything after
        a3m_path = pdb_path.split('_unrelaxed')[0] + '.a3m'
        return a3m_path