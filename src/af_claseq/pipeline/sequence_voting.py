"""
Sequence voting module for AF-ClaSeq pipeline.

This module provides classes to:
1. Run sequence voting analysis on sampling results
2. Plot metric bin distributions
3. Process sequence votes for downstream analysis
"""

import os
import logging
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict, Counter
from tqdm import tqdm
from pathlib import Path

from af_claseq.utils.structure_analysis import StructureAnalyzer
from af_claseq.utils.logging_utils import get_logger


class VotingAnalyzer:
    """
    Class that performs the core sequence voting calculations.
    
    This class provides methods to process sampling directories,
    create metric bins, and collect votes from sequences.
    """
    
    def __init__(self, max_workers: int = 64):
        """
        Initialize the VotingAnalyzer.
        
        Args:
            max_workers: Maximum number of concurrent workers for processing
        """
        self.max_workers = max_workers
        self.structure_analyzer = StructureAnalyzer()
        self.logger = get_logger(__name__)

    def process_sampling_dirs(self, 
                              base_dir: str, 
                              filter_criteria: List[Dict[str, Any]], 
                              basics: Dict[str, Any],
                              precomputed_metrics_file: Optional[str] = None,
                              plddt_threshold: float = 0) -> Dict[str, Dict[str, float]]:
        """Process sampling directories to calculate metrics in parallel.
        
        Args:
            base_dir: Base directory containing sampling results
            filter_criteria: List of criteria for filtering structures
            basics: Basic configuration parameters
            precomputed_metrics_file: Optional path to precomputed metrics CSV/directory
            plddt_threshold: Minimum pLDDT score threshold
            hierarchical: Whether to process hierarchical sampling directories
            
        Returns:
            Dictionary mapping PDB paths to their metric values
        """
        results = {}
        metric_names = [criterion['name'] for criterion in filter_criteria]

        # Try to load precomputed metrics first
        if precomputed_metrics_file:
            results = self._load_precomputed_metrics(
                precomputed_metrics_file, metric_names, plddt_threshold
            )
            if results:
                return results

        # If no precomputed metrics or they couldn't be loaded, collect PDB files
        pdb_files = self._collect_pdb_files(base_dir)

        if not pdb_files:
            self.logger.warning("No PDB files found in sampling directories")
            return results

        # Process PDB files in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(
                    self._process_pdb_file, 
                    pdb_path, filter_criteria, basics, plddt_threshold
                ) for pdb_path in pdb_files
            ]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing PDB files"):
                result = future.result()
                if result:
                    pdb_path, metrics = result
                    results[pdb_path] = metrics
                    
        self.logger.info(f"Calculated metrics for {len(results)} structures")
        return results

    def _process_pdb_file(self, pdb_path, filter_criteria, basics, plddt_threshold):
        """Process a single PDB file and return metrics if valid."""
        # Use the improved StructureAnalyzer method
        result = self.structure_analyzer.process_single_pdb(
            pdb_path, filter_criteria, basics, plddt_threshold
        )
        
        if result:
            # Extract only the required metrics for voting
            metrics = {
                criterion['name']: result[criterion['name']] 
                for criterion in filter_criteria 
                if criterion['name'] in result
            }
            
            return pdb_path, metrics
        
        return None

    def _load_precomputed_metrics(self, 
                                 precomputed_metrics_file: str, 
                                 metric_names: List[str],
                                 plddt_threshold: float) -> Dict[str, Dict[str, float]]:
        """Load metrics from precomputed CSV files."""
        results = {}
        
        # Handle directory with multiple CSV files
        if os.path.isdir(precomputed_metrics_file):
            self.logger.info(f"Looking for precomputed metrics in directory: {precomputed_metrics_file}")
            csv_files = [f for f in os.listdir(precomputed_metrics_file) if f.endswith('.csv')]
            
            if not csv_files:
                self.logger.warning(f"No CSV files found in {precomputed_metrics_file}")
                return results
                
            self.logger.info(f"Found {len(csv_files)} CSV files")
            
            for csv_file in csv_files:
                try:
                    csv_path = os.path.join(precomputed_metrics_file, csv_file)
                    self._process_metrics_csv(csv_path, metric_names, plddt_threshold, results)
                except Exception as e:
                    self.logger.error(f"Error processing CSV file {csv_file}: {str(e)}")
        
        # Handle single CSV file
        elif os.path.exists(precomputed_metrics_file) and precomputed_metrics_file.endswith('.csv'):
            self.logger.info(f"Loading precomputed metrics from {precomputed_metrics_file}")
            try:
                self._process_metrics_csv(precomputed_metrics_file, metric_names, plddt_threshold, results)
            except Exception as e:
                self.logger.error(f"Error loading precomputed metrics: {str(e)}")
        
        if results:
            self.logger.info(f"Loaded metrics for {len(results)} structures from precomputed files")
            
        return results
    
    def _process_metrics_csv(self, 
                            csv_path: str, 
                            metric_names: List[str], 
                            plddt_threshold: float,
                            results: Dict[str, Dict[str, float]]) -> None:
        """Process a single metrics CSV file and update results dictionary."""
        metrics_df = pd.read_csv(csv_path)
        
        for _, row in metrics_df.iterrows():
            if 'PDB' not in row:
                continue
                
            pdb_path = row['PDB']
            
            # Skip if pLDDT is below threshold
            if 'plddt' in row and row['plddt'] <= plddt_threshold:
                continue
                
            # Add metrics to results
            if pdb_path not in results:
                results[pdb_path] = {}
                
            for metric_name in metric_names:
                if metric_name in row:
                    results[pdb_path][metric_name] = row[metric_name]

    def _collect_pdb_files(self, base_dir: str) -> List[str]:
        """Collect PDB files from single or multi-round sampling directories."""
        pdb_files = []
        
        # Check if we have a rounds-based directory structure
        round_dirs = [d for d in os.listdir(base_dir) if d.startswith('round_')]
        
        if round_dirs:
            # Multi-round structure
            for round_dir in round_dirs:
                round_path = os.path.join(base_dir, round_dir)
                sampling_path = os.path.join(round_path, '02_sampling')
                
                if not os.path.exists(sampling_path):
                    continue
                    
                sampling_dirs = [d for d in os.listdir(sampling_path) if d.startswith('sampling_')]
                
                for sampling_dir in sampling_dirs:
                    dir_path = os.path.join(sampling_path, sampling_dir)
                    for group_dir in os.listdir(dir_path):
                        if group_dir.endswith('.a3m'):
                            a3m_path = os.path.join(dir_path, group_dir)
                            base_name = os.path.splitext(a3m_path)[0]
                            pdb_name = f"{base_name}_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_042.pdb"
                            if os.path.exists(pdb_name):
                                pdb_files.append(pdb_name)
        else:
            # Standard directory structure (single round)
            sampling_dirs = [d for d in os.listdir(base_dir) if d.startswith('sampling_')]
            
            for sampling_dir in sampling_dirs:
                dir_path = os.path.join(base_dir, sampling_dir)
                for f in os.listdir(dir_path):
                    if f.endswith('.a3m'):
                        base_name = os.path.splitext(os.path.join(dir_path, f))[0]
                        pdb_name = f"{base_name}_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_042.pdb"
                        if os.path.exists(pdb_name):
                            pdb_files.append(pdb_name)
                        
        return pdb_files
    
    def _extract_indices(self, indices_spec, default_indices):
        """Extract indices from specification or use defaults."""
        if not indices_spec:
            return default_indices
            
        if isinstance(indices_spec, list):
            result = []
            for range_dict in indices_spec:
                result.extend(range(range_dict['start'], range_dict['end']+1))
            return result
        else:
            return list(range(indices_spec['start'], indices_spec['end'] + 1))
    
    def create_1d_metric_bins(self, 
                           results: Dict[str, Dict[str, float]], 
                           metric_name: str, 
                           num_bins: int = 20,
                           min_value: Optional[float] = None,
                           max_value: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, int]]:
        """Create 1D bins for a specific metric and assign PDBs to bins.
        
        Args:
            results: Dictionary mapping PDB IDs to their metric values
            metric_name: Name of the metric to bin
            num_bins: Number of bins to create
            min_value: Optional minimum value for binning range. If None, uses min of data
            max_value: Optional maximum value for binning range. If None, uses max of data
            
        Returns:
            Tuple containing:
            - Array of bin edges
            - Dictionary mapping PDB IDs to bin indices
        """
        if not results:
            raise ValueError("No results provided for binning")
            
        metric_values = [metrics[metric_name] for metrics in results.values() if metric_name in metrics]
        if not metric_values:
            raise ValueError(f"No valid values found for metric {metric_name}. Check if metric calculation succeeded.")
            
        # Use provided min/max if given, otherwise use data min/max
        min_val = min_value if min_value is not None else min(metric_values)
        max_val = max_value if max_value is not None else max(metric_values)
        
        # Validate values are within range if min/max were provided
        if min_value is not None:
            assert all(v >= min_value for v in metric_values), f"Some {metric_name} values are below minimum {min_value}"
        if max_value is not None:
            assert all(v <= max_value for v in metric_values), f"Some {metric_name} values are above maximum {max_value}"
            
        bins = np.linspace(min_val, max_val, num_bins + 1)
        
        # Assign each PDB to a bin
        pdb_bins = {}
        for pdb, metrics in results.items():
            if metric_name in metrics:
                bin_idx = np.digitize(metrics[metric_name], bins) - 1
                pdb_bins[pdb] = bin_idx
            
        return bins, pdb_bins
    
    def create_focused_1d_bins(self,
                             results: Dict[str, Dict[str, float]],
                             metric_name: str,
                             num_bins: int,
                             min_value: Optional[float] = None,
                             max_value: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, int]]:
        """Create 1D bins with focus range, assigning outliers to edge bins.
        
        Args:
            results: Dictionary mapping PDB IDs to their metric values
            metric_name: Name of the metric to bin
            num_bins: Number of bins to create within focus range
            min_value: Minimum value of focus range (optional)
            max_value: Maximum value of focus range (optional)
            
        Returns:
            Tuple containing:
            - Array of bin edges including outlier bins
            - Dictionary mapping PDB IDs to bin indices
        """
        if not results:
            raise ValueError("No results provided for binning")
            
        metric_values = [metrics[metric_name] for metrics in results.values() if metric_name in metrics]
        if not metric_values:
            raise ValueError(f"No valid values found for metric {metric_name}")
            
        # Use provided min/max if given, otherwise use data min/max
        min_val = min_value if min_value is not None else min(metric_values)
        max_val = max_value if max_value is not None else max(metric_values)
        
        # Create bins within focus range
        focus_bins = np.linspace(min_val, max_val, num_bins + 1)
        
        # Add edge bins for outliers
        bins = np.concatenate([[float('-inf')], focus_bins, [float('inf')]])
        
        # Assign each PDB to a bin
        pdb_bins = {}
        for pdb, metrics in results.items():
            if metric_name in metrics:
                value = metrics[metric_name]
                if value < min_val:
                    pdb_bins[pdb] = 0  # Below focus range
                elif value > max_val:
                    pdb_bins[pdb] = num_bins + 1  # Above focus range
                else:
                    # Within focus range - subtract 1 to account for the below-range bin
                    bin_idx = np.digitize(value, focus_bins) 
                    pdb_bins[pdb] = bin_idx
                    
        return bins, pdb_bins

    def create_2d_metric_bins(self,
                            results: Dict[str, Dict[str, float]],
                            metric_names: List[str],
                            num_bins: int = 10) -> Tuple[List[np.ndarray], Dict[str, Tuple[int, int]]]:
        """Create 2D bins using two metrics and assign PDBs to 2D bin coordinates."""
        if not results or len(metric_names) != 2:
            raise ValueError("Need results and exactly two metric names for 2D binning")

        metric1_name, metric2_name = metric_names
        
        # Get values for both metrics
        metric1_values = [metrics[metric1_name] for metrics in results.values() if metric1_name in metrics]
        metric2_values = [metrics[metric2_name] for metrics in results.values() if metric2_name in metrics]
        
        if not metric1_values or not metric2_values:
            raise ValueError("No valid values found for one or both metrics")

        # Create bins for each dimension
        bins1 = np.linspace(min(metric1_values), max(metric1_values), num_bins + 1)
        bins2 = np.linspace(min(metric2_values), max(metric2_values), num_bins + 1)

        # Assign each PDB to a 2D bin coordinate
        pdb_bins = {}
        for pdb, metrics in results.items():
            if metric1_name in metrics and metric2_name in metrics:
                bin_idx1 = np.digitize(metrics[metric1_name], bins1) - 1
                bin_idx2 = np.digitize(metrics[metric2_name], bins2) - 1
                pdb_bins[pdb] = (bin_idx1, bin_idx2)

        return [bins1, bins2], pdb_bins

    def create_3d_metric_bins(self,
                            results: Dict[str, Dict[str, float]],
                            metric_names: List[str],
                            num_bins: int = 10) -> Tuple[List[np.ndarray], Dict[str, Tuple[int, int, int]]]:
        """Create 3D bins using three metrics and assign PDBs to 3D bin coordinates."""
        if not results or len(metric_names) != 3:
            raise ValueError("Need results and exactly three metric names for 3D binning")

        metric1_name, metric2_name, metric3_name = metric_names
        
        # Get values for all three metrics
        metric1_values = [metrics[metric1_name] for metrics in results.values() if metric1_name in metrics]
        metric2_values = [metrics[metric2_name] for metrics in results.values() if metric2_name in metrics]
        metric3_values = [metrics[metric3_name] for metrics in results.values() if metric3_name in metrics]
        
        if not metric1_values or not metric2_values or not metric3_values:
            raise ValueError("No valid values found for one or more metrics")

        # Create bins for each dimension
        bins1 = np.linspace(min(metric1_values), max(metric1_values), num_bins + 1)
        bins2 = np.linspace(min(metric2_values), max(metric2_values), num_bins + 1)
        bins3 = np.linspace(min(metric3_values), max(metric3_values), num_bins + 1)

        # Assign each PDB to a 3D bin coordinate
        pdb_bins = {}
        for pdb, metrics in results.items():
            if all(m in metrics for m in metric_names):
                bin_idx1 = np.digitize(metrics[metric1_name], bins1) - 1
                bin_idx2 = np.digitize(metrics[metric2_name], bins2) - 1
                bin_idx3 = np.digitize(metrics[metric3_name], bins3) - 1
                pdb_bins[pdb] = (bin_idx1, bin_idx2, bin_idx3)

        return [bins1, bins2, bins3], pdb_bins
        
    def get_sequence_votes(self, 
                         source_msa: str, 
                         sampling_base_dir: str, 
                         pdb_bins: Dict[str, Union[int, Tuple]],
                         is_2d: bool = False,
                         is_3d: bool = False,
                         vote_threshold: float = 0.0,
                         hierarchical: bool = False) -> Tuple[Dict, Dict]:
        """Get votes for each sequence based on metric bins."""
        if not os.path.exists(source_msa):
            raise FileNotFoundError(f"Source MSA file not found: {source_msa}")
            
        with open(source_msa) as f:
            source_headers = {line.strip()[1:].split()[0] for line in f if line.startswith('>')}
            
        if not source_headers:
            raise ValueError("No headers found in source MSA file")
        
        # Collect all A3M files and their corresponding PDB files
        a3m_files = self._collect_a3m_files(sampling_base_dir, hierarchical)

        if not a3m_files:
            raise ValueError("No valid A3M/PDB file pairs found")

        self.logger.info(f"Processing {len(a3m_files)} A3M files")

        # Process files in batches for better parallelization
        batch_size = max(1, len(a3m_files) // (self.max_workers * 4))
        self.logger.info(f"batch_size: {batch_size}")
        
        batches = [a3m_files[i:i + batch_size] for i in range(0, len(a3m_files), batch_size)]
        
        all_votes = defaultdict(list)
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            batch_args = [(batch, pdb_bins, source_headers, is_2d, is_3d) for batch in batches]
            futures = [executor.submit(self._process_a3m_batch, args) for args in batch_args]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing A3M batches"):
                batch_votes = future.result()
                for header, votes in batch_votes.items():
                    all_votes[header].extend(votes)

        # Process votes to find consensus
        sequence_votes = self._process_votes(all_votes, is_2d, is_3d, vote_threshold)

        if not sequence_votes:
            self.logger.warning("No sequence votes met the threshold criteria")

        return sequence_votes, dict(all_votes)
    
    def _collect_a3m_files(self, sampling_base_dir: str, hierarchical: bool) -> List[Tuple[str, str]]:
        """Collect A3M files and their corresponding PDB files from single or multi-round sampling."""
        a3m_files = []
        
        # Check if we have a rounds-based directory structure
        round_dirs = [d for d in os.listdir(sampling_base_dir) if d.startswith('round_')]
        
        if round_dirs:
            # Multi-round structure
            self.logger.info(f"Found {len(round_dirs)} sampling rounds")
            for round_dir in round_dirs:
                round_path = os.path.join(sampling_base_dir, round_dir)
                sampling_path = os.path.join(round_path, '02_sampling')
                
                if os.path.exists(sampling_path):
                    sampling_dirs = [d for d in os.listdir(sampling_path) if d.startswith('sampling_')]
                    
                    for sampling_dir in sampling_dirs:
                        dir_path = os.path.join(sampling_path, sampling_dir)
                        for group_file in os.listdir(dir_path):
                            if group_file.endswith('.a3m'):
                                a3m_path = os.path.join(dir_path, group_file)
                                base_name = os.path.splitext(group_file)[0]
                                pdb_name = f"{base_name}_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_042.pdb"
                                pdb_file = os.path.join(dir_path, pdb_name)
                                if os.path.exists(pdb_file):
                                    a3m_files.append((a3m_path, pdb_file))
        else:
            # Single round structure (flat directory)
            sampling_dirs = [d for d in os.listdir(sampling_base_dir) if d.startswith('sampling_')]
            
            for sampling_dir in sampling_dirs:
                dir_path = os.path.join(sampling_base_dir, sampling_dir)
                for a3m_file in os.listdir(dir_path):
                    if a3m_file.endswith('.a3m'):
                        a3m_path = os.path.join(dir_path, a3m_file)
                        base_name = os.path.splitext(a3m_file)[0]
                        pdb_name = f"{base_name}_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_042.pdb"
                        pdb_file = os.path.join(dir_path, pdb_name)
                        if os.path.exists(pdb_file):
                            a3m_files.append((a3m_path, pdb_file))
                            
        self.logger.info(f"Collected {len(a3m_files)} A3M/PDB file pairs")
        return a3m_files
    
    def _process_votes(self, all_votes, is_2d, is_3d, vote_threshold):
        """Process votes to find consensus for each sequence."""
        sequence_votes = {}

        for header, votes in all_votes.items():
            if not votes:
                continue
                
            total_votes = len(votes)
            
            if is_3d or is_2d:
                # Handle multi-dimensional votes
                vote_array = np.array(votes)
                unique_votes, vote_counts = np.unique(vote_array, axis=0, return_counts=True)
                most_common_idx = np.argmax(vote_counts)
                most_common_bin = tuple(unique_votes[most_common_idx])
                most_common_count = vote_counts[most_common_idx]
            else:
                # Handle 1D votes
                vote_array = np.array(votes)
                unique_votes, vote_counts = np.unique(vote_array, return_counts=True)
                most_common_idx = np.argmax(vote_counts)
                most_common_bin = unique_votes[most_common_idx]
                most_common_count = vote_counts[most_common_idx]
            
            # Only include votes that meet the threshold
            if most_common_count / total_votes >= vote_threshold:
                sequence_votes[header] = (most_common_bin, most_common_count, total_votes)

        return sequence_votes
    
    def _process_a3m_batch(self, batch_args):
        """Process a batch of A3M files in parallel"""
        a3m_files_batch, pdb_bins, source_headers, is_2d, is_3d = batch_args
        batch_votes = defaultdict(list)
        
        # Log sample information for debugging
        if source_headers:
            sample_headers = list(source_headers)[:5]  # Take first 5 headers as sample
            logging.info(f"Processing batch with {len(source_headers)} source headers")
            logging.info(f"Sample source headers: {sample_headers}...")
            
            if pdb_bins:
                sample_bins = list(pdb_bins.items())[:3]  # Take first 3 bin assignments as sample
                logging.info(f"PDB bins dictionary contains {len(pdb_bins)} entries")
                logging.info(f"Sample PDB bins: {sample_bins}...")
        else:
            logging.warning("No source headers found in the batch")
        
        for a3m_path, pdb_file in a3m_files_batch:
            if pdb_file in pdb_bins:
                try:
                    with open(a3m_path) as f:
                        headers = [line.strip()[1:].split('\t')[0] for line in f if line.startswith('>')]
                        
                    for header in headers:
                        if header in source_headers:
                            bin_assignment = pdb_bins[pdb_file]
                            
                            # Validate bin_assignment format based on dimensionality
                            if is_3d and (not isinstance(bin_assignment, tuple) or len(bin_assignment) != 3):
                                logging.warning(f"Expected 3D bin assignment for {pdb_file}, got {bin_assignment}")
                                continue
                            elif is_2d and (not isinstance(bin_assignment, tuple) or len(bin_assignment) != 2):
                                logging.warning(f"Expected 2D bin assignment for {pdb_file}, got {bin_assignment}")
                                continue
                            
                            # Add the bin assignment to the votes for this header
                            batch_votes[header].append(bin_assignment)
                except Exception as e:
                    logging.error(f"Error processing {a3m_path}: {str(e)}")
                        
        return dict(batch_votes)


class SequenceVotingPlotter:
    """
    Class for creating bin distribution plots from voting results.
    
    This class handles the creation of visualizations for sequence voting
    results, showing the distribution of sequences across metric bins.
    """
    
    def __init__(
        self,
        results_file: str,
        output_dir: str,
        initial_color: str = '#d3b0b0',
        end_color: Optional[str] = None,
        figsize: Tuple[float, float] = (10, 5),
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        x_ticks: Optional[List[int]] = None,
        num_bins: Optional[int] = None
    ):
        """Initialize the plotter with configuration parameters."""
        self.results_file = results_file
        self.output_dir = output_dir
        self.initial_color = initial_color
        self.end_color = end_color
        self.figsize = figsize
        self.y_min = y_min
        self.y_max = y_max
        self.x_ticks = x_ticks
        self.num_bins = num_bins
        self.logger = get_logger(__name__)
        
        # Configure matplotlib for publication-quality plots
        plt.rcParams.update({
            'font.family': ['sans-serif'],
            'font.sans-serif': ['DejaVu Sans'],
            'font.size': 24,
            'axes.labelsize': 24,
            'axes.titlesize': 24,
            'xtick.labelsize': 24,
            'ytick.labelsize': 24
        })
    
    def plot(self) -> Optional[str]:
        """Create distribution plot from voting results."""
        try:
            # Load results dataframe
            if not os.path.exists(self.results_file):
                self.logger.error(f"Results file not found: {self.results_file}")
                return None
                
            results_df = pd.read_csv(self.results_file)
            
            # Create output directory
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Check for 1D, 2D, or 3D results
            if "Bin_Assignment_1" in results_df.columns and "Bin_Assignment_2" in results_df.columns:
                if "Bin_Assignment_3" in results_df.columns:
                    # 3D voting results
                    self.logger.warning("3D voting results plotting not yet implemented")
                    return None
                else:
                    # 2D voting results
                    self.logger.warning("2D voting results plotting not yet implemented")
                    return None
            else:
                # 1D voting results
                if self.num_bins is not None:
                    num_bins = self.num_bins
                else:
                    num_bins = int(results_df['Bin_Assignment'].max())
                return self._plot_1d_distribution(results_df, num_bins)
            
        except Exception as e:
            self.logger.error(f"Error creating distribution plot: {str(e)}", exc_info=True)
            return None
    
    def _hex2color(self, hex_str: str) -> Tuple[float, float, float]:
        """Convert hex string to RGB tuple."""
        hex_str = hex_str.lstrip('#')
        return tuple(int(hex_str[i:i+2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore
    
    def _plot_1d_distribution(self, results_df: pd.DataFrame, num_bins: int) -> str:
        """Create 1D bin distribution plot."""
        plt.figure(figsize=self.figsize, dpi=600)
        
        bins = np.arange(1, num_bins + 2)  # 1 to num_bins+1
        
        # Create histogram data without plotting
        counts, bins, _ = plt.hist(results_df['Bin_Assignment'], bins=bins)
        plt.clf()  # Clear the figure
        
        if self.end_color is not None:  # Only create gradient if end_color is provided
            # Create color gradient
            initial_rgb = self._hex2color(self.initial_color)
            end_rgb = self._hex2color(self.end_color)
            
            # Generate gradient colors
            colors = []
            for i in range(len(bins)-1):
                ratio = i / (len(bins)-2)  # Linear gradient from 0 to 1
                color = tuple(initial_rgb[j] + (end_rgb[j] - initial_rgb[j]) * ratio for j in range(3))
                colors.append(color)
            
            # Plot each bar centered on bin numbers with gradient colors
            for i, (count, color) in enumerate(zip(counts, colors)):
                plt.bar(bins[i], count, width=1.0, align='center',
                       color=color, edgecolor=None)
        else:
            # Use single color if no end_color provided
            plt.bar(bins[:-1], counts, width=1.0, align='center',
                   color=self._hex2color(self.initial_color), edgecolor=None)
        
        # Add vertical dashed lines at bin boundaries
        for bin_edge in bins:
            plt.axvline(x=bin_edge - 0.5, color='gray', linestyle='--', alpha=0.5)
        
        plt.xlabel('Bin Assignment')
        plt.ylabel('Count')
        plt.yscale('log')
        plt.xlim(0.5, num_bins + 0.5)
        
        # Set custom x ticks if provided, otherwise use all bins
        if self.x_ticks is not None:
            plt.xticks(self.x_ticks)
        
        # Set y-axis limits if provided
        if self.y_min is not None and self.y_max is not None:
            plt.ylim(self.y_min, self.y_max)
        
        # Save plot
        os.makedirs(self.output_dir, exist_ok=True)
        plot_path = os.path.join(self.output_dir, 'sequence_voting_distribution.png')
        plt.savefig(plot_path, dpi=600, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Created distribution plot: {plot_path}")
        return plot_path


class SequenceVotingRunner:
    """
    Runner class for sequence voting analysis.
    
    This class handles the process of analyzing sampling results,
    creating metric bins, and assigning votes to sequences based on
    their structural properties.
    """
    
    def __init__(
        self,
        sampling_dir: str,
        source_msa: str,
        config_path: str,
        output_dir: str,
        num_bins: int = 20,
        max_workers: int = 32,
        vote_threshold: float = 0.0,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        use_focused_bins: bool = False,
        precomputed_metrics: Optional[str] = None,
        plddt_threshold: float = 0,
        filter_criterion: Optional[str] = None
    ):
        """
        Initialize the voting runner with configuration parameters.
        
        Args:
            sampling_dir: Directory containing sampling results
            source_msa: Path to source MSA file
            config_path: Path to configuration JSON file
            output_dir: Output directory for voting results
            num_bins: Number of bins for voting
            max_workers: Maximum number of concurrent workers
            vote_threshold: Threshold for vote filtering (0-1)
            min_value: Minimum value for metric binning range
            max_value: Maximum value for metric binning range
            use_focused_bins: Whether to use focused 1D binning with outlier bins
            precomputed_metrics: Path to precomputed metrics CSV file or directory
            plddt_threshold: pLDDT threshold for filtering structures
            filter_criterion: Specific filter criterion name to process
        """
        self.sampling_dir = sampling_dir
        self.source_msa = source_msa
        self.config_path = config_path
        self.output_dir = output_dir
        self.num_bins = num_bins
        self.max_workers = max_workers
        self.vote_threshold = vote_threshold
        self.min_value = min_value
        self.max_value = max_value
        self.use_focused_bins = use_focused_bins
        self.precomputed_metrics = precomputed_metrics
        self.plddt_threshold = plddt_threshold
        self.filter_criterion = filter_criterion
        
        # Initialize output directories
        self.voting_dir = self.output_dir
        os.makedirs(self.voting_dir, exist_ok=True)
        
        # Set up logger
        self.logger = get_logger(__name__)
        
        # Load configuration
        self._load_config()
        
        # Initialize voting analyzer
        self.analyzer = VotingAnalyzer(max_workers=self.max_workers)
    def _load_config(self) -> None:
        """Load configuration from JSON file."""
        try:
            with open(self.config_path) as f:
                self.config = json.load(f)
            
            # If a specific filter criterion is specified, filter the config
            if self.filter_criterion:
                # Find the specified criterion in the filter_criteria list
                filtered_criteria = [c for c in self.config['filter_criteria'] 
                                    if c.get('name') == self.filter_criterion]
                
                if not filtered_criteria:
                    self.logger.error(f"Filter criterion '{self.filter_criterion}' not found in config")
                    raise ValueError(f"Filter criterion '{self.filter_criterion}' not found in config")
                
                # Replace the filter_criteria list with just the selected criterion
                self.config['filter_criteria'] = filtered_criteria
            
            # Always use 1D voting by default
            self.is_2d = False
            self.is_3d = False
            
        except Exception as e:
            self.logger.error(f"Error reading config file: {str(e)}")
            raise ValueError(f"Invalid configuration file: {str(e)}")
    
    def setup_logging(self, log_file: Optional[str] = None) -> None:
        """
        Set up logging configuration.
        
        Args:
            log_file: Path to log file
        """
        if log_file is None:
            log_file = os.path.join(self.voting_dir, "sequence_voting.log")
            
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def run(self) -> Optional[str]:
        """
        Run the sequence voting analysis process.
        
        Returns:
            Path to results CSV file or None if process fails
        """
        self.logger.info(f"Starting sequence voting analysis for criterion: {self.filter_criterion or 'All'}")
        results_file = os.path.join(self.voting_dir, "voting_results.csv")
        
        try:
            # Check if results file already exists
            if os.path.exists(results_file):
                self.logger.info("Found existing results file, skipping computation")
                return results_file
            
            # Validate paths
            for path, name in [
                (self.sampling_dir, "Sampling directory"),
                (self.source_msa, "Source MSA file"),
                (self.config_path, "Config file")
            ]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"{name} not found: {path}")
            
            # Process sampling directories and get metrics
            self.logger.info("Processing sampling directories...")
            results = self.analyzer.process_sampling_dirs(
                self.sampling_dir, 
                self.config['filter_criteria'], 
                self.config['basics'],
                precomputed_metrics_file=self.precomputed_metrics,
                plddt_threshold=self.plddt_threshold
            )
            
            if not results:
                self.logger.warning("No valid results found in sampling directories")
                return None
                
            # Create bins for the current criterion
            self.logger.info("Creating metric bins...")
            criterion_name = self.config['filter_criteria'][0]['name']
            
            if self.use_focused_bins:
                bins, pdb_bins = self.analyzer.create_focused_1d_bins(
                    results,
                    criterion_name,
                    self.num_bins,
                    self.min_value,
                    self.max_value
                )
            else:
                bins, pdb_bins = self.analyzer.create_1d_metric_bins(
                    results,
                    criterion_name,
                    num_bins=self.num_bins,
                    min_value=self.min_value,
                    max_value=self.max_value
                )
            
            # Get sequence votes
            self.logger.info("Processing sequence votes...")
            sequence_votes, all_votes = self.analyzer.get_sequence_votes(
                self.source_msa,
                self.sampling_dir,
                pdb_bins,
                is_2d=self.is_2d,
                is_3d=self.is_3d,
                vote_threshold=self.vote_threshold
            )
            
            if not sequence_votes:
                self.logger.warning("No sequence votes met the threshold criteria")
                return None
                
            # Save raw sequence votes for later analysis
            self._save_raw_votes(all_votes)
            
            # Create and save results DataFrame
            results_df = self._create_results_dataframe(sequence_votes)
            results_df.to_csv(results_file, index=False)
            
            self.logger.info(f"Processed {len(sequence_votes)} sequences for criterion: {self.filter_criterion or 'All'}")
            self.logger.info(f"Saved voting results to {results_file}")
            
            return results_file
            
        except Exception as e:
            self.logger.error(f"An error occurred in sequence voting: {str(e)}", exc_info=True)
            return None
    
    def _save_raw_votes(self, all_votes):
        """Save raw sequence votes to JSON file."""
        raw_votes_file = os.path.join(self.voting_dir, "raw_sequence_votes.json")
        
        # Convert votes to serializable format
        serializable_votes = {}
        for header, votes in all_votes.items():
            if self.is_3d:
                serializable_votes[header] = [[int(v[0]), int(v[1]), int(v[2])] for v in votes]
            elif self.is_2d:
                serializable_votes[header] = [[int(v[0]), int(v[1])] for v in votes]
            else:
                serializable_votes[header] = [int(v) for v in votes]
            
        with open(raw_votes_file, 'w') as f:
            json.dump(serializable_votes, f)
        
        self.logger.info(f"Raw sequence votes saved to {raw_votes_file}")
    
    def _create_results_dataframe(self, sequence_votes):
        """Create DataFrame from sequence votes."""
        if self.is_3d:
            return pd.DataFrame([
                {
                    "Sequence_Header": header,
                    "Bin_Assignment_1": bin_nums[0] if isinstance(bin_nums, tuple) else bin_nums,
                    "Bin_Assignment_2": bin_nums[1] if isinstance(bin_nums, tuple) else bin_nums,
                    "Bin_Assignment_3": bin_nums[2] if isinstance(bin_nums, tuple) else bin_nums,
                    "Vote_Count": vote_count,
                    "Total_Votes": total_votes
                }
                for header, (bin_nums, vote_count, total_votes) in sequence_votes.items()
            ])
        elif self.is_2d:
            return pd.DataFrame([
                {
                    "Sequence_Header": header,
                    "Bin_Assignment_1": bin_nums[0] if isinstance(bin_nums, tuple) else bin_nums,
                    "Bin_Assignment_2": bin_nums[1] if isinstance(bin_nums, tuple) else bin_nums,
                    "Vote_Count": vote_count,
                    "Total_Votes": total_votes
                }
                for header, (bin_nums, vote_count, total_votes) in sequence_votes.items()
            ])
        else:
            return pd.DataFrame([
                {
                    "Sequence_Header": header,
                    "Bin_Assignment": bin_num,
                    "Vote_Count": vote_count,
                    "Total_Votes": total_votes
                }
                for header, (bin_num, vote_count, total_votes) in sequence_votes.items()
            ])