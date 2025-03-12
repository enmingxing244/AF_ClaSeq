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
from typing import Dict, List, Tuple, Any, Optional, Union
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from concurrent.futures import as_completed

from af_claseq.utils.structure_analysis import StructureAnalyzer


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
        hierarchical_sampling: bool = False
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
            precomputed_metrics: Path to precomputed metrics CSV file
            plddt_threshold: pLDDT threshold for filtering structures
            hierarchical_sampling: Whether sampling directories have hierarchical structure
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
        self.hierarchical_sampling = hierarchical_sampling
        
        # Initialize output directories
        self.voting_dir = os.path.join(self.output_dir)
        os.makedirs(self.voting_dir, exist_ok=True)
        
        # Set up logger
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self._load_config()
        
        # Initialize voting analyzer
        self.analyzer = VotingAnalyzer(max_workers=self.max_workers)
    
    def _load_config(self) -> None:
        """Load configuration from JSON file."""
        try:
            with open(self.config_path) as f:
                self.config = json.load(f)
            
            # Determine binning dimensionality
            self.is_2d = len(self.config['filter_criteria']) >= 2
            self.is_3d = len(self.config['filter_criteria']) >= 3
            
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
        self.logger.info("Starting sequence voting analysis")
        results_file = os.path.join(self.voting_dir, "sequence_voting_results.csv")
        
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
                plddt_threshold=self.plddt_threshold,
                hierarchical=self.hierarchical_sampling
            )
            
            if not results:
                self.logger.warning("No valid results found in sampling directories")
                return None
                
            # Create bins based on dimensionality
            self.logger.info("Creating metric bins...")
            pdb_bins = self._create_metric_bins(results)
            
            # Get sequence votes
            self.logger.info("Processing sequence votes...")
            sequence_votes, all_votes = self.analyzer.get_sequence_votes(
                self.source_msa,
                self.sampling_dir,
                pdb_bins,
                is_2d=self.is_2d,
                is_3d=self.is_3d,
                vote_threshold=self.vote_threshold,
                hierarchical=self.hierarchical_sampling
            )
            
            if not sequence_votes:
                self.logger.warning("No sequence votes met the threshold criteria")
                return None
                
            # Save raw sequence votes for later analysis
            self._save_raw_votes(all_votes)
            
            # Create and save results DataFrame
            results_df = self._create_results_dataframe(sequence_votes)
            results_df.to_csv(results_file, index=False)
            
            self.logger.info(f"Processed {len(sequence_votes)} sequences")
            self.logger.info(f"Saved voting results to {results_file}")
            
            return results_file
            
        except Exception as e:
            self.logger.error(f"An error occurred in sequence voting: {str(e)}", exc_info=True)
            return None
    
    def _create_metric_bins(self, results):
        """Create metric bins based on dimensionality."""
        if self.is_3d:
            metric_names = [criterion['name'] for criterion in self.config['filter_criteria'][:3]]
            bins, pdb_bins = self.analyzer.create_3d_metric_bins(
                results,
                metric_names,
                num_bins=self.num_bins
            )
        elif self.is_2d:
            metric_names = [criterion['name'] for criterion in self.config['filter_criteria'][:2]]
            bins, pdb_bins = self.analyzer.create_2d_metric_bins(
                results,
                metric_names,
                num_bins=self.num_bins
            )
        else:
            if self.use_focused_bins:
                bins, pdb_bins = self.analyzer.create_focused_1d_bins(
                    results,
                    self.config['filter_criteria'][0]['name'],
                    self.num_bins,
                    self.min_value,
                    self.max_value
                )
            else:
                bins, pdb_bins = self.analyzer.create_1d_metric_bins(
                    results,
                    self.config['filter_criteria'][0]['name'],
                    num_bins=self.num_bins,
                    min_value=self.min_value,
                    max_value=self.max_value
                )
        
        self.bins = bins
        return pdb_bins
    
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
            results_df = pd.DataFrame([
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
            results_df = pd.DataFrame([
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
            results_df = pd.DataFrame([
                {
                    "Sequence_Header": header,
                    "Bin_Assignment": bin_num,
                    "Vote_Count": vote_count,
                    "Total_Votes": total_votes
                }
                for header, (bin_num, vote_count, total_votes) in sequence_votes.items()
            ])
        
        return results_df


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
        x_ticks: Optional[List[int]] = None
    ):
        """
        Initialize the plotter with configuration parameters.
        
        Args:
            results_file: Path to voting results CSV file
            output_dir: Directory to save plots
            initial_color: Initial color for bin distribution plot (hex)
            end_color: End color for bin distribution plot (hex)
            figsize: Figure size in inches (width, height)
            y_min: Minimum y-axis value
            y_max: Maximum y-axis value
            x_ticks: Custom x-axis tick positions
        """
        self.results_file = results_file
        self.output_dir = output_dir
        self.initial_color = initial_color
        self.end_color = end_color
        self.figsize = figsize
        self.y_min = y_min
        self.y_max = y_max
        self.x_ticks = x_ticks
        
        # Set up logger
        self.logger = logging.getLogger(__name__)
        
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
        """
        Create distribution plot from voting results.
        
        Returns:
            Path to created plot file or None if process fails
        """
        try:
            # Load results dataframe
            if not os.path.exists(self.results_file):
                self.logger.error(f"Results file not found: {self.results_file}")
                return None
                
            results_df = pd.read_csv(self.results_file)
            
            # Determine if it's 2D or 3D from column names
            is_2d = 'Bin_Assignment_1' in results_df.columns and 'Bin_Assignment_2' in results_df.columns
            is_3d = is_2d and 'Bin_Assignment_3' in results_df.columns
            
            # Create output directory
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Determine number of bins from the data
            if is_3d or is_2d:
                self.logger.info("Multi-dimensional plotting not implemented yet")
                return None
            else:
                num_bins = int(results_df['Bin_Assignment'].max())
                plot_path = self._plot_1d_distribution(results_df, num_bins)
                
            return plot_path
            
        except Exception as e:
            self.logger.error(f"Error creating distribution plot: {str(e)}", exc_info=True)
            return None
    
    def _hex2color(self, hex_str: str) -> Tuple[float, float, float]:
        """Convert hex string to RGB tuple."""
        hex_str = hex_str.lstrip('#')
        # Create explicit tuple to satisfy the type checker
        r = int(hex_str[0:2], 16) / 255.0
        g = int(hex_str[2:4], 16) / 255.0
        b = int(hex_str[4:6], 16) / 255.0
        return (r, g, b)
    
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
            colors = []
            for i in range(len(bins)-1):
                ratio = i / (len(bins)-2)  # Linear gradient from 0 to 1
                r = initial_rgb[0] + (end_rgb[0] - initial_rgb[0]) * ratio
                g = initial_rgb[1] + (end_rgb[1] - initial_rgb[1]) * ratio
                b = initial_rgb[2] + (end_rgb[2] - initial_rgb[2]) * ratio
                colors.append((r, g, b))
            
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
        
        # Set x-axis limits and ticks
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


class VotingAnalyzer:
    def __init__(self, max_workers: int = 32):
        """
        Initialize the VotingAnalyzer.
        
        Args:
            max_workers: Maximum number of concurrent workers for processing
        """
        self.max_workers = max_workers
        self._setup_logging()
        self.structure_analyzer = StructureAnalyzer()

    def _setup_logging(self):
        """Set up basic logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def process_sampling_dirs(self, 
                              base_dir: str, 
                              filter_criteria: List[Dict[str, Any]], 
                              basics: Dict[str, Any],
                              precomputed_metrics_file: str = None,
                              plddt_threshold: float = 0,
                              hierarchical: bool = False) -> Dict[str, Dict[str, float]]:
        """Process sampling directories to calculate metrics in parallel.
        
        Args:
            base_dir: Base directory containing sampling results
            filter_criteria: List of criteria for filtering structures
            basics: Basic configuration parameters
            precomputed_metrics_file: Optional path to precomputed metrics CSV
            plddt_threshold: Minimum pLDDT score threshold
            hierarchical: Whether to process hierarchical sampling directories
        """
        results = {}

        # Check if precomputed metrics file exists and should be used
        if precomputed_metrics_file and os.path.exists(precomputed_metrics_file):
            logging.info(f"Loading precomputed metrics from {precomputed_metrics_file}")
            metrics_df = pd.read_csv(precomputed_metrics_file)
            
            # Convert DataFrame to expected results format
            for _, row in metrics_df.iterrows():
                pdb_path = row['PDB'] if 'PDB' in row else row.name
                
                # Skip if pLDDT is below threshold
                if 'plddt' in row and row['plddt'] <= plddt_threshold:
                    continue
                    
                metrics = {}
                for criterion in filter_criteria:
                    metric_name = criterion['name']
                    if metric_name in row:
                        metrics[metric_name] = row[metric_name]
                results[pdb_path] = metrics
                
            return results

        # Collect all PDB files
        pdb_files = []
        
        if hierarchical:
            # Get all sampling round directories
            sampling_rounds = [d for d in os.listdir(base_dir) if d.startswith('sampling_round_')]
            
            for round_dir in sampling_rounds:
                sampling_path = os.path.join(base_dir, round_dir, '02_sampling')
                if not os.path.exists(sampling_path):
                    continue
                    
                sampling_dirs = [d for d in os.listdir(sampling_path) if d.startswith('sampling_')]
                
                for sampling_dir in sampling_dirs:
                    dir_path = os.path.join(sampling_path, sampling_dir)
                    for group_dir in os.listdir(dir_path):
                        if group_dir.startswith('group_'):
                            group_path = os.path.join(dir_path, group_dir)
                            if group_path.endswith('.a3m'):
                                base_name = os.path.splitext(group_path)[0]
                                pdb_name = f"{base_name}_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_042.pdb"
                                if os.path.exists(pdb_name):
                                    pdb_files.append((pdb_name, filter_criteria, basics, plddt_threshold))
        else:
            # Get regular sampling directories
            sampling_dirs = [d for d in os.listdir(base_dir) if d.startswith('sampling_')]
            
            for sampling_dir in sampling_dirs:
                dir_path = os.path.join(base_dir, sampling_dir)
                for f in os.listdir(dir_path):
                    if f.endswith('.a3m'):
                        base_name = os.path.splitext(f)[0]
                        pdb_name = f"{base_name}_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_042.pdb"
                        pdb_path = os.path.join(dir_path, pdb_name)
                        if os.path.exists(pdb_path):
                            pdb_files.append((pdb_path, filter_criteria, basics, plddt_threshold))

        if not pdb_files:
            logging.warning("No PDB files found in sampling directories")
            return results

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._process_pdb_file, args) for args in pdb_files]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing PDB files"):
                result = future.result()
                if result:
                    pdb_path, metrics = result
                    results[pdb_path] = metrics
                    
        if not results:
            logging.warning("No valid results obtained from PDB processing")
            
        return results

    def _process_pdb_file(self, args):
        """Process a single PDB file and return metrics if valid."""
        pdb_path, filter_criteria, basics, plddt_threshold = args
        
        try:
            # Get pLDDT score using StructureAnalyzer
            plddt = self.structure_analyzer.plddt_process(pdb_path, list(range(basics['full_index']['start'], basics['full_index']['end'] + 1)))
            if not plddt:
                logging.warning(f"Could not calculate pLDDT for {pdb_path}")
                return None
                
            if plddt and plddt > plddt_threshold:
                metrics = {}
                for criterion in filter_criteria:
                    if criterion['type'] == 'distance':
                        indices1 = criterion['indices']['set1']
                        indices2 = criterion['indices']['set2']
                        distance = self.structure_analyzer.calculate_residue_distance(pdb_path, 'A', indices1, indices2)
                        metrics[criterion['name']] = distance
                    elif criterion['type'] == 'angle':
                        angle = self.structure_analyzer.calculate_angle(
                            pdb_path,
                            criterion['indices']['domain1'],
                            criterion['indices']['domain2'],
                            criterion['indices']['hinge']
                        )
                        metrics[criterion['name']] = angle
                    elif criterion['type'] == 'rmsd':
                        superposition_indices = range(
                            criterion['superposition_indices']['start'],
                            criterion['superposition_indices']['end'] + 1
                        )
                        rmsd_indices = []
                        if isinstance(criterion['rmsd_indices'], list):
                            for range_dict in criterion['rmsd_indices']:
                                rmsd_indices.extend(range(range_dict['start'], range_dict['end'] + 1))
                        else:
                            rmsd_indices = range(
                                criterion['rmsd_indices']['start'],
                                criterion['rmsd_indices']['end'] + 1
                            )
                        rmsd = self.structure_analyzer.calculate_ca_rmsd(
                            basics['reference_pdb'],
                            pdb_path,
                            list(superposition_indices),
                            list(rmsd_indices),
                            chain_id='A'
                        )
                        if rmsd is not None:
                            metrics[criterion['name']] = rmsd
                        else:
                            logging.warning(f"Could not calculate RMSD for {pdb_path}")
                            return None
                    elif criterion['type'] == 'tmscore':
                        tm_score = self.structure_analyzer.calculate_tm_score(pdb_path, basics['reference_pdb'])
                        metrics[criterion['name']] = tm_score

                return pdb_path, metrics
        except Exception as e:
            logging.error(f"Error processing {pdb_path}: {str(e)}")
            return None
            
        return None
    
    def create_1d_metric_bins(self, 
                           results: Dict[str, Dict[str, float]], 
                           metric_name: str, 
                           num_bins: int = 20,
                           min_value: float = None,
                           max_value: float = None) -> Tuple[np.ndarray, Dict[str, int]]:
        """Create 1D bins for a specific metric and assign PDBs to bins.
        
        Args:
            results: Dictionary mapping PDB IDs to their metric values
            metric_name: Name of the metric to bin
            num_bins: Number of bins to create
            min_value: Optional minimum value for binning range. If None, uses min of data
            max_value: Optional maximum value for binning range. If None, uses max of data
            
        Raises:
            ValueError: If no results provided or no valid metric values found
        """
        if not results:
            raise ValueError("No results provided for binning")
            
        metric_values = [metrics[metric_name] for metrics in results.values() if metric_name in metrics]
        if not metric_values:
            raise ValueError(f"No valid values found for metric {metric_name}. Check if metric calculation succeeded.")
            
        # Use provided min/max if given, otherwise use data min/max
        min_val = min_value if min_value is not None else min(metric_values)
        max_val = max_value if max_value is not None else max(metric_values)
        
        # Assert all values are within range if min/max were provided
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
                             x_min: float,
                             x_max: float) -> Tuple[np.ndarray, Dict[str, int]]:
        """Create 1D bins with focus range, assigning outliers to edge bins.
        
        Args:
            results: Dictionary mapping PDB IDs to their metric values
            metric_name: Name of the metric to bin
            num_bins: Number of bins to create within focus range
            x_min: Minimum value of focus range
            x_max: Maximum value of focus range
            
        Returns:
            Tuple containing:
            - Array of bin edges including outlier bins
            - Dictionary mapping PDB IDs to bin indices
            
        Raises:
            ValueError: If no results provided or no valid metric values found
        """
        if not results:
            raise ValueError("No results provided for binning")
            
        metric_values = [metrics[metric_name] for metrics in results.values() if metric_name in metrics]
        if not metric_values:
            raise ValueError(f"No valid values found for metric {metric_name}")
            
        # Create bins within focus range
        focus_bins = np.linspace(x_min, x_max, num_bins + 1)
        
        # Add edge bins for outliers
        bins = np.concatenate([[float('-inf')], focus_bins, [float('inf')]])
        
        # Assign each PDB to a bin
        pdb_bins = {}
        for pdb, metrics in results.items():
            if metric_name in metrics:
                value = metrics[metric_name]
                if value < x_min:
                    pdb_bins[pdb] = 0  # Below focus range
                elif value > x_max:
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
        """Create 3D bins using three metrics and assign PDBs to 3D bin coordinates.
        
        Args:
            results: Dictionary mapping PDB IDs to their metric values
            metric_names: List of three metric names to use for binning
            num_bins: Number of bins to create for each dimension
            
        Returns:
            Tuple containing:
            - List of three bin edge arrays (one for each dimension)
            - Dictionary mapping PDB IDs to 3D bin coordinates (x,y,z)
            
        Raises:
            ValueError: If input data is invalid
        """
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
                         pdb_bins: Union[Dict[str, int], Dict[str, Tuple[int, int]], Dict[str, Tuple[int, int, int]]],
                         is_2d: bool = False,
                         is_3d: bool = False,
                         vote_threshold: float = 0.0,
                         hierarchical: bool = False) -> Tuple[Dict[str, Union[Tuple[int, int, int], Tuple[Tuple[int, int], int, int], Tuple[Tuple[int, int, int], int, int]]], Dict[str, List]]:
        """Get votes for each sequence based on metric bins (1D, 2D or 3D)."""
        if not os.path.exists(source_msa):
            raise FileNotFoundError(f"Source MSA file not found: {source_msa}")
            
        with open(source_msa) as f:
            source_headers = {line.strip()[1:].split('\t')[0] for line in f if line.startswith('>')}
            
        if not source_headers:
            raise ValueError("No headers found in source MSA file")
        
        a3m_files = []
        if hierarchical:
            # Handle hierarchical directory structure
            sampling_rounds = [d for d in os.listdir(sampling_base_dir) if d.startswith('sampling_round_')]
            
            for round_dir in sampling_rounds:
                sampling_path = os.path.join(sampling_base_dir, round_dir, '02_sampling')
                if not os.path.exists(sampling_path):
                    continue
                    
                sampling_dirs = [d for d in os.listdir(sampling_path) if d.startswith('sampling_')]
                for sampling_dir in sampling_dirs:
                    dir_path = os.path.join(sampling_path, sampling_dir)
                    for f in os.listdir(dir_path):
                        if f.endswith('.a3m'):
                            a3m_files.append(os.path.join(dir_path, f))
        else:
            # Handle flat directory structure
            sampling_dirs = [d for d in os.listdir(sampling_base_dir) if d.startswith('sampling_')]
            for sampling_dir in sampling_dirs:
                dir_path = os.path.join(sampling_base_dir, sampling_dir)
                for f in os.listdir(dir_path):
                    if f.endswith('.a3m'):
                        a3m_files.append(os.path.join(dir_path, f))
        
        if not a3m_files:
            raise ValueError(f"No A3M files found in sampling directory: {sampling_base_dir}")
            
        # Process all A3M files and collect sequence votes
        header_votes = {}
        
        for a3m_file in tqdm(a3m_files, desc="Processing A3M files for votes"):
            # Determine corresponding PDB file
            base_name = os.path.splitext(os.path.basename(a3m_file))[0]
            pdb_path = os.path.join(os.path.dirname(a3m_file), f"{base_name}_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_042.pdb")
            
            if pdb_path not in pdb_bins:
                continue
                
            # Get bin assignment for this PDB
            bin_assignment = pdb_bins[pdb_path]
            
            # Read A3M file headers
            try:
                with open(a3m_file) as f:
                    lines = f.readlines()
                    
                headers = [line.strip()[1:].split('\t')[0] for line in lines if line.startswith('>')]
                if len(headers) <= 1:  # Skip if only the query sequence is present
                    continue
                    
                # Assign votes to each header in this A3M
                for header in headers[1:]:  # Skip the first header (query)
                    if header in source_headers:
                        if header not in header_votes:
                            header_votes[header] = []
                        header_votes[header].append(bin_assignment)
            except Exception as e:
                logging.warning(f"Error processing {a3m_file}: {str(e)}")
        
        # Calculate the most common bin for each sequence
        sequence_votes = {}
        
        for header, votes in header_votes.items():
            if not votes:
                continue
                
            # Different handling based on dimensionality
            if is_3d:
                # For 3D, count votes for each bin combination
                vote_counts = {}
                for vote in votes:
                    vote_str = str(vote)  # Convert tuple to string for counting
                    if vote_str not in vote_counts:
                        vote_counts[vote_str] = 0
                    vote_counts[vote_str] += 1
                
                # Find the most common bin
                most_common_vote = max(vote_counts.items(), key=lambda x: x[1])
                most_common_bin = eval(most_common_vote[0])  # Convert string back to tuple
                vote_count = most_common_vote[1]
                
            elif is_2d:
                # For 2D, count votes for each bin combination
                vote_counts = {}
                for vote in votes:
                    vote_str = str(vote)  # Convert tuple to string for counting
                    if vote_str not in vote_counts:
                        vote_counts[vote_str] = 0
                    vote_counts[vote_str] += 1
                
                # Find the most common bin
                most_common_vote = max(vote_counts.items(), key=lambda x: x[1])
                most_common_bin = eval(most_common_vote[0])  # Convert string back to tuple
                vote_count = most_common_vote[1]
                
            else:
                # For 1D, use simple Counter
                from collections import Counter
                vote_counter = Counter(votes)
                most_common_bin, vote_count = vote_counter.most_common(1)[0]
            
            total_votes = len(votes)
            vote_ratio = vote_count / total_votes
            
            # Only include if vote ratio meets threshold
            if vote_ratio >= vote_threshold:
                sequence_votes[header] = (most_common_bin, vote_count, total_votes)
        
        return sequence_votes, header_votes