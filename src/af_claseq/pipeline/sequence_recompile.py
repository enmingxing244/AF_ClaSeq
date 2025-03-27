import os
import logging
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, hex2color
from collections import Counter
import pandas as pd

from af_claseq.utils.sequence_processing import read_a3m_to_dict, get_protein_sequence


class SequenceRecompiler:
    """
    Class for recompiling sequences from specific bins and preparing for prediction.
    
    This class handles:
    - Extracting sequences by bin from voting results
    - Creating prediction MSA files from selected bins
    - Creating control MSA files with random sequences
    - Generating vote distribution plots
    """
    
    def __init__(
        self,
        output_dir: str,
        source_msa: str,
        default_pdb: str,
        voting_results: str,
        bin_numbers: Union[List[int], int],
        num_total_bins: int = 20,
        initial_color: str = '#2486b9',
        combine_bins: bool = False,
        ratio_colorbar_min_max: Optional[Tuple[float, float]] = None,
        raw_votes_json: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the SequenceRecompiler.
        
        Args:
            output_dir: Output directory for recompiled sequences
            source_msa: Path to source MSA file
            default_pdb: Path to reference PDB file for getting query sequence
            voting_results: Path to voting results CSV file
            bin_numbers: List of bin numbers or single bin number to compile sequences from
            num_total_bins: Number of total bins to include in vote distribution plots
            initial_color: Initial color for vote distribution plots in hex format
            combine_bins: Whether to combine sequences from all input bins into one directory
            ratio_colorbar_min_max: Min and max values for ratio colorbar in sequence vote ratio plots
            raw_votes_json: Path to raw votes JSON file (optional)
            logger: Logger instance (optional, will create new one if not provided)
        """
        self.output_dir = output_dir
        self.source_msa = source_msa
        self.default_pdb = default_pdb
        self.voting_results = voting_results
        # Convert bin_numbers to list if it's an integer
        self.bin_numbers = bin_numbers if isinstance(bin_numbers, list) else [bin_numbers]
        self.num_total_bins = num_total_bins
        self.initial_color = initial_color
        self.combine_bins = combine_bins
        self.ratio_colorbar_min_max = ratio_colorbar_min_max
        self.raw_votes_json = raw_votes_json
        
        # Set up logging
        self.logger = logger or self._setup_logging()
        
        # Set up matplotlib parameters
        self._setup_plot_params()
        
        # Directories to store outputs
        self.dirs = {}
        
        # Cache for query sequence
        self._query_seq = None
        
        # Raw votes data
        self.raw_votes_data = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Set up basic logging configuration."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def _setup_plot_params(self) -> None:
        """Set up matplotlib plot parameters."""
        plt.rcParams.update({
            'font.size': 16,
            'axes.labelsize': 16,
            'axes.titlesize': 18,
            'xtick.labelsize': 16,
            'ytick.labelsize': 16,
            'legend.fontsize': 16
        })
    
    def setup_directories(self) -> Dict[str, str]:
        """
        Create necessary directories for output.
        
        Returns:
            Dictionary of created directory paths
        """
        prediction_dir = os.path.join(self.output_dir, "prediction")
        control_prediction_dir = os.path.join(self.output_dir, "control_prediction")
        plots_dir = os.path.join(self.output_dir, "plots")
        
        for dir_path in [self.output_dir, prediction_dir, control_prediction_dir, plots_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        self.dirs = {
            "prediction": prediction_dir,
            "control_prediction": control_prediction_dir,
            "plots": plots_dir
        }
        
        return self.dirs
    
    def load_raw_votes_data(self) -> bool:
        """
        Load raw votes data from JSON file if available.
        
        Returns:
            Boolean indicating success
        """
        if not self.raw_votes_json:
            return False
        
        try:
            with open(self.raw_votes_json, 'r') as f:
                self.raw_votes_data = json.load(f)
            self.logger.info(f"Loaded raw votes data from {self.raw_votes_json}")
            return True
        except Exception as e:
            self.logger.warning(f"Could not load raw votes data: {e}")
            return False
    
    def get_bin_headers(self, bin_numbers: Optional[List[int]] = None) -> List[str]:
        """
        Extract headers for sequences in specified bins.
        
        Args:
            bin_numbers: List of bin numbers to filter by (uses self.bin_numbers if None)
            
        Returns:
            List of sequence headers in the specified bins
        """
        bin_numbers = bin_numbers or self.bin_numbers
        bin_headers = []
        
        # Try to use pandas for faster processing if file is large
        try:
            df = pd.read_csv(self.voting_results)
            # Check column names to handle different formats
            if 'Bin_Assignment' in df.columns:
                bin_col = 'Bin_Assignment'
                header_col = 'Sequence_Header'
            elif 'bin' in df.columns:
                bin_col = 'bin'
                header_col = 'Sequence_Header' if 'Sequence_Header' in df.columns else 'sequence_id'
            else:
                raise ValueError(f"Could not determine column names in {self.voting_results}")
                
            filtered_df = df[df[bin_col].isin(bin_numbers)]
            bin_headers = filtered_df[header_col].tolist()
            
        except Exception as e:
            # Fall back to manual parsing if pandas fails
            self.logger.warning(f"Falling back to manual CSV parsing: {e}")
            with open(self.voting_results) as f:
                header_line = next(f).strip().split(',')
                # Try to determine column indices
                try:
                    bin_idx = header_line.index('Bin_Assignment')
                    header_idx = header_line.index('Sequence_Header')
                except ValueError:
                    try:
                        bin_idx = header_line.index('bin')
                        header_idx = (
                            header_line.index('Sequence_Header') 
                            if 'Sequence_Header' in header_line 
                            else header_line.index('sequence_id')
                        )
                    except ValueError:
                        # Assume first column is header, second is bin
                        header_idx, bin_idx = 0, 1
                
                for line in f:
                    parts = line.strip().split(',')
                    header = parts[header_idx]
                    try:
                        bin_num = int(parts[bin_idx])
                        if bin_num in bin_numbers:
                            bin_headers.append(header)
                    except (IndexError, ValueError):
                        continue
        
        self.logger.info(f"Found {len(bin_headers)} sequences in specified bins")
        return bin_headers
    
    def get_query_sequence(self) -> str:
        """
        Get query sequence from PDB file with caching.
        
        Returns:
            Query sequence string
        """
        if self._query_seq is None:
            self._query_seq = get_protein_sequence(self.default_pdb)
        return self._query_seq
    
    def compile_sequences(self, bin_headers: List[str], output_file: str) -> None:
        """
        Extract and compile sequences from source MSA.
        
        Args:
            bin_headers: List of sequence headers to include
            output_file: Path to output A3M file
        """
        query_seq = self.get_query_sequence()
        
        # Read all sequences from MSA file
        sequences = read_a3m_to_dict(self.source_msa)
        
        # Create a set of headers for faster lookup
        header_set = set(bin_headers)
        
        # Extract sequences for selected bin headers
        bin_seqs = []
        for header, seq in sequences.items():
            if header[1:] in header_set:  # Remove '>' from header for comparison
                bin_seqs.append(f"{header}\n{seq}\n")
        
        # Write compiled sequences with query first
        with open(output_file, "w") as f:
            f.write(f">query\n{query_seq}\n")
            for seq in bin_seqs:
                f.write(seq)
        
        self.logger.info(f"Compiled {len(bin_seqs)} sequences to {output_file}")
    
    def create_control_file(self, bin_headers: List[str], output_file: str) -> None:
        """
        Create control prediction file with random sequences.
        
        Args:
            bin_headers: List of sequence headers (used for count only)
            output_file: Path to output A3M file
        """
        query_seq = self.get_query_sequence()
        
        # Read sequences
        sequences = read_a3m_to_dict(self.source_msa)
        
        # Convert to list of (header, sequence) tuples for random sampling
        sequence_items = list(sequences.items())
        
        # Randomly select sequences
        if len(sequence_items) < len(bin_headers):
            self.logger.warning(
                f"Not enough sequences in MSA. Using {len(sequence_items)} instead of {len(bin_headers)}"
            )
            selected_items = sequence_items
        else:
            selected_items = random.sample(sequence_items, len(bin_headers))
        
        # Write sequences to file with query first
        with open(output_file, 'w') as f:
            f.write(f">query\n{query_seq}\n")
            for header, seq in selected_items:
                f.write(f"{header}\n{seq}\n")
        
        self.logger.info(f"Created control file with {len(selected_items)} random sequences")
    
    def plot_sequence_vote_ratios(
        self,
        bin_headers: List[str],
        target_bin: int,
        limit_sequences: int = 100
    ) -> None:
        """
        Create vote distribution plot for sequences in a specific bin.
        
        Args:
            bin_headers: List of sequence headers to include in plot
            target_bin: Bin number for the plot title
            limit_sequences: Maximum number of sequences to include in plot
        """
        if not self.raw_votes_data:
            self.logger.warning("No raw votes data available for plotting")
            return
        
        total_bins = list(range(1, self.num_total_bins + 1))
        
        # Calculate ratios for sequences in this bin only
        bin_ratios = []
        seq_names = []
        
        # Limit the number of sequences to avoid huge plots
        headers_to_plot = bin_headers[:limit_sequences]
        if len(bin_headers) > limit_sequences:
            self.logger.info(f"Limiting plot to {limit_sequences} sequences out of {len(bin_headers)}")
        
        for header in headers_to_plot:
            if header in self.raw_votes_data:
                value = self.raw_votes_data[header]
                # Count occurrences of each bin
                bin_counts = Counter(value)
                total_votes = len(value)
                
                # Calculate ratio for selected bins only
                ratios = []
                for bin_num in total_bins:
                    ratio = bin_counts.get(bin_num, 0) / total_votes
                    ratios.append(ratio)
                    
                bin_ratios.append(ratios)
                seq_names.append(header)
        
        if not bin_ratios:
            self.logger.warning("No sequences with vote data found for plotting")
            return
        
        # Convert to numpy array
        bin_ratios = np.array(bin_ratios)
        
        # Calculate figure size based on number of bins
        # When num_bins=20, we want width=8
        # Scale width proportionally for different num_bins
        width = 8 * (self.num_total_bins/20)
        
        # Adjust height based on sequence count with minimum and maximum constraints
        # Ensure height per sequence is reasonable even with few sequences
        if len(seq_names) <= 5:
            # For very few sequences, use fixed height per sequence
            height = max(4, len(seq_names) * 0.8)  # Minimum height of 4
        else:
            # Scale height with sequence count but with decreasing per-sequence height as count increases
            height = min(40, max(6, len(seq_names) * 0.3))  # Cap at 40, minimum of 6
        
        # Create figure
        fig, ax = plt.subplots(figsize=(width, height))
        # Convert hex color to RGB
        r, g, b = hex2color(self.initial_color)
        
        # Create custom non-linear colormap
        def nonlinear_gradient(x):
            return np.power(x, 0.8)
        
        N = 256
        vals = np.ones((N, 4))
        vals[:, 0] = np.linspace(1, r, N)
        vals[:, 1] = np.linspace(1, g, N)
        vals[:, 2] = np.linspace(1, b, N)
        vals = nonlinear_gradient(vals)
        custom_cmap = LinearSegmentedColormap.from_list("custom", vals)
        
        # Create heatmap with optional min/max values
        if self.ratio_colorbar_min_max is not None:
            im = ax.imshow(bin_ratios, cmap=custom_cmap, aspect='auto', 
                          vmin=self.ratio_colorbar_min_max[0], 
                          vmax=self.ratio_colorbar_min_max[1])
        else:
            im = ax.imshow(bin_ratios, cmap=custom_cmap, aspect='auto')
        plt.colorbar(im, ax=ax, label='Ratio of votes')
        
        # Customize axes
        ax.set_xlabel('Bin Index', fontsize=12)
        ax.set_ylabel('Sequence Header', fontsize=12)
        
        # Adjust tick frequency based on number of bins
        tick_step = max(1, self.num_total_bins // 20)  # Show at most ~20 ticks
        tick_positions = range(0, len(total_bins), tick_step)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([total_bins[i] for i in tick_positions])
        plt.xticks(rotation=45)
        
        # Only show a subset of y-ticks if there are many sequences
        if len(seq_names) > 30:
            tick_interval = max(1, len(seq_names) // 30)
            y_positions = range(0, len(seq_names), tick_interval)
            ax.set_yticks(y_positions)
            ax.set_yticklabels([seq_names[i] for i in y_positions], fontsize=10)
        else:
            ax.set_yticks(range(len(seq_names)))
            ax.set_yticklabels(seq_names, fontsize=10)
        
        plt.title(f'Vote Distribution for Bin {target_bin} Sequences')
        plt.tight_layout()
        
        # Save plot
        plot_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f'{target_bin}_sequence_vote_ratios.png'), 
                   dpi=600, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Created vote distribution plot for bin {target_bin}")
    
    def process_combined_bins(self) -> None:
        """Process combined bins into a single directory."""
        # Create combined bin name (e.g. "bin_5_6_7")
        bin_name = f"bin_{'_'.join(str(b) for b in sorted(self.bin_numbers))}"
        self.logger.info(f"Processing combined bins as {bin_name}")
        
        # Get headers for all specified bins
        self.logger.info(f"Extracting headers for combined bins")
        bin_headers = self.get_bin_headers()
        
        # Create prediction directory for combined bins
        prediction_bin_dir = os.path.join(self.dirs["prediction"], bin_name)
        os.makedirs(prediction_bin_dir, exist_ok=True)
        
        # Compile sequences for prediction
        prediction_file = os.path.join(prediction_bin_dir, f"{bin_name}_sequences.a3m")
        self.logger.info(f"Compiling sequences for prediction")
        self.compile_sequences(bin_headers=bin_headers, output_file=prediction_file)
        
        # Create control prediction with random sequences
        self.logger.info(f"Creating control prediction file")
        control_prediction_bin_dir = os.path.join(self.dirs["control_prediction"], bin_name)
        os.makedirs(control_prediction_bin_dir, exist_ok=True)
        
        control_prediction_file = os.path.join(control_prediction_bin_dir, f"{bin_name}_random_seq.a3m")
        self.create_control_file(bin_headers=bin_headers, output_file=control_prediction_file)
        
        self.logger.info(f"Completed processing combined bins")
    
    def process_individual_bins(self) -> None:
        """Process each bin individually."""
        # Check if bin_numbers is a single integer wrapped in a list
        if len(self.bin_numbers) == 1:
            
            bin_number = self.bin_numbers[0]
            self.logger.info(f"Processing bin {bin_number}")
            
            # Get headers for current bin
            self.logger.info(f"Extracting headers for bin {bin_number}")
            bin_headers = self.get_bin_headers(bin_numbers=[bin_number])
            
            # Create vote distribution plot if raw votes data is available
            if self.raw_votes_data:
                self.logger.info(f"Creating vote distribution plot for bin {bin_number}")
                try:
                    self.plot_sequence_vote_ratios(
                        bin_headers=bin_headers,
                        target_bin=bin_number
                    )
                except Exception as e:
                    self.logger.error(f"Error creating vote distribution plot for bin {bin_number}: {e}")
            
            # Create prediction directory for this bin
            prediction_bin_dir = os.path.join(self.dirs["prediction"], f"bin_{bin_number}")
            os.makedirs(prediction_bin_dir, exist_ok=True)
            
            # Compile sequences for prediction
            prediction_file = os.path.join(prediction_bin_dir, f"bin{bin_number}_sequences.a3m")
            self.logger.info(f"Compiling sequences for prediction")
            self.compile_sequences(bin_headers=bin_headers, output_file=prediction_file)
            
            # Create control prediction with random sequences
            self.logger.info(f"Creating control prediction file")
            control_prediction_bin_dir = os.path.join(self.dirs["control_prediction"], f"bin_{bin_number}")
            os.makedirs(control_prediction_bin_dir, exist_ok=True)
            
            control_prediction_file = os.path.join(control_prediction_bin_dir, f"bin{bin_number}_random_seq.a3m")
            self.create_control_file(bin_headers=bin_headers, output_file=control_prediction_file)
            
            self.logger.info(f"Completed processing bin {bin_number}")
        else:
            # Process multiple bins with a for loop
            for bin_number in self.bin_numbers:
                self.logger.info(f"Processing bin {bin_number}")
                
                # Get headers for current bin
                self.logger.info(f"Extracting headers for bin {bin_number}")
                bin_headers = self.get_bin_headers(bin_numbers=[bin_number])
                
                # Create vote distribution plot if raw votes data is available
                if self.raw_votes_data:
                    self.logger.info(f"Creating vote distribution plot for bin {bin_number}")
                    try:
                        self.plot_sequence_vote_ratios(
                            bin_headers=bin_headers,
                            target_bin=bin_number
                        )
                    except Exception as e:
                        self.logger.error(f"Error creating vote distribution plot for bin {bin_number}: {e}")
                
                # Create prediction directory for this bin
                prediction_bin_dir = os.path.join(self.dirs["prediction"], f"bin_{bin_number}")
                os.makedirs(prediction_bin_dir, exist_ok=True)
                
                # Compile sequences for prediction
                prediction_file = os.path.join(prediction_bin_dir, f"bin{bin_number}_sequences.a3m")
                self.logger.info(f"Compiling sequences for prediction")
                self.compile_sequences(bin_headers=bin_headers, output_file=prediction_file)
                
                # Create control prediction with random sequences
                self.logger.info(f"Creating control prediction file")
                control_prediction_bin_dir = os.path.join(self.dirs["control_prediction"], f"bin_{bin_number}")
                os.makedirs(control_prediction_bin_dir, exist_ok=True)
                
                control_prediction_file = os.path.join(control_prediction_bin_dir, f"bin{bin_number}_random_seq.a3m")
                self.create_control_file(bin_headers=bin_headers, output_file=control_prediction_file)
                
                self.logger.info(f"Completed processing bin {bin_number}")
    
    def recompile_sequences(self) -> Dict[str, str]:
        """
        Main method to recompile sequences from specific bins.
        
        Returns:
            Dictionary with output directory paths
        """
        # Set up directories
        self.setup_directories()
        
        # Load raw votes data if available
        self.load_raw_votes_data()
        
        # Process bins according to configuration
        if self.combine_bins:
            self.process_combined_bins()
        else:
            self.process_individual_bins()
        
        self.logger.info("Completed recompilation for all bins")
        return self.dirs
