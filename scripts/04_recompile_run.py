import os
import argparse
import logging
import shutil
from pathlib import Path
import random
from typing import List, Dict
import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, hex2color
from collections import Counter
from af_claseq.sequence_processing import read_sequences, process_sequences, get_protein_sequence


plt.rcParams.update({

    'font.size': 16,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16
})


def parse_args():
    parser = argparse.ArgumentParser(description='Recompile sequences from specific bin and prepare for prediction')
    parser.add_argument('--output_dir', required=True,
                      help='Output directory for recompiled sequences')
    parser.add_argument('--bin_numbers', type=int, nargs='+', required=True,
                      help='List of bin numbers to compile sequences from (e.g. 4 5 6 7 8 9)')
    parser.add_argument('--source_msa', required=True,
                      help='Path to source MSA file')
    parser.add_argument('--num_selections', type=int, default=20,
                      help='Number of random selections to create')
    parser.add_argument('--default_pdb', required=True,
                      help='Path to reference PDB file for getting query sequence')
    parser.add_argument('--seq_num_per_selection', type=int, default=12,
                      help='Number of sequences per random selection')
    parser.add_argument('--voting_results_path', required=True,
                      help='Path to voting results CSV file')
    parser.add_argument('--raw_votes_json', required=True,
                      help='Path to raw sequence votes JSON file')
    parser.add_argument('--initial_color', type=str, default='#2486b9',
                      help='Initial color for vote distribution plots in hex format')
    parser.add_argument('--num_total_bins', type=int, default=20,
                      help='Number of total bins to include in vote distribution plots')
    parser.add_argument('--combine_bins', action='store_true',
                      help='Whether to combine sequences from all input bins into one directory')
    parser.add_argument('--ratio_colorbar_min_max', type=float, nargs=2, default=None,
                      help='Min and max values for ratio colorbar in sequence vote ratio plots')
    
    return parser.parse_args()

def plot_sequence_vote_ratios(data: Dict, 
                              bin_headers: List[str], 
                              target_bin: int, 
                              output_dir: str, 
                              initial_color: str,
                              num_bins: int,
                              ratio_min_max: List[float] | None = None):
    """Create vote distribution plot for sequences in a specific bin"""
    total_bins = list(range(1, num_bins + 1))
    
    # Calculate ratios for sequences in this bin only
    bin_ratios = []
    seq_names = []

    for header in bin_headers:
        if header in data:
            value = data[header]
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
        return

    # Convert to numpy array
    bin_ratios = np.array(bin_ratios)

    # Calculate figure size based on number of bins
    # When num_bins=20, we want width=8
    # Scale width proportionally for different num_bins
    width = 8 * (num_bins/20) 
    height = len(seq_names)*0.3  # Keep height scaling the same

    # Create figure
    fig, ax = plt.subplots(figsize=(width, height))

    # Convert hex color to RGB
    r, g, b = hex2color(initial_color)

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
    if ratio_min_max is not None:
        im = ax.imshow(bin_ratios, cmap=custom_cmap, aspect='auto', vmin=ratio_min_max[0], vmax=ratio_min_max[1])
    else:
        im = ax.imshow(bin_ratios, cmap=custom_cmap, aspect='auto')
    plt.colorbar(im, ax=ax, label='Ratio of votes')

    # Customize axes
    ax.set_xlabel('Bin Index', fontsize=12)
    ax.set_ylabel('Sequence Header', fontsize=12)
    
    # Adjust tick frequency based on number of bins
    tick_step = max(1, num_bins // 20)  # Show at most ~20 ticks
    tick_positions = range(0, len(total_bins), tick_step)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([total_bins[i] for i in tick_positions])
    plt.xticks(rotation=45)
    
    ax.set_yticks(range(len(seq_names)))
    ax.set_yticklabels(seq_names, fontsize=10)

    plt.title(f'Vote Distribution for Bin {bin_num} Sequences')
    plt.tight_layout()

    # Save plot
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f'{target_bin}_sequence_vote_ratios.png'), dpi=600, bbox_inches='tight')
    plt.close()

def setup_directories(output_dir: str) -> Dict[str, str]:
    """
    Create necessary directories for output
    
    Output structure:
    output_dir/
        prediction/
            bin4/
                bin4_sequences.a3m
            bin5/
                bin5_sequences.a3m
            ...
        control_prediction/
            bin_4/
                bin4_random_seq.a3m
            bin_5/
                bin5_random_seq.a3m
            ...
        random_selection/
            bin_4/
                selection_1.a3m
                selection_2.a3m
                ...
            bin_5/
                selection_1.a3m
                ...
        control_random_selection/
            bin_4/
                selection_1.a3m
                selection_2.a3m
                ...
            bin_5/
                selection_1.a3m
                ...
        plots/
            bin4_sequence_vote_ratios.png
            bin5_sequence_vote_ratios.png
            ...
    """
    prediction_dir = os.path.join(output_dir, "prediction")
    control_prediction_dir = os.path.join(output_dir, "control_prediction")
    random_selection_dir = os.path.join(output_dir, "random_selection")
    control_random_selection_dir = os.path.join(output_dir, "control_random_selection")
    plots_dir = os.path.join(output_dir, "plots")
    
    for dir_path in [output_dir, prediction_dir, control_prediction_dir, 
                    random_selection_dir, control_random_selection_dir, plots_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    return {
        "prediction": prediction_dir,
        "control_prediction": control_prediction_dir,
        "random_selection": random_selection_dir,
        "control_random_selection": control_random_selection_dir,
        "plots": plots_dir
    }

def get_bin_headers(voting_results: str, 
                    bin_numbers: List[int]) -> List[str]:
    """Extract headers for sequences in specified bins"""
    bin_headers = []
    with open(voting_results) as f:
        next(f)  # Skip header line
        for line in f:
            parts = line.strip().split(',')
            header = parts[0]
            bin_num = int(parts[1])
            if bin_num in bin_numbers:
                bin_headers.append(header)
    return bin_headers

def compile_sequences(source_msa: str, 
                      bin_headers: List[str], 
                      output_file: str, 
                      default_pdb: str) -> None:
    """Extract and compile sequences from source MSA"""
    # Get query sequence from PDB file
    query_seq = get_protein_sequence(default_pdb)
    
    # Read all sequences from MSA file
    sequences = read_sequences(source_msa)

    
    # Extract sequences for selected bin headers
    bin_seqs = []
    for seq in sequences:
        header = seq.split('\n')[0][1:] # Remove '>' from header
        if header in bin_headers:
            bin_seqs.append(seq)
            
    # Write compiled sequences with query first
    with open(output_file, "w") as f:
        f.write(f">query\n{query_seq}\n")
        for seq in bin_seqs:
            f.write(seq)

def create_random_selected_groups(input_msa: str, 
                                  output_dir: str, 
                                  num_selections: int,
                                  seq_num_per_selection: int,
                                  default_pdb: str) -> None:
    """
    Create random sequence selections by randomly picking sequences multiple times.
    
    Args:
        input_msa: Path to input MSA file
        output_dir: Directory to write output files
        num_selections: Number of random selections to perform
        seq_num_per_selection: Number of sequences per random selection
        default_pdb: Path to PDB file for query sequence
    """
    # Get query sequence from PDB
    query_seq = get_protein_sequence(default_pdb)
    
    # Read sequences
    sequences = read_sequences(input_msa)
    non_query_seqs = sequences[1:]  # Skip first sequence since we'll use PDB query
    
    # Perform random selections
    for i in range(1, num_selections + 1):
        selected_seqs = random.sample(non_query_seqs, seq_num_per_selection)
        output_file = os.path.join(output_dir, f"selection_{i}.a3m")
        
        # Write sequences to file with query first
        with open(output_file, 'w') as f:
            f.write(f">query\n{query_seq}\n")
            for seq in selected_seqs:
                f.write(seq)

def main():
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s - %(levelname)s - %(message)s')

    # Create directories 
    dirs = setup_directories(output_dir=args.output_dir)
    
    # Load raw votes data
    with open(args.raw_votes_json, 'r') as f:
        raw_votes_data = json.load(f)
    
    if args.combine_bins:
        logging.info(f"Combining bins {args.bin_numbers}")
        logging.info("At this case, the plot will not be plot since the combined bins were used")
        # Create combined bin name (e.g. "bin_5_6_7")
        bin_name = f"bin_{'_'.join(str(b) for b in sorted(args.bin_numbers))}"
        logging.info(f"Processing combined bins as {bin_name}")
        
        # Get headers for all specified bins
        logging.info(f"Extracting headers for combined bins")
        bin_headers = get_bin_headers(voting_results=args.voting_results_path, 
                                    bin_numbers=args.bin_numbers)
        
        # Create prediction directory for combined bins
        prediction_bin_dir = os.path.join(dirs["prediction"], bin_name)
        os.makedirs(prediction_bin_dir, exist_ok=True)
        
        # Compile sequences for prediction
        prediction_file = os.path.join(prediction_bin_dir, f"{bin_name}_sequences.a3m")
        logging.info(f"Compiling sequences for prediction")
        compile_sequences(source_msa=args.source_msa,
                        bin_headers=bin_headers,
                        output_file=prediction_file,
                        default_pdb=args.default_pdb)
        
        # Create control prediction with random sequences
        logging.info(f"Creating control prediction file")
        control_prediction_bin_dir = os.path.join(dirs["control_prediction"], bin_name)
        os.makedirs(control_prediction_bin_dir, exist_ok=True)
        
        # Get query sequence from PDB
        query_seq = get_protein_sequence(args.default_pdb)
        
        # Read sequences
        sequences = read_sequences(args.source_msa)
        non_query_seqs = sequences[1:]  # Skip first sequence since we'll use PDB query
        
        # Randomly select sequences
        selected_seqs = random.sample(non_query_seqs, len(bin_headers))
        
        # Write sequences to file with query first
        control_prediction_file = os.path.join(control_prediction_bin_dir, f"{bin_name}_random_seq.a3m")
        with open(control_prediction_file, 'w') as f:
            f.write(f">query\n{query_seq}\n")
            for seq in selected_seqs:
                f.write(seq)
        
        # Create random selection groups under bin directory
        random_selection_bin_dir = os.path.join(dirs["random_selection"], bin_name)
        os.makedirs(random_selection_bin_dir, exist_ok=True)
        
        logging.info(f"Creating {args.num_selections} random selections")
        create_random_selected_groups(input_msa=prediction_file,
                                   output_dir=random_selection_bin_dir,
                                   num_selections=args.num_selections,
                                   seq_num_per_selection=args.seq_num_per_selection,
                                   default_pdb=args.default_pdb)
        
        # Create control random selections from source MSA
        control_random_selection_bin_dir = os.path.join(dirs["control_random_selection"], bin_name)
        os.makedirs(control_random_selection_bin_dir, exist_ok=True)
        
        logging.info(f"Creating {args.num_selections} control random selections")
        create_random_selected_groups(input_msa=args.source_msa,
                                   output_dir=control_random_selection_bin_dir,
                                   num_selections=args.num_selections,
                                   seq_num_per_selection=args.seq_num_per_selection,
                                   default_pdb=args.default_pdb)
        
        logging.info(f"Completed processing combined bins")
        
    else:
        # Original behavior - process each bin separately
        for bin_number in args.bin_numbers:
            logging.info(f"Processing bin {bin_number}")
            
            # Get headers for current bin
            logging.info(f"Extracting headers for bin {bin_number}")
            bin_headers = get_bin_headers(voting_results=args.voting_results_path, 
                                        bin_numbers=[bin_number])
            
            # Create vote distribution plot
            logging.info(f"Creating vote distribution plot for bin {bin_number}")
            try:
                plot_sequence_vote_ratios(data=raw_votes_data,
                                    bin_headers=bin_headers, 
                                    target_bin=bin_number,
                                    output_dir=args.output_dir,
                                    initial_color=args.initial_color,
                                        num_bins=args.num_total_bins,
                                        ratio_min_max=args.ratio_colorbar_min_max)
            except Exception as e:
                logging.error(f"Error creating vote distribution plot for bin {bin_number}: {e}")
            
            # Create prediction directory for this bin
            prediction_bin_dir = os.path.join(dirs["prediction"], f"bin_{bin_number}")
            os.makedirs(prediction_bin_dir, exist_ok=True)
            
            # Compile sequences for prediction
            prediction_file = os.path.join(prediction_bin_dir, f"bin{bin_number}_sequences.a3m")
            logging.info(f"Compiling sequences for prediction")
            compile_sequences(source_msa=args.source_msa,
                            bin_headers=bin_headers,
                            output_file=prediction_file,
                            default_pdb=args.default_pdb)
            
            # Create control prediction with random sequences
            logging.info(f"Creating control prediction file")
            control_prediction_bin_dir = os.path.join(dirs["control_prediction"], f"bin_{bin_number}")
            os.makedirs(control_prediction_bin_dir, exist_ok=True)
            
            # Get query sequence from PDB
            query_seq = get_protein_sequence(args.default_pdb)
            
            # Read sequences
            sequences = read_sequences(args.source_msa)
            non_query_seqs = sequences[1:]  # Skip first sequence since we'll use PDB query
            
            # Randomly select sequences
            selected_seqs = random.sample(non_query_seqs, len(bin_headers))
            
            # Write sequences to file with query first
            control_prediction_file = os.path.join(control_prediction_bin_dir, f"bin{bin_number}_random_seq.a3m")
            with open(control_prediction_file, 'w') as f:
                f.write(f">query\n{query_seq}\n")
                for seq in selected_seqs:
                    f.write(seq)
            
            # Create random selection groups under bin directory
            random_selection_bin_dir = os.path.join(dirs["random_selection"], f"bin_{bin_number}")
            os.makedirs(random_selection_bin_dir, exist_ok=True)
            
            logging.info(f"Creating {args.num_selections} random selections")
            create_random_selected_groups(input_msa=prediction_file,
                                       output_dir=random_selection_bin_dir,
                                       num_selections=args.num_selections,
                                       seq_num_per_selection=args.seq_num_per_selection,
                                       default_pdb=args.default_pdb)
            
            # Create control random selections from source MSA
            control_random_selection_bin_dir = os.path.join(dirs["control_random_selection"], f"bin_{bin_number}")
            os.makedirs(control_random_selection_bin_dir, exist_ok=True)
            
            logging.info(f"Creating {args.num_selections} control random selections")
            create_random_selected_groups(input_msa=args.source_msa,
                                       output_dir=control_random_selection_bin_dir,
                                       num_selections=args.num_selections,
                                       seq_num_per_selection=args.seq_num_per_selection,
                                       default_pdb=args.default_pdb)
            
            logging.info(f"Completed processing bin {bin_number}")
    
    logging.info("Completed recompilation and random selections for all bins")

if __name__ == "__main__":
    main()