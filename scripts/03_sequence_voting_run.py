import os
import logging
import json
import argparse
from typing import Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from af_claseq.voting import VotingAnalyzer

def hex2color(hex_str):
    """Convert hex string to RGB tuple"""
    hex_str = hex_str.lstrip('#')
    return tuple(int(hex_str[i:i+2], 16)/255.0 for i in (0, 2, 4))

def plot_bin_distribution(results_df, 
                          output_dir, 
                          n_plot_bins, 
                          initial_color,
                          end_color=None,  # Default is None - will use initial_color if not provided
                          is_2d=False, 
                          figsize: tuple = (10, 5),
                          y_min: float = None,
                          y_max: float = None,
                          x_ticks: list = None):
    if is_2d:
        print("2D bin distribution not implemented yet")
    else:
        plt.figure(figsize=figsize, dpi=600)

        # Set font parameters
        plt.rcParams.update({
            'font.size': 24,
            'axes.labelsize': 24,
            'axes.titlesize': 24,
            'xtick.labelsize': 24,
            'ytick.labelsize': 24,
            'legend.fontsize': 24
        })

        bins = np.arange(1, n_plot_bins + 2)  # 1 to n_plot_bins+1
        
        # Create histogram data without plotting
        counts, bins, _ = plt.hist(results_df['Bin_Assignment'], bins=bins)
        plt.clf()  # Clear the figure

        if end_color is not None:  # Only create gradient if end_color is provided
            # Create color gradient
            initial_rgb = hex2color(initial_color)
            end_rgb = hex2color(end_color)
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
                   color=hex2color(initial_color), edgecolor=None)

        # Add vertical dashed lines at bin boundaries
        for bin_edge in bins:
            plt.axvline(x=bin_edge - 0.5, color='gray', linestyle='--', alpha=0.5)

        plt.xlabel('Bin Assignment')
        plt.ylabel('Count')
        # plt.title('Distribution of Bin Assignments')
        
        plt.yscale('log')
        
        # Set x-axis limits and ticks
        plt.xlim(0.5, n_plot_bins + 0.5)
        
        # Set custom x ticks if provided, otherwise use all bins
        if x_ticks is not None:
            plt.xticks(x_ticks)

            
        # Set y-axis limits if provided
        if y_min is not None and y_max is not None:
            plt.ylim(y_min, y_max)
        
        # Save plot
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, '03_voting_distribution.png')
        # plt.tight_layout()
        plt.savefig(plot_path, dpi=600, bbox_inches='tight')
        plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description='Run voting analysis')

    parser.add_argument('--sampling_dir', required=True,
                      help='Directory containing sampling results')
    parser.add_argument('--source_msa', required=True,
                      help='Path to source MSA file')
    parser.add_argument('--config_path', required=True,
                      help='Path to configuration JSON file')
    parser.add_argument('--num_bins', type=int, default=20,
                      help='Number of bins for voting')
    parser.add_argument('--max_workers', type=int, default=88,
                      help='Maximum number of concurrent workers')
    parser.add_argument('--vote_threshold', type=float, default=0.0,
                      help='Threshold for vote filtering (between 0 and 1)')
    parser.add_argument('--output_dir', required=True,
                      help='Output directory where the voting results will be saved')
    parser.add_argument('--vote_min_value', type=float, default=None,
                      help='Minimum value for 1D metric binning range')
    parser.add_argument('--vote_min_value', type=float, default=None,
                      help='Maximum value for 1D metric binning range')
    parser.add_argument('--initial_color', type=str, default='#d3b0b0',
                      help='Initial color for bin distribution plot')
    parser.add_argument('--end_color', type=str, default=None,
                      help='End color for bin distribution plot')
    parser.add_argument('--use_focused_bins', action='store_true',
                      help='Use focused 1D binning with outlier bins')
    parser.add_argument('--precomputed_metrics', type=str, default=None,
                      help='Path to precomputed metrics CSV file')
    parser.add_argument('--plddt_threshold', type=float, default=0,
                      help='pLDDT threshold for filtering structures (default: 0, no filtering)')
    parser.add_argument('--figsize', type=float, nargs=2, default=(10, 5),
                      help='Figure size in inches (width, height) (default: 10 5)')
    parser.add_argument('--y_min', type=float, default=None,
                      help='Minimum y-axis value for distribution plot')
    parser.add_argument('--y_max', type=float, default=None,
                      help='Maximum y-axis value for distribution plot')
    parser.add_argument('--x_ticks', type=int, nargs='+', default=None,
                      help='Custom x-axis tick positions')
    parser.add_argument('--hierarchical_sampling', action='store_true',
                      help='Use hierarchical sampling directories structure')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    voting_dir = os.path.join(args.output_dir)
    results_file = os.path.join(voting_dir, "03_voting_results.csv")

    # Setup logging
    os.makedirs(voting_dir, exist_ok=True)
    log_file = os.path.join(voting_dir, "03_voting_run.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    # Load configuration
    try:
        with open(args.config_path) as f:
            config = json.load(f)
    except Exception as e:
        logging.error(f"Error reading config file: {str(e)}")
        return

    # Determine if we need 2D binning based on config
    is_2d = len(config['filter_criteria']) >= 2
    is_3d = len(config['filter_criteria']) >= 3

    # Check if results file already exists
    if os.path.exists(results_file):
        logging.info("Found existing results file, loading and plotting distribution...")
        results_df = pd.read_csv(results_file)
        plot_bin_distribution(results_df, 
                              voting_dir, 
                              args.num_bins, 
                              args.initial_color, 
                              args.end_color, 
                              is_2d, 
                              args.figsize,
                              args.y_min,
                              args.y_max,
                              args.x_ticks)
        logging.info("Distribution plot created")
        return
    
    # Validate paths
    for path, name in [
        (args.sampling_dir, "Sampling directory"),
        (args.source_msa, "Source MSA file"),
        (args.config_path, "Config file")
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found: {path}")

    # Initialize VotingAnalyzer
    analyzer = VotingAnalyzer(max_workers=args.max_workers)

    # Process sampling directories and get metrics
    logging.info("Processing sampling directories...")
    results = analyzer.process_sampling_dirs(args.sampling_dir, 
                                          config['filter_criteria'], 
                                          config['basics'],
                                          precomputed_metrics_file=args.precomputed_metrics,
                                          plddt_threshold=args.plddt_threshold,
                                          hierarchical=args.hierarchical_sampling)

    # Create bins based on dimensionality
    logging.info("Creating metric bins...")
    if is_3d:
        metric_names = [criterion['name'] for criterion in config['filter_criteria'][:3]]
        bins, pdb_bins = analyzer.create_3d_metric_bins(results,
                                                      metric_names, 
                                                      num_bins=args.num_bins)
    elif is_2d:
        metric_names = [criterion['name'] for criterion in config['filter_criteria'][:2]]
        bins, pdb_bins = analyzer.create_2d_metric_bins(results,
                                                      metric_names, 
                                                      num_bins=args.num_bins)
    else:
        if args.use_focused_bins:
            bins, pdb_bins = analyzer.create_focused_1d_bins(results,
                                                           config['filter_criteria'][0]['name'],
                                                           args.num_bins,
                                                           args.min_value,
                                                           args.max_value)
        else:
            bins, pdb_bins = analyzer.create_1d_metric_bins(results, 
                                                          config['filter_criteria'][0]['name'], 
                                                          num_bins=args.num_bins,
                                                          min_value=args.min_value,
                                                          max_value=args.max_value)

    # Get sequence votes
    logging.info("Processing sequence votes...")
    sequence_votes, all_votes = analyzer.get_sequence_votes(args.source_msa, 
                                                          args.sampling_dir, 
                                                          pdb_bins, 
                                                          is_2d=is_2d,
                                                          is_3d=is_3d,
                                                          vote_threshold=args.vote_threshold,
                                                          hierarchical=args.hierarchical_sampling)
    
    # Save raw sequence votes for later analysis
    logging.info("Saving raw sequence votes...")
    raw_votes_file = os.path.join(voting_dir, "raw_sequence_votes.json")
    
    # Convert votes to serializable format
    serializable_votes = {}
    for header, votes in all_votes.items():
        if is_3d:
            serializable_votes[header] = [[int(v[0]), int(v[1]), int(v[2])] for v in votes]
        elif is_2d:
            serializable_votes[header] = [[int(v[0]), int(v[1])] for v in votes]
        else:
            serializable_votes[header] = [int(v) for v in votes]
        
    with open(raw_votes_file, 'w') as f:
        json.dump(serializable_votes, f)
    
    logging.info(f"Raw sequence votes saved to {raw_votes_file}")

    # Save voting results
    if is_3d:
        results_df = pd.DataFrame([
            {
                "Sequence_Header": header,
                "Bin_Assignment_1": bin_nums if isinstance(bin_nums, int) else bin_nums[0],
                "Bin_Assignment_2": bin_nums[1] if not isinstance(bin_nums, int) else bin_nums,
                "Bin_Assignment_3": bin_nums[2] if not isinstance(bin_nums, int) else bin_nums,
                "Vote_Count": vote_count,
                "Total_Votes": total_votes
            }
            for header, (bin_nums, vote_count, total_votes) in sequence_votes.items()
        ])
    elif is_2d:
        results_df = pd.DataFrame([
            {
                "Sequence_Header": header,
                "Bin_Assignment_1": bin_nums if isinstance(bin_nums, int) else bin_nums[0],
                "Bin_Assignment_2": bin_nums[1] if not isinstance(bin_nums, int) else bin_nums,
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
    results_df.to_csv(results_file, index=False)

    # Create distribution plot
    plot_bin_distribution(results_df, 
                          voting_dir, 
                          args.num_bins, 
                          args.initial_color, 
                          args.end_color, 
                          is_2d, 
                          args.figsize,
                          args.y_min,
                          args.y_max,
                          args.x_ticks)

    logging.info(f"Processed {len(sequence_votes)} sequences")
    logging.info(f"Metric bins: {bins}")
    logging.info("Completed voting analysis and created distribution plot")

if __name__ == "__main__":
    main()