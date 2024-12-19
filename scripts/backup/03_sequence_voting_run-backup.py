import os
import logging
import json
import argparse
from typing import Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from af_vote.voting import VotingAnalyzer

def plot_bin_distribution(results_df, output_dir):
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['Bin_Assignment'], bins=20, edgecolor='black')
    plt.xlabel('Bin Assignment')
    plt.ylabel('Count')
    plt.title('Distribution of Bin Assignments')
    plt.savefig(os.path.join(output_dir, '03_voting_distribution.png'))
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description='Run voting analysis')
    parser.add_argument('--base_dir', required=True,
                      help='Base directory for output')
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
    parser.add_argument('--output_dir', required=True,
                      help='Output directory where the voting results will be saved')
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

    # Check if results file already exists
    if os.path.exists(results_file):
        logging.info("Found existing results file, loading and plotting distribution...")
        results_df = pd.read_csv(results_file)
        plot_bin_distribution(results_df, voting_dir)
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
    
    # Load configuration
    try:
        with open(args.config_path) as f:
            config = json.load(f)
    except Exception as e:
        logging.error(f"Error reading config file: {str(e)}")
        return

    # Initialize VotingAnalyzer
    analyzer = VotingAnalyzer(max_workers=args.max_workers)

    # Process sampling directories and get metrics
    logging.info("Processing sampling directories...")
    results = analyzer.process_sampling_dirs(args.sampling_dir, config['filter_criteria'], config['basics'])

    # Create bins for the specified metric
    logging.info("Creating metric bins...")
    bins, pdb_bins = analyzer.create_metric_bins(results, config['filter_criteria'][0]['name'], num_bins=args.num_bins)

    # Get sequence votes
    logging.info("Processing sequence votes...")
    sequence_votes, all_votes = analyzer.get_sequence_votes(args.source_msa, args.sampling_dir, pdb_bins)
    
    # Save raw sequence votes for later analysis
    logging.info("Saving raw sequence votes...")
    raw_votes_file = os.path.join(voting_dir, "raw_sequence_votes.json")
    
    # Convert votes to serializable format
    serializable_votes = {}
    for header, votes in all_votes.items():
        serializable_votes[header] = [int(v) for v in votes]  # Convert numpy ints to regular ints
        
    with open(raw_votes_file, 'w') as f:
        json.dump(serializable_votes, f)
    
    logging.info(f"Raw sequence votes saved to {raw_votes_file}")

    # Save voting results
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
    plot_bin_distribution(results_df, voting_dir)

    logging.info(f"Processed {len(sequence_votes)} sequences")
    logging.info(f"Metric bins: {bins}")
    logging.info("Completed voting analysis and created distribution plot")

if __name__ == "__main__":
    main()