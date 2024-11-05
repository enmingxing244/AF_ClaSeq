import os
import logging
import json
import argparse
from typing import Dict, Any
import csv
import pandas as pd
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from af_vote.voting import VotingAnalyzer
from af_vote.sequence_processing import read_a3m_to_dict
from af_vote.slurm_utils import SlurmJobSubmitter

def parse_args():
    parser = argparse.ArgumentParser(description='Run voting analysis and recompile sequences')
    parser.add_argument('--base-dir', required=True,
                      help='Base directory for output')
    parser.add_argument('--sampling-dir', required=True,
                      help='Directory containing sampling results')
    parser.add_argument('--source-msa', required=True,
                      help='Path to source MSA file')
    parser.add_argument('--config-path', required=True,
                      help='Path to configuration JSON file')
    parser.add_argument('--num-bins', type=int, default=20,
                      help='Number of bins for voting')
    
    # SLURM arguments
    parser.add_argument('--conda-env-path', default="/fs/ess/PAA0203/xing244/.conda/envs/colabfold",
                      help='Path to conda environment')
    parser.add_argument('--slurm-account', default="PAA0203",
                      help='SLURM account name')
    parser.add_argument('--slurm-output', default="/dev/null",
                      help='SLURM output file path')
    parser.add_argument('--slurm-error', default="/dev/null",
                      help='SLURM error file path')
    parser.add_argument('--slurm-nodes', type=int, default=1,
                      help='Number of nodes per SLURM job')
    parser.add_argument('--slurm-gpus-per-task', type=int, default=1,
                      help='Number of GPUs per task')
    parser.add_argument('--slurm-tasks', type=int, default=1,
                      help='Number of tasks per SLURM job')
    parser.add_argument('--slurm-cpus-per-task', type=int, default=4,
                      help='Number of CPUs per task')
    parser.add_argument('--slurm-time', default='04:00:00',
                      help='Wall time limit for SLURM jobs')
    parser.add_argument('--random-seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--num-models', type=int, default=1,
                      help='Number of models to generate')
    parser.add_argument('--check-interval', type=int, default=60,
                      help='Interval to check job status in seconds')
    parser.add_argument('--max-workers', type=int, default=64,
                      help='Maximum number of concurrent workers')
    return parser.parse_args()

def recompile_sequences(voting_results: Dict[str, tuple], source_msa: str, bin_num: int, output_file: str):
    """Recompile sequences for a specific bin into new A3M file"""
    # Get headers for specified bin
    bin_headers = [header for header, (bin_assignment, _, _) in voting_results.items() 
                  if bin_assignment == bin_num]
    
    # Read source MSA and extract sequences
    try:
        sequences = read_a3m_to_dict(source_msa)
    except FileNotFoundError:
        logging.error(f"Source MSA file not found: {source_msa}")
        return
    except Exception as e:
        logging.error(f"Error reading MSA file {source_msa}: {str(e)}")
        return
        
    query_seq = sequences.get('>seq')
    if not query_seq:
        logging.error("No query sequence (>seq) found in MSA file")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write new A3M file
    try:
        with open(output_file, 'w') as f:
            # Write query sequence first
            f.write('>seq\n')
            f.write(query_seq + '\n')
            
            # Write bin sequences
            for header in bin_headers:
                if header in sequences:
                    f.write(f'>{header}\n')
                    f.write(sequences[header] + '\n')
    except IOError as e:
        logging.error(f"Error writing output file {output_file}: {str(e)}")
        return

def main():
    args = parse_args()
    
    # Setup logging
    os.makedirs(args.base_dir, exist_ok=True)
    log_file = os.path.join(args.base_dir, "voting_run.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
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

    # Initialize analyzer
    analyzer = VotingAnalyzer(config, num_bins=args.num_bins)
    
    # Get PDB files
    pdb_files = [
        str(p) for p in Path(args.sampling_dir).rglob('*.pdb')
        if 'non_a3m' not in str(p)
    ]
    
    if not pdb_files:
        logging.error("No PDB files found in sampling directory")
        return

    # Setup directories
    voting_dir = os.path.join(args.base_dir, "03_voting")
    os.makedirs(voting_dir, exist_ok=True)

    # Process structures
    bins, pdb_assignments, pdb_properties = analyzer.process_structures(pdb_files)
    
    # Save structure criteria
    criteria_data = {
        'pdb_file': list(pdb_assignments.keys()),
        'bin_assignment': list(pdb_assignments.values())
    }
    criteria_file = os.path.join(voting_dir, "structure_criteria.csv")
    pd.DataFrame(criteria_data).to_csv(criteria_file, index=False)

    # Get source headers from MSA
    source_headers = set()
    with open(args.source_msa) as f:
        for line in f:
            if line.startswith('>'):
                header = line.strip()[1:]
                if header != 'seq':
                    source_headers.add(header)

    # Process A3M files in batches
    a3m_files = [(root, f) for root, _, files in os.walk(args.sampling_dir) 
                 for f in files if f.endswith('.a3m')]
    
    batch_votes = analyzer.process_a3m_batch(a3m_files, pdb_assignments, source_headers)
    
    # Process voting results
    voting_results = {}
    for header, votes in batch_votes.items():
        if votes:
            vote_counts = defaultdict(int)
            for vote in votes:
                vote_counts[vote] += 1
            most_common_bin = max(vote_counts.items(), key=lambda x: x[1])[0]
            voting_results[header] = (most_common_bin, vote_counts[most_common_bin], len(votes))

    # Save voting results
    results_df = pd.DataFrame([
        {
            "Sequence_Header": header,
            "Bin_Assignment": bin_num,
            "Vote_Count": vote_count,
            "Total_Votes": total_votes
        }
        for header, (bin_num, vote_count, total_votes) in voting_results.items()
    ])
    results_df.to_csv(os.path.join(voting_dir, "voting_results.csv"), index=False)

    # Process bins and submit predictions
    recompile_dir = os.path.join(args.base_dir, "04_recompiled")
    os.makedirs(recompile_dir, exist_ok=True)

    slurm_submitter = SlurmJobSubmitter(
        conda_env_path=args.conda_env_path,
        slurm_account=args.slurm_account,
        slurm_output=args.slurm_output,
        slurm_error=args.slurm_error,
        slurm_nodes=args.slurm_nodes,
        slurm_gpus_per_task=args.slurm_gpus_per_task,
        slurm_tasks=args.slurm_tasks,
        slurm_cpus_per_task=args.slurm_cpus_per_task,
        slurm_time=args.slurm_time,
        random_seed=args.random_seed,
        num_models=args.num_models,
        check_interval=args.check_interval
    )

    # Process bins in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        for bin_num in range(args.num_bins):
            output_file = os.path.join(recompile_dir, f"bin{bin_num}_sequences.a3m")
            recompile_sequences(voting_results, args.source_msa, bin_num, output_file)
            
            output_dir = os.path.join(args.base_dir, f"05_predictions/bin{bin_num}")
            os.makedirs(output_dir, exist_ok=True)
            
            futures.append(executor.submit(
                slurm_submitter.process_folders_concurrently,
                [output_file],
                [f"predict_bin{bin_num}"],
                output_dir=output_dir,
                max_workers=1  # Process one folder at a time within each bin
            ))
            
        # Wait for all jobs to complete
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error in bin processing: {str(e)}")
                continue

    logging.info("Completed voting analysis and submitted prediction jobs")

if __name__ == "__main__":
    main()