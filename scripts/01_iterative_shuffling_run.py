import argparse
import logging
import os
import random
from af_vote.m_fold_sampling import process_iterations
from af_vote.sequence_processing import (
    read_a3m_to_dict,
    filter_a3m_by_coverage, 
    write_a3m,
    process_all_sequences,
    read_sequences,
    process_sequences
)

def parse_args():
    parser = argparse.ArgumentParser(description='Perform iterative sequence shuffling')
    parser.add_argument('--input_a3m', required=True,
                      help='Path to input A3M file as the source for shuffling')
    parser.add_argument('--default_pdb', required=True,
                      help='Path to reference PDB file')
    parser.add_argument('--base_dir', required=True,
                      help='Base directory for output')
    parser.add_argument('--group_size', type=int, default=10,
                      help='Size of each sequence group during shuffling')
    parser.add_argument('--num_shuffles', type=int, default=10,
                      help='Number of shuffles')
    parser.add_argument('--coverage_threshold', type=float, default=0.8,
                      help='Minimum required sequence coverage (default: 0.8)')
    
    parser.add_argument('--conda_env_path', 
                      default="/fs/ess/PAA0203/xing244/.conda/envs/colabfold",
                      help='Path to conda environment')
    parser.add_argument('--slurm_account', default="PAA0203",
                      help='SLURM account name')
    parser.add_argument('--slurm_output', default="/dev/null",
                      help='SLURM output file path')
    parser.add_argument('--slurm_error', default="/dev/null",
                      help='SLURM error file path')
    parser.add_argument('--slurm_nodes', type=int, default=1,
                      help='Number of nodes per SLURM job')
    parser.add_argument('--slurm_gpus_per_task', type=int, default=1,
                      help='Number of GPUs per task')
    parser.add_argument('--slurm_tasks', type=int, default=1,
                      help='Number of tasks per SLURM job')
    parser.add_argument('--slurm_cpus_per_task', type=int, default=4,
                      help='Number of CPUs per task')
    parser.add_argument('--slurm_time', default='04:00:00',
                      help='Wall time limit for SLURM jobs')
    parser.add_argument('--random_seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--num_models', type=int, default=1,
                      help='Number of models to generate')
    parser.add_argument('--check_interval', type=int, default=60,
                      help='Interval to check job status in seconds')
    parser.add_argument('--max_workers', type=int, default=64,
                      help='Maximum number of concurrent workers')
    return parser.parse_args()

def setup_logging(log_file: str) -> None:
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def log_arguments(args) -> None:
    """Log all input arguments"""
    logging.info("=== Input Arguments ===")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")
    logging.info("====================")

def main():
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.base_dir, exist_ok=True)
    iter_shuffle_base = os.path.join(args.base_dir, "01_iterative_shuffling")
    os.makedirs(iter_shuffle_base, exist_ok=True)

    # Setup logging
    log_file = os.path.join(iter_shuffle_base, "01_iterative_shuffling.log")
    setup_logging(log_file)
    log_arguments(args)
    
    try:
        # Read and filter sequences
        sequences = read_a3m_to_dict(args.input_a3m)
        filtered_sequences = filter_a3m_by_coverage(sequences, args.coverage_threshold)
        logging.info(f"Filtered sequences from {len(sequences)} to {len(filtered_sequences)} "
                    f"based on coverage threshold {args.coverage_threshold}")
        
        if not filtered_sequences:
            raise ValueError("No sequences remained after filtering")
            
        # Write filtered sequences
        filtered_a3m_path = os.path.join(iter_shuffle_base, "filtered_sequences.a3m")
        write_a3m(filtered_sequences, filtered_a3m_path, reference_pdb=args.default_pdb)

        # Process all sequences with shuffling
        process_all_sequences(
            dir_path=iter_shuffle_base,
            file_path=filtered_a3m_path,
            num_shuffles=args.num_shuffles,
            seq_num_per_shuffle=args.group_size,
            reference_pdb=args.default_pdb
        )

        # Submit SLURM jobs
        process_iterations(
            base_dir=iter_shuffle_base,
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
            check_interval=args.check_interval,
            max_workers=args.max_workers
        )
        logging.info("Submitted all SLURM jobs")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()