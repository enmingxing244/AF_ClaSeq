import argparse
import logging
import os
from af_vote.m_fold_sampling import (
    initial_random_split,
    create_sampling_splits,
    process_iterations
)
from af_vote.sequence_processing import read_a3m_to_dict, filter_a3m_by_coverage, write_a3m

def parse_args():
    parser = argparse.ArgumentParser(description='Perform Bayesian sampling of sequences')
    parser.add_argument('--input_a3m', required=True,
                      help='Path to input A3M file as the source for M-fold sampling')
    parser.add_argument('--default_pdb', required=True,
                      help='Path to reference PDB file')
    parser.add_argument('--base_dir', required=True,
                      help='Base directory for output')
    parser.add_argument('--group_size', type=int, default=10,
                      help='Size of each sequence group')
    parser.add_argument('--coverage_threshold', type=float, default=0.8,
                      help='Minimum required sequence coverage (default: 0.8)')
    
    parser.add_argument('--conda_env_path', default="/fs/ess/PAA0203/xing244/.conda/envs/colabfold",
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

def main():

    args = parse_args()

    os.makedirs(args.base_dir, exist_ok=True)
    
    log_file = os.path.join(args.base_dir, "bayesian_sampling_run.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    # Log all input arguments
    logging.info("=== Input Arguments ===")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")
    logging.info("====================")
    
    # Create directory structure
    init_dir = os.path.join(args.base_dir, "01_init_random_split")
    os.makedirs(init_dir, exist_ok=True)
    sampling_base_dir = os.path.join(args.base_dir, "02_sampling")
    os.makedirs(sampling_base_dir, exist_ok=True)
    
    # Read and filter sequences based on coverage
    sequences = read_a3m_to_dict(args.input_a3m)
    filtered_sequences = filter_a3m_by_coverage(sequences, args.coverage_threshold)
    logging.info(f"Filtered sequences from {len(sequences)} to {len(filtered_sequences)} based on coverage threshold {args.coverage_threshold}")
    
    # Write filtered sequences to temporary a3m file
    filtered_a3m_path = os.path.join(init_dir, "filtered_sequences.a3m")
    write_a3m(filtered_sequences, filtered_a3m_path, reference_pdb=args.default_pdb)
    
    # Step 1: Initial random split with filtered sequences
    num_groups = initial_random_split(filtered_a3m_path, init_dir, args.default_pdb, args.group_size)
    logging.info(f"Created {num_groups} initial groups")

    # Step 2: Create sampling splits
    create_sampling_splits(init_dir, sampling_base_dir, args.default_pdb, args.group_size)
    logging.info(f"Created {num_groups-1} sampling splits")

    # Step 3: Submit SLURM jobs
    process_iterations(
        base_dir=sampling_base_dir,
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

if __name__ == "__main__":
    main()