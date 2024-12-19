import argparse
import logging
import os
from af_vote.slurm_utils import SlurmJobSubmitter
from af_vote.sequence_processing import (
    read_a3m_to_dict,
    filter_a3m_by_coverage,
    write_a3m,
    process_all_sequences,
    collect_a3m_files,
    concatenate_a3m_content
)
from af_vote import structure_analysis
from typing import Dict, Any, List, Optional

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
    parser.add_argument('--num_iterations', type=int, default=3,
                      help='Number of iterations to run')
    parser.add_argument('--quantile', type=float, default=0.6,
                      help='Quantile threshold for filtering')
    parser.add_argument('--config_file', required=True,
                      help='Path to config file with filter criteria and indices')
    parser.add_argument('--resume_from_iter', type=int,
                      help='Resume from a specific iteration number (e.g., 2 to resume from iteration 2 using iteration 1 results)')
    parser.add_argument('--plddt_threshold', type=int, default=75,
                      help='pLDDT threshold for filtering (default: 75)')
    
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

def process_iteration(args,
                      iteration: int, 
                      input_a3m: str, 
                      iter_base_dir: str, 
                      config: Dict[str, Any]) -> Optional[str]:
    """Process a single iteration"""
    iter_dir = os.path.join(iter_base_dir, f'Iteration_{iteration}')
    os.makedirs(iter_dir, exist_ok=True)
    logging.info(f'Processing iteration {iteration}...')

    # Process sequences with shuffling
    process_all_sequences(
        dir_path=iter_dir,
        file_path=input_a3m,
        num_shuffles=args.num_shuffles,
        seq_num_per_shuffle=args.group_size,
        reference_pdb=args.default_pdb
    )

    # Get job name prefix from base directory path
    base_path_parts = args.base_dir.split(os.sep)
    try:
        results_idx = base_path_parts.index('results')
        job_name_prefix = base_path_parts[results_idx + 1] if results_idx + 1 < len(base_path_parts) else "fold"
    except ValueError:
        job_name_prefix = "fold"

    # Submit SLURM jobs
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
        check_interval=args.check_interval,
        job_name_prefix=job_name_prefix
    )

    # Get shuffle directories and submit jobs
    shuffle_dirs = [d for d in os.listdir(iter_dir) if d.startswith('shuffle_')]
    job_folders = [f"shuffle_{i+1}" for i, _ in enumerate(shuffle_dirs)]
    all_paths = [os.path.join(iter_dir, d) for d in shuffle_dirs]
    slurm_submitter.process_folders_concurrently(all_paths, job_folders, max_workers=args.max_workers)
    
    # Get results and apply filters
    result_df = structure_analysis.get_result_df(
        parent_dir=iter_dir,
        filter_criteria=config['filter_criteria'],
        basics=config['basics']
    )
    
    result_df_filtered = result_df[result_df['plddt'] > args.plddt_threshold]
    filtered_df = structure_analysis.apply_filters(
        df_threshold=result_df_filtered,
        df_operate=result_df_filtered,
        filter_criteria=config['filter_criteria'],
        quantile=args.quantile
    )

    # Save filtered DataFrame to CSV
    filtered_df_path = os.path.join(iter_dir, f'filtered_results_iteration_{iteration}.csv')
    filtered_df.to_csv(filtered_df_path, index=False)
    logging.info(f'Saved filtered results to {filtered_df_path}')

    # Collect and concatenate filtered sequences
    a3m_files = collect_a3m_files([filtered_df])
    concatenated_a3m_path = os.path.join(iter_dir, f'combined_filtered_iteration_{iteration}.a3m')
    concatenate_a3m_content(a3m_files, args.default_pdb, concatenated_a3m_path)
    
    return concatenated_a3m_path

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
        # Load config file
        config = structure_analysis.load_filter_modes(args.config_file)
        
        # Initial sequence filter by coverage
        if not args.resume_from_iter:
            sequences = read_a3m_to_dict(args.input_a3m)
            filtered_sequences = filter_a3m_by_coverage(sequences, args.coverage_threshold)
            logging.info(f"Filtered sequences from {len(sequences)} to {len(filtered_sequences)} "
                    f"based on coverage threshold {args.coverage_threshold}")
        
            if not filtered_sequences:
                raise ValueError("No sequences remained after filtering")

            filtered_a3m_path = os.path.join(iter_shuffle_base, "filtered_sequences.a3m")
            write_a3m(filtered_sequences, filtered_a3m_path, reference_pdb=args.default_pdb)

            # Iterative processing
            current_input = filtered_a3m_path
            start_iter = 1

        elif args.resume_from_iter:
            start_iter = args.resume_from_iter
            prev_iter_a3m = os.path.join(iter_shuffle_base, f'Iteration_{args.resume_from_iter - 1}', 
                                        f'combined_filtered_iteration_{args.resume_from_iter - 1}.a3m')
            if os.path.exists(prev_iter_a3m):
                current_input = prev_iter_a3m
                logging.info(f"Resuming from iteration {args.resume_from_iter}")
            else:
                raise ValueError(f"Cannot resume - file not found: {prev_iter_a3m}")

        for iteration in range(start_iter, args.num_iterations + 1):
            logging.info(f"Starting iteration {iteration}")
            current_input = process_iteration(
                iteration=iteration,
                input_a3m=current_input,
                args=args,
                iter_base_dir=iter_shuffle_base,
                config=config
            )
            if current_input is None:
                break
                
        logging.info("All iterations completed successfully")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()