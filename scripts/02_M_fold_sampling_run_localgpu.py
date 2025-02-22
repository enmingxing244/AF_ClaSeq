import argparse
import logging
import os
from af_claseq.m_fold_sampling import (
    initial_random_split,
    create_sampling_splits
)
from af_claseq.sequence_processing import read_a3m_to_dict, write_a3m, filter_a3m_by_coverage
from af_claseq.local_gpu_utils import LocalGPUSubmitter

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
    parser.add_argument('--coverage_threshold', type=float, default=0.0,
                      help='Minimum required sequence coverage (default: 0.0, no filtering)')
    parser.add_argument('--random_select_num_seqs', type=int, default=None,
                      help='Number of sequences to randomly select after coverage filtering')
    
    # Local GPU configuration arguments
    parser.add_argument('--conda_env_path', default="/data/xing244/.conda/envs/colabfold",
                      help='Path to conda environment')
    parser.add_argument('--num_gpus', type=int, default=1,
                      help='Number of available GPUs')
    parser.add_argument('--processes_per_gpu', type=int, default=1,
                      help='Number of processes to run per GPU')
    parser.add_argument('--check_interval', type=int, default=60,
                      help='Interval to check job status in seconds')
    parser.add_argument('--random_seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--num_models', type=int, default=1,
                      help='Number of models to generate')
    parser.add_argument('--num_seeds', type=int, default=1,
                      help='Number of seeds to use')
    return parser.parse_args()

def setup_logging(base_dir: str):
    """Set up logging configuration"""
    log_file = os.path.join(base_dir, "bayesian_sampling_run.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def create_directory_structure(base_dir: str) -> tuple:
    """Create and return required directory paths"""
    init_dir = os.path.join(base_dir, "02_init_random_split")
    sampling_base_dir = os.path.join(base_dir, "02_sampling")
    os.makedirs(init_dir, exist_ok=True)
    os.makedirs(sampling_base_dir, exist_ok=True)
    return init_dir, sampling_base_dir

def main():
    args = parse_args()
    os.makedirs(args.base_dir, exist_ok=True)
    
    # Setup logging
    setup_logging(args.base_dir)
    logging.info("=== Input Arguments ===")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")
    logging.info("====================")
    
    # Create directories
    init_dir, sampling_base_dir = create_directory_structure(args.base_dir)

    # Initial sequence filter by coverage if threshold > 0
    input_a3m = args.input_a3m
    if args.coverage_threshold == 0:
        logging.info("Coverage threshold is not set (0.0). Will use sequences as they are without filtering.")
    else:
        sequences = read_a3m_to_dict(args.input_a3m)
        filtered_sequences = filter_a3m_by_coverage(sequences, args.coverage_threshold)
        logging.info(f"Filtered sequences from {len(sequences)} to {len(filtered_sequences)} "
                    f"based on coverage threshold {args.coverage_threshold}")
        
        if not filtered_sequences:
            raise ValueError("No sequences remained after filtering")

        filtered_a3m_path = os.path.join(args.base_dir, "filtered_sequences.a3m")
        write_a3m(filtered_sequences, filtered_a3m_path, reference_pdb=args.default_pdb)
        input_a3m = filtered_a3m_path
    
    # Initial random split
    num_groups = initial_random_split(
        input_a3m, 
        init_dir, 
        args.default_pdb, 
        args.group_size,
        args.random_select_num_seqs
    )
    logging.info(f"Created {num_groups} initial groups")

    # Create sampling splits
    create_sampling_splits(
        init_dir, 
        sampling_base_dir, 
        args.default_pdb, 
        args.group_size
    )
    logging.info(f"Created {num_groups-1} sampling splits")

    # Get job name prefix from base directory path
    base_path_parts = args.base_dir.split(os.sep)
    try:
        results_idx = base_path_parts.index('results')
        job_name_prefix = base_path_parts[results_idx + 1] if results_idx + 1 < len(base_path_parts) else "fold"
    except ValueError:
        job_name_prefix = "fold"

    # Initialize Local GPU submitter
    gpu_submitter = LocalGPUSubmitter(
        conda_env_path=args.conda_env_path,
        num_gpus=args.num_gpus,
        processes_per_gpu=args.processes_per_gpu,
        check_interval=args.check_interval,
        job_name_prefix=job_name_prefix,
        num_models=args.num_models,
        num_seeds=args.num_seeds,
        random_seed=args.random_seed
    )

    # Get sampling directories and process with local GPUs
    sampling_dirs = [d for d in os.listdir(sampling_base_dir) if d.startswith('sampling_')]
    sampling_paths = [os.path.join(sampling_base_dir, d) for d in sampling_dirs]
    job_ids = [f"sample_{i+1}" for i in range(len(sampling_dirs))]
    
    gpu_submitter.process_folders_concurrently(
        folders=sampling_paths,
        job_ids=job_ids
    )
    logging.info("Submitted all local GPU jobs")

if __name__ == "__main__":
    main()