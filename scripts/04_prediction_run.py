import os
import argparse
import logging
import time
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from af_claseq.slurm_utils import SlurmJobSubmitter

def parse_args():
    parser = argparse.ArgumentParser(description='Submit prediction jobs for sequences')
    parser.add_argument('--base_dir', required=True,
                      help='Base directory, e.g. /fs/ess/PAA0203/xing244/AF_Vote/results/KAD_ECOLI/run2/04_recompile/rmsd_1ake')
    parser.add_argument('--bin_numbers', type=int, nargs='+', required=True,
                      help='List of bin numbers to submit predictions for (e.g. 4 5 6)')
    parser.add_argument('--combine_bins', action='store_true',
                      help='Whether to combine sequences from all input bins into one directory')
    
    parser.add_argument('--conda_env_path', 
                      default="/fs/ess/PAA0203/xing244/.conda/envs/colabfold",
                      help='Path to conda environment')
    parser.add_argument('--slurm_account', default="PAA0203",
                      help='Slurm account name')
    parser.add_argument('--slurm_output', default="/dev/null",
                      help='Path for slurm output files')
    parser.add_argument('--slurm_error', default="/dev/null",
                      help='Path for slurm error files')
    parser.add_argument('--slurm_nodes', type=int, default=1,
                      help='Number of nodes per job')
    parser.add_argument('--slurm_gpus_per_task', type=int, default=1,
                      help='GPUs per task')
    parser.add_argument('--slurm_tasks', type=int, default=1,
                      help='Number of tasks per job')
    parser.add_argument('--slurm_cpus_per_task', type=int, default=4,
                      help='CPUs per task')
    parser.add_argument('--slurm_time', default='04:00:00',
                      help='Time limit for slurm jobs')
    parser.add_argument('--prediction_num_model', type=int, default=5,
                      help='Number of models to predict for prediction jobs')
    parser.add_argument('--prediction_num_seed', type=int, default=8,
                      help='Random seed for prediction jobs')
    parser.add_argument('--check_interval', type=int, default=60,
                      help='Seconds between job status checks')
    parser.add_argument('--max_workers', type=int, default=64,
                      help='Maximum number of concurrent slurm jobs')
    return parser.parse_args()

def submit_prediction_jobs(base_dir: str,
                         bin_numbers: List[int],
                         combine_bins: bool,
                         submitter: SlurmJobSubmitter,
                         max_workers: int) -> None:
    """Submit prediction jobs for sequences in prediction and control_prediction directories"""
    
    if not os.path.exists(base_dir):
        logging.error(f"Base directory not found: {base_dir}")
        return
        
    if not bin_numbers:
        logging.error("No bin numbers provided")
        return
    
    all_jobs = []
    all_job_ids = []
    job_types = []
    
    # Define directory types to process (only prediction and control_prediction)
    dir_types = {
        "prediction": ("pred", "prediction"),
        "control_prediction": ("ctrl_pred", "prediction")
    }
    
    # Process each directory type
    for dir_name, (job_prefix, job_type) in dir_types.items():
        dir_path = os.path.join(base_dir, dir_name)
        if not os.path.exists(dir_path):
            logging.error(f"{dir_name} directory not found: {dir_path}")
            continue
            
        logging.info(f"Collecting jobs from {dir_name} directory: {dir_path}")
        
        if combine_bins:
            # Create combined bin name (e.g. "bin_5_6_7")
            bin_name = f"bin_{'_'.join(str(b) for b in sorted(bin_numbers))}"
            bin_dir = os.path.join(dir_path, bin_name)
            
            if os.path.exists(bin_dir):
                job_id = f"{job_prefix}_{bin_name}"
                all_jobs.append(bin_dir)
                all_job_ids.append(job_id)
                job_types.append(job_type)
                logging.info(f"Added {dir_name} job for combined bins: {bin_dir}")
            else:
                logging.warning(f"Combined bin directory not found: {bin_dir}")
        else:
            # Process each bin directory separately
            for bin_num in bin_numbers:
                bin_dir = os.path.join(dir_path, f"bin_{bin_num}")
                if os.path.exists(bin_dir):
                    job_id = f"{job_prefix}_bin{bin_num}"
                    all_jobs.append(bin_dir)
                    all_job_ids.append(job_id)
                    job_types.append(job_type)
                    logging.info(f"Added {dir_name} job for bin {bin_num}: {bin_dir}")
                else:
                    logging.warning(f"Bin directory not found: {bin_dir}")

    # Process all jobs concurrently
    if all_jobs:
        logging.info(f"Processing {len(all_jobs)} jobs concurrently...")
        submitter.process_folders_concurrently(
            folders=all_jobs,
            job_ids=all_job_ids,
            max_workers=max_workers,
            job_types=job_types
        )
    else:
        logging.warning("No jobs found to process")

def main():
    args = parse_args()
    
    logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s - %(levelname)s - %(message)s')

    # Get job name prefix from base directory path
    base_path_parts = args.base_dir.split(os.sep)
    try:
        results_idx = base_path_parts.index('results')
        job_name_prefix = base_path_parts[results_idx + 1] if results_idx + 1 < len(base_path_parts) else "fold"
    except ValueError:
        job_name_prefix = "fold"
        logging.warning("Could not extract job prefix from base_dir path, using default: 'fold'")

    # Initialize slurm submitter with typed mode
    submitter = SlurmJobSubmitter(
        conda_env_path=args.conda_env_path,
        slurm_account=args.slurm_account,
        slurm_output=args.slurm_output,
        slurm_error=args.slurm_error,
        slurm_nodes=args.slurm_nodes,
        slurm_gpus_per_task=args.slurm_gpus_per_task,
        slurm_tasks=args.slurm_tasks,
        slurm_cpus_per_task=args.slurm_cpus_per_task,
        slurm_time=args.slurm_time,
        check_interval=args.check_interval,
        job_name_prefix=job_name_prefix,
        prediction_num_model=args.prediction_num_model,
        prediction_num_seed=args.prediction_num_seed
    )

    # Submit prediction jobs
    submit_prediction_jobs(
        base_dir=args.base_dir,
        bin_numbers=args.bin_numbers,
        combine_bins=args.combine_bins,
        submitter=submitter,
        max_workers=args.max_workers
    )
    
    logging.info("Completed submitting all prediction jobs")

if __name__ == "__main__":
    main()
