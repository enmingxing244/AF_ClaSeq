import os
import argparse
import logging
import subprocess
import time
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

class SlurmSubmitter:
    def __init__(self,
                 conda_env_path: str,
                 slurm_account: str,
                 slurm_output: str,
                 slurm_error: str,
                 slurm_nodes: int,
                 slurm_gpus_per_task: int,
                 slurm_tasks: int,
                 slurm_cpus_per_task: int,
                 slurm_time: str,
                 prediction_num_model: int,
                 prediction_num_seed: int,
                 shuffling_num_model: int,
                 shuffling_num_seed: int,
                 check_interval: int,
                 job_name_prefix: str):
        """
        Initialize SlurmSubmitter
        """
        self.conda_env_path = conda_env_path
        self.slurm_account = slurm_account
        self.slurm_output = slurm_output
        self.slurm_error = slurm_error
        self.slurm_nodes = slurm_nodes
        self.slurm_gpus_per_task = slurm_gpus_per_task
        self.slurm_tasks = slurm_tasks
        self.slurm_cpus_per_task = slurm_cpus_per_task
        self.slurm_time = slurm_time
        self.prediction_num_model = prediction_num_model
        self.prediction_num_seed = prediction_num_seed
        self.shuffling_num_model = shuffling_num_model
        self.shuffling_num_seed = shuffling_num_seed
        self.check_interval = check_interval
        self.job_name_prefix = job_name_prefix

    def submit_job(self, 
                   input_path: str, 
                   job_id: str, 
                   is_prediction: bool = True) -> str:
        """Submit a single job to slurm"""
        job_name = f"{self.job_name_prefix}_{job_id}"

        # Use different model/seed parameters based on job type
        num_model = self.prediction_num_model if is_prediction else self.shuffling_num_model
        num_seed = self.prediction_num_seed if is_prediction else self.shuffling_num_seed

        # Build colabfold command with num_models and random_seed
        colabfold_cmd = (
            f"colabfold_batch "
            f"--num-recycle 3 "
            f"--num-models {num_model} "
            f"--num-seeds {num_seed} "
            f"{input_path} {input_path}"
        )

        # Build sbatch command
        sbatch_cmd = [
            "sbatch",
            f"--account={self.slurm_account}",
            f"--job-name={job_name}",
            f"--output={self.slurm_output}",
            f"--error={self.slurm_error}",
            f"--nodes={self.slurm_nodes}",
            f"--gpus-per-task={self.slurm_gpus_per_task}",
            f"--ntasks={self.slurm_tasks}",
            f"--cpus-per-task={self.slurm_cpus_per_task}",
            f"--time={self.slurm_time}",
            "--wrap", f"module reset && module load openmpi cuda miniconda3 && conda init && conda activate {self.conda_env_path} && {colabfold_cmd}"
        ]

        try:
            result = subprocess.run(sbatch_cmd, capture_output=True, text=True, check=True)
            job_id = result.stdout.strip().split()[-1]
            logging.info(f"Submitted job {job_name} with ID {job_id}")
            return job_id
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to submit job {job_name}: {e}")
            return None

    def check_job_status(self, job_id: str) -> bool:
        """Check if a job is still running"""
        try:
            result = subprocess.run(["squeue", "-j", job_id], capture_output=True, text=True)
            return job_id in result.stdout
        except subprocess.CalledProcessError:
            return False

    def process_folder(self, 
                       colabfold_input_path: str, 
                       job_id: str, 
                       is_prediction: bool = True) -> None:
        """Process a single folder"""
        slurm_job_id = self.submit_job(colabfold_input_path, job_id, is_prediction)
        if slurm_job_id:
            while self.check_job_status(slurm_job_id):
                time.sleep(self.check_interval)
            logging.info(f"Job {job_id} completed")
        else:
            logging.error(f"Failed to submit job for {colabfold_input_path}")

    def process_folders_concurrently(self, 
                                     colabfold_input_paths: List[str], 
                                     job_ids: List[str], 
                                     is_predictions: List[bool], 
                                     max_workers: int) -> None:
        """Process multiple folders concurrently"""
        if not colabfold_input_paths or not job_ids or not is_predictions:
            logging.error("Empty input lists provided to process_folders_concurrently")
            return
            
        if not (len(colabfold_input_paths) == len(job_ids) == len(is_predictions)):
            logging.error("Input lists have different lengths")
            return

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for colabfold_input_path, job_id, is_prediction in zip(colabfold_input_paths, job_ids, is_predictions):
                if not os.path.exists(colabfold_input_path):
                    logging.error(f"Input path does not exist: {colabfold_input_path}")
                    continue
                    
                future = executor.submit(self.process_folder, colabfold_input_path, job_id, is_prediction)
                futures.append(future)

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Error processing folder: {str(e)}")

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
    parser.add_argument('--prediction_num_model', type=int, default=1,
                      help='Number of models to predict for prediction jobs')
    parser.add_argument('--prediction_num_seed', type=int, default=1,
                      help='Random seed for prediction jobs')
    parser.add_argument('--shuffling_num_model', type=int, default=1,
                      help='Number of models to predict for shuffling jobs')
    parser.add_argument('--shuffling_num_seed', type=int, default=1,
                      help='Random seed for shuffling jobs')
    parser.add_argument('--check_interval', type=int, default=60,
                      help='Seconds between job status checks')
    parser.add_argument('--max_workers', type=int, default=64,
                      help='Maximum number of concurrent slurm jobs')
    return parser.parse_args()

def submit_prediction_jobs(base_dir: str,
                         bin_numbers: List[int],
                         combine_bins: bool,
                         submitter: SlurmSubmitter,
                         max_workers: int) -> None:
    """Submit prediction jobs for sequences in prediction and random selection directories"""
    
    if not os.path.exists(base_dir):
        logging.error(f"Base directory not found: {base_dir}")
        return
        
    if not bin_numbers:
        logging.error("No bin numbers provided")
        return
    
    all_jobs = []
    all_job_ids = []
    is_predictions = []
    
    # Define all directory types to process
    dir_types = {
        "prediction": ("pred", True),
        "random_selection": ("sel", False),
        "control_prediction": ("ctrl_pred", True),
        "control_random_selection": ("ctrl_sel", False)
    }
    
    # Process each directory type
    for dir_name, (job_prefix, is_prediction) in dir_types.items():
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
                is_predictions.append(is_prediction)
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
                    is_predictions.append(is_prediction)
                    logging.info(f"Added {dir_name} job for bin {bin_num}: {bin_dir}")
                else:
                    logging.warning(f"Bin directory not found: {bin_dir}")

    # Process all jobs concurrently
    if all_jobs:
        logging.info(f"Processing {len(all_jobs)} jobs concurrently...")
        submitter.process_folders_concurrently(
            colabfold_input_paths=all_jobs,
            job_ids=all_job_ids,
            is_predictions=is_predictions,
            max_workers=max_workers
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

    # Initialize slurm submitter
    submitter = SlurmSubmitter(
        conda_env_path=args.conda_env_path,
        slurm_account=args.slurm_account,
        slurm_output=args.slurm_output,
        slurm_error=args.slurm_error,
        slurm_nodes=args.slurm_nodes,
        slurm_gpus_per_task=args.slurm_gpus_per_task,
        slurm_tasks=args.slurm_tasks,
        slurm_cpus_per_task=args.slurm_cpus_per_task,
        slurm_time=args.slurm_time,
        prediction_num_model=args.prediction_num_model,
        prediction_num_seed=args.prediction_num_seed,
        shuffling_num_model=args.shuffling_num_model,
        shuffling_num_seed=args.shuffling_num_seed,
        check_interval=args.check_interval,
        job_name_prefix=job_name_prefix
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
