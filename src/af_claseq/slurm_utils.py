import os
import subprocess
import logging
import time
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict, Any, Union


class SlurmJobSubmitter:
    """A unified class to manage SLURM job submissions and monitoring."""

    def __init__(
        self,
        conda_env_path: str,
        slurm_account: str,
        slurm_output: str = "/dev/null",
        slurm_error: str = "/dev/null",
        slurm_nodes: int = 1,
        slurm_gpus_per_task: int = 1,
        slurm_tasks: int = 1,
        slurm_cpus_per_task: int = 4,
        slurm_time: str = "04:00:00",
        check_interval: int = 60,
        job_name_prefix: str = "fold",
        **kwargs
    ):
        """
        Initialize the SlurmJobSubmitter with configuration parameters.

        Args:
            conda_env_path (str): Path to the Conda environment.
            slurm_account (str): SLURM account name.
            slurm_output (str): SLURM job output path.
            slurm_error (str): SLURM job error path.
            slurm_nodes (int): Number of nodes.
            slurm_gpus_per_task (int): Number of GPUs per task.
            slurm_tasks (int): Number of tasks.
            slurm_cpus_per_task (int): Number of CPUs per task.
            slurm_time (str): SLURM job time limit.
            check_interval (int): Time in seconds between status checks.
            job_name_prefix (str): Prefix for job names.
            **kwargs: Additional arguments for different modes:
                For simple mode:
                    - num_models (int): Number of models to generate
                    - num_seeds (int): Number of seeds to use
                    - random_seed (int): Random seed for reproducibility
                For typed mode:
                    - prediction_num_model (int): Number of models for prediction
                    - prediction_num_seed (int): Number of seeds for prediction
                    - shuffling_num_model (int): Number of models for shuffling
                    - shuffling_num_seed (int): Number of seeds for shuffling
        """
        # Basic SLURM configuration
        self.conda_env_path = conda_env_path
        self.slurm_account = slurm_account
        self.slurm_output = slurm_output
        self.slurm_error = slurm_error
        self.slurm_nodes = slurm_nodes
        self.slurm_gpus_per_task = slurm_gpus_per_task
        self.slurm_tasks = slurm_tasks
        self.slurm_cpus_per_task = slurm_cpus_per_task
        self.slurm_time = slurm_time
        self.check_interval = check_interval
        self.job_name_prefix = job_name_prefix

        # Determine mode based on kwargs
        if any(k.startswith(('prediction_', 'shuffling_')) for k in kwargs):
            self.mode = 'typed'
            self.job_configs = {
                'prediction': {
                    'num_models': kwargs.get('prediction_num_model', 1),
                    'num_seeds': kwargs.get('prediction_num_seed', 1)
                },
                'shuffling': {
                    'num_models': kwargs.get('shuffling_num_model', 1),
                    'num_seeds': kwargs.get('shuffling_num_seed', 1)
                }
            }
        else:
            self.mode = 'simple'
            self.num_models = kwargs.get('num_models', 1)
            self.num_seeds = kwargs.get('num_seeds', 1)
            self.random_seed = kwargs.get('random_seed')

    def _get_job_config(self, job_type: Optional[str] = None) -> Dict[str, int]:
        """Get job configuration based on mode and job type."""
        if self.mode == 'typed':
            if not job_type or job_type not in self.job_configs:
                raise ValueError(f"Invalid job type: {job_type}")
            return self.job_configs[job_type]
        else:
            config = {'num_models': self.num_models, 'num_seeds': self.num_seeds}
            if self.random_seed is not None:
                config['random_seed'] = self.random_seed
            return config

    def submit_job(self, task_dir: str, job_id: str, job_type: Optional[str] = None) -> Optional[str]:
        """Submit a SLURM job for the given task directory."""
        if not os.path.exists(task_dir):
            logging.error(f"Task directory not found: {task_dir}")
            return None

        config = self._get_job_config(job_type)
        
        # Build colabfold command
        colabfold_cmd = [
            "colabfold_batch",
            "--num-recycle", "3",
            "--num-models", str(config['num_models']),
            "--num-seeds", str(config['num_seeds'])
        ]
        
        if 'random_seed' in config:
            colabfold_cmd.extend(["--random-seed", str(config['random_seed'])])
            
        colabfold_cmd.extend([task_dir, task_dir])
        colabfold_cmd = " ".join(colabfold_cmd)

        # Build environment setup
        env_setup = (
            "module reset && module load openmpi cuda miniconda3 && "
            f"conda init && conda activate {self.conda_env_path}"
        )

        # Build sbatch command
        job_name = f"{self.job_name_prefix}_{job_id}"
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
            "--wrap", f"{env_setup} && {colabfold_cmd}"
        ]

        job_type_str = f" ({job_type})" if job_type else ""
        logging.info(f"Submitting job for {task_dir}{job_type_str} (Job ID: {job_id})")
        
        try:
            result = subprocess.run(sbatch_cmd, capture_output=True, text=True, check=True)
            submitted_job_id = result.stdout.strip().split()[-1]
            logging.info(f"Submitted job {submitted_job_id} for task directory {task_dir}")
            return submitted_job_id
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to submit job for {task_dir}: {e}")
            return None

    def check_job_status(self, job_id: str) -> bool:
        """Check if a job is still running."""
        try:
            result = subprocess.run(["squeue", "-j", job_id], capture_output=True, text=True)
            return job_id in result.stdout
        except subprocess.CalledProcessError:
            return False

    def wait_for_completion(self, job_id: str) -> None:
        """Wait for a job to complete."""
        logging.info(f"Waiting for job {job_id} to complete")
        while self.check_job_status(job_id):
            time.sleep(self.check_interval)
        logging.info(f"Job {job_id} completed")

    def process_folder(self, task_dir: str, job_id: str, job_type: Optional[str] = None) -> None:
        """Process a single folder."""
        job_type_str = f" ({job_type})" if job_type else ""
        logging.info(f"Processing folder: {task_dir}{job_type_str}")

        while not self._check_pdb_files(task_dir):
            current_job_id = self.submit_job(task_dir, job_id, job_type)
            if not current_job_id:
                return

            self.wait_for_completion(current_job_id)

            if self._check_log_file(task_dir):
                logging.info(f"All PDB files generated for {task_dir}")
                break
            else:
                logging.warning(f"PDB files missing in {task_dir}. Resubmitting job.")
                self._backup_log_file(task_dir, current_job_id)

    def process_folders_concurrently(self, 
                                   folders: List[str], 
                                   job_ids: List[str], 
                                   max_workers: int,
                                   job_types: Optional[List[str]] = None) -> None:
        """Process multiple folders concurrently."""
        if not folders or not job_ids:
            logging.error("Empty input lists provided")
            return

        if len(folders) != len(job_ids):
            logging.error("Input lists have different lengths")
            return

        if job_types and len(job_types) != len(folders):
            logging.error("Job types list length doesn't match folders list length")
            return

        logging.info(f"Processing {len(folders)} folders concurrently")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, (folder, job_id) in enumerate(zip(folders, job_ids)):
                job_type = job_types[i] if job_types else None
                future = executor.submit(self.process_folder, folder, job_id, job_type)
                futures.append(future)

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Error processing folder: {str(e)}")

    def _check_pdb_files(self, task_dir: str) -> bool:
        """Check if all required PDB files exist."""
        pdb_files = [
            os.path.splitext(f)[0] + '.pdb'
            for f in os.listdir(task_dir) if f.endswith('.a3m')
        ]
        missing_files = [f for f in pdb_files if not os.path.exists(os.path.join(task_dir, f))]
        if missing_files:
            logging.debug(f"Missing PDB files in {task_dir}: {missing_files}")
            return False
        return True

    def _check_log_file(self, task_dir: str) -> bool:
        """Check if the log file indicates completion."""
        log_file = os.path.join(task_dir, 'log.txt')
        if not os.path.exists(log_file):
            return False
        try:
            with open(log_file, 'r') as f:
                return 'Done' in f.read()
        except Exception as e:
            logging.error(f"Error reading log file {log_file}: {e}")
            return False

    def _backup_log_file(self, task_dir: str, job_id: str) -> None:
        """Backup the log file before resubmitting."""
        log_file = os.path.join(task_dir, 'log.txt')
        if not os.path.exists(log_file):
            return
            
        backup_dir = os.path.join(task_dir, 'log_backups')
        os.makedirs(backup_dir, exist_ok=True)
        backup_file = os.path.join(backup_dir, f'log_{job_id}.txt')
        
        try:
            shutil.copy2(log_file, backup_file)
            os.remove(log_file)
            logging.info(f"Backed up log file to {backup_file}")
        except Exception as e:
            logging.error(f"Error backing up log file: {e}")

