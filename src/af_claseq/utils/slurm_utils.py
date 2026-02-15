import os
import subprocess
import time
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from af_claseq.utils.logging_utils import get_logger

# Initialize module logger
logger = get_logger("slurm_utils")


class JobState(Enum):
    """SLURM job states."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"
    UNKNOWN = "UNKNOWN"


@dataclass
class JobInfo:
    """Information about a submitted job."""
    job_id: str
    job_name: str
    command: str
    task_dir: str
    state: JobState = JobState.PENDING
    submitted_time: Optional[float] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None


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
        slurm_partition: str = "nextgen",
        check_interval: int = 60,
        job_name_prefix: str = "fold",
        num_recycle: int = 3,
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
            slurm_partition (str): SLURM partition to use.
            check_interval (int): Time in seconds between status checks.
            job_name_prefix (str): Prefix for job names.
            **kwargs: Additional arguments for different modes:
                For batch_pred mode:
                    - num_models (int): Number of models to generate
                    - num_seeds (int): Number of seeds to use
                    - random_seed (int): Random seed for reproducibility
                For pure_pred mode:
                    - prediction_num_model (int): Number of models for prediction
                    - prediction_num_seed (int): Number of seeds for prediction
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
        self.slurm_partition = slurm_partition
        self.check_interval = check_interval
        self.job_name_prefix = job_name_prefix
        self.num_recycle = num_recycle

        # Determine mode based on kwargs
        if any(k.startswith('prediction_') for k in kwargs):
            self.mode = 'pure_pred'
            self.job_configs = {
                'prediction': {
                    'num_models': kwargs.get('prediction_num_model', 1),
                    'num_seeds': kwargs.get('prediction_num_seed', 1)
                },
                'control_prediction': {
                    'num_models': kwargs.get('prediction_num_model', 1),
                    'num_seeds': kwargs.get('prediction_num_seed', 1)
                }
                
            }
        else:
            self.mode = 'batch_pred'
            self.num_models = kwargs.get('num_models', 1)
            self.num_seeds = kwargs.get('num_seeds', 1)
            self.random_seed = kwargs.get('random_seed')
            
        # Job tracking
        self.active_jobs: Dict[str, JobInfo] = {}
        self.completed_jobs: Dict[str, JobInfo] = {}

    def _get_job_config(self, job_type: Optional[str] = None) -> Dict[str, int]:
        """Get job configuration based on mode and job type."""
        if self.mode == 'pure_pred':
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
            logger.error(f"Task directory not found: {task_dir}")
            return None

        config = self._get_job_config(job_type)
        
        # Build colabfold command
        colabfold_cmd = [
            "colabfold_batch",
            "--num-recycle", str(self.num_recycle),
            "--num-models", str(config['num_models']),
            "--num-seeds", str(config['num_seeds']),
            # "--templates" , "--custom-template-path",
            # "/fs/ess/PAA0203/xing244/AF_ClaSeq/results_updated/BCCIP/deepmsa_default/finalMSAs-alpha/template_test/template"
        ]
        
        if 'random_seed' in config:
            colabfold_cmd.extend(["--random-seed", str(config['random_seed'])])
            
        colabfold_cmd.extend([task_dir, task_dir])
        colabfold_cmd = " ".join(colabfold_cmd)

        # Build environment setup
        env_setup = (
            "module reset && module load cuda/12.4.1 miniconda3/24.1.2-py310 && "
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
            f"--partition={self.slurm_partition}",
            f"--gres=gpu:{self.slurm_gpus_per_task}",
            "--wrap", f"{env_setup} && {colabfold_cmd}"
        ]

        job_type_str = f" ({job_type})" if job_type else ""
        logger.info(f"Submitting job for {task_dir}{job_type_str} (Job ID: {job_id})")
        
        try:
            result = subprocess.run(sbatch_cmd, capture_output=True, text=True, check=True)
            submitted_job_id = result.stdout.strip().split()[-1]
            logger.info(f"Submitted job {submitted_job_id} for task directory {task_dir}")
            return submitted_job_id
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to submit job for {task_dir}: {e}")
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
        logger.info(f"Waiting for job {job_id} to complete")
        while self.check_job_status(job_id):
            time.sleep(self.check_interval)
        logger.info(f"Job {job_id} completed")

    def process_folder(self, task_dir: str, job_id: str, job_type: Optional[str] = None) -> None:
        """Process a single folder."""
        job_type_str = f" ({job_type})" if job_type else ""
        logger.info(f"Processing folder: {task_dir}{job_type_str}")

        while not self._check_pdb_files(task_dir):
            current_job_id = self.submit_job(task_dir, job_id, job_type)
            if not current_job_id:
                return

            self.wait_for_completion(current_job_id)

            if self._check_log_file(task_dir):
                logger.info(f"All PDB files generated for {task_dir}")
                break
            else:
                logger.warning(f"PDB files missing in {task_dir}. Resubmitting job.")
                self._backup_log_file(task_dir, current_job_id)

    def process_folders_concurrently(self, 
                                   folders: List[str], 
                                   job_ids: List[str], 
                                   max_workers: int,
                                   job_types: Optional[List[str]] = None) -> None:
        """Process multiple folders concurrently."""
        if not folders or not job_ids:
            logger.error("Empty input lists provided")
            return

        if len(folders) != len(job_ids):
            logger.error("Input lists have different lengths")
            return

        if job_types and len(job_types) != len(folders):
            logger.error("Job types list length doesn't match folders list length")
            return

        logger.info(f"Processing {len(folders)} folders concurrently")
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
                    logger.error(f"Error processing folder: {str(e)}")

    def _check_pdb_files(self, task_dir: str) -> bool:
        """Check if all required PDB files exist."""
        pdb_files = [
            os.path.splitext(f)[0] + '.pdb'
            for f in os.listdir(task_dir) if f.endswith('.a3m')
        ]
        missing_files = [f for f in pdb_files if not os.path.exists(os.path.join(task_dir, f))]
        if missing_files:
            logger.debug(f"Missing PDB files in {task_dir}: {missing_files}")
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
            logger.error(f"Error reading log file {log_file}: {e}")
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
            logger.info(f"Backed up log file to {backup_file}")
        except Exception as e:
            logger.error(f"Error backing up log file: {e}")

    # ===== Hit Expand Enhancement Methods =====
    
    def submit_custom_job(self, 
                         job_name: str,
                         command: str,
                         task_dir: str,
                         memory: str = "32G",
                         gres: Optional[str] = None,
                         delay: float = 0.0) -> Optional[str]:
        """
        Submit a custom SLURM job with arbitrary command.
        
        Args:
            job_name: Name for the SLURM job
            command: Command to execute
            task_dir: Working directory for the job
            memory: Memory requirement (default: 32G)
            gres: GPU resources (optional, e.g., "gpu:1")
            delay: Delay before submission (seconds)
            
        Returns:
            SLURM job ID if successful, None otherwise
        """
        if delay > 0:
            time.sleep(delay)
            
        # Build sbatch command
        sbatch_cmd = [
            "sbatch",
            f"--account={self.slurm_account}",
            f"--job-name={job_name}",
            f"--output={self.slurm_output}",
            f"--error={self.slurm_error}",
            f"--nodes={self.slurm_nodes}",
            f"--ntasks={self.slurm_tasks}",
            f"--cpus-per-task={self.slurm_cpus_per_task}",
            f"--time={self.slurm_time}",
            f"--partition={self.slurm_partition}",
            f"--memory={memory}"
        ]
        
        if gres:
            sbatch_cmd.append(f"--gres={gres}")
            
        sbatch_cmd.extend(["--wrap", command])
        
        logger.info(f"Submitting custom job: {job_name}")
        
        try:
            result = subprocess.run(sbatch_cmd, capture_output=True, text=True, check=True)
            submitted_job_id = result.stdout.strip().split()[-1]
            
            # Track the job
            job_info = JobInfo(
                job_id=submitted_job_id,
                job_name=job_name,
                command=command,
                task_dir=task_dir,
                state=JobState.PENDING,
                submitted_time=time.time()
            )
            self.active_jobs[submitted_job_id] = job_info
            
            logger.info(f"Submitted custom job {submitted_job_id}: {job_name}")
            return submitted_job_id
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to submit custom job {job_name}: {e}")
            return None
    
    def get_job_state(self, job_id: str) -> JobState:
        """
        Get the current state of a SLURM job.
        
        Args:
            job_id: SLURM job ID
            
        Returns:
            Current job state
        """
        try:
            result = subprocess.run(
                ["squeue", "-j", job_id, "-h", "-o", "%T"],
                capture_output=True, text=True, check=False
            )
            
            if result.returncode == 0 and result.stdout.strip():
                state_str = result.stdout.strip()
                
                # Map SLURM states to our enum
                state_mapping = {
                    'PENDING': JobState.PENDING,
                    'RUNNING': JobState.RUNNING,
                    'COMPLETED': JobState.COMPLETED,
                    'FAILED': JobState.FAILED,
                    'CANCELLED': JobState.CANCELLED,
                    'TIMEOUT': JobState.TIMEOUT,
                    'R': JobState.RUNNING,
                    'PD': JobState.PENDING,
                    'CD': JobState.COMPLETED,
                    'F': JobState.FAILED,
                    'CA': JobState.CANCELLED,
                    'TO': JobState.TIMEOUT
                }
                
                return state_mapping.get(state_str, JobState.UNKNOWN)
            else:
                # Job not in queue, check if it completed
                return JobState.COMPLETED
                
        except Exception as e:
            logger.error(f"Error checking job state for {job_id}: {e}")
            return JobState.UNKNOWN
    
    def monitor_jobs(self, 
                    job_ids: List[str], 
                    check_interval: float = 60.0,
                    timeout: Optional[float] = None) -> Dict[str, JobState]:
        """
        Monitor multiple jobs until completion.
        
        Args:
            job_ids: List of SLURM job IDs to monitor
            check_interval: Time between status checks (seconds)
            timeout: Maximum time to wait (seconds), None for no timeout
            
        Returns:
            Dictionary mapping job IDs to final states
        """
        start_time = time.time()
        completed_jobs = {}
        remaining_jobs = set(job_ids)
        
        logger.info(f"Monitoring {len(job_ids)} jobs with check interval {check_interval}s")
        
        while remaining_jobs:
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                logger.warning(f"Monitoring timeout reached ({timeout}s)")
                for job_id in remaining_jobs:
                    completed_jobs[job_id] = JobState.TIMEOUT
                break
            
            # Check job states
            jobs_to_remove = []
            for job_id in remaining_jobs:
                state = self.get_job_state(job_id)
                
                if state in [JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED, JobState.TIMEOUT]:
                    completed_jobs[job_id] = state
                    jobs_to_remove.append(job_id)
                    
                    # Update job tracking
                    if job_id in self.active_jobs:
                        self.active_jobs[job_id].state = state
                        self.active_jobs[job_id].end_time = time.time()
                        self.completed_jobs[job_id] = self.active_jobs.pop(job_id)
            
            # Remove completed jobs from monitoring
            for job_id in jobs_to_remove:
                remaining_jobs.remove(job_id)
                logger.info(f"Job {job_id} completed with state: {completed_jobs[job_id]}")
            
            if remaining_jobs:
                logger.debug(f"{len(remaining_jobs)} jobs still running...")
                time.sleep(check_interval)
        
        logger.info(f"All jobs completed. States: {completed_jobs}")
        return completed_jobs
    
    def submit_batch_jobs(self,
                         job_specs: List[Dict[str, Any]],
                         max_concurrent: int = 50,
                         delay_between_jobs: float = 1.0) -> List[str]:
        """
        Submit multiple jobs in batches.
        
        Args:
            job_specs: List of job specifications, each containing:
                      {'name': str, 'command': str, 'task_dir': str, 'memory': str, 'gres': str}
            max_concurrent: Maximum number of concurrent jobs
            delay_between_jobs: Delay between job submissions
            
        Returns:
            List of submitted job IDs
        """
        submitted_jobs = []
        
        for i, spec in enumerate(job_specs):
            if len(self.active_jobs) >= max_concurrent:
                # Wait for some jobs to complete
                logger.info(f"Reached max concurrent jobs ({max_concurrent}), waiting...")
                active_job_ids = list(self.active_jobs.keys())
                self.monitor_jobs(active_job_ids[:10], check_interval=30.0)  # Wait for 10 jobs
            
            job_id = self.submit_custom_job(
                job_name=spec['name'],
                command=spec['command'],
                task_dir=spec['task_dir'],
                memory=spec.get('memory', '32G'),
                gres=spec.get('gres'),
                delay=delay_between_jobs if i > 0 else 0.0
            )
            
            if job_id:
                submitted_jobs.append(job_id)
            else:
                logger.error(f"Failed to submit job: {spec['name']}")
        
        logger.info(f"Submitted {len(submitted_jobs)}/{len(job_specs)} jobs")
        return submitted_jobs
    
    def get_job_statistics(self) -> Dict[str, Any]:
        """Get statistics about submitted jobs."""
        total_jobs = len(self.active_jobs) + len(self.completed_jobs)
        
        if total_jobs == 0:
            return {"total_jobs": 0}
        
        state_counts = {}
        for job in self.completed_jobs.values():
            state_counts[job.state.value] = state_counts.get(job.state.value, 0) + 1
        
        for job in self.active_jobs.values():
            current_state = self.get_job_state(job.job_id)
            state_counts[current_state.value] = state_counts.get(current_state.value, 0) + 1
        
        return {
            "total_jobs": total_jobs,
            "active_jobs": len(self.active_jobs),
            "completed_jobs": len(self.completed_jobs),
            "state_counts": state_counts
        }
