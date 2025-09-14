"""
ColabFold job management for the divide-and-conquer workflow.
Handles SLURM job submission and monitoring for structure prediction.
"""

import os
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging

from .utils import (
    validate_file_exists, create_directory, find_files_with_pattern
)
from af_claseq.utils.exceptions import WorkflowError


class ColabFoldManager:
    """
    Manages ColabFold job submission and monitoring using SLURM.
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize ColabFoldManager.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        
        # Extract ColabFold configuration
        self.colabfold_config = config.get('colabfold', {})
        self.slurm_config = config.get('slurm', {})
        
        # ColabFold parameters
        self.conda_env = self.colabfold_config.get('conda_env', 'colabfold')
        self.num_models = self.colabfold_config.get('num_models', 1)
        self.num_seeds = self.colabfold_config.get('num_seeds', 1)
        self.max_concurrent_jobs = self.colabfold_config.get('max_concurrent_jobs', 20)
        
        # SLURM parameters
        self.account = self.slurm_config.get('account', 'PAA0203')
        self.partition = self.slurm_config.get('partition', 'nextgen')
        self.time_limit = self.slurm_config.get('time', '03:00:00')
        self.memory = self.slurm_config.get('memory', '32G')
        self.cpus = self.slurm_config.get('cpus', 8)
        
        self.submitted_jobs = []
        self.job_metadata = {}
        
        self.logger.info(f"ColabFold configuration:")
        self.logger.info(f"  Conda environment: {self.conda_env}")
        self.logger.info(f"  Number of models: {self.num_models}")
        self.logger.info(f"  Number of seeds: {self.num_seeds}")
        self.logger.info(f"  Max concurrent jobs: {self.max_concurrent_jobs}")
        self.logger.info(f"  SLURM account: {self.account}")
        self.logger.info(f"  SLURM partition: {self.partition}")
    
    def clean_directory_for_colabfold(self, input_dir: str) -> None:
        """
        Clean directory of non-A3M files before ColabFold submission.
        
        Args:
            input_dir: Directory to clean
        """
        input_path = Path(input_dir)
        if not input_path.is_dir():
            return
        
        cleaned_files = []
        for file in input_path.iterdir():
            if file.is_file() and file.suffix.lower() not in ['.a3m', '.fasta', '.fas']:
                self.logger.debug(f"Removing non-sequence file: {file}")
                file.unlink(missing_ok=True)
                cleaned_files.append(str(file))
        
        if cleaned_files:
            self.logger.info(f"Cleaned {len(cleaned_files)} non-sequence files from {input_dir}")
    
    def submit_single_job(self, input_path: str, job_name: str, 
                         metadata: Optional[Dict] = None) -> Optional[str]:
        """
        Submit a single ColabFold job to SLURM.
        
        Args:
            input_path: Path to input directory containing A3M files
            job_name: Name for the SLURM job
            metadata: Optional metadata for the job
            
        Returns:
            SLURM job ID if successful, None otherwise
        """
        if not os.path.exists(input_path):
            self.logger.error(f"Input path does not exist: {input_path}")
            return None
        
        # Clean directory before submission
        self.clean_directory_for_colabfold(input_path)
        
        # Environment setup
        env_setup = f"source activate {self.conda_env}"
        
        # ColabFold command - using same directory as input and output
        colabfold_cmd = (
            f"colabfold_batch --num-models {self.num_models} "
            f"--num-seeds {self.num_seeds} --overwrite "
            f"{input_path} {input_path}"
        )
        
        # Full command
        full_command = f"{env_setup} && {colabfold_cmd}"
        
        # SLURM sbatch command
        sbatch_cmd = [
            'sbatch',
            f'--account={self.account}',
            f'--partition={self.partition}',
            f'--time={self.time_limit}',
            f'--mem={self.memory}',
            f'--cpus-per-task={self.cpus}',
            f'--job-name={job_name}',
            '--nodes=1',
            '--gres=gpu:1',
            '--ntasks-per-node=1',
            '--output=/dev/null',  # Redirect stdout to /dev/null
            '--error=/dev/null',   # Redirect stderr to /dev/null
            '--wrap',
            full_command
        ]
        
        try:
            result = subprocess.run(sbatch_cmd, capture_output=True, text=True, check=True)
            job_id = result.stdout.strip().split()[-1]
            
            self.submitted_jobs.append(job_id)
            self.job_metadata[job_id] = {
                'job_name': job_name,
                'input_path': input_path,
                'num_models': self.num_models,
                'num_seeds': self.num_seeds,
                'metadata': metadata or {}
            }
            
            self.logger.info(f"Submitted ColabFold job {job_id}: {job_name}")
            return job_id
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to submit job {job_name}: {e}")
            if e.stderr:
                self.logger.error(f"SLURM error: {e.stderr}")
            return None
        
        except Exception as e:
            self.logger.error(f"Unexpected error submitting job {job_name}: {e}")
            return None
    
    def submit_all_jobs(self, shuffle_dirs: List[str]) -> List[str]:
        """
        Submit ColabFold jobs for all shuffle directories.
        
        Args:
            shuffle_dirs: List of shuffle directory paths
            
        Returns:
            List of submitted job IDs
        """
        self.logger.info("=" * 50)
        self.logger.info("COLABFOLD JOB SUBMISSION STARTED")
        self.logger.info("=" * 50)
        
        submitted_job_ids = []
        failed_submissions = []
        
        # Submit jobs in batches to respect max_concurrent_jobs limit
        for i, shuffle_dir in enumerate(shuffle_dirs):
            # Wait if we've reached the concurrent job limit
            if len(submitted_job_ids) >= self.max_concurrent_jobs:
                self.logger.info(f"Reached max concurrent jobs limit ({self.max_concurrent_jobs})")
                self.logger.info("Waiting for some jobs to complete before submitting more...")
                
                # Wait for at least one job to complete
                self._wait_for_job_completion(submitted_job_ids[:self.max_concurrent_jobs//2])
            
            # Generate job name
            clade_name = os.path.basename(os.path.dirname(shuffle_dir))
            shuffle_name = os.path.basename(shuffle_dir)
            job_name = f"cf_{clade_name}_{shuffle_name}"
            
            # Count A3M files in the directory
            a3m_files = find_files_with_pattern(shuffle_dir, "*.a3m")
            
            if not a3m_files:
                self.logger.warning(f"No A3M files found in {shuffle_dir}, skipping")
                failed_submissions.append(shuffle_dir)
                continue
            
            self.logger.info(f"Submitting job {i+1}/{len(shuffle_dirs)}: {job_name}")
            self.logger.info(f"  Input directory: {shuffle_dir}")
            self.logger.info(f"  A3M files found: {len(a3m_files)}")
            
            # Submit job
            job_id = self.submit_single_job(
                input_path=shuffle_dir,
                job_name=job_name,
                metadata={
                    'clade': clade_name,
                    'shuffle': shuffle_name,
                    'a3m_count': len(a3m_files)
                }
            )
            
            if job_id:
                submitted_job_ids.append(job_id)
            else:
                failed_submissions.append(shuffle_dir)
        
        # Log submission summary
        self.logger.info("=" * 50)
        self.logger.info("COLABFOLD JOB SUBMISSION COMPLETED")
        self.logger.info(f"Successfully submitted: {len(submitted_job_ids)} jobs")
        self.logger.info(f"Failed submissions: {len(failed_submissions)} jobs")
        
        if failed_submissions:
            self.logger.warning("Failed submissions:")
            for failed_dir in failed_submissions:
                self.logger.warning(f"  - {failed_dir}")
        
        self.logger.info("=" * 50)
        
        return submitted_job_ids
    
    def _wait_for_job_completion(self, job_ids: List[str], timeout: int = 3600) -> None:
        """
        Wait for at least half of the specified jobs to complete.
        
        Args:
            job_ids: List of job IDs to monitor
            timeout: Maximum time to wait in seconds
        """
        start_time = time.time()
        target_completed = len(job_ids) // 2
        
        while time.time() - start_time < timeout:
            completed_jobs = self._count_completed_jobs(job_ids)
            
            if completed_jobs >= target_completed:
                self.logger.info(f"{completed_jobs}/{len(job_ids)} jobs completed")
                return
            
            self.logger.info(f"Waiting for jobs to complete: {completed_jobs}/{len(job_ids)} done")
            time.sleep(60)  # Check every minute
        
        self.logger.warning(f"Timeout reached while waiting for job completion")
    
    def _count_completed_jobs(self, job_ids: List[str]) -> int:
        """
        Count how many jobs in the list have completed.
        
        Args:
            job_ids: List of job IDs to check
            
        Returns:
            Number of completed jobs
        """
        if not job_ids:
            return 0
        
        try:
            result = subprocess.run(
                ['squeue', '-j', ','.join(job_ids), '-h', '-o', '%i %T'],
                capture_output=True, text=True
            )
            
            running_jobs = set()
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        job_id, status = parts[0], parts[1]
                        if status not in ['COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT']:
                            running_jobs.add(job_id)
            
            # Jobs not in squeue are considered completed
            completed_count = len([jid for jid in job_ids if jid not in running_jobs])
            return completed_count
            
        except subprocess.CalledProcessError:
            # If squeue fails, assume no jobs are completed
            return 0
    
    def monitor_jobs(self, job_ids: List[str], check_interval: int = 10) -> Dict[str, Any]:
        """
        Monitor job completion with detailed progress tracking.
        
        Args:
            job_ids: List of job IDs to monitor
            check_interval: Time between checks in seconds
            
        Returns:
            Dictionary with completion statistics
        """
        if not job_ids:
            self.logger.warning("No jobs to monitor")
            return {'completed_jobs': [], 'failed_jobs': [], 'total_jobs': 0}
        
        self.logger.info("=" * 50)
        self.logger.info("COLABFOLD JOB MONITORING STARTED")
        self.logger.info(f"Monitoring {len(job_ids)} jobs...")
        self.logger.info("=" * 50)
        
        completed_jobs = []
        failed_jobs = []
        start_time = time.time()
        
        while True:
            remaining_jobs = [jid for jid in job_ids 
                            if jid not in completed_jobs and jid not in failed_jobs]
            
            if not remaining_jobs:
                break
            
            # Check job status using squeue
            try:
                result = subprocess.run(
                    ['squeue', '-j', ','.join(remaining_jobs), '-h', '-o', '%i %T %M %L'],
                    capture_output=True, text=True
                )
                
                running_jobs = {}
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 4:
                            job_id, status, elapsed, remaining = parts[0], parts[1], parts[2], parts[3]
                            running_jobs[job_id] = {
                                'status': status,
                                'elapsed': elapsed,
                                'remaining': remaining
                            }
                            
                            if status in ['COMPLETED']:
                                completed_jobs.append(job_id)
                            elif status in ['FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY']:
                                failed_jobs.append(job_id)
                
                # Jobs not in squeue output are considered completed
                for job_id in remaining_jobs:
                    if job_id not in running_jobs and job_id not in completed_jobs and job_id not in failed_jobs:
                        completed_jobs.append(job_id)
                
                # Log progress
                elapsed_time = time.time() - start_time
                progress_msg = (
                    f"Job status: {len(completed_jobs)} completed, "
                    f"{len(failed_jobs)} failed, {len(remaining_jobs)} remaining "
                    f"(elapsed: {elapsed_time/3600:.1f}h)"
                )
                self.logger.info(progress_msg)
                
                if remaining_jobs:
                    time.sleep(check_interval)
                    
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Error checking job status: {e}")
                time.sleep(check_interval)
        
        total_time = time.time() - start_time
        
        self.logger.info("=" * 50)
        self.logger.info("COLABFOLD JOB MONITORING COMPLETED")
        self.logger.info(f"Total monitoring time: {total_time/3600:.1f} hours")
        self.logger.info(f"Completed jobs: {len(completed_jobs)}")
        self.logger.info(f"Failed jobs: {len(failed_jobs)}")
        self.logger.info("=" * 50)
        
        if failed_jobs:
            self.logger.warning("Failed jobs:")
            for job_id in failed_jobs:
                if job_id in self.job_metadata:
                    job_name = self.job_metadata[job_id]['job_name']
                    self.logger.warning(f"  {job_id}: {job_name}")
        
        return {
            'completed_jobs': completed_jobs,
            'failed_jobs': failed_jobs,
            'total_jobs': len(job_ids),
            'monitoring_time': total_time
        }