#!/usr/bin/env python3
"""
High-quality SLURM job management for ColabFold structure prediction jobs.
Based on submit_abl1_jobs.py but with enterprise-grade code quality.
"""

import os
import subprocess
import logging
import time
import json
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


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
class SlurmJobConfig:
    """Configuration for SLURM job submission."""
    account: str = "PAA0203"
    partition: str = "nextgen"
    time_limit: str = "02:00:00"
    memory: str = "32G"
    cpus_per_task: str = "8"
    nodes: int = 1
    ntasks_per_node: int = 1
    gres: Optional[str] = None  # Can be None for CPU-only jobs
    conda_env_path: str = "/fs/ess/PAA0203/xing244/.conda/envs/colabfold"
    output_file: str = "/dev/null"
    error_file: str = "/dev/null"


@dataclass
class JobInfo:
    """Information about a submitted job."""
    job_id: str
    job_name: str
    input_dir: str
    submit_time: float
    state: JobState = JobState.PENDING


class SlurmJobError(Exception):
    """Raised when SLURM job operations fail."""
    pass


class SlurmJobManager:
    """Manages SLURM job submission and monitoring for ColabFold jobs."""
    
    def __init__(self, config: SlurmJobConfig):
        """
        Initialize SLURM job manager.
        
        Args:
            config: SLURM job configuration
        """
        self.config = config
        self.submitted_jobs: Dict[str, JobInfo] = {}
        
        # Validate SLURM availability
        self._validate_slurm_availability()
        
        logger.info("SLURM job manager initialized")
    
    def _validate_slurm_availability(self) -> None:
        """Validate that SLURM commands are available."""
        try:
            subprocess.run(['squeue', '--version'], 
                         capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise SlurmJobError("SLURM commands not available")
    
    def submit_colabfold_job(self, input_dir: Union[str, Path], 
                           job_name: str,
                           custom_config: Optional[SlurmJobConfig] = None) -> str:
        """
        Submit a ColabFold job to SLURM.
        
        Args:
            input_dir: Directory containing A3M files for ColabFold
            job_name: Name for the SLURM job
            custom_config: Optional custom configuration (overrides default)
            
        Returns:
            SLURM job ID
            
        Raises:
            SlurmJobError: If job submission fails
        """
        input_dir = Path(input_dir)
        
        if not input_dir.exists():
            raise SlurmJobError(f"Input directory not found: {input_dir}")
        
        # Check for A3M files
        a3m_files = list(input_dir.glob("*.a3m"))
        if not a3m_files:
            raise SlurmJobError(f"No A3M files found in {input_dir}")
        
        # Use custom config if provided
        config = custom_config or self.config
        
        logger.info(f"Submitting ColabFold job: {job_name}")
        logger.info(f"Input directory: {input_dir}")
        logger.info(f"A3M files: {len(a3m_files)}")
        
        try:
            # Build job command
            job_command = self._build_colabfold_command(input_dir, config)
            
            # Build sbatch command
            sbatch_cmd = self._build_sbatch_command(job_name, job_command, config)
            
            # Submit job
            result = subprocess.run(sbatch_cmd, capture_output=True, text=True, check=True)
            
            # Extract job ID
            job_id = result.stdout.strip().split()[-1]
            
            # Store job information
            job_info = JobInfo(
                job_id=job_id,
                job_name=job_name,
                input_dir=str(input_dir),
                submit_time=time.time()
            )
            self.submitted_jobs[job_id] = job_info
            
            logger.info(f"Successfully submitted job {job_id}")
            return job_id
            
        except subprocess.CalledProcessError as e:
            error_context = {
                'command': ' '.join(sbatch_cmd),
                'exit_code': e.returncode,
                'stdout': e.stdout,
                'stderr': e.stderr,
                'job_name': job_name,
                'input_dir': str(input_dir)
            }
            logger.error(f"SLURM job submission failed", extra=error_context)
            raise SlurmJobError(
                f"Failed to submit job '{job_name}' (exit code {e.returncode}): {e.stderr}"
            ) from e
    
    def _build_colabfold_command(self, input_dir: Path, 
                               config: SlurmJobConfig) -> str:
        """Build the ColabFold command to execute."""
        # Environment setup (remove CUDA module for CPU-only)
        env_setup = (
            "module reset && "
            "module load cuda/12.4.1  miniconda3/24.1.2-py310 && "
            f"conda activate {config.conda_env_path}"
        )
        
       
        colabfold_cmd = (
            f"colabfold_batch {input_dir} {input_dir} "
            "--num-models 1 --num-seeds 1 "
            
        )
        
        # Full command
        return f"{env_setup} && {colabfold_cmd}"
    
    def _build_sbatch_command(self, job_name: str, job_command: str,
                            config: SlurmJobConfig) -> List[str]:
        """Build the sbatch command."""
        sbatch_cmd = [
            "sbatch",
            f"--account={config.account}",
            f"--job-name={job_name}",
            f"--output={config.output_file}",
            f"--error={config.error_file}",
            f"--nodes={config.nodes}",
            f"--ntasks-per-node={config.ntasks_per_node}",
            f"--cpus-per-task={config.cpus_per_task}",
            f"--mem={config.memory}",
            f"--time={config.time_limit}",
            f"--partition={config.partition}"
        ]
        
        # Only add gres if it's specified (not null/empty)
        if config.gres and config.gres.strip():
            sbatch_cmd.append(f"--gres={config.gres}")
        
        sbatch_cmd.extend(["--wrap", job_command])
        
        return sbatch_cmd
    
    def submit_batch_jobs(self, batch_dir: Union[str, Path], 
                         job_prefix: str = "colabfold",
                         delay_between_jobs: float = 1.0) -> List[str]:
        """
        Submit jobs for all batches in a directory.
        
        Args:
            batch_dir: Directory containing batch subdirectories
            job_prefix: Prefix for job names
            delay_between_jobs: Delay in seconds between job submissions
            
        Returns:
            List of submitted job IDs
        """
        batch_dir = Path(batch_dir)
        
        if not batch_dir.exists():
            raise SlurmJobError(f"Batch directory not found: {batch_dir}")
        
        # Find batch directories
        batch_dirs = sorted([d for d in batch_dir.iterdir() 
                           if d.is_dir() and d.name.startswith('batch_')])
        
        if not batch_dirs:
            raise SlurmJobError(f"No batch directories found in {batch_dir}")
        
        logger.info(f"Submitting jobs for {len(batch_dirs)} batches")
        
        submitted_job_ids = []
        
        for batch_subdir in batch_dirs:
            batch_name = batch_subdir.name
            job_name = f"{job_prefix}_{batch_name}"
            
            try:
                # Check if batch has A3M files
                a3m_files = list(batch_subdir.glob("*.a3m"))
                if not a3m_files:
                    logger.warning(f"Skipping empty batch: {batch_name}")
                    continue
                
                logger.info(f"Submitting {batch_name}: {len(a3m_files)} files")
                
                # Submit job
                job_id = self.submit_colabfold_job(batch_subdir, job_name)
                submitted_job_ids.append(job_id)
                
                # Delay between submissions
                if delay_between_jobs > 0:
                    time.sleep(delay_between_jobs)
                
            except Exception as e:
                logger.error(f"Failed to submit job for {batch_name}: {e}")
                continue
        
        logger.info(f"Successfully submitted {len(submitted_job_ids)} jobs")
        return submitted_job_ids
    
    def get_job_state(self, job_id: str) -> JobState:
        """
        Get current state of a SLURM job.
        
        Args:
            job_id: SLURM job ID
            
        Returns:
            Current job state
        """
        try:
            result = subprocess.run(
                ['squeue', '-j', job_id, '-h', '-o', '%T'],
                capture_output=True, text=True, check=True
            )
            
            if result.stdout.strip():
                state_str = result.stdout.strip()
                # Map SLURM states to our enum
                state_mapping = {
                    'PENDING': JobState.PENDING,
                    'RUNNING': JobState.RUNNING,
                    'COMPLETED': JobState.COMPLETED,
                    'FAILED': JobState.FAILED,
                    'CANCELLED': JobState.CANCELLED,
                    'TIMEOUT': JobState.TIMEOUT,
                    'PD': JobState.PENDING,
                    'R': JobState.RUNNING,
                    'CG': JobState.RUNNING,  # Completing
                    'CD': JobState.COMPLETED,
                    'F': JobState.FAILED,
                    'CA': JobState.CANCELLED,
                    'TO': JobState.TIMEOUT
                }
                
                return state_mapping.get(state_str, JobState.UNKNOWN)
            else:
                # Job not in queue, check if completed
                return self._check_completed_job(job_id)
                
        except subprocess.CalledProcessError:
            # Job might be completed or not exist
            return self._check_completed_job(job_id)
    
    def _check_completed_job(self, job_id: str) -> JobState:
        """Check if job has completed by looking at sacct."""
        try:
            result = subprocess.run(
                ['sacct', '-j', job_id, '-n', '-o', 'State'],
                capture_output=True, text=True, check=True
            )
            
            if result.stdout.strip():
                state_lines = result.stdout.strip().split('\n')
                # Take the last state (most recent)
                last_state = state_lines[-1].strip()
                
                if 'COMPLETED' in last_state:
                    return JobState.COMPLETED
                elif 'FAILED' in last_state:
                    return JobState.FAILED
                elif 'CANCELLED' in last_state:
                    return JobState.CANCELLED
                elif 'TIMEOUT' in last_state:
                    return JobState.TIMEOUT
            
            return JobState.UNKNOWN
            
        except subprocess.CalledProcessError:
            return JobState.UNKNOWN
    
    def wait_for_jobs(self, job_ids: List[str], 
                     check_interval: float = 60.0,
                     timeout: Optional[float] = None) -> Dict[str, JobState]:
        """
        Wait for jobs to complete.
        
        Args:
            job_ids: List of job IDs to monitor
            check_interval: Time between status checks in seconds
            timeout: Maximum time to wait in seconds (None for no timeout)
            
        Returns:
            Dictionary mapping job IDs to final states
        """
        logger.info(f"Monitoring {len(job_ids)} jobs")
        
        start_time = time.time()
        final_states = {}
        remaining_jobs = set(job_ids)
        
        while remaining_jobs:
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                logger.warning(f"Timeout reached after {timeout} seconds")
                break
            
            completed_jobs = set()
            
            for job_id in remaining_jobs:
                state = self.get_job_state(job_id)
                
                # Update stored job info
                if job_id in self.submitted_jobs:
                    self.submitted_jobs[job_id].state = state
                
                # Check if job is finished
                if state in [JobState.COMPLETED, JobState.FAILED, 
                           JobState.CANCELLED, JobState.TIMEOUT]:
                    final_states[job_id] = state
                    completed_jobs.add(job_id)
                    
                    job_name = (self.submitted_jobs[job_id].job_name 
                              if job_id in self.submitted_jobs else job_id)
                    logger.info(f"Job {job_name} ({job_id}) finished: {state.value}")
            
            # Remove completed jobs from monitoring
            remaining_jobs -= completed_jobs
            
            if remaining_jobs:
                logger.info(f"{len(remaining_jobs)} jobs still running...")
                time.sleep(check_interval)
        
        # Add any remaining jobs as unknown state
        for job_id in remaining_jobs:
            final_states[job_id] = JobState.UNKNOWN
        
        return final_states
    
    def get_job_summary(self) -> Dict[str, Any]:
        """
        Get summary of all submitted jobs.
        
        Returns:
            Dictionary with job summary statistics
        """
        if not self.submitted_jobs:
            return {"total_jobs": 0}
        
        # Update job states
        for job_id in self.submitted_jobs:
            self.submitted_jobs[job_id].state = self.get_job_state(job_id)
        
        # Count by state
        state_counts = {}
        for state in JobState:
            state_counts[state.value] = sum(
                1 for job in self.submitted_jobs.values() 
                if job.state == state
            )
        
        return {
            "total_jobs": len(self.submitted_jobs),
            "state_counts": state_counts,
            "jobs": {job_id: asdict(job) for job_id, job in self.submitted_jobs.items()}
        }
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a SLURM job.
        
        Args:
            job_id: Job ID to cancel
            
        Returns:
            True if cancellation was successful
        """
        try:
            subprocess.run(['scancel', job_id], check=True)
            
            # Update job state
            if job_id in self.submitted_jobs:
                self.submitted_jobs[job_id].state = JobState.CANCELLED
            
            logger.info(f"Successfully cancelled job {job_id}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False
    
    def cancel_all_jobs(self) -> int:
        """
        Cancel all submitted jobs.
        
        Returns:
            Number of successfully cancelled jobs
        """
        cancelled_count = 0
        
        for job_id in self.submitted_jobs:
            if self.cancel_job(job_id):
                cancelled_count += 1
        
        return cancelled_count
    
    def save_job_info(self, output_file: Union[str, Path]) -> None:
        """
        Save job information to JSON file.
        
        Args:
            output_file: Path to output JSON file
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        job_summary = self.get_job_summary()
        
        with open(output_file, 'w') as f:
            json.dump(job_summary, f, indent=2, default=str)
        
        logger.info(f"Job information saved to {output_file}")
    
    def load_job_info(self, input_file: Union[str, Path]) -> None:
        """
        Load job information from JSON file.
        
        Args:
            input_file: Path to input JSON file
        """
        input_file = Path(input_file)
        
        if not input_file.exists():
            raise FileNotFoundError(f"Job info file not found: {input_file}")
        
        with open(input_file, 'r') as f:
            job_data = json.load(f)
        
        # Reconstruct job objects
        self.submitted_jobs = {}
        for job_id, job_dict in job_data.get('jobs', {}).items():
            job_info = JobInfo(
                job_id=job_dict['job_id'],
                job_name=job_dict['job_name'],
                input_dir=job_dict['input_dir'],
                submit_time=job_dict['submit_time'],
                state=JobState(job_dict['state'])
            )
            self.submitted_jobs[job_id] = job_info
        
        logger.info(f"Loaded information for {len(self.submitted_jobs)} jobs")


def create_default_slurm_config() -> SlurmJobConfig:
    """Create default SLURM configuration."""
    return SlurmJobConfig()


def submit_colabfold_jobs(batch_dir: Union[str, Path], 
                         job_prefix: str = "colabfold",
                         config: Optional[SlurmJobConfig] = None) -> Tuple[SlurmJobManager, List[str]]:
    """
    Convenience function to submit ColabFold jobs for all batches.
    
    Args:
        batch_dir: Directory containing batch subdirectories
        job_prefix: Prefix for job names
        config: SLURM configuration (uses default if None)
        
    Returns:
        Tuple of (job_manager, submitted_job_ids)
    """
    if config is None:
        config = create_default_slurm_config()
    
    manager = SlurmJobManager(config)
    job_ids = manager.submit_batch_jobs(batch_dir, job_prefix)
    
    return manager, job_ids