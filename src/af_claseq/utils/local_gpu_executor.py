import os
import subprocess
import time
import shutil
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict, Any

from af_claseq.utils.logging_utils import get_logger

logger = get_logger("local_gpu_executor")


class LocalGPUExecutor:
    """Execute ColabFold jobs locally on multi-GPU machines.

    Mirrors the SlurmJobSubmitter interface so all workflows work unchanged.
    Pins one ColabFold job per GPU via CUDA_VISIBLE_DEVICES.
    """

    def __init__(
        self,
        cuda_visible_devices: str,
        num_recycle: int = 3,
        job_name_prefix: str = "fold",
        check_interval: int = 10,
        **kwargs
    ):
        # Parse GPU IDs
        self.gpu_ids = [g.strip() for g in cuda_visible_devices.split(",") if g.strip()]
        if not self.gpu_ids:
            raise ValueError("cuda_visible_devices must contain at least one GPU ID")

        self.num_recycle = num_recycle
        self.job_name_prefix = job_name_prefix
        self.check_interval = check_interval

        # ColabFold mode config — same logic as SlurmJobSubmitter
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

        # GPU pool for concurrency control
        self.gpu_queue = queue.Queue()
        for gpu_id in self.gpu_ids:
            self.gpu_queue.put(gpu_id)

        logger.info(f"LocalGPUExecutor initialized with {len(self.gpu_ids)} GPUs: {self.gpu_ids}")

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

    def _run_and_wait(self, task_dir: str, job_id: str, job_type: Optional[str] = None) -> bool:
        """Run a ColabFold job on a GPU and wait for completion.

        Acquires a GPU from the pool, runs colabfold_batch with
        CUDA_VISIBLE_DEVICES pinned, and releases the GPU when done.
        Returns True if the process exited successfully.
        """
        if not os.path.exists(task_dir):
            logger.error(f"Task directory not found: {task_dir}")
            return False

        config = self._get_job_config(job_type)

        colabfold_cmd = [
            "colabfold_batch",
            "--num-recycle", str(self.num_recycle),
            "--num-models", str(config['num_models']),
            "--num-seeds", str(config['num_seeds']),
        ]
        if 'random_seed' in config:
            colabfold_cmd.extend(["--random-seed", str(config['random_seed'])])
        colabfold_cmd.extend([task_dir, task_dir])

        gpu_id = self.gpu_queue.get()

        job_type_str = f" ({job_type})" if job_type else ""
        logger.info(f"Running job {job_id}{job_type_str} on GPU {gpu_id}: {task_dir}")

        try:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            process = subprocess.run(
                colabfold_cmd,
                env=env,
                capture_output=True,
                text=True,
                cwd=task_dir
            )

            if process.returncode != 0:
                logger.warning(f"Job {job_id} on GPU {gpu_id} exited with code {process.returncode}")
                if process.stderr:
                    logger.warning(f"stderr: {process.stderr[:500]}")
                return False

            logger.info(f"Job {job_id} completed on GPU {gpu_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to run job for {task_dir}: {e}")
            return False

        finally:
            self.gpu_queue.put(gpu_id)  # Always release GPU

    def submit_job(self, task_dir: str, job_id: str, job_type: Optional[str] = None) -> Optional[str]:
        """Submit a local ColabFold job. Runs synchronously and returns job_id on success."""
        if not os.path.exists(task_dir):
            logger.error(f"Task directory not found: {task_dir}")
            return None

        logger.info(f"Submitting local job for {task_dir} (Job ID: {job_id})")
        return job_id

    def process_folder(self, task_dir: str, job_id: str, job_type: Optional[str] = None) -> None:
        """Process a single folder with retry logic.
        Same behavior as SlurmJobSubmitter.process_folder.
        """
        logger.info(f"Processing folder: {task_dir}")

        while not self._check_pdb_files(task_dir):
            success = self._run_and_wait(task_dir, job_id, job_type)

            if self._check_log_file(task_dir):
                logger.info(f"All PDB files generated for {task_dir}")
                break
            else:
                logger.warning(f"PDB files missing in {task_dir}. Retrying.")
                self._backup_log_file(task_dir, job_id)

    def process_folders_concurrently(
        self,
        folders: List[str],
        job_ids: List[str],
        max_workers: int,
        job_types: Optional[List[str]] = None
    ) -> None:
        """Process multiple folders concurrently.
        Same interface as SlurmJobSubmitter.process_folders_concurrently.
        max_workers is capped to the number of GPUs.
        """
        if not folders or not job_ids:
            logger.error("Empty input lists provided")
            return

        if len(folders) != len(job_ids):
            logger.error("Input lists have different lengths")
            return

        if job_types and len(job_types) != len(folders):
            logger.error("Job types list length doesn't match folders list length")
            return

        # Cap concurrency to number of GPUs
        effective_workers = min(max_workers, len(self.gpu_ids))
        logger.info(
            f"Processing {len(folders)} folders with {effective_workers} workers "
            f"({len(self.gpu_ids)} GPUs available)"
        )

        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
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

    # --- File checking methods (same logic as SlurmJobSubmitter) ---

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
        """Backup the log file before retrying."""
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

    # --- Monitoring stubs for workflows that call these ---

    def check_job_status(self, job_id: str) -> bool:
        """Local jobs are synchronous, always returns False (not running)."""
        return False

    def wait_for_completion(self, job_id: str) -> None:
        """Local jobs complete synchronously in _run_and_wait. No-op."""
        pass

    def monitor_jobs(
        self,
        job_ids: List[str],
        check_interval: float = 60.0,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Return all jobs as COMPLETED since local jobs are synchronous."""
        from af_claseq.utils.slurm_utils import JobState
        return {jid: JobState.COMPLETED for jid in job_ids}
