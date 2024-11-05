import os
import subprocess
import logging
import time
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List


class SlurmJobSubmitter:
    """
    A class to manage SLURM job submissions and monitoring.
    """

    def __init__(
        self,
        conda_env_path: str,
        slurm_account: str,
        slurm_output: str,
        slurm_error: str,
        slurm_nodes: int,
        slurm_gpus_per_task: int,
        slurm_tasks: int,
        slurm_cpus_per_task: int,
        slurm_time: str,
        random_seed: int,
        num_models: int,
        check_interval: int,
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
            random_seed (int): Random seed for the job.
            num_models (int): Number of models for the job.
            check_interval (int): Time in seconds between status checks.
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
        self.random_seed = random_seed
        self.num_models = num_models
        self.check_interval = check_interval

    def submit_job(self, task_dir: str, job_id: str) -> str:
        """
        Submit a SLURM job for the given task directory.

        Args:
            task_dir (str): The directory of the task.
            job_id (str): The identifier for the job.

        Returns:
            str: The submitted job ID.
        """
        env_setup = (
            "module reset && module load openmpi cuda miniconda3 && "
            f"conda init && conda activate {self.conda_env_path} && which python"
        )
        slurm_command = (
            f"sbatch --parsable -A {self.slurm_account} "
            f"--output={self.slurm_output} --error={self.slurm_error} "
            f"--nodes={self.slurm_nodes} --gpus-per-task={self.slurm_gpus_per_task} "
            f"--ntasks={self.slurm_tasks} --cpus-per-task={self.slurm_cpus_per_task} "
            f"--time={self.slurm_time}"
        )

        job_name = f"fold_{job_id}"
        command = (
            f"{slurm_command} --job-name={job_name} "
            f"--wrap='{env_setup} && colabfold_batch {task_dir} {task_dir} "
            f"--random-seed {self.random_seed} --num-models {self.num_models}'"
        )

        logging.info(f"Submitting job for {task_dir} (Job ID: {job_id})")
        try:
            submitted_job_id = subprocess.check_output(command, shell=True).strip().decode()
            logging.info(f"Submitted job {submitted_job_id} for task directory {task_dir}")
            return submitted_job_id
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to submit job for {task_dir}: {e}")
            raise

    def wait_for_completion(self, job_id: str):
        """
        Wait for the given SLURM job to complete.

        Args:
            job_id (str): The SLURM job ID.
        """
        logging.info(f"Waiting for job {job_id} to complete.")
        while True:
            try:
                job_statuses = subprocess.check_output(['squeue', '-j', job_id], text=True).strip().split()
                # If job_id is not in the queue, it's completed
                if job_id not in job_statuses:
                    logging.info(f"Job {job_id} has finished.")
                    return
                logging.debug(f"Job {job_id} is still running. Checking again in {self.check_interval} seconds.")
            except subprocess.CalledProcessError as e:
                logging.error(f"Error checking status for job {job_id}: {e}")
                raise
            time.sleep(self.check_interval)

    def process_folder(self, folder_path: str, job_id: str):
        """
        Process a single folder by submitting and monitoring SLURM jobs until completion.

        Args:
            folder_path (str): Path to the folder to process.
            job_id (str): Identifier for the job.
        """
        logging.info(f"Starting process for folder: {folder_path} (Job ID: {job_id})")

        while not self._check_pdb_files(folder_path):
            current_job_id = self.submit_job(folder_path, job_id)
            self.wait_for_completion(current_job_id)

            if self._check_log_file(folder_path):
                logging.info(f"All PDB files generated for {folder_path}. 'Done' found in log.txt.")
                break
            else:
                logging.warning(f"PDB files missing in {folder_path}. Resubmitting job.")
                self._backup_log_file(folder_path, current_job_id)

        logging.info(f"Completed processing for folder: {folder_path} (Job ID: {job_id})")

    def process_folders_concurrently(self, folders: List[str], job_ids: List[str], max_workers: int):
        """
        Process multiple folders concurrently.

        Args:
            folders (List[str]): List of folder paths to process.
            job_ids (List[str]): Corresponding list of job IDs.
            max_workers (int): Maximum number of concurrent threads.
        """
        logging.info(f"Processing {len(folders)} folders concurrently.")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.process_folder, folder, job_id): job_id
                for folder, job_id in zip(folders, job_ids)
            }
            for future in as_completed(futures):
                job_id = futures[future]
                try:
                    future.result()
                    logging.info(f"Successfully processed job ID: {job_id}")
                except Exception as exc:
                    logging.error(f"Job ID {job_id} generated an exception: {exc}")

    def _check_pdb_files(self, task_dir: str) -> bool:
        """
        Check if all required PDB files exist in the task directory.

        Args:
            task_dir (str): The task directory path.

        Returns:
            bool: True if all PDB files exist, False otherwise.
        """
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
        """
        Check if the log file contains the word "Done".

        Args:
            task_dir (str): The task directory path.

        Returns:
            bool: True if "Done" is found in log.txt, False otherwise.
        """
        log_file_path = os.path.join(task_dir, 'log.txt')
        if not os.path.exists(log_file_path):
            logging.debug(f"log.txt does not exist in {task_dir}")
            return False
        try:
            with open(log_file_path, 'r') as log_file:
                for line in log_file:
                    if "Done" in line:
                        return True
        except Exception as e:
            logging.error(f"Error reading log file {log_file_path}: {e}")
        return False

    def _backup_log_file(self, task_dir: str, job_id: str):
        """
        Backup the existing log file by renaming it.

        Args:
            task_dir (str): The task directory path.
            job_id (str): The job ID associated with the log.
        """
        base_backup_path = os.path.join(task_dir, 'log.backup')
        backup_counter = 1
        backup_file = f"{base_backup_path}.{backup_counter}"
        while os.path.exists(backup_file):
            backup_counter += 1
            backup_file = f"{base_backup_path}.{backup_counter}"
        try:
            os.rename(os.path.join(task_dir, 'log.txt'), backup_file)
            logging.info(f"Created backup log file: {backup_file} (Job ID: {job_id})")
        except OSError as e:
            logging.error(f"Failed to backup log file in {task_dir}: {e}")


class BatchFolderProcessor:
    """
    A class to handle processing of shuffle and round folders.
    """

    def __init__(self, slurm_submitter: SlurmJobSubmitter):
        """
        Initialize the BatchFolderProcessor with a SlurmJobSubmitter instance.

        Args:
            slurm_submitter (SlurmJobSubmitter): An instance of SlurmJobSubmitter.
        """
        self.slurm_submitter = slurm_submitter

    def process_shuffle_folders(self, shuffle_dir: str, num_shuffles: int):
        """
        Process shuffle folders concurrently.

        Args:
            shuffle_dir (str): The directory containing shuffle folders.
            num_shuffles (int): Number of shuffle folders to process.
        """
        logging.info(f"Processing shuffle folders in: {shuffle_dir}")
        shuffle_folders = [os.path.join(shuffle_dir, f"shuffle_{i}") for i in range(1, num_shuffles + 1)]
        job_ids = [f"shuffle_{i}" for i in range(1, num_shuffles + 1)]

        self.slurm_submitter.process_folders_concurrently(shuffle_folders, job_ids, max_workers=num_shuffles)

    def process_round_folders(self, round_dir: str):
        """
        Process round folders, organizing files and submitting jobs.

        Args:
            round_dir (str): The directory containing round folders.
        """
        logging.info(f"Processing round folders in: {round_dir}")
        round_subdirs = sorted([
            d for d in os.listdir(round_dir)
            if d.endswith('_cluster-msa') and d.startswith('Round')
        ])

        if not round_subdirs:
            logging.warning(f"No round subdirectories found in {round_dir}.")
            return

        round_dirs = [os.path.join(round_dir, subdir) for subdir in round_subdirs]
        round_ids = [subdir.split('_')[0] for subdir in round_subdirs]

        for subdir_path, round_id in zip(round_dirs, round_ids):
            self._organize_round_subdir(subdir_path, round_id)

        self.slurm_submitter.process_folders_concurrently(
            round_dirs,
            round_ids,
            # max_workers=len(round_dirs),
            max_workers=8
        )

    def _organize_round_subdir(self, subdir_path: str, round_id: str):
        """
        Organize files within a round subdirectory.

        Args:
            subdir_path (str): Path to the round subdirectory.
            round_id (str): Identifier for the round.
        """
        non_a3m_dir = os.path.join(subdir_path, 'non_a3m')
        os.makedirs(non_a3m_dir, exist_ok=True)

        for filename in os.listdir(subdir_path):
            file_path = os.path.join(subdir_path, filename)
            if os.path.isfile(file_path) and not filename.endswith('.a3m'):
                try:
                    shutil.move(file_path, os.path.join(non_a3m_dir, filename))
                    logging.info(f"Moved {filename} to non_a3m directory in {subdir_path}")
                except shutil.Error as e:
                    logging.error(f"Failed to move {filename} in {subdir_path}: {e}")


def slurm_submission(
    shuffle_dir: Optional[str],
    round_dir: Optional[str],
    num_shuffles: int,
    conda_env_path: str,
    slurm_account: str,
    slurm_output: str,
    slurm_error: str,
    slurm_nodes: int,
    slurm_gpus_per_task: int,
    slurm_tasks: int,
    slurm_cpus_per_task: int,
    slurm_time: str,
    random_seed: int,
    num_models: int,
    check_interval: int,
):
    """
    Entry point for submitting SLURM jobs for shuffle and/or round directories.

    Args:
        shuffle_dir (Optional[str]): Directory containing shuffle folders.
        round_dir (Optional[str]): Directory containing round folders.
        num_shuffles (int): Number of shuffle folders to process.
        conda_env_path (str): Path to the Conda environment.
        slurm_account (str): SLURM account name.
        slurm_output (str): SLURM job output path.
        slurm_error (str): SLURM job error path.
        slurm_nodes (int): Number of nodes.
        slurm_gpus_per_task (int): Number of GPUs per task.
        slurm_tasks (int): Number of tasks.
        slurm_cpus_per_task (int): Number of CPUs per task.
        slurm_time (str): SLURM job time limit.
        random_seed (int): Random seed for the job.
        num_models (int): Number of models for the job.
        check_interval (int): Time in seconds between status checks.
    """
    slurm_submitter = SlurmJobSubmitter(
        conda_env_path=conda_env_path,
        slurm_account=slurm_account,
        slurm_output=slurm_output,
        slurm_error=slurm_error,
        slurm_nodes=slurm_nodes,
        slurm_gpus_per_task=slurm_gpus_per_task,
        slurm_tasks=slurm_tasks,
        slurm_cpus_per_task=slurm_cpus_per_task,
        slurm_time=slurm_time,
        random_seed=random_seed,
        num_models=num_models,
        check_interval=check_interval,
    )
    batch_processor = BatchFolderProcessor(slurm_submitter)

    if shuffle_dir:
        batch_processor.process_shuffle_folders(shuffle_dir, num_shuffles)

    if round_dir:
        batch_processor.process_round_folders(round_dir)

    if not shuffle_dir and not round_dir:
        logging.warning("No directories provided for processing. Please specify shuffle_dir or round_dir.")