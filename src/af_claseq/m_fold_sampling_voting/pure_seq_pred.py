"""
Pure Sequence Prediction module for AF-ClaSeq pipeline.

This module provides a class for submitting structure prediction jobs 
for sequence bins through SLURM, handling both regular predictions
and control predictions concurrently.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any

from af_claseq.utils.slurm_utils import SlurmJobSubmitter


class PureSequenceAF2Prediction:
    """
    Manages structure prediction for sequence bins.
    
    This class handles submitting prediction jobs for sequences in
    the prediction and control_prediction directories, using SLURM
    as the job scheduler.
    """
    
    def __init__(self, 
                 config: Dict[str, Any],  # Using Dict to accept config from pipeline
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the sequence prediction manager.
        
        Args:
            config: Configuration options for prediction from pipeline
            logger: Optional logger, will create one if not provided
        """
        self.config = config
        self.logger = logger or self._setup_logger()
        self.submitter = self._init_slurm_submitter()
        self.job_configs = {}
        
        # Convert bin_numbers to list if it's a single integer
        if isinstance(self.config['bin_numbers'], int):
            self.config['bin_numbers'] = [self.config['bin_numbers']]
        
    def _setup_logger(self) -> logging.Logger:
        """Set up a default logger if none was provided."""
        logger = logging.getLogger("af_claseq.pure_seq_pred")
        
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
        return logger
    
    def _init_slurm_submitter(self) -> SlurmJobSubmitter:
        """Initialize the SLURM job submitter with configuration options."""
        # Extract job name prefix from base directory if not specified
        job_prefix = self.config.get('job_name_prefix')
        if not job_prefix:
            base_dir = Path(self.config['pure_seq_pred_base_dir'])
            try:
                # Convert Path to parts safely
                base_path_parts = base_dir.parts
                results_idx = list(base_path_parts).index('results')
                job_prefix = base_path_parts[results_idx + 1] if results_idx + 1 < len(base_path_parts) else "fold"
            except (ValueError, AttributeError, IndexError):
                job_prefix = "fold"
                self.logger.warning("Could not extract job prefix from pure_seq_pred_base_dir path, using default: 'fold'")
        
        return SlurmJobSubmitter(
            conda_env_path=self.config['conda_env_path'],
            slurm_account=self.config['slurm_account'],
            slurm_output=self.config['slurm_output'],
            slurm_error=self.config['slurm_error'],
            slurm_nodes=self.config['slurm_nodes'],
            slurm_gpus_per_task=self.config['slurm_gpus_per_task'],
            slurm_tasks=self.config['slurm_tasks'],
            slurm_cpus_per_task=self.config['slurm_cpus_per_task'],
            slurm_time=self.config['slurm_time'],
            slurm_partition=self.config['slurm_partition'],
            check_interval=self.config['check_interval'],
            job_name_prefix=job_prefix,
            prediction_num_model=self.config['prediction_num_model'],
            prediction_num_seed=self.config['prediction_num_seed']
        )
        
    def collect_job_configs(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Collect all job configurations for prediction directories.
        
        Returns:
            Tuple containing lists of job folders, job IDs, and job types
        """
        all_jobs = []
        all_job_ids = []
        job_types = []
        
        base_dir = self.config['pure_seq_pred_base_dir']
        if not os.path.exists(base_dir):
            self.logger.error(f"Base directory not found: {base_dir}")
            return all_jobs, all_job_ids, job_types
            
        bin_numbers = self.config['bin_numbers']
        if not bin_numbers:
            self.logger.error("No bin numbers provided")
            return all_jobs, all_job_ids, job_types
        
        # Define directory types to process (prediction and control_prediction)
        dir_types = {
            "prediction": ("pred", "prediction"),
            "control_prediction": ("ctrl_pred", "control_prediction")
        }
        
        # Process each directory type
        for dir_name, (job_prefix, job_type) in dir_types.items():
            dir_path = os.path.join(base_dir, dir_name)
            if not os.path.exists(dir_path):
                self.logger.warning(f"{dir_name} directory not found: {dir_path}")
                continue
                
            self.logger.info(f"Collecting jobs from {dir_name} directory: {dir_path}")
            
            if self.config['combine_bins']:
                # Create combined bin name (e.g. "bin_5_6_7")
                bin_name = f"bin_{'_'.join(str(b) for b in sorted(bin_numbers))}"
                bin_dir = os.path.join(dir_path, bin_name)
                
                if os.path.exists(bin_dir):
                    job_id = f"{job_prefix}_{bin_name}"
                    all_jobs.append(bin_dir)
                    all_job_ids.append(job_id)
                    job_types.append(job_type)
                    self.logger.info(f"Added {dir_name} job for combined bins: {bin_dir}")
                else:
                    self.logger.warning(f"Combined bin directory not found: {bin_dir}")
            else:
                # Process each bin directory separately
                for bin_num in bin_numbers:
                    bin_dir = os.path.join(dir_path, f"bin_{bin_num}")
                    if os.path.exists(bin_dir):
                        job_id = f"{job_prefix}_bin{bin_num}"
                        all_jobs.append(bin_dir)
                        all_job_ids.append(job_id)
                        job_types.append(job_type)
                        self.logger.info(f"Added {dir_name} job for bin {bin_num}: {bin_dir}")
                    else:
                        self.logger.warning(f"Bin directory not found: {bin_dir}")
                        
        self.job_configs = {
            "folders": all_jobs,
            "job_ids": all_job_ids,
            "job_types": job_types
        }
        
        return all_jobs, all_job_ids, job_types
    
    def submit_prediction_jobs(self) -> bool:
        """
        Submit all prediction jobs to SLURM.
        
        Returns:
            Boolean indicating success
        """
        # Collect job configurations if not already done
        if not self.job_configs:
            all_jobs, all_job_ids, job_types = self.collect_job_configs()
        else:
            all_jobs = self.job_configs["folders"]
            all_job_ids = self.job_configs["job_ids"]
            job_types = self.job_configs["job_types"]
        
        # Process all jobs concurrently
        if all_jobs:
            self.logger.info(f"Processing {len(all_jobs)} jobs concurrently...")
            try:
                max_workers = self.config.get('max_workers', 4)  # Default to 4 if not specified
                self.submitter.process_folders_concurrently(
                    folders=all_jobs,
                    job_ids=all_job_ids,
                    max_workers=max_workers,
                    job_types=job_types
                )
                self.logger.info("All prediction jobs submitted successfully")
                return True
            except Exception as e:
                self.logger.error(f"Error submitting prediction jobs: {str(e)}")
                return False
        else:
            self.logger.warning("No jobs found to process")
            return False
            
    def run(self) -> bool:
        """
        Run the prediction process from start to finish.
        
        Returns:
            Boolean indicating success
        """
        try:
            self.logger.info("Starting prediction process...")
            self.collect_job_configs()
            result = self.submit_prediction_jobs()
            
            if result:
                self.logger.info("Prediction process completed successfully")
            else:
                self.logger.error("Prediction process failed")
                
            return result
            
        except Exception as e:
            self.logger.error(f"Unexpected error in prediction process: {str(e)}")
            return False