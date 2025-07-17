"""
Init bootstrapping module for AF-ClaSeq pipeline.

This module provides a quick preview of the hit expand process with fewer sequences
to assess sequence distribution before running the full pipeline.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import sys
from dataclasses import asdict

from tqdm import tqdm

from af_claseq.pipeline.config import InitBootstrappingConfig
from af_claseq.modules.subset_generator import SubsetGenerator
from af_claseq.utils.sequence_processing import A3MParser, validate_a3m_file
from af_claseq.utils.slurm_utils import SlurmJobSubmitter
from af_claseq.utils.logging_utils import get_logger


class InitBootstrappingError(Exception):
    """Raised when init bootstrapping pipeline fails."""
    pass


class InitBootstrappingRunner:
    """
    Quick bootstrapping preview of the hit expand pipeline.
    
    Runs a subset of the hit expand workflow with reduced parameters
    to provide a quick preview of sequence distribution.
    """
    
    def __init__(self,
                 config: InitBootstrappingConfig,
                 slurm_submitter: SlurmJobSubmitter,
                 base_dir: Path,
                 input_msa: Path,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize init bootstrapping runner.
        
        Args:
            config: Init bootstrapping configuration
            slurm_submitter: SLURM job submitter instance
            base_dir: Base output directory (should be 00_init_bootstrapping)
            input_msa: Path to input MSA file
            logger: Optional logger instance
        """
        self.config = config
        self.slurm_submitter = slurm_submitter
        self.base_dir = Path(base_dir)
        self.input_msa = Path(input_msa)
        self.logger = logger or get_logger(__name__)
        
        # Create output directory
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.parser = A3MParser(strict_validation=False)
        
        # Set up paths
        self.subsets_dir = self.base_dir / "01_subsets"
        self.analysis_dir = self.base_dir / "02_analysis"
        
    def _write_done_file(self, stage_dir: Path, stage_name: str) -> None:
        """Write DONE file for a completed stage."""
        done_file = stage_dir / f"{stage_name}.DONE"
        done_file.write_text(f"Stage {stage_name} completed successfully\n")
        self.logger.info(f"Created DONE file: {done_file}")
        
    def _check_done_file(self, stage_dir: Path, stage_name: str) -> bool:
        """Check if a stage has already been completed."""
        done_file = stage_dir / f"{stage_name}.DONE"
        return done_file.exists()
    
    
    def _run_subset_generation_stage(self, input_msa: Path) -> Dict[str, Any]:
        """
        Generate random subsets directly from input MSA.
        
        Args:
            input_msa: Path to input MSA file
            
        Returns:
            Dictionary with subset generation results
        """
        self.logger.info("=== STAGE 1: INIT SUBSET GENERATION ===")
        self.subsets_dir.mkdir(parents=True, exist_ok=True)
        
        # Parse input MSA
        input_sequences = self.parser.parse_file(input_msa)
        self.logger.info(f"Loaded {len(input_sequences)} sequences from input MSA")
        
        # Initialize subset generator
        subset_generator = SubsetGenerator(
            num_random_sequences=self.config.init_num_random_sequences,
            coverage_threshold=0.8,  # Use default coverage threshold
            ensure_query_first=self.config.ensure_query_first,
            random_seed=self.config.random_seed,
            logger=self.logger
        )
        
        # Generate subsets with progress bar
        self.logger.info(f"Generating {self.config.init_num_subsets} subsets...")
        subset_paths = subset_generator.generate_multiple_subsets(
            a3m_file=input_msa,
            num_subsets=self.config.init_num_subsets,
            output_dir=self.subsets_dir,
            output_prefix=self.config.init_output_prefix
        )
        
        # Organize into batches
        self.logger.info(f"Organizing {len(subset_paths)} subsets into {self.config.init_num_batches} batches...")
        batch_info = subset_generator._organize_into_batches(
            subset_paths=subset_paths,
            output_dir=self.subsets_dir,
            num_batches=self.config.init_num_batches,
            batch_prefix=self.config.init_batch_prefix
        )
        
        # Save subset generation summary
        summary_file = self.subsets_dir / "subset_generation_summary.json"
        summary = {
            "num_input_sequences": len(input_sequences),
            "num_subsets": self.config.init_num_subsets,
            "num_random_sequences": self.config.init_num_random_sequences,
            "num_batches": self.config.init_num_batches,
            "batch_info": batch_info
        }
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Write DONE file
        self._write_done_file(self.subsets_dir, "01_init_subsets")
        
        return {
            "subset_paths": subset_paths,
            "batch_info": batch_info,
            "input_sequences": input_sequences
        }
    
    def _run_structure_prediction_stage(self, subset_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit ColabFold batch jobs for structure prediction.
        
        Args:
            subset_results: Results from subset generation
            
        Returns:
            Dictionary with job submission results
        """
        self.logger.info("=== STAGE 3: INIT STRUCTURE PREDICTION ===")
        
        batch_info = subset_results["batch_info"]
        job_specs = []
        submitted_jobs = []
        
        # Submit jobs for each batch with progress bar
        with tqdm(total=len(batch_info["batch_dirs"]), desc="Submitting batch jobs") as pbar:
            for batch_name, batch_dir in batch_info["batch_dirs"].items():
                batch_path = Path(batch_dir)
                
                # Check for existing results
                existing_pdbs = list(batch_path.glob("**/*_unrelaxed_rank_001_*.pdb"))
                if existing_pdbs and self.config.skip_if_exists:
                    self.logger.info(f"Skipping {batch_name}: found {len(existing_pdbs)} existing structures")
                    pbar.update(1)
                    continue
                
                # Create job spec
                job_spec = {
                    "job_name": f"init_cf_{batch_name}",
                    "batch_dir": str(batch_path),
                    "num_models": 1,  # Quick preview - only 1 model
                    "use_gpu": True,
                    "num_recycle": 3
                }
                job_specs.append(job_spec)
                
                # Submit job
                try:
                    job_id = self.slurm_submitter.submit_colabfold_batch(
                        batch_dir=batch_path,
                        job_name=job_spec["job_name"],
                        num_models=1,
                        use_gpu=True,
                        num_recycle=3
                    )
                    
                    if job_id:
                        submitted_jobs.append({
                            "job_id": job_id,
                            "batch_name": batch_name,
                            "batch_dir": str(batch_path)
                        })
                        self.logger.info(f"Submitted job {job_id} for {batch_name}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to submit job for {batch_name}: {e}")
                
                pbar.update(1)
        
        # Save job submission info
        job_info_file = self.subsets_dir / "job_submission_info.json"
        job_info = {
            "num_jobs_submitted": len(submitted_jobs),
            "submitted_jobs": submitted_jobs,
            "job_specs": job_specs
        }
        with open(job_info_file, 'w') as f:
            json.dump(job_info, f, indent=2)
        
        self.logger.info(f"Submitted {len(submitted_jobs)} jobs for structure prediction")
        
        # Monitor jobs if configured
        if self.config.monitor_jobs and submitted_jobs:
            self.logger.info("Monitoring job progress...")
            job_ids = [job["job_id"] for job in submitted_jobs]
            completed_jobs = self.slurm_submitter.monitor_jobs(
                job_ids=job_ids,
                check_interval=self.config.job_check_interval,
                max_wait_time=self.config.max_job_wait_time
            )
            self.logger.info(f"Completed {len(completed_jobs)} out of {len(job_ids)} jobs")
        
        # Write DONE file
        self._write_done_file(self.subsets_dir, "02_init_structure_prediction")
        
        return {
            "job_specs": job_specs,
            "submitted_jobs": submitted_jobs,
            "prediction_dir": self.subsets_dir
        }
    
    def _run_structure_analysis(self, prediction_results: Dict[str, Any]) -> None:
        """
        Analyze predicted structures and create plots.
        
        Args:
            prediction_results: Results from structure prediction
        """
        self.logger.info("=== STAGE 4: INIT STRUCTURE ANALYSIS ===")
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Import required modules
        from af_claseq.utils.structure_analysis import StructureAnalyzer
        from af_claseq.utils.plotting_manager import PlottingManager
        
        # Initialize analyzer
        filter_config_path = Path(self.config.filter_config_path)
        if not filter_config_path.exists():
            self.logger.error(f"Filter config not found: {filter_config_path}")
            return
        
        analyzer = StructureAnalyzer(
            filter_config_path=filter_config_path,
            plddt_threshold=self.config.init_plddt_threshold,
            logger=self.logger
        )
        
        # Find all predicted structures
        prediction_dir = Path(prediction_results["prediction_dir"])
        pdb_files = list(prediction_dir.glob("**/init_subset_*_unrelaxed_rank_001_*.pdb"))
        
        self.logger.info(f"Found {len(pdb_files)} predicted structures")
        
        if not pdb_files:
            self.logger.warning("No structures found for analysis")
            return
        
        # Analyze structures with progress bar
        analysis_results = {}
        with tqdm(total=len(pdb_files), desc="Analyzing structures") as pbar:
            for pdb_file in pdb_files:
                try:
                    # Temporarily suppress logging
                    original_level = logging.getLogger('af_claseq.utils.sequence_processing').level
                    logging.getLogger('af_claseq.utils.sequence_processing').setLevel(logging.WARNING)
                    
                    result = analyzer.process_single_pdb(str(pdb_file))
                    
                    # Restore logging
                    logging.getLogger('af_claseq.utils.sequence_processing').setLevel(original_level)
                    
                    if result:
                        subset_name = pdb_file.parent.name
                        analysis_results[subset_name] = result
                except Exception as e:
                    self.logger.debug(f"Failed to analyze {pdb_file}: {e}")
                pbar.update(1)
        
        self.logger.info(f"Successfully analyzed {len(analysis_results)} structures")
        
        # Save analysis results
        results_file = self.analysis_dir / "init_structure_analysis_results.json"
        with open(results_file, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        # Create CSV file
        if analysis_results:
            import pandas as pd
            
            # Convert to DataFrame
            df_data = []
            for subset_name, result in analysis_results.items():
                row = {
                    'subset': subset_name,
                    'plddt': result.get('plddt', 0)
                }
                # Add dynamic metrics
                for metric_name, metric_value in result.items():
                    if metric_name != 'plddt':
                        row[metric_name] = metric_value
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            csv_file = self.analysis_dir / "init_structure_analysis_results.csv"
            df.to_csv(csv_file, index=False)
            self.logger.info(f"Saved analysis results to {csv_file}")
            
            # Create plots
            plots_dir = self.analysis_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            plotting_manager = PlottingManager(logger=self.logger)
            
            # Get metric names from filter config
            with open(filter_config_path, 'r') as f:
                filter_config = json.load(f)
            
            filter_criteria = filter_config.get('filter_criteria', [])
            if len(filter_criteria) >= 2:
                metric1_name = filter_criteria[0]['name']
                metric2_name = filter_criteria[1]['name']
                
                # Create 2D scatter plot
                self.logger.info(f"Creating 2D scatter plot for {metric1_name} vs {metric2_name}")
                plotting_manager.create_2d_scatter_plot(
                    df=df,
                    metric1_col=metric1_name,
                    metric2_col=metric2_name,
                    color_col='plddt',
                    output_path=plots_dir / f"init_{metric1_name}_vs_{metric2_name}_scatter.png",
                    xlabel=metric1_name.replace('_', ' ').title(),
                    ylabel=metric2_name.replace('_', ' ').title(),
                    title=f"Init Bootstrapping: {metric1_name} vs {metric2_name}",
                    figsize=self.config.init_plot_figsize,
                    x_min=self.config.init_plot_metric1_min,
                    x_max=self.config.init_plot_metric1_max,
                    y_min=self.config.init_plot_metric2_min,
                    y_max=self.config.init_plot_metric2_max,
                    x_ticks=self.config.init_plot_metric1_ticks,
                    y_ticks=self.config.init_plot_metric2_ticks
                )
        
        # Write DONE file
        self._write_done_file(self.analysis_dir, "03_init_analysis")
    
    def run(self) -> bool:
        """
        Run the init bootstrapping pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("=== STARTING INIT BOOTSTRAPPING PIPELINE ===")
            self.logger.info(f"Input MSA: {self.input_msa}")
            self.logger.info(f"Output directory: {self.base_dir}")
            
            # Save configuration
            config_file = self.base_dir / "init_bootstrapping_config.json"
            config_dict = asdict(self.config)
            config_dict["input_msa"] = str(self.input_msa)
            with open(config_file, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            # Check if all stages are complete
            all_done = (
                self._check_done_file(self.subsets_dir, "01_init_subsets") and
                self._check_done_file(self.subsets_dir, "02_init_structure_prediction") and
                self._check_done_file(self.analysis_dir, "03_init_analysis")
            )
            
            if all_done and self.config.skip_if_exists:
                self.logger.info("=== ALL INIT BOOTSTRAPPING STAGES ALREADY COMPLETED ===")
                self.logger.info("Use --force to rerun or delete DONE files to rerun specific stages")
                return True
            
            # Validate input MSA
            if not validate_a3m_file(self.input_msa, strict=False):
                raise InitBootstrappingError(f"Invalid input MSA: {self.input_msa}")
            
            # Stage 1: Subset generation
            if self._check_done_file(self.subsets_dir, "01_init_subsets"):
                self.logger.info("=== STAGE 1: SUBSET GENERATION ALREADY COMPLETED - SKIPPING ===")
                # Load subset results
                summary_file = self.subsets_dir / "subset_generation_summary.json"
                if summary_file.exists():
                    with open(summary_file, 'r') as f:
                        summary = json.load(f)
                        subset_results = {
                            "batch_info": summary["batch_info"],
                            "input_sequences": {}  # Will be loaded if needed
                        }
                else:
                    raise InitBootstrappingError("Cannot find subset generation summary")
            else:
                subset_results = self._run_subset_generation_stage(self.input_msa)
            
            # Stage 2: Structure prediction
            if self._check_done_file(self.subsets_dir, "02_init_structure_prediction"):
                self.logger.info("=== STAGE 2: STRUCTURE PREDICTION ALREADY COMPLETED - SKIPPING ===")
                prediction_results = {"prediction_dir": self.subsets_dir}
            else:
                prediction_results = self._run_structure_prediction_stage(subset_results)
            
            # Stage 3: Structure analysis (if structures exist)
            if self._check_done_file(self.analysis_dir, "03_init_analysis"):
                self.logger.info("=== STAGE 3: STRUCTURE ANALYSIS ALREADY COMPLETED - SKIPPING ===")
            else:
                self._run_structure_analysis(prediction_results)
            
            self.logger.info("=== INIT BOOTSTRAPPING PIPELINE COMPLETED SUCCESSFULLY ===")
            self.logger.info(f"Results saved to: {self.base_dir}")
            
            # Create final summary
            self._create_final_summary()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Init bootstrapping pipeline failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False
    
    def _create_final_summary(self):
        """Create a final summary of the init bootstrapping results."""
        summary_file = self.base_dir / "init_bootstrapping_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("INIT BOOTSTRAPPING SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Input MSA: {self.input_msa}\n")
            f.write(f"Output directory: {self.base_dir}\n\n")
            
            # Input MSA summary
            f.write("INPUT MSA:\n")
            f.write(f"  Direct random subsetting (no clustering)\n\n")
            
            # Subset generation summary
            subset_summary = self.subsets_dir / "subset_generation_summary.json"
            if subset_summary.exists():
                with open(subset_summary, 'r') as sf:
                    subset_data = json.load(sf)
                    f.write("SUBSET GENERATION RESULTS:\n")
                    f.write(f"  Number of subsets: {subset_data.get('num_subsets', 'N/A')}\n")
                    f.write(f"  Sequences per subset: {subset_data.get('num_random_sequences', 'N/A')}\n")
                    f.write(f"  Number of batches: {subset_data.get('num_batches', 'N/A')}\n\n")
            
            # Structure prediction summary
            job_info_file = self.subsets_dir / "job_submission_info.json"
            if job_info_file.exists():
                with open(job_info_file, 'r') as jf:
                    job_data = json.load(jf)
                    f.write("STRUCTURE PREDICTION RESULTS:\n")
                    f.write(f"  Jobs submitted: {job_data.get('num_jobs_submitted', 'N/A')}\n\n")
            
            # Analysis summary
            analysis_file = self.analysis_dir / "init_structure_analysis_results.json"
            if analysis_file.exists():
                with open(analysis_file, 'r') as af:
                    analysis_data = json.load(af)
                    f.write("STRUCTURE ANALYSIS RESULTS:\n")
                    f.write(f"  Structures analyzed: {len(analysis_data)}\n")
                    
                    # Calculate average metrics
                    if analysis_data:
                        plddt_values = [v.get('plddt', 0) for v in analysis_data.values()]
                        avg_plddt = sum(plddt_values) / len(plddt_values)
                        f.write(f"  Average pLDDT: {avg_plddt:.2f}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("Init bootstrapping completed. Review results before running full pipeline.\n")
        
        self.logger.info(f"Created final summary: {summary_file}")


def run_init_bootstrapping(
    config: InitBootstrappingConfig,
    slurm_submitter: SlurmJobSubmitter,
    base_dir: Path,
    input_msa: Path,
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Run the init bootstrapping pipeline.
    
    Args:
        config: Init bootstrapping configuration
        slurm_submitter: SLURM job submitter
        base_dir: Base output directory
        input_msa: Path to input MSA file
        logger: Optional logger
        
    Returns:
        True if successful, False otherwise
    """
    runner = InitBootstrappingRunner(
        config=config,
        slurm_submitter=slurm_submitter,
        base_dir=base_dir,
        input_msa=input_msa,
        logger=logger
    )
    return runner.run()