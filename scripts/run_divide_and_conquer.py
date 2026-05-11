#!/usr/bin/env python3
"""
Main orchestrator for the divide-and-conquer phylogenetic workflow.

This script implements a comprehensive pipeline for:
1. Phylogenetic tree construction and clade-based sequence splitting
2. Random shuffling and grouping within clades
3. ColabFold structure prediction job management
4. Multi-metric structure analysis
5. Publication-quality plot generation

Usage:
    python main.py [--config config.json] [--resume-from STEP] [--dry-run]
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Import from af_claseq package structure
from af_claseq.divide_and_conquer import (
    PhylogeneticProcessor, ShuffleManager,
    StructureAnalyzer, PlotGenerator
)
from af_claseq.divide_and_conquer.utils import setup_logging, load_config, find_files_with_pattern
from af_claseq.utils.exceptions import WorkflowError, ValidationError
from af_claseq.utils.slurm_utils import SlurmJobSubmitter


class WorkflowOrchestrator:
    """
    Main orchestrator for the complete workflow.
    """
    
    def __init__(self, config_file: str):
        """
        Initialize the workflow orchestrator.
        
        Args:
            config_file: Path to configuration file
        """
        self.config = load_config(config_file)
        self.logger = setup_logging(self.config)
        self.start_time = time.time()
        
        # Initialize components
        self.phylo_processor = None
        self.shuffle_manager = None
        self.structure_analyzer = None
        self.plot_generator = None
        
        # Workflow state
        self.query_header = None
        self.query_sequence = None
        self.clade_dirs = None
        self.shuffle_dirs = None
        self.analysis_results = None
    
    def step_1_phylogenetic_processing(self) -> None:
        """Step 1: Phylogenetic tree construction and clade splitting."""
        self.logger.info("STEP 1: PHYLOGENETIC PROCESSING")
        
        a3m_file = self.config['input']['a3m_file']
        
        # Initialize phylogenetic processor
        self.phylo_processor = PhylogeneticProcessor(self.config, self.logger)
        
        # Run complete phylogenetic processing
        self.clade_dirs, self.query_header, self.query_sequence = \
            self.phylo_processor.process_complete(a3m_file)
        
        self.logger.info(f"Step 1 completed: {len(self.clade_dirs)} clades generated")
    
    def step_2_shuffle_management(self) -> None:
        """Step 2: Random shuffling and grouping within clades."""
        self.logger.info("STEP 2: SHUFFLE MANAGEMENT")
        
        if not self.clade_dirs:
            raise WorkflowError("No clade directories available. Run Step 1 first.")
        
        # Initialize shuffle manager
        self.shuffle_manager = ShuffleManager(
            self.config, self.logger, self.query_header, self.query_sequence
        )
        
        # Process all clades
        self.shuffle_dirs = self.shuffle_manager.process_all_clades(self.clade_dirs)
        
        # Validate results
        validation_passed = self.shuffle_manager.validate_shuffle_results(self.shuffle_dirs)
        
        if not validation_passed:
            raise WorkflowError("Shuffle validation failed")
        
        self.logger.info(f"Step 2 completed: {len(self.shuffle_dirs)} shuffle directories created")
    
    def step_3_colabfold_jobs(self) -> None:
        """Step 3: ColabFold structure prediction job submission and monitoring."""
        self.logger.info("STEP 3: COLABFOLD STRUCTURE PREDICTION")

        if not self.shuffle_dirs:
            raise WorkflowError("No shuffle directories available. Run Step 2 first.")

        # Extract configuration parameters
        colabfold_config = self.config.get('colabfold', {})
        slurm_config = self.config.get('slurm', {})

        # Initialize SlurmJobSubmitter with ColabFold parameters
        slurm_submitter = SlurmJobSubmitter(
            conda_env_path=colabfold_config.get('conda_env', 'colabfold'),
            slurm_account=slurm_config.get('account', 'PAA0203'),
            slurm_partition=slurm_config.get('partition', 'nextgen'),
            slurm_time=slurm_config.get('time', '00:30:00'),
            slurm_cpus_per_task=slurm_config.get('cpus', 8),
            job_name_prefix="cf",
            num_models=colabfold_config.get('num_models', 1),
            num_seeds=colabfold_config.get('num_seeds', 1),
            num_recycle=colabfold_config.get('num_recycle', 3)
        )

        self.logger.info("==" * 25)
        self.logger.info("COLABFOLD JOB SUBMISSION STARTED")
        self.logger.info("==" * 25)

        # Prepare directories and job IDs for batch processing
        valid_dirs = []
        job_ids = []
        failed_prep = []

        for shuffle_dir in self.shuffle_dirs:
            # Check for A3M files
            a3m_files = find_files_with_pattern(shuffle_dir, "*.a3m")

            if not a3m_files:
                self.logger.warning(f"No A3M files found in {shuffle_dir}, skipping")
                failed_prep.append(shuffle_dir)
                continue

            # Clean directory of non-sequence files before submission
            self._clean_directory_for_colabfold(shuffle_dir)

            # Generate job ID
            clade_name = os.path.basename(os.path.dirname(shuffle_dir))
            shuffle_name = os.path.basename(shuffle_dir)
            job_id = f"cf_{clade_name}_{shuffle_name}"

            valid_dirs.append(shuffle_dir)
            job_ids.append(job_id)
            self.logger.info(f"Prepared: {job_id} ({len(a3m_files)} A3M files)")

        if not valid_dirs:
            raise WorkflowError("No valid shuffle directories found for ColabFold submission")

        self.logger.info(f"Processing {len(valid_dirs)} directories concurrently")
        self.logger.info(f"Failed preparations: {len(failed_prep)}")

        # Use SlurmJobSubmitter's high-level concurrent processing method
        max_workers = colabfold_config.get('max_concurrent_jobs', 90)
        slurm_submitter.process_folders_concurrently(
            folders=valid_dirs,
            job_ids=job_ids,
            max_workers=max_workers
        )

        # process_folders_concurrently is a blocking method that handles everything
        self.logger.info("==" * 25)
        self.logger.info("COLABFOLD JOB PROCESSING COMPLETED")
        self.logger.info(f"Processed {len(valid_dirs)} directories")
        self.logger.info(f"Failed preparations: {len(failed_prep)}")
        self.logger.info("==" * 25)

    def _clean_directory_for_colabfold(self, input_dir: str) -> None:
        """
        Move non-A3M files to a .colabfold_staging subdir instead of deleting.

        Args:
            input_dir: Directory to clean
        """
        input_path = Path(input_dir)
        if not input_path.is_dir():
            return

        staging_dir = input_path / '.colabfold_staging'
        moved_files = []
        for file in input_path.iterdir():
            if file.is_file() and file.suffix.lower() not in ['.a3m', '.fasta', '.fas']:
                staging_dir.mkdir(exist_ok=True)
                dest = staging_dir / file.name
                # Avoid overwriting previously staged files
                if dest.exists():
                    stem, suffix = file.stem, file.suffix
                    counter = 1
                    while dest.exists():
                        dest = staging_dir / f"{stem}_{counter}{suffix}"
                        counter += 1
                self.logger.debug(f"Moving non-sequence file to staging: {file}")
                file.rename(dest)
                moved_files.append(str(file))

        if moved_files:
            self.logger.info(f"Staged {len(moved_files)} non-sequence files from {input_dir} to {staging_dir}")
    
    def step_4_structure_analysis(self) -> None:
        """Step 4: Multi-metric structure analysis."""
        self.logger.info("STEP 4: STRUCTURE ANALYSIS")
        
        if not self.shuffle_dirs:
            raise WorkflowError("No shuffle directories available. Run Steps 1-3 first.")
        
        # Initialize structure analyzer
        self.structure_analyzer = StructureAnalyzer(self.config, self.logger)
        
        # Analyze structures - use working_dir for output file
        working_dir = self.config.get('output', {}).get('working_dir', '.')
        output_file = os.path.join(working_dir, "structure_analysis_results.csv")
        self.analysis_results = self.structure_analyzer.analyze_complete(
            self.shuffle_dirs, output_file
        )
        
        if self.analysis_results.empty:
            raise WorkflowError("No structures passed analysis criteria")
        
        self.logger.info(f"Step 4 completed: {len(self.analysis_results)} structures analyzed")
    
    def step_5_plot_generation(self) -> None:
        """Step 5: Publication-quality plot generation."""
        self.logger.info("STEP 5: PLOT GENERATION")
        
        if self.analysis_results is None or self.analysis_results.empty:
            raise WorkflowError("No analysis results available. Run Step 4 first.")
        
        # Initialize plot generator
        self.plot_generator = PlotGenerator(self.config, self.logger)
        
        # Generate all plots
        plot_files = self.plot_generator.generate_all_plots(self.analysis_results)
        
        # Save plot summary
        self.plot_generator.save_plot_summary(plot_files)
        
        self.logger.info(f"Step 5 completed: {len(plot_files)} plots generated")
    
    def run_complete_workflow(self, resume_from: str = None, dry_run: bool = False) -> None:
        """
        Run the complete workflow or resume from a specific step.
        
        Args:
            resume_from: Step to resume from ('step1', 'step2', etc.)
            dry_run: If True, only validate configuration without running
        """
        steps = {
            'step1': self.step_1_phylogenetic_processing,
            'step2': self.step_2_shuffle_management,
            'step3': self.step_3_colabfold_jobs,
            'step4': self.step_4_structure_analysis,
            'step5': self.step_5_plot_generation
        }
        
        if dry_run:
            self.logger.info("=" * 60)
            self.logger.info("DRY RUN MODE - Configuration Validation")
            self.logger.info("=" * 60)
            self._validate_configuration()
            self.logger.info("Configuration validation completed successfully")
            return
        
        self.logger.info("=" * 60)
        self.logger.info("DIVIDE AND CONQUER WORKFLOW STARTED")
        self.logger.info("=" * 60)
        
        # Determine which steps to run
        if resume_from:
            resume_from = resume_from.lower()
            if resume_from not in steps:
                raise WorkflowError(f"Invalid resume step: {resume_from}")
            
            self.logger.info(f"Resuming workflow from {resume_from}")
            
            # Load previous state if resuming (placeholder for future enhancement)
            self._load_previous_state(resume_from)
            
            # Run from resume point
            step_keys = list(steps.keys())
            start_idx = step_keys.index(resume_from)
            steps_to_run = step_keys[start_idx:]
        else:
            steps_to_run = list(steps.keys())
        
        # Execute workflow steps
        try:
            for step_name in steps_to_run:
                self.logger.info(f"\n{'-' * 20} {step_name.upper()} {'-' * 20}")
                steps[step_name]()
                self._save_checkpoint(step_name)
            
            # Final summary
            self._log_final_summary()
            
        except Exception as e:
            self.logger.error(f"Workflow failed at step: {step_name if 'step_name' in locals() else 'unknown'}")
            self.logger.error(f"Error: {e}")
            raise
    
    def _validate_configuration(self) -> None:
        """Validate workflow configuration."""
        # Validate input file
        a3m_file = self.config['input']['a3m_file']
        if not os.path.exists(a3m_file):
            raise WorkflowError(f"Input A3M file not found: {a3m_file}")
        
        # Validate FastTree binary
        fasttree_binary = self.config['input']['fasttree_binary']
        if not os.path.exists(fasttree_binary):
            raise WorkflowError(f"FastTree binary not found: {fasttree_binary}")
        
        # Validate structure analysis config if specified
        if 'structure_analysis' in self.config:
            config_file = self.config['structure_analysis'].get('config_file')
            if config_file and not os.path.exists(config_file):
                raise WorkflowError(f"Structure analysis config not found: {config_file}")
        
        self.logger.info("All configuration validations passed")
    
    def _load_previous_state(self, resume_step: str) -> None:
        """Load previous workflow state for resuming."""
        self.logger.info(f"Loading previous state for {resume_step}")
        
        # Use working_dir directly as specified in config
        working_dir = self.config.get('output', {}).get('working_dir', '.')
        
        if resume_step in ['step2', 'step3', 'step4', 'step5']:
            # Load clade directories
            clades_dir = os.path.join(working_dir, 'clades')
            if os.path.exists(clades_dir):
                self.clade_dirs = []
                for item in os.listdir(clades_dir):
                    item_path = os.path.join(clades_dir, item)
                    if os.path.isdir(item_path) and item.startswith('clade_'):
                        self.clade_dirs.append(item_path)
                    elif item.endswith('.a3m') and item.startswith('clade_'):
                        clade_name = os.path.splitext(item)[0]
                        clade_dir_path = os.path.join(clades_dir, clade_name)
                        # Skip if directory already exists (already processed)
                        if os.path.isdir(clade_dir_path):
                            if clade_dir_path not in self.clade_dirs:
                                self.clade_dirs.append(clade_dir_path)
                            continue
                        os.makedirs(clade_dir_path)
                        import shutil
                        src_path = os.path.join(clades_dir, item)
                        dst_path = os.path.join(clade_dir_path, item)
                        shutil.copy2(src_path, dst_path)
                        self.clade_dirs.append(clade_dir_path)
                self.clade_dirs.sort()
                self.logger.info(f"Loaded {len(self.clade_dirs)} clade directories")
                
                # Load query information from preprocessed file
                a3m_file = self.config['input']['a3m_file']
                file_stem = os.path.splitext(os.path.basename(a3m_file))[0]
                preprocessed_file = os.path.join(working_dir, f"{file_stem}_preprocessed.a3m")
                
                if os.path.exists(preprocessed_file):
                    # Read first sequence as query
                    with open(preprocessed_file, 'r') as f:
                        lines = f.readlines()
                        if len(lines) >= 2:
                            self.query_header = lines[0].strip()[1:]  # Remove '>'
                            self.query_sequence = lines[1].strip()
                            # Remove lowercase letters from query sequence
                            self.query_sequence = ''.join(c for c in self.query_sequence if c.isupper() or c == '-')
                            self.logger.info(f"Loaded query: {self.query_header}")
                        else:
                            raise WorkflowError(f"Invalid preprocessed file: {preprocessed_file}")
                else:
                    raise WorkflowError(f"Preprocessed file not found: {preprocessed_file}")
            else:
                raise WorkflowError(f"Clades directory not found: {clades_dir}")
        
        if resume_step in ['step3', 'step4', 'step5']:
            # Load shuffle directories
            shuffle_dirs = []
            for clade_dir in self.clade_dirs:
                if os.path.exists(clade_dir):
                    for item in os.listdir(clade_dir):
                        item_path = os.path.join(clade_dir, item)
                        if os.path.isdir(item_path) and item.startswith('shuffle_'):
                            shuffle_dirs.append(item_path)
            self.shuffle_dirs = sorted(shuffle_dirs)
            if self.shuffle_dirs:
                self.logger.info(f"Loaded {len(self.shuffle_dirs)} shuffle directories")
        
        if resume_step in ['step4', 'step5']:
            # Check for analysis results
            results_file = os.path.join(working_dir, "structure_analysis_results.csv")
            if os.path.exists(results_file):
                import pandas as pd
                self.analysis_results = pd.read_csv(results_file)
                self.logger.info(f"Loaded analysis results: {len(self.analysis_results)} structures")
    
    def _save_checkpoint(self, step_name: str) -> None:
        """Save workflow checkpoint (placeholder)."""
        # This is a placeholder for future enhancement
        self.logger.debug(f"Checkpoint saved for {step_name}")
    
    def _log_final_summary(self) -> None:
        """Log final workflow summary."""
        total_time = time.time() - self.start_time
        
        self.logger.info("=" * 60)
        self.logger.info("WORKFLOW COMPLETED SUCCESSFULLY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total execution time: {total_time/3600:.2f} hours")
        
        if self.clade_dirs:
            self.logger.info(f"Clades generated: {len(self.clade_dirs)}")
        if self.shuffle_dirs:
            self.logger.info(f"Shuffle directories: {len(self.shuffle_dirs)}")
        if self.analysis_results is not None:
            self.logger.info(f"Structures analyzed: {len(self.analysis_results)}")
        
        self.logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Divide-and-conquer phylogenetic workflow for structure prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config', 
        default='config.yaml',
        help='Configuration file path (default: config.yaml)'
    )
    
    parser.add_argument(
        '--resume-from',
        choices=['step1', 'step2', 'step3', 'step4', 'step5'],
        help='Resume workflow from specific step'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration without running workflow'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize and run workflow
        orchestrator = WorkflowOrchestrator(args.config)
        orchestrator.run_complete_workflow(
            resume_from=args.resume_from,
            dry_run=args.dry_run
        )
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except WorkflowError as e:
        print(f"Workflow Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nWorkflow interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()