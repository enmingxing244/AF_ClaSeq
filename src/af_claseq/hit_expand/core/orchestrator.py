#!/usr/bin/env python3
"""
Master pipeline orchestrator for MSA clustering and structure-guided optimization.
Coordinates all pipeline steps from clustering to final MSA generation.
"""

import logging
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

from ..core.sequence_io import parse_sequence_file, write_a3m_file, SequenceFilter
from ..core.subset_generator import MSASubsetGenerator, BatchOrganizer
from ..core.similarity_search import SimilaritySearchPipeline
from ..structure.analyzer import MSAStructureAnalyzer, HitIdentifier
from ..jobs.slurm_manager import SlurmJobManager, SlurmJobConfig, JobState
from ..config.settings import PipelineConfig
from ..config.utils import load_structure_config, ConfigurationError
from ..visualization.plotting import create_structure_analysis_plots

logger = logging.getLogger(__name__)


@dataclass
class PipelineState:
    """Tracks the current state of the pipeline execution."""
    step: str = "initialized"
    start_time: float = 0.0
    current_step_start: float = 0.0
    completed_steps: List[str] = None
    failed_steps: List[str] = None
    
    def __post_init__(self):
        if self.completed_steps is None:
            self.completed_steps = []
        if self.failed_steps is None:
            self.failed_steps = []


class PipelineError(Exception):
    """Raised when pipeline execution fails."""
    pass


class MSAOptimizationPipeline:
    """Main pipeline orchestrator for MSA clustering and structure-guided optimization."""
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize pipeline orchestrator.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.state = PipelineState()
        self.output_dir = Path(config.output_dir)
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._initialize_components()
        
        logger.info(f"Pipeline '{config.name}' initialized")
        logger.info(f"Output directory: {self.output_dir}")
    
    def _initialize_components(self) -> None:
        """Initialize pipeline components."""
        # Initialize components (clustering is now skipped)
        self.subset_generator = MSASubsetGenerator(self.config.subsets)
        self.batch_organizer = BatchOrganizer(self.config.batches.num_batches)
        self.similarity_searcher = SimilaritySearchPipeline(self.config.similarity_search)
        
        # SLURM manager will be initialized when needed
        self.job_manager: Optional[SlurmJobManager] = None
        
        logger.info("Pipeline components initialized (clustering is provided externally)")
    
    def run_full_pipeline(self, input_file: Optional[str] = None, source_msa_file: Optional[str] = None) -> Path:
        """
        Run the complete pipeline from start to finish.
        
        Args:
            input_file: Input clustered representative A3M file (provided externally)
            source_msa_file: Source MSA file for similarity search
            
        Returns:
            Path to final optimized MSA file
        """
        self.state.start_time = time.time()
        self.state.step = "starting"
        
        input_file = input_file or self.config.input_file
        if not input_file:
            raise PipelineError("No input file specified")
        
        input_path = Path(input_file)
        if not input_path.exists():
            raise PipelineError(f"Input file not found: {input_path}")
        
        # Store source MSA file for similarity search and query extraction
        self.source_msa_file = source_msa_file
        if not source_msa_file:
            raise PipelineError("Source MSA file is required for query sequence extraction")
        if not Path(source_msa_file).exists():
            raise PipelineError(f"Source MSA file not found: {source_msa_file}")
        
        logger.info(f"Starting pipeline with clustered representatives: {input_path}")
        if source_msa_file:
            logger.info(f"Source MSA for similarity search: {source_msa_file}")
        
        try:
            # Clustering is skipped - input is already clustered representatives
            logger.info("Using externally provided clustered representatives")
            clustered_file = input_path
            
            # Step 1: ALWAYS check for existing batch folders first (safety measure)
            batch_dir = self._check_and_handle_batch_folders(clustered_file)
            
            # Step 4: Structure analysis (optional)
            if self.config.skip_structure_analysis:
                logger.info("Skipping structure analysis step")
                hit_a3m_files = []
            else:
                hit_a3m_files = self._run_structure_analysis_step(batch_dir)
            
            # Step 5: Hit expansion (optional)
            if self.config.skip_hit_expansion or not hit_a3m_files:
                logger.info("Skipping hit expansion step")
                final_msa_file = clustered_file
            else:
                final_msa_file = self._run_hit_expansion_step(hit_a3m_files)
                
                # Step 6: Second round structure prediction with expanded sequences (optional)
                if self._should_run_second_round(final_msa_file):
                    self._run_second_round_structure_prediction(final_msa_file)
            
            # Finalize pipeline
            final_output = self._finalize_pipeline(final_msa_file)
            
            total_time = time.time() - self.state.start_time
            logger.info(f"Pipeline completed successfully in {total_time:.1f}s")
            logger.info(f"Final output: {final_output}")
            
            return final_output
            
        except Exception as e:
            self.state.failed_steps.append(self.state.step)
            logger.error(f"Pipeline failed at step '{self.state.step}': {e}")
            raise PipelineError(f"Pipeline execution failed: {e}")
    
    
    def _run_subset_to_batch_step(self, clustered_file: Path) -> Path:
        """Generate subsets directly into multiple batch folders."""
        self._start_step("subset_to_batch")
        
        logger.info("Step 1: Generating subsets directly into batch folders")
        
        # Create batches directory in 00_initial_prediction/
        initial_dir = self.output_dir / "00_initial_prediction"
        batches_dir = initial_dir / "batches"
        batches_dir.mkdir(parents=True, exist_ok=True)
        
        # Create batch subdirectories
        batch_folders = []
        for i in range(1, self.config.batches.num_batches + 1):
            batch_folder = batches_dir / f"{self.config.batches.batch_prefix}_{i:02d}"
            batch_folder.mkdir(exist_ok=True)
            batch_folders.append(batch_folder)
        
        # Generate subsets directly into batch folders (round-robin distribution)
        subset_files = self._generate_subsets_to_batches(clustered_file, batch_folders)
        
        logger.info(f"Generated {len(subset_files)} subset files distributed into {len(batch_folders)} batch folders")
        
        self._complete_step("subset_to_batch")
        return batches_dir
    
    def _generate_subsets_to_batches(self, clustered_file: Path, batch_folders: List[Path]) -> List[Path]:
        """Generate subsets directly into batch folders using round-robin distribution."""
        from ..core.sequence_io import parse_sequence_file, write_a3m_file
        import random
        
        # Parse clustered representatives file
        sequences, headers = parse_sequence_file(clustered_file)
        
        if not sequences:
            raise PipelineError(f"No sequences found in {clustered_file}")
        
        # Get query sequence from source MSA file (first sequence)
        query_sequence, query_header = self._get_query_from_source_msa()
        
        # Use all clustered representatives as the sampling pool
        pool_sequences = sequences
        pool_headers = headers
        
        if len(pool_sequences) < self.config.subsets.num_random_sequences:
            raise PipelineError(
                f"Not enough sequences for subset generation: "
                f"need {self.config.subsets.num_random_sequences}, got {len(pool_sequences)}"
            )
        
        # Set random seed for reproducibility
        random.seed(self.config.subsets.random_seed)
        
        # Generate subsets and distribute to batch folders
        subset_files = []
        num_batches = len(batch_folders)
        
        for subset_idx in range(self.config.subsets.num_subsets):
            # Select random sequences
            random_indices = random.sample(range(len(pool_sequences)), 
                                         self.config.subsets.num_random_sequences)
            selected_sequences = [pool_sequences[i] for i in random_indices]
            selected_headers = [pool_headers[i] for i in random_indices]
            
            # Combine query with selected sequences
            subset_sequences = [query_sequence] + selected_sequences
            subset_headers = [query_header] + selected_headers
            
            # Determine which batch folder to use (round-robin)
            batch_idx = subset_idx % num_batches
            batch_folder = batch_folders[batch_idx]
            
            # Create subset file
            subset_filename = f"{self.config.subsets.output_prefix}_{subset_idx+1:05d}.a3m"
            subset_file = batch_folder / subset_filename
            
            # Write subset file
            write_a3m_file(subset_sequences, subset_headers, subset_file)
            subset_files.append(subset_file)
        
        logger.info(f"Distributed {len(subset_files)} subsets across {num_batches} batch folders")
        return subset_files
    
    def _get_query_from_source_msa(self) -> Tuple[str, str]:
        """
        Get the query sequence from the source MSA file.
        
        Returns:
            Tuple of (query_sequence, query_header)
            
        Raises:
            PipelineError: If source MSA file is not available or has no sequences
        """
        if not self.source_msa_file:
            raise PipelineError("Source MSA file is required for query sequence extraction")
        
        from ..core.sequence_io import parse_sequence_file
        
        # Parse source MSA file
        sequences, headers = parse_sequence_file(self.source_msa_file)
        
        if not sequences:
            raise PipelineError(f"No sequences found in source MSA file: {self.source_msa_file}")
        
        # First sequence is the query
        query_sequence = sequences[0]
        query_header = headers[0]
        
        logger.info(f"Using query sequence from source MSA: {query_header}")
        return query_sequence, query_header
    
    def _check_and_handle_batch_folders(self, clustered_file: Path) -> Path:
        """
        ALWAYS check for existing batch folders first to prevent accidental overwriting.
        This is a safety measure that protects existing work regardless of skip settings.
        
        Args:
            clustered_file: Path to clustered representatives file (only used if creating new batches)
            
        Returns:
            Path to batches directory
            
        Raises:
            PipelineError: If structure prediction is needed but cannot proceed
        """
        # Updated path: batches now go in 00_initial_prediction/
        initial_dir = self.output_dir / "00_initial_prediction"
        batches_dir = initial_dir / "batches"
        
        # Check if batches directory exists and has content
        if batches_dir.exists():
            # Check for any subdirectories or files
            existing_items = list(batches_dir.iterdir())
            
            if existing_items:
                # Count batch folders and files
                batch_folders = [d for d in existing_items if d.is_dir() and d.name.startswith(self.config.batches.batch_prefix)]
                other_items = [item for item in existing_items if not (item.is_dir() and item.name.startswith(self.config.batches.batch_prefix))]
                
                total_a3m_files = 0
                total_pdb_files = 0
                for batch_folder in batch_folders:
                    total_a3m_files += len(list(batch_folder.glob("*.a3m")))
                    total_pdb_files += len(list(batch_folder.glob("*.pdb")))
                
                logger.warning("🚨 EXISTING BATCH CONTENT DETECTED 🚨")
                logger.warning(f"Found existing content in {batches_dir}:")
                logger.warning(f"  - {len(batch_folders)} batch folders")
                logger.warning(f"  - {total_a3m_files} A3M files")
                logger.warning(f"  - {total_pdb_files} PDB files")
                if other_items:
                    logger.warning(f"  - {len(other_items)} other items")
                
                logger.warning("🛡️  PROTECTION MODE: Will NOT overwrite existing batch content")
                logger.warning("   Skipping subset generation and batch creation")
                
                # Determine what to do about structure prediction
                if not self.config.skip_structure_prediction:
                    if total_pdb_files > 0:
                        logger.warning("   Found existing PDB files - skipping structure prediction automatically")
                        logger.warning("   (Set skip_structure_prediction: true to suppress this warning)")
                    else:
                        logger.warning("   No PDB files found - structure prediction may be needed")
                        logger.warning("   but cannot proceed due to existing A3M files")
                        raise PipelineError(
                            f"Existing batch folders found but no PDB structures detected. "
                            f"Cannot safely proceed with structure prediction as it would overwrite existing A3M files. "
                            f"Either: 1) Set skip_structure_prediction: true, or 2) Remove/backup {batches_dir}"
                        )
                
                logger.info("✅ Proceeding with existing batch folders (completely untouched)")
                return batches_dir
        
        # No existing content - safe to create new batches
        if self.config.skip_structure_prediction:
            raise PipelineError(
                f"skip_structure_prediction is true but no existing batch folders found at {batches_dir}. "
                f"Cannot skip structure prediction without existing batches."
            )
        
        # Create new batches and run structure prediction
        logger.info("📁 No existing batch content found - creating new batches")
        # Ensure initial_dir exists
        initial_dir.mkdir(parents=True, exist_ok=True)
        batch_dir = self._run_subset_to_batch_step(clustered_file)
        self._run_structure_prediction_step(batch_dir)
        return batch_dir
    
    def _run_structure_prediction_step(self, batch_dir: Path) -> None:
        """Run structure prediction step using SLURM."""
        self._start_step("structure_prediction")
        
        logger.info("Step 2: Checking structure prediction job status")
        
        # Check if all ColabFold jobs are already complete (if enabled)
        if self.config.check_existing_jobs:
            if self._are_colabfold_jobs_complete(batch_dir):
                logger.info("All ColabFold jobs are already complete. Skipping job submission.")
                self._complete_step("structure_prediction")
                return
        
        logger.info("Some ColabFold jobs are incomplete. Submitting new jobs.")
        
        # Initialize SLURM manager
        slurm_config = SlurmJobConfig(
            account=self.config.slurm.account,
            partition=self.config.slurm.partition,
            time_limit=self.config.slurm.time_limit,
            memory=self.config.slurm.memory,
            cpus_per_task=self.config.slurm.cpus_per_task,
            gres=self.config.slurm.gres,
            conda_env_path=self.config.slurm.conda_env_path
        )
        
        self.job_manager = SlurmJobManager(slurm_config)
        
        # Submit jobs for each batch directory
        job_ids = self.job_manager.submit_batch_jobs(
            batch_dir, 
            job_prefix=f"{self.config.name}_colabfold",
            delay_between_jobs=self.config.slurm.delay_between_jobs
        )
        
        logger.info(f"Submitted {len(job_ids)} ColabFold jobs")
        
        # Monitor jobs if requested
        if self.config.monitor_jobs:
            logger.info("Monitoring job completion...")
            final_states = self.job_manager.wait_for_jobs(
                job_ids,
                check_interval=self.config.job_check_interval,
                timeout=self.config.job_timeout
            )
            
            # Check job results
            completed_jobs = sum(1 for state in final_states.values() 
                               if state == JobState.COMPLETED)
            failed_jobs = sum(1 for state in final_states.values() 
                            if state == JobState.FAILED)
            
            logger.info(f"Job completion: {completed_jobs} completed, {failed_jobs} failed")
            
            if failed_jobs > 0:
                logger.warning(f"{failed_jobs} jobs failed - results may be incomplete")
        
        # Save job information
        job_info_file = self.output_dir / "job_summary.json"
        if self.job_manager:
            self.job_manager.save_job_info(job_info_file)
        
        self._complete_step("structure_prediction")
    
    def _run_structure_analysis_step(self, batch_dir: Path) -> List[Path]:
        """Run structure analysis step."""
        self._start_step("structure_analysis")
        
        logger.info("Step 5: Analyzing predicted structures")
        
        # Load structure analysis configuration using standardized utilities
        try:
            analysis_config = load_structure_config(self.config)
        except ConfigurationError as e:
            raise PipelineError(f"Failed to load structure analysis configuration: {e}")
        
        # Initialize analyzer
        analyzer = MSAStructureAnalyzer(analysis_config)
        
        # Find predicted structures
        pdb_files = analyzer.find_predicted_structures(batch_dir)
        
        if not pdb_files:
            logger.warning("No predicted structures found")
            self._complete_step("structure_analysis")
            return []
        
        # Analyze structures
        results_df = analyzer.analyze_structures_parallel(pdb_files)
        
        if results_df.empty:
            logger.warning("No structures successfully analyzed")
            self._complete_step("structure_analysis")
            return []
        
        # Save analysis results in 00_initial_prediction/
        initial_dir = self.output_dir / "00_initial_prediction"
        results_file = initial_dir / "analysis_results.csv"
        results_df.to_csv(results_file, index=False)
        
        # Create visualization plots in 00_initial_prediction/plots/
        plots_dir = initial_dir / "plots"
        self._create_analysis_plots(results_file, analysis_config, plots_dir)
        
        # Get best structures using user-specified criteria and thresholds
        best_structures_df = analyzer.get_best_structures(
            results_df,  # Use all results, filtering is done in the method
            top_n=10
        )
        
        # Identify corresponding A3M files
        hit_a3m_files = HitIdentifier.identify_hit_a3m_files(best_structures_df)
        
        logger.info(f"Identified {len(hit_a3m_files)} hit A3M files from structure analysis")
        
        # Save hit information in 00_initial_prediction/
        hit_info_file = initial_dir / "hit_structures.csv"
        best_structures_df.to_csv(hit_info_file, index=False)
        
        self._complete_step("structure_analysis")
        return hit_a3m_files
    
    def _run_hit_expansion_step(self, hit_a3m_files: List[Path]) -> Path:
        """Run hit expansion step using BLOSUM62 similarity search."""
        self._start_step("hit_expansion")
        
        logger.info("Step 5: Expanding hits with similar sequences using BLOSUM62")
        
        if not self.source_msa_file:
            logger.warning("No source MSA file provided for similarity search")
            # Return the first hit file if available
            if hit_a3m_files:
                self._complete_step("hit_expansion")
                return hit_a3m_files[0]
            else:
                raise PipelineError("No hit files available and no source MSA provided")
        
        # Create hit_sequences directory in 01_hit_expansion/
        expansion_dir = self.output_dir / "01_hit_expansion"
        expansion_dir.mkdir(parents=True, exist_ok=True)
        
        hit_dir = expansion_dir / "hit_sequences"
        hit_dir.mkdir(exist_ok=True)
        
        # Copy hit A3M files to hit directory
        for hit_file in hit_a3m_files:
            if hit_file.exists():
                dst_file = hit_dir / hit_file.name
                dst_file.write_text(hit_file.read_text())
        
        # Run BLOSUM62-based similarity search - save to expanded_msa/
        expanded_msa_dir = expansion_dir / "expanded_msa"
        expanded_msa_dir.mkdir(exist_ok=True)
        expanded_msa_file = expanded_msa_dir / "expanded_hit_sequences.a3m"
        
        self.similarity_searcher.search_hits_in_source_msa(
            hit_dir, self.source_msa_file, expanded_msa_file
        )
        
        logger.info(f"Hit expansion complete: {expanded_msa_file}")
        
        self._complete_step("hit_expansion")
        return expanded_msa_file
    
    def _should_run_second_round(self, final_msa_file: Path) -> bool:
        """
        Determine if we should run the second round of structure prediction.
        
        Args:
            final_msa_file: Path to the final MSA file from hit expansion
            
        Returns:
            True if second round should be run, False otherwise
        """
        # Only run second round if hit expansion was actually performed
        # and the final MSA file is the expanded sequences
        expanded_file = self.output_dir / "01_hit_expansion" / "expanded_msa" / "expanded_hit_sequences.a3m"
        
        if final_msa_file == expanded_file and expanded_file.exists():
            logger.info("Hit expansion was performed - second round structure prediction enabled")
            return True
        else:
            logger.info("Hit expansion was skipped - no second round structure prediction")
            return False
    
    def _run_second_round_structure_prediction(self, expanded_msa_file: Path) -> None:
        """
        Run second round of structure prediction using expanded hit sequences.
        
        This step:
        1. Creates batch_after_expand/ folder
        2. Generates subsets from expanded_hit_sequences.a3m (350 sequences)
        3. Runs structure prediction on new subsets
        4. Runs af_claseq structure analysis
        5. Creates visualization plots
        6. Stops (no further hit expansion)
        
        Args:
            expanded_msa_file: Path to expanded hit sequences A3M file
        """
        self._start_step("second_round_structure_prediction")
        
        logger.info("🔄 Step 6: Second round structure prediction with expanded sequences")
        logger.info(f"Input: {expanded_msa_file} (expanded hit sequences)")
        
        # Step 6a: Create batch_after_expand directory and generate subsets
        batch_after_expand_dir = self._run_expanded_subset_to_batch_step(expanded_msa_file)
        
        # Step 6b: Run structure prediction on new batches
        self._run_expanded_structure_prediction_step(batch_after_expand_dir)
        
        # Step 6c: Run structure analysis on new predictions
        expanded_hit_a3m_files = self._run_expanded_structure_analysis_step(batch_after_expand_dir)
        
        logger.info("🛑 Second round complete - stopping pipeline (no further hit expansion)")
        self._complete_step("second_round_structure_prediction")
    
    def _run_expanded_subset_to_batch_step(self, expanded_msa_file: Path) -> Path:
        """Generate subsets from expanded sequences directly into batch_after_expand folders."""
        logger.info("Step 6a: Generating subsets from expanded sequences into batch_after_expand")
        
        # Create batch_after_expand directory in 01_hit_expansion/
        expansion_dir = self.output_dir / "01_hit_expansion"
        batch_after_expand_dir = expansion_dir / "batch_after_expand"
        batch_after_expand_dir.mkdir(parents=True, exist_ok=True)
        
        # Create batch subdirectories
        batch_folders = []
        for i in range(1, self.config.batches.num_batches + 1):
            batch_folder = batch_after_expand_dir / f"{self.config.batches.batch_prefix}_{i:02d}"
            batch_folder.mkdir(exist_ok=True)
            batch_folders.append(batch_folder)
        
        # Generate subsets from expanded sequences
        subset_files = self._generate_expanded_subsets_to_batches(expanded_msa_file, batch_folders)
        
        logger.info(f"Generated {len(subset_files)} subset files from expanded sequences")
        logger.info(f"Distributed into {len(batch_folders)} batch folders in batch_after_expand/")
        
        return batch_after_expand_dir
    
    def _generate_expanded_subsets_to_batches(self, expanded_msa_file: Path, batch_folders: List[Path]) -> List[Path]:
        """Generate subsets from expanded sequences using same logic as original subset generation."""
        from ..core.sequence_io import parse_sequence_file, write_a3m_file
        import random
        
        # Parse expanded MSA file (350 sequences)
        sequences, headers = parse_sequence_file(expanded_msa_file)
        
        if not sequences:
            raise PipelineError(f"No sequences found in {expanded_msa_file}")
        
        logger.info(f"Loaded {len(sequences)} expanded sequences for subset generation")
        
        # Extract query sequence (first sequence in expanded MSA)
        query_sequence = sequences[0]
        query_header = headers[0]
        
        # Use remaining sequences as the sampling pool (exclude query)
        pool_sequences = sequences[1:]
        pool_headers = headers[1:]
        
        if len(pool_sequences) < self.config.subsets.num_random_sequences:
            logger.warning(
                f"Not enough pool sequences for subset generation: "
                f"need {self.config.subsets.num_random_sequences}, got {len(pool_sequences)}. "
                f"Will use all available sequences with replacement if needed."
            )
        
        # Set random seed for reproducibility
        random.seed(self.config.subsets.random_seed)
        
        # Generate subsets and distribute to batch folders
        subset_files = []
        num_batches = len(batch_folders)
        
        for subset_idx in range(self.config.subsets.num_subsets):
            # Select random sequences (with replacement if necessary)
            if len(pool_sequences) >= self.config.subsets.num_random_sequences:
                random_indices = random.sample(range(len(pool_sequences)), 
                                             self.config.subsets.num_random_sequences)
            else:
                # Use all sequences with replacement to reach target count
                random_indices = [random.randint(0, len(pool_sequences) - 1) 
                                for _ in range(self.config.subsets.num_random_sequences)]
            
            selected_sequences = [pool_sequences[i] for i in random_indices]
            selected_headers = [pool_headers[i] for i in random_indices]
            
            # Combine query with selected sequences (query always first)
            subset_sequences = [query_sequence] + selected_sequences
            subset_headers = [query_header] + selected_headers
            
            # Determine which batch folder to use (round-robin)
            batch_idx = subset_idx % num_batches
            batch_folder = batch_folders[batch_idx]
            
            # Create subset file
            subset_filename = f"{self.config.subsets.output_prefix}_{subset_idx+1:05d}.a3m"
            subset_file = batch_folder / subset_filename
            
            # Write subset file
            write_a3m_file(subset_sequences, subset_headers, subset_file)
            subset_files.append(subset_file)
        
        logger.info(f"Distributed {len(subset_files)} subsets across {num_batches} batch folders in batch_after_expand/")
        return subset_files
    
    def _run_expanded_structure_prediction_step(self, batch_after_expand_dir: Path) -> None:
        """Run structure prediction on batch_after_expand subsets."""
        logger.info("Step 6b: Running structure prediction on expanded batch subsets")
        
        # Reuse existing structure prediction logic but with new batch directory
        self._start_step("expanded_structure_prediction")
        
        # Check if jobs are already complete
        if self.config.check_existing_jobs:
            if self._are_colabfold_jobs_complete(batch_after_expand_dir):
                logger.info("All expanded ColabFold jobs are already complete. Skipping job submission.")
                self._complete_step("expanded_structure_prediction")
                return
        
        logger.info("Some expanded ColabFold jobs are incomplete. Submitting new jobs.")
        
        # Initialize SLURM manager if not already done
        if not self.job_manager:
            slurm_config = SlurmJobConfig(
                account=self.config.slurm.account,
                partition=self.config.slurm.partition,
                time_limit=self.config.slurm.time_limit,
                memory=self.config.slurm.memory,
                cpus_per_task=self.config.slurm.cpus_per_task,
                gres=self.config.slurm.gres,
                conda_env_path=self.config.slurm.conda_env_path
            )
            self.job_manager = SlurmJobManager(slurm_config)
        
        # Submit jobs for expanded batch directories
        job_ids = self.job_manager.submit_batch_jobs(
            batch_after_expand_dir, 
            job_prefix=f"{self.config.name}_expanded_colabfold",
            delay_between_jobs=self.config.slurm.delay_between_jobs
        )
        
        logger.info(f"Submitted {len(job_ids)} expanded ColabFold jobs")
        
        # Monitor jobs if requested
        if self.config.monitor_jobs:
            logger.info("Monitoring expanded job completion...")
            final_states = self.job_manager.wait_for_jobs(
                job_ids,
                check_interval=self.config.job_check_interval,
                timeout=self.config.job_timeout
            )
            
            # Check job results
            completed_jobs = sum(1 for state in final_states.values() 
                               if state == JobState.COMPLETED)
            failed_jobs = sum(1 for state in final_states.values() 
                            if state == JobState.FAILED)
            
            logger.info(f"Expanded job completion: {completed_jobs} completed, {failed_jobs} failed")
            
            if failed_jobs > 0:
                logger.warning(f"{failed_jobs} expanded jobs failed - results may be incomplete")
        
        self._complete_step("expanded_structure_prediction")
    
    def _run_expanded_structure_analysis_step(self, batch_after_expand_dir: Path) -> List[Path]:
        """Run structure analysis on expanded predictions."""
        logger.info("Step 6c: Analyzing expanded predicted structures")
        
        self._start_step("expanded_structure_analysis")
        
        # Load structure analysis configuration using standardized utilities
        try:
            analysis_config = load_structure_config(self.config)
        except ConfigurationError as e:
            raise PipelineError(f"Failed to load structure analysis configuration: {e}")
        
        # Initialize analyzer
        analyzer = MSAStructureAnalyzer(analysis_config)
        
        # Find predicted structures in expanded batch directory
        pdb_files = analyzer.find_predicted_structures(batch_after_expand_dir)
        
        if not pdb_files:
            logger.warning("No expanded predicted structures found")
            self._complete_step("expanded_structure_analysis")
            return []
        
        # Analyze expanded structures
        results_df = analyzer.analyze_structures_parallel(pdb_files)
        
        if results_df.empty:
            logger.warning("No expanded structures successfully analyzed")
            self._complete_step("expanded_structure_analysis")
            return []
        
        # Save expanded analysis results in 01_hit_expansion/expanded_analysis/
        expansion_dir = self.output_dir / "01_hit_expansion"
        expanded_analysis_dir = expansion_dir / "expanded_analysis"
        expanded_analysis_dir.mkdir(parents=True, exist_ok=True)
        
        expanded_results_file = expanded_analysis_dir / "expanded_structure_analysis_results.csv"
        results_df.to_csv(expanded_results_file, index=False)
        
        # Create expanded visualization plots in 01_hit_expansion/plots/
        expanded_plots_dir = expansion_dir / "plots"
        self._create_expanded_analysis_plots(expanded_results_file, analysis_config, expanded_plots_dir)
        
        # Get best expanded structures
        best_structures_df = analyzer.get_best_structures(
            results_df,
            top_n=10
        )
        
        # Identify corresponding A3M files (for potential future use, but we won't expand further)
        hit_a3m_files = HitIdentifier.identify_hit_a3m_files(best_structures_df)
        
        logger.info(f"Identified {len(hit_a3m_files)} expanded hit A3M files (but stopping here)")
        
        # Save expanded hit information in 01_hit_expansion/expanded_analysis/
        expanded_hit_info_file = expanded_analysis_dir / "expanded_hit_structures.csv"
        best_structures_df.to_csv(expanded_hit_info_file, index=False)
        
        self._complete_step("expanded_structure_analysis")
        return hit_a3m_files
    
    def _create_expanded_analysis_plots(self, expanded_results_file: Path, analysis_config, expanded_plots_dir: Path) -> None:
        """Create visualization plots for expanded structure analysis results."""
        logger.info("Creating expanded structure analysis visualization plots")
        
        try:
            # Create expanded plots directory
            expanded_plots_dir.mkdir(parents=True, exist_ok=True)
            
            # Create all plots for expanded results
            saved_plots = create_structure_analysis_plots(
                results_file=expanded_results_file,
                output_dir=expanded_plots_dir,
                use_af_claseq=True,
                filter_criteria_threshold=analysis_config.filter_criteria_threshold,
                plddt_threshold=analysis_config.plddt_threshold
            )
            
            logger.info(f"Created expanded structure analysis plots:")
            for plot_type, plot_path in saved_plots.items():
                if isinstance(plot_path, list):
                    logger.info(f"  {plot_type}: {len(plot_path)} plots")
                else:
                    logger.info(f"  {plot_type}: {plot_path}")
            
        except Exception as e:
            logger.error(f"Failed to create expanded analysis plots: {e}")
            # Continue pipeline execution even if plotting fails
    
    def _are_colabfold_jobs_complete(self, batch_dir: Path) -> bool:
        """
        Check if all ColabFold jobs are complete by looking for output PDB files.
        
        Args:
            batch_dir: Directory containing batch folders
            
        Returns:
            True if all expected PDB files exist, False otherwise
        """
        logger.info("Checking ColabFold job completion status...")
        
        # Get all batch folders
        batch_folders = [d for d in batch_dir.iterdir() if d.is_dir() and d.name.startswith('batch_')]
        
        if not batch_folders:
            logger.warning("No batch folders found")
            return False
        
        total_subsets = 0
        completed_subsets = 0
        
        for batch_folder in batch_folders:
            # Get all subset files in this batch
            subset_files = list(batch_folder.glob("subset_*.a3m"))
            
            for subset_file in subset_files:
                total_subsets += 1
                
                # Check if corresponding PDB file exists
                subset_name = subset_file.stem  # e.g., "subset_00001"
                expected_pdb = batch_folder / f"{subset_name}_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb"
                
                if expected_pdb.exists():
                    completed_subsets += 1
                else:
                    logger.debug(f"Missing PDB file: {expected_pdb}")
        
        completion_rate = completed_subsets / total_subsets if total_subsets > 0 else 0
        
        logger.info(f"ColabFold completion status: {completed_subsets}/{total_subsets} ({completion_rate:.1%})")
        
        # Consider jobs complete if all expected PDB files exist
        if completed_subsets == total_subsets and total_subsets > 0:
            logger.info("All ColabFold jobs are complete!")
            return True
        else:
            logger.info(f"Missing {total_subsets - completed_subsets} PDB files")
            return False
    
    def _create_analysis_plots(self, results_file: Path, analysis_config, plots_dir: Path) -> None:
        """Create visualization plots for structure analysis results."""
        logger.info("Creating structure analysis visualization plots")
        
        try:
            # Create plots directory
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            # Create all plots using af_claseq plotting utilities and config thresholds
            saved_plots = create_structure_analysis_plots(
                results_file=results_file,
                output_dir=plots_dir,
                use_af_claseq=True,
                filter_criteria_threshold=analysis_config.filter_criteria_threshold,
                plddt_threshold=analysis_config.plddt_threshold
            )
            
            logger.info(f"Created structure analysis plots:")
            for plot_type, plot_path in saved_plots.items():
                if isinstance(plot_path, list):
                    logger.info(f"  {plot_type}: {len(plot_path)} plots")
                else:
                    logger.info(f"  {plot_type}: {plot_path}")
            
        except Exception as e:
            logger.error(f"Failed to create analysis plots: {e}")
            # Continue pipeline execution even if plotting fails
    
    def _finalize_pipeline(self, final_msa_file: Path) -> Path:
        """Finalize pipeline and create final output."""
        self._start_step("finalization")
        
        # Copy final MSA to standard output location (keep at root level)
        final_output = self.output_dir / "final_optimized_msa.a3m"
        
        if final_msa_file != final_output:
            final_output.write_text(final_msa_file.read_text())
        
        # Create pipeline summary (keep at root level)
        summary = self._create_pipeline_summary()
        summary_file = self.output_dir / "pipeline_summary.json"
        
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self._complete_step("finalization")
        return final_output
    
    def _create_pipeline_summary(self) -> Dict[str, Any]:
        """Create pipeline execution summary."""
        total_time = time.time() - self.state.start_time
        
        summary = {
            "pipeline_name": self.config.name,
            "pipeline_version": self.config.version,
            "execution_time_seconds": total_time,
            "completed_steps": self.state.completed_steps,
            "failed_steps": self.state.failed_steps,
            "configuration": {
                "num_subsets": self.config.subsets.num_subsets,
                "num_batches": self.config.batches.num_batches,
                "plddt_threshold": self.config.structure_analysis.plddt_threshold,
                "filter_criteria_threshold": self.config.structure_analysis.filter_criteria_threshold,
                "similarity_threshold": self.config.similarity_search.similarity_threshold,
                "clustering": "External (provided by user)"
            },
            "output_files": {
                "final_msa": "final_optimized_msa.a3m",
                "initial_round": {
                    "structure_analysis": "00_initial_prediction/analysis_results.csv",
                    "hit_structures": "00_initial_prediction/hit_structures.csv",
                    "plots": "00_initial_prediction/plots/",
                    "batches": "00_initial_prediction/batches/"
                },
                "hit_expansion": {
                    "hit_sequences": "01_hit_expansion/hit_sequences/",
                    "expanded_msa": "01_hit_expansion/expanded_msa/expanded_hit_sequences.a3m",
                    "expanded_analysis": "01_hit_expansion/expanded_analysis/",
                    "expanded_plots": "01_hit_expansion/plots/",
                    "batch_after_expand": "01_hit_expansion/batch_after_expand/"
                },
                "job_summary": "job_summary.json"
            }
        }
        
        return summary
    
    def _start_step(self, step_name: str) -> None:
        """Mark the start of a pipeline step."""
        self.state.step = step_name
        self.state.current_step_start = time.time()
        logger.info(f"Starting step: {step_name}")
    
    def _complete_step(self, step_name: str) -> None:
        """Mark the completion of a pipeline step."""
        step_time = time.time() - self.state.current_step_start
        self.state.completed_steps.append(step_name)
        logger.info(f"Completed step '{step_name}' in {step_time:.1f}s")
    
    def cleanup(self) -> None:
        """Clean up pipeline resources."""
        if self.job_manager:
            # Save final job information
            job_info_file = self.output_dir / "final_job_summary.json"
            self.job_manager.save_job_info(job_info_file)
        
        logger.info("Pipeline cleanup completed")


def run_pipeline(config: PipelineConfig, input_file: Optional[str] = None, source_msa_file: Optional[str] = None) -> Path:
    """
    Convenience function to run the complete pipeline.
    
    Args:
        config: Pipeline configuration
        input_file: Input clustered representative A3M file (uses config if None)
        source_msa_file: Source MSA file for similarity search
        
    Returns:
        Path to final optimized MSA file
    """
    pipeline = MSAOptimizationPipeline(config)
    
    try:
        return pipeline.run_full_pipeline(input_file, source_msa_file)
    finally:
        pipeline.cleanup()