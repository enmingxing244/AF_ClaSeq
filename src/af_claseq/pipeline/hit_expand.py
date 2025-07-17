#!/usr/bin/env python3
"""
Hit Expand Pipeline Orchestrator.

This module orchestrates the complete hit expand workflow:
1. Sequence clustering with MMseqs2
2. Subset generation for structure prediction
3. Structure prediction job submission and monitoring
4. Structure analysis and filtering
5. Similarity search with BLOSUM62 to expand good sequences
6. Expanded subset generation from final MSA
"""

import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import shutil
import pandas as pd
from tqdm import tqdm

from af_claseq.utils.logging_utils import get_logger
from af_claseq.utils.slurm_utils import SlurmJobSubmitter
from af_claseq.utils.structure_analysis import StructureAnalyzer
from af_claseq.utils.sequence_processing import A3MParser, validate_a3m_file
from af_claseq.utils.plotting_manager import (
    plot_1d_distribution, 
    create_2d_scatter_plot,
    create_joint_plot
)

from af_claseq.modules.mmseqs_wrapper import MMseqsWrapper, MMseqsConfig
from af_claseq.modules.similarity_search import BLOSUM62SimilaritySearch, SimilaritySearchConfig
from af_claseq.modules.subset_generator import SubsetGenerator, SubsetConfig

from af_claseq.pipeline.config import HitExpandConfig

logger = get_logger(__name__)


class HitExpandError(Exception):
    """Raised when hit expand pipeline fails."""
    pass


class HitExpandRunner:
    """
    Main orchestrator for the hit expand pipeline.
    
    Coordinates all components of the hit expand workflow to transform
    an input MSA into expanded sequences ready for structure prediction.
    """
    
    def __init__(self,
                 config: HitExpandConfig,
                 slurm_submitter: SlurmJobSubmitter,
                 base_dir: Path,
                 config_file: str,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize hit expand runner.
        
        Args:
            config: Hit expand configuration
            slurm_submitter: SLURM job submitter instance
            base_dir: Base output directory
            config_file: Path to JSON config file for structure analysis
            logger: Optional logger instance
        """
        self.config = config
        self.slurm_submitter = slurm_submitter
        self.base_dir = Path(base_dir)
        self.config_file = config_file
        self.logger = logger or get_logger(__name__)
        
        # Create base directory
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._init_components()
        
        # Track workflow state
        self.workflow_state = {
            "clustering_completed": False,
            "similarity_search_completed": False,
            "subset_generation_completed": False,
            "structure_prediction_completed": False,
            "structure_analysis_completed": False,
            "expanded_subset_generation_completed": False
        }
        
        self.logger.info(f"Hit expand runner initialized for {base_dir}")
    
    def _init_components(self):
        """Initialize pipeline components."""
        # MMseqs2 wrapper
        mmseqs_config = MMseqsConfig(
            bin_path=self.config.mmseqs_bin,
            coverage=self.config.mmseqs_coverage,
            min_seq_id=self.config.mmseqs_min_seq_id,
            cov_mode=self.config.mmseqs_cov_mode,
            cluster_mode=self.config.mmseqs_cluster_mode,
            threads=self.config.mmseqs_threads,
            tmp_dir=self.config.mmseqs_tmp_dir
        )
        self.mmseqs_wrapper = MMseqsWrapper(mmseqs_config)
        
        # Similarity search
        similarity_config = SimilaritySearchConfig(
            top_k=self.config.similarity_top_k,
            similarity_threshold=self.config.similarity_threshold,
            exclude_query_headers=self.config.exclude_query_headers
        )
        self.similarity_search = BLOSUM62SimilaritySearch(similarity_config)
        
        # Subset generator
        subset_config = SubsetConfig(
            num_subsets=self.config.num_subsets,
            num_random_sequences=self.config.num_random_sequences,
            num_batches=self.config.num_batches,
            batch_prefix=self.config.batch_prefix,
            output_prefix=self.config.output_prefix,
            random_seed=self.config.random_seed
        )
        self.subset_generator = SubsetGenerator(subset_config)
        
        # Structure analyzer
        self.structure_analyzer = StructureAnalyzer()
        
        self.logger.info("All pipeline components initialized")
    
    def run(self) -> Optional[Path]:
        """
        Run the complete multi-round hit expand pipeline.
        
        Returns:
            Path to final MSA file if successful, None otherwise
        """
        try:
            self.logger.info("=== STARTING MULTI-ROUND HIT EXPAND PIPELINE ===")
            self.logger.info(f"Running {self.config.rounds} rounds of hit expansion")
            
            # Validate input MSA
            input_msa = Path(self.config.input_msa)
            if not validate_a3m_file(input_msa, strict=False):
                raise HitExpandError(f"Invalid input MSA: {input_msa}")
            
            # Stage 1: Clustering (done once, shared across all rounds)
            clustering_dir = self.base_dir / "01_clustering"
            if not self.config.skip_clustering:
                if self._check_done_file(clustering_dir, "01_clustering"):
                    self.logger.info("=== STAGE 1: CLUSTERING ALREADY COMPLETED - SKIPPING ===")
                    representative_sequences = self._load_clustered_sequences(clustering_dir)
                else:
                    representative_sequences = self._run_clustering_stage(input_msa)
            else:
                self.logger.info("Skipping clustering stage")
                parser = A3MParser(strict_validation=False)
                representative_sequences = parser.parse_file(input_msa)
            
            # Run multiple rounds
            final_msa_path = None
            for round_num in range(1, self.config.rounds + 1):
                self.logger.info(f"=== STARTING HIT EXPAND ROUND {round_num}/{self.config.rounds} ===")
                
                round_dir = self.base_dir / f"round_{round_num}"
                round_dir.mkdir(exist_ok=True)
                
                # Determine input for this round
                if round_num == 1:
                    # Round 1: Use clustered sequences
                    round_input_sequences = representative_sequences
                    round_input_type = "clustered"
                else:
                    # Round 2+: Use previous round's expanded subsets
                    prev_round_dir = self.base_dir / f"round_{round_num - 1}"
                    expanded_subsets_dir = prev_round_dir / "06_expanded_subsets"
                    
                    if not expanded_subsets_dir.exists():
                        self.logger.warning(f"No expanded subsets found from round {round_num - 1}. Ending at round {round_num - 1}")
                        break
                    
                    round_input_sequences = self._load_expanded_subsets(expanded_subsets_dir)
                    round_input_type = "expanded"
                
                # Run the round
                round_msa = self._run_single_round(
                    round_num=round_num,
                    round_dir=round_dir,
                    input_sequences=round_input_sequences,
                    input_type=round_input_type
                )
                
                if round_msa:
                    final_msa_path = round_msa
                    
                    # Check if we found new sequences in this round
                    if round_num > 1:
                        new_sequences_found = self._check_new_sequences_found(round_dir)
                        if not new_sequences_found:
                            self.logger.info(f"No new sequences found in round {round_num}. Stopping early.")
                            break
                else:
                    self.logger.error(f"Round {round_num} failed")
                    break
            
            self.logger.info(f"=== MULTI-ROUND HIT EXPAND PIPELINE COMPLETED ===")
            self.logger.info(f"Final MSA: {final_msa_path}")
            
            return final_msa_path
            
        except Exception as e:
            self.logger.error(f"Multi-round hit expand pipeline failed: {e}")
            raise HitExpandError(f"Pipeline failed: {e}")
    
    def _run_clustering_stage(self, input_msa: Path) -> Dict[str, str]:
        """Run MMseqs2 clustering stage."""
        self.logger.info("=== STAGE 1: SEQUENCE CLUSTERING ===")
        
        clustering_dir = self.base_dir / "01_clustering"
        clustering_dir.mkdir(exist_ok=True)
        
        # Run clustering directly on A3M file (MMseqs2 easy-cluster handles A3M)
        cluster_results = self.mmseqs_wrapper.cluster_sequences(
            input_file=input_msa,
            output_dir=clustering_dir,
            prefix="clustered"
        )
        
        # Load representative sequences from the _rep_seq.fasta output
        rep_fasta = Path(cluster_results["representative_sequences"])
        clustered_sequences = {}
        
        with open(rep_fasta, 'r') as f:
            current_header = None
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    current_header = line
                elif current_header and line:
                    clustered_sequences[current_header] = line
                    current_header = None
        
        self.workflow_state["clustering_completed"] = True
        self.logger.info(f"Clustering completed: {len(clustered_sequences)} representative sequences")
        
        # Create DONE file to mark completion
        self._create_done_file(clustering_dir, "01_clustering")
        
        return clustered_sequences
    
    def _run_similarity_search_stage(self, 
                                   good_sequences: Dict[str, str],
                                   source_msa: Path) -> Path:
        """Run similarity search stage using good sequences from structure analysis."""
        self.logger.info("=== STAGE 5: SIMILARITY SEARCH & EXPANSION ===")
        
        similarity_dir = self.base_dir / "05_similarity_search"
        similarity_dir.mkdir(exist_ok=True)
        
        # Run similarity search to expand good sequences
        expanded_msa = self.similarity_search.search_and_expand(
            representative_sequences=good_sequences,  # These are now the "good" sequences
            source_msa=source_msa,
            output_dir=similarity_dir
        )
        
        # Copy to final location
        final_msa_path = self.base_dir / "hit_expand_final_msa.a3m"
        shutil.copy2(expanded_msa, final_msa_path)
        
        self.workflow_state["similarity_search_completed"] = True
        self.logger.info(f"Similarity search completed: {final_msa_path}")
        
        # Create DONE file to mark completion
        self._create_done_file(similarity_dir, "05_similarity_search")
        
        return final_msa_path
    
    def _run_expanded_subset_generation_stage(self, final_msa: Path) -> Dict[str, Any]:
        """Run expanded subset generation stage using final expanded MSA."""
        self.logger.info("=== STAGE 6: EXPANDED SUBSET GENERATION ===")
        
        expanded_subsets_dir = self.base_dir / "06_expanded_subsets"
        expanded_subsets_dir.mkdir(exist_ok=True)
        
        # Check if existing results exist
        existing_pdbs = list(expanded_subsets_dir.rglob("*.pdb"))
        if existing_pdbs:
            self.logger.info(f"Found {len(existing_pdbs)} existing structure prediction results in {expanded_subsets_dir}")
            self.logger.info("Skipping subset generation and structure prediction - proceeding directly to plotting")
            
            # Load existing subset results for plotting
            subset_results = self._load_existing_subset_results(expanded_subsets_dir)
            
            # Run plotting directly on existing results
            self._run_expanded_subset_plotting(expanded_subsets_dir, subset_results)
            
            return subset_results
        
        # Parse sequences from final MSA
        parser = A3MParser(strict_validation=False)
        expanded_sequences = parser.parse_file(final_msa)
        
        self.logger.info(f"Parsed {len(expanded_sequences)} sequences from final MSA")
        
        # Get query sequence from the expanded sequences
        query_header, query_sequence = parser.get_query_sequence(expanded_sequences)
        
        # Generate subsets using the same configuration as original subsets
        subset_results = self.subset_generator.generate_subsets_with_query(
            expanded_msa=final_msa,
            query_header=query_header,
            query_sequence=query_sequence,
            output_dir=expanded_subsets_dir
        )
        
        self.logger.info(f"Expanded subset generation completed: {len(subset_results['subset_paths'])} subsets")
        
        # Run structure prediction on expanded subsets (similar to stage 3)
        prediction_results = self._run_expanded_structure_prediction(subset_results, expanded_subsets_dir)
        
        # Run plotting on the prediction results
        self._run_expanded_subset_plotting(expanded_subsets_dir, subset_results)
        
        self.workflow_state["expanded_subset_generation_completed"] = True
        
        # Create DONE file to mark completion
        self._create_done_file(expanded_subsets_dir, "06_expanded_subsets")
        
        return subset_results
    
    def _run_expanded_structure_prediction(self, subset_results: Dict[str, Any], expanded_subsets_dir: Path) -> Dict[str, Any]:
        """Run structure prediction on expanded subsets."""
        self.logger.info("=== EXPANDED STRUCTURE PREDICTION ===")
        
        # Reuse the generic structure prediction logic with different parameters
        return self._run_generic_structure_prediction(
            subset_results=subset_results,
            base_dir=expanded_subsets_dir,
            job_prefix="expanded",
            stage_name="expanded structure prediction"
        )
    
    def _run_expanded_subset_plotting(self, expanded_subsets_dir: Path, subset_results: Dict[str, Any]):
        """Run plotting for expanded subset structure analysis results."""
        self.logger.info("=== EXPANDED SUBSET PLOTTING ===")
        
        # Create plots directory within expanded subsets
        plots_dir = expanded_subsets_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Find all PDB files in the expanded subsets directory
        pdb_files = list(expanded_subsets_dir.rglob("*.pdb"))
        
        if not pdb_files:
            self.logger.warning("No PDB files found for expanded subset plotting")
            return
        
        self.logger.info(f"Found {len(pdb_files)} PDB files for analysis")
        
        # Load filter configuration for metrics to calculate
        filter_config_path = Path(self.config_file)
        if filter_config_path.exists():
            with open(filter_config_path, 'r') as f:
                filter_config = json.load(f)
        else:
            self.logger.warning(f"Filter config not found: {filter_config_path}")
            return
        
        # Run structure analysis on all PDB files
        analysis_results = {}
        
        # Get analysis parameters (same as main structure analysis)
        hit_expand_plddt_threshold = getattr(self.config, 'plddt_threshold', 75.0)
        all_filter_criteria = filter_config.get("filter_criteria", [])
        
        # Temporarily suppress INFO logging from sequence_processing module
        seq_processing_logger = logging.getLogger('af_claseq.sequence_processing')
        original_seq_level = seq_processing_logger.level
        seq_processing_logger.setLevel(logging.WARNING)
        
        # Create progress bar for structure analysis
        pbar = tqdm(pdb_files, desc="Analyzing expanded structures", unit="structure")
        
        for pdb_file in pbar:
            pbar.set_postfix({"current": pdb_file.name})
            
            try:
                # Run structure analysis using the same method as main analysis
                metrics = self.structure_analyzer.process_single_pdb(
                    pdb_path=str(pdb_file),
                    filter_criteria=all_filter_criteria,
                    basics=filter_config.get("basics", {}),
                    plddt_threshold=hit_expand_plddt_threshold
                )
                
                if metrics and "error" not in metrics:
                    analysis_results[str(pdb_file)] = metrics
                    
            except Exception as e:
                self.logger.debug(f"Error analyzing {pdb_file}: {e}")
        
        # Close progress bar
        pbar.close()
        
        # Restore original logging level
        seq_processing_logger.setLevel(original_seq_level)
        
        self.logger.info(f"Successfully analyzed {len(analysis_results)} expanded structures")
        
        # Save results as CSV and create plots (no filtering needed)
        all_filter_criteria = filter_config.get("filter_criteria", [])
        csv_file, plot_files = self._save_results_and_create_plots(
            analysis_results, 
            analysis_results,  # Use same results for both (no filtering)
            expanded_subsets_dir, 
            all_filter_criteria
        )
        
        self.logger.info(f"Saved expanded subset analysis to: {csv_file}")
        self.logger.info(f"Created {len(plot_files)} plots in {plots_dir}")
    
    def _run_subset_generation_stage(self, representative_sequences: Dict[str, str]) -> Dict[str, Any]:
        """Run subset generation stage using representative sequences."""
        self.logger.info("=== STAGE 2: SUBSET GENERATION ===")
        
        subsets_dir = self.base_dir / "02_subsets"
        subsets_dir.mkdir(exist_ok=True)
        
        # Get query sequence from original source MSA
        input_msa = Path(self.config.input_msa)
        parser = A3MParser(strict_validation=False)
        source_sequences = parser.parse_file(input_msa)
        query_header, query_sequence = parser.get_query_sequence(source_sequences)
        
        self.logger.info(f"Source query sequence: {query_header}")
        
        # Save representative sequences as temporary MSA for subset generation
        temp_msa = subsets_dir / "representative_sequences.a3m"
        parser.write_sequences(representative_sequences, temp_msa)
        
        # Generate subsets with query sequence included
        subset_results = self.subset_generator.generate_subsets_with_query(
            expanded_msa=temp_msa,
            query_header=query_header,
            query_sequence=query_sequence,
            output_dir=subsets_dir
        )
        
        # Validate subsets
        validation_results = self.subset_generator.validate_subsets(
            subset_results["subset_paths"]
        )
        
        subset_results["validation"] = validation_results
        subset_results["representative_sequences"] = representative_sequences
        subset_results["query_header"] = query_header
        subset_results["query_sequence"] = query_sequence
        
        self.workflow_state["subset_generation_completed"] = True
        self.logger.info(f"Subset generation completed: {len(subset_results['subset_paths'])} subsets")
        
        # Create DONE file to mark completion
        self._create_done_file(subsets_dir, "02_subsets")
        
        return subset_results
    
    def _run_generic_structure_prediction(self, 
                                         subset_results: Dict[str, Any],
                                         base_dir: Path,
                                         job_prefix: str = "hit_expand",
                                         stage_name: str = "structure prediction") -> Dict[str, Any]:
        """Generic structure prediction logic that can be reused for different stages."""
        # Create job specifications for batch directories
        job_specs = []
        for batch_name, subset_paths in subset_results["batch_info"].items():
            batch_dir = base_dir / batch_name
            job_spec = {
                "name": f"{job_prefix}_{batch_name}",
                "task_dir": str(batch_dir),
                "gres": "gpu:1",
                "num_subsets": len(subset_paths)
            }
            job_specs.append(job_spec)
        
        self.logger.info(f"Created {len(job_specs)} {stage_name} job specifications")
        
        # Submit and monitor jobs using the generic function
        submitted_jobs = self._submit_and_monitor_structure_jobs(
            job_specs,
            job_prefix=job_prefix,
            stage_name=stage_name
        )
        
        return {
            "job_specs": job_specs,
            "submitted_jobs": submitted_jobs,
            "prediction_dir": base_dir
        }
    
    def _submit_and_monitor_structure_jobs(self, 
                                          job_specs: List[Dict[str, Any]], 
                                          job_prefix: str = "hit_expand",
                                          stage_name: str = "structure prediction") -> List[str]:
        """Generic function to submit and monitor structure prediction jobs."""
        submitted_jobs = []
        
        # Temporarily suppress INFO logging from slurm_utils module
        slurm_logger = logging.getLogger('af_claseq.slurm_utils')
        original_slurm_level = slurm_logger.level
        slurm_logger.setLevel(logging.WARNING)
        
        # Create progress bar for job submission
        pbar = tqdm(job_specs, desc=f"Submitting {stage_name} jobs", unit="batch")
        
        for job_spec in pbar:
            batch_dir = job_spec["task_dir"]
            job_name = job_spec["name"]
            
            # Update progress bar description
            pbar.set_postfix({"current": job_name})
            
            # Check for existing results if requested
            if self.config.check_existing_jobs:
                task_dir = Path(batch_dir)
                existing_pdbs = list(task_dir.rglob("*.pdb"))
                if existing_pdbs:
                    self.logger.info(f"Existing results found in {task_dir}, skipping job submission")
                    continue
            
            # Extract batch identifier from job name for job_id
            batch_id = job_name.replace(f"{job_prefix}_", "")
            
            # Submit using the existing submit_job method
            job_id = self.slurm_submitter.submit_job(
                task_dir=batch_dir,
                job_id=batch_id
            )
            
            if job_id:
                submitted_jobs.append(job_id)
            else:
                self.logger.error(f"Failed to submit job for {batch_dir}")
        
        # Close progress bar
        pbar.close()
        
        # Restore original logging level
        slurm_logger.setLevel(original_slurm_level)
        
        self.logger.info(f"Submitted {len(submitted_jobs)}/{len(job_specs)} {stage_name} jobs")
        
        # Monitor jobs if requested
        if self.config.monitor_jobs and submitted_jobs:
            self.logger.info(f"Monitoring {len(submitted_jobs)} {stage_name} jobs")
            job_states = self.slurm_submitter.monitor_jobs(
                job_ids=submitted_jobs,
                check_interval=self.config.job_check_interval,
                timeout=self.config.job_timeout
            )
            
            # Log job completion statistics
            completed_jobs = sum(1 for state in job_states.values() if state.value == "COMPLETED")
            self.logger.info(f"{stage_name} completed: {completed_jobs}/{len(submitted_jobs)} jobs successful")
        
        return submitted_jobs
    
    def _run_structure_prediction_stage(self, subset_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run structure prediction stage."""
        self.logger.info("=== STAGE 3: STRUCTURE PREDICTION ===")
        
        # Use the subset directories directly (no separate prediction directory)
        subsets_dir = self.base_dir / "02_subsets"
        
        # Reuse the generic structure prediction logic
        prediction_results = self._run_generic_structure_prediction(
            subset_results=subset_results,
            base_dir=subsets_dir,
            job_prefix="hit_expand",
            stage_name="structure prediction"
        )
        
        self.workflow_state["structure_prediction_completed"] = True
        
        # Create DONE file to mark completion
        self._create_done_file(subsets_dir, "03_structure_prediction")
        
        return prediction_results
    
    def _run_structure_analysis_stage(self, prediction_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run structure analysis stage."""
        self.logger.info("=== STAGE 4: STRUCTURE ANALYSIS ===")
        
        analysis_dir = self.base_dir / "04_structure_analysis"
        analysis_dir.mkdir(exist_ok=True)
        
        # Collect all PDB files from prediction results
        prediction_dir = prediction_results["prediction_dir"]
        pdb_files = list(prediction_dir.rglob("*.pdb"))
        
        if not pdb_files:
            self.logger.warning("No PDB files found for structure analysis")
            return {"pdb_files": [], "analysis_results": {}}
        
        self.logger.info(f"Found {len(pdb_files)} PDB files for analysis")
        
        # Load filter configuration from hit_expand config and JSON file
        with open(self.config_file, 'r') as f:
            filter_config = json.load(f)
        
        # Get hit_expand specific configuration
        hit_expand_plddt_threshold = getattr(self.config, 'plddt_threshold', 75.0)
        filter_criteria_name = getattr(self.config, 'filter_criteria', None)
        
        # Get filter criteria from JSON - the JSON contains filter_criteria as a list
        all_filter_criteria = filter_config.get("filter_criteria", [])
        
        # For ANALYSIS: Calculate ALL metrics from JSON config
        analysis_filter_criteria = all_filter_criteria
        
        # For FILTERING: Use only the specified criteria name (e.g., "2qke_tmscore")
        selected_filter_criteria = []
        if filter_criteria_name:
            # Find criteria matching the specified name
            for criterion in all_filter_criteria:
                if criterion.get("name") == filter_criteria_name:
                    selected_filter_criteria.append(criterion)
            
            if selected_filter_criteria:
                self.logger.info(f"Found {len(selected_filter_criteria)} criteria matching '{filter_criteria_name}' for filtering")
            else:
                self.logger.warning(f"Filter criteria '{filter_criteria_name}' not found in config file!")
                available_names = [c.get("name") for c in all_filter_criteria]
                self.logger.warning(f"Available criteria names: {available_names}")
                self.logger.warning("NO STRUCTURE FILTERING WILL BE APPLIED - ALL STRUCTURES WILL PASS!")
        else:
            # Use all available criteria if no specific name specified
            selected_filter_criteria = all_filter_criteria
            self.logger.info(f"Using all {len(selected_filter_criteria)} available filter criteria for filtering")
        
        self.logger.info(f"Will calculate ALL {len(analysis_filter_criteria)} metrics from JSON config: {[c.get('name') for c in analysis_filter_criteria]}")
        self.logger.info(f"Will use {len(selected_filter_criteria)} criteria for filtering: {[c.get('name') for c in selected_filter_criteria]}")
        
        self.logger.info(f"Using pLDDT threshold: {hit_expand_plddt_threshold}")
        self.logger.info(f"Using filter criteria: {filter_criteria_name}")
        
        # Debug: Print the actual filter criteria configuration and resolve reference PDB paths
        # Do this for ALL criteria that will be calculated
        if analysis_filter_criteria:
            config_dir = Path(self.config_file).parent  # Directory containing the JSON config file
            self.logger.info("=== ALL METRICS TO BE CALCULATED ===")
            for i, criterion in enumerate(analysis_filter_criteria):
                ref_pdb = criterion.get('ref_pdb', 'N/A')
                
                # Resolve relative paths relative to config file directory
                if ref_pdb != 'N/A' and not Path(ref_pdb).is_absolute():
                    resolved_ref_pdb = config_dir / ref_pdb
                    criterion['ref_pdb'] = str(resolved_ref_pdb)  # Update the criterion with absolute path
                    self.logger.info(f"  Metric {i+1}: {criterion.get('name')} ({criterion.get('type')}) method={criterion.get('method', 'below')} ref_pdb={resolved_ref_pdb}")
                    
                    # Check if reference PDB file exists
                    if not resolved_ref_pdb.exists():
                        self.logger.error(f"Reference PDB file not found: {resolved_ref_pdb}")
                        self.logger.error("TM-score calculation will fail!")
                    else:
                        self.logger.info(f"Reference PDB file found: {resolved_ref_pdb}")
                else:
                    self.logger.info(f"  Metric {i+1}: {criterion.get('name')} ({criterion.get('type')}) method={criterion.get('method', 'below')} ref_pdb={ref_pdb}")
        else:
            self.logger.warning("No filter criteria loaded - structures will only be filtered by pLDDT!")
        
        # Analyze structures using existing StructureAnalyzer methods with progress bar
        analysis_results = {}
        
        # Temporarily suppress INFO logging from sequence_processing module
        seq_processing_logger = logging.getLogger('af_claseq.sequence_processing')
        original_level = seq_processing_logger.level
        seq_processing_logger.setLevel(logging.WARNING)
        
        # Set up progress bar
        pbar = tqdm(pdb_files, desc="Analyzing structures", unit="structure")
        
        for pdb_file in pbar:
            try:
                # Update progress bar with current structure name
                pbar.set_postfix({"current": Path(pdb_file).name[:30] + "..."})
                
                # Use the existing process_single_pdb method with ALL criteria for calculation
                metrics = self.structure_analyzer.process_single_pdb(
                    pdb_path=str(pdb_file),
                    filter_criteria=analysis_filter_criteria,  # Calculate ALL metrics
                    basics=filter_config.get("basics", {}),
                    plddt_threshold=hit_expand_plddt_threshold
                )
                
                if metrics:
                    analysis_results[str(pdb_file)] = metrics
                    # Debug: Show what metrics were actually calculated
                    metric_names = [k for k in metrics.keys() if k not in ['PDB', 'seq_count']]
                    self.logger.debug(f"Structure {pdb_file}: calculated metrics {metric_names}")
                    
                    # Specifically check for TM-score metrics (but only log to debug level)
                    tmscore_metrics = [k for k in metric_names if 'tmscore' in k.lower()]
                    if tmscore_metrics:
                        for tm_metric in tmscore_metrics:
                            tm_value = metrics.get(tm_metric)
                            self.logger.debug(f"Structure {Path(pdb_file).name}: {tm_metric} = {tm_value}")
                    else:
                        self.logger.debug(f"Structure {Path(pdb_file).name}: NO TM-score metrics calculated!")
                else:
                    self.logger.debug(f"Structure {pdb_file} did not meet basic criteria (pLDDT threshold)")
                
            except Exception as e:
                self.logger.warning(f"Structure analysis failed for {pdb_file}: {e}")
                analysis_results[str(pdb_file)] = {"error": str(e)}
        
        # Close progress bar
        pbar.close()
        
        # Restore original logging level for sequence_processing module
        seq_processing_logger.setLevel(original_level)
        
        # Debug: Check what metrics were actually calculated across all structures
        all_metrics = set()
        successful_structures = 0
        for pdb_path, metrics in analysis_results.items():
            if "error" not in metrics:
                all_metrics.update(metrics.keys())
                successful_structures += 1
        
        self.logger.info(f"Successfully analyzed {successful_structures} structures out of {len(pdb_files)}")
        self.logger.info(f"Available metrics: {sorted(all_metrics)}")
        
        # Check specifically for the metrics defined in ALL filter criteria
        expected_metrics = [criterion.get('name') for criterion in analysis_filter_criteria if criterion.get('name')]
        missing_metrics = [m for m in expected_metrics if m not in all_metrics]
        if missing_metrics:
            self.logger.warning(f"Missing expected metrics: {missing_metrics}")
        else:
            self.logger.info(f"All expected metrics found: {expected_metrics} - plots should be created")
        
        # Filter structures based on criteria
        filtered_results = self._filter_structures(analysis_results, filter_config)
        
        # Save results as CSV and create plots
        # Use ALL filter criteria from JSON config for plotting, not just the selected ones for filtering
        all_filter_criteria = filter_config.get("filter_criteria", [])
        csv_file, plot_files = self._save_results_and_create_plots(
            analysis_results, 
            filtered_results, 
            analysis_dir, 
            all_filter_criteria  # Use all criteria for plotting
        )
        
        self.workflow_state["structure_analysis_completed"] = True
        self.logger.info(f"Structure analysis completed: {len(filtered_results)} structures passed filters")
        
        # Create DONE file to mark completion
        self._create_done_file(analysis_dir, "04_structure_analysis")
        
        return {
            "analysis_results": analysis_results,
            "filtered_results": filtered_results,
            "csv_file": csv_file,
            "plot_files": plot_files
        }
    
    def _extract_good_sequences(self, 
                               analysis_results: Dict[str, Any], 
                               subset_results: Dict[str, Any]) -> Dict[str, str]:
        """
        Extract sequences from subsets that correspond to good structures.
        
        Args:
            analysis_results: Results from structure analysis
            subset_results: Results from subset generation
            
        Returns:
            Dictionary of good sequences
        """
        self.logger.info("=== EXTRACTING GOOD SEQUENCES ===")
        
        filtered_results = analysis_results["filtered_results"]
        good_sequences = {}
        
        if not filtered_results:
            self.logger.warning("No structures passed filtering criteria - using representative sequences")
            return subset_results.get("representative_sequences", {})
        
        # Extract sequences only from subsets that produced good structures
        parser = A3MParser(strict_validation=False)
        
        for pdb_path_str in filtered_results.keys():
            try:
                pdb_path = Path(pdb_path_str)
                
                # Construct A3M file path from PDB file path
                # PDB files are named like: subset_000001_unrelaxed_rank_001_...
                # A3M files are named like: subset_000001.a3m and stored in batch folder
                pdb_name = pdb_path.name
                a3m_name = pdb_name.split('_unrelaxed')[0] + '.a3m'
                
                # A3M file is in the same directory as the PDB file (batch folder)
                a3m_path = pdb_path.parent / a3m_name
                
                if a3m_path.exists():
                    # Parse sequences from this A3M file
                    sequences = parser.parse_file(a3m_path)
                    
                    # Add all sequences from this subset to good sequences
                    for header, sequence in sequences.items():
                        if header not in good_sequences:  # Avoid duplicates
                            good_sequences[header] = sequence
                    
                    self.logger.debug(f"Added {len(sequences)} sequences from {a3m_name}")
                else:
                    self.logger.warning(f"A3M file not found for structure {pdb_path}: {a3m_path}")
                    
            except Exception as e:
                self.logger.warning(f"Error processing structure path {pdb_path_str}: {e}")
        
        if not good_sequences:
            self.logger.warning("No good sequences extracted - using representative sequences")
            good_sequences = subset_results.get("representative_sequences", {})
        
        self.logger.info(f"Extracted {len(good_sequences)} good sequences from {len(filtered_results)} good structures")
        
        return good_sequences
    
    def _save_results_and_create_plots(self,
                                      analysis_results: Dict[str, Any],
                                      filtered_results: Dict[str, Any],
                                      output_dir: Path,
                                      filter_criteria: List[Dict[str, Any]]) -> tuple:
        """
        Save analysis results to CSV and create plots.
        
        Args:
            analysis_results: All structure analysis results
            filtered_results: Filtered analysis results
            output_dir: Directory to save outputs
            filter_criteria: List of filter criteria used
            
        Returns:
            Tuple of (csv_file_path, list_of_plot_paths)
        """
        try:
            # Convert results to DataFrame
            all_data = []
            for pdb_path, metrics in analysis_results.items():
                if "error" not in metrics:
                    row = {"PDB": Path(pdb_path).name}
                    row.update(metrics)
                    all_data.append(row)
            
            if not all_data:
                self.logger.warning("No valid data for CSV/plotting")
                return None, []
            
            df = pd.DataFrame(all_data)
            
            
            # Save to CSV
            csv_file = output_dir / "structure_analysis_results.csv"
            df.to_csv(csv_file, index=False)
            self.logger.info(f"Saved analysis results to CSV: {csv_file}")
            
            # Create plots directory
            plots_dir = output_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            plot_files = []
            
            # Extract metric names from filter criteria
            metric_names = [criterion.get('name') for criterion in filter_criteria if criterion.get('name')]
            self.logger.info(f"Expected metrics from filter criteria: {metric_names}")
            
            # Check which metrics are actually available in the DataFrame
            available_metrics = [m for m in metric_names if m in df.columns]
            missing_metrics = [m for m in metric_names if m not in df.columns]
            
            if missing_metrics:
                self.logger.warning(f"Metrics missing from DataFrame: {missing_metrics}")
            self.logger.info(f"Available metrics for plotting: {available_metrics}")
            
            # Extract plotting configuration from config
            plotting_config = {
                'scatter_plot_metric1_min': getattr(self.config, 'scatter_plot_metric1_min', 0.0),
                'scatter_plot_metric1_max': getattr(self.config, 'scatter_plot_metric1_max', 1.0),
                'scatter_plot_metric2_min': getattr(self.config, 'scatter_plot_metric2_min', 0.0),
                'scatter_plot_metric2_max': getattr(self.config, 'scatter_plot_metric2_max', 1.0),
                'scatter_plot_metric1_ticks': getattr(self.config, 'scatter_plot_metric1_ticks', [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
                'scatter_plot_metric2_ticks': getattr(self.config, 'scatter_plot_metric2_ticks', [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
                'plddt_plot_min': getattr(self.config, 'plddt_plot_min', 0),
                'plddt_plot_max': getattr(self.config, 'plddt_plot_max', 100),
                'plddt_plot_ticks': getattr(self.config, 'plddt_plot_ticks', [0, 20, 40, 60, 80, 100])
            }
            
            # Use available metrics instead of expected metrics
            metric_names = available_metrics
            
            # Create 1D distribution plots for each metric
            for metric_name in metric_names:
                if metric_name in df.columns:
                    try:
                        plot_path = plot_1d_distribution(
                            results_df=df,
                            metric_name=metric_name,
                            output_dir=str(plots_dir),
                            n_plot_bins=20,
                            logger=self.logger
                        )
                        if plot_path:
                            plot_files.append(plot_path)
                    except Exception as e:
                        self.logger.warning(f"Failed to create 1D plot for {metric_name}: {e}")
            
            # Also create 1D plots for pLDDT metrics
            for plddt_metric in ['plddt', 'local_plddt']:
                if plddt_metric in df.columns:
                    try:
                        plot_path = plot_1d_distribution(
                            results_df=df,
                            metric_name=plddt_metric,
                            output_dir=str(plots_dir),
                            n_plot_bins=20,
                            logger=self.logger
                        )
                        if plot_path:
                            plot_files.append(plot_path)
                    except Exception as e:
                        self.logger.warning(f"Failed to create 1D plot for {plddt_metric}: {e}")
            
            # Create 2D scatter plots for metric combinations
            if len(metric_names) >= 2 and all(m in df.columns for m in metric_names[:2]):
                try:
                    # Create scatter plot colored by pLDDT using the standard plotting style
                    plot_path = create_2d_scatter_plot(
                        results_df=df,
                        metric_name1=metric_names[0],
                        metric_name2=metric_names[1],
                        output_dir=str(plots_dir),
                        color_metric='plddt',
                        title=f'{metric_names[0]} vs {metric_names[1]}',
                        x_min=plotting_config['scatter_plot_metric1_min'],
                        x_max=plotting_config['scatter_plot_metric1_max'],
                        y_min=plotting_config['scatter_plot_metric2_min'],
                        y_max=plotting_config['scatter_plot_metric2_max'],
                        x_ticks=plotting_config['scatter_plot_metric1_ticks'],
                        y_ticks=plotting_config['scatter_plot_metric2_ticks'],
                        logger=self.logger
                    )
                    if plot_path:
                        plot_files.append(plot_path)
                    
                    # Create joint plot with marginal distributions
                    plot_path = create_joint_plot(
                        results_df=df,
                        metric_name1=metric_names[0],
                        metric_name2=metric_names[1],
                        output_dir=str(plots_dir),
                        color_metric='plddt',
                        x_min=plotting_config['scatter_plot_metric1_min'],
                        x_max=plotting_config['scatter_plot_metric1_max'],
                        y_min=plotting_config['scatter_plot_metric2_min'],
                        y_max=plotting_config['scatter_plot_metric2_max'],
                        logger=self.logger
                    )
                    if plot_path:
                        plot_files.append(plot_path)
                        
                    # Create local_plddt colored version if available
                    if 'local_plddt' in df.columns:
                        plot_path = create_2d_scatter_plot(
                            results_df=df,
                            metric_name1=metric_names[0],
                            metric_name2=metric_names[1],
                            output_dir=str(plots_dir),
                            color_metric='local_plddt',
                            title=f'{metric_names[0]} vs {metric_names[1]} (Local pLDDT)',
                            x_min=plotting_config['scatter_plot_metric1_min'],
                            x_max=plotting_config['scatter_plot_metric1_max'],
                            y_min=plotting_config['scatter_plot_metric2_min'],
                            y_max=plotting_config['scatter_plot_metric2_max'],
                            x_ticks=plotting_config['scatter_plot_metric1_ticks'],
                            y_ticks=plotting_config['scatter_plot_metric2_ticks'],
                            logger=self.logger
                        )
                        if plot_path:
                            plot_files.append(plot_path)
                            
                except Exception as e:
                    self.logger.warning(f"Failed to create 2D plots: {e}")
            
            # Create pLDDT vs first metric scatter plot if available
            if metric_names and metric_names[0] in df.columns:
                try:
                    plot_path = create_2d_scatter_plot(
                        results_df=df,
                        metric_name1='plddt',
                        metric_name2=metric_names[0],
                        output_dir=str(plots_dir),
                        color_metric='plddt',
                        title=f'pLDDT vs {metric_names[0]}',
                        x_min=plotting_config['plddt_plot_min'],
                        x_max=plotting_config['plddt_plot_max'],
                        y_min=plotting_config['scatter_plot_metric1_min'],
                        y_max=plotting_config['scatter_plot_metric1_max'],
                        x_ticks=plotting_config['plddt_plot_ticks'],
                        y_ticks=plotting_config['scatter_plot_metric1_ticks'],
                        logger=self.logger
                    )
                    if plot_path:
                        plot_files.append(plot_path)
                except Exception as e:
                    self.logger.warning(f"Failed to create pLDDT scatter plot: {e}")
            
            # Add filtered results summary
            if filtered_results:
                filtered_df = df[df['PDB'].isin([Path(p).name for p in filtered_results.keys()])]
                filtered_csv = output_dir / "filtered_structures.csv"
                filtered_df.to_csv(filtered_csv, index=False)
                self.logger.info(f"Saved filtered structures to CSV: {filtered_csv}")
            
            self.logger.info(f"Created {len(plot_files)} plots in {plots_dir}")
            
            return csv_file, plot_files
            
        except Exception as e:
            self.logger.error(f"Error saving results and creating plots: {e}")
            return None, []
    
    def _filter_structures(self, 
                          analysis_results: Dict[str, Any],
                          filter_config: Dict[str, Any]) -> Dict[str, Any]:
        """Filter structures based on analysis criteria using hit_expand configuration."""
        filtered_results = {}
        
        # Get hit_expand specific thresholds
        hit_expand_plddt_threshold = getattr(self.config, 'plddt_threshold', 75.0)
        filter_criteria_threshold = getattr(self.config, 'filter_criteria_threshold', 0.8)
        filter_criteria_name = getattr(self.config, 'filter_criteria', None)
        
        self.logger.info(f"Filtering with pLDDT >= {hit_expand_plddt_threshold}")
        self.logger.info(f"Filtering with {filter_criteria_name} threshold: {filter_criteria_threshold}")
        
        # Get the selected filter criteria configuration (same logic as analysis stage)
        all_filter_criteria = filter_config.get("filter_criteria", [])
        selected_filter_criteria = []
        if filter_criteria_name:
            # Find criteria matching the specified name
            for criterion in all_filter_criteria:
                if criterion.get("name") == filter_criteria_name:
                    selected_filter_criteria.append(criterion)
        else:
            selected_filter_criteria = all_filter_criteria
        
        self.logger.info(f"Filter criteria for filtering stage: {len(selected_filter_criteria)} criteria")
        if not selected_filter_criteria:
            self.logger.warning("NO CRITERIA-BASED FILTERING - only pLDDT filtering will be applied!")
        
        for pdb_file, metrics in analysis_results.items():
            if "error" in metrics:
                continue
            
            # Check pLDDT threshold (already filtered in process_single_pdb, but double-check)
            plddt_score = metrics.get("plddt", 0.0)
            if plddt_score < hit_expand_plddt_threshold:
                self.logger.debug(f"Structure {pdb_file} filtered out: pLDDT {plddt_score} < {hit_expand_plddt_threshold}")
                continue
            
            # Check filter criteria against threshold
            passes_filter = True
            for criterion in selected_filter_criteria:
                criterion_name = criterion.get("name")
                criterion_type = criterion.get("type", "")
                criterion_method = criterion.get("method", "below")  # default filtering method
                
                if criterion_name in metrics:
                    criterion_value = metrics[criterion_name]
                    
                    # Apply threshold based on criterion type and method
                    if criterion_method == "above":
                        # For metrics where higher is better (e.g., TM-score)
                        if criterion_value < filter_criteria_threshold:
                            self.logger.debug(f"Structure {pdb_file} filtered out: {criterion_name} {criterion_value} < {filter_criteria_threshold}")
                            passes_filter = False
                            break
                        else:
                            self.logger.debug(f"Structure {pdb_file} passed: {criterion_name} {criterion_value} >= {filter_criteria_threshold}")
                    else:  # method == "below" (default)
                        # For metrics where lower is better (e.g., RMSD)
                        if criterion_value > filter_criteria_threshold:
                            self.logger.debug(f"Structure {pdb_file} filtered out: {criterion_name} {criterion_value} > {filter_criteria_threshold}")
                            passes_filter = False
                            break
                        else:
                            self.logger.debug(f"Structure {pdb_file} passed: {criterion_name} {criterion_value} <= {filter_criteria_threshold}")
                else:
                    # CRITICAL: This criterion was not calculated!
                    self.logger.warning(f"Structure {pdb_file}: Missing criterion '{criterion_name}' - STRUCTURE PASSES BY DEFAULT!")
                    self.logger.warning(f"Available metrics: {list(metrics.keys())}")
                    # Could choose to fail here instead: passes_filter = False
            
            if passes_filter:
                self.logger.debug(f"Structure {pdb_file} passed all filters")
                filtered_results[pdb_file] = metrics
        
        self.logger.info(f"Structures passed filtering: {len(filtered_results)}/{len(analysis_results)}")
        return filtered_results
    
    
    def _load_existing_subset_results(self, subsets_dir: Path) -> Dict[str, Any]:
        """
        Load existing subset results from subsets directory.
        
        Args:
            subsets_dir: Path to subsets directory
            
        Returns:
            Dictionary with subset results compatible with pipeline
        """
        self.logger.info("Loading existing subset results...")
        
        # Find all A3M files in subsets directory
        subset_paths = list(subsets_dir.rglob("*.a3m"))
        
        # Find batch directories
        batch_dirs = [d for d in subsets_dir.iterdir() if d.is_dir() and d.name.startswith("batch_")]
        batch_info = {}
        
        for batch_dir in batch_dirs:
            batch_name = batch_dir.name
            batch_a3m_files = list(batch_dir.glob("*.a3m"))
            batch_info[batch_name] = batch_a3m_files
        
        # Try to load metadata if it exists
        metadata_file = subsets_dir / "subset_generation_metadata.json"
        statistics = {}
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    statistics = json.load(f)
                self.logger.info(f"Loaded existing metadata from {metadata_file}")
            except Exception as e:
                self.logger.warning(f"Could not load metadata: {e}")
        
        subset_results = {
            "subset_paths": subset_paths,
            "batch_info": batch_info,
            "statistics": statistics,
            "representative_sequences": {},  # Will be loaded if needed
            "validation": {"total_subsets": len(subset_paths), "valid_subsets": len(subset_paths)}
        }
        
        self.logger.info(f"Loaded {len(subset_paths)} subset files in {len(batch_info)} batches")
        return subset_results
    
    def _run_analysis_only_workflow(self, subset_results: Dict[str, Any]) -> Optional[Path]:
        """
        Run analysis-only workflow for existing subset results.
        
        Args:
            subset_results: Existing subset results
            
        Returns:
            Path to final MSA file if successful, None otherwise
        """
        self.logger.info("=== RUNNING ANALYSIS-ONLY WORKFLOW ===")
        
        # Use subsets directory as prediction directory (since we use it directly now)
        subsets_dir = self.base_dir / "02_subsets"
        prediction_results = {
            "job_specs": [],
            "submitted_jobs": [],
            "prediction_dir": subsets_dir
        }
        
        # Stage 4: Structure analysis and filtering (if not skipped)
        if not self.config.skip_structure_analysis:
            analysis_results = self._run_structure_analysis_stage(prediction_results)
            good_sequences = self._extract_good_sequences(analysis_results, subset_results)
            
            # Check if any structures passed filtering
            filtered_results = analysis_results.get("filtered_results", {})
            if not filtered_results:
                self.logger.warning("=== NO STRUCTURES PASSED FILTERING CRITERIA ===")
                self.logger.warning("Aborting similarity search - no good sequences to expand")
                self.logger.info("Saving sequences from first available subset as final result")
                
                # Use sequences from first available subset as fallback
                if subset_results["subset_paths"]:
                    parser = A3MParser(strict_validation=False)
                    fallback_sequences = parser.parse_file(subset_results["subset_paths"][0])
                    final_msa = self.base_dir / "hit_expand_final_msa.a3m"
                    parser.write_sequences(fallback_sequences, final_msa)
                else:
                    # Create empty result if no subsets available
                    final_msa = self.base_dir / "hit_expand_final_msa.a3m"
                    parser = A3MParser(strict_validation=False)
                    parser.write_sequences({}, final_msa)
                
                self.logger.info(f"=== ANALYSIS-ONLY WORKFLOW COMPLETED (NO EXPANSION) ===")
                return final_msa
                
        else:
            self.logger.info("Skipping structure analysis stage")
            # Use representative sequences if available, otherwise parse from first subset
            good_sequences = subset_results.get("representative_sequences", {})
            if not good_sequences:
                # Parse sequences from first available subset as fallback
                if subset_results["subset_paths"]:
                    parser = A3MParser(strict_validation=False)
                    good_sequences = parser.parse_file(subset_results["subset_paths"][0])
        
        # Stage 5: Similarity search to expand good sequences (if not skipped)
        if not self.config.skip_hit_expansion:
            # Need original input MSA for similarity search
            input_msa = Path(self.config.input_msa)
            if input_msa.exists():
                final_msa = self._run_similarity_search_stage(good_sequences, input_msa)
            else:
                self.logger.warning(f"Input MSA not found: {input_msa}, saving good sequences directly")
                final_msa = self.base_dir / "hit_expand_final_msa.a3m"
                parser = A3MParser(strict_validation=False)
                parser.write_sequences(good_sequences, final_msa)
        else:
            self.logger.info("Skipping hit expansion stage")
            final_msa = self.base_dir / "hit_expand_final_msa.a3m"
            parser = A3MParser(strict_validation=False)
            parser.write_sequences(good_sequences, final_msa)
        
        # Stage 6: Generate expanded subsets from final MSA
        expanded_subset_results = self._run_expanded_subset_generation_stage(final_msa)
        
        self.logger.info(f"=== ANALYSIS-ONLY WORKFLOW COMPLETED ===")
        self.logger.info(f"Final MSA: {final_msa}")
        self.logger.info(f"Expanded subsets: {len(expanded_subset_results['subset_paths'])} subsets in {len(expanded_subset_results['batch_info'])} batches")
        
        return final_msa

    def _create_done_file(self, stage_dir: Path, stage_name: str):
        """Create a DONE file to mark stage completion."""
        done_file = stage_dir / f"{stage_name}.DONE"
        with open(done_file, 'w') as f:
            f.write(f"Stage {stage_name} completed successfully\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.logger.info(f"Created DONE file: {done_file}")
    
    def _check_done_file(self, stage_dir: Path, stage_name: str) -> bool:
        """Check if a DONE file exists for the stage."""
        done_file = stage_dir / f"{stage_name}.DONE"
        return done_file.exists()
    
    def _remove_done_file(self, stage_dir: Path, stage_name: str):
        """Remove a DONE file (for cleanup or re-runs)."""
        done_file = stage_dir / f"{stage_name}.DONE"
        if done_file.exists():
            done_file.unlink()
            self.logger.info(f"Removed DONE file: {done_file}")

    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status and statistics."""
        return {
            "workflow_state": self.workflow_state,
            "base_dir": str(self.base_dir),
            "config": {
                "input_msa": self.config.input_msa,
                "num_subsets": self.config.num_subsets,
                "num_batches": self.config.num_batches,
                "similarity_threshold": self.config.similarity_threshold,
                "plddt_threshold": self.config.plddt_threshold
            }
        }
    
    def setup_logging(self):
        """Set up logging for the hit expand pipeline."""
        log_dir = self.base_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / "hit_expand_pipeline.log"
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"Logging configured for hit expand pipeline: {log_file}")
    
    def log_parameters(self):
        """Log all pipeline parameters."""
        self.logger.info("=== HIT EXPAND PIPELINE PARAMETERS ===")
        self.logger.info(f"Input MSA: {self.config.input_msa}")
        self.logger.info(f"Base directory: {self.base_dir}")
        self.logger.info(f"Config file: {self.config_file}")
        self.logger.info(f"Number of subsets: {self.config.num_subsets}")
        self.logger.info(f"Sequences per subset: {self.config.num_random_sequences}")
        self.logger.info(f"Number of batches: {self.config.num_batches}")
        self.logger.info(f"Similarity threshold: {self.config.similarity_threshold}")
        self.logger.info(f"pLDDT threshold: {self.config.plddt_threshold}")
        self.logger.info(f"Filter criteria threshold: {self.config.filter_criteria_threshold}")
        self.logger.info(f"Random seed: {self.config.random_seed}")
        self.logger.info(f"Number of rounds: {self.config.rounds}")
        self.logger.info(f"Cumulative expansion: {self.config.cumulative_expansion}")
        self.logger.info("========================================")
    
    # Multi-round helper methods
    def _load_clustered_sequences(self, clustering_dir: Path) -> Dict[str, str]:
        """Load representative sequences from clustering results."""
        rep_fasta = clustering_dir / "clustered_rep_seq.fasta"
        representative_sequences = {}
        
        if rep_fasta.exists():
            with open(rep_fasta, 'r') as f:
                current_header = None
                for line in f:
                    line = line.strip()
                    if line.startswith('>'):
                        current_header = line
                    elif current_header and line:
                        representative_sequences[current_header] = line
                        current_header = None
        else:
            # Fallback: parse input MSA directly
            parser = A3MParser(strict_validation=False)
            representative_sequences = parser.parse_file(Path(self.config.input_msa))
        
        self.logger.info(f"Loaded {len(representative_sequences)} representative sequences")
        return representative_sequences
    
    def _load_expanded_subsets(self, expanded_subsets_dir: Path) -> Dict[str, str]:
        """Load all sequences from expanded subset A3M files."""
        all_sequences = {}
        a3m_files = list(expanded_subsets_dir.glob("*.a3m"))
        
        with tqdm(total=len(a3m_files), desc="Loading expanded subsets") as pbar:
            for a3m_file in a3m_files:
                sequences = self.parser.parse_file(a3m_file)
                all_sequences.update(sequences)
                pbar.update(1)
        
        self.logger.info(f"Loaded {len(all_sequences)} sequences from {len(a3m_files)} expanded subset files")
        return all_sequences
    
    def _run_single_round(self, round_num: int, round_dir: Path, 
                         input_sequences: Dict[str, str], input_type: str) -> Optional[Path]:
        """Run a single round of hit expand."""
        self.logger.info(f"Running round {round_num} with {len(input_sequences)} input sequences ({input_type})")
        
        try:
            if round_num == 1:
                # ROUND 1: Full pipeline
                # Stage 2: Generate subsets from clustered sequences
                subsets_dir = round_dir / "02_subsets"
                if self._check_done_file(subsets_dir, "02_subsets"):
                    self.logger.info("=== STAGE 2: SUBSET GENERATION ALREADY COMPLETED - SKIPPING ===")
                    subset_results = self._load_existing_subset_results(subsets_dir)
                else:
                    subset_results = self._run_subset_generation_stage_for_round(input_sequences, subsets_dir)
                
                # Stage 3: Structure prediction
                prediction_dir = subsets_dir  # Using subsets directory directly
                if not self.config.skip_structure_prediction:
                    if self._check_done_file(prediction_dir, "03_structure_prediction"):
                        self.logger.info("=== STAGE 3: STRUCTURE PREDICTION ALREADY COMPLETED - SKIPPING ===")
                        prediction_results = {
                            "job_specs": [],
                            "submitted_jobs": [],
                            "prediction_dir": prediction_dir
                        }
                    else:
                        prediction_results = self._run_structure_prediction_stage(subset_results)
                else:
                    self.logger.info("Skipping structure prediction stage")
                    prediction_results = None
                
                # Stage 4: Structure analysis
                analysis_dir = round_dir / "04_structure_analysis"
                if not self.config.skip_structure_analysis and prediction_results:
                    if self._check_done_file(analysis_dir, "04_structure_analysis"):
                        self.logger.info("=== STAGE 4: STRUCTURE ANALYSIS ALREADY COMPLETED - SKIPPING ===")
                        analysis_results = {"analysis_results": {}, "filtered_results": {}}
                        good_sequences = subset_results.get("representative_sequences", {})
                    else:
                        analysis_results = self._run_structure_analysis_stage_for_round(prediction_results, analysis_dir)
                        good_sequences = self._extract_good_sequences(analysis_results, subset_results)
                else:
                    self.logger.info("Skipping structure analysis stage")
                    good_sequences = input_sequences
                
            else:
                # ROUND 2+: Reuse existing predictions
                # Directly analyze structures from previous round's 06_expanded_subsets
                prev_round_dir = self.base_dir / f"round_{round_num - 1}"
                prev_expanded_dir = prev_round_dir / "06_expanded_subsets"
                
                # Stage 4: Analyze existing structures (no prediction needed)
                analysis_dir = round_dir / "04_structure_analysis"
                if self._check_done_file(analysis_dir, "04_structure_analysis"):
                    self.logger.info("=== STAGE 4: STRUCTURE ANALYSIS ALREADY COMPLETED - SKIPPING ===")
                    analysis_results = {"analysis_results": {}, "filtered_results": {}}
                    good_sequences = input_sequences
                else:
                    analysis_results, good_sequences = self._analyze_existing_expanded_structures(
                        prev_expanded_dir, analysis_dir
                    )
            
            # Check if any structures passed filtering
            filtered_results = analysis_results.get("filtered_results", {})
            if not filtered_results:
                self.logger.warning(f"=== ROUND {round_num}: NO STRUCTURES PASSED FILTERING CRITERIA ===")
                if round_num == 1:
                    self.logger.warning("Aborting similarity search - no good sequences to expand")
                    self.logger.info("Saving representative sequences as final result")
                    final_msa = round_dir / "hit_expand_final_msa.a3m"
                    self.parser.write_sequences(input_sequences, final_msa)
                    return final_msa
                else:
                    self.logger.info("No new hits found in this round")
                    return None
            
            # Stage 5: Similarity search against ORIGINAL source
            if not self.config.skip_hit_expansion:
                search_dir = round_dir / "05_similarity_search"
                
                # Get previously found sequences to exclude
                previously_found = self._get_all_previous_sequences(round_num)
                
                # Search for NEW sequences only
                newly_found_sequences = self._run_similarity_search_stage_for_round(
                    good_sequences, search_dir, round_num, exclude_sequences=previously_found
                )
                
                if not newly_found_sequences:
                    self.logger.info(f"No new sequences found in round {round_num}")
                    return None
                
                # Stage 6: Generate expanded subsets and predict ONLY new sequences
                expanded_subsets_dir = round_dir / "06_expanded_subsets"
                if self._check_done_file(expanded_subsets_dir, "06_expanded_subsets"):
                    self.logger.info("=== STAGE 6: EXPANDED SUBSET GENERATION ALREADY COMPLETED - SKIPPING ===")
                else:
                    if round_num > 1:
                        # Copy good structures from previous round
                        self._copy_good_structures_from_previous_round(
                            prev_expanded_dir, expanded_subsets_dir, good_sequences
                        )
                    
                    # Generate subsets and predict only new sequences
                    start_index = self._get_next_subset_index(expanded_subsets_dir)
                    self._run_expanded_subset_generation_stage_for_round(
                        newly_found_sequences, expanded_subsets_dir, start_index
                    )
                
                # Create final MSA for this round
                final_msa_name = f"hit_expand_final_msa{'_round_' + str(round_num) if round_num > 1 else ''}.a3m"
                final_msa = round_dir / final_msa_name
                
                # Combine sequences for final MSA
                if self.config.cumulative_expansion and round_num > 1:
                    # Merge with previous rounds
                    final_sequences = self._merge_sequences_across_rounds(round_num, good_sequences, newly_found_sequences)
                else:
                    final_sequences = {**good_sequences, **newly_found_sequences}
                
                self.parser.write_sequences(final_sequences, final_msa)
                self.logger.info(f"Round {round_num} completed with {len(newly_found_sequences)} new sequences")
                
                return final_msa
            
            else:
                self.logger.info("Skipping hit expansion stage")
                final_msa = round_dir / f"hit_expand_final_msa{'_round_' + str(round_num) if round_num > 1 else ''}.a3m"
                self.parser.write_sequences(good_sequences, final_msa)
                return final_msa
        
        except Exception as e:
            self.logger.error(f"Round {round_num} failed: {e}")
            return None
    
    def _analyze_existing_expanded_structures(self, prev_expanded_dir: Path, 
                                             analysis_dir: Path) -> Tuple[Dict, Dict]:
        """Analyze already-predicted structures from previous round."""
        self.logger.info(f"Analyzing existing structures from {prev_expanded_dir}")
        
        # Initialize structure analyzer
        analyzer = StructureAnalyzer(
            filter_config_path=self.config_file,
            plddt_threshold=self.config.plddt_threshold,
            logger=self.logger
        )
        
        # Find all PDB files
        pdb_files = list(prev_expanded_dir.glob("**/*_unrelaxed_rank_001_*.pdb"))
        self.logger.info(f"Found {len(pdb_files)} existing structures to analyze")
        
        if not pdb_files:
            self.logger.warning("No PDB files found to analyze")
            return ({}, {})
        
        all_results = {}
        filtered_results = {}
        good_sequences = {}
        
        # Temporarily suppress logging
        original_level = logging.getLogger('af_claseq.utils.sequence_processing').level
        logging.getLogger('af_claseq.utils.sequence_processing').setLevel(logging.WARNING)
        
        try:
            with tqdm(total=len(pdb_files), desc="Analyzing existing structures") as pbar:
                for pdb_file in pdb_files:
                    # Get corresponding A3M file
                    a3m_file = Path(str(pdb_file).split('_unrelaxed')[0] + '.a3m')
                    
                    if a3m_file.exists():
                        # Analyze structure
                        result = analyzer.process_single_pdb(str(pdb_file))
                        
                        if result:
                            subset_name = pdb_file.parent.name
                            all_results[subset_name] = result
                            
                            # Check if passes filters
                            if self._passes_filter_criteria(result):
                                filtered_results[subset_name] = result
                                
                                # Extract sequences from A3M
                                sequences = self.parser.parse_file(a3m_file)
                                good_sequences.update(sequences)
                    
                    pbar.update(1)
        
        finally:
            # Restore logging
            logging.getLogger('af_claseq.utils.sequence_processing').setLevel(original_level)
        
        # Save analysis results
        self._save_results_and_create_plots(all_results, filtered_results, analysis_dir, [])
        
        self.logger.info(f"Successfully analyzed {len(all_results)} structures, {len(filtered_results)} passed filters")
        
        return ({"all_results": all_results, "filtered_results": filtered_results}, good_sequences)
    
    def _get_all_previous_sequences(self, current_round: int) -> set:
        """Get all sequences found in previous rounds to avoid duplicates."""
        all_sequences = set()
        
        # Collect from all previous rounds
        for round_num in range(1, current_round):
            round_dir = self.base_dir / f"round_{round_num}"
            
            # Check both similarity search results and expanded subsets
            for subdir in ["05_similarity_search", "06_expanded_subsets"]:
                search_dir = round_dir / subdir
                if search_dir.exists():
                    for a3m_file in search_dir.glob("*.a3m"):
                        try:
                            sequences = self.parser.parse_file(a3m_file)
                            all_sequences.update(sequences.keys())
                        except Exception as e:
                            self.logger.debug(f"Failed to parse {a3m_file}: {e}")
        
        self.logger.info(f"Found {len(all_sequences)} sequences from previous {current_round - 1} rounds")
        return all_sequences
    
    def _copy_good_structures_from_previous_round(self, prev_expanded_dir: Path, 
                                                 new_expanded_dir: Path, 
                                                 good_sequences: Dict[str, str]):
        """Copy structures that passed filters to new round directory."""
        import shutil
        
        new_expanded_dir.mkdir(parents=True, exist_ok=True)
        
        # Get subset names from good sequences
        subset_names = set()
        for header in good_sequences.keys():
            # Extract subset name from header
            if "_subset_" in header:
                subset_name = header.split("_subset_")[1].split("_")[0]
                subset_names.add(f"subset_{subset_name}")
        
        copied_count = 0
        # Copy files for each subset
        for subset_name in subset_names:
            pattern = f"*{subset_name}*"
            for file in prev_expanded_dir.glob(pattern):
                try:
                    dest = new_expanded_dir / file.name
                    shutil.copy2(file, dest)
                    copied_count += 1
                except Exception as e:
                    self.logger.debug(f"Failed to copy {file}: {e}")
        
        self.logger.info(f"Copied {copied_count} structure files from previous round")
    
    def _get_next_subset_index(self, expanded_subsets_dir: Path) -> int:
        """Get the next subset index to avoid overwriting existing files."""
        if not expanded_subsets_dir.exists():
            return 0
        
        existing_files = list(expanded_subsets_dir.glob("subset_*.a3m"))
        if not existing_files:
            return 0
        
        # Extract indices and find max
        indices = []
        for f in existing_files:
            try:
                idx = int(f.stem.split('_')[1])
                indices.append(idx)
            except:
                pass
        
        return max(indices) + 1 if indices else 0
    
    def _check_new_sequences_found(self, round_dir: Path) -> bool:
        """Check if new sequences were found in this round."""
        search_dir = round_dir / "05_similarity_search"
        if not search_dir.exists():
            return False
        
        # Check if any A3M files exist in similarity search results
        a3m_files = list(search_dir.glob("*.a3m"))
        if not a3m_files:
            return False
        
        # Count total sequences in similarity search results
        total_sequences = 0
        for a3m_file in a3m_files:
            try:
                sequences = self.parser.parse_file(a3m_file)
                total_sequences += len(sequences)
            except Exception as e:
                self.logger.debug(f"Failed to parse {a3m_file}: {e}")
        
        return total_sequences > 0
    
    def _merge_sequences_across_rounds(self, current_round: int, 
                                     good_sequences: Dict[str, str], 
                                     newly_found_sequences: Dict[str, str]) -> Dict[str, str]:
        """Merge sequences from all rounds if cumulative expansion is enabled."""
        all_sequences = {}
        
        # Add sequences from all previous rounds
        for round_num in range(1, current_round):
            round_dir = self.base_dir / f"round_{round_num}"
            final_msa_name = f"hit_expand_final_msa{'_round_' + str(round_num) if round_num > 1 else ''}.a3m"
            final_msa = round_dir / final_msa_name
            
            if final_msa.exists():
                try:
                    sequences = self.parser.parse_file(final_msa)
                    all_sequences.update(sequences)
                except Exception as e:
                    self.logger.debug(f"Failed to parse {final_msa}: {e}")
        
        # Add current round sequences
        all_sequences.update(good_sequences)
        all_sequences.update(newly_found_sequences)
        
        self.logger.info(f"Merged {len(all_sequences)} sequences across {current_round} rounds")
        return all_sequences
    
    def _run_subset_generation_stage_for_round(self, sequences: Dict[str, str], subsets_dir: Path) -> Dict[str, Any]:
        """Run subset generation for a specific round."""
        return self._run_subset_generation_stage(sequences, subsets_dir)
    
    def _run_structure_analysis_stage_for_round(self, prediction_results: Dict[str, Any], analysis_dir: Path) -> Dict[str, Any]:
        """Run structure analysis for a specific round."""
        return self._run_structure_analysis_stage(prediction_results, analysis_dir)
    
    def _run_similarity_search_stage_for_round(self, good_sequences: Dict[str, str], 
                                             search_dir: Path, round_num: int,
                                             exclude_sequences: set) -> Dict[str, str]:
        """Search for NEW sequences not found in previous rounds."""
        from af_claseq.modules.similarity_search import SimilaritySearcher
        
        # Initialize similarity searcher
        searcher = SimilaritySearcher(
            source_msa_path=Path(self.config.input_msa),  # Always search original!
            top_k=self.config.similarity_top_k,
            threshold=self.config.similarity_threshold,
            exclude_query_headers=self.config.exclude_query_headers,
            logger=self.logger
        )
        
        # Search for similar sequences
        search_results = searcher.search_and_expand(
            query_sequences=good_sequences,
            output_dir=search_dir
        )
        
        # Filter out sequences already found in previous rounds
        newly_found = {}
        for header, seq in search_results.items():
            if header not in exclude_sequences:
                newly_found[header] = seq
        
        excluded_count = len(search_results) - len(newly_found)
        self.logger.info(f"Round {round_num}: Found {len(newly_found)} NEW sequences "
                        f"(excluded {excluded_count} duplicates from previous rounds)")
        
        return newly_found
    
    def _run_expanded_subset_generation_stage_for_round(self, sequences: Dict[str, str], 
                                                      expanded_subsets_dir: Path, 
                                                      start_index: int = 0) -> Dict[str, Any]:
        """Generate expanded subsets for a specific round, starting from a given index."""
        # Create a temporary A3M file with the new sequences
        temp_a3m = expanded_subsets_dir / "temp_new_sequences.a3m"
        temp_a3m.parent.mkdir(parents=True, exist_ok=True)
        
        self.parser.write_sequences(sequences, temp_a3m)
        
        # Generate subsets with custom starting index
        subset_generator = SubsetGenerator(
            num_random_sequences=self.config.num_random_sequences,
            coverage_threshold=0.8,
            ensure_query_first=True,
            random_seed=self.config.random_seed,
            logger=self.logger
        )
        
        # Generate subsets
        subset_paths = subset_generator.generate_multiple_subsets(
            a3m_file=temp_a3m,
            num_subsets=len(sequences) // self.config.num_random_sequences + 1,
            output_dir=expanded_subsets_dir,
            output_prefix="subset",
            start_index=start_index
        )
        
        # Organize into batches
        batch_info = subset_generator._organize_into_batches(
            subset_paths=subset_paths,
            output_dir=expanded_subsets_dir,
            num_batches=min(self.config.num_batches, len(subset_paths)),
            batch_prefix=self.config.batch_prefix
        )
        
        # Submit ColabFold jobs for the new subsets
        self._submit_jobs_with_existing_check(batch_info)
        
        # Clean up temporary file
        temp_a3m.unlink()
        
        # Write DONE file
        self._write_done_file(expanded_subsets_dir, "06_expanded_subsets")
        
        return {
            "subset_paths": subset_paths,
            "batch_info": batch_info
        }
