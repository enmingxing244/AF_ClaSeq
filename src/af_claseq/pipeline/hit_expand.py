#!/usr/bin/env python3
"""
Hit Expand Pipeline Orchestrator.

This module orchestrates the complete hit expand workflow:
1. Sequence clustering with MMseqs2
2. Similarity search with BLOSUM62
3. Subset generation for structure prediction
4. Structure prediction job submission and monitoring
5. Structure analysis and filtering
"""

import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import tempfile
import shutil

from af_claseq.utils.logging_utils import get_logger
from af_claseq.utils.slurm_utils import SlurmJobSubmitter
from af_claseq.utils.structure_analysis import StructureAnalyzer
from af_claseq.utils.sequence_processing import A3MParser, validate_a3m_file

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
            "structure_analysis_completed": False
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
        Run the complete hit expand pipeline.
        
        Returns:
            Path to final MSA file if successful, None otherwise
        """
        try:
            self.logger.info("=== STARTING HIT EXPAND PIPELINE ===")
            
            # Validate input MSA
            input_msa = Path(self.config.input_msa)
            if not validate_a3m_file(input_msa, strict=False):
                raise HitExpandError(f"Invalid input MSA: {input_msa}")
            
            # Stage 1: Clustering (optional)
            if not self.config.skip_clustering:
                clustered_sequences = self._run_clustering_stage(input_msa)
            else:
                self.logger.info("Skipping clustering stage")
                # Parse input MSA directly
                parser = A3MParser(strict_validation=False)
                clustered_sequences = parser.parse_file(input_msa)
            
            # Stage 2: Hit expansion via similarity search
            if not self.config.skip_hit_expansion:
                expanded_msa = self._run_similarity_search_stage(clustered_sequences, input_msa)
            else:
                self.logger.info("Skipping hit expansion stage")
                expanded_msa = self._save_sequences_as_msa(clustered_sequences, "skipped_expansion.a3m")
            
            # Stage 3: Subset generation
            subset_results = self._run_subset_generation_stage(expanded_msa)
            
            # Stage 4: Structure prediction
            if not self.config.skip_structure_prediction:
                prediction_results = self._run_structure_prediction_stage(subset_results)
            else:
                self.logger.info("Skipping structure prediction stage")
                prediction_results = None
            
            # Stage 5: Structure analysis
            if not self.config.skip_structure_analysis and prediction_results:
                analysis_results = self._run_structure_analysis_stage(prediction_results)
                final_msa = self._create_final_msa(analysis_results, expanded_msa)
            else:
                self.logger.info("Skipping structure analysis stage")
                final_msa = expanded_msa
            
            self.logger.info(f"=== HIT EXPAND PIPELINE COMPLETED ===")
            self.logger.info(f"Final MSA: {final_msa}")
            
            return final_msa
            
        except Exception as e:
            self.logger.error(f"Hit expand pipeline failed: {e}")
            raise HitExpandError(f"Pipeline failed: {e}")
    
    def _run_clustering_stage(self, input_msa: Path) -> Dict[str, str]:
        """Run MMseqs2 clustering stage."""
        self.logger.info("=== STAGE 1: SEQUENCE CLUSTERING ===")
        
        clustering_dir = self.base_dir / "01_clustering"
        clustering_dir.mkdir(exist_ok=True)
        
        # Convert A3M to FASTA for MMseqs2
        fasta_file = clustering_dir / "input_sequences.fasta"
        self.mmseqs_wrapper.convert_a3m_to_fasta(input_msa, fasta_file)
        
        # Run clustering
        cluster_results = self.mmseqs_wrapper.cluster_sequences(
            input_fasta=fasta_file,
            output_dir=clustering_dir,
            prefix="clustered"
        )
        
        # Load representative sequences
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
        
        return clustered_sequences
    
    def _run_similarity_search_stage(self, 
                                   representative_sequences: Dict[str, str],
                                   source_msa: Path) -> Path:
        """Run similarity search stage."""
        self.logger.info("=== STAGE 2: SIMILARITY SEARCH ===")
        
        similarity_dir = self.base_dir / "02_similarity_search"
        similarity_dir.mkdir(exist_ok=True)
        
        # Run similarity search
        expanded_msa = self.similarity_search.search_and_expand(
            representative_sequences=representative_sequences,
            source_msa=source_msa,
            output_dir=similarity_dir
        )
        
        self.workflow_state["similarity_search_completed"] = True
        self.logger.info(f"Similarity search completed: {expanded_msa}")
        
        return expanded_msa
    
    def _run_subset_generation_stage(self, expanded_msa: Path) -> Dict[str, Any]:
        """Run subset generation stage."""
        self.logger.info("=== STAGE 3: SUBSET GENERATION ===")
        
        subsets_dir = self.base_dir / "03_subsets"
        subsets_dir.mkdir(exist_ok=True)
        
        # Generate subsets
        subset_results = self.subset_generator.generate_subsets(
            expanded_msa=expanded_msa,
            output_dir=subsets_dir
        )
        
        # Validate subsets
        validation_results = self.subset_generator.validate_subsets(
            subset_results["subset_paths"]
        )
        
        subset_results["validation"] = validation_results
        
        self.workflow_state["subset_generation_completed"] = True
        self.logger.info(f"Subset generation completed: {len(subset_results['subset_paths'])} subsets")
        
        return subset_results
    
    def _run_structure_prediction_stage(self, subset_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run structure prediction stage."""
        self.logger.info("=== STAGE 4: STRUCTURE PREDICTION ===")
        
        prediction_dir = self.base_dir / "04_structure_prediction"
        prediction_dir.mkdir(exist_ok=True)
        
        # Create job specifications
        job_specs = self.subset_generator.create_structure_prediction_jobs(
            batch_info=subset_results["batch_info"],
            base_output_dir=prediction_dir
        )
        
        # Submit jobs
        submitted_jobs = []
        if not self.config.check_existing_jobs:
            # Submit all jobs
            submitted_jobs = self.slurm_submitter.submit_batch_jobs(
                job_specs=job_specs,
                max_concurrent=self.config.max_workers,
                delay_between_jobs=1.0
            )
        else:
            # Check for existing results and only submit necessary jobs
            submitted_jobs = self._submit_jobs_with_existing_check(job_specs, prediction_dir)
        
        # Monitor jobs if requested
        if self.config.monitor_jobs and submitted_jobs:
            self.logger.info(f"Monitoring {len(submitted_jobs)} structure prediction jobs")
            job_states = self.slurm_submitter.monitor_jobs(
                job_ids=submitted_jobs,
                check_interval=self.config.job_check_interval,
                timeout=self.config.job_timeout
            )
            
            # Log job completion statistics
            completed_jobs = sum(1 for state in job_states.values() if state.value == "COMPLETED")
            self.logger.info(f"Structure prediction completed: {completed_jobs}/{len(submitted_jobs)} jobs successful")
        
        prediction_results = {
            "job_specs": job_specs,
            "submitted_jobs": submitted_jobs,
            "prediction_dir": prediction_dir
        }
        
        self.workflow_state["structure_prediction_completed"] = True
        return prediction_results
    
    def _run_structure_analysis_stage(self, prediction_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run structure analysis stage."""
        self.logger.info("=== STAGE 5: STRUCTURE ANALYSIS ===")
        
        analysis_dir = self.base_dir / "05_structure_analysis"
        analysis_dir.mkdir(exist_ok=True)
        
        # Collect all PDB files from prediction results
        prediction_dir = prediction_results["prediction_dir"]
        pdb_files = list(prediction_dir.rglob("*.pdb"))
        
        if not pdb_files:
            self.logger.warning("No PDB files found for structure analysis")
            return {"pdb_files": [], "analysis_results": {}}
        
        self.logger.info(f"Found {len(pdb_files)} PDB files for analysis")
        
        # Load filter configuration
        with open(self.config_file, 'r') as f:
            filter_config = json.load(f)
        
        # Analyze structures
        analysis_results = {}
        for pdb_file in pdb_files:
            try:
                # Calculate metrics for this structure
                metrics = self.structure_analyzer.calculate_structure_metrics(
                    pdb_file=str(pdb_file),
                    config_file=self.config_file
                )
                
                analysis_results[str(pdb_file)] = metrics
                
            except Exception as e:
                self.logger.warning(f"Structure analysis failed for {pdb_file}: {e}")
                analysis_results[str(pdb_file)] = {"error": str(e)}
        
        # Filter structures based on criteria
        filtered_results = self._filter_structures(analysis_results, filter_config)
        
        # Save analysis results
        results_file = analysis_dir / "structure_analysis_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "all_results": analysis_results,
                "filtered_results": filtered_results,
                "filter_config": filter_config
            }, f, indent=2)
        
        self.workflow_state["structure_analysis_completed"] = True
        self.logger.info(f"Structure analysis completed: {len(filtered_results)} structures passed filters")
        
        return {
            "analysis_results": analysis_results,
            "filtered_results": filtered_results,
            "results_file": results_file
        }
    
    def _filter_structures(self, 
                          analysis_results: Dict[str, Any],
                          filter_config: Dict[str, Any]) -> Dict[str, Any]:
        """Filter structures based on analysis criteria."""
        filtered_results = {}
        
        for pdb_file, metrics in analysis_results.items():
            if "error" in metrics:
                continue
            
            # Check pLDDT threshold
            plddt_score = metrics.get("plddt", 0.0)
            if plddt_score < self.config.plddt_threshold:
                continue
            
            # Check filter criteria
            passes_filter = True
            for criterion in filter_config.get("filter_criteria", []):
                criterion_name = criterion.get("name")
                if criterion_name in metrics:
                    criterion_value = metrics[criterion_name]
                    if criterion_value > self.config.filter_criteria_threshold:
                        passes_filter = False
                        break
            
            if passes_filter:
                filtered_results[pdb_file] = metrics
        
        return filtered_results
    
    def _create_final_msa(self, analysis_results: Dict[str, Any], expanded_msa: Path) -> Path:
        """Create final MSA based on structure analysis results."""
        final_msa_path = self.base_dir / "hit_expand_final_msa.a3m"
        
        # For now, just copy the expanded MSA as the final result
        # In a more sophisticated implementation, you could filter sequences
        # based on structure analysis results
        shutil.copy2(expanded_msa, final_msa_path)
        
        self.logger.info(f"Created final MSA: {final_msa_path}")
        return final_msa_path
    
    def _submit_jobs_with_existing_check(self, 
                                       job_specs: List[Dict[str, Any]], 
                                       prediction_dir: Path) -> List[str]:
        """Submit jobs with check for existing results."""
        submitted_jobs = []
        
        for job_spec in job_specs:
            task_dir = Path(job_spec["task_dir"])
            
            # Check if results already exist
            existing_pdbs = list(task_dir.rglob("*.pdb"))
            if existing_pdbs:
                self.logger.info(f"Existing results found in {task_dir}, skipping job submission")
                continue
            
            # Submit job
            job_id = self.slurm_submitter.submit_custom_job(
                job_name=job_spec["name"],
                command=job_spec["command"],
                task_dir=job_spec["task_dir"],
                memory=job_spec.get("memory", "32G"),
                gres=job_spec.get("gres", "gpu:1")
            )
            
            if job_id:
                submitted_jobs.append(job_id)
        
        return submitted_jobs
    
    def _save_sequences_as_msa(self, sequences: Dict[str, str], filename: str) -> Path:
        """Save sequences dictionary as A3M file."""
        output_path = self.base_dir / filename
        parser = A3MParser(strict_validation=False)
        parser.write_sequences(sequences, output_path)
        return output_path
    
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
        self.logger.info("========================================")


# Legacy compatibility class
class HitExpandRunner_Legacy:
    """Legacy compatibility wrapper for old hit expand interface."""
    
    def __init__(self, *args, **kwargs):
        """Initialize legacy wrapper."""
        self.logger = get_logger(__name__)
        self.logger.warning("Using legacy HitExpandRunner interface - consider upgrading")
    
    def run(self):
        """Legacy run method."""
        raise NotImplementedError("Legacy interface not implemented - use new HitExpandRunner")