"""
Simplified Hit Expand Pipeline Module

This is a streamlined version of the hit expand pipeline that reduces complexity
while maintaining full API compatibility with the original implementation.
"""

import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
import shutil
import pandas as pd
from tqdm import tqdm
from functools import wraps
from contextlib import contextmanager

# Consolidated imports - single location for all dependencies
from af_claseq.utils.logging_utils import get_logger
from af_claseq.utils.slurm_utils import SlurmJobSubmitter
from af_claseq.utils.structure_analysis import StructureAnalyzer
from af_claseq.utils.sequence_processing import A3MParser, validate_a3m_file
from af_claseq.utils.plotting_manager import (
    plot_m_fold_sampling_1d,
    plot_m_fold_sampling_2d,
    create_2d_scatter_plot,
    load_results_df
)
from af_claseq.modules.mmseqs_wrapper import MMseqsWrapper, MMseqsConfig
from af_claseq.modules.similarity_search import BLOSUM62SimilaritySearch, SimilaritySearchConfig
from af_claseq.modules.cluster_based_expansion import ClusterBasedExpansion, ClusterExpansionError
from af_claseq.modules.subset_generator import SubsetGenerator, SubsetConfig
from af_claseq.pipeline.config import HitExpandConfig


class HitExpandError(Exception):
    """Exception raised for hit expand pipeline errors."""
    pass


# =============================================================================
# Helper Decorators and Context Managers
# =============================================================================

def stage_completion(stage_name: str):
    """Decorator to handle DONE file management for pipeline stages."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            stage_dir = kwargs.get('stage_dir') or args[0] if args else None
            if stage_dir and self._check_done_file(stage_dir, stage_name):
                self.logger.info(f"Stage {stage_name} already completed, skipping")
                return True
            
            try:
                result = func(self, *args, **kwargs)
                if result and stage_dir:
                    self._create_done_file(stage_dir, stage_name)
                return result
            except Exception as e:
                self.logger.error(f"Stage {stage_name} failed: {e}")
                if stage_dir:
                    self._remove_done_file(stage_dir, stage_name)
                raise
        return wrapper
    return decorator


@contextmanager
def progress_reporter(items, description: str, logger=None):
    """Context manager for consistent progress reporting."""
    if logger is None:
        logger = get_logger(__name__)
    
    pbar = tqdm(items, desc=description)
    try:
        for item in pbar:
            pbar.set_postfix({"current": getattr(item, 'name', str(item)[:20])})
            yield item
    finally:
        pbar.close()


# =============================================================================
# Results Manager Class
# =============================================================================

class HitExpandResultsManager:
    """Handles CSV generation and results file management."""
    
    def __init__(self, logger: logging.Logger, config: HitExpandConfig):
        self.logger = logger
        self.config = config
        self.parser = A3MParser(strict_validation=False)
    
    def save_clustering_results(self, cluster_file: Path, output_dir: Path) -> str:
        """Save clustering results and create plots."""
        csv_path = output_dir / "clustering_results.csv"
        
        # Read cluster results
        cluster_data = []
        with open(cluster_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    cluster_data.append({
                        'sequence_id': parts[0],
                        'cluster_id': parts[1]
                    })
        
        # Save to CSV
        df = pd.DataFrame(cluster_data)
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Saved clustering results to {csv_path}")
        return str(csv_path)
    
    def save_structure_analysis_results(self, analysis_data: Dict[str, Any], 
                                      output_dir: Path) -> List[str]:
        """Save structure analysis results and create plots."""
        csv_dir = output_dir / "csv"
        csv_dir.mkdir(exist_ok=True)
        
        saved_files = []
        
        # Save main results to CSV
        if "all_results" in analysis_data:
            all_results_df = pd.DataFrame(analysis_data["all_results"])
            csv_path = csv_dir / "structure_analysis_all.csv"
            all_results_df.to_csv(csv_path, index=False)
            saved_files.append(str(csv_path))
        
        if "filtered_results" in analysis_data:
            filtered_results_df = pd.DataFrame(analysis_data["filtered_results"])
            csv_path = csv_dir / "structure_analysis_filtered.csv"
            filtered_results_df.to_csv(csv_path, index=False)
            saved_files.append(str(csv_path))
        
        return saved_files
    
    def create_analysis_plots(self, analysis_data: Dict[str, Any], 
                            output_dir: Path, config_file: str) -> List[str]:
        """Create analysis plots using existing plotting utilities."""
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        plot_paths = []
        
        # Load configuration for metrics
        with open(config_file, 'r') as f:
            filter_config = json.load(f)
        
        filter_criteria = filter_config.get('filter_criteria', [])
        
        if len(filter_criteria) == 1:
            # Single metric - create 1D plot
            metric_name = filter_criteria[0].get('name')
            if metric_name:
                plot_path = plot_m_fold_sampling_1d(
                    results_dir=str(output_dir),
                    metric_name=metric_name,
                    output_dir=str(plots_dir),
                    csv_dir=str(output_dir / "csv"),
                    config_file=config_file,
                    logger=self.logger
                )
                plot_paths.extend(plot_path)
        
        elif len(filter_criteria) == 2:
            # Two metrics - create 2D plots
            metric1 = filter_criteria[0].get('name')
            metric2 = filter_criteria[1].get('name')
            if metric1 and metric2:
                plot_paths_2d = plot_m_fold_sampling_2d(
                    results_dir=str(output_dir),
                    metric_name1=metric1,
                    metric_name2=metric2,
                    output_dir=str(plots_dir),
                    csv_dir=str(output_dir / "csv"),
                    config_file=config_file,
                    logger=self.logger
                )
                plot_paths.extend(plot_paths_2d)
        
        return plot_paths


# =============================================================================
# Stage Runner Base Class
# =============================================================================

class StageRunner:
    """Base class for common stage execution patterns."""
    
    def __init__(self, logger: logging.Logger, config: HitExpandConfig, 
                 slurm_submitter: SlurmJobSubmitter):
        self.logger = logger
        self.config = config
        self.slurm_submitter = slurm_submitter
        self.parser = A3MParser(strict_validation=False)
    
    def submit_and_monitor_jobs(self, job_scripts: List[Path], 
                              job_name: str) -> bool:
        """Submit jobs and monitor completion."""
        if not job_scripts:
            self.logger.warning(f"No job scripts found for {job_name}")
            return False
        
        # Submit jobs
        job_ids = []
        for script in job_scripts:
            job_id = self.slurm_submitter.submit_job(str(script))
            if job_id:
                job_ids.append(job_id)
        
        if not job_ids:
            self.logger.error(f"Failed to submit any jobs for {job_name}")
            return False
        
        self.logger.info(f"Submitted {len(job_ids)} jobs for {job_name}")
        
        # Monitor jobs
        if self.config.monitor_jobs:
            return self.slurm_submitter.wait_for_jobs(
                job_ids, 
                check_interval=self.config.job_check_interval,
                timeout=self.config.job_timeout
            )
        
        return True


# =============================================================================
# Simplified HitExpandRunner Class
# =============================================================================

class HitExpandRunner:
    """
    Simplified Hit Expand Pipeline Runner.
    
    This is a streamlined version that maintains full API compatibility
    while reducing code complexity and improving maintainability.
    """
    
    def __init__(self, 
                 input_msa: str,
                 output_dir: str,
                 config: HitExpandConfig,
                 slurm_account: str,
                 conda_env_path: str,
                 config_file: str,
                 max_workers: int = 96,
                 logger: Optional[logging.Logger] = None):
        """Initialize the Hit Expand Runner with simplified configuration."""
        
        self.input_msa = Path(input_msa)
        self.output_dir = Path(output_dir)
        self.config = config
        self.slurm_account = slurm_account
        self.conda_env_path = conda_env_path
        self.config_file = config_file
        self.max_workers = max_workers
        self.logger = logger or get_logger(__name__)
        
        # Initialize helper classes
        self.results_manager = HitExpandResultsManager(self.logger, self.config)
        self.parser = A3MParser(strict_validation=False)
        
        # Initialize SLURM submitter
        self.slurm_submitter = SlurmJobSubmitter(
            account=slurm_account,
            conda_env_path=conda_env_path,
            logger=self.logger
        )
        
        # Initialize stage runner
        self.stage_runner = StageRunner(self.logger, self.config, self.slurm_submitter)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized HitExpandRunner for {self.input_msa}")
        self.logger.info(f"Output directory: {self.output_dir}")
    
    # ==========================================================================
    # Public API Methods (unchanged signatures for compatibility)
    # ==========================================================================
    
    def run(self, rounds: int = 1, return_expanded_msa: bool = True) -> Union[str, List[Dict[str, str]]]:
        """
        Main entry point for hit expand pipeline.
        Maintains exact same API as original implementation.
        """
        self.logger.info(f"=== STARTING HIT EXPAND PIPELINE ({rounds} rounds) ===")
        
        try:
            # Validate input
            if not self.input_msa.exists():
                raise HitExpandError(f"Input MSA file not found: {self.input_msa}")
            
            if not validate_a3m_file(self.input_msa):
                raise HitExpandError(f"Invalid A3M file format: {self.input_msa}")
            
            # Execute rounds
            final_result = None
            for round_num in range(1, rounds + 1):
                self.logger.info(f"=== ROUND {round_num}/{rounds} ===")
                
                if round_num == 1:
                    result = self._execute_round_one(return_expanded_msa)
                else:
                    result = self._execute_subsequent_rounds(round_num, return_expanded_msa)
                
                if result is not None:
                    final_result = result
            
            self.logger.info("=== HIT EXPAND PIPELINE COMPLETED ===")
            return final_result or []
            
        except Exception as e:
            self.logger.error(f"Hit expand pipeline failed: {e}", exc_info=True)
            raise HitExpandError(f"Pipeline execution failed: {e}")
    
    def run_single_workflow(self, input_msa: Optional[str] = None, 
                          round_num: int = 1) -> bool:
        """
        Run single workflow - maintains API compatibility.
        """
        if input_msa:
            self.input_msa = Path(input_msa)
        
        try:
            if round_num == 1:
                return self._execute_round_one(return_expanded_msa=False) is not None
            else:
                return self._execute_subsequent_rounds(round_num, False) is not None
        except Exception as e:
            self.logger.error(f"Single workflow failed: {e}")
            return False
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get workflow status - maintains API compatibility."""
        return {
            "input_msa": str(self.input_msa),
            "output_dir": str(self.output_dir),
            "rounds_completed": self._count_completed_rounds(),
            "total_sequences": self._count_total_sequences(),
            "stages_completed": self._get_completed_stages()
        }
    
    # ==========================================================================
    # Round Execution Methods (simplified logic)
    # ==========================================================================
    
    def _execute_round_one(self, return_expanded_msa: bool = True) -> Union[str, List[Dict[str, str]], None]:
        """Execute round 1 with simplified workflow."""
        round_dir = self.output_dir / "round_1"
        round_dir.mkdir(exist_ok=True)
        
        try:
            # Stage 1: Clustering
            if not self.config.skip_clustering:
                clustering_success = self._run_clustering(round_dir)
                if not clustering_success:
                    self.logger.error("Clustering failed in round 1")
                    return None
            
            # Stage 2: Subset Generation  
            subset_success = self._run_subset_generation(round_dir)
            if not subset_success:
                self.logger.error("Subset generation failed in round 1")
                return None
            
            # Stage 3: Structure Prediction (runs in same dir as subset generation)
            if not self.config.skip_structure_prediction:
                prediction_success = self._run_structure_prediction(round_dir)
                if not prediction_success:
                    self.logger.error("Structure prediction failed in round 1")
                    return None
            
            # Stage 4: Structure Analysis
            if not self.config.skip_structure_analysis:
                analysis_success = self._run_structure_analysis(round_dir)
                if not analysis_success:
                    self.logger.error("Structure analysis failed in round 1")
                    return None
            
            # Stage 5: Similarity Search
            if not self.config.skip_hit_expansion:
                similarity_result = self._run_similarity_search(round_dir, 1, return_expanded_msa)
                return similarity_result
            
            return []
            
        except Exception as e:
            self.logger.error(f"Round 1 execution failed: {e}")
            return None
    
    def _execute_subsequent_rounds(self, round_num: int, 
                                 return_expanded_msa: bool = True) -> Union[str, List[Dict[str, str]], None]:
        """Execute rounds 2+ with simplified workflow."""
        round_dir = self.output_dir / f"round_{round_num}"
        round_dir.mkdir(exist_ok=True)
        
        try:
            # Use previous round's output as input
            prev_round_dir = self.output_dir / f"round_{round_num - 1}"
            prev_expanded_file = prev_round_dir / "05_similarity_search" / "expanded_sequences.a3m"
            
            if not prev_expanded_file.exists():
                self.logger.error(f"Previous round output not found: {prev_expanded_file}")
                return None
            
            # Temporarily update input MSA for this round
            original_input = self.input_msa
            self.input_msa = prev_expanded_file
            
            try:
                # Execute same workflow as round 1 but with different input
                result = self._execute_round_one(return_expanded_msa)
                return result
            finally:
                # Restore original input
                self.input_msa = original_input
                
        except Exception as e:
            self.logger.error(f"Round {round_num} execution failed: {e}")
            return None
    
    # ==========================================================================
    # Stage Methods (simplified with decorators)
    # ==========================================================================
    
    @stage_completion("clustering")
    def _run_clustering(self, round_dir: Path) -> bool:
        """Run MMSeqs2 clustering with simplified logic."""
        stage_dir = round_dir / "01_clustering"
        stage_dir.mkdir(exist_ok=True)
        
        try:
            # Configure MMSeqs2
            mmseqs_config = MMseqsConfig(
                binary_path=self.config.mmseqs_bin,
                coverage=self.config.mmseqs_coverage,
                min_seq_id=self.config.mmseqs_min_seq_id,
                cov_mode=self.config.mmseqs_cov_mode,
                cluster_mode=self.config.mmseqs_cluster_mode,
                threads=self.config.mmseqs_threads,
                tmp_dir=self.config.mmseqs_tmp_dir
            )
            
            mmseqs = MMseqsWrapper(mmseqs_config, logger=self.logger)
            
            # Run clustering
            cluster_file = mmseqs.cluster_sequences(
                input_fasta=str(self.input_msa),
                output_dir=str(stage_dir),
                cluster_name="hit_expand_clusters"
            )
            
            if cluster_file:
                # Save results
                self.results_manager.save_clustering_results(Path(cluster_file), stage_dir)
                self.logger.info(f"Clustering completed: {cluster_file}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Clustering failed: {e}")
            return False
    
    @stage_completion("subset_generation")
    def _run_subset_generation(self, round_dir: Path) -> bool:
        """Run subset generation with simplified logic."""
        stage_dir = round_dir / "02_subset_generation"
        stage_dir.mkdir(exist_ok=True)
        
        try:
            # Configure subset generator
            subset_config = SubsetConfig(
                num_subsets=self.config.num_subsets,
                num_random_sequences=self.config.num_random_sequences,
                num_batches=self.config.num_batches,
                batch_prefix=self.config.batch_prefix,
                output_prefix=self.config.output_prefix,
                random_seed=self.config.random_seed
            )
            
            generator = SubsetGenerator(subset_config, logger=self.logger)
            
            # Load sequences
            sequences = self.parser.parse_file(self.input_msa)
            
            # Generate subsets
            subset_files = generator.generate_subsets(
                sequences=sequences,
                output_dir=stage_dir
            )
            
            if subset_files:
                self.logger.info(f"Generated {len(subset_files)} subset files")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Subset generation failed: {e}")
            return False
    
    def _run_structure_prediction(self, round_dir: Path) -> bool:
        """Run structure prediction with batch directory as input/output."""
        batch_dir = round_dir / "02_subset_generation"  # Same directory as subset generation
        
        try:
            # Find subset files
            subset_files = list(batch_dir.glob("*.a3m"))
            
            if not subset_files:
                self.logger.error("No subset files found for structure prediction")
                return False
            
            # Create prediction jobs that use batch_dir as both input and output
            job_scripts = self._create_prediction_jobs(batch_dir)
            
            # Submit and monitor
            return self.stage_runner.submit_and_monitor_jobs(job_scripts, "structure_prediction")
            
        except Exception as e:
            self.logger.error(f"Structure prediction failed: {e}")
            return False
    
    @stage_completion("structure_analysis")
    def _run_structure_analysis(self, round_dir: Path) -> bool:
        """Run structure analysis with simplified logic."""
        stage_dir = round_dir / "04_structure_analysis"
        stage_dir.mkdir(exist_ok=True)
        
        try:
            # Find PDB files (they should be in the batch directory after prediction)
            batch_dir = round_dir / "02_subset_generation"
            pdb_files = list(batch_dir.glob("**/*.pdb"))
            
            if not pdb_files:
                self.logger.error("No PDB files found for structure analysis")
                return False
            
            # Configure analyzer
            analyzer = StructureAnalyzer()
            
            # Load filter criteria
            with open(self.config_file, 'r') as f:
                filter_config = json.load(f)
            
            filter_criteria = filter_config.get('filter_criteria', [])
            basics = filter_config.get('basics', {})
            
            # Analyze structures
            results_df = analyzer.get_result_df(
                parent_dir=str(batch_dir),
                filter_criteria=filter_criteria,
                basics=basics
            )
            
            # Filter by pLDDT threshold
            filtered_df = results_df[results_df['plddt'] >= self.config.plddt_threshold]
            
            # Prepare analysis data
            analysis_data = {
                "all_results": results_df.to_dict('records'),
                "filtered_results": filtered_df.to_dict('records')
            }
            
            # Save results and create plots
            csv_files = self.results_manager.save_structure_analysis_results(analysis_data, stage_dir)
            plot_files = self.results_manager.create_analysis_plots(analysis_data, stage_dir, self.config_file)
            
            # Save analysis summary
            analysis_file = stage_dir / "structure_analysis_results.json"
            with open(analysis_file, 'w') as f:
                json.dump(analysis_data, f, indent=2)
            
            self.logger.info(f"Structure analysis completed: {len(filtered_df)} good structures")
            return True
            
        except Exception as e:
            self.logger.error(f"Structure analysis failed: {e}")
            return False
    
    def _run_similarity_search(self, round_dir: Path, round_num: int, 
                             return_expanded_msa: bool = True) -> Union[str, List[Dict[str, str]], None]:
        """Run similarity search with simplified logic."""
        stage_dir = round_dir / "05_similarity_search"
        stage_dir.mkdir(exist_ok=True)
        
        try:
            # Get good sequences from structure analysis
            analysis_dir = round_dir / "04_structure_analysis"
            analysis_file = analysis_dir / "structure_analysis_results.json"
            
            if not analysis_file.exists():
                self.logger.error("Structure analysis results not found")
                return None
            
            with open(analysis_file, 'r') as f:
                analysis_data = json.load(f)
            
            good_structures = analysis_data.get("filtered_results", [])
            
            if not good_structures:
                self.logger.warning("No good structures found for similarity search")
                return []
            
            # Extract good sequences
            good_sequences = self._extract_good_sequences(good_structures, round_dir)
            
            if not good_sequences:
                self.logger.warning("No sequences extracted from good structures")
                return []
            
            # Perform similarity search based on configuration
            if self.config.expansion_method == "BLOSUM62":
                expanded_sequences = self._run_blosum62_search(good_sequences, stage_dir)
            else:
                expanded_sequences = self._run_mmseqs_expansion(good_sequences, stage_dir)
            
            # Always save expanded sequences to MSA file (regardless of return_expanded_msa)
            expanded_msa_path = stage_dir / "expanded_sequences.a3m"
            if expanded_sequences:
                self.parser.write_sequences(expanded_sequences, expanded_msa_path)
                self.logger.info(f"Saved {len(expanded_sequences)} expanded sequences to {expanded_msa_path}")
            
            # Return sequences directly if not returning MSA
            if not return_expanded_msa:
                self.logger.info(f"Round {round_num}: Found {len(expanded_sequences)} sequences")
                return expanded_sequences
            
            # Return MSA file path
            return str(expanded_msa_path) if expanded_msa_path.exists() else None
            
        except Exception as e:
            self.logger.error(f"Similarity search failed: {e}")
            return None
    
    # ==========================================================================
    # Helper Methods (simplified and consolidated)
    # ==========================================================================
    
    def _extract_good_sequences(self, good_structures: List[Dict], round_dir: Path) -> Dict[str, str]:
        """Extract sequences from good structures."""
        good_sequences = {}
        
        # Find original subset files to extract sequences from (same as batch dir)
        batch_dir = round_dir / "02_subset_generation"
        
        for structure in good_structures:
            pdb_name = structure.get('PDB', '')
            if not pdb_name:
                continue
            
            # Find corresponding A3M file
            subset_name = pdb_name.replace('_unrelaxed_rank_001_alphafold2_ptm', '')
            a3m_file = batch_dir / f"{subset_name}.a3m"
            
            if a3m_file.exists():
                try:
                    sequences = self.parser.parse_file(a3m_file)
                    good_sequences.update(sequences)
                except Exception as e:
                    self.logger.warning(f"Failed to parse {a3m_file}: {e}")
        
        return good_sequences
    
    def _run_blosum62_search(self, good_sequences: Dict[str, str], stage_dir: Path) -> Dict[str, str]:
        """Run BLOSUM62 similarity search."""
        config = SimilaritySearchConfig(
            threshold=self.config.similarity_threshold,
            top_k=self.config.similarity_top_k,
            exclude_query_headers=self.config.exclude_query_headers
        )
        
        search = BLOSUM62SimilaritySearch(config, logger=self.logger)
        
        # Load original MSA
        original_sequences = self.parser.parse_file(self.input_msa)
        
        # Perform search
        expanded_sequences = search.find_similar_sequences(
            query_sequences=good_sequences,
            target_sequences=original_sequences
        )
        
        return expanded_sequences
    
    def _run_mmseqs_expansion(self, good_sequences: Dict[str, str], stage_dir: Path) -> Dict[str, str]:
        """Run MMSeqs2-based cluster expansion."""
        try:
            expansion = ClusterBasedExpansion(
                mmseqs_bin=self.config.mmseqs_bin,
                max_sequences_per_cluster=self.config.max_sequences_per_cluster,
                logger=self.logger
            )
            
            # Load original MSA  
            original_sequences = self.parser.parse_file(self.input_msa)
            
            # Perform expansion
            expanded_sequences = expansion.expand_from_good_sequences(
                good_sequences=good_sequences,
                original_msa_sequences=original_sequences,
                output_dir=stage_dir
            )
            
            return expanded_sequences
            
        except ClusterExpansionError as e:
            self.logger.error(f"Cluster expansion failed: {e}")
            return {}
    
    def _create_prediction_jobs(self, batch_dir: Path) -> List[Path]:
        """Create ColabFold prediction job scripts using batch directory as input/output."""
        job_scripts = []
        
        # Create single job that processes the entire batch directory
        job_name = f"colabfold_batch_{batch_dir.name}"
        
        script_content = f"""#!/bin/bash
#SBATCH --account={self.slurm_account}
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output={batch_dir}/slurm_%j.out
#SBATCH --error={batch_dir}/slurm_%j.err

source {self.conda_env_path}

# ColabFold batch mode: input and output directory are the same
colabfold_batch \\
    --num-models {self.config.num_models} \\
    --random-seed {self.config.random_seed} \\
    {batch_dir} \\
    {batch_dir}
"""
        
        script_path = batch_dir / f"{job_name}.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        script_path.chmod(0o755)
        
        job_scripts.append(script_path)
        
        return job_scripts
    
    def _count_completed_rounds(self) -> int:
        """Count number of completed rounds."""
        count = 0
        for i in range(1, 100):  # Reasonable upper limit
            round_dir = self.output_dir / f"round_{i}"
            if round_dir.exists():
                count += 1
            else:
                break
        return count
    
    def _count_total_sequences(self) -> int:
        """Count total sequences in final MSA."""
        try:
            from af_claseq.utils.sequence_processing import count_sequences_in_a3m
            
            # Find the latest expanded sequences file
            for i in range(self._count_completed_rounds(), 0, -1):
                expanded_file = self.output_dir / f"round_{i}" / "05_similarity_search" / "expanded_sequences.a3m"
                if expanded_file.exists():
                    return count_sequences_in_a3m(str(expanded_file))
            
            # Fallback to input MSA
            return count_sequences_in_a3m(str(self.input_msa))
            
        except Exception:
            return 0
    
    def _get_completed_stages(self) -> List[str]:
        """Get list of completed stages."""
        stages = []
        
        for round_num in range(1, self._count_completed_rounds() + 1):
            round_dir = self.output_dir / f"round_{round_num}"
            
            stage_names = [
                "01_clustering", "02_subset_generation", 
                "04_structure_analysis", "05_similarity_search"
            ]
            
            for stage in stage_names:
                stage_dir = round_dir / stage
                if stage_dir.exists() and self._check_done_file(stage_dir, stage.split('_', 1)[1]):
                    stages.append(f"round_{round_num}_{stage}")
        
        return stages
    
    # DONE file management (simplified inline methods)
    def _create_done_file(self, stage_dir: Path, stage_name: str) -> None:
        """Create DONE file for completed stage."""
        done_file = stage_dir / f"{stage_name}.done"
        done_file.touch()
    
    def _check_done_file(self, stage_dir: Path, stage_name: str) -> bool:
        """Check if DONE file exists for stage."""
        done_file = stage_dir / f"{stage_name}.done"
        return done_file.exists()
    
    def _remove_done_file(self, stage_dir: Path, stage_name: str) -> None:
        """Remove DONE file for stage."""
        done_file = stage_dir / f"{stage_name}.done"
        if done_file.exists():
            done_file.unlink()