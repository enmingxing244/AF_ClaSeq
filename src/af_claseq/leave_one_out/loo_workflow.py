"""
Leave-One-Out Workflow Manager

This module implements the leave-one-out impact analysis workflow,
leveraging existing AF_ClaSeq infrastructure for maximum code reuse.
"""

import os
import json
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

# Use existing AF_ClaSeq utilities (priority order: utils > m_fold_sampling_voting)
from af_claseq.utils.structure_analysis import StructureAnalyzer, load_filter_modes
from af_claseq.utils.executor_factory import create_executor
from af_claseq.utils.sequence_processing import read_a3m_to_dict, write_a3m
from af_claseq.utils.logging_utils import get_logger
from af_claseq.utils.exceptions import WorkflowError

from .config import WorkflowConfig


def _analyze_group_parallel(group_info: Dict[str, Any], structure_config: Dict[str, Any],
                           impact_metric: str) -> List[Dict]:
    """
    Parallel worker function for analyzing a single group's impact.
    This function must be defined at module level for multiprocessing.
    """
    try:
        # Initialize structure analyzer for this worker
        structure_analyzer = StructureAnalyzer()

        # Extract config components
        filter_criteria = structure_config.get('filter_criteria', [])
        basics = structure_config.get('basics', {})
        composite_metrics = structure_config.get('composite_metrics', [])

        # Analyze this group
        results = []
        group_dir = Path(group_info['group_dir'])
        group_id = group_info['group_id']

        # Find full group PDB files
        full_pdbs = list(group_dir.glob(f"{group_id}_full_unrelaxed_rank_*.pdb"))
        if not full_pdbs:
            return results

        # Calculate full group metrics
        full_metrics = _calculate_metrics_for_pdbs_parallel(
            structure_analyzer, full_pdbs, filter_criteria, basics, composite_metrics
        )
        if not full_metrics or impact_metric not in full_metrics[0]:
            return results

        full_metric_values = [result[impact_metric] for result in full_metrics if result.get(impact_metric) is not None]
        if not full_metric_values:
            return results

        full_mean = np.mean(full_metric_values)

        # Analyze each LOO subset
        for loo_subset in group_info['loo_subsets']:
            loo_id = loo_subset['loo_id']

            # Find LOO PDB files
            loo_pdbs = list(group_dir.glob(f"{group_id}_{loo_id}_unrelaxed_rank_*.pdb"))
            if not loo_pdbs:
                continue

            # Calculate LOO metrics
            loo_metrics = _calculate_metrics_for_pdbs_parallel(
                structure_analyzer, loo_pdbs, filter_criteria, basics, composite_metrics
            )
            if not loo_metrics:
                continue

            loo_metric_values = [result[impact_metric] for result in loo_metrics if result.get(impact_metric) is not None]
            if not loo_metric_values:
                continue

            loo_mean = np.mean(loo_metric_values)

            # Calculate impact score (loo_mean - full_mean)
            impact_score = loo_mean - full_mean

            result = {
                'group_id': group_id,
                'loo_id': loo_id,
                'impact_score': impact_score,
                'full_mean': full_mean,
                'loo_mean': loo_mean,
                'left_out_header': loo_subset['left_out_header'],
                'left_out_sequence': loo_subset['left_out_sequence']
            }

            results.append(result)

        return results

    except Exception as e:
        # Return error info for debugging
        return [{'error': str(e), 'group_id': group_info.get('group_id', 'unknown')}]


def _calculate_metrics_for_pdbs_parallel(structure_analyzer, pdb_files: List[Path],
                                       filter_criteria: List[Dict], basics: Dict,
                                       composite_metrics: List[Dict]) -> List[Dict]:
    """Calculate structure metrics for a list of PDB files (parallel version)"""
    results = []

    for pdb_file in pdb_files:
        try:
            result = structure_analyzer.process_single_pdb(
                str(pdb_file),
                filter_criteria,
                basics,
                composite_metrics=composite_metrics
            )
            if result:
                results.append(result)
        except Exception:
            # Silently skip failed PDB processing in parallel mode
            pass

    return results


class LeaveOneOutManager:
    """
    Main class for managing leave-one-out impact analysis workflow.

    This class orchestrates the complete LOO workflow:
    1. Random sequence grouping
    2. LOO subset generation
    3. ColabFold structure prediction
    4. Impact analysis
    5. Results filtering and visualization
    """

    def __init__(self, config: WorkflowConfig):
        """
        Initialize the Leave-One-Out workflow manager.

        Args:
            config: Complete workflow configuration
        """
        self.config = config
        self.logger = get_logger("loo_workflow")

        # Set random seed for reproducibility
        random.seed(config.general.random_seed)
        np.random.seed(config.general.random_seed)

        # Setup output directories
        self.output_dir = config.get_output_dir()
        self.groups_dir = self.output_dir / "groups"
        self.results_dir = self.output_dir / "results"
        self.plots_dir = config.get_plots_dir()

        # Create directories
        for dir_path in [self.output_dir, self.groups_dir, self.results_dir, self.plots_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize structure analyzer
        self.structure_analyzer = StructureAnalyzer()

        # Load structure analysis configuration
        self.structure_config = self._load_structure_config()

        # Initialize SLURM job submitter
        self.job_submitter = self._initialize_job_submitter()

        self.logger.info(f"Initialized LeaveOneOutManager for protein: {config.general.protein_name}")
        self.logger.info(f"Output directory: {self.output_dir}")

    def _load_structure_config(self) -> Dict[str, Any]:
        """Load and validate structure analysis configuration"""
        config_path = self.config.general.structure_analysis_config

        try:
            structure_config = load_filter_modes(config_path)

            # Validate that impact metric exists
            impact_metric = self.config.leave_one_out.impact_metric_name
            filter_criteria = structure_config.get('filter_criteria', [])
            composite_metrics = structure_config.get('composite_metrics', [])

            # Check if metric exists in filter criteria or composite metrics
            available_metrics = [criterion.get('name') for criterion in filter_criteria]
            available_metrics.extend([composite.get('name') for composite in composite_metrics])

            if impact_metric not in available_metrics:
                raise WorkflowError(
                    f"Impact metric '{impact_metric}' not found in structure config. "
                    f"Available metrics: {available_metrics}"
                )

            self.logger.info(f"Loaded structure config with {len(filter_criteria)} metrics and {len(composite_metrics)} composite metrics")
            self.logger.info(f"Using impact metric: {impact_metric}")

            return structure_config

        except Exception as e:
            raise WorkflowError(f"Failed to load structure analysis config: {e}")

    def _initialize_job_submitter(self):
        """Initialize job executor (SLURM or local GPU) from configuration"""
        raw_config = {}
        if self.config.slurm is not None:
            slurm_config = self.config.slurm
            raw_config['slurm'] = {
                'conda_env_path': slurm_config.conda_env_path,
                'slurm_account': slurm_config.account,
                'slurm_partition': slurm_config.partition,
                'slurm_time': slurm_config.time,
                'slurm_cpus_per_task': slurm_config.cpus,
            }
            extra_kwargs = {
                'num_models': slurm_config.num_models,
                'num_seeds': slurm_config.num_seeds,
                'job_name_prefix': "loo",
            }
        elif self.config.local_gpu is not None:
            raw_config['local_gpu'] = {
                'cuda_visible_devices': self.config.local_gpu.cuda_visible_devices,
            }
            extra_kwargs = {
                'num_models': 5,
                'num_seeds': 1,
                'job_name_prefix': "loo",
            }
        else:
            raise ValueError("No execution mode configured (need 'slurm' or 'local_gpu')")

        return create_executor(raw_config, **extra_kwargs)

    def run_complete_workflow(self) -> Dict[str, Any]:
        """
        Execute the complete leave-one-out workflow.

        Returns:
            Dictionary with workflow results and summary
        """
        self.logger.info("=" * 50)
        self.logger.info("STARTING LEAVE-ONE-OUT WORKFLOW")
        self.logger.info("=" * 50)

        workflow_results = {}

        try:
            # Step 1: Generate LOO groups and subsets
            self.logger.info("Step 1: Generating leave-one-out groups and subsets...")
            group_info = self.generate_loo_groups()
            workflow_results['group_generation'] = group_info

            # Step 2: Submit ColabFold jobs
            self.logger.info("Step 2: Submitting ColabFold prediction jobs...")
            job_results = self.submit_colabfold_jobs(group_info['groups'])
            workflow_results['job_submission'] = job_results

            # Step 3: Monitor job completion
            self.logger.info("Step 3: Monitoring job completion...")
            completion_results = self.monitor_jobs(job_results['job_ids'])
            workflow_results['job_monitoring'] = completion_results

            # Step 4: Analyze impact scores
            self.logger.info("Step 4: Analyzing impact scores...")
            impact_results = self.analyze_impact_scores(group_info['groups'])
            workflow_results['impact_analysis'] = impact_results

            # Step 5: Generate visualizations (required)
            self.logger.info("Step 5: Generating impact visualizations...")
            plot_results = self.generate_impact_plots(impact_results['all_results'])
            workflow_results['plotting'] = plot_results

            # Step 6: Save final results
            self.logger.info("Step 6: Saving final results...")
            final_results = self.save_final_results(impact_results)
            workflow_results['final_output'] = final_results

            self.logger.info("=" * 50)
            self.logger.info("LEAVE-ONE-OUT WORKFLOW COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 50)

            return workflow_results

        except Exception as e:
            self.logger.error(f"Workflow failed: {e}")
            raise WorkflowError(f"Leave-one-out workflow failed: {e}")

    def generate_loo_groups(self) -> Dict[str, Any]:
        """
        Generate leave-one-out groups and subsets from input A3M file.

        Returns:
            Dictionary with group information and metadata
        """
        self.logger.info(f"Reading sequences from: {self.config.general.source_a3m}")

        # Read sequences using existing utility
        sequences_dict = read_a3m_to_dict(self.config.general.source_a3m)
        sequence_items = list(sequences_dict.items())

        if len(sequence_items) < 2:
            raise WorkflowError(f"Need at least 2 sequences (query + 1), found: {len(sequence_items)}")

        # Extract query sequence (first one) and other sequences
        query_header, query_sequence = sequence_items[0]
        other_sequences = sequence_items[1:]

        self.logger.info(f"Found {len(other_sequences)} non-query sequences for grouping")

        # Check minimum requirements
        min_seqs = self.config.leave_one_out.min_sequences_for_loo
        if len(other_sequences) < min_seqs:
            raise WorkflowError(f"Need at least {min_seqs} non-query sequences, found: {len(other_sequences)}")

        # Randomly shuffle other sequences
        random.shuffle(other_sequences)

        # Create groups
        groups = self._create_sequence_groups(other_sequences)

        # Generate A3M files for each group
        group_metadata = []
        for group_idx, group_sequences in enumerate(groups):
            group_id = f"group_{group_idx+1:03d}"
            group_metadata.append(self._create_group_files(group_id, query_header, query_sequence, group_sequences))

        # Save metadata
        metadata = {
            'total_sequences': len(other_sequences),
            'groups_created': len(groups),
            'sequences_per_group': [len(group) for group in groups],
            'group_details': group_metadata
        }


        self.logger.info(f"Generated {len(groups)} groups with LOO subsets")
        return {'groups': group_metadata, 'metadata': metadata}

    def _create_sequence_groups(self, sequences: List[Tuple[str, str]]) -> List[List[Tuple[str, str]]]:
        """Create groups of specified size from sequences"""
        group_size = self.config.leave_one_out.num_seq_per_group
        groups = []

        # Create groups of specified size
        for i in range(0, len(sequences), group_size):
            group = sequences[i:i + group_size]
            groups.append(group)

        # Handle remainder: merge last small group with previous one if needed
        if len(groups) > 1 and len(groups[-1]) < group_size:
            remainder = groups.pop()
            groups[-1].extend(remainder)
            self.logger.info(f"Merged remainder ({len(remainder)} sequences) with previous group")

        # Log group sizes
        for i, group in enumerate(groups):
            self.logger.info(f"Group {i+1}: {len(group)} sequences")

        return groups

    def _create_group_files(self, group_id: str, query_header: str, query_sequence: str,
                           group_sequences: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Create A3M files for a group (full group + LOO subsets)"""
        group_dir = self.groups_dir / group_id
        group_dir.mkdir(exist_ok=True)

        # Create full group A3M (query + all group sequences)
        full_sequences = {query_header: query_sequence}
        for header, sequence in group_sequences:
            full_sequences[header] = sequence

        full_a3m_path = group_dir / f"{group_id}_full.a3m"
        write_a3m(full_sequences, str(full_a3m_path))

        # Create LOO subsets (remove one sequence at a time)
        loo_subsets = []
        for i, (left_out_header, left_out_sequence) in enumerate(group_sequences):
            loo_id = f"loo_{i+1:02d}"

            # Create subset without the left-out sequence
            loo_sequences = {query_header: query_sequence}
            for header, sequence in group_sequences:
                if header != left_out_header:
                    loo_sequences[header] = sequence

            loo_a3m_path = group_dir / f"{group_id}_{loo_id}.a3m"
            write_a3m(loo_sequences, str(loo_a3m_path))

            loo_subsets.append({
                'loo_id': loo_id,
                'loo_a3m_path': str(loo_a3m_path),
                'left_out_header': left_out_header,
                'left_out_sequence': left_out_sequence
            })

        group_info = {
            'group_id': group_id,
            'group_dir': str(group_dir),
            'full_a3m_path': str(full_a3m_path),
            'sequences_in_group': len(group_sequences),
            'loo_subsets': loo_subsets
        }

        self.logger.info(f"Created {len(loo_subsets)} LOO subsets for {group_id}")
        return group_info

    def submit_colabfold_jobs(self, groups: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Submit ColabFold batch jobs for all groups"""
        submitted_jobs = []

        for group_info in groups:
            group_id = group_info['group_id']
            group_dir = group_info['group_dir']

            try:
                # Submit directory-based batch job (input and output same directory)
                job_id = self.job_submitter.submit_job(
                    task_dir=group_dir,
                    job_id=f"loo_{group_id}"
                )

                if job_id:
                    submitted_jobs.append({
                        'job_id': job_id,
                        'group_id': group_id,
                        'group_dir': group_dir
                    })
                    self.logger.info(f"Submitted job {job_id} for group {group_id}")
                else:
                    self.logger.warning(f"Failed to submit job for group {group_id}")

            except Exception as e:
                self.logger.error(f"Error submitting job for group {group_id}: {e}")

        self.logger.info(f"Submitted {len(submitted_jobs)} ColabFold jobs")
        return {
            'job_ids': [job['job_id'] for job in submitted_jobs],
            'job_details': submitted_jobs
        }

    def monitor_jobs(self, job_ids: List[str]) -> Dict[str, Any]:
        """Monitor job completion with periodic status checks"""
        if not job_ids:
            return {'status': 'no_jobs', 'completed_jobs': []}

        self.logger.info(f"Monitoring {len(job_ids)} jobs for completion...")

        # Use existing job monitoring from SlurmJobSubmitter
        completed_jobs = self.job_submitter.monitor_jobs(job_ids, check_interval=60)

        self.logger.info(f"All jobs completed. Success rate: {len(completed_jobs)}/{len(job_ids)}")
        return {
            'status': 'completed',
            'total_jobs': len(job_ids),
            'completed_jobs': completed_jobs,
            'success_rate': len(completed_jobs) / len(job_ids)
        }

    def analyze_impact_scores(self, groups: List[Dict[str, Any]], use_parallel: bool = True) -> Dict[str, Any]:
        """Analyze impact scores for all groups"""
        self.logger.info("Starting impact analysis...")

        # Temporarily reduce verbosity of sequence processing logger
        import logging
        seq_logger = logging.getLogger('af_claseq.sequence_processing')
        original_level = seq_logger.level
        seq_logger.setLevel(logging.WARNING)

        all_results = []
        impact_metric = self.config.leave_one_out.impact_metric_name

        if use_parallel and len(groups) > 1:
            # Determine number of processes (use 75% of available CPUs)
            num_processes = max(1, int(cpu_count() * 0.75))
            self.logger.info(f"Running impact analysis using {num_processes} parallel processes")

            # Create worker function with fixed parameters
            worker_func = partial(_analyze_group_parallel, structure_config=self.structure_config,
                                 impact_metric=impact_metric)

            # Use multiprocessing with progress bar
            with Pool(processes=num_processes) as pool:
                with tqdm(total=len(groups), desc="Analyzing groups", unit="group") as pbar:
                    # Submit all jobs
                    async_results = []
                    for group_info in groups:
                        async_result = pool.apply_async(worker_func, (group_info,))
                        async_results.append((group_info['group_id'], async_result))

                    # Collect results as they complete
                    for group_id, async_result in async_results:
                        try:
                            group_results = async_result.get(timeout=300)  # 5 minute timeout per group
                            # Filter out any error results
                            valid_results = [r for r in group_results if 'error' not in r]
                            error_results = [r for r in group_results if 'error' in r]

                            if error_results:
                                self.logger.error(f"Error analyzing group {group_id}: {error_results[0]['error']}")
                                pbar.set_postfix(current=group_id, status="✗")
                            else:
                                all_results.extend(valid_results)
                                pbar.set_postfix(current=group_id, status=f"✓ {len(valid_results)}")
                        except Exception as e:
                            self.logger.error(f"Error analyzing group {group_id}: {e}")
                            pbar.set_postfix(current=group_id, status="✗")

                        pbar.update(1)

        else:
            # Fallback to sequential processing
            self.logger.info("Running impact analysis sequentially")
            with tqdm(total=len(groups), desc="Analyzing groups", unit="group") as pbar:
                for group_info in groups:
                    group_id = group_info['group_id']
                    pbar.set_postfix(current=group_id)

                    try:
                        group_results = _analyze_group_parallel(
                            group_info, self.structure_config, impact_metric
                        )
                        # Filter out any error results
                        valid_results = [r for r in group_results if 'error' not in r]
                        error_results = [r for r in group_results if 'error' in r]

                        if error_results:
                            self.logger.error(f"Error analyzing group {group_id}: {error_results[0]['error']}")
                            pbar.set_postfix(current=group_id, status="✗")
                        else:
                            all_results.extend(valid_results)
                            pbar.set_postfix(current=group_id, status="✓")
                    except Exception as e:
                        self.logger.error(f"Error analyzing group {group_id}: {e}")
                        pbar.set_postfix(current=group_id, status="✗")

                    pbar.update(1)

        # Restore original logging level
        seq_logger.setLevel(original_level)

        # Apply filtering
        filtered_results = self._apply_impact_filters(all_results)

        # Save results
        self._save_impact_results(all_results, filtered_results)

        self.logger.info(f"Impact analysis complete: {len(filtered_results)}/{len(all_results)} sequences passed filters")

        return {
            'all_results': all_results,
            'filtered_results': filtered_results,
            'total_analyzed': len(all_results),
            'significant_sequences': len(filtered_results)
        }


    def _apply_impact_filters(self, all_results: List[Dict]) -> List[Dict]:
        """Apply dual filtering criteria: impact score and full group mean"""
        config = self.config.leave_one_out
        filtered = []

        for result in all_results:
            impact_score = result['impact_score']
            full_mean = result['full_mean']

            # Apply impact score filter
            passes_impact = False
            if config.cutoff_method == 'above':
                passes_impact = impact_score > config.impact_threshold
            elif config.cutoff_method == 'below':
                passes_impact = impact_score < config.impact_threshold

            # Apply full group mean filter
            passes_full_mean = False
            if config.full_mean_cutoff_method == 'above':
                passes_full_mean = full_mean > config.full_group_mean_threshold
            elif config.full_mean_cutoff_method == 'below':
                passes_full_mean = full_mean < config.full_group_mean_threshold

            # Both filters must pass
            if passes_impact and passes_full_mean:
                filtered.append(result)

        # Sort by impact score
        filtered.sort(key=lambda x: x['impact_score'], reverse=(config.cutoff_method == 'above'))

        return filtered

    def _save_impact_results(self, all_results: List[Dict], filtered_results: List[Dict]):
        """Save impact analysis results to CSV files"""
        # Save all results
        all_results_df = pd.DataFrame(all_results)
        all_results_file = self.results_dir / "all_impact_results.csv"
        all_results_df.to_csv(all_results_file, index=False)

        # Save filtered results
        filtered_results_df = pd.DataFrame(filtered_results)
        filtered_results_file = self.results_dir / "significant_impact_sequences.csv"
        filtered_results_df.to_csv(filtered_results_file, index=False)

        self.logger.info(f"Saved impact results to: {all_results_file} and {filtered_results_file}")

    def generate_impact_plots(self, all_results: List[Dict]) -> Dict[str, Any]:
        """Generate required impact visualization plots"""
        if not all_results:
            self.logger.warning("No results to plot")
            return {'status': 'no_data'}

        # Import plotting module and generate plots
        from .plotting import ImpactPlotter

        plotter = ImpactPlotter(self.config, self.logger)
        plot_files = plotter.create_impact_plots(all_results, self.plots_dir)

        self.logger.info(f"Generated {len(plot_files)} impact visualization plots")
        return {
            'status': 'completed',
            'plot_files': plot_files,
            'plots_directory': str(self.plots_dir)
        }

    def save_final_results(self, impact_results: Dict[str, Any]) -> Dict[str, Any]:
        """Save final A3M file with significant impact sequences"""
        filtered_results = impact_results['filtered_results']

        if not filtered_results:
            self.logger.warning("No significant sequences to save")
            return {'status': 'no_significant_sequences'}

        # Create A3M file with significant sequences
        significant_sequences = {}

        # Add query sequence first
        query_header, query_sequence = self._get_query_sequence()
        significant_sequences[query_header] = query_sequence

        # Add filtered sequences with enhanced headers
        for result in filtered_results:
            header = result['left_out_header']
            sequence = result['left_out_sequence']
            impact = result['impact_score']

            # Enhance header with impact information
            enhanced_header = f"{header} [impact={impact:.3f}]"
            significant_sequences[enhanced_header] = sequence

        # Save A3M file
        output_a3m = self.results_dir / "significant_impact_sequences.a3m"
        write_a3m(significant_sequences, str(output_a3m))

        self.logger.info(f"Saved {len(filtered_results)} significant sequences to: {output_a3m}")

        return {
            'status': 'completed',
            'output_file': str(output_a3m),
            'sequences_count': len(filtered_results)
        }

    def _get_query_sequence(self) -> Tuple[str, str]:
        """Get query sequence from source A3M file"""
        sequences_dict = read_a3m_to_dict(self.config.general.source_a3m)
        query_header, query_sequence = list(sequences_dict.items())[0]
        return query_header, query_sequence