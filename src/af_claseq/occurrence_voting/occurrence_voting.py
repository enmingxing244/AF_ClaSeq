"""
Occurrence Voting Workflow Manager

This module implements the complete occurrence voting workflow:
1. Batch organization for SLURM submission
2. Structure prediction via ColabFold
3. Structure filtering by metrics
4. Sequence collection and occurrence counting
5. Ranking by frequency and final selection
"""

import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import Counter, defaultdict

# Use existing AF_ClaSeq utilities
from af_claseq.utils.slurm_utils import SlurmJobSubmitter
from af_claseq.utils.plotting_manager import create_2d_scatter_plot, plot_1d_distribution, create_joint_plot
from af_claseq.utils.sequence_processing import read_a3m_to_dict, write_a3m
from af_claseq.utils.logging_utils import get_logger
from af_claseq.utils.exceptions import WorkflowError
from af_claseq.utils.structure_analysis import StructureAnalyzer, load_filter_modes

from .sampler import SequenceSampler


class OccurrenceVotingManager:
    """
    Main workflow manager for occurrence voting analysis.

    This class orchestrates the complete workflow from random sampling
    through occurrence counting to final sequence selection.
    """

    def __init__(self, config):
        """
        Initialize the occurrence voting manager.

        Args:
            config: OccurrenceVotingConfig object
        """
        self.config = config
        self.logger = get_logger("occurrence_voting")

        # Extract homodimer mode from structure prediction config
        self.homodimer_mode = config.structure_prediction.prediction_mode == "homodimer"

        # Setup output directories
        self.output_dir = config.get_output_dir()
        self.batches_dir = config.get_batches_dir()
        self.results_dir = config.get_results_dir()

        # Create directories (groups_dir no longer needed - A3M files created directly in batch dirs)
        for dir_path in [self.output_dir, self.batches_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize sampler
        self.sampler = SequenceSampler(config, self.logger)

        # Initialize SLURM job submitter
        self.job_submitter = self._initialize_job_submitter()

        self.logger.info(f"Initialized OccurrenceVotingManager for protein: {config.general.protein_name}")
        self.logger.info(f"Output directory: {self.output_dir}")

    def _initialize_job_submitter(self) -> SlurmJobSubmitter:
        """Initialize SLURM job submitter with configuration"""
        slurm_config = self.config.slurm
        structure_config = self.config.structure_prediction

        # Initialize for batch prediction mode
        submitter = SlurmJobSubmitter(
            conda_env_path=slurm_config.conda_env_path,
            slurm_account=slurm_config.account,
            slurm_partition=slurm_config.partition,
            slurm_time=slurm_config.time,
            slurm_cpus_per_task=slurm_config.cpus,
            num_models=structure_config.num_models,
            num_seeds=structure_config.num_seeds,
            num_recycle=structure_config.num_recycle,
            job_name_prefix="occ_vote"
        )

        return submitter

    def run_complete_workflow(self) -> Dict[str, Any]:
        """
        Execute the complete occurrence voting workflow.

        Returns:
            Dictionary with workflow results and summary
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING OCCURRENCE VOTING WORKFLOW")
        self.logger.info("=" * 60)

        workflow_results = {}

        try:
            # Step 1: Create batches directly (combines sampling and organization)
            self.logger.info("Step 1: Creating random sequence groups organized into batches...")
            batches_info = self.sampler.create_random_groups()
            sampling_summary = self._get_batches_summary(batches_info)
            workflow_results['sampling'] = sampling_summary


            # Step 2: Submit ColabFold jobs
            self.logger.info("Step 2: Submitting ColabFold prediction jobs...")
            job_results = self.submit_colabfold_jobs(batches_info)
            workflow_results['job_submission'] = job_results

            # Step 3: Monitor job completion
            self.logger.info("Step 3: Monitoring job completion...")
            completion_results = self.monitor_jobs(job_results['job_ids'])
            workflow_results['job_monitoring'] = completion_results

            # Step 4: Analyze predicted structures
            self.logger.info("Step 4: Analyzing predicted structures...")
            analysis_results = self.analyze_predicted_structures(batches_info)
            workflow_results['structure_analysis'] = analysis_results

            # Step 5: Generate structure metrics plots
            self.logger.info("Step 5: Generating structure analysis plots...")
            plotting_results = self.generate_structure_plots(analysis_results['results_csv'])
            workflow_results['plotting'] = plotting_results

            # Step 6: Filter structures and count occurrences
            self.logger.info("Step 6: Filtering structures and counting sequence occurrences...")
            voting_results = self.perform_occurrence_voting(analysis_results['results_csv'])
            workflow_results['occurrence_voting'] = voting_results

            # Step 7: Generate sequence occurrence plots
            self.logger.info("Step 7: Generating sequence occurrence plots...")
            occurrence_plotting_results = self.generate_occurrence_plots(voting_results)
            workflow_results['occurrence_plotting'] = occurrence_plotting_results

            # Step 8: Generate final results
            self.logger.info("Step 8: Generating final results...")
            final_results = self.generate_final_results(voting_results)
            workflow_results['final_results'] = final_results

            self.logger.info("=" * 60)
            self.logger.info("OCCURRENCE VOTING WORKFLOW COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 60)

            return workflow_results

        except Exception as e:
            self.logger.error(f"Workflow failed: {e}")
            raise WorkflowError(f"Occurrence voting workflow failed: {e}")

    def run_step(self, step_name: str) -> Dict[str, Any]:
        """
        Run a single workflow step.

        Args:
            step_name: Name of the step to run ('sampling', 'colabfold', 'analysis', 'voting')

        Returns:
            Dictionary with step results
        """
        self.logger.info(f"=" * 60)
        self.logger.info(f"RUNNING SINGLE STEP: {step_name.upper()}")
        self.logger.info(f"=" * 60)

        if step_name == 'sampling':
            return self._run_sampling_step()
        elif step_name == 'colabfold':
            return self._run_colabfold_step()
        elif step_name == 'analysis':
            return self._run_analysis_step()
        elif step_name == 'voting':
            return self._run_voting_step()
        else:
            raise ValueError(f"Unknown step: {step_name}")

    def resume_from_step(self, step_name: str) -> Dict[str, Any]:
        """
        Resume workflow from a specific step.

        Args:
            step_name: Name of the step to resume from

        Returns:
            Dictionary with workflow results
        """
        self.logger.info(f"=" * 60)
        self.logger.info(f"RESUMING WORKFLOW FROM: {step_name.upper()}")
        self.logger.info(f"=" * 60)

        workflow_results = {}

        if step_name == 'sampling':
            return self.run_complete_workflow()

        elif step_name == 'colabfold':
            # Check if batches exist
            if not self._check_batches_exist():
                raise WorkflowError("No batches found. Run sampling step first.")

            batches_info = self._load_existing_batches()
            workflow_results['existing_batches'] = batches_info

            # Run ColabFold and subsequent steps
            job_results = self.submit_colabfold_jobs(batches_info)
            workflow_results['job_submission'] = job_results

            completion_results = self.monitor_jobs(job_results['job_ids'])
            workflow_results['job_monitoring'] = completion_results

            analysis_results = self.analyze_predicted_structures(batches_info)
            workflow_results['structure_analysis'] = analysis_results

            voting_results = self.perform_occurrence_voting(analysis_results['results_csv'])
            workflow_results['occurrence_voting'] = voting_results

            final_results = self.generate_final_results(voting_results)
            workflow_results['final_results'] = final_results

        elif step_name == 'analysis':
            # Check if structure predictions exist
            if not self._check_predictions_exist():
                raise WorkflowError("No structure predictions found. Run ColabFold step first.")

            batches_info = self._load_existing_batches()
            analysis_results = self.analyze_predicted_structures(batches_info)
            workflow_results['structure_analysis'] = analysis_results

            plotting_results = self.generate_structure_plots(analysis_results['results_csv'])
            workflow_results['plotting'] = plotting_results

            voting_results = self.perform_occurrence_voting(analysis_results['results_csv'])
            workflow_results['occurrence_voting'] = voting_results

            occurrence_plotting_results = self.generate_occurrence_plots(voting_results)
            workflow_results['occurrence_plotting'] = occurrence_plotting_results

            final_results = self.generate_final_results(voting_results)
            workflow_results['final_results'] = final_results

        elif step_name == 'voting':
            # Check if analysis results exist
            analysis_csv = self.results_dir / "structure_analysis_results.csv"
            if not analysis_csv.exists():
                raise WorkflowError("No structure analysis results found. Run analysis step first.")

            voting_results = self.perform_occurrence_voting(str(analysis_csv))
            workflow_results['occurrence_voting'] = voting_results

            occurrence_plotting_results = self.generate_occurrence_plots(voting_results)
            workflow_results['occurrence_plotting'] = occurrence_plotting_results

            final_results = self.generate_final_results(voting_results)
            workflow_results['final_results'] = final_results

        else:
            raise ValueError(f"Unknown step: {step_name}")

        self.logger.info(f"=" * 60)
        self.logger.info(f"WORKFLOW RESUMED FROM {step_name.upper()} COMPLETED")
        self.logger.info(f"=" * 60)

        return workflow_results

    def _get_batches_summary(self, batches_info: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate summary of created batches.

        Args:
            batches_info: List of batch information dictionaries

        Returns:
            Dictionary with batch summary information
        """
        total_groups = sum(batch['groups_count'] for batch in batches_info)
        total_files = sum(len(batch['groups']) for batch in batches_info)

        summary = {
            'total_batches_created': len(batches_info),
            'total_groups_created': total_groups,
            'total_a3m_files_created': total_files,
            'target_groups': self.config.sampling.num_groups,
            'sequences_per_group': self.config.sampling.group_size,
            'batches_directory': str(self.config.get_batches_dir()),
            'random_seed_used': self.config.general.random_seed
        }

        self.logger.info("Sampling Summary:")
        self.logger.info(f"  Batches created: {len(batches_info)}")
        self.logger.info(f"  Groups created: {total_groups}")
        self.logger.info(f"  A3M files created: {total_files}")
        self.logger.info(f"  Sequences per group: {self.config.sampling.group_size}")

        return summary

    def _run_sampling_step(self) -> Dict[str, Any]:
        """Run only the sampling step."""
        self.logger.info("Step 1: Creating random sequence groups organized into batches...")
        batches_info = self.sampler.create_random_groups()
        sampling_summary = self._get_batches_summary(batches_info)


        return {'sampling': sampling_summary}

    def _run_colabfold_step(self) -> Dict[str, Any]:
        """Run only the ColabFold submission and monitoring step."""
        # Load existing batches
        batches_info = self._load_existing_batches()

        self.logger.info("Step 2: Submitting ColabFold prediction jobs...")
        job_results = self.submit_colabfold_jobs(batches_info)

        self.logger.info("Step 3: Monitoring job completion...")
        completion_results = self.monitor_jobs(job_results['job_ids'])

        return {
            'job_submission': job_results,
            'job_monitoring': completion_results
        }

    def _run_analysis_step(self) -> Dict[str, Any]:
        """Run only the structure analysis step."""
        # Load existing batches
        batches_info = self._load_existing_batches()

        self.logger.info("Step 4: Analyzing predicted structures...")
        analysis_results = self.analyze_predicted_structures(batches_info)

        self.logger.info("Step 5: Generating structure analysis plots...")
        plotting_results = self.generate_structure_plots(analysis_results['results_csv'])

        return {
            'structure_analysis': analysis_results,
            'plotting': plotting_results
        }

    def _run_voting_step(self) -> Dict[str, Any]:
        """Run only the occurrence voting step."""
        # Check if analysis results exist
        analysis_csv = self.results_dir / "structure_analysis_results.csv"
        if not analysis_csv.exists():
            raise WorkflowError("No structure analysis results found. Run analysis step first.")

        self.logger.info("Step 6: Filtering structures and counting sequence occurrences...")
        voting_results = self.perform_occurrence_voting(str(analysis_csv))

        self.logger.info("Step 7: Generating sequence occurrence plots...")
        occurrence_plotting_results = self.generate_occurrence_plots(voting_results)

        self.logger.info("Step 8: Generating final results...")
        final_results = self.generate_final_results(voting_results)

        return {
            'occurrence_voting': voting_results,
            'occurrence_plotting': occurrence_plotting_results,
            'final_results': final_results
        }

    def _load_existing_batches(self) -> List[Dict[str, Any]]:
        """Load existing batch information by reconstructing from directory structure."""
        return self._reconstruct_batches_from_dirs()

    def _reconstruct_batches_from_dirs(self) -> List[Dict[str, Any]]:
        """Reconstruct batch information from existing directory structure."""
        import glob

        batches_dir = self.config.get_batches_dir()
        if not batches_dir.exists():
            raise WorkflowError("No batches directory found. Run sampling step first.")

        batches_info = []
        batch_dirs = sorted(glob.glob(str(batches_dir / "batch_*")))

        for batch_dir in batch_dirs:
            batch_path = Path(batch_dir)
            batch_id = batch_path.name

            # Get A3M files in this batch
            a3m_files = list(batch_path.glob("*.a3m"))
            batch_groups = []

            for a3m_file in a3m_files:
                group_id = a3m_file.stem
                batch_groups.append({
                    'group_id': group_id,
                    'a3m_file': str(a3m_file),
                    'sequences_count': 9  # Default: query + 8 sequences
                })

            batch_info = {
                'batch_id': batch_id,
                'batch_dir': str(batch_path),
                'groups': batch_groups,
                'groups_count': len(batch_groups)
            }

            batches_info.append(batch_info)

        self.logger.info(f"Reconstructed {len(batches_info)} batches from directory structure")


        return batches_info

    def _check_batches_exist(self) -> bool:
        """Check if batch directories exist."""
        batches_dir = self.config.get_batches_dir()
        return batches_dir.exists() and any(batches_dir.glob("batch_*"))

    def _check_predictions_exist(self) -> bool:
        """Check if structure predictions exist."""
        batches_dir = self.config.get_batches_dir()
        if not batches_dir.exists():
            return False

        # Check if any batch has PDB files
        for batch_dir in batches_dir.glob("batch_*"):
            if any(batch_dir.glob("*.pdb")):
                return True

        return False

    def submit_colabfold_jobs(self, batches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Submit ColabFold batch jobs for all batches"""
        submitted_jobs = []

        for batch_info in batches:
            batch_id = batch_info['batch_id']
            batch_dir = batch_info['batch_dir']

            try:
                # Submit directory-based batch job
                job_id = self.job_submitter.submit_job(
                    task_dir=str(batch_dir),
                    job_id=f"occ_vote_{batch_id}"
                )

                if job_id:
                    submitted_jobs.append({
                        'job_id': job_id,
                        'batch_id': batch_id,
                        'batch_dir': batch_dir
                    })
                    self.logger.info(f"Submitted job {job_id} for {batch_id}")
                else:
                    self.logger.warning(f"Failed to submit job for {batch_id}")

            except Exception as e:
                self.logger.error(f"Error submitting job for {batch_id}: {e}")

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

    def analyze_predicted_structures(self, batches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze all predicted structures using existing structure analysis utilities.

        Args:
            batches: List of batch information dictionaries

        Returns:
            Dictionary with structure analysis results
        """
        self.logger.info("Starting structure analysis of predicted structures...")

        # Initialize structure analyzer
        structure_analyzer = StructureAnalyzer()

        # Load structure analysis configuration
        structure_config = load_filter_modes(self.config.structure_analysis.config_json)
        filter_criteria = structure_config.get('filter_criteria', [])
        basics = structure_config.get('basics', {})
        composite_metrics = structure_config.get('composite_metrics', [])
        plddt_threshold = self.config.structure_analysis.plddt_threshold

        self.logger.info(f"Structure analysis configuration:")
        self.logger.info(f"  Filter criteria: {len(filter_criteria)} metrics")
        self.logger.info(f"  Composite metrics: {len(composite_metrics)} metrics")
        self.logger.info(f"  pLDDT threshold: {plddt_threshold}")

        # Collect all PDB files from all batches
        all_pdb_files = []
        for batch_info in batches:
            batch_dir = Path(batch_info['batch_dir'])
            pdb_files = list(batch_dir.glob("*.pdb"))
            all_pdb_files.extend([str(pdb) for pdb in pdb_files])

        if not all_pdb_files:
            raise WorkflowError("No PDB files found for structure analysis")

        self.logger.info(f"Found {len(all_pdb_files)} PDB files for analysis")

        # Run structure analysis in parallel
        try:
            analysis_results = structure_analyzer.process_pdbs_parallel(
                pdb_files=all_pdb_files,
                filter_criteria=filter_criteria,
                basics=basics,
                plddt_threshold=plddt_threshold,
                n_jobs=-1,  # Use all available cores
                composite_metrics=composite_metrics
            )

            # Convert results to DataFrame
            results_list = [result for result in analysis_results.values() if result is not None]

            if not results_list:
                raise WorkflowError("No structures passed analysis criteria")

            results_df = pd.DataFrame(results_list)

            # Save results to CSV
            results_csv = self.results_dir / "structure_analysis_results.csv"
            results_df.to_csv(results_csv, index=False)

            self.logger.info(f"Structure analysis complete:")
            self.logger.info(f"  Analyzed structures: {len(results_df)}")
            self.logger.info(f"  Results saved to: {results_csv}")

            return {
                'total_analyzed': len(all_pdb_files),
                'structures_passed': len(results_df),
                'results_csv': str(results_csv),
                'results_dataframe': results_df
            }

        except Exception as e:
            raise WorkflowError(f"Structure analysis failed: {e}")

    def perform_occurrence_voting(self, results_csv: str) -> Dict[str, Any]:
        """
        Perform structure filtering and sequence occurrence counting.

        Args:
            results_csv: Path to structure analysis results CSV file

        Returns:
            Dictionary with voting results
        """
        self.logger.info("Starting occurrence voting analysis...")

        # Step 1: Load structure analysis results
        if not os.path.exists(results_csv):
            raise FileNotFoundError(f"Structure analysis results not found: {results_csv}")

        df = pd.read_csv(results_csv)
        self.logger.info(f"Loaded {len(df)} structure analysis results")

        # Step 2: Filter structures by metric criteria
        filtered_df = self._filter_structures(df)
        if len(filtered_df) == 0:
            raise WorkflowError("No structures passed filtering criteria")

        # Step 3: Collect sequences from filtered structures
        sequence_counts = self._collect_and_count_sequences(filtered_df)

        # Step 4: Rank sequences by occurrence frequency
        ranked_sequences = self._rank_sequences_by_frequency(sequence_counts)

        voting_results = {
            'total_structures_analyzed': len(df),
            'structures_passed_filter': len(filtered_df),
            'unique_sequences_found': len(sequence_counts),
            'sequence_occurrence_counts': sequence_counts,
            'ranked_sequences': ranked_sequences
        }

        self.logger.info(f"Occurrence voting complete:")
        self.logger.info(f"  Structures analyzed: {len(df)}")
        self.logger.info(f"  Structures passed filter: {len(filtered_df)}")
        self.logger.info(f"  Unique sequences found: {len(sequence_counts)}")

        return voting_results

    def _filter_structures(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter structures based on metric criteria"""
        metric_name = self.config.filtering.metric_name
        cutoff_value = self.config.filtering.cutoff_value
        cutoff_method = self.config.filtering.cutoff_method

        if metric_name not in df.columns:
            raise ValueError(f"Metric '{metric_name}' not found in results CSV")

        # Apply filtering
        if cutoff_method == "below":
            filtered_df = df[df[metric_name] < cutoff_value].copy()
        else:  # "above"
            filtered_df = df[df[metric_name] > cutoff_value].copy()

        self.logger.info(f"Filtering: {metric_name} {cutoff_method} {cutoff_value}")
        self.logger.info(f"Structures passing filter: {len(filtered_df)} out of {len(df)}")

        return filtered_df

    def _collect_and_count_sequences(self, filtered_df: pd.DataFrame) -> Dict[str, int]:
        """Collect sequences from A3M files corresponding to filtered structures"""
        self.logger.info("Collecting sequences from filtered structures...")

        # Get PDB paths from filtered results
        pdb_paths = filtered_df['PDB'].tolist()
        sequence_counts = Counter()

        for i, pdb_path in enumerate(pdb_paths, 1):
            # Convert PDB path to A3M path
            a3m_path = self._pdb_to_a3m_path(pdb_path)

            if not os.path.exists(a3m_path):
                self.logger.warning(f"A3M file not found: {a3m_path}")
                continue

            try:
                # Read A3M file
                sequences_dict = read_a3m_to_dict(a3m_path)

                # Skip query (first sequence) and count others
                sequence_items = list(sequences_dict.items())
                non_query_sequences = sequence_items[1:]  # Skip query

                for header, sequence in non_query_sequences:
                    # Count occurrence of each sequence
                    sequence_counts[sequence] += 1

                if i % 100 == 0:
                    self.logger.info(f"Processed {i}/{len(pdb_paths)} A3M files")

            except Exception as e:
                self.logger.warning(f"Error reading {a3m_path}: {e}")
                continue

        self.logger.info(f"Collected sequences from {len(pdb_paths)} A3M files")
        self.logger.info(f"Found {len(sequence_counts)} unique sequences")

        return dict(sequence_counts)

    def _pdb_to_a3m_path(self, pdb_path: str) -> str:
        """Convert PDB path to corresponding A3M path"""
        pdb_path = str(pdb_path)

        # Handle ColabFold naming pattern
        if '_unrelaxed' in pdb_path:
            base_path = pdb_path.split('_unrelaxed')[0]
            return base_path + '.a3m'

        # Fallback: replace .pdb extension with .a3m
        return os.path.splitext(pdb_path)[0] + '.a3m'

    def _rank_sequences_by_frequency(self, sequence_counts: Dict[str, int]) -> List[Tuple[str, int]]:
        """Rank sequences by occurrence frequency"""
        # Sort by count (descending) then by sequence (for deterministic ordering)
        ranked_sequences = sorted(
            sequence_counts.items(),
            key=lambda x: (-x[1], x[0])  # Sort by count desc, then sequence asc
        )

        self.logger.info("Top 10 most frequent sequences:")
        for i, (sequence, count) in enumerate(ranked_sequences[:10], 1):
            seq_preview = sequence[:50] + "..." if len(sequence) > 50 else sequence
            self.logger.info(f"  {i:2d}. Count: {count:4d} - {seq_preview}")

        return ranked_sequences

    def generate_final_results(self, voting_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final results and output files"""
        ranked_sequences = voting_results['ranked_sequences']
        top_n = self.config.voting.top_n_sequences

        # Select top N sequences
        top_sequences = ranked_sequences[:top_n]

        self.logger.info(f"Selecting top {top_n} sequences from {len(ranked_sequences)} total")

        # Save occurrence counts to CSV
        occurrence_df = pd.DataFrame([
            {'rank': i+1, 'sequence': seq, 'occurrence_count': count, 'sequence_length': len(seq)}
            for i, (seq, count) in enumerate(ranked_sequences)
        ])
        occurrence_csv = self.results_dir / "occurrence_counts.csv"
        occurrence_df.to_csv(occurrence_csv, index=False)

        # Create final A3M file with top sequences
        final_a3m_path = self._create_final_a3m(top_sequences)

        # Generate summary report
        summary_path = self._create_summary_report(voting_results, top_sequences)

        final_results = {
            'top_n_selected': len(top_sequences),
            'final_a3m_file': str(final_a3m_path),
            'occurrence_counts_csv': str(occurrence_csv),
            'summary_report': str(summary_path)
        }

        self.logger.info(f"Final results generated:")
        self.logger.info(f"  Top sequences A3M: {final_a3m_path}")
        self.logger.info(f"  Occurrence counts CSV: {occurrence_csv}")
        self.logger.info(f"  Summary report: {summary_path}")

        return final_results

    def _create_final_a3m(self, top_sequences: List[Tuple[str, int]]) -> Path:
        """Create final A3M file with top sequences"""
        # Get query sequence from source A3M
        source_sequences = read_a3m_to_dict(self.config.general.source_a3m)
        query_header, query_sequence = list(source_sequences.items())[0]

        # Create final sequences dictionary
        final_sequences = {query_header: query_sequence}

        for i, (sequence, count) in enumerate(top_sequences, 1):
            # Create enhanced header with rank and occurrence count
            header = f"rank_{i:03d}_count_{count:04d}_seq_{i:06d}"
            final_sequences[header] = sequence

        # Write final A3M file
        final_a3m_path = self.results_dir / "top_sequences.a3m"
        write_a3m(final_sequences, str(final_a3m_path), homodimer_mode=self.homodimer_mode)

        return final_a3m_path

    def _create_summary_report(self, voting_results: Dict[str, Any],
                              top_sequences: List[Tuple[str, int]]) -> Path:
        """Create detailed summary report"""
        summary_path = self.results_dir / "occurrence_voting_summary.txt"

        with open(summary_path, 'w') as f:
            f.write("Occurrence Voting Analysis Summary\n")
            f.write("=" * 50 + "\n\n")

            # Configuration summary
            f.write("Configuration:\n")
            f.write(f"  Source A3M: {self.config.general.source_a3m}\n")
            f.write(f"  Protein: {self.config.general.protein_name}\n")
            f.write(f"  Random seed: {self.config.general.random_seed}\n")
            f.write(f"  Groups created: {self.config.sampling.num_groups}\n")
            f.write(f"  Group size: {self.config.sampling.group_size}\n")
            f.write(f"  Number of batches: {self.config.sampling.num_batches}\n\n")

            # Filtering criteria
            f.write("Filtering Criteria:\n")
            f.write(f"  Metric: {self.config.filtering.metric_name}\n")
            f.write(f"  Cutoff: {self.config.filtering.cutoff_method} {self.config.filtering.cutoff_value}\n\n")

            # Results summary
            f.write("Results Summary:\n")
            f.write(f"  Structures analyzed: {voting_results['total_structures_analyzed']}\n")
            f.write(f"  Structures passed filter: {voting_results['structures_passed_filter']}\n")
            f.write(f"  Unique sequences found: {voting_results['unique_sequences_found']}\n")
            f.write(f"  Top N selected: {len(top_sequences)}\n\n")

            # Top sequences
            f.write(f"Top {len(top_sequences)} Most Frequent Sequences:\n")
            for i, (sequence, count) in enumerate(top_sequences, 1):
                seq_preview = sequence[:60] + "..." if len(sequence) > 60 else sequence
                f.write(f"  {i:3d}. Count: {count:4d} - {seq_preview}\n")

        return summary_path

    def _create_metric_mapping(self, metrics: List[str]) -> Dict[str, str]:
        """
        Create mapping from actual metric names to generic names (metric1, metric2, etc.).

        Args:
            metrics: List of metric names

        Returns:
            Dictionary mapping actual metric names to generic names
        """
        mapping = {}
        for i, metric in enumerate(metrics, 1):
            generic_name = f"metric{i}"
            mapping[metric] = generic_name
        return mapping

    def _get_generic_metric_name(self, metric: str, metric_mapping: Dict[str, str]) -> str:
        """
        Get the generic name (metric1, metric2, etc.) for a metric.

        Args:
            metric: Actual metric name
            metric_mapping: Mapping dictionary

        Returns:
            Generic metric name, or original name if not found in mapping
        """
        return metric_mapping.get(metric, metric)

    def _get_metric_plot_params(self, metric: str, metric_mapping: Dict[str, str]) -> Dict[str, Any]:
        """
        Get plotting parameters for a specific metric (1D plots).

        Args:
            metric: Metric name
            metric_mapping: Mapping dictionary

        Returns:
            Dictionary of plotting parameters
        """
        params = {}

        # Default parameters
        params.update({
            'n_plot_bins': 50,
            'log_scale': False,
            'figsize': (10, 6),
            'initial_color': '#87CEEB',
            'end_color': '#FFFFFF',
            'show_bin_lines': False
        })

        # Check if plotting config exists
        if not (self.config.plotting and hasattr(self.config.plotting, 'plot_params')):
            return params

        # Override with global plot parameters
        if hasattr(self.config.plotting, 'plot_params') and self.config.plotting.plot_params:
            plot_params = self.config.plotting.plot_params
            if '1d' in plot_params:
                params.update(plot_params['1d'])

        # Get generic metric name for config lookup
        generic_metric = self._get_generic_metric_name(metric, metric_mapping)

        # Override with metric-specific parameters
        if hasattr(self.config.plotting, 'plot_params') and self.config.plotting.plot_params:
            plot_params = self.config.plotting.plot_params
            if metric in plot_params:
                params.update(plot_params[metric])
            elif generic_metric in plot_params:
                params.update(plot_params[generic_metric])

        # Add metric range if specified
        if hasattr(self.config.plotting, 'metric_ranges') and self.config.plotting.metric_ranges:
            metric_ranges = self.config.plotting.metric_ranges
            range_config = None

            if metric in metric_ranges:
                range_config = metric_ranges[metric]
            elif generic_metric in metric_ranges:
                range_config = metric_ranges[generic_metric]

            if range_config:
                params.update({
                    'x_min': range_config.get('min'),
                    'x_max': range_config.get('max'),
                    'x_ticks': range_config.get('ticks')
                })

        # Add colors if specified
        if hasattr(self.config.plotting, 'colors') and self.config.plotting.colors:
            colors_config = self.config.plotting.colors
            colors = None

            if metric in colors_config:
                colors = colors_config[metric]
            elif generic_metric in colors_config:
                colors = colors_config[generic_metric]

            if colors and isinstance(colors, list) and len(colors) >= 2:
                params.update({
                    'initial_color': colors[0],
                    'end_color': colors[1]
                })

        return params

    def _get_2d_plot_params(self, metric1: str, metric2: str, metric_mapping: Dict[str, str]) -> Dict[str, Any]:
        """
        Get plotting parameters for 2D plots.

        Args:
            metric1: First metric name
            metric2: Second metric name
            metric_mapping: Mapping dictionary

        Returns:
            Dictionary of plotting parameters
        """
        params = {}

        # Default parameters
        params.update({
            'color_metric': 'plddt'
        })

        # Check if plotting config exists
        if not (self.config.plotting and hasattr(self.config.plotting, 'plot_params')):
            return params

        # Override with global 2D parameters
        if hasattr(self.config.plotting, 'plot_params') and self.config.plotting.plot_params:
            plot_params = self.config.plotting.plot_params
            if '2d' in plot_params:
                params.update(plot_params['2d'])

        # Add metric ranges (dynamically assign to x or y axis based on position)
        if hasattr(self.config.plotting, 'metric_ranges') and self.config.plotting.metric_ranges:
            metric_ranges = self.config.plotting.metric_ranges

            for i, metric in enumerate([metric1, metric2], 1):
                generic_metric = self._get_generic_metric_name(metric, metric_mapping)

                range_config = None
                if metric in metric_ranges:
                    range_config = metric_ranges[metric]
                elif generic_metric in metric_ranges:
                    range_config = metric_ranges[generic_metric]

                if range_config:
                    if i == 1:  # First metric (x-axis)
                        params.update({
                            'x_min': range_config.get('min'),
                            'x_max': range_config.get('max'),
                            'x_ticks': range_config.get('ticks')
                        })
                    else:  # Second metric (y-axis)
                        params.update({
                            'y_min': range_config.get('min'),
                            'y_max': range_config.get('max'),
                            'y_ticks': range_config.get('ticks')
                        })

        return params

    def generate_structure_plots(self, results_csv: str) -> Dict[str, Any]:
        """
        Generate structure analysis plots including scatter plots and distributions.

        Args:
            results_csv: Path to structure analysis results CSV

        Returns:
            Dictionary with plotting results
        """
        self.logger.info("Generating structure analysis plots...")

        # Load results data
        results_df = pd.read_csv(results_csv)
        if results_df.empty:
            self.logger.warning("Empty results DataFrame, skipping structure plots")
            return {'plot_files': [], 'plots_directory': str(self.results_dir / "plots")}

        # Create plots directory
        plots_dir = self.results_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        plot_files = []

        # Get metrics to plot from configuration
        if self.config.plotting and self.config.plotting.metrics_to_plot:
            # Use specified metrics from config
            specified_metrics = self.config.plotting.metrics_to_plot
            available_metrics = [m for m in specified_metrics if m in results_df.columns]
            if len(specified_metrics) != len(available_metrics):
                missing = set(specified_metrics) - set(available_metrics)
                self.logger.warning(f"Some specified metrics not found in data: {missing}")
            self.logger.info(f"Using {len(available_metrics)} specified metrics: {available_metrics}")
        else:
            # Use all available metrics from structure analysis config
            structure_config = load_filter_modes(self.config.structure_analysis.config_json)
            available_metrics = []

            # Collect all metric names from filter criteria and composite metrics
            for criterion in structure_config.get('filter_criteria', []):
                metric_name = criterion.get('name')
                if metric_name and metric_name in results_df.columns:
                    available_metrics.append(metric_name)

            for composite in structure_config.get('composite_metrics', []):
                metric_name = composite.get('name')
                if metric_name and metric_name in results_df.columns:
                    available_metrics.append(metric_name)

            self.logger.info(f"Found {len(available_metrics)} metrics to plot: {available_metrics}")

        if len(available_metrics) < 2:
            self.logger.warning("Need at least 2 metrics for comprehensive plotting")

        # Create metric mapping for configuration lookup
        metric_mapping = self._create_metric_mapping(available_metrics)
        self.logger.info(f"Created metric mapping: {metric_mapping}")

        # Check which plot types to generate
        plot_types = ['1d', '2d']  # Default plot types
        if self.config.plotting and self.config.plotting.plot_types:
            plot_types = self.config.plotting.plot_types

        # Generate 1D distribution plots
        if '1d' in plot_types:
            for metric in available_metrics:
                if metric in results_df.columns:
                    try:
                        self.logger.info(f"  Generating distribution plot for {metric}")

                        # Get metric-specific parameters
                        metric_params = self._get_metric_plot_params(metric, metric_mapping)

                        plot_path = plot_1d_distribution(
                            results_df=results_df,
                            metric_name=metric,
                            output_dir=str(plots_dir),
                            **metric_params,
                            logger=self.logger
                        )
                        if plot_path:
                            plot_files.append(plot_path)
                            self.logger.info(f"    Saved distribution plot: {plot_path}")
                    except Exception as e:
                        self.logger.warning(f"Failed to generate distribution plot for {metric}: {e}")

        # Generate 2D scatter plots for metric pairs
        if '2d' in plot_types and len(available_metrics) >= 2:
            for i, metric1 in enumerate(available_metrics):
                for metric2 in available_metrics[i+1:]:
                    if metric1 in results_df.columns and metric2 in results_df.columns:
                        try:
                            self.logger.info(f"  Generating scatter plot for {metric1} vs {metric2}")

                            # Get plot parameters for this metric pair
                            plot_params = self._get_2d_plot_params(metric1, metric2, metric_mapping)

                            plot_path = create_2d_scatter_plot(
                                results_df=results_df,
                                metric_name1=metric1,
                                metric_name2=metric2,
                                output_dir=str(plots_dir),
                                title=None,
                                **plot_params,
                                logger=self.logger
                            )
                            if plot_path:
                                plot_files.append(plot_path)
                                self.logger.info(f"    Saved scatter plot: {plot_path}")

                            # Generate joint plot if requested
                            if 'joint' in plot_types:
                                joint_path = create_joint_plot(
                                    results_df=results_df,
                                    metric_name1=metric1,
                                    metric_name2=metric2,
                                    output_dir=str(plots_dir),
                                    **plot_params,
                                    logger=self.logger
                                )

                                if joint_path:
                                    plot_files.append(joint_path)
                                    self.logger.info(f"    Saved joint plot: {joint_path}")

                        except Exception as e:
                            self.logger.warning(f"Failed to generate scatter plot for {metric1} vs {metric2}: {e}")

        # Generate correlation plots
        if 'correlation' in plot_types:
            self.logger.info("Generating correlation plots...")
            plddt_metrics = ['plddt', 'local_plddt']

            for plddt_metric in plddt_metrics:
                if plddt_metric not in results_df.columns:
                    continue

                self.logger.info(f"  Generating correlation plots with {plddt_metric}")

                for metric in available_metrics:
                    if metric == plddt_metric or metric not in results_df.columns:
                        continue

                    try:
                        plot_params = self._get_2d_plot_params(metric, plddt_metric, metric_mapping)

                        # Override color_metric in plot_params to use the plddt metric
                        plot_params_corr = plot_params.copy()
                        plot_params_corr['color_metric'] = plddt_metric

                        correlation_path = create_2d_scatter_plot(
                            results_df=results_df,
                            metric_name1=metric,
                            metric_name2=plddt_metric,
                            output_dir=str(plots_dir),
                            title=None,
                            **plot_params_corr,
                            logger=self.logger
                        )

                        if correlation_path:
                            plot_files.append(correlation_path)
                            self.logger.info(f"    Saved correlation plot: {correlation_path}")

                    except Exception as e:
                        self.logger.warning(f"Failed to generate correlation plot for {metric} vs {plddt_metric}: {e}")

        self.logger.info(f"Generated {len(plot_files)} structure analysis plots")

        return {
            'plot_files': plot_files,
            'plots_directory': str(plots_dir),
            'metrics_plotted': available_metrics
        }

    def generate_occurrence_plots(self, voting_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate sequence occurrence bar graphs and related plots.

        Args:
            voting_results: Results from occurrence voting

        Returns:
            Dictionary with plotting results
        """
        self.logger.info("Generating sequence occurrence plots...")

        # Create plots directory
        plots_dir = self.results_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        plot_files = []

        # Load occurrence counts data
        occurrence_csv = self.results_dir / "occurrence_counts.csv"
        if not occurrence_csv.exists():
            self.logger.warning("No occurrence counts CSV found, skipping occurrence plots")
            return {'plot_files': [], 'plots_directory': str(plots_dir)}

        occurrence_df = pd.read_csv(occurrence_csv)
        if occurrence_df.empty:
            self.logger.warning("Empty occurrence data, skipping occurrence plots")
            return {'plot_files': [], 'plots_directory': str(plots_dir)}

        # Plot 1: Enhanced sequence occurrence bar chart
        try:
            # Determine optimal number of sequences to show
            total_sequences = len(occurrence_df)
            if total_sequences <= 20:
                top_n = total_sequences
            elif total_sequences <= 100:
                top_n = min(30, total_sequences)
            else:
                top_n = min(50, total_sequences)

            top_sequences = occurrence_df.head(top_n)

            # Create figure with improved styling
            fig, ax = plt.subplots(figsize=(16, 10))

            # Create color gradient based on occurrence counts
            max_count = top_sequences['occurrence_count'].max()
            min_count = top_sequences['occurrence_count'].min()

            # Use a more sophisticated color scheme
            colors = []
            for count in top_sequences['occurrence_count']:
                # Normalize between 0 and 1
                normalized = (count - min_count) / (max_count - min_count) if max_count > min_count else 0.5
                # Use a custom color map: blue for low, red for high
                colors.append(plt.cm.coolwarm(0.3 + normalized * 0.7))

            # Create bars with enhanced styling
            bars = ax.bar(range(len(top_sequences)), top_sequences['occurrence_count'],
                         color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)

            # Add value labels on top of bars for top sequences
            for i, (bar, count) in enumerate(zip(bars, top_sequences['occurrence_count'])):
                if i < 10:  # Only annotate top 10 for clarity
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_count*0.01,
                           f'{count}', ha='center', va='bottom', fontsize=9, fontweight='bold')

            # Enhanced styling
            ax.set_xlabel('Sequence Rank', fontsize=14, fontweight='bold')
            ax.set_ylabel('Occurrence Count', fontsize=14, fontweight='bold')
            ax.set_title(f'Top {top_n} Sequence Occurrences in Filtered Structures\n'
                        f'Total: {total_sequences} unique sequences',
                        fontsize=16, fontweight='bold', pad=20)

            # Improve x-axis ticks
            if top_n <= 20:
                ax.set_xticks(range(len(top_sequences)))
                ax.set_xticklabels([f'{i+1}' for i in range(len(top_sequences))])
            else:
                step = max(1, len(top_sequences)//10)
                tick_positions = range(0, len(top_sequences), step)
                ax.set_xticks(tick_positions)
                ax.set_xticklabels([f'{i+1}' for i in tick_positions])

            # Add grid for better readability
            ax.grid(True, axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)

            # Add statistical annotations
            mean_count = top_sequences['occurrence_count'].mean()
            median_count = top_sequences['occurrence_count'].median()

            # Add horizontal lines for mean and median
            ax.axhline(y=mean_count, color='orange', linestyle='-', alpha=0.7, linewidth=2,
                      label=f'Mean: {mean_count:.1f}')
            ax.axhline(y=median_count, color='green', linestyle='--', alpha=0.7, linewidth=2,
                      label=f'Median: {median_count:.1f}')

            # Add legend
            ax.legend(loc='upper right', fontsize=11)

            # Add text box with summary statistics
            stats_text = f'Statistics:\nMax: {max_count}\nMin: {min_count}\nStd: {top_sequences["occurrence_count"].std():.1f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            plt.tight_layout()
            occurrence_bar_plot = plots_dir / "top_sequence_occurrences.png"
            plt.savefig(occurrence_bar_plot, dpi=300, bbox_inches='tight', facecolor='white')
            plt.savefig(str(occurrence_bar_plot).replace('.png', '.svg'), format='svg', bbox_inches='tight', facecolor='white')
            plt.close()
            plot_files.append(str(occurrence_bar_plot))
            self.logger.info(f"  Generated enhanced occurrence bar chart: {occurrence_bar_plot}")

        except Exception as e:
            self.logger.warning(f"Failed to generate occurrence bar chart: {e}")


        self.logger.info(f"Generated {len(plot_files)} occurrence plots")

        return {
            'plot_files': plot_files,
            'plots_directory': str(plots_dir),
            'total_sequences_analyzed': len(occurrence_df)
        }