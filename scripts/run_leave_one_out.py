#!/usr/bin/env python3
"""
Leave-One-Out Impact Analysis Workflow

This script executes the complete leave-one-out workflow for sequence impact analysis.
It evaluates how individual sequences affect ensemble structure predictions using
configurable metrics and thresholds.

Usage:
    python scripts/run_leave_one_out.py /path/to/config.yaml

Example:
    python scripts/run_leave_one_out.py config_examples/leave_one_out_config.yaml
"""

import sys
import argparse
import traceback
from pathlib import Path

from af_claseq.leave_one_out import load_config, LeaveOneOutManager
from af_claseq.utils.logging_utils import get_logger, setup_logger
from tqdm import tqdm


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run Leave-One-Out Impact Analysis Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s config.yaml                    # Run complete workflow
  %(prog)s --validate-only config.yaml   # Validate configuration only
  %(prog)s --analysis-only config.yaml   # Run analysis after ColabFold jobs complete
  %(prog)s --resume config.yaml          # Resume from last step

Configuration file should contain:
  - general: source_a3m, base_dir, structure_analysis_config, protein_name
  - leave_one_out: num_seq_per_group, impact_metric_name, thresholds
  - slurm: conda_env_path, account, partition, job settings
  - plotting: output settings (optional)
        """
    )

    parser.add_argument(
        'config_file',
        type=str,
        help='Path to YAML configuration file'
    )

    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate configuration without running workflow'
    )

    parser.add_argument(
        '--analysis-only',
        action='store_true',
        help='Run only impact analysis (skip group creation and ColabFold jobs)'
    )

    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level (default: INFO)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Prepare workflow but do not submit jobs'
    )

    return parser.parse_args()


def display_workflow_summary(config):
    """Display a summary of the workflow configuration"""
    logger = get_logger("workflow_summary")

    logger.info("=" * 60)
    logger.info("LEAVE-ONE-OUT WORKFLOW SUMMARY")
    logger.info("=" * 60)

    logger.info("General Configuration:")
    logger.info(f"  Protein: {config.general.protein_name}")
    logger.info(f"  Source A3M: {config.general.source_a3m}")
    logger.info(f"  Output directory: {config.general.base_dir}")
    logger.info(f"  Random seed: {config.general.random_seed}")

    logger.info("Leave-One-Out Configuration:")
    logger.info(f"  Sequences per group: {config.leave_one_out.num_seq_per_group}")
    logger.info(f"  Impact metric: {config.leave_one_out.impact_metric_name}")
    logger.info(f"  Impact threshold: {config.leave_one_out.impact_threshold} ({config.leave_one_out.cutoff_method})")
    logger.info(f"  Full group threshold: {config.leave_one_out.full_group_mean_threshold} ({config.leave_one_out.full_mean_cutoff_method})")

    logger.info("SLURM Configuration:")
    logger.info(f"  Account: {config.slurm.account}")
    logger.info(f"  Partition: {config.slurm.partition}")
    logger.info(f"  Models per prediction: {config.slurm.num_models}")
    logger.info(f"  Max concurrent jobs: {config.slurm.max_concurrent_jobs}")

    logger.info("Output Configuration:")
    logger.info(f"  Plots directory: {config.plotting.output_dir}")
    logger.info(f"  Figure size: {config.plotting.figsize}")

    logger.info("=" * 60)


def run_analysis_only(loo_manager):
    """Run only the analysis portion of the workflow (assumes ColabFold jobs are complete)"""
    logger = get_logger("analysis_only")

    try:
        # Check if groups directory exists
        groups_dir = loo_manager.groups_dir
        if not groups_dir.exists():
            logger.error(f"Groups directory not found: {groups_dir}")
            logger.error("Please run the complete workflow first to create groups and submit ColabFold jobs.")
            return {'status': 'failed', 'error': 'groups_not_found'}

        # Reconstruct group information from existing directories
        logger.info("Reconstructing group information from existing directories...")
        groups = []
        group_dirs = sorted(groups_dir.glob("group_*"))

        with tqdm(total=len(group_dirs), desc="Scanning groups", unit="group") as pbar:
            for group_dir in group_dirs:
                if group_dir.is_dir():
                    # Check if this group has PDB files (indicating ColabFold completion)
                    pdb_files = list(group_dir.glob("*.pdb"))
                    if not pdb_files:
                        pbar.set_postfix(current=group_dir.name, status="no PDBs")
                        pbar.update(1)
                        continue

                    # Reconstruct group info (simplified - we only need group_id and group_dir for analysis)
                    group_info = {
                        'group_id': group_dir.name,
                        'group_dir': str(group_dir),
                        'loo_subsets': []
                    }

                    # Find LOO subsets by looking for LOO-specific PDB files
                    loo_patterns = set()
                    for pdb_file in pdb_files:
                        if "_loo_" in pdb_file.name:
                            loo_id = pdb_file.name.split("_loo_")[1].split("_")[0]
                            loo_patterns.add(f"loo_{loo_id}")

                    # Get actual sequences from LOO A3M files
                    for loo_pattern in sorted(loo_patterns):
                        # Find the corresponding A3M file
                        loo_a3m_files = list(group_dir.glob(f"{group_dir.name}_{loo_pattern}.a3m"))
                        if loo_a3m_files:
                            # Read the A3M file to get the query sequence (first sequence)
                            try:
                                from af_claseq.utils.sequence_processing import read_a3m_to_dict
                                sequences = read_a3m_to_dict(str(loo_a3m_files[0]))
                                if sequences:
                                    # Get the query sequence (first one)
                                    first_header, first_sequence = next(iter(sequences.items()))

                                    # Find what sequence was left out by comparing with full group
                                    full_a3m = group_dir / f"{group_dir.name}_full.a3m"
                                    if full_a3m.exists():
                                        full_sequences = read_a3m_to_dict(str(full_a3m))
                                        loo_headers = set(sequences.keys())
                                        full_headers = set(full_sequences.keys())
                                        left_out_headers = full_headers - loo_headers

                                        if left_out_headers:
                                            left_out_header = list(left_out_headers)[0]
                                            left_out_sequence = full_sequences[left_out_header]
                                        else:
                                            # Fallback to first sequence if we can't determine
                                            left_out_header = first_header
                                            left_out_sequence = first_sequence
                                    else:
                                        # Use first sequence as fallback
                                        left_out_header = first_header
                                        left_out_sequence = first_sequence
                                else:
                                    left_out_header = f'unknown_{loo_pattern}'
                                    left_out_sequence = 'unknown'
                            except Exception:
                                left_out_header = f'unknown_{loo_pattern}'
                                left_out_sequence = 'unknown'
                        else:
                            left_out_header = f'unknown_{loo_pattern}'
                            left_out_sequence = 'unknown'

                        group_info['loo_subsets'].append({
                            'loo_id': loo_pattern,
                            'left_out_header': left_out_header,
                            'left_out_sequence': left_out_sequence
                        })

                    if group_info['loo_subsets']:
                        groups.append(group_info)
                        pbar.set_postfix(current=group_dir.name, status=f"✓ {len(group_info['loo_subsets'])} LOO")
                    else:
                        pbar.set_postfix(current=group_dir.name, status="no LOO")

                pbar.update(1)

        if not groups:
            logger.error("No valid groups with completed ColabFold jobs found.")
            return {'status': 'failed', 'error': 'no_completed_groups'}

        logger.info(f"Found {len(groups)} groups ready for analysis")

        # Run impact analysis
        logger.info("Starting impact analysis...")
        impact_results = loo_manager.analyze_impact_scores(groups)

        # Generate plots
        logger.info("Generating impact visualization plots...")
        plotting_results = loo_manager.generate_impact_plots(impact_results['all_results'])

        # Save final results
        logger.info("Saving final results...")
        final_results = loo_manager.save_final_results(impact_results)

        return {
            'status': 'completed',
            'groups_analyzed': len(groups),
            'impact_analysis': impact_results,
            'plotting': plotting_results,
            'final_output': final_results
        }

    except Exception as e:
        logger.error(f"Analysis-only mode failed: {e}")
        return {'status': 'failed', 'error': str(e)}


def main():
    """Main execution function"""
    args = parse_arguments()

    # Setup logging
    setup_logger()
    logger = get_logger("run_leave_one_out")

    try:
        # Load config once — validation and workflow use the same object
        config = load_config(args.config_file)
        logger.info("Configuration loaded successfully.")

        if args.validate_only:
            logger.info("Configuration validation completed successfully.")
            return 0

        # Display workflow summary
        display_workflow_summary(config)

        if args.dry_run:
            logger.info("Dry run completed. No jobs were submitted.")
            return 0

        # Initialize workflow manager
        logger.info("Initializing Leave-One-Out workflow manager...")
        loo_manager = LeaveOneOutManager(config)

        # Execute workflow
        if args.analysis_only:
            logger.info("Running analysis-only mode (skipping group creation and ColabFold jobs)...")
            results = run_analysis_only(loo_manager)
        else:
            logger.info("Starting complete leave-one-out workflow...")
            results = loo_manager.run_complete_workflow()

        # Display final summary
        logger.info("=" * 60)
        logger.info("WORKFLOW COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)

        if 'impact_analysis' in results:
            impact_results = results['impact_analysis']
            logger.info(f"Total sequences analyzed: {impact_results['total_analyzed']}")
            logger.info(f"Significant sequences identified: {impact_results['significant_sequences']}")

        if 'final_output' in results and results['final_output']['status'] == 'completed':
            logger.info(f"Final results saved to: {results['final_output']['output_file']}")

        if 'plotting' in results and results['plotting']['status'] == 'completed':
            logger.info(f"Visualizations saved to: {results['plotting']['plots_directory']}")

        logger.info("Leave-one-out impact analysis completed successfully!")
        return 0

    except KeyboardInterrupt:
        logger.info("Workflow interrupted by user")
        return 1

    except Exception as e:
        logger.error(f"Workflow failed with error: {e}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)