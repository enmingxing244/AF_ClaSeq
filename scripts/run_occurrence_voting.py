#!/usr/bin/env python3
"""
Occurrence Voting Workflow

This script executes the complete occurrence voting workflow for sequence selection.
It performs random sampling, structure prediction, filtering, and occurrence-based voting
to identify the most frequently occurring sequences in high-quality structures.

Usage:
    python scripts/run_occurrence_voting.py /path/to/config.yaml

Example:
    python scripts/run_occurrence_voting.py config_examples/occurrence_voting_config.yaml
"""

import sys
import argparse
import traceback
import logging
from pathlib import Path

from af_claseq.occurrence_voting import load_config, OccurrenceVotingManager # type: ignore
from af_claseq.utils.logging_utils import get_logger, setup_logger


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run Occurrence Voting Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s config.yaml
  %(prog)s --validate-only config.yaml
  %(prog)s --dry-run config.yaml
  %(prog)s --sampling-only config.yaml
  %(prog)s --step-only colabfold config.yaml
  %(prog)s --resume-from analysis config.yaml

Workflow Steps:
  1. sampling: Create random sequence groups organized into batches
  2. colabfold: Submit ColabFold structure prediction jobs
  3. analysis: Analyze predicted structures, calculate metrics, and generate plots
  4. voting: Perform occurrence voting, generate occurrence plots, and final results

Configuration file should contain:
  - general: source_a3m, base_dir, protein_name, random_seed
  - sampling: num_groups, group_size, num_batches
  - structure_prediction: num_models, num_seeds
  - structure_analysis: config_json, plddt_threshold
  - filtering: metric_name, cutoff_value, cutoff_method
  - voting: top_n_sequences
  - slurm: conda_env_path, account, partition, job settings
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
        '--dry-run',
        action='store_true',
        help='Prepare workflow but do not submit jobs'
    )

    parser.add_argument(
        '--sampling-only',
        action='store_true',
        help='Only perform random sampling step'
    )

    parser.add_argument(
        '--resume-from',
        choices=['sampling', 'colabfold', 'analysis', 'voting'],
        help='Resume workflow from a specific step'
    )

    parser.add_argument(
        '--step-only',
        choices=['sampling', 'colabfold', 'analysis', 'voting'],
        help='Run only a specific step'
    )

    return parser.parse_args()


def validate_configuration(config_file: str) -> bool:
    """
    Validate configuration file and dependencies.

    Args:
        config_file: Path to configuration file

    Returns:
        True if configuration is valid, False otherwise
    """
    logger = get_logger("config_validation")

    try:
        logger.info("Validating configuration...")

        # Load configuration
        config = load_config(config_file)

        # Validation is performed during config loading
        logger.info("✓ Configuration file loaded successfully")
        logger.info(f"✓ Source A3M file: {config.general.source_a3m}")
        logger.info(f"✓ Base directory: {config.general.base_dir}")
        logger.info(f"✓ Sampling: {config.sampling.num_groups} groups of {config.sampling.group_size} sequences")
        logger.info(f"✓ Filtering: {config.filtering.metric_name} {config.filtering.cutoff_method} {config.filtering.cutoff_value}")

        # Additional validations
        if config.sampling.num_groups <= 0:
            logger.error("num_groups must be positive")
            return False

        if config.sampling.group_size <= 0:
            logger.error("group_size must be positive")
            return False

        if config.voting.top_n_sequences <= 0:
            logger.error("top_n_sequences must be positive")
            return False

        logger.info("✓ All configuration parameters are valid")
        return True

    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False


def display_workflow_summary(config):
    """Display a summary of the workflow configuration"""
    logger = get_logger("workflow_summary")

    logger.info("=" * 70)
    logger.info("OCCURRENCE VOTING WORKFLOW SUMMARY")
    logger.info("=" * 70)

    logger.info("General Configuration:")
    logger.info(f"  Protein: {config.general.protein_name}")
    logger.info(f"  Source A3M: {config.general.source_a3m}")
    logger.info(f"  Output directory: {config.general.base_dir}")
    logger.info(f"  Random seed: {config.general.random_seed}")

    logger.info("Sampling Configuration:")
    logger.info(f"  Number of groups: {config.sampling.num_groups}")
    logger.info(f"  Sequences per group: {config.sampling.group_size}")
    logger.info(f"  Number of batches: {config.sampling.num_batches}")

    total_sequences = config.sampling.num_groups * config.sampling.group_size
    logger.info(f"  Total sequences to sample: {total_sequences}")

    logger.info("Structure Prediction:")
    logger.info(f"  Models per prediction: {config.structure_prediction.num_models}")
    logger.info(f"  Seeds per prediction: {config.structure_prediction.num_seeds}")

    logger.info("Filtering Configuration:")
    logger.info(f"  Structure analysis JSON: {config.structure_analysis.config_json}")
    logger.info(f"  Metric: {config.filtering.metric_name}")
    logger.info(f"  Cutoff: {config.filtering.cutoff_method} {config.filtering.cutoff_value}")

    logger.info("Voting Configuration:")
    logger.info(f"  Top N sequences: {config.voting.top_n_sequences}")

    logger.info("SLURM Configuration:")
    logger.info(f"  Account: {config.slurm.account}")
    logger.info(f"  Partition: {config.slurm.partition}")
    logger.info(f"  Max concurrent jobs: {config.slurm.max_concurrent_jobs}")

    logger.info("=" * 70)


def setup_logging(config_file: str) -> logging.Logger:
    """Set up logging for occurrence voting workflow"""
    # Load config to get base directory
    config = load_config(config_file)
    base_dir = Path(config.general.base_dir)

    # Create logs directory
    log_dir = base_dir / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)

    log_file = log_dir / "occurrence_voting.log"

    # Set up the root logger for the whole package
    return setup_logger(
        name="af_claseq",  # Root logger for the package
        log_file=log_file,
        level=logging.INFO,
        propagate=False,  # Root logger doesn't propagate
        add_console_handler=True
    )


def main():
    """Main execution function"""
    args = parse_arguments()

    # Setup logging (must be done before using any loggers)
    try:
        setup_logging(args.config_file)
        logger = get_logger("run_occurrence_voting")
        logger.info("Starting Occurrence Voting workflow...")
    except Exception as e:
        print(f"Failed to setup logging: {e}")
        return 1

    try:
        # Validate configuration
        if not validate_configuration(args.config_file):
            logger.error("Configuration validation failed. Exiting.")
            return 1

        if args.validate_only:
            logger.info("Configuration validation completed successfully.")
            return 0

        # Load configuration
        config = load_config(args.config_file)

        # Display workflow summary
        display_workflow_summary(config)

        if args.dry_run:
            logger.info("Dry run completed. No jobs were submitted.")
            return 0

        # Initialize workflow manager
        logger.info("Initializing Occurrence Voting workflow manager...")
        voting_manager = OccurrenceVotingManager(config)

        # Execute workflow based on options
        if args.sampling_only:
            logger.info("Running sampling-only mode...")
            batches_info = voting_manager.sampler.create_random_groups()
            sampling_summary = voting_manager._get_batches_summary(batches_info)

            logger.info("Sampling completed successfully!")
            logger.info(f"Created {sampling_summary['total_groups_created']} groups in {sampling_summary['total_batches_created']} batches")
            logger.info(f"Output directory: {sampling_summary['batches_directory']}")
            return 0

        elif args.step_only:
            if args.dry_run:
                logger.info(f"Dry run for single step: {args.step_only}")
                logger.info("Dry run completed. No step was executed.")
                return 0

            logger.info(f"Running single step: {args.step_only}")
            results = voting_manager.run_step(args.step_only)

        elif args.resume_from:
            if args.dry_run:
                logger.info(f"Dry run for resuming from: {args.resume_from}")
                logger.info("Dry run completed. No workflow was resumed.")
                return 0

            logger.info(f"Resuming workflow from: {args.resume_from}")
            results = voting_manager.resume_from_step(args.resume_from)

        else:
            logger.info("Starting complete occurrence voting workflow...")
            results = voting_manager.run_complete_workflow()

        # Display final summary
        logger.info("=" * 70)
        logger.info("WORKFLOW COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)

        if 'sampling' in results:
            sampling = results['sampling']
            logger.info(f"Groups created: {sampling['total_groups_created']}")

        if 'occurrence_voting' in results:
            voting = results['occurrence_voting']
            logger.info(f"Structures analyzed: {voting['total_structures_analyzed']}")
            logger.info(f"Structures passed filter: {voting['structures_passed_filter']}")
            logger.info(f"Unique sequences found: {voting['unique_sequences_found']}")

        if 'final_results' in results:
            final = results['final_results']
            logger.info(f"Top sequences selected: {final['top_n_selected']}")
            logger.info(f"Final A3M file: {final['final_a3m_file']}")
            logger.info(f"Summary report: {final['summary_report']}")

        logger.info("Occurrence voting analysis completed successfully!")
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