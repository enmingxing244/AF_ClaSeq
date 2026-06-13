#!/usr/bin/env python3
"""
Coverage filter script for A3M files.

This script filters sequences in A3M files based on coverage compared to the query sequence.
Coverage is calculated as: (1 - gaps/query_length) >= threshold

Usage:
    python coverage_filter.py input.a3m --threshold 0.8 --output filtered.a3m
    python coverage_filter.py input.a3m -t 0.7 -o filtered.a3m --verbose
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict

from af_claseq.utils.sequence_processing import read_a3m_to_dict, write_a3m, filter_a3m_by_coverage
from af_claseq.utils.logging_utils import get_logger

# Initialize logger
logger = get_logger("coverage_filter")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Filter A3M sequences based on coverage threshold',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Filter sequences with at least 80% coverage
  python coverage_filter.py input.a3m --output filtered.a3m

  # Filter with 70% coverage threshold
  python coverage_filter.py input.a3m -t 0.7 -o filtered.a3m

  # Verbose output with custom threshold
  python coverage_filter.py input.a3m -t 0.9 -o high_coverage.a3m --verbose
        """
    )

    parser.add_argument(
        'input_a3m',
        help='Input A3M file path'
    )

    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output A3M file path'
    )

    parser.add_argument(
        '-t', '--threshold',
        type=float,
        default=0.8,
        help='Coverage threshold (0.0-1.0). Default: 0.8 (80%% coverage)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    return parser.parse_args()


def validate_threshold(threshold: float) -> None:
    """Validate coverage threshold value."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Coverage threshold must be between 0.0 and 1.0, got: {threshold}")


def calculate_coverage_stats(sequences: Dict[str, str], query_length: int) -> Dict[str, float]:
    """Calculate coverage statistics for all sequences."""
    coverage_stats = {}

    for header, seq in sequences.items():
        gaps = seq.count('-')
        coverage = 1 - (gaps / query_length)
        coverage_stats[header] = coverage

    return coverage_stats


def write_coverage_log(log_path: str, input_file: str, threshold: float,
                      total_sequences: int, filtered_sequences: int,
                      output_path: str, coverage_stats: Dict[str, float] = None):
    """Write a detailed log file of the coverage filtering operation."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_path, 'w') as f:
        f.write("COVERAGE FILTERING LOG\n")
        f.write("=" * 50 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Script: coverage_filter.py\n\n")

        f.write("INPUT PARAMETERS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Input A3M file: {input_file}\n")
        f.write(f"Coverage threshold: {threshold:.2f} ({threshold*100:.1f}%)\n")
        f.write(f"Output A3M file: {output_path}\n\n")

        f.write("FILTERING RESULTS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total sequences: {total_sequences}\n")
        f.write(f"Sequences passing filter: {filtered_sequences}\n")
        f.write(f"Filtering rate: {filtered_sequences/total_sequences*100:.2f}%\n")
        f.write(f"Sequences removed: {total_sequences - filtered_sequences}\n\n")

        if coverage_stats:
            coverages = list(coverage_stats.values())
            f.write("COVERAGE STATISTICS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Min coverage: {min(coverages):.4f} ({min(coverages)*100:.2f}%)\n")
            f.write(f"Max coverage: {max(coverages):.4f} ({max(coverages)*100:.2f}%)\n")
            f.write(f"Mean coverage: {sum(coverages)/len(coverages):.4f} ({sum(coverages)/len(coverages)*100:.2f}%)\n\n")

        f.write("FILES GENERATED:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Output A3M: {output_path}\n")
        f.write(f"Coverage log: {log_path}\n")


def main():
    """Main function."""
    args = parse_arguments()

    try:
        logger.info("=" * 60)
        logger.info("COVERAGE FILTERING SCRIPT")
        logger.info("=" * 60)

        # Validate inputs
        if not os.path.exists(args.input_a3m):
            raise FileNotFoundError(f"Input A3M file not found: {args.input_a3m}")

        validate_threshold(args.threshold)

        logger.info(f"Input file: {args.input_a3m}")
        logger.info(f"Coverage threshold: {args.threshold:.2f} ({args.threshold*100:.1f}%)")
        logger.info(f"Output file: {args.output}")

        # Read A3M file
        logger.info("Reading A3M file...")
        sequences = read_a3m_to_dict(args.input_a3m)

        if not sequences:
            raise ValueError("No sequences found in input A3M file")

        total_sequences = len(sequences)
        logger.info(f"Loaded {total_sequences} sequences")

        # Get query sequence info
        query_header, query_seq = next(iter(sequences.items()))
        query_length = len(query_seq)
        logger.info(f"Query sequence length: {query_length}")
        logger.info(f"Query header: {query_header}")

        # Calculate coverage statistics for all sequences
        if args.verbose:
            logger.info("Calculating coverage statistics...")
            coverage_stats = calculate_coverage_stats(sequences, query_length)

            logger.info("Coverage distribution:")
            sorted_coverages = sorted(coverage_stats.items(), key=lambda x: x[1], reverse=True)

            # Show top 5 and bottom 5
            logger.info("  Top 5 highest coverage:")
            for i, (header, coverage) in enumerate(sorted_coverages[:5], 1):
                logger.info(f"    {i}. {coverage:.4f} ({coverage*100:.2f}%) - {header}")

            if len(sorted_coverages) > 10:
                logger.info("  ...")

            logger.info("  Bottom 5 lowest coverage:")
            for i, (header, coverage) in enumerate(sorted_coverages[-5:], 1):
                logger.info(f"    {i}. {coverage:.4f} ({coverage*100:.2f}%) - {header}")
        else:
            coverage_stats = calculate_coverage_stats(sequences, query_length)

        # Apply coverage filter
        logger.info(f"Applying coverage filter (threshold >= {args.threshold:.2f})...")
        filtered_sequences = filter_a3m_by_coverage(sequences, args.threshold)

        filtered_count = len(filtered_sequences)
        removed_count = total_sequences - filtered_count

        logger.info(f"Sequences passing filter: {filtered_count}/{total_sequences}")
        logger.info(f"Sequences removed: {removed_count} ({removed_count/total_sequences*100:.2f}%)")

        if filtered_count == 0:
            logger.warning("No sequences meet the coverage threshold!")
            return

        # Write filtered A3M file
        logger.info("Writing filtered A3M file...")
        write_a3m(filtered_sequences, args.output)

        # Generate log file path (same directory as output, with .log extension)
        output_path = Path(args.output)
        log_path = output_path.parent / (output_path.stem + "_coverage.log")

        # Write detailed log file
        write_coverage_log(
            log_path=str(log_path),
            input_file=args.input_a3m,
            threshold=args.threshold,
            total_sequences=total_sequences,
            filtered_sequences=filtered_count,
            output_path=args.output,
            coverage_stats=coverage_stats
        )

        logger.info("=" * 60)
        logger.info("COVERAGE FILTERING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Input sequences: {total_sequences}")
        logger.info(f"Output sequences: {filtered_count}")
        logger.info(f"Coverage threshold: {args.threshold:.2f} ({args.threshold*100:.1f}%)")
        logger.info(f"Output A3M file: {args.output}")
        logger.info(f"Coverage log file: {log_path}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()