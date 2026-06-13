#!/usr/bin/env python3
"""
Sequence filtering script based on structure analysis results.

Simple workflow:
1. Read structure analysis CSV file
2. Filter PDB structures by metric cutoffs (above/below thresholds)
3. Convert PDB paths to corresponding A3M files
4. Collect unique sequences from filtered A3M files
5. Write sequences to new A3M file
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

from af_claseq.utils.sequence_processing import read_a3m_to_dict, write_a3m
from af_claseq.utils.logging_utils import get_logger

# Initialize logger
logger = get_logger("filter_sequences")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Filter sequences based on structure analysis results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Filter by criteria: 6xr6_composite_rmsd < 3.0 AND 6xrg_composite_rmsd < 3.0
  python filter_sequences.py results.csv \\
    --criteria "6xr6_composite_rmsd<3.0" "6xrg_composite_rmsd<3.0" \\
    --output filtered_sequences.a3m

  # OR logic: high performers in either metric
  python filter_sequences.py results.csv \\
    --criteria "6xr6_composite_rmsd<2.0" "6xrg_composite_rmsd<2.0" --combine_method any \\
    --output high_performers.a3m

  # Top-N selection: best 100 structures by composite RMSD (lowest values)
  python filter_sequences.py results.csv \\
    --top_n 6xr6_composite_rmsd:100:min \\
    --output top_100_sequences.a3m

  # Percentile selection: bottom 5%% by RMSD (lowest values)
  python filter_sequences.py results.csv \\
    --percentile "6nc7_rmsd:5:min" \\
    --output bottom_5pct.a3m

  # Percentile selection: top 10%% by TM-score (highest values)
  python filter_sequences.py results.csv \\
    --percentile "6nc7_tmscore:10:max" "6nc9_tmscore:10:max" \\
    --output top_10pct.a3m
        """
    )
    
    parser.add_argument(
        'csv_file',
        help='Path to structure analysis results CSV file'
    )
    
    # Create mutually exclusive group for filtering methods
    filter_group = parser.add_mutually_exclusive_group(required=True)
    
    # Criteria-based filtering
    filter_group.add_argument(
        '--criteria',
        nargs='+',
        help='Metric criteria in format "metric_name>value" or "metric_name<value" (e.g., "5jyt_tmscore>0.8" "2qke_tmscore<0.5")'
    )
    
    # Top-N selection
    filter_group.add_argument(
        '--top_n',
        nargs='+',
        help='Top-N selection in format "metric_name:N:direction" where direction is "max" or "min" (e.g., "5jyt_tmscore:100:max" "2qke_tmscore:50:min")'
    )

    # Percentile selection
    filter_group.add_argument(
        '--percentile',
        nargs='+',
        help='Percentile selection in format "metric_name:P:direction" where P is the percentile (1-99) and direction is "max" (top P%%) or "min" (bottom P%%) (e.g., "6nc7_rmsd:5:min" selects bottom 5%% by RMSD)'
    )
    
    parser.add_argument(
        '--output',
        required=True,
        help='Output A3M file path'
    )
    
    parser.add_argument(
        '--query_a3m',
        help='Path to A3M file containing the query sequence (first entry). '
             'If provided, the query is prepended to the output. '
             'If not provided, the query is auto-extracted from the first filtered A3M file.'
    )

    parser.add_argument(
        '--combine_method',
        choices=['all', 'any'],
        default='all',
        help='For multiple metrics: "all" (AND logic) or "any" (OR logic). Default: all'
    )

    parser.add_argument(
        '--min_occurrence',
        type=int,
        default=1,
        help='Minimum number of times a sequence must appear across filtered A3M files to be kept. Default: 1 (no filtering)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def parse_criteria(criteria_list: List[str]) -> List[Tuple[str, str, float]]:
    """Parse criteria strings into (metric, operator, value) tuples."""
    parsed_criteria = []

    for criterion in criteria_list:
        if '>' in criterion:
            metric, value_str = criterion.split('>', 1)
            operator = '>'
        elif '<' in criterion:
            metric, value_str = criterion.split('<', 1)
            operator = '<'
        else:
            raise ValueError(f"Invalid criterion format: {criterion}")

        try:
            value = float(value_str.strip())
        except ValueError:
            raise ValueError(f"Invalid numeric value: {criterion}")

        parsed_criteria.append((metric.strip(), operator, value))

    return parsed_criteria


def parse_top_n(top_n_list: List[str]) -> List[Tuple[str, int, str]]:
    """Parse top-N strings into (metric, N, direction) tuples."""
    parsed_top_n = []

    for top_n_spec in top_n_list:
        parts = top_n_spec.split(':')
        if len(parts) != 3:
            raise ValueError(f"Invalid top-N format (expected metric:N:direction): {top_n_spec}")

        metric, n_str, direction = [part.strip() for part in parts]

        try:
            n = int(n_str)
            if n <= 0:
                raise ValueError(f"N must be positive: {n}")
        except ValueError:
            raise ValueError(f"Invalid N value: {top_n_spec}")

        if direction not in ['max', 'min']:
            raise ValueError(f"Invalid direction (must be 'max' or 'min'): {direction}")

        parsed_top_n.append((metric, n, direction))

    return parsed_top_n


def load_structure_results(csv_file: str) -> pd.DataFrame:
    """Load structure analysis results from CSV file."""
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    df = pd.read_csv(csv_file)
    logger.info(f"Loaded {len(df)} structure results from {csv_file}")

    return df


def filter_structures_criteria(
    df: pd.DataFrame,
    criteria: List[Tuple[str, str, float]],
    combine_method: str = 'all'
) -> pd.DataFrame:
    """Filter structures based on metric criteria."""
    # Check if all metrics exist
    metrics = [criterion[0] for criterion in criteria]
    missing_metrics = [m for m in metrics if m not in df.columns]
    if missing_metrics:
        raise ValueError(f"Metrics not found in CSV: {missing_metrics}")

    # Apply filtering logic
    conditions = []
    for metric, operator, cutoff in criteria:
        condition = df[metric] > cutoff if operator == '>' else df[metric] < cutoff
        conditions.append(condition)
        logger.info(f"Filtering: {metric} {operator} {cutoff}")

    # Combine conditions
    if len(conditions) == 1:
        final_condition = conditions[0]
    elif combine_method == 'all':
        final_condition = conditions[0]
        for condition in conditions[1:]:
            final_condition &= condition
        logger.info("Combining criteria with AND logic")
    else:  # any
        final_condition = conditions[0]
        for condition in conditions[1:]:
            final_condition |= condition
        logger.info("Combining criteria with OR logic")

    filtered_df = df[final_condition].copy()
    logger.info(f"Structures passing filter: {len(filtered_df)} out of {len(df)}")

    return filtered_df


def filter_structures_top_n(
    df: pd.DataFrame,
    top_n_specs: List[Tuple[str, int, str]]
) -> pd.DataFrame:
    """Filter structures by top-N selection (union of all top-N selections)."""
    # Check if all metrics exist
    metrics = [spec[0] for spec in top_n_specs]
    missing_metrics = [m for m in metrics if m not in df.columns]
    if missing_metrics:
        raise ValueError(f"Metrics not found in CSV: {missing_metrics}")

    all_selected_indices = set()

    # Get top-N from each metric and combine them
    for metric, n, direction in top_n_specs:
        logger.info(f"Selecting top {n} by {metric} ({'highest' if direction == 'max' else 'lowest'} values)")

        # Sort by metric
        ascending = direction == 'min'
        sorted_df = df.sort_values(metric, ascending=ascending)

        # Take top N indices
        n_actual = min(n, len(sorted_df))
        top_n_indices = sorted_df.head(n_actual).index

        logger.info(f"Selected {n_actual} structures from {metric}")
        all_selected_indices.update(top_n_indices)

    # Create filtered DataFrame from combined indices
    filtered_df = df.loc[list(all_selected_indices)].copy()
    logger.info(f"Combined selection: {len(filtered_df)} unique structures out of {len(df)} total")

    return filtered_df


def parse_percentile(percentile_list: List[str]) -> List[Tuple[str, float, str]]:
    """Parse percentile strings into (metric, percentile, direction) tuples."""
    parsed = []

    for spec in percentile_list:
        parts = spec.split(':')
        if len(parts) != 3:
            raise ValueError(f"Invalid percentile format (expected metric:P:direction): {spec}")

        metric, pct_str, direction = [part.strip() for part in parts]

        try:
            pct = float(pct_str)
            if not (0 < pct < 100):
                raise ValueError()
        except ValueError:
            raise ValueError(f"Percentile must be a number between 0 and 100 (exclusive): {spec}")

        if direction not in ['max', 'min']:
            raise ValueError(f"Invalid direction (must be 'max' or 'min'): {direction}")

        parsed.append((metric, pct, direction))

    return parsed


def filter_structures_percentile(
    df: pd.DataFrame,
    percentile_specs: List[Tuple[str, float, str]]
) -> pd.DataFrame:
    """Filter structures by percentile selection (union of all percentile selections).

    For direction='min': selects structures in the bottom P% (lowest values).
    For direction='max': selects structures in the top P% (highest values).
    """
    metrics = [spec[0] for spec in percentile_specs]
    missing_metrics = [m for m in metrics if m not in df.columns]
    if missing_metrics:
        raise ValueError(f"Metrics not found in CSV: {missing_metrics}")

    all_selected_indices = set()

    for metric, pct, direction in percentile_specs:
        if direction == 'min':
            threshold = df[metric].quantile(pct / 100)
            selected = df[df[metric] <= threshold]
            logger.info(f"Percentile: bottom {pct}% of {metric} (threshold={threshold:.4f})")
        else:
            threshold = df[metric].quantile(1 - pct / 100)
            selected = df[df[metric] >= threshold]
            logger.info(f"Percentile: top {pct}% of {metric} (threshold={threshold:.4f})")

        logger.info(f"Selected {len(selected)} structures from {metric}")
        all_selected_indices.update(selected.index)

    filtered_df = df.loc[list(all_selected_indices)].copy()
    logger.info(f"Combined selection: {len(filtered_df)} unique structures out of {len(df)} total")

    return filtered_df


def pdb_to_a3m_path(pdb_path: str) -> str:
    """Convert PDB path to corresponding A3M path."""
    pdb_path = str(pdb_path)

    # Split by '_unrelaxed' and take the first part (common ColabFold pattern)
    if '_unrelaxed' in pdb_path:
        base_path = pdb_path.split('_unrelaxed')[0]
        return base_path + '.a3m'

    # Fallback: replace .pdb extension with .a3m
    return os.path.splitext(pdb_path)[0] + '.a3m'


def collect_sequences(pdb_paths: List[str], verbose: bool = False,
                      min_occurrence: int = 1) -> Tuple[Tuple[str, str], Dict[str, str]]:
    """Collect unique non-query sequences from A3M files corresponding to PDB paths.

    Args:
        pdb_paths: List of PDB file paths to find corresponding A3M files
        verbose: Enable verbose logging
        min_occurrence: Minimum number of A3M files a sequence must appear in to be kept

    Returns:
        Tuple of (query_header, query_sequence) and dict of filtered unique sequences.
    """
    from collections import Counter

    seq_counts = Counter()
    seq_to_header = {}
    missing_files = []
    query_entry = None

    logger.info(f"Collecting sequences from {len(pdb_paths)} A3M files...")
    if min_occurrence > 1:
        logger.info(f"Minimum occurrence threshold: {min_occurrence}")

    for i, pdb_path in enumerate(pdb_paths, 1):
        a3m_path = pdb_to_a3m_path(pdb_path)

        if verbose:
            logger.info(f"{i:3d}. Processing: {os.path.basename(a3m_path)}")

        if not os.path.exists(a3m_path):
            missing_files.append(a3m_path)
            if verbose:
                logger.warning(f"A3M file not found: {a3m_path}")
            continue

        try:
            sequences_dict = read_a3m_to_dict(a3m_path)

            if not sequences_dict:
                if verbose:
                    logger.warning(f"No sequences in {a3m_path}")
                continue

            sequence_items = list(sequences_dict.items())

            if query_entry is None:
                query_entry = sequence_items[0]

            non_query_sequences = sequence_items[1:]

            seen_in_this_file = set()
            for header, sequence in non_query_sequences:
                if sequence not in seen_in_this_file:
                    seen_in_this_file.add(sequence)
                    seq_counts[sequence] += 1
                    if sequence not in seq_to_header:
                        seq_to_header[sequence] = header

        except Exception as e:
            if verbose:
                logger.error(f"Error reading {a3m_path}: {e}")
            continue

    if missing_files:
        logger.warning(f"{len(missing_files)} A3M files were not found")
        if verbose:
            for missing_file in missing_files[:5]:
                logger.warning(f"Missing: {missing_file}")
            if len(missing_files) > 5:
                logger.warning(f"... and {len(missing_files)-5} more")

    total_unique = len(seq_counts)
    unique_sequences = {}
    for sequence, count in seq_counts.items():
        if count >= min_occurrence:
            header = seq_to_header[sequence]
            unique_sequences[header] = sequence

    logger.info(f"Total unique sequences found: {total_unique}")
    if min_occurrence > 1:
        logger.info(f"Sequences with >= {min_occurrence} occurrences: {len(unique_sequences)}")
        max_count = max(seq_counts.values()) if seq_counts else 0
        logger.info(f"Occurrence distribution: max={max_count}, "
                     f">=2: {sum(1 for c in seq_counts.values() if c >= 2)}, "
                     f">=5: {sum(1 for c in seq_counts.values() if c >= 5)}, "
                     f">=10: {sum(1 for c in seq_counts.values() if c >= 10)}")

    return query_entry, unique_sequences


def write_filter_log(log_path: str, csv_file: str, filter_type: str, filter_details: str,
                     total_structures: int, filtered_structures: int, sequences_written: int,
                     output_path: str, metrics_summary: Dict[str, str] = None):
    """Write a detailed log file of the filtering operation."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_path, 'w') as f:
        f.write("SEQUENCE FILTERING LOG\n")
        f.write("=" * 50 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Script: filter_sequences.py\n\n")

        f.write("INPUT PARAMETERS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Input CSV file: {csv_file}\n")
        f.write(f"Filter type: {filter_type}\n")
        f.write(f"Filter details: {filter_details}\n")
        f.write(f"Output A3M file: {output_path}\n\n")

        f.write("FILTERING RESULTS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total structures in CSV: {total_structures}\n")
        f.write(f"Structures passing filter: {filtered_structures}\n")
        f.write(f"Filtering rate: {filtered_structures/total_structures*100:.2f}%\n")
        f.write(f"Unique sequences written: {sequences_written}\n\n")

        if metrics_summary:
            f.write("METRICS SUMMARY:\n")
            f.write("-" * 20 + "\n")
            for metric, summary in metrics_summary.items():
                f.write(f"{metric}: {summary}\n")
            f.write("\n")

        f.write("FILES GENERATED:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Output A3M: {output_path}\n")
        f.write(f"Filter log: {log_path}\n")


def main():
    """Main function."""
    args = parse_arguments()

    try:
        logger.info("=" * 60)
        logger.info("SEQUENCE FILTERING SCRIPT")
        logger.info("=" * 60)

        # Load structure analysis results
        df = load_structure_results(args.csv_file)

        # Filter structures based on mode
        if args.criteria:
            criteria = parse_criteria(args.criteria)
            filtered_df = filter_structures_criteria(df, criteria, args.combine_method)
            metrics_used = [criterion[0] for criterion in criteria]
            filter_type = "criteria"
            filter_details = f"Criteria: {args.criteria}, Combine method: {args.combine_method}"
        elif args.top_n:
            top_n_specs = parse_top_n(args.top_n)
            filtered_df = filter_structures_top_n(df, top_n_specs)
            metrics_used = [spec[0] for spec in top_n_specs]
            filter_type = "top_n"
            filter_details = f"Top-N selection: {args.top_n}"
        else:
            percentile_specs = parse_percentile(args.percentile)
            filtered_df = filter_structures_percentile(df, percentile_specs)
            metrics_used = [spec[0] for spec in percentile_specs]
            filter_type = "percentile"
            filter_details = f"Percentile selection: {args.percentile}"

        if len(filtered_df) == 0:
            logger.warning("No structures meet the filtering criteria!")
            return

        # Show filtering summary
        logger.info("Filtering Summary:")
        metrics_summary = {}
        for metric in metrics_used:
            if metric in filtered_df.columns:
                values = filtered_df[metric]
                summary = f"{values.min():.4f} - {values.max():.4f} (mean: {values.mean():.4f})"
                metrics_summary[metric] = summary
                logger.info(f"  {metric}: {summary}")

        # Get PDB paths that meet criteria
        pdb_paths = filtered_df['PDB'].tolist()

        if args.verbose:
            logger.info("Filtered PDB files:")
            for i, pdb_path in enumerate(pdb_paths[:10], 1):  # Show first 10
                logger.info(f"  {i:2d}. {os.path.basename(pdb_path)}")
            if len(pdb_paths) > 10:
                logger.info(f"  ... and {len(pdb_paths)-10} more")

        # Collect sequences from corresponding A3M files
        query_entry, sequences = collect_sequences(pdb_paths, args.verbose,
                                                     min_occurrence=args.min_occurrence)

        if not sequences:
            logger.warning("No sequences were collected!")
            return

        # Resolve query sequence: explicit --query_a3m takes priority, else auto-extracted
        if args.query_a3m:
            query_seqs = read_a3m_to_dict(args.query_a3m)
            if not query_seqs:
                logger.error(f"No sequences found in query A3M: {args.query_a3m}")
                return
            q_header, q_seq = list(query_seqs.items())[0]
            logger.info(f"Query sequence from --query_a3m: {q_header}")
        elif query_entry:
            q_header, q_seq = query_entry
            logger.info(f"Query sequence auto-extracted: {q_header}")
        else:
            logger.error("No query sequence found. Provide --query_a3m.")
            return

        # Prepend query sequence to output
        output_sequences = {q_header: q_seq}
        output_sequences.update(sequences)
        logger.info(f"Output: 1 query + {len(sequences)} enriched sequences = {len(output_sequences)} total")

        # Write combined A3M file using existing utility
        write_a3m(output_sequences, args.output)

        # Generate log file path (same directory as output, with .log extension)
        output_path = Path(args.output)
        log_path = output_path.parent / (output_path.stem + "_filter.log")

        # Write detailed log file
        write_filter_log(
            log_path=str(log_path),
            csv_file=args.csv_file,
            filter_type=filter_type,
            filter_details=filter_details,
            total_structures=len(df),
            filtered_structures=len(filtered_df),
            sequences_written=len(output_sequences),
            output_path=args.output,
            metrics_summary=metrics_summary
        )

        logger.info("=" * 60)
        logger.info("FILTERING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Structures filtered: {len(filtered_df)} out of {len(df)}")
        logger.info(f"Unique sequences collected: {len(output_sequences)} (1 query + {len(sequences)} enriched)")
        logger.info(f"Output A3M file: {args.output}")
        logger.info(f"Filter log file: {log_path}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()