#!/usr/bin/env python3
"""
Simple Structure Plotting Tool

A utility script that analyzes PDB files in a directory using structure metrics
and generates a single 2D scatter plot of the first two metrics from config JSON,
colored by pLDDT with customizable axis limits and ticks.

Usage:
    python scripts/simple_structure_plot_tool.py /path/to/config.json /path/to/pdb_folder
    python scripts/simple_structure_plot_tool.py /path/to/config.json /path/to/pdb_folder --output-dir /path/to/plots
    python scripts/simple_structure_plot_tool.py /path/to/config.json /path/to/pdb_folder --x-lim 0 1 --y-lim 0 100 --x-ticks 0 0.2 0.4 0.6 0.8 1.0

Example:
    python scripts/simple_structure_plot_tool.py config_examples/structure_analysis_config.json results/predictions/ --x-lim 0 1 --y-lim 50 100
"""

import sys
import os
import argparse
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# AF_ClaSeq imports
from af_claseq.utils.structure_analysis import StructureAnalyzer, load_filter_modes
from af_claseq.utils.plotting_manager import create_2d_scatter_plot
from af_claseq.utils.logging_utils import get_logger


def setup_simple_logger() -> logging.Logger:
    """Set up a simple logger for the tool"""
    logger = logging.getLogger("simple_structure_plot_tool")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def find_pdb_files(pdb_folder: str) -> List[str]:
    """
    Find all PDB files in the given folder (including subdirectories).

    Args:
        pdb_folder: Path to folder containing PDB files

    Returns:
        List of PDB file paths
    """
    pdb_folder = Path(pdb_folder)
    if not pdb_folder.exists():
        raise FileNotFoundError(f"PDB folder not found: {pdb_folder}")

    # Find PDB files recursively
    pdb_files = []
    for pattern in ['*.pdb', '**/*.pdb']:
        pdb_files.extend(list(pdb_folder.glob(pattern)))

    # Convert to strings and remove duplicates
    pdb_files = list(set(str(pdb) for pdb in pdb_files))

    if not pdb_files:
        raise ValueError(f"No PDB files found in {pdb_folder}")

    return sorted(pdb_files)


def analyze_structures(pdb_files: List[str], config_file: str, logger: logging.Logger) -> pd.DataFrame:
    """
    Analyze structures using the provided configuration.

    Args:
        pdb_files: List of PDB file paths
        config_file: Path to structure analysis config JSON
        logger: Logger instance

    Returns:
        DataFrame with analysis results
    """
    logger.info(f"Analyzing {len(pdb_files)} PDB files...")

    # Load structure analysis configuration
    try:
        structure_config = load_filter_modes(config_file)
    except Exception as e:
        raise ValueError(f"Failed to load structure config: {e}")

    filter_criteria = structure_config.get('filter_criteria', [])
    basics = structure_config.get('basics', {})
    composite_metrics = structure_config.get('composite_metrics', [])

    if not filter_criteria:
        raise ValueError("No filter criteria found in configuration file")

    logger.info(f"Using {len(filter_criteria)} filter criteria from config")

    # Initialize structure analyzer
    structure_analyzer = StructureAnalyzer()

    # Run structure analysis
    try:
        analysis_results = structure_analyzer.process_pdbs_parallel(
            pdb_files=pdb_files,
            filter_criteria=filter_criteria,
            basics=basics,
            plddt_threshold=0,  # Include all structures
            n_jobs=-1,  # Use all available cores
            composite_metrics=composite_metrics
        )

        # Convert results to DataFrame
        results_list = []
        for pdb_path, result in analysis_results.items():
            if result is not None:
                result['PDB'] = pdb_path
                results_list.append(result)

        if not results_list:
            raise ValueError("No structures passed analysis criteria")

        results_df = pd.DataFrame(results_list)
        logger.info(f"Successfully analyzed {len(results_df)} structures")

        return results_df

    except Exception as e:
        raise RuntimeError(f"Structure analysis failed: {e}")


def generate_single_plot(results_df: pd.DataFrame, config_file: str, output_dir: str,
                        x_lim: List[float] = None, y_lim: List[float] = None,
                        x_ticks: List[float] = None, y_ticks: List[float] = None,
                        logger: logging.Logger = None) -> str:
    """
    Generate a single 2D scatter plot using the first two metrics from config.

    Args:
        results_df: DataFrame with structure analysis results
        config_file: Path to structure analysis config JSON
        output_dir: Output directory for plots
        x_lim: X-axis limits [min, max]
        y_lim: Y-axis limits [min, max]
        x_ticks: Custom X-axis tick positions
        y_ticks: Custom Y-axis tick positions
        logger: Logger instance

    Returns:
        Path to generated plot file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load structure analysis configuration to get metric order
    try:
        structure_config = load_filter_modes(config_file)
        filter_criteria = structure_config.get('filter_criteria', [])

        if len(filter_criteria) < 2:
            raise ValueError(f"Config must have at least 2 filter criteria. Found: {len(filter_criteria)}")

        # Get first two metrics from config
        metric1 = filter_criteria[0]['name']
        metric2 = filter_criteria[1]['name']

    except Exception as e:
        raise ValueError(f"Failed to get metrics from config: {e}")

    # Verify metrics exist in results
    if metric1 not in results_df.columns:
        raise ValueError(f"Metric '{metric1}' not found in analysis results")
    if metric2 not in results_df.columns:
        raise ValueError(f"Metric '{metric2}' not found in analysis results")

    logger.info(f"Creating plot: {metric1} (X) vs {metric2} (Y)")

    try:
        # Create plot parameters
        plot_params = {}
        if x_lim is not None:
            plot_params['x_min'] = x_lim[0]
            plot_params['x_max'] = x_lim[1]
        if y_lim is not None:
            plot_params['y_min'] = y_lim[0]
            plot_params['y_max'] = y_lim[1]
        if x_ticks is not None:
            plot_params['x_ticks'] = x_ticks
        if y_ticks is not None:
            plot_params['y_ticks'] = y_ticks

        # Create 2D scatter plot using plotting manager
        plot_path = create_2d_scatter_plot(
            results_df=results_df,
            metric_name1=metric1,
            metric_name2=metric2,
            output_dir=str(output_dir),
            color_metric='plddt',  # Color by pLDDT
            title=None,
            logger=logger,
            **plot_params
        )

        if plot_path:
            logger.info(f"Plot saved: {plot_path}")
            return plot_path
        else:
            raise RuntimeError("Failed to generate plot")

    except Exception as e:
        raise RuntimeError(f"Failed to generate plot for {metric1} vs {metric2}: {e}")


def save_results_csv(results_df: pd.DataFrame, output_dir: str, logger: logging.Logger) -> str:
    """
    Save analysis results to CSV file.

    Args:
        results_df: DataFrame with results
        output_dir: Output directory
        logger: Logger instance

    Returns:
        Path to saved CSV file
    """
    output_dir = Path(output_dir)
    csv_path = output_dir / "structure_analysis_results.csv"

    try:
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to: {csv_path}")
        return str(csv_path)
    except Exception as e:
        logger.warning(f"Failed to save results CSV: {e}")
        return ""


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Simple Structure Plotting Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s config.json /path/to/pdbs/
  %(prog)s config.json /path/to/pdbs/ --output-dir /path/to/plots/
  %(prog)s config.json /path/to/pdbs/ --x-lim 0 1 --y-lim 50 100
  %(prog)s config.json /path/to/pdbs/ --x-ticks 0 0.2 0.4 0.6 0.8 1.0 --y-ticks 50 60 70 80 90 100
  %(prog)s config.json /path/to/pdbs/ --x-lim 0 1 --y-lim 0 100 --save-csv

This tool will:
1. Find all PDB files in the specified folder (recursively)
2. Analyze them using the provided structure metrics config
3. Generate ONE 2D scatter plot using the first two metrics from config
4. Color the plot by pLDDT values
5. Allow custom axis limits and tick positions
6. Optionally save analysis results to CSV
        """
    )

    parser.add_argument(
        'config_file',
        type=str,
        help='Path to structure analysis configuration JSON file'
    )

    parser.add_argument(
        'pdb_folder',
        type=str,
        help='Path to folder containing PDB files'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='structure_plots',
        help='Output directory for plots (default: structure_plots)'
    )

    parser.add_argument(
        '--x-lim',
        type=float,
        nargs=2,
        metavar=('MIN', 'MAX'),
        help='X-axis limits (e.g., --x-lim 0 1)'
    )

    parser.add_argument(
        '--y-lim',
        type=float,
        nargs=2,
        metavar=('MIN', 'MAX'),
        help='Y-axis limits (e.g., --y-lim 50 100)'
    )

    parser.add_argument(
        '--x-ticks',
        type=float,
        nargs='+',
        help='Custom X-axis tick positions (e.g., --x-ticks 0 0.2 0.4 0.6 0.8 1.0)'
    )

    parser.add_argument(
        '--y-ticks',
        type=float,
        nargs='+',
        help='Custom Y-axis tick positions (e.g., --y-ticks 50 60 70 80 90 100)'
    )

    parser.add_argument(
        '--save-csv',
        action='store_true',
        help='Save analysis results to CSV file'
    )

    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_arguments()

    # Setup logging
    logger = setup_simple_logger()
    logger.info("Starting Simple Structure Plotting Tool...")

    try:
        # Validate inputs
        config_file = Path(args.config_file)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        pdb_folder = Path(args.pdb_folder)
        if not pdb_folder.exists():
            raise FileNotFoundError(f"PDB folder not found: {pdb_folder}")

        logger.info(f"Config file: {config_file}")
        logger.info(f"PDB folder: {pdb_folder}")
        logger.info(f"Output directory: {args.output_dir}")

        # Find PDB files
        pdb_files = find_pdb_files(str(pdb_folder))
        logger.info(f"Found {len(pdb_files)} PDB files")

        # Analyze structures
        results_df = analyze_structures(pdb_files, str(config_file), logger)

        # Generate single plot
        plot_path = generate_single_plot(
            results_df=results_df,
            config_file=str(config_file),
            output_dir=args.output_dir,
            x_lim=args.x_lim,
            y_lim=args.y_lim,
            x_ticks=args.x_ticks,
            y_ticks=args.y_ticks,
            logger=logger
        )

        # Save CSV if requested
        csv_path = ""
        if args.save_csv:
            csv_path = save_results_csv(results_df, args.output_dir, logger)

        # Display summary
        logger.info("=" * 50)
        logger.info("ANALYSIS COMPLETED SUCCESSFULLY")
        logger.info("=" * 50)
        logger.info(f"Structures analyzed: {len(results_df)}")
        logger.info(f"Plot generated: {plot_path}")
        logger.info(f"Output directory: {args.output_dir}")
        if csv_path:
            logger.info(f"Results CSV: {csv_path}")

        # Show plot parameters
        if args.x_lim:
            logger.info(f"X-axis limits: {args.x_lim}")
        if args.y_lim:
            logger.info(f"Y-axis limits: {args.y_lim}")
        if args.x_ticks:
            logger.info(f"X-axis ticks: {args.x_ticks}")
        if args.y_ticks:
            logger.info(f"Y-axis ticks: {args.y_ticks}")

        return 0

    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        return 1

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)