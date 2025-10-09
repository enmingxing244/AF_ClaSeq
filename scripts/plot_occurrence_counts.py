#!/usr/bin/env python3
"""
Plot Occurrence Counts Tool

A simple script that reads an occurrence_counts.csv file and creates a bar graph
showing sequence occurrence counts with the top 16 sequences highlighted.

Usage:
    python scripts/plot_occurrence_counts.py /path/to/occurrence_counts.csv
    python scripts/plot_occurrence_counts.py /path/to/occurrence_counts.csv --output-dir plots
    python scripts/plot_occurrence_counts.py /path/to/occurrence_counts.csv --top-n 20

Example:
    python scripts/plot_occurrence_counts.py results/occurrence_counts.csv --top-n 16
"""

import sys
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add the src directory to the path so we can import from af_claseq
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from af_claseq.utils.plotting_manager import save_ai_compatible_plot


def plot_occurrence_counts(csv_file: str, output_dir: str = ".", top_n: int = 16,
                          max_display: int = 100) -> str:
    """
    Plot occurrence counts bar graph with top N sequences highlighted.

    Args:
        csv_file: Path to occurrence_counts.csv file
        output_dir: Output directory for the plot
        top_n: Number of top sequences to highlight
        max_display: Maximum number of sequences to display in plot

    Returns:
        Path to saved plot file
    """
    # Load data
    csv_path = Path(csv_file)
    if not csv_path.exists():
        raise FileNotFoundError(f"Occurrence counts CSV not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}")

    # Validate required columns
    required_cols = ['rank', 'occurrence_count']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")

    if df.empty:
        raise ValueError("Empty occurrence counts data")

    # Sort by rank to ensure proper order
    df = df.sort_values('rank').reset_index(drop=True)

    # Limit display to max_display sequences
    display_df = df.head(max_display)
    total_sequences = len(df)

    print(f"Loaded {total_sequences} sequences from {csv_path}")
    print(f"Displaying top {len(display_df)} sequences")
    print(f"Highlighting top {top_n} sequences")

    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 10))

    # Create x-axis labels (seq_1, seq_2, etc.)
    x_labels = [f"seq_{i+1}" for i in range(len(display_df))]
    x_positions = np.arange(len(display_df))

    # Create colors: highlight top N sequences
    colors = []
    for i in range(len(display_df)):
        if i < top_n:
            colors.append('#FF6B6B')  # Red for top N
        else:
            colors.append('#4ECDC4')  # Teal for others

    # Create bars
    bars = ax.bar(x_positions, display_df['occurrence_count'],
                  color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)

    # Add value labels on top of bars for top sequences
    max_count = display_df['occurrence_count'].max()
    for i, (bar, count) in enumerate(zip(bars, display_df['occurrence_count'])):
        if i < top_n:  # Only annotate top N sequences
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_count*0.01,
                   f'{count}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Styling
    ax.set_xlabel('Sequence Rank', fontsize=14, fontweight='bold')
    ax.set_ylabel('Occurrence Count', fontsize=14, fontweight='bold')
    ax.set_title(f'Sequence Occurrence Counts (Top {top_n} Highlighted)\n'
                f'Showing {len(display_df)} of {total_sequences} total sequences',
                fontsize=16, fontweight='bold', pad=20)

    # Set x-axis ticks and labels
    if len(display_df) <= 50:
        # Show every sequence for small datasets
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
    else:
        # Show every 5th or 10th sequence for larger datasets
        step = max(1, len(display_df) // 20)  # Show ~20 labels max
        tick_positions = x_positions[::step]
        tick_labels = [x_labels[i] for i in range(0, len(x_labels), step)]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF6B6B', edgecolor='black', label=f'Top {top_n} sequences'),
        Patch(facecolor='#4ECDC4', edgecolor='black', label='Other sequences')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

    # Add grid
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add statistics text box
    stats_text = (f'Statistics:\n'
                 f'Max: {display_df["occurrence_count"].max()}\n'
                 f'Min: {display_df["occurrence_count"].min()}\n'
                 f'Mean: {display_df["occurrence_count"].mean():.1f}\n'
                 f'Median: {display_df["occurrence_count"].median():.1f}')

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Adjust layout
    plt.tight_layout()

    # Save plot
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_filename = f"occurrence_counts_top{top_n}_highlighted.png"
    plot_path = output_dir / plot_filename

    # Save plot using AI-compatible format
    base_path = str(plot_path).replace('.png', '')
    save_ai_compatible_plot(plt.gcf(), base_path, dpi=300)
    plt.show()  # Display the plot

    print(f"Plot saved to: {base_path}.png, {base_path}.pdf, {base_path}.svg")
    return str(plot_path)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Plot Occurrence Counts with Top N Highlighted",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s occurrence_counts.csv
  %(prog)s occurrence_counts.csv --top-n 20
  %(prog)s occurrence_counts.csv --output-dir plots --top-n 16
  %(prog)s occurrence_counts.csv --max-display 50 --top-n 10

This tool will:
1. Read the occurrence_counts.csv file
2. Create a bar graph with x-axis as seq_1, seq_2, etc.
3. Y-axis shows occurrence counts
4. Highlight the top N sequences in a different color
5. Add value labels on top of highlighted bars
        """
    )

    parser.add_argument(
        'csv_file',
        type=str,
        help='Path to occurrence_counts.csv file'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='.',
        help='Output directory for the plot (default: current directory)'
    )

    parser.add_argument(
        '--top-n',
        type=int,
        default=16,
        help='Number of top sequences to highlight (default: 16)'
    )

    parser.add_argument(
        '--max-display',
        type=int,
        default=100,
        help='Maximum number of sequences to display in plot (default: 100)'
    )

    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_arguments()

    try:
        print("Starting Occurrence Counts Plotting...")
        print(f"Input CSV: {args.csv_file}")
        print(f"Output directory: {args.output_dir}")
        print(f"Highlighting top {args.top_n} sequences")

        # Generate plot
        plot_path = plot_occurrence_counts(
            csv_file=args.csv_file,
            output_dir=args.output_dir,
            top_n=args.top_n,
            max_display=args.max_display
        )

        print("=" * 50)
        print("PLOTTING COMPLETED SUCCESSFULLY")
        print("=" * 50)
        print(f"Plot saved to: {plot_path}")

        return 0

    except KeyboardInterrupt:
        print("Plotting interrupted by user")
        return 1

    except Exception as e:
        print(f"Plotting failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)