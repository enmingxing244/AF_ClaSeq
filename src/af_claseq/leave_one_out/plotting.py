"""
Impact Analysis Plotting Module

This module provides visualization functions for leave-one-out impact analysis,
following existing AF_ClaSeq plotting patterns and conventions.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Optional

from af_claseq.utils.logging_utils import get_logger
from af_claseq.utils.plotting_manager import save_ai_compatible_plot

# Set publication-quality font defaults (following existing patterns)
plt.rcParams.update({
    'font.family': ['sans-serif'],
    'font.sans-serif': ['DejaVu Sans'],
    'font.size': 24,
    'axes.labelsize': 24,
    'axes.titlesize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24
})


class ImpactPlotter:
    """
    Handles all impact analysis visualization following AF_ClaSeq patterns.
    """

    def __init__(self, config, logger=None):
        """
        Initialize the impact plotter.

        Args:
            config: WorkflowConfig object
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or get_logger("impact_plotter")
        self.plotting_config = config.plotting

    def create_impact_plots(self, results: List[Dict], output_dir: Path) -> List[str]:
        """
        Create all required impact visualization plots.

        Args:
            results: List of impact analysis results
            output_dir: Directory to save plots

        Returns:
            List of created plot file paths
        """
        if not results:
            self.logger.warning("No results provided for plotting")
            return []

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        plot_files = []

        try:
            # Convert results to DataFrame for easier plotting
            df = pd.DataFrame(results)

            # Plot 1: Impact score distribution
            hist_file = self._create_impact_histogram(df, output_dir)
            if hist_file:
                plot_files.append(hist_file)

            # Plot 2: Full vs LOO scatter plot with impact coloring
            scatter_file = self._create_scatter_plot(df, output_dir)
            if scatter_file:
                plot_files.append(scatter_file)

            # Plot 3: Combined impact analysis plot
            combined_file = self._create_combined_plot(df, output_dir)
            if combined_file:
                plot_files.append(combined_file)

            # Plot 4: Top impact sequences bar plot
            if len(df) > 0:
                bar_file = self._create_top_sequences_plot(df, output_dir)
                if bar_file:
                    plot_files.append(bar_file)

            self.logger.info(f"Created {len(plot_files)} impact visualization plots")

        except Exception as e:
            self.logger.error(f"Error creating impact plots: {e}")

        return plot_files

    def _create_impact_histogram(self, df: pd.DataFrame, output_dir: Path) -> Optional[str]:
        """Create histogram of impact scores"""
        try:
            fig, ax = plt.subplots(figsize=self.plotting_config.figsize)

            # Create histogram
            impacts = df['impact_score'].values
            ax.hist(impacts, bins=30, alpha=0.7, color='steelblue', edgecolor='black')

            # Add vertical lines for thresholds
            threshold = self.config.leave_one_out.impact_threshold
            ax.axvline(0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Neutral Impact')
            ax.axvline(threshold, color='orange', linestyle='--', alpha=0.7, linewidth=2,
                      label=f'Threshold ({threshold})')
            ax.axvline(-threshold, color='orange', linestyle='--', alpha=0.7, linewidth=2)

            # Formatting
            ax.set_xlabel('Impact Score')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Impact Scores')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add statistics text
            stats_text = f'Mean: {np.mean(impacts):.3f}\nMedian: {np.median(impacts):.3f}\nStd: {np.std(impacts):.3f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.tight_layout()

            # Save plot using AI-compatible format
            base_path = str(output_dir / "impact_score_distribution")
            save_ai_compatible_plot(plt.gcf(), base_path, dpi=self.plotting_config.dpi, logger=self.logger)
            plt.close()

            self.logger.info(f"Created impact histogram: {base_path}")
            return base_path

        except Exception as e:
            self.logger.error(f"Error creating impact histogram: {e}")
            return None

    def _create_scatter_plot(self, df: pd.DataFrame, output_dir: Path) -> Optional[str]:
        """Create scatter plot of full vs LOO means colored by impact score"""
        try:
            fig, ax = plt.subplots(figsize=self.plotting_config.figsize)

            # Create scatter plot
            scatter = ax.scatter(
                df['full_mean'], df['loo_mean'],
                c=df['impact_score'], cmap='RdBu_r',
                alpha=0.7, s=100, edgecolors='black', linewidth=0.5
            )

            # Add diagonal line (y = x)
            min_val = min(df['full_mean'].min(), df['loo_mean'].min())
            max_val = max(df['full_mean'].max(), df['loo_mean'].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5,
                   linewidth=2, label='y = x (No Impact)')

            # Formatting
            metric_name = self.config.leave_one_out.impact_metric_name
            ax.set_xlabel(f'Full Group Mean {metric_name}')
            ax.set_ylabel(f'Leave-One-Out Mean {metric_name}')
            ax.set_title('Full vs LOO Means (colored by Impact Score)')
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Set equal aspect ratio
            ax.set_aspect('equal', adjustable='box')

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.solids.set_rasterized(True)
            cbar.set_label('Impact Score', rotation=270, labelpad=20)

            plt.tight_layout()

            # Save plot using AI-compatible format
            base_path = str(output_dir / "full_vs_loo_scatter")
            save_ai_compatible_plot(plt.gcf(), base_path, dpi=self.plotting_config.dpi, logger=self.logger)
            plt.close()

            self.logger.info(f"Created scatter plot: {base_path}")
            return base_path

        except Exception as e:
            self.logger.error(f"Error creating scatter plot: {e}")
            return None

    def _create_combined_plot(self, df: pd.DataFrame, output_dir: Path) -> Optional[str]:
        """Create combined plot with histogram and scatter plot side by side"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # Left plot: Histogram of impact scores
            impacts = df['impact_score'].values
            ax1.hist(impacts, bins=30, alpha=0.7, color='steelblue', edgecolor='black')

            threshold = self.config.leave_one_out.impact_threshold
            ax1.axvline(0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Neutral')
            ax1.axvline(threshold, color='orange', linestyle='--', alpha=0.7, linewidth=2,
                       label=f'Threshold ({threshold})')
            ax1.axvline(-threshold, color='orange', linestyle='--', alpha=0.7, linewidth=2)

            ax1.set_xlabel('Impact Score')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Impact Score Distribution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Right plot: Scatter plot
            scatter = ax2.scatter(
                df['full_mean'], df['loo_mean'],
                c=df['impact_score'], cmap='RdBu_r',
                alpha=0.7, s=80, edgecolors='black', linewidth=0.5
            )

            min_val = min(df['full_mean'].min(), df['loo_mean'].min())
            max_val = max(df['full_mean'].max(), df['loo_mean'].max())
            ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5,
                    linewidth=2, label='y = x')

            metric_name = self.config.leave_one_out.impact_metric_name
            ax2.set_xlabel(f'Full Group Mean {metric_name}')
            ax2.set_ylabel(f'Leave-One-Out Mean {metric_name}')
            ax2.set_title('Full vs LOO Means')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            ax2.set_aspect('equal', adjustable='box')

            # Add colorbar for scatter plot
            cbar = plt.colorbar(scatter, ax=ax2)
            cbar.solids.set_rasterized(True)
            cbar.set_label('Impact Score', rotation=270, labelpad=20)

            plt.tight_layout()

            # Save plot using AI-compatible format
            base_path = str(output_dir / "combined_impact_analysis")
            save_ai_compatible_plot(plt.gcf(), base_path, dpi=self.plotting_config.dpi, logger=self.logger)
            plt.close()

            self.logger.info(f"Created combined plot: {base_path}")
            return base_path

        except Exception as e:
            self.logger.error(f"Error creating combined plot: {e}")
            return None

    def _create_top_sequences_plot(self, df: pd.DataFrame, output_dir: Path, top_n: int = 20) -> Optional[str]:
        """Create bar plot of top impact sequences"""
        try:
            # Sort by impact score and take top N
            cutoff_method = self.config.leave_one_out.cutoff_method
            ascending = (cutoff_method == 'below')
            top_df = df.nlargest(top_n, 'impact_score') if not ascending else df.nsmallest(top_n, 'impact_score')

            if len(top_df) == 0:
                return None

            fig, ax = plt.subplots(figsize=(12, 8))

            # Create bar plot
            y_pos = np.arange(len(top_df))
            colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(top_df)))

            bars = ax.barh(y_pos, top_df['impact_score'], color=colors, alpha=0.8, edgecolor='black')

            # Add threshold line
            threshold = self.config.leave_one_out.impact_threshold
            ax.axvline(threshold, color='red', linestyle='--', alpha=0.7, linewidth=2,
                      label=f'Threshold ({threshold})')

            # Format sequence headers for display (truncate long headers)
            sequence_labels = []
            for header in top_df['left_out_header']:
                # Remove '>' if present and truncate if too long
                clean_header = header.replace('>', '').strip()
                if len(clean_header) > 30:
                    clean_header = clean_header[:27] + '...'
                sequence_labels.append(clean_header)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(sequence_labels, fontsize=10)
            ax.set_xlabel('Impact Score')
            ax.set_title(f'Top {len(top_df)} Impact Sequences')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='x')

            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, top_df['impact_score'])):
                ax.text(value + 0.01 * (max(top_df['impact_score']) - min(top_df['impact_score'])),
                       bar.get_y() + bar.get_height()/2,
                       f'{value:.3f}', va='center', fontsize=8)

            plt.tight_layout()

            # Save plot using AI-compatible format
            base_path = str(output_dir / f"top_{top_n}_impact_sequences")
            save_ai_compatible_plot(plt.gcf(), base_path, dpi=self.plotting_config.dpi, logger=self.logger)
            plt.close()

            self.logger.info(f"Created top sequences plot: {base_path}")
            return base_path

        except Exception as e:
            self.logger.error(f"Error creating top sequences plot: {e}")
            return None

    def create_summary_report(self, results: List[Dict], filtered_results: List[Dict],
                            output_dir: Path) -> Optional[str]:
        """Create a text summary report of the impact analysis"""
        try:
            report_file = output_dir / "impact_analysis_summary.txt"

            with open(report_file, 'w') as f:
                f.write("Leave-One-Out Impact Analysis Summary\n")
                f.write("=" * 50 + "\n\n")

                # Basic statistics
                if results:
                    impacts = [r['impact_score'] for r in results]
                    f.write(f"Total sequences analyzed: {len(results)}\n")
                    f.write(f"Significant sequences identified: {len(filtered_results)}\n")
                    f.write(f"Significance rate: {len(filtered_results)/len(results)*100:.1f}%\n\n")

                    f.write("Impact Score Statistics:\n")
                    f.write(f"  Mean: {np.mean(impacts):.3f}\n")
                    f.write(f"  Median: {np.median(impacts):.3f}\n")
                    f.write(f"  Std Dev: {np.std(impacts):.3f}\n")
                    f.write(f"  Min: {np.min(impacts):.3f}\n")
                    f.write(f"  Max: {np.max(impacts):.3f}\n\n")

                # Filter criteria
                config = self.config.leave_one_out
                f.write("Filter Criteria:\n")
                f.write(f"  Impact metric: {config.impact_metric_name}\n")
                f.write(f"  Impact threshold: {config.impact_threshold} ({config.cutoff_method})\n")
                f.write(f"  Full group mean threshold: {config.full_group_mean_threshold} ({config.full_mean_cutoff_method})\n\n")

                # Top significant sequences
                if filtered_results:
                    f.write("Top 10 Significant Sequences:\n")
                    cutoff_method = config.cutoff_method
                    sorted_results = sorted(filtered_results, key=lambda x: x['impact_score'],
                                          reverse=(cutoff_method == 'above'))

                    for i, result in enumerate(sorted_results[:10], 1):
                        header = result['left_out_header'].replace('>', '').strip()
                        if len(header) > 50:
                            header = header[:47] + '...'
                        f.write(f"  {i:2d}. {header} (impact: {result['impact_score']:.3f})\n")

            self.logger.info(f"Created summary report: {report_file}")
            return str(report_file)

        except Exception as e:
            self.logger.error(f"Error creating summary report: {e}")
            return None