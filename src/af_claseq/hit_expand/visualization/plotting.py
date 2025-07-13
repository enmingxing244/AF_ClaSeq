#!/usr/bin/env python3
"""
Plotting utilities for MSA pipeline structure analysis visualization.
Creates scatter plots of RMSD values color-coded by pLDDT scores using af_claseq plotting utilities.
"""

import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict, Any

from af_claseq.utils.plotting_manager import (
    create_2d_scatter_plot,
    create_joint_plot,
    create_correlation_plots,
    load_results_df,
    PLOT_PARAMS,
    COLORS
)
   

logger = logging.getLogger(__name__)


class StructureAnalysisPlotter:
    """Creates visualizations for structure analysis results."""
    
    def __init__(self, use_af_claseq: bool = True):
        """
        Initialize structure analysis plotter.
        
        Args:
            use_af_claseq: Whether to use af_claseq plotting utilities
        """
        self.use_af_claseq = use_af_claseq
        
        if self.use_af_claseq:
            logger.info("Initialized with af_claseq plotting utilities")
        else:
            logger.info("Initialized with matplotlib/seaborn plotting")
    
    def create_rmsd_scatter_plot(self, 
                               results_df: pd.DataFrame,
                               output_file: Union[str, Path],
                               title: str = "Structure Analysis: Filter Criteria vs pLDDT",
                               figsize: Tuple[int, int] = (12, 8),
                               filter_criteria_threshold: float = 6.0,
                               plddt_threshold: float = 75.0) -> Path:
        """
        Create scatter plot of filter criteria values color-coded by pLDDT scores.
        Works with RMSD, TM-score, distance, angle, or any other criteria.
        
        Args:
            results_df: DataFrame with structure analysis results
            output_file: Path to save the plot
            title: Plot title
            figsize: Figure size (width, height)
            filter_criteria_threshold: Threshold for filter criteria (from config)
            plddt_threshold: pLDDT threshold (from config)
            
        Returns:
            Path to saved plot file
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Find filter criteria columns (excluding basic columns)
        basic_columns = {'PDB', 'plddt', 'local_plddt', 'seq_count'}
        criteria_columns = [col for col in results_df.columns if col not in basic_columns]
        
        if len(criteria_columns) < 1:
            raise ValueError(f"Need at least 1 filter criteria column, found {len(criteria_columns)}: {criteria_columns}")
        
        # Use first criteria column for scatter plot, or two if available
        if len(criteria_columns) >= 2:
            criteria_x = criteria_columns[0]
            criteria_y = criteria_columns[1]
        else:
            criteria_x = criteria_columns[0]
            criteria_y = 'plddt'  # Use pLDDT as second axis if only one criteria
        
        # Check for pLDDT column
        plddt_col = None
        for col in results_df.columns:
            if 'plddt' in col.lower():
                plddt_col = col
                break
        
        if plddt_col is None:
            raise ValueError("No pLDDT column found in results")
        
        logger.info(f"Creating scatter plot: {criteria_x} vs {criteria_y}, colored by {plddt_col}")
        
        # Filter out invalid values
        valid_mask = (
            results_df[criteria_x].notna() & 
            results_df[criteria_y].notna() & 
            results_df[plddt_col].notna()
        )
        
        plot_df = results_df[valid_mask].copy()
        
        if plot_df.empty:
            logger.warning("No valid data points for plotting")
            return output_file
        
        logger.info(f"Plotting {len(plot_df)} valid data points")
        
        if self.use_af_claseq:
            return self._create_plot_with_af_claseq(plot_df, criteria_x, criteria_y, plddt_col, 
                                                   output_file, title, figsize)
        else:
            return self._create_plot_with_matplotlib(plot_df, criteria_x, criteria_y, plddt_col, 
                                                   output_file, title, figsize,
                                                   filter_criteria_threshold, plddt_threshold)
    
    def _create_plot_with_af_claseq(self, 
                                  plot_df: pd.DataFrame,
                                  criteria_x: str, 
                                  criteria_y: str,
                                  plddt_col: str,
                                  output_file: Path,
                                  title: str,
                                  figsize: Tuple[int, int]) -> Path:
        """Create plot using af_claseq plotting utilities."""
        try:
            # Use the create_2d_scatter_plot function directly
            plot_path = create_2d_scatter_plot(
                results_df=plot_df,
                metric_name1=criteria_x,
                metric_name2=criteria_y,
                output_dir=output_file.parent,
                color_metric=plddt_col,
                logger=logger
            )
            
            logger.info(f"Saved af_claseq scatter plot: {plot_path}")
            return Path(plot_path)
            
        except Exception as e:
            logger.error(f"af_claseq plotting failed: {e}")
            # Fall back to matplotlib
            return self._create_plot_with_matplotlib(plot_df, criteria_x, criteria_y, plddt_col, 
                                                   output_file, title, figsize,
                                                   6.0, 75.0)  # Default fallback values
    
    def _create_plot_with_matplotlib(self, 
                                   plot_df: pd.DataFrame,
                                   criteria_x: str, 
                                   criteria_y: str,
                                   plddt_col: str,
                                   output_file: Path,
                                   title: str,
                                   figsize: Tuple[int, int],
                                   filter_criteria_threshold: float,
                                   plddt_threshold: float) -> Path:
        """Create plot using matplotlib/seaborn."""
        plt.figure(figsize=figsize)
        
        # Create scatter plot
        scatter = plt.scatter(
            plot_df[criteria_x], 
            plot_df[criteria_y], 
            c=plot_df[plddt_col], 
            cmap='viridis',
            alpha=0.7,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label(plddt_col, rotation=270, labelpad=15)
        
        # Determine units for axis labels
        x_unit = "Score" if "score" in criteria_x.lower() else ("Å" if "rmsd" in criteria_x.lower() else "")
        y_unit = "Score" if "score" in criteria_y.lower() else ("Å" if "rmsd" in criteria_y.lower() else "")
        
        # Labels and title
        plt.xlabel(f'{criteria_x} {("(" + x_unit + ")") if x_unit else ""}')
        plt.ylabel(f'{criteria_y} {("(" + y_unit + ")") if y_unit else ""}')
        plt.title(title)
        
        # Add quality thresholds (using values from config)
        self._add_quality_thresholds(plt.gca(), plot_df, criteria_x, criteria_y, plddt_col,
                                    filter_criteria_threshold, plddt_threshold)
        
        # Grid and styling
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved matplotlib scatter plot: {output_file}")
        return output_file
    
    def _add_quality_thresholds(self, ax, plot_df: pd.DataFrame, 
                              criteria_x: str, criteria_y: str, plddt_col: str,
                              filter_criteria_threshold: float = 6.0,
                              plddt_threshold: float = 75.0) -> None:
        """Add quality threshold lines to the plot."""
        # Determine threshold direction based on criteria type
        is_score_metric = "score" in criteria_x.lower() or "score" in criteria_y.lower()
        
        if is_score_metric:
            # For scores (TM-score, etc.), higher is better - threshold is minimum
            threshold_label = f'Min threshold: {filter_criteria_threshold}'
        else:
            # For RMSD/distance, lower is better - threshold is maximum  
            threshold_label = f'Max threshold: {filter_criteria_threshold}'
        
        # Add threshold lines
        ax.axhline(y=filter_criteria_threshold, color='red', linestyle='--', alpha=0.7, 
                  label=threshold_label)
        ax.axvline(x=filter_criteria_threshold, color='red', linestyle='--', alpha=0.7)
        
        # Add legend
        ax.legend()
    
    def create_quality_distribution_plots(self, 
                                        results_df: pd.DataFrame,
                                        output_dir: Union[str, Path],
                                        prefix: str = "quality_dist") -> List[Path]:
        """
        Create distribution plots for quality metrics.
        
        Args:
            results_df: DataFrame with structure analysis results
            output_dir: Directory to save plots
            prefix: Prefix for plot filenames
            
        Returns:
            List of paths to saved plot files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_plots = []
        
        # pLDDT distribution
        plddt_col = None
        for col in results_df.columns:
            if 'plddt' in col.lower():
                plddt_col = col
                break
        
        if plddt_col:
            plddt_plot = self._create_distribution_plot(
                results_df, plddt_col, 
                output_dir / f"{prefix}_plddt_distribution.png",
                "pLDDT Score Distribution"
            )
            saved_plots.append(plddt_plot)
        
        # RMSD distributions
        rmsd_columns = [col for col in results_df.columns if 'rmsd' in col.lower()]
        
        for rmsd_col in rmsd_columns:
            rmsd_plot = self._create_distribution_plot(
                results_df, rmsd_col,
                output_dir / f"{prefix}_{rmsd_col}_distribution.png",
                f"{rmsd_col} Distribution"
            )
            saved_plots.append(rmsd_plot)
        
        return saved_plots
    
    def _create_distribution_plot(self, 
                                results_df: pd.DataFrame,
                                column: str,
                                output_file: Path,
                                title: str) -> Path:
        """Create a distribution plot for a single metric."""
        plt.figure(figsize=(10, 6))
        
        # Filter valid values
        valid_data = results_df[column].dropna()
        
        if valid_data.empty:
            logger.warning(f"No valid data for {column}")
            return output_file
        
        # Create histogram with KDE
        plt.hist(valid_data, bins=50, alpha=0.7, density=True, color='skyblue')
        
        # Add KDE curve
        try:
            sns.kdeplot(valid_data, color='red', linewidth=2)
        except:
            pass  # Skip KDE if it fails
        
        # Add statistics
        mean_val = valid_data.mean()
        median_val = valid_data.median()
        
        plt.axvline(mean_val, color='green', linestyle='--', 
                   label=f'Mean: {mean_val:.2f}')
        plt.axvline(median_val, color='orange', linestyle='--', 
                   label=f'Median: {median_val:.2f}')
        
        plt.xlabel(column)
        plt.ylabel('Density')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved distribution plot: {output_file}")
        return output_file
    
    def create_quality_summary_plot(self, 
                                  results_df: pd.DataFrame,
                                  output_file: Union[str, Path],
                                  plddt_threshold: float = 75.0,
                                  filter_criteria_threshold: float = 6.0) -> Path:
        """
        Create a summary plot showing quality metrics and thresholds.
        
        Args:
            results_df: DataFrame with structure analysis results
            output_file: Path to save the plot
            plddt_threshold: pLDDT threshold for quality filtering
            filter_criteria_threshold: Filter criteria threshold for quality filtering
            
        Returns:
            Path to saved plot file
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Find columns
        plddt_col = None
        rmsd_columns = []
        
        for col in results_df.columns:
            if 'plddt' in col.lower():
                plddt_col = col
            elif 'rmsd' in col.lower():
                rmsd_columns.append(col)
        
        if not plddt_col or not rmsd_columns:
            logger.warning("Cannot create summary plot - missing required columns")
            return output_file
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Structure Analysis Quality Summary', fontsize=16)
        
        # Plot 1: pLDDT vs first RMSD
        ax1 = axes[0, 0]
        scatter = ax1.scatter(results_df[plddt_col], results_df[rmsd_columns[0]], 
                             alpha=0.6, s=30)
        ax1.axhline(y=filter_criteria_threshold, color='red', linestyle='--', alpha=0.7)
        ax1.axvline(x=plddt_threshold, color='red', linestyle='--', alpha=0.7)
        ax1.set_xlabel(plddt_col)
        ax1.set_ylabel(f'{rmsd_columns[0]} (Å)')
        ax1.set_title(f'{plddt_col} vs {rmsd_columns[0]}')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Quality categories
        ax2 = axes[0, 1]
        
        # Categorize structures
        high_quality = (
            (results_df[plddt_col] > plddt_threshold) & 
            (results_df[rmsd_columns[0]] < filter_criteria_threshold)
        ).sum()
        
        total_structures = len(results_df)
        low_quality = total_structures - high_quality
        
        categories = ['High Quality', 'Low Quality']
        counts = [high_quality, low_quality]
        colors = ['green', 'red']
        
        ax2.bar(categories, counts, color=colors, alpha=0.7)
        ax2.set_ylabel('Number of Structures')
        ax2.set_title('Quality Categories')
        
        # Add count labels
        for i, count in enumerate(counts):
            ax2.text(i, count + 0.5, str(count), ha='center', va='bottom')
        
        # Plot 3: pLDDT distribution
        ax3 = axes[1, 0]
        ax3.hist(results_df[plddt_col].dropna(), bins=30, alpha=0.7, color='skyblue')
        ax3.axvline(plddt_threshold, color='red', linestyle='--', alpha=0.7,
                   label=f'Threshold: {plddt_threshold}')
        ax3.set_xlabel(plddt_col)
        ax3.set_ylabel('Count')
        ax3.set_title(f'{plddt_col} Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: RMSD distribution
        ax4 = axes[1, 1]
        ax4.hist(results_df[rmsd_columns[0]].dropna(), bins=30, alpha=0.7, color='lightcoral')
        ax4.axvline(filter_criteria_threshold, color='red', linestyle='--', alpha=0.7,
                   label=f'Threshold: {filter_criteria_threshold} Å')
        ax4.set_xlabel(f'{rmsd_columns[0]} (Å)')
        ax4.set_ylabel('Count')
        ax4.set_title(f'{rmsd_columns[0]} Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved quality summary plot: {output_file}")
        return output_file


def create_structure_analysis_plots(results_file: Union[str, Path],
                                  output_dir: Union[str, Path],
                                  use_af_claseq: bool = True,
                                  filter_criteria_threshold: float = 6.0,
                                  plddt_threshold: float = 75.0) -> Dict[str, Path]:
    """
    Convenience function to create all structure analysis plots.
    
    Args:
        results_file: Path to structure analysis results CSV
        output_dir: Directory to save plots
        use_af_claseq: Whether to use af_claseq plotting utilities
        filter_criteria_threshold: Threshold for filter criteria (from config)
        plddt_threshold: pLDDT threshold (from config)
        
    Returns:
        Dictionary mapping plot types to saved file paths
    """
    results_file = Path(results_file)
    output_dir = Path(output_dir)
    
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    # Load results
    results_df = pd.read_csv(results_file)
    
    # Initialize plotter
    plotter = StructureAnalysisPlotter(use_af_claseq=use_af_claseq)
    
    # Create plots
    saved_plots = {}
    
    # Main scatter plot
    scatter_plot = plotter.create_rmsd_scatter_plot(
        results_df, 
        output_dir / "filter_criteria_scatter_plot.png",
        filter_criteria_threshold=filter_criteria_threshold,
        plddt_threshold=plddt_threshold
    )
    saved_plots['scatter'] = scatter_plot
    
    # Distribution plots
    dist_plots = plotter.create_quality_distribution_plots(
        results_df, 
        output_dir
    )
    saved_plots['distributions'] = dist_plots
    
    # Summary plot
    summary_plot = plotter.create_quality_summary_plot(
        results_df,
        output_dir / "quality_summary.png",
        plddt_threshold=plddt_threshold,
        filter_criteria_threshold=filter_criteria_threshold
    )
    saved_plots['summary'] = summary_plot
    
    logger.info(f"Created {len(saved_plots)} plot types in {output_dir}")
    return saved_plots