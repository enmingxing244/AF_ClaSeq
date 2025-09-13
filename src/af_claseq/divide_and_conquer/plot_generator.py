"""
Plot generation for the divide-and-conquer workflow.
Wraps existing AF_ClaSeq plotting functionality.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import logging

# Add AF_ClaSeq to path for imports
sys.path.append('/fs/ess/PAA0203/xing244/AF_ClaSeq/src')

from af_claseq.utils.plotting_manager import (
    plot_1d_distribution, create_2d_scatter_plot, create_joint_plot,
    plot_m_fold_sampling_1d, plot_m_fold_sampling_2d
)

from .utils import create_directory, WorkflowError


class PlotGenerator:
    """
    Wraps AF_ClaSeq plotting functionality for the workflow.
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize PlotGenerator.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.plotting_config = config.get('plotting', {})
        
        # Extract plotting parameters
        self.metrics = self.plotting_config.get('metrics', [])
        self.plot_types = self.plotting_config.get('plot_types', ['1d', '2d'])
        self.output_dir = self.plotting_config.get('output_dir', 'plots')
        
        # Plot customization options
        self.colors = self.plotting_config.get('colors', {})
        self.metric_ranges = self.plotting_config.get('metric_ranges', {})
        self.plot_params = self.plotting_config.get('plot_params', {})
        
        # Create output directory
        create_directory(self.output_dir)
        
        self.logger.info(f"Plot generation configuration:")
        self.logger.info(f"  Output directory: {self.output_dir}")
        self.logger.info(f"  Metrics to plot: {self.metrics}")
        self.logger.info(f"  Plot types: {self.plot_types}")
    
    def generate_1d_plots(self, results_df: pd.DataFrame) -> List[str]:
        """
        Generate 1D distribution plots for all specified metrics.
        
        Args:
            results_df: Analysis results DataFrame
            
        Returns:
            List of generated plot file paths
        """
        if results_df.empty:
            self.logger.warning("Empty results DataFrame, skipping 1D plots")
            return []
        
        self.logger.info("Generating 1D distribution plots...")
        plot_files = []
        
        for metric in self.metrics:
            if metric not in results_df.columns:
                self.logger.warning(f"Metric '{metric}' not found in results, skipping")
                continue
            
            self.logger.info(f"  Generating 1D plot for {metric}")
            
            try:
                # Get metric-specific parameters
                metric_params = self._get_metric_plot_params(metric)
                
                plot_path = plot_1d_distribution(
                    results_df=results_df,
                    metric_name=metric,
                    output_dir=self.output_dir,
                    **metric_params,
                    logger=self.logger
                )
                
                if plot_path:
                    plot_files.append(plot_path)
                    
            except Exception as e:
                self.logger.error(f"Failed to generate 1D plot for {metric}: {e}")
                continue
        
        self.logger.info(f"Generated {len(plot_files)} 1D distribution plots")
        return plot_files
    
    def generate_2d_plots(self, results_df: pd.DataFrame) -> List[str]:
        """
        Generate 2D scatter plots for metric pairs.
        
        Args:
            results_df: Analysis results DataFrame
            
        Returns:
            List of generated plot file paths
        """
        if results_df.empty:
            self.logger.warning("Empty results DataFrame, skipping 2D plots")
            return []
        
        self.logger.info("Generating 2D scatter plots...")
        plot_files = []
        
        # Generate plots for all possible metric pairs
        available_metrics = [m for m in self.metrics if m in results_df.columns]
        
        if len(available_metrics) < 2:
            self.logger.warning("Need at least 2 metrics for 2D plots")
            return []
        
        # Generate pairwise plots
        for i, metric1 in enumerate(available_metrics):
            for metric2 in available_metrics[i+1:]:
                self.logger.info(f"  Generating 2D plot for {metric1} vs {metric2}")
                
                try:
                    # Get plot parameters for this metric pair
                    plot_params = self._get_2d_plot_params(metric1, metric2)
                    
                    # Generate scatter plot
                    scatter_path = create_2d_scatter_plot(
                        results_df=results_df,
                        metric_name1=metric1,
                        metric_name2=metric2,
                        output_dir=self.output_dir,
                        **plot_params,
                        logger=self.logger
                    )
                    
                    if scatter_path:
                        plot_files.append(scatter_path)
                    
                    # Generate joint plot if requested
                    if 'joint' in self.plot_types:
                        joint_path = create_joint_plot(
                            results_df=results_df,
                            metric_name1=metric1,
                            metric_name2=metric2,
                            output_dir=self.output_dir,
                            **plot_params,
                            logger=self.logger
                        )
                        
                        if joint_path:
                            plot_files.append(joint_path)
                            
                except Exception as e:
                    self.logger.error(f"Failed to generate 2D plot for {metric1} vs {metric2}: {e}")
                    continue
        
        self.logger.info(f"Generated {len(plot_files)} 2D plots")
        return plot_files
    
    def generate_correlation_plots(self, results_df: pd.DataFrame) -> List[str]:
        """
        Generate correlation plots between pLDDT and other metrics.
        
        Args:
            results_df: Analysis results DataFrame
            
        Returns:
            List of generated plot file paths
        """
        if results_df.empty:
            self.logger.warning("Empty results DataFrame, skipping correlation plots")
            return []
        
        plot_files = []
        
        # Generate plots correlating metrics with pLDDT scores
        plddt_metrics = ['plddt', 'local_plddt']
        
        for plddt_metric in plddt_metrics:
            if plddt_metric not in results_df.columns:
                continue
            
            self.logger.info(f"Generating correlation plots with {plddt_metric}")
            
            for metric in self.metrics:
                if metric == plddt_metric or metric not in results_df.columns:
                    continue
                
                try:
                    plot_params = self._get_2d_plot_params(metric, plddt_metric)
                    
                    correlation_path = create_2d_scatter_plot(
                        results_df=results_df,
                        metric_name1=metric,
                        metric_name2=plddt_metric,
                        output_dir=self.output_dir,
                        color_metric=plddt_metric,
                        **plot_params,
                        logger=self.logger
                    )
                    
                    if correlation_path:
                        plot_files.append(correlation_path)
                        
                except Exception as e:
                    self.logger.error(f"Failed to generate correlation plot for {metric} vs {plddt_metric}: {e}")
                    continue
        
        self.logger.info(f"Generated {len(plot_files)} correlation plots")
        return plot_files
    
    def _get_metric_plot_params(self, metric: str) -> Dict[str, Any]:
        """
        Get plotting parameters for a specific metric.
        
        Args:
            metric: Metric name
            
        Returns:
            Dictionary of plotting parameters
        """
        params = {}
        
        # Default parameters
        params.update({
            'n_plot_bins': 50,
            'log_scale': False,
            'figsize': (10, 5),
            'initial_color': '#87CEEB',
            'end_color': '#FFFFFF',
            'show_bin_lines': False
        })
        
        # Override with global plot parameters
        params.update(self.plot_params.get('1d', {}))
        
        # Override with metric-specific parameters
        if metric in self.plot_params:
            params.update(self.plot_params[metric])
        
        # Add metric range if specified (for 1D plots, this becomes x-axis range)
        if metric in self.metric_ranges:
            range_config = self.metric_ranges[metric]
            params.update({
                'x_min': range_config.get('min'),
                'x_max': range_config.get('max'),
                'x_ticks': range_config.get('ticks')
            })
        
        # Add colors if specified
        if metric in self.colors:
            colors = self.colors[metric]
            if isinstance(colors, list) and len(colors) >= 2:
                params.update({
                    'initial_color': colors[0],
                    'end_color': colors[1]
                })
        
        return params
    
    def _get_2d_plot_params(self, metric1: str, metric2: str) -> Dict[str, Any]:
        """
        Get plotting parameters for 2D plots.
        
        Args:
            metric1: First metric name
            metric2: Second metric name
            
        Returns:
            Dictionary of plotting parameters
        """
        params = {}
        
        # Default parameters
        params.update({
            'color_metric': 'plddt'
        })
        
        # Override with global 2D parameters
        params.update(self.plot_params.get('2d', {}))
        
        # Add metric ranges (dynamically assign to x or y axis based on position)
        for i, metric in enumerate([metric1, metric2], 1):
            if metric in self.metric_ranges:
                range_config = self.metric_ranges[metric]
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
    
    def generate_all_plots(self, results_df: pd.DataFrame) -> List[str]:
        """
        Generate all requested plot types.
        
        Args:
            results_df: Analysis results DataFrame
            
        Returns:
            List of all generated plot file paths
        """
        self.logger.info("=" * 50)
        self.logger.info("PLOT GENERATION STARTED")
        self.logger.info("=" * 50)
        
        all_plot_files = []
        
        try:
            # Generate 1D plots
            if '1d' in self.plot_types:
                plot_files_1d = self.generate_1d_plots(results_df)
                all_plot_files.extend(plot_files_1d)
            
            # Generate 2D plots
            if '2d' in self.plot_types:
                plot_files_2d = self.generate_2d_plots(results_df)
                all_plot_files.extend(plot_files_2d)
            
            # Generate correlation plots
            if 'correlation' in self.plot_types:
                plot_files_corr = self.generate_correlation_plots(results_df)
                all_plot_files.extend(plot_files_corr)
            
            self.logger.info("=" * 50)
            self.logger.info("PLOT GENERATION COMPLETED")
            self.logger.info(f"Total plots generated: {len(all_plot_files)}")
            
            # List generated plots
            if all_plot_files:
                self.logger.info("Generated plot files:")
                for i, plot_file in enumerate(all_plot_files, 1):
                    self.logger.info(f"  {i}. {os.path.basename(plot_file)}")
            
            self.logger.info("=" * 50)
            
            return all_plot_files
            
        except Exception as e:
            self.logger.error("=" * 50)
            self.logger.error("PLOT GENERATION FAILED")
            self.logger.error(f"Error: {e}")
            self.logger.error("=" * 50)
            raise WorkflowError(f"Plot generation failed: {e}")
    
    def save_plot_summary(self, plot_files: List[str], summary_file: str = "plot_summary.txt") -> None:
        """
        Save a summary of generated plots.
        
        Args:
            plot_files: List of plot file paths
            summary_file: Summary file name
        """
        summary_path = os.path.join(self.output_dir, summary_file)
        
        try:
            with open(summary_path, 'w') as f:
                f.write("Plot Generation Summary\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total plots generated: {len(plot_files)}\n")
                f.write(f"Output directory: {self.output_dir}\n\n")
                
                # Group plots by type
                plot_types = {}
                for plot_file in plot_files:
                    filename = os.path.basename(plot_file)
                    if '1d_distribution' in filename:
                        plot_type = '1D Distribution'
                    elif 'scatter' in filename:
                        plot_type = '2D Scatter'
                    elif 'joint' in filename:
                        plot_type = 'Joint Plot'
                    else:
                        plot_type = 'Other'
                    
                    if plot_type not in plot_types:
                        plot_types[plot_type] = []
                    plot_types[plot_type].append(filename)
                
                # Write summary by type
                for plot_type, files in plot_types.items():
                    f.write(f"{plot_type} ({len(files)}):\n")
                    for filename in sorted(files):
                        f.write(f"  - {filename}\n")
                    f.write("\n")
            
            self.logger.info(f"Plot summary saved to: {summary_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save plot summary: {e}")