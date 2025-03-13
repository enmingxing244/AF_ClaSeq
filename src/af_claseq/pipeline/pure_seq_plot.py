"""
Pure Sequence Plotting module for AF-ClaSeq pipeline.

This module provides functionality for visualizing structure prediction results
from pure sequence predictions, comparing metrics between prediction and control sets.
"""

import os
import glob
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from matplotlib.colors import LinearSegmentedColormap
from concurrent.futures import ProcessPoolExecutor, as_completed
import functools

from af_claseq.utils.structure_analysis import StructureAnalyzer, get_result_df
from af_claseq.utils.plotting_manager import (
    PLOT_PARAMS, COLORS, _load_or_calculate_metrics,
    _load_data_for_2d_plot, _set_axis_limits_and_ticks
)

@dataclass
class PureSeqPlotConfig:
    """Configuration for pure sequence plotting."""
    base_dir: str
    output_dir: str
    config_file: str
    color_prediction: str = COLORS.get('default_prediction', '#468F8B')
    color_control: str = COLORS.get('default_control', '#AFD2D0')
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    x_ticks: Optional[List[float]] = None
    y_ticks: Optional[List[float]] = None
    plddt_threshold: float = 70.0
    figsize: Tuple[int, int] = (15, 7)
    dpi: int = 300
    max_workers: int = 8


class PureSequencePlotter:
    """
    Visualizes structure prediction results from pure sequence predictions.
    
    This class handles the generation of plots comparing metrics between
    prediction and control prediction sets, with options for customizing
    plot appearance and filtering results.
    """
    
    def __init__(self, 
                 config: PureSeqPlotConfig,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the pure sequence plotter.
        
        Args:
            config: Configuration options for plotting
            logger: Optional logger, will create one if not provided
        """
        self.config = config
        self.logger = logger or self._setup_logger()
        self.structure_analyzer = StructureAnalyzer()
        
        # Apply plot style configuration
        plt.rcParams.update({k: v for k, v in PLOT_PARAMS.items() 
                            if k not in ['scatter_size', 'scatter_alpha', 'scatter_edge']})
        
        # Load configuration file
        self.filter_config = self._load_filter_config()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up a default logger if none was provided."""
        logger = logging.getLogger("PureSequencePlotter")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _load_filter_config(self) -> Dict:
        """Load the filter configuration from JSON file."""
        try:
            config_path = Path(self.config.config_file)
            if not config_path.exists():
                self.logger.error(f"Filter config file does not exist: {config_path}")
                return {}
                
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Validate config has required sections
            if 'filter_criteria' not in config:
                self.logger.error("Filter config is missing required 'filter_criteria' section")
                return {}
                
            return config
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON in filter config file: {config_path}")
            return {}
        except Exception as e:
            self.logger.error(f"Error loading filter config: {str(e)}")
            return {}
    
    def process_prediction_dir(self, pred_dir: str) -> Dict[str, pd.DataFrame]:
        """
        Process prediction directory containing PDB files with seed information.
        
        Args:
            pred_dir: Directory containing prediction PDB files
            
        Returns:
            Dict mapping bin patterns to DataFrames with processed metrics
        """
        data_by_bin = {}
        bin_dirs = glob.glob(os.path.join(pred_dir, 'bin_*'))
        
        if not bin_dirs:
            self.logger.warning(f"No bin directories found in {pred_dir}")
            return {}
            
        # Use only the first two criteria if available
        criteria = self.filter_config.get('filter_criteria', [])[:2]
        if not criteria:
            self.logger.error("No filter criteria found in config")
            return {}
        
        # Process each bin directory
        for bin_dir in bin_dirs:
            try:
                bin_pattern = os.path.basename(bin_dir)
                
                # Use get_result_df from structure_analysis module
                results_df = get_result_df(
                    parent_dir=bin_dir,
                    filter_criteria=criteria,
                    basics=self.filter_config.get('basics', {})
                )
                
                if results_df.empty:
                    self.logger.warning(f"No results found in {bin_dir}")
                    continue
                
                # Extract seed numbers from PDB filenames
                results_df['seed'] = results_df['PDB'].apply(
                    lambda x: int(os.path.basename(x).split('seed_')[1].split('.')[0])
                    if 'seed_' in os.path.basename(x) else 0
                )
                
                # Add bin pattern for reference
                results_df['bin_pattern'] = bin_pattern
                
                # Rename columns to match expected format
                column_mapping = {
                    criteria[0]['name']: 'metric1'
                }
                
                if len(criteria) > 1:
                    column_mapping[criteria[1]['name']] = 'metric2'
                
                # Create a copy with mapped columns
                bin_data = results_df.rename(columns=column_mapping)
                
                data_by_bin[bin_pattern] = bin_data
                        
            except Exception as e:
                self.logger.error(f"Error processing bin directory {bin_dir}: {str(e)}")
                
        return data_by_bin
    
    def plot_prediction_results(self, 
                               pred_data: pd.DataFrame, 
                               control_pred_data: pd.DataFrame, 
                               output_path: str, 
                               bin_pattern: str) -> None:
        """
        Create scatter plots comparing prediction results with control for a specific bin pattern.
        
        Args:
            pred_data: DataFrame with prediction metrics for this bin pattern
            control_pred_data: DataFrame with control prediction metrics for this bin pattern
            output_path: Path to save plot
            bin_pattern: Bin pattern being plotted (e.g. bin_5 or bin_5_6_7)
        """
        if pred_data.empty and control_pred_data.empty:
            self.logger.warning(f"No data available for plotting prediction results for {bin_pattern}")
            return
            
        plt.figure(figsize=self.config.figsize)
        prediction_palette = LinearSegmentedColormap.from_list(
            'custom', 
            [self.config.color_prediction, self.config.color_control], 
            N=16
        )
        
        # Get metric name from config
        metric_name = "Unknown"
        if self.filter_config.get('filter_criteria') and len(self.filter_config['filter_criteria']) > 0:
            metric_name = self.filter_config['filter_criteria'][0]['name']
        
        # Plot predictions
        plt.subplot(1, 2, 1)
        self._plot_predictions(
            pred_data, 
            prediction_palette, 
            metric_name, 
            f"Prediction Results - {bin_pattern}"
        )
        
        # Plot control predictions  
        plt.subplot(1, 2, 2)
        self._plot_predictions(
            control_pred_data, 
            prediction_palette, 
            metric_name, 
            f"Control Prediction - {bin_pattern}"
        )
        
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=self.config.dpi)
        plt.close()
    
    def _plot_predictions(self, 
                         data: pd.DataFrame, 
                         palette: LinearSegmentedColormap, 
                         metric_name: str, 
                         title: str) -> None:
        """Helper function for prediction plotting."""
        if data.empty:
            plt.text(0.5, 0.5, 'No data available', 
                    horizontalalignment='center',
                    verticalalignment='center')
            return
            
        unique_seeds = sorted(list(set(data['seed'])))
        
        for i, seed in enumerate(unique_seeds):
            mask = (data['seed'] == seed)
            plt.scatter(
                data[mask]['metric1'], 
                data[mask]['plddt'],
                color=palette(i/len(unique_seeds)),
                label=f'Seed {seed:03d}',
                s=PLOT_PARAMS['scatter_size'],
                alpha=PLOT_PARAMS['scatter_alpha'],
                edgecolor=PLOT_PARAMS['scatter_edge']
            )
        
        plt.xlabel(metric_name)
        plt.ylabel('pLDDT Score')
        plt.title(title, pad=20)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
        
        # Set axis limits and ticks using utility function
        self._set_plot_limits_and_ticks(plt)
    
    def _set_plot_limits_and_ticks(self, plot_obj):
        """Set axis limits and ticks for a plot."""
        # Set axis limits if provided
        if self.config.x_min is not None and self.config.x_max is not None:
            plot_obj.xlim(self.config.x_min, self.config.x_max)
        if self.config.y_min is not None and self.config.y_max is not None:
            plot_obj.ylim(self.config.y_min, self.config.y_max)
            
        # Set axis ticks if provided
        if self.config.x_ticks is not None:
            plot_obj.xticks(self.config.x_ticks)
        if self.config.y_ticks is not None:
            plot_obj.yticks(self.config.y_ticks)
    
    def plot_metric_correlation(self, 
                               data: pd.DataFrame, 
                               output_path: str, 
                               title: str, 
                               bin_pattern: str) -> None:
        """
        Create scatter plot showing correlation between two metrics colored by pLDDT.
        
        Args:
            data: DataFrame with metrics to correlate for this bin pattern
            output_path: Path to save plot
            title: Plot title
            bin_pattern: Bin pattern being plotted
        """
        if data.empty or 'metric2' not in data.columns or data['metric2'].isna().all():
            self.logger.warning(f"No correlation data available for {bin_pattern}")
            return
            
        plt.figure(figsize=(7, 6))
        
        # Use correlation colormap from COLORS dictionary
        custom_cmap = LinearSegmentedColormap.from_list('custom', COLORS['correlation_cmap'])
        
        # Sort data by pLDDT in ascending order so higher pLDDT points appear on top
        data = data.sort_values('plddt', ascending=True)
        
        # Get metric names from config
        metric1_name = self.filter_config['filter_criteria'][0]['name']
        metric2_name = self.filter_config['filter_criteria'][1]['name']
        
        scatter = plt.scatter(
            data['metric1'],
            data['metric2'],
            c=data['plddt'],
            cmap=custom_cmap,
            s=PLOT_PARAMS['scatter_size'],
            alpha=PLOT_PARAMS['scatter_alpha'],
            vmin=50,
            vmax=100
        )
        
        plt.colorbar(scatter, label='pLDDT Score')
        plt.xlabel(metric1_name)
        plt.ylabel(metric2_name) 
        plt.title(f"{title} - {bin_pattern}", pad=20)
        
        # Set axis limits and ticks
        self._set_plot_limits_and_ticks(plt)
        
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=self.config.dpi)
        plt.close()
    
    def run(self) -> bool:
        """
        Run the plotting pipeline for all bin patterns.
        
        Returns:
            Boolean indicating success
        """
        try:
            self.logger.info("Starting pure sequence plotting")
            
            # Create output directory
            os.makedirs(self.config.output_dir, exist_ok=True)
            
            # Process data by bin pattern
            pred_data_by_bin = self.process_prediction_dir(
                os.path.join(self.config.base_dir, 'prediction')
            )
            
            control_pred_data_by_bin = self.process_prediction_dir(
                os.path.join(self.config.base_dir, 'control_prediction')
            )
            
            # Get all unique bin patterns
            all_bin_patterns = set(list(pred_data_by_bin.keys()) + 
                                  list(control_pred_data_by_bin.keys()))
            
            if not all_bin_patterns:
                self.logger.error("No bin patterns found in prediction directories")
                return False
            
            # Check if we have two criteria for correlation plots
            has_second_criterion = len(self.filter_config.get('filter_criteria', [])) > 1
            
            # Generate plots for each bin pattern
            for bin_pattern in sorted(all_bin_patterns):
                bin_pred_data = pred_data_by_bin.get(bin_pattern, pd.DataFrame())
                bin_control_pred_data = control_pred_data_by_bin.get(bin_pattern, pd.DataFrame())
                
                # Create bin-specific output directory
                bin_output_dir = os.path.join(self.config.output_dir, bin_pattern)
                os.makedirs(bin_output_dir, exist_ok=True)
                
                # Generate prediction plots
                self.plot_prediction_results(
                    bin_pred_data,
                    bin_control_pred_data,
                    os.path.join(bin_output_dir, 'prediction_results.png'),
                    bin_pattern
                )
                
                # Generate correlation plots if second metric exists
                if has_second_criterion:
                    for data, name in [
                        (bin_pred_data, 'prediction'),
                        (bin_control_pred_data, 'control_prediction')
                    ]:
                        # Generate correlation plot
                        self.plot_metric_correlation(
                            data,
                            os.path.join(bin_output_dir, f'{name}_metric_correlation.png'),
                            f'{name.replace("_", " ").title()} Metric Correlation',
                            bin_pattern
                        )
                        
                        # Save metrics to CSV
                        if not data.empty:
                            metric1_name = self.filter_config['filter_criteria'][0]['name']
                            metric2_name = self.filter_config['filter_criteria'][1]['name']
                            
                            metrics_df = pd.DataFrame({
                                metric1_name: data['metric1'],
                                metric2_name: data['metric2'],
                                'pLDDT': data['plddt'],
                                'pdb_path': data['PDB']
                            })
                            csv_path = os.path.join(bin_output_dir, f'{name}_metrics.csv')
                            metrics_df.to_csv(csv_path, index=False)
            
            self.logger.info("Pure sequence plotting completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in pure sequence plotting: {str(e)}", exc_info=True)
            return False


def create_pure_seq_plot_config_from_dict(config_dict: Dict) -> PureSeqPlotConfig:
    """
    Create a PureSeqPlotConfig from a dictionary.
    
    Args:
        config_dict: Dictionary with configuration parameters
        
    Returns:
        PureSeqPlotConfig object
    """
    return PureSeqPlotConfig(
        base_dir=config_dict['base_dir'],
        output_dir=config_dict['output_dir'],
        config_file=config_dict['config_file'],
        color_prediction=config_dict.get('color_prediction', COLORS.get('default_prediction', '#468F8B')),
        color_control=config_dict.get('color_control', COLORS.get('default_control', '#AFD2D0')),
        x_min=config_dict.get('x_min'),
        x_max=config_dict.get('x_max'),
        y_min=config_dict.get('y_min'),
        y_max=config_dict.get('y_max'),
        x_ticks=config_dict.get('x_ticks'),
        y_ticks=config_dict.get('y_ticks'),
        plddt_threshold=config_dict.get('plddt_threshold', 70.0),
        figsize=config_dict.get('figsize', (15, 7)),
        dpi=config_dict.get('dpi', 300),
        max_workers=config_dict.get('max_workers', 8)
    )