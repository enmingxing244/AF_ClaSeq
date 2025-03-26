"""
Pure Sequence Plotting module for AF-ClaSeq pipeline.

This module provides functionality for visualizing structure prediction results
from pure sequence predictions, comparing metrics between prediction and control sets.
"""

import os
import glob
import json
import logging
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import centralized plotting utilities
from af_claseq.utils.plotting_manager import (
    load_results_df, set_axis_limits_and_ticks,
    create_2d_scatter_plot, create_joint_plot, PLOT_PARAMS, COLORS
)
from af_claseq.utils.structure_analysis import StructureAnalyzer
from af_claseq.utils.logging_utils import get_logger
from af_claseq.pipeline.config import PureSequencePlottingConfig


@dataclass
class PureSeqPlotConfig:
    """Configuration for pure sequence plotting."""
    base_dir: str
    output_dir: str
    config_file: str
    color_prediction: str = COLORS['default_prediction']
    color_control: Optional[str] = COLORS['default_control']
    metric1_min: Optional[float] = None
    metric1_max: Optional[float] = None
    metric2_min: Optional[float] = None
    metric2_max: Optional[float] = None
    metric1_ticks: Optional[List[float]] = None
    metric2_ticks: Optional[List[float]] = None
    plddt_threshold: float = 70.0
    figsize: Tuple[int, int] = (15, 7)
    dpi: int = PLOT_PARAMS['dpi']
    max_workers: int = 8


def create_pure_seq_plot_config_from_dict(config_dict: Dict[str, Any]) -> PureSeqPlotConfig:
    """Create a PureSeqPlotConfig from a dictionary."""
    return PureSeqPlotConfig(**config_dict)


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
        """Initialize the pure sequence plotter."""
        self.config = config
        self.logger = logger or get_logger("pure_seq_plot")
        
        # Load configuration file
        self.filter_config = self._load_filter_config()
    
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
                
                # Load and process data
                results_df = load_results_df(
                    results_dir=bin_dir,
                    metric_names=[criterion['name'] for criterion in criteria],
                    config_file=self.config.config_file,
                    plddt_threshold=self.config.plddt_threshold,
                    logger=self.logger
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
                
                # Store the original DataFrame without renaming columns
                data_by_bin[bin_pattern] = results_df
                        
            except Exception as e:
                self.logger.error(f"Error processing bin directory {bin_dir}: {str(e)}")
                
        return data_by_bin
    
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
                
               
                # Generate correlation plots if second metric exists
                if has_second_criterion:
                    for data, name in [
                        (bin_pred_data, 'prediction'),
                        (bin_control_pred_data, 'control_prediction')
                    ]:
                        # Get metric names from config
                        metric1_name = self.filter_config['filter_criteria'][0]['name']
                        metric2_name = self.filter_config['filter_criteria'][1]['name']
                        
                        if data.empty or metric2_name not in data.columns or data[metric2_name].isna().all():
                            self.logger.warning(f"No correlation data available for {bin_pattern} - {name}")
                            continue
                            
                        # Use central utility function for 2D scatter plot
                        output_dir = os.path.join(bin_output_dir, name)
                        title = f"{name.replace('_', ' ').title()} Metric Correlation - {bin_pattern}"
                        
                        # Create scatter plot
                        create_2d_scatter_plot(
                            results_df=data,
                            metric_name1=metric1_name,
                            metric_name2=metric2_name,
                            output_dir=output_dir,
                            color_metric='plddt',
                            x_min=self.config.metric1_min,
                            x_max=self.config.metric1_max,
                            y_min=self.config.metric2_min,
                            y_max=self.config.metric2_max,
                            x_ticks=self.config.metric1_ticks,
                            y_ticks=self.config.metric2_ticks,
                            title=title,
                            logger=self.logger
                        )
                        
                        # Save metrics to CSV
                        metrics_df = pd.DataFrame({
                            metric1_name: data[metric1_name],
                            metric2_name: data[metric2_name],
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
