"""
Structure analysis for the divide-and-conquer workflow.
Wraps existing AF_ClaSeq structure analysis functionality.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import logging

# Add AF_ClaSeq to path for imports
sys.path.append('/fs/ess/PAA0203/xing244/AF_ClaSeq/src')

from af_claseq.utils.structure_analysis import StructureAnalyzer as BaseStructureAnalyzer
from af_claseq.utils.structure_analysis import load_filter_modes, apply_filters

from .utils import (
    validate_file_exists, create_directory, find_files_with_pattern, WorkflowError
)


class StructureAnalyzer:
    """
    Wraps AF_ClaSeq structure analysis functionality for the workflow.
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize StructureAnalyzer.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.analysis_config = config.get('structure_analysis', {})
        
        # Load structure analysis configuration
        self.structure_config_file = self.analysis_config.get('config_file')
        if self.structure_config_file:
            validate_file_exists(self.structure_config_file, "Structure analysis config file")
            
        self.plddt_threshold = self.analysis_config.get('plddt_threshold', 70)
        
        # Initialize the base analyzer
        self.analyzer = BaseStructureAnalyzer()
        
        self.logger.info(f"Structure analysis configuration:")
        self.logger.info(f"  Config file: {self.structure_config_file}")
        self.logger.info(f"  pLDDT threshold: {self.plddt_threshold}")
    
    def load_structure_config(self) -> Dict[str, Any]:
        """
        Load structure analysis configuration from JSON file.
        
        Returns:
            Structure analysis configuration dictionary
        """
        if not self.structure_config_file:
            raise WorkflowError("Structure analysis config file not specified")
        
        try:
            with open(self.structure_config_file, 'r') as f:
                structure_config = json.load(f)
            
            self.logger.info(f"Loaded structure analysis config from {self.structure_config_file}")
            
            # Validate required sections
            if 'filter_criteria' not in structure_config:
                raise WorkflowError("filter_criteria not found in structure config")
            
            return structure_config
            
        except Exception as e:
            raise WorkflowError(f"Failed to load structure config: {e}")
    
    def find_pdb_files(self, shuffle_dirs: List[str]) -> List[str]:
        """
        Find all PDB files in shuffle directories.
        
        Args:
            shuffle_dirs: List of shuffle directory paths
            
        Returns:
            List of PDB file paths
        """
        all_pdb_files = []
        
        for shuffle_dir in shuffle_dirs:
            if not os.path.exists(shuffle_dir):
                self.logger.warning(f"Shuffle directory not found: {shuffle_dir}")
                continue
            
            # Find PDB files in the shuffle directory
            pdb_files = find_files_with_pattern(shuffle_dir, "*.pdb")
            
            # Filter out non-structure PDB files if needed
            filtered_pdb_files = []
            for pdb_file in pdb_files:
                pdb_path = str(pdb_file)
                # Skip if it's in a subdirectory we want to avoid
                if 'non_a3m' not in pdb_path:
                    filtered_pdb_files.append(pdb_path)
            
            all_pdb_files.extend(filtered_pdb_files)
            
            self.logger.debug(f"Found {len(filtered_pdb_files)} PDB files in {shuffle_dir}")
        
        self.logger.info(f"Total PDB files found: {len(all_pdb_files)}")
        return all_pdb_files
    
    def analyze_structures(self, shuffle_dirs: List[str]) -> pd.DataFrame:
        """
        Analyze structures using the existing AF_ClaSeq functionality.
        
        Args:
            shuffle_dirs: List of shuffle directory paths
            
        Returns:
            DataFrame with analysis results
        """
        self.logger.info("=" * 50)
        self.logger.info("STRUCTURE ANALYSIS STARTED")
        self.logger.info("=" * 50)
        
        # Load structure analysis configuration
        structure_config = self.load_structure_config()
        
        # Find all PDB files
        pdb_files = self.find_pdb_files(shuffle_dirs)
        
        if not pdb_files:
            raise WorkflowError("No PDB files found for analysis")
        
        # Extract configuration components
        filter_criteria = structure_config.get('filter_criteria', [])
        basics = structure_config.get('basics', {})
        composite_metrics = structure_config.get('composite_metrics', [])
        
        self.logger.info(f"Analysis configuration:")
        self.logger.info(f"  Filter criteria: {len(filter_criteria)} metrics")
        self.logger.info(f"  Composite metrics: {len(composite_metrics)} metrics")
        self.logger.info(f"  pLDDT threshold: {self.plddt_threshold}")
        
        # Log metric names being calculated
        metric_names = [criterion.get('name', 'unknown') for criterion in filter_criteria]
        composite_names = [composite.get('name', 'unknown') for composite in composite_metrics]
        
        self.logger.info(f"Metrics to calculate:")
        for name in metric_names:
            self.logger.info(f"  - {name}")
        if composite_names:
            self.logger.info(f"Composite metrics:")
            for name in composite_names:
                self.logger.info(f"  - {name}")
        
        try:
            # Use the parallel processing method from the base analyzer
            analysis_results = self.analyzer.process_pdbs_parallel(
                pdb_files=pdb_files,
                filter_criteria=filter_criteria,
                basics=basics,
                plddt_threshold=self.plddt_threshold,
                n_jobs=-1,  # Use all available cores
                composite_metrics=composite_metrics
            )
            
            # Convert results to DataFrame
            results_list = [result for result in analysis_results.values() if result is not None]
            
            if not results_list:
                raise WorkflowError("No structures passed analysis criteria")
            
            results_df = pd.DataFrame(results_list)
            
            self.logger.info("=" * 50)
            self.logger.info("STRUCTURE ANALYSIS COMPLETED")
            self.logger.info(f"Analyzed structures: {len(results_df)}")
            self.logger.info(f"Structures passing pLDDT threshold: {len(results_df)}")
            self.logger.info("=" * 50)
            
            # Log analysis summary
            self._log_analysis_summary(results_df, metric_names + composite_names)
            
            return results_df
            
        except Exception as e:
            self.logger.error("=" * 50)
            self.logger.error("STRUCTURE ANALYSIS FAILED")
            self.logger.error(f"Error: {e}")
            self.logger.error("=" * 50)
            raise WorkflowError(f"Structure analysis failed: {e}")
    
    def _log_analysis_summary(self, results_df: pd.DataFrame, metric_names: List[str]) -> None:
        """
        Log summary statistics for the analysis results.
        
        Args:
            results_df: Analysis results DataFrame
            metric_names: List of metric names
        """
        self.logger.info("Analysis Summary:")
        self.logger.info("-" * 30)
        
        # General statistics
        if 'plddt' in results_df.columns:
            plddt_mean = results_df['plddt'].mean()
            plddt_median = results_df['plddt'].median()
            self.logger.info(f"pLDDT - Mean: {plddt_mean:.2f}, Median: {plddt_median:.2f}")
        
        if 'local_plddt' in results_df.columns:
            local_plddt_mean = results_df['local_plddt'].mean()
            local_plddt_median = results_df['local_plddt'].median()
            self.logger.info(f"Local pLDDT - Mean: {local_plddt_mean:.2f}, Median: {local_plddt_median:.2f}")
        
        # Metric statistics
        for metric in metric_names:
            if metric in results_df.columns:
                values = results_df[metric].dropna()
                if len(values) > 0:
                    mean_val = values.mean()
                    median_val = values.median()
                    min_val = values.min()
                    max_val = values.max()
                    self.logger.info(
                        f"{metric} - Mean: {mean_val:.3f}, Median: {median_val:.3f}, "
                        f"Range: [{min_val:.3f}, {max_val:.3f}]"
                    )
    
    def apply_quality_filters(self, results_df: pd.DataFrame, 
                            quantile: float = 0.1) -> pd.DataFrame:
        """
        Apply quality filters to results based on quantile thresholds.
        
        Args:
            results_df: Analysis results DataFrame
            quantile: Quantile value for filtering (default: 0.1 for top 10%)
            
        Returns:
            Filtered DataFrame
        """
        if results_df.empty:
            return results_df
        
        self.logger.info(f"Applying quality filters with quantile threshold: {quantile}")
        
        # Load structure configuration to get filter criteria
        structure_config = self.load_structure_config()
        filter_criteria = structure_config.get('filter_criteria', [])
        
        # Apply filters using the existing functionality
        try:
            filtered_df = apply_filters(
                df_threshold=results_df,  # Use same df for threshold calculation
                df_operate=results_df,
                filter_criteria=filter_criteria,
                quantile=quantile
            )
            
            self.logger.info(f"Filtering results:")
            self.logger.info(f"  Before filtering: {len(results_df)} structures")
            self.logger.info(f"  After filtering: {len(filtered_df)} structures")
            self.logger.info(f"  Retention rate: {len(filtered_df)/len(results_df)*100:.1f}%")
            
            return filtered_df
            
        except Exception as e:
            self.logger.warning(f"Failed to apply quality filters: {e}")
            self.logger.warning("Returning unfiltered results")
            return results_df
    
    def save_results(self, results_df: pd.DataFrame, output_file: str) -> None:
        """
        Save analysis results to CSV file.
        
        Args:
            results_df: Analysis results DataFrame
            output_file: Output file path
        """
        try:
            # Create output directory
            output_dir = os.path.dirname(output_file)
            if output_dir:
                create_directory(output_dir)
            
            # Save to CSV
            results_df.to_csv(output_file, index=False)
            
            self.logger.info(f"Analysis results saved to: {output_file}")
            self.logger.info(f"Saved {len(results_df)} structure analysis records")
            
        except Exception as e:
            raise WorkflowError(f"Failed to save results: {e}")
    
    def analyze_complete(self, shuffle_dirs: List[str], 
                        output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Complete structure analysis workflow.
        
        Args:
            shuffle_dirs: List of shuffle directory paths
            output_file: Optional output CSV file path
            
        Returns:
            Analysis results DataFrame
        """
        try:
            # Analyze structures
            results_df = self.analyze_structures(shuffle_dirs)
            
            # Save results if output file specified
            if output_file:
                self.save_results(results_df, output_file)
            
            return results_df
            
        except Exception as e:
            self.logger.error(f"Complete structure analysis failed: {e}")
            raise