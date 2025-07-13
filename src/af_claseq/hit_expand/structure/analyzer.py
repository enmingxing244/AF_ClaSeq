#!/usr/bin/env python3
"""
High-quality structure analysis for protein structure prediction results.
Uses af_claseq.utils.structure_analysis for RMSD calculations with config_6xr6_6xrg_rmsd.json.
"""

import json
import logging
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
import time

import numpy as np
import pandas as pd
from af_claseq.utils import structure_analysis
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings

from af_claseq.hit_expand.config.settings import StructureAnalysisConfig
from af_claseq.hit_expand.config.utils import ConfigurationError

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class StructureAnalysisError(Exception):
    """Raised when structure analysis fails."""
    pass


class MSAStructureAnalyzer:
    """High-level structure analyzer using af_claseq.utils.structure_analysis."""
    
    def __init__(self, config: StructureAnalysisConfig):
        """
        Initialize structure analyzer.
        
        Args:
            config: Analysis configuration
        """
        self.config = config
        self.af_analyzer = structure_analysis.StructureAnalyzer()
        
        # Load filter configuration only if config_file is provided
        if config.config_file:
            self.filter_config = self._load_filter_config()
        else:
            self.filter_config = None
        
        # Set number of cores
        if config.n_cores is None:
            self.n_cores = min(64, mp.cpu_count())
        else:
            self.n_cores = config.n_cores
        
        logger.info(f"Structure analyzer initialized with {self.n_cores} cores")
        if config.config_file:
            logger.info(f"Using config file: {config.config_file}")
        else:
            logger.info("No config file provided - using default parameters")
    
    def _load_filter_config(self) -> Dict[str, Any]:
        """Load filter configuration from JSON file."""
        if not self.config.config_file:
            return {}
            
        config_path = Path(self.config.config_file)
        
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        logger.info(f"Loaded filter configuration with {len(config_data.get('filter_criteria', []))} criteria")
        return config_data
    
    def find_predicted_structures(self, batch_dir: Union[str, Path]) -> List[Path]:
        """
        Find all predicted PDB structures in batch directories.
        
        Args:
            batch_dir: Directory containing batch subdirectories
            
        Returns:
            List of PDB file paths
        """
        batch_dir = Path(batch_dir)
        
        # Pattern for ColabFold output files
        pattern = "*/*unrelaxed_rank_00*.pdb"
        pdb_files = list(batch_dir.glob(pattern))
        
        logger.info(f"Found {len(pdb_files)} predicted structures in {batch_dir}")
        return sorted(pdb_files)
    
    def analyze_structures_parallel(self, pdb_files: List[Path]) -> pd.DataFrame:
        """
        Analyze structures using af_claseq structure analysis.
        
        Args:
            pdb_files: List of PDB files to analyze
            
        Returns:
            DataFrame with analysis results
        """
        if not pdb_files:
            logger.warning("No PDB files to analyze")
            return pd.DataFrame()
        
        logger.info(f"Analyzing {len(pdb_files)} structures using af_claseq")
        
        start_time = time.time()
        
        # Process PDB files in parallel using af_claseq approach
        logger.info(f"Processing {len(pdb_files)} structures using {self.n_cores} cores")
        
        if not self.filter_config:
            logger.error("No filter configuration available. Cannot analyze structures.")
            return pd.DataFrame()
        
        results = Parallel(n_jobs=self.n_cores)(
            delayed(self.af_analyzer.process_single_pdb)(
                str(pdb_file),
                self.filter_config['filter_criteria'],
                self.filter_config['basics'],
                self.config.plddt_threshold
            ) for pdb_file in tqdm(pdb_files, desc="Analyzing structures")
        )
        
        # Filter out None results
        results = [r for r in results if r is not None]
        
        # Convert results to DataFrame
        df = pd.DataFrame(results) if results else pd.DataFrame()
        
        end_time = time.time()
        logger.info(f"Analysis completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Successfully analyzed {len(df)}/{len(pdb_files)} structures")
        
        return df
    
    
    def get_best_structures(self, df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """
        Get best structures using af_claseq filtering logic with user-specified criteria and thresholds.
        Reads the af_claseq config to find the specified filter criteria by name,
        then applies the user-provided threshold using the method from the config.
        
        Args:
            df: DataFrame with analysis results
            top_n: Number of top structures to return
            
        Returns:
            DataFrame with best structures (sorted by pLDDT descending)
        """
        if df.empty:
            return df
        
        # Apply pLDDT threshold first (user-specified)
        filtered_df = df.copy()
        plddt_filter = filtered_df['plddt'] > self.config.plddt_threshold
        filtered_df = filtered_df[plddt_filter]
        
        logger.info(f"pLDDT filtering: {len(df)} → {len(filtered_df)} structures (pLDDT > {self.config.plddt_threshold})")
        
        # Apply user-specified filter criteria from af_claseq config
        if self.filter_config and 'filter_criteria' in self.filter_config:
            # Find the user-specified criterion by name
            target_criterion = None
            for criterion in self.filter_config['filter_criteria']:
                if criterion['name'] == self.config.filter_criteria:
                    target_criterion = criterion
                    break
            
            if target_criterion:
                # Use the method from af_claseq config but threshold from user
                criterion_name = target_criterion['name']
                method = target_criterion['method']  # Read method from af_claseq config
                user_threshold = self.config.filter_criteria_threshold  # Use user-provided threshold
                
                if criterion_name in filtered_df.columns:
                    if method == 'below':
                        # Filter structures with value below threshold (e.g., RMSD < threshold)
                        criterion_filter = filtered_df[criterion_name] < user_threshold
                        operator_str = "<"
                    elif method == 'above':
                        # Filter structures with value above threshold (e.g., distance > threshold)
                        criterion_filter = filtered_df[criterion_name] > user_threshold
                        operator_str = ">"
                    else:
                        logger.warning(f"Unknown filter method: {method}. Defaulting to 'below'")
                        criterion_filter = filtered_df[criterion_name] < user_threshold
                        operator_str = "<"
                    
                    filtered_df = filtered_df[criterion_filter]
                    logger.info(f"Applied {criterion_name} filter: {criterion_name} {operator_str} {user_threshold}")
                    logger.info(f"  Method '{method}' from af_claseq config: {target_criterion.get('ref_pdb', 'N/A')}")
                else:
                    logger.warning(f"Criterion column '{criterion_name}' not found in data")
            else:
                logger.warning(f"Target criterion '{self.config.filter_criteria}' not found in af_claseq filter_criteria")
                logger.info(f"Available criteria: {[c['name'] for c in self.filter_config['filter_criteria']]}")
        else:
            logger.warning("No af_claseq filter_criteria available - skipping criterion filtering")
        
        logger.info(f"Final filtering: {len(df)} → {len(filtered_df)} structures")
        logger.info(f"Applied filters: pLDDT > {self.config.plddt_threshold} AND {self.config.filter_criteria} threshold {self.config.filter_criteria_threshold}")
        
        if filtered_df.empty:
            logger.warning("No structures passed filtering criteria")
            return filtered_df
        
        # Sort by pLDDT (higher is better) and return top N
        sorted_df = filtered_df.sort_values('plddt', ascending=False)
        return sorted_df.head(top_n)
    
    def apply_af_claseq_filters(self, df: pd.DataFrame, quantile: float = 0.1) -> pd.DataFrame:
        """
        Apply af_claseq-style quantile-based filtering using the filter_criteria.
        
        Args:
            df: DataFrame with analysis results
            quantile: Quantile threshold for filtering (0.1 = top 10%)
            
        Returns:
            DataFrame with filtered structures
        """
        if df.empty or not self.filter_config:
            return df
            
        # Use af_claseq apply_filters function
        from af_claseq.utils.structure_analysis import apply_filters
        
        filter_criteria = self.filter_config.get('filter_criteria', [])
        if not filter_criteria:
            logger.warning("No filter_criteria available for quantile filtering")
            return df
        
        logger.info(f"Applying af_claseq quantile filtering with {quantile*100}% threshold")
        
        try:
            # Apply quantile-based filtering using af_claseq logic
            filtered_df = apply_filters(df, df, filter_criteria, quantile)
            
            logger.info(f"af_claseq filtering: {len(df)} → {len(filtered_df)} structures")
            return filtered_df
            
        except Exception as e:
            logger.error(f"af_claseq filtering failed: {e}")
            return df


class HitIdentifier:
    """Identifies hit structures and maps them back to A3M files."""
    
    @staticmethod
    def pdb_to_a3m_path(pdb_path: Union[str, Path]) -> Path:
        """
        Convert PDB file path to corresponding A3M file path.
        
        Args:
            pdb_path: Path to PDB file
            
        Returns:
            Path to corresponding A3M file
        """
        pdb_path = Path(pdb_path)
        
        # Remove the '_unrelaxed...' suffix and change extension
        stem = pdb_path.stem
        if '_unrelaxed' in stem:
            a3m_stem = stem.split('_unrelaxed')[0]
        else:
            a3m_stem = stem
        
        a3m_path = pdb_path.parent / f"{a3m_stem}.a3m"
        return a3m_path
    
    @staticmethod
    def identify_hit_a3m_files(best_structures_df: pd.DataFrame) -> List[Path]:
        """
        Identify A3M files corresponding to best structures.
        
        Args:
            best_structures_df: DataFrame with best structures
            
        Returns:
            List of A3M file paths
        """
        hit_a3m_files = []
        
        for _, row in best_structures_df.iterrows():
            pdb_path = Path(row['PDB'])  # af_claseq uses 'PDB' column name
            a3m_path = HitIdentifier.pdb_to_a3m_path(pdb_path)
            
            if a3m_path.exists():
                hit_a3m_files.append(a3m_path)
            else:
                logger.warning(f"A3M file not found for hit structure: {a3m_path}")
        
        logger.info(f"Identified {len(hit_a3m_files)} hit A3M files")
        return hit_a3m_files


def load_analysis_config(config_file: Union[str, Path], 
                        base_config: Optional[StructureAnalysisConfig] = None) -> StructureAnalysisConfig:
    """
    Load analysis configuration for af_claseq structure analysis.
    
    Args:
        config_file: Path to configuration file (config_6xr6_6xrg_rmsd.json)
        base_config: Base configuration to merge with (optional)
        
    Returns:
        StructureAnalysisConfig object
    """
    config_file = Path(config_file)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    # Use base config values if provided, otherwise use defaults
    if base_config is not None:
        return StructureAnalysisConfig(
            config_file=str(config_file),
            plddt_threshold=base_config.plddt_threshold,
            filter_criteria_threshold=base_config.filter_criteria_threshold,
            filter_criteria=base_config.filter_criteria,
            n_cores=base_config.n_cores
        )
    else:
        return StructureAnalysisConfig(
            config_file=str(config_file),
            plddt_threshold=75.0,  # Default threshold
            filter_criteria_threshold=6.0,    # Default threshold
            filter_criteria="6xrg_rmsd"  # Default criteria
        )


def get_available_criteria_names(config_file: Union[str, Path]) -> List[str]:
    """
    Get list of available filter criteria names from af_claseq config file.
    
    Args:
        config_file: Path to af_claseq configuration file
        
    Returns:
        List of available criteria names
    """
    config_file = Path(config_file)
    
    if not config_file.exists():
        raise FileNotFoundError(f"af_claseq config file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config_data = json.load(f)
    
    filter_criteria = config_data.get('filter_criteria', [])
    criteria_names = [criterion.get('name', 'unknown') for criterion in filter_criteria]
    
    logger.info(f"Available filter criteria in {config_file}: {criteria_names}")
    return criteria_names