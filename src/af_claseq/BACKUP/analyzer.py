"""
Hit expand structure analysis module.

This module provides the HitExpandAnalyzer class that analyzes structure
prediction results for the hit expansion process.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from af_claseq.utils.structure_analysis import StructureAnalyzer
from af_claseq.utils.logging_utils import get_logger

logger = get_logger(__name__)


class HitExpandAnalyzer:
    """Analyzes structure prediction results for hit expansion."""
    
    def __init__(self, config_file: str, logger: Optional[logging.Logger] = None):
        """
        Initialize the HitExpandAnalyzer.
        
        Args:
            config_file: Path to AF-ClaSeq configuration JSON file
            logger: Optional logger instance
        """
        self.config_file = config_file
        self.logger = logger or get_logger(__name__)
        self.structure_analyzer = StructureAnalyzer()
        
        # Load configuration
        self._load_config()
        
    def _load_config(self) -> None:
        """Load configuration from JSON file."""
        import json
        
        try:
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
                self.filter_criteria = self.config.get('filter_criteria', [])
                self.references = self.config.get('references', [])
                
                # Extract metric information
                self.metric_configs = {}
                for criterion in self.filter_criteria:
                    metric_name = criterion.get('name', '')
                    self.metric_configs[metric_name] = criterion
                    
            self.logger.info(f"Loaded configuration with {len(self.filter_criteria)} filter criteria")
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            self.config = {}
            self.filter_criteria = []
            self.references = []
            self.metric_configs = {}
    
    def analyze_structures(self, structures_dir: Path, plddt_threshold: float = 70.0) -> pd.DataFrame:
        """
        Analyze predicted structures against reference structures.
        
        Args:
            structures_dir: Directory containing predicted PDB files
            plddt_threshold: Minimum pLDDT score threshold
            
        Returns:
            DataFrame with analysis results
        """
        self.logger.info(f"Analyzing structures in {structures_dir}")
        
        # Find all PDB files
        pdb_files = list(structures_dir.glob("**/*.pdb"))
        
        if not pdb_files:
            self.logger.warning("No PDB files found for analysis")
            return pd.DataFrame()
        
        self.logger.info(f"Found {len(pdb_files)} PDB files to analyze")
        
        # Analyze each structure
        results = []
        for pdb_file in pdb_files:
            try:
                result = self._analyze_single_structure(pdb_file)
                if result:
                    results.append(result)
            except Exception as e:
                self.logger.warning(f"Failed to analyze {pdb_file}: {str(e)}")
                continue
        
        if not results:
            self.logger.warning("No structures successfully analyzed")
            return pd.DataFrame()
        
        # Create DataFrame and filter by pLDDT
        df = pd.DataFrame(results)
        df_filtered = df[df['plddt'] >= plddt_threshold]
        
        self.logger.info(f"Analyzed {len(df)} structures, {len(df_filtered)} passed pLDDT threshold")
        
        return df_filtered
    
    def _analyze_single_structure(self, pdb_file: Path) -> Optional[Dict[str, Any]]:
        """
        Analyze a single structure file.
        
        Args:
            pdb_file: Path to PDB file
            
        Returns:
            Dictionary with analysis results
        """
        result = {
            'pdb_file': str(pdb_file),
            'a3m_file': str(pdb_file.with_suffix('.a3m')),
            'structure_name': pdb_file.stem
        }
        
        # Extract pLDDT score
        plddt_score = self._extract_plddt_score(pdb_file)
        if plddt_score is None:
            return None
        result['plddt'] = plddt_score
        
        # Calculate metrics for each filter criterion
        for criterion in self.filter_criteria:
            metric_name = criterion.get('name', '')
            metric_type = criterion.get('metric', '')
            
            if metric_type == 'TM-score':
                value = self._calculate_tm_score(pdb_file, criterion)
            elif metric_type == 'RMSD':
                value = self._calculate_rmsd(pdb_file, criterion)
            elif metric_type == 'distance':
                value = self._calculate_distance(pdb_file, criterion)
            elif metric_type == 'angle':
                value = self._calculate_angle(pdb_file, criterion)
            else:
                value = None
            
            if value is not None:
                result[metric_name] = value
        
        return result
    
    def _extract_plddt_score(self, pdb_file: Path) -> Optional[float]:
        """Extract average pLDDT score from PDB file."""
        try:
            structure = self.structure_analyzer.pdb_parser.get_structure("protein", pdb_file)
            
            plddt_scores = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        for atom in residue:
                            if atom.name == 'CA':  # Only consider CA atoms
                                plddt_scores.append(atom.bfactor)
            
            return sum(plddt_scores) / len(plddt_scores) if plddt_scores else None
            
        except Exception as e:
            self.logger.error(f"Error extracting pLDDT score from {pdb_file}: {str(e)}")
            return None
    
    def _calculate_tm_score(self, pdb_file: Path, criterion: Dict[str, Any]) -> Optional[float]:
        """Calculate TM-score against reference structure."""
        try:
            ref_index = criterion.get('ref_index', 0)
            if ref_index < len(self.references):
                ref_pdb = self.references[ref_index].get('PDB', '')
                if ref_pdb and Path(ref_pdb).exists():
                    # Use structure analyzer to calculate TM-score
                    tm_score = self.structure_analyzer.calculate_tm_score(
                        str(pdb_file), 
                        ref_pdb
                    )
                    return tm_score
            return None
        except Exception as e:
            self.logger.debug(f"Error calculating TM-score: {str(e)}")
            return None
    
    def _calculate_rmsd(self, pdb_file: Path, criterion: Dict[str, Any]) -> Optional[float]:
        """Calculate RMSD against reference structure."""
        try:
            ref_index = criterion.get('ref_index', 0)
            rmsd_type = criterion.get('rmsd_type', 'ca')
            
            if ref_index < len(self.references):
                ref_pdb = self.references[ref_index].get('PDB', '')
                if ref_pdb and Path(ref_pdb).exists():
                    # Use structure analyzer to calculate RMSD
                    if rmsd_type == 'ca':
                        rmsd = self.structure_analyzer.calculate_ca_rmsd(
                            str(pdb_file), 
                            ref_pdb
                        )
                    else:
                        rmsd = self.structure_analyzer.calculate_all_atom_rmsd(
                            str(pdb_file), 
                            ref_pdb
                        )
                    return rmsd
            return None
        except Exception as e:
            self.logger.debug(f"Error calculating RMSD: {str(e)}")
            return None
    
    def _calculate_distance(self, pdb_file: Path, criterion: Dict[str, Any]) -> Optional[float]:
        """Calculate distance between specified residues."""
        try:
            residue_indices = criterion.get('residue_indices', [])
            if len(residue_indices) == 2:
                distance = self.structure_analyzer.calculate_distance(
                    str(pdb_file),
                    residue_indices[0],
                    residue_indices[1]
                )
                return distance
            return None
        except Exception as e:
            self.logger.debug(f"Error calculating distance: {str(e)}")
            return None
    
    def _calculate_angle(self, pdb_file: Path, criterion: Dict[str, Any]) -> Optional[float]:
        """Calculate angle between specified residues."""
        try:
            residue_indices = criterion.get('residue_indices', [])
            if len(residue_indices) == 3:
                angle = self.structure_analyzer.calculate_angle(
                    str(pdb_file),
                    residue_indices[0],
                    residue_indices[1],
                    residue_indices[2]
                )
                return angle
            return None
        except Exception as e:
            self.logger.debug(f"Error calculating angle: {str(e)}")
            return None
    
    def filter_by_criteria(self, df: pd.DataFrame, filter_threshold: float = 0.5) -> pd.DataFrame:
        """
        Apply filtering criteria to analysis results.
        
        Args:
            df: DataFrame with analysis results
            filter_threshold: Threshold value for filter criteria
            
        Returns:
            Filtered DataFrame
        """
        if df.empty:
            return df
        
        # Apply filters based on metric configurations
        df_filtered = df.copy()
        
        for metric_name, config in self.metric_configs.items():
            if metric_name in df.columns:
                operator = config.get('operator', '>')
                threshold = config.get('threshold', filter_threshold)
                
                if operator == '>':
                    df_filtered = df_filtered[df_filtered[metric_name] > threshold]
                elif operator == '>=':
                    df_filtered = df_filtered[df_filtered[metric_name] >= threshold]
                elif operator == '<':
                    df_filtered = df_filtered[df_filtered[metric_name] < threshold]
                elif operator == '<=':
                    df_filtered = df_filtered[df_filtered[metric_name] <= threshold]
                elif operator == '==':
                    df_filtered = df_filtered[df_filtered[metric_name] == threshold]
        
        self.logger.info(f"Filtered {len(df)} structures to {len(df_filtered)} based on criteria")
        
        return df_filtered
    
    def get_hit_structures(self, df: pd.DataFrame) -> List[Path]:
        """
        Get list of hit structure A3M files from analysis results.
        
        Args:
            df: DataFrame with analysis results
            
        Returns:
            List of A3M file paths
        """
        hit_files = []
        
        for _, row in df.iterrows():
            a3m_file = Path(row['a3m_file'])
            if a3m_file.exists():
                hit_files.append(a3m_file)
        
        return hit_files
    
    def save_results(self, df: pd.DataFrame, output_file: Path) -> None:
        """
        Save analysis results to CSV file.
        
        Args:
            df: DataFrame with analysis results
            output_file: Path to output CSV file
        """
        df.to_csv(output_file, index=False)
        self.logger.info(f"Analysis results saved to {output_file}")