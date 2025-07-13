#!/usr/bin/env python3
"""
Configuration utilities for MSA pipeline.
Provides standardized configuration loading, validation, and management functions.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Union, Optional, Tuple, List

from af_claseq.hit_expand.config.settings import PipelineConfig, StructureAnalysisConfig

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration is invalid or cannot be loaded."""
    pass


class ConfigurationUtils:
    """Utilities for consistent configuration management across the pipeline."""
    
    @staticmethod
    def load_pipeline_config(config_file: Union[str, Path]) -> PipelineConfig:
        """
        Load and validate pipeline configuration.
        
        Args:
            config_file: Path to pipeline configuration file
            
        Returns:
            Validated PipelineConfig object
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        try:
            # Use ConfigManager directly to avoid circular imports
            from .settings import ConfigManager
            manager = ConfigManager()
            config = manager.load_config(config_file)
            ConfigurationUtils.validate_pipeline_config(config)
            logger.info(f"Loaded pipeline configuration from {config_file}")
            return config
        except Exception as e:
            raise ConfigurationError(f"Failed to load pipeline configuration: {e}")
    
    @staticmethod
    def load_structure_config(pipeline_config: PipelineConfig) -> StructureAnalysisConfig:
        """
        Load structure analysis configuration with af_claseq integration.
        
        Args:
            pipeline_config: Main pipeline configuration
            
        Returns:
            StructureAnalysisConfig object ready for use
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        try:
            if pipeline_config.structure_analysis.config_file:
                # Load with af_claseq config file
                config_file = Path(pipeline_config.structure_analysis.config_file)
                if not config_file.exists():
                    raise FileNotFoundError(f"af_claseq config file not found: {config_file}")
                
                # Import here to avoid circular imports
                from ..structure.analyzer import load_analysis_config
                analysis_config = load_analysis_config(config_file, pipeline_config.structure_analysis)
                logger.info(f"Loaded structure analysis config with af_claseq integration: {config_file}")
            else:
                # Use pipeline configuration directly
                analysis_config = pipeline_config.structure_analysis
                logger.info("Using structure analysis config from pipeline (no af_claseq config)")
            
            ConfigurationUtils.validate_structure_analysis_config(analysis_config)
            return analysis_config
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load structure analysis configuration: {e}")
    
    @staticmethod
    def get_available_filter_criteria(config_file: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Get available filter criteria from af_claseq configuration file.
        
        Args:
            config_file: Path to af_claseq configuration file
            
        Returns:
            List of available filter criteria with their names, types, and methods
            
        Raises:
            ConfigurationError: If configuration file cannot be read
        """
        try:
            config_file = Path(config_file)
            if not config_file.exists():
                raise FileNotFoundError(f"af_claseq config file not found: {config_file}")
            
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            filter_criteria = config_data.get('filter_criteria', [])
            
            # Extract key information for each criterion
            criteria_info = []
            for criterion in filter_criteria:
                info = {
                    'name': criterion.get('name', 'unknown'),
                    'type': criterion.get('type', 'unknown'),
                    'method': criterion.get('method', 'unknown'),
                    'ref_pdb': criterion.get('ref_pdb', 'N/A')
                }
                criteria_info.append(info)
            
            logger.info(f"Found {len(criteria_info)} filter criteria in {config_file}")
            return criteria_info
            
        except Exception as e:
            raise ConfigurationError(f"Failed to read filter criteria from config: {e}")
    
    @staticmethod
    def validate_pipeline_config(config: PipelineConfig) -> None:
        """
        Validate pipeline configuration.
        
        Args:
            config: Pipeline configuration to validate
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        try:
            # Configuration validation is done in __post_init__, but we can add extra checks
            
            # Check af_claseq config file if specified
            if config.structure_analysis.config_file:
                config_file = Path(config.structure_analysis.config_file)
                if not config_file.exists():
                    raise FileNotFoundError(f"af_claseq config file not found: {config_file}")
                
                # Validate af_claseq config file format
                ConfigurationUtils._validate_af_claseq_config_file(config_file)
            
            logger.debug("Pipeline configuration validation passed")
            
        except Exception as e:
            raise ConfigurationError(f"Pipeline configuration validation failed: {e}")
    
    @staticmethod
    def validate_structure_analysis_config(config: StructureAnalysisConfig) -> None:
        """
        Validate structure analysis configuration.
        
        Args:
            config: Structure analysis configuration to validate
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        try:
            # Check thresholds
            if not 0.0 <= config.plddt_threshold <= 100.0:
                raise ValueError("pLDDT threshold must be between 0.0 and 100.0")
            
            if config.filter_criteria_threshold <= 0.0:
                raise ValueError("Filter criteria threshold must be positive")
            
            # Check af_claseq config file if specified
            if config.config_file:
                config_file = Path(config.config_file)
                if not config_file.exists():
                    raise FileNotFoundError(f"af_claseq config file not found: {config_file}")
                
                ConfigurationUtils._validate_af_claseq_config_file(config_file)
            
            logger.debug("Structure analysis configuration validation passed")
            
        except Exception as e:
            raise ConfigurationError(f"Structure analysis configuration validation failed: {e}")
    
    @staticmethod
    def _validate_af_claseq_config_file(config_file: Path) -> None:
        """
        Validate af_claseq configuration file format.
        
        Args:
            config_file: Path to af_claseq configuration file
            
        Raises:
            ConfigurationError: If file format is invalid
        """
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Check required sections
            if "basics" not in config_data:
                raise ValueError("Missing 'basics' section")
            
            if "filter_criteria" not in config_data:
                raise ValueError("Missing 'filter_criteria' section")
            
            # Check basics structure
            basics = config_data["basics"]
            if "full_index" not in basics or "local_index" not in basics:
                raise ValueError("Missing required indices in 'basics' section")
            
            # Check filter criteria structure
            filter_criteria = config_data["filter_criteria"]
            if not isinstance(filter_criteria, list):
                raise ValueError("'filter_criteria' must be a list")
            
            for i, criterion in enumerate(filter_criteria):
                if not isinstance(criterion, dict):
                    raise ValueError(f"Filter criterion {i} must be a dictionary")
                
                required_fields = ["name", "type", "method"]
                for field in required_fields:
                    if field not in criterion:
                        raise ValueError(f"Filter criterion {i} missing required field: {field}")
            
            logger.debug(f"af_claseq configuration file validation passed: {config_file}")
            
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in af_claseq config file: {e}")
        except Exception as e:
            raise ConfigurationError(f"af_claseq config file validation failed: {e}")
    
    @staticmethod
    def get_config_summary(config: PipelineConfig) -> Dict[str, Any]:
        """
        Get a summary of the current configuration.
        
        Args:
            config: Pipeline configuration
            
        Returns:
            Dictionary with configuration summary
        """
        summary = {
            "pipeline": {
                "name": config.name,
                "version": config.version,
                "description": config.description
            },
            "processing": {
                "num_subsets": config.subsets.num_subsets,
                "num_batches": config.batches.num_batches,
                "random_sequences_per_subset": config.subsets.num_random_sequences
            },
            "structure_analysis": {
                "plddt_threshold": config.structure_analysis.plddt_threshold,
                "filter_criteria_threshold": config.structure_analysis.filter_criteria_threshold,
                "filter_criteria": config.structure_analysis.filter_criteria,
                "af_claseq_config": config.structure_analysis.config_file
            },
            "similarity_search": {
                "top_k": config.similarity_search.top_k,
                "similarity_threshold": config.similarity_search.similarity_threshold
            },
            "options": {
                "skip_structure_prediction": config.skip_structure_prediction,
                "skip_structure_analysis": config.skip_structure_analysis,
                "skip_hit_expansion": config.skip_hit_expansion,
                "check_existing_jobs": config.check_existing_jobs
            }
        }
        
        return summary


# ============================================================================
# Public API Functions (Clear, Non-Redundant Names)
# ============================================================================

def load_config(config_file: Union[str, Path]) -> PipelineConfig:
    """
    Load and validate pipeline configuration.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Validated PipelineConfig object
    """
    return ConfigurationUtils.load_pipeline_config(config_file)


def load_structure_config(pipeline_config: PipelineConfig) -> StructureAnalysisConfig:
    """
    Load structure analysis configuration with af_claseq integration.
    
    Args:
        pipeline_config: Main pipeline configuration
        
    Returns:
        StructureAnalysisConfig object
    """
    return ConfigurationUtils.load_structure_config(pipeline_config)


def get_filter_criteria(config_file: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Get available filter criteria from af_claseq configuration file.
    
    Args:
        config_file: Path to af_claseq configuration file
        
    Returns:
        List of available filter criteria with their names, types, and methods
    """
    return ConfigurationUtils.get_available_filter_criteria(config_file)


def create_default_config(output_file: Optional[Union[str, Path]] = None) -> PipelineConfig:
    """
    Create default pipeline configuration.
    
    Args:
        output_file: Optional path to save default config
        
    Returns:
        Default PipelineConfig object
    """
    from .settings import ConfigManager
    manager = ConfigManager()
    return manager.create_default_config(output_file)