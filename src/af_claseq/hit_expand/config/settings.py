#!/usr/bin/env python3
"""
Configuration management for the MSA clustering and structure optimization pipeline.
Provides validation, defaults, and type-safe configuration handling.
"""

import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Union, Optional, List
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class ConfigFormat(Enum):
    """Supported configuration file formats."""
    JSON = "json"
    YAML = "yaml"




@dataclass
class SubsetConfig:
    """Configuration for subset generation."""
    num_subsets: int = 2000
    num_random_sequences: int = 8
    random_seed: int = 42
    output_prefix: str = "subset"
    ensure_query_first: bool = True


@dataclass
class BatchConfig:
    """Configuration for batch organization."""
    num_batches: int = 50
    batch_prefix: str = "batch"


@dataclass
class SlurmConfig:
    """Configuration for SLURM job submission."""
    account: str = "PAA0203"
    partition: str = "nextgen"
    time_limit: str = "02:00:00"
    memory: str = "32G"
    cpus_per_task: str = "8"
    nodes: int = 1
    gres: str = "gpu:1"
    conda_env_path: str = "/fs/ess/PAA0203/xing244/.conda/envs/colabfold"
    delay_between_jobs: float = 1.0


@dataclass
class StructureAnalysisConfig:
    """Configuration for structure analysis using af_claseq."""
    plddt_threshold: float = 75.0
    filter_criteria_threshold: float = 6.0  # Threshold for filter criteria (RMSD, distance, angle, etc.)
    filter_criteria: str = "6xrg_rmsd"  # Which filter criteria to use for filtering
    n_cores: Optional[int] = None
    config_file: Optional[str] = None  # Path to af_claseq config file


@dataclass
class SimilaritySearchConfig:
    """Configuration for BLOSUM62-based similarity search."""
    top_k: int = 50
    similarity_threshold: float = 0.7
    exclude_query_headers: bool = True


@dataclass
class PipelineConfig:
    """Main pipeline configuration."""
    # Core configurations (clustering removed - provided externally)
    subsets: SubsetConfig = field(default_factory=SubsetConfig)
    batches: BatchConfig = field(default_factory=BatchConfig)
    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    structure_analysis: StructureAnalysisConfig = field(default_factory=StructureAnalysisConfig)
    similarity_search: SimilaritySearchConfig = field(default_factory=SimilaritySearchConfig)
    
    # Pipeline settings
    name: str = "msa_structure_pipeline"
    version: str = "1.0.0"
    description: str = "MSA structure-guided optimization pipeline (clustering external)"
    
    # I/O settings
    input_file: Optional[str] = None  # Clustered representative sequences
    source_msa_file: Optional[str] = None  # Source MSA for similarity search
    output_dir: str = "pipeline_output"
    
    # Processing options (clustering removed)
    skip_structure_prediction: bool = False
    skip_structure_analysis: bool = False
    skip_hit_expansion: bool = False
    check_existing_jobs: bool = True  # Check if ColabFold jobs are already complete
    
    # Monitoring options
    monitor_jobs: bool = True
    job_check_interval: float = 60.0
    job_timeout: Optional[float] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        # Validate subset config
        if self.subsets.num_subsets <= 0:
            raise ValueError("Number of subsets must be positive")
        
        if self.subsets.num_random_sequences <= 0:
            raise ValueError("Number of random sequences must be positive")
        
        # Validate batch config
        if self.batches.num_batches <= 0:
            raise ValueError("Number of batches must be positive")
        
        # Validate structure analysis config
        if not 0.0 <= self.structure_analysis.plddt_threshold <= 100.0:
            raise ValueError("pLDDT threshold must be between 0.0 and 100.0")
        
        if self.structure_analysis.filter_criteria_threshold <= 0.0:
            raise ValueError("Filter criteria threshold must be positive")
        
        # Validate similarity search config
        if self.similarity_search.top_k <= 0:
            raise ValueError("Top-k value must be positive")
        
        if not 0.0 <= self.similarity_search.similarity_threshold <= 1.0:
            raise ValueError("Similarity threshold must be between 0.0 and 1.0")


class ConfigManager:
    """Manages configuration loading, saving, and validation."""
    
    def __init__(self):
        """Initialize configuration manager."""
        self._config_cache: Dict[str, PipelineConfig] = {}
    
    def load_config(self, config_file: Union[str, Path], 
                   config_format: Optional[ConfigFormat] = None) -> PipelineConfig:
        """
        Load configuration from file.
        
        Args:
            config_file: Path to configuration file
            config_format: File format (auto-detected if None)
            
        Returns:
            PipelineConfig object
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config format is unsupported or invalid
        """
        config_file = Path(config_file)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        # Auto-detect format if not specified
        if config_format is None:
            config_format = self._detect_format(config_file)
        
        logger.info(f"Loading configuration from {config_file} ({config_format.value})")
        
        try:
            with open(config_file, 'r') as f:
                if config_format == ConfigFormat.JSON:
                    config_data = json.load(f)
                elif config_format == ConfigFormat.YAML:
                    config_data = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported config format: {config_format}")
            
            # Convert to PipelineConfig object
            config = self._dict_to_config(config_data)
            
            # Cache the config
            cache_key = str(config_file.absolute())
            self._config_cache[cache_key] = config
            
            logger.info("Configuration loaded successfully")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise ValueError(f"Invalid configuration file: {e}")
    
    def save_config(self, config: PipelineConfig, 
                   output_file: Union[str, Path],
                   config_format: Optional[ConfigFormat] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            config: PipelineConfig object to save
            output_file: Path to output file
            config_format: File format (auto-detected if None)
        """
        output_file = Path(output_file)
        
        # Auto-detect format if not specified
        if config_format is None:
            config_format = self._detect_format(output_file)
        
        # Create output directory
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving configuration to {output_file} ({config_format.value})")
        
        try:
            config_data = asdict(config)
            
            with open(output_file, 'w') as f:
                if config_format == ConfigFormat.JSON:
                    json.dump(config_data, f, indent=2)
                elif config_format == ConfigFormat.YAML:
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
                else:
                    raise ValueError(f"Unsupported config format: {config_format}")
            
            logger.info("Configuration saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise ValueError(f"Failed to save configuration: {e}")
    
    def create_default_config(self, output_file: Optional[Union[str, Path]] = None) -> PipelineConfig:
        """
        Create default configuration.
        
        Args:
            output_file: Optional path to save default config
            
        Returns:
            Default PipelineConfig object
        """
        config = PipelineConfig()
        
        if output_file:
            self.save_config(config, output_file)
        
        return config
    
    def _detect_format(self, file_path: Path) -> ConfigFormat:
        """Detect configuration file format from extension."""
        suffix = file_path.suffix.lower()
        
        if suffix == '.json':
            return ConfigFormat.JSON
        elif suffix in ['.yaml', '.yml']:
            return ConfigFormat.YAML
        else:
            # Default to JSON
            return ConfigFormat.JSON
    
    def _dict_to_config(self, config_data: Dict[str, Any]) -> PipelineConfig:
        """Convert dictionary to PipelineConfig object."""
        # Extract nested configurations (clustering removed)
        subsets_data = config_data.get('subsets', {})
        batches_data = config_data.get('batches', {})
        slurm_data = config_data.get('slurm', {})
        structure_analysis_data = config_data.get('structure_analysis', {})
        similarity_search_data = config_data.get('similarity_search', {})
        
        # Create configuration objects
        subsets_config = SubsetConfig(**subsets_data)
        batches_config = BatchConfig(**batches_data)
        slurm_config = SlurmConfig(**slurm_data)
        structure_analysis_config = StructureAnalysisConfig(**structure_analysis_data)
        similarity_search_config = SimilaritySearchConfig(**similarity_search_data)
        
        # Extract main pipeline settings
        pipeline_data = {k: v for k, v in config_data.items() 
                        if k not in ['subsets', 'batches', 'slurm', 
                                   'structure_analysis', 'similarity_search']}
        
        # Create main config
        config = PipelineConfig(
            subsets=subsets_config,
            batches=batches_config,
            slurm=slurm_config,
            structure_analysis=structure_analysis_config,
            similarity_search=similarity_search_config,
            **pipeline_data
        )
        
        return config
    
    def validate_paths(self, config: PipelineConfig) -> List[str]:
        """
        Validate file paths in configuration (all paths must be absolute).
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Check that output_dir is absolute
        if not Path(config.output_dir).is_absolute():
            errors.append(f"Output directory must be absolute path: {config.output_dir}")
        
        # Check input file (clustered representatives)
        if config.input_file:
            if not Path(config.input_file).is_absolute():
                errors.append(f"Input file must be absolute path: {config.input_file}")
            elif not Path(config.input_file).exists():
                errors.append(f"Clustered representatives file not found: {config.input_file}")
        
        # Check source MSA file
        if config.source_msa_file:
            if not Path(config.source_msa_file).is_absolute():
                errors.append(f"Source MSA file must be absolute path: {config.source_msa_file}")
            elif not Path(config.source_msa_file).exists():
                errors.append(f"Source MSA file not found: {config.source_msa_file}")
        
        # Check af_claseq config file
        if config.structure_analysis.config_file:
            config_path = Path(config.structure_analysis.config_file)
            if not config_path.is_absolute():
                errors.append(f"af_claseq config file must be absolute path: {config_path}")
            elif not config_path.exists():
                errors.append(f"af_claseq config file not found: {config_path}")
        
        # Check conda environment
        conda_path = Path(config.slurm.conda_env_path)
        if not conda_path.is_absolute():
            errors.append(f"Conda environment path must be absolute: {conda_path}")
        elif not conda_path.exists():
            errors.append(f"Conda environment not found: {conda_path}")
        
        return errors


# Configuration loading functions moved to utils.py to avoid naming conflicts
# Use: from msa_pipeline.config.utils import load_config, create_default_config