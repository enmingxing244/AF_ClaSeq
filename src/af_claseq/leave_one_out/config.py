"""
Configuration module for Leave-One-Out workflow.

This module provides dataclasses for leave-one-out configuration sections
and functions to load configuration from YAML files, following AF-ClaSeq patterns.
"""

import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path

from af_claseq.utils.logging_utils import get_logger

# Initialize module logger
logger = get_logger("loo_config")


@dataclass
class GeneralConfig:
    """General configuration options for leave-one-out workflow"""
    source_a3m: str
    base_dir: str
    structure_analysis_config: str  # JSON file with metrics configuration
    protein_name: str
    random_seed: int = 42


@dataclass
class LeaveOneOutConfig:
    """Leave-one-out specific configuration"""
    num_seq_per_group: int = 8  # Number of sequences per group
    impact_metric_name: str = "composite_rmsd"  # Metric to use for impact analysis
    impact_threshold: float = 1.0  # Threshold for significant impact
    cutoff_method: str = "above"  # "above" or "below" threshold
    full_group_mean_threshold: float = 3.0  # Additional filter on full group mean
    full_mean_cutoff_method: str = "below"  # "above" or "below" for full group mean
    min_sequences_for_loo: int = 3  # Minimum sequences needed for LOO analysis

    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.cutoff_method not in ["above", "below"]:
            raise ValueError(f"cutoff_method must be 'above' or 'below', got: {self.cutoff_method}")
        if self.full_mean_cutoff_method not in ["above", "below"]:
            raise ValueError(f"full_mean_cutoff_method must be 'above' or 'below', got: {self.full_mean_cutoff_method}")
        if self.num_seq_per_group < 2:
            raise ValueError(f"num_seq_per_group must be at least 2, got: {self.num_seq_per_group}")


@dataclass
class SlurmConfig:
    """SLURM configuration options for ColabFold jobs"""
    conda_env_path: str
    account: str
    partition: str = "nextgen"
    time: str = "00:30:00"
    memory: str = "32G"
    cpus: int = 8
    max_concurrent_jobs: int = 90
    num_models: int = 5  # Required for LOO analysis
    num_seeds: int = 1


@dataclass
class LocalGPUConfig:
    """Local GPU execution configuration"""
    cuda_visible_devices: str


@dataclass
class PlottingConfig:
    """Plotting configuration for impact visualization"""
    output_dir: Optional[str] = None  # If None, will use base_dir/leave_one_out/plots
    figsize: Tuple[float, float] = (12, 6)
    dpi: int = 150
    show_plots: bool = False  # Whether to display plots interactively


@dataclass
class WorkflowConfig:
    """Complete workflow configuration"""
    general: GeneralConfig
    leave_one_out: LeaveOneOutConfig
    slurm: Optional[SlurmConfig] = None
    local_gpu: Optional[LocalGPUConfig] = None
    plotting: PlottingConfig = field(default_factory=PlottingConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'WorkflowConfig':
        """Load configuration from YAML file"""
        yaml_path = Path(yaml_path)

        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        logger.info(f"Loading configuration from: {yaml_path}")

        with open(yaml_path, 'r') as f:
            config_data = yaml.safe_load(f)

        # Validate execution mode: exactly one of slurm or local_gpu
        has_slurm = 'slurm' in config_data and config_data['slurm'] is not None
        has_local_gpu = 'local_gpu' in config_data and config_data['local_gpu'] is not None

        if has_slurm and has_local_gpu:
            raise ValueError(
                "Config error: Cannot specify both 'slurm' and 'local_gpu' sections. "
                "Please choose one execution mode."
            )
        if not has_slurm and not has_local_gpu:
            raise ValueError(
                "Config error: Must specify either 'slurm' or 'local_gpu' section "
                "to define the execution mode."
            )

        # Validate other required sections
        for section in ['general', 'leave_one_out']:
            if section not in config_data:
                raise ValueError(f"Missing required configuration section: {section}")

        # Create configuration objects
        general_config = GeneralConfig(**config_data['general'])
        loo_config = LeaveOneOutConfig(**config_data['leave_one_out'])
        slurm_config = SlurmConfig(**config_data['slurm']) if has_slurm else None
        local_gpu_config = LocalGPUConfig(**config_data['local_gpu']) if has_local_gpu else None

        # Optional plotting config
        plotting_data = config_data.get('plotting', {})
        plotting_config = PlottingConfig(**plotting_data)

        # Set default plotting output directory if not specified
        if plotting_config.output_dir is None:
            plotting_config.output_dir = str(Path(general_config.base_dir) / "plots")

        workflow_config = cls(
            general=general_config,
            leave_one_out=loo_config,
            slurm=slurm_config,
            local_gpu=local_gpu_config,
            plotting=plotting_config
        )

        # Validate file paths
        workflow_config._validate_paths()

        logger.info("Configuration loaded successfully")
        return workflow_config

    def _validate_paths(self):
        """Validate that required files and directories exist"""
        # Check source A3M file
        source_a3m = Path(self.general.source_a3m)
        if not source_a3m.exists():
            raise FileNotFoundError(f"Source A3M file not found: {source_a3m}")

        # Check structure analysis config
        structure_config = Path(self.general.structure_analysis_config)
        if not structure_config.exists():
            raise FileNotFoundError(f"Structure analysis config not found: {structure_config}")

        # Create base directory if it doesn't exist
        base_dir = Path(self.general.base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        # Create plotting directory
        plotting_dir = Path(self.plotting.output_dir)
        plotting_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Validated paths - base_dir: {base_dir}, plotting_dir: {plotting_dir}")

    def get_output_dir(self) -> Path:
        """Get the main output directory for leave-one-out workflow"""
        return Path(self.general.base_dir)

    def get_plots_dir(self) -> Path:
        """Get the plots output directory"""
        return Path(self.plotting.output_dir)


def load_config(yaml_path: str) -> WorkflowConfig:
    """
    Load leave-one-out workflow configuration from YAML file.

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        WorkflowConfig object with validated configuration

    Raises:
        FileNotFoundError: If configuration file doesn't exist
        ValueError: If configuration is invalid
    """
    return WorkflowConfig.from_yaml(yaml_path)