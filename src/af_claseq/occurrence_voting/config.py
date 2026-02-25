"""
Configuration module for Occurrence Voting workflow.

This module provides dataclasses for occurrence voting configuration,
following AF_ClaSeq patterns and leveraging existing utilities.
"""

import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path

from af_claseq.utils.logging_utils import get_logger

# Initialize module logger
logger = get_logger("occurrence_voting_config")


@dataclass
class GeneralConfig:
    """General configuration options for occurrence voting workflow"""
    source_a3m: str
    base_dir: str
    protein_name: str
    random_seed: int = 42


@dataclass
class SamplingConfig:
    """Random sampling configuration"""
    num_groups: int = 1000  # Number of random samples to create
    group_size: int = 8     # Number of sequences per group
    num_batches: int = 10   # Number of SLURM batches to organize groups into

    def __post_init__(self):
        """Validate sampling configuration"""
        if self.num_groups <= 0:
            raise ValueError(f"num_groups must be positive, got: {self.num_groups}")
        if self.group_size <= 0:
            raise ValueError(f"group_size must be positive, got: {self.group_size}")
        if self.num_batches <= 0:
            raise ValueError(f"num_batches must be positive, got: {self.num_batches}")


@dataclass
class StructurePredictionConfig:
    """Structure prediction configuration for ColabFold"""
    num_models: int = 1     # Number of models per prediction
    num_seeds: int = 1      # Number of seeds per prediction
    prediction_mode: str = "monomer"  # Prediction mode: "monomer" or "homodimer"
    num_recycle: int = 3    # Number of recycle iterations (default: 3)

    def __post_init__(self):
        """Validate structure prediction configuration"""
        if self.prediction_mode not in ["monomer", "homodimer"]:
            raise ValueError(f"Invalid prediction_mode: '{self.prediction_mode}'. Must be 'monomer' or 'homodimer'")
        if self.num_recycle < 0:
            raise ValueError(f"num_recycle must be non-negative, got: {self.num_recycle}")


@dataclass
class StructureAnalysisConfig:
    """Structure analysis configuration"""
    config_json: str                    # Path to structure analysis JSON config file
    plddt_threshold: float = 0          # Minimum pLDDT threshold for analysis

    def __post_init__(self):
        """Validate structure analysis configuration"""
        if self.plddt_threshold < 0:
            raise ValueError(f"plddt_threshold must be non-negative, got: {self.plddt_threshold}")


@dataclass
class FilteringConfig:
    """Structure filtering configuration"""
    metric_name: str                    # Metric to use for filtering (from structure analysis)
    cutoff_value: float                 # Threshold value
    cutoff_method: str = "below"        # "above" or "below"

    def __post_init__(self):
        """Validate filtering configuration"""
        if self.cutoff_method not in ["above", "below"]:
            raise ValueError(f"cutoff_method must be 'above' or 'below', got: {self.cutoff_method}")


@dataclass
class VotingConfig:
    """Occurrence voting configuration"""
    top_n_sequences: int = 100  # Number of top sequences to select

    def __post_init__(self):
        """Validate voting configuration"""
        if self.top_n_sequences <= 0:
            raise ValueError(f"top_n_sequences must be positive, got: {self.top_n_sequences}")


@dataclass
class PlottingConfig:
    """Plotting configuration for occurrence voting"""
    enabled: bool = True                           # Enable/disable plotting
    metrics_to_plot: List[str] = None             # Specific metrics to plot (None = all available)
    plot_types: List[str] = None                  # Plot types: ['1d', '2d', 'correlation', 'joint', 'occurrence']
    output_subdir: str = "plots"                  # Subdirectory for plots

    # Advanced plotting configuration (new fields for sophisticated plotting)
    colors: Optional[Dict[str, List[str]]] = None           # Colors per metric (metric1, metric2, etc.)
    metric_ranges: Optional[Dict[str, Dict[str, Any]]] = None  # Ranges per metric (min, max, ticks)
    plot_params: Optional[Dict[str, Dict[str, Any]]] = None    # Plot-specific parameters (1d, 2d, etc.)

    def __post_init__(self):
        """Set defaults and validate plotting configuration"""
        if self.metrics_to_plot is None:
            self.metrics_to_plot = []  # Empty list means plot all available
        if self.plot_types is None:
            self.plot_types = ['1d', '2d', 'occurrence']  # Default to all plot types
        if self.colors is None:
            self.colors = {}  # Empty dict means use default colors
        if self.metric_ranges is None:
            self.metric_ranges = {}  # Empty dict means auto-determine ranges
        if self.plot_params is None:
            self.plot_params = {}  # Empty dict means use default parameters


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


@dataclass
class LocalGPUConfig:
    """Local GPU execution configuration"""
    cuda_visible_devices: str


@dataclass
class OccurrenceVotingConfig:
    """Complete occurrence voting workflow configuration"""
    general: GeneralConfig
    sampling: SamplingConfig
    structure_prediction: StructurePredictionConfig
    structure_analysis: StructureAnalysisConfig
    filtering: FilteringConfig
    voting: VotingConfig
    slurm: Optional[SlurmConfig] = None
    local_gpu: Optional[LocalGPUConfig] = None
    plotting: PlottingConfig = None

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'OccurrenceVotingConfig':
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
        for section in ['general', 'sampling', 'structure_analysis', 'filtering']:
            if section not in config_data:
                raise ValueError(f"Missing required configuration section: {section}")

        # Create configuration objects
        general_config = GeneralConfig(**config_data['general'])
        sampling_config = SamplingConfig(**config_data['sampling'])
        structure_analysis_config = StructureAnalysisConfig(**config_data['structure_analysis'])
        filtering_config = FilteringConfig(**config_data['filtering'])
        slurm_config = SlurmConfig(**config_data['slurm']) if has_slurm else None
        local_gpu_config = LocalGPUConfig(**config_data['local_gpu']) if has_local_gpu else None

        # Optional sections with defaults
        structure_prediction_data = config_data.get('structure_prediction', {})
        structure_prediction_config = StructurePredictionConfig(**structure_prediction_data)

        voting_data = config_data.get('voting', {})
        voting_config = VotingConfig(**voting_data)

        plotting_data = config_data.get('plotting', {})
        if plotting_data:
            # Filter plotting_data to only include fields that PlottingConfig accepts
            valid_fields = {'enabled', 'metrics_to_plot', 'plot_types', 'output_subdir',
                          'colors', 'metric_ranges', 'plot_params'}
            filtered_plotting_data = {k: v for k, v in plotting_data.items() if k in valid_fields}
            plotting_config = PlottingConfig(**filtered_plotting_data)
        else:
            plotting_config = PlottingConfig()

        workflow_config = cls(
            general=general_config,
            sampling=sampling_config,
            structure_prediction=structure_prediction_config,
            structure_analysis=structure_analysis_config,
            filtering=filtering_config,
            voting=voting_config,
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

        # Check structure analysis config JSON file
        structure_config = Path(self.structure_analysis.config_json)
        if not structure_config.exists():
            raise FileNotFoundError(f"Structure analysis config not found: {structure_config}")

        # Create base directory if it doesn't exist
        base_dir = Path(self.general.base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Validated paths - base_dir: {base_dir}, structure_config: {structure_config}")

    def get_output_dir(self) -> Path:
        """Get the main output directory for occurrence voting workflow"""
        return Path(self.general.base_dir)

    def get_groups_dir(self) -> Path:
        """Get the groups output directory"""
        return self.get_output_dir() / "groups"

    def get_batches_dir(self) -> Path:
        """Get the batches output directory"""
        return self.get_output_dir() / "batches"

    def get_results_dir(self) -> Path:
        """Get the results output directory"""
        return self.get_output_dir() / "results"


def load_config(yaml_path: str) -> OccurrenceVotingConfig:
    """
    Load occurrence voting workflow configuration from YAML file.

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        OccurrenceVotingConfig object with validated configuration

    Raises:
        FileNotFoundError: If configuration file doesn't exist
        ValueError: If configuration is invalid
    """
    return OccurrenceVotingConfig.from_yaml(yaml_path)