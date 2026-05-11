"""
Configuration module for AF-ClaSeq pipeline.

This module provides dataclasses for different configuration sections
and functions to load configuration from YAML files.
"""

import math
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from af_claseq.utils.plotting_manager import COLORS


@dataclass
class MetricBinConfig:
    """Per-metric binning parameters for unit-based binning.

    When bin_width is set, min and max are required — the total number of bins
    is computed as ceil((max - min) / bin_width).
    When bin_width is None, falls back to a default bin count of 30.
    """
    bin_width: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None

    def __post_init__(self):
        if self.bin_width is not None:
            if self.bin_width <= 0:
                raise ValueError(f"bin_width must be positive, got {self.bin_width}")
            if self.min is None or self.max is None:
                raise ValueError(
                    "When bin_width is specified, both 'min' and 'max' are required. "
                    "Please define the metric range explicitly."
                )
            if self.max <= self.min:
                raise ValueError(f"max ({self.max}) must be greater than min ({self.min})")

    def compute_num_bins(self) -> Optional[int]:
        """Compute the number of bins from bin_width and range. Returns None if bin_width not set."""
        if self.bin_width is None:
            return None
        return max(1, math.ceil((self.max - self.min) / self.bin_width))

@dataclass
class GeneralConfig:
    """General configuration options"""
    source_a3m: str
    base_dir: str
    config_file: str  # This refers to the JSON filter criteria file
    protein_name: str
    default_pdb: Optional[str] = None  # Optional - query sequence will be read from source_a3m if not provided
    coverage_threshold: float = 0.8
    num_models: int = 1
    random_seed: int = 42
    metric1_color: List[str] = field(default_factory=lambda: ["#87CEEB", "#FFFFFF"])  # [start, end] color gradient for metric 1
    metric2_color: List[str] = field(default_factory=lambda: ["#FFB6C1", "#8B0000"])  # [start, end] color gradient for metric 2
    
    # Explicit metric selection
    use_composite_metrics: bool = False
    metric1_name: Optional[str] = None
    metric2_name: Optional[str] = None
    metric1_label: Optional[str] = None
    metric2_label: Optional[str] = None
    metric_bin_configs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SlurmConfig:
    """SLURM configuration options"""
    conda_env_path: str
    slurm_account: str
    slurm_output: str
    slurm_error: str
    slurm_nodes: int
    slurm_gpus_per_task: int
    slurm_tasks: int
    slurm_cpus_per_task: int
    slurm_time: str
    slurm_partition: str
    max_workers: int


@dataclass
class PipelineControlConfig:
    """Pipeline control options"""
    stages: List[str] = field(default_factory=lambda: [
        "01_M_FOLD_SAMPLING_RUN", "01_M_FOLD_SAMPLING_PLOT",
        "02_VOTING_RUN", "03_RECOMPILE_PREDICT_RUN",
        "04_PURE_SEQ_PLOT_RUN"
    ])
    check_interval: int = 60



@dataclass
class MFoldSamplingConfig:
    """Stage 01: M-fold Sampling parameters"""
    m_fold_samp_input_a3m: str
    m_fold_group_size: int = 10
    m_fold_random_select: Optional[int] = None
    m_fold_plddt_threshold: float = 75
    m_fold_log_scale: bool = False
    m_fold_gradient_ascending: bool = False
    m_fold_linear_gradient: bool = False
    m_fold_figsize: Tuple[float, float] = (10, 5)
    m_fold_show_bin_lines: bool = False
    m_fold_count_min: Optional[float] = None
    m_fold_count_max: Optional[float] = None
    m_fold_metric1_min: Optional[float] = None
    m_fold_metric1_max: Optional[float] = None
    m_fold_metric2_min: Optional[float] = None
    m_fold_metric2_max: Optional[float] = None
    m_fold_metric1_ticks: Optional[List[float]] = None
    m_fold_metric2_ticks: Optional[List[float]] = None
    rounds: int = 1  # New parameter for number of sampling rounds



@dataclass
class SequenceVotingConfig:
    """Stage 02: Sequence Voting parameters"""
    vote_threshold: float = 0.0
    vote_min_value: Optional[float] = None
    vote_max_value: Optional[float] = None
    vote_figsize: Tuple[float, float] = (10, 5)
    vote_y_min: Optional[float] = None
    vote_y_max: Optional[float] = None
    vote_x_ticks: Optional[List[int]] = None
    use_focused_bins: bool = False
    

@dataclass
class RecompilePredictConfig:
    """Stage 03: Recompilation & Prediction parameters"""
    bin_numbers_1: Union[List[int], int] = field(default_factory=list)
    bin_numbers_2: Union[List[int], int] = field(default_factory=list)
    combine_bins: bool = False
    metric_name_1: Optional[str] = None
    metric_name_2: Optional[str] = None
    prediction_num_model: int = 5
    prediction_num_seed: int = 8

    # NEW: Specify which metrics to process in stages 03 & 04
    # If None or empty → process all metrics (backward compatible)
    # If ["bound_rmsd"] → only process that metric
    # If ["bound_rmsd", "apo_rmsd"] → process both
    metrics_to_process: Optional[List[str]] = None

@dataclass
class PureSequencePlottingConfig:
    """Configuration for pure sequence plotting."""
    metric1_min: Optional[float] = None
    metric1_max: Optional[float] = None
    metric2_min: Optional[float] = None
    metric2_max: Optional[float] = None
    metric1_ticks: Optional[List[float]] = None
    metric2_ticks: Optional[List[float]] = None
    plddt_threshold: float = 70.0
    figsize: Tuple[int, int] = (15, 7)  # Updated to match PureSequencePlottingConfig
    dpi: int = 300  # Updated to match PureSequencePlottingConfig
    max_workers: int = 8





@dataclass
class PipelineConfig:
    """Complete pipeline configuration"""
    general: GeneralConfig
    slurm: SlurmConfig
    pipeline_control: PipelineControlConfig
    m_fold_sampling: MFoldSamplingConfig
    sequence_voting: SequenceVotingConfig
    recompile_predict: RecompilePredictConfig
    pure_sequence_plotting: PureSequencePlottingConfig


def get_selected_metrics(general_config: GeneralConfig) -> List[str]:
    """
    Get the list of selected metrics based on configuration.
    
    Args:
        general_config: General configuration object
        
    Returns:
        List of metric names to use (in order: metric1, metric2, ...)
    """
    import json
    
    selected_metrics = []
    
    # If explicit metric names are specified, use them
    if general_config.metric1_name:
        selected_metrics.append(general_config.metric1_name)
    if general_config.metric2_name:
        selected_metrics.append(general_config.metric2_name)
    
    # If no explicit names specified, fall back to default behavior
    if not selected_metrics:
        # Load JSON config to get available metrics
        with open(general_config.config_file, 'r') as f:
            json_config = json.load(f)
            
        if general_config.use_composite_metrics:
            # Use composite metrics (first 2 by default)
            composite_metrics = json_config.get('composite_metrics', [])
            selected_metrics = [comp['name'] for comp in composite_metrics[:2]]
        else:
            # Use regular filter criteria (first 2 by default)
            filter_criteria = json_config.get('filter_criteria', [])
            selected_metrics = [crit['name'] for crit in filter_criteria[:2]]
    
    return selected_metrics


def validate_metric_names(general_config: GeneralConfig) -> None:
    """
    Validate that specified metric names exist in the JSON configuration file.
    
    Args:
        general_config: General configuration object containing metric selections
        
    Raises:
        ValueError: If specified metrics don't exist in the JSON config
    """
    import json
    import os
    
    if not os.path.exists(general_config.config_file):
        raise ValueError(f"JSON config file not found: {general_config.config_file}")
    
    # Load JSON config to check available metrics
    with open(general_config.config_file, 'r') as f:
        json_config = json.load(f)
    
    # Get available metric names based on use_composite_metrics flag
    if general_config.use_composite_metrics:
        available_metrics = [comp['name'] for comp in json_config.get('composite_metrics', [])]
        metric_type = "composite_metrics"
        source_section = "composite_metrics"
    else:
        available_metrics = [crit['name'] for crit in json_config.get('filter_criteria', [])]
        metric_type = "filter_criteria"  
        source_section = "filter_criteria"
    
    # Validate metric1_name
    if general_config.metric1_name:
        if general_config.metric1_name not in available_metrics:
            raise ValueError(
                f"metric1_name '{general_config.metric1_name}' not found in JSON config section '{source_section}'. "
                f"Available {metric_type} metrics: {available_metrics}"
            )
    
    # Validate metric2_name
    if general_config.metric2_name:
        if general_config.metric2_name not in available_metrics:
            raise ValueError(
                f"metric2_name '{general_config.metric2_name}' not found in JSON config section '{source_section}'. "
                f"Available {metric_type} metrics: {available_metrics}"
            )


def load_pipeline_config(yaml_input: str) -> PipelineConfig:
    """
    Load configuration from YAML file and create config objects
    
    Args:
        yaml_input: Path to YAML configuration file with pipeline parameters
        
    Returns:
        PipelineConfig object with all configuration options
    """
    with open(yaml_input, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    # Create individual config objects
    general_config = GeneralConfig(**yaml_config.get('general', {}))
    
    # Validate metric names if specified
    if general_config.metric1_name or general_config.metric2_name:
        validate_metric_names(general_config)
    
    slurm_config = SlurmConfig(**yaml_config.get('slurm', {}))
    pipeline_control_config = PipelineControlConfig(**yaml_config.get('pipeline_control', {}))
    m_fold_sampling_config = MFoldSamplingConfig(**yaml_config.get('m_fold_sampling', {}))
    sequence_voting_config = SequenceVotingConfig(**yaml_config.get('sequence_voting', {}))
    recompile_predict_config = RecompilePredictConfig(**yaml_config.get('recompile_predict', {}))
    pure_sequence_plotting_config = PureSequencePlottingConfig(**yaml_config.get('pure_sequence_plotting', {}))

    # Combine into a single config object
    return PipelineConfig(
        general=general_config,
        slurm=slurm_config,
        pipeline_control=pipeline_control_config,
        m_fold_sampling=m_fold_sampling_config,
        sequence_voting=sequence_voting_config,
        recompile_predict=recompile_predict_config,
        pure_sequence_plotting=pure_sequence_plotting_config
    )


def get_metric_bin_config(general_config: GeneralConfig, metric_name: str) -> Optional[MetricBinConfig]:
    """Look up the MetricBinConfig for a given metric name, or None if not configured."""
    raw = general_config.metric_bin_configs.get(metric_name)
    if raw is None:
        return None
    if isinstance(raw, MetricBinConfig):
        return raw
    return MetricBinConfig(**{k: v for k, v in raw.items() if k in MetricBinConfig.__dataclass_fields__})