"""
Configuration module for AF-ClaSeq pipeline.

This module provides dataclasses for different configuration sections
and functions to load configuration from YAML files.
"""

import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from af_claseq.utils.plotting_manager import COLORS
# HitExpandConfig now defined in this file

@dataclass
class GeneralConfig:
    """General configuration options"""
    source_a3m: str
    default_pdb: str
    base_dir: str
    config_file: str  # This refers to the JSON filter criteria file
    protein_name: str
    coverage_threshold: float = 0.8
    num_models: int = 1
    random_seed: int = 42
    num_bins: int = 30
    metric1_color: List[str] = field(default_factory=lambda: ["#87CEEB", "#FFFFFF"])  # [start, end] color gradient for metric 1
    metric2_color: List[str] = field(default_factory=lambda: ["#FFB6C1", "#8B0000"])  # [start, end] color gradient for metric 2
    
    # Explicit metric selection
    use_composite_metrics: bool = False
    metric1_name: Optional[str] = None
    metric2_name: Optional[str] = None


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
        "01_HIT_EXPAND_RUN", 
        "02_M_FOLD_SAMPLING_RUN", "02_M_FOLD_SAMPLING_PLOT", 
        "03_VOTING_RUN", "04_RECOMPILE_PREDICT_RUN", 
        "05_PURE_SEQ_PLOT_RUN"
    ])
    check_interval: int = 60



@dataclass
class MFoldSamplingConfig:
    """Stage 02: M-fold Sampling parameters"""
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
    """Stage 03: Sequence Voting parameters"""
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
    """Stage 04: Recompilation & Prediction parameters"""
    bin_numbers_1: Union[List[int], int] = field(default_factory=list)
    bin_numbers_2: Union[List[int], int] = field(default_factory=list)
    combine_bins: bool = False
    metric_name_1: Optional[str] = None
    metric_name_2: Optional[str] = None
    prediction_num_model: int = 5
    prediction_num_seed: int = 8

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
class InitBootstrappingConfig:
    """Configuration for initialization bootstrapping step."""
    
    # Subset generation parameters (reduced for quick preview)
    init_num_subsets: int = 200
    init_num_random_sequences: int = 8
    init_num_batches: int = 10
    init_batch_prefix: str = "init_batch"
    init_output_prefix: str = "init_subset"
    
    # Structure analysis parameters
    init_plddt_threshold: float = 75.0
    init_plot_all_structures: bool = True
    
    # Plotting configuration
    init_plot_metric1_min: float = 0.0
    init_plot_metric1_max: float = 1.0
    init_plot_metric2_min: float = 0.0
    init_plot_metric2_max: float = 1.0
    init_plot_metric1_ticks: Optional[List[float]] = field(default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    init_plot_metric2_ticks: Optional[List[float]] = field(default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    init_plot_figsize: Tuple[int, int] = (10, 8)
    
    # Control flags
    skip_if_exists: bool = True
    random_seed: int = 42
    ensure_query_first: bool = True
    
    # Job monitoring (reuse from hit_expand)
    monitor_jobs: bool = True
    job_check_interval: float = 60.0
    max_job_wait_time: float = 14400.0  # 4 hours
    
    # Integration with filter config
    filter_config_path: str = ""  # Will be set from general.config_file
    
    def __post_init__(self):
        """Post-initialization processing."""
        # This will be set later when the full config is loaded
        pass


@dataclass
class HitExpandConfig:
    """Configuration for hit expand pipeline step."""
    
    # Input MSA file (source A3M) - can reference general.source_a3m
    input_msa: str = ""
    
    # MMseqs2 clustering configuration
    mmseqs_bin: str = "/fs/ess/PAA0203/xing244/packages/mmseqs/bin/mmseqs"
    mmseqs_coverage: float = 0.8
    mmseqs_min_seq_id: float = 0.7
    mmseqs_cov_mode: int = 0
    mmseqs_cluster_mode: int = 0
    mmseqs_threads: int = 8
    mmseqs_tmp_dir: str = "/tmp"
    
    # Subset generation configuration
    num_subsets: int = 2000
    num_random_sequences: int = 8
    num_batches: int = 80
    batch_prefix: str = "batch"
    
    # Similarity search configuration  
    similarity_top_k: int = 50
    similarity_threshold: float = 0.7
    exclude_query_headers: bool = True
    
    # Structure analysis configuration
    plddt_threshold: float = 75.0
    filter_criteria_threshold: float = 0.8
    filter_criteria: str = "default"
    
    # Job monitoring
    monitor_jobs: bool = True
    job_check_interval: float = 60.0
    job_timeout: Optional[float] = None
    check_existing_jobs: bool = True
    
    # Processing control
    skip_structure_prediction: bool = False
    skip_structure_analysis: bool = False
    skip_hit_expansion: bool = False
    skip_clustering: bool = False
    
    # Output configuration
    output_prefix: str = "subset"
    
    # Plotting configuration
    plot_num_cols: int = 5
    plot_x_min: float = 0
    plot_x_max: float = 20
    plot_y_min: float = 0.8
    plot_y_max: float = 10000
    plot_xticks: Optional[List[float]] = None
    plot_bin_step: float = 0.2
    
    # Scatter plot configuration (for structure analysis plots)
    scatter_plot_metric1_min: float = 0.0
    scatter_plot_metric1_max: float = 1.0
    scatter_plot_metric2_min: float = 0.0
    scatter_plot_metric2_max: float = 1.0
    scatter_plot_metric1_ticks: Optional[List[float]] = field(default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    scatter_plot_metric2_ticks: Optional[List[float]] = field(default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    # pLDDT plot configuration
    plddt_plot_min: int = 0
    plddt_plot_max: int = 100
    plddt_plot_ticks: Optional[List[int]] = field(default_factory=lambda: [0, 20, 40, 60, 80, 100])
    
    # Multi-round configuration
    rounds: int = 1                         # Number of iterative expansion rounds
    cumulative_expansion: bool = True       # Accumulate sequences across rounds
    
    # Expansion method configuration
    expansion_method: str = "BLOSUM62"      # Options: "BLOSUM62" or "mmseqs_result"
    max_sequences_per_cluster: int = 50     # Maximum sequences per cluster in MMseqs2 expansion
    
    # Integration parameters
    random_seed: int = 42
    max_workers: int = 96
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.mmseqs_coverage < 0 or self.mmseqs_coverage > 1:
            raise ValueError("mmseqs_coverage must be between 0 and 1")
        if self.mmseqs_min_seq_id < 0 or self.mmseqs_min_seq_id > 1:
            raise ValueError("mmseqs_min_seq_id must be between 0 and 1")
        if self.similarity_threshold < 0 or self.similarity_threshold > 1:
            raise ValueError("similarity_threshold must be between 0 and 1")
        if self.expansion_method not in ["BLOSUM62", "mmseqs_result"]:
            raise ValueError("expansion_method must be either 'BLOSUM62' or 'mmseqs_result'")
        if self.plddt_threshold < 0 or self.plddt_threshold > 100:
            raise ValueError("plddt_threshold must be between 0 and 100")

@dataclass
class PipelineConfig:
    """Complete pipeline configuration"""
    general: GeneralConfig
    slurm: SlurmConfig
    pipeline_control: PipelineControlConfig
    init_bootstrapping: InitBootstrappingConfig
    hit_expand: HitExpandConfig
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
    
    # Create init_bootstrapping config with filter_config_path from general config
    init_bootstrapping_dict = yaml_config.get('init_bootstrapping', {})
    init_bootstrapping_dict['filter_config_path'] = general_config.config_file
    init_bootstrapping_config = InitBootstrappingConfig(**init_bootstrapping_dict)
    
    hit_expand_config = HitExpandConfig(**yaml_config.get('hit_expand', {}))
    m_fold_sampling_config = MFoldSamplingConfig(**yaml_config.get('m_fold_sampling', {}))
    sequence_voting_config = SequenceVotingConfig(**yaml_config.get('sequence_voting', {}))
    recompile_predict_config = RecompilePredictConfig(**yaml_config.get('recompile_predict', {}))
    pure_sequence_plotting_config = PureSequencePlottingConfig(**yaml_config.get('pure_sequence_plotting', {}))
    
    # Combine into a single config object
    return PipelineConfig(
        general=general_config,
        slurm=slurm_config,
        pipeline_control=pipeline_control_config,
        init_bootstrapping=init_bootstrapping_config,
        hit_expand=hit_expand_config,
        m_fold_sampling=m_fold_sampling_config,
        sequence_voting=sequence_voting_config,
        recompile_predict=recompile_predict_config,
        pure_sequence_plotting=pure_sequence_plotting_config
    )