"""
Configuration module for AF-ClaSeq pipeline.

This module provides dataclasses for different configuration sections
and functions to load configuration from YAML files.
"""

import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from af_claseq.utils.plotting_manager import COLORS

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
    plot_initial_color: str = "#87CEEB"
    plot_end_color: str = "#FFFFFF"


@dataclass
class SlurmConfig:
    """SLURM configuration options"""
    conda_env_path: str = "/fs/ess/PAA0203/xing244/.conda/envs/colabfold"
    slurm_account: str = "PAA0203"
    slurm_output: str = "/dev/null"
    slurm_error: str = "/dev/null"
    slurm_nodes: int = 1
    slurm_gpus_per_task: int = 1
    slurm_tasks: int = 1
    slurm_cpus_per_task: int = 4
    slurm_time: str = "04:00:00"
    slurm_partition: str = "nextgen"
    max_workers: int = 64


@dataclass
class PipelineControlConfig:
    """Pipeline control options"""
    stages: List[str] = field(default_factory=lambda: [
        "01_RUN", "01_ANALYSIS", "02_RUN", "02_ANALYSIS", "03", "04"
    ])
    check_interval: int = 60


@dataclass
class IterativeShufflingConfig:
    """Stage 01: Iterative Shuffling parameters"""
    iter_shuf_input_a3m: str
    num_iterations: int = 8
    num_shuffles: int = 10
    seq_num_per_shuffle: int = 16
    plddt_threshold: int = 75
    quantile: float = 0.2
    resume_from_iter: Optional[int] = None
    iter_shuf_plot_num_cols: int = 5
    iter_shuf_plot_x_min: float = 0
    iter_shuf_plot_x_max: float = 20
    iter_shuf_plot_y_min: float = 0.8
    iter_shuf_plot_y_max: float = 10000
    iter_shuf_plot_xticks: Optional[List[float]] = None
    iter_shuf_plot_bin_step: float = 0.2
    iter_shuf_combine_threshold: float = 0.5

@dataclass
class MFoldSamplingConfig:
    """Stage 02: M-fold Sampling parameters"""
    m_fold_samp_input_a3m: str
    m_fold_group_size: int = 10
    m_fold_random_select: Optional[int] = None
    m_fold_plddt_threshold: float = 75
    m_fold_initial_color: str = "#87CEEB"
    m_fold_end_color: str = "#FFFFFF"
    m_fold_log_scale: bool = False
    m_fold_n_plot_bins: int = 50
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
    vote_hierarchical_sampling: bool = False
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
class PipelineConfig:
    """Complete pipeline configuration"""
    general: GeneralConfig
    slurm: SlurmConfig
    pipeline_control: PipelineControlConfig
    iterative_shuffling: IterativeShufflingConfig
    m_fold_sampling: MFoldSamplingConfig
    sequence_voting: SequenceVotingConfig
    recompile_predict: RecompilePredictConfig
    pure_sequence_plotting: PureSequencePlottingConfig


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
    slurm_config = SlurmConfig(**yaml_config.get('slurm', {}))
    pipeline_control_config = PipelineControlConfig(**yaml_config.get('pipeline_control', {}))
    iterative_shuffling_config = IterativeShufflingConfig(**yaml_config.get('iterative_shuffling', {}))
    m_fold_sampling_config = MFoldSamplingConfig(**yaml_config.get('m_fold_sampling', {}))
    sequence_voting_config = SequenceVotingConfig(**yaml_config.get('sequence_voting', {}))
    recompile_predict_config = RecompilePredictConfig(**yaml_config.get('recompile_predict', {}))
    pure_sequence_plotting_config = PureSequencePlottingConfig(**yaml_config.get('pure_sequence_plotting', {}))
    
    # Combine into a single config object
    return PipelineConfig(
        general=general_config,
        slurm=slurm_config,
        pipeline_control=pipeline_control_config,
        iterative_shuffling=iterative_shuffling_config,
        m_fold_sampling=m_fold_sampling_config,
        sequence_voting=sequence_voting_config,
        recompile_predict=recompile_predict_config,
        pure_sequence_plotting=pure_sequence_plotting_config
    )