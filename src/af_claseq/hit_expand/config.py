"""
Configuration module for hit expand pipeline.

This module provides configuration dataclasses for the hit expand
functionality that integrates MSA pipeline capabilities into AF-ClaSeq.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union, Tuple


@dataclass
class HitExpandConfig:
    """Configuration for hit expand pipeline step."""
    
    # Input MSA file (source A3M)
    input_msa: str
    
    # MMseqs2 clustering configuration
    mmseqs_bin: str = "/fs/ess/PAA0203/xing244/packages/mmseqs/bin/mmseqs"
    mmseqs_coverage: float = 0.8
    mmseqs_min_seq_id: float = 0.7
    mmseqs_cov_mode: int = 0
    mmseqs_cluster_mode: int = 0
    mmseqs_threads: int = 8
    mmseqs_tmp_dir: str = "/tmp"
    
    # MSA pipeline configuration
    msa_pipeline_config: Optional[str] = None
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
        if self.plddt_threshold < 0 or self.plddt_threshold > 100:
            raise ValueError("plddt_threshold must be between 0 and 100")


@dataclass
class HitExpandPlottingConfig:
    """Configuration for hit expand plotting and visualization."""
    
    # Plot dimensions and styling
    figsize: Tuple[int, int] = (15, 7)
    dpi: int = 300
    
    # Color configuration
    initial_color: str = "#87CEEB"
    end_color: str = "#FFFFFF"
    
    # Axis configuration
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    x_ticks: Optional[List[float]] = None
    y_ticks: Optional[List[float]] = None
    
    # Analysis thresholds
    plddt_threshold: float = 75.0
    filter_criteria_threshold: float = 0.8
    
    # Plot types to generate
    generate_quality_plots: bool = True
    generate_scatter_plots: bool = True
    generate_distribution_plots: bool = True
    generate_summary_plots: bool = True
    
    # Performance settings
    max_workers: int = 8