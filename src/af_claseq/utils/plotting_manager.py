# af_claseq/utils/plotting_manager.py

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, hex2color
from typing import List, Dict, Any, Optional, Union, Tuple, Callable

# Import the unified logging utility
from af_claseq.utils.logging_utils import get_logger

# Set publication-quality font defaults
plt.rcParams.update({
    'font.family': ['sans-serif'],
    'font.sans-serif': ['DejaVu Sans'],
    'font.size': 24,
    'axes.labelsize': 24,
    'axes.titlesize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24
})

# Standard plot parameters
PLOT_PARAMS = {
    'scatter_size': 100,
    'scatter_alpha': 0.8,
    'scatter_edge': 'none',
    'dpi': 500,
    'correlation_cmap': ['#72b3d4', '#dadada', '#bc6565']
}

# Color schemes
COLORS = {
    'default_prediction': '#468F8B',
    'default_control': '#AFD2D0',
    'correlation_cmap': ['#72b3d4', '#dadada', '#bc6565']
}
def load_results_df(
    results_dir: Union[str, List[str]],
    metric_names: List[str],
    csv_dir: Optional[str] = None,
    config_file: Optional[str] = None,
    plddt_threshold: float = 0,
    logger: Optional[Any] = None
) -> pd.DataFrame:
    """
    Load results dataframe from CSV or calculate metrics if needed.
    
    Args:
        results_dir: Directory or list of directories containing structure results
        metric_names: Names of metrics to include
        csv_dir: Directory for CSV files (optional)
        config_file: Path to configuration JSON file (optional, required if no CSV)
        plddt_threshold: Minimum pLDDT value for filtering structures
        logger: Optional logger to use
        
    Returns:
        DataFrame with metric values
    """
    # Use provided logger or create one
    log = logger or get_logger(__name__)
    
    # Create CSV directory if needed and if path provided
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
        csv_filename = f"{'_'.join(metric_names)}_values.csv"
        csv_path = os.path.join(csv_dir, csv_filename)
        
        # Check if CSV already exists
        if os.path.exists(csv_path):
            log.info(f"Loading existing results from {csv_path}")
            results_df = pd.read_csv(csv_path)
            
            # Filter by pLDDT threshold if specified
            if plddt_threshold > 0:
                results_df = results_df[results_df['local_plddt'] > plddt_threshold]
                log.info(f"Filtered to {len(results_df)} structures with pLDDT > {plddt_threshold}")
                
            return results_df
    
    # If no CSV exists, we need to calculate the metrics
    if not config_file:
        raise ValueError("Either existing CSV or config_file must be provided")
    
    # Load config
    log.info(f"Calculating metrics for {', '.join(metric_names)}")
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Handle multiple results directories
    if isinstance(results_dir, list):
        log.info(f"Processing results from {len(results_dir)} directories")
        all_results = []
        
        for dir_index, directory in enumerate(results_dir):
            log.info(f"Processing directory {dir_index+1}/{len(results_dir)}: {directory}")
            # Process each directory and add a round identifier
            dir_results = calculate_metric_values(
                parent_dir=directory,
                config=config,
                metric_names=metric_names,
                logger=log
            )
            
            # Add a column to identify which round this data came from
            round_num = os.path.basename(directory).replace("round_", "")
            dir_results['round'] = round_num
            all_results.append(dir_results)
            
        # Combine all results into a single DataFrame
        results_df = pd.concat(all_results, ignore_index=True)
        log.info(f"Combined {len(results_df)} results from all directories")
    else:
        # Process a single directory (original behavior)
        results_df = calculate_metric_values(
            parent_dir=results_dir,
            config=config,
            metric_names=metric_names,
            logger=log
        )
    
    # Save to CSV if path provided
    if csv_dir:
        results_df.to_csv(csv_path, index=False)
        log.info(f"Saved results to {csv_path}")
    
    # Filter by pLDDT threshold if specified
    if plddt_threshold > 0:
        results_df = results_df[results_df['local_plddt'] > plddt_threshold]
        log.info(f"Filtered to {len(results_df)} structures with pLDDT > {plddt_threshold}")
    
    return results_df
def calculate_metric_values(
    parent_dir: str,
    config: Dict[str, Any],
    metric_names: Optional[List[str]] = None,
    logger: Optional[Any] = None
) -> pd.DataFrame:
    """
    Calculate metric values for all PDB files in a directory.
    
    Args:
        parent_dir: Root directory containing PDB files to analyze
        config: Configuration dictionary with filter criteria
        metric_names: Optional list of specific metrics to calculate (defaults to all)
        logger: Optional logger to use
        
    Returns:
        DataFrame with calculated metrics
    """
    log = logger or get_logger(__name__)
    
    # Lazy import to avoid circular imports
    from af_claseq.utils.structure_analysis import StructureAnalyzer
    
    # Extract filter criteria for requested metrics
    filter_criteria = []
    all_criteria = config.get('filter_criteria', [])
    
    if metric_names:
        # Only include requested metrics
        for criterion in all_criteria:
            if criterion.get('name') in metric_names:
                filter_criteria.append(criterion)
    else:
        # Include all metrics
        filter_criteria = all_criteria
    
    if not filter_criteria:
        log.warning(f"No filter criteria found for metrics: {metric_names}")
        return pd.DataFrame()
    
    basics = config.get('basics', {})
    
    # Get results using StructureAnalyzer
    analyzer = StructureAnalyzer()
    results_df = analyzer.get_result_df(
        parent_dir=parent_dir,
        filter_criteria=filter_criteria,
        basics=basics
    )
    
    # Extract and combine data
    data = {
        'PDB': results_df['PDB'],
        'plddt': results_df['plddt'].dropna()
    }
    
    # Add local_plddt if available
    if 'local_plddt' in results_df.columns:
        data['local_plddt'] = results_df['local_plddt'].dropna()
    
    # Add metrics for requested criteria
    for criterion in filter_criteria:
        metric_name = criterion.get('name')
        if metric_name in results_df.columns:
            data[metric_name] = results_df[metric_name].dropna()
        else:
            log.warning(f"Metric '{metric_name}' not found in results")
    
    # Add sequence counts if available
    if 'seq_count' in results_df.columns:
        data['seq_count'] = results_df['seq_count']
    
    return pd.DataFrame(data)

def set_axis_limits_and_ticks(
    plt_obj,
    x_min: Optional[float] = None, 
    x_max: Optional[float] = None, 
    y_min: Optional[float] = None, 
    y_max: Optional[float] = None, 
    x_ticks: Optional[List[float]] = None, 
    y_ticks: Optional[List[float]] = None
) -> None:
    """
    Set axis limits and ticks for plots.
    
    Args:
        plt_obj: Matplotlib plotting object
        x_min: Minimum x-axis value
        x_max: Maximum x-axis value
        y_min: Minimum y-axis value
        y_max: Maximum y-axis value
        x_ticks: Custom x-axis tick values
        y_ticks: Custom y-axis tick values
    """
    # Set x-axis limits
    if x_min is not None and x_max is not None:
        plt_obj.xlim(x_min, x_max)
        if x_ticks is not None:
            plt_obj.xticks(x_ticks)
        else:
            plt_obj.xticks(np.arange(x_min, x_max + 1, 5))
        # Move x-axis ticks away from axis
        plt_obj.tick_params(axis='x', which='major', pad=10)
            
    # Set y-axis limits
    if y_min is not None and y_max is not None:
        plt_obj.ylim(y_min, y_max)
        if y_ticks is not None:
            plt_obj.yticks(y_ticks)
        else:
            plt_obj.yticks(np.arange(y_min, y_max + 1, 5))
        # Move y-axis ticks away from axis    
        plt_obj.tick_params(axis='y', which='major', pad=10)

def plot_1d_distribution(
    results_df: pd.DataFrame,
    metric_name: str,
    output_dir: str,
    initial_color: Union[str, Tuple[float, float, float]] = '#87CEEB',
    end_color: Union[str, Tuple[float, float, float]] = '#FFFFFF',
    n_plot_bins: int = 50,
    log_scale: bool = False,
    gradient_ascending: bool = False,
    linear_gradient: bool = False,
    show_bin_lines: bool = False,
    figsize: Tuple[float, float] = (10, 5),
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    x_ticks: Optional[List[float]] = None,
    logger: Optional[Any] = None
) -> str:
    """
    Create 1D distribution plot for a specific metric.
    
    Args:
        results_df: DataFrame containing metric values
        metric_name: Name of the metric to plot
        output_dir: Directory to save plot output
        initial_color: Initial color for gradient (hex or RGB)
        end_color: End color for gradient (hex or RGB)
        n_plot_bins: Number of bins for histogram
        log_scale: Whether to use log scale for y-axis
        gradient_ascending: Whether to use ascending gradient
        linear_gradient: Whether to use linear gradient
        show_bin_lines: Whether to show bin lines
        figsize: Figure size in inches
        x_min: Minimum value for metric (x-axis)
        x_max: Maximum value for metric (x-axis)
        y_min: Minimum value for count (y-axis)
        y_max: Maximum value for count (y-axis)
        x_ticks: Custom tick values for metric (x-axis)
        logger: Optional logger to use
        
    Returns:
        Path to saved plot
    """
    log = logger or get_logger(__name__)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process colors
    if isinstance(initial_color, str):
        initial_color = hex2color(initial_color)
    if isinstance(end_color, str):
        end_color = hex2color(end_color)
    
    # Skip if metric is not in results
    if metric_name not in results_df.columns:
        log.warning(f"Metric '{metric_name}' not found in results, skipping plot.")
        return ""
    
    log.info(f"Generating 1D distribution plot for {metric_name}")
    
    # Create plot
    plt.figure(figsize=figsize, dpi=PLOT_PARAMS['dpi'])
    
    # Determine range for the metric
    metric_x_min = results_df[metric_name].min() if x_min is None else x_min
    metric_x_max = results_df[metric_name].max() if x_max is None else x_max
    
    bins = np.linspace(metric_x_min, metric_x_max, n_plot_bins + 1)

    # Create histogram data without plotting
    counts, bin_edges, _ = plt.hist(results_df[metric_name], bins=bins.tolist())
    plt.clf()  # Clear the figure

    # Create color gradient
    colors = []
    for i in range(len(bin_edges)-1):
        if linear_gradient:
            ratio = i / (len(bin_edges)-2) if gradient_ascending else 1 - i / (len(bin_edges)-2)
        else:
            ratio = 1 - i / (len(bin_edges)-2) if gradient_ascending else i / (len(bin_edges)-2)
            
        if gradient_ascending:
            r = end_color[0] + (initial_color[0] - end_color[0]) * ratio
            g = end_color[1] + (initial_color[1] - end_color[1]) * ratio 
            b = end_color[2] + (initial_color[2] - end_color[2]) * ratio
        else:
            r = initial_color[0] + (end_color[0] - initial_color[0]) * ratio
            g = initial_color[1] + (end_color[1] - initial_color[1]) * ratio
            b = initial_color[2] + (end_color[2] - initial_color[2]) * ratio
            
        colors.append((r, g, b))

    # Plot each bar with its own color
    plt.bar(
        bin_edges[:-1], 
        counts, 
        width=np.diff(bin_edges), 
        align='edge', 
        color=colors, 
        edgecolor=None
    )

    # Add vertical dashed lines at bin boundaries if requested
    if show_bin_lines:
        for bin_edge in bin_edges:
            plt.axvline(x=bin_edge, color='gray', linestyle='--', alpha=0.5)
    
    plt.xlabel(f'{metric_name}')
    plt.ylabel('Count')
    
    if log_scale:
        plt.yscale('log')
    
    # Set axis limits and ticks
    set_axis_limits_and_ticks(
        plt, 
        x_min=x_min, 
        x_max=x_max, 
        y_min=y_min, 
        y_max=y_max, 
        x_ticks=x_ticks
    )

    # Save plot
    plot_path = os.path.join(output_dir, f'{metric_name}_1d_distribution.png')
    plt.savefig(plot_path, dpi=PLOT_PARAMS['dpi'], bbox_inches='tight')
    plt.close()

    log.info(f"Saved 1D distribution plot to: {plot_path}")
    log.info(f"Mean {metric_name}: {results_df[metric_name].mean():.2f}")
    log.info(f"Median {metric_name}: {results_df[metric_name].median():.2f}")
    
    return plot_path

def create_2d_scatter_plot(
    results_df: pd.DataFrame, 
    metric_name1: str, 
    metric_name2: str, 
    output_dir: str,
    color_metric: str = 'plddt', 
    cmap_colors: List[str] = None,
    x_min: Optional[float] = None, 
    x_max: Optional[float] = None, 
    y_min: Optional[float] = None, 
    y_max: Optional[float] = None, 
    x_ticks: Optional[List[float]] = None, 
    y_ticks: Optional[List[float]] = None,
    title: Optional[str] = None,
    logger: Optional[Any] = None
) -> str:
    """
    Create a 2D scatter plot with the specified parameters.
    
    Args:
        results_df: DataFrame containing results
        metric_name1: Name of metric for x-axis
        metric_name2: Name of metric for y-axis
        output_dir: Directory to save output plot
        color_metric: Metric to use for point coloring (default: 'plddt')
        cmap_colors: List of colors for colormap (default: None, uses standard colors)
        x_min: Minimum x-axis value
        x_max: Maximum x-axis value
        y_min: Minimum y-axis value
        y_max: Maximum y-axis value
        x_ticks: Custom x-axis tick values
        y_ticks: Custom y-axis tick values
        title: Optional title for the plot
        logger: Optional logger to use
        
    Returns:
        Path to saved plot
    """
    log = logger or get_logger(__name__)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Use provided colormap or default
    if cmap_colors is None:
        cmap_colors = PLOT_PARAMS['correlation_cmap']
    
    custom_cmap = LinearSegmentedColormap.from_list('custom', cmap_colors)
    
    plt.figure(figsize=(8, 7))
    scatter = plt.scatter(
        results_df[metric_name1],
        results_df[metric_name2], 
        c=results_df[color_metric],
        cmap=custom_cmap,
        s=PLOT_PARAMS['scatter_size'],
        alpha=PLOT_PARAMS['scatter_alpha'],
        vmin=50,
        vmax=100,
        edgecolors=None,
        linewidths=0
    )

    plt.colorbar(scatter, label=f'{color_metric.replace("plddt", "pLDDT")} Score')
    plt.xlabel(metric_name1)
    plt.ylabel(metric_name2)
    
    if title:
        plt.title(title)

    set_axis_limits_and_ticks(
        plt, 
        x_min=x_min, 
        x_max=x_max, 
        y_min=y_min, 
        y_max=y_max, 
        x_ticks=x_ticks, 
        y_ticks=y_ticks
    )

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{metric_name1}_{metric_name2}_scatter_{color_metric}.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=PLOT_PARAMS['dpi'])
    plt.close()
    
    log.info(f"Saved 2D scatter plot to: {plot_path}")
    
    return plot_path

def create_joint_plot(
    results_df: pd.DataFrame, 
    metric_name1: str, 
    metric_name2: str, 
    output_dir: str,
    color_metric: str = 'plddt', 
    cmap_colors: List[str] = None,
    x_min: Optional[float] = None, 
    x_max: Optional[float] = None, 
    y_min: Optional[float] = None, 
    y_max: Optional[float] = None,
    logger: Optional[Any] = None
) -> str:
    """
    Create a joint plot with scatter plot and marginal distributions.
    
    Args:
        results_df: DataFrame containing results
        metric_name1: Name of metric for x-axis
        metric_name2: Name of metric for y-axis
        output_dir: Directory to save output plot
        color_metric: Metric to use for point coloring (default: 'plddt')
        cmap_colors: List of colors for colormap (default: None, uses standard colors)
        x_min: Minimum x-axis value
        x_max: Maximum x-axis value
        y_min: Minimum y-axis value
        y_max: Maximum y-axis value
        logger: Optional logger to use
        
    Returns:
        Path to saved plot
    """
    log = logger or get_logger(__name__)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Use provided colormap or default
    if cmap_colors is None:
        cmap_colors = PLOT_PARAMS['correlation_cmap']
    
    custom_cmap = LinearSegmentedColormap.from_list('custom', cmap_colors)
    
    g = sns.JointGrid(
        data=results_df, 
        x=metric_name1,
        y=metric_name2,
        height=10
    )

    g.plot_joint(
        plt.scatter,
        c=results_df[color_metric],
        cmap=custom_cmap,
        s=PLOT_PARAMS['scatter_size'],
        alpha=PLOT_PARAMS['scatter_alpha'],
        vmin=50,
        vmax=100
    )

    g.plot_marginals(
        sns.kdeplot,
        color='gray',
        alpha=0.5,
        log_scale=False
    )

    cax = g.figure.add_axes((.95, .4, .02, .2))
    plt.colorbar(g.ax_joint.collections[0], cax=cax, label=f'{color_metric.replace("plddt", "pLDDT")} Score')

    g.ax_joint.set_xlabel(metric_name1)
    g.ax_joint.set_ylabel(metric_name2)

    if x_min is not None and x_max is not None:
        g.ax_joint.set_xlim(x_min, x_max)
    if y_min is not None and y_max is not None:
        g.ax_joint.set_ylim(y_min, y_max)

    plot_path = os.path.join(output_dir, f'{metric_name1}_{metric_name2}_joint_{color_metric}.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=PLOT_PARAMS['dpi'])
    plt.close()
    
    log.info(f"Saved joint plot to: {plot_path}")
    
    return plot_path
def plot_m_fold_sampling_1d(
    results_dir: Union[str, List[str]],
    metric_name: str,
    output_dir: str,
    csv_dir: Optional[str] = None,
    config_file: Optional[str] = None,
    initial_color: Union[str, Tuple[float, float, float]] = '#87CEEB',
    end_color: Union[str, Tuple[float, float, float]] = '#FFFFFF',
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    x_ticks: Optional[List[float]] = None,
    log_scale: bool = False,
    n_plot_bins: int = 50,
    gradient_ascending: bool = False,
    linear_gradient: bool = False,
    plddt_threshold: float = 0,
    figsize: Tuple[float, float] = (10, 5),
    show_bin_lines: bool = False,
    logger: Optional[Any] = None
) -> List[str]:
    """
    Plot 1D sampling distribution for M-fold sampling metrics.
    
    Args:
        results_dir: Directory or list of directories containing M-fold sampling results
        metric_name: Name of the metric to plot
        output_dir: Directory for output plots
        csv_dir: Directory for CSV files (optional)
        config_file: Path to config JSON file (required if CSV doesn't exist)
        initial_color: Initial color for gradient
        end_color: End color for gradient
        x_min: Minimum value for x-axis
        x_max: Maximum value for x-axis
        y_min: Minimum value for y-axis
        y_max: Maximum value for y-axis
        x_ticks: Custom tick values for x-axis
        log_scale: Use log scale for y-axis
        n_plot_bins: Number of bins for histogram
        gradient_ascending: Use ascending gradient (low to high values)
        linear_gradient: Use linear instead of logarithmic gradient
        plddt_threshold: pLDDT threshold for filtering structures
        figsize: Figure size (width, height)
        show_bin_lines: Show vertical lines at bin boundaries
        logger: Optional logger
        
    Returns:
        List of paths to saved plots
    """
    log = logger or get_logger(__name__)
    
    # Load results DataFrame - now handling multiple directories
    results_df = load_results_df(
        results_dir=results_dir,
        metric_names=[metric_name],
        csv_dir=csv_dir,
        config_file=config_file,
        plddt_threshold=plddt_threshold,
        logger=log
    )
    
    # Generate the plot
    plot_path = plot_1d_distribution(
        results_df=results_df,
        metric_name=metric_name,
        output_dir=output_dir,
        initial_color=initial_color,
        end_color=end_color,
        n_plot_bins=n_plot_bins,
        log_scale=log_scale,
        gradient_ascending=gradient_ascending,
        linear_gradient=linear_gradient,
        show_bin_lines=show_bin_lines,
        figsize=figsize,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        x_ticks=x_ticks,
        logger=log
    )
    
    return [plot_path]

def plot_m_fold_sampling_2d(
    results_dir: Union[str, List[str]],
    metric_name1: str,
    metric_name2: str,
    output_dir: str,
    csv_dir: Optional[str] = None,
    config_file: Optional[str] = None,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    x_ticks: Optional[List[float]] = None,
    y_ticks: Optional[List[float]] = None,
    plddt_threshold: float = 0,
    include_joint_plot: bool = True,
    logger: Optional[Any] = None
) -> List[str]:
    """
    Plot 2D sampling distribution for M-fold sampling metrics.
    
    This function is a convenience wrapper around the more general plotting 
    functions, specifically for M-fold sampling analysis.

    Args:
        results_dir: Directory or list of directories containing M-fold sampling results
        metric_name1: Name of the first metric (x-axis)
        metric_name2: Name of the second metric (y-axis)
        output_dir: Path to directory for saving plot outputs
        csv_dir: Path to directory for saving/loading CSV results (optional)
        config_file: Path to config JSON file (required if CSV doesn't exist)
        x_min: Minimum value for first metric (x-axis)
        x_max: Maximum value for first metric (x-axis)
        y_min: Minimum value for second metric (y-axis)
        y_max: Maximum value for second metric (y-axis)
        x_ticks: Custom tick values for first metric (x-axis)
        y_ticks: Custom tick values for second metric (y-axis)
        plddt_threshold: pLDDT threshold for filtering structures
        include_joint_plot: Whether to create joint plots with marginal distributions
        logger: Optional logger to use
        
    Returns:
        List of paths to saved plots
    """
    log = logger or get_logger(__name__)
    
    # Load results DataFrame - now handling multiple directories
    results_df = load_results_df(
        results_dir=results_dir,
        metric_names=[metric_name1, metric_name2],
        csv_dir=csv_dir,
        config_file=config_file,
        plddt_threshold=plddt_threshold,
        logger=log
    )
    
    # Sort by pLDDT to have points with higher pLDDT on top
    results_df = results_df.sort_values('plddt', ascending=True)
    
    plot_paths = []
    
    # Create scatter plots for pLDDT coloring
    plot_paths.append(create_2d_scatter_plot(
        results_df=results_df,
        metric_name1=metric_name1,
        metric_name2=metric_name2,
        output_dir=output_dir,
        color_metric='plddt',
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        x_ticks=x_ticks,
        y_ticks=y_ticks,
        logger=log
    ))
    
    # Create scatter plots for local_pLDDT coloring if available
    if 'local_plddt' in results_df.columns:
        plot_paths.append(create_2d_scatter_plot(
            results_df=results_df,
            metric_name1=metric_name1,
            metric_name2=metric_name2,
            output_dir=output_dir,
            color_metric='local_plddt',
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            logger=log
        ))
    # Create joint plots if requested
    if include_joint_plot:
        joint_plot_path = create_joint_plot(
            results_df=results_df,
            metric_name1=metric_name1,
            metric_name2=metric_name2,
            output_dir=output_dir,
            color_metric='plddt',
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            logger=log
        )
        plot_paths.append(joint_plot_path)
        
        if 'local_plddt' in results_df.columns:
            local_joint_plot_path = create_joint_plot(
                results_df=results_df,
                metric_name1=metric_name1,
                metric_name2=metric_name2,
                output_dir=output_dir,
                color_metric='local_plddt',
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                logger=log
            )
            plot_paths.append(local_joint_plot_path)
    
    return plot_paths