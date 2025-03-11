import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from af_claseq.structure_analysis import get_result_df, load_filter_modes
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import List, Dict, Any, Optional, Union, Tuple

# Set publication-quality font
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['font.size'] = 24
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24

# Plot appearance constants
PLOT_PARAMS = {
    'scatter_size': 100,
    'scatter_alpha': 0.8,
    'scatter_edge': 'none',
    'dpi': 500,
    'prediction_figsize': (15, 7),
    'selection_figsize': (14, 8)
}

# Color schemes
COLORS = {
    'default_selection': '#468F8B',
    'default_control': '#AFD2D0',
    'correlation_cmap': ['#72b3d4', '#dadada', '#bc6565']
    # 'correlation_cmap': ['#e77260', '#dadada', '#f9ba51']
}


def calculate_metric_values(
    parent_dir: str,
    config: Dict[str, Any],
    iteration_dirs: Optional[List[str]] = None,
    csv_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Calculate metric values for all PDB files in a directory structure and save to CSV.
    
    Args:
        parent_dir: Root directory containing PDB files to analyze
        config: Configuration dictionary containing filter criteria and settings
        iteration_dirs: Optional list of iteration subdirectories to process
        csv_dir: Optional output directory path for saving CSV results
        
    Returns:
        DataFrame containing calculated metrics for all processed structures
    """
    all_data = []
    
    # Determine directories to process
    dirs_to_process = [parent_dir] if iteration_dirs is None else [
        os.path.join(parent_dir, d) for d in iteration_dirs
    ]

    # Extract filter criteria from config
    filter_criteria = [config['filter_criteria'][0]]
    if len(config['filter_criteria']) > 1:
        filter_criteria.append(config['filter_criteria'][1])
    basics = config['basics']
    
    # Get metric names from filter criteria
    metric_names = [criterion['name'] for criterion in filter_criteria]
    
    # Process each directory
    for dir_path in dirs_to_process:
        # Get results dataframe
        results_df = get_result_df(
            parent_dir=dir_path,
            filter_criteria=filter_criteria,
            basics=basics
        )
        
        # Extract data from results
        data = {
            'PDB': results_df['PDB'],
            'plddt': results_df['plddt'].dropna()
        }
        
        # Add local_plddt if available
        if 'local_plddt' in results_df.columns:
            data['local_plddt'] = results_df['local_plddt'].dropna()
        
        # Add metrics for all criteria
        for metric_name in metric_names:
            metric_values = results_df[metric_name].dropna()
            if not metric_values.empty:
                data[metric_name] = metric_values
            
        # Add iteration info if processing multiple iterations
        if iteration_dirs is not None:
            data['iteration'] = pd.Series([os.path.basename(dir_path).split('_')[1]] * len(data['PDB']))
        
        # Add sequence counts if available
        if 'seq_count' in results_df.columns:
            data['seq_count'] = results_df['seq_count']
            
        iteration_data = pd.DataFrame(data)
        all_data.append(iteration_data)
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Save to CSV if path provided
    if csv_dir is not None:
        os.makedirs(csv_dir, exist_ok=True)
        csv_filename = f"{metric_names[0]}_{metric_names[1] if len(metric_names) > 1 else metric_names[0]}_values.csv"
        csv_path = os.path.join(csv_dir, csv_filename)
        combined_df.to_csv(csv_path, index=False)
    
    return combined_df


def _load_or_calculate_metrics(results_dir: str, config_file: str, csv_dir: str) -> pd.DataFrame:
    """
    Load metrics from CSV if available, otherwise calculate them.
    
    Args:
        results_dir: Directory containing structure results
        config_file: Path to configuration JSON file
        csv_dir: Directory for CSV files
        
    Returns:
        DataFrame with metric values
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Get metric names from config
    metric_names = [criterion['name'] for criterion in config['filter_criteria']]
    metric_name1 = metric_names[0]
    metric_name2 = metric_names[1] if len(metric_names) > 1 else metric_name1
    
    # Check if CSV already exists
    csv_path = os.path.join(csv_dir, f'{metric_name1}_{metric_name2}_values.csv')
    if os.path.exists(csv_path):
        print(f"Loading existing results from {csv_path}")
        return pd.read_csv(csv_path)
    else:
        # Calculate metrics
        return calculate_metric_values(
            parent_dir=results_dir,
            config=config,
            csv_dir=csv_dir
        )


def _load_data_for_2d_plot(
    results_dir: str, 
    config_file: str, 
    csv_dir: str, 
    plddt_threshold: float
) -> Tuple[pd.DataFrame, Tuple[str, str]]:
    """
    Load data for 2D plots and return dataframe and metric names.
    
    Args:
        results_dir: Directory containing structure results
        config_file: Path to configuration JSON file
        csv_dir: Directory for CSV files
        plddt_threshold: Minimum pLDDT value to include structures
        
    Returns:
        Tuple of (DataFrame with filtered results, tuple of metric names)
    """
    # Load config
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Get metric names from config
    metric_name1 = config['filter_criteria'][0]['name']
    metric_name2 = config['filter_criteria'][1]['name'] 
    
    # Check if CSV already exists
    csv_path = os.path.join(csv_dir, f'{metric_name1}_{metric_name2}_values.csv')
    if os.path.exists(csv_path):
        print(f"Loading existing results from {csv_path}")
        results_df = pd.read_csv(csv_path)
    else:
        # Calculate metrics
        results_df = calculate_metric_values(
            parent_dir=results_dir,
            config=config,
            csv_dir=csv_dir
        )

    # Filter by pLDDT threshold if specified and sort by pLDDT
    results_df = results_df.sort_values('plddt', ascending=True)
    if plddt_threshold > 0:
        results_df = results_df[results_df['plddt'] > plddt_threshold]
        
    return results_df, (metric_name1, metric_name2)


def _set_axis_limits_and_ticks(
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
    # Set x-axis limits explicitly
    if x_min is not None and x_max is not None:
        plt_obj.xlim(x_min, x_max)
        if x_ticks is not None:
            plt_obj.xticks(x_ticks)
        else:
            plt_obj.xticks(np.arange(x_min, x_max + 1, 5))
        # Move x-axis ticks away from axis
        plt_obj.tick_params(axis='x', which='major', pad=10)
            
    if y_min is not None and y_max is not None:
        plt_obj.ylim(y_min, y_max)
        if y_ticks is not None:
            plt_obj.yticks(y_ticks)
        else:
            plt_obj.yticks(np.arange(y_min, y_max + 1, 5))
        # Move y-axis ticks away from axis    
        plt_obj.tick_params(axis='y', which='major', pad=10)


def _create_2d_scatter_plot(
    results_df: pd.DataFrame, 
    metric_name1: str, 
    metric_name2: str, 
    color_metric: str, 
    custom_cmap: LinearSegmentedColormap, 
    output_dir: str, 
    x_min: Optional[float] = None, 
    x_max: Optional[float] = None, 
    y_min: Optional[float] = None, 
    y_max: Optional[float] = None, 
    x_ticks: Optional[List[float]] = None, 
    y_ticks: Optional[List[float]] = None
) -> None:
    """
    Create a 2D scatter plot with the specified parameters.
    
    Args:
        results_df: DataFrame containing results
        metric_name1: Name of metric for x-axis
        metric_name2: Name of metric for y-axis
        color_metric: Metric to use for point coloring
        custom_cmap: Colormap for scatter points
        output_dir: Directory to save output plot
        x_min: Minimum x-axis value
        x_max: Maximum x-axis value
        y_min: Minimum y-axis value
        y_max: Maximum y-axis value
        x_ticks: Custom x-axis tick values
        y_ticks: Custom y-axis tick values
    """
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

    _set_axis_limits_and_ticks(plt, x_min, x_max, y_min, y_max, x_ticks, y_ticks)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{metric_name1}_{metric_name2}_scatter_{color_metric}.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=PLOT_PARAMS['dpi'])
    plt.close()


def _create_joint_plot(
    results_df: pd.DataFrame, 
    metric_name1: str, 
    metric_name2: str, 
    color_metric: str, 
    custom_cmap: LinearSegmentedColormap, 
    output_dir: str, 
    x_min: Optional[float] = None, 
    x_max: Optional[float] = None, 
    y_min: Optional[float] = None, 
    y_max: Optional[float] = None
) -> None:
    """
    Create a joint plot with the specified parameters.
    
    Args:
        results_df: DataFrame containing results
        metric_name1: Name of metric for x-axis
        metric_name2: Name of metric for y-axis
        color_metric: Metric to use for point coloring
        custom_cmap: Colormap for scatter points
        output_dir: Directory to save output plot
        x_min: Minimum x-axis value
        x_max: Maximum x-axis value
        y_min: Minimum y-axis value
        y_max: Maximum y-axis value
    """
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


def plot_m_fold_sampling_1d(
    results_dir: str,
    config_file: str,
    output_dir: str,
    csv_dir: str,
    gradient_ascending: bool,
    initial_color: Tuple[float, float, float] = (135/255, 206/255, 235/255),
    end_color: Tuple[float, float, float] = (1, 1, 1),
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    log_scale: bool = False,
    n_plot_bins: int = 50,
    linear_gradient: bool = False,
    plddt_threshold: float = 0,
    figsize: Tuple[int, int] = (10, 5),
    show_bin_lines: bool = False,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    x_ticks: Optional[List[float]] = None
) -> None:
    """
    Generate 1D distribution plot for metric values.
    
    Args:
        results_dir: Path to directory containing results
        config_file: Path to config JSON file
        output_dir: Path to directory for saving plot outputs
        csv_dir: Path to directory for saving CSV results
        gradient_ascending: If True, gradient goes from end_color to initial_color
        initial_color: RGB values for initial color in gradient (default: skyblue)
        end_color: RGB values for end color in gradient (default: white)
        x_min: Minimum x-axis value
        x_max: Maximum x-axis value
        log_scale: Whether to use log scale for y-axis
        n_plot_bins: Number of bins for histogram
        linear_gradient: If True, use linear color transition
        plddt_threshold: pLDDT threshold for filtering structures
        figsize: Figure size in inches (width, height)
        show_bin_lines: Whether to show vertical dashed lines at bin boundaries
        y_min: Minimum y-axis value
        y_max: Maximum y-axis value
        x_ticks: List of x-axis tick values
    """
    # Load data from CSV or calculate metrics
    results_df = _load_or_calculate_metrics(results_dir, config_file, csv_dir)
    
    # Get metric names from config
    with open(config_file, 'r') as f:
        config = json.load(f)
    filter_criteria = config['filter_criteria']
    
    # Filter by pLDDT threshold if specified
    if plddt_threshold > 0:
        results_df = results_df[results_df['plddt'] > plddt_threshold]
    
    # Process each metric in the filter criteria
    for criterion in filter_criteria:
        metric_name = criterion['name']
        
        # Skip if metric is not in results
        if metric_name not in results_df.columns:
            print(f"Warning: Metric '{metric_name}' not found in results, skipping.")
            continue
            
        # Create plot
        plt.figure(figsize=figsize, dpi=PLOT_PARAMS['dpi'])
        
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
        _set_axis_limits_and_ticks(plt, x_min, x_max, y_min, y_max, x_ticks)

        # Save plot
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, f'{metric_name}_1d_distribution.png')
        plt.savefig(plot_path, dpi=PLOT_PARAMS['dpi'], bbox_inches='tight')
        plt.close()

        print(f"Generated plot for {metric_name} saved to: {plot_path}")
        print(f"Total structures analyzed{f' (pLDDT > {plddt_threshold})' if plddt_threshold > 0 else ''}: {len(results_df)}")
        print(f"Mean {metric_name}: {results_df[metric_name].mean():.2f}")
        print(f"Median {metric_name}: {results_df[metric_name].median():.2f}")


def plot_m_fold_sampling_2d(
    results_dir: str,
    config_file: str,
    output_dir: str,
    csv_dir: str,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    plddt_threshold: float = 0,
    x_ticks: Optional[List[float]] = None,
    y_ticks: Optional[List[float]] = None
) -> None:
    """
    Plot 2D sampling distribution of two metrics from M-fold sampling results.

    Args:
        results_dir: Path to directory containing M-fold sampling results
        config_file: Path to config JSON file specifying metrics to analyze
        output_dir: Path to directory for saving plot outputs
        csv_dir: Path to directory for saving/loading CSV results
        x_min: Minimum x-axis value
        x_max: Maximum x-axis value
        y_min: Minimum y-axis value
        y_max: Maximum y-axis value
        plddt_threshold: pLDDT threshold for filtering structures
        x_ticks: List of x-axis tick values
        y_ticks: List of y-axis tick values
        
    The function creates two 2D scatter plots showing the distribution of two metrics 
    specified in the config file. Each point represents a structure from M-fold sampling, 
    one colored by pLDDT score and one by local pLDDT.
    """
    # Load data and get metric names
    results_df, metric_names = _load_data_for_2d_plot(results_dir, config_file, csv_dir, plddt_threshold)
    metric_name1, metric_name2 = metric_names

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create custom colormap from initial to end color
    custom_cmap = LinearSegmentedColormap.from_list('custom', COLORS['correlation_cmap'])
    
    # Plot with pLDDT coloring
    _create_2d_scatter_plot(
        results_df, metric_name1, metric_name2, 'plddt', 
        custom_cmap, output_dir, x_min, x_max, y_min, y_max, 
        x_ticks, y_ticks
    )

    # Only plot local pLDDT if it exists in the dataframe
    if 'local_plddt' in results_df.columns:
        _create_2d_scatter_plot(
            results_df, metric_name1, metric_name2, 'local_plddt', 
            custom_cmap, output_dir, x_min, x_max, y_min, y_max, 
            x_ticks, y_ticks
        )

    print(f"Generated scatter plots saved to: {output_dir}")
    print(f"Total structures analyzed{f' (pLDDT > {plddt_threshold})' if plddt_threshold > 0 else ''}: {len(results_df)}")


def plot_m_fold_sampling_2d_joint(
    results_dir: str,
    config_file: str,
    output_dir: str,
    csv_dir: str,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    plddt_threshold: float = 0
) -> None:
    """
    Plot 2D joint plots (scatter + histograms) of two metrics from M-fold sampling results.

    Args:
        results_dir: Path to directory containing M-fold sampling results
        config_file: Path to config JSON file specifying metrics to analyze
        output_dir: Path to directory for saving plot outputs
        csv_dir: Path to directory for saving/loading CSV results
        x_min: Minimum x-axis value
        x_max: Maximum x-axis value
        y_min: Minimum y-axis value
        y_max: Maximum y-axis value
        plddt_threshold: pLDDT threshold for filtering structures
        
    The function creates two joint plots using seaborn showing:
    - Center: 2D scatter plot of two metrics colored by either pLDDT or local pLDDT
    - Top and Right: Kernel density estimation plots of marginal distributions
    """
    # Load data and get metric names
    results_df, metric_names = _load_data_for_2d_plot(results_dir, config_file, csv_dir, plddt_threshold)
    metric_name1, metric_name2 = metric_names

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Create custom colormap
    custom_cmap = LinearSegmentedColormap.from_list('custom', COLORS['correlation_cmap'])

    # Plot with pLDDT coloring
    _create_joint_plot(
        results_df, metric_name1, metric_name2, 'plddt', 
        custom_cmap, output_dir, x_min, x_max, y_min, y_max
    )

    # Only plot local pLDDT if it exists in the dataframe
    if 'local_plddt' in results_df.columns:
        _create_joint_plot(
            results_df, metric_name1, metric_name2, 'local_plddt', 
            custom_cmap, output_dir, x_min, x_max, y_min, y_max
        )

    print(f"Generated joint plots saved to: {output_dir}")
    print(f"Total structures analyzed{f' (pLDDT > {plddt_threshold})' if plddt_threshold > 0 else ''}: {len(results_df)}")


def plot_iteration_shuffling_distribution(
    results_df: pd.DataFrame, 
    output_dir: str, 
    initial_color: Tuple[float, float, float], 
    x_min: Optional[float] = None, 
    x_max: Optional[float] = None
) -> None:
    """
    Create distribution plots from metric data for each iteration.
    
    Args:
        results_df: DataFrame containing results with iteration information
        output_dir: Directory to save output plots
        initial_color: RGB color for histogram bars
        x_min: Minimum x-axis value
        x_max: Maximum x-axis value
    """
    iterations = sorted(results_df['iteration'].unique())
    
    n_iterations = len(iterations)
    n_cols = min(3, n_iterations)
    n_rows = (n_iterations + n_cols - 1) // n_cols
    
    # Get metric name from dataframe columns
    metric_name = [col for col in results_df.columns if col not in ['iteration', 'PDB', 'plddt', 'seq_count']][0]
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(-1, 1) if n_cols == 1 else axes.reshape(1, -1)
    
    # Plot each iteration
    for idx, iteration in enumerate(iterations):
        row = idx // n_cols
        col = idx % n_cols
        
        iteration_data = results_df[results_df['iteration'] == iteration][metric_name]
        
        axes[row, col].hist(iteration_data, bins=80, color=initial_color, edgecolor=None, alpha=0.7)
        axes[row, col].set_title(f'Iteration {iteration}')
        axes[row, col].set_xlabel(metric_name)
        if x_min is not None and x_max is not None:
            axes[row, col].set_xlim(x_min, x_max)
        axes[row, col].set_yscale('log')
        axes[row, col].set_ylabel('Counts of predicted structures')
    
    # Remove empty subplots if any
    for idx in range(len(iterations), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row, col])
    
    # Adjust layout and save plot
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{metric_name}_distribution_by_iteration.png')
    plt.savefig(output_path, dpi=PLOT_PARAMS['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f"Distribution plots by iteration saved to {output_path}")


def plot_scatter(
    results_dir: str,
    config_file: str,
    output_dir: str,
    csv_dir: str,
    initial_color: Tuple[float, float, float] = (135/255, 206/255, 235/255),
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None
) -> None:
    """
    Generate 2D scatter plot of metric vs pLDDT scores.
    
    Args:
        results_dir: Path to directory containing results
        config_file: Path to config JSON file
        output_dir: Path to directory for saving plot outputs
        csv_dir: Path to directory for saving CSV results
        initial_color: RGB color for scatter points
        x_min: Minimum x-axis value
        x_max: Maximum x-axis value
        y_min: Minimum y-axis value
        y_max: Maximum y-axis value
    """
    # Load config
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Get metric name from config
    metric_name = config['filter_criteria'][0]['name']
    
    # Get results dataframe
    results_df = calculate_metric_values(
        parent_dir=results_dir,
        config=config,
        csv_dir=csv_dir
    )

    # Create scatter plot
    plt.figure(figsize=(6, 6), dpi=600)
    
    plt.scatter(results_df[metric_name], results_df['plddt'], 
               c=[initial_color], alpha=0.5)
    
    plt.xlabel(f'{metric_name}')
    plt.ylabel('pLDDT Score')
    plt.title(f'{metric_name} vs pLDDT Scores')
    
    # Set axis limits if provided
    if x_min is not None:
        plt.xlim(left=x_min)
    if x_max is not None:
        plt.xlim(right=x_max)
    if y_min is not None:
        plt.ylim(bottom=y_min)
    if y_max is not None:
        plt.ylim(top=y_max)

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f'{metric_name}_plddt_scatter.png')
    plt.savefig(plot_path, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"Generated scatter plot saved to: {plot_path}")

