import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from af_vote.structure_analysis import get_result_df, load_filter_modes
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

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

def calculate_metric_values(parent_dir: str,
                          iteration_dirs: list = None,
                          config: dict = None,
                          csv_dir: str = None):
    """
    Calculate metric values for all PDB files in a directory structure and save to CSV.
    Can process either a single directory or multiple iteration directories from iterative
    shuffling or M-fold sampling approaches.
    
    Args:
        parent_dir (str): Root directory containing PDB files to analyze
        iteration_dirs (list): Optional list of iteration subdirectories to process
        config (dict): Configuration dictionary containing filter criteria and settings
        csv_dir (str): Optional output directory path for saving CSV results
    """
    all_data = []
    
    # determine if to plot the iterative shuffling dir or the M-Fold sampling dir 
    dirs_to_process = [parent_dir] if iteration_dirs is None else [os.path.join(parent_dir, d) for d in iteration_dirs]

    for dir_path in dirs_to_process:
        # Get filter criteria from either config or filter_modes
        # Get either one or two filter criteria depending on what's in config
        filter_criteria = [config['filter_criteria'][0]]
        if len(config['filter_criteria']) > 1:
            filter_criteria.append(config['filter_criteria'][1])
        basics = config['basics']
        
        # Get results dataframe
        results_df = get_result_df(
            parent_dir=dir_path,
            filter_criteria=filter_criteria,
            basics=basics
        )
        
        # Get metric names from both filter criteria
        metric_names = []
        for criterion in filter_criteria:
            metric_names.append(criterion['name'])
        
        # Get values from results
        pdb_paths = results_df['PDB']
        plddt_values = results_df['plddt'].dropna()
        local_plddt_values = results_df['local_plddt'].dropna() if 'local_plddt' in results_df.columns else None
        seq_counts = results_df['seq_count'] if 'seq_count' in results_df.columns else None
        
        data = {
            'PDB': pdb_paths,
            'plddt': plddt_values
        }
        
        # Add local_plddt if available
        if local_plddt_values is not None:
            data['local_plddt'] = local_plddt_values
        
        # Add metrics for both criteria
        for metric_name in metric_names:
            metric_values = results_df[metric_name].dropna()
            if not metric_values.empty:
                data[metric_name] = metric_values
            
        # Add iteration info if processing multiple iterations
        if iteration_dirs is not None:
            data['iteration'] = os.path.basename(dir_path).split('_')[1]
        
        # Add sequence counts if available
        if seq_counts is not None:
            data['seq_count'] = seq_counts
            
        iteration_data = pd.DataFrame(data)
        all_data.append(iteration_data)
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Save to CSV if path provided
    if csv_dir is not None:
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, f'{metric_names[0]}_{metric_names[1]}_values.csv' if len(metric_names) > 1 else f'{metric_names[0]}_values.csv')
        combined_df.to_csv(csv_path, index=False)
    
    return combined_df

def plot_m_fold_sampling_1d(results_dir: str,
                     config_path: str,
                     output_dir: str,
                     csv_dir: str,
                     gradient_ascending: bool,
                     initial_color: tuple = (135/255, 206/255, 235/255),
                     end_color: tuple = (1, 1, 1),  # Default white
                     x_min: float = None,
                     x_max: float = None,
                     log_scale: bool = False,
                     n_plot_bins: int = 50,
                     linear_gradient: bool = False,
                     plddt_threshold: float = 0,
                     figsize: tuple = (10, 5),
                     show_bin_lines: bool = False,
                     y_min: float = None,
                     y_max: float = None,
                     x_ticks: list = None):
    """
    Generate 1D distribution plot for metric values.
    
    Args:
        results_dir (str): Path to directory containing results
        config_path (str): Path to config JSON file
        metric_name (str): the name of the metric to plot adn analysis, e.g. '2qke_tmscore'
        output_dir (str): Path to directory for saving plot outputs
        csv_dir (str): Path to directory for saving CSV results
        initial_color (tuple): RGB values for initial color in gradient (default: skyblue)
        end_color (tuple): RGB values for end color in gradient (default: white)
        x_min (float): Minimum x-axis value
        x_max (float): Maximum x-axis value
        log_scale (bool): Whether to use log scale for y-axis
        n_plot_bins (int): Number of bins for histogram (default: 50)
        gradient_ascending (bool): If True, gradient goes from end_color to initial_color.
                                 If False, gradient goes from initial_color to end_color.
        linear_gradient (bool): If True, use linear color transition. If False, use exponential transition.
        plddt_threshold (float): pLDDT threshold for filtering structures (default: 0, no filtering)
        figsize (tuple): Figure size in inches (width, height) (default: (10, 5))
        show_bin_lines (bool): Whether to show vertical dashed lines at bin boundaries (default: False)
        x_ticks (list): List of x-axis tick values (default: None)
    """
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Get metric name from config
    metric_name = config['filter_criteria'][0]['name']
    
    # Check if CSV already exists
    csv_path = os.path.join(csv_dir, f'{metric_name}_values.csv')
    if os.path.exists(csv_path):
        print(f"Loading existing results from {csv_path}")
        results_df = pd.read_csv(csv_path)
    else:
        # Get results dataframe
        results_df = calculate_metric_values(
            parent_dir=results_dir,
            config=config,
            csv_dir=csv_dir
        )

    # Filter by pLDDT threshold if specified
    if plddt_threshold > 0:
        results_df = results_df[results_df['plddt'] > plddt_threshold]

    # Create plot
    plt.figure(figsize=figsize, dpi=600)
    
    x_min = results_df[metric_name].min() if x_min is None else x_min
    x_max = results_df[metric_name].max() if x_max is None else x_max
    
    bins = np.linspace(x_min, x_max, n_plot_bins + 1)  # n_bins + 1 edges makes n_bins bins

    # Create histogram data without plotting
    counts, bins, _ = plt.hist(results_df[metric_name], bins=bins)
    plt.clf()  # Clear the figure

    # Create color gradient
    colors = []
    for i in range(len(bins)-1):
        if linear_gradient:
            ratio = i / (len(bins)-2) if gradient_ascending else 1 - i / (len(bins)-2)
        else:
            if gradient_ascending:
                ratio = 1- i / (len(bins)-2)  # Linear ascending gradient
            else:
                ratio = i / (len(bins)-2)  # Linear descending gradient
            
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
    plt.bar(bins[:-1], counts, width=np.diff(bins), align='edge', 
            color=colors, edgecolor=None)

    # Add vertical dashed lines at bin boundaries if show_bin_lines is True
    if show_bin_lines:
        for bin_edge in bins:
            plt.axvline(x=bin_edge, color='gray', linestyle='--', alpha=0.5)
    
    plt.xlabel(f'{metric_name}')
    plt.ylabel('Count')
    
    # # Set title based on pLDDT threshold
    # title = f'Distribution of {metric_name} values'
    # if plddt_threshold > 0:
    #     title += f' (pLDDT > {plddt_threshold})'
    # plt.title(title)
    
    if log_scale:
        plt.yscale('log')
    
    # Set x-axis limits explicitly
    if x_min is not None and x_max is not None:
        plt.xlim(x_min, x_max)
    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)
        
    # Set x-axis ticks if provided
    if x_ticks is not None:
        plt.xticks(x_ticks)

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f'{metric_name}_1d_distribution.png')
    plt.savefig(plot_path, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"Generated plot saved to: {plot_path}")
    print(f"Total structures analyzed{f' (pLDDT > {plddt_threshold})' if plddt_threshold > 0 else ''}: {len(results_df)}")
    print(f"Mean {metric_name}: {results_df[metric_name].mean():.2f}")
    print(f"Median {metric_name}: {results_df[metric_name].median():.2f}")

def plot_m_fold_sampling_2d(results_dir: str,
                     config_path: str,
                     output_dir: str,
                     csv_dir: str,
                     x_min: float = None,
                     x_max: float = None,
                     y_min: float = None,
                     y_max: float = None,
                     plddt_threshold: float = 0,
                     x_ticks: list = None,
                     y_ticks: list = None):

    """Plot 2D sampling distribution of two metrics from M-fold sampling results.

    Args:
        results_dir (str): Path to directory containing M-fold sampling results
        config_path (str): Path to config JSON file specifying metrics to analyze
        output_dir (str): Path to directory for saving plot outputs
        csv_dir (str): Path to directory for saving/loading CSV results
        x_min (float, optional): Minimum x-axis value. Defaults to None.
        x_max (float, optional): Maximum x-axis value. Defaults to None.
        y_min (float, optional): Minimum y-axis value. Defaults to None.
        y_max (float, optional): Maximum y-axis value. Defaults to None.
        plddt_threshold (float, optional): pLDDT threshold for filtering structures. 
            Structures with pLDDT below this value will be excluded. Defaults to 0 (no filtering).
        x_ticks (list, optional): List of x-axis tick values. Defaults to None.
        y_ticks (list, optional): List of y-axis tick values. Defaults to None.
    The function creates two 2D scatter plots showing the distribution of two metrics specified in the config file.
    Each point represents a structure from M-fold sampling, one colored by pLDDT score and one by local pLDDT.
    Results are cached in a CSV file for faster replotting.
    """

    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Get metric name from config
    metric_name1 = config['filter_criteria'][0]['name']
    metric_name2 = config['filter_criteria'][1]['name'] 
    
    # Check if CSV already exists
    csv_path = os.path.join(csv_dir, f'{metric_name1}_{metric_name2}_values.csv')
    if os.path.exists(csv_path):
        print(f"Loading existing results from {csv_path}")
        results_df = pd.read_csv(csv_path)
    else:
        # Get results dataframe
        results_df = calculate_metric_values(
            parent_dir=results_dir,
            config=config,
            csv_dir=csv_dir
        )

    # Filter by pLDDT threshold if specified and sort by pLDDT in descending order
    results_df = results_df.sort_values('plddt', ascending=True)
    if plddt_threshold > 0:
        results_df = results_df[results_df['plddt'] > plddt_threshold]

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create custom colormap from initial to end color
    custom_cmap = LinearSegmentedColormap.from_list('custom', COLORS['correlation_cmap'])
    
    # Plot with pLDDT coloring
    plt.figure(figsize=(8,7))
    scatter = plt.scatter(
        results_df[metric_name1],
        results_df[metric_name2], 
        c=results_df['plddt'],
        cmap=custom_cmap,
        s=PLOT_PARAMS['scatter_size'],
        alpha=PLOT_PARAMS['scatter_alpha'],
        vmin=50,
        vmax=100,
        edgecolors=None,
        linewidths=0
    )

    plt.colorbar(scatter, label='pLDDT Score')
    plt.xlabel(metric_name1)
    plt.ylabel(metric_name2)

    if x_min is not None and x_max is not None:
        plt.xlim(x_min, x_max)
        if x_ticks is not None:
            plt.xticks(x_ticks)
        else:
            plt.xticks(np.arange(x_min, x_max + 1, 5))
        # Move x-axis ticks away from axis
        plt.tick_params(axis='x', which='major', pad=10)
            
    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)
        if y_ticks is not None:
            plt.yticks(y_ticks)
        else:
            plt.yticks(np.arange(y_min, y_max + 1, 5))
        # Move y-axis ticks away from axis    
        plt.tick_params(axis='y', which='major', pad=10)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{metric_name1}_{metric_name2}_scatter_plddt.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=500)
    plt.close()

    # Only plot local pLDDT if it exists in the dataframe
    if 'local_plddt' in results_df.columns:
        plt.figure(figsize=(8,7))
        scatter = plt.scatter(
            results_df[metric_name1],
            results_df[metric_name2], 
            c=results_df['local_plddt'],
            cmap=custom_cmap,
            s=PLOT_PARAMS['scatter_size'],
            alpha=PLOT_PARAMS['scatter_alpha'],
            vmin=50,
            vmax=100,
            edgecolors=None,
            linewidths=0
        )

        plt.colorbar(scatter, label='Local pLDDT Score')
        plt.xlabel(metric_name1)
        plt.ylabel(metric_name2)

        if x_min is not None and x_max is not None:
            plt.xlim(x_min, x_max)
            if x_ticks is not None:
                plt.xticks(x_ticks)
            else:
                plt.xticks(np.arange(x_min, x_max + 1, 5))
            # Move x-axis ticks away from axis
            plt.tick_params(axis='x', which='major', pad=10)
                
        if y_min is not None and y_max is not None:
            plt.ylim(y_min, y_max)
            if y_ticks is not None:
                plt.yticks(y_ticks)
            else:
                plt.yticks(np.arange(y_min, y_max + 1, 5))
            # Move y-axis ticks away from axis
            plt.tick_params(axis='y', which='major', pad=10)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'{metric_name1}_{metric_name2}_scatter_local_plddt.png')
        plt.savefig(plot_path, bbox_inches='tight', dpi=500)
        plt.close()

    print(f"Generated scatter plots saved to: {output_dir}")
    print(f"Total structures analyzed{f' (pLDDT > {plddt_threshold})' if plddt_threshold > 0 else ''}: {len(results_df)}")

def plot_m_fold_sampling_2d_joint(results_dir: str,
                     config_path: str,
                     output_dir: str,
                     csv_dir: str,
                     x_min: float = None,
                     x_max: float = None,
                     y_min: float = None,
                     y_max: float = None,
                     plddt_threshold: float = 0):
    """Plot 2D joint plots (scatter + histograms) of two metrics from M-fold sampling results.

    Args:
        results_dir (str): Path to directory containing M-fold sampling results
        config_path (str): Path to config JSON file specifying metrics to analyze
        output_dir (str): Path to directory for saving plot outputs
        csv_dir (str): Path to directory for saving/loading CSV results
        x_min (float, optional): Minimum x-axis value. Defaults to None.
        x_max (float, optional): Maximum x-axis value. Defaults to None.
        y_min (float, optional): Minimum y-axis value. Defaults to None.
        y_max (float, optional): Maximum y-axis value. Defaults to None.
        plddt_threshold (float, optional): pLDDT threshold for filtering structures.
            Structures with pLDDT below this value will be excluded. Defaults to 0 (no filtering).

    The function creates two joint plots using seaborn showing:
    - Center: 2D scatter plot of two metrics colored by either pLDDT or local pLDDT
    - Top and Right: Kernel density estimation plots of marginal distributions
    Results are cached in a CSV file for faster replotting.
    """

    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Get metric names from config
    metric_name1 = config['filter_criteria'][0]['name']
    metric_name2 = config['filter_criteria'][1]['name']
    
    # Check if CSV exists
    csv_path = os.path.join(csv_dir, f'{metric_name1}_{metric_name2}_values.csv')
    if os.path.exists(csv_path):
        print(f"Loading existing results from {csv_path}")
        results_df = pd.read_csv(csv_path)
    else:
        # Get results dataframe
        results_df = calculate_metric_values(
            parent_dir=results_dir,
            config=config,
            csv_dir=csv_dir
        )

    # Filter by pLDDT threshold if specified and sort by pLDDT
    results_df = results_df.sort_values('plddt', ascending=True)
    if plddt_threshold > 0:
        results_df = results_df[results_df['plddt'] > plddt_threshold]

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Create custom colormap
    custom_cmap = LinearSegmentedColormap.from_list('custom', COLORS['correlation_cmap'])

    # Plot with pLDDT coloring
    g = sns.JointGrid(data=results_df, 
                     x=metric_name1,
                     y=metric_name2,
                     height=10)

    g.plot_joint(plt.scatter,
                 c=results_df['plddt'],
                 cmap=custom_cmap,
                 s=PLOT_PARAMS['scatter_size'],
                 alpha=PLOT_PARAMS['scatter_alpha'],
                 vmin=50,
                 vmax=100)

    g.plot_marginals(sns.kdeplot,
                    color='gray',
                    alpha=0.5,
                    log_scale=False)

    cax = g.figure.add_axes([.95, .4, .02, .2])
    plt.colorbar(g.ax_joint.collections[0], cax=cax, label='pLDDT Score')

    g.ax_joint.set_xlabel(metric_name1)
    g.ax_joint.set_ylabel(metric_name2)

    if x_min is not None and x_max is not None:
        g.ax_joint.set_xlim(x_min, x_max)
    if y_min is not None and y_max is not None:
        g.ax_joint.set_ylim(y_min, y_max)

    plot_path = os.path.join(output_dir, f'{metric_name1}_{metric_name2}_joint_plddt.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=500)
    plt.close()

    # Only plot local pLDDT if it exists in the dataframe
    if 'local_plddt' in results_df.columns:
        g = sns.JointGrid(data=results_df, 
                         x=metric_name1,
                         y=metric_name2,
                         height=10)

        g.plot_joint(plt.scatter,
                     c=results_df['local_plddt'],
                     cmap=custom_cmap,
                     s=PLOT_PARAMS['scatter_size'],
                     alpha=PLOT_PARAMS['scatter_alpha'],
                     vmin=50,
                     vmax=100)

        g.plot_marginals(sns.kdeplot,
                        color='gray',
                        alpha=0.5,
                        log_scale=False)

        cax = g.figure.add_axes([.95, .4, .02, .2])
        plt.colorbar(g.ax_joint.collections[0], cax=cax, label='Local pLDDT Score')

        g.ax_joint.set_xlabel(metric_name1)
        g.ax_joint.set_ylabel(metric_name2)

        if x_min is not None and x_max is not None:
            g.ax_joint.set_xlim(x_min, x_max)
        if y_min is not None and y_max is not None:
            g.ax_joint.set_ylim(y_min, y_max)

        plot_path = os.path.join(output_dir, f'{metric_name1}_{metric_name2}_joint_local_plddt.png')
        plt.savefig(plot_path, bbox_inches='tight', dpi=500)
        plt.close()

    print(f"Generated joint plots saved to: {output_dir}")
    print(f"Total structures analyzed{f' (pLDDT > {plddt_threshold})' if plddt_threshold > 0 else ''}: {len(results_df)}")


def plot_iteration_shuffling_distribution(results_df, output_dir, initial_color, x_min=None, x_max=None):
    """Create distribution plots from metric data for each iteration"""
    iterations = results_df['iteration'].unique()
    iterations.sort()
    
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
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"Distribution plots by iteration saved to {output_path}")

def plot_scatter(results_dir: str,
                config_path: str,
                output_dir: str,
                csv_dir: str,
                initial_color: tuple = (135/255, 206/255, 235/255),
                x_min: float = None,
                x_max: float = None,
                y_min: float = None,
                y_max: float = None):
    """
    Generate 2D scatter plot of metric vs pLDDT scores.
    
    Args:
        results_dir (str): Path to directory containing results
        config_path (str): Path to config JSON file
        output_dir (str): Path to directory for saving plot outputs
        csv_dir (str): Path to directory for saving CSV results
        initial_color (tuple): RGB color for scatter points
        x_min (float): Minimum x-axis value
        x_max (float): Maximum x-axis value
        y_min (float): Minimum y-axis value
        y_max (float): Maximum y-axis value
    """
    # Load config
    with open(config_path, 'r') as f:
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

