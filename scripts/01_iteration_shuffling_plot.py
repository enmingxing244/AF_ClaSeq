import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from af_claseq.structure_analysis import get_result_df, load_filter_modes
import matplotlib.ticker as ticker

# Set publication-quality font
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['font.size'] = 24
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24

def calculate_metric_values(parent_dir, config_path, iteration_dirs):
    """Calculate metric values (RMSD or TM-score) for all iterations and save to CSV"""
    filter_modes = load_filter_modes(config_path)
    all_data = []
    
    metric_type = filter_modes['filter_criteria'][0]['type']
    metric_name = filter_modes['filter_criteria'][0]['name']
    
    for iteration_dir in tqdm(iteration_dirs, desc="Processing iterations"):
        iteration_path = os.path.join(parent_dir, iteration_dir)
        
        # Get results dataframe for this iteration
        results_df = get_result_df(
            iteration_path,
            filter_modes['filter_criteria'],
            filter_modes['basics']
        )
        
        # Get values from results
        metric_values = results_df[metric_name].dropna()
        plddt_values = results_df['plddt'].dropna()
        pdb_paths = results_df['PDB']
        seq_counts = results_df['seq_count']
        
        if not metric_values.empty:
            iteration_data = pd.DataFrame({
                'iteration': iteration_dir.split('_')[1],
                'PDB': pdb_paths,
                'seq_count': seq_counts,
                'plddt': plddt_values,
                metric_name: metric_values
            })
            all_data.append(iteration_data)
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Save to CSV
    csv_path = os.path.join(parent_dir, f'analysis/plot/{metric_name}_values.csv')
    combined_df.to_csv(csv_path, index=False)
    
    return combined_df, metric_type

def plot_distribution(df, parent_dir, metric_name, metric_type, num_cols, x_min, x_max, y_min, y_max, xticks, bin_step):
    """Create distribution plots from metric data"""
    iterations = sorted(df['iteration'].unique())
    
    n_iterations = len(iterations)
    n_cols = min(num_cols, n_iterations)
    n_rows = (n_iterations + n_cols - 1) // n_cols
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, 
                             n_cols, 
                             figsize=(5*n_cols, 4*n_rows), 
                             sharex=True, 
                             sharey=True
                             )
    
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(-1, 1) if n_cols == 1 else axes.reshape(1, -1)
    
    bins = np.arange(x_min, x_max + bin_step, bin_step)
    
    # Find global y limits
    max_count = 0
    for iteration in iterations:
        iteration_data = df[df['iteration'] == iteration][metric_name]
        counts, _ = np.histogram(iteration_data, bins=bins)
        max_count = max(max_count, max(counts))
    
    # Plot each iteration
    for idx, iteration in enumerate(iterations):
        row = idx // n_cols
        col = idx % n_cols
        
        iteration_data = df[df['iteration'] == iteration][metric_name]

        quantile = 0.2 if metric_type == 'rmsd' else 0.8
        quantile_value = np.quantile(iteration_data, quantile)
        axes[row, col].hist(iteration_data, bins=bins, color='skyblue', edgecolor=None, alpha=0.7)
        
        unit = 'Å' if metric_type == 'rmsd' else ''
        axes[row, col].axvline(x=quantile_value, color='red', linestyle='--', alpha=0.7,
                             label=f'{quantile_value:.3f}{unit}')

        axes[row, col].set_title(f'Iteration {iteration}')
        axes[row, col].legend(fontsize=16, loc='upper right')
            
        axes[row, col].set_xlim(x_min, x_max)
        axes[row, col].set_xticks(xticks)
        # Format x-axis tick labels to show exact values given
        axes[row, col].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: str(int(x)) if x.is_integer() else f'{x:.1f}'))
        axes[row, col].set_yscale('log')
        axes[row, col].set_ylim(y_min, y_max)
        axes[row, col].yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
        axes[row, col].yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=list(np.arange(2, 10) * 0.1), numticks=10))
        axes[row, col].tick_params(axis='y', which='major', length=8, width=1)
        axes[row, col].tick_params(axis='y', which='minor', length=5, width=1)
        axes[row, col].tick_params(axis='x', which='major', length=5, width=1)

    # Add single x and y labels for entire figure
    x_label = f'RMSD to {metric_name.split("_")[0]} (Å)' if metric_type == 'rmsd' else f'TM-score to {metric_name.split("_")[0]}'
    # fig.text(0.5, 0.02, x_label, ha='center', va='center')
    # fig.text(0.02, 0.5, 'Counts of predicted structures', ha='center', va='center', rotation='vertical')

    for idx in range(len(iterations), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row, col])
    
    plt.subplots_adjust(wspace=0.14, hspace=0.2, left=0.1, bottom=0.1)
    
    output_path = os.path.join(parent_dir, f'analysis/plot/{metric_name}_distribution_by_iteration.png')
    plt.savefig(output_path, dpi=600, bbox_inches='tight')

    plt.close()
    print(f"{metric_type.upper()} distribution plots saved to {output_path}")

def main():
    # Get inputs from command line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', required=True, help='Base directory for all output files')
    parser.add_argument('--config_path', required=True, help='Path to config file')
    parser.add_argument('--iter_shuf_plot_num_cols', type=int, default=5, help='Number of columns in plot grid')
    parser.add_argument('--iter_shuf_plot_x_min', type=float, default=0, help='Minimum x-axis value')
    parser.add_argument('--iter_shuf_plot_x_max', type=float, default=20, help='Maximum x-axis value')
    parser.add_argument('--iter_shuf_plot_y_min', type=float, default=0.8, help='Minimum y-axis value')
    parser.add_argument('--iter_shuf_plot_y_max', type=float, default=10000, help='Maximum y-axis value')
    parser.add_argument('--iter_shuf_plot_xticks', type=float, nargs='+', help='List of x-axis tick positions')
    parser.add_argument('--iter_shuf_plot_bin_step', type=float, default=0.2, help='Step size for binning')
    args = parser.parse_args()
    
    # Set parent_dir to be base_dir/01_iterative_shuffling
    parent_dir = os.path.join(args.base_dir, '01_iterative_shuffling')
    
    # Get list of iteration directories
    iteration_dirs = [d for d in os.listdir(parent_dir) if d.startswith('Iteration_')]
    iteration_dirs.sort(key=lambda x: int(x.split('_')[1]))
    
    if not iteration_dirs:
        print("No iterations found.")
        return
    
    # Load filter modes to determine metric type
    filter_modes = load_filter_modes(args.config_path)
    metric_type = filter_modes['filter_criteria'][0]['type']
    metric_name = filter_modes['filter_criteria'][0]['name']
    
    # Check if values CSV exists
    csv_path = os.path.join(args.parent_dir, f'analysis/plot/{metric_name}_values.csv')
    if os.path.exists(csv_path):
        print(f"Loading pre-calculated {metric_type} values...")
        df = pd.read_csv(csv_path)
        metric_type = 'rmsd' if 'rmsd' in metric_name.lower() else 'tmscore'
    else:
        print(f"Calculating {metric_type} values...")
        df, metric_type = calculate_metric_values(args.parent_dir, args.config_path, iteration_dirs)
    
    # Adjust x-axis limits and ticks based on metric type
    if metric_type == 'tmscore':
        args.x_min = 0
        args.x_max = 1
        if args.xticks is None:
            args.xticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        args.bin_step = 0.02
    
    # Create plots
    plot_distribution(df,
                     args.parent_dir,
                     metric_name,
                     metric_type,
                     args.num_cols,
                     args.x_min,
                     args.x_max,
                     args.y_min,
                     args.y_max,
                     args.xticks,
                     args.bin_step)

if __name__ == "__main__":
    main()