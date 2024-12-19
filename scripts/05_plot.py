import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from af_vote.structure_analysis import StructureAnalyzer, DEFAULT_TMALIGN_PATH

# Plot style configuration
PLOT_STYLE = {
    'font.size': 20,
    'axes.labelsize': 20, 
    'axes.titlesize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20
}
plt.rcParams.update(PLOT_STYLE)

# Plot appearance constants
PLOT_PARAMS = {
    'scatter_size': 140,
    'scatter_alpha': 0.9,
    'scatter_edge': 'none',
    'dpi': 600,
    'prediction_figsize': (15, 7),
    'selection_figsize': (14, 8)
}

# Color schemes
COLORS = {
    'default_selection': '#468F8B',
    'default_control': '#AFD2D0',
    'correlation_cmap': ['#72b3d4', '#dadada', '#bc6565']
}

def get_plddt_from_pdb(pdb_file, full_index):
    """
    Extract average pLDDT score from PDB file
    
    Args:
        pdb_file: Path to PDB file
        full_index: Dict with start/end indices or list of indices
        
    Returns:
        float: Average pLDDT score, 0 if calculation fails
    """
    try:
        analyzer = StructureAnalyzer()
        
        if isinstance(full_index, dict) and 'start' in full_index and 'end' in full_index:
            indices = list(range(full_index['start'], full_index['end']+1))
        elif isinstance(full_index, list):
            indices = [int(i) for i in full_index]
        else:
            raise ValueError("full_index must be either a dict with 'start'/'end' or a list")
            
        plddt = analyzer.plddt_process(pdb_file, indices)
        return plddt if plddt is not None else 0
    except Exception as e:
        print(f"Error processing pLDDT for {pdb_file}: {str(e)}")
        return 0

def get_metric_from_structure(pdb_file, config, criterion_name=None):
    """
    Calculate structure metric based on config specification
    
    Args:
        pdb_file: Path to PDB file to analyze
        config: Configuration dictionary
        criterion_name: Optional name of specific criterion to calculate
        
    Returns:
        float: Calculated metric value
    """
    try:
        analyzer = StructureAnalyzer()
        
        for criterion in config['filter_criteria']:
            if criterion_name and criterion['name'] != criterion_name:
                continue
                
            if criterion['type'] == 'distance':
                return analyzer.calculate_residue_distance(
                    pdb_file,
                    'A', 
                    criterion['indices']['set1'],
                    criterion['indices']['set2']
                )
                
            elif criterion['type'] == 'angle':
                return analyzer.calculate_angle(
                    pdb_file,
                    criterion['indices']['domain1'],
                    criterion['indices']['domain2'], 
                    criterion['indices']['hinge']
                )
                
            elif criterion['type'] == 'rmsd':
                sup_indices, rmsd_indices = _get_rmsd_indices(criterion, config)
                return analyzer.calculate_ca_rmsd(
                    reference_pdb=criterion['ref_pdb'],
                    target_pdb=pdb_file,
                    superposition_indices=sup_indices,
                    rmsd_indices=rmsd_indices,
                    chain_id='A'
                )
                
            elif criterion['type'] == 'tmscore':
                return analyzer.calculate_tm_score(
                    target_pdb=pdb_file,
                    reference_pdb=criterion['ref_pdb'],
                    tm_align_path=DEFAULT_TMALIGN_PATH
                )
                
        return None
    except Exception as e:
        print(f"Error calculating metric for {pdb_file}: {str(e)}")
        return None

def _get_rmsd_indices(criterion, config):
    """Helper function to extract RMSD calculation indices"""
    if 'superposition_indices' in criterion and 'rmsd_indices' in criterion:
        sup_indices = list(range(
            criterion['superposition_indices']['start'],
            criterion['superposition_indices']['end'] + 1
        ))
        
        rmsd_indices = []
        if isinstance(criterion['rmsd_indices'], list):
            for range_dict in criterion['rmsd_indices']:
                rmsd_indices.extend(range(range_dict['start'], range_dict['end']+1))
        else:
            rmsd_indices = list(range(
                criterion['rmsd_indices']['start'],
                criterion['rmsd_indices']['end'] + 1
            ))
    else:
        full_range = list(range(
            config['basics']['full_index']['start'],
            config['basics']['full_index']['end'] + 1
        ))
        sup_indices = rmsd_indices = full_range
        
    return sup_indices, rmsd_indices

def process_prediction_dir(pred_dir, config):
    """
    Process prediction directory containing PDB files with seed information
    
    Args:
        pred_dir: Directory containing prediction PDB files
        config: Configuration dictionary
        
    Returns:
        Dict mapping bin patterns to DataFrames with processed metrics
    """
    data_by_bin = {}
    bin_dirs = glob.glob(os.path.join(pred_dir, 'bin_*'))
    
    if not bin_dirs:
        print(f"Warning: No bin directories found in {pred_dir}")
        return {}
        
    criteria = config['filter_criteria'][:2]
    
    for bin_dir in bin_dirs:
        try:
            bin_pattern = os.path.basename(bin_dir)  # Keep full bin pattern (e.g. bin_29 or bin_29_30_31)
            pdb_pattern = os.path.join(bin_dir, '**', '*seed_*.pdb')
            pdb_files = glob.glob(pdb_pattern, recursive=True)
            
            if not pdb_files:
                print(f"Warning: No PDB files found in {bin_dir}")
                continue
                
            bin_data = []
            for pdb_file in pdb_files:
                try:
                    seed_num = int(os.path.basename(pdb_file).split('seed_')[1].split('.')[0])
                    
                    entry = {
                        'plddt': get_plddt_from_pdb(pdb_file, config['basics']['full_index']),
                        'metric1': get_metric_from_structure(pdb_file, config, criteria[0]['name']),
                        'metric2': None,
                        'seed': seed_num,
                        'bin_pattern': bin_pattern
                    }
                    
                    if len(criteria) > 1:
                        entry['metric2'] = get_metric_from_structure(
                            pdb_file, config, criteria[1]['name']
                        )
                        
                    if entry['metric1'] is not None:  # Only add if metric calculation succeeded
                        bin_data.append(entry)
                except Exception as e:
                    print(f"Error processing file {pdb_file}: {str(e)}")
                    continue
                    
            if bin_data:
                data_by_bin[bin_pattern] = pd.DataFrame(bin_data)
                    
        except Exception as e:
            print(f"Error processing bin directory {bin_dir}: {str(e)}")
            continue
                
    return data_by_bin

def process_selection_dir(sel_dir, config):
    """
    Process selection directory containing PDB files without seed information
    
    Args:
        sel_dir: Directory containing selection PDB files  
        config: Configuration dictionary
        
    Returns:
        Dict mapping bin patterns to DataFrames with processed metrics
    """
    data_by_bin = {}
    bin_dirs = glob.glob(os.path.join(sel_dir, 'bin_*'))
    
    if not bin_dirs:
        print(f"Warning: No bin directories found in {sel_dir}")
        return {}
        
    criteria = config['filter_criteria'][:2]
    
    for bin_dir in bin_dirs:
        try:
            bin_pattern = os.path.basename(bin_dir)  # Keep full bin pattern
            pdb_pattern = os.path.join(bin_dir, '**', '*.pdb')
            pdb_files = glob.glob(pdb_pattern, recursive=True)
            
            if not pdb_files:
                print(f"Warning: No PDB files found in {bin_dir}")
                continue
                
            bin_data = []
            for pdb_file in pdb_files:
                try:
                    entry = {
                        'plddt': get_plddt_from_pdb(pdb_file, config['basics']['full_index']),
                        'metric1': get_metric_from_structure(pdb_file, config, criteria[0]['name']),
                        'metric2': None,
                        'bin_pattern': bin_pattern
                    }
                    
                    if len(criteria) > 1:
                        entry['metric2'] = get_metric_from_structure(
                            pdb_file, config, criteria[1]['name']
                        )
                        
                    if entry['metric1'] is not None:  # Only add if metric calculation succeeded
                        bin_data.append(entry)
                except Exception as e:
                    print(f"Error processing file {pdb_file}: {str(e)}")
                    continue
                    
            if bin_data:
                data_by_bin[bin_pattern] = pd.DataFrame(bin_data)
                    
        except Exception as e:
            print(f"Error processing bin directory {bin_dir}: {str(e)}")
            continue
                
    return data_by_bin

def plot_prediction_results(pred_data, control_pred_data, output_path, title, color1, color2, 
                          metric_name, bin_pattern, xlim=None, ylim=None, xticks=None, yticks=None):
    """
    Create scatter plots comparing prediction results with control for a specific bin pattern
    
    Args:
        pred_data: DataFrame with prediction metrics for this bin pattern
        control_pred_data: DataFrame with control prediction metrics for this bin pattern
        output_path: Path to save plot
        title: Plot title
        color1, color2: Colors for prediction and control
        metric_name: Name of metric being plotted
        bin_pattern: Bin pattern being plotted (e.g. bin_29 or bin_29_30_31)
        xlim, ylim: Optional axis limits
        xticks, yticks: Optional tick locations
    """
    if pred_data.empty and control_pred_data.empty:
        print(f"Warning: No data available for plotting prediction results for {bin_pattern}")
        return
        
    plt.figure(figsize=PLOT_PARAMS['prediction_figsize'])
    prediction_palette = LinearSegmentedColormap.from_list('custom', [color1, color2], N=16)
    
    # Plot predictions
    plt.subplot(1, 2, 1)
    _plot_predictions(pred_data, prediction_palette, metric_name, f"{title} - {bin_pattern}", 
                     xlim, ylim, xticks, yticks)
    
    # Plot control predictions  
    plt.subplot(1, 2, 2)
    _plot_predictions(control_pred_data, prediction_palette, metric_name, 
                     f"{title} - Control - {bin_pattern}", xlim, ylim, xticks, yticks)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=PLOT_PARAMS['dpi'])
    plt.close()

def _plot_predictions(data, palette, metric_name, title, xlim=None, ylim=None, xticks=None, yticks=None):
    """Helper function for prediction plotting"""
    if data.empty:
        plt.text(0.5, 0.5, 'No data available', 
                horizontalalignment='center',
                verticalalignment='center')
        return
        
    unique_seeds = sorted(list(set(data['seed'])))
    
    for i, seed in enumerate(unique_seeds):
        mask = (data['seed'] == seed)
        plt.scatter(
            data[mask]['metric1'], 
            data[mask]['plddt'],
            color=palette(i/len(unique_seeds)),
            label=f'Seed {seed:03d}',
            s=PLOT_PARAMS['scatter_size'],
            alpha=PLOT_PARAMS['scatter_alpha'],
            edgecolor=PLOT_PARAMS['scatter_edge']
        )
    
    plt.xlabel(metric_name)
    plt.ylabel('pLDDT Score')
    plt.title(f"{title}", pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
    if xlim: plt.xlim(xlim)
    if ylim: plt.ylim(ylim)
    if xticks is not None: plt.xticks(xticks)
    if yticks is not None: plt.yticks(yticks)

def plot_selection_results(sel_data, control_sel_data, output_path, title, color1, color2,
                         metric_name, bin_pattern, xlim=None, ylim=None, xticks=None, yticks=None):
    """
    Create scatter plots comparing selection results with control for a specific bin pattern
    
    Args:
        sel_data: DataFrame with selection metrics for this bin pattern
        control_sel_data: DataFrame with control selection metrics for this bin pattern
        output_path: Path to save plot
        title: Plot title
        color1, color2: Colors for selection and control
        metric_name: Name of metric being plotted
        bin_pattern: Bin pattern being plotted
        xlim, ylim: Optional axis limits
        xticks, yticks: Optional tick locations
    """
    if sel_data.empty and control_sel_data.empty:
        print(f"Warning: No data available for plotting selection results for {bin_pattern}")
        return
        
    plt.figure(figsize=PLOT_PARAMS['selection_figsize'])
    
    # Plot selections
    plt.subplot(1, 2, 1)
    _plot_selections(sel_data, color1, metric_name, f"{title} - Selection - {bin_pattern}", 
                    xlim, ylim, xticks, yticks)
    
    # Plot control selections
    plt.subplot(1, 2, 2) 
    _plot_selections(control_sel_data, color2, metric_name, 
                    f"{title} - Control Selection - {bin_pattern}", xlim, ylim, xticks, yticks)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=PLOT_PARAMS['dpi'])
    plt.close()

def _plot_selections(data, color, metric_name, title, xlim=None, ylim=None, xticks=None, yticks=None):
    """Helper function for selection plotting"""
    if data.empty:
        plt.text(0.5, 0.5, 'No data available', 
                horizontalalignment='center',
                verticalalignment='center')
        return
        
    plt.scatter(
        data['metric1'],
        data['plddt'],
        color=color,
        s=PLOT_PARAMS['scatter_size'],
        alpha=PLOT_PARAMS['scatter_alpha'],
        edgecolor=PLOT_PARAMS['scatter_edge']
    )
    
    plt.xlabel(metric_name)
    plt.ylabel('pLDDT Score')
    plt.title(title, pad=20)
    if xlim: plt.xlim(xlim)
    if ylim: plt.ylim(ylim)
    if xticks is not None: plt.xticks(xticks)
    if yticks is not None: plt.yticks(yticks)

def plot_metric_correlation(data, 
                          output_path, 
                          title, 
                          metric1_name, 
                          metric2_name,
                          bin_pattern,
                          xlim=None, 
                          ylim=None,
                          xticks=None,
                          yticks=None):
    """
    Create scatter plot showing correlation between two metrics colored by pLDDT for a specific bin pattern
    
    Args:
        data: DataFrame with metrics to correlate for this bin pattern
        output_path: Path to save plot
        title: Plot title
        metric1_name, metric2_name: Names of metrics being correlated
        bin_pattern: Bin pattern being plotted
        xlim, ylim: Optional axis limits
        xticks, yticks: Optional tick locations
    """
    if data.empty:
        print(f"Warning: No data available for plotting metric correlation for {title} {bin_pattern}")
        return
        
    plt.figure(figsize=(7,6))
    
    custom_cmap = LinearSegmentedColormap.from_list('custom', COLORS['correlation_cmap'])
    # Sort data by pLDDT in ascending order so lower pLDDT points appear behind higher ones
    data = data.sort_values('plddt', ascending=True)
    
    scatter = plt.scatter(
        data['metric1'],
        data['metric2'],
        c=data['plddt'],
        cmap=custom_cmap,
        s=PLOT_PARAMS['scatter_size'],
        alpha=PLOT_PARAMS['scatter_alpha'],
        vmin=50,
        vmax=100
    )
    
    plt.colorbar(scatter, label='pLDDT Score')
    plt.xlabel(metric1_name)
    plt.ylabel(metric2_name) 
    plt.title(f"{title} - {bin_pattern}")
    if xlim: plt.xlim(xlim)
    if ylim: plt.ylim(ylim)
    if xticks is not None: plt.xticks(xticks)
    if yticks is not None: plt.yticks(yticks)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=PLOT_PARAMS['dpi'])
    plt.close()

def main():
    """Main function to run plotting pipeline"""
    parser = argparse.ArgumentParser(description='Generate plots from structure prediction data')
    
    # Required arguments
    parser.add_argument('--base_dir', required=True,
                      help='Base directory containing prediction results')
    parser.add_argument('--output_dir', required=True, 
                      help='Output directory for plots')
    parser.add_argument('--config', required=True,
                      help='Path to config JSON file')
    
    # Optional arguments
    parser.add_argument('--color1', default=COLORS['default_selection'],
                      help='Color for selection plots')
    parser.add_argument('--color2', default=COLORS['default_control'],
                      help='Color for control selection plots')
    parser.add_argument('--xlim', nargs=2, type=float,
                      help='X-axis limits [min max] for single metric plots')
    parser.add_argument('--ylim', nargs=2, type=float,
                      help='Y-axis limits [min max] for single metric plots')
    parser.add_argument('--xlim2', nargs=2, type=float,
                      help='X-axis limits [min max] for correlation plots')
    parser.add_argument('--ylim2', nargs=2, type=float,
                      help='Y-axis limits [min max] for correlation plots')
    parser.add_argument('--xticks', nargs='*', type=float,
                      help='X-axis tick locations')
    parser.add_argument('--yticks', nargs='*', type=float,
                      help='Y-axis tick locations')
    parser.add_argument('--xticks2', nargs='*', type=float,
                      help='X-axis tick locations for correlation plots')
    parser.add_argument('--yticks2', nargs='*', type=float,
                      help='Y-axis tick locations for correlation plots')
                      
    args = parser.parse_args()
    
    try:
        # Load and validate config
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        has_second_criterion = len(config['filter_criteria']) > 1
        
        # Get metric names
        metric1_name = config['filter_criteria'][0]['name']
        metric2_name = config['filter_criteria'][1]['name'] if has_second_criterion else None
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Process data by bin pattern
        pred_data_by_bin = process_prediction_dir(
            os.path.join(args.base_dir, 'prediction'),
            config
        )
        
        control_pred_data_by_bin = process_prediction_dir(
            os.path.join(args.base_dir, 'control_prediction'),
            config
        )
        
        sel_data_by_bin = process_selection_dir(
            os.path.join(args.base_dir, 'random_selection'),
            config
        )
        
        control_sel_data_by_bin = process_selection_dir(
            os.path.join(args.base_dir, 'control_random_selection'),
            config
        )
        
        # Get all unique bin patterns
        all_bin_patterns = set()
        for data_dict in [pred_data_by_bin, control_pred_data_by_bin, 
                         sel_data_by_bin, control_sel_data_by_bin]:
            all_bin_patterns.update(data_dict.keys())
        
        # Generate plots for each bin pattern
        for bin_pattern in sorted(all_bin_patterns):
            bin_pred_data = pred_data_by_bin.get(bin_pattern, pd.DataFrame())
            bin_control_pred_data = control_pred_data_by_bin.get(bin_pattern, pd.DataFrame())
            bin_sel_data = sel_data_by_bin.get(bin_pattern, pd.DataFrame())
            bin_control_sel_data = control_sel_data_by_bin.get(bin_pattern, pd.DataFrame())
            
            # Create bin-specific output directory using full pattern
            bin_output_dir = os.path.join(args.output_dir, bin_pattern)
            os.makedirs(bin_output_dir, exist_ok=True)
            
            # Generate standard plots
            plot_prediction_results(
                bin_pred_data,
                bin_control_pred_data,
                os.path.join(bin_output_dir, 'prediction_results.png'),
                'Prediction Results',
                args.color1,
                args.color2,
                metric1_name,
                bin_pattern,
                args.xlim,
                args.ylim,
                args.xticks,
                args.yticks
            )
            
            plot_selection_results(
                bin_sel_data,
                bin_control_sel_data,
                os.path.join(bin_output_dir, 'selection_results.png'),
                'Selection Results', 
                args.color2,
                args.color1,
                metric1_name,
                bin_pattern,
                args.xlim,
                args.ylim,
                args.xticks,
                args.yticks
            )
            
            # Generate correlation plots if second metric exists
            if metric2_name:
                for data, name in [
                    (bin_pred_data, 'prediction'),
                    (bin_control_pred_data, 'control_prediction'),
                    (bin_sel_data, 'selection'),
                    (bin_control_sel_data, 'control_selection')
                ]:
                    plot_metric_correlation(
                        data,
                        os.path.join(bin_output_dir, f'{name}_metric_correlation.png'),
                        f'{name.replace("_", " ").title()} Metric Correlation',
                        metric1_name,
                        metric2_name,
                        bin_pattern,
                        args.xlim2,
                        args.ylim2,
                        args.xticks2,
                        args.yticks2
                    )
                
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
