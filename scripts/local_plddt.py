import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from af_vote.structure_analysis import StructureAnalyzer
import argparse

# Plot style configuration 
PLOT_STYLE = {
    'font.size': 16,
    'axes.labelsize': 16,
    'axes.titlesize': 18, 
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16
}
plt.rcParams.update(PLOT_STYLE)

def get_local_plddt_from_pdb(pdb_file, indices):
    """Extract local pLDDT scores for specific residue indices from PDB file"""
    try:
        analyzer = StructureAnalyzer()
        plddt = analyzer.plddt_process(pdb_file, indices)
        return plddt if plddt is not None else None
    except Exception as e:
        print(f"Error processing pLDDT for {pdb_file}: {str(e)}")
        return None

def process_directory(directory, indices):
    """Process directory containing PDB files and extract local pLDDT scores"""
    plddt_scores = []
    pdb_files = glob.glob(os.path.join(directory, '**', '*.pdb'), recursive=True)
    
    for pdb_file in pdb_files:
        plddt = get_local_plddt_from_pdb(pdb_file, indices)
        if plddt is not None:
            plddt_scores.append(plddt)
            
    return plddt_scores

def plot_plddt_comparison(data_dict, output_path, title):
    """Create violin plot comparing pLDDT distributions"""
    plt.figure(figsize=(10, 6))
    
    # Convert data to format suitable for violin plot
    plot_data = []
    for category, scores in data_dict.items():
        for score in scores:
            plot_data.append({
                'Category': category,
                'pLDDT': score
            })
    df = pd.DataFrame(plot_data)
    
    # Create violin plot
    sns.violinplot(data=df, x='Category', y='pLDDT')
    
    plt.title(title)
    plt.ylabel('Local pLDDT Score')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate local pLDDT comparison plots')
    
    parser.add_argument('--base_dir', required=True,
                      help='Base directory containing prediction results')
    parser.add_argument('--output_dir', required=True,
                      help='Output directory for plots')
    parser.add_argument('--config', required=True,
                      help='Path to config JSON file')
    
    args = parser.parse_args()
    
    try:
        # Load config
        with open(args.config, 'r') as f:
            config = json.load(f)
            
        # Get residue indices
        indices = list(range(
            config['basics']['local_index']['start'],
            config['basics']['local_index']['end'] + 1
        ))
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Process each results directory
        plddt_data = {}
        for dir_name in ['prediction', 'control_prediction']:
            dir_path = os.path.join(args.base_dir, dir_name)
            if os.path.exists(dir_path):
                plddt_data[dir_name] = process_directory(dir_path, indices)
        
        # Generate plot
        plot_plddt_comparison(
            plddt_data,
            os.path.join(args.output_dir, 'local_plddt_comparison.png'),
            'Local pLDDT Score Comparison'
        )
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
