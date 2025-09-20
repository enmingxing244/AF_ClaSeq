#!/usr/bin/env python3
"""
Quick script to analyze PDB files in top_pred folder and create scatter plot
of 6xr6 vs 6xrg composite RMSD values.
"""

import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import glob

# Import AF_ClaSeq modules
from af_claseq.utils.structure_analysis import StructureAnalyzer

def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_top_pred.py <pdb_directory>")
        print("Example: python analyze_top_pred.py /path/to/top_pred")
        sys.exit(1)

    # Get pdb_dir from command line argument
    pdb_dir = sys.argv[1]

    # Fixed config path
    config_path = "/fs/ess/PAA0203/xing244/AF_ClaSeq/results_updated/ABL1/ABL1_6xrg_6xr6_composite_rmsd.json"

    print(f"Loading config from: {config_path}")
    print(f"Analyzing PDB files in: {pdb_dir}")

    # Load filter configuration directly
    with open(config_path, 'r') as f:
        filter_config = json.load(f)

    # Get all PDB files
    pdb_files = glob.glob(f"{pdb_dir}/*.pdb")
    print(f"Found {len(pdb_files)} PDB files")

    if not pdb_files:
        print("No PDB files found!")
        return

    # Initialize structure analyzer
    analyzer = StructureAnalyzer()

    # Analyze each PDB file
    results = []
    for pdb_file in pdb_files:
        pdb_name = Path(pdb_file).stem
        print(f"Analyzing {pdb_name}...")

        try:
            # Analyze structure using process_single_pdb with correct parameters
            analysis_result = analyzer.process_single_pdb(
                pdb_path=pdb_file,
                filter_criteria=filter_config['filter_criteria'],
                basics=filter_config['basics'],
                composite_metrics=filter_config.get('composite_metrics', [])
            )

            # Extract composite RMSD values
            rmsd_6xr6 = analysis_result.get('6xr6_composite_rmsd')
            rmsd_6xrg = analysis_result.get('6xrg_composite_rmsd')

            if rmsd_6xr6 is not None and rmsd_6xrg is not None:
                results.append({
                    'pdb_name': pdb_name,
                    '6xr6_composite_rmsd': rmsd_6xr6,
                    '6xrg_composite_rmsd': rmsd_6xrg
                })
                print(f"  6xr6: {rmsd_6xr6:.3f}, 6xrg: {rmsd_6xrg:.3f}")
            else:
                print(f"  Warning: Missing RMSD values for {pdb_name}")

        except Exception as e:
            print(f"  Error analyzing {pdb_name}: {e}")

    if not results:
        print("No valid results to plot!")
        return

    # Create DataFrame
    df = pd.DataFrame(results)
    print(f"\nSuccessfully analyzed {len(df)} structures")

    # Create scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(df['6xr6_composite_rmsd'], df['6xrg_composite_rmsd'],
                alpha=0.7, s=50, c='blue', edgecolors='black', linewidth=0.5)

    plt.xlabel('6xr6 Composite RMSD (Å)', fontsize=12)
    plt.ylabel('6xrg Composite RMSD (Å)', fontsize=12)
    plt.title('6xr6 vs 6xrg Composite RMSD\nTop Predicted Structures', fontsize=14)
    plt.grid(True, alpha=0.3)

    # Add diagonal line for reference
    min_val = min(df['6xr6_composite_rmsd'].min(), df['6xrg_composite_rmsd'].min())
    max_val = max(df['6xr6_composite_rmsd'].max(), df['6xrg_composite_rmsd'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='y=x')

    plt.legend()
    plt.tight_layout()

    # Save plot as SVG
    output_plot = f"{pdb_dir}/6xr6_vs_6xrg_composite_rmsd_scatter.svg"
    plt.savefig(output_plot, format='svg', bbox_inches='tight')
    print(f"\nScatter plot saved to: {output_plot}")

    # Save data
    output_csv = f"{pdb_dir}/6xr6_vs_6xrg_composite_rmsd_data.csv"
    df.to_csv(output_csv, index=False)
    print(f"Data saved to: {output_csv}")

    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"6xr6 RMSD: mean={df['6xr6_composite_rmsd'].mean():.3f}, std={df['6xr6_composite_rmsd'].std():.3f}")
    print(f"6xrg RMSD: mean={df['6xrg_composite_rmsd'].mean():.3f}, std={df['6xrg_composite_rmsd'].std():.3f}")

    plt.show()

if __name__ == "__main__":
    main()