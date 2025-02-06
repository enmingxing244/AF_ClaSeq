#!/usr/bin/env python3
import os
import pandas as pd
import argparse
import json
from af_vote.sequence_processing import read_a3m_to_dict, concatenate_a3m_content
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def get_pdb_files(base_dir):
    """Recursively find all PDB files in directory"""
    pdb_files = [
        os.path.join(root, file)
        for root, dirs, files in os.walk(base_dir)
        for file in files
        if file.endswith('.pdb')
    ]
    return pdb_files

def get_a3m_path_from_pdb(pdb_path):
    """Convert PDB path to corresponding A3M path"""
    # Remove _unrelaxed and everything after
    a3m_path = pdb_path.split('_unrelaxed')[0] + '.a3m'
    return a3m_path

def process_pdb(pdb, metric_dict, threshold, metric_type):
    """Process a single PDB file and return A3M path if it meets criteria"""
    value = metric_dict.get(pdb)
    if value is not None:
        if metric_type == 'tmscore' and value > threshold:
            return get_a3m_path_from_pdb(pdb)
        elif metric_type == 'rmsd' and value < threshold:
            return get_a3m_path_from_pdb(pdb)
    return None

def parallel_process_pdbs(pdb_files, metric_dict, metric_type, threshold, max_workers=32):
    """Process PDB files in parallel using threads"""
    a3m_files = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_pdb, pdb, metric_dict, threshold, metric_type): pdb for pdb in pdb_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing PDBs"):
            try:
                a3m_path = future.result()
                if a3m_path:
                    a3m_files.append(a3m_path)
            except Exception as e:
                pdb = futures[future]
                print(f"Error processing PDB {pdb}: {e}")
    return a3m_files

def parse_args():
    parser = argparse.ArgumentParser(description='Combine A3M files based on TM-score filtering')
    parser.add_argument('--parent_dir', type=str, required=True,
                      help='Parent directory containing all iterations')
    parser.add_argument('--config_path', type=str, required=True,
                      help='Path to config JSON file containing filter criteria')
    parser.add_argument('--default_pdb', type=str, required=True,
                      help='Path to reference PDB file, this is used to get the query sequence')
    parser.add_argument('--max_workers', type=int, default=32,
                      help='Maximum number of parallel workers')
    parser.add_argument('--threshold', type=float, required=True,
                      help='Threshold for filtering (TM-score > threshold or RMSD < threshold)')
    return parser.parse_args()

def main():
    args = parse_args()

    # Load config file to get filter criteria
    with open(args.config_path) as f:
        config = json.load(f)
    filter_name = config['filter_criteria'][0]['name']
    metric_type = config['filter_criteria'][0]['type']

    # Read metric values from CSV into a dictionary for faster lookup
    csv_path = os.path.join(args.parent_dir, f'analysis/plot/{filter_name}_values.csv')
    print(f"Reading metric values from {csv_path}")
    print("Threshold for filtering (TM-score > threshold or RMSD < threshold)")
    print(f"Using metric: {filter_name} set threshold to {args.threshold}")

    metric_df = pd.read_csv(csv_path)
    metric_dict = pd.Series(metric_df[f'{filter_name}'].values, index=metric_df['PDB']).to_dict()
    
    # Get all PDB files
    pdb_files = get_pdb_files(args.parent_dir)
    
    # Process PDBs in parallel and get filtered A3M files
    a3m_files = parallel_process_pdbs(pdb_files, metric_dict, 
                                    metric_type=metric_type,
                                    threshold=args.threshold,
                                    max_workers=args.max_workers)
    
    print(f"Number of filtered PDBs/A3M files: {len(a3m_files)}")
    
    # Combine sequences from all A3M files
    output_dir = os.path.join(args.parent_dir, 'analysis/a3m_combine')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'gathered_seq_after_iter_shuffling.a3m')
    
    # Concatenate A3M content using utility function
    concatenate_a3m_content(a3m_files, args.default_pdb, output_file)

if __name__ == "__main__":
    main()
