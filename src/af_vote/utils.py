import logging
from Bio.PDB import PDBParser, PPBuilder
import json
import pandas as pd
from typing import List, Dict, Any
import os
import argparse

def setup_logging(base_dir, args, iterations_base_dir=None):
    iterations_base_dir = args.iterations_base_dir if hasattr(args, 'iterations_base_dir') else os.path.join(args.base_dir, "Iterations")
    log_dir = iterations_base_dir if iterations_base_dir else base_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, 'run_iterations.log' if iterations_base_dir else 'run.log')

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def get_protein_sequence(pdb_filename: str) -> str:
    """
    Extract the protein sequence from a PDB file.

    Args:
        pdb_filename (str): Path to the PDB file.

    Returns:
        str: The full protein sequence.

    Raises:
        Exception: If there's an error in processing the PDB file.
    """
    try:
        pdb_parser = PDBParser(QUIET=True)
        structure = pdb_parser.get_structure("Protein", pdb_filename)
        ppb = PPBuilder()
        sequences = []
        for pp in ppb.build_peptides(structure):
            sequence = pp.get_sequence()
            sequences.append(str(sequence))

        full_sequence = ''.join(sequences)
        return full_sequence

    except Exception as e:
        logging.error(f"Error getting protein sequence: {e}")
        raise

def count_sequences_in_a3m(a3m_file: str) -> int:
    """
    Count the number of sequences in an A3M file.

    Args:
        a3m_file (str): Path to the A3M file.

    Returns:
        int: The number of sequences in the A3M file.
    """
    count = 0
    try:
        with open(a3m_file, 'r') as file:
            for line in file:
                if line.startswith('>'):
                    count += 1
    except FileNotFoundError:
        logging.error(f"A3M file not found: {a3m_file}")
    return count

def load_filter_modes(file_path: str) -> Dict:
    """
    Load filter modes from a JSON file.

    Args:
        file_path (str): Path to the JSON file containing filter modes.

    Returns:
        Dict: The loaded filter modes.
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def apply_filters(df_threshold: pd.DataFrame, 
                  df_operate: pd.DataFrame, 
                  filter_criteria: List[Dict[str, Any]], 
                  quantile: float) -> pd.DataFrame:
    """
    Apply filters to the dataframe based on specified criteria and quantile thresholds.

    Args:
        df_threshold (pd.DataFrame): DataFrame used for calculating thresholds.
        df_operate (pd.DataFrame): DataFrame to apply filters on.
        filter_criteria (List[Dict[str, Any]]): List of filter criteria dictionaries.
        quantile (float): Quantile value for threshold calculation.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    filtered_df = df_operate.copy()
    
    for criterion in filter_criteria:
        column_name = criterion['name']
        method = criterion['method']
        
        if method not in ['above', 'below']:
            logging.warning(f"Invalid filter method: {method}. Skipping this criterion.")
            continue
        
        if column_name not in df_threshold.columns:
            logging.warning(f"Column {column_name} not found in threshold DataFrame. Skipping this criterion.")
            continue
        
        if method == 'above':
            threshold_value = df_threshold[column_name].quantile(1 - quantile)
            filtered_df = filtered_df[filtered_df[column_name] > threshold_value]
            logging.info(f"Filtering {column_name} above {(1-quantile)*100}% quantile value: {threshold_value}")
        elif method == 'below':
            threshold_value = df_threshold[column_name].quantile(quantile)
            filtered_df = filtered_df[filtered_df[column_name] < threshold_value]
            logging.info(f"Filtering {column_name} below {quantile*100}% quantile value: {threshold_value}")
    
    return filtered_df

def parse_arguments():
    parser = argparse.ArgumentParser(description='AF Path recursive clustering script.')

    # Bootstrap Shuffling Arguments
    bootstrap_group = parser.add_argument_group('Bootstrap Shuffling')
    bootstrap_group.add_argument('--run_random_bootstrap_shuffling', action='store_true', help='Run bootstrap shuffling process.')
    bootstrap_group.add_argument('--num_shuffles', type=int, default=8, help='Number of shuffles.')
    bootstrap_group.add_argument('--seq_num_per_shuffle', type=int, default=16, help='Number of sequences per shuffle.')
    bootstrap_group.add_argument('--quantile', type=float, help='Quantile for filtering.')
    bootstrap_group.add_argument('--bootstrap_ratio', type=float, help='Bootstrap ratio for subsampling.')

    # Recursive Clustering Arguments (only needed for process_random_bootstrap_shuffling)
    recursive_clustering_group = parser.add_argument_group('Recursive Clustering Arguments (only needed when not using --run_bootstrap_shuffling)')
    recursive_clustering_group.add_argument('--num_seqs_per_cluster', type=int, default=16, help='Number of sequences per cluster.')
    recursive_clustering_group.add_argument('--min_seq_id_initial', type=float, help='Initial minimum sequence identity.')
    recursive_clustering_group.add_argument('--min_seq_id_cutoff', type=float, help='Minimum sequence identity cutoff.')
    recursive_clustering_group.add_argument('--coverage', type=float, help='Sequence coverage.')
    recursive_clustering_group.add_argument('--cov_mode', type=int, default=0, help='Coverage mode.')
    recursive_clustering_group.add_argument('--cluster_reassign', type=int, default=1, help='Cluster reassign flag.')
    recursive_clustering_group.add_argument('--similarity_type', type=int, default=1, help='Similarity type.')
    recursive_clustering_group.add_argument('--quantile_last', type=float, help='Quantile for last_round_shuffle.')

    # Required Global Arguments
    global_group = parser.add_argument_group('Required Global Arguments')
    global_group.add_argument('--base_dir', type=str, help='Base directory for outputs.')
    global_group.add_argument('--reference_pdb', type=str, help='Reference PDB file.')
    global_group.add_argument('--a3m_file_path', type=str, help='Path to input A3M file.')
    global_group.add_argument('--iterations', type=int, help='Number of iterations.')
    global_group.add_argument('--iterations_base_dir', type=str, help='Directory for the current iteration.')
    global_group.add_argument('--config_file', type=str, default='config.json', help='Path to configuration JSON file.')

    # Arguments for controlling execution flow
    flow_group = parser.add_argument_group('Execution Control Arguments')
    flow_group.add_argument('--run_recursive_clustering', action='store_true', help='Run recursive clustering.')
    flow_group.add_argument('--run_last_round_shuffle', action='store_true', help='Run last_round_shuffle SLURM jobs.')
    flow_group.add_argument('--run_iteration_shuffles', action='store_true', help='Run iteration shuffles.')
    flow_group.add_argument('--skip_slurm', action='store_true', help='Skip SLURM job submission.')
    flow_group.add_argument('--skip_analysis', action='store_true', help='Skip analysis steps.')

    # SLURM-related arguments
    slurm_group = parser.add_argument_group('SLURM Arguments on submitting Colabfold jobs')
    slurm_group.add_argument('--conda_env_path', type=str, default="/fs/ess/PAA0203/xing244/.conda/envs/colabfold", help='Path to Conda environment')
    slurm_group.add_argument('--slurm_account', type=str, default="PAA0203", help='SLURM account name')
    slurm_group.add_argument('--slurm_output', type=str, default="/dev/null", help='SLURM job output path')
    slurm_group.add_argument('--slurm_error', type=str, default="/dev/null", help='SLURM job error path')
    slurm_group.add_argument('--slurm_nodes', type=int, default=1, help='Number of nodes for SLURM job')
    slurm_group.add_argument('--slurm_gpus_per_task', type=int, default=1, help='Number of GPUs per task for SLURM job')
    slurm_group.add_argument('--slurm_tasks', type=int, default=1, help='Number of tasks for SLURM job')
    slurm_group.add_argument('--slurm_cpus_per_task', type=int, default=4, help='Number of CPUs per task for SLURM job')
    slurm_group.add_argument('--slurm_time', type=str, default="10:00:00", help='Time limit for SLURM job')
    slurm_group.add_argument('--random_seed', type=int, default=42, help='Random seed for SLURM job')
    slurm_group.add_argument('--num_models', type=int, default=1, help='Number of models for SLURM job')
    slurm_group.add_argument('--check_interval', type=int, default=60, help='Interval for checking SLURM job status')

    args = parser.parse_args()

    return args

def validate_args(args):
    logging.info("Now validating arguments...")
    
    if args.run_random_bootstrap_shuffling:
        if any(getattr(args, attr) is None for attr in ['a3m_file_path', 'reference_pdb', 'quantile']):
            raise ValueError("a3m_file_path, reference_pdb, and quantile are required when running bootstrap shuffling.")
    
    if args.run_recursive_clustering:
        if any(getattr(args, attr) is None for attr in ['min_seq_id_initial', 
                                                        'min_seq_id_cutoff', 
                                                        'coverage',
                                                        'a3m_file_path',
                                                        'reference_pdb'
                                                        ]):
            raise ValueError("min_seq_id_initial, min_seq_id_cutoff, coverage, a3m_file_path, and reference_pdb are required when running recursive clustering.")
        if not args.a3m_file_path:
            raise ValueError("a3m_file_path is required when running recursive clustering.")

    if args.run_iteration_shuffles:
        if args.iterations <= 0:
            raise ValueError("iterations must be greater than 0 when running iteration shuffles.")

    logging.info("Arguments successfully validated! Moving on...")

    return True

def log_parameters(args):
    logging.info("==  Welcome to AF_Path ==")
    logging.info("=== Script parameters ===")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")
    logging.info("=" * 30)

def check_file_exists(file_path, description):
    if os.path.isfile(file_path):
        return True 

    dir_path = os.path.dirname(file_path)
    if dir_path and not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Directory for {description} not found: {dir_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{description} not found: {file_path}")
