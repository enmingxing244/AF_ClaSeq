import logging
import os
import random
from typing import Dict, List

from af_vote.sequence_processing import read_a3m_to_dict, write_a3m
from af_vote.slurm_utils import SlurmJobSubmitter

def split_into_groups(sequences: Dict[str, str], 
                     group_size: int) -> List[Dict[str, str]]:
    """Split sequences into groups of specified size.
    
    Args:
        sequences: Dictionary mapping sequence headers to sequences
        group_size: Size of each group, specified via command line args
        
    Returns:
        List of dictionaries, where each dictionary contains a group of sequences
    """
    groups = []
    headers = list(sequences.keys())
    
    for i in range(0, len(headers), group_size):
        group = {}
        group_headers = headers[i:i + group_size]
        
        if len(group_headers) < group_size and groups:
            groups[-1].update({h: sequences[h] for h in group_headers})
        else:
            group = {h: sequences[h] for h in group_headers}
            groups.append(group)
            
    return groups

def initial_random_split(input_a3m: str, 
                        output_dir: str,
                        reference_pdb: str,
                        group_size: int) -> int:  # Added missing group_size parameter
    """Perform initial random split of sequences.
    
    Args:
        input_a3m: Path to input A3M file
        output_dir: Directory to write output files
        reference_pdb: Path to reference PDB file
        group_size: Size of each sequence group
        
    Returns:
        Number of groups created
    """
    os.makedirs(output_dir, exist_ok=True)
    sequences = read_a3m_to_dict(input_a3m)
    print(f"Number of sequences: {len(sequences)}")
    
    headers = list(sequences.keys())
    random.shuffle(headers)
    shuffled_sequences = {h: sequences[h] for h in headers}
    
    groups = split_into_groups(shuffled_sequences, group_size)  # Pass group_size
    
    for i, group in enumerate(groups, 1):
        output_file = os.path.join(output_dir, f'group_{i}.a3m')
        write_a3m(group, output_file, reference_pdb=reference_pdb)
    
    return len(groups)

def create_sampling_splits(init_dir: str, 
                         output_base_dir: str, 
                         reference_pdb: str,
                         group_size: int):  # Added missing group_size parameter
    """Create sampling splits by exhaustively leaving one group out at a time.
    
    Args:
        init_dir: Directory containing initial groups
        output_base_dir: Base directory for output sampling splits
        reference_pdb: Path to reference PDB file
        group_size: Size of each sequence group
    """
    groups = [f for f in os.listdir(init_dir) if f.endswith('.a3m')]
    num_groups = len(groups)
    
    for i in range(num_groups):
        sample_dir = os.path.join(output_base_dir, f'sampling_{i+1}')
        os.makedirs(sample_dir, exist_ok=True)
        
        all_sequences = {}
        for j, group in enumerate(groups):
            if j != i:
                group_path = os.path.join(init_dir, group)
                group_sequences = read_a3m_to_dict(group_path)
                all_sequences.update(group_sequences)
        
        headers = list(all_sequences.keys())
        random.shuffle(headers)
        shuffled_sequences = {h: all_sequences[h] for h in headers}
        new_groups = split_into_groups(shuffled_sequences, group_size)  # Pass group_size
        
        for j, group in enumerate(new_groups, 1):
            output_file = os.path.join(sample_dir, f'group_{j}.a3m')
            write_a3m(group, output_file, reference_pdb=reference_pdb)

# def process_iterations(base_dir: str, 
#                       conda_env_path: str,
#                       slurm_account: str,
#                       slurm_output: str,
#                       slurm_error: str, 
#                       slurm_nodes: int,
#                       slurm_gpus_per_task: int,
#                       slurm_tasks: int,
#                       slurm_cpus_per_task: int, 
#                       slurm_time: str,
#                       random_seed: int,
#                       num_models: int,
#                       check_interval: int,
#                       max_workers: int):
#     """Process all sampling directories with SLURM jobs.
    
#     Args:
#         base_dir: Base directory containing sampling splits
#         conda_env_path: Path to conda environment
#         slurm_account: SLURM account name
#         slurm_output: Path for SLURM output files
#         slurm_error: Path for SLURM error files
#         slurm_nodes: Number of nodes per job
#         slurm_gpus_per_task: GPUs per task
#         slurm_tasks: Number of tasks per job
#         slurm_cpus_per_task: CPUs per task
#         slurm_time: Wall time limit
#         random_seed: Random seed for reproducibility
#         num_models: Number of models to generate
#         check_interval: Interval to check job status
#         max_workers: Maximum concurrent workers
#     """
#     slurm_submitter = SlurmJobSubmitter(
#         conda_env_path=conda_env_path,
#         slurm_account=slurm_account,
#         slurm_output=slurm_output,
#         slurm_error=slurm_error,
#         slurm_nodes=slurm_nodes,
#         slurm_gpus_per_task=slurm_gpus_per_task,
#         slurm_tasks=slurm_tasks,
#         slurm_cpus_per_task=slurm_cpus_per_task,
#         slurm_time=slurm_time,
#         random_seed=random_seed,
#         num_models=num_models,
#         check_interval=check_interval
#     )
    
#     # Get both sampling and shuffle directories
#     sampling_dirs = [d for d in os.listdir(base_dir) if d.startswith('sampling_')]
#     shuffle_dirs = [d for d in os.listdir(base_dir) if d.startswith('shuffle_')]
#     all_dirs = sampling_dirs + shuffle_dirs
    
#     # Create job folder names
#     job_folders = []
#     for i, d in enumerate(all_dirs):
#         if d.startswith('sampling_'):
#             job_folders.append(f"sample_{i+1}")
#         else:
#             job_folders.append(f"shuffle_{i+1}")
    
#     # Get full paths
#     all_paths = [os.path.join(base_dir, d) for d in all_dirs]
#     slurm_submitter.process_folders_concurrently(all_paths, job_folders, max_workers=max_workers)