import logging
import os
import random
from typing import Dict, List, Optional

from af_claseq.sequence_processing import read_a3m_to_dict, write_a3m
from af_claseq.slurm_utils import SlurmJobSubmitter

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
                        group_size: int,
                        random_select_num_seqs: Optional[int] = None) -> int:
    """Perform initial random split of sequences.
    
    Args:
        input_a3m: Path to input A3M file
        output_dir: Directory to write output files
        reference_pdb: Path to reference PDB file
        group_size: Size of each sequence group
        random_select_num_seqs: Number of sequences to randomly select, if specified
        
    Returns:
        Number of groups created
    """
    os.makedirs(output_dir, exist_ok=True)
    sequences = read_a3m_to_dict(input_a3m)
    logging.info(f"Number of sequences: {len(sequences)}")
    
    headers = list(sequences.keys())
    random.shuffle(headers)

    if random_select_num_seqs is not None:
        headers = headers[:random_select_num_seqs]
        logging.info(f"Randomly selected {len(headers)} sequences")
    
    shuffled_sequences = {h: sequences[h] for h in headers}
    
    groups = split_into_groups(shuffled_sequences, group_size)
    
    for i, group in enumerate(groups, 1):
        output_file = os.path.join(output_dir, f'group_{i}.a3m')
        write_a3m(group, output_file, reference_pdb=reference_pdb)
    
    return len(groups)

def create_sampling_splits(init_dir: str, 
                         output_base_dir: str, 
                         reference_pdb: str,
                         group_size: int):
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
        new_groups = split_into_groups(shuffled_sequences, group_size)
        
        for j, group in enumerate(new_groups, 1):
            output_file = os.path.join(sample_dir, f'group_{j}.a3m')
            write_a3m(group, output_file, reference_pdb=reference_pdb)
