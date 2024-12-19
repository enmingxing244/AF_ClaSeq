import os
import argparse
import logging
import shutil
from pathlib import Path
import random
from typing import List, Dict
import subprocess
from af_vote.sequence_processing import read_sequences, process_sequences, get_protein_sequence

def parse_args():
    parser = argparse.ArgumentParser(description='Recompile sequences from specific bin and prepare for prediction')
    parser.add_argument('--output_dir', required=True,
                      help='Output directory for recompiled sequences')
    parser.add_argument('--bin_numbers', type=int, nargs='+', required=True,
                      help='List of bin numbers to compile sequences from (e.g. 4 5 6 7 8 9)')
    parser.add_argument('--source_msa', required=True,
                      help='Path to source MSA file')
    parser.add_argument('--num_shuffles', type=int, default=8,
                      help='Number of shuffled groups to create (creates shuffle_1 to shuffle_40 dirs)')
    parser.add_argument('--default_pdb', required=True,
                      help='Path to reference PDB file for getting query sequence')
    parser.add_argument('--seq_num_per_shuffle', type=int, default=6,
                      help='Number of sequences per shuffle (creates group_1 to group_N in each shuffle dir)')
    parser.add_argument('--voting_results_path', required=True,
                      help='Path to voting results CSV file')
    return parser.parse_args()

def setup_directories(output_dir: str) -> Dict[str, str]:
    """
    Create necessary directories for output
    
    Output structure:
    output_dir/
        prediction/
            bin4_sequences.a3m
            bin5_sequences.a3m
            ...
        shuffling/
            bin4/
                shuffle_1/
                    group_1.a3m
                    group_2.a3m
                    ...
                    shuffle_1.shuf
                shuffle_2/
                    group_1.a3m
                    ...
                    shuffle_2.shuf
                ...
                shuffle_40/
            bin5/
                shuffle_1/
                ...
            ...
            bin9/
    """
    prediction_dir = os.path.join(output_dir, "prediction")
    shuffling_dir = os.path.join(output_dir, "shuffling")
    
    for dir_path in [output_dir, prediction_dir, shuffling_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    return {
        "prediction": prediction_dir,
        "shuffling": shuffling_dir
    }

def get_bin_headers(voting_results: str, 
                    bin_number: int) -> List[str]:
    """Extract headers for sequences in specified bin"""
    bin_headers = []
    with open(voting_results) as f:
        next(f)  # Skip header line
        for line in f:
            parts = line.strip().split(',')
            header = parts[0]
            bin_num = int(parts[1])
            if bin_num == bin_number:
                bin_headers.append(header)
    return bin_headers

def compile_sequences(source_msa: str, 
                      bin_headers: List[str], 
                      output_file: str, 
                      default_pdb: str) -> None:
    """Extract and compile sequences from source MSA"""
    # Get query sequence from PDB file
    query_seq = get_protein_sequence(default_pdb)
    query_header = ">seq"
    
    # Extract sequences for selected bin headers
    bin_seqs = {}
    current_header = ""
    current_seq = ""

    with open(source_msa) as f:
        for line in f:
            if line.startswith(">"):
                if current_header and current_header[1:] in bin_headers:
                    bin_seqs[current_header] = current_seq
                current_header = line.strip()
                current_seq = ""
            else:
                current_seq += line.strip()
        
        # Handle last sequence
        if current_header and current_header[1:] in bin_headers:
            bin_seqs[current_header] = current_seq

    # Write compiled sequences
    with open(output_file, "w") as f:
        f.write(f"{query_header}\n{query_seq}\n")
        for header, seq in bin_seqs.items():
            f.write(f"{header}\n{seq}\n")

def create_shuffled_groups(input_msa: str, 
                           output_dir: str, 
                           num_shuffles: int,
                           seq_num_per_shuffle: int) -> None:
    """
    Create shuffled groups of sequences using process_sequences from af_vote.sequence_processing
    
    For each bin directory (e.g. bin4/):
        Creates shuffle_1/ to shuffle_40/ subdirectories
        In each shuffle_X/:
            Creates:
            - group_1.a3m to group_N.a3m files containing subsets of sequences
            - shuffle_X.shuf containing all sequences for this shuffle
    """
    # Read all sequences from input MSA
    sequences = read_sequences(input_msa)
    
    # Get query sequence (first sequence)
    query_seq = sequences[0].split('\n')[1]
    
    # Create shuffled groups using process_sequences
    for i in range(1, num_shuffles + 1):
        process_sequences(
            dir_path=output_dir,
            sequences=sequences[1:], # Skip query sequence
            shuffle_num=i,
            seq_num_per_shuffle=seq_num_per_shuffle,
            protein_sequence=query_seq
        )

def main():
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s - %(levelname)s - %(message)s')

    # Create directories
    dirs = setup_directories(args.output_dir)
    
    # Process each bin number (e.g. bins 4-9)
    for bin_number in args.bin_numbers:
        logging.info(f"Processing bin {bin_number}")
        
        # Get headers for current bin
        logging.info(f"Extracting headers for bin {bin_number}")
        bin_headers = get_bin_headers(args.voting_results_path, bin_number)
        
        # Compile sequences for prediction
        prediction_file = os.path.join(dirs["prediction"], f"bin{bin_number}_sequences.a3m")
        logging.info(f"Compiling sequences for prediction")
        compile_sequences(args.source_msa, 
                          bin_headers, 
                          prediction_file, 
                          args.default_pdb)
        
        # Create shuffled groups under bin directory
        shuffling_bin_dir = os.path.join(dirs["shuffling"], f"bin{bin_number}")
        os.makedirs(shuffling_bin_dir, exist_ok=True)
        
        logging.info(f"Creating {args.num_shuffles} shuffle directories with sequence groups")
        create_shuffled_groups(prediction_file, 
                               shuffling_bin_dir, 
                               args.num_shuffles, 
                               args.seq_num_per_shuffle)
        
        logging.info(f"Completed processing bin {bin_number}")
    
    logging.info("Completed recompilation and shuffling for all bins")

if __name__ == "__main__":
    main()