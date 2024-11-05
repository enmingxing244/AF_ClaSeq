import logging
import os

from Bio import SeqIO
from Bio.PDB import PDBParser, PPBuilder
from typing import List, Dict, Any


def read_a3m_to_dict(a3m_file_path: str) -> Dict[str, str]:
    """
    Reads an A3M file and returns a dictionary mapping headers to sequences.

    Args:
        a3m_file_path (str): Path to the A3M file.

    Returns:
        Dict[str, str]: A dictionary with headers as keys and sequences as values.
    """
    try:
        sequences = {}
        with open(a3m_file_path, 'r') as file:
            current_header = ''
            for line in file:
                line = line.strip()
                if line.startswith('>'):
                    if '\t' in line:
                        current_header = line.split('\t')[0]
                    else:
                        current_header = line.split(' ')[0]
                    sequences[current_header] = ''
                else:
                    sequences[current_header] += ''.join([char for char in line if not char.islower()])
        return sequences
    except Exception as e:
        logging.error(f"Error reading A3M file: {e}")
        raise


def write_a3m(sequences: Dict[str, str], file_path: str, reference_pdb: str) -> None:
    """
    Writes sequences to an A3M file, attaching the reference protein sequence.

    Args:
        sequences (Dict[str, str]): Dictionary of sequences to write.
        file_path (str): Path to the output A3M file.
        reference_pdb (str): Reference PDB identifier.
    """
    try:
        protein_sequence = get_protein_sequence(reference_pdb)
        with open(file_path, 'w') as a3m_file:
            a3m_file.write('>101\n')
            a3m_file.write(protein_sequence + '\n')
            for header, sequence in sequences.items():
                a3m_file.write(f'{header}\n{sequence}\n')
    except Exception as e:
        logging.error(f"Error writing A3M file: {e}")
        raise


def filter_a3m_by_coverage(sequences: Dict[str, str], 
                           coverage_threshold: float = 0.8) -> Dict[str, str]:
    """
    Filter sequences based on coverage compared to the query sequence.
    
    Args:
        sequences (Dict[str, str]): Dictionary of sequences
        coverage_threshold (float): Minimum required coverage (default: 0.8 or 80%)
    
    Returns:
        Dict[str, str]: Filtered sequences dictionary
    """
    try:
        # Get the query sequence (first sequence)
        query_seq = next(iter(sequences.values()))
        query_length = len(query_seq)
        
        # Filter sequences based on coverage
        filtered_sequences = {}
        for header, seq in sequences.items():
            gap_count = seq.count('-')
            coverage = 1 - (gap_count / query_length)
            
            if coverage >= coverage_threshold:
                filtered_sequences[header] = seq
                
        return filtered_sequences
    except Exception as e:
        logging.error(f"Error filtering sequences by coverage: {e}")
        raise



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