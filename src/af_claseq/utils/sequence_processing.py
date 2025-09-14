import os
import random
import re
from Bio import SeqIO
from Bio.PDB import PDBParser, PPBuilder
from typing import List, Dict, Any, Tuple, Set, Optional
from pathlib import Path
from collections import defaultdict

from af_claseq.utils.logging_utils import get_logger

# Initialize module logger
logger = get_logger("sequence_processing")

def get_query_sequence_from_a3m(a3m_file: str) -> Tuple[str, str]:
    """
    Extract the first header and sequence (query sequence) from an A3M file.

    Args:
        a3m_file: Path to the source A3M file

    Returns:
        Tuple of (header, sequence) for the query sequence

    Raises:
        FileNotFoundError: If A3M file doesn't exist
        ValueError: If A3M file is empty or malformed
    """
    if not os.path.exists(a3m_file):
        raise FileNotFoundError(f"Source A3M file not found: {a3m_file}")

    try:
        with open(a3m_file, 'r') as f:
            lines = f.readlines()

        if not lines:
            raise ValueError(f"Source A3M file is empty: {a3m_file}")

        # Find first header line
        header = None
        sequence_lines = []
        in_sequence = False

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith('>'):
                if header is not None:  # We found the first sequence already
                    break
                header = line[1:]  # Remove '>'
                in_sequence = True
            elif in_sequence and header is not None:
                sequence_lines.append(line)

        if header is None:
            raise ValueError(f"No valid sequence found in A3M file: {a3m_file}")

        sequence = ''.join(sequence_lines)
        if not sequence:
            raise ValueError(f"Query sequence is empty in A3M file: {a3m_file}")

        return header, sequence

    except Exception as e:
        logger.error(f"Error extracting query sequence from {a3m_file}: {e}")
        raise

def read_a3m_to_dict(a3m_file_path: str) -> Dict[str, str]:
    """
    Reads an A3M file and returns a dictionary mapping headers to sequences.

    Args:
        a3m_file_path (str): Path to the A3M file.

    Returns:
        Dict[str, str]: A dictionary with headers as keys and sequences as values.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        Exception: For any other errors during file processing.
    """
    try:
        sequences = {}
        current_header = None
        
        with open(a3m_file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                if line.startswith('>'):
                    # Process header line - take first part before space or tab
                    current_header = line.split()[0] if ' ' in line else line.split('\t')[0] if '\t' in line else line
                    sequences[current_header] = ''
                elif current_header is not None:
                    # Process sequence line - filter out lowercase letters (insertions)
                    sequences[current_header] += ''.join(char for char in line if not char.islower())
                    
        return sequences
    except FileNotFoundError:
        logger.error(f"File not found: {a3m_file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading A3M file: {e}")
        raise

def write_a3m(sequences: Dict[str, str],
               file_path: str,
               source_a3m: Optional[str] = None,
               prepend_query: bool = False) -> None:
    """
    Write sequences to an A3M file, optionally prepending query sequence from source A3M.

    Args:
        sequences: Dictionary of sequences to write
        file_path: Path to the output A3M file
        source_a3m: Source A3M file to extract query sequence from (if prepend_query=True)
        prepend_query: Whether to prepend query sequence from source_a3m

    Raises:
        FileNotFoundError: If source_a3m doesn't exist when prepend_query=True
        ValueError: If prepend_query=True but source_a3m is None
        Exception: If there's an error writing the file
    """
    try:
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w') as a3m_file:
            # Handle query sequence prepending
            if prepend_query:
                if source_a3m is None:
                    raise ValueError("source_a3m must be provided when prepend_query=True")

                # Extract query sequence from source A3M
                query_header, query_sequence = get_query_sequence_from_a3m(source_a3m)
                a3m_file.write(f'>{query_header}\n')
                a3m_file.write(f"{query_sequence}\n")

            # Write all other sequences
            for header, sequence in sequences.items():
                # Ensure header starts with '>' if not already
                header_line = header if header.startswith('>') else f'>{header}'
                a3m_file.write(f'{header_line}\n{sequence}\n')

    except Exception as e:
        logger.error(f"Error writing A3M file: {e}")
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
    
    Raises:
        ValueError: If sequences dictionary is empty
        Exception: For any other errors during processing
    """
    if not sequences:
        raise ValueError("Empty sequences dictionary provided")
        
    # Get the query sequence (first sequence)
    query_seq = next(iter(sequences.values()))
    query_length = len(query_seq)
    
    # Filter sequences based on coverage
    filtered_sequences = {
        header: seq for header, seq in sequences.items() 
        if (1 - (seq.count('-') / query_length)) >= coverage_threshold
    }
            
    return filtered_sequences

def get_protein_sequence(pdb_filename: str) -> str:
    """
    Extract the protein sequence from a PDB file.

    Args:
        pdb_filename (str): Path to the PDB file.

    Returns:
        str: The full protein sequence.

    Raises:
        FileNotFoundError: If the PDB file does not exist.
        Exception: If there's an error in processing the PDB file.
    """
    if not os.path.exists(pdb_filename):
        raise FileNotFoundError(f"PDB file not found: {pdb_filename}")
        
    pdb_parser = PDBParser(QUIET=True)
    structure = pdb_parser.get_structure("Protein", pdb_filename)
    ppb = PPBuilder()
    
    # Extract sequences from all peptides
    sequences = [str(pp.get_sequence()) for pp in ppb.build_peptides(structure)]
    
    # Join all sequences
    full_sequence = ''.join(sequences)
    
    if not full_sequence:
        logger.warning(f"No protein sequence found in {pdb_filename}")
        
    return full_sequence

def map_and_extract(headers: List[str], sequences: Dict[str, str]) -> Dict[str, str]:
    """
    Maps and extracts sequences based on provided headers.

    Args:
        headers (List[str]): List of header identifiers.
        sequences (Dict[str, str]): Dictionary of available sequences.

    Returns:
        Dict[str, str]: Extracted sequences matching the headers.
        
    Raises:
        Exception: For any errors during processing
    """
    # Use dictionary comprehension to efficiently extract matching sequences
    extracted = {header: sequences[header] for header in headers if header in sequences}
    
    # Log if some headers weren't found
    missing_count = len(set(headers) - set(extracted.keys()))
    if missing_count:
        logger.warning(f"Could not find sequences for {missing_count} headers")
        
    return extracted

def combine_sequences(extracted_sequences: Dict[str, str], output_file: str) -> None:
    """
    Combines and writes extracted sequences to an output file.

    Args:
        extracted_sequences (Dict[str, str]): Dictionary of extracted sequences.
        output_file (str): Path to the output file.
        
    Raises:
        Exception: For any errors during file writing
    """
    # Ensure output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        for header, seq in extracted_sequences.items():
            f.write(f">{header}\n{seq}\n")
            
    logger.info(f"Successfully wrote {len(extracted_sequences)} sequences to {output_file}")

def count_sequences_in_a3m(a3m_file: str) -> int:
    """
    Count the number of sequences in an A3M file.

    Args:
        a3m_file (str): Path to the A3M file.

    Returns:
        int: The number of sequences in the A3M file.
        
    Raises:
        FileNotFoundError: If the A3M file doesn't exist (logged but not raised)
    """
    if not os.path.exists(a3m_file):
        logger.error(f"A3M file not found: {a3m_file}")
        return 0
        
    try:
        count = 0
        with open(a3m_file, 'r') as file:
            for line in file:
                if line.startswith('>'):
                    count += 1
                    
        logger.info(f"Found {count} sequences in {a3m_file}")
        return count
    except Exception as e:
        logger.error(f"Error counting sequences in A3M file: {e}")
        return 0

