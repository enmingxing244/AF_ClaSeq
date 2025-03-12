import logging
import os
import random
from Bio import SeqIO
from Bio.PDB import PDBParser, PPBuilder
from typing import List, Dict, Any, Tuple, Set, Optional
from pathlib import Path
from collections import defaultdict

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
                    # Process header line
                    if '\t' in line or ' ' in line:
                        current_header = line.split('\t')[0] if '\t' in line else line.split()[0]
                    else:
                        current_header = line
                    sequences[current_header] = ''
                elif current_header is not None:
                    # Process sequence line - filter out lowercase letters (insertions)
                    sequences[current_header] += ''.join(char for char in line if not char.islower())
                    
        return sequences
    except FileNotFoundError:
        logging.error(f"File not found: {a3m_file_path}")
        raise
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

    Raises:
        Exception: If there's an error writing the file or getting the protein sequence.
    """
    try:
        protein_sequence = get_protein_sequence(reference_pdb)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        with open(file_path, 'w') as a3m_file:
            # Write reference sequence first
            a3m_file.write('>101\n')
            a3m_file.write(f"{protein_sequence}\n")
            
            # Write all other sequences
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
    
    Raises:
        ValueError: If sequences dictionary is empty
        Exception: For any other errors during processing
    """
    try:
        if not sequences:
            raise ValueError("Empty sequences dictionary provided")
            
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
    except ValueError as ve:
        logging.error(f"Value error in filter_a3m_by_coverage: {ve}")
        raise
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
        FileNotFoundError: If the PDB file does not exist.
        Exception: If there's an error in processing the PDB file.
    """
    try:
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
            logging.warning(f"No protein sequence found in {pdb_filename}")
            
        return full_sequence

    except FileNotFoundError as fnf:
        logging.error(f"PDB file not found: {fnf}")
        raise
    except Exception as e:
        logging.error(f"Error getting protein sequence: {e}")
        raise

def convert_fasta_to_a3m(fasta_file: str, a3m_file: str, output_file: str) -> None:
    """
    Converts FASTA files to A3M format by mapping and extracting relevant sequences.

    Args:
        fasta_file (str): Path to the FASTA file.
        a3m_file (str): Path to the A3M file.
        output_file (str): Path to the output A3M file.
        
    Raises:
        FileNotFoundError: If input files don't exist
        Exception: For any other errors during processing
    """
    try:
        # Check if input files exist
        for file_path in [fasta_file, a3m_file]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Input file not found: {file_path}")
        
        # Process files
        fasta_headers = read_fasta_headers(fasta_file)
        a3m_sequences = read_a3m_sequences(a3m_file)
        extracted_sequences = map_and_extract(fasta_headers, a3m_sequences)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Write output
        combine_sequences(extracted_sequences, output_file)
        
        logging.info(f"Successfully converted {fasta_file} to A3M format at {output_file}")
    except FileNotFoundError as fnf:
        logging.error(f"File not found: {fnf}")
        raise
    except Exception as e:
        logging.error(f"Error converting FASTA to A3M: {e}")
        raise

def read_fasta_headers(fasta_file: str) -> List[str]:
    """
    Reads headers from a FASTA file.

    Args:
        fasta_file (str): Path to the FASTA file.

    Returns:
        List[str]: A list of header identifiers.
        
    Raises:
        FileNotFoundError: If the FASTA file doesn't exist
        Exception: For any other errors during processing
    """
    try:
        if not os.path.exists(fasta_file):
            raise FileNotFoundError(f"FASTA file not found: {fasta_file}")
            
        headers = [record.id for record in SeqIO.parse(fasta_file, "fasta")]
        return headers
    except FileNotFoundError as fnf:
        logging.error(f"FASTA file not found: {fnf}")
        raise
    except Exception as e:
        logging.error(f"Error reading FASTA headers: {e}")
        raise

def read_a3m_sequences(a3m_file: str) -> Dict[str, str]:
    """
    Reads sequences from an A3M file.

    Args:
        a3m_file (str): Path to the A3M file.

    Returns:
        Dict[str, str]: A dictionary with headers as keys and sequences as values.
        
    Raises:
        FileNotFoundError: If the A3M file doesn't exist
        Exception: For any other errors during processing
    """
    try:
        if not os.path.exists(a3m_file):
            raise FileNotFoundError(f"A3M file not found: {a3m_file}")
            
        sequences = {record.id: str(record.seq) for record in SeqIO.parse(a3m_file, "fasta")}
        return sequences
    except FileNotFoundError as fnf:
        logging.error(f"A3M file not found: {fnf}")
        raise
    except Exception as e:
        logging.error(f"Error reading A3M sequences: {e}")
        raise

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
    try:
        # Use dictionary comprehension to efficiently extract matching sequences
        extracted = {header: sequences[header] for header in headers if header in sequences}
        
        # Log if some headers weren't found
        missing_headers = set(headers) - set(extracted.keys())
        if missing_headers:
            logging.warning(f"Could not find sequences for {len(missing_headers)} headers")
            
        return extracted
    except Exception as e:
        logging.error(f"Error mapping and extracting sequences: {e}")
        raise

def combine_sequences(extracted_sequences: Dict[str, str], output_file: str) -> None:
    """
    Combines and writes extracted sequences to an output file.

    Args:
        extracted_sequences (Dict[str, str]): Dictionary of extracted sequences.
        output_file (str): Path to the output file.
        
    Raises:
        Exception: For any errors during file writing
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        with open(output_file, "w") as f:
            for header, seq in extracted_sequences.items():
                f.write(f">{header}\n{seq}\n")
                
        logging.info(f"Successfully wrote {len(extracted_sequences)} sequences to {output_file}")
    except Exception as e:
        logging.error(f"Error combining sequences: {e}")
        raise

def process_sequences(
    dir_path: str,
    sequences: List[str],
    shuffle_num: int,
    seq_num_per_shuffle: int,
    protein_sequence: str
) -> None:
    """
    Processes and writes shuffled sequences into separate files and groups.

    Args:
        dir_path (str): Directory to store shuffled files.
        sequences (List[str]): List of sequences to process.
        shuffle_num (int): Shuffle iteration number.
        seq_num_per_shuffle (int): Number of sequences per shuffle.
        protein_sequence (str): Reference protein sequence.
        
    Raises:
        ValueError: If sequences list is empty
        Exception: For any other errors during processing
    """
    try:
        if not sequences:
            raise ValueError("Empty sequences list provided")
            
        # Create a copy to avoid modifying the original list
        sequences_copy = sequences.copy()
        random.shuffle(sequences_copy)
        
        # If total sequences is less than seq_num_per_shuffle, use all sequences in one group
        actual_seq_per_shuffle = min(seq_num_per_shuffle, len(sequences_copy))
        
        # Create groups of sequences
        groups = [
            sequences_copy[x:x + actual_seq_per_shuffle]
            for x in range(0, len(sequences_copy), actual_seq_per_shuffle)
        ]
        
        # Create shuffle directory
        shuffle_dir = os.path.join(dir_path, f'shuffle_{shuffle_num}')
        os.makedirs(shuffle_dir, exist_ok=True)

        # Write all sequences to a single shuffle file
        shuffle_file_path = os.path.join(shuffle_dir, f'shuffle_{shuffle_num}.shuf')
        with open(shuffle_file_path, 'w') as f:
            for seq in sequences_copy:
                f.write(seq)

        # Write each group to a separate A3M file
        for i, group in enumerate(groups, start=1):
            group_file_path = os.path.join(shuffle_dir, f'group_{i}.a3m')
            with open(group_file_path, 'w') as g:
                g.write('>101\n')
                g.write(f"{protein_sequence}\n")
                for seq in group:
                    g.write(seq)
                    
        logging.info(f"Successfully processed {len(sequences_copy)} sequences into {len(groups)} groups")
    except ValueError as ve:
        logging.error(f"Value error in process_sequences: {ve}")
        raise
    except Exception as e:
        logging.error(f"Error processing sequences: {e}")
        raise

def process_all_sequences(
    dir_path: str,
    file_path: str,
    num_shuffles: int,
    seq_num_per_shuffle: int,
    reference_pdb: str
) -> None:
    """
    Processes all sequences by reading, shuffling, and writing them.

    Args:
        dir_path (str): Directory to store shuffled files.
        file_path (str): Path to the input sequence file.
        num_shuffles (int): Number of shuffles to perform.
        seq_num_per_shuffle (int): Number of sequences per shuffle.
        reference_pdb (str): Reference PDB identifier.
        
    Raises:
        FileNotFoundError: If input files don't exist
        ValueError: If no sequences are found
        Exception: For any other errors during processing
    """
    try:
        # Check if input files exist
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")
        if not os.path.exists(reference_pdb):
            raise FileNotFoundError(f"Reference PDB file not found: {reference_pdb}")
            
        # Ensure output directory exists
        os.makedirs(dir_path, exist_ok=True)
        
        # Read sequences from A3M file
        sequences_dict = read_a3m_to_dict(file_path)
        
        if not sequences_dict:
            raise ValueError(f"No sequences found in {file_path}")
            
        # Convert dictionary to list of formatted sequences
        sequences = [f"{header}\n{seq}\n" for header, seq in sequences_dict.items()]
        
        # Get protein sequence from reference PDB
        protein_sequence = get_protein_sequence(reference_pdb)
        
        # Process sequences for each shuffle
        for i in range(1, num_shuffles + 1):
            process_sequences(dir_path, sequences, i, seq_num_per_shuffle, protein_sequence)
            
        logging.info(f"Successfully processed {len(sequences)} sequences for {num_shuffles} shuffles")
    except FileNotFoundError as fnf:
        logging.error(f"File not found: {fnf}")
        raise
    except ValueError as ve:
        logging.error(f"Value error in process_all_sequences: {ve}")
        raise
    except Exception as e:
        logging.error(f"Error processing all sequences: {e}")
        raise

def collect_a3m_files(df_list: List[Dict[str, str]]) -> List[str]:
    """
    Collects A3M file paths from a list of dataframes.

    Args:
        df_list (List[Dict[str, str]]): List of dataframes containing PDB information.

    Returns:
        List[str]: List of A3M file paths.
        
    Raises:
        ValueError: If df_list is empty or invalid
        Exception: For any other errors during processing
    """
    try:
        if not df_list:
            raise ValueError("Empty dataframe list provided")
            
        a3m_list = []
        
        for i, df in enumerate(df_list):
            if 'PDB' not in df:
                logging.warning(f"DataFrame at index {i} does not contain 'PDB' column, skipping")
                continue
                
            logging.info(f'Processing DataFrame {i+1}/{len(df_list)}')
            
            for pdb in df['PDB']:
                if not isinstance(pdb, str):
                    logging.warning(f"Skipping non-string PDB entry: {pdb}")
                    continue
                    
                a3m = pdb.split('_unrelaxed')[0] + '.a3m'
                a3m_list.append(a3m)
                logging.info(f"Added A3M file: {a3m}")
                
        logging.info(f"Collected {len(a3m_list)} A3M files")
        return a3m_list
    except ValueError as ve:
        logging.error(f"Value error in collect_a3m_files: {ve}")
        raise
    except Exception as e:
        logging.error(f"Error collecting A3M files: {e}")
        raise

def concatenate_a3m_content(
    a3m_list: List[str],
    reference_pdb: str,
    a3m_path: str
) -> None:
    """
    Concatenates content from multiple A3M files into a single file.

    Args:
        a3m_list (List[str]): List of A3M file paths.
        reference_pdb (str): Reference PDB identifier.
        a3m_path (str): Path to the output concatenated A3M file.
        
    Raises:
        FileNotFoundError: If reference PDB file doesn't exist
        ValueError: If a3m_list is empty
        Exception: For any other errors during processing
    """
    try:
        if not a3m_list:
            raise ValueError("Empty A3M file list provided")
            
        if not os.path.exists(reference_pdb):
            raise FileNotFoundError(f"Reference PDB file not found: {reference_pdb}")
            
        # Get reference protein sequence
        query = get_protein_sequence(reference_pdb)
        
        # Track unique entries to avoid duplicates
        seen_entries = set()
        concatenated_content = []
        
        # Process each A3M file
        for file_name in a3m_list:
            try:
                if not os.path.exists(file_name):
                    logging.warning(f"File not found, skipping: {file_name}")
                    continue
                    
                with open(file_name, "r") as file:
                    current_header = None
                    current_sequence = ""
                    
                    for line in file:
                        line = line.strip()
                        if not line:
                            continue
                            
                        if line.startswith('>') and not line.startswith('>101') and not line.startswith(query) and not line.startswith('#'):
                            # Process previous entry if exists
                            if current_header and current_sequence:
                                entry = (current_header, current_sequence)
                                if entry not in seen_entries:
                                    concatenated_content.append(f"{current_header}\n{current_sequence}\n")
                                    seen_entries.add(entry)
                            
                            # Start new entry
                            current_header = line
                            current_sequence = ""
                        elif current_header:
                            # Add to current sequence, filtering out lowercase letters
                            current_sequence = "".join([char for char in line if char.isupper() or char == '-'])
                    
                    # Process the last entry in the file
                    if current_header and current_sequence:
                        entry = (current_header, current_sequence)
                        if entry not in seen_entries:
                            concatenated_content.append(f"{current_header}\n{current_sequence}\n")
                            seen_entries.add(entry)
                            
            except FileNotFoundError:
                logging.warning(f"File not found, skipping: {file_name}")
            except Exception as e:
                logging.warning(f"Error processing file {file_name}: {e}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(a3m_path)), exist_ok=True)
        
        # Write concatenated content to output file
        try:
            with open(a3m_path, "w") as output_file:
                output_file.writelines(concatenated_content)
                
            logging.info(f"Successfully wrote {len(seen_entries)} unique sequences to {a3m_path}")
        except Exception as e:
            logging.error(f"Error writing concatenated A3M file: {e}")
            raise
            
    except FileNotFoundError as fnf:
        logging.error(f"File not found: {fnf}")
        raise
    except ValueError as ve:
        logging.error(f"Value error in concatenate_a3m_content: {ve}")
        raise
    except Exception as e:
        logging.error(f"Error concatenating A3M content: {e}")
        raise

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
    count = 0
    try:
        if not os.path.exists(a3m_file):
            logging.error(f"A3M file not found: {a3m_file}")
            return 0
            
        with open(a3m_file, 'r') as file:
            for line in file:
                if line.startswith('>'):
                    count += 1
                    
        logging.info(f"Found {count} sequences in {a3m_file}")
        return count
    except FileNotFoundError:
        logging.error(f"A3M file not found: {a3m_file}")
        return 0
    except Exception as e:
        logging.error(f"Error counting sequences in A3M file: {e}")
        return 0