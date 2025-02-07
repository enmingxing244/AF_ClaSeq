import logging
import os
import random
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
        current_header = None
        with open(a3m_file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith('>'):
                    if '\t' in line or ' ' in line:
                        current_header = line.split('\t')[0] if '\t' in line else line.split()[0]
                        # current_header = line
                    else:
                        current_header = line
                    sequences[current_header] = ''
                elif current_header is not None:  # Only add sequence if we have a header
                    sequences[current_header] += "".join([char for char in line if not char.islower()])
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


def convert_fasta_to_a3m(fasta_file: str, a3m_file: str, output_file: str) -> None:
    """
    Converts FASTA files to A3M format by mapping and extracting relevant sequences.

    Args:
        fasta_file (str): Path to the FASTA file.
        a3m_file (str): Path to the A3M file.
        output_file (str): Path to the output A3M file.
    """
    try:
        fasta_headers = read_fasta_headers(fasta_file)
        a3m_sequences = read_a3m_sequences(a3m_file)
        extracted_sequences = map_and_extract(fasta_headers, a3m_sequences)
        combine_sequences(extracted_sequences, output_file)
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
    """
    try:
        headers = [record.id for record in SeqIO.parse(fasta_file, "fasta")]
        return headers
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
    """
    try:
        sequences = {record.id: str(record.seq) for record in SeqIO.parse(a3m_file, "fasta")}
        return sequences
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
    """
    try:
        return {header: sequences[header] for header in headers if header in sequences}
    except Exception as e:
        logging.error(f"Error mapping and extracting sequences: {e}")
        raise


def combine_sequences(extracted_sequences: Dict[str, str], output_file: str) -> None:
    """
    Combines and writes extracted sequences to an output file.

    Args:
        extracted_sequences (Dict[str, str]): Dictionary of extracted sequences.
        output_file (str): Path to the output file.
    """
    try:
        with open(output_file, "w") as f:
            for header, seq in extracted_sequences.items():
                f.write(f">{header}\n{seq}\n")
    except Exception as e:
        logging.error(f"Error combining sequences: {e}")
        raise

def read_sequences(file_path: str) -> List[str]:
    """
    Reads all sequences from a file, keeping only uppercase letters and gaps.

    Args:
        file_path (str): Path to the sequence file.

    Returns:
        List[str]: A list of sequence strings with only uppercase letters and gaps.
    """
    try:
        sequences = []
        current_sequence = ""
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('>'):
                    if current_sequence:
                        sequences.append(current_sequence)
                        current_sequence = ""
                    # Split header by space or tab and take first part
                    header = line.strip().split()[0] + '\n'  # split() handles both space and tab
                    current_sequence += header
                else:
                    # Keep only uppercase letters and gaps
                    current_sequence += ''.join([char for char in line if not char.islower()])
            if current_sequence:
                sequences.append(current_sequence)
        return sequences
    except Exception as e:
        logging.error(f"Error reading sequences: {e}")
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
    """
    try:
        random.shuffle(sequences)
        
        # If total sequences is less than seq_num_per_shuffle, use all sequences in one group
        actual_seq_per_shuffle = min(seq_num_per_shuffle, len(sequences))
        
        groups = [
            sequences[x:x + actual_seq_per_shuffle]
            for x in range(0, len(sequences), actual_seq_per_shuffle)
        ]
        
        shuffle_dir = os.path.join(dir_path, f'shuffle_{shuffle_num}')
        os.makedirs(shuffle_dir, exist_ok=True)

        shuffle_file_path = os.path.join(shuffle_dir, f'shuffle_{shuffle_num}.shuf')
        with open(shuffle_file_path, 'w') as f:
            for seq in sequences:
                f.write(seq)

        for i, group in enumerate(groups, start=1):
            group_file_path = os.path.join(shuffle_dir, f'group_{i}.a3m')
            with open(group_file_path, 'w') as g:
                g.write('>101\n')
                g.write(protein_sequence + '\n')
                for seq in group:
                    g.write(seq)
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
    """
    try:
        sequences = read_sequences(file_path)
        protein_sequence = get_protein_sequence(reference_pdb)
        for i in range(1, num_shuffles + 1):
            process_sequences(dir_path, sequences.copy(), i, seq_num_per_shuffle, protein_sequence)
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
    """
    try:
        a3m_list = []
        for df in df_list:
            logging.info('Processing DataFrame')
            for pdb in df['PDB']:
                a3m = pdb.split('_unrelaxed')[0] + '.a3m'
                a3m_list.append(a3m)
                logging.info(a3m)
        return a3m_list
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
    """
    query = get_protein_sequence(reference_pdb)
    concatenated_content = ""
    seen_entries = set()

    for file_name in a3m_list:
        try:
            with open(file_name, "r") as file:
                current_header = None
                for line in file:
                    if line.startswith('>') and not line.startswith('>101') and not line.startswith(query) and not line.startswith('#'):
                        current_header = line.strip()
                    elif current_header:
                        sequence = "".join([char for char in line if char.isupper() or char == '-'])
                        entry = (current_header, sequence)
                        

                        if entry not in seen_entries:
                            concatenated_content += f"{current_header}\n{sequence}\n"
                            seen_entries.add(entry)
                            
        except FileNotFoundError:
            logging.error(f"File not found: {file_name}")
        except Exception as e:
            logging.error(f"Error processing file {file_name}: {e}")

    try:
        with open(a3m_path, "w") as output_file:
            output_file.write(concatenated_content)
    except Exception as e:
        logging.error(f"Error writing concatenated A3M file: {e}")
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