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
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as a3m_file:
            # Write reference sequence first
            a3m_file.write('>101\n')
            a3m_file.write(f"{protein_sequence}\n")
            
            # Write all other sequences
            for header, sequence in sequences.items():
                a3m_file.write(f'{header}\n{sequence}\n')
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
    shuffle_dir = Path(dir_path) / f'shuffle_{shuffle_num}'
    shuffle_dir.mkdir(parents=True, exist_ok=True)

    # Write all sequences to a single shuffle file
    shuffle_file_path = shuffle_dir / f'shuffle_{shuffle_num}.shuf'
    with open(shuffle_file_path, 'w') as f:
        for seq in sequences_copy:
            f.write(seq)

    # Write each group to a separate A3M file
    for i, group in enumerate(groups, start=1):
        group_file_path = shuffle_dir / f'group_{i}.a3m'
        with open(group_file_path, 'w') as g:
            g.write('>101\n')
            g.write(f"{protein_sequence}\n")
            for seq in group:
                g.write(seq)
                
    logger.info(f"Successfully processed {len(sequences_copy)} sequences into {len(groups)} groups")

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
        file_path (str): Path to the input a3m file.
        num_shuffles (int): Number of shuffles to perform.
        seq_num_per_shuffle (int): Number of sequences per shuffle.
        reference_pdb (str): Reference PDB identifier.
        
    Raises:
        FileNotFoundError: If input files don't exist
        ValueError: If no sequences are found
        Exception: For any other errors during processing
    """
    # Check if input files exist
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
    if not os.path.exists(reference_pdb):
        raise FileNotFoundError(f"Reference PDB file not found: {reference_pdb}")
        
    # Ensure output directory exists
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    
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
        
    logger.info(f"Successfully processed {len(sequences)} sequences for {num_shuffles} shuffles")

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
    if not df_list:
        raise ValueError("Empty dataframe list provided")
        
    a3m_list = []
    
    for i, df in enumerate(df_list):
        if 'PDB' not in df:
            logger.warning(f"DataFrame at index {i} does not contain 'PDB' column, skipping")
            continue
            
        logger.info(f'Processing DataFrame {i+1}/{len(df_list)}')
        
        for pdb in df['PDB']:
            if not isinstance(pdb, str):
                logger.warning(f"Skipping non-string PDB entry: {pdb}")
                continue
                
            a3m = pdb.split('_unrelaxed')[0] + '.a3m'
            a3m_list.append(a3m)
            
    logger.info(f"Collected {len(a3m_list)} A3M files")
    return a3m_list

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
        if not os.path.exists(file_name):
            logger.warning(f"File not found, skipping: {file_name}")
            continue
            
        try:
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
                        current_sequence += "".join(char for char in line if char.isupper() or char == '-')
                
                # Process the last entry in the file
                if current_header and current_sequence:
                    entry = (current_header, current_sequence)
                    if entry not in seen_entries:
                        concatenated_content.append(f"{current_header}\n{current_sequence}\n")
                        seen_entries.add(entry)
                        
        except Exception as e:
            logger.warning(f"Error processing file {file_name}: {e}")
    
    # Ensure output directory exists
    Path(a3m_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Write concatenated content to output file
    with open(a3m_path, "w") as output_file:
        output_file.writelines(concatenated_content)
            
    logger.info(f"Successfully wrote {len(seen_entries)} unique sequences to {a3m_path}")

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


# ===== Enhanced A3M Processing for Hit Expand =====

# Constants for sequence validation
A3M_EXTENSION = ".a3m"
FASTA_EXTENSIONS = [".fasta", ".fa", ".fas"]
MAX_SEQUENCE_LENGTH = 50000
MIN_SEQUENCE_LENGTH = 10
STANDARD_AMINO_ACIDS = set("ARNDCQEGHILKMFPSTWYV")


class SequenceFormatError(Exception):
    """Raised when sequence format is invalid."""
    pass


class A3MParser:
    """Enhanced A3M file parser with validation and error handling."""
    
    def __init__(self, strict_validation: bool = True, filter_lowercase: bool = True):
        """
        Initialize A3M parser.
        
        Args:
            strict_validation: If True, raises exceptions on validation errors.
                             If False, logs warnings and attempts to continue.
            filter_lowercase: If True, removes lowercase letters (insertions) during parsing.
        """
        self.strict_validation = strict_validation
        self.filter_lowercase = filter_lowercase
        # Pattern to match valid amino acid sequences
        self._aa_pattern = re.compile(r'^[ARNDCQEGHILKMFPSTWYVXarndcqeghilkmfpstwyv\-\.]*$')
    
    def parse_file(self, file_path: Path) -> Dict[str, str]:
        """
        Parse A3M file and return sequences.
        
        Args:
            file_path: Path to A3M file
            
        Returns:
            Dictionary mapping headers to sequences
            
        Raises:
            SequenceFormatError: If file format is invalid
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"A3M file not found: {file_path}")
        
        if file_path.suffix.lower() != A3M_EXTENSION:
            logger.warning(f"File does not have A3M extension: {file_path}")
        
        sequences = {}
        current_header = None
        current_sequence = ""
        line_num = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    if not line:  # Skip empty lines
                        continue
                    
                    if line.startswith('>'):
                        # Save previous sequence if exists
                        if current_header is not None:
                            processed_seq = self._process_sequence(current_sequence, current_header, line_num)
                            if processed_seq:
                                sequences[current_header] = processed_seq
                        
                        # Start new sequence
                        current_header = line
                        current_sequence = ""
                    else:
                        # Accumulate sequence lines
                        current_sequence += line
                
                # Don't forget the last sequence
                if current_header is not None:
                    processed_seq = self._process_sequence(current_sequence, current_header, line_num)
                    if processed_seq:
                        sequences[current_header] = processed_seq
        
        except UnicodeDecodeError as e:
            raise SequenceFormatError(f"File encoding error at line {line_num}: {e}")
        except Exception as e:
            raise SequenceFormatError(f"Parse error at line {line_num}: {e}")
        
        if not sequences:
            raise SequenceFormatError(f"No valid sequences found in {file_path}")
        
        logger.info(f"Parsed {len(sequences)} sequences from {file_path}")
        return sequences
    
    def _process_sequence(self, sequence: str, header: str, line_num: int) -> Optional[str]:
        """
        Process and validate a sequence.
        
        Args:
            sequence: Raw sequence string
            header: Sequence header
            line_num: Current line number for error reporting
            
        Returns:
            Processed sequence or None if invalid
        """
        if not sequence:
            if self.strict_validation:
                raise SequenceFormatError(f"Empty sequence for header {header} at line {line_num}")
            else:
                logger.warning(f"Empty sequence for header {header} at line {line_num}")
                return None
        
        # Filter lowercase letters if requested (MSA insertions)
        if self.filter_lowercase:
            sequence = ''.join(char for char in sequence if not char.islower())
        
        # Validate sequence content
        if not self._aa_pattern.match(sequence):
            invalid_chars = set(sequence) - set("ARNDCQEGHILKMFPSTWYVXarndcqeghilkmfpstwyv-.")
            if self.strict_validation:
                raise SequenceFormatError(
                    f"Invalid characters in sequence {header}: {invalid_chars}"
                )
            else:
                logger.warning(
                    f"Invalid characters in sequence {header}: {invalid_chars}"
                )
                return None
        
        # Validate sequence length
        if len(sequence) < MIN_SEQUENCE_LENGTH:
            if self.strict_validation:
                raise SequenceFormatError(
                    f"Sequence {header} too short: {len(sequence)} < {MIN_SEQUENCE_LENGTH}"
                )
            else:
                logger.warning(
                    f"Sequence {header} too short: {len(sequence)} < {MIN_SEQUENCE_LENGTH}"
                )
                return None
        
        if len(sequence) > MAX_SEQUENCE_LENGTH:
            if self.strict_validation:
                raise SequenceFormatError(
                    f"Sequence {header} too long: {len(sequence)} > {MAX_SEQUENCE_LENGTH}"
                )
            else:
                logger.warning(
                    f"Sequence {header} too long: {len(sequence)} > {MAX_SEQUENCE_LENGTH}"
                )
                return None
        
        return sequence
    
    def write_sequences(self, sequences: Dict[str, str], output_path: Path, 
                       ensure_query_first: bool = True) -> None:
        """
        Write sequences to A3M file.
        
        Args:
            sequences: Dictionary mapping headers to sequences
            output_path: Output file path
            ensure_query_first: If True, ensures first sequence (query) comes first
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Sort sequences, optionally putting query first
        sequence_items = list(sequences.items())
        if ensure_query_first and sequence_items:
            # Assume first sequence is query, move it to front if not already
            query_header, query_seq = sequence_items[0]
            other_sequences = sequence_items[1:]
            sequence_items = [(query_header, query_seq)] + other_sequences
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for header, sequence in sequence_items:
                    f.write(f"{header}\n{sequence}\n")
            
            logger.info(f"Wrote {len(sequences)} sequences to {output_path}")
        
        except Exception as e:
            raise SequenceFormatError(f"Error writing sequences to {output_path}: {e}")
    
    def get_query_sequence(self, sequences: Dict[str, str]) -> Tuple[str, str]:
        """
        Get the query sequence (assumed to be first).
        
        Args:
            sequences: Dictionary of sequences
            
        Returns:
            Tuple of (header, sequence) for query
            
        Raises:
            SequenceFormatError: If no sequences found
        """
        if not sequences:
            raise SequenceFormatError("No sequences available")
        
        # First sequence is assumed to be query
        query_header = list(sequences.keys())[0]
        query_sequence = sequences[query_header]
        
        return query_header, query_sequence


def validate_a3m_file(file_path: Path, strict: bool = True) -> bool:
    """
    Validate A3M file format and content.
    
    Args:
        file_path: Path to A3M file
        strict: If True, raise exceptions on errors
        
    Returns:
        True if valid, False otherwise
        
    Raises:
        SequenceFormatError: If strict=True and validation fails
    """
    try:
        parser = A3MParser(strict_validation=strict)
        sequences = parser.parse_file(file_path)
        return len(sequences) > 0
    except Exception as e:
        if strict:
            raise
        else:
            logger.error(f"A3M validation failed for {file_path}: {e}")
            return False


def filter_sequences_by_coverage(sequences: Dict[str, str], 
                                query_sequence: str,
                                min_coverage: float = 0.7) -> Dict[str, str]:
    """
    Filter sequences by coverage relative to query.
    
    Args:
        sequences: Dictionary of sequences
        query_sequence: Reference query sequence
        min_coverage: Minimum coverage threshold (0.0-1.0)
        
    Returns:
        Filtered sequences dictionary
    """
    if not (0.0 <= min_coverage <= 1.0):
        raise ValueError("Coverage threshold must be between 0.0 and 1.0")
    
    query_length = len(query_sequence.replace('-', ''))
    filtered_sequences = {}
    
    for header, sequence in sequences.items():
        # Calculate coverage (non-gap positions)
        non_gap_length = len(sequence.replace('-', ''))
        coverage = non_gap_length / query_length if query_length > 0 else 0.0
        
        if coverage >= min_coverage:
            filtered_sequences[header] = sequence
        else:
            logger.debug(f"Filtered out {header}: coverage {coverage:.3f} < {min_coverage}")
    
    logger.info(f"Filtered {len(filtered_sequences)}/{len(sequences)} sequences by coverage >= {min_coverage}")
    return filtered_sequences


def split_sequences_by_similarity(sequences: Dict[str, str],
                                 query_sequence: str,
                                 similarity_threshold: float = 0.7) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Split sequences into similar and dissimilar groups based on query.
    
    Args:
        sequences: Dictionary of sequences
        query_sequence: Reference query sequence
        similarity_threshold: Similarity threshold (0.0-1.0)
        
    Returns:
        Tuple of (similar_sequences, dissimilar_sequences)
    """
    if not (0.0 <= similarity_threshold <= 1.0):
        raise ValueError("Similarity threshold must be between 0.0 and 1.0")
    
    similar_sequences = {}
    dissimilar_sequences = {}
    
    for header, sequence in sequences.items():
        # Simple similarity calculation (identity percentage)
        if len(sequence) == len(query_sequence):
            matches = sum(1 for a, b in zip(sequence, query_sequence) if a == b and a != '-')
            total_positions = len(query_sequence.replace('-', ''))
            similarity = matches / total_positions if total_positions > 0 else 0.0
        else:
            # For different lengths, use a more complex alignment (simplified here)
            similarity = 0.0
        
        if similarity >= similarity_threshold:
            similar_sequences[header] = sequence
        else:
            dissimilar_sequences[header] = sequence
    
    logger.info(f"Split sequences: {len(similar_sequences)} similar, {len(dissimilar_sequences)} dissimilar")
    return similar_sequences, dissimilar_sequences