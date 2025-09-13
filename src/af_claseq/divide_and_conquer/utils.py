"""
Common utilities for the divide-and-conquer phylogenetic workflow
"""

import os
import json
import yaml
import re
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict, OrderedDict


def setup_logging(config: Dict[str, Any], log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration for the workflow.
    
    Args:
        config: Configuration dictionary
        log_file: Optional specific log file path
        
    Returns:
        Configured logger instance
    """
    if log_file is None:
        log_file = os.path.join("logs", "dac_workflow.log")
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("dac_workflow")
    logger.info("=" * 50)
    logger.info("DAC Phylogenetic Workflow Started")
    logger.info("=" * 50)
    
    return logger


def load_config(config_file: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_file: Path to configuration file (.yaml/.yml or .json)
        
    Returns:
        Configuration dictionary
    """
    file_ext = Path(config_file).suffix.lower()
    
    with open(config_file, 'r') as f:
        if file_ext in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif file_ext == '.json':
            config = json.load(f)
        else:
            # Try to detect format by content
            content = f.read()
            f.seek(0)
            try:
                if content.strip().startswith('{'):
                    config = json.load(f)
                else:
                    config = yaml.safe_load(f)
            except:
                raise ValueError(f"Could not parse config file: {config_file}")
    
    # Validate required sections
    required_sections = ['input', 'output', 'clade_splitting', 'shuffling', 'colabfold']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Required configuration section '{section}' not found")
    
    return config


def normalize_header(header: str) -> str:
    """
    Normalize sequence headers by replacing spaces and special characters.
    
    Example:
    >UniRef100_A0A182ASR4 Circadian clock protein KaiB n=1 Tax=Cyanobium sp. NIES-981 TaxID=1851505 RepID=A0A182ASR4_9CYAN
    becomes:
    >UniRef100_A0A182ASR4_Circadian_clock_protein_KaiB_n1_TaxCyanobium_sp_NIES-981_TaxID1851505_RepIDA0A182ASR4_9CYAN
    
    Args:
        header: Original header string
        
    Returns:
        Normalized header string
    """
    # Remove the '>' if present
    clean_header = header.lstrip('>')
    
    # Replace spaces with underscores and remove special characters
    clean_header = re.sub(r'[^a-zA-Z0-9_-]', '_', clean_header)
    
    # Remove multiple consecutive underscores
    clean_header = re.sub(r'_+', '_', clean_header)
    
    # Remove trailing underscores
    clean_header = clean_header.strip('_')
    
    return clean_header


def process_sequences_with_header_conflicts(sequences: Dict[str, str], logger: logging.Logger) -> Dict[str, str]:
    """
    Handle sequences that may have identical headers but different content.
    
    Process:
    1. Normalize all headers
    2. Detect header conflicts (same header, different sequence)
    3. Add suffixes to conflicting headers
    4. Remove true sequence duplicates (same content)
    
    Args:
        sequences: Dict mapping original headers to sequences
        logger: Logger instance
        
    Returns:
        Dict mapping processed headers to unique sequences
    """
    logger.info("Processing sequences with potential header conflicts...")
    
    # Step 1: Normalize headers and group by content
    normalized_sequences = {}
    content_to_headers = defaultdict(list)
    
    for original_header, sequence in sequences.items():
        normalized_header = normalize_header(original_header)
        content_to_headers[sequence].append(normalized_header)
    
    # Step 2: For each unique sequence content, pick one representative header
    content_to_final_header = {}
    for sequence, headers in content_to_headers.items():
        if len(headers) == 1:
            # No header conflict for this sequence
            content_to_final_header[sequence] = headers[0]
        else:
            # Multiple headers for same sequence - keep the first one (shortest usually)
            sorted_headers = sorted(headers, key=len)
            content_to_final_header[sequence] = sorted_headers[0]
            logger.debug(f"Multiple headers for same sequence, using: {sorted_headers[0]}")
    
    # Step 3: Check for header conflicts (same header, different content)
    header_to_sequences = defaultdict(list)
    for sequence, header in content_to_final_header.items():
        header_to_sequences[header].append(sequence)
    
    # Step 4: Resolve header conflicts by adding suffixes
    final_sequences = {}
    conflicts_resolved = 0
    
    for header, sequences_list in header_to_sequences.items():
        if len(sequences_list) == 1:
            # No conflict
            final_sequences[header] = sequences_list[0]
        else:
            # Header conflict - add suffixes
            conflicts_resolved += len(sequences_list) - 1
            for i, sequence in enumerate(sequences_list):
                if i == 0:
                    final_header = header
                else:
                    final_header = f"{header}_{i+1}"
                final_sequences[final_header] = sequence
            
            logger.info(f"Resolved header conflict for '{header}': {len(sequences_list)} different sequences")
    
    original_count = len(sequences)
    final_count = len(final_sequences)
    duplicates_removed = original_count - final_count
    
    logger.info(f"Sequence processing summary:")
    logger.info(f"  Original sequences: {original_count}")
    logger.info(f"  Header conflicts resolved: {conflicts_resolved}")
    logger.info(f"  Duplicate sequences removed: {duplicates_removed}")
    logger.info(f"  Final unique sequences: {final_count}")
    
    return final_sequences


def read_a3m(filename: str) -> Dict[str, str]:
    """
    Read a3m alignment file.
    
    Args:
        filename: Path to a3m file
        
    Returns:
        Dictionary mapping sequence IDs to sequences
    """
    records = OrderedDict()
    with open(filename, 'r') as f:
        current_id = None
        current_seq = []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id:
                    records[current_id] = ''.join(current_seq)
                current_id = line[1:]  # Keep full header initially
                current_seq = []
            else:
                current_seq.append(line)
        if current_id:
            records[current_id] = ''.join(current_seq)
    return records


def write_a3m(records: Dict[str, str], filename: str) -> None:
    """
    Write sequences to a3m format file.
    
    Args:
        records: Dictionary mapping sequence IDs to sequences
        filename: Output file path
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        for seq_id, seq in records.items():
            f.write(f'>{seq_id}\n')
            f.write(seq + '\n')


def count_sequences_in_a3m(a3m_file: str) -> int:
    """
    Count the number of sequences in an a3m file.
    
    Args:
        a3m_file: Path to a3m file
        
    Returns:
        Number of sequences
    """
    try:
        with open(a3m_file, 'r') as f:
            return sum(1 for line in f if line.startswith('>'))
    except FileNotFoundError:
        return 0


def validate_file_exists(file_path: str, description: str = "File") -> None:
    """
    Validate that a file exists, raise exception if not.
    
    Args:
        file_path: Path to check
        description: Description of file for error message
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{description} not found: {file_path}")


def create_directory(dir_path: str, description: str = "Directory") -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        dir_path: Directory path to create
        description: Description for logging
    """
    os.makedirs(dir_path, exist_ok=True)


def get_file_stem(file_path: str) -> str:
    """
    Get file stem (filename without extension).
    
    Args:
        file_path: File path
        
    Returns:
        File stem
    """
    return Path(file_path).stem


def find_files_with_pattern(directory: str, pattern: str) -> List[str]:
    """
    Find all files matching a pattern in a directory.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern
        
    Returns:
        List of matching file paths
    """
    return list(Path(directory).rglob(pattern))


class WorkflowError(Exception):
    """Custom exception for workflow errors."""
    pass


class ValidationError(WorkflowError):
    """Custom exception for validation errors."""
    pass