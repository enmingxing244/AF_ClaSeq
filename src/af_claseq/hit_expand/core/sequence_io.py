#!/usr/bin/env python3
"""
High-quality sequence I/O operations for MSA processing.
Handles A3M format parsing, filtering, and validation.
"""

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from af_claseq.hit_expand.core.constants import (
    A3M_EXTENSION, 
    FASTA_EXTENSIONS,
    MAX_SEQUENCE_LENGTH,
    MIN_SEQUENCE_LENGTH,
    STANDARD_AMINO_ACIDS
)

logger = logging.getLogger(__name__)


class SequenceFormatError(Exception):
    """Raised when sequence format is invalid."""
    pass


class A3MParser:
    """High-quality A3M file parser with validation and error handling."""
    
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
        # Updated pattern to handle MSA format:
        # - Uppercase: standard amino acids aligned to query
        # - Lowercase: insertions relative to query
        # - X: unknown/ambiguous amino acids
        # - -: gaps
        # - .: alternative gap character
        self._aa_pattern = re.compile(r'^[ARNDCQEGHILKMFPSTWYVXarndcqeghilkmfpstwyv\-\.]*$')
    
    def parse_file(self, file_path: Union[str, Path]) -> Tuple[List[str], List[str]]:
        """
        Parse A3M file and return sequences with headers.
        
        Args:
            file_path: Path to A3M file
            
        Returns:
            Tuple of (sequences, headers) lists
            
        Raises:
            FileNotFoundError: If file doesn't exist
            SequenceFormatError: If format is invalid (when strict_validation=True)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"A3M file not found: {file_path}")
        
        logger.info(f"Parsing A3M file: {file_path}")
        
        sequences = []
        headers = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeDecodeError as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            raise SequenceFormatError(f"Invalid file encoding: {e}")
        
        # Parse alternating header/sequence lines, skipping comment lines
        line_number = 0
        while line_number < len(lines):
            # Skip comment lines (ColabFold annotations starting with #)
            while line_number < len(lines) and lines[line_number].strip().startswith('#'):
                logger.debug(f"Skipping comment line {line_number + 1}: {lines[line_number].strip()}")
                line_number += 1
            
            if line_number + 1 >= len(lines):
                break
                
            header_line = lines[line_number].strip()
            sequence_line = lines[line_number + 1].strip()
            
            # Validate header format
            if not header_line.startswith('>'):
                if self.strict_validation:
                    raise SequenceFormatError(
                        f"Invalid header format at line {line_number + 1}: {header_line}"
                    )
                else:
                    logger.warning(f"Skipping invalid header at line {line_number + 1}")
                    line_number += 2
                    continue
            
            # Validate sequence format (after filtering if enabled)
            validation_sequence = sequence_line
            if self.filter_lowercase:
                validation_sequence = self._filter_sequence(sequence_line)
            
            if not self._validate_sequence_filtered(validation_sequence):
                if self.strict_validation:
                    raise SequenceFormatError(
                        f"Invalid sequence format at line {line_number + 2}: {sequence_line[:50]}..."
                    )
                else:
                    logger.warning(f"Skipping invalid sequence at line {line_number + 2}")
                    line_number += 2
                    continue
            
            # Filter sequence if requested
            if self.filter_lowercase:
                filtered_sequence = self._filter_sequence(sequence_line)
                sequences.append(filtered_sequence)
            else:
                sequences.append(sequence_line)
            
            headers.append(header_line)
            line_number += 2
        
        # ASSERT: All sequences must have same length after filtering (critical requirement)
        if self.filter_lowercase and sequences:
            expected_length = len(sequences[0])
            inconsistent_seqs = []
            
            for i, seq in enumerate(sequences):
                if len(seq) != expected_length:
                    inconsistent_seqs.append((i, len(seq)))
            
            if inconsistent_seqs:
                error_details = []
                for seq_idx, seq_len in inconsistent_seqs[:5]:  # Show first 5 errors
                    error_details.append(f"seq{seq_idx}: length {seq_len}")
                
                error_msg = (
                    f"ASSERTION FAILED: All sequences must have same length after filtering! "
                    f"Expected length: {expected_length}, but found {len(inconsistent_seqs)} "
                    f"sequences with different lengths. Examples: {', '.join(error_details)}"
                )
                
                # Always raise error - this is a critical requirement
                raise SequenceFormatError(error_msg)
        
        logger.info(f"Successfully parsed {len(sequences)} sequences from {file_path}")
        
        if self.filter_lowercase and sequences:
            logger.info(f"All sequences filtered to length {len(sequences[0])} (uppercase + gaps only)")
        
        if len(sequences) == 0:
            logger.warning(f"No valid sequences found in {file_path}")
        
        return sequences, headers
    
    def _validate_sequence(self, sequence: str) -> bool:
        """
        Validate sequence contains only valid amino acids and gaps.
        
        Args:
            sequence: Sequence string to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not sequence:
            return False
        
        # Check for valid amino acid characters (including lowercase for insertions)
        return bool(self._aa_pattern.match(sequence))
    
    def _validate_sequence_filtered(self, sequence: str) -> bool:
        """
        Validate filtered sequence contains only uppercase amino acids, gaps, and X.
        
        Args:
            sequence: Filtered sequence string to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not sequence:
            return False
        
        # Pattern for filtered sequences (uppercase amino acids including X and gaps only)
        filtered_pattern = re.compile(r'^[ARNDCQEGHILKMFPSTWYV\-\.X]*$')
        return bool(filtered_pattern.match(sequence))
    
    @staticmethod
    def _filter_sequence(sequence: str) -> str:
        """Filter sequence to keep only uppercase letters and gaps, removing lowercase insertions."""
        # Keep uppercase letters (including X for unknown amino acids) and gaps (- and .)
        # Remove ONLY lowercase letters (insertions)
        return ''.join(c for c in sequence if c.isupper() or c in '-.')


class SequenceFilter:
    """Filters and processes sequences according to AlphaFold requirements."""
    
    @staticmethod
    def filter_for_alphafold(sequence: str) -> str:
        """
        Filter sequence to only keep uppercase letters and gaps.
        This removes insertion characters (lowercase) as required by AlphaFold.
        
        Args:
            sequence: Input sequence string
            
        Returns:
            Filtered sequence with only uppercase amino acids and gaps
        """
        return ''.join(c for c in sequence if c.isupper() or c == '-')
    
    @staticmethod
    def remove_gap_only_positions(sequences: List[str]) -> List[str]:
        """
        Remove positions that are gaps in all sequences.
        
        Args:
            sequences: List of aligned sequences
            
        Returns:
            List of sequences with gap-only positions removed
        """
        if not sequences:
            return sequences
        
        # Find positions that have at least one non-gap character
        seq_length = len(sequences[0])
        keep_positions = []
        
        for pos in range(seq_length):
            has_residue = any(
                pos < len(seq) and seq[pos] != '-' 
                for seq in sequences
            )
            if has_residue:
                keep_positions.append(pos)
        
        # Filter sequences to keep only non-gap positions
        filtered_sequences = []
        for seq in sequences:
            filtered_seq = ''.join(
                seq[pos] if pos < len(seq) else '-' 
                for pos in keep_positions
            )
            filtered_sequences.append(filtered_seq)
        
        logger.info(f"Removed {seq_length - len(keep_positions)} gap-only positions")
        return filtered_sequences
    
    @staticmethod
    def remove_duplicates(sequences: List[str], headers: List[str]) -> Tuple[List[str], List[str]]:
        """
        Remove duplicate sequences while preserving order.
        
        Args:
            sequences: List of sequences
            headers: List of corresponding headers
            
        Returns:
            Tuple of (unique_sequences, unique_headers)
        """
        seen_sequences = set()
        unique_sequences = []
        unique_headers = []
        
        for seq, header in zip(sequences, headers):
            if seq not in seen_sequences:
                seen_sequences.add(seq)
                unique_sequences.append(seq)
                unique_headers.append(header)
        
        removed_count = len(sequences) - len(unique_sequences)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} duplicate sequences")
        
        return unique_sequences, unique_headers


class A3MWriter:
    """High-quality A3M file writer with validation."""
    
    @staticmethod
    def write_file(sequences: List[str], headers: List[str], 
                   output_path: Union[str, Path], 
                   validate: bool = True) -> None:
        """
        Write sequences to A3M format file.
        
        Args:
            sequences: List of sequences
            headers: List of corresponding headers
            output_path: Output file path
            validate: Whether to validate inputs before writing
            
        Raises:
            ValueError: If sequences and headers length mismatch
            SequenceFormatError: If validation fails
        """
        output_path = Path(output_path)
        
        if len(sequences) != len(headers):
            raise ValueError(
                f"Sequences ({len(sequences)}) and headers ({len(headers)}) count mismatch"
            )
        
        if validate:
            A3MWriter._validate_inputs(sequences, headers)
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Writing {len(sequences)} sequences to {output_path}")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for header, sequence in zip(headers, sequences):
                    f.write(f"{header}\n{sequence}\n")
        except IOError as e:
            logger.error(f"Failed to write file {output_path}: {e}")
            raise
        
        logger.info(f"Successfully wrote A3M file: {output_path}")
    
    @staticmethod
    def _validate_inputs(sequences: List[str], headers: List[str]) -> None:
        """
        Validate input sequences and headers.
        
        Args:
            sequences: List of sequences to validate
            headers: List of headers to validate
            
        Raises:
            SequenceFormatError: If validation fails
        """
        if not sequences:
            raise SequenceFormatError("No sequences provided")
        
        if not headers:
            raise SequenceFormatError("No headers provided")
        
        # Check header format
        for i, header in enumerate(headers):
            if not header.startswith('>'):
                raise SequenceFormatError(f"Invalid header format at index {i}: {header}")
        
        # Check sequence format (allow standard amino acids including X and gaps only)
        aa_pattern = re.compile(r'^[ARNDCQEGHILKMFPSTWYV\-\.X]*$', re.IGNORECASE)
        for i, sequence in enumerate(sequences):
            if not sequence:
                raise SequenceFormatError(f"Empty sequence at index {i}")
            
            if not aa_pattern.match(sequence):
                raise SequenceFormatError(f"Invalid sequence at index {i}: {sequence[:50]}...")


def parse_fasta_file(file_path: Union[str, Path]) -> Tuple[List[str], List[str]]:
    """
    Parse FASTA file and return sequences with headers.
    
    Args:
        file_path: Path to FASTA file
        
    Returns:
        Tuple of (sequences, headers)
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {file_path}")
    
    logger.info(f"Parsing FASTA file: {file_path}")
    
    sequences = []
    headers = []
    current_header = ""
    current_sequence = ""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    # Save previous sequence if exists
                    if current_header and current_sequence:
                        headers.append(current_header)
                        sequences.append(current_sequence)
                    
                    # Start new sequence
                    current_header = line
                    current_sequence = ""
                elif line:
                    # Accumulate sequence lines
                    current_sequence += line
            
            # Save last sequence
            if current_header and current_sequence:
                headers.append(current_header)
                sequences.append(current_sequence)
                
    except UnicodeDecodeError as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        raise SequenceFormatError(f"Invalid file encoding: {e}")
    
    logger.info(f"Parsed {len(sequences)} sequences from FASTA file")
    return sequences, headers


def parse_sequence_file(file_path: Union[str, Path], 
                       strict: bool = True,
                       filter_lowercase: bool = True) -> Tuple[List[str], List[str]]:
    """
    Parse sequence file (A3M or FASTA format) automatically based on extension.
    
    Args:
        file_path: Path to sequence file (.a3m or .fasta/.fa/.fas)
        strict: Whether to use strict validation (A3M only)
        filter_lowercase: Whether to filter out lowercase letters (A3M only)
        
    Returns:
        Tuple of (sequences, headers)
    """
    file_path = Path(file_path)
    
    # Validate input
    if not file_path.exists():
        raise SequenceFormatError(f"File not found: {file_path}")
    
    if file_path.stat().st_size == 0:
        raise SequenceFormatError(f"File is empty: {file_path}")
    
    if file_path.stat().st_size > 1024 * 1024 * 1024:  # 1GB limit
        raise SequenceFormatError(f"File too large (>1GB): {file_path}")
    
    # Determine format from extension
    if file_path.suffix.lower() in FASTA_EXTENSIONS:
        return parse_fasta_file(file_path)
    elif file_path.suffix.lower() == A3M_EXTENSION:
        parser = A3MParser(strict_validation=strict, filter_lowercase=filter_lowercase)
        return parser.parse_file(file_path)
    else:
        # Try to auto-detect format by content
        logger.warning(f"Unknown file extension: {file_path.suffix}. Attempting auto-detection...")
        try:
            # First try A3M format
            parser = A3MParser(strict_validation=False, filter_lowercase=filter_lowercase)
            return parser.parse_file(file_path)
        except:
            # Fall back to FASTA format
            logger.info("A3M parsing failed, trying FASTA format...")
            return parse_fasta_file(file_path)


def parse_a3m_file(file_path: Union[str, Path], 
                   strict: bool = True,
                   filter_lowercase: bool = True) -> Tuple[List[str], List[str]]:
    """
    Convenience function to parse A3M file.
    
    Args:
        file_path: Path to A3M file
        strict: Whether to use strict validation
        filter_lowercase: Whether to filter out lowercase letters (insertions)
        
    Returns:
        Tuple of (sequences, headers)
    """
    parser = A3MParser(strict_validation=strict, filter_lowercase=filter_lowercase)
    return parser.parse_file(file_path)


def write_a3m_file(sequences: List[str], headers: List[str], 
                   output_path: Union[str, Path]) -> None:
    """
    Convenience function to write A3M file.
    
    Args:
        sequences: List of sequences
        headers: List of headers
        output_path: Output file path
    """
    A3MWriter.write_file(sequences, headers, output_path)