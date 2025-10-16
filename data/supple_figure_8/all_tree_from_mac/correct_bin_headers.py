#!/usr/bin/env python3
"""
Script to verify and correct headers in bin a3m files against preprocessed a3m files.
For each bin a3m file, it checks if the sequences match those in the preprocessed file,
and if headers differ, creates a corrected copy.
"""

import os
import sys
from pathlib import Path
from collections import defaultdict


def parse_a3m_file(filepath):
    """
    Parse an a3m file and return a dictionary mapping sequences to headers.

    Args:
        filepath: Path to the a3m file

    Returns:
        dict: {sequence: header} mapping
    """
    seq_to_header = {}
    current_header = None
    current_seq = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('>'):
                # Save previous sequence if exists
                if current_header is not None:
                    seq = ''.join(current_seq)
                    seq_to_header[seq] = current_header

                # Start new sequence
                current_header = line[1:]  # Remove '>' prefix
                current_seq = []
            else:
                current_seq.append(line)

        # Save last sequence
        if current_header is not None:
            seq = ''.join(current_seq)
            seq_to_header[seq] = current_header

    return seq_to_header


def parse_a3m_file_ordered(filepath):
    """
    Parse an a3m file and return sequences in order with their headers.

    Args:
        filepath: Path to the a3m file

    Returns:
        list: [(header, sequence), ...] in order
    """
    sequences = []
    current_header = None
    current_seq = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('>'):
                # Save previous sequence if exists
                if current_header is not None:
                    seq = ''.join(current_seq)
                    sequences.append((current_header, seq))

                # Start new sequence
                current_header = line[1:]  # Remove '>' prefix
                current_seq = []
            else:
                current_seq.append(line)

        # Save last sequence
        if current_header is not None:
            seq = ''.join(current_seq)
            sequences.append((current_header, seq))

    return sequences


def write_a3m_file(filepath, sequences):
    """
    Write sequences to an a3m file.

    Args:
        filepath: Path to output file
        sequences: list of (header, sequence) tuples
    """
    with open(filepath, 'w') as f:
        for header, seq in sequences:
            f.write(f'>{header}\n')
            f.write(f'{seq}\n')


def process_bin_file(bin_path, preprocessed_path):
    """
    Process a single bin file and correct headers if needed.

    Args:
        bin_path: Path to bin a3m file
        preprocessed_path: Path to preprocessed a3m file

    Returns:
        tuple: (corrected_count, total_count, corrected_file_path or None)
    """
    print(f"\nProcessing: {bin_path}")

    # Parse both files
    bin_sequences = parse_a3m_file_ordered(bin_path)
    preprocessed_seq_to_header = parse_a3m_file(preprocessed_path)

    total_count = len(bin_sequences)
    corrected_count = 0
    needs_correction = False
    corrected_sequences = []

    # Check each sequence in bin file
    for bin_header, bin_seq in bin_sequences:
        if bin_seq in preprocessed_seq_to_header:
            preprocessed_header = preprocessed_seq_to_header[bin_seq]

            if bin_header != preprocessed_header:
                print(f"  Found mismatch:")
                print(f"    Bin header: {bin_header}")
                print(f"    Preprocessed header: {preprocessed_header}")
                corrected_sequences.append((preprocessed_header, bin_seq))
                corrected_count += 1
                needs_correction = True
            else:
                # Headers match, keep as is
                corrected_sequences.append((bin_header, bin_seq))
        else:
            # Sequence not found in preprocessed file
            print(f"  WARNING: Sequence with header '{bin_header}' not found in preprocessed file")
            corrected_sequences.append((bin_header, bin_seq))

    # Write corrected file if needed
    corrected_file_path = None
    if needs_correction:
        corrected_file_path = bin_path.replace('.a3m', '_corrected.a3m')
        write_a3m_file(corrected_file_path, corrected_sequences)
        print(f"  Created corrected file: {corrected_file_path}")
    else:
        print(f"  All headers match - no correction needed")

    return corrected_count, total_count, corrected_file_path


def main():
    """Main function to process all directories."""
    current_dir = Path('.')

    # Find all subdirectories
    subdirs = [d for d in current_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

    print(f"Found {len(subdirs)} directories to process")

    total_files_processed = 0
    total_files_corrected = 0
    total_headers_corrected = 0

    for subdir in sorted(subdirs):
        # Find bin and preprocessed files
        bin_files = list(subdir.glob('*bin*.a3m'))
        preprocessed_files = list(subdir.glob('*preprocessed*.a3m'))

        if not bin_files:
            continue

        if not preprocessed_files:
            print(f"\nWARNING: {subdir.name} has bin files but no preprocessed file")
            continue

        if len(preprocessed_files) > 1:
            print(f"\nWARNING: {subdir.name} has multiple preprocessed files: {preprocessed_files}")
            continue

        preprocessed_file = preprocessed_files[0]

        print(f"\n{'='*60}")
        print(f"Directory: {subdir.name}")
        print(f"Preprocessed file: {preprocessed_file.name}")
        print(f"Found {len(bin_files)} bin file(s)")

        for bin_file in sorted(bin_files):
            corrected_count, total_count, corrected_file = process_bin_file(
                str(bin_file),
                str(preprocessed_file)
            )

            total_files_processed += 1
            if corrected_file:
                total_files_corrected += 1
                total_headers_corrected += corrected_count

            print(f"  Summary: {corrected_count}/{total_count} headers corrected")

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total bin files processed: {total_files_processed}")
    print(f"Files requiring correction: {total_files_corrected}")
    print(f"Total headers corrected: {total_headers_corrected}")


if __name__ == '__main__':
    main()
