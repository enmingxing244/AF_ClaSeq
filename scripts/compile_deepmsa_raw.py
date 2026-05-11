#!/usr/bin/env python3

import glob
import re
import argparse
import os
from collections import defaultdict
from pathlib import Path
import datetime

def clean_sequence(seq):
    """Remove lowercase letters, keep only uppercase letters and gaps"""
    return ''.join(c for c in seq if c.isupper() or c == '-')

def clean_header(header):
    """Clean header to keep only first part before space or tab"""
    # Remove the > prefix first
    if header.startswith('>'):
        header = header[1:]
    # Split by space or tab and take first part
    parts = re.split(r'[\s\t]', header)
    return '>' + parts[0]


def process_a3m_files(input_files=None, output_filename="compiled_unique_sequences.a3m"):
    """Process specified .a3m files and extract unique sequences

    Args:
        input_files: List of A3M file paths to process. If None, process all *.a3m files in current directory
        output_filename: Name for the output file
    """

    # Initialize logging
    log_filename = os.path.splitext(output_filename)[0] + "_processing_log.txt"
    file_stats = {}  # Track stats for each file
    duplicate_details = []  # Track duplicate sequences

    # Step 1: Determine input files
    all_sequences = []

    # Use provided list of files
    a3m_files = []
    for file_path in input_files:
        if os.path.exists(file_path):
            a3m_files.append(file_path)
        else:
            print(f"Warning: File not found: {file_path}")

    if not a3m_files:
        print("Error: No valid A3M files found")
        return
    
    for filename in a3m_files:
        print(f"Processing {filename}...")
        file_sequences = []

        with open(filename, 'r') as f:
            current_header = None
            current_seq = ""

            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    # Save previous sequence if exists
                    if current_header is not None:
                        file_sequences.append((current_header, current_seq))
                        all_sequences.append((current_header, current_seq, filename))

                    # Start new sequence
                    current_header = line
                    current_seq = ""
                else:
                    current_seq += line

            # Save last sequence
            if current_header is not None:
                file_sequences.append((current_header, current_seq))
                all_sequences.append((current_header, current_seq, filename))

        # Record stats for this file
        file_stats[filename] = {
            'total_sequences': len(file_sequences),
            'sequences': file_sequences
        }
    
    print(f"Read {len(all_sequences)} total sequences from {len(a3m_files)} files")
    
    # Step 2: Clean headers and sequences, remove duplicates (both header and sequence must be unique)
    seen_headers = set()
    seen_sequences = set()
    unique_sequences = {}
    total_duplicates = 0

    for header, seq, source_file in all_sequences:
        cleaned_header = clean_header(header)
        cleaned_seq = clean_sequence(seq)

        # Check if either header or sequence is already seen
        is_duplicate = False
        duplicate_reason = []

        if cleaned_header in seen_headers:
            is_duplicate = True
            duplicate_reason.append("duplicate header")

        if cleaned_seq in seen_sequences:
            is_duplicate = True
            duplicate_reason.append("duplicate sequence")

        if is_duplicate:
            # Skip this entry - it's a duplicate
            total_duplicates += 1
            duplicate_details.append({
                'skipped_header': cleaned_header,
                'sequence': cleaned_seq[:50] + "..." if len(cleaned_seq) > 50 else cleaned_seq,
                'reason': " and ".join(duplicate_reason),
                'source_file': source_file
            })
            print(f"  Skipping '{cleaned_header}': {' and '.join(duplicate_reason)}")
        else:
            # Keep this entry - both header and sequence are unique
            seen_headers.add(cleaned_header)
            seen_sequences.add(cleaned_seq)
            unique_sequences[cleaned_header] = cleaned_seq
    
    print(f"Found {len(unique_sequences)} unique sequences")

    # Step 3: Write output
    with open(output_filename, 'w') as f:
        for header, seq in unique_sequences.items():
            f.write(f"{header}\n{seq}\n")

    print(f"Written unique sequences to {output_filename}")

    # Calculate included sequences per file
    for filename in file_stats:
        included_count = 0
        for header, seq in unique_sequences.items():
            # Check if this sequence came from this file
            for orig_header, orig_seq, source_file in all_sequences:
                cleaned_header = clean_header(orig_header)
                cleaned_seq = clean_sequence(orig_seq)
                if source_file == filename and cleaned_header == header and cleaned_seq == seq:
                    included_count += 1
                    break
        file_stats[filename]['included_sequences'] = included_count
        file_stats[filename]['duplicated_sequences'] = file_stats[filename]['total_sequences'] - included_count

    # Write comprehensive log file
    with open(log_filename, 'w') as log_file:
        log_file.write(f"A3M Processing Log\n")
        log_file.write(f"==================\n")
        log_file.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Output file: {output_filename}\n\n")

        log_file.write(f"SUMMARY:\n")
        log_file.write(f"- Input files processed: {len(a3m_files)}\n")
        log_file.write(f"- Total sequences read: {len(all_sequences)}\n")
        log_file.write(f"- Total duplicates removed: {total_duplicates}\n")
        log_file.write(f"- Final unique sequences: {len(unique_sequences)}\n\n")

        log_file.write(f"FILE DETAILS:\n")
        log_file.write(f"{'='*80}\n")
        for filename, stats in file_stats.items():
            log_file.write(f"\nFile: {filename}\n")
            log_file.write(f"  - Total sequences in file: {stats['total_sequences']}\n")
            log_file.write(f"  - Sequences included in output: {stats['included_sequences']}\n")
            log_file.write(f"  - Sequences duplicated/excluded: {stats['duplicated_sequences']}\n")

        if duplicate_details:
            log_file.write(f"\nDUPLICATE DETAILS:\n")
            log_file.write(f"{'='*80}\n")
            for i, dup in enumerate(duplicate_details, 1):
                log_file.write(f"\nDuplicate {i}:\n")
                log_file.write(f"  - Skipped header: {dup['skipped_header']}\n")
                log_file.write(f"  - Sequence preview: {dup['sequence']}\n")
                log_file.write(f"  - Reason: {dup['reason']}\n")
                log_file.write(f"  - Source file: {dup['source_file']}\n")

    print(f"Detailed log written to {log_filename}")

    # Print summary
    print("\nSummary:")
    print(f"- Input files: {len(a3m_files)}")
    print(f"- Total sequences read: {len(all_sequences)}")
    print(f"- Total duplicates removed: {total_duplicates}")
    print(f"- Final unique sequences: {len(unique_sequences)}")

def main():
    parser = argparse.ArgumentParser(
        description="Process A3M files and extract unique sequences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all *.a3m files in current directory
  python compile_deepmsa_raw.py

  # Process specific files with default output name
  python compile_deepmsa_raw.py -i file1.a3m file2.a3m file3.a3m

  # Process specific files with custom output name
  python compile_deepmsa_raw.py -i file1.a3m file2.a3m -o my_compiled_sequences.a3m

  # Process files from different directories
  python compile_deepmsa_raw.py -i /path/to/file1.a3m /path/to/file2.a3m -o combined.a3m

  # Read file list from text file (one file path per line)
  python compile_deepmsa_raw.py --file-list input_files.txt -o output.a3m
        """
    )

    parser.add_argument(
        '-i', '--input-files',
        nargs='+',
        help='List of A3M files to process. If not specified, processes all *.a3m files in current directory'
    )

    parser.add_argument(
        '--file-list',
        help='Text file containing list of A3M file paths (one per line)'
    )

    parser.add_argument(
        '-o', '--output',
        default='compiled_unique_sequences.a3m',
        help='Output filename (default: compiled_unique_sequences.a3m)'
    )

    args = parser.parse_args()

    # Determine input files
    input_files = None
    if args.file_list:
        try:
            with open(args.file_list, 'r') as f:
                input_files = [line.strip() for line in f if line.strip()]
            print(f"Read {len(input_files)} file paths from {args.file_list}")
        except Exception as e:
            print(f"Error reading file list: {e}")
            return 1
    elif args.input_files:
        input_files = args.input_files
    else:
        import glob
        input_files = sorted(glob.glob('*.a3m'))
        if not input_files:
            print("Error: No A3M files found in current directory and no input files specified.")
            return 1
        print(f"Auto-discovered {len(input_files)} A3M files in current directory")

    # Validate output directory
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")
        except Exception as e:
            print(f"Error creating output directory: {e}")
            return 1

    # Process files
    try:
        process_a3m_files(input_files, args.output)
        return 0
    except Exception as e:
        print(f"Error processing files: {e}")
        return 1


if __name__ == "__main__":
    exit(main())