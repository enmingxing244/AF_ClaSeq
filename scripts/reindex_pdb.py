#!/usr/bin/env python3
"""
Reindex residue numbers in a PDB file by applying an offset.
"""

import argparse
import sys


def reindex_pdb(input_pdb, offset, output_pdb, chain=None):
    """
    Apply an offset to all residue numbers in a PDB file.

    Args:
        input_pdb: Path to input PDB file
        offset: Integer offset to apply (can be positive or negative)
        output_pdb: Path to output PDB file
        chain: Optional chain ID to filter (if None, process all chains)
    """
    try:
        with open(input_pdb, 'r') as infile, open(output_pdb, 'w') as outfile:
            in_model_1 = False
            has_models = False
            atoms_written = 0
            atoms_skipped = 0

            for line in infile:
                # Track if we're in MODEL 1
                if line.startswith('MODEL'):
                    has_models = True
                    model_num = int(line[10:14].strip())
                    if model_num == 1:
                        in_model_1 = True
                    else:
                        in_model_1 = False
                    continue

                # Stop processing after MODEL 1 ends
                if line.startswith('ENDMDL') and in_model_1:
                    break

                # If file has models and we're not in model 1, skip
                if has_models and not in_model_1:
                    continue

                # Process only ATOM records (not HETATM or other records)
                if line.startswith('ATOM'):
                    # Chain ID is at position 21 (column 22)
                    chain_id = line[21]

                    # Filter by chain if specified
                    if chain is not None and chain_id != chain:
                        continue

                    # Residue number is in columns 23-26 (positions 22-26 in string)
                    prefix = line[:22]
                    resnum_str = line[22:26].strip()
                    suffix = line[26:]

                    try:
                        # Parse residue number and apply offset
                        resnum = int(resnum_str)
                        new_resnum = resnum + offset

                        # Skip residues with reindexed number < 1
                        if new_resnum < 1:
                            atoms_skipped += 1
                            continue

                        # Format with proper width (right-aligned in 4 characters)
                        new_line = f"{prefix}{new_resnum:>4}{suffix}"
                        outfile.write(new_line)
                        atoms_written += 1
                    except ValueError:
                        # If residue number can't be parsed, skip it
                        atoms_skipped += 1
                        continue

        print(f"Successfully reindexed {input_pdb}")
        print(f"Applied offset: {offset:+d}")
        if chain:
            print(f"Filtered chain: {chain}")
        print(f"Atoms written: {atoms_written}")
        print(f"Atoms skipped: {atoms_skipped}")
        print(f"Output saved to: {output_pdb}")

    except FileNotFoundError:
        print(f"Error: Input file '{input_pdb}' not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Reindex residue numbers in a PDB file by applying an offset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Input PDB file'
    )

    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output PDB file'
    )

    parser.add_argument(
        '--offset',
        type=int,
        required=True,
        help='Offset to apply to residue numbers (can be positive or negative)'
    )

    parser.add_argument(
        '-c', '--chain',
        type=str,
        default=None,
        help='Chain ID to extract (if not specified, all chains are processed)'
    )

    args = parser.parse_args()

    reindex_pdb(args.input, args.offset, args.output, args.chain)


if __name__ == '__main__':
    main()
