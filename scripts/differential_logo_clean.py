#!/usr/bin/env python3
"""
Create a clean differential logo comparing two A3M files
Usage: python differential_logo_clean.py file1.a3m file2.a3m [output_name]
"""

import logomaker
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import argparse
import sys
import os

# Add the src directory to the path so we can import from af_claseq
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from af_claseq.utils.plotting_manager import save_ai_compatible_plot

def parse_a3m_file(filename):
    """Parse A3M file and extract sequences"""
    sequences = []
    with open(filename, 'r') as f:
        lines = f.readlines()

    current_seq = ""
    for line in lines:
        line = line.strip()
        if line.startswith('>'):
            if current_seq:
                sequences.append(current_seq)
                current_seq = ""
        elif line and not line.startswith('#'):
            current_seq += line

    if current_seq:
        sequences.append(current_seq)

    return sequences

def create_alignment_matrix(sequences):
    """Strip lowercase insertions then validate aligned length for logomaker.

    Lowercase chars are A3M insertion states that don't correspond to alignment
    columns. They must be removed before computing per-column frequencies, while
    gap characters ('-') are preserved to maintain column correspondence.
    """
    if not sequences:
        raise ValueError("No sequences found")

    # Strip lowercase insertion characters first
    cleaned = [''.join(c for c in seq if not c.islower()) for seq in sequences]

    seq_length = len(cleaned[0])
    aligned_sequences = []
    for seq in cleaned:
        if len(seq) != seq_length:
            seq = seq[:seq_length] if len(seq) > seq_length else seq + '-' * (seq_length - len(seq))
        aligned_sequences.append(seq)

    return aligned_sequences

def sequences_to_matrix(sequences):
    """Convert sequences to frequency matrix"""
    if not sequences:
        raise ValueError("No sequences provided")

    seq_length = len(sequences[0])
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

    matrix_data = {aa: [0] * seq_length for aa in amino_acids}

    for pos in range(seq_length):
        position_counts = Counter()
        total_valid = 0

        for seq in sequences:
            if pos < len(seq) and seq[pos] in amino_acids:
                position_counts[seq[pos]] += 1
                total_valid += 1

        if total_valid > 0:
            for aa in amino_acids:
                matrix_data[aa][pos] = position_counts.get(aa, 0) / total_valid

    df = pd.DataFrame(matrix_data)
    df.index = range(1, seq_length + 1)
    return df

def calculate_differential_matrix(matrix1, matrix2):
    """Calculate difference matrix between two frequency matrices"""
    common_positions = matrix1.index.intersection(matrix2.index)
    diff_matrix = matrix1.loc[common_positions] - matrix2.loc[common_positions]
    return diff_matrix

def highlight_most_different_positions(logo, diff_matrix, n_positions=5, color='red', alpha=0.5):
    """Highlight positions with largest differences"""
    position_max_diff = diff_matrix.abs().max(axis=1)
    top_different = position_max_diff.nlargest(n_positions).index

    for pos in top_different:
        logo.highlight_position(p=pos, color=color, alpha=alpha)

    return top_different, position_max_diff

def create_differential_logo(file1, file2, output_name=None):
    """Create clean differential logo"""

    # Parse files
    sequences1 = parse_a3m_file(file1)
    sequences2 = parse_a3m_file(file2)

    # Extract query sequence (first sequence after header)
    query_seq = sequences1[0].replace('-', '')  # Remove gaps from query

    # Create frequency matrices
    aligned_sequences1 = create_alignment_matrix(sequences1)
    aligned_sequences2 = create_alignment_matrix(sequences2)

    freq_matrix1 = sequences_to_matrix(aligned_sequences1)
    freq_matrix2 = sequences_to_matrix(aligned_sequences2)

    # Calculate differential matrix
    diff_matrix = calculate_differential_matrix(freq_matrix1, freq_matrix2)

    # Create Nature-style figure with clean formatting
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 12,
        'axes.linewidth': 1.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.major.width': 1,
        'ytick.major.width': 1
    })

    fig, ax = plt.subplots(figsize=(24, 5))

    # Create logo with chemical functional group coloring
    logo = logomaker.Logo(diff_matrix,
                         ax=ax,
                         color_scheme='dmslogo_funcgroup',
                         vpad=0.05,
                         width=0.9,
                         flip_below=False)  # Prevent letter flipping for negative values

    # Highlight most different positions
    top_different, position_diffs = highlight_most_different_positions(
        logo, diff_matrix, n_positions=5, color='#FF4444', alpha=0.6
    )

    # Set axis limits with more space for amino acid letters
    max_diff = max(abs(diff_matrix.min().min()), abs(diff_matrix.max().max()))
    y_limit = max_diff * 2  # Increased from 1.05 to 1.3 for proper letter display

    ax.set_ylim(-y_limit, y_limit)

    # Nature-style axis formatting
    ax.set_ylabel('Enrichment', fontsize=14, fontweight='normal')
    ax.set_xlabel('Position', fontsize=14, fontweight='normal')

    # Add query sequence as secondary x-axis labels
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    positions = list(range(1, len(query_seq) + 1))
    ax2.set_xticks(positions)
    ax2.set_xticklabels(list(query_seq), fontsize=10, fontfamily='monospace')
    ax2.set_xlabel('Query sequence', fontsize=12, fontweight='normal')
    ax2.tick_params(axis='x', which='major', length=0)  # Remove tick marks

    # Add horizontal line at y=0 with Nature-style formatting
    ax.axhline(y=0, color='#000000', linestyle='-', linewidth=0.8)

    # Clean Nature-style axis appearance
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(axis='both', which='major', labelsize=11, color='#000000')
    ax.tick_params(axis='both', which='minor', length=0)

    plt.tight_layout()

    # Save the figure
    if output_name is None:
        output_name = 'differential_logo'

    # Save plot using AI-compatible format
    save_ai_compatible_plot(plt.gcf(), output_name, dpi=600)
    plt.close()

    # Print summary to stdout
    print(f"Processed {len(sequences1)} vs {len(sequences2)} sequences")
    print(f"Top different positions: {list(top_different)}")
    print(f"Files saved: {output_name}.png, {output_name}.pdf, {output_name}.svg")

    return {
        'diff_matrix': diff_matrix,
        'top_different': top_different,
        'position_diffs': position_diffs
    }

def main():
    parser = argparse.ArgumentParser(description='Create differential sequence logo from two A3M files')
    parser.add_argument('file1', help='First A3M file')
    parser.add_argument('file2', help='Second A3M file')
    parser.add_argument('-o', '--output', help='Output filename prefix (default: differential_logo)',
                       default='differential_logo')

    args = parser.parse_args()

    try:
        results = create_differential_logo(args.file1, args.file2, args.output)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing files: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()