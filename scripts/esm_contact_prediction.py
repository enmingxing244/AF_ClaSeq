#!/usr/bin/env python3
"""
ESM Contact Prediction Script
Predicts contact maps from A3M formatted MSA files using ESM transformer
"""

import torch
import esm
import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO
from typing import List, Tuple, Optional
import re
import argparse
from pathlib import Path

def remove_insertions(sequence: str) -> str:
    """Remove insertion characters (lowercase) from sequence"""
    return re.sub('[a-z]', '', sequence)

def read_a3m_msa(filename: str) -> List[Tuple[str, str]]:
    """
    Read A3M formatted MSA file
    Returns list of (description, sequence) tuples
    """
    sequences = []
    try:
        for record in SeqIO.parse(filename, "fasta"):
            # Remove insertions (lowercase characters) from sequence
            clean_seq = remove_insertions(str(record.seq))
            sequences.append((record.description, clean_seq))
            # print(sequences)
    except Exception as e:
        print(f"Error reading MSA file {filename}: {e}")
        return []

    print(f"Loaded {len(sequences)} sequences from {filename}")
    return sequences


def predict_contacts(msa: List[Tuple[str, str]]) -> torch.Tensor:
    """
    Predict contact map using ESM MSA transformer
    """
    print("Loading ESM MSA transformer model: esm_msa1b_t12_100M_UR50S")

    # Load ESM MSA transformer model
    model, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval().to(device)

    print(f"Using device: {device}")

    # For MSA transformer
    batch_converter = alphabet.get_batch_converter()
    # MSA transformer expects the full MSA
    batch_labels, batch_strs, batch_tokens = batch_converter([msa])
    batch_tokens = batch_tokens.to(device)
    print(f"MSA shape: {batch_tokens.shape}")

    # Predict contacts using MSA transformer
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[model.num_layers], return_contacts=True)
        contacts = results["contacts"][0].cpu()

    return contacts

def plot_contact_map(contacts: torch.Tensor, title: str = "Contact Map", save_path: Optional[str] = None):
    """
    Plot contact map
    """
    # Set font size globally
    plt.rcParams.update({'font.size': 24})

    plt.figure(figsize=(10, 8))
    plt.imshow(contacts.numpy(), cmap='Blues', origin='lower')
    plt.colorbar(label='Contact Probability')
    # plt.title(title)
    plt.xlabel('Residue Position')
    plt.ylabel('Residue Position')

    if save_path:
        # Save PNG with high DPI
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"Contact map saved to: {save_path}")

        # Save SVG version
        svg_path = save_path.replace('.png', '.svg')
        plt.savefig(svg_path, format='svg', bbox_inches='tight')
        print(f"Contact map saved to: {svg_path}")

    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Predict contact maps from A3M MSA files using ESM')
    parser.add_argument('msa_file', help='Path to A3M formatted MSA file')
    parser.add_argument('--max_seqs', type=int, default=1024,
                       help='Maximum number of sequences allowed in MSA (error if exceeded)')
    parser.add_argument('--output', help='Output file path for contact map plot')

    args = parser.parse_args()

    # Check if file exists
    if not Path(args.msa_file).exists():
        print(f"Error: MSA file {args.msa_file} not found")
        return

    # Load MSA
    print(f"Loading MSA from: {args.msa_file}")
    msa = read_a3m_msa(args.msa_file)

    if not msa:
        print("Error: No sequences loaded from MSA file")
        return

    # Check if MSA exceeds max_seqs limit
    if len(msa) > args.max_seqs:
        print(f"Error: MSA has {len(msa)} sequences, which exceeds the maximum limit of {args.max_seqs}")
        print(f"Please increase --max_seqs parameter or reduce the MSA size")
        return

    print(f"Using all {len(msa)} sequences from MSA")

    # Predict contacts
    print("Predicting contacts...")
    try:
        contacts = predict_contacts(msa)
        print(f"Contact map shape: {contacts.shape}")

        # Plot contact map
        output_path = args.output if args.output else f"{Path(args.msa_file).stem}_contacts.png"
        title = f"Contact Map - {Path(args.msa_file).stem}"
        plot_contact_map(contacts, title=title, save_path=output_path)

        # Save contact map as numpy array
        np_output = f"{Path(args.msa_file).stem}_contacts.npy"
        np.save(np_output, contacts.numpy())
        print(f"Contact map saved as numpy array: {np_output}")

    except Exception as e:
        print(f"Error during contact prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()