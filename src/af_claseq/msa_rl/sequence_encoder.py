"""
Sequence encoding utilities for MSA RL.

This module provides functions to convert protein sequences into 
feature vectors suitable for RL training.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from collections import Counter
import logging

logger = logging.getLogger(__name__)

class SequenceEncoder:
    """
    Encodes protein sequences into feature vectors for RL training.
    
    Features include:
    - Amino acid composition
    - Sequence identity to query
    - Phylogenetic diversity metrics
    - Secondary structure predictions (simplified)
    """
    
    def __init__(self, query_sequence: str, embedding_dim: int = 64):
        """
        Initialize the sequence encoder.
        
        Args:
            query_sequence: Reference query sequence
            embedding_dim: Dimension of sequence embeddings
        """
        self.query_sequence = query_sequence
        self.embedding_dim = embedding_dim
        self.aa_vocab = {
            'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7,
            'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15,
            'T': 16, 'W': 17, 'Y': 18, 'V': 19, '-': 20, 'X': 21
        }
        
        # Amino acid properties
        self.aa_properties = self._get_aa_properties()
        
    def _get_aa_properties(self) -> Dict[str, np.ndarray]:
        """Get amino acid physicochemical properties."""
        # Simplified properties: [hydrophobicity, charge, size, aromaticity]
        properties = {
            'A': [1.8, 0, 1, 0], 'R': [-4.5, 1, 4, 0], 'N': [-3.5, 0, 2, 0],
            'D': [-3.5, -1, 2, 0], 'C': [2.5, 0, 2, 0], 'Q': [-3.5, 0, 3, 0],
            'E': [-3.5, -1, 3, 0], 'G': [-0.4, 0, 1, 0], 'H': [-3.2, 0, 3, 1],
            'I': [4.5, 0, 3, 0], 'L': [3.8, 0, 3, 0], 'K': [-3.9, 1, 4, 0],
            'M': [1.9, 0, 3, 0], 'F': [2.8, 0, 4, 1], 'P': [-1.6, 0, 2, 0],
            'S': [-0.8, 0, 1, 0], 'T': [-0.7, 0, 2, 0], 'W': [-0.9, 0, 5, 1],
            'Y': [-1.3, 0, 4, 1], 'V': [4.2, 0, 2, 0], '-': [0, 0, 0, 0],
            'X': [0, 0, 0, 0]
        }
        return {aa: np.array(props, dtype=np.float32) for aa, props in properties.items()}
    
    def encode_sequence(self, sequence: str, header: str = "") -> np.ndarray:
        """
        Encode a single sequence into a feature vector.
        
        Args:
            sequence: Protein sequence string
            header: Sequence header (optional, for additional features)
            
        Returns:
            Feature vector of shape (feature_dim,)
        """
        # Clean sequence
        clean_seq = ''.join(c.upper() if c.upper() in self.aa_vocab else 'X' for c in sequence)
        
        features = []
        
        # 1. Amino acid composition (22 features)
        aa_counts = Counter(clean_seq)
        seq_length = len(clean_seq)
        aa_composition = np.zeros(22, dtype=np.float32)
        for aa, count in aa_counts.items():
            if aa in self.aa_vocab:
                aa_composition[self.aa_vocab[aa]] = count / seq_length
        features.extend(aa_composition)
        
        # 2. Sequence identity to query (1 feature)
        identity = self._calculate_sequence_identity(clean_seq, self.query_sequence)
        features.append(identity)
        
        # 3. Coverage (1 feature)
        coverage = self._calculate_coverage(clean_seq)
        features.append(coverage)
        
        # 4. Physicochemical properties (4 features)
        avg_properties = np.zeros(4, dtype=np.float32)
        valid_residues = 0
        for aa in clean_seq:
            if aa in self.aa_properties and aa != '-':
                avg_properties += self.aa_properties[aa]
                valid_residues += 1
        if valid_residues > 0:
            avg_properties /= valid_residues
        features.extend(avg_properties)
        
        # 5. Gap statistics (3 features)
        gap_stats = self._calculate_gap_statistics(clean_seq)
        features.extend(gap_stats)
        
        # 6. Sequence length (1 feature, normalized)
        normalized_length = len(clean_seq) / len(self.query_sequence)
        features.append(normalized_length)
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_sequence_identity(self, seq1: str, seq2: str) -> float:
        """Calculate sequence identity between two sequences."""
        if len(seq1) != len(seq2):
            # Handle different lengths by aligning to shorter
            min_len = min(len(seq1), len(seq2))
            seq1, seq2 = seq1[:min_len], seq2[:min_len]
        
        if len(seq1) == 0:
            return 0.0
            
        matches = sum(1 for a, b in zip(seq1, seq2) if a == b and a != '-' and b != '-')
        valid_positions = sum(1 for a, b in zip(seq1, seq2) if a != '-' and b != '-')
        
        return matches / valid_positions if valid_positions > 0 else 0.0
    
    def _calculate_coverage(self, sequence: str) -> float:
        """Calculate sequence coverage (non-gap ratio)."""
        if len(sequence) == 0:
            return 0.0
        return (len(sequence) - sequence.count('-')) / len(sequence)
    
    def _calculate_gap_statistics(self, sequence: str) -> List[float]:
        """Calculate gap-related statistics."""
        if len(sequence) == 0:
            return [0.0, 0.0, 0.0]
            
        gap_ratio = sequence.count('-') / len(sequence)
        
        # Calculate gap runs
        gap_runs = []
        current_run = 0
        for char in sequence:
            if char == '-':
                current_run += 1
            else:
                if current_run > 0:
                    gap_runs.append(current_run)
                    current_run = 0
        if current_run > 0:
            gap_runs.append(current_run)
        
        avg_gap_length = np.mean(gap_runs) if gap_runs else 0.0
        num_gap_runs = len(gap_runs) / len(sequence)  # Normalized
        
        return [gap_ratio, avg_gap_length / len(sequence), num_gap_runs]
    
    def encode_msa_subset(self, sequences: Dict[str, str]) -> np.ndarray:
        """
        Encode an MSA subset into a matrix representation.
        
        Args:
            sequences: Dictionary of header -> sequence
            
        Returns:
            Matrix of shape (num_sequences, feature_dim)
        """
        if not sequences:
            return np.empty((0, self.get_feature_dim()), dtype=np.float32)
            
        encoded_seqs = []
        for header, sequence in sequences.items():
            encoded_seq = self.encode_sequence(sequence, header)
            encoded_seqs.append(encoded_seq)
            
        return np.array(encoded_seqs, dtype=np.float32)
    
    def encode_subset_summary(self, sequences: Dict[str, str]) -> np.ndarray:
        """
        Encode MSA subset into a summary feature vector.
        
        Args:
            sequences: Dictionary of header -> sequence
            
        Returns:
            Summary feature vector
        """
        if not sequences:
            return np.zeros(self.embedding_dim, dtype=np.float32)
            
        # Get individual sequence encodings
        seq_matrix = self.encode_msa_subset(sequences)
        
        if len(seq_matrix) == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        # Compute summary statistics
        summary_features = []
        
        # Mean, std, min, max across sequences
        summary_features.extend(np.mean(seq_matrix, axis=0))
        summary_features.extend(np.std(seq_matrix, axis=0))
        summary_features.extend(np.min(seq_matrix, axis=0))
        summary_features.extend(np.max(seq_matrix, axis=0))
        
        # Diversity metrics
        num_sequences = len(sequences)
        summary_features.append(num_sequences / 1000.0)  # Normalized count
        
        # Pairwise identity statistics
        identities = []
        seq_list = list(sequences.values())
        for i in range(len(seq_list)):
            for j in range(i + 1, len(seq_list)):
                identity = self._calculate_sequence_identity(seq_list[i], seq_list[j])
                identities.append(identity)
        
        if identities:
            summary_features.extend([
                np.mean(identities),
                np.std(identities),
                np.min(identities),
                np.max(identities)
            ])
        else:
            summary_features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Truncate or pad to embedding_dim
        summary_array = np.array(summary_features, dtype=np.float32)
        if len(summary_array) > self.embedding_dim:
            return summary_array[:self.embedding_dim]
        elif len(summary_array) < self.embedding_dim:
            padded = np.zeros(self.embedding_dim, dtype=np.float32)
            padded[:len(summary_array)] = summary_array
            return padded
        else:
            return summary_array
    
    def get_feature_dim(self) -> int:
        """Get the dimension of individual sequence features."""
        # 22 (AA composition) + 1 (identity) + 1 (coverage) + 4 (properties) + 3 (gaps) + 1 (length)
        return 32


class SequenceEmbeddingNetwork(nn.Module):
    """
    Neural network for learning sequence embeddings.
    """
    
    def __init__(self, input_dim: int, embedding_dim: int, hidden_dim: int = 128):
        """
        Initialize the embedding network.
        
        Args:
            input_dim: Input feature dimension
            embedding_dim: Output embedding dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.embedding_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Tanh()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.embedding_net(x)


def create_sequence_features(msa_file: str, query_sequence: str) -> Tuple[Dict[str, np.ndarray], SequenceEncoder]:
    """
    Create sequence features for all sequences in an MSA file.
    
    Args:
        msa_file: Path to MSA file
        query_sequence: Query sequence string
        
    Returns:
        Dictionary mapping headers to feature vectors and encoder instance
    """
    from af_claseq.utils.sequence_processing import read_a3m_to_dict
    
    # Read MSA
    sequences = read_a3m_to_dict(msa_file)
    
    # Create encoder
    encoder = SequenceEncoder(query_sequence)
    
    # Encode all sequences
    features = {}
    for header, sequence in sequences.items():
        features[header] = encoder.encode_sequence(sequence, header)
    
    logger.info(f"Created features for {len(features)} sequences")
    return features, encoder