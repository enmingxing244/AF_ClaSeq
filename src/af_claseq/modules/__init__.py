"""
AF-ClaSeq specialized modules for hit expand functionality.

This package contains specialized components for the hit expand pipeline:
- MMseqs2 wrapper for sequence clustering
- BLOSUM62-based similarity search
- Subset generation for structure prediction
"""

from .mmseqs_wrapper import MMseqsWrapper
from .similarity_search import BLOSUM62SimilaritySearch
from .subset_generator import SubsetGenerator

__all__ = [
    'MMseqsWrapper',
    'BLOSUM62SimilaritySearch', 
    'SubsetGenerator'
]