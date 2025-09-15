"""
Divide and Conquer Phylogenetic Workflow for Protein Structure Prediction

This package implements a comprehensive workflow for:
1. Phylogenetic tree construction and clade-based sequence splitting
2. Random shuffling and grouping within clades  
3. ColabFold structure prediction job management
4. Multi-metric structure analysis
5. Publication-quality plot generation
"""

from .phylogenetic_processor import PhylogeneticProcessor
from .shuffle_manager import ShuffleManager
from .structure_analyzer import StructureAnalyzer
from .plot_generator import PlotGenerator
from .utils import setup_logging, load_config

__version__ = "1.0.0"
__author__ = "DAC Workflow"

__all__ = [
    "PhylogeneticProcessor",
    "ShuffleManager",
    "StructureAnalyzer",
    "PlotGenerator",
    "setup_logging",
    "load_config"
]