"""
Hit expansion module for AF-ClaSeq pipeline.

This module provides functionality to expand hit sequences using MSA pipeline
techniques, including clustering, similarity search, and structure analysis.
"""

from af_claseq.hit_expand.runner import HitExpandRunner
from af_claseq.hit_expand.config import HitExpandConfig
from af_claseq.hit_expand.hit_expansion import HitExpandSampler
from af_claseq.hit_expand.analyzer import HitExpandAnalyzer
from af_claseq.hit_expand.output_manager import HitExpandOutputManager
from af_claseq.hit_expand.plotting import HitExpandPlotter

__all__ = [
    'HitExpandRunner',
    'HitExpandConfig', 
    'HitExpandSampler',
    'HitExpandAnalyzer',
    'HitExpandOutputManager',
    'HitExpandPlotter'
]