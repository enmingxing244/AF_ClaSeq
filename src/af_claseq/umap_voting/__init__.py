"""
UMAP Voting Module

VAE-embed structures, jointly UMAP-project with references, bin on the UMAP grid
using Option F (top-K most frequent sequences per reference bin), fold with
ColabFold, and evaluate predictions by RMSD scatter.
"""

from af_claseq.umap_voting.config import (
    UmapVotingConfig,
    VaeTrainConfig,
    load_config,
)


def __getattr__(name: str):
    if name == "VaeTrainer":
        from af_claseq.umap_voting.vae.train import VaeTrainer
        return VaeTrainer
    if name == "UmapVotingManager":
        from af_claseq.umap_voting.workflow import UmapVotingManager
        return UmapVotingManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "VaeTrainConfig",
    "UmapVotingConfig",
    "load_config",
    "VaeTrainer",
    "UmapVotingManager",
]
