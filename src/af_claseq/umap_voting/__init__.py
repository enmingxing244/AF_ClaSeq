"""UMAP Voting module — VAE-embed + UMAP projection + Option F voting."""

from .config import VaeTrainConfig, UmapVotingConfig, load_config

__all__ = ["VaeTrainConfig", "UmapVotingConfig", "load_config", "VaeTrainer", "UmapVotingManager"]


def __getattr__(name):
    if name == "VaeTrainer":
        from .vae.train import VaeTrainer
        return VaeTrainer
    if name == "UmapVotingManager":
        from .workflow import UmapVotingManager
        return UmapVotingManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
