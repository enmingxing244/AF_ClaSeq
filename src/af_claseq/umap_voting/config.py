"""Config dataclasses for umap_voting (VAE embedding + UMAP voting pipeline)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from af_claseq.utils.logging_utils import get_logger

logger = get_logger("umap_voting_config")

ALLOWED_COORD_TARGET = {"local", "global"}
ALLOWED_PREDICTION_MODE = {"monomer", "homodimer"}


def _require(d: Dict[str, Any], path: str, key: str) -> Any:
    if key not in d:
        raise ValueError(f"missing required field: {path}.{key}")
    return d[key]


# ---- shared sections ----

@dataclass
class GeneralSection:
    protein_name: str
    base_dir: str
    random_seed: int
    device: str = "cpu"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> GeneralSection:
        return cls(
            protein_name=_require(d, "general", "protein_name"),
            base_dir=_require(d, "general", "base_dir"),
            random_seed=_require(d, "general", "random_seed"),
            device=d.get("device", "cpu"),
        )


@dataclass
class StructureAnalysisSection:
    config_json: str
    coord_target: Optional[str] = None
    metrics: Optional[List[str]] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any], *, for_vae: bool) -> StructureAnalysisSection:
        config_json = _require(d, "structure_analysis", "config_json")
        coord_target = d.get("coord_target")
        if for_vae:
            if coord_target not in ALLOWED_COORD_TARGET:
                raise ValueError(
                    f"structure_analysis.coord_target must be one of "
                    f"{ALLOWED_COORD_TARGET}, got {coord_target!r}"
                )
        metrics = d.get("metrics")
        if not for_vae and (metrics is None or len(metrics) == 0):
            raise ValueError(
                "structure_analysis.metrics must be a non-empty list"
            )
        return cls(
            config_json=config_json,
            coord_target=coord_target if for_vae else None,
            metrics=list(metrics) if metrics else None,
        )


# ---- VAE training config ----

@dataclass
class VaeModelSection:
    latent_dim: int
    hidden_channels: List[int]
    use_residual: bool

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> VaeModelSection:
        return cls(
            latent_dim=_require(d, "vae.model", "latent_dim"),
            hidden_channels=_require(d, "vae.model", "hidden_channels"),
            use_residual=_require(d, "vae.model", "use_residual"),
        )


@dataclass
class VaeTrainingSection:
    epochs: int
    batch_size: int
    learning_rate: float
    kl_weight: float
    val_split: float
    save_best_only: bool
    early_stopping_patience: int

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> VaeTrainingSection:
        return cls(
            epochs=_require(d, "vae.training", "epochs"),
            batch_size=_require(d, "vae.training", "batch_size"),
            learning_rate=_require(d, "vae.training", "learning_rate"),
            kl_weight=_require(d, "vae.training", "kl_weight"),
            val_split=_require(d, "vae.training", "val_split"),
            save_best_only=_require(d, "vae.training", "save_best_only"),
            early_stopping_patience=_require(d, "vae.training", "early_stopping_patience"),
        )


@dataclass
class VaeSection:
    model: VaeModelSection
    training: VaeTrainingSection

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> VaeSection:
        return cls(
            model=VaeModelSection.from_dict(_require(d, "vae", "model")),
            training=VaeTrainingSection.from_dict(_require(d, "vae", "training")),
        )


@dataclass
class VaeInputsSection:
    structures_csv: str
    references_csv: str

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> VaeInputsSection:
        return cls(
            structures_csv=_require(d, "inputs", "structures_csv"),
            references_csv=_require(d, "inputs", "references_csv"),
        )


@dataclass
class VaeOutputSection:
    embedding_filename: str = "embedding.npz"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> VaeOutputSection:
        return cls(
            embedding_filename=d.get("embedding_filename", "embedding.npz"),
        )


@dataclass
class CoordExtractionSection:
    alignment_ref_pdb: Optional[str] = None
    alignment_ref_chain: str = "A"
    target_chain: str = "A"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> CoordExtractionSection:
        return cls(
            alignment_ref_pdb=d.get("alignment_ref_pdb"),
            alignment_ref_chain=d.get("alignment_ref_chain", "A"),
            target_chain=d.get("target_chain", "A"),
        )


@dataclass
class VaeTrainConfig:
    general: GeneralSection
    inputs: VaeInputsSection
    structure_analysis: StructureAnalysisSection
    vae: VaeSection
    output: VaeOutputSection
    coord_extraction: CoordExtractionSection

    @classmethod
    def from_yaml(cls, path: str | Path) -> VaeTrainConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(
            general=GeneralSection.from_dict(_require(raw, "root", "general")),
            inputs=VaeInputsSection.from_dict(_require(raw, "root", "inputs")),
            structure_analysis=StructureAnalysisSection.from_dict(
                _require(raw, "root", "structure_analysis"), for_vae=True
            ),
            vae=VaeSection.from_dict(_require(raw, "root", "vae")),
            output=VaeOutputSection.from_dict(raw.get("output", {})),
            coord_extraction=CoordExtractionSection.from_dict(
                raw.get("coord_extraction", {})
            ),
        )


# ---- UMAP voting config ----

@dataclass
class VotingInputsSection:
    embedding_npz: str
    references_csv: str
    query_a3m: str
    rmsd_vs_refs_csv: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> VotingInputsSection:
        return cls(
            embedding_npz=_require(d, "inputs", "embedding_npz"),
            references_csv=_require(d, "inputs", "references_csv"),
            query_a3m=_require(d, "inputs", "query_a3m"),
            rmsd_vs_refs_csv=d.get("rmsd_vs_refs_csv"),
        )


@dataclass
class UmapSection:
    n_neighbors: int
    min_dist: float
    n_components: int = 2
    metric: str = "euclidean"
    umap1_range: Optional[Tuple[float, float]] = None
    umap2_range: Optional[Tuple[float, float]] = None


@dataclass
class BinningSection:
    bin_size: float
    top_k: int
    min_records_per_bin: int = 50


@dataclass
class StructurePredictionSection:
    num_models: int
    num_seeds: int
    num_recycle: int = 3
    prediction_mode: str = "monomer"
    rank: str = "plddt"
    random_seed: int = 0

    def __post_init__(self):
        if self.prediction_mode not in ALLOWED_PREDICTION_MODE:
            raise ValueError(
                f"prediction_mode must be one of {ALLOWED_PREDICTION_MODE}, "
                f"got {self.prediction_mode!r}"
            )


@dataclass
class SlurmSection:
    conda_env_path: str
    account: str
    partition: str = "nextgen"
    time: str = "00:30:00"
    gpus_per_task: int = 1
    cpus_per_task: int = 4
    check_interval: int = 60


@dataclass
class PlottingSection:
    formats: List[str] = field(default_factory=lambda: ["png", "pdf"])
    metric_ranges: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    colors: Dict[str, str] = field(default_factory=dict)
    panels_per_row: int = 2


@dataclass
class UmapVotingConfig:
    general: GeneralSection
    inputs: VotingInputsSection
    umap: UmapSection
    binning: BinningSection
    structure_prediction: StructurePredictionSection
    slurm: SlurmSection
    structure_analysis: StructureAnalysisSection
    plotting: PlottingSection

    @classmethod
    def from_yaml(cls, path: str | Path) -> UmapVotingConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        gen = _require(raw, "root", "general")
        inp = _require(raw, "root", "inputs")
        umap_d = _require(raw, "root", "umap")
        bin_d = _require(raw, "root", "binning")
        sp = _require(raw, "root", "structure_prediction")
        sl = _require(raw, "root", "slurm")
        sa = _require(raw, "root", "structure_analysis")
        pl = raw.get("plotting", {})

        return cls(
            general=GeneralSection.from_dict(gen),
            inputs=VotingInputsSection.from_dict(inp),
            umap=UmapSection(
                n_neighbors=_require(umap_d, "umap", "n_neighbors"),
                min_dist=_require(umap_d, "umap", "min_dist"),
                n_components=umap_d.get("n_components", 2),
                metric=umap_d.get("metric", "euclidean"),
                umap1_range=(
                    tuple(umap_d["umap1_range"])
                    if umap_d.get("umap1_range")
                    else None
                ),
                umap2_range=(
                    tuple(umap_d["umap2_range"])
                    if umap_d.get("umap2_range")
                    else None
                ),
            ),
            binning=BinningSection(
                bin_size=_require(bin_d, "binning", "bin_size"),
                top_k=_require(bin_d, "binning", "top_k"),
                min_records_per_bin=bin_d.get("min_records_per_bin", 50),
            ),
            structure_prediction=StructurePredictionSection(
                num_models=_require(sp, "structure_prediction", "num_models"),
                num_seeds=_require(sp, "structure_prediction", "num_seeds"),
                num_recycle=sp.get("num_recycle", 3),
                prediction_mode=sp.get("prediction_mode", "monomer"),
                rank=sp.get("rank", "plddt"),
                random_seed=sp.get("random_seed", 0),
            ),
            slurm=SlurmSection(
                conda_env_path=_require(sl, "slurm", "conda_env_path"),
                account=_require(sl, "slurm", "account"),
                partition=sl.get("partition", "nextgen"),
                time=sl.get("time", "00:30:00"),
                gpus_per_task=sl.get("gpus_per_task", 1),
                cpus_per_task=sl.get("cpus_per_task", 4),
                check_interval=sl.get("check_interval", 60),
            ),
            structure_analysis=StructureAnalysisSection.from_dict(sa, for_vae=False),
            plotting=PlottingSection(
                formats=pl.get("formats", ["png", "pdf"]),
                metric_ranges=pl.get("metric_ranges", {}),
                colors=pl.get("colors", {}),
                panels_per_row=pl.get("panels_per_row", 2),
            ),
        )


def load_config(path: str | Path, *, kind: str) -> VaeTrainConfig | UmapVotingConfig:
    """Load config from YAML. kind must be 'vae' or 'voting'."""
    if kind == "vae":
        return VaeTrainConfig.from_yaml(path)
    if kind == "voting":
        return UmapVotingConfig.from_yaml(path)
    raise ValueError(f"kind must be 'vae' or 'voting', got {kind!r}")
