"""Configuration dataclasses for the VAE-UMAP-Vote pipeline."""

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from af_claseq.utils.logging_utils import get_logger
from af_claseq.utils.exceptions import ConfigurationError

logger = get_logger("umap_voting_config")


# ---------------------------------------------------------------------------
# Shared sections
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GeneralSection:
    protein_name: str
    base_dir: str
    random_seed: int = 42
    device: str = "cpu"

    @classmethod
    def from_dict(cls, d: dict) -> "GeneralSection":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class StructureAnalysisSection:
    config_json: str
    coord_target: str = "local"
    metrics: List[str] = field(default_factory=lambda: ["local", "global"])

    def __post_init__(self):
        if self.coord_target not in ("local", "global"):
            raise ConfigurationError(
                f"coord_target must be 'local' or 'global', got '{self.coord_target}'"
            )

    @classmethod
    def from_dict(cls, d: dict) -> "StructureAnalysisSection":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# VAE training config sections
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VaeInputsSection:
    structures_csv: str
    references_csv: str

    @classmethod
    def from_dict(cls, d: dict) -> "VaeInputsSection":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class CoordExtractionSection:
    alignment_ref_pdb: Optional[str] = None
    alignment_ref_chain: str = "A"
    target_chain: str = "A"
    min_superposition_atoms: int = 30

    @classmethod
    def from_dict(cls, d: dict) -> "CoordExtractionSection":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class VaeModelSection:
    latent_dim: int = 6
    hidden_channels: List[int] = field(default_factory=lambda: [32, 64])
    use_residual: bool = True

    def __post_init__(self):
        if self.latent_dim < 1:
            raise ConfigurationError(f"latent_dim must be >= 1, got {self.latent_dim}")

    @classmethod
    def from_dict(cls, d: dict) -> "VaeModelSection":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class VaeTrainingSection:
    epochs: int = 300
    batch_size: int = 256
    learning_rate: float = 1e-3
    kl_weight: float = 0.05
    val_split: float = 0.1
    save_best_only: bool = True
    early_stopping_patience: int = 30
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    lr_scheduler_factor: float = 1.0
    lr_scheduler_patience: int = 30
    normalization_mode: str = "global"

    def __post_init__(self):
        if self.normalization_mode not in ("global", "per_residue", "center_only"):
            raise ConfigurationError(
                f"normalization_mode must be 'global', 'per_residue', or 'center_only', "
                f"got '{self.normalization_mode}'"
            )
        if not 0.0 <= self.val_split < 1.0:
            raise ConfigurationError(f"val_split must be in [0, 1), got {self.val_split}")

    @classmethod
    def from_dict(cls, d: dict) -> "VaeTrainingSection":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class VaeSection:
    model: VaeModelSection = field(default_factory=VaeModelSection)
    training: VaeTrainingSection = field(default_factory=VaeTrainingSection)

    @classmethod
    def from_dict(cls, d: dict) -> "VaeSection":
        model = VaeModelSection.from_dict(d.get("model", {}))
        training = VaeTrainingSection.from_dict(d.get("training", {}))
        return cls(model=model, training=training)


@dataclass(frozen=True)
class VaeOutputSection:
    embedding_filename: str = "embedding.npz"

    @classmethod
    def from_dict(cls, d: dict) -> "VaeOutputSection":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Top-level VAE training config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VaeTrainConfig:
    general: GeneralSection
    inputs: VaeInputsSection
    structure_analysis: StructureAnalysisSection
    coord_extraction: CoordExtractionSection = field(default_factory=CoordExtractionSection)
    vae: VaeSection = field(default_factory=VaeSection)
    output: VaeOutputSection = field(default_factory=VaeOutputSection)

    @classmethod
    def from_dict(cls, d: dict) -> "VaeTrainConfig":
        general = GeneralSection.from_dict(d["general"])
        inputs = VaeInputsSection.from_dict(d["inputs"])
        sa = StructureAnalysisSection.from_dict(d["structure_analysis"])
        coord = CoordExtractionSection.from_dict(d.get("coord_extraction", {}))
        vae = VaeSection.from_dict(d.get("vae", {}))
        output = VaeOutputSection.from_dict(d.get("output", {}))
        return cls(
            general=general,
            inputs=inputs,
            structure_analysis=sa,
            coord_extraction=coord,
            vae=vae,
            output=output,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VaeTrainConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        logger.info(f"Loading VAE config from: {path}")
        with open(path) as f:
            data = yaml.safe_load(f)
        for section in ("general", "inputs", "structure_analysis"):
            if section not in data:
                raise ConfigurationError(f"Missing required section: '{section}'")
        cfg = cls.from_dict(data)
        cfg._validate_paths()
        return cfg

    def _validate_paths(self):
        sa_json = Path(self.structure_analysis.config_json)
        if not sa_json.exists():
            raise FileNotFoundError(f"Structure analysis config not found: {sa_json}")
        structs = Path(self.inputs.structures_csv)
        if not structs.exists():
            raise FileNotFoundError(f"Structures CSV not found: {structs}")
        refs = Path(self.inputs.references_csv)
        if not refs.exists():
            raise FileNotFoundError(f"References CSV not found: {refs}")
        if self.coord_extraction.alignment_ref_pdb is not None:
            ref_pdb = Path(self.coord_extraction.alignment_ref_pdb)
            if not ref_pdb.exists():
                raise FileNotFoundError(f"Alignment ref PDB not found: {ref_pdb}")
        Path(self.general.base_dir).mkdir(parents=True, exist_ok=True)

    def get_vae_dir(self) -> Path:
        return Path(self.general.base_dir) / "vae"


# ---------------------------------------------------------------------------
# UMAP voting config sections
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VotingInputsSection:
    embedding_npz: str
    references_csv: str
    query_a3m: str
    rmsd_vs_refs_csv: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> "VotingInputsSection":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class UmapSection:
    n_neighbors: int = 30
    min_dist: float = 0.1
    n_components: int = 2
    metric: str = "euclidean"
    umap1_range: Optional[List[float]] = None
    umap2_range: Optional[List[float]] = None

    @classmethod
    def from_dict(cls, d: dict) -> "UmapSection":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class BinningSection:
    bin_size: float = 1.0
    top_k: int = 16
    min_records_per_bin: int = 50

    def __post_init__(self):
        if self.bin_size <= 0:
            raise ConfigurationError(f"bin_size must be positive, got {self.bin_size}")
        if self.top_k < 1:
            raise ConfigurationError(f"top_k must be >= 1, got {self.top_k}")

    @classmethod
    def from_dict(cls, d: dict) -> "BinningSection":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class StructurePredictionSection:
    num_models: int = 5
    num_seeds: int = 8
    num_recycle: int = 3
    prediction_mode: str = "monomer"
    rank: str = "plddt"
    random_seed: int = 0

    def __post_init__(self):
        if self.prediction_mode not in ("monomer", "homodimer"):
            raise ConfigurationError(
                f"prediction_mode must be 'monomer' or 'homodimer', "
                f"got '{self.prediction_mode}'"
            )

    @classmethod
    def from_dict(cls, d: dict) -> "StructurePredictionSection":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class SlurmSection:
    conda_env_path: str
    account: str
    partition: str = "nextgen"
    time: str = "00:30:00"
    gpus_per_task: int = 1
    cpus_per_task: int = 4
    check_interval: int = 60

    @classmethod
    def from_dict(cls, d: dict) -> "SlurmSection":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class PlottingSection:
    formats: List[str] = field(default_factory=lambda: ["png", "pdf"])
    metric_ranges: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    colors: Dict[str, str] = field(default_factory=dict)
    panels_per_row: int = 2

    @classmethod
    def from_dict(cls, d: dict) -> "PlottingSection":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Top-level UMAP voting config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UmapVotingConfig:
    general: GeneralSection
    inputs: VotingInputsSection
    umap: UmapSection = field(default_factory=UmapSection)
    binning: BinningSection = field(default_factory=BinningSection)
    structure_prediction: StructurePredictionSection = field(
        default_factory=StructurePredictionSection
    )
    slurm: Optional[SlurmSection] = None
    structure_analysis: Optional[StructureAnalysisSection] = None
    plotting: PlottingSection = field(default_factory=PlottingSection)

    @classmethod
    def from_dict(cls, d: dict) -> "UmapVotingConfig":
        general = GeneralSection.from_dict(d["general"])
        inputs = VotingInputsSection.from_dict(d["inputs"])
        umap = UmapSection.from_dict(d.get("umap", {}))
        binning = BinningSection.from_dict(d.get("binning", {}))
        sp = StructurePredictionSection.from_dict(d.get("structure_prediction", {}))
        slurm = SlurmSection.from_dict(d["slurm"]) if "slurm" in d else None
        sa = (
            StructureAnalysisSection.from_dict(d["structure_analysis"])
            if "structure_analysis" in d
            else None
        )
        plotting = PlottingSection.from_dict(d.get("plotting", {}))
        return cls(
            general=general,
            inputs=inputs,
            umap=umap,
            binning=binning,
            structure_prediction=sp,
            slurm=slurm,
            structure_analysis=sa,
            plotting=plotting,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "UmapVotingConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        logger.info(f"Loading UMAP voting config from: {path}")
        with open(path) as f:
            data = yaml.safe_load(f)
        for section in ("general", "inputs"):
            if section not in data:
                raise ConfigurationError(f"Missing required section: '{section}'")
        cfg = cls.from_dict(data)
        cfg._validate_paths()
        return cfg

    def _validate_paths(self):
        emb = Path(self.inputs.embedding_npz)
        if not emb.exists():
            raise FileNotFoundError(f"Embedding NPZ not found: {emb}")
        refs = Path(self.inputs.references_csv)
        if not refs.exists():
            raise FileNotFoundError(f"References CSV not found: {refs}")
        qa3m = Path(self.inputs.query_a3m)
        if not qa3m.exists():
            raise FileNotFoundError(f"Query A3M not found: {qa3m}")
        if self.structure_analysis is not None:
            sa_json = Path(self.structure_analysis.config_json)
            if not sa_json.exists():
                raise FileNotFoundError(
                    f"Structure analysis config not found: {sa_json}"
                )
        Path(self.general.base_dir).mkdir(parents=True, exist_ok=True)

    def get_umap_dir(self) -> Path:
        return Path(self.general.base_dir) / "umap"

    def get_voting_dir(self) -> Path:
        return Path(self.general.base_dir) / "voting"

    def get_predictions_dir(self) -> Path:
        return Path(self.general.base_dir) / "predictions"

    def get_scatter_dir(self) -> Path:
        return Path(self.general.base_dir) / "scatter"


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def load_config(path: str | Path, kind: str = "vae") -> VaeTrainConfig | UmapVotingConfig:
    if kind == "vae":
        return VaeTrainConfig.from_yaml(path)
    if kind == "voting":
        return UmapVotingConfig.from_yaml(path)
    raise ConfigurationError(f"Unknown config kind: '{kind}'. Use 'vae' or 'voting'.")
