"""UMAP fit (joint sampling + refs) + bin assignment + grid persistence."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import umap

from af_claseq.utils.exceptions import WorkflowError
from af_claseq.utils.logging_utils import get_logger

logger = get_logger("umap_voting.projector")


def embedding_sha256(npz_path: str | Path) -> str:
    """Order-independent hash: sort rows by pdb_path then hash all fields."""
    d = np.load(npz_path, allow_pickle=False)
    pdb = d["pdb_paths"].astype(str)
    order = np.argsort(pdb)
    h = hashlib.sha256()
    h.update(d["mu"][order].tobytes())
    h.update(pdb[order].tobytes())
    h.update(d["a3m_paths"].astype(str)[order].tobytes())
    h.update(d["is_reference"][order].tobytes())
    h.update(d["ref_label"].astype(str)[order].tobytes())
    return h.hexdigest()


@dataclass
class Projector:
    embedding_npz: str | Path
    output_dir: str | Path
    n_neighbors: int = 30
    min_dist: float = 0.1
    n_components: int = 2
    metric: str = "euclidean"
    random_state: int = 42

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _model_path(self) -> Path:
        return self.output_dir / "umap_model.joblib"

    def _coords_path(self) -> Path:
        return self.output_dir / "umap_coords.csv"

    def _config_path(self) -> Path:
        return self.output_dir / "umap_config_used.json"

    def _hash_sidecar(self) -> Path:
        return self.output_dir / "umap_inputs.json"

    def _grid_path(self) -> Path:
        return self.output_dir / "grid.json"

    def fit(self, refit: bool = False) -> pd.DataFrame:
        d = np.load(self.embedding_npz, allow_pickle=False)
        pdb = d["pdb_paths"].astype(str)
        order = np.argsort(pdb)
        mu = d["mu"][order]
        a3m = d["a3m_paths"].astype(str)[order]
        is_ref = d["is_reference"][order]
        ref_label = d["ref_label"].astype(str)[order]
        pdb = pdb[order]

        if not np.isfinite(mu).all():
            raise WorkflowError(
                "embedding contains NaN/Inf - cannot fit UMAP"
            )

        current_hash = embedding_sha256(self.embedding_npz)

        if self._model_path().exists() and not refit:
            prior_hash = None
            if self._hash_sidecar().exists():
                prior_hash = json.loads(
                    self._hash_sidecar().read_text()
                ).get("sha256")
            if prior_hash and prior_hash != current_hash:
                raise RuntimeError(
                    "embedding has drifted vs. persisted UMAP "
                    "- pass --refit-umap to refit"
                )
            # joblib is the standard serialization for scikit-learn/UMAP models;
            # the file is only loaded from the user's own output directory.
            reducer = joblib.load(self._model_path())
            coords = reducer.embedding_
            logger.info(f"reloaded UMAP from {self._model_path()}")
        else:
            reducer = umap.UMAP(
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                n_components=self.n_components,
                metric=self.metric,
                random_state=self.random_state,
            )
            reducer.fit(mu)
            coords = reducer.embedding_
            joblib.dump(reducer, self._model_path())
            self._hash_sidecar().write_text(
                json.dumps(
                    {"sha256": current_hash, "n_rows": int(len(pdb))},
                    indent=2,
                )
            )
            self._config_path().write_text(
                json.dumps(
                    {
                        "n_neighbors": self.n_neighbors,
                        "min_dist": self.min_dist,
                        "n_components": self.n_components,
                        "metric": self.metric,
                        "random_state": self.random_state,
                        "umap_version": umap.__version__,
                    },
                    indent=2,
                )
            )
            logger.info(f"fit UMAP on {len(pdb)} rows; wrote {self._model_path()}")

        df = pd.DataFrame(
            {
                "pdb_path": pdb,
                "a3m_path": a3m,
                "UMAP1": coords[:, 0],
                "UMAP2": coords[:, 1],
                "is_reference": is_ref,
                "ref_label": ref_label,
            }
        )
        df.to_csv(self._coords_path(), index=False)
        return df

    def assign_bins(
        self,
        bin_size: float,
        umap1_range: Optional[Tuple[float, float]] = None,
        umap2_range: Optional[Tuple[float, float]] = None,
        margin_frac: float = 0.05,
    ) -> pd.DataFrame:
        df = pd.read_csv(self._coords_path())
        grid_path = self._grid_path()

        if grid_path.exists():
            prior = json.loads(grid_path.read_text())
            if abs(prior["bin_size"] - bin_size) > 1e-9:
                raise WorkflowError(
                    f"grid.json bin_size={prior['bin_size']} "
                    f"!= requested {bin_size}; pass --refit-umap"
                )
            if umap1_range and tuple(prior["umap1_range"]) != tuple(umap1_range):
                raise WorkflowError(
                    "umap1_range conflicts with pinned grid.json; "
                    "pass --refit-umap to override"
                )
            if umap2_range and tuple(prior["umap2_range"]) != tuple(umap2_range):
                raise WorkflowError(
                    "umap2_range conflicts with pinned grid.json; "
                    "pass --refit-umap to override"
                )
            umap1_range = tuple(prior["umap1_range"])
            umap2_range = tuple(prior["umap2_range"])

        if umap1_range is None:
            lo, hi = df["UMAP1"].min(), df["UMAP1"].max()
            pad = (hi - lo) * margin_frac
            umap1_range = (lo - pad, hi + pad)
        if umap2_range is None:
            lo, hi = df["UMAP2"].min(), df["UMAP2"].max()
            pad = (hi - lo) * margin_frac
            umap2_range = (lo - pad, hi + pad)

        grid_path.write_text(
            json.dumps(
                {
                    "umap1_range": list(umap1_range),
                    "umap2_range": list(umap2_range),
                    "bin_size": float(bin_size),
                },
                indent=2,
            )
        )

        x_in = (df["UMAP1"] >= umap1_range[0]) & (df["UMAP1"] < umap1_range[1])
        y_in = (df["UMAP2"] >= umap2_range[0]) & (df["UMAP2"] < umap2_range[1])
        in_range = x_in & y_in

        refs_out = df[df["is_reference"] & ~in_range]
        if len(refs_out) > 0:
            labels = ", ".join(refs_out["ref_label"].tolist())
            raise WorkflowError(
                f"reference(s) outside grid range: {labels}. "
                "Widen umap1_range/umap2_range or pass --refit-umap."
            )

        oor = df.loc[~in_range].copy()
        oor.to_csv(self.output_dir / "out_of_range.csv", index=False)
        if len(oor) > 0:
            logger.warning(
                f"dropped {len(oor)} sampling rows outside grid range"
            )

        df = df.loc[in_range].copy()
        df["bin_ix"] = np.floor(
            (df["UMAP1"] - umap1_range[0]) / bin_size
        ).astype(int)
        df["bin_iy"] = np.floor(
            (df["UMAP2"] - umap2_range[0]) / bin_size
        ).astype(int)
        df.to_csv(self._coords_path(), index=False)
        return df
