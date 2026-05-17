"""Per-prediction RMSD computation + scatter plot assembly."""
from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from af_claseq.utils.logging_utils import get_logger
from af_claseq.utils.structure_analysis import StructureAnalyzer

logger = get_logger("umap_voting.scatter")


def _parse_indices(spec: Any) -> List[int]:
    """Convert a start/end dict or list of dicts to a flat list of ints."""
    if isinstance(spec, dict):
        return list(range(int(spec["start"]), int(spec["end"]) + 1))
    if isinstance(spec, list):
        out = []
        for item in spec:
            if isinstance(item, dict):
                out.extend(range(int(item["start"]), int(item["end"]) + 1))
            else:
                out.append(int(item))
        return out
    return [int(spec)]


def _load_rmsd_specs(
    sa_json_path: str | Path, metrics: List[str]
) -> Dict[str, Dict[str, List[int]]]:
    """Load superposition + rmsd indices from structure-analysis JSON.

    For 'global', uses basics.full_index for both superposition and rmsd.
    For 'local', uses basics.full_index for superposition and
    basics.local_index for rmsd.
    """
    with open(sa_json_path) as f:
        sa = json.load(f)
    basics = sa.get("basics", sa)

    full = basics.get("full_index")
    full_indices = _parse_indices(full) if full else []
    local = basics.get("local_index")
    local_indices = _parse_indices(local) if local else full_indices

    specs = {}
    for m in metrics:
        if m == "global":
            specs[m] = {"superposition": full_indices, "rmsd": full_indices}
        else:
            specs[m] = {"superposition": full_indices, "rmsd": local_indices}
    return specs


@dataclass
class ScatterBuilder:
    predictions_dir: str | Path
    references_csv: str | Path
    structure_analysis_json: str | Path
    output_dir: str | Path
    metrics: List[str] = field(default_factory=lambda: ["local", "global"])
    formats: List[str] = field(default_factory=lambda: ["png", "pdf"])
    metric_ranges: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    colors: Dict[str, str] = field(default_factory=dict)
    panels_per_row: int = 2

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _compute_rmsd(
        self,
        pred_path: Path,
        refs: pd.DataFrame,
        analyzer: StructureAnalyzer,
        specs: Dict[str, Dict[str, List[int]]],
    ) -> Dict[str, float]:
        results: Dict[str, float] = {}
        for metric, sp in specs.items():
            for _, ref in refs.iterrows():
                try:
                    # ColabFold predictions are always chain A; references may
                    # live on a different chain (per references.csv ref_chain).
                    rmsd = analyzer.calculate_ca_rmsd(
                        reference_pdb=str(ref["ref_pdb"]),
                        target_pdb=str(pred_path),
                        superposition_indices=sp["superposition"],
                        rmsd_indices=sp["rmsd"],
                        chain_id="A",
                        reference_chain_id=str(ref.get("ref_chain", "A") or "A"),
                    )
                    results[f"{metric}_rmsd_{ref['ref_label']}"] = rmsd
                except Exception as e:
                    logger.warning(
                        f"RMSD failed for {pred_path.name} vs "
                        f"{ref['ref_label']} ({metric}): {e}"
                    )
        return results

    def build(self) -> pd.DataFrame:
        refs = pd.read_csv(self.references_csv)
        analyzer = StructureAnalyzer()
        specs = _load_rmsd_specs(self.structure_analysis_json, self.metrics)

        rows: List[Dict[str, Any]] = []
        preds_dir = Path(self.predictions_dir)
        for ref_bin_dir in sorted(preds_dir.iterdir()):
            if not ref_bin_dir.is_dir():
                continue
            ref_bin_label = ref_bin_dir.name
            # bin_{ix}_{iy}_{label} — extract label after 3rd underscore
            parts = ref_bin_label.split("_", 3)
            source_ref = parts[3] if len(parts) > 3 else ref_bin_label
            for pdb in sorted(ref_bin_dir.glob("*.pdb")):
                row: Dict[str, Any] = {
                    "ref_bin_label": ref_bin_label,
                    "source_ref": source_ref,
                    "pred_path": str(pdb),
                }
                row.update(self._compute_rmsd(pdb, refs, analyzer, specs))
                rows.append(row)

        per_pred = pd.DataFrame(rows)
        if per_pred.empty:
            logger.warning("no prediction PDBs found; writing empty outputs")
            per_pred.to_csv(self.output_dir / "per_pred.csv", index=False)
            pd.DataFrame(
                columns=["ref_label", "n", "med_target", "med_other",
                         "pct_lt_2A", "pct_lt_1p5A"]
            ).to_csv(self.output_dir / "summary.csv", index=False)
            return per_pred
        per_pred.to_csv(self.output_dir / "per_pred.csv", index=False)

        summary_metric = "local" if "local" in self.metrics else self.metrics[0]
        summary_rows: List[Dict[str, Any]] = []
        for label in refs["ref_label"]:
            in_bin = per_pred[per_pred["source_ref"] == label]
            tgt_col = f"{summary_metric}_rmsd_{label}"
            other_cols = [
                c
                for c in in_bin.columns
                if c.startswith(f"{summary_metric}_rmsd_") and c != tgt_col
            ]
            if in_bin.empty or tgt_col not in in_bin:
                summary_rows.append(
                    dict(
                        ref_label=label,
                        n=0,
                        med_target=float("nan"),
                        med_other=float("nan"),
                        pct_lt_2A=0.0,
                        pct_lt_1p5A=0.0,
                    )
                )
                continue
            n = len(in_bin)
            med_tgt = float(in_bin[tgt_col].median())
            med_other = (
                float(in_bin[other_cols].median().median())
                if other_cols
                else float("nan")
            )
            pct_2 = 100.0 * (in_bin[tgt_col] < 2.0).sum() / n
            pct_15 = 100.0 * (in_bin[tgt_col] < 1.5).sum() / n
            summary_rows.append(
                dict(
                    ref_label=label,
                    n=n,
                    med_target=med_tgt,
                    med_other=med_other,
                    pct_lt_2A=pct_2,
                    pct_lt_1p5A=pct_15,
                )
            )
        pd.DataFrame(summary_rows).to_csv(
            self.output_dir / "summary.csv", index=False
        )

        for metric in self.metrics:
            self._draw_scatter(per_pred, refs, metric)
        return per_pred

    def _draw_scatter(
        self, per_pred: pd.DataFrame, refs: pd.DataFrame, metric: str
    ) -> None:
        labels = list(refs["ref_label"])
        pairs = list(itertools.combinations(labels, 2))
        if not pairs:
            return
        n = len(pairs)
        ncols = min(self.panels_per_row, n)
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False
        )
        rng = self.metric_ranges.get(metric, {})
        for k, (a, b) in enumerate(pairs):
            ax = axes[k // ncols][k % ncols]
            for label in labels:
                sub = per_pred[per_pred["source_ref"] == label]
                x_col = f"{metric}_rmsd_{a}"
                y_col = f"{metric}_rmsd_{b}"
                if x_col not in sub.columns or y_col not in sub.columns:
                    continue
                valid = sub[[x_col, y_col]].dropna()
                ax.scatter(
                    valid[x_col],
                    valid[y_col],
                    c=self.colors.get(label, "#888"),
                    label=f"bin>{label}",
                    s=18,
                    alpha=0.7,
                    edgecolor="white",
                    linewidth=0.3,
                )
            ax.set_xlabel(f"{metric} RMSD vs {a} (A)")
            ax.set_ylabel(f"{metric} RMSD vs {b} (A)")
            if "min" in rng:
                ax.set_xlim(rng["min"], rng["max"])
                ax.set_ylim(rng["min"], rng["max"])
            ax.legend(fontsize=8, frameon=False)
        for k in range(n, nrows * ncols):
            axes[k // ncols][k % ncols].set_visible(False)
        fig.suptitle(
            f"{metric.title()} RMSD scatter - "
            "predictions colored by source ref-bin"
        )
        fig.tight_layout()
        for fmt in self.formats:
            fig.savefig(
                self.output_dir / f"scatter_{metric}.{fmt}",
                dpi=150,
                bbox_inches="tight",
            )
        plt.close(fig)
