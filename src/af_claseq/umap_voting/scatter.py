"""RMSD computation vs references + scatter plot generation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from af_claseq.utils.logging_utils import get_logger
from af_claseq.utils.structure_analysis import StructureAnalyzer

from .config import UmapVotingConfig
from .coords import _parse_index_spec

logger = get_logger("umap_voting.scatter")


def _load_rmsd_specs(
    config_json: str | Path,
) -> List[Dict[str, Any]]:
    """Load superposition + RMSD index specs from structure-analysis JSON."""
    with open(config_json) as f:
        sa = json.load(f)
    specs = []
    for crit in sa.get("filter_criteria", []):
        specs.append({
            "name": crit["name"],
            "superposition_indices": _parse_index_spec(crit["superposition_indices"]),
            "rmsd_indices": _parse_index_spec(crit["rmsd_indices"]),
            "ref_pdb": crit["ref_pdb"],
        })
    return specs


class ScatterBuilder:
    def __init__(self, config: UmapVotingConfig):
        self.cfg = config
        self.scatter_dir = config.get_scatter_dir()
        self.scatter_dir.mkdir(parents=True, exist_ok=True)

    def _compute_rmsd(
        self,
        pred_pdb: str | Path,
        references: pd.DataFrame,
        specs: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Compute CA-RMSD of pred_pdb vs each reference for each spec."""
        analyzer = StructureAnalyzer()
        results: Dict[str, float] = {}

        for spec in specs:
            ref_label = None
            ref_chain = "A"
            for _, rrow in references.iterrows():
                if rrow["ref_pdb"] == spec["ref_pdb"] or spec["name"].startswith(rrow["ref_label"]):
                    ref_label = rrow["ref_label"]
                    ref_chain = rrow.get("ref_chain", "A")
                    break
            if ref_label is None:
                continue

            metric_prefix = "local" if spec["rmsd_indices"] != spec["superposition_indices"] else "global"
            col = f"{metric_prefix}_rmsd_{ref_label}"

            try:
                rmsd = analyzer.calculate_ca_rmsd(
                    reference_pdb=spec["ref_pdb"],
                    target_pdb=str(pred_pdb),
                    superposition_indices=spec["superposition_indices"],
                    rmsd_indices=spec["rmsd_indices"],
                    chain_id="A",
                    reference_chain_id=ref_chain,
                )
                results[col] = rmsd
            except Exception as e:
                logger.warning(f"RMSD failed for {pred_pdb} vs {ref_label}: {e}")
                results[col] = float("nan")

        return results

    def build(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Walk prediction dirs, compute RMSDs, write CSVs + plots."""
        pred_dir = self.cfg.get_predictions_dir()
        refs = pd.read_csv(self.cfg.inputs.references_csv)
        sa_cfg = self.cfg.structure_analysis
        if sa_cfg is None:
            raise RuntimeError("structure_analysis config required for scatter")

        specs = _load_rmsd_specs(sa_cfg.config_json)

        bin_pattern = re.compile(r"^bin_(\d+)_(\d+)_(.+)$")
        rows = []

        for sub in sorted(pred_dir.iterdir()):
            if not sub.is_dir():
                continue
            m = bin_pattern.match(sub.name)
            if m is None:
                continue
            source_ref = m.group(3)
            pdbs = sorted(sub.glob("*.pdb"))
            for pdb in pdbs:
                rmsds = self._compute_rmsd(pdb, refs, specs)
                rmsds["ref_bin_label"] = sub.name
                rmsds["source_ref"] = source_ref
                rmsds["pred_path"] = str(pdb)
                rows.append(rmsds)

        if not rows:
            logger.warning("No prediction PDBs found")
            empty = pd.DataFrame()
            return empty, empty

        per_pred = pd.DataFrame(rows)
        per_pred.to_csv(self.scatter_dir / "per_pred.csv", index=False)
        logger.info(f"per_pred.csv: {len(per_pred)} rows")

        summary = self._build_summary(per_pred, refs)
        summary.to_csv(self.scatter_dir / "summary.csv", index=False)

        self._plot_scatter(per_pred, refs)

        return per_pred, summary

    def _build_summary(self, per_pred: pd.DataFrame, refs: pd.DataFrame) -> pd.DataFrame:
        """Build summary.csv with per-ref-label statistics."""
        ref_labels = refs["ref_label"].tolist()
        summary_rows = []

        for label in ref_labels:
            subset = per_pred[per_pred["source_ref"] == label]
            n = len(subset)
            if n == 0:
                continue

            # Find target RMSD column (any metric containing this ref_label)
            target_cols = [c for c in per_pred.columns if c.endswith(f"_rmsd_{label}")]
            other_labels = [l for l in ref_labels if l != label]
            other_cols = []
            for ol in other_labels:
                other_cols.extend(c for c in per_pred.columns if c.endswith(f"_rmsd_{ol}"))

            row: Dict[str, Any] = {"ref_label": label, "n": n}

            for tc in target_cols:
                vals = subset[tc].dropna()
                prefix = tc.replace(f"_rmsd_{label}", "")
                row[f"med_target_{prefix}"] = float(vals.median()) if len(vals) > 0 else float("nan")
                row[f"pct_lt_2A_{prefix}"] = float((vals < 2.0).mean()) if len(vals) > 0 else 0.0
                row[f"pct_lt_1p5A_{prefix}"] = float((vals < 1.5).mean()) if len(vals) > 0 else 0.0

            for oc in other_cols:
                vals = subset[oc].dropna()
                row[f"med_{oc}"] = float(vals.median()) if len(vals) > 0 else float("nan")

            summary_rows.append(row)

        return pd.DataFrame(summary_rows)

    def _plot_scatter(self, per_pred: pd.DataFrame, refs: pd.DataFrame) -> None:
        """Generate one scatter plot per metric."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_cfg = self.cfg.plotting
        ref_labels = refs["ref_label"].tolist()
        colors_map = plot_cfg.colors

        if len(ref_labels) < 2:
            logger.warning("Need >= 2 references for scatter plot")
            return

        sa_cfg = self.cfg.structure_analysis
        metrics = sa_cfg.metrics if sa_cfg else ["local", "global"]

        for metric in metrics:
            cols = {l: f"{metric}_rmsd_{l}" for l in ref_labels}
            missing = [l for l, c in cols.items() if c not in per_pred.columns]
            if missing:
                logger.warning(f"Scatter {metric}: missing columns for {missing}, skipping")
                continue

            fig, ax = plt.subplots(figsize=(7, 7))

            for label in ref_labels:
                subset = per_pred[per_pred["source_ref"] == label]
                if subset.empty:
                    continue

                other_labels = [l for l in ref_labels if l != label]
                x_col = cols[label]
                y_cols = [cols[l] for l in other_labels]

                color = colors_map.get(label, None)
                for y_col in y_cols:
                    plot_df = subset[[x_col, y_col]].dropna()
                    ax.scatter(
                        plot_df[x_col], plot_df[y_col],
                        c=color, s=15, alpha=0.6, label=label, edgecolors="none",
                    )

            ax.set_xlabel(f"Target RMSD ({metric}, A)")
            ax.set_ylabel(f"Other RMSD ({metric}, A)")

            mr = plot_cfg.metric_ranges.get(metric, {})
            if "min" in mr and "max" in mr:
                ax.set_xlim(mr["min"], mr["max"])
                ax.set_ylim(mr["min"], mr["max"])
            if "ticks" in mr:
                ax.set_xticks(mr["ticks"])
                ax.set_yticks(mr["ticks"])

            ax.plot([0, 20], [0, 20], "k--", alpha=0.3, linewidth=0.8)
            ax.set_aspect("equal")

            handles, labels_legend = ax.get_legend_handles_labels()
            by_label = dict(zip(labels_legend, handles))
            ax.legend(by_label.values(), by_label.keys(), loc="best")

            ax.set_title(f"Prediction RMSD scatter ({metric})")
            plt.tight_layout()

            for fmt in plot_cfg.formats:
                p = self.scatter_dir / f"scatter_{metric}.{fmt}"
                fig.savefig(p, dpi=300, bbox_inches="tight")
            plt.close(fig)

        logger.info(f"Generated scatter plots for metrics: {metrics}")
