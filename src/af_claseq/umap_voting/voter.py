"""Option F top-K sequence voting per reference bin."""

from __future__ import annotations

import json
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from af_claseq.utils.logging_utils import get_logger

from .config import UmapVotingConfig

logger = get_logger("umap_voting.voter")


def _parse_one(a3m_path: str) -> List[str]:
    """Parse a single A3M, return non-query sequences (uppercase only)."""
    seqs: List[str] = []
    current_seq_lines: List[str] = []
    header_count = 0

    with open(a3m_path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if header_count > 0 and current_seq_lines:
                    raw = "".join(current_seq_lines)
                    seqs.append("".join(ch for ch in raw if not ch.islower()))
                current_seq_lines = []
                header_count += 1
            else:
                if header_count > 0:
                    current_seq_lines.append(line)

    if header_count > 0 and current_seq_lines:
        raw = "".join(current_seq_lines)
        seqs.append("".join(ch for ch in raw if not ch.islower()))

    # skip first (query)
    return seqs[1:] if len(seqs) > 1 else []


class Voter:
    def __init__(self, config: UmapVotingConfig):
        self.cfg = config
        self.voting_dir = config.get_voting_dir()
        self.voting_dir.mkdir(parents=True, exist_ok=True)
        (self.voting_dir / "a3ms").mkdir(exist_ok=True)
        (self.voting_dir / "votes").mkdir(exist_ok=True)

    def vote(
        self,
        umap_coords_csv: str | Path,
        query_header: Optional[str] = None,
        query_seq: Optional[str] = None,
    ) -> pd.DataFrame:
        """Run Option F voting and return voting_summary DataFrame."""
        df = pd.read_csv(umap_coords_csv)
        refs = df[df["is_reference"]].copy()
        sampling = df[~df["is_reference"]].copy()

        top_k = self.cfg.binning.top_k
        min_recs = self.cfg.binning.min_records_per_bin

        # Group sampling A3Ms by bin
        bin_groups: Dict[Tuple[int, int], List[str]] = {}
        for _, row in sampling.iterrows():
            key = (int(row["bin_ix"]), int(row["bin_iy"]))
            a3m = row["a3m_path"]
            if not pd.isna(a3m) and str(a3m).strip() != "":
                bin_groups.setdefault(key, []).append(str(a3m))

        summary_rows = []

        for _, ref_row in refs.iterrows():
            label = ref_row["ref_label"]
            bix, biy = int(ref_row["bin_ix"]), int(ref_row["bin_iy"])
            bin_key = (bix, biy)

            a3ms_in_bin = bin_groups.get(bin_key, [])
            n_a3ms = len(a3ms_in_bin)
            low_density = n_a3ms < min_recs

            if n_a3ms == 0:
                logger.warning(f"Bin ({bix},{biy}) for {label}: no sampling A3Ms")
                summary_rows.append({
                    "ref_label": label, "bin_ix": bix, "bin_iy": biy,
                    "n_a3ms": 0, "n_records": 0, "n_distinct": 0,
                    "top1_count": 0, "top_k_used": 0, "low_density": True,
                })
                continue

            if low_density:
                logger.warning(f"Bin ({bix},{biy}) for {label}: {n_a3ms} A3Ms < {min_recs} minimum")

            # Parse in parallel
            with Pool(processes=min(4, n_a3ms)) as pool:
                all_seqs_nested = pool.map(_parse_one, a3ms_in_bin)

            counter: Counter = Counter()
            for seq_list in all_seqs_nested:
                counter.update(seq_list)

            n_records = sum(counter.values())
            n_distinct = len(counter)

            ranked = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
            top = ranked[:top_k]
            top_k_used = len(top)
            top1_count = top[0][1] if top else 0

            # Write votes CSV
            votes_path = self.voting_dir / "votes" / f"bin_{bix:02d}_{biy:02d}_{label}.csv"
            votes_df = pd.DataFrame([
                {"rank": i + 1, "seq": s, "count": c, "header": f"rank_{i+1:03d}_count_{c:04d}"}
                for i, (s, c) in enumerate(ranked)
            ])
            votes_df.to_csv(votes_path, index=False)

            # Write output A3M
            a3m_path = self.voting_dir / "a3ms" / f"bin_{bix:02d}_{biy:02d}_{label}.a3m"
            with open(a3m_path, "w") as fh:
                if query_header and query_seq:
                    fh.write(f">{query_header}\n{query_seq}\n")
                for i, (seq, count) in enumerate(top, 1):
                    fh.write(f">rank_{i:03d}_count_{count:04d}\n{seq}\n")

            summary_rows.append({
                "ref_label": label, "bin_ix": bix, "bin_iy": biy,
                "n_a3ms": n_a3ms, "n_records": n_records, "n_distinct": n_distinct,
                "top1_count": top1_count, "top_k_used": top_k_used,
                "low_density": low_density,
            })
            logger.info(
                f"Bin ({bix},{biy}) {label}: {n_a3ms} A3Ms, "
                f"{n_records} records, {n_distinct} distinct, top1={top1_count}"
            )

        summary = pd.DataFrame(summary_rows)
        summary.to_csv(self.voting_dir / "voting_summary.csv", index=False)
        logger.info(f"Voting summary saved ({len(summary)} refs)")
        return summary

    # ------------------------------------------------------------------
    # Bin visualization
    # ------------------------------------------------------------------

    def plot_voting_bins(self, umap_coords_csv: str | Path) -> List[Path]:
        """UMAP scatter with bin rectangles for each reference."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        df = pd.read_csv(umap_coords_csv)
        grid_path = self.cfg.get_umap_dir() / "grid.json"
        grid = json.loads(grid_path.read_text())
        bs = grid["bin_size"]
        u1_min = grid["umap1_range"][0]
        u2_min = grid["umap2_range"][0]

        refs = df[df["is_reference"]]
        sampling = df[~df["is_reference"]]
        colors_map = self.cfg.plotting.colors

        fig, ax = plt.subplots(figsize=(9, 8))
        ax.scatter(sampling["UMAP1"], sampling["UMAP2"],
                   c="lightgrey", s=3, alpha=0.4, zorder=1)

        for _, rr in refs.iterrows():
            label = rr["ref_label"]
            color = colors_map.get(label, "black")
            bix, biy = int(rr["bin_ix"]), int(rr["bin_iy"])
            x0 = u1_min + bix * bs
            y0 = u2_min + biy * bs

            rect = Rectangle((x0, y0), bs, bs,
                              linewidth=2, edgecolor=color,
                              facecolor=color, alpha=0.15, zorder=5)
            ax.add_patch(rect)

            ax.scatter(rr["UMAP1"], rr["UMAP2"], c=color,
                       marker="*", s=250, edgecolors="black", linewidths=0.8,
                       zorder=10, label=label)

        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_title("UMAP with voting bins")
        ax.legend(loc="best")
        plt.tight_layout()

        plots: List[Path] = []
        for fmt in self.cfg.plotting.formats:
            p = self.voting_dir / f"voting_bins.{fmt}"
            fig.savefig(p, dpi=300, bbox_inches="tight")
            plots.append(p)
        plt.close(fig)

        logger.info(f"Generated voting bin plot(s)")
        return plots
