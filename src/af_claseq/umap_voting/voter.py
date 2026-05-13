"""Option F voting: top-K most frequent sequences in each ref's bin."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from af_claseq.utils.logging_utils import get_logger
from af_claseq.utils.sequence_processing import read_a3m_to_dict

logger = get_logger("umap_voting.voter")


def _parse_one(path: str) -> Tuple[str, Optional[List[Tuple[str, str]]]]:
    try:
        d = read_a3m_to_dict(path)
    except Exception as e:
        logger.warning(f"A3M parse failed for {path}: {e}")
        return path, None
    items = list(d.items())
    if not items:
        return path, None
    # skip the first record (query) by index
    return path, items[1:]


@dataclass
class Voter:
    umap_coords_csv: str | Path
    references_csv: str | Path
    output_dir: str | Path
    top_k: int = 16
    min_records_per_bin: int = 50
    query_header: str = ""
    query_seq: str = ""
    n_workers: int = 4

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        (self.output_dir / "a3ms").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "votes").mkdir(parents=True, exist_ok=True)

    def vote(self) -> pd.DataFrame:
        coords = pd.read_csv(self.umap_coords_csv)
        refs = pd.read_csv(self.references_csv)
        sampling = coords[~coords["is_reference"]].copy()
        sampling = sampling[sampling["a3m_path"].astype(str).str.len() > 0]

        bin_to_a3ms: Dict[Tuple[int, int], set] = {}
        for _, row in sampling.iterrows():
            bk = (int(row["bin_ix"]), int(row["bin_iy"]))
            bin_to_a3ms.setdefault(bk, set()).add(str(row["a3m_path"]))

        summary_rows: List[Dict] = []
        for _, ref in refs.iterrows():
            label = str(ref["ref_label"])
            ref_rows = coords[
                (coords["is_reference"]) & (coords["ref_label"] == label)
            ]
            if ref_rows.empty:
                logger.warning(
                    f"reference {label} not in UMAP coords; skipping"
                )
                summary_rows.append(
                    dict(
                        ref_label=label,
                        bin_ix=-1,
                        bin_iy=-1,
                        n_a3ms=0,
                        n_records=0,
                        n_distinct=0,
                        top1_count=0,
                        top_k_used=0,
                        low_density=True,
                    )
                )
                continue

            bk = (
                int(ref_rows.iloc[0]["bin_ix"]),
                int(ref_rows.iloc[0]["bin_iy"]),
            )
            paths = sorted(bin_to_a3ms.get(bk, set()))
            base = f"bin_{bk[0]:02d}_{bk[1]:02d}_{label}"

            if not paths:
                logger.warning(f"{label}: empty bin {bk}; skipping")
                summary_rows.append(
                    dict(
                        ref_label=label,
                        bin_ix=bk[0],
                        bin_iy=bk[1],
                        n_a3ms=0,
                        n_records=0,
                        n_distinct=0,
                        top1_count=0,
                        top_k_used=0,
                        low_density=True,
                    )
                )
                continue

            counter: Counter = Counter()
            n_records = 0
            header_for_seq: Dict[str, str] = {}

            with Pool(min(self.n_workers, len(paths))) as pool:
                for _, records in pool.imap_unordered(
                    _parse_one, paths, chunksize=4
                ):
                    if records is None:
                        continue
                    for hdr, seq in records:
                        if not seq:
                            continue
                        counter[seq] += 1
                        n_records += 1
                        if seq not in header_for_seq:
                            header_for_seq[seq] = hdr

            # deterministic tie-breaking: sort by (-count, sequence)
            ranked = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[
                : self.top_k
            ]

            top1 = ranked[0][1] if ranked else 0
            low = n_records < self.min_records_per_bin

            a3m_out = self.output_dir / "a3ms" / f"{base}.a3m"
            with open(a3m_out, "w") as f:
                if self.query_seq:
                    f.write(f"#{len(self.query_seq)}\t1\n")
                    hdr = self.query_header or "query"
                    f.write(f">{hdr}\n{self.query_seq}\n")
                for r, (seq, k) in enumerate(ranked, 1):
                    safe = re.sub(
                        r"\s+", "_", header_for_seq.get(seq, "")
                    ).replace(">", "")
                    f.write(f">rank{r}_count{k}_{safe}\n{seq}\n")

            votes_out = self.output_dir / "votes" / f"{base}.csv"
            pd.DataFrame(
                [
                    dict(
                        rank=r,
                        seq=s,
                        count=k,
                        header=header_for_seq.get(s, ""),
                    )
                    for r, (s, k) in enumerate(ranked, 1)
                ]
            ).to_csv(votes_out, index=False)

            summary_rows.append(
                dict(
                    ref_label=label,
                    bin_ix=bk[0],
                    bin_iy=bk[1],
                    n_a3ms=len(paths),
                    n_records=n_records,
                    n_distinct=len(counter),
                    top1_count=top1,
                    top_k_used=len(ranked),
                    low_density=low,
                )
            )

        summary = pd.DataFrame(summary_rows)
        summary.to_csv(self.output_dir / "voting_summary.csv", index=False)
        return summary
