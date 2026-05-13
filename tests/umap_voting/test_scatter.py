import shutil
from pathlib import Path

import pandas as pd
import pytest

from af_claseq.umap_voting.scatter import ScatterBuilder

FIXT = Path(__file__).parent / "fixtures"
PDBS = FIXT / "synthetic"


@pytest.fixture
def mini_preds_layout(tmp_path):
    preds = tmp_path / "predictions"
    for ref in ("bin_00_00_ref_a", "bin_00_00_ref_b"):
        d = preds / ref
        d.mkdir(parents=True)
        for n in range(4):
            shutil.copy(
                PDBS / f"struct_{n + 1:04d}.pdb",
                d / f"unrelaxed_rank_{n + 1:03d}.pdb",
            )
    refs_csv = tmp_path / "refs.csv"
    pd.DataFrame(
        [
            {
                "ref_label": "ref_a",
                "ref_pdb": str(PDBS / "ref_A.pdb"),
                "ref_chain": "A",
            },
            {
                "ref_label": "ref_b",
                "ref_pdb": str(PDBS / "ref_B.pdb"),
                "ref_chain": "A",
            },
        ]
    ).to_csv(refs_csv, index=False)
    sa_json = tmp_path / "sa.json"
    sa_json.write_text(
        '{"basics":{"full_index":{"start":1,"end":30},'
        '"local_index":{"start":5,"end":20}}}'
    )
    return preds, refs_csv, sa_json


def test_scatter_builds_per_pred_csv(mini_preds_layout, tmp_path):
    preds, refs_csv, sa_json = mini_preds_layout
    sb = ScatterBuilder(
        predictions_dir=preds,
        references_csv=refs_csv,
        structure_analysis_json=sa_json,
        output_dir=tmp_path / "scatter",
        metrics=["local", "global"],
        formats=["png"],
        metric_ranges={
            "local": {"min": 0, "max": 30, "ticks": [0, 10, 20, 30]},
            "global": {"min": 0, "max": 30, "ticks": [0, 10, 20, 30]},
        },
        colors={"ref_a": "#C73E3A", "ref_b": "#2E7AB8"},
        panels_per_row=2,
    )
    sb.build()
    pp = pd.read_csv(tmp_path / "scatter" / "per_pred.csv")
    assert len(pp) == 8
    for col in (
        "local_rmsd_ref_a",
        "local_rmsd_ref_b",
        "global_rmsd_ref_a",
        "global_rmsd_ref_b",
    ):
        assert col in pp.columns
    summary = pd.read_csv(tmp_path / "scatter" / "summary.csv")
    assert {
        "ref_label",
        "n",
        "med_target",
        "med_other",
        "pct_lt_2A",
        "pct_lt_1p5A",
    }.issubset(summary.columns)
    assert (tmp_path / "scatter" / "scatter_local.png").exists()
    assert (tmp_path / "scatter" / "scatter_global.png").exists()
