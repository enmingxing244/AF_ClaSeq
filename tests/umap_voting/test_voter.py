from pathlib import Path

import pandas as pd
import pytest

from af_claseq.umap_voting.voter import Voter

FIXT = Path(__file__).parent / "fixtures"
A3MS = FIXT / "a3ms"


@pytest.fixture
def mini_coords(tmp_path):
    rows = []
    for i in range(10):
        rows.append(
            {
                "pdb_path": f"/tmp/s_{i:02d}.pdb",
                "a3m_path": str(A3MS / f"a3m_{i:02d}.a3m"),
                "UMAP1": 1.5,
                "UMAP2": 1.5,
                "is_reference": False,
                "ref_label": "",
                "bin_ix": 0,
                "bin_iy": 0,
            }
        )
    rows.append(
        {
            "pdb_path": "/tmp/ref_A.pdb",
            "a3m_path": "",
            "UMAP1": 1.4,
            "UMAP2": 1.4,
            "is_reference": True,
            "ref_label": "ref_a",
            "bin_ix": 0,
            "bin_iy": 0,
        }
    )
    df = pd.DataFrame(rows)
    p = tmp_path / "umap_coords.csv"
    df.to_csv(p, index=False)
    return p


def test_voter_writes_top_k_a3m(mini_coords, tmp_path):
    refs = pd.DataFrame(
        [{"ref_label": "ref_a", "ref_pdb": "/tmp/ref_A.pdb", "ref_chain": "A"}]
    )
    refs_csv = tmp_path / "refs.csv"
    refs.to_csv(refs_csv, index=False)
    v = Voter(
        umap_coords_csv=mini_coords,
        references_csv=refs_csv,
        output_dir=tmp_path / "voting",
        top_k=3,
        min_records_per_bin=5,
        query_seq="M" * 28,
    )
    summary = v.vote()
    assert len(summary) == 1

    out_a3m = tmp_path / "voting" / "a3ms" / "bin_00_00_ref_a.a3m"
    assert out_a3m.exists()
    body = out_a3m.read_text().splitlines()
    # query + 3 ranked sequences = 4 header lines
    assert len([line for line in body if line.startswith(">")]) == 4

    votes_csv = tmp_path / "voting" / "votes" / "bin_00_00_ref_a.csv"
    votes = pd.read_csv(votes_csv)
    assert list(votes["rank"]) == [1, 2, 3]
    # COMMON_A appears in all 10 A3Ms
    assert votes.iloc[0]["count"] == 10
    # COMMON_B appears in 7 A3Ms (i < 7)
    assert votes.iloc[1]["count"] == 7
    # COMMON_C appears in 4 A3Ms (i < 4)
    assert votes.iloc[2]["count"] == 4


def test_voter_skips_ref_with_empty_bin(tmp_path):
    df = pd.DataFrame(
        [
            {
                "pdb_path": "/tmp/ref.pdb",
                "a3m_path": "",
                "UMAP1": 99,
                "UMAP2": 99,
                "is_reference": True,
                "ref_label": "lonely",
                "bin_ix": 99,
                "bin_iy": 99,
            }
        ]
    )
    coords = tmp_path / "c.csv"
    df.to_csv(coords, index=False)
    refs = pd.DataFrame(
        [{"ref_label": "lonely", "ref_pdb": "/x.pdb", "ref_chain": "A"}]
    )
    refs_csv = tmp_path / "r.csv"
    refs.to_csv(refs_csv, index=False)
    v = Voter(
        umap_coords_csv=coords,
        references_csv=refs_csv,
        output_dir=tmp_path / "voting",
        top_k=3,
        min_records_per_bin=5,
        query_seq="M" * 28,
    )
    summary = v.vote()
    assert len(summary) == 1
    assert summary.iloc[0]["n_a3ms"] == 0
    assert bool(summary.iloc[0]["low_density"]) is True


def test_voter_excludes_query_from_top_k(mini_coords, tmp_path):
    """The query sequence (first record in each A3M) must not be voted on."""
    refs = pd.DataFrame(
        [{"ref_label": "ref_a", "ref_pdb": "/tmp/ref_A.pdb", "ref_chain": "A"}]
    )
    refs_csv = tmp_path / "refs.csv"
    refs.to_csv(refs_csv, index=False)
    query = "MTEYKLVVVGAGGVGKSALTIQLIQNHFV"
    v = Voter(
        umap_coords_csv=mini_coords,
        references_csv=refs_csv,
        output_dir=tmp_path / "voting",
        top_k=100,
        min_records_per_bin=1,
        query_seq=query,
    )
    summary = v.vote()
    votes_csv = tmp_path / "voting" / "votes" / "bin_00_00_ref_a.csv"
    votes = pd.read_csv(votes_csv)
    # the query sequence should not appear in the voted sequences
    assert query not in votes["seq"].values
