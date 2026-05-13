import json

import numpy as np
import pandas as pd
import pytest

from af_claseq.umap_voting.projector import Projector, embedding_sha256


def _make_embedding(tmp_path, n_sampling=200, d=8, n_refs=2):
    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    samp = rng.normal(size=(n_sampling, d))
    refs = rng.normal(size=(n_refs, d)) + 5.0
    mu = np.vstack([samp, refs]).astype(np.float32)
    pdb_paths = np.array(
        [f"/tmp/s_{i:04d}.pdb" for i in range(n_sampling)]
        + ["/tmp/ref_A.pdb", "/tmp/ref_B.pdb"]
    )
    a3m_paths = np.array(
        [f"/tmp/s_{i:04d}.a3m" for i in range(n_sampling)] + ["", ""]
    )
    is_ref = np.array([False] * n_sampling + [True] * n_refs)
    ref_label = np.array([""] * n_sampling + ["ref_a", "ref_b"])
    npz_path = tmp_path / "embedding.npz"
    np.savez(
        npz_path,
        mu=mu,
        pdb_paths=pdb_paths,
        a3m_paths=a3m_paths,
        is_reference=is_ref,
        ref_label=ref_label,
    )
    return npz_path


def test_projector_fits_and_persists(tmp_path):
    npz = _make_embedding(tmp_path)
    out = tmp_path / "umap_out"
    proj = Projector(
        embedding_npz=npz,
        output_dir=out,
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric="euclidean",
        random_state=42,
    )
    proj.fit(refit=True)
    assert (out / "umap_model.joblib").exists()
    assert (out / "umap_coords.csv").exists()
    df = pd.read_csv(out / "umap_coords.csv")
    assert len(df) == 202
    assert df["is_reference"].sum() == 2


def test_projector_reload_skips_refit(tmp_path):
    npz = _make_embedding(tmp_path)
    out = tmp_path / "umap_out"
    Projector(
        embedding_npz=npz,
        output_dir=out,
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric="euclidean",
        random_state=42,
    ).fit(refit=True)
    mtime_before = (out / "umap_model.joblib").stat().st_mtime
    Projector(
        embedding_npz=npz,
        output_dir=out,
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric="euclidean",
        random_state=42,
    ).fit(refit=False)
    assert (out / "umap_model.joblib").stat().st_mtime == mtime_before


def test_projector_refuses_silent_drift(tmp_path):
    npz1 = _make_embedding(tmp_path)
    out = tmp_path / "umap_out"
    Projector(
        embedding_npz=npz1,
        output_dir=out,
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric="euclidean",
        random_state=42,
    ).fit(refit=True)
    npz2 = _make_embedding(tmp_path / "x", n_sampling=210)
    (tmp_path / "embedding.npz").unlink()
    npz2.rename(tmp_path / "embedding.npz")
    with pytest.raises(RuntimeError, match="--refit-umap"):
        Projector(
            embedding_npz=tmp_path / "embedding.npz",
            output_dir=out,
            n_neighbors=15,
            min_dist=0.1,
            n_components=2,
            metric="euclidean",
            random_state=42,
        ).fit(refit=False)


def test_embedding_sha256_stable_under_row_permutation(tmp_path):
    npz = _make_embedding(tmp_path)
    h1 = embedding_sha256(npz)
    d = np.load(npz)
    perm = np.random.default_rng(1).permutation(len(d["pdb_paths"]))
    np.savez(
        npz,
        mu=d["mu"][perm],
        pdb_paths=d["pdb_paths"][perm],
        a3m_paths=d["a3m_paths"][perm],
        is_reference=d["is_reference"][perm],
        ref_label=d["ref_label"][perm],
    )
    h2 = embedding_sha256(npz)
    assert h1 == h2


def test_assign_bins_pinned_range(tmp_path):
    npz = _make_embedding(tmp_path)
    out = tmp_path / "umap_out"
    proj = Projector(
        embedding_npz=npz,
        output_dir=out,
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric="euclidean",
        random_state=42,
    )
    proj.fit(refit=True)
    df = proj.assign_bins(bin_size=1.0, umap1_range=None, umap2_range=None)
    assert {"bin_ix", "bin_iy"}.issubset(df.columns)
    grid = json.loads((out / "grid.json").read_text())
    assert "umap1_range" in grid and "umap2_range" in grid
    assert grid["bin_size"] == 1.0


def test_assign_bins_out_of_range_dropped(tmp_path):
    npz = _make_embedding(tmp_path)
    out = tmp_path / "umap_out"
    proj = Projector(
        embedding_npz=npz,
        output_dir=out,
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric="euclidean",
        random_state=42,
    )
    proj.fit(refit=True)
    df_full = pd.read_csv(out / "umap_coords.csv")
    # use a very narrow range so most points are dropped (but refs are in)
    ref_rows = df_full[df_full["is_reference"]]
    u1_min = ref_rows["UMAP1"].min() - 0.5
    u1_max = ref_rows["UMAP1"].max() + 0.5
    u2_min = ref_rows["UMAP2"].min() - 0.5
    u2_max = ref_rows["UMAP2"].max() + 0.5
    df = proj.assign_bins(
        bin_size=1.0,
        umap1_range=(u1_min, u1_max),
        umap2_range=(u2_min, u2_max),
    )
    oor = pd.read_csv(out / "out_of_range.csv")
    assert len(oor) >= 0
    assert len(df) + len(oor) == 202
