"""Tests for UMAP projection: fit, cache, drift detection, binning."""

import json
import pytest
import numpy as np
import yaml
from pathlib import Path

from af_claseq.umap_voting.projector import embedding_sha256


class TestEmbeddingSha256:
    def test_order_independent(self, tmp_path):
        mu = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        p1 = tmp_path / "emb1.npz"
        np.savez(p1, mu=mu)

        mu_shuffled = mu[[2, 0, 1]]
        p2 = tmp_path / "emb2.npz"
        np.savez(p2, mu=mu_shuffled)

        assert embedding_sha256(p1) == embedding_sha256(p2)

    def test_different_data_different_hash(self, tmp_path):
        p1 = tmp_path / "a.npz"
        np.savez(p1, mu=np.array([[1.0, 2.0]]))
        p2 = tmp_path / "b.npz"
        np.savez(p2, mu=np.array([[3.0, 4.0]]))
        assert embedding_sha256(p1) != embedding_sha256(p2)


class TestProjectorBinning:
    def test_bin_assignment(self, tmp_path):
        import pandas as pd
        from af_claseq.umap_voting.config import UmapVotingConfig

        emb = tmp_path / "emb.npz"
        np.savez(emb, mu=np.random.randn(10, 3),
                 pdb_paths=np.array([f"p{i}" for i in range(10)]),
                 a3m_paths=np.array([f"a{i}" for i in range(10)]),
                 is_reference=np.array([False]*8 + [True]*2),
                 ref_label=np.array([""]*8 + ["rA", "rB"]))
        refs = tmp_path / "refs.csv"
        refs.write_text("ref_pdb,ref_label,ref_chain\n/a,rA,A\n/b,rB,A\n")
        qa3m = tmp_path / "q.a3m"
        qa3m.write_text(">q\nACDEF\n")

        cfg_data = {
            "general": {"protein_name": "T", "base_dir": str(tmp_path / "out")},
            "inputs": {
                "embedding_npz": str(emb),
                "references_csv": str(refs),
                "query_a3m": str(qa3m),
            },
            "binning": {"bin_size": 2.0},
        }
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(yaml.dump(cfg_data))
        config = UmapVotingConfig.from_yaml(cfg_path)

        from af_claseq.umap_voting.projector import Projector
        proj = Projector(config)

        # Make a fake df with known UMAP coords
        df = pd.DataFrame({
            "pdb_path": [f"p{i}" for i in range(10)],
            "a3m_path": [f"a{i}" for i in range(10)],
            "UMAP1": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 1.0, 5.0],
            "UMAP2": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 1.0, 5.0],
            "is_reference": [False]*8 + [True, True],
            "ref_label": [""]*8 + ["rA", "rB"],
        })

        result = proj.assign_bins(df)
        assert "bin_ix" in result.columns
        assert "bin_iy" in result.columns
        assert (config.get_umap_dir() / "grid.json").exists()
        assert (config.get_umap_dir() / "umap_coords.csv").exists()
