"""Tests for predictor: command building and dry-run mode."""

import pytest
import yaml
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock


@pytest.fixture
def predictor_config(tmp_path):
    emb = tmp_path / "emb.npz"
    np.savez(emb, mu=np.zeros((5, 3)))
    refs = tmp_path / "refs.csv"
    refs.write_text("ref_pdb,ref_label,ref_chain\n")
    qa3m = tmp_path / "q.a3m"
    qa3m.write_text(">q\nACDEF\n")

    cfg = {
        "general": {"protein_name": "T", "base_dir": str(tmp_path / "out")},
        "inputs": {
            "embedding_npz": str(emb),
            "references_csv": str(refs),
            "query_a3m": str(qa3m),
        },
        "slurm": {
            "conda_env_path": "/fake/env",
            "account": "TEST",
            "partition": "debug",
        },
        "structure_prediction": {
            "num_models": 1,
            "num_seeds": 1,
            "prediction_mode": "monomer",
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.dump(cfg))

    from af_claseq.umap_voting.config import UmapVotingConfig
    return UmapVotingConfig.from_yaml(cfg_path)


class TestPredictor:
    def test_build_colabfold_cmd_monomer(self, predictor_config):
        from af_claseq.umap_voting.predictor import Predictor
        pred = Predictor(predictor_config)
        cmd = pred._build_colabfold_cmd("/a.a3m", "/out")
        assert "alphafold2_ptm" in cmd
        assert "--num-models 1" in cmd

    def test_build_colabfold_cmd_homodimer(self, tmp_path):
        emb = tmp_path / "emb.npz"
        np.savez(emb, mu=np.zeros((5, 3)))
        refs = tmp_path / "refs.csv"
        refs.write_text("ref_pdb,ref_label,ref_chain\n")
        qa3m = tmp_path / "q.a3m"
        qa3m.write_text(">q\nACDEF\n")

        cfg = {
            "general": {"protein_name": "T", "base_dir": str(tmp_path / "out")},
            "inputs": {
                "embedding_npz": str(emb),
                "references_csv": str(refs),
                "query_a3m": str(qa3m),
            },
            "slurm": {"conda_env_path": "/e", "account": "A"},
            "structure_prediction": {"prediction_mode": "homodimer"},
        }
        cfg_path = tmp_path / "c.yaml"
        cfg_path.write_text(yaml.dump(cfg))
        from af_claseq.umap_voting.config import UmapVotingConfig
        from af_claseq.umap_voting.predictor import Predictor
        config = UmapVotingConfig.from_yaml(cfg_path)
        pred = Predictor(config)
        cmd = pred._build_colabfold_cmd("/a.a3m", "/out")
        assert "alphafold2_multimer_v3" in cmd

    def test_dry_run_no_submission(self, predictor_config, tmp_path):
        from af_claseq.umap_voting.predictor import Predictor
        pred = Predictor(predictor_config)

        a3m_dir = predictor_config.get_voting_dir() / "a3ms"
        a3m_dir.mkdir(parents=True, exist_ok=True)
        (a3m_dir / "bin_00_00_test.a3m").write_text(">q\nACDEF\n")

        manifest = pred.run(dry_run=True)
        assert len(manifest) == 1
        assert manifest.iloc[0]["state"] == "dry_run"
