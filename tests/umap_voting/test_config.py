"""Tests for umap_voting config parsing and validation."""

import pytest
import yaml
from pathlib import Path

from af_claseq.umap_voting.config import (
    GeneralSection,
    VaeTrainConfig,
    UmapVotingConfig,
    StructureAnalysisSection,
    VaeTrainingSection,
    BinningSection,
    StructurePredictionSection,
)
from af_claseq.utils.exceptions import ConfigurationError


class TestGeneralSection:
    def test_from_dict_defaults(self):
        g = GeneralSection.from_dict({"protein_name": "X", "base_dir": "/tmp"})
        assert g.random_seed == 42
        assert g.device == "cpu"

    def test_from_dict_override(self):
        g = GeneralSection.from_dict({
            "protein_name": "Y", "base_dir": "/tmp", "device": "cuda"
        })
        assert g.device == "cuda"


class TestStructureAnalysisSection:
    def test_valid_targets(self):
        StructureAnalysisSection.from_dict({"config_json": "/tmp/x.json", "coord_target": "local"})
        StructureAnalysisSection.from_dict({"config_json": "/tmp/x.json", "coord_target": "global"})

    def test_invalid_target(self):
        with pytest.raises(ConfigurationError, match="coord_target"):
            StructureAnalysisSection.from_dict({"config_json": "/x", "coord_target": "invalid"})


class TestVaeTrainingSection:
    def test_invalid_normalization_mode(self):
        with pytest.raises(ConfigurationError, match="normalization_mode"):
            VaeTrainingSection(normalization_mode="wrong")

    def test_invalid_val_split(self):
        with pytest.raises(ConfigurationError, match="val_split"):
            VaeTrainingSection(val_split=1.5)

    def test_defaults(self):
        t = VaeTrainingSection()
        assert t.normalization_mode == "global"
        assert t.kl_weight == 0.05
        assert t.grad_clip == 1.0


class TestBinningSection:
    def test_invalid_bin_size(self):
        with pytest.raises(ConfigurationError, match="bin_size"):
            BinningSection(bin_size=-1.0)

    def test_invalid_top_k(self):
        with pytest.raises(ConfigurationError, match="top_k"):
            BinningSection(top_k=0)


class TestStructurePredictionSection:
    def test_invalid_mode(self):
        with pytest.raises(ConfigurationError, match="prediction_mode"):
            StructurePredictionSection(prediction_mode="trimer")


class TestVaeTrainConfigFromYaml:
    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            VaeTrainConfig.from_yaml("/nonexistent/config.yaml")

    def test_missing_section(self, tmp_path):
        cfg = tmp_path / "bad.yaml"
        cfg.write_text(yaml.dump({"general": {"protein_name": "X", "base_dir": str(tmp_path)}}))
        with pytest.raises(ConfigurationError, match="Missing required section"):
            VaeTrainConfig.from_yaml(cfg)

    def test_valid_minimal(self, tmp_path):
        sa_json = tmp_path / "sa.json"
        sa_json.write_text('{"basics": {"full_index": {"start": 1, "end": 10}}}')
        structs_csv = tmp_path / "structures.csv"
        structs_csv.write_text("pdb_path,a3m_path\n")
        refs_csv = tmp_path / "references.csv"
        refs_csv.write_text("ref_pdb,ref_label,ref_chain\n")
        (tmp_path / "out").mkdir()

        cfg_data = {
            "general": {"protein_name": "TEST", "base_dir": str(tmp_path / "out")},
            "inputs": {
                "structures_csv": str(structs_csv),
                "references_csv": str(refs_csv),
            },
            "structure_analysis": {
                "config_json": str(sa_json),
                "coord_target": "global",
            },
        }
        cfg_path = tmp_path / "vae.yaml"
        cfg_path.write_text(yaml.dump(cfg_data))

        config = VaeTrainConfig.from_yaml(cfg_path)
        assert config.general.protein_name == "TEST"
        assert config.vae.model.latent_dim == 6
        assert config.vae.training.normalization_mode == "global"


class TestUmapVotingConfigFromYaml:
    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            UmapVotingConfig.from_yaml("/nonexistent/config.yaml")

    def test_valid_minimal(self, tmp_path):
        emb = tmp_path / "embedding.npz"
        import numpy as np
        np.savez(emb, mu=np.zeros((5, 3)))
        refs_csv = tmp_path / "refs.csv"
        refs_csv.write_text("ref_pdb,ref_label,ref_chain\n")
        qa3m = tmp_path / "query.a3m"
        qa3m.write_text(">q\nACDEF\n")

        cfg_data = {
            "general": {"protein_name": "T", "base_dir": str(tmp_path / "out")},
            "inputs": {
                "embedding_npz": str(emb),
                "references_csv": str(refs_csv),
                "query_a3m": str(qa3m),
            },
        }
        cfg_path = tmp_path / "umap.yaml"
        cfg_path.write_text(yaml.dump(cfg_data))

        config = UmapVotingConfig.from_yaml(cfg_path)
        assert config.binning.top_k == 16
        assert config.umap.n_neighbors == 30
