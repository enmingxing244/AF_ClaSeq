from pathlib import Path

import pytest

from af_claseq.umap_voting.config import UmapVotingConfig, VaeTrainConfig

FIXT = Path(__file__).parent / "fixtures"


def test_vae_config_roundtrip():
    cfg = VaeTrainConfig.from_yaml(FIXT / "minimal_vae_train.yaml")
    assert cfg.general.protein_name == "TEST"
    assert cfg.general.random_seed == 42
    assert cfg.vae.model.latent_dim == 8
    assert cfg.vae.training.epochs == 5
    assert cfg.structure_analysis.coord_target == "local"


def test_voting_config_roundtrip():
    cfg = UmapVotingConfig.from_yaml(FIXT / "minimal_umap_voting.yaml")
    assert cfg.umap.n_neighbors == 30
    assert cfg.binning.bin_size == 1.0
    assert cfg.binning.top_k == 16
    assert cfg.structure_prediction.num_models == 5
    assert cfg.slurm.partition == "nextgen"
    assert cfg.plotting.panels_per_row == 2
    assert cfg.inputs.query_a3m == "/tmp/query.a3m"


def test_voting_config_missing_required_field(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("general:\n  protein_name: TEST\n")
    with pytest.raises(ValueError, match="missing required"):
        UmapVotingConfig.from_yaml(bad)


def test_vae_config_coord_target_validated(tmp_path):
    src = (FIXT / "minimal_vae_train.yaml").read_text()
    bad = src.replace('coord_target: "local"', 'coord_target: "elbow"')
    p = tmp_path / "bad.yaml"
    p.write_text(bad)
    with pytest.raises(ValueError, match="coord_target"):
        VaeTrainConfig.from_yaml(p)
