from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from af_claseq.umap_voting.config import VaeTrainConfig
from af_claseq.umap_voting.vae.train import VaeTrainer

FIXT = Path(__file__).parent / "fixtures"
PDBS = FIXT / "synthetic"


@pytest.fixture
def vae_cfg(tmp_path):
    structures = pd.DataFrame(
        {
            "pdb_path": sorted(str(p) for p in PDBS.glob("struct_*.pdb")),
            "a3m_path": [""] * 20,
        }
    )
    refs = pd.DataFrame(
        {
            "ref_label": ["ref_a", "ref_b"],
            "ref_pdb": [str(PDBS / "ref_A.pdb"), str(PDBS / "ref_B.pdb")],
            "ref_chain": ["A", "A"],
        }
    )
    s_csv = tmp_path / "structures.csv"
    structures.to_csv(s_csv, index=False)
    r_csv = tmp_path / "refs.csv"
    refs.to_csv(r_csv, index=False)
    sa_json = tmp_path / "sa.json"
    sa_json.write_text(
        '{"basics":{"full_index":{"start":1,"end":30},'
        '"local_index":{"start":5,"end":20}}}'
    )
    out_dir = tmp_path / "vae_out"
    yml = tmp_path / "vae.yaml"
    yml.write_text(
        f"""
general: {{protein_name: T, base_dir: "{out_dir}", random_seed: 42, device: cpu}}
inputs: {{structures_csv: "{s_csv}", references_csv: "{r_csv}"}}
structure_analysis: {{config_json: "{sa_json}", coord_target: local}}
vae:
  model: {{latent_dim: 4, hidden_channels: [8, 16], use_residual: true}}
  training: {{epochs: 3, batch_size: 4, learning_rate: 1.0e-3, kl_weight: 0.1,
              val_split: 0.1, save_best_only: true, early_stopping_patience: 50}}
output: {{embedding_filename: embedding.npz, save_checkpoints_every: 200}}
"""
    )
    return VaeTrainConfig.from_yaml(yml)


def test_trainer_writes_embedding_npz(vae_cfg):
    VaeTrainer(vae_cfg).train()
    npz_path = Path(vae_cfg.general.base_dir) / "vae" / "embedding.npz"
    assert npz_path.exists()
    data = np.load(npz_path, allow_pickle=False)
    # 20 sampling + 2 refs
    assert data["mu"].shape == (22, 4)
    assert data["is_reference"].sum() == 2
    ref_labels = data["ref_label"][data["is_reference"]]
    assert set(ref_labels) == {"ref_a", "ref_b"}


def test_trainer_writes_checkpoint_and_norm(vae_cfg):
    VaeTrainer(vae_cfg).train()
    vae_dir = Path(vae_cfg.general.base_dir) / "vae"
    # PyTorch model state_dict checkpoint
    assert (vae_dir / "protein_vae_best.pth").exists()
    assert (vae_dir / "normalization_params.npz").exists()
