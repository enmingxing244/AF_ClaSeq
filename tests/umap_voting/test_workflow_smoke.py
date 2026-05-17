from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from af_claseq.umap_voting.config import UmapVotingConfig
from af_claseq.umap_voting.workflow import UmapVotingManager

FIXT = Path(__file__).parent / "fixtures"


@pytest.fixture
def smoke_cfg(tmp_path):
    rng = np.random.default_rng(0)
    n = 20
    mu = np.vstack(
        [rng.normal(size=(n, 4)), rng.normal(size=(2, 4)) + 5]
    ).astype(np.float32)
    pdb_paths = [f"/tmp/s_{i:02d}.pdb" for i in range(n)] + [
        "/tmp/ref_A.pdb",
        "/tmp/ref_B.pdb",
    ]
    a3m_paths = [
        str(FIXT / "a3ms" / f"a3m_{i % 10:02d}.a3m") for i in range(n)
    ] + ["", ""]
    is_ref = np.array([False] * n + [True, True])
    ref_label = np.array([""] * n + ["ref_a", "ref_b"])
    emb = tmp_path / "embedding.npz"
    np.savez(
        emb,
        mu=mu,
        pdb_paths=np.array(pdb_paths),
        a3m_paths=np.array(a3m_paths),
        is_reference=is_ref,
        ref_label=ref_label,
    )
    refs = tmp_path / "refs.csv"
    pd.DataFrame(
        [
            {
                "ref_label": "ref_a",
                "ref_pdb": str(FIXT / "synthetic" / "ref_A.pdb"),
                "ref_chain": "A",
            },
            {
                "ref_label": "ref_b",
                "ref_pdb": str(FIXT / "synthetic" / "ref_B.pdb"),
                "ref_chain": "A",
            },
        ]
    ).to_csv(refs, index=False)
    query_a3m = tmp_path / "query.a3m"
    query_a3m.write_text(">query\nMTEYKLVVVGAGGVGKSALTIQLIQNHFV\n")
    sa = tmp_path / "sa.json"
    sa.write_text(
        '{"basics":{"full_index":{"start":1,"end":30},'
        '"local_index":{"start":5,"end":20}}}'
    )
    yml = tmp_path / "voting.yaml"
    yml.write_text(
        f"""
general: {{protein_name: T, base_dir: "{tmp_path / 'voting_out'}", random_seed: 42}}
inputs:
  embedding_npz: "{emb}"
  references_csv: "{refs}"
  query_a3m: "{query_a3m}"
umap: {{n_neighbors: 5, min_dist: 0.1, n_components: 2, metric: euclidean,
        umap1_range: null, umap2_range: null}}
binning: {{bin_size: 5.0, top_k: 3, min_records_per_bin: 1}}
structure_prediction: {{num_models: 1, num_seeds: 1, num_recycle: 1,
                        prediction_mode: monomer, rank: plddt, random_seed: 0}}
slurm: {{conda_env_path: /tmp, account: T, partition: P, time: '00:30:00',
         gpus_per_task: 1, cpus_per_task: 4, check_interval: 1}}
structure_analysis: {{config_json: "{sa}", metrics: [local]}}
plotting: {{formats: [png], metric_ranges: {{local: {{min: 0, max: 30, ticks: [0,15,30]}}}},
            colors: {{ref_a: '#C73E3A', ref_b: '#2E7AB8'}}, panels_per_row: 1}}
"""
    )
    return UmapVotingConfig.from_yaml(yml)


def test_workflow_runs_to_vote(smoke_cfg):
    mgr = UmapVotingManager(smoke_cfg)
    mgr.run(stop_after="vote")
    base = Path(smoke_cfg.general.base_dir)
    assert (base / "umap" / "umap_coords.csv").exists()
    assert (base / "umap" / "grid.json").exists()
    assert (base / "voting" / "voting_summary.csv").exists()
    summary = pd.read_csv(base / "voting" / "voting_summary.csv")
    assert set(summary["ref_label"]) == {"ref_a", "ref_b"}


def test_workflow_skips_completed_stages(smoke_cfg):
    mgr = UmapVotingManager(smoke_cfg)
    mgr.run(stop_after="project")
    coords_mtime = (
        Path(smoke_cfg.general.base_dir) / "umap" / "umap_coords.csv"
    ).stat().st_mtime
    UmapVotingManager(smoke_cfg).run(start_from="vote", stop_after="vote")
    assert (
        Path(smoke_cfg.general.base_dir) / "umap" / "umap_coords.csv"
    ).stat().st_mtime == coords_mtime
