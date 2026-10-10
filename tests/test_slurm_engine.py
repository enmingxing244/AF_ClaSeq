"""Tests for prediction-engine selection in SlurmJobSubmitter (colabfold | openfold).

subprocess.run is mocked (no real sbatch) so we can assert on the generated --wrap payload.
"""
import os
from unittest.mock import MagicMock, patch

import pytest

from af_claseq.utils.slurm_utils import SlurmJobSubmitter
from af_claseq.utils import openfold_utils as ofu


def _completed(stdout="Submitted batch job 12345"):
    m = MagicMock()
    m.stdout = stdout
    m.returncode = 0
    return m


def _wrap_of(sbatch_cmd):
    return sbatch_cmd[sbatch_cmd.index("--wrap") + 1]


# ----------------------------------------------------------------- construction / validation
def test_default_engine_is_colabfold():
    s = SlurmJobSubmitter("/envs/cf", "A", num_models=1, num_seeds=1)
    assert s.prediction_engine == "colabfold"


def test_invalid_engine_raises():
    with pytest.raises(ValueError):
        SlurmJobSubmitter("/e", "A", prediction_engine="rosetta")


def test_invalid_openfold_config_raises():
    with pytest.raises(ValueError):
        SlurmJobSubmitter("/e", "A", prediction_engine="openfold", openfold_config="bogus")


def test_openfold_env_is_distinct_from_conda_env_path():
    s = SlurmJobSubmitter("/envs/cf", "A", prediction_engine="openfold")
    assert s.openfold_conda_env == ofu.DEFAULT_OPENFOLD_CONDA_ENV
    assert s.conda_env_path == "/envs/cf"  # the ColabFold env is left untouched


def test_openfold_paths_overridable():
    s = SlurmJobSubmitter(
        "/envs/cf", "A", prediction_engine="openfold",
        openfold_conda_env="/custom/of2", openfold_dir="/custom/of",
    )
    assert s.openfold_conda_env == "/custom/of2"
    assert s.openfold_dir == "/custom/of"


# ----------------------------------------------------------------- colabfold submit (unchanged)
def test_colabfold_submit_builds_colabfold_wrap(tmp_path):
    task = tmp_path / "t"
    task.mkdir()
    (task / "g.a3m").write_text(">q\nACDE\n")
    s = SlurmJobSubmitter("/envs/cf", "A", num_models=2, num_seeds=3)
    with patch("af_claseq.utils.slurm_utils.subprocess.run", return_value=_completed()) as run:
        jid = s.submit_job(str(task), "job0")
    assert jid == "12345"
    wrap = _wrap_of(run.call_args[0][0])
    assert "colabfold_batch" in wrap
    assert "conda activate /envs/cf" in wrap
    assert f"{task} {task}" in wrap
    assert "--num-models 2" in wrap and "--num-seeds 3" in wrap
    assert "conda init" not in wrap  # the removed bugfix stays removed


# ----------------------------------------------------------------- openfold submit
def test_openfold_submit_builds_openfold_wrap_and_prepares_input(tmp_path):
    task = tmp_path / "t"
    task.mkdir()
    (task / "group_1.a3m").write_text(">q\nACDEFGHIK\n>h\nACDEFGHIK\n")
    s = SlurmJobSubmitter("/envs/cf", "A", num_models=1, num_seeds=1, prediction_engine="openfold")
    with patch("af_claseq.utils.slurm_utils.subprocess.run", return_value=_completed()) as run:
        jid = s.submit_job(str(task), "job0")
    assert jid == "12345"
    wrap = _wrap_of(run.call_args[0][0])
    assert "run_pretrained_openfold.py" in wrap
    assert f"conda activate {ofu.DEFAULT_OPENFOLD_CONDA_ENV}" in wrap
    assert "set -eo pipefail" in wrap
    assert "--use_deepspeed_evoformer_attention" in wrap and "--precision bf16" in wrap
    assert "colabfold_batch" not in wrap
    # driver-side a3m -> OpenFold conversion happened:
    assert (task / "_openfold_work" / "fasta" / "group_1.fasta").exists()
    assert (task / "_openfold_work" / "alignments" / "group_1" / "bfd_uniclust_hits.a3m").exists()


def test_openfold_collect_renames_with_configured_seed(tmp_path):
    task = tmp_path / "t"
    task.mkdir()
    s = SlurmJobSubmitter(
        "/envs/cf", "A", num_models=1, num_seeds=1, random_seed=42, prediction_engine="openfold"
    )
    pred = task / "_openfold_work" / "output" / "predictions"
    pred.mkdir(parents=True)
    (pred / "group_1_model_3_ptm_unrelaxed.pdb").write_text("ATOM\n")

    s._collect_openfold(str(task))

    assert (task / "group_1_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_042.pdb").exists()
    assert "Done" in (task / "log.txt").read_text()
    assert not (task / "_openfold_work").exists()
