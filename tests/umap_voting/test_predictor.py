from unittest.mock import MagicMock, patch

import pytest

from af_claseq.umap_voting.predictor import Predictor


def _slurm_kwargs():
    return dict(
        conda_env_path="/fake/env",
        account="A",
        partition="P",
        time="00:30:00",
        gpus_per_task=1,
        cpus_per_task=4,
        check_interval=1,
    )


def _pred_kwargs():
    return dict(
        num_models=1,
        num_seeds=1,
        num_recycle=1,
        prediction_mode="monomer",
        rank="plddt",
        random_seed=0,
    )


def test_predictor_submits_one_task_per_a3m(tmp_path):
    a3ms_dir = tmp_path / "a3ms"
    a3ms_dir.mkdir()
    for n in ("bin_00_00_ref_a.a3m", "bin_00_00_ref_b.a3m"):
        (a3ms_dir / n).write_text("#1\t1\n>q\nM\n")
    predictions_dir = tmp_path / "preds"

    with patch("af_claseq.umap_voting.predictor.SlurmJobSubmitter") as Mock:
        m = MagicMock()
        Mock.return_value = m
        m.submit_custom_job.side_effect = ["111", "222"]
        from af_claseq.utils.slurm_utils import JobState

        m.monitor_jobs.return_value = {
            "111": JobState.COMPLETED,
            "222": JobState.COMPLETED,
        }
        pred = Predictor(
            a3ms_dir=a3ms_dir,
            predictions_dir=predictions_dir,
            structure_prediction=_pred_kwargs(),
            slurm=_slurm_kwargs(),
        )
        manifest = pred.run()

    assert len(manifest) == 2
    assert set(manifest["state"]) == {"COMPLETED"}
    assert m.submit_custom_job.call_count == 2


def test_predictor_records_failed_state(tmp_path):
    a3ms_dir = tmp_path / "a3ms"
    a3ms_dir.mkdir()
    (a3ms_dir / "bin_00_00_x.a3m").write_text("#1\t1\n>q\nM\n")

    with patch("af_claseq.umap_voting.predictor.SlurmJobSubmitter") as Mock:
        m = MagicMock()
        Mock.return_value = m
        m.submit_custom_job.return_value = "999"
        from af_claseq.utils.slurm_utils import JobState

        m.monitor_jobs.return_value = {"999": JobState.FAILED}
        pred = Predictor(
            a3ms_dir=a3ms_dir,
            predictions_dir=tmp_path / "preds",
            structure_prediction=_pred_kwargs(),
            slurm=_slurm_kwargs(),
        )
        manifest = pred.run()

    assert manifest.iloc[0]["state"] == "FAILED"
