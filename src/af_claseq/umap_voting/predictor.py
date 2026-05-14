"""ColabFold structure prediction via SLURM."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from af_claseq.utils.logging_utils import get_logger
from af_claseq.utils.slurm_utils import SlurmJobSubmitter

from .config import UmapVotingConfig

logger = get_logger("umap_voting.predictor")


class Predictor:
    def __init__(self, config: UmapVotingConfig):
        self.cfg = config
        self.pred_dir = config.get_predictions_dir()
        self.pred_dir.mkdir(parents=True, exist_ok=True)

    def _build_colabfold_cmd(self, a3m_path: str, output_dir: str) -> str:
        sp = self.cfg.structure_prediction
        if sp.prediction_mode == "homodimer":
            model_type = "alphafold2_multimer_v3"
        else:
            model_type = "alphafold2_ptm"

        return (
            f"colabfold_batch {a3m_path} {output_dir} "
            f"--model-type {model_type} "
            f"--num-models {sp.num_models} "
            f"--num-seeds {sp.num_seeds} "
            f"--num-recycle {sp.num_recycle} "
            f"--rank {sp.rank} "
            f"--random-seed {sp.random_seed}"
        )

    def run(self, dry_run: bool = False) -> pd.DataFrame:
        """Submit ColabFold jobs for all voted A3Ms."""
        a3m_dir = self.cfg.get_voting_dir() / "a3ms"
        a3ms = sorted(a3m_dir.glob("bin_*.a3m"))
        if not a3ms:
            logger.warning("No voted A3M files found")
            return pd.DataFrame()

        slurm_cfg = self.cfg.slurm
        if slurm_cfg is None:
            raise RuntimeError("SLURM config required for predictions")

        submitter = SlurmJobSubmitter(
            conda_env_path=slurm_cfg.conda_env_path,
            slurm_account=slurm_cfg.account,
            slurm_partition=slurm_cfg.partition,
            slurm_time=slurm_cfg.time,
            slurm_cpus_per_task=slurm_cfg.cpus_per_task,
            num_models=self.cfg.structure_prediction.num_models,
            num_seeds=self.cfg.structure_prediction.num_seeds,
            num_recycle=self.cfg.structure_prediction.num_recycle,
            job_name_prefix="umap_vote",
        )

        rows = []
        for a3m in a3ms:
            job_name = a3m.stem
            out_dir = self.pred_dir / job_name
            out_dir.mkdir(parents=True, exist_ok=True)

            cmd = self._build_colabfold_cmd(str(a3m), str(out_dir))
            logger.info(f"Job {job_name}: {cmd}")

            if dry_run:
                rows.append({
                    "job_name": job_name,
                    "a3m_path": str(a3m),
                    "output_dir": str(out_dir),
                    "state": "dry_run",
                    "slurm_job_id": "",
                })
                continue

            try:
                job_id = submitter.submit_job(
                    task_dir=str(out_dir),
                    job_id=job_name,
                )
                rows.append({
                    "job_name": job_name,
                    "a3m_path": str(a3m),
                    "output_dir": str(out_dir),
                    "state": "submitted",
                    "slurm_job_id": str(job_id) if job_id else "",
                })
            except Exception as e:
                logger.error(f"Failed to submit {job_name}: {e}")
                rows.append({
                    "job_name": job_name,
                    "a3m_path": str(a3m),
                    "output_dir": str(out_dir),
                    "state": "failed",
                    "slurm_job_id": "",
                })

        manifest = pd.DataFrame(rows)
        manifest.to_csv(self.pred_dir / "job_manifest.csv", index=False)
        logger.info(f"Job manifest: {len(manifest)} jobs ({manifest['state'].value_counts().to_dict()})")

        if not dry_run:
            job_ids = [r["slurm_job_id"] for r in rows if r["slurm_job_id"]]
            if job_ids:
                logger.info(f"Monitoring {len(job_ids)} SLURM jobs...")
                submitter.monitor_jobs(
                    job_ids, check_interval=slurm_cfg.check_interval
                )

        return manifest
