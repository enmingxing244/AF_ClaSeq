"""ColabFold submission wrapper using utils/slurm_utils.SlurmJobSubmitter."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from af_claseq.utils.logging_utils import get_logger
from af_claseq.utils.slurm_utils import SlurmJobSubmitter

logger = get_logger("umap_voting.predictor")


@dataclass
class Predictor:
    a3ms_dir: str | Path
    predictions_dir: str | Path
    structure_prediction: Dict[str, Any]
    slurm: Dict[str, Any]

    def __post_init__(self) -> None:
        self.a3ms_dir = Path(self.a3ms_dir)
        self.predictions_dir = Path(self.predictions_dir)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

    def _build_colabfold_cmd(self, a3m: Path, out_dir: Path) -> str:
        sp = self.structure_prediction
        cf = Path(self.slurm["conda_env_path"]) / "bin" / "colabfold_batch"
        flags = [
            f"--num-models {sp['num_models']}",
            f"--num-seeds {sp['num_seeds']}",
            f"--num-recycle {sp['num_recycle']}",
            "--model-type alphafold2_ptm",
            f"--rank {sp['rank']}",
            f"--random-seed {sp['random_seed']}",
        ]
        return f'"{cf}" {" ".join(flags)} "{a3m}" "{out_dir}"'

    def run(self, dry_run: bool = False) -> pd.DataFrame:
        submitter = SlurmJobSubmitter(
            conda_env_path=self.slurm["conda_env_path"],
            slurm_account=self.slurm["account"],
            slurm_partition=self.slurm["partition"],
            slurm_time=self.slurm["time"],
            slurm_gpus_per_task=self.slurm["gpus_per_task"],
            slurm_cpus_per_task=self.slurm["cpus_per_task"],
            slurm_nodes=1,
            slurm_tasks=1,
            check_interval=self.slurm["check_interval"],
            job_name_prefix="umap_voting",
            num_recycle=self.structure_prediction["num_recycle"],
            num_models=self.structure_prediction["num_models"],
            num_seeds=self.structure_prediction["num_seeds"],
        )

        a3ms = sorted(self.a3ms_dir.glob("bin_*.a3m"))
        if not a3ms:
            raise RuntimeError(f"no A3Ms under {self.a3ms_dir}")

        manifest_rows: List[Dict[str, Any]] = []
        job_ids: Dict[str, Dict[str, Any]] = {}

        for a3m in a3ms:
            ref_slug = a3m.stem
            out_dir = self.predictions_dir / ref_slug
            out_dir.mkdir(parents=True, exist_ok=True)
            cmd = self._build_colabfold_cmd(a3m, out_dir)

            if dry_run:
                manifest_rows.append(
                    dict(
                        ref_slug=ref_slug,
                        a3m=str(a3m),
                        out_dir=str(out_dir),
                        slurm_job_id="DRY_RUN",
                        state="DRY_RUN",
                    )
                )
                continue

            jid = submitter.submit_custom_job(
                job_name=ref_slug,
                command=cmd,
                task_dir=str(out_dir),
                gres=f"gpu:{self.slurm['gpus_per_task']}",
            )
            if jid is None:
                logger.error(f"failed to submit ColabFold for {a3m.name}")
                manifest_rows.append(
                    dict(
                        ref_slug=ref_slug,
                        a3m=str(a3m),
                        out_dir=str(out_dir),
                        slurm_job_id="",
                        state="SUBMIT_FAILED",
                    )
                )
                continue
            job_ids[jid] = dict(
                ref_slug=ref_slug, a3m=str(a3m), out_dir=str(out_dir)
            )
            logger.info(f"submitted {ref_slug} as job {jid}")

        if job_ids:
            final = submitter.monitor_jobs(list(job_ids.keys()))
            for jid, state in final.items():
                meta = job_ids[jid]
                manifest_rows.append(
                    dict(slurm_job_id=jid, state=str(state.value), **meta)
                )

        manifest = pd.DataFrame(manifest_rows)
        manifest.to_csv(
            self.predictions_dir / "job_manifest.csv", index=False
        )
        return manifest
