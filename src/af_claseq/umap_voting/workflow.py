"""End-to-end orchestrator: project -> vote -> predict -> scatter."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from af_claseq.umap_voting.config import UmapVotingConfig
from af_claseq.umap_voting.predictor import Predictor
from af_claseq.umap_voting.projector import Projector
from af_claseq.umap_voting.scatter import ScatterBuilder
from af_claseq.umap_voting.voter import Voter
from af_claseq.utils.exceptions import WorkflowError
from af_claseq.utils.logging_utils import get_logger
from af_claseq.utils.sequence_processing import get_query_sequence_from_a3m

logger = get_logger("umap_voting.workflow")

STAGES = ["project", "vote", "predict", "scatter"]


@dataclass
class UmapVotingManager:
    cfg: UmapVotingConfig

    def _base(self) -> Path:
        return Path(self.cfg.general.base_dir)

    def _stage_done(self, stage: str) -> bool:
        b = self._base()
        return {
            "project": (
                (b / "umap" / "umap_coords.csv").exists()
                and (b / "umap" / "grid.json").exists()
            ),
            "vote": (b / "voting" / "voting_summary.csv").exists(),
            "predict": (b / "predictions" / "job_manifest.csv").exists(),
            "scatter": (b / "scatter" / "per_pred.csv").exists(),
        }[stage]

    def run(
        self,
        start_from: Optional[str] = None,
        stop_after: Optional[str] = None,
        refit_umap: bool = False,
    ) -> None:
        first_idx = STAGES.index(start_from) if start_from else 0
        last_idx = STAGES.index(stop_after) if stop_after else len(STAGES) - 1
        for idx in range(first_idx, last_idx + 1):
            stage = STAGES[idx]
            if idx > first_idx and self._stage_done(stage):
                logger.info(f"skip {stage}: outputs already present")
                continue
            logger.info(f"=== stage {stage} ===")
            try:
                getattr(self, f"_run_{stage}")(refit_umap=refit_umap)
            except WorkflowError:
                raise
            except Exception as e:
                raise WorkflowError(f"stage '{stage}' failed: {e}") from e

    def _run_project(self, refit_umap: bool = False, **_kw: object) -> None:
        u = self.cfg.umap
        proj = Projector(
            embedding_npz=self.cfg.inputs.embedding_npz,
            output_dir=self._base() / "umap",
            n_neighbors=u.n_neighbors,
            min_dist=u.min_dist,
            n_components=u.n_components,
            metric=u.metric,
            random_state=self.cfg.general.random_seed,
        )
        proj.fit(refit=refit_umap)
        proj.assign_bins(
            bin_size=self.cfg.binning.bin_size,
            umap1_range=u.umap1_range,
            umap2_range=u.umap2_range,
        )

    def _query_info(self) -> tuple[str, str]:
        """Get query header and sequence from the configured query A3M."""
        query_a3m = self.cfg.inputs.query_a3m
        if query_a3m and Path(query_a3m).exists():
            header, seq = get_query_sequence_from_a3m(query_a3m)
            return header, seq
        return "", ""

    def _run_vote(self, **_kw: object) -> None:
        header, seq = self._query_info()
        v = Voter(
            umap_coords_csv=self._base() / "umap" / "umap_coords.csv",
            references_csv=self.cfg.inputs.references_csv,
            output_dir=self._base() / "voting",
            top_k=self.cfg.binning.top_k,
            min_records_per_bin=self.cfg.binning.min_records_per_bin,
            query_header=header,
            query_seq=seq,
        )
        v.vote()

    def _run_predict(self, **_kw: object) -> None:
        sp = self.cfg.structure_prediction
        sl = self.cfg.slurm
        pred = Predictor(
            a3ms_dir=self._base() / "voting" / "a3ms",
            predictions_dir=self._base() / "predictions",
            structure_prediction=dict(
                num_models=sp.num_models,
                num_seeds=sp.num_seeds,
                num_recycle=sp.num_recycle,
                prediction_mode=sp.prediction_mode,
                rank=sp.rank,
                random_seed=sp.random_seed,
            ),
            slurm=dict(
                conda_env_path=sl.conda_env_path,
                account=sl.account,
                partition=sl.partition,
                time=sl.time,
                gpus_per_task=sl.gpus_per_task,
                cpus_per_task=sl.cpus_per_task,
                check_interval=sl.check_interval,
            ),
        )
        pred.run()

    def _run_scatter(self, **_kw: object) -> None:
        pl = self.cfg.plotting
        sb = ScatterBuilder(
            predictions_dir=self._base() / "predictions",
            references_csv=self.cfg.inputs.references_csv,
            structure_analysis_json=self.cfg.structure_analysis.config_json,
            output_dir=self._base() / "scatter",
            metrics=self.cfg.structure_analysis.metrics or ["local"],
            formats=pl.formats,
            metric_ranges=pl.metric_ranges,
            colors=pl.colors,
            panels_per_row=pl.panels_per_row,
        )
        sb.build()
