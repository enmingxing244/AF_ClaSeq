"""UmapVotingManager — orchestrates project → vote → predict → scatter."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from af_claseq.utils.logging_utils import get_logger
from af_claseq.utils.exceptions import WorkflowError
from af_claseq.utils.sequence_processing import get_query_sequence_from_a3m

from .config import UmapVotingConfig
from .predictor import Predictor
from .projector import Projector
from .scatter import ScatterBuilder
from .voter import Voter

logger = get_logger("umap_voting.workflow")


class UmapVotingManager:
    STAGES = ["project", "vote", "predict", "scatter"]

    def __init__(self, config: UmapVotingConfig):
        self.cfg = config
        logger.info(f"UmapVotingManager for {config.general.protein_name}")

    def run(
        self,
        start_from: Optional[str] = None,
        stop_after: Optional[str] = None,
        refit_umap: bool = False,
    ) -> None:
        stages = self.STAGES
        if start_from:
            if start_from not in stages:
                raise WorkflowError(f"Unknown stage: {start_from}")
            stages = stages[stages.index(start_from):]
        if stop_after:
            if stop_after not in stages:
                raise WorkflowError(f"Unknown stage: {stop_after}")
            stages = stages[: stages.index(stop_after) + 1]

        logger.info(f"Running stages: {stages}")

        for stage in stages:
            logger.info(f"{'=' * 60}")
            logger.info(f"STAGE: {stage.upper()}")
            logger.info(f"{'=' * 60}")

            if stage == "project":
                self._run_project(refit_umap)
            elif stage == "vote":
                self._run_vote()
            elif stage == "predict":
                self._run_predict()
            elif stage == "scatter":
                self._run_scatter()

        logger.info("All requested stages complete.")

    def _run_project(self, refit_umap: bool) -> None:
        projector = Projector(self.cfg)
        df = projector.fit(refit=refit_umap)
        df = projector.assign_bins(df)
        df = projector.compute_sampling_rmsds(df)
        projector.plot_umap_diagnostic(df)

    def _run_vote(self) -> None:
        umap_csv = self.cfg.get_umap_dir() / "umap_coords.csv"
        if not umap_csv.exists():
            raise WorkflowError("umap_coords.csv not found — run 'project' first")

        query_header, query_seq = None, None
        try:
            query_header, query_seq = get_query_sequence_from_a3m(
                self.cfg.inputs.query_a3m
            )
        except Exception as e:
            logger.warning(f"Could not read query A3M: {e}")

        voter = Voter(self.cfg)
        voter.vote(umap_csv, query_header, query_seq)
        voter.plot_voting_bins(umap_csv)

    def _run_predict(self) -> None:
        predictor = Predictor(self.cfg)
        predictor.run(dry_run=False)

    def _run_scatter(self) -> None:
        builder = ScatterBuilder(self.cfg)
        builder.build()
