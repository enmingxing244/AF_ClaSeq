"""Smoke test: verify UmapVotingManager stage dispatch works."""

import pytest
from af_claseq.umap_voting.workflow import UmapVotingManager
from af_claseq.utils.exceptions import WorkflowError


class TestStageValidation:
    def test_invalid_start_from(self):
        manager = UmapVotingManager.__new__(UmapVotingManager)
        manager.cfg = None
        with pytest.raises(WorkflowError, match="Unknown stage"):
            manager.run(start_from="invalid")

    def test_invalid_stop_after(self):
        manager = UmapVotingManager.__new__(UmapVotingManager)
        manager.cfg = None
        with pytest.raises(WorkflowError, match="Unknown stage"):
            manager.run(stop_after="invalid")

    def test_stages_constant(self):
        assert UmapVotingManager.STAGES == ["project", "vote", "predict", "scatter"]
