"""Tests for scatter RMSD computation and plot generation."""

import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from af_claseq.umap_voting.coords import _parse_index_spec


class TestParseIndexSpec:
    def test_single_dict(self):
        assert _parse_index_spec({"start": 1, "end": 5}) == [1, 2, 3, 4, 5]

    def test_list_of_dicts(self):
        result = _parse_index_spec([{"start": 1, "end": 3}, {"start": 7, "end": 9}])
        assert result == [1, 2, 3, 7, 8, 9]

    def test_bare_int(self):
        assert _parse_index_spec(5) == [5]

    def test_discontinuous_dedup(self):
        result = _parse_index_spec([{"start": 1, "end": 5}, {"start": 3, "end": 7}])
        assert result == [1, 2, 3, 4, 5, 6, 7]


class TestLoadRmsdSpecs:
    def test_loads_from_json(self, tmp_path):
        from af_claseq.umap_voting.scatter import _load_rmsd_specs
        sa_json = tmp_path / "sa.json"
        sa_json.write_text(json.dumps({
            "basics": {"full_index": {"start": 1, "end": 10}},
            "filter_criteria": [{
                "name": "ref_a_rmsd",
                "superposition_indices": {"start": 1, "end": 10},
                "rmsd_indices": {"start": 3, "end": 7},
                "ref_pdb": "/fake/ref_a.pdb",
            }],
        }))
        specs = _load_rmsd_specs(sa_json)
        assert len(specs) == 1
        assert specs[0]["name"] == "ref_a_rmsd"
        assert specs[0]["rmsd_indices"] == [3, 4, 5, 6, 7]


class TestScatterBuilderSummary:
    def test_summary_from_per_pred(self):
        from af_claseq.umap_voting.scatter import ScatterBuilder

        per_pred = pd.DataFrame({
            "ref_bin_label": ["bin_00_00_rA"] * 5 + ["bin_01_01_rB"] * 5,
            "source_ref": ["rA"] * 5 + ["rB"] * 5,
            "pred_path": [f"p{i}.pdb" for i in range(10)],
            "local_rmsd_rA": [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 4.5, 5.0, 5.5, 6.0],
            "local_rmsd_rB": [5.0, 4.5, 4.0, 3.5, 3.0, 1.0, 1.5, 2.0, 2.5, 3.0],
        })
        refs = pd.DataFrame({
            "ref_pdb": ["/a.pdb", "/b.pdb"],
            "ref_label": ["rA", "rB"],
            "ref_chain": ["A", "A"],
        })

        # Call _build_summary directly (it's a pure function on DataFrames)
        builder = ScatterBuilder.__new__(ScatterBuilder)
        summary = builder._build_summary(per_pred, refs)
        assert len(summary) == 2
        assert "ref_label" in summary.columns
