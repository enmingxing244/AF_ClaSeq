"""Tests for the metric-threshold-based recompile bin selection
(``MetricBinConfig.bins_within_threshold``) added on feat-recompile-metric-threshold.

The selection uses the "fully-below"/"fully-above" rule: a bin qualifies only when its
ENTIRE value range lies on the reference side of the cutoff. Returned indices must match
the voting stage's convention (focused -> 1-based + sentinels; non-focused -> 0-based).
"""

import pytest

from af_claseq.m_fold_sampling_voting.config import (
    MetricBinConfig,
    RecompilePredictConfig,
)

# AdK-style RMSD binning: bin_width 0.2, range 0..8 -> 40 bins, edge k = 0.2*k.
RMSD = dict(bin_width=0.2, min=0.0, max=8.0)
# TM-score-style binning: bin_width 0.02, range 0..1 -> 50 bins, edge k = 0.02*k.
TM = dict(bin_width=0.02, min=0.0, max=1.0)


@pytest.mark.parametrize("cfg,threshold,direction,focused,expected", [
    # RMSD < 2.0 -> bins covering [0, 2.0): non-focused 0..9, focused 1..10 (+ sentinel 0).
    (RMSD, 2.0, "below", False, list(range(0, 10))),
    (RMSD, 2.0, "below", True,  [0] + list(range(1, 11))),
    # TM > 0.9 -> bins covering [0.90, 1.00): non-focused 45..49, focused 46..50 (+ sentinel 51).
    (TM,   0.9, "above", False, list(range(45, 50))),
    (TM,   0.9, "above", True,  list(range(46, 51)) + [51]),
])
def test_bins_exact_edge(cfg, threshold, direction, focused, expected):
    """Cutoff lands exactly on a bin edge — every option agrees on the boundary."""
    assert MetricBinConfig(**cfg).bins_within_threshold(threshold, direction, focused) == expected


class TestStraddleAndEmpty:
    def test_straddling_bin_excluded_below(self):
        # RMSD < 1.95: the bin [1.8, 2.0) straddles the cutoff and must be excluded.
        bins = MetricBinConfig(**RMSD).bins_within_threshold(1.95, "below", focused=False)
        assert bins == list(range(0, 9))  # bins 0..8 cover [0, 1.8)
        assert 9 not in bins

    def test_straddling_bin_excluded_above(self):
        # TM > 0.91: bin [0.90, 0.92) straddles -> excluded; first fully-above is [0.92, 0.94).
        bins = MetricBinConfig(**TM).bins_within_threshold(0.91, "above", focused=False)
        assert min(bins) == 46  # left edge 0.92
        assert 45 not in bins

    def test_empty_when_cutoff_inside_first_bin_below(self):
        # RMSD < 0.1 is inside bin 0 [0, 0.2): no bin is wholly below -> empty (non-focused).
        bins = MetricBinConfig(**RMSD).bins_within_threshold(0.1, "below", focused=False)
        assert bins == []

    def test_empty_when_cutoff_inside_last_bin_above(self):
        # TM > 0.99 is inside the last bin [0.98, 1.00): no bin wholly above -> empty (non-focused).
        bins = MetricBinConfig(**TM).bins_within_threshold(0.99, "above", focused=False)
        assert bins == []

    def test_focused_only_sentinel_when_cutoff_inside_first_bin(self):
        # Non-focused gives [] (no in-range bin qualifies); focused gives [0] (below-min sentinel).
        bins = MetricBinConfig(**RMSD).bins_within_threshold(0.1, "below", focused=True)
        assert bins == [0]


class TestConventionRelationship:
    def test_focused_offset_by_one_vs_nonfocused(self):
        # The two conventions differ by exactly +1 on the in-range bins.
        nf = MetricBinConfig(**RMSD).bins_within_threshold(2.0, "below", focused=False)
        f = MetricBinConfig(**RMSD).bins_within_threshold(2.0, "below", focused=True)
        f_inrange = [b for b in f if b != 0]
        assert f_inrange == [b + 1 for b in nf]


class TestBinsWithinThresholdErrors:
    def test_requires_bin_width(self):
        with pytest.raises(ValueError):
            MetricBinConfig().bins_within_threshold(2.0, "below", focused=False)

    def test_bad_direction(self):
        with pytest.raises(ValueError):
            MetricBinConfig(**RMSD).bins_within_threshold(2.0, "sideways", focused=False)


class TestRecompileConfigBackCompat:
    def test_new_threshold_fields_default_none(self):
        # Threshold fields are additive and optional -> existing configs are unchanged.
        cfg = RecompilePredictConfig(bin_numbers_1=[26], metric_name_1="m1")
        assert cfg.threshold_1 is None
        assert cfg.threshold_direction_1 is None
        assert cfg.bin_numbers_1 == [26]

    def test_threshold_fields_accepted(self):
        cfg = RecompilePredictConfig(
            metric_name_1="m1", threshold_1=2.0, threshold_direction_1="below"
        )
        assert cfg.threshold_1 == 2.0
        assert cfg.threshold_direction_1 == "below"

    def test_invalid_direction_rejected_at_construction(self):
        # __post_init__ validates threshold_direction_* at config-load time.
        with pytest.raises(ValueError):
            RecompilePredictConfig(threshold_direction_1="bellow")
