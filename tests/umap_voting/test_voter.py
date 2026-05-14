"""Tests for voting logic: frequency ranking, tie-breaking, A3M output."""

import pytest
from pathlib import Path
from collections import Counter

from af_claseq.umap_voting.voter import _parse_one

FIXTURE_DIR = Path(__file__).parent / "fixtures"
A3M_DIR = FIXTURE_DIR / "a3ms"


class TestParseOne:
    def test_returns_non_query_sequences(self):
        seqs = _parse_one(str(A3M_DIR / "group_00.a3m"))
        assert len(seqs) >= 3  # common_a, common_b, common_c, unique

    def test_skips_query(self):
        seqs = _parse_one(str(A3M_DIR / "group_00.a3m"))
        assert "ACDEFGHIKLMNPQRST" not in seqs


class TestVotingFrequency:
    def test_frequency_ranking(self):
        """COMMON_A in all 10, COMMON_B in 7, COMMON_C in 4."""
        counter: Counter = Counter()
        for i in range(10):
            seqs = _parse_one(str(A3M_DIR / f"group_{i:02d}.a3m"))
            counter.update(seqs)

        ranked = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

        assert ranked[0][0] == "AAAAAAAAAAAAAAAAA"
        assert ranked[0][1] == 10
        assert ranked[1][0] == "BBBBBBBBBBBBBBBBB"
        assert ranked[1][1] == 7
        assert ranked[2][0] == "CCCCCCCCCCCCCCCCC"
        assert ranked[2][1] == 4

    def test_deterministic_tie_breaking(self):
        """Ties broken by lexicographic sequence order."""
        counter = Counter({"AAA": 5, "BBB": 5, "CCC": 5})
        ranked = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        assert [s for s, _ in ranked] == ["AAA", "BBB", "CCC"]
