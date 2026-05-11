"""Tests for af_claseq.utils.sequence_processing — validates Tier 0 bug fixes."""

import os
import tempfile
import pytest
from pathlib import Path

FIXTURE_DIR = Path(__file__).parent / "fixtures"
SAMPLE_A3M = str(FIXTURE_DIR / "sample.a3m")

from af_claseq.utils.sequence_processing import (
    read_a3m_to_dict,
    get_query_sequence_from_a3m,
    write_a3m,
    filter_a3m_by_coverage,
    count_sequences_in_a3m,
)


class TestReadA3mToDict:
    def test_keys_have_gt_prefix(self):
        seqs = read_a3m_to_dict(SAMPLE_A3M)
        for key in seqs:
            assert key.startswith(">"), f"Key missing > prefix: {key}"

    def test_lowercase_stripped(self):
        seqs = read_a3m_to_dict(SAMPLE_A3M)
        for header, seq in seqs.items():
            assert seq == seq.upper() or set(seq) <= set("ACDEFGHIKLMNPQRSTVWY-"), (
                f"Lowercase found in {header}: {seq}"
            )

    def test_header_truncated_at_whitespace(self):
        seqs = read_a3m_to_dict(SAMPLE_A3M)
        assert ">seq_004" in seqs, "Header with tab annotation should be truncated"
        assert ">seq_004\textra_annotation" not in seqs

    def test_sequence_count(self):
        seqs = read_a3m_to_dict(SAMPLE_A3M)
        assert len(seqs) == 5

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            read_a3m_to_dict("/nonexistent/path.a3m")


class TestGetQuerySequence:
    def test_returns_first_sequence(self):
        header, seq = get_query_sequence_from_a3m(SAMPLE_A3M)
        assert header == "query_protein"

    def test_strips_lowercase(self):
        header, seq = get_query_sequence_from_a3m(SAMPLE_A3M)
        assert seq == seq.upper() or all(c in "ACDEFGHIKLMNPQRSTVWY-" for c in seq), (
            f"Lowercase in query: {seq}"
        )

    def test_matches_read_a3m_to_dict(self):
        header, seq = get_query_sequence_from_a3m(SAMPLE_A3M)
        seqs = read_a3m_to_dict(SAMPLE_A3M)
        first_key = list(seqs.keys())[0]
        first_val = seqs[first_key]
        assert seq == first_val, "get_query_sequence_from_a3m and read_a3m_to_dict should agree"


class TestWriteA3m:
    def test_prepend_query_appears_once(self, tmp_path):
        seqs = read_a3m_to_dict(SAMPLE_A3M)
        items = list(seqs.items())
        subset = dict(items[1:3])

        out = str(tmp_path / "out.a3m")
        write_a3m(subset, out, source_a3m=SAMPLE_A3M, prepend_query=True)

        result = read_a3m_to_dict(out)
        result_items = list(result.items())

        query_header, query_seq = get_query_sequence_from_a3m(SAMPLE_A3M)
        query_count = sum(1 for h, s in result_items if s == query_seq)
        assert query_count == 1, f"Query should appear exactly once, found {query_count}"

    def test_plain_write_roundtrip(self, tmp_path):
        seqs = read_a3m_to_dict(SAMPLE_A3M)
        out = str(tmp_path / "out.a3m")
        write_a3m(seqs, out)

        result = read_a3m_to_dict(out)
        assert len(result) == len(seqs)
        for k in seqs:
            assert k in result
            assert result[k] == seqs[k]


class TestFilterA3mByCoverage:
    def test_query_always_kept(self):
        seqs = read_a3m_to_dict(SAMPLE_A3M)
        filtered = filter_a3m_by_coverage(seqs, coverage_threshold=0.99)
        first_key = list(seqs.keys())[0]
        assert first_key in filtered, "Query should always be kept"

    def test_high_threshold_removes_low_coverage(self):
        seqs = read_a3m_to_dict(SAMPLE_A3M)
        filtered = filter_a3m_by_coverage(seqs, coverage_threshold=0.99)
        assert len(filtered) <= len(seqs)


class TestCountSequences:
    def test_correct_count(self):
        assert count_sequences_in_a3m(SAMPLE_A3M) == 5

    def test_missing_file_returns_zero(self):
        assert count_sequences_in_a3m("/nonexistent.a3m") == 0
