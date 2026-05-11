"""Tests for af_claseq.utils.config_validator — validates Tier 2 preflight checks."""

import tempfile
import pytest
from pathlib import Path

from af_claseq.utils.config_validator import validate_config


class TestValidateConfig:
    def test_missing_file(self):
        errors = validate_config("/nonexistent/config.yaml")
        assert any("does not exist" in e or "not found" in e.lower() for e in errors)

    def test_invalid_yaml(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("{{invalid yaml: [")
        errors = validate_config(str(bad))
        assert any("parse" in e.lower() or "yaml" in e.lower() for e in errors)

    def test_missing_general_section(self, tmp_path):
        cfg = tmp_path / "no_general.yaml"
        cfg.write_text("slurm:\n  account: test\n")
        errors = validate_config(str(cfg))
        assert any("general" in e.lower() or "input" in e.lower() for e in errors)

    def test_worktree_path_flagged(self, tmp_path):
        cfg = tmp_path / "worktree.yaml"
        cfg.write_text(
            "general:\n"
            "  source_a3m: /some/path/.worktrees/feat-branch/data/file.a3m\n"
            "  base_dir: /tmp/test\n"
        )
        errors = validate_config(str(cfg))
        assert any(".worktrees" in e for e in errors)

    def test_missing_source_a3m_flagged(self, tmp_path):
        cfg = tmp_path / "missing_a3m.yaml"
        cfg.write_text(
            "general:\n"
            "  source_a3m: /definitely/nonexistent/file.a3m\n"
            "  base_dir: /tmp/test\n"
        )
        errors = validate_config(str(cfg))
        assert any("nonexistent" in e or "not found" in e.lower() or "does not exist" in e.lower() for e in errors)

    def test_valid_minimal_config(self, tmp_path):
        a3m = tmp_path / "test.a3m"
        a3m.write_text(">query\nACDEF\n>seq1\nACDEL\n")
        cfg = tmp_path / "valid.yaml"
        cfg.write_text(
            f"general:\n"
            f"  source_a3m: {a3m}\n"
            f"  base_dir: {tmp_path}\n"
        )
        errors = validate_config(str(cfg))
        path_errors = [e for e in errors if "not found" in e.lower() or "does not exist" in e.lower()]
        assert len(path_errors) == 0, f"Valid config should have no path errors: {path_errors}"
