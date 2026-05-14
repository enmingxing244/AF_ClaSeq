"""Tests for CLI entry points: help flags and validation-only mode."""

import subprocess
import sys
import pytest


class TestVaeEmbeddingCli:
    def test_help_flag(self):
        result = subprocess.run(
            [sys.executable, "scripts/run_vae_embedding.py", "--help"],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        assert "VAE" in result.stdout or "vae" in result.stdout.lower()

    def test_missing_config(self):
        result = subprocess.run(
            [sys.executable, "scripts/run_vae_embedding.py", "/nonexistent.yaml"],
            capture_output=True, text=True
        )
        assert result.returncode != 0


class TestUmapVotingCli:
    def test_help_flag(self):
        result = subprocess.run(
            [sys.executable, "scripts/run_umap_voting.py", "--help"],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        assert "umap" in result.stdout.lower() or "UMAP" in result.stdout

    def test_missing_config(self):
        result = subprocess.run(
            [sys.executable, "scripts/run_umap_voting.py", "/nonexistent.yaml"],
            capture_output=True, text=True
        )
        assert result.returncode != 0
