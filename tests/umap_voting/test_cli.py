import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_vae_cli_help():
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts/run_vae_embedding.py"), "--help"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0
    assert "--validate-only" in r.stdout


def test_vae_cli_validate_only_bad_config(tmp_path):
    cfg = tmp_path / "vae.yaml"
    cfg.write_text("general: {}")
    r = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/run_vae_embedding.py"),
            "--validate-only",
            str(cfg),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0
    assert "missing required" in (r.stdout + r.stderr).lower()


def test_voting_cli_help():
    r = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/run_umap_voting.py"),
            "--help",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0
    assert "--start-from" in r.stdout
    assert "--stop-after" in r.stdout
    assert "--refit-umap" in r.stdout


def test_voting_cli_validate_only_bad_config(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("general: {}")
    r = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/run_umap_voting.py"),
            "--validate-only",
            str(bad),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0
