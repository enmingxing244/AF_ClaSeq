#!/usr/bin/env python3
"""UMAP + Option F voting + ColabFold + RMSD scatter pipeline.

Usage:
  python scripts/run_umap_voting.py CONFIG.yaml
  python scripts/run_umap_voting.py --dry-run CONFIG.yaml
  python scripts/run_umap_voting.py --start-from vote --stop-after vote CONFIG.yaml
"""
import argparse
import sys
import traceback

from af_claseq.umap_voting import UmapVotingConfig, UmapVotingManager
from af_claseq.utils.logging_utils import setup_logger

STAGES = ["project", "vote", "predict", "scatter"]


def main() -> int:
    p = argparse.ArgumentParser(description="UMAP voting pipeline")
    p.add_argument("config_file", type=str)
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--start-from", choices=STAGES, default=None)
    p.add_argument("--stop-after", choices=STAGES, default=None)
    p.add_argument(
        "--refit-umap",
        action="store_true",
        help="Refit UMAP even if a persisted model exists",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run through project + vote; skip predict + scatter",
    )
    args = p.parse_args()

    try:
        cfg = UmapVotingConfig.from_yaml(args.config_file)
    except Exception as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2

    if args.validate_only:
        print(
            f"config OK: protein={cfg.general.protein_name} "
            f"bin_size={cfg.binning.bin_size}"
        )
        return 0

    setup_logger(name="af_claseq", log_file=None)

    stop_after = "vote" if args.dry_run else args.stop_after
    try:
        UmapVotingManager(cfg).run(
            start_from=args.start_from,
            stop_after=stop_after,
            refit_umap=args.refit_umap,
        )
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
