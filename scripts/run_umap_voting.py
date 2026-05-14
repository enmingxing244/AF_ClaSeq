#!/usr/bin/env python3
"""CLI entry point for UMAP voting pipeline (project → vote → predict → scatter)."""

import sys
import argparse
import logging
from pathlib import Path

from af_claseq.umap_voting import UmapVotingConfig
from af_claseq.umap_voting.workflow import UmapVotingManager
from af_claseq.utils.logging_utils import setup_logger, get_logger


STAGES = UmapVotingManager.STAGES


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run UMAP voting pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Stages: {', '.join(STAGES)}",
    )
    parser.add_argument("config", help="Path to UMAP voting YAML config")
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate config and exit")
    parser.add_argument("--start-from", choices=STAGES,
                        help="Resume from this stage")
    parser.add_argument("--stop-after", choices=STAGES,
                        help="Stop after this stage")
    parser.add_argument("--refit-umap", action="store_true",
                        help="Force UMAP refit even if cached")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run through project+vote only (no SLURM)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        config = UmapVotingConfig.from_yaml(args.config)
    except Exception as e:
        print(f"Config error: {e}")
        return 1

    log_dir = Path(config.general.base_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logger("af_claseq", log_file=log_dir / "umap_voting.log",
                 level=logging.INFO, propagate=False, add_console_handler=True)
    logger = get_logger("run_umap_voting")

    if args.validate_only:
        logger.info("Config validation passed.")
        return 0

    stop_after = args.stop_after
    if args.dry_run:
        stop_after = "vote"

    logger.info(f"Protein: {config.general.protein_name}")

    try:
        manager = UmapVotingManager(config)
        manager.run(
            start_from=args.start_from,
            stop_after=stop_after,
            refit_umap=args.refit_umap,
        )
        logger.info("Pipeline complete.")
        return 0
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
