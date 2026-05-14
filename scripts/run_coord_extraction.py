#!/usr/bin/env python3
"""CLI entry point for parallel Kabsch-aligned coordinate extraction.

Run this as a high-CPU SLURM job before VAE training. The output
coords.npz is cached — as long as the superposition indices, target
indices, and input CSVs don't change, it won't re-extract.
"""

import sys
import argparse
import logging
from pathlib import Path

from af_claseq.umap_voting import VaeTrainConfig
from af_claseq.umap_voting.coords import extract_all_parallel
from af_claseq.utils.logging_utils import setup_logger, get_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract Kabsch-aligned Calpha coordinates (parallel, cached)"
    )
    parser.add_argument("config", help="Path to VAE training YAML config")
    parser.add_argument("--n-jobs", type=int, default=64,
                        help="Number of parallel workers (default: 64)")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if cache is valid")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        config = VaeTrainConfig.from_yaml(args.config)
    except Exception as e:
        print(f"Config error: {e}")
        return 1

    log_dir = Path(config.general.base_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logger("af_claseq", log_file=log_dir / "coord_extraction.log",
                 level=logging.INFO, propagate=False, add_console_handler=True)
    logger = get_logger("run_coord_extraction")

    if args.force:
        coords_path = config.get_vae_dir() / "coords.npz"
        if coords_path.exists():
            coords_path.unlink()
            logger.info("Removed existing coords.npz (--force)")

    logger.info(f"Protein: {config.general.protein_name}")
    logger.info(f"Workers: {args.n_jobs}")
    logger.info(f"Target: {config.structure_analysis.coord_target}")

    try:
        out = extract_all_parallel(config, n_jobs=args.n_jobs)
        logger.info(f"Done. Coords saved to {out}")
        return 0
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
