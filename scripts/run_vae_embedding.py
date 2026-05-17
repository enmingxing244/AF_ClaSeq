#!/usr/bin/env python3
"""VAE training + structure-to-embedding encoder.

Usage:
  python scripts/run_vae_embedding.py CONFIG.yaml
  python scripts/run_vae_embedding.py --validate-only CONFIG.yaml
"""
import argparse
import sys
import traceback

from af_claseq.umap_voting import VaeTrainConfig
from af_claseq.umap_voting.vae.train import VaeTrainer
from af_claseq.utils.logging_utils import setup_logger


def main() -> int:
    p = argparse.ArgumentParser(description="VAE training + embedding encoder")
    p.add_argument("config_file", type=str)
    p.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate config and exit without training",
    )
    p.add_argument("--device", default=None, help="Override device (cpu|cuda)")
    args = p.parse_args()

    try:
        cfg = VaeTrainConfig.from_yaml(args.config_file)
    except Exception as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2

    if args.device:
        cfg.general.device = args.device

    if args.validate_only:
        print(
            f"config OK: protein={cfg.general.protein_name} "
            f"epochs={cfg.vae.training.epochs}"
        )
        return 0

    setup_logger(name="af_claseq", log_file=None)
    try:
        VaeTrainer(cfg).train()
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
