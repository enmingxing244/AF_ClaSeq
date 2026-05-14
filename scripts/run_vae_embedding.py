#!/usr/bin/env python3
"""CLI entry point for VAE embedding training."""

import sys
import argparse
import logging
from pathlib import Path

from af_claseq.umap_voting import VaeTrainConfig
from af_claseq.umap_voting.vae.train import VaeTrainer
from af_claseq.utils.logging_utils import setup_logger, get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train ProteinVAE and save embedding")
    parser.add_argument("config", help="Path to VAE training YAML config")
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate config and exit")
    parser.add_argument("--device", choices=["cpu", "cuda"],
                        help="Override device from config")
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
    setup_logger("af_claseq", log_file=log_dir / "vae_embedding.log",
                 level=logging.INFO, propagate=False, add_console_handler=True)
    logger = get_logger("run_vae_embedding")

    if args.validate_only:
        logger.info("Config validation passed.")
        return 0

    if args.device:
        config = VaeTrainConfig.from_dict({
            **{"general": {**config.general.__dict__, "device": args.device}},
            "inputs": config.inputs.__dict__,
            "structure_analysis": config.structure_analysis.__dict__,
            "coord_extraction": config.coord_extraction.__dict__,
            "vae": {"model": config.vae.model.__dict__,
                    "training": config.vae.training.__dict__},
            "output": config.output.__dict__,
        })

    logger.info(f"Protein: {config.general.protein_name}")
    logger.info(f"Device: {config.general.device}")
    logger.info(f"Latent dim: {config.vae.model.latent_dim}")

    try:
        trainer = VaeTrainer(config)
        emb_path = trainer.train()
        logger.info(f"Done. Embedding saved to {emb_path}")
        return 0
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
