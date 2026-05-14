"""VAE training pipeline: coord extraction → normalize → train → encode → save."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from af_claseq.utils.logging_utils import get_logger

from ..config import VaeTrainConfig
from ..coords import CoordExtractor, extract_aligned_coords, load_cached_coords
from .model import ProteinVAE

logger = get_logger("umap_voting.vae_train")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_all_coords(
    extractor: CoordExtractor,
    structures_csv: str | Path,
    references_csv: str | Path,
) -> Tuple[
    np.ndarray,       # coords (N, L, 3)
    List[str],         # pdb_paths
    List[str],         # a3m_paths
    np.ndarray,        # is_reference bool
    List[str],         # ref_labels ("" for sampling)
]:
    structs = pd.read_csv(structures_csv)
    refs = pd.read_csv(references_csv)

    all_coords, pdb_paths, a3m_paths, is_ref, ref_labels = [], [], [], [], []

    for _, row in structs.iterrows():
        c = extractor.extract(row["pdb_path"])
        if c is None:
            logger.warning(f"Skipping sampling structure: {row['pdb_path']}")
            continue
        all_coords.append(c)
        pdb_paths.append(row["pdb_path"])
        a3m_paths.append(row["a3m_path"])
        is_ref.append(False)
        ref_labels.append("")

    logger.info(f"Extracted coords from {len(all_coords)}/{len(structs)} sampling structures")

    for _, row in refs.iterrows():
        ref_chain = row.get("ref_chain", extractor.chain_id)
        c = extract_aligned_coords(
            row["ref_pdb"],
            ref_chain,
            extractor.residue_indices,
            extractor.superposition_indices,
            extractor.alignment_ref_coords,
            extractor.min_superposition_atoms,
        )
        if c is None:
            raise RuntimeError(f"Reference extraction failed: {row['ref_pdb']}")
        all_coords.append(c)
        pdb_paths.append(row["ref_pdb"])
        a3m_paths.append("")
        is_ref.append(True)
        ref_labels.append(row["ref_label"])

    logger.info(f"Extracted coords from {len(refs)} reference structures")
    coords = np.stack(all_coords)
    return coords, pdb_paths, a3m_paths, np.array(is_ref), ref_labels


class VaeTrainer:
    def __init__(self, config: VaeTrainConfig):
        self.cfg = config

    def train(self) -> Path:
        """Run the full training pipeline and return path to embedding.npz."""
        cfg = self.cfg
        _seed_everything(cfg.general.random_seed)

        vae_dir = cfg.get_vae_dir()
        vae_dir.mkdir(parents=True, exist_ok=True)

        device = torch.device(cfg.general.device)
        logger.info(f"Device: {device}")

        # ---- Load coordinates (cached or fresh) ----
        coords_npz = vae_dir / "coords.npz"
        if coords_npz.exists():
            logger.info("Loading pre-extracted coords from coords.npz")
            coords, pdb_paths, a3m_paths, is_ref, ref_labels = load_cached_coords(cfg)
        else:
            logger.info("No cached coords — extracting inline (consider run_coord_extraction.py)")
            extractor = CoordExtractor(cfg.structure_analysis, cfg.coord_extraction)
            coords, pdb_paths, a3m_paths, is_ref, ref_labels = _load_all_coords(
                extractor, cfg.inputs.structures_csv, cfg.inputs.references_csv
            )

        n_total, n_residues, _ = coords.shape
        n_sampling = int((~is_ref).sum())
        n_refs = int(is_ref.sum())
        logger.info(f"Total: {n_total} ({n_sampling} sampling + {n_refs} refs), "
                     f"{n_residues} residues")

        # ---- Normalize (from sampling only) ----
        sampling_coords = coords[~is_ref]
        mode = cfg.vae.training.normalization_mode

        if mode == "global":
            mean = sampling_coords.mean(axis=(0, 1))  # (3,)
            std = sampling_coords.std(axis=(0, 1))     # (3,)
            std[std < 1e-8] = 1.0
        elif mode == "per_residue":
            mean = sampling_coords.mean(axis=0)  # (L, 3)
            std = sampling_coords.std(axis=0)    # (L, 3)
            std[std < 1e-8] = 1.0
        else:  # center_only
            mean = sampling_coords.mean(axis=(0, 1))  # (3,)
            std = np.ones_like(mean)

        np.savez(
            vae_dir / "normalization_params.npz",
            mean=mean, std=std, mode=np.array(mode),
        )
        logger.info(f"Normalization mode: {mode}, mean shape: {mean.shape}")

        coords_normed = (coords - mean) / std

        # ---- Train/val split (sampling only; refs always train) ----
        sampling_idx = np.where(~is_ref)[0]
        ref_idx = np.where(is_ref)[0]
        rng = np.random.RandomState(cfg.general.random_seed)
        perm = rng.permutation(len(sampling_idx))

        n_val = max(1, int(len(sampling_idx) * cfg.vae.training.val_split))
        val_idx = sampling_idx[perm[:n_val]]
        train_samp_idx = sampling_idx[perm[n_val:]]
        train_idx = np.concatenate([train_samp_idx, ref_idx])

        train_tensor = torch.from_numpy(coords_normed[train_idx]).float()
        val_tensor = torch.from_numpy(coords_normed[val_idx]).float()

        train_loader = DataLoader(
            TensorDataset(train_tensor),
            batch_size=cfg.vae.training.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(val_tensor),
            batch_size=cfg.vae.training.batch_size,
        )

        logger.info(f"Train: {len(train_idx)} ({len(train_samp_idx)} sampling + "
                     f"{len(ref_idx)} refs), Val: {len(val_idx)}")

        # ---- Build model ----
        model = ProteinVAE(
            n_residues=n_residues,
            latent_dim=cfg.vae.model.latent_dim,
            hidden_channels=list(cfg.vae.model.hidden_channels),
            use_residual=cfg.vae.model.use_residual,
        ).to(device)

        opt_kwargs: Dict = {"lr": cfg.vae.training.learning_rate}
        if cfg.vae.training.weight_decay > 0:
            opt_kwargs["weight_decay"] = cfg.vae.training.weight_decay
        optimizer = torch.optim.Adam(model.parameters(), **opt_kwargs)

        scheduler = None
        if cfg.vae.training.lr_scheduler_factor < 1.0:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=cfg.vae.training.lr_scheduler_factor,
                patience=cfg.vae.training.lr_scheduler_patience,
            )

        kl_w = cfg.vae.training.kl_weight
        best_val = float("inf")
        patience_counter = 0
        ckpt_path = vae_dir / "protein_vae_best.pth"

        # ---- Training loop ----
        for epoch in range(1, cfg.vae.training.epochs + 1):
            model.train()
            train_recon_sum, train_kl_sum, train_n = 0.0, 0.0, 0
            for (batch,) in train_loader:
                batch = batch.to(device)
                out = model(batch)
                losses = ProteinVAE.loss(batch, out["recon"], out["mu"], out["logvar"], kl_w)

                optimizer.zero_grad()
                losses["total"].backward()
                if cfg.vae.training.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.vae.training.grad_clip
                    )
                optimizer.step()

                bs = batch.size(0)
                train_recon_sum += losses["recon"].item() * bs
                train_kl_sum += losses["kl"].item() * bs
                train_n += bs

            # ---- Validation ----
            model.eval()
            val_loss_sum, val_n = 0.0, 0
            with torch.no_grad():
                for (batch,) in val_loader:
                    batch = batch.to(device)
                    out = model(batch)
                    losses = ProteinVAE.loss(
                        batch, out["recon"], out["mu"], out["logvar"], kl_w
                    )
                    val_loss_sum += losses["total"].item() * batch.size(0)
                    val_n += batch.size(0)

            val_loss = val_loss_sum / val_n if val_n > 0 else float("inf")

            if scheduler is not None:
                scheduler.step(val_loss)

            if epoch % 50 == 0 or epoch == 1:
                logger.info(
                    f"Epoch {epoch:4d}  "
                    f"train_recon={train_recon_sum / train_n:.4f}  "
                    f"train_kl={train_kl_sum / train_n:.4f}  "
                    f"val_loss={val_loss:.4f}"
                )

            if val_loss < best_val:
                best_val = val_loss
                patience_counter = 0
                if cfg.vae.training.save_best_only:
                    torch.save(model.state_dict(), ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= cfg.vae.training.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch} (best val={best_val:.4f})")
                    break

        if not cfg.vae.training.save_best_only:
            torch.save(model.state_dict(), ckpt_path)

        # ---- Reload best and encode in batches ----
        if ckpt_path.exists():
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()

        all_tensor = torch.from_numpy(coords_normed).float()
        mu_list = []
        encode_bs = cfg.vae.training.batch_size * 4
        with torch.no_grad():
            for i in range(0, len(all_tensor), encode_bs):
                chunk = all_tensor[i : i + encode_bs].to(device)
                mu_list.append(model.encode(chunk).cpu().numpy())
        mu = np.concatenate(mu_list, axis=0)

        # ---- Save embedding ----
        emb_path = vae_dir / cfg.output.embedding_filename
        np.savez(
            emb_path,
            mu=mu,
            pdb_paths=np.array(pdb_paths, dtype=object),
            a3m_paths=np.array(a3m_paths, dtype=object),
            is_reference=is_ref,
            ref_label=np.array(ref_labels, dtype=object),
        )
        logger.info(f"Saved embedding ({mu.shape}) to {emb_path}")
        return emb_path
