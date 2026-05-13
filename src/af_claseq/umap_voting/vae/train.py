"""VAE training + structure-to-embedding encoding."""
from __future__ import annotations

import random as _random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from af_claseq.umap_voting.config import VaeTrainConfig
from af_claseq.umap_voting.coords import CoordExtractor
from af_claseq.umap_voting.vae.model import ProteinVAE
from af_claseq.utils.logging_utils import get_logger

logger = get_logger("umap_voting.vae_train")


def _seed_everything(seed: int) -> None:
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_all_coords(
    structures_df: pd.DataFrame,
    references_df: pd.DataFrame,
    extractor: CoordExtractor,
) -> Tuple[np.ndarray, List[str], List[str], np.ndarray, List[str]]:
    samp_coords, samp_pdb, samp_a3m = [], [], []
    for _, row in structures_df.iterrows():
        c = extractor.extract(row["pdb_path"])
        if c is None:
            logger.warning(f"skipping {row['pdb_path']}: coord extraction failed")
            continue
        samp_coords.append(c)
        samp_pdb.append(str(row["pdb_path"]))
        samp_a3m.append(str(row.get("a3m_path", "") or ""))

    ref_coords, ref_pdb, ref_a3m, ref_labels = [], [], [], []
    for _, row in references_df.iterrows():
        c = extractor.extract(
            str(row["ref_pdb"]), chain_id=str(row["ref_chain"])
        )
        if c is None:
            raise ValueError(
                f"could not extract coords for reference {row['ref_label']}"
            )
        ref_coords.append(c)
        ref_pdb.append(str(row["ref_pdb"]))
        ref_a3m.append("")
        ref_labels.append(str(row["ref_label"]))

    all_coords = np.stack(samp_coords + ref_coords)
    all_pdb = samp_pdb + ref_pdb
    all_a3m = samp_a3m + ref_a3m
    is_ref = np.array([False] * len(samp_pdb) + [True] * len(ref_pdb))
    ref_label_full = [""] * len(samp_pdb) + ref_labels
    return all_coords, all_pdb, all_a3m, is_ref, ref_label_full


class VaeTrainer:
    def __init__(self, cfg: VaeTrainConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.general.device)
        self.out_dir = Path(cfg.general.base_dir) / "vae"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> None:
        _seed_everything(self.cfg.general.random_seed)

        extractor = CoordExtractor(
            sa_json_path=self.cfg.structure_analysis.config_json,
            coord_target=self.cfg.structure_analysis.coord_target,
            alignment_ref_pdb=self.cfg.coord_extraction.alignment_ref_pdb,
            alignment_ref_chain=self.cfg.coord_extraction.alignment_ref_chain,
            target_chain=self.cfg.coord_extraction.target_chain,
        )

        structures = pd.read_csv(self.cfg.inputs.structures_csv)
        refs = pd.read_csv(self.cfg.inputs.references_csv)
        coords, pdb_paths, a3m_paths, is_ref, ref_labels = _load_all_coords(
            structures, refs, extractor
        )
        logger.info(
            f"loaded {(~is_ref).sum()} sampling + {is_ref.sum()} ref structures"
        )

        mean = coords[~is_ref].mean(axis=(0, 1))
        std = coords[~is_ref].std(axis=(0, 1)) + 1e-6
        coords_n = (coords - mean) / std
        np.savez(
            self.out_dir / "normalization_params.npz",
            coords_mean=mean,
            coords_std=std,
        )

        x_train = torch.from_numpy(coords_n[~is_ref])
        n_train = x_train.shape[0]
        if n_train < 2:
            raise ValueError(
                f"need at least 2 sampling structures for VAE training, "
                f"got {n_train}"
            )
        n_val = max(1, int(self.cfg.vae.training.val_split * n_train))
        train_ds, val_ds = random_split(
            TensorDataset(x_train),
            [n_train - n_val, n_val],
            generator=torch.Generator().manual_seed(self.cfg.general.random_seed),
        )
        train_dl = DataLoader(
            train_ds, batch_size=self.cfg.vae.training.batch_size, shuffle=True
        )
        val_dl = DataLoader(val_ds, batch_size=self.cfg.vae.training.batch_size)

        model = ProteinVAE(
            n_residues=coords.shape[1],
            latent_dim=self.cfg.vae.model.latent_dim,
            hidden_channels=self.cfg.vae.model.hidden_channels,
            use_residual=self.cfg.vae.model.use_residual,
        ).to(self.device)
        opt = torch.optim.Adam(
            model.parameters(), lr=self.cfg.vae.training.learning_rate
        )

        best_val = float("inf")
        patience = self.cfg.vae.training.early_stopping_patience
        bad_epochs = 0
        for epoch in range(1, self.cfg.vae.training.epochs + 1):
            model.train(True)
            train_total = 0.0
            for (xb,) in train_dl:
                xb = xb.to(self.device)
                loss = model.loss(xb, kl_weight=self.cfg.vae.training.kl_weight)
                opt.zero_grad()
                loss["total"].backward()
                opt.step()
                train_total += float(loss["total"]) * xb.shape[0]
            train_total /= n_train - n_val

            model.train(False)
            val_total = 0.0
            with torch.no_grad():
                for (xb,) in val_dl:
                    xb = xb.to(self.device)
                    vl = model.loss(
                        xb, kl_weight=self.cfg.vae.training.kl_weight
                    )
                    val_total += float(vl["total"]) * xb.shape[0]
            val_total /= n_val

            logger.info(
                f"epoch {epoch}/{self.cfg.vae.training.epochs} "
                f"train={train_total:.4f} val={val_total:.4f}"
            )

            if val_total < best_val:
                best_val = val_total
                bad_epochs = 0
                torch.save(
                    model.state_dict(),
                    self.out_dir / "protein_vae_best.pth",
                )
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    logger.info(f"early stop at epoch {epoch}")
                    break

        model.load_state_dict(
            torch.load(
                self.out_dir / "protein_vae_best.pth",
                weights_only=True,
            )
        )
        model.train(False)
        with torch.no_grad():
            mu = (
                model.encode(torch.from_numpy(coords_n).to(self.device))
                .cpu()
                .numpy()
            )
        np.savez(
            self.out_dir / self.cfg.output.embedding_filename,
            mu=mu.astype(np.float32),
            pdb_paths=np.array(pdb_paths),
            a3m_paths=np.array(a3m_paths),
            is_reference=is_ref,
            ref_label=np.array(ref_labels),
        )
        logger.info(
            f"wrote {self.out_dir / self.cfg.output.embedding_filename}"
        )
