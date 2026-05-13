"""ResNet-style VAE on aligned Calpha coordinates of shape (n_residues, 3)."""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(F.relu(self.conv1(x)))


class ProteinVAE(nn.Module):
    """VAE: (B, n_residues, 3) -> latent (B, latent_dim) -> (B, n_residues, 3)."""

    def __init__(
        self,
        n_residues: int,
        latent_dim: int,
        hidden_channels: List[int],
        use_residual: bool = True,
    ):
        super().__init__()
        self.n_residues = n_residues
        self.latent_dim = latent_dim
        self._last_hidden = hidden_channels[-1]

        enc_layers: List[nn.Module] = []
        in_c = 3
        for h in hidden_channels:
            enc_layers.append(nn.Conv1d(in_c, h, kernel_size=3, padding=1))
            enc_layers.append(nn.ReLU(inplace=True))
            if use_residual:
                enc_layers.append(ResBlock1D(h))
            in_c = h
        self.encoder = nn.Sequential(*enc_layers)

        flat = hidden_channels[-1] * n_residues
        self.fc_mu = nn.Linear(flat, latent_dim)
        self.fc_logvar = nn.Linear(flat, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, flat)

        dec_layers: List[nn.Module] = []
        rev = list(reversed(hidden_channels))
        in_c = rev[0]
        for h in rev[1:]:
            dec_layers.append(nn.Conv1d(in_c, h, kernel_size=3, padding=1))
            dec_layers.append(nn.ReLU(inplace=True))
            if use_residual:
                dec_layers.append(ResBlock1D(h))
            in_c = h
        dec_layers.append(nn.Conv1d(in_c, 3, kernel_size=3, padding=1))
        self.decoder = nn.Sequential(*dec_layers)

    def _encode_features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x.transpose(1, 2))
        return h.flatten(1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_mu(self._encode_features(x))

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z)
        h = h.view(z.shape[0], self._last_hidden, self.n_residues)
        recon = self.decoder(h)
        return recon.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self._encode_features(x)
        mu = self.fc_mu(feat)
        logvar = self.fc_logvar(feat)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return {"recon": recon, "mu": mu, "logvar": logvar, "z": z}

    def loss(
        self, x: torch.Tensor, kl_weight: float
    ) -> Dict[str, torch.Tensor]:
        out = self.forward(x)
        recon = F.mse_loss(out["recon"], x, reduction="mean")
        kl = -0.5 * torch.mean(
            1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp()
        )
        total = recon + kl_weight * kl
        return {"recon": recon, "kl": kl, "total": total}
