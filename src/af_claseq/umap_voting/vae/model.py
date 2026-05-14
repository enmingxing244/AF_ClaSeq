"""1-D convolutional VAE for protein Calpha coordinates."""

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
        residual = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return F.relu(out + residual)


class ProteinVAE(nn.Module):
    """1-D convolutional VAE for ``(B, n_residues, 3)`` Calpha coordinates."""

    def __init__(
        self,
        n_residues: int,
        latent_dim: int = 6,
        hidden_channels: List[int] | None = None,
        use_residual: bool = True,
    ):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [32, 64]

        self.n_residues = n_residues
        self.latent_dim = latent_dim

        # --- Encoder ---
        enc_layers: list[nn.Module] = []
        in_ch = 3
        for out_ch in hidden_channels:
            enc_layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1))
            enc_layers.append(nn.ReLU())
            if use_residual:
                enc_layers.append(ResBlock1D(out_ch))
            in_ch = out_ch
        self.encoder = nn.Sequential(*enc_layers)

        self._enc_flat = hidden_channels[-1] * n_residues
        self.fc_mu = nn.Linear(self._enc_flat, latent_dim)
        self.fc_logvar = nn.Linear(self._enc_flat, latent_dim)

        # --- Decoder ---
        self.fc_dec = nn.Linear(latent_dim, self._enc_flat)
        dec_layers: list[nn.Module] = []
        rev_channels = list(reversed(hidden_channels))
        in_ch = rev_channels[0]
        for out_ch in rev_channels[1:]:
            if use_residual:
                dec_layers.append(ResBlock1D(in_ch))
            dec_layers.append(nn.ConvTranspose1d(in_ch, out_ch, kernel_size=3, padding=1))
            dec_layers.append(nn.ReLU())
            in_ch = out_ch
        dec_layers.append(nn.ConvTranspose1d(in_ch, 3, kernel_size=3, padding=1))
        self.decoder = nn.Sequential(*dec_layers)

    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``mu`` only (inference path)."""
        # x: (B, n_residues, 3) -> (B, 3, n_residues)
        h = self.encoder(x.transpose(1, 2))
        h = h.reshape(h.size(0), -1)
        return self.fc_mu(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.encoder(x.transpose(1, 2))
        h = h.reshape(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), min=-10.0, max=10.0)
        z = self.reparameterize(mu, logvar)

        dec_in = self.fc_dec(z).reshape(z.size(0), -1, self.n_residues)
        recon = self.decoder(dec_in).transpose(1, 2)  # (B, n_residues, 3)
        return {"recon": recon, "mu": mu, "logvar": logvar, "z": z}

    # ------------------------------------------------------------------
    @staticmethod
    def loss(
        x: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        kl_weight: float = 0.05,
    ) -> Dict[str, torch.Tensor]:
        """Sum-then-divide loss so kl_weight is a true beta."""
        bs = x.size(0)
        recon_loss = F.mse_loss(recon, x, reduction="sum") / bs
        kl = torch.sum(-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())) / bs
        return {"recon": recon_loss, "kl": kl, "total": recon_loss + kl_weight * kl}
