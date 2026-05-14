"""Tests for ProteinVAE model: forward, encode, loss."""

import pytest
import torch
from af_claseq.umap_voting.vae.model import ProteinVAE, ResBlock1D


class TestResBlock1D:
    def test_shape_preserved(self):
        block = ResBlock1D(16)
        x = torch.randn(2, 16, 10)
        out = block(x)
        assert out.shape == x.shape


class TestProteinVAE:
    @pytest.fixture
    def model(self):
        return ProteinVAE(n_residues=15, latent_dim=4, hidden_channels=[16, 32])

    def test_forward_shapes(self, model):
        x = torch.randn(8, 15, 3)
        out = model(x)
        assert out["recon"].shape == (8, 15, 3)
        assert out["mu"].shape == (8, 4)
        assert out["logvar"].shape == (8, 4)
        assert out["z"].shape == (8, 4)

    def test_encode_returns_mu(self, model):
        x = torch.randn(4, 15, 3)
        mu = model.encode(x)
        assert mu.shape == (4, 4)

    def test_logvar_clamped(self, model):
        x = torch.randn(4, 15, 3) * 100
        out = model(x)
        assert out["logvar"].min() >= -10.0
        assert out["logvar"].max() <= 10.0

    def test_loss_sum_then_divide(self, model):
        x = torch.randn(4, 15, 3)
        out = model(x)
        losses = ProteinVAE.loss(x, out["recon"], out["mu"], out["logvar"], kl_weight=0.05)
        assert "recon" in losses
        assert "kl" in losses
        assert "total" in losses
        expected = losses["recon"] + 0.05 * losses["kl"]
        assert torch.allclose(losses["total"], expected, atol=1e-5)

    def test_loss_kl_nonnegative(self, model):
        x = torch.randn(4, 15, 3)
        out = model(x)
        losses = ProteinVAE.loss(x, out["recon"], out["mu"], out["logvar"])
        assert losses["kl"].item() >= 0

    def test_no_residual(self):
        model = ProteinVAE(n_residues=15, latent_dim=4,
                           hidden_channels=[16, 32], use_residual=False)
        x = torch.randn(2, 15, 3)
        out = model(x)
        assert out["recon"].shape == (2, 15, 3)

    def test_three_layer(self):
        model = ProteinVAE(n_residues=15, latent_dim=8,
                           hidden_channels=[16, 32, 64])
        x = torch.randn(2, 15, 3)
        out = model(x)
        assert out["mu"].shape == (2, 8)
