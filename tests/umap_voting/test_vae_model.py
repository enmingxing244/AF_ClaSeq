import torch

from af_claseq.umap_voting.vae.model import ProteinVAE


def test_vae_forward_shapes():
    n_residues, latent_dim, batch = 16, 8, 4
    model = ProteinVAE(
        n_residues=n_residues,
        latent_dim=latent_dim,
        hidden_channels=[16, 32],
        use_residual=True,
    )
    x = torch.randn(batch, n_residues, 3)
    out = model(x)
    assert out["recon"].shape == (batch, n_residues, 3)
    assert out["mu"].shape == (batch, latent_dim)
    assert out["logvar"].shape == (batch, latent_dim)


def test_vae_encode_only_returns_mu():
    model = ProteinVAE(
        n_residues=16, latent_dim=8, hidden_channels=[16, 32]
    )
    x = torch.randn(2, 16, 3)
    mu = model.encode(x)
    assert mu.shape == (2, 8)


def test_vae_loss_components():
    model = ProteinVAE(
        n_residues=16, latent_dim=8, hidden_channels=[16, 32]
    )
    x = torch.randn(2, 16, 3)
    losses = model.loss(x, kl_weight=0.1)
    assert "recon" in losses and "kl" in losses and "total" in losses
    assert losses["total"].requires_grad
