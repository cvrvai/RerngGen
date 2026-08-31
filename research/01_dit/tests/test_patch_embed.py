"""Unit tests for PatchEmbed module."""

import pytest
import torch
from src.patch_embed import PatchEmbed


def test_patch_embed_basic_shape():
    """Verify [B, C, H, W] -> [B, N, D] transformation with B=2, C=4, H=32, W=32, P=2, D=384."""
    B, C, H, W = 2, 4, 32, 32
    patch_size = 2
    hidden_size = 384

    patch_embed = PatchEmbed(
        latent_size=H,
        patch_size=patch_size,
        in_channels=C,
        hidden_size=hidden_size,
    )

    x = torch.randn(B, C, H, W)
    out = patch_embed(x)

    expected_N = (H // patch_size) * (W // patch_size)  # (16 * 16) = 256
    assert out.shape == (B, expected_N, hidden_size), (
        f"Expected shape {(B, expected_N, hidden_size)}, but got {out.shape}"
    )
    assert patch_embed.num_patches == expected_N
    assert patch_embed.grid_size == (16, 16)


def test_patch_embed_parameter_count():
    """Verify analytical parameter count: (P * P * C * D) + D bias."""
    P, C, D = 2, 4, 384
    patch_embed = PatchEmbed(latent_size=32, patch_size=P, in_channels=C, hidden_size=D, bias=True)

    expected_params = (P * P * C * D) + D  # 2*2*4*384 + 384 = 6144 + 384 = 6528
    actual_params = sum(p.numel() for p in patch_embed.parameters())
    assert actual_params == expected_params, (
        f"Expected {expected_params} parameters, but found {actual_params}"
    )


def test_patch_embed_gradient_flow():
    """Verify backward gradient computation is finite and updates weights."""
    B, C, H, W = 2, 4, 16, 16
    patch_embed = PatchEmbed(latent_size=H, patch_size=2, in_channels=C, hidden_size=128)

    x = torch.randn(B, C, H, W, requires_grad=True)
    out = patch_embed(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all(), "Non-finite gradients detected in input x."
    assert torch.isfinite(patch_embed.proj.weight.grad).all(), (
        "Non-finite gradients detected in projection weight."
    )


def test_patch_embed_invalid_spatial_dimensions():
    """Ensure ValueError is raised if H or W is not divisible by patch_size."""
    patch_embed = PatchEmbed(latent_size=32, patch_size=2, in_channels=4, hidden_size=128)

    x_invalid = torch.randn(2, 4, 33, 32)
    with pytest.raises(ValueError, match="divisible by patch_size"):
        patch_embed(x_invalid)


def test_patch_embed_invalid_channels():
    """Ensure ValueError is raised if input channels do not match in_channels."""
    patch_embed = PatchEmbed(latent_size=32, patch_size=2, in_channels=4, hidden_size=128)

    x_invalid = torch.randn(2, 3, 32, 32)
    with pytest.raises(ValueError, match="Expected input channels 4"):
        patch_embed(x_invalid)
