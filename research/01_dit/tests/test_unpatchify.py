"""Unit tests for Unpatchify module."""

import pytest
import torch
from src.unpatchify import Unpatchify, unpatchify


def test_unpatchify_basic_shape():
    """Verify [B, N, P*P*C] -> [B, C, H, W] with B=2, N=256, P=2, C=4 -> [2, 4, 32, 32]."""
    B, N, P, C = 2, 256, 2, 4
    D_patch = P * P * C  # 16
    x = torch.randn(B, N, D_patch)

    out = unpatchify(x, patch_size=P, out_channels=C)

    assert out.shape == (B, C, 32, 32), f"Expected shape (2, 4, 32, 32), got {out.shape}"


def test_unpatchify_spatial_exact_roundtrip():
    """Verify exact spatial reconstruction by comparing manual pixel patching and unpatchify."""
    B, C, H, W = 1, 2, 4, 4
    P = 2
    h, w = H // P, W // P  # 2, 2
    N = h * w  # 4

    # Create distinct pixel values: 0 to 31
    orig = torch.arange(B * C * H * W, dtype=torch.float32).reshape(B, C, H, W)

    # Manually extract patches in [B, h, w, P, P, C] layout
    # orig is [B, C, H, W] -> reshape to [B, C, h, P, w, P] -> permute to [B, h, w, P, P, C] -> flatten(1, 2).flatten(2)
    patches = (
        orig.view(B, C, h, P, w, P)
        .permute(0, 2, 4, 3, 5, 1)
        .contiguous()
        .view(B, N, P * P * C)
    )

    reconstructed = unpatchify(patches, patch_size=P, out_channels=C)

    assert torch.equal(reconstructed, orig), (
        "Reconstructed tensor does not match original tensor pixel-by-pixel!\n"
        f"Original:\n{orig}\nReconstructed:\n{reconstructed}"
    )


def test_unpatchify_gradient_flow():
    """Verify gradients propagate backwards through unpatchify without disruption."""
    B, N, P, C = 2, 64, 2, 4
    x = torch.randn(B, N, P * P * C, requires_grad=True)

    out = unpatchify(x, patch_size=P, out_channels=C)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    # Since unpatchify is a pure permutation/reshape, every gradient element should be exactly 1.0
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_unpatchify_rectangular_grid():
    """Verify unpatchify works on non-square latent grids when grid_size is provided."""
    B, C, H, W = 2, 4, 16, 32
    P = 2
    h, w = H // P, W // P  # 8, 16
    N = h * w  # 128
    x = torch.randn(B, N, P * P * C)

    out = unpatchify(x, patch_size=P, out_channels=C, grid_size=(h, w))
    assert out.shape == (B, C, 16, 32)


def test_unpatchify_dimension_mismatch_error():
    """Verify ValueError is raised when passing D_model (e.g. 384) directly without linear projection."""
    B, N, D_model = 2, 256, 384
    x = torch.randn(B, N, D_model)

    with pytest.raises(ValueError, match="Expected token feature dimension"):
        unpatchify(x, patch_size=2, out_channels=4)


def test_unpatchify_module_wrapper():
    """Verify the nn.Module Unpatchify wrapper produces identical results."""
    B, N, P, C = 2, 256, 2, 4
    x = torch.randn(B, N, P * P * C)

    module = Unpatchify(patch_size=P, out_channels=C, latent_size=(32, 32))
    out = module(x)
    assert out.shape == (B, C, 32, 32)
