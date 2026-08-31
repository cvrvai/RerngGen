"""Unit tests for FinalLayer Head module."""

import pytest
import torch
import torch.nn as nn
from src.final_layer import FinalLayer
from src.unpatchify import unpatchify


def test_final_layer_basic_shape():
    """Verify [B, N, D] + [B, D] -> [B, N, P^2 * C_out] with B=2, N=256, D=384, P=2, C_out=4."""
    B, N, D, P, C = 2, 256, 384, 2, 4
    final_layer = FinalLayer(hidden_size=D, patch_size=P, out_channels=C)

    x = torch.randn(B, N, D)
    c = torch.randn(B, D)
    out = final_layer(x, c)

    expected_out_dim = P * P * C  # 16
    assert out.shape == (B, N, expected_out_dim), f"Expected shape {(B, N, expected_out_dim)}, got {out.shape}"
    assert final_layer.out_features == 16


def test_final_layer_exact_parameter_count():
    """Verify analytical parameter count: adaLN (295,680) + Linear (6,160) = 301,840."""
    D, P, C = 384, 2, 4
    out_dim = P * P * C  # 16
    final_layer = FinalLayer(hidden_size=D, patch_size=P, out_channels=C)

    # adaLN: 384 * 768 + 768 = 295,680
    # Linear: 384 * 16 + 16 = 6,160
    # Total = 301,840
    expected_adaln = (D * 2 * D) + (2 * D)
    expected_linear = (D * out_dim) + out_dim
    expected_total = expected_adaln + expected_linear

    actual_total = sum(p.numel() for p in final_layer.parameters())
    assert actual_total == expected_total == 301840, (
        f"Expected {expected_total} parameters, but found {actual_total}"
    )


def test_final_layer_strict_zero_initialization():
    """Verify that at step 0, FinalLayer outputs IDENTICALLY ZERO velocity predictions."""
    B, N, D = 4, 128, 384
    final_layer = FinalLayer(hidden_size=D, patch_size=2, out_channels=4)

    x = torch.randn(B, N, D)
    c = torch.randn(B, D)

    out = final_layer(x, c)

    assert torch.equal(out, torch.zeros_like(out)), (
        f"FinalLayer output is not exactly 0.0 at initialization! Max abs: {out.abs().max().item()}"
    )


def test_final_layer_layernorm_no_elementwise_affine():
    """Verify norm_final has elementwise_affine=False and absent parameters."""
    final_layer = FinalLayer(hidden_size=384, patch_size=2, out_channels=4)

    assert final_layer.norm_final.elementwise_affine is False
    assert final_layer.norm_final.weight is None, "norm_final has a static weight parameter."
    assert final_layer.norm_final.bias is None, "norm_final has a static bias parameter."


def test_final_layer_plus_unpatchify_end_to_end():
    """Verify [B, N, D] -> FinalLayer -> [B, N, 16] -> Unpatchify -> [B, 4, 32, 32]."""
    B, N, D, P, C = 2, 256, 384, 2, 4
    final_layer = FinalLayer(hidden_size=D, patch_size=P, out_channels=C)

    # Make weights non-zero to test unpatchify reconstruction with rich features
    nn.init.normal_(final_layer.linear.weight, std=0.02)
    nn.init.normal_(final_layer.linear.bias, std=0.02)

    x = torch.randn(B, N, D)
    c = torch.randn(B, D)

    projected_patches = final_layer(x, c)  # [2, 256, 16]
    reconstructed_latent = unpatchify(projected_patches, patch_size=P, out_channels=C)  # [2, 4, 32, 32]

    assert reconstructed_latent.shape == (B, C, 32, 32), (
        f"Expected final reconstructed shape {(B, C, 32, 32)}, got {reconstructed_latent.shape}"
    )


def test_final_layer_gradient_flow_after_update():
    """Verify gradient propagation through FinalLayer when weights are active."""
    final_layer = FinalLayer(hidden_size=384, patch_size=2, out_channels=4)

    nn.init.normal_(final_layer.linear.weight, std=0.02)
    nn.init.normal_(final_layer.adaLN_modulation[-1].weight, std=0.02)

    x = torch.randn(2, 64, 384, requires_grad=True)
    c = torch.randn(2, 384, requires_grad=True)

    out = final_layer(x, c)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert c.grad is not None and torch.isfinite(c.grad).all()
    assert final_layer.linear.weight.grad is not None
    assert torch.isfinite(final_layer.linear.weight.grad).all()


def test_final_layer_dtypes():
    """Verify FinalLayer operates across float16, bfloat16, float32, and float64."""
    for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
        final_layer = FinalLayer(hidden_size=384, patch_size=2, out_channels=4).to(dtype=dtype)
        x = torch.randn(2, 32, 384, dtype=dtype)
        c = torch.randn(2, 384, dtype=dtype)
        out = final_layer(x, c)
        assert out.dtype == dtype, f"Expected {dtype}, got {out.dtype}"


def test_final_layer_dimension_mismatch_errors():
    """Verify ValueError is raised on mismatched input dimensions."""
    final_layer = FinalLayer(hidden_size=384, patch_size=2, out_channels=4)

    with pytest.raises(ValueError, match="Expected input feature dimension"):
        final_layer(torch.randn(2, 64, 380), torch.randn(2, 384))

    with pytest.raises(ValueError, match="Expected condition vector shape"):
        final_layer(torch.randn(2, 64, 384), torch.randn(2, 256))
