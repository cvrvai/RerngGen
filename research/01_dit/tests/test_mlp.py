"""Unit tests for Transformer MLP module."""

import math
import pytest
import torch
import torch.nn.functional as F
from src.mlp import Mlp


def test_mlp_basic_shape():
    """Verify [B, N, D] -> [B, N, D] with expansion to 4D (1536) with B=2, N=256, D=384."""
    B, N, D = 2, 256, 384
    mlp = Mlp(in_features=D, hidden_features=1536)

    x = torch.randn(B, N, D)
    out = mlp(x)

    assert out.shape == (B, N, D), f"Expected shape {(B, N, D)}, but got {out.shape}"
    assert mlp.hidden_features == 1536


def test_mlp_exact_parameter_count():
    """Verify analytical parameter count: fc1 (D * 4D + 4D) + fc2 (4D * D + D)."""
    D = 384
    hidden_D = 4 * D  # 1536
    mlp = Mlp(in_features=D, hidden_features=hidden_D, bias=True)

    # fc1: 384 * 1536 + 1536 = 591,360
    # fc2: 1536 * 384 + 384 = 590,208
    # Total = 1,181,568
    expected_fc1 = (D * hidden_D) + hidden_D
    expected_fc2 = (hidden_D * D) + D
    expected_total = expected_fc1 + expected_fc2

    actual_total = sum(p.numel() for p in mlp.parameters())
    assert actual_total == expected_total == 1181568, (
        f"Expected {expected_total} parameters, but found {actual_total}"
    )


def test_mlp_token_independence_property():
    """Verify pointwise property: token i output depends solely on token i input.

    Altering token j (where j != i) must have ZERO effect on token i.
    """
    D = 384
    mlp = Mlp(in_features=D)
    mlp.eval()

    x1 = torch.randn(2, 64, D)
    x2 = x1.clone()

    # Alter token index 10 and 20 in x2
    x2[:, 10, :] += 5.0
    x2[:, 20, :] -= 3.0

    with torch.no_grad():
        out1 = mlp(x1)
        out2 = mlp(x2)

    # For all unaffected tokens (e.g. token 0 to 9, 11 to 19, 21 to 63), outputs must be identical
    unaffected_indices = [idx for idx in range(64) if idx not in (10, 20)]
    assert torch.equal(out1[:, unaffected_indices, :], out2[:, unaffected_indices, :]), (
        "Pointwise MLP property violated! Modifying token j affected token i output."
    )
    # The modified tokens must differ
    assert not torch.allclose(out1[:, 10, :], out2[:, 10, :])


def test_mlp_finite_outputs_and_gradients():
    """Verify forward outputs and backward gradients are finite and non-zero."""
    B, N, D = 2, 64, 384
    mlp = Mlp(in_features=D)

    x = torch.randn(B, N, D, requires_grad=True)
    out = mlp(x)

    assert torch.isfinite(out).all(), "Non-finite values detected in MLP output."

    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all(), "Non-finite gradients in input x."

    for name, param in mlp.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient."
        assert torch.isfinite(param.grad).all(), f"Parameter {name} has non-finite gradient."


def test_mlp_batch_and_sequence_length_variations():
    """Verify MLP operates across multiple batch sizes and sequence lengths."""
    mlp = Mlp(in_features=384)
    mlp.eval()

    with torch.no_grad():
        for B in [1, 2, 4]:
            for N in [16, 64, 256]:
                x = torch.randn(B, N, 384)
                out = mlp(x)
                assert out.shape == (B, N, 384)


def test_mlp_dtypes():
    """Verify MLP functions across float16, bfloat16, float32, and float64."""
    for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
        mlp = Mlp(in_features=384).to(dtype=dtype)
        x = torch.randn(2, 32, 384, dtype=dtype)
        out = mlp(x)
        assert out.dtype == dtype, f"Expected {dtype}, got {out.dtype}"


def test_mlp_dimension_mismatch_error():
    """Verify ValueError is raised if input feature dimension does not match in_features."""
    mlp = Mlp(in_features=384)
    x_invalid = torch.randn(2, 64, 380)

    with pytest.raises(ValueError, match="Expected input feature dimension"):
        mlp(x_invalid)
