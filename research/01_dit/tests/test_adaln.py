"""Unit tests for adaLN-Zero Conditioning and Modulation module."""

import pytest
import torch
from src.adaln import AdaLNZero, modulate


def test_adaln_zero_shapes():
    """Verify condition vector c [B, D] chunks into 6 vectors [B, D] with B=2, D=384."""
    B, D = 2, 384
    adaln = AdaLNZero(hidden_size=D, num_modulations=6)

    c = torch.randn(B, D)
    chunks = adaln(c)

    assert len(chunks) == 6
    for chunk in chunks:
        assert chunk.shape == (B, D)

    shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = chunks
    assert shift_attn.shape == (B, D)
    assert scale_attn.shape == (B, D)
    assert gate_attn.shape == (B, D)
    assert shift_mlp.shape == (B, D)
    assert scale_mlp.shape == (B, D)
    assert gate_mlp.shape == (B, D)


def test_adaln_zero_exact_parameter_count():
    """Verify analytical parameter count: D * (6D) + 6D = 384 * 2304 + 2304 = 887,040."""
    D = 384
    adaln = AdaLNZero(hidden_size=D, num_modulations=6, bias=True)

    expected_params = (D * 6 * D) + (6 * D)
    actual_params = sum(p.numel() for p in adaln.parameters())

    assert actual_params == expected_params == 887040, (
        f"Expected {expected_params} parameters, but found {actual_params}"
    )


def test_adaln_zero_strict_initialization():
    """Verify that upon initialization, ALL 6 modulation vectors are EXACTLY 0.0."""
    B, D = 4, 384
    adaln = AdaLNZero(hidden_size=D, num_modulations=6)

    # Random non-zero conditioning inputs
    c = torch.randn(B, D)
    chunks = adaln(c)

    for idx, chunk in enumerate(chunks):
        assert torch.equal(chunk, torch.zeros_like(chunk)), (
            f"Modulation vector index {idx} is non-zero at initialization! "
            f"Max abs value: {chunk.abs().max().item()}"
        )


def test_modulate_function_identity_and_scaling():
    """Verify modulate(x, shift, scale) = x * (1 + scale) + shift behavior."""
    B, N, D = 2, 64, 384
    x = torch.randn(B, N, D)

    # 1. Identity case: shift=0, scale=0 -> output == x
    zero_shift = torch.zeros(B, D)
    zero_scale = torch.zeros(B, D)
    out_identity = modulate(x, zero_shift, zero_scale)
    assert torch.equal(out_identity, x), "modulate with (0, 0) did not produce exact identity."

    # 2. Linear scaling case: shift=2.0, scale=3.0 -> output == 4*x + 2.0
    shift = torch.full((B, D), 2.0)
    scale = torch.full((B, D), 3.0)
    out_scaled = modulate(x, shift, scale)
    expected_scaled = x * (1.0 + 3.0) + 2.0
    assert torch.allclose(out_scaled, expected_scaled, atol=1e-6)


def test_residual_gating_identity():
    """Verify that when gate alpha is 0, residual branch acts as an exact identity function."""
    B, N, D = 2, 64, 384
    x = torch.randn(B, N, D)
    h = torch.randn(B, N, D)  # Attention or MLP branch output

    gate = torch.zeros(B, D)  # alpha = 0

    # Residual addition: x' = x + alpha * h
    x_out = x + gate.unsqueeze(1) * h

    assert torch.equal(x_out, x), "Zero gate alpha did not preserve exact residual identity."


def test_adaln_zero_gradient_flow():
    """Verify gradients propagate backwards through AdaLNZero into conditioning vector c and weights."""
    adaln = AdaLNZero(hidden_size=384)
    # Give non-zero weights for gradient test
    torch.nn.init.normal_(adaln.linear.weight, std=0.02)
    torch.nn.init.zeros_(adaln.linear.bias)

    c = torch.randn(2, 384, requires_grad=True)
    chunks = adaln(c)
    loss = sum(chunk.sum() for chunk in chunks)
    loss.backward()

    assert c.grad is not None
    assert torch.isfinite(c.grad).all(), "Non-finite gradient in condition vector c."
    assert torch.isfinite(adaln.linear.weight.grad).all(), "Non-finite gradient in linear weight."


def test_adaln_zero_dtypes():
    """Verify AdaLNZero operates across float16, bfloat16, float32, and float64."""
    for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
        adaln = AdaLNZero(hidden_size=384).to(dtype=dtype)
        c = torch.randn(2, 384, dtype=dtype)
        chunks = adaln(c)
        for chunk in chunks:
            assert chunk.dtype == dtype, f"Expected {dtype}, got {chunk.dtype}"


def test_adaln_zero_dimension_mismatch_error():
    """Verify ValueError is raised if condition vector dimension does not match hidden_size."""
    adaln = AdaLNZero(hidden_size=384)
    c_invalid = torch.randn(2, 256)

    with pytest.raises(ValueError, match="Expected conditioning dimension"):
        adaln(c_invalid)
