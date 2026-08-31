"""Unit tests for single DiTBlock module."""

import pytest
import torch
import torch.nn as nn
from src.block import DiTBlock


def test_dit_block_basic_shape():
    """Verify [B, N, D] + [B, D] -> [B, N, D] with B=2, N=256, D=384, H=6."""
    B, N, D, H = 2, 256, 384, 6
    block = DiTBlock(hidden_size=D, num_heads=H)

    x = torch.randn(B, N, D)
    c = torch.randn(B, D)
    out = block(x, c)

    assert out.shape == (B, N, D), f"Expected shape {(B, N, D)}, but got {out.shape}"


def test_dit_block_exact_parameter_count():
    """Verify analytical parameter count: adaLN (887,040) + Attn (591,360) + MLP (1,181,568) = 2,659,968."""
    D, H = 384, 6
    block = DiTBlock(hidden_size=D, num_heads=H, mlp_ratio=4.0)

    expected_adaln = (D * 6 * D) + (6 * D)               # 887,040
    expected_attn = (D * 3 * D + 3 * D) + (D * D + D)    # 591,360
    expected_mlp = (D * 4 * D + 4 * D) + (4 * D * D + D) # 1,181,568
    expected_total = expected_adaln + expected_attn + expected_mlp  # 2,659,968

    actual_total = sum(p.numel() for p in block.parameters())
    assert actual_total == expected_total == 2659968, (
        f"Expected {expected_total} parameters, but found {actual_total}"
    )


def test_dit_block_strict_zero_init_identity():
    """Verify that at step 0 initialization, DiTBlock(x, c) evaluates as an EXACT IDENTITY mapping."""
    B, N, D = 4, 128, 384
    block = DiTBlock(hidden_size=D, num_heads=6)

    # Random non-zero inputs
    x = torch.randn(B, N, D)
    c = torch.randn(B, D)

    out = block(x, c)

    # The block output must be EXACTLY equal to input x because gates alpha_1 and alpha_2 are 0
    assert torch.equal(out, x), (
        f"DiTBlock did not evaluate to exact identity at initialization!\n"
        f"Max difference: {(out - x).abs().max().item()}"
    )


def test_dit_block_layernorm_no_elementwise_affine():
    """Verify that both LayerNorm modules in DiTBlock have elementwise_affine=False."""
    block = DiTBlock(hidden_size=384, num_heads=6)

    assert block.norm1.elementwise_affine is False, "norm1 must have elementwise_affine=False."
    assert block.norm2.elementwise_affine is False, "norm2 must have elementwise_affine=False."

    assert block.norm1.weight is None, "norm1 has static trainable weights."
    assert block.norm1.bias is None, "norm1 has static trainable biases."
    assert block.norm2.weight is None, "norm2 has static trainable weights."
    assert block.norm2.bias is None, "norm2 has static trainable biases."


def test_dit_block_gradient_flow():
    """Verify finite gradients propagate through DiTBlock to x, c, and all sub-modules."""
    block = DiTBlock(hidden_size=384, num_heads=6)

    # Give non-zero gates to test active backpropagation through both branches
    nn.init.normal_(block.adaLN_modulation.linear.weight, std=0.02)
    nn.init.zeros_(block.adaLN_modulation.linear.bias)

    x = torch.randn(2, 64, 384, requires_grad=True)
    c = torch.randn(2, 384, requires_grad=True)

    out = block(x, c)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    assert c.grad is not None
    assert torch.isfinite(c.grad).all()

    for name, param in block.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient."
        assert torch.isfinite(param.grad).all(), f"Parameter {name} has non-finite gradient."


def test_dit_block_batch_and_sequence_length_variations():
    """Verify DiTBlock operates across multiple batch sizes and sequence lengths."""
    block = DiTBlock(hidden_size=384, num_heads=6)
    block.eval()

    with torch.no_grad():
        for B in [1, 2, 4]:
            for N in [16, 64, 256]:
                x = torch.randn(B, N, 384)
                c = torch.randn(B, 384)
                out = block(x, c)
                assert out.shape == (B, N, 384)


def test_dit_block_dtypes():
    """Verify DiTBlock executes across float16, bfloat16, float32, and float64."""
    for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
        block = DiTBlock(hidden_size=384, num_heads=6).to(dtype=dtype)
        x = torch.randn(2, 32, 384, dtype=dtype)
        c = torch.randn(2, 384, dtype=dtype)
        out = block(x, c)
        assert out.dtype == dtype, f"Expected {dtype}, got {out.dtype}"


def test_dit_block_dimension_mismatch_errors():
    """Verify ValueError is raised on mismatched x or c dimensions."""
    block = DiTBlock(hidden_size=384, num_heads=6)

    # Invalid x feature dimension
    with pytest.raises(ValueError, match="Expected input feature dimension"):
        block(torch.randn(2, 64, 380), torch.randn(2, 384))

    # Invalid c condition dimension
    with pytest.raises(ValueError, match="Expected condition vector shape"):
        block(torch.randn(2, 64, 384), torch.randn(2, 256))
