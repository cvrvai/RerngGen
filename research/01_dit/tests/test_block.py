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
    """Verify that both LayerNorm modules have elementwise_affine=False and absent parameters."""
    block = DiTBlock(hidden_size=384, num_heads=6)

    assert block.norm1.elementwise_affine is False, "norm1 must have elementwise_affine=False."
    assert block.norm2.elementwise_affine is False, "norm2 must have elementwise_affine=False."

    # Verify that trainable parameters do NOT exist (are None)
    assert block.norm1.weight is None, "norm1 has a weight parameter."
    assert block.norm1.bias is None, "norm1 has a bias parameter."
    assert block.norm2.weight is None, "norm2 has a weight parameter."
    assert block.norm2.bias is None, "norm2 has a bias parameter."


def test_dit_block_gradient_at_strict_initialization():
    """Verify exact gradient propagation mechanics under strict zero initialization.

    At initialization:
    - x.grad is non-zero (flowing directly through identity residual path x' = x + 0)
    - adaLN gate rows receive non-zero gradients
    - attention and MLP branch weights receive exactly zero gradient (multiplied by gate=0)
    - c.grad is exactly 0.0 (multiplied by zero modulation weights)
    """
    block = DiTBlock(hidden_size=384, num_heads=6)

    x = torch.randn(2, 64, 384, requires_grad=True)
    c = torch.randn(2, 384, requires_grad=True)

    out = block(x, c)
    loss = out.sum()
    loss.backward()

    # 1. x.grad flows through the identity residual path
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert (x.grad != 0).any(), "x.grad must be non-zero through identity path."
    assert torch.allclose(x.grad, torch.ones_like(x.grad)), "x.grad should be 1.0 through identity path."

    # 2. c.grad is exactly 0.0 because adaLN linear weights are initialized to 0
    assert c.grad is not None
    assert torch.isfinite(c.grad).all()
    assert torch.equal(c.grad, torch.zeros_like(c.grad)), "c.grad must be 0.0 at zero-init."

    # 3. Attention and MLP weights receive zero gradients because their outputs are multiplied by alpha=0
    assert block.attn.qkv.weight.grad is not None
    assert torch.equal(block.attn.qkv.weight.grad, torch.zeros_like(block.attn.qkv.weight.grad)), (
        "attn weights must have 0.0 gradient at zero-init because gate alpha_1 is 0."
    )
    assert block.mlp.fc1.weight.grad is not None
    assert torch.equal(block.mlp.fc1.weight.grad, torch.zeros_like(block.mlp.fc1.weight.grad)), (
        "mlp weights must have 0.0 gradient at zero-init because gate alpha_2 is 0."
    )

    # 4. Gate-related rows of adaLN receive non-zero learning signals
    adaln_weight_grad = block.adaLN_modulation.linear.weight.grad
    assert adaln_weight_grad is not None
    assert torch.isfinite(adaln_weight_grad).all()
    # Chunk weights by 6 modulations: shift1, scale1, gate1, shift2, scale2, gate2
    gate1_row_grad = adaln_weight_grad[2 * 384 : 3 * 384, :]
    gate2_row_grad = adaln_weight_grad[5 * 384 : 6 * 384, :]
    assert (gate1_row_grad != 0).any(), "Gate 1 row must receive non-zero gradient."
    assert (gate2_row_grad != 0).any(), "Gate 2 row must receive non-zero gradient."


def test_dit_block_gradient_after_gates_open():
    """Verify that once gates/weights become non-zero, gradients propagate into all branches."""
    block = DiTBlock(hidden_size=384, num_heads=6)

    # Simulate opened gates and non-zero modulation weights
    nn.init.normal_(block.adaLN_modulation.linear.weight, std=0.02)
    nn.init.zeros_(block.adaLN_modulation.linear.bias)

    x = torch.randn(2, 64, 384, requires_grad=True)
    c = torch.randn(2, 384, requires_grad=True)

    out = block(x, c)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None and (x.grad != 0).any()
    assert c.grad is not None and (c.grad != 0).any(), "c.grad must be non-zero once weights are non-zero."
    assert block.attn.qkv.weight.grad is not None and (block.attn.qkv.weight.grad != 0).any()
    assert block.mlp.fc1.weight.grad is not None and (block.mlp.fc1.weight.grad != 0).any()


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
