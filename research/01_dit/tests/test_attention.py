"""Unit tests for Multi-Head Self-Attention module."""

import math
import pytest
import torch
import torch.nn.functional as F
from src.attention import Attention


def test_attention_basic_shape():
    """Verify [B, N, D] -> [B, N, D] transformation with B=2, N=256, D=384, H=6."""
    B, N, D, H = 2, 256, 384, 6
    attn = Attention(hidden_size=D, num_heads=H)

    x = torch.randn(B, N, D)
    out = attn(x)

    assert out.shape == (B, N, D), f"Expected shape {(B, N, D)}, but got {out.shape}"
    assert attn.head_dim == 64


def test_attention_exact_parameter_count():
    """Verify analytical parameter count: QKV (D * 3D + 3D) + Proj (D * D + D)."""
    D, H = 384, 6
    attn = Attention(hidden_size=D, num_heads=H, qkv_bias=True, out_bias=True)

    # QKV: 384 * 1152 + 1152 = 443,520
    # Proj: 384 * 384 + 384 = 147,840
    # Total = 591,360
    expected_qkv = (D * 3 * D) + (3 * D)
    expected_proj = (D * D) + D
    expected_total = expected_qkv + expected_proj

    actual_total = sum(p.numel() for p in attn.parameters())
    assert actual_total == expected_total == 591360, (
        f"Expected {expected_total} parameters, but found {actual_total}"
    )


def test_attention_finite_outputs_and_gradients():
    """Verify forward outputs and backward gradients are finite and non-zero."""
    B, N, D, H = 2, 64, 384, 6
    attn = Attention(hidden_size=D, num_heads=H)

    x = torch.randn(B, N, D, requires_grad=True)
    out = attn(x)

    assert torch.isfinite(out).all(), "Non-finite values detected in attention output."

    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all(), "Non-finite gradients in input x."

    for name, param in attn.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient."
        assert torch.isfinite(param.grad).all(), f"Parameter {name} has non-finite gradient."


def test_attention_invalid_divisibility_error():
    """Verify ValueError is raised if hidden_size is not divisible by num_heads."""
    with pytest.raises(ValueError, match="must be divisible by num_heads"):
        Attention(hidden_size=384, num_heads=7)


def test_attention_batch_size_variation():
    """Verify module operates correctly across multiple batch sizes."""
    attn = Attention(hidden_size=384, num_heads=6)
    attn.eval()

    with torch.no_grad():
        for B in [1, 2, 4, 8]:
            x = torch.randn(B, 128, 384)
            out = attn(x)
            assert out.shape == (B, 128, 384)


def test_attention_sequence_length_variation():
    """Verify module operates correctly across variable sequence lengths."""
    attn = Attention(hidden_size=384, num_heads=6)
    attn.eval()

    with torch.no_grad():
        for N in [16, 64, 256, 1024]:
            x = torch.randn(2, N, 384)
            out = attn(x)
            assert out.shape == (2, N, 384)


def test_attention_dtypes():
    """Verify attention operates across float16, bfloat16, float32, and float64."""
    for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
        attn = Attention(hidden_size=384, num_heads=6).to(dtype=dtype)
        x = torch.randn(2, 64, 384, dtype=dtype)
        out = attn(x)
        assert out.dtype == dtype, f"Expected {dtype}, got {out.dtype}"


def test_attention_deterministic_behavior():
    """Verify eval mode produces deterministic outputs on identical inputs."""
    attn = Attention(hidden_size=384, num_heads=6)
    attn.eval()

    x = torch.randn(2, 64, 384)
    with torch.no_grad():
        out1 = attn(x)
        out2 = attn(x)

    assert torch.equal(out1, out2), "Attention output is not deterministic."


def test_attention_numerical_reference():
    """Verify module output matches step-by-step unrolled mathematical attention."""
    B, N, D, H = 2, 16, 64, 4
    d_head = D // H  # 16
    attn = Attention(hidden_size=D, num_heads=H)
    attn.eval()

    x = torch.randn(B, N, D, dtype=torch.float64)
    attn = attn.to(dtype=torch.float64)

    with torch.no_grad():
        # Module forward pass
        out_module = attn(x)

        # Step-by-step manual reference computation:
        # 1. Linear QKV projection
        qkv = F.linear(x, attn.qkv.weight, attn.qkv.bias)  # [B, N, 3*D]
        qkv = qkv.reshape(B, N, 3, H, d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, N, d_head]

        # 2. Attention weights matrix: [B, H, N, N]
        scale = 1.0 / math.sqrt(d_head)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        weights = F.softmax(scores, dim=-1)

        # 3. Context aggregation: [B, H, N, d_head]
        context = torch.matmul(weights, v)

        # 4. Head concatenation & output projection: [B, N, D]
        context = context.transpose(1, 2).reshape(B, N, D)
        out_manual = F.linear(context, attn.proj.weight, attn.proj.bias)

        assert torch.allclose(out_module, out_manual, atol=1e-7), (
            f"Module output diverged from mathematical reference calculation!\n"
            f"Max diff: {(out_module - out_manual).abs().max().item()}"
        )
