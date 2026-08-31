"""Unit tests for Timestep Embedding module."""

import pytest
import torch
from src.timestep_embedding import TimestepEmbedder, sinusoidal_timestep_embedding


def test_timestep_embedding_shape():
    """Verify [B] -> [B, D] mapping with B=4, D=384, frequency_dim=256."""
    B, D, freq_dim = 4, 384, 256
    embedder = TimestepEmbedder(hidden_size=D, frequency_embedding_size=freq_dim)

    t = torch.rand(B)  # Continuous timesteps in [0, 1]
    out = embedder(t)

    assert out.shape == (B, D), f"Expected shape {(B, D)}, but got {out.shape}"


def test_timestep_embedding_2d_input():
    """Verify shape [B, 1] is handled identically to [B]."""
    embedder = TimestepEmbedder(hidden_size=384)

    t_1d = torch.tensor([0.2, 0.5, 0.8])
    t_2d = t_1d.unsqueeze(1)

    out_1d = embedder(t_1d)
    out_2d = embedder(t_2d)

    assert torch.allclose(out_1d, out_2d, atol=1e-6)


def test_timestep_embedding_parameter_count():
    """Verify analytical parameter count: (freq*D + D) + (D*D + D)."""
    freq_dim, D = 256, 384
    embedder = TimestepEmbedder(hidden_size=D, frequency_embedding_size=freq_dim)

    # Linear 1: 256 * 384 + 384 = 98,688
    # Linear 2: 384 * 384 + 384 = 147,840
    # Total = 246,528
    expected_params = (freq_dim * D + D) + (D * D + D)
    actual_params = sum(p.numel() for p in embedder.parameters())

    assert actual_params == expected_params, (
        f"Expected {expected_params} parameters, but found {actual_params}"
    )


def test_timestep_embedding_smoothness_and_uniqueness():
    """Verify continuous smooth transition across time t in [0, 1]."""
    embedder = TimestepEmbedder(hidden_size=384)
    embedder.eval()

    with torch.no_grad():
        t_near_1 = torch.tensor([0.50])
        t_near_2 = torch.tensor([0.51])
        t_far = torch.tensor([0.05])

        emb_near_1 = embedder(t_near_1)
        emb_near_2 = embedder(t_near_2)
        emb_far = embedder(t_far)

        # Normalize to compute cosine similarities
        sim_near = torch.cosine_similarity(emb_near_1, emb_near_2).item()
        sim_far = torch.cosine_similarity(emb_near_1, emb_far).item()

        # Close timesteps must have higher similarity than distant timesteps
        assert sim_near > sim_far, (
            f"Expected near similarity ({sim_near:.4f}) > far similarity ({sim_far:.4f})"
        )
        assert sim_near > 0.95, f"Expected close timesteps to have high similarity, got {sim_near:.4f}"


def test_timestep_embedding_gradient_flow():
    """Verify backward gradient flow computes finite gradients for all MLP parameters."""
    embedder = TimestepEmbedder(hidden_size=384)
    t = torch.tensor([0.1, 0.4, 0.7, 0.9], requires_grad=True)

    out = embedder(t)
    loss = out.sum()
    loss.backward()

    assert t.grad is not None
    assert torch.isfinite(t.grad).all()

    for name, param in embedder.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient."
        assert torch.isfinite(param.grad).all(), f"Parameter {name} has non-finite gradient."


def test_timestep_embedding_dtypes():
    """Verify module functions with float16, bfloat16, float32, and float64."""
    for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
        embedder = TimestepEmbedder(hidden_size=384).to(dtype=dtype)
        t = torch.tensor([0.25, 0.75])
        out = embedder(t)
        assert out.dtype == dtype, f"Expected output dtype {dtype}, got {out.dtype}"


def test_timestep_embedding_invalid_shape_error():
    """Verify ValueError is raised if passing 3D tensor [B, N, D] instead of 1D scalar."""
    embedder = TimestepEmbedder(hidden_size=384)
    t_invalid = torch.randn(2, 4, 384)

    with pytest.raises(ValueError, match="Expected timesteps tensor"):
        embedder(t_invalid)
