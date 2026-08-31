"""Unit tests for 2D Sinusoidal Positional Embedding module."""

import pytest
import torch
import numpy as np
from src.positional_embedding import (
    PositionalEmbedding2D,
    get_2d_sincos_pos_embed,
    get_1d_sincos_pos_embed_from_grid,
)


def test_pos_embed_shape():
    """Verify positional embedding buffer matches [1, N, D] where N = h * w."""
    grid_size = (16, 16)
    embed_dim = 384
    pos_embed_mod = PositionalEmbedding2D(embed_dim=embed_dim, grid_size=grid_size)

    assert pos_embed_mod.pos_embed.shape == (1, 256, 384)
    assert pos_embed_mod.num_patches == 256


def test_pos_embed_no_trainable_parameters():
    """Verify fixed 2D sinusoidal embeddings contain 0 learnable parameters."""
    pos_embed_mod = PositionalEmbedding2D(embed_dim=384, grid_size=(16, 16))
    learnable_params = sum(p.numel() for p in pos_embed_mod.parameters() if p.requires_grad)
    assert learnable_params == 0, f"Expected 0 learnable parameters, found {learnable_params}."


def test_pos_embed_coordinate_uniqueness():
    """Verify that every 2D patch coordinate (y, x) gets a unique positional embedding."""
    pos_embed = get_2d_sincos_pos_embed(embed_dim=384, grid_size=(16, 16))
    pos_tensor = torch.from_numpy(pos_embed).float()  # [256, 384]

    # Normalize vectors and compute pairwise cosine similarity matrix [256, 256]
    norm_pos = pos_tensor / pos_tensor.norm(dim=-1, keepdim=True)
    sim_matrix = torch.matmul(norm_pos, norm_pos.T)

    # Diagonal elements must be 1.0 (self-similarity)
    diag = torch.diagonal(sim_matrix)
    assert torch.allclose(diag, torch.ones_like(diag), atol=1e-5)

    # Mask diagonal and verify no distinct positions share identical embeddings
    off_diag = sim_matrix - torch.eye(sim_matrix.shape[0])
    max_off_diag = off_diag.max().item()
    assert max_off_diag < 0.999, (
        f"Positional embeddings are not sufficiently unique: max off-diagonal similarity is {max_off_diag}"
    )


def test_pos_embed_spatial_symmetry():
    """Verify Y-axis and X-axis embeddings follow identical sinusoidal formulas on symmetric coordinates."""
    pos_embed = get_2d_sincos_pos_embed(embed_dim=384, grid_size=(16, 16))
    pos_grid = pos_embed.reshape(16, 16, 384)

    # Patch (y=3, x=7) vs Patch (y=7, x=3)
    # The Y half of (3, 7) should match the X half of (7, 3)
    emb_3_7 = pos_grid[3, 7]
    emb_7_3 = pos_grid[7, 3]

    y_half_3_7 = emb_3_7[:192]
    x_half_7_3 = emb_7_3[192:]

    assert np.allclose(y_half_3_7, x_half_7_3), (
        "Y-coordinate features and X-coordinate features did not match across symmetric coordinates."
    )


def test_pos_embed_forward_broadcast_add():
    """Verify [B, N, D] + [1, N, D] broadcast addition and gradient passthrough."""
    B, N, D = 2, 256, 384
    pos_embed_mod = PositionalEmbedding2D(embed_dim=D, grid_size=(16, 16))

    x = torch.randn(B, N, D, requires_grad=True)
    out = pos_embed_mod(x)

    assert out.shape == (B, N, D)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    # Gradient of (x + pos_embed) w.r.t x is 1.0
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_pos_embed_rectangular_grid():
    """Verify 2D sinusoidal embeddings work on non-square grids (e.g. 8x16)."""
    grid_size = (8, 16)
    embed_dim = 384
    pos_embed_mod = PositionalEmbedding2D(embed_dim=embed_dim, grid_size=grid_size)

    assert pos_embed_mod.pos_embed.shape == (1, 128, 384)
    assert pos_embed_mod.num_patches == 128


def test_pos_embed_dimension_divisibility_error():
    """Verify ValueError is raised if embed_dim is not divisible by 4."""
    with pytest.raises(ValueError, match="divisible by 4"):
        get_2d_sincos_pos_embed(embed_dim=382, grid_size=(16, 16))


def test_pos_embed_sequence_length_mismatch_error():
    """Verify ValueError is raised if input sequence length does not match grid size."""
    pos_embed_mod = PositionalEmbedding2D(embed_dim=384, grid_size=(16, 16))
    x_invalid = torch.randn(2, 100, 384)

    with pytest.raises(ValueError, match="Sequence length mismatch"):
        pos_embed_mod(x_invalid)


def test_pos_embed_dtype_and_device_casting():
    """Verify positional embedding safely matches input tensor dtype (float16, bfloat16, float64)."""
    pos_embed_mod = PositionalEmbedding2D(embed_dim=384, grid_size=(16, 16))

    for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
        x = torch.randn(2, 256, 384, dtype=dtype)
        out = pos_embed_mod(x)
        assert out.dtype == dtype, f"Expected output dtype {dtype}, got {out.dtype}"

