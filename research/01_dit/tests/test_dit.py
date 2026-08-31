"""Unit tests for complete Diffusion Transformer (DiT) model."""

from pathlib import Path
import pytest
import torch
import torch.nn as nn
from src.block import DiTBlock
from src.dit import DiT


def test_dit_complete_forward_shape():
    """Verify [B, 4, 32, 32] + [B] -> [B, 4, 32, 32] end-to-end tensor pipeline."""
    B, C, H, W = 2, 4, 32, 32
    model = DiT(
        in_channels=C,
        out_channels=C,
        latent_size=H,
        patch_size=2,
        hidden_size=384,
        depth=8,
        num_heads=6,
        mlp_ratio=4.0,
    )

    x = torch.randn(B, C, H, W)
    t = torch.tensor([0.2, 0.8])
    v_pred = model(x, t)

    assert v_pred.shape == (B, C, H, W), f"Expected shape {(B, C, H, W)}, but got {v_pred.shape}"


def test_dit_exact_model_parameter_count():
    """Verify exact total parameter count across all model components.

    Breakdown:
        - PatchEmbed: 6,528
        - PosEmbed: 0
        - TimestepEmbedder: 246,528
        - 8 x DiTBlock: 8 * 2,659,968 = 21,279,744
        - FinalLayer: 301,840
        - Unpatchify: 0
        Total = 21,834,640
    """
    model = DiT(
        in_channels=4,
        out_channels=4,
        latent_size=32,
        patch_size=2,
        hidden_size=384,
        depth=8,
        num_heads=6,
        mlp_ratio=4.0,
    )

    # 1. PatchEmbed params: (2*2*4*384) + 384 = 6,528
    patch_embed_params = sum(p.numel() for p in model.x_embed.parameters())
    assert patch_embed_params == 6528

    # 2. TimestepEmbedder params: (256*384+384) + (384*384+384) = 246,528
    timestep_params = sum(p.numel() for p in model.t_embedder.parameters())
    assert timestep_params == 246528

    # 3. 8 x DiTBlock params: 8 * 2,659,968 = 21,279,744
    blocks_params = sum(p.numel() for p in model.blocks.parameters())
    assert blocks_params == 21279744

    # 4. FinalLayer params: (384*768+768) + (384*16+16) = 301,840
    final_layer_params = sum(p.numel() for p in model.final_layer.parameters())
    assert final_layer_params == 301840

    # 5. Total model parameters
    total_params = sum(p.numel() for p in model.parameters())
    expected_total = 6528 + 246528 + 21279744 + 301840  # 21,834,640
    assert total_params == expected_total == 21834640, (
        f"Expected {expected_total} parameters, but got {total_params}"
    )


def test_dit_strict_zero_initialization():
    """Verify that at step 0, the complete DiT model outputs identically zero velocity fields."""
    B, C, H, W = 2, 4, 32, 32
    model = DiT(
        in_channels=C,
        out_channels=C,
        latent_size=H,
        patch_size=2,
        hidden_size=384,
        depth=8,
        num_heads=6,
    )

    x = torch.randn(B, C, H, W)
    t = torch.tensor([0.1, 0.9])
    v_pred = model(x, t)

    assert torch.equal(v_pred, torch.zeros_like(v_pred)), (
        f"Model output is not exactly 0.0 at initialization! Max abs: {v_pred.abs().max().item()}"
    )


def test_dit_blocks_count_and_types():
    """Verify that model contains exactly depth=8 DiTBlock modules."""
    model = DiT(depth=8)
    assert len(model.blocks) == 8
    for idx, block in enumerate(model.blocks):
        assert isinstance(block, DiTBlock), f"Block {idx} is not an instance of DiTBlock."


def test_dit_condition_sensitivity_after_opening_gates():
    """Verify that once weights are perturbed from zero, different timesteps produce distinct outputs."""
    model = DiT(depth=8)

    # Perturb zero-initialized output and modulation weights
    for block in model.blocks:
        nn.init.normal_(block.adaLN_modulation.linear.weight, std=0.02)
    nn.init.normal_(model.final_layer.linear.weight, std=0.02)
    nn.init.normal_(model.final_layer.adaLN_modulation[-1].weight, std=0.02)

    x = torch.randn(1, 4, 32, 32)
    t1 = torch.tensor([0.1])
    t2 = torch.tensor([0.9])

    with torch.no_grad():
        v1 = model(x, t1)
        v2 = model(x, t2)

    assert not torch.allclose(v1, v2, atol=1e-4), (
        "Model outputs with t=0.1 and t=0.9 are unexpectedly identical after opening gates."
    )


def test_dit_batch_size_variations():
    """Verify DiT executes across variable batch sizes B in {1, 2, 4}."""
    model = DiT(depth=4)
    model.eval()

    with torch.no_grad():
        for B in [1, 2, 4]:
            x = torch.randn(B, 4, 32, 32)
            t = torch.linspace(0.0, 1.0, steps=B)
            out = model(x, t)
            assert out.shape == (B, 4, 32, 32)


def test_dit_invalid_input_shapes():
    """Verify ValueError is raised on invalid latent channel or spatial resolution."""
    model = DiT(in_channels=4, latent_size=32)

    # Invalid channels (3 instead of 4)
    with pytest.raises(ValueError, match="Expected input channels"):
        model(torch.randn(2, 3, 32, 32), torch.tensor([0.5, 0.5]))

    # Invalid spatial resolution (16 instead of 32)
    with pytest.raises(ValueError, match="Expected spatial resolution"):
        model(torch.randn(2, 4, 16, 16), torch.tensor([0.5, 0.5]))


def test_dit_full_backward_gradient_flow_after_update():
    """Verify finite gradients flow throughout all components in the complete DiT model."""
    model = DiT(depth=2, hidden_size=128, num_heads=2)

    # Activate weights
    for block in model.blocks:
        nn.init.normal_(block.adaLN_modulation.linear.weight, std=0.02)
    nn.init.normal_(model.final_layer.linear.weight, std=0.02)
    nn.init.normal_(model.final_layer.adaLN_modulation[-1].weight, std=0.02)

    x = torch.randn(2, 4, 32, 32, requires_grad=True)
    t = torch.tensor([0.25, 0.75])

    out = model(x, t)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient."
            assert torch.isfinite(param.grad).all(), f"Parameter {name} has non-finite gradient."


def test_dit_dtypes():
    """Verify DiT executes across float16, bfloat16, float32, and float64."""
    for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
        model = DiT(depth=2, hidden_size=128, num_heads=2).to(dtype=dtype)
        x = torch.randn(1, 4, 32, 32, dtype=dtype)
        t = torch.tensor([0.5], dtype=dtype)
        out = model(x, t)
        assert out.dtype == dtype, f"Expected {dtype}, got {out.dtype}"


def test_dit_from_config_yaml():
    """Verify instantiation of DiT from configs/debug.yaml."""
    config_path = Path("research/01_dit/configs/debug.yaml")
    assert config_path.exists(), "configs/debug.yaml not found."

    model = DiT.from_config(config_path)

    assert model.in_channels == 4
    assert model.out_channels == 4
    assert model.latent_size == 32
    assert model.patch_size == 2
    assert model.hidden_size == 384
    assert model.depth == 8
    assert model.num_heads == 6
    assert model.mlp_ratio == 4.0
    assert model.time_scale == 1000.0
    assert sum(p.numel() for p in model.parameters()) == 21834640
