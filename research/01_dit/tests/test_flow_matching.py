"""Unit tests for Frozen Baseline Flow Matching Objective."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.dit import DiT
from src.flow_matching import (
    FlowMatchingObjective,
    compute_interpolated_state,
    compute_target_velocity,
)


def test_flow_matching_endpoints_and_midpoint():
    """Verify t=0 gives noise, t=1 gives data, and t=0.5 gives exact midpoint."""
    B, C, H, W = 2, 4, 32, 32
    x_data = torch.randn(B, C, H, W)
    x_noise = torch.randn(B, C, H, W)

    # 1. t = 0 -> x_t = x_noise
    t_0 = torch.zeros(B)
    x_t0 = compute_interpolated_state(x_data, x_noise, t_0)
    assert torch.equal(x_t0, x_noise), "x_t at t=0 must equal x_noise exactly."

    # 2. t = 1 -> x_t = x_data
    t_1 = torch.ones(B)
    x_t1 = compute_interpolated_state(x_data, x_noise, t_1)
    assert torch.equal(x_t1, x_data), "x_t at t=1 must equal x_data exactly."

    # 3. t = 0.5 -> x_t = 0.5 * (x_noise + x_data)
    t_half = torch.full((B,), 0.5)
    x_thalf = compute_interpolated_state(x_data, x_noise, t_half)
    expected_midpoint = 0.5 * (x_noise + x_data)
    assert torch.allclose(x_thalf, expected_midpoint, atol=1e-6), "x_t at t=0.5 must equal midpoint."


def test_flow_matching_target_velocity():
    """Verify analytical target velocity formula v_target = x_data - x_noise."""
    B, C, H, W = 2, 4, 32, 32
    x_data = torch.randn(B, C, H, W)
    x_noise = torch.randn(B, C, H, W)

    v_target = compute_target_velocity(x_data, x_noise)
    assert torch.equal(v_target, x_data - x_noise), "v_target must be exactly x_data - x_noise."


def test_flow_matching_numerical_finite_difference():
    """Verify analytical velocity agrees with finite-difference numerical derivative dx/dt."""
    B, C, H, W = 2, 4, 32, 32
    x_data = torch.randn(B, C, H, W, dtype=torch.float64)
    x_noise = torch.randn(B, C, H, W, dtype=torch.float64)

    t_val = 0.35
    dt = 1e-5

    t1 = torch.full((B,), t_val, dtype=torch.float64)
    t2 = torch.full((B,), t_val + dt, dtype=torch.float64)

    x_t1 = compute_interpolated_state(x_data, x_noise, t1)
    x_t2 = compute_interpolated_state(x_data, x_noise, t2)

    numerical_velocity = (x_t2 - x_t1) / dt
    analytical_velocity = compute_target_velocity(x_data, x_noise)

    assert torch.allclose(numerical_velocity, analytical_velocity, atol=1e-5), (
        "Finite difference derivative does not match analytical straight-line velocity target."
    )


def test_flow_matching_objective_shapes_and_metrics():
    """Verify loss calculation, metric dictionary keys, and shapes."""
    B, C, H, W, D = 2, 4, 32, 32, 384
    model = DiT(in_channels=C, out_channels=C, latent_size=H, hidden_size=D, depth=2, num_heads=2)
    objective = FlowMatchingObjective()

    x_data = torch.randn(B, C, H, W)
    text_embed = torch.randn(B, D)

    loss, metrics = objective(model, x_data, text_embed=text_embed)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0, "Loss must be a scalar."
    assert torch.isfinite(loss), "Loss must be finite."

    assert metrics["v_pred"].shape == (B, C, H, W)
    assert metrics["v_target"].shape == (B, C, H, W)
    assert metrics["x_t"].shape == (B, C, H, W)
    assert metrics["t"].shape == (B,)


def test_flow_matching_zero_loss_when_prediction_matches_target():
    """Verify MSE loss evaluates to exactly 0.0 when predicted velocity equals target."""
    v_target = torch.randn(2, 4, 32, 32)
    v_pred = v_target.clone()

    loss = F.mse_loss(v_pred, v_target)
    assert loss.item() == 0.0, "MSE loss must be 0.0 when prediction matches target."


def test_flow_matching_deterministic_reproducibility():
    """Verify supplying fixed noise and t yields 100% reproducible deterministic loss."""
    B, C, H, W = 2, 4, 32, 32
    model = DiT(depth=2, hidden_size=128, num_heads=2)
    model.eval()
    objective = FlowMatchingObjective()

    x_data = torch.randn(B, C, H, W)
    fixed_noise = torch.randn(B, C, H, W)
    fixed_t = torch.tensor([0.25, 0.75])

    with torch.no_grad():
        loss1, metrics1 = objective(model, x_data, noise=fixed_noise, t=fixed_t)
        loss2, metrics2 = objective(model, x_data, noise=fixed_noise, t=fixed_t)

    assert torch.equal(loss1, loss2), "Deterministic inputs did not produce identical loss."
    assert torch.equal(metrics1["x_t"], metrics2["x_t"])
    assert torch.equal(metrics1["v_target"], metrics2["v_target"])


def test_flow_matching_text_embed_passthrough():
    """Verify pooled text conditioning propagates through objective into DiT."""
    model = DiT(depth=2, hidden_size=128, num_heads=2)
    # Open gates to detect text influence
    for block in model.blocks:
        nn.init.normal_(block.adaLN_modulation.linear.weight, std=0.02)
    nn.init.normal_(model.final_layer.linear.weight, std=0.02)

    objective = FlowMatchingObjective()

    x_data = torch.randn(2, 4, 32, 32)
    fixed_noise = torch.randn(2, 4, 32, 32)
    fixed_t = torch.tensor([0.4, 0.6])

    text1 = torch.randn(2, 128)
    text2 = torch.randn(2, 128)

    with torch.no_grad():
        loss1, m1 = objective(model, x_data, text_embed=text1, noise=fixed_noise, t=fixed_t)
        loss2, m2 = objective(model, x_data, text_embed=text2, noise=fixed_noise, t=fixed_t)

    assert not torch.allclose(m1["v_pred"], m2["v_pred"]), (
        "Different text embeddings should yield distinct velocity predictions."
    )


def test_flow_matching_dtypes():
    """Verify FlowMatchingObjective functions across float16, bfloat16, float32, and float64."""
    for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
        model = DiT(depth=2, hidden_size=128, num_heads=2).to(dtype=dtype)
        objective = FlowMatchingObjective()

        x_data = torch.randn(1, 4, 32, 32, dtype=dtype)
        loss, metrics = objective(model, x_data)

        assert loss.dtype == dtype, f"Expected loss dtype {dtype}, got {loss.dtype}"
        assert metrics["v_pred"].dtype == dtype
        assert metrics["v_target"].dtype == dtype
        assert metrics["x_t"].dtype == dtype


def test_flow_matching_invalid_shapes():
    """Verify ValueError is raised on shape, dimension, or batch mismatches."""
    model = DiT(depth=2, hidden_size=128, num_heads=2)
    objective = FlowMatchingObjective()

    # 1. 3D data instead of 4D
    with pytest.raises(ValueError, match="Expected 4D data tensor"):
        objective(model, torch.randn(2, 4, 32))

    # 2. Mismatched noise shape
    with pytest.raises(ValueError, match="Shape mismatch"):
        objective(model, torch.randn(2, 4, 32, 32), noise=torch.randn(2, 4, 16, 16))

    # 3. Mismatched timestep batch size
    with pytest.raises(ValueError, match="Batch size mismatch"):
        objective(model, torch.randn(2, 4, 32, 32), t=torch.tensor([0.1, 0.2, 0.3, 0.4]))


def test_flow_matching_backward_gradient_flow():
    """Verify backward loss backpropagates finite gradients into model parameters."""
    model = DiT(depth=2, hidden_size=128, num_heads=2)
    # Open gates to allow active gradient propagation
    for block in model.blocks:
        nn.init.normal_(block.adaLN_modulation.linear.weight, std=0.02)
    nn.init.normal_(model.final_layer.linear.weight, std=0.02)
    nn.init.normal_(model.final_layer.adaLN_modulation[-1].weight, std=0.02)

    objective = FlowMatchingObjective()

    x_data = torch.randn(2, 4, 32, 32, requires_grad=True)
    text_embed = torch.randn(2, 128, requires_grad=True)

    loss, _ = objective(model, x_data, text_embed=text_embed)
    loss.backward()

    assert torch.isfinite(loss)
    assert x_data.grad is not None and torch.isfinite(x_data.grad).all()
    assert text_embed.grad is not None and torch.isfinite(text_embed.grad).all()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient."
            assert torch.isfinite(param.grad).all(), f"Parameter {name} has non-finite gradient."
