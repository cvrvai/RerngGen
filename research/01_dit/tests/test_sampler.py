"""Unit tests for Deterministic Euler ODE Sampler."""

from typing import List, Optional
import pytest
import torch
import torch.nn as nn
from src.dit import DiT
from src.sampler import EulerSampler


class ZeroVelocityModel(nn.Module):
    """Mock model that returns exactly zero velocity."""
    def forward(self, x: torch.Tensor, t: torch.Tensor, text_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        return torch.zeros_like(x)


class ConstantVelocityModel(nn.Module):
    """Mock model that returns a constant velocity c."""
    def __init__(self, c: torch.Tensor) -> None:
        super().__init__()
        self.c = c

    def forward(self, x: torch.Tensor, t: torch.Tensor, text_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.c.expand_as(x)


class MockOracleVelocityModel(nn.Module):
    """Mock model that returns exact straight-line velocity target v = x1 - x0."""
    def __init__(self, x0: torch.Tensor, x1: torch.Tensor) -> None:
        super().__init__()
        self.v = x1 - x0

    def forward(self, x: torch.Tensor, t: torch.Tensor, text_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.v


class TrackerModel(nn.Module):
    """Mock model that logs all timesteps, call counts, and text embeddings."""
    def __init__(self) -> None:
        super().__init__()
        self.recorded_timesteps: List[float] = []
        self.recorded_text_embeds: List[Optional[torch.Tensor]] = []
        self.call_count = 0

    def forward(self, x: torch.Tensor, t: torch.Tensor, text_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.recorded_timesteps.append(t[0].item())
        self.recorded_text_embeds.append(text_embed)
        self.call_count += 1
        return torch.zeros_like(x)


def test_sampler_zero_velocity_field():
    """Verify that a zero vector field leaves initial noise exactly unchanged."""
    B, C, H, W = 2, 4, 32, 32
    noise = torch.randn(B, C, H, W)
    model = ZeroVelocityModel()
    sampler = EulerSampler(num_steps=20)

    out = sampler.sample(model, noise)
    assert torch.equal(out, noise), "Zero velocity field must preserve initial state exactly."


def test_sampler_constant_velocity_step_invariance():
    """Verify that constant velocity v(x,t)=c produces x_final = x0 + c for any step count."""
    B, C, H, W = 2, 4, 32, 32
    noise = torch.randn(B, C, H, W)
    c = torch.randn(1, C, H, W)
    model = ConstantVelocityModel(c)

    expected_final = noise + c.expand_as(noise)

    for num_steps in [1, 5, 20, 50, 100]:
        sampler = EulerSampler(num_steps=num_steps)
        out = sampler.sample(model, noise)
        assert torch.allclose(out, expected_final, atol=1e-5), (
            f"Constant velocity integration failed for num_steps={num_steps}."
        )


def test_sampler_oracle_path_exact_reconstruction():
    """Verify oracle straight-line velocity reaches target x1 exactly."""
    B, C, H, W = 2, 4, 32, 32
    x0 = torch.randn(B, C, H, W)
    x1 = torch.randn(B, C, H, W)
    model = MockOracleVelocityModel(x0, x1)

    sampler = EulerSampler(num_steps=10)
    out, trajectory = sampler.sample(model, x0, return_trajectory=True)

    assert len(trajectory) == 11, "Trajectory must contain N+1 states."
    assert torch.equal(trajectory[0], x0), "Trajectory start must be x0."
    assert torch.allclose(out, x1, atol=1e-5), "Oracle integration did not arrive at exact target x1."
    assert torch.allclose(trajectory[-1], x1, atol=1e-5)


def test_sampler_time_schedule_and_call_count():
    """Verify model receives exact time schedule [0, 1/N, ..., (N-1)/N] and is called N times."""
    num_steps = 10
    sampler = EulerSampler(num_steps=num_steps)
    tracker = TrackerModel()
    noise = torch.randn(2, 4, 32, 32)
    text_embed = torch.randn(2, 128)

    sampler.sample(tracker, noise, text_embed=text_embed)

    assert tracker.call_count == num_steps, f"Expected {num_steps} calls, got {tracker.call_count}."
    expected_timesteps = [i / float(num_steps) for i in range(num_steps)]

    for actual_t, exp_t in zip(tracker.recorded_timesteps, expected_timesteps):
        assert abs(actual_t - exp_t) < 1e-6, f"Time schedule mismatch: {actual_t} vs {exp_t}."

    for passed_text in tracker.recorded_text_embeds:
        assert torch.equal(passed_text, text_embed), "text_embed was not passed identically to step."


def test_sampler_batch_dtype_and_device_preservation():
    """Verify sampler preserves batch size, spatial shape, and dtypes."""
    for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
        noise = torch.randn(3, 4, 16, 16, dtype=dtype)
        model = ZeroVelocityModel()
        sampler = EulerSampler(num_steps=5)

        out = sampler.sample(model, noise)
        assert out.shape == (3, 4, 16, 16)
        assert out.dtype == dtype


def test_sampler_invalid_parameters():
    """Verify clear error handling for invalid step counts and shapes."""
    sampler = EulerSampler(num_steps=10)
    model = ZeroVelocityModel()

    # 1. Invalid step count in init
    with pytest.raises(ValueError, match="num_steps must be a positive integer"):
        EulerSampler(num_steps=0)
    with pytest.raises(ValueError, match="num_steps must be a positive integer"):
        EulerSampler(num_steps=-5)

    # 2. Invalid step count in sample()
    with pytest.raises(ValueError, match="num_steps must be a positive integer"):
        sampler.sample(model, torch.randn(2, 4, 32, 32), num_steps=0)

    # 3. Invalid noise dimension
    with pytest.raises(ValueError, match="Expected 4D noise latent"):
        sampler.sample(model, torch.randn(2, 4, 32))

    # 4. Text batch mismatch
    with pytest.raises(ValueError, match="Batch size mismatch"):
        sampler.sample(model, torch.randn(2, 4, 32, 32), text_embed=torch.randn(4, 128))


def test_sampler_deterministic_reproducibility():
    """Verify that identical initial noise produces identical sampled outputs."""
    model = DiT(depth=2, hidden_size=128, num_heads=2)
    # Open weights
    for block in model.blocks:
        nn.init.normal_(block.adaLN_modulation.linear.weight, std=0.02)
    nn.init.normal_(model.final_layer.linear.weight, std=0.02)

    sampler = EulerSampler(num_steps=10)
    noise = torch.randn(2, 4, 32, 32)
    text = torch.randn(2, 128)

    out1 = sampler.sample(model, noise, text_embed=text)
    out2 = sampler.sample(model, noise, text_embed=text)

    assert torch.equal(out1, out2), "Sampling with fixed noise must be 100% deterministic."


def test_sampler_real_untrained_dit_zero_init_behavior():
    """Verify that running the real untrained zero-initialized DiT leaves initial noise unchanged.

    This demonstrates the mathematical property of strict zero initialization at inference,
    confirming initialization sanity rather than generative quality.
    """
    model = DiT(depth=4, hidden_size=192, num_heads=3)
    model.eval()

    sampler = EulerSampler(num_steps=20)
    noise = torch.randn(2, 4, 32, 32)

    out = sampler.sample(model, noise)
    assert torch.equal(out, noise), (
        "Real untrained zero-initialized DiT must output zero velocity, leaving initial noise unchanged."
    )
