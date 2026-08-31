"""Training Smoke Test and Single-Batch Overfitting Experiments.

Demonstrates that the DiT architecture paired with the Flow Matching objective
can actively learn from gradient descent, reducing loss substantially towards zero
on a fixed synthetic batch.
"""

import time
import pytest
import torch
import torch.nn as nn
from src.dit import DiT
from src.flow_matching import FlowMatchingObjective


def test_training_smoke_multistep():
    """Verify that multiple optimizer steps execute with finite loss, finite gradients, and parameter updates."""
    torch.manual_seed(42)
    model = DiT(depth=2, hidden_size=128, num_heads=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    objective = FlowMatchingObjective()

    initial_param_clones = [p.clone().detach() for p in model.parameters()]

    losses = []
    for step in range(5):
        optimizer.zero_grad()
        x_data = torch.randn(2, 4, 32, 32)
        text_embed = torch.randn(2, 128)

        loss, _ = objective(model, x_data, text_embed=text_embed)
        assert torch.isfinite(loss), f"Step {step} produced non-finite loss: {loss.item()}"
        losses.append(loss.item())

        loss.backward()

        # Verify all gradients are finite
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name} at step {step}."

        optimizer.step()

    # Verify that model parameters were updated and moved away from initialization
    has_changed = any(
        not torch.equal(p, p_init) for p, p_init in zip(model.parameters(), initial_param_clones)
    )
    assert has_changed, "Model parameters did not change after 5 optimizer steps."


def test_intentional_fixed_batch_overfit():
    """Verify that DiT can completely overfit a fixed synthetic batch with >90% loss reduction.

    Experimental Protocol:
        - Fixed seed: 42
        - Fixed batch: B=2, C=4, H=32, W=32
        - Fixed noise: sampled once from N(0, I)
        - Fixed timesteps: t = [0.25, 0.75]
        - Fixed text embeddings: [2, 192]
        - Optimizer: AdamW(lr=2e-3, weight_decay=0.0)
        - Target: Loss must reduce by >90% within 100 optimization steps.
    """
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate model
    model = DiT(
        in_channels=4,
        out_channels=4,
        latent_size=32,
        patch_size=2,
        hidden_size=192,
        depth=4,
        num_heads=3,
        mlp_ratio=4.0,
    ).to(device)

    objective = FlowMatchingObjective()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.0)

    # Freeze single synthetic batch
    x_data = torch.randn(2, 4, 32, 32, device=device)
    fixed_noise = torch.randn(2, 4, 32, 32, device=device)
    fixed_t = torch.tensor([0.25, 0.75], device=device)
    fixed_text = torch.randn(2, 192, device=device)

    loss_history = []
    grad_norm_history = []

    start_time = time.time()
    num_steps = 100

    for step in range(num_steps):
        optimizer.zero_grad()
        loss, metrics = objective(
            model,
            x_data,
            text_embed=fixed_text,
            noise=fixed_noise,
            t=fixed_t,
        )
        loss_val = loss.item()
        loss_history.append(loss_val)

        loss.backward()

        # Measure gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        grad_norm_history.append(total_norm)

        # Gradient clipping for training stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    runtime = time.time() - start_time
    initial_loss = loss_history[0]
    final_loss = loss_history[-1]
    loss_reduction_pct = ((initial_loss - final_loss) / initial_loss) * 100.0

    print("\n--- Fixed-Batch Overfit Metrics ---")
    print(f"Device: {device}")
    print(f"Steps: {num_steps}")
    print(f"Runtime: {runtime:.2f}s")
    print(f"Initial Loss (Step 0): {initial_loss:.6f}")
    print(f"Step 20 Loss: {loss_history[20]:.6f}")
    print(f"Step 50 Loss: {loss_history[50]:.6f}")
    print(f"Final Loss (Step {num_steps-1}): {final_loss:.6f}")
    print(f"Loss Reduction: {loss_reduction_pct:.2f}%")
    print(f"Initial Grad Norm: {grad_norm_history[0]:.4f}")
    print(f"Final Grad Norm: {grad_norm_history[-1]:.4f}")

    # Assertions
    assert initial_loss > 0.5, f"Expected non-trivial initial loss, got {initial_loss}"
    assert final_loss < 0.05, f"Expected final loss < 0.05, but got {final_loss}"
    assert loss_reduction_pct > 90.0, (
        f"Expected >90% loss reduction, but achieved {loss_reduction_pct:.2f}%"
    )
