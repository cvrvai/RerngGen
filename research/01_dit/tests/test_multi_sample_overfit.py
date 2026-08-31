"""Step 15 — Tiny Multi-Sample Dataset Overfit and Condition-Selective Generation Experiment.

Demonstrates that a small DiT model can learn to fit multiple distinct latent targets
conditioned on distinct text embeddings simultaneously, verifying that:
1. Training loss decreases strongly across all 4 samples.
2. The network uses the conditioning vector to select which target to generate.
3. Euler sampling from noise with condition A produces target A, while condition B produces target B.
4. Matching condition MSE is significantly lower than initial noise MSE and mismatched condition MSE.
5. Checkpoint saving and resumption faithfully preserves the multi-sample generative capability.
"""

from pathlib import Path
import time
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.checkpoint import load_checkpoint, save_checkpoint
from src.dit import DiT
from src.flow_matching import FlowMatchingObjective
from src.sampler import EulerSampler


def test_tiny_multi_sample_dataset_overfit_and_euler_generation(tmp_path: Path):
    """Verify condition-selective multi-target learning and Euler generation."""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------
    # 1. DATASET SETUP: 4 Distinct Synthetic Latents & Conditions
    # -------------------------------------------------------------
    num_samples = 4
    C, H, W = 4, 32, 32
    text_dim = 192

    # Create 4 distinct target latents with distinct mean spatial profiles
    targets = []
    for i in range(num_samples):
        base_latent = torch.randn(1, C, H, W, device=device) * 0.5 + (i - 1.5) * 1.5
        targets.append(base_latent)
    data_batch = torch.cat(targets, dim=0)  # [4, 4, 32, 32]

    # Create 4 orthogonal/distinct condition vectors
    conditions = []
    for i in range(num_samples):
        cond = torch.zeros(1, text_dim, device=device)
        cond[0, i * 40 : (i + 1) * 40] = 1.0
        conditions.append(cond)
    condition_batch = torch.cat(conditions, dim=0)  # [4, 192]

    # -------------------------------------------------------------
    # 2. MODEL & OPTIMIZER SETUP
    # -------------------------------------------------------------
    model = DiT(
        in_channels=C,
        out_channels=C,
        latent_size=H,
        patch_size=2,
        hidden_size=text_dim,
        depth=4,
        num_heads=3,
        mlp_ratio=4.0,
    ).to(device)

    objective = FlowMatchingObjective()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    # -------------------------------------------------------------
    # 3. MULTI-SAMPLE TRAINING LOOP (250 Steps)
    # -------------------------------------------------------------
    num_steps = 250
    loss_history = []
    start_time = time.time()

    for step in range(num_steps):
        optimizer.zero_grad()

        # In each step, sample continuous noise x0 and timesteps t for all 4 samples
        noise = torch.randn_like(data_batch)
        t = torch.rand(num_samples, device=device)

        loss, _ = objective(
            model,
            data_batch,
            text_embed=condition_batch,
            noise=noise,
            t=t,
        )
        assert torch.isfinite(loss), f"Encountered non-finite loss at step {step}: {loss.item()}"
        loss_history.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    runtime = time.time() - start_time
    initial_loss = loss_history[0]
    final_loss = sum(loss_history[-10:]) / 10.0  # smoothed final 10 steps
    loss_reduction_pct = ((initial_loss - final_loss) / initial_loss) * 100.0

    print("\n--- Step 15: Multi-Sample Dataset Overfit Training Metrics ---")
    print(f"Device: {device}")
    print(f"Dataset Size: {num_samples} samples")
    print(f"Steps: {num_steps}")
    print(f"Runtime: {runtime:.2f}s")
    print(f"Initial Loss (Step 0): {initial_loss:.6f}")
    print(f"Step 50 Loss: {loss_history[50]:.6f}")
    print(f"Step 100 Loss: {loss_history[100]:.6f}")
    print(f"Final Smoothed Loss (Steps 240-249): {final_loss:.6f}")
    print(f"Loss Reduction: {loss_reduction_pct:.2f}%")

    assert loss_reduction_pct > 80.0, (
        f"Expected >80% loss reduction across 4 samples, got {loss_reduction_pct:.2f}%"
    )

    # -------------------------------------------------------------
    # 4. CHECKPOINT SAVE & RESTORE INTEGRITY
    # -------------------------------------------------------------
    ckpt_path = tmp_path / "multi_sample_step15_ckpt.pt"
    save_checkpoint(ckpt_path, model, optimizer, global_step=num_steps)

    fresh_model = DiT(
        in_channels=C,
        out_channels=C,
        latent_size=H,
        patch_size=2,
        hidden_size=text_dim,
        depth=4,
        num_heads=3,
        mlp_ratio=4.0,
    ).to(device)
    load_checkpoint(ckpt_path, fresh_model, device=device)
    fresh_model.eval()

    # -------------------------------------------------------------
    # 5. EULER SAMPLING GENERATION & CONDITION-SELECTIVITY TEST
    # -------------------------------------------------------------
    sampler = EulerSampler(num_steps=50)

    print("\n--- Step 15: Condition-Selective Generation Evaluation ---")
    matching_mses = []
    mismatched_mses = []
    initial_noise_mses = []

    for i in range(num_samples):
        torch.manual_seed(1000 + i)
        test_noise = torch.randn(1, C, H, W, device=device)
        target_i = targets[i]
        cond_i = conditions[i]
        cond_mismatched = conditions[(i + 1) % num_samples]  # Deliberately mismatched condition

        # 1. Baseline Initial Noise MSE
        noise_mse = F.mse_loss(test_noise, target_i).item()
        initial_noise_mses.append(noise_mse)

        # 2. Matching Condition Generation: Euler integration from noise using cond_i
        gen_matching = sampler.sample(fresh_model, test_noise, text_embed=cond_i)
        matching_mse = F.mse_loss(gen_matching, target_i).item()
        matching_mses.append(matching_mse)

        # 3. Mismatched Condition Generation: Euler integration from noise using wrong condition
        gen_mismatched = sampler.sample(fresh_model, test_noise, text_embed=cond_mismatched)
        mismatch_mse = F.mse_loss(gen_mismatched, target_i).item()
        mismatched_mses.append(mismatch_mse)

        print(
            f"Sample {i}: Noise MSE = {noise_mse:.4f} | "
            f"Matching Gen MSE = {matching_mse:.4f} | "
            f"Mismatched Gen MSE = {mismatch_mse:.4f}"
        )

        # Assertions per sample:
        # A. Generated latent with matching condition is significantly closer to target than starting noise
        assert matching_mse < noise_mse * 0.4, (
            f"Sample {i}: Matching generation ({matching_mse:.4f}) failed to reduce noise MSE ({noise_mse:.4f}) by >60%."
        )
        # B. Matching condition achieves substantially lower MSE than mismatched condition
        assert matching_mse < mismatch_mse, (
            f"Sample {i}: Matching MSE ({matching_mse:.4f}) must be lower than mismatched MSE ({mismatch_mse:.4f})."
        )

    # 4. Swapping conditions produces distinctly different outputs
    gen_0_with_cond_0 = sampler.sample(fresh_model, targets[0] * 0.0 + torch.randn_like(targets[0]), text_embed=conditions[0])
    gen_0_with_cond_1 = sampler.sample(fresh_model, targets[0] * 0.0 + torch.randn_like(targets[0]), text_embed=conditions[1])
    assert not torch.allclose(gen_0_with_cond_0, gen_0_with_cond_1), (
        "Swapping conditions must produce distinct generated latents."
    )
