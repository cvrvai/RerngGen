"""Unit tests for Checkpoint Save and Resume Integrity."""

import os
from pathlib import Path
import random
import pytest
import torch
import torch.nn as nn
from src.checkpoint import load_checkpoint, save_checkpoint
from src.dit import DiT
from src.flow_matching import FlowMatchingObjective


def test_checkpoint_model_and_optimizer_exact_restoration(tmp_path: Path):
    """Verify that model parameters and optimizer state are bitwise identical after save and load."""
    ckpt_file = tmp_path / "model_ckpt.pt"

    # Create model and optimizer with non-zero state
    model_orig = DiT(depth=2, hidden_size=128, num_heads=2)
    optimizer_orig = torch.optim.AdamW(model_orig.parameters(), lr=1e-3)

    # Perform a dummy training step to populate optimizer state (exp_avg, exp_avg_sq)
    x = torch.randn(2, 4, 32, 32)
    t = torch.tensor([0.3, 0.7])
    out = model_orig(x, t)
    loss = out.sum()
    loss.backward()
    optimizer_orig.step()

    # Save checkpoint
    saved_path = save_checkpoint(
        path=ckpt_file,
        model=model_orig,
        optimizer=optimizer_orig,
        global_step=42,
        config={"hidden_size": 128, "depth": 2},
        extra_state={"best_loss": 0.123},
    )
    assert saved_path.exists()

    # Create fresh model and optimizer
    model_loaded = DiT(depth=2, hidden_size=128, num_heads=2)
    optimizer_loaded = torch.optim.AdamW(model_loaded.parameters(), lr=1e-3)

    # Ensure models differ initially before loading
    assert any(
        not torch.equal(p1, p2)
        for p1, p2 in zip(model_orig.parameters(), model_loaded.parameters())
    )

    # Load checkpoint
    meta = load_checkpoint(
        path=ckpt_file,
        model=model_loaded,
        optimizer=optimizer_loaded,
    )

    # Verify metadata
    assert meta["global_step"] == 42
    assert meta["config"] == {"hidden_size": 128, "depth": 2}
    assert meta["extra_state"] == {"best_loss": 0.123}

    # Verify model parameters match bitwise
    for p_orig, p_loaded in zip(model_orig.parameters(), model_loaded.parameters()):
        assert torch.equal(p_orig, p_loaded), "Loaded parameter does not match saved parameter exactly."

    # Verify optimizer state matches
    opt_state_orig = optimizer_orig.state_dict()
    opt_state_loaded = optimizer_loaded.state_dict()
    assert len(opt_state_orig["state"]) == len(opt_state_loaded["state"])
    for key in opt_state_orig["state"]:
        for subkey in ["exp_avg", "exp_avg_sq"]:
            t_orig = opt_state_orig["state"][key][subkey]
            t_loaded = opt_state_loaded["state"][key][subkey]
            assert torch.equal(t_orig, t_loaded), f"Optimizer {subkey} state mismatch."


def test_checkpoint_rng_state_restoration(tmp_path: Path):
    """Verify that saving and restoring RNG state reproduces identical future random draws."""
    ckpt_file = tmp_path / "rng_ckpt.pt"
    model = DiT(depth=2, hidden_size=128, num_heads=2)

    # Seed and draw some random numbers
    torch.manual_seed(777)
    random.seed(777)
    _ = torch.randn(10)
    _ = random.random()

    # Save checkpoint with current RNG state
    save_checkpoint(ckpt_file, model)

    # Reference continuation: draw next 5 random numbers
    ref_torch_draw = torch.randn(5)
    ref_py_draw = random.random()

    # Discard RNG state by drawing random numbers
    _ = torch.randn(100)
    _ = random.random()

    # Reload checkpoint and restore RNG state
    fresh_model = DiT(depth=2, hidden_size=128, num_heads=2)
    load_checkpoint(ckpt_file, fresh_model, restore_rng=True)

    # Resumed continuation: draw next 5 random numbers
    resumed_torch_draw = torch.randn(5)
    resumed_py_draw = random.random()

    assert torch.equal(ref_torch_draw, resumed_torch_draw), "Torch CPU RNG draw mismatch after resume."
    assert ref_py_draw == resumed_py_draw, "Python RNG draw mismatch after resume."


def test_checkpoint_cuda_rng_state_restoration(tmp_path: Path):
    """Verify CUDA RNG state restoration when CUDA is available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this environment.")

    ckpt_file = tmp_path / "cuda_rng_ckpt.pt"
    model = DiT(depth=2, hidden_size=128, num_heads=2).cuda()

    torch.cuda.manual_seed_all(999)
    _ = torch.randn(10, device="cuda")

    save_checkpoint(ckpt_file, model)

    ref_cuda_draw = torch.randn(5, device="cuda")

    # Disturb RNG
    _ = torch.randn(100, device="cuda")

    fresh_model = DiT(depth=2, hidden_size=128, num_heads=2).cuda()
    load_checkpoint(ckpt_file, fresh_model, restore_rng=True, device="cuda")

    resumed_cuda_draw = torch.randn(5, device="cuda")

    assert torch.equal(ref_cuda_draw, resumed_cuda_draw), "CUDA RNG draw mismatch after resume."


def test_checkpoint_missing_or_corrupted_file_errors(tmp_path: Path):
    """Verify clear error handling on missing or corrupt checkpoint files."""
    model = DiT(depth=2, hidden_size=128, num_heads=2)

    # 1. Missing file
    with pytest.raises(FileNotFoundError, match="Checkpoint file does not exist"):
        load_checkpoint(tmp_path / "non_existent.pt", model)

    # 2. Corrupt file
    corrupt_file = tmp_path / "corrupt.pt"
    with open(corrupt_file, "wb") as f:
        f.write(b"not a valid pytorch checkpoint dictionary")

    with pytest.raises(Exception):
        load_checkpoint(corrupt_file, model)


def test_interrupted_vs_uninterrupted_training_equivalence(tmp_path: Path):
    """The Gold Standard Equivalence Test.

    Proves that interrupting a deterministic training run midway (at step J=5),
    checkpointing, restoring into a fresh model & optimizer, and training to step K=10
    reproduces the uninterrupted run's loss trajectory and final model weights BITWISE EXACTLY.
    """
    ckpt_file = tmp_path / "interrupted_ckpt.pt"
    num_total_steps = 10
    interrupt_step = 5

    # -------------------------------------------------------------
    # 1. UNINTERRUPTED REFERENCE RUN
    # -------------------------------------------------------------
    torch.manual_seed(1234)
    random.seed(1234)

    ref_model = DiT(depth=2, hidden_size=128, num_heads=2)
    ref_opt = torch.optim.AdamW(ref_model.parameters(), lr=1e-3, weight_decay=0.0)
    objective = FlowMatchingObjective()

    ref_losses = []
    for step in range(num_total_steps):
        ref_opt.zero_grad()
        # Synthetic sample generated deterministically from current RNG state
        x_data = torch.randn(2, 4, 32, 32)
        loss, _ = objective(ref_model, x_data)
        ref_losses.append(loss.item())
        loss.backward()
        ref_opt.step()

    ref_final_params = [p.clone().detach() for p in ref_model.parameters()]

    # -------------------------------------------------------------
    # 2. INTERRUPTED AND RESUMED RUN
    # -------------------------------------------------------------
    # Reset identical starting seed
    torch.manual_seed(1234)
    random.seed(1234)

    resumed_model = DiT(depth=2, hidden_size=128, num_heads=2)
    resumed_opt = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3, weight_decay=0.0)

    resumed_losses = []

    # Phase 1: Train for interrupt_step steps (0 to 4)
    for step in range(interrupt_step):
        resumed_opt.zero_grad()
        x_data = torch.randn(2, 4, 32, 32)
        loss, _ = objective(resumed_model, x_data)
        resumed_losses.append(loss.item())
        loss.backward()
        resumed_opt.step()

    # Save checkpoint at step 5
    save_checkpoint(
        path=ckpt_file,
        model=resumed_model,
        optimizer=resumed_opt,
        global_step=interrupt_step,
    )

    # Completely destroy model, optimizer, and disturb RNG
    del resumed_model
    del resumed_opt
    _ = torch.randn(500)
    _ = random.random()

    # Recreate fresh model and optimizer
    new_model = DiT(depth=2, hidden_size=128, num_heads=2)
    new_opt = torch.optim.AdamW(new_model.parameters(), lr=1e-3, weight_decay=0.0)

    # Restore from checkpoint
    meta = load_checkpoint(
        path=ckpt_file,
        model=new_model,
        optimizer=new_opt,
        restore_rng=True,
    )
    assert meta["global_step"] == interrupt_step

    # Phase 2: Train remaining steps (5 to 9)
    for step in range(interrupt_step, num_total_steps):
        new_opt.zero_grad()
        x_data = torch.randn(2, 4, 32, 32)
        loss, _ = objective(new_model, x_data)
        resumed_losses.append(loss.item())
        loss.backward()
        new_opt.step()

    # -------------------------------------------------------------
    # 3. EXACT NUMERICAL EQUIVALENCE VERIFICATION
    # -------------------------------------------------------------
    assert len(resumed_losses) == len(ref_losses) == num_total_steps

    # Check loss trajectories match exactly at every single step
    for step_idx, (l_ref, l_res) in enumerate(zip(ref_losses, resumed_losses)):
        assert l_ref == l_res, f"Loss mismatch at step {step_idx}: ref={l_ref}, resumed={l_res}"

    # Check final weights match bitwise exactly
    for p_ref, p_res in zip(ref_final_params, new_model.parameters()):
        assert torch.equal(p_ref, p_res), "Final model parameters do not match uninterrupted run bitwise!"
