"""End-to-end training preflight and TinyDiT smoke training execution engine."""

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn

from rernggen.models.dit.model import TinyDiT
from rernggen.training.checkpoint import save_training_checkpoint
from rernggen.training.dataset import SnapshotTrainingDataset, create_snapshot_dataloader
from rernggen.training.diffusion import DiffusionSchedule
from rernggen.training.preflight import run_preflight_checks
from rernggen.training.provenance import TrainingRunManager


def seed_everything(seed: int) -> None:
    """Sets deterministic seeds across Python, NumPy, and PyTorch (CPU & CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_tiny_dit_smoke(
    run_id: str,
    training_root: Union[str, Path] = "training_runs",
    dataset_root: Union[str, Path] = "datasets",
    device: str = "cpu",
    max_steps: int = 2,
    save_checkpoint: bool = True,
) -> Dict[str, Any]:
    """Executes preflight verification and an end-to-end TinyDiT smoke training loop."""
    # 1. Execute Preflight Checks
    preflight = run_preflight_checks(
        run_id=run_id,
        training_root=training_root,
        dataset_root=dataset_root,
        device=device,
    )

    if not preflight.passed:
        raise ValueError(
            f"Training preflight failed for run '{run_id}': {'; '.join(preflight.errors)}"
        )

    mgr = TrainingRunManager(training_root=training_root, dataset_root=dataset_root)

    # 2. Lifecycle State Transition: PLANNED -> RUNNING
    run = mgr.start_run(run_id)

    step_losses: List[float] = []
    ckpt_path: Optional[str] = None
    cur_step = 0

    try:
        # 3. Deterministic Seed Enforcement
        seed_everything(run.spec.seed)
        target_device = torch.device(device)

        # 4. Instantiate TinyDiT Architecture
        model = TinyDiT.from_config(run.spec.model_config).to(target_device)
        param_counts = model.get_parameter_count()

        # 5. Instantiate Diffusion Schedule
        num_timesteps = int(run.spec.training_config.get("num_timesteps", 100))
        schedule = DiffusionSchedule(num_timesteps=num_timesteps)

        # 6. Instantiate Dataset and DataLoader from Frozen Snapshot
        dataset = SnapshotTrainingDataset(
            dataset_id=run.spec.dataset_id,
            snapshot_version=run.spec.snapshot_version,
            dataset_root=dataset_root,
        )
        batch_size = int(run.spec.training_config.get("batch_size", 2))
        dl_generator = torch.Generator()
        dl_generator.manual_seed(run.spec.seed)
        dataloader = create_snapshot_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=dl_generator,
        )

        # 7. Optimizer Setup
        lr = float(run.spec.training_config.get("learning_rate", run.spec.training_config.get("lr", 1e-4)))
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        # 8. Training Steps
        data_iter = iter(dataloader)
        for step in range(1, max_steps + 1):
            cur_step = step
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            x_0 = batch["latent"].to(target_device)
            text_emb = batch["text_embedding"].to(target_device)
            B = x_0.shape[0]

            t = torch.randint(0, schedule.num_timesteps, (B,), device=target_device)

            # Forward pass & Loss
            loss, eps_hat, eps = schedule.training_loss(model, x_0, t, text_emb)

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss detected at step {step}: {loss.item()}")
            if loss.ndim != 0:
                raise ValueError(f"Loss must be scalar (ndim 0), got shape {loss.shape}")

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Verify Gradients
            grad_found = False
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if not torch.isfinite(param.grad).all():
                        raise FloatingPointError(f"Non-finite gradient in parameter '{name}'.")
                    grad_found = True

            if not grad_found:
                raise RuntimeError("No model parameters received gradients during backward pass.")

            # Optimizer step
            optimizer.step()
            step_losses.append(loss.item())

        # 9. Checkpoint Saving
        if save_checkpoint:
            ckpt_dir = run.run_dir / "checkpoints"
            ckpt_file = ckpt_dir / f"step_{max_steps:06d}.pt"
            save_training_checkpoint(
                checkpoint_path=ckpt_file,
                model=model,
                optimizer=optimizer,
                step=max_steps,
                run_spec=run.spec,
                snapshot_metadata_sha256=run.spec.snapshot_metadata_sha256,
                loss=step_losses[-1] if step_losses else None,
                dataloader_generator=dl_generator,
            )
            ckpt_path = str(ckpt_file)

        # 10. Lifecycle State Transition: RUNNING -> COMPLETED
        mgr.complete_run(
            run_id=run_id,
            current_step=max_steps,
            last_checkpoint=ckpt_path,
        )

        return {
            "run_id": run_id,
            "status": "COMPLETED",
            "device": device,
            "steps_completed": max_steps,
            "step_losses": step_losses,
            "parameter_counts": param_counts,
            "checkpoint_path": ckpt_path,
            "preflight_result": preflight.to_dict(),
        }

    except Exception as e:
        # Fail-closed lifecycle transition: RUNNING -> FAILED
        mgr.fail_run(
            run_id=run_id,
            failure_reason=str(e),
            current_step=cur_step,
            last_checkpoint=ckpt_path,
        )
        raise
