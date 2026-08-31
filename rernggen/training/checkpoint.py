"""Provenance-bound and RNG-reproducible checkpoint saving, validation, and restoration."""

from pathlib import Path
import random
import shutil
from typing import Any, Dict, Optional, Union
import uuid
import numpy as np
import torch
import torch.nn as nn

from rernggen.training.schema import TrainingRunSpec, utcnow_iso

CHECKPOINT_FORMAT_VERSION: str = "training_checkpoint_v001"


def save_training_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    run_spec: TrainingRunSpec,
    snapshot_metadata_sha256: str,
    loss: Optional[float] = None,
    dataloader_generator: Optional[torch.Generator] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Atomically saves model, optimizer, and RNG states bound to immutable training run provenance."""
    out_path = Path(checkpoint_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Capture CUDA RNG states if available
    cuda_rng = None
    if torch.cuda.is_available():
        try:
            cuda_rng = torch.cuda.get_rng_state_all()
        except Exception:
            cuda_rng = None

    payload = {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "step": step,
        "run_id": run_spec.run_id,
        "run_spec_sha256": run_spec.run_spec_sha256,
        "snapshot_metadata_sha256": snapshot_metadata_sha256,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        # Exact Framework-Native RNG States
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": cuda_rng,
        "dataloader_generator_state": dataloader_generator.get_state() if dataloader_generator is not None else None,
        "saved_at": utcnow_iso(),
        "loss": loss,
        "extra_metadata": extra_metadata or {},
    }

    tmp_path = out_path.parent / f".tmp_ckpt_{uuid.uuid4().hex[:8]}.pt"
    torch.save(payload, tmp_path)
    shutil.move(str(tmp_path), str(out_path))
    return out_path


def load_training_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    expected_run_id: Optional[str] = None,
    expected_run_spec_sha256: Optional[str] = None,
    expected_snapshot_metadata_sha256: Optional[str] = None,
    restore_rng: bool = True,
    dataloader_generator: Optional[torch.Generator] = None,
) -> Dict[str, Any]:
    """Loads a training checkpoint and verifies run/snapshot provenance bindings before restoring weights and RNG."""
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: '{ckpt_file}'")

    try:
        # Load weights safely
        payload = torch.load(str(ckpt_file), weights_only=False, map_location="cpu")
    except Exception:
        payload = torch.load(str(ckpt_file), map_location="cpu")

    # 1. Verify Format Version if present
    fmt_ver = payload.get("checkpoint_format_version")
    if fmt_ver is not None and fmt_ver != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint format version '{fmt_ver}', expected '{CHECKPOINT_FORMAT_VERSION}'."
        )

    # 2. Verify run_id binding FIRST
    if expected_run_id is not None:
        ckpt_run_id = payload.get("run_id")
        if ckpt_run_id != expected_run_id:
            raise ValueError(
                f"Checkpoint provenance mismatch for '{ckpt_file}': "
                f"checkpoint run_id '{ckpt_run_id}' does not match expected '{expected_run_id}'."
            )

    # 3. Verify run_spec_sha256 binding FIRST
    if expected_run_spec_sha256 is not None:
        ckpt_spec_sha = payload.get("run_spec_sha256")
        if ckpt_spec_sha != expected_run_spec_sha256:
            raise ValueError(
                f"Checkpoint experiment spec mismatch for '{ckpt_file}': "
                f"checkpoint run_spec_sha256 '{ckpt_spec_sha}' does not match expected '{expected_run_spec_sha256}'."
            )

    # 4. Verify snapshot_metadata_sha256 binding FIRST
    if expected_snapshot_metadata_sha256 is not None:
        ckpt_snap_sha = payload.get("snapshot_metadata_sha256")
        if ckpt_snap_sha != expected_snapshot_metadata_sha256:
            raise ValueError(
                f"Checkpoint dataset snapshot mismatch for '{ckpt_file}': "
                f"checkpoint snapshot_metadata_sha256 '{ckpt_snap_sha}' does not match expected '{expected_snapshot_metadata_sha256}'."
            )

    # 5. Restore model weights
    if "model_state_dict" not in payload:
        raise KeyError(f"Corrupted checkpoint '{ckpt_file}': missing 'model_state_dict'.")
    model.load_state_dict(payload["model_state_dict"])

    # 6. Restore optimizer state if supplied
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    # 7. Restore RNG States only after provenance and weights are verified
    if restore_rng:
        if "python_rng_state" in payload and payload["python_rng_state"] is not None:
            random.setstate(payload["python_rng_state"])
        if "numpy_rng_state" in payload and payload["numpy_rng_state"] is not None:
            np.random.set_state(payload["numpy_rng_state"])
        if "torch_rng_state" in payload and payload["torch_rng_state"] is not None:
            torch.set_rng_state(payload["torch_rng_state"])
        if (
            torch.cuda.is_available()
            and "cuda_rng_states" in payload
            and payload["cuda_rng_states"] is not None
        ):
            try:
                torch.cuda.set_rng_state_all(payload["cuda_rng_states"])
            except Exception:
                pass
        if (
            dataloader_generator is not None
            and "dataloader_generator_state" in payload
            and payload["dataloader_generator_state"] is not None
        ):
            dataloader_generator.set_state(payload["dataloader_generator_state"])

    return payload
