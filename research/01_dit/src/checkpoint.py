"""Checkpoint Save and Resume Utilities for Diffusion Transformers (DiT).

Provides atomic checkpoint serialization and exact restoration of:
    1. Model weights
    2. Optimizer state (momentum / Adam second moments)
    3. Global step counter
    4. Experiment configuration
    5. Python, PyTorch CPU, and PyTorch CUDA RNG states
"""

import os
from pathlib import Path
import random
import time
from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn


def save_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    global_step: int = 0,
    config: Optional[Dict[str, Any]] = None,
    extra_state: Optional[Dict[str, Any]] = None,
) -> Path:
    """Atomically serializes the complete training state to disk.

    Uses a temporary file and atomic rename (os.replace) to guarantee
    zero risk of leaving a corrupt/truncated checkpoint file if interrupted.

    Args:
        path (Union[str, Path]): Target filepath for the checkpoint (e.g. "ckpt.pt").
        model (nn.Module): The model instance to save.
        optimizer (Optional[torch.optim.Optimizer]): Optimizer instance to save.
        global_step (int): Current global training step index.
        config (Optional[Dict[str, Any]]): Model/training configuration dictionary.
        extra_state (Optional[Dict[str, Any]]): Additional metadata to persist.

    Returns:
        Path: Path to the finalized checkpoint file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Capture all pseudo-random number generator states
    rng_state = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

    checkpoint_dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "global_step": global_step,
        "config": config,
        "rng_state": rng_state,
        "extra_state": extra_state or {},
    }

    # Write to a unique temporary file in the same directory
    tmp_path = path.with_name(f"{path.stem}_tmp_{os.getpid()}_{time.time_ns()}.pt")
    torch.save(checkpoint_dict, tmp_path)

    # Atomic rename / replace: guarantees atomic replacement on POSIX and Windows NTFS
    os.replace(tmp_path, path)

    return path


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    restore_rng: bool = True,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Deserializes and restores the complete training state from disk.

    Args:
        path (Union[str, Path]): Path to the checkpoint file.
        model (nn.Module): Target model instance to load weights into.
        optimizer (Optional[torch.optim.Optimizer]): Optional target optimizer.
        restore_rng (bool): Whether to restore Python, CPU, and CUDA RNG states. Default: True.
        device (Optional[Union[str, torch.device]]): Map location device. Default: "cpu".

    Returns:
        Dict[str, Any]: Dictionary containing "global_step", "config", and "extra_state".
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {path}")

    checkpoint = torch.load(path, map_location=device or "cpu")

    # 1. Restore model parameters
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint at {path} is missing 'model_state_dict'.")
    model.load_state_dict(checkpoint["model_state_dict"])

    # 2. Restore optimizer states (momenta, second moments, step counters)
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # 3. Restore RNG states
    if restore_rng and "rng_state" in checkpoint and checkpoint["rng_state"] is not None:
        rng_state = checkpoint["rng_state"]
        if "python" in rng_state and rng_state["python"] is not None:
            random.setstate(rng_state["python"])
        if "torch_cpu" in rng_state and rng_state["torch_cpu"] is not None:
            cpu_state = rng_state["torch_cpu"]
            if isinstance(cpu_state, torch.Tensor):
                cpu_state = cpu_state.cpu()
            torch.set_rng_state(cpu_state)
        if (
            torch.cuda.is_available()
            and "torch_cuda" in rng_state
            and rng_state["torch_cuda"] is not None
        ):
            cuda_states = rng_state["torch_cuda"]
            if isinstance(cuda_states, list):
                cuda_states = [s.cpu() if isinstance(s, torch.Tensor) else s for s in cuda_states]
            elif isinstance(cuda_states, torch.Tensor):
                cuda_states = cuda_states.cpu()
            torch.cuda.set_rng_state_all(cuda_states)

    return {
        "global_step": checkpoint.get("global_step", 0),
        "config": checkpoint.get("config"),
        "extra_state": checkpoint.get("extra_state", {}),
    }
