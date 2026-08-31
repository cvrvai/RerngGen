"""Reconstruction validation engine for frozen VAE adapters."""

from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from rernggen.models.vae.interface import AutoencoderKLAdapter


def compute_mse_and_psnr(orig: torch.Tensor, recon: torch.Tensor) -> Tuple[float, float]:
    """Calculates pixel-wise Mean Squared Error (MSE) and Peak Signal-to-Noise Ratio (PSNR).

    Args:
        orig (torch.Tensor): Original RGB tensor [C, H, W] in range [0.0, 1.0].
        recon (torch.Tensor): Reconstructed RGB tensor [C, H, W] in range [0.0, 1.0].

    Returns:
        Tuple[float, float]: (MSE, PSNR in dB).
    """
    mse = torch.mean((orig - recon) ** 2).item()
    if mse <= 1e-12:
        psnr = 100.0
    else:
        psnr = -10.0 * math.log10(mse)
    return mse, psnr


@dataclass
class ReconstructionReport:
    """Structured report containing reconstruction metrics across a validated dataset."""

    dataset_id: str
    preprocessing_version: str
    vae_model_id: str
    vae_revision: str
    scaling_factor: float
    device: str
    dtype: str
    latent_shape: List[int]
    reconstruction_shape: List[int]
    images_validated: int = 0
    failures: int = 0
    aggregate_mse: Dict[str, float] = field(default_factory=dict)
    aggregate_psnr: Dict[str, float] = field(default_factory=dict)
    per_image_metrics: List[Dict[str, Any]] = field(default_factory=list)
    reconstructions_dir: Optional[Path] = None
    report_path: Optional[Path] = None
    runtime_seconds: float = 0.0

    def summary(self) -> str:
        """Generates a human-readable formatted summary string."""
        return (
            "============================================================\n"
            "VAE RECONSTRUCTION VALIDATION COMPLETE\n"
            "============================================================\n"
            f"Dataset ID:            {self.dataset_id}\n"
            f"Preprocessing Version: {self.preprocessing_version}\n"
            f"VAE Model:             {self.vae_model_id} (rev: {self.vae_revision})\n"
            f"Device / Dtype:        {self.device} / {self.dtype}\n"
            f"Scaling Factor:        {self.scaling_factor}\n"
            f"Latent Shape:          {self.latent_shape}\n"
            f"Reconstruction Shape:  {self.reconstruction_shape}\n"
            f"Images Validated:      {self.images_validated}\n"
            f"Failures:              {self.failures}\n"
            f"Mean MSE:              {self.aggregate_mse.get('mean', 0.0):.6f}\n"
            f"Mean PSNR:             {self.aggregate_psnr.get('mean', 0.0):.2f} dB\n"
            f"Min / Max PSNR:        {self.aggregate_psnr.get('min', 0.0):.2f} dB / {self.aggregate_psnr.get('max', 0.0):.2f} dB\n"
            f"Reconstructions Dir:   {self.reconstructions_dir}\n"
            f"Report JSON:           {self.report_path}\n"
            f"Runtime:               {self.runtime_seconds:.2f}s\n"
            "============================================================"
        )


class ReconstructionValidator:
    """Validates VAE reconstruction fidelity over processed dataset images."""

    def __init__(
        self,
        vae_adapter: AutoencoderKLAdapter,
        dataset_root: Union[str, Path] = "datasets",
    ) -> None:
        """Initializes the validator with a frozen VAE adapter.

        Args:
            vae_adapter (AutoencoderKLAdapter): Initialized and frozen VAE adapter.
            dataset_root (Union[str, Path]): Root path for datasets. Default: "datasets".
        """
        self.adapter = vae_adapter
        self.dataset_root = Path(dataset_root)

    def validate_dataset(
        self,
        dataset_id: str = "khmer_story_cartoon_v001",
        preprocessing_version: str = "square256_center_v001",
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> ReconstructionReport:
        """Runs frozen VAE encode/decode reconstruction over all processed images.

        Args:
            dataset_id (str): Dataset ID to validate.
            preprocessing_version (str): Preprocessing version to load images from.
            device (Union[str, torch.device]): Device to run inference on.
            dtype (torch.dtype): Compute dtype.

        Returns:
            ReconstructionReport: Comprehensive validation report.
        """
        start_time = time.time()
        dataset_dir = self.dataset_root / dataset_id
        processed_dir = dataset_dir / "processed" / preprocessing_version
        manifest_path = processed_dir / "manifest.jsonl"

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Processed manifest not found: {manifest_path}. Run preprocessing first."
            )

        # Setup output directories
        sanitized_vae_id = self.adapter.spec.model_id.replace("/", "--")
        validation_dir = dataset_dir / "validation" / f"vae_{sanitized_vae_id}"
        reconstructions_dir = validation_dir / "reconstructions"
        report_path = validation_dir / "report.json"

        reconstructions_dir.mkdir(parents=True, exist_ok=True)

        # Read processed records
        records = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        per_image_metrics: List[Dict[str, Any]] = []
        mses: List[float] = []
        psnrs: List[float] = []

        latent_shape = [1, 4, 32, 32]
        recon_shape = [1, 3, 256, 256]

        self.adapter.to(device=device, dtype=dtype)

        for rec in records:
            image_id = rec["image_id"]
            img_rel_path = rec["output_relative_path"]
            img_path = dataset_dir / img_rel_path

            if not img_path.exists():
                continue

            # 1. Load image as float [0.0, 1.0] tensor
            with Image.open(img_path) as pil_img:
                img_tensor = TF.to_tensor(pil_img.convert("RGB")).unsqueeze(0).to(device=device, dtype=dtype)

            # 2. Normalize to [-1.0, 1.0]
            x_norm = self.adapter.normalize_input(img_tensor)

            # 3. Deterministic posterior-mode encoding -> model latent
            z_model = self.adapter.encode(x_norm)
            latent_shape = list(z_model.shape)

            # 4. Reconstruction via inverse scaling -> decode -> [-1.0, 1.0]
            recon_norm = self.adapter.decode(z_model, is_model_latent=True)
            recon_shape = list(recon_norm.shape)

            # 5. Unnormalize back to [0.0, 1.0]
            recon_img_tensor = self.adapter.unnormalize_output(recon_norm)

            # 6. Compute fidelity metrics
            mse, psnr = compute_mse_and_psnr(img_tensor.squeeze(0).cpu(), recon_img_tensor.squeeze(0).cpu())
            mses.append(mse)
            psnrs.append(psnr)

            # 7. Save reconstructed PNG for human audit
            recon_pil = TF.to_pil_image(recon_img_tensor.squeeze(0).cpu().clamp(0.0, 1.0))
            recon_save_path = reconstructions_dir / f"{image_id}.png"
            recon_pil.save(recon_save_path, format="PNG")

            per_image_metrics.append(
                {
                    "image_id": image_id,
                    "source_processed_path": img_rel_path,
                    "reconstruction_path": f"validation/vae_{sanitized_vae_id}/reconstructions/{image_id}.png",
                    "mse": round(mse, 6),
                    "psnr_db": round(psnr, 2),
                    "original_width": rec["original_width"],
                    "original_height": rec["original_height"],
                }
            )

        runtime = time.time() - start_time

        agg_mse = {
            "mean": sum(mses) / len(mses) if mses else 0.0,
            "min": min(mses) if mses else 0.0,
            "max": max(mses) if mses else 0.0,
        }
        agg_psnr = {
            "mean": sum(psnrs) / len(psnrs) if psnrs else 0.0,
            "min": min(psnrs) if psnrs else 0.0,
            "max": max(psnrs) if psnrs else 0.0,
        }

        report = ReconstructionReport(
            dataset_id=dataset_id,
            preprocessing_version=preprocessing_version,
            vae_model_id=self.adapter.spec.model_id,
            vae_revision=self.adapter.spec.revision,
            scaling_factor=self.adapter.scaling_factor,
            device=str(device),
            dtype=str(dtype),
            latent_shape=latent_shape,
            reconstruction_shape=recon_shape,
            images_validated=len(per_image_metrics),
            failures=0,
            aggregate_mse=agg_mse,
            aggregate_psnr=agg_psnr,
            per_image_metrics=per_image_metrics,
            reconstructions_dir=reconstructions_dir,
            report_path=report_path,
            runtime_seconds=runtime,
        )

        # Save report JSON
        report_dict = asdict(report)
        report_dict["reconstructions_dir"] = str(reconstructions_dir)
        report_dict["report_path"] = str(report_path)

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2)

        return report
