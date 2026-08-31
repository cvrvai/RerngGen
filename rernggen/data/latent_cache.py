"""Deterministic permanent latent cache generator and loader for RerngGen.

Encodes processed 256x256 RGB images once with a frozen VAE into scaled model latents [4, 32, 32],
persisting them as standalone safetensors files with cryptographic provenance tracking,
diagnostics, and atomic manifest management.
"""

import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union
from PIL import Image
from safetensors.torch import load_file, save_file
import torch
import torchvision.transforms.functional as TF
from rernggen.data.importer import compute_sha256
from rernggen.data.schema import LatentCacheReport, LatentRecord
from rernggen.models.vae.interface import AutoencoderKLAdapter


class LatentCacheGenerator:
    """Generates and manages versioned permanent latent caches from processed images."""

    def __init__(
        self,
        vae_adapter: AutoencoderKLAdapter,
        cache_version: str = "vae_sd_mse_square256_v001",
        dataset_root: Union[str, Path] = "datasets",
    ) -> None:
        """Initializes the latent cache generator.

        Args:
            vae_adapter (AutoencoderKLAdapter): Frozen VAE adapter.
            cache_version (str): Subdirectory identifier under cache/latents/.
            dataset_root (Union[str, Path]): Root path for datasets. Default: "datasets".
        """
        self.adapter = vae_adapter
        self.cache_version = cache_version
        self.dataset_root = Path(dataset_root)

    def generate_cache(
        self,
        dataset_id: str = "khmer_story_cartoon_v001",
        preprocessing_version: str = "square256_center_v001",
        force: bool = False,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> LatentCacheReport:
        """Generates permanent scaled latent files for all processed images in the dataset.

        Args:
            dataset_id (str): Dataset ID.
            preprocessing_version (str): Preprocessing version to load images from.
            force (bool): If True, re-encodes latents even if already present. Default: False.
            device (Union[str, torch.device]): Device for VAE inference.
            dtype (torch.dtype): Compute dtype for VAE inference.

        Returns:
            LatentCacheReport: Execution summary metrics and manifest records.
        """
        start_time = time.time()
        dataset_dir = self.dataset_root / dataset_id
        processed_dir = dataset_dir / "processed" / preprocessing_version
        processed_manifest_path = processed_dir / "manifest.jsonl"

        if not processed_manifest_path.exists():
            raise FileNotFoundError(
                f"Processed manifest not found: {processed_manifest_path}. Run preprocessing first."
            )

        cache_dir = dataset_dir / "cache" / "latents" / self.cache_version
        cache_manifest_path = cache_dir / "manifest.jsonl"
        cache_dir.mkdir(parents=True, exist_ok=True)

        self.adapter.to(device=device, dtype=dtype)
        self.adapter.eval()

        report = LatentCacheReport(
            dataset_id=dataset_id,
            preprocessing_version=preprocessing_version,
            cache_version=self.cache_version,
            cache_dir=cache_dir,
            manifest_path=cache_manifest_path,
            vae_provenance={
                "model_id": self.adapter.spec.model_id,
                "revision": self.adapter.spec.revision,
                "scaling_factor": self.adapter.scaling_factor,
                "weights_sha256": self.adapter.spec.weights_sha256,
                "config_sha256": self.adapter.spec.config_sha256,
                "posterior_policy": self.adapter.spec.posterior_policy,
                "latent_channels": self.adapter.spec.latent_channels,
            },
        )

        # Load existing cache manifest for idempotency
        existing_cache_map: Dict[str, Dict[str, Any]] = {}
        if cache_manifest_path.exists() and not force:
            with open(cache_manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        existing_cache_map[rec["image_id"]] = rec

        # Read processed records
        processed_records: List[Dict[str, Any]] = []
        with open(processed_manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    processed_records.append(json.loads(line))

        report.total_images_in_dataset = len(processed_records)
        final_records: List[LatentRecord] = []

        for proc_rec in processed_records:
            image_id = proc_rec["image_id"]
            img_rel = proc_rec["output_relative_path"]
            img_path = dataset_dir / img_rel
            proc_sha = proc_rec["processed_sha256"]

            latent_filename = f"{image_id}.safetensors"
            latent_file = cache_dir / latent_filename
            latent_rel = f"cache/latents/{self.cache_version}/{latent_filename}"

            # Idempotency check: verify existing latent matches image hash and VAE identity
            if (
                not force
                and image_id in existing_cache_map
                and latent_file.exists()
            ):
                cached_rec = existing_cache_map[image_id]
                if (
                    cached_rec.get("source_processed_sha256") == proc_sha
                    and cached_rec.get("vae_weights_sha256") == self.adapter.spec.weights_sha256
                    and cached_rec.get("vae_scaling_factor") == self.adapter.scaling_factor
                    and compute_sha256(latent_file) == cached_rec.get("latent_sha256")
                ):
                    report.valid_cache_hits += 1
                    report.total_cache_bytes += latent_file.stat().st_size
                    final_records.append(LatentRecord(**cached_rec))
                    continue

            # Validate input image file exists
            if not img_path.exists():
                report.failures += 1
                report.failure_details.append(
                    {"image_id": image_id, "error": f"Processed image not found: {img_path}"}
                )
                continue

            try:
                # 1. Load image and normalize to [-1.0, 1.0]
                with Image.open(img_path) as pil_img:
                    img_tensor = TF.to_tensor(pil_img.convert("RGB")).unsqueeze(0).to(device=device, dtype=dtype)

                x_norm = self.adapter.normalize_input(img_tensor)

                # 2. Encode to scaled model latent [1, 4, 32, 32]
                z_model = self.adapter.encode(x_norm, return_raw=False)
                latent_tensor = z_model.squeeze(0).contiguous().to(torch.float32).cpu()

                if not torch.all(torch.isfinite(latent_tensor)):
                    raise ValueError(f"Non-finite values encountered in latent for {image_id}")

                # 3. Calculate tensor diagnostics
                min_val = float(latent_tensor.min().item())
                max_val = float(latent_tensor.max().item())
                mean_val = float(latent_tensor.mean().item())
                std_val = float(latent_tensor.std().item())
                l2_norm = float(torch.linalg.norm(latent_tensor).item())

                # 4. Atomic safetensors write (temp file + os.replace on same filesystem)
                tmp_latent_file = cache_dir / f"{image_id}_tmp_{os.getpid()}_{time.time_ns()}.safetensors"
                save_file({"latent": latent_tensor}, tmp_latent_file)
                os.replace(tmp_latent_file, latent_file)

                latent_size = latent_file.stat().st_size
                report.total_cache_bytes += latent_size
                latent_sha = compute_sha256(latent_file)

                # 5. Build LatentRecord with exact governance passthrough
                lat_rec = LatentRecord(
                    image_id=image_id,
                    dataset_id=dataset_id,
                    dataset_version=proc_rec.get("dataset_version", "v001"),
                    source_processed_sha256=proc_sha,
                    preprocessing_version=preprocessing_version,
                    vae_model_id=self.adapter.spec.model_id,
                    vae_revision=self.adapter.spec.revision,
                    vae_weights_sha256=self.adapter.spec.weights_sha256 or "",
                    vae_config_sha256=self.adapter.spec.config_sha256 or "",
                    vae_scaling_factor=self.adapter.scaling_factor,
                    posterior_policy=self.adapter.spec.posterior_policy,
                    latent_shape=list(latent_tensor.shape),
                    latent_dtype=str(latent_tensor.dtype).replace("torch.", ""),
                    latent_sha256=latent_sha,
                    latent_relative_path=latent_rel,
                    min_val=round(min_val, 4),
                    max_val=round(max_val, 4),
                    mean_val=round(mean_val, 4),
                    std_val=round(std_val, 4),
                    l2_norm=round(l2_norm, 4),
                    training_allowed=proc_rec.get("training_allowed"),
                    commercial_allowed=proc_rec.get("commercial_allowed"),
                    license_id=proc_rec.get("license_id"),
                    cache_version=self.cache_version,
                    status="CACHED",
                )

                final_records.append(lat_rec)
                report.latents_created += 1

            except Exception as e:
                report.failures += 1
                report.failure_details.append({"image_id": image_id, "error": str(e)})

        # 6. Atomic Manifest Write
        if final_records:
            tmp_manifest = cache_dir / f"manifest_tmp_{os.getpid()}_{time.time_ns()}.jsonl"
            with open(tmp_manifest, "w", encoding="utf-8") as f:
                for r in final_records:
                    f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
            os.replace(tmp_manifest, cache_manifest_path)

        report.records = final_records
        report.elapsed_time_seconds = time.time() - start_time
        return report


class LatentCacheLoader:
    """Fast, parameter-free latent tensor loader for training pipelines.

    Reads standalone safetensors files directly from disk without instantiating or invoking a VAE.
    """

    def __init__(
        self,
        dataset_dir: Union[str, Path],
        cache_version: str = "vae_sd_mse_square256_v001",
    ) -> None:
        """Initializes the cache loader.

        Args:
            dataset_dir (Union[str, Path]): Path to dataset directory (e.g. datasets/khmer_story_cartoon_v001).
            cache_version (str): Cache version identifier.
        """
        self.dataset_dir = Path(dataset_dir)
        self.cache_version = cache_version
        self.cache_dir = self.dataset_dir / "cache" / "latents" / cache_version
        self.manifest_path = self.cache_dir / "manifest.jsonl"

    def load_manifest(self) -> List[LatentRecord]:
        """Loads all latent records from the cache manifest."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Cache manifest not found: {self.manifest_path}")

        records: List[LatentRecord] = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(LatentRecord(**json.loads(line)))
        return records

    def load_latent(
        self,
        image_id_or_record: Union[str, LatentRecord],
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Loads a cached latent tensor [4, 32, 32] directly into memory without invoking a VAE.

        Args:
            image_id_or_record (Union[str, LatentRecord]): Image ID or LatentRecord.
            device (Union[str, torch.device]): Target tensor device.
            dtype (torch.dtype): Target tensor dtype.

        Returns:
            torch.Tensor: Scaled model latent [4, 32, 32].
        """
        if isinstance(image_id_or_record, LatentRecord):
            latent_path = self.dataset_dir / image_id_or_record.latent_relative_path
        else:
            latent_path = self.cache_dir / f"{image_id_or_record}.safetensors"

        if not latent_path.exists():
            raise FileNotFoundError(f"Cached latent file not found: {latent_path}")

        tensors = load_file(str(latent_path))
        if "latent" not in tensors:
            raise KeyError(f"Expected key 'latent' in safetensors file {latent_path}")

        latent = tensors["latent"].to(device=device, dtype=dtype)
        return latent
