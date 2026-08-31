"""Deterministic image preprocessing pipeline for RerngGen.

Transforms raw original images into canonical training-ready derivatives
(e.g., 256x256 RGB center-cropped lossless PNGs) with complete transformation metadata
and strict source immutability.
"""

import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image
from rernggen.data.importer import compute_sha256
from rernggen.data.schema import PreprocessingReport, ProcessedRecord


def preprocess_image_to_square(
    img: Image.Image,
    target_size: int = 256,
    resample: Image.Resampling = Image.Resampling.LANCZOS,
) -> Tuple[Image.Image, Dict[str, int]]:
    """Transforms an image to exact target_size x target_size RGB via aspect-preserving resize + center crop.

    Algorithm:
        1. Convert mode to RGB (handling RGBA/LA by compositing on white background).
        2. Compute scaling factor such that the shortest side becomes target_size.
        3. Resize image using high-quality resampling filter (LANCZOS).
        4. Deterministically extract a centered target_size x target_size crop.

    Args:
        img (Image.Image): Input PIL Image.
        target_size (int): Output square resolution (width and height). Default: 256.
        resample (Image.Resampling): Resampling filter. Default: LANCZOS.

    Returns:
        Tuple[Image.Image, Dict[str, int]]:
            - Preprocessed PIL Image [target_size, target_size, RGB].
            - Transformation metadata dictionary.
    """
    orig_w, orig_h = img.size

    # 1. Color space conversion to RGB
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        rgb_img = background
    else:
        rgb_img = img.convert("RGB")

    # 2. Aspect-ratio preserving resize (scale shortest side to target_size)
    if orig_w <= orig_h:
        scale = target_size / float(orig_w)
        resized_w = target_size
        resized_h = max(target_size, int(round(orig_h * scale)))
    else:
        scale = target_size / float(orig_h)
        resized_h = target_size
        resized_w = max(target_size, int(round(orig_w * scale)))

    resized_img = rgb_img.resize((resized_w, resized_h), resample=resample)

    # 3. Deterministic center crop
    crop_left = (resized_w - target_size) // 2
    crop_top = (resized_h - target_size) // 2
    crop_right = crop_left + target_size
    crop_bottom = crop_top + target_size

    cropped_img = resized_img.crop((crop_left, crop_top, crop_right, crop_bottom))

    metadata = {
        "original_width": orig_w,
        "original_height": orig_h,
        "resized_width": resized_w,
        "resized_height": resized_h,
        "crop_left": crop_left,
        "crop_top": crop_top,
        "crop_width": target_size,
        "crop_height": target_size,
        "output_width": target_size,
        "output_height": target_size,
    }

    return cropped_img, metadata


class ImagePreprocessor:
    """Batch preprocessor that transforms dataset originals into versioned derivatives."""

    def __init__(
        self,
        target_size: int = 256,
        version: str = "square256_center_v001",
        dataset_root: Union[str, Path] = "datasets",
    ) -> None:
        """Initializes the image preprocessor.

        Args:
            target_size (int): Target square dimension (e.g. 256). Default: 256.
            version (str): Subdirectory name under processed/ for this derivative version.
            dataset_root (Union[str, Path]): Root path of datasets. Default: "datasets".
        """
        self.target_size = target_size
        self.version = version
        self.dataset_root = Path(dataset_root)

    def process_dataset(
        self,
        dataset_id: str = "khmer_story_cartoon_v001",
        force: bool = False,
    ) -> PreprocessingReport:
        """Processes all valid images in a dataset into standardized derivatives.

        Args:
            dataset_id (str): Identifier of the dataset to process.
            force (bool): If True, re-processes images even if already present. Default: False.

        Returns:
            PreprocessingReport: Summary metrics and record metadata.
        """
        dataset_dir = self.dataset_root / dataset_id
        source_manifest_path = dataset_dir / "manifests" / "manifest.jsonl"
        processed_dir = dataset_dir / "processed" / self.version
        processed_manifest_path = processed_dir / "manifest.jsonl"

        if not source_manifest_path.exists():
            raise FileNotFoundError(
                f"Source manifest not found: {source_manifest_path}. Import dataset first."
            )

        processed_dir.mkdir(parents=True, exist_ok=True)

        report = PreprocessingReport(
            dataset_id=dataset_id,
            preprocessing_version=self.version,
            target_size=self.target_size,
            processed_dir=processed_dir,
            manifest_path=processed_manifest_path,
        )

        start_time = time.time()

        # Load existing processed manifest for idempotency
        existing_processed_map: Dict[str, Dict[str, Any]] = {}
        if processed_manifest_path.exists() and not force:
            with open(processed_manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        existing_processed_map[rec["image_id"]] = rec

        # Read source manifest records
        source_records: List[Dict[str, Any]] = []
        with open(source_manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    source_records.append(json.loads(line))

        report.total_images_in_dataset = len(source_records)
        final_records: List[ProcessedRecord] = []

        for src_rec in source_records:
            image_id = src_rec["image_id"]
            stored_rel = src_rec["stored_relative_path"]
            source_file = dataset_dir / stored_rel
            source_sha = src_rec["sha256"]

            out_filename = f"{image_id}.png"
            out_file = processed_dir / out_filename
            output_rel = f"processed/{self.version}/{out_filename}"

            # Idempotency check: reuse existing derivative if file exists and hash matches
            if (
                not force
                and image_id in existing_processed_map
                and out_file.exists()
                and existing_processed_map[image_id].get("source_sha256") == source_sha
            ):
                report.images_skipped_idempotent += 1
                report.total_output_bytes += out_file.stat().st_size
                final_records.append(
                    ProcessedRecord(**existing_processed_map[image_id])
                )
                continue

            # Validate that original file exists
            if not source_file.exists():
                report.failures += 1
                report.failure_details.append(
                    {"image_id": image_id, "error": f"Original file not found: {source_file}"}
                )
                continue

            # Process image
            try:
                with Image.open(source_file) as img:
                    processed_img, meta = preprocess_image_to_square(
                        img=img,
                        target_size=self.target_size,
                    )

                # Save lossless PNG derivative
                processed_img.save(out_file, format="PNG", optimize=False)
                out_size = out_file.stat().st_size
                report.total_output_bytes += out_size
                processed_sha = compute_sha256(out_file)

                # Record metadata with exact governance passthrough
                proc_rec = ProcessedRecord(
                    image_id=image_id,
                    dataset_id=dataset_id,
                    preprocessing_version=self.version,
                    source_sha256=source_sha,
                    processed_sha256=processed_sha,
                    original_width=meta["original_width"],
                    original_height=meta["original_height"],
                    resized_width=meta["resized_width"],
                    resized_height=meta["resized_height"],
                    crop_left=meta["crop_left"],
                    crop_top=meta["crop_top"],
                    crop_width=meta["crop_width"],
                    crop_height=meta["crop_height"],
                    output_width=meta["output_width"],
                    output_height=meta["output_height"],
                    output_mode="RGB",
                    output_relative_path=output_rel,
                    training_allowed=src_rec.get("training_allowed"),
                    commercial_allowed=src_rec.get("commercial_allowed"),
                    license_id=src_rec.get("license_id"),
                    status="PROCESSED",
                )

                final_records.append(proc_rec)
                report.images_processed += 1

            except Exception as e:
                report.failures += 1
                report.failure_details.append({"image_id": image_id, "error": str(e)})

        # Atomic Manifest Write in processed directory
        if final_records:
            tmp_manifest = processed_dir / f"manifest_tmp_{os.getpid()}_{time.time_ns()}.jsonl"
            with open(tmp_manifest, "w", encoding="utf-8") as f:
                for rec in final_records:
                    f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
            os.replace(tmp_manifest, processed_manifest_path)

        report.records = final_records
        report.elapsed_time_seconds = time.time() - start_time
        return report


def preprocess_dataset(
    dataset_id: str = "khmer_story_cartoon_v001",
    target_size: int = 256,
    version: str = "square256_center_v001",
    dataset_root: Union[str, Path] = "datasets",
    force: bool = False,
) -> PreprocessingReport:
    """Convenience function to preprocess a dataset."""
    preprocessor = ImagePreprocessor(
        target_size=target_size,
        version=version,
        dataset_root=dataset_root,
    )
    return preprocessor.process_dataset(dataset_id=dataset_id, force=force)
