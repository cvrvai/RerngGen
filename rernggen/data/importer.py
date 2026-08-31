"""Image dataset importer for RerngGen.

Scans, validates, deduplicates, and copies source images into versioned dataset repositories
with atomic manifest generation and strict source immutability.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from PIL import Image
from rernggen.data.schema import ImportReport, ManifestRecord

SUPPORTED_EXTENSIONS: Set[str] = {".png", ".jpg", ".jpeg", ".webp"}


def compute_sha256(path: Path) -> str:
    """Calculates SHA-256 hash over raw file bytes using chunked streaming."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class DatasetImporter:
    """Manages the intake and indexing of source images into a standardized RerngGen dataset."""

    def __init__(
        self,
        dataset_id: str = "khmer_story_cartoon_v001",
        dataset_root: Union[str, Path] = "datasets",
    ) -> None:
        """Initializes the dataset importer.

        Args:
            dataset_id (str): Unique dataset identifier (e.g. "khmer_story_cartoon_v001").
            dataset_root (Union[str, Path]): Root path for dataset storage. Default: "datasets".
        """
        self.dataset_id = dataset_id
        self.dataset_root = Path(dataset_root)
        self.dataset_dir = self.dataset_root / dataset_id

        # Define subdirectories
        self.originals_dir = self.dataset_dir / "originals"
        self.processed_dir = self.dataset_dir / "processed"
        self.captions_dir = self.dataset_dir / "captions"
        self.manifests_dir = self.dataset_dir / "manifests"
        self.cache_dir = self.dataset_dir / "cache"
        self.manifest_path = self.manifests_dir / "manifest.jsonl"

        # Extract dataset version from ID (e.g. "v001" from "khmer_story_cartoon_v001")
        version_match = re.search(r"v\d+", dataset_id)
        self.dataset_version = version_match.group(0) if version_match else "v001"

    def _ensure_directories(self) -> None:
        """Ensures all standard dataset subdirectories exist."""
        for d in [
            self.originals_dir,
            self.processed_dir,
            self.captions_dir,
            self.manifests_dir,
            self.cache_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def _load_existing_manifest(self) -> Tuple[Dict[str, str], List[Dict[str, Any]], int]:
        """Loads existing manifest to support idempotent incremental ingestion.

        Returns:
            Tuple[Dict[str, str], List[Dict[str, Any]], int]:
                - known_hashes: mapping from sha256 to image_id
                - existing_records: list of already imported manifest dictionary records
                - max_id_index: highest numerical index among existing image IDs
        """
        known_hashes: Dict[str, str] = {}
        existing_records: List[Dict[str, Any]] = []
        max_id_index = 0

        if self.manifest_path.exists():
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    existing_records.append(record)
                    img_id = record.get("image_id", "")
                    sha = record.get("sha256", "")
                    if sha:
                        known_hashes[sha] = img_id

                    # Extract numerical portion from e.g. "IMG-000042"
                    id_match = re.search(r"IMG-(\d+)", img_id)
                    if id_match:
                        idx = int(id_match.group(1))
                        if idx > max_id_index:
                            max_id_index = idx

        return known_hashes, existing_records, max_id_index

    def import_directory(self, source_dir: Union[str, Path]) -> ImportReport:
        """Recursively scans and imports images from the given source directory.

        Guarantees:
            1. Source files are NEVER modified, moved, renamed, or deleted.
            2. Content-addressable SHA-256 deduplication.
            3. Atomic manifest writes using os.replace in the target directory.
            4. Idempotency on repeated executions.

        Args:
            source_dir (Union[str, Path]): Path to source image folder.

        Returns:
            ImportReport: Structured report with exact execution metrics.
        """
        source_dir = Path(source_dir).resolve()
        if not source_dir.exists() or not source_dir.is_dir():
            raise FileNotFoundError(f"Source directory does not exist or is not a directory: {source_dir}")

        self._ensure_directories()
        known_hashes, existing_records, current_id_counter = self._load_existing_manifest()

        report = ImportReport(
            source_dir=source_dir,
            dataset_id=self.dataset_id,
            dataset_root=self.dataset_root,
            manifest_path=self.manifest_path,
            originals_path=self.originals_dir,
        )

        start_time = time.time()
        new_records: List[ManifestRecord] = []

        # Recursively collect all candidate files
        candidate_paths: List[Path] = []
        for root, _, files in os.walk(source_dir):
            for fname in files:
                candidate_paths.append(Path(root) / fname)

        report.files_discovered = len(candidate_paths)

        # Sort candidate paths for deterministic processing order across platforms
        candidate_paths.sort(key=lambda p: str(p).lower())

        for file_path in candidate_paths:
            ext = file_path.suffix.lower()

            # 1. Skip unsupported file extensions
            if ext not in SUPPORTED_EXTENSIONS:
                report.unsupported_files += 1
                continue

            report.supported_candidates += 1

            # 2. Verify decodability and extract dimensions using PIL
            try:
                with Image.open(file_path) as img:
                    img.verify()
                with Image.open(file_path) as img:
                    width, height = img.size
                    mode = img.mode
                    img_format = img.format or ext.lstrip(".").upper()
            except Exception as e:
                report.corrupt_images += 1
                report.corrupt_details.append(
                    {"source_path": str(file_path), "error": str(e)}
                )
                continue

            report.valid_images += 1

            # 3. Calculate SHA-256 over raw source file bytes
            file_hash = compute_sha256(file_path)

            # 4. Check for duplicate content
            if file_hash in known_hashes:
                existing_id = known_hashes[file_hash]
                report.duplicate_images_skipped += 1
                report.duplicate_details.append(
                    {
                        "source_path": str(file_path),
                        "existing_image_id": existing_id,
                        "sha256": file_hash,
                    }
                )
                continue

            # 5. Assign sequential image ID
            current_id_counter += 1
            image_id = f"IMG-{current_id_counter:06d}"
            dest_filename = f"{image_id}{ext}"
            dest_path = self.originals_dir / dest_filename

            # 6. Copy exact original bytes (source is strictly read-only)
            shutil.copy2(file_path, dest_path)
            file_size = file_path.stat().st_size
            report.total_bytes_copied += file_size

            # Register hash
            known_hashes[file_hash] = image_id

            # 7. Compute relative paths
            try:
                source_rel = str(file_path.relative_to(source_dir))
            except ValueError:
                source_rel = str(file_path)

            stored_rel = f"originals/{dest_filename}"

            record = ManifestRecord(
                image_id=image_id,
                dataset_id=self.dataset_id,
                dataset_version=self.dataset_version,
                original_filename=file_path.name,
                source_relative_path=source_rel,
                stored_relative_path=stored_rel,
                sha256=file_hash,
                width=width,
                height=height,
                format=img_format,
                mode=mode,
                caption=None,
                tags=[],
                source="local_owned_dataset",
                license_id=None,
                training_allowed=None,
                commercial_allowed=None,
                split=None,
                status="IMPORTED",
            )
            new_records.append(record)
            report.imported_images += 1

        # 8. Atomic Manifest Write
        if new_records:
            all_records = existing_records + [r.to_dict() for r in new_records]
            tmp_manifest = self.manifests_dir / f"manifest_tmp_{os.getpid()}_{time.time_ns()}.jsonl"

            with open(tmp_manifest, "w", encoding="utf-8") as f:
                for rec in all_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # Atomic commit on the same filesystem
            os.replace(tmp_manifest, self.manifest_path)

        report.elapsed_time_seconds = time.time() - start_time
        return report


def import_image_directory(
    source_dir: Union[str, Path],
    dataset_id: str = "khmer_story_cartoon_v001",
    dataset_root: Union[str, Path] = "datasets",
) -> ImportReport:
    """Convenience functional API to import an image directory."""
    importer = DatasetImporter(dataset_id=dataset_id, dataset_root=dataset_root)
    return importer.import_directory(source_dir=source_dir)
