"""Caption management and serialization engine for RerngGen datasets."""

import hashlib
import json
import os
from pathlib import Path
import time
from typing import List, Optional, Union
from rernggen.data.schema import CaptionRecord


def compute_caption_sha256(text: str) -> str:
    """Computes SHA-256 hash of UTF-8 encoded caption text string."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


class CaptionManager:
    """Manages creation, validation, and versioned storage of image captions."""

    def __init__(self, dataset_root: Union[str, Path] = "datasets") -> None:
        """Initializes the caption manager.

        Args:
            dataset_root (Union[str, Path]): Root path for datasets. Default: "datasets".
        """
        self.dataset_root = Path(dataset_root)

    def create_caption_record(
        self,
        image_id: str,
        dataset_id: str,
        caption: str,
        caption_source: str = "manual",
        caption_version: str = "captions_v001",
        language: str = "en",
        review_status: str = "reviewed",
        training_allowed: Optional[bool] = None,
        commercial_allowed: Optional[bool] = None,
        license_id: Optional[str] = None,
    ) -> CaptionRecord:
        """Constructs and validates a CaptionRecord.

        Args:
            image_id: Image identifier (e.g. IMG-000001).
            dataset_id: Dataset ID.
            caption: Non-empty semantic description of image content.
            caption_source: Origin of caption ('manual', 'ai_generated', etc.).
            caption_version: Version identifier (default: 'captions_v001').
            language: ISO language code (default: 'en').
            review_status: 'reviewed' or 'unreviewed'.
            training_allowed: Governance flag.
            commercial_allowed: Governance flag.
            license_id: License identifier.

        Returns:
            CaptionRecord: Validated record with computed SHA-256 hash.
        """
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError(f"Caption for {image_id} must be a non-empty string.")

        caption_clean = caption.strip()
        sha = compute_caption_sha256(caption_clean)

        return CaptionRecord(
            image_id=image_id,
            dataset_id=dataset_id,
            caption=caption_clean,
            caption_source=caption_source,
            caption_version=caption_version,
            caption_sha256=sha,
            language=language,
            review_status=review_status,
            training_allowed=training_allowed,
            commercial_allowed=commercial_allowed,
            license_id=license_id,
        )

    def save_captions(
        self,
        dataset_id: str,
        captions: List[CaptionRecord],
        version: str = "captions_v001",
    ) -> Path:
        """Saves caption records atomically to versioned dataset directory.

        Args:
            dataset_id: Target dataset ID.
            captions: List of CaptionRecord instances.
            version: Caption version identifier.

        Returns:
            Path: Path to written manifest.jsonl file.
        """
        if not captions:
            raise ValueError("Cannot save empty caption list.")

        target_dir = self.dataset_root / dataset_id / "captions" / version
        target_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = target_dir / "manifest.jsonl"

        tmp_manifest = target_dir / f"manifest_tmp_{os.getpid()}_{time.time_ns()}.jsonl"
        with open(tmp_manifest, "w", encoding="utf-8") as f:
            for rec in captions:
                f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")

        os.replace(tmp_manifest, manifest_path)
        return manifest_path

    def load_captions(
        self,
        dataset_id: str,
        version: str = "captions_v001",
    ) -> List[CaptionRecord]:
        """Loads caption records from a versioned manifest.

        Args:
            dataset_id: Target dataset ID.
            version: Caption version identifier.

        Returns:
            List[CaptionRecord]: Parsed caption records.
        """
        manifest_path = self.dataset_root / dataset_id / "captions" / version / "manifest.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Caption manifest not found: {manifest_path}")

        records: List[CaptionRecord] = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(CaptionRecord(**json.loads(line)))

        return records
