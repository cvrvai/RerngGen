"""Deterministic Human Caption Review & Invalidation Workflow for RerngGen.

Provides versioned, auditable human quality and acceptance review for dataset captions,
cryptographically binding human approval to exact caption content hashes and enforcing
strict invalidation if captions or sample identities change.
"""

from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from rernggen.data.captions import CaptionManager
from rernggen.data.schema import CaptionReviewRecord


class CaptionReviewStatus(str, Enum):
    """Explicit review decision status for a dataset caption."""

    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    INVALIDATED = "INVALIDATED"


def parse_caption_review_status(val: Union[str, CaptionReviewStatus]) -> str:
    """Parses and validates a caption review status into standard string representation.

    Args:
        val: Input status string or CaptionReviewStatus enum.

    Returns:
        str: Standard uppercase status string ('PENDING', 'APPROVED', 'REJECTED', 'INVALIDATED').

    Raises:
        ValueError: If input is not a recognized caption review status.
    """
    if isinstance(val, CaptionReviewStatus):
        return val.value

    if isinstance(val, str):
        val_upper = val.strip().upper()
        if val_upper in ("PENDING", "APPROVED", "REJECTED", "INVALIDATED"):
            return val_upper

    raise ValueError(
        f"Invalid caption review status '{val}'. "
        f"Must be one of: 'PENDING', 'APPROVED', 'REJECTED', 'INVALIDATED'."
    )


def compute_caption_review_record_sha256(record: Union[Dict[str, Any], CaptionReviewRecord]) -> str:
    """Computes a deterministic SHA-256 hash for a CaptionReviewRecord excluding the hash itself.

    Args:
        record: CaptionReviewRecord dataclass or dictionary.

    Returns:
        str: 64-character hexadecimal SHA-256 hash.
    """
    data = record.to_dict() if isinstance(record, CaptionReviewRecord) else dict(record)
    # Exclude record_sha256 to avoid circularity
    payload_dict = {k: v for k, v in data.items() if k != "record_sha256"}
    serialized = json.dumps(payload_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class CaptionReviewManager:
    """Manages versioned, cryptographic caption review manifests for dataset quality verification."""

    def __init__(self, dataset_root: Union[str, Path] = "datasets") -> None:
        """Initializes the CaptionReviewManager.

        Args:
            dataset_root: Root path containing dataset repositories (default: 'datasets').
        """
        self.dataset_root = Path(dataset_root)

    def get_review_dir(self, dataset_id: str, version: str) -> Path:
        """Returns the absolute directory path for a caption review version.

        Args:
            dataset_id: Dataset identifier.
            version: Review version identifier (e.g. 'caption_review_v001').

        Returns:
            Path: Directory path.
        """
        return self.dataset_root / dataset_id / "caption_reviews" / version

    def list_versions(self, dataset_id: str) -> List[str]:
        """Lists all existing caption review versions for a dataset sorted lexicographically.

        Args:
            dataset_id: Dataset identifier.

        Returns:
            List[str]: List of version directory names containing manifest.jsonl.
        """
        review_base = self.dataset_root / dataset_id / "caption_reviews"
        if not review_base.exists():
            return []

        versions = []
        for p in sorted(review_base.iterdir()):
            if p.is_dir() and (p / "manifest.jsonl").exists():
                versions.append(p.name)
        return versions

    def compute_manifest_sha256(self, dataset_id: str, version: str) -> str:
        """Computes SHA-256 hash of a caption review version manifest file for training provenance.

        Args:
            dataset_id: Dataset identifier.
            version: Review version identifier.

        Returns:
            str: 64-character hexadecimal SHA-256 hash.
        """
        manifest_path = self.get_review_dir(dataset_id, version) / "manifest.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Caption review manifest not found: {manifest_path}")
        return hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    def load_reviews(self, dataset_id: str, version: str) -> List[CaptionReviewRecord]:
        """Loads and verifies all caption review records from a versioned manifest.

        Args:
            dataset_id: Dataset identifier.
            version: Caption review version identifier.

        Returns:
            List[CaptionReviewRecord]: Parsed and verified review records.

        Raises:
            FileNotFoundError: If the manifest file does not exist.
            ValueError: If manifest is corrupt or contains duplicate image IDs.
        """
        manifest_path = self.get_review_dir(dataset_id, version) / "manifest.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Caption review manifest not found: {manifest_path}")

        records: List[CaptionReviewRecord] = []
        seen_ids = set()

        with open(manifest_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                clean_line = line.strip()
                if not clean_line:
                    continue
                try:
                    data = json.loads(clean_line)
                    rec = CaptionReviewRecord(**data)
                except Exception as e:
                    raise ValueError(f"Corrupt JSON at line {line_idx} in {manifest_path}: {e}")

                if rec.image_id in seen_ids:
                    raise ValueError(
                        f"Duplicate image_id '{rec.image_id}' found at line {line_idx} in {manifest_path}"
                    )
                seen_ids.add(rec.image_id)
                records.append(rec)

        return records

    def save_reviews(
        self,
        dataset_id: str,
        version: str,
        records: Sequence[CaptionReviewRecord],
        _allow_test_overwrite: bool = False,
    ) -> Path:
        """Saves caption review records atomically with cryptographic record hashing.

        Finalized versions are strictly immutable. Standard callers cannot overwrite existing version manifests.

        Args:
            dataset_id: Target dataset identifier.
            version: Target caption review version (e.g. 'caption_review_v001').
            records: Sequence of CaptionReviewRecord instances.
            _allow_test_overwrite: Internal test flag only. Standard workflows must create a new version.

        Returns:
            Path: Path to saved manifest.jsonl.

        Raises:
            FileExistsError: If version already exists and _allow_test_overwrite is False.
            ValueError: If record list is empty, contains duplicates, or has invalid fields.
        """
        if not records:
            raise ValueError("Cannot save empty caption review record list.")

        target_dir = self.get_review_dir(dataset_id, version)
        manifest_path = target_dir / "manifest.jsonl"

        if manifest_path.exists() and not _allow_test_overwrite:
            raise FileExistsError(
                f"Caption review version '{version}' already exists at {manifest_path}. "
                f"Review versions are finalized and strictly immutable. "
                f"To record new reviews, create a new version (e.g. caption_review_v002) with base_version='{version}'."
            )

        target_dir.mkdir(parents=True, exist_ok=True)

        # Validate uniqueness & recalculate hashes
        seen_ids = set()
        final_records: List[CaptionReviewRecord] = []

        for r in records:
            if r.image_id in seen_ids:
                raise ValueError(f"Duplicate image_id '{r.image_id}' in caption review record set.")
            seen_ids.add(r.image_id)

            if not r.reviewed_by or not r.reviewed_by.strip():
                raise ValueError(f"CaptionReviewRecord for '{r.image_id}' missing reviewed_by.")
            if not r.review_source or not r.review_source.strip():
                raise ValueError(f"CaptionReviewRecord for '{r.image_id}' missing review_source.")
            if not r.reviewed_at or not r.reviewed_at.strip():
                raise ValueError(f"CaptionReviewRecord for '{r.image_id}' missing reviewed_at timestamp.")
            if not r.caption_sha256 or not r.caption_sha256.strip():
                raise ValueError(f"CaptionReviewRecord for '{r.image_id}' missing caption_sha256.")
            if r.review_status not in ("PENDING", "APPROVED", "REJECTED", "INVALIDATED"):
                raise ValueError(f"Invalid status '{r.review_status}' for '{r.image_id}'.")

            # Recompute SHA-256 for deterministic integrity
            record_hash = compute_caption_review_record_sha256(r)
            r.record_sha256 = record_hash
            final_records.append(r)

        # Atomic write on same filesystem
        tmp_manifest = target_dir / f"manifest_tmp_{os.getpid()}_{time.time_ns()}.jsonl"
        with open(tmp_manifest, "w", encoding="utf-8") as f:
            for r in final_records:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

        os.replace(tmp_manifest, manifest_path)
        return manifest_path

    def review_samples(
        self,
        dataset_id: str,
        review_version: str,
        image_ids: Union[str, Sequence[str]],
        review_status: Union[str, CaptionReviewStatus],
        reviewed_by: str,
        review_source: str,
        caption_manager: Optional[CaptionManager] = None,
        caption_version: str = "captions_v002",
        reason: str = "",
        base_version: Optional[str] = None,
        all_dataset_ids: Optional[Sequence[str]] = None,
        explicit_caption_hashes: Optional[Dict[str, str]] = None,
        _allow_test_overwrite: bool = False,
    ) -> Tuple[Path, List[CaptionReviewRecord]]:
        """Records explicit human caption review decisions bound to exact caption hashes.

        Args:
            dataset_id: Target dataset identifier.
            review_version: Version name to write into (e.g. 'caption_review_v001').
            image_ids: Single image ID, list of IDs, or 'ALL' if all_dataset_ids provided.
            review_status: Explicit decision ('APPROVED', 'REJECTED', 'INVALIDATED', 'PENDING').
            reviewed_by: Mandatory explicit human reviewer identifier.
            review_source: Mandatory explicit review workflow/provenance source.
            caption_manager: Optional CaptionManager instance to look up current captions.
            caption_version: Caption version identifier to verify hashes against.
            reason: Human audit rationale / rejection reason / notes.
            base_version: Optional existing review version to clone/update records from.
            all_dataset_ids: Optional complete list of image IDs in dataset when executing --all.
            explicit_caption_hashes: Optional precomputed map of image_id -> caption_sha256.
            _allow_test_overwrite: Internal test flag only.

        Returns:
            Tuple[Path, List[CaptionReviewRecord]]: Manifest path and saved review records.
        """
        # 1. Validate mandatory reviewer identity and source
        if not isinstance(reviewed_by, str) or not reviewed_by.strip():
            raise ValueError(
                "reviewed_by is required and must be a non-empty string explicitly identifying the reviewer."
            )
        if not isinstance(review_source, str) or not review_source.strip():
            raise ValueError(
                "review_source is required and must be a non-empty string explicitly identifying the review origin."
            )

        status_str = parse_caption_review_status(review_status)

        # 2. Resolve target image IDs
        if isinstance(image_ids, str):
            if image_ids.strip().upper() == "ALL":
                if not all_dataset_ids:
                    raise ValueError(
                        "Cannot review 'ALL' without providing explicit all_dataset_ids list."
                    )
                target_ids = list(all_dataset_ids)
            else:
                target_ids = [image_ids.strip()]
        else:
            target_ids = [img.strip() for img in image_ids if img.strip()]

        if not target_ids:
            raise ValueError("No valid image IDs specified for caption review.")

        # 3. Enforce version immutability
        target_dir = self.get_review_dir(dataset_id, review_version)
        manifest_path = target_dir / "manifest.jsonl"
        if manifest_path.exists() and not _allow_test_overwrite:
            raise FileExistsError(
                f"Caption review version '{review_version}' already exists and is finalized. "
                f"To record new review decisions, specify a new version (e.g. caption_review_v002) "
                f"with base_version='{review_version}'."
            )

        # 4. Resolve current caption hashes for target samples
        caption_hashes: Dict[str, str] = {}
        if explicit_caption_hashes is not None:
            caption_hashes.update(explicit_caption_hashes)

        # If any target IDs lack hash, look up in caption manifest
        missing_hash_ids = [img_id for img_id in target_ids if img_id not in caption_hashes]
        if missing_hash_ids:
            cap_mgr = caption_manager or CaptionManager(dataset_root=self.dataset_root)
            try:
                cap_records = cap_mgr.load_captions(dataset_id, caption_version)
                cap_map = {r.image_id: r.caption_sha256 for r in cap_records}
                for img_id in missing_hash_ids:
                    if img_id not in cap_map:
                        raise ValueError(
                            f"Image ID '{img_id}' not found in caption manifest for version '{caption_version}'."
                        )
                    caption_hashes[img_id] = cap_map[img_id]
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Caption manifest for version '{caption_version}' not found for dataset '{dataset_id}'."
                )

        # 5. Load baseline records if available
        record_map: Dict[str, CaptionReviewRecord] = {}
        if base_version is not None:
            for r in self.load_reviews(dataset_id, base_version):
                record_map[r.image_id] = r
        elif manifest_path.exists() and _allow_test_overwrite:
            for r in self.load_reviews(dataset_id, review_version):
                record_map[r.image_id] = r

        # 6. Apply updates with explicit timestamp
        timestamp = datetime.now(timezone.utc).isoformat()

        for img_id in target_ids:
            prev_version = record_map[img_id].review_version if img_id in record_map else base_version
            rec = CaptionReviewRecord(
                image_id=img_id,
                dataset_id=dataset_id,
                review_version=review_version,
                caption_sha256=caption_hashes[img_id],
                review_status=status_str,
                reviewed_by=reviewed_by.strip(),
                review_source=review_source.strip(),
                reviewed_at=timestamp,
                reason=reason,
                previous_review_version=prev_version,
            )
            record_map[img_id] = rec

        # 7. Deterministically sort records by image_id
        sorted_records = [record_map[k] for k in sorted(record_map.keys())]

        # 8. Save atomically
        saved_manifest_path = self.save_reviews(
            dataset_id=dataset_id,
            version=review_version,
            records=sorted_records,
            _allow_test_overwrite=_allow_test_overwrite,
        )

        return saved_manifest_path, sorted_records
