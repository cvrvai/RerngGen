"""Deterministic, immutable dataset snapshot generator and loader for RerngGen.

Provides deterministic integrity validation and freeze of eligible training samples, binding
exact captions, latents, text embeddings, governance records, caption reviews, and eligibility policies
into a tamper-evident, permanent training input artifact.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union
from rernggen.data.dataset import GovernanceMode, PairedLatentTextDataset
from rernggen.data.eligibility import TRAINING_ELIGIBILITY_POLICY_VERSION, TrainingEligibilityEvaluator
from rernggen.data.importer import compute_sha256
from rernggen.data.schema import DatasetSnapshotMetadata, DatasetSnapshotRecord


class SnapshotStatus(str, Enum):
    """Lifecycle status for a dataset snapshot."""

    DRAFT = "DRAFT"
    FROZEN = "FROZEN"
    SUPERSEDED = "SUPERSEDED"


def serialize_snapshot_record(record: Union[Dict[str, Any], DatasetSnapshotRecord]) -> str:
    """Canonical JSON serialization for a DatasetSnapshotRecord with deterministic key ordering.

    Includes the finalized record_sha256 field.

    Args:
        record: DatasetSnapshotRecord dataclass or dictionary.

    Returns:
        str: Deterministic JSON string.
    """
    data = record.to_dict() if isinstance(record, DatasetSnapshotRecord) else dict(record)
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_snapshot_record_sha256(record: Union[Dict[str, Any], DatasetSnapshotRecord]) -> str:
    """Computes a deterministic SHA-256 hash for a DatasetSnapshotRecord excluding record_sha256.

    Args:
        record: DatasetSnapshotRecord dataclass or dictionary.

    Returns:
        str: 64-character hexadecimal SHA-256 hash.
    """
    data = record.to_dict() if isinstance(record, DatasetSnapshotRecord) else dict(record)
    payload_dict = {k: v for k, v in data.items() if k != "record_sha256"}
    serialized = json.dumps(payload_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def compute_snapshot_metadata_sha256(metadata: Union[Dict[str, Any], DatasetSnapshotMetadata]) -> str:
    """Computes a deterministic SHA-256 hash for DatasetSnapshotMetadata excluding metadata_sha256.

    Args:
        metadata: DatasetSnapshotMetadata dataclass or dictionary.

    Returns:
        str: 64-character hexadecimal SHA-256 hash.
    """
    data = metadata.to_dict() if isinstance(metadata, DatasetSnapshotMetadata) else dict(metadata)
    payload_dict = {k: v for k, v in data.items() if k != "metadata_sha256"}
    serialized = json.dumps(payload_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass
class DatasetSnapshotCandidate:
    """Candidate inspection report generated prior to freezing a snapshot."""

    dataset_id: str
    snapshot_version: str
    total_samples: int
    eligible_count: int
    ineligible_count: int
    reason_counts: Dict[str, int]
    records: List[DatasetSnapshotRecord]
    governance_version: Optional[str]
    governance_manifest_sha256: Optional[str]
    caption_review_version: Optional[str]
    caption_review_manifest_sha256: Optional[str]
    eligibility_policy_version: str
    latent_cache_version: str
    text_cache_version: str
    caption_version: str
    can_freeze: bool

    def summary(self) -> str:
        """Formats candidate plan into a human-readable audit string."""
        return (
            "============================================================\n"
            "DATASET SNAPSHOT CANDIDATE PLAN\n"
            "============================================================\n"
            f"Dataset ID:                 {self.dataset_id}\n"
            f"Candidate Snapshot Version: {self.snapshot_version}\n"
            f"Policy Version:             {self.eligibility_policy_version}\n"
            f"Governance Version:         {self.governance_version or '(none)'}\n"
            f"Governance Manifest SHA:    {self.governance_manifest_sha256 or '(none)'}\n"
            f"Caption Review Version:     {self.caption_review_version or '(none)'}\n"
            f"Caption Review SHA:         {self.caption_review_manifest_sha256 or '(none)'}\n"
            f"Latent Cache Version:       {self.latent_cache_version}\n"
            f"Text Cache Version:         {self.text_cache_version}\n"
            f"Caption Version:            {self.caption_version}\n"
            "------------------------------------------------------------\n"
            f"Total Paired Samples:       {self.total_samples}\n"
            f"Eligible (Admitted):        {self.eligible_count}\n"
            f"Ineligible (Excluded):      {self.ineligible_count}\n"
            f"Can Freeze Snapshot:        {self.can_freeze}\n"
            "------------------------------------------------------------\n"
            f"Reason Breakdown:           {self.reason_counts}\n"
            "============================================================"
        )


@dataclass
class DatasetSnapshot:
    """Loaded immutable dataset snapshot containing verified metadata and frozen records."""

    metadata: DatasetSnapshotMetadata
    records: List[DatasetSnapshotRecord]
    manifest_path: Path
    metadata_path: Path
    records_by_id: Dict[str, DatasetSnapshotRecord] = field(init=False)

    def __post_init__(self) -> None:
        """Builds lookup index by sample_id."""
        self.records_by_id = {r.sample_id: r for r in self.records}

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[DatasetSnapshotRecord]:
        return iter(self.records)

    def __getitem__(self, index_or_id: Union[int, str]) -> DatasetSnapshotRecord:
        if isinstance(index_or_id, int):
            return self.records[index_or_id]
        elif isinstance(index_or_id, str):
            if index_or_id not in self.records_by_id:
                raise KeyError(f"Sample '{index_or_id}' not present in snapshot '{self.metadata.snapshot_version}'.")
            return self.records_by_id[index_or_id]
        else:
            raise TypeError(f"Invalid index type '{type(index_or_id)}'. Must be int or str.")

    def get_sample(self, sample_id: str) -> Optional[DatasetSnapshotRecord]:
        """Returns snapshot record for sample_id or None if absent."""
        return self.records_by_id.get(sample_id)

    def to_provenance_dict(self) -> Dict[str, Any]:
        """Returns training-provenance dictionary for embedding into model checkpoints."""
        return {
            "dataset_id": self.metadata.dataset_id,
            "snapshot_version": self.metadata.snapshot_version,
            "snapshot_status": self.metadata.status,
            "sample_count": self.metadata.sample_count,
            "snapshot_metadata_sha256": self.metadata.metadata_sha256,
            "snapshot_manifest_sha256": self.metadata.snapshot_manifest_sha256,
            "governance_version": self.metadata.governance_version,
            "governance_manifest_sha256": self.metadata.governance_manifest_sha256,
            "caption_review_version": self.metadata.caption_review_version,
            "caption_review_manifest_sha256": self.metadata.caption_review_manifest_sha256,
            "eligibility_policy_version": self.metadata.eligibility_policy_version,
            "created_at": self.metadata.created_at,
            "created_by": self.metadata.created_by,
            "creation_source": self.metadata.creation_source,
        }


class DatasetSnapshotManager:
    """Manages creation, immutability enforcement, and verification of dataset snapshots."""

    def __init__(self, dataset_root: Union[str, Path] = "datasets") -> None:
        """Initializes the dataset snapshot manager.

        Args:
            dataset_root: Root path containing dataset repositories (default: 'datasets').
        """
        self.dataset_root = Path(dataset_root)

    def get_snapshot_dir(self, dataset_id: str, snapshot_version: str) -> Path:
        """Returns directory path for a snapshot version.

        Args:
            dataset_id: Dataset identifier.
            snapshot_version: Snapshot version identifier (e.g. 'dataset_snapshot_v001').

        Returns:
            Path: Snapshot directory path.
        """
        return self.dataset_root / dataset_id / "snapshots" / snapshot_version

    def list_snapshots(self, dataset_id: str, verify_integrity: bool = True) -> List[DatasetSnapshotMetadata]:
        """Discovers and returns all snapshot metadata for a dataset in version order.

        Args:
            dataset_id: Dataset identifier.
            verify_integrity: If True, validates metadata SHA and dataset_id before returning.

        Returns:
            List[DatasetSnapshotMetadata]: Discovered valid snapshot metadata records.
        """
        snapshots_root = self.dataset_root / dataset_id / "snapshots"
        if not snapshots_root.exists():
            return []

        snapshots: List[DatasetSnapshotMetadata] = []
        for version_dir in sorted(snapshots_root.iterdir()):
            if version_dir.is_dir():
                meta_p = version_dir / "metadata.json"
                if meta_p.exists():
                    try:
                        with open(meta_p, "r", encoding="utf-8") as f:
                            meta_data = json.load(f)
                            meta = DatasetSnapshotMetadata(**meta_data)
                            if verify_integrity:
                                if not meta.metadata_sha256 or compute_snapshot_metadata_sha256(meta) != meta.metadata_sha256:
                                    continue
                                if meta.dataset_id != dataset_id or meta.snapshot_version != version_dir.name:
                                    continue
                            snapshots.append(meta)
                    except Exception:
                        continue
        return snapshots

    def plan_snapshot(
        self,
        dataset_id: str,
        snapshot_version: str = "candidate_plan",
        governance_version: Optional[str] = None,
        caption_review_version: Optional[str] = None,
        latent_cache_version: str = "vae_sd_mse_square256_v001",
        text_cache_version: str = "clip_b32_v001",
        caption_version: str = "captions_v002",
    ) -> DatasetSnapshotCandidate:
        """Evaluates dataset eligibility and builds a candidate snapshot plan (read-only).

        Args:
            dataset_id: Dataset identifier.
            snapshot_version: Target snapshot version name.
            governance_version: Governance version to evaluate.
            caption_review_version: Caption review version to evaluate.
            latent_cache_version: Latent cache subdirectory name.
            text_cache_version: Text embedding cache subdirectory name.
            caption_version: Caption subdirectory name.

        Returns:
            DatasetSnapshotCandidate: Candidate inspection report with admitted records.
        """
        ds_dir = self.dataset_root / dataset_id
        if not ds_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {ds_dir}")

        dataset = PairedLatentTextDataset(
            dataset_dir=ds_dir,
            latent_cache_version=latent_cache_version,
            text_cache_version=text_cache_version,
            caption_version=caption_version,
            governance_version=governance_version,
            caption_review_version=caption_review_version,
            governance_mode=GovernanceMode.DEVELOPMENT_AUDIT,
        )

        candidate_records: List[DatasetSnapshotRecord] = []
        for img_id in sorted(dataset.paired_image_ids):
            decision = dataset.training_eligibility(img_id)
            if decision.training_allowed:
                cap_rec = dataset.caption_records[img_id]
                lat_rec = dataset.latent_records[img_id]
                text_rec = dataset.text_records[img_id]

                rec = DatasetSnapshotRecord(
                    sample_id=img_id,
                    dataset_id=dataset_id,
                    snapshot_version=snapshot_version,
                    caption=cap_rec.caption,
                    caption_sha256=decision.caption_sha256 or cap_rec.caption_sha256,
                    caption_version=caption_version,
                    caption_review_version=caption_review_version or "",
                    governance_version=governance_version or "",
                    latent_relative_path=lat_rec.latent_relative_path,
                    latent_sha256=lat_rec.latent_sha256,
                    latent_shape=lat_rec.latent_shape,
                    latent_cache_version=latent_cache_version,
                    text_embedding_relative_path=text_rec.embedding_relative_path,
                    text_embedding_sha256=text_rec.embedding_sha256,
                    text_embedding_shape=text_rec.embedding_shape,
                    text_cache_version=text_cache_version,
                    eligibility_policy_version=decision.policy_version,
                )
                rec.record_sha256 = compute_snapshot_record_sha256(rec)
                candidate_records.append(rec)

        summary = dataset.eligibility_summary
        prov = dataset.eligibility_provenance

        return DatasetSnapshotCandidate(
            dataset_id=dataset_id,
            snapshot_version=snapshot_version,
            total_samples=summary["total_samples"],
            eligible_count=len(candidate_records),
            ineligible_count=summary["total_samples"] - len(candidate_records),
            reason_counts=summary["reason_counts"],
            records=candidate_records,
            governance_version=governance_version,
            governance_manifest_sha256=prov["governance_manifest_sha256"],
            caption_review_version=caption_review_version,
            caption_review_manifest_sha256=prov["caption_review_manifest_sha256"],
            eligibility_policy_version=dataset.eligibility_evaluator.policy_version,
            latent_cache_version=latent_cache_version,
            text_cache_version=text_cache_version,
            caption_version=caption_version,
            can_freeze=(len(candidate_records) > 0 and governance_version is not None and caption_review_version is not None),
        )

    def freeze_snapshot(
        self,
        dataset_id: str,
        snapshot_version: str,
        governance_version: str,
        caption_review_version: str,
        created_by: str,
        creation_source: str,
        latent_cache_version: str = "vae_sd_mse_square256_v001",
        text_cache_version: str = "clip_b32_v001",
        caption_version: str = "captions_v002",
        previous_snapshot_version: Optional[str] = None,
        notes: str = "",
        _allow_test_overwrite: bool = False,
    ) -> DatasetSnapshot:
        """Evaluates dataset eligibility, binds exact sample identities, and freezes an immutable snapshot.

        Args:
            dataset_id: Dataset identifier.
            snapshot_version: Target immutable snapshot version identifier.
            governance_version: Mandatory explicit governance version.
            caption_review_version: Mandatory explicit caption review version.
            created_by: Mandatory explicit human or system identifier.
            creation_source: Mandatory explicit operational source context.
            latent_cache_version: Latent cache version to freeze.
            text_cache_version: Text embedding cache version to freeze.
            caption_version: Caption version to freeze.
            previous_snapshot_version: Optional previous snapshot version in lineage.
            notes: Optional contextual notes.
            _allow_test_overwrite: Private hook for test isolation only.

        Returns:
            DatasetSnapshot: Loaded and verified immutable snapshot.

        Raises:
            FileExistsError: If the snapshot version is already frozen and _allow_test_overwrite is False.
            ValueError: If attribution is invalid, evidence versions missing, or 0 eligible samples exist.
        """
        # 1. Validate version and attribution contracts
        if not isinstance(snapshot_version, str) or not snapshot_version.strip() or snapshot_version.strip().lower() in ("latest", "current"):
            raise ValueError(f"Invalid snapshot_version '{snapshot_version}'. Must be an explicit version identifier.")

        if not isinstance(governance_version, str) or not governance_version.strip():
            raise ValueError("Production snapshot freeze requires an explicit governance_version.")

        if not isinstance(caption_review_version, str) or not caption_review_version.strip():
            raise ValueError("Production snapshot freeze requires an explicit caption_review_version.")

        # 2. Check for existing frozen snapshot
        target_dir = self.get_snapshot_dir(dataset_id, snapshot_version)
        if target_dir.exists() and not _allow_test_overwrite:
            raise FileExistsError(
                f"Snapshot version '{snapshot_version}' already exists for dataset '{dataset_id}'. "
                f"Frozen snapshots are strictly immutable and cannot be overwritten."
            )

        # 3. Plan candidate
        candidate = self.plan_snapshot(
            dataset_id=dataset_id,
            snapshot_version=snapshot_version,
            governance_version=governance_version,
            caption_review_version=caption_review_version,
            latent_cache_version=latent_cache_version,
            text_cache_version=text_cache_version,
            caption_version=caption_version,
        )

        # 4. Fail-closed on zero eligible samples
        if candidate.eligible_count == 0:
            raise ValueError(
                f"Cannot freeze snapshot '{snapshot_version}': dataset '{dataset_id}' has 0 eligible samples "
                f"(total: {candidate.total_samples}, ineligible: {candidate.ineligible_count}). "
                f"Rejection breakdown: {candidate.reason_counts}."
            )

        if not candidate.governance_manifest_sha256:
            raise ValueError(f"Governance manifest for version '{governance_version}' could not be resolved.")

        if not candidate.caption_review_manifest_sha256:
            raise ValueError(f"Caption review manifest for version '{caption_review_version}' could not be resolved.")

        # 5. Atomic write to staging directory with canonical serialization
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(tempfile.mkdtemp(prefix=f"snapshot_{snapshot_version}_", dir=target_dir.parent))

        try:
            manifest_path = staging_dir / "manifest.jsonl"
            with open(manifest_path, "w", encoding="utf-8", newline="\n") as f:
                for rec in candidate.records:
                    f.write(serialize_snapshot_record(rec) + "\n")

            manifest_sha256 = compute_sha256(manifest_path)
            created_at_utc = datetime.now(timezone.utc).isoformat()

            metadata = DatasetSnapshotMetadata(
                dataset_id=dataset_id,
                snapshot_version=snapshot_version,
                status=SnapshotStatus.FROZEN.value,
                sample_count=len(candidate.records),
                created_at=created_at_utc,
                created_by=created_by.strip(),
                creation_source=creation_source.strip(),
                governance_version=governance_version,
                governance_manifest_sha256=candidate.governance_manifest_sha256,
                caption_review_version=caption_review_version,
                caption_review_manifest_sha256=candidate.caption_review_manifest_sha256,
                eligibility_policy_version=candidate.eligibility_policy_version,
                snapshot_manifest_sha256=manifest_sha256,
                latent_cache_version=latent_cache_version,
                text_cache_version=text_cache_version,
                caption_version=caption_version,
                previous_snapshot_version=previous_snapshot_version,
                notes=notes,
            )
            metadata.metadata_sha256 = compute_snapshot_metadata_sha256(metadata)

            meta_path = staging_dir / "metadata.json"
            with open(meta_path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)
                f.write("\n")

            # 6. Commit atomic directory swap
            if target_dir.exists() and _allow_test_overwrite:
                import shutil
                shutil.rmtree(target_dir)

            staging_dir.rename(target_dir)

        except Exception:
            if staging_dir.exists():
                import shutil
                shutil.rmtree(staging_dir, ignore_errors=True)
            raise

        # 7. Return verified loaded snapshot
        return self.load_snapshot(dataset_id, snapshot_version, verify_integrity=True)

    def load_snapshot(
        self,
        dataset_id: str,
        snapshot_version: str,
        verify_integrity: bool = True,
    ) -> DatasetSnapshot:
        """Loads a frozen dataset snapshot with deterministic integrity and coherence validation.

        Args:
            dataset_id: Expected dataset identifier.
            snapshot_version: Expected snapshot version identifier.
            verify_integrity: If True, validates metadata SHA, manifest SHA, record SHAs,
                path coherence, FROZEN status, sample uniqueness, and canonical ordering.

        Returns:
            DatasetSnapshot: Loaded and verified snapshot.

        Raises:
            FileNotFoundError: If snapshot files or directories do not exist.
            ValueError: If integrity verification or coherence check fails (tampering or mismatch detected).
        """
        snapshot_dir = self.get_snapshot_dir(dataset_id, snapshot_version)
        if not snapshot_dir.exists():
            raise FileNotFoundError(f"Snapshot directory not found: {snapshot_dir}")

        meta_path = snapshot_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Snapshot metadata not found: {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)
        metadata = DatasetSnapshotMetadata(**meta_data)

        # Path / Metadata identity coherence (Issues 3 & 4)
        if metadata.dataset_id != dataset_id:
            raise ValueError(
                f"Snapshot metadata dataset_id '{metadata.dataset_id}' does not match requested dataset_id '{dataset_id}' "
                f"at path '{snapshot_dir}'."
            )

        if metadata.snapshot_version != snapshot_version:
            raise ValueError(
                f"Snapshot metadata snapshot_version '{metadata.snapshot_version}' does not match requested "
                f"snapshot_version '{snapshot_version}' at path '{snapshot_dir}'."
            )

        if metadata.status != SnapshotStatus.FROZEN.value:
            raise ValueError(
                f"Snapshot '{snapshot_version}' has status '{metadata.status}', expected 'FROZEN' for training usage."
            )

        if verify_integrity:
            # Issue 1: Verify metadata_sha256 BEFORE trusting metadata provenance fields
            if not metadata.metadata_sha256 or not metadata.metadata_sha256.strip():
                raise ValueError(
                    f"Snapshot integrity error for '{snapshot_version}': metadata_sha256 is missing or empty."
                )

            actual_metadata_sha = compute_snapshot_metadata_sha256(metadata)
            if actual_metadata_sha != metadata.metadata_sha256:
                raise ValueError(
                    f"Snapshot integrity error for '{snapshot_version}': metadata SHA-256 '{actual_metadata_sha}' "
                    f"does not match recorded metadata_sha256 '{metadata.metadata_sha256}'. "
                    f"Snapshot metadata has been modified or corrupted."
                )

        manifest_path = snapshot_dir / "manifest.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Snapshot manifest not found: {manifest_path}")

        if verify_integrity:
            actual_manifest_sha = compute_sha256(manifest_path)
            if actual_manifest_sha != metadata.snapshot_manifest_sha256:
                raise ValueError(
                    f"Snapshot integrity error for '{snapshot_version}': manifest SHA-256 '{actual_manifest_sha}' "
                    f"does not match recorded metadata SHA '{metadata.snapshot_manifest_sha256}'. "
                    f"Snapshot manifest has been modified or corrupted."
                )

        records: List[DatasetSnapshotRecord] = []
        seen_sample_ids = set()
        sample_ids_order = []

        with open(manifest_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                if line.strip():
                    rec_dict = json.loads(line)
                    rec = DatasetSnapshotRecord(**rec_dict)

                    if verify_integrity:
                        # Record-to-snapshot coherence (Issue 5)
                        if rec.dataset_id != metadata.dataset_id:
                            raise ValueError(
                                f"Snapshot record coherence error for '{snapshot_version}' line {line_idx}: "
                                f"sample '{rec.sample_id}' has dataset_id '{rec.dataset_id}' "
                                f"which differs from snapshot metadata dataset_id '{metadata.dataset_id}'."
                            )

                        if rec.snapshot_version != metadata.snapshot_version:
                            raise ValueError(
                                f"Snapshot record coherence error for '{snapshot_version}' line {line_idx}: "
                                f"sample '{rec.sample_id}' has snapshot_version '{rec.snapshot_version}' "
                                f"which differs from snapshot metadata snapshot_version '{metadata.snapshot_version}'."
                            )

                        # Uniqueness check (Issue 5)
                        if rec.sample_id in seen_sample_ids:
                            raise ValueError(
                                f"Snapshot record integrity error for '{snapshot_version}' line {line_idx}: "
                                f"duplicate sample_id '{rec.sample_id}' detected."
                            )
                        seen_sample_ids.add(rec.sample_id)
                        sample_ids_order.append(rec.sample_id)

                        # Record SHA checksum
                        actual_rec_sha = compute_snapshot_record_sha256(rec)
                        if actual_rec_sha != rec.record_sha256:
                            raise ValueError(
                                f"Snapshot integrity error for '{snapshot_version}' line {line_idx}: sample '{rec.sample_id}' "
                                f"record SHA-256 mismatch (actual: '{actual_rec_sha}', recorded: '{rec.record_sha256}'). "
                                f"Sample record has been modified."
                            )
                    else:
                        seen_sample_ids.add(rec.sample_id)
                        sample_ids_order.append(rec.sample_id)

                    records.append(rec)

        if verify_integrity:
            # Canonical ordering verification (Issue 5)
            if sample_ids_order != sorted(sample_ids_order):
                raise ValueError(
                    f"Snapshot record ordering error for '{snapshot_version}': sample IDs in manifest are not in "
                    f"canonical sorted order (expected: {sorted(sample_ids_order)}, found: {sample_ids_order})."
                )

            if len(records) != metadata.sample_count:
                raise ValueError(
                    f"Snapshot integrity error for '{snapshot_version}': loaded {len(records)} sample(s) "
                    f"but metadata sample_count is {metadata.sample_count}."
                )

        return DatasetSnapshot(
            metadata=metadata,
            records=records,
            manifest_path=manifest_path,
            metadata_path=meta_path,
        )


