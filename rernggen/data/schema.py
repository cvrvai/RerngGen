"""Typed schema definitions for RerngGen dataset management."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ManifestRecord:
    """Schema for a single verified dataset image record in manifest.jsonl."""

    image_id: str
    dataset_id: str
    dataset_version: str
    original_filename: str
    source_relative_path: str
    stored_relative_path: str
    sha256: str
    width: int
    height: int
    format: str
    mode: str
    caption: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    source: str = "local_owned_dataset"
    license_id: Optional[str] = None
    training_allowed: Optional[bool] = None
    commercial_allowed: Optional[bool] = None
    split: Optional[str] = None
    status: str = "IMPORTED"

    def to_dict(self) -> Dict[str, Any]:
        """Converts dataclass record to a JSON-serializable dictionary."""
        return asdict(self)


@dataclass
class ProcessedRecord:
    """Schema for a single preprocessed derivative image record."""

    image_id: str
    dataset_id: str
    preprocessing_version: str
    source_sha256: str
    processed_sha256: str
    original_width: int
    original_height: int
    resized_width: int
    resized_height: int
    crop_left: int
    crop_top: int
    crop_width: int
    crop_height: int
    output_width: int
    output_height: int
    output_mode: str
    output_relative_path: str
    training_allowed: Optional[bool] = None
    commercial_allowed: Optional[bool] = None
    license_id: Optional[str] = None
    status: str = "PROCESSED"

    def to_dict(self) -> Dict[str, Any]:
        """Converts dataclass record to a JSON-serializable dictionary."""
        return asdict(self)


@dataclass
class ImportReport:
    """Summary report produced upon completion of a dataset import operation."""

    source_dir: Path
    dataset_id: str
    dataset_root: Path
    manifest_path: Path
    originals_path: Path
    files_discovered: int = 0
    supported_candidates: int = 0
    valid_images: int = 0
    imported_images: int = 0
    duplicate_images_skipped: int = 0
    corrupt_images: int = 0
    unsupported_files: int = 0
    total_bytes_copied: int = 0
    elapsed_time_seconds: float = 0.0
    duplicate_details: List[Dict[str, str]] = field(default_factory=list)
    corrupt_details: List[Dict[str, str]] = field(default_factory=list)

    def summary(self) -> str:
        """Generates a human-readable formatted summary string."""
        bytes_formatted = self._format_bytes(self.total_bytes_copied)
        return (
            "============================================================\n"
            "DATASET IMPORT COMPLETE\n"
            "============================================================\n"
            f"Dataset ID:           {self.dataset_id}\n"
            f"Source Directory:     {self.source_dir}\n"
            f"Manifest Path:        {self.manifest_path}\n"
            f"Originals Path:       {self.originals_path}\n"
            f"Files Discovered:     {self.files_discovered}\n"
            f"Supported Candidates: {self.supported_candidates}\n"
            f"Valid Images:         {self.valid_images}\n"
            f"Imported Images:      {self.imported_images}\n"
            f"Duplicates Skipped:   {self.duplicate_images_skipped}\n"
            f"Corrupt Images:       {self.corrupt_images}\n"
            f"Unsupported Files:    {self.unsupported_files}\n"
            f"Total Bytes Copied:   {bytes_formatted}\n"
            f"Elapsed Time:         {self.elapsed_time_seconds:.2f}s\n"
            "============================================================"
        )

    @staticmethod
    def _format_bytes(size: int) -> str:
        """Formats bytes into human readable binary unit string."""
        if size < 1024:
            return f"{size} B"
        elif size < 1024**2:
            return f"{size / 1024:.2f} KB"
        elif size < 1024**3:
            return f"{size / (1024**2):.2f} MB"
        else:
            return f"{size / (1024**3):.2f} GB"


@dataclass
class PreprocessingReport:
    """Summary report produced upon completion of an image preprocessing operation."""

    dataset_id: str
    preprocessing_version: str
    target_size: int
    processed_dir: Path
    manifest_path: Path
    total_images_in_dataset: int = 0
    images_processed: int = 0
    images_skipped_idempotent: int = 0
    failures: int = 0
    total_output_bytes: int = 0
    elapsed_time_seconds: float = 0.0
    records: List[ProcessedRecord] = field(default_factory=list)
    failure_details: List[Dict[str, str]] = field(default_factory=list)

    def summary(self) -> str:
        """Generates a human-readable formatted summary string."""
        bytes_formatted = ImportReport._format_bytes(self.total_output_bytes)
        return (
            "============================================================\n"
            "DATASET PREPROCESSING COMPLETE\n"
            "============================================================\n"
            f"Dataset ID:            {self.dataset_id}\n"
            f"Preprocessing Version: {self.preprocessing_version}\n"
            f"Target Size:           {self.target_size}x{self.target_size}\n"
            f"Processed Directory:   {self.processed_dir}\n"
            f"Manifest Path:         {self.manifest_path}\n"
            f"Total Originals:       {self.total_images_in_dataset}\n"
            f"Images Processed:      {self.images_processed}\n"
            f"Skipped (Idempotent):  {self.images_skipped_idempotent}\n"
            f"Failures:              {self.failures}\n"
            f"Total Output Bytes:    {bytes_formatted}\n"
            f"Elapsed Time:          {self.elapsed_time_seconds:.2f}s\n"
            "============================================================"
        )


@dataclass
class LatentRecord:
    """Schema for a single cached VAE latent tensor record in cache manifest.jsonl."""

    image_id: str
    dataset_id: str
    dataset_version: str
    source_processed_sha256: str
    preprocessing_version: str
    vae_model_id: str
    vae_revision: str
    vae_weights_sha256: str
    vae_config_sha256: str
    vae_scaling_factor: float
    posterior_policy: str
    latent_shape: List[int]
    latent_dtype: str
    latent_sha256: str
    latent_relative_path: str
    min_val: float
    max_val: float
    mean_val: float
    std_val: float
    l2_norm: float
    training_allowed: Optional[bool] = None
    commercial_allowed: Optional[bool] = None
    license_id: Optional[str] = None
    cache_version: str = "vae_sd_mse_square256_v001"
    status: str = "CACHED"

    def to_dict(self) -> Dict[str, Any]:
        """Converts dataclass record to a JSON-serializable dictionary."""
        return asdict(self)


@dataclass
class LatentCacheReport:
    """Summary report produced upon completion of latent cache generation."""

    dataset_id: str
    preprocessing_version: str
    cache_version: str
    cache_dir: Path
    manifest_path: Path
    total_images_in_dataset: int = 0
    latents_created: int = 0
    valid_cache_hits: int = 0
    failures: int = 0
    total_cache_bytes: int = 0
    elapsed_time_seconds: float = 0.0
    vae_provenance: Dict[str, Any] = field(default_factory=dict)
    records: List[LatentRecord] = field(default_factory=list)
    failure_details: List[Dict[str, str]] = field(default_factory=list)

    def summary(self) -> str:
        """Generates a human-readable formatted summary string."""
        bytes_formatted = ImportReport._format_bytes(self.total_cache_bytes)
        return (
            "============================================================\n"
            "LATENT CACHE GENERATION COMPLETE\n"
            "============================================================\n"
            f"Dataset ID:            {self.dataset_id}\n"
            f"Preprocessing Version: {self.preprocessing_version}\n"
            f"Cache Version:         {self.cache_version}\n"
            f"Cache Directory:       {self.cache_dir}\n"
            f"Manifest Path:         {self.manifest_path}\n"
            f"Total Images:          {self.total_images_in_dataset}\n"
            f"Latents Created:       {self.latents_created}\n"
            f"Valid Cache Hits:      {self.valid_cache_hits}\n"
            f"Failures:              {self.failures}\n"
            f"Total Cache Bytes:     {bytes_formatted}\n"
            f"Elapsed Time:          {self.elapsed_time_seconds:.2f}s\n"
            "============================================================"
        )


@dataclass
class CaptionRecord:
    """Schema for a single caption entry in captions manifest.jsonl."""

    image_id: str
    dataset_id: str
    caption: str
    caption_source: str = "manual"
    caption_version: str = "captions_v001"
    caption_sha256: str = ""
    language: str = "en"
    review_status: str = "reviewed"
    training_allowed: Optional[bool] = None
    commercial_allowed: Optional[bool] = None
    license_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts dataclass record to a JSON-serializable dictionary."""
        return asdict(self)


@dataclass
class TextEmbeddingRecord:
    """Schema for a single frozen text encoder pooled embedding in cache manifest.jsonl."""

    image_id: str
    dataset_id: str
    dataset_version: str
    caption_version: str
    caption_sha256: str
    text_encoder_id: str
    text_encoder_revision: str
    text_encoder_weights_sha256: str
    text_encoder_config_sha256: str
    tokenizer_class: str
    tokenizer_config_sha256: str = ""
    vocab_sha256: str = ""
    merges_sha256: str = ""
    special_tokens_map_sha256: str = ""
    tokenizer_identity_sha256: str = ""
    max_token_length: int = 77
    pooling_policy: str = "eos_token"
    embedding_shape: List[int] = field(default_factory=list)
    embedding_dtype: str = "float32"
    embedding_sha256: str = ""
    embedding_relative_path: str = ""
    min_val: float = 0.0
    max_val: float = 0.0
    mean_val: float = 0.0
    std_val: float = 0.0
    l2_norm: float = 0.0
    token_count: int = 0
    truncated: bool = False
    training_allowed: Optional[bool] = None
    commercial_allowed: Optional[bool] = None
    license_id: Optional[str] = None
    cache_version: str = "clip_b32_v001"
    status: str = "CACHED"

    def to_dict(self) -> Dict[str, Any]:
        """Converts dataclass record to a JSON-serializable dictionary."""
        return asdict(self)


@dataclass
class TextEmbeddingCacheReport:
    """Summary report produced upon completion of frozen text embedding cache generation."""

    dataset_id: str
    caption_version: str
    cache_version: str
    cache_dir: Path
    manifest_path: Path
    total_captions_in_dataset: int = 0
    embeddings_created: int = 0
    valid_cache_hits: int = 0
    failures: int = 0
    total_cache_bytes: int = 0
    elapsed_time_seconds: float = 0.0
    text_encoder_provenance: Dict[str, Any] = field(default_factory=dict)
    records: List[TextEmbeddingRecord] = field(default_factory=list)
    failure_details: List[Dict[str, str]] = field(default_factory=list)

    def summary(self) -> str:
        """Generates a human-readable formatted summary string."""
        bytes_formatted = ImportReport._format_bytes(self.total_cache_bytes)
        return (
            "============================================================\n"
            "TEXT EMBEDDING CACHE GENERATION COMPLETE\n"
            "============================================================\n"
            f"Dataset ID:            {self.dataset_id}\n"
            f"Caption Version:       {self.caption_version}\n"
            f"Cache Version:         {self.cache_version}\n"
            f"Cache Directory:       {self.cache_dir}\n"
            f"Manifest Path:         {self.manifest_path}\n"
            f"Total Captions:        {self.total_captions_in_dataset}\n"
            f"Embeddings Created:    {self.embeddings_created}\n"
            f"Valid Cache Hits:      {self.valid_cache_hits}\n"
            f"Failures:              {self.failures}\n"
            f"Total Cache Bytes:     {bytes_formatted}\n"
            f"Elapsed Time:          {self.elapsed_time_seconds:.2f}s\n"
            "============================================================"
        )


@dataclass
class GovernanceRecord:
    """Schema for an explicit, versioned dataset item authorization record in governance manifest.jsonl."""

    image_id: str
    dataset_id: str
    governance_version: str
    training_allowed: Optional[bool]  # True (ALLOW), False (DENY), None (UNKNOWN)
    commercial_allowed: Optional[bool]  # True (ALLOW), False (DENY), None (UNKNOWN)
    authorization_source: str  # Mandatory: explicitly declared source (e.g. 'human_audit', 'license_verified')
    authorized_at: str  # Mandatory: ISO-8601 UTC timestamp string
    license_id: Optional[str] = None
    authorization_note: str = ""
    evidence_reference: Optional[str] = None
    record_sha256: str = ""
    status: str = "ACTIVE"  # "ACTIVE", "SUPERSEDED", "REVOKED"
    previous_governance_version: Optional[str] = None

    def __post_init__(self) -> None:
        """Validates record fields upon instantiation."""
        if not isinstance(self.authorization_source, str) or not self.authorization_source.strip():
            raise ValueError(
                f"GovernanceRecord for '{self.image_id}' must have a non-empty authorization_source."
            )
        if not isinstance(self.authorized_at, str) or not self.authorized_at.strip():
            raise ValueError(
                f"GovernanceRecord for '{self.image_id}' must have a non-empty authorized_at timestamp."
            )
        if self.status not in ("ACTIVE", "SUPERSEDED", "REVOKED"):
            raise ValueError(
                f"Invalid governance status '{self.status}'. Must be 'ACTIVE', 'SUPERSEDED', or 'REVOKED'."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Converts dataclass record to a JSON-serializable dictionary."""
        return asdict(self)


@dataclass
class CaptionReviewRecord:
    """Schema for a versioned human caption review and quality acceptance record."""

    image_id: str
    dataset_id: str
    review_version: str
    caption_sha256: str
    review_status: str  # "PENDING", "APPROVED", "REJECTED", "INVALIDATED"
    reviewed_by: str  # Mandatory reviewer identifier (e.g. 'reviewer_alice')
    review_source: str  # Mandatory origin (e.g. 'human_audit', 'batch_review_01')
    reviewed_at: str  # Mandatory ISO-8601 UTC timestamp string
    reason: str = ""
    record_sha256: str = ""
    previous_review_version: Optional[str] = None

    def __post_init__(self) -> None:
        """Validates caption review record fields upon instantiation."""
        if not isinstance(self.reviewed_by, str) or not self.reviewed_by.strip():
            raise ValueError(
                f"CaptionReviewRecord for '{self.image_id}' must have a non-empty reviewed_by."
            )
        if not isinstance(self.review_source, str) or not self.review_source.strip():
            raise ValueError(
                f"CaptionReviewRecord for '{self.image_id}' must have a non-empty review_source."
            )
        if not isinstance(self.reviewed_at, str) or not self.reviewed_at.strip():
            raise ValueError(
                f"CaptionReviewRecord for '{self.image_id}' must have a non-empty reviewed_at timestamp."
            )
        if not isinstance(self.caption_sha256, str) or not self.caption_sha256.strip():
            raise ValueError(
                f"CaptionReviewRecord for '{self.image_id}' must have a non-empty caption_sha256."
            )
        if self.review_status not in ("PENDING", "APPROVED", "REJECTED", "INVALIDATED"):
            raise ValueError(
                f"Invalid caption review status '{self.review_status}'. "
                f"Must be 'PENDING', 'APPROVED', 'REJECTED', or 'INVALIDATED'."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Converts dataclass record to a JSON-serializable dictionary."""
        return asdict(self)


DUMMY_ATTRIBUTION_VALUES = frozenset({"human", "system", "manual", "unknown", "human_declared"})


@dataclass
class DatasetSnapshotRecord:
    """Schema for a single frozen sample in an immutable dataset snapshot manifest."""

    sample_id: str
    dataset_id: str
    snapshot_version: str
    caption: str
    caption_sha256: str
    caption_version: str
    caption_review_version: str
    governance_version: str
    latent_relative_path: str
    latent_sha256: str
    latent_shape: List[int]
    latent_cache_version: str
    text_embedding_relative_path: str
    text_embedding_sha256: str
    text_embedding_shape: List[int]
    text_cache_version: str
    eligibility_policy_version: str
    record_sha256: str = ""

    def __post_init__(self) -> None:
        """Validates snapshot record fields upon instantiation."""
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ValueError("DatasetSnapshotRecord must have a non-empty sample_id.")
        if not isinstance(self.caption_sha256, str) or not self.caption_sha256.strip():
            raise ValueError(f"DatasetSnapshotRecord for '{self.sample_id}' must have a valid caption_sha256.")
        if not isinstance(self.latent_sha256, str) or not self.latent_sha256.strip():
            raise ValueError(f"DatasetSnapshotRecord for '{self.sample_id}' must have a valid latent_sha256.")
        if not isinstance(self.text_embedding_sha256, str) or not self.text_embedding_sha256.strip():
            raise ValueError(f"DatasetSnapshotRecord for '{self.sample_id}' must have a valid text_embedding_sha256.")

    def to_dict(self) -> Dict[str, Any]:
        """Converts dataclass record to a JSON-serializable dictionary."""
        return asdict(self)


@dataclass
class DatasetSnapshotMetadata:
    """Schema for top-level metadata and cryptographic provenance of a dataset snapshot."""

    dataset_id: str
    snapshot_version: str
    status: str  # "DRAFT", "FROZEN", "SUPERSEDED"
    sample_count: int
    created_at: str  # ISO-8601 UTC timestamp
    created_by: str  # Mandatory explicit creator identifier
    creation_source: str  # Mandatory explicit origin
    governance_version: str
    governance_manifest_sha256: str
    caption_review_version: str
    caption_review_manifest_sha256: str
    eligibility_policy_version: str
    snapshot_manifest_sha256: str
    latent_cache_version: str = "vae_sd_mse_square256_v001"
    text_cache_version: str = "clip_b32_v001"
    caption_version: str = "captions_v002"
    previous_snapshot_version: Optional[str] = None
    notes: str = ""
    metadata_sha256: str = ""

    def __post_init__(self) -> None:
        """Validates snapshot metadata fields upon instantiation."""
        if not isinstance(self.snapshot_version, str) or not self.snapshot_version.strip() or self.snapshot_version.strip().lower() in ("latest", "current"):
            raise ValueError(f"DatasetSnapshotMetadata requires an explicit snapshot_version (cannot be '{self.snapshot_version}').")
        if not isinstance(self.created_by, str) or not self.created_by.strip() or self.created_by.strip().lower() in DUMMY_ATTRIBUTION_VALUES:
            raise ValueError(f"DatasetSnapshotMetadata requires an explicit, non-dummy created_by identifier (got '{self.created_by}').")
        if not isinstance(self.creation_source, str) or not self.creation_source.strip() or self.creation_source.strip().lower() in DUMMY_ATTRIBUTION_VALUES:
            raise ValueError(f"DatasetSnapshotMetadata requires an explicit, non-dummy creation_source identifier (got '{self.creation_source}').")
        if not isinstance(self.created_at, str) or not self.created_at.strip():
            raise ValueError("DatasetSnapshotMetadata requires a non-empty ISO-8601 created_at timestamp.")
        if self.status not in ("DRAFT", "FROZEN", "SUPERSEDED"):
            raise ValueError(f"Invalid snapshot status '{self.status}'. Must be 'DRAFT', 'FROZEN', or 'SUPERSEDED'.")

    def to_dict(self) -> Dict[str, Any]:
        """Converts dataclass record to a JSON-serializable dictionary."""
        return asdict(self)






