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
