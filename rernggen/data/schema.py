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
