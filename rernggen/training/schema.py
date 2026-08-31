"""Typed schema definitions for RerngGen training run provenance and state management."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import enum
from typing import Any, Dict, Optional


def utcnow_iso() -> str:
    """Returns current UTC timestamp in ISO 8601 format with Z suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class TrainingRunStatus(str, enum.Enum):
    """Lifecycle states for a training run."""

    PLANNED = "PLANNED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"


@dataclass
class TrainingRunSpec:
    """Immutable experiment identity and specification for a training run."""

    run_id: str
    dataset_id: str
    snapshot_version: str
    snapshot_metadata_sha256: str
    model_family: str
    model_config: Dict[str, Any]
    training_config: Dict[str, Any]
    seed: int
    git_commit: Optional[str] = None
    git_dirty: Optional[bool] = None
    git_branch: Optional[str] = None
    created_at: str = field(default_factory=utcnow_iso)
    created_by: str = ""
    creation_source: str = ""
    notes: Optional[str] = None
    parent_run_id: Optional[str] = None
    experiment_name: Optional[str] = None
    model_config_sha256: Optional[str] = None
    training_config_sha256: Optional[str] = None
    run_spec_sha256: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts dataclass to a JSON-serializable dictionary."""
        return asdict(self)


@dataclass
class TrainingRunState:
    """Mutable runtime lifecycle state for an executing or completed training run."""

    run_id: str
    status: str = TrainingRunStatus.PLANNED.value
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    current_step: int = 0
    last_checkpoint: Optional[str] = None
    failure_reason: Optional[str] = None
    updated_at: str = field(default_factory=utcnow_iso)

    def to_dict(self) -> Dict[str, Any]:
        """Converts dataclass to a JSON-serializable dictionary."""
        return asdict(self)


@dataclass
class TrainingEnvironmentRecord:
    """Captured environment telemetry and provenance for a training run execution."""

    python_version: str
    platform: str
    platform_release: str
    machine_architecture: str
    torch_version: Optional[str] = None
    cuda_version: Optional[str] = None
    device_type: str = "cpu"
    device_name: Optional[str] = None
    gpu_count: int = 0
    rernggen_version: Optional[str] = None
    dependency_lock_sha256: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts dataclass to a JSON-serializable dictionary."""
        return asdict(self)
