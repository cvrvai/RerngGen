"""RerngGen training provenance, run lifecycle, and experiment reproducibility framework."""

from rernggen.training.checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from rernggen.training.dataset import (
    SnapshotTrainingDataset,
    create_snapshot_dataloader,
    snapshot_collate_fn,
)
from rernggen.training.diffusion import DiffusionSchedule
from rernggen.training.preflight import (
    TrainingPreflightResult,
    run_preflight_checks,
)
from rernggen.training.provenance import (
    TrainingRun,
    TrainingRunManager,
    collect_environment_record,
    collect_git_provenance,
    compute_config_sha256,
    compute_training_run_spec_sha256,
    serialize_training_run_spec,
    validate_attribution,
    validate_run_id,
    validate_seed,
)
from rernggen.training.schema import (
    TrainingEnvironmentRecord,
    TrainingRunSpec,
    TrainingRunState,
    TrainingRunStatus,
    utcnow_iso,
)
from rernggen.training.smoke import run_tiny_dit_smoke, seed_everything

__all__ = [
    "TrainingRunStatus",
    "TrainingRunSpec",
    "TrainingRunState",
    "TrainingEnvironmentRecord",
    "TrainingRun",
    "TrainingRunManager",
    "validate_run_id",
    "validate_seed",
    "validate_attribution",
    "serialize_training_run_spec",
    "compute_config_sha256",
    "compute_training_run_spec_sha256",
    "collect_git_provenance",
    "collect_environment_record",
    "utcnow_iso",
    "TrainingPreflightResult",
    "run_preflight_checks",
    "SnapshotTrainingDataset",
    "create_snapshot_dataloader",
    "snapshot_collate_fn",
    "DiffusionSchedule",
    "save_training_checkpoint",
    "load_training_checkpoint",
    "run_tiny_dit_smoke",
    "seed_everything",
]
