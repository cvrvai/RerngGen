"""Authoritative preflight verification gate executed before starting any training run."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch

from rernggen.data.snapshot import DatasetSnapshotManager
from rernggen.models.dit.model import TinyDiT
from rernggen.training.dataset import SnapshotTrainingDataset, create_snapshot_dataloader
from rernggen.training.provenance import (
    TrainingRunManager,
    collect_environment_record,
    collect_git_provenance,
)
from rernggen.training.schema import TrainingRunStatus


@dataclass
class TrainingPreflightResult:
    """Structured report returned by the training preflight gate."""

    run_id: str
    passed: bool
    checks: Dict[str, bool] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    device: str = "cpu"
    sample_count: int = 0
    batch_size: int = 1
    seed: int = 0
    snapshot_metadata_sha256: str = ""
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Converts result to JSON-serializable dictionary."""
        return asdict(self)


def run_preflight_checks(
    run_id: str,
    training_root: Union[str, Path] = "training_runs",
    dataset_root: Union[str, Path] = "datasets",
    device: str = "cpu",
) -> TrainingPreflightResult:
    """Executes fail-closed preflight validation before training initialization."""
    checks: Dict[str, bool] = {
        "RUN_VERIFIED": False,
        "RUN_STATE_VALID": False,
        "SNAPSHOT_VERIFIED": False,
        "SNAPSHOT_SHA_MATCH": False,
        "DATASET_NON_EMPTY": False,
        "ARTIFACTS_LOADABLE": False,
        "MODEL_CONFIG_VALID": False,
        "TRAINING_CONFIG_VALID": False,
        "DEVICE_AVAILABLE": False,
        "BATCH_CONSTRUCTIBLE": False,
    }
    warnings: List[str] = []
    errors: List[str] = []

    # 1. Device Availability Check
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            errors.append(f"Requested device '{device}' but CUDA is not available.")
            checks["DEVICE_AVAILABLE"] = False
        else:
            checks["DEVICE_AVAILABLE"] = True
    else:
        checks["DEVICE_AVAILABLE"] = True

    # 2. Run Verification Check
    mgr = TrainingRunManager(training_root=training_root, dataset_root=dataset_root)
    try:
        run = mgr.load_verified_run(run_id, verify_integrity=True)
        checks["RUN_VERIFIED"] = True
    except Exception as e:
        errors.append(f"Training run verification failed: {e}")
        return TrainingPreflightResult(
            run_id=run_id,
            passed=False,
            checks=checks,
            warnings=warnings,
            device=device,
            errors=errors,
        )

    # 3. Run State Lifecycle Check
    if run.state.status != TrainingRunStatus.PLANNED.value:
        errors.append(
            f"Run '{run_id}' has state '{run.state.status}', expected '{TrainingRunStatus.PLANNED.value}'."
        )
        checks["RUN_STATE_VALID"] = False
    else:
        checks["RUN_STATE_VALID"] = True

    # 4. Snapshot Verification & SHA Match Check
    snap_mgr = DatasetSnapshotManager(dataset_root=dataset_root)
    try:
        snapshot = snap_mgr.load_snapshot(
            dataset_id=run.spec.dataset_id,
            snapshot_version=run.spec.snapshot_version,
            verify_integrity=True,
        )
        checks["SNAPSHOT_VERIFIED"] = True

        if snapshot.metadata.metadata_sha256 != run.spec.snapshot_metadata_sha256:
            errors.append(
                f"Snapshot metadata SHA mismatch: actual '{snapshot.metadata.metadata_sha256}' "
                f"vs spec '{run.spec.snapshot_metadata_sha256}'."
            )
            checks["SNAPSHOT_SHA_MATCH"] = False
        else:
            checks["SNAPSHOT_SHA_MATCH"] = True
    except Exception as e:
        errors.append(f"Snapshot verification failed: {e}")
        checks["SNAPSHOT_VERIFIED"] = False

    # 5. Dataset Non-Empty Check
    if checks["SNAPSHOT_VERIFIED"]:
        sample_count = len(snapshot)
        if sample_count == 0:
            errors.append(f"Referenced snapshot '{run.spec.snapshot_version}' contains 0 samples.")
            checks["DATASET_NON_EMPTY"] = False
        else:
            checks["DATASET_NON_EMPTY"] = True
    else:
        sample_count = 0

    # 6. Artifact Loadability Check
    if checks["SNAPSHOT_VERIFIED"] and checks["DATASET_NON_EMPTY"]:
        try:
            dataset = SnapshotTrainingDataset(snapshot=snapshot, dataset_root=dataset_root)
            # Verify every record in snapshot loads and matches SHA
            for idx in range(len(dataset)):
                sample = dataset[idx]
                if torch.isnan(sample["latent"]).any() or torch.isinf(sample["latent"]).any():
                    raise ValueError(f"NaN or Inf detected in latent for sample '{sample['sample_id']}'.")
                if torch.isnan(sample["text_embedding"]).any() or torch.isinf(sample["text_embedding"]).any():
                    raise ValueError(f"NaN or Inf detected in text embedding for sample '{sample['sample_id']}'.")
            checks["ARTIFACTS_LOADABLE"] = True
        except Exception as e:
            errors.append(f"Artifact integrity check failed: {e}")
            checks["ARTIFACTS_LOADABLE"] = False

    # 7. Model Config Validation Check
    try:
        test_model = TinyDiT.from_config(run.spec.model_config)
        checks["MODEL_CONFIG_VALID"] = True
    except Exception as e:
        errors.append(f"Model config instantiation failed: {e}")
        checks["MODEL_CONFIG_VALID"] = False

    # 8. Training Config Validation Check
    batch_size = run.spec.training_config.get("batch_size", 1)
    if not isinstance(batch_size, int) or batch_size <= 0:
        errors.append(f"Invalid batch_size '{batch_size}' in training_config.")
        checks["TRAINING_CONFIG_VALID"] = False
    else:
        checks["TRAINING_CONFIG_VALID"] = True

    # 9. Batch Constructibility Check
    if checks["ARTIFACTS_LOADABLE"] and checks["TRAINING_CONFIG_VALID"]:
        try:
            dl = create_snapshot_dataloader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
            )
            first_batch = next(iter(dl))
            if first_batch["latent"].shape[0] == 0:
                raise ValueError("Batch construction produced empty tensor.")
            checks["BATCH_CONSTRUCTIBLE"] = True
        except Exception as e:
            errors.append(f"Batch constructibility test failed: {e}")
            checks["BATCH_CONSTRUCTIBLE"] = False

    # 10. Execution-time Revalidation & Drift Warnings
    cur_commit, cur_dirty, _ = collect_git_provenance()
    if run.spec.git_commit is not None and cur_commit != run.spec.git_commit:
        warnings.append(
            f"Git HEAD commit drift: run spec created at '{run.spec.git_commit}', execution at '{cur_commit}'."
        )
    if run.spec.git_dirty is not None and cur_dirty != run.spec.git_dirty:
        warnings.append(
            f"Git dirty state drift: run spec created with dirty={run.spec.git_dirty}, execution dirty={cur_dirty}."
        )

    all_passed = all(checks.values()) and (len(errors) == 0)

    return TrainingPreflightResult(
        run_id=run_id,
        passed=all_passed,
        checks=checks,
        warnings=warnings,
        device=device,
        sample_count=sample_count,
        batch_size=batch_size,
        seed=run.spec.seed,
        snapshot_metadata_sha256=run.spec.snapshot_metadata_sha256,
        errors=errors,
    )
