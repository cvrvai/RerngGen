"""Deterministic, fail-closed training run provenance and experiment reproducibility management."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

import torch

import rernggen
from rernggen.data.snapshot import DatasetSnapshotManager
from rernggen.training.schema import (
    TrainingEnvironmentRecord,
    TrainingRunSpec,
    TrainingRunState,
    TrainingRunStatus,
    utcnow_iso,
)

# Reject dummy or uninformative attributions
DISALLOWED_ATTRIBUTION_VALUES = {
    "human",
    "system",
    "manual",
    "unknown",
    "human_declared",
    "default",
    "none",
    "null",
    "undefined",
    "auto",
    "anonymous",
}

RUN_ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]+$")
MAX_SEED_VALUE = (1 << 32) - 1  # 4294967295


def validate_run_id(run_id: str) -> None:
    """Validates run_id against path safety, naming conventions, and reserved keywords."""
    if not isinstance(run_id, str):
        raise ValueError(f"run_id must be a string, got {type(run_id).__name__}")
    stripped = run_id.strip()
    if not stripped:
        raise ValueError("run_id cannot be empty or whitespace")
    if run_id != stripped:
        raise ValueError(f"run_id cannot contain leading or trailing whitespace: '{run_id}'")
    if run_id.lower() in ("latest", "current", "default", "none", "null"):
        raise ValueError(f"run_id cannot be reserved keyword '{run_id}'")
    if not RUN_ID_REGEX.match(run_id):
        raise ValueError(
            f"run_id '{run_id}' contains invalid characters. Must match regex '^[a-zA-Z0-9_-]+$'."
        )


def validate_seed(seed: Any) -> int:
    """Validates that seed is an explicit, non-boolean integer within [0, 2^32 - 1]."""
    if seed is None:
        raise ValueError("Random seed is mandatory and cannot be None.")
    if type(seed) is bool:
        raise ValueError("Random seed must be an integer, got bool.")
    if not isinstance(seed, int):
        raise ValueError(f"Random seed must be an integer, got {type(seed).__name__}.")
    if seed < 0 or seed > MAX_SEED_VALUE:
        raise ValueError(f"Random seed must be in range [0, {MAX_SEED_VALUE}], got {seed}.")
    return seed


def validate_attribution(created_by: str, creation_source: str) -> None:
    """Enforces non-empty and non-dummy attribution identity."""
    if not isinstance(created_by, str) or not created_by.strip():
        raise ValueError("created_by is mandatory and cannot be empty.")
    if not isinstance(creation_source, str) or not creation_source.strip():
        raise ValueError("creation_source is mandatory and cannot be empty.")

    norm_user = created_by.strip().lower()
    norm_source = creation_source.strip().lower()

    if norm_user in DISALLOWED_ATTRIBUTION_VALUES:
        raise ValueError(
            f"created_by cannot be placeholder '{created_by}'. Please provide an explicit user/agent identifier."
        )
    if norm_source in DISALLOWED_ATTRIBUTION_VALUES:
        raise ValueError(
            f"creation_source cannot be placeholder '{creation_source}'. Please provide an explicit source description."
        )


def serialize_canonical_json(payload: Any) -> str:
    """Serializes payload to deterministic canonical JSON with sorted keys and compact separators."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_config_sha256(config_dict: Dict[str, Any]) -> str:
    """Computes deterministic SHA-256 digest of a configuration dictionary."""
    if not isinstance(config_dict, dict):
        raise ValueError(f"Config must be a dictionary, got {type(config_dict).__name__}")
    serialized = serialize_canonical_json(config_dict)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def serialize_training_run_spec(spec: Union[TrainingRunSpec, Dict[str, Any]]) -> str:
    """Serializes a TrainingRunSpec to canonical JSON string."""
    payload = spec.to_dict() if isinstance(spec, TrainingRunSpec) else dict(spec)
    return serialize_canonical_json(payload)


def compute_training_run_spec_sha256(spec: Union[TrainingRunSpec, Dict[str, Any]]) -> str:
    """Computes deterministic SHA-256 digest of immutable experiment specification fields."""
    payload = spec.to_dict() if isinstance(spec, TrainingRunSpec) else dict(spec)
    # Exclude the digest itself from calculation
    payload.pop("run_spec_sha256", None)
    serialized = serialize_canonical_json(payload)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def collect_git_provenance(
    cwd: Optional[Union[str, Path]] = None,
) -> Tuple[Optional[str], Optional[bool], Optional[str]]:
    """Safely inspects git status and HEAD commit without raising exceptions."""
    work_dir = str(cwd) if cwd else None
    git_commit: Optional[str] = None
    git_dirty: Optional[bool] = None
    git_branch: Optional[str] = None

    try:
        # Check commit
        res_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if res_commit.returncode == 0:
            commit_text = res_commit.stdout.strip()
            if commit_text and len(commit_text) >= 7:
                git_commit = commit_text

        # Check dirty state
        if git_commit is not None:
            res_status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if res_status.returncode == 0:
                git_dirty = bool(res_status.stdout.strip())

            # Check branch
            res_branch = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if res_branch.returncode == 0:
                branch_text = res_branch.stdout.strip()
                if branch_text and branch_text != "HEAD":
                    git_branch = branch_text
    except Exception:
        # Silently fall back to None if git is not installed or outside git tree
        pass

    return git_commit, git_dirty, git_branch


def collect_environment_record(
    project_root: Optional[Union[str, Path]] = None,
) -> TrainingEnvironmentRecord:
    """Collects strictly allow-listed environment telemetry without dumping environment variables."""
    # Find lock / dependency file hash
    dep_hash: Optional[str] = None
    search_root = Path(project_root) if project_root else Path.cwd()

    candidate_files = [
        "requirements.lock",
        "uv.lock",
        "poetry.lock",
        "requirements.txt",
        "pyproject.toml",
    ]
    for fname in candidate_files:
        candidate_path = search_root / fname
        if candidate_path.is_file():
            try:
                dep_hash = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
                break
            except Exception:
                pass

    # Hardware & PyTorch device telemetry
    torch_ver: Optional[str] = None
    cuda_ver: Optional[str] = None
    dev_type = "cpu"
    dev_name: Optional[str] = None
    gpu_cnt = 0

    try:
        torch_ver = torch.__version__
        cuda_ver = torch.version.cuda
        if torch.cuda.is_available():
            dev_type = "cuda"
            gpu_cnt = torch.cuda.device_count()
            if gpu_cnt > 0:
                dev_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    rerng_ver = getattr(rernggen, "__version__", None)

    return TrainingEnvironmentRecord(
        python_version=sys.version.split()[0],
        platform=platform.system(),
        platform_release=platform.release(),
        machine_architecture=platform.machine(),
        torch_version=torch_ver,
        cuda_version=cuda_ver,
        device_type=dev_type,
        device_name=dev_name,
        gpu_count=gpu_cnt,
        rernggen_version=rerng_ver,
        dependency_lock_sha256=dep_hash,
    )


@dataclass
class TrainingRun:
    """Loaded training run representation encapsulating spec, state, and environment."""

    spec: TrainingRunSpec
    state: TrainingRunState
    environment: TrainingEnvironmentRecord
    run_dir: Path

    def to_provenance_dict(self) -> Dict[str, Any]:
        """Exports compact root provenance dictionary for checkpoint embedding."""
        return {
            "run_id": self.spec.run_id,
            "run_spec_sha256": self.spec.run_spec_sha256,
            "dataset_id": self.spec.dataset_id,
            "snapshot_version": self.spec.snapshot_version,
            "snapshot_metadata_sha256": self.spec.snapshot_metadata_sha256,
            "model_family": self.spec.model_family,
            "seed": self.spec.seed,
            "git_commit": self.spec.git_commit,
            "git_dirty": self.spec.git_dirty,
        }


class TrainingRunManager:
    """Authoritative manager for training run creation, lifecycle state transitions, and verification."""

    def __init__(
        self,
        training_root: Union[str, Path] = "training_runs",
        dataset_root: Union[str, Path] = "datasets",
    ) -> None:
        self.training_root = Path(training_root)
        self.dataset_root = Path(dataset_root)
        self.snapshot_manager = DatasetSnapshotManager(dataset_root=self.dataset_root)

    def get_run_dir(self, run_id: str) -> Path:
        """Resolves run directory safely within training_root."""
        validate_run_id(run_id)
        run_path = self.training_root / run_id
        # Defense against path traversal
        resolved_root = self.training_root.resolve()
        resolved_path = run_path.resolve()
        try:
            resolved_path.relative_to(resolved_root)
        except ValueError:
            raise ValueError(f"Run ID '{run_id}' resolves outside training root directory.")
        return run_path

    def create_run(
        self,
        run_id: str,
        dataset_id: str,
        snapshot_version: str,
        model_family: str,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
        seed: int,
        created_by: str,
        creation_source: str,
        expected_snapshot_metadata_sha256: Optional[str] = None,
        notes: Optional[str] = None,
        parent_run_id: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ) -> TrainingRun:
        """Creates an immutable training run specification from a verified frozen dataset snapshot."""
        validate_run_id(run_id)
        valid_seed = validate_seed(seed)
        validate_attribution(created_by, creation_source)

        if not isinstance(model_family, str) or not model_family.strip():
            raise ValueError("model_family is mandatory and cannot be empty.")
        if not isinstance(model_config, dict):
            raise ValueError(f"model_config must be a dict, got {type(model_config).__name__}")
        if not isinstance(training_config, dict):
            raise ValueError(f"training_config must be a dict, got {type(training_config).__name__}")

        run_dir = self.get_run_dir(run_id)
        if run_dir.exists():
            raise FileExistsError(f"Training run '{run_id}' already exists at '{run_dir}'.")

        # 1. Load and cryptographically verify the referenced frozen dataset snapshot
        snapshot = self.snapshot_manager.load_snapshot(
            dataset_id=dataset_id,
            snapshot_version=snapshot_version,
            verify_integrity=True,
        )

        if snapshot.metadata.status != "FROZEN":
            raise ValueError(
                f"Referenced snapshot '{snapshot_version}' has status '{snapshot.metadata.status}', expected 'FROZEN'."
            )

        actual_snap_meta_sha = snapshot.metadata.metadata_sha256
        if not actual_snap_meta_sha:
            raise ValueError(f"Snapshot '{snapshot_version}' has missing or empty metadata_sha256.")

        if expected_snapshot_metadata_sha256 is not None:
            if actual_snap_meta_sha != expected_snapshot_metadata_sha256:
                raise ValueError(
                    f"Snapshot metadata SHA mismatch for '{snapshot_version}': "
                    f"actual '{actual_snap_meta_sha}' vs expected '{expected_snapshot_metadata_sha256}'."
                )

        # 2. Collect git and environment provenance
        git_commit, git_dirty, git_branch = collect_git_provenance()
        env_record = collect_environment_record()

        # 3. Construct immutable TrainingRunSpec
        model_cfg_sha = compute_config_sha256(model_config)
        train_cfg_sha = compute_config_sha256(training_config)

        spec = TrainingRunSpec(
            run_id=run_id,
            dataset_id=dataset_id,
            snapshot_version=snapshot_version,
            snapshot_metadata_sha256=actual_snap_meta_sha,
            model_family=model_family.strip(),
            model_config=model_config,
            training_config=training_config,
            seed=valid_seed,
            git_commit=git_commit,
            git_dirty=git_dirty,
            git_branch=git_branch,
            created_at=utcnow_iso(),
            created_by=created_by.strip(),
            creation_source=creation_source.strip(),
            notes=notes,
            parent_run_id=parent_run_id,
            experiment_name=experiment_name,
            model_config_sha256=model_cfg_sha,
            training_config_sha256=train_cfg_sha,
        )
        spec.run_spec_sha256 = compute_training_run_spec_sha256(spec)

        # 4. Construct initial TrainingRunState (PLANNED)
        state = TrainingRunState(
            run_id=run_id,
            status=TrainingRunStatus.PLANNED.value,
            updated_at=spec.created_at,
        )

        # 5. Atomic persistence via staging directory
        self.training_root.mkdir(parents=True, exist_ok=True)
        staging_dir = self.training_root / f".tmp_run_{run_id}_{uuid.uuid4().hex[:8]}"
        staging_dir.mkdir(parents=True, exist_ok=False)

        try:
            # Write spec.json
            with open(staging_dir / "spec.json", "w", encoding="utf-8") as f:
                f.write(serialize_training_run_spec(spec))

            # Write state.json
            with open(staging_dir / "state.json", "w", encoding="utf-8") as f:
                json.dump(state.to_dict(), f, indent=2, sort_keys=True)

            # Write environment.json
            with open(staging_dir / "environment.json", "w", encoding="utf-8") as f:
                json.dump(env_record.to_dict(), f, indent=2, sort_keys=True)

            # Rename staging directory to final run directory
            shutil.move(str(staging_dir), str(run_dir))
        finally:
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)

        return TrainingRun(
            spec=spec,
            state=state,
            environment=env_record,
            run_dir=run_dir,
        )

    def load_run(self, run_id: str) -> TrainingRun:
        """Loads a training run from disk without cryptographic verification."""
        run_dir = self.get_run_dir(run_id)
        if not run_dir.exists() or not run_dir.is_dir():
            raise FileNotFoundError(f"Training run directory '{run_dir}' does not exist.")

        spec_file = run_dir / "spec.json"
        state_file = run_dir / "state.json"
        env_file = run_dir / "environment.json"

        if not spec_file.exists():
            raise FileNotFoundError(f"Missing spec.json in training run '{run_id}'.")
        if not state_file.exists():
            raise FileNotFoundError(f"Missing state.json in training run '{run_id}'.")

        with open(spec_file, "r", encoding="utf-8") as f:
            spec_dict = json.load(f)
        spec = TrainingRunSpec(**spec_dict)

        with open(state_file, "r", encoding="utf-8") as f:
            state_dict = json.load(f)
        state = TrainingRunState(**state_dict)

        if env_file.exists():
            with open(env_file, "r", encoding="utf-8") as f:
                env_dict = json.load(f)
            env_record = TrainingEnvironmentRecord(**env_dict)
        else:
            env_record = TrainingEnvironmentRecord(
                python_version="unknown",
                platform="unknown",
                platform_release="unknown",
                machine_architecture="unknown",
            )

        return TrainingRun(spec=spec, state=state, environment=env_record, run_dir=run_dir)

    def load_verified_run(
        self,
        run_id: str,
        verify_integrity: bool = True,
        dataset_root: Optional[Union[str, Path]] = None,
    ) -> TrainingRun:
        """Loads a training run and verifies spec digest, ID coherence, and referenced snapshot."""
        run = self.load_run(run_id)
        spec = run.spec
        state = run.state

        # 1. Verify spec ID coherence
        if spec.run_id != run_id:
            raise ValueError(
                f"Training run spec run_id '{spec.run_id}' does not match requested run_id '{run_id}'."
            )
        if state.run_id != run_id:
            raise ValueError(
                f"Training run state run_id '{state.run_id}' does not match run_id '{run_id}'."
            )

        # 2. Verify state status value
        valid_statuses = {s.value for s in TrainingRunStatus}
        if state.status not in valid_statuses:
            raise ValueError(
                f"Invalid training run status '{state.status}'. Allowed values: {valid_statuses}."
            )

        # 3. Verify spec SHA-256 integrity
        if not spec.run_spec_sha256:
            raise ValueError(f"Training run '{run_id}' has missing or empty run_spec_sha256.")

        actual_spec_sha = compute_training_run_spec_sha256(spec)
        if actual_spec_sha != spec.run_spec_sha256:
            raise ValueError(
                f"Training run spec integrity error for '{run_id}': "
                f"computed SHA-256 '{actual_spec_sha}' does not match recorded run_spec_sha256 '{spec.run_spec_sha256}'. "
                "Experiment specification has been modified or corrupted."
            )

        # 4. Verify referenced frozen dataset snapshot
        if verify_integrity:
            d_root = dataset_root or self.dataset_root
            snap_mgr = DatasetSnapshotManager(dataset_root=d_root)
            snapshot = snap_mgr.load_snapshot(
                dataset_id=spec.dataset_id,
                snapshot_version=spec.snapshot_version,
                verify_integrity=True,
            )

            if snapshot.metadata.status != "FROZEN":
                raise ValueError(
                    f"Referenced snapshot '{spec.snapshot_version}' has status '{snapshot.metadata.status}', expected 'FROZEN'."
                )

            if snapshot.metadata.metadata_sha256 != spec.snapshot_metadata_sha256:
                raise ValueError(
                    f"Snapshot metadata SHA mismatch for '{spec.snapshot_version}': "
                    f"actual '{snapshot.metadata.metadata_sha256}' vs run spec '{spec.snapshot_metadata_sha256}'."
                )

        return run

    def _persist_state(self, run_dir: Path, state: TrainingRunState) -> None:
        """Atomically persists updated runtime state to state.json."""
        state_file = run_dir / "state.json"
        tmp_file = run_dir / f".tmp_state_{uuid.uuid4().hex[:8]}.json"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, indent=2, sort_keys=True)
        shutil.move(str(tmp_file), str(state_file))

    def start_run(self, run_id: str) -> TrainingRun:
        """Transitions a PLANNED run to RUNNING state."""
        run = self.load_verified_run(run_id, verify_integrity=False)
        if run.state.status != TrainingRunStatus.PLANNED.value:
            raise ValueError(
                f"Cannot start training run '{run_id}': status is '{run.state.status}', expected 'PLANNED'."
            )

        now = utcnow_iso()
        run.state.status = TrainingRunStatus.RUNNING.value
        run.state.started_at = now
        run.state.updated_at = now

        self._persist_state(run.run_dir, run.state)
        return run

    def complete_run(
        self,
        run_id: str,
        current_step: Optional[int] = None,
        last_checkpoint: Optional[str] = None,
    ) -> TrainingRun:
        """Transitions a RUNNING run to COMPLETED state."""
        run = self.load_verified_run(run_id, verify_integrity=False)
        if run.state.status != TrainingRunStatus.RUNNING.value:
            raise ValueError(
                f"Cannot complete training run '{run_id}': status is '{run.state.status}', expected 'RUNNING'."
            )

        now = utcnow_iso()
        run.state.status = TrainingRunStatus.COMPLETED.value
        run.state.completed_at = now
        run.state.updated_at = now
        if current_step is not None:
            run.state.current_step = current_step
        if last_checkpoint is not None:
            run.state.last_checkpoint = last_checkpoint

        self._persist_state(run.run_dir, run.state)
        return run

    def fail_run(
        self,
        run_id: str,
        failure_reason: str,
        current_step: Optional[int] = None,
        last_checkpoint: Optional[str] = None,
    ) -> TrainingRun:
        """Transitions a RUNNING run to FAILED state."""
        run = self.load_verified_run(run_id, verify_integrity=False)
        if run.state.status != TrainingRunStatus.RUNNING.value:
            raise ValueError(
                f"Cannot fail training run '{run_id}': status is '{run.state.status}', expected 'RUNNING'."
            )

        now = utcnow_iso()
        run.state.status = TrainingRunStatus.FAILED.value
        run.state.failure_reason = str(failure_reason)
        run.state.completed_at = now
        run.state.updated_at = now
        if current_step is not None:
            run.state.current_step = current_step
        if last_checkpoint is not None:
            run.state.last_checkpoint = last_checkpoint

        self._persist_state(run.run_dir, run.state)
        return run

    def abort_run(
        self,
        run_id: str,
        abort_reason: Optional[str] = None,
    ) -> TrainingRun:
        """Transitions a PLANNED or RUNNING run to ABORTED state."""
        run = self.load_verified_run(run_id, verify_integrity=False)
        allowed_statuses = (TrainingRunStatus.PLANNED.value, TrainingRunStatus.RUNNING.value)
        if run.state.status not in allowed_statuses:
            raise ValueError(
                f"Cannot abort training run '{run_id}': status is '{run.state.status}', expected one of {allowed_statuses}."
            )

        now = utcnow_iso()
        run.state.status = TrainingRunStatus.ABORTED.value
        if abort_reason:
            run.state.failure_reason = str(abort_reason)
        run.state.completed_at = now
        run.state.updated_at = now

        self._persist_state(run.run_dir, run.state)
        return run

    def verify_run(
        self,
        run_id: str,
        dataset_root: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Performs full integrity audit and returns structured report."""
        errors: List[str] = []
        run: Optional[TrainingRun] = None

        try:
            run = self.load_verified_run(
                run_id=run_id,
                verify_integrity=True,
                dataset_root=dataset_root,
            )
        except Exception as e:
            errors.append(str(e))

        return {
            "run_id": run_id,
            "valid": len(errors) == 0,
            "errors": errors,
            "spec": run.spec.to_dict() if run else None,
            "state": run.state.to_dict() if run else None,
        }

    def list_runs(
        self,
        verify_integrity: bool = False,
        dataset_id: Optional[str] = None,
    ) -> List[TrainingRun]:
        """Lists all training runs in training_root, optionally filtering and verifying integrity."""
        if not self.training_root.exists():
            return []

        runs: List[TrainingRun] = []
        for child in self.training_root.iterdir():
            if child.is_dir() and not child.name.startswith("."):
                try:
                    if verify_integrity:
                        run = self.load_verified_run(child.name, verify_integrity=True)
                    else:
                        run = self.load_run(child.name)

                    if dataset_id is not None and run.spec.dataset_id != dataset_id:
                        continue
                    runs.append(run)
                except Exception:
                    if not verify_integrity:
                        # Skip corrupted runs on plain list or omit when corrupt
                        pass

        runs.sort(key=lambda r: r.spec.run_id)
        return runs
