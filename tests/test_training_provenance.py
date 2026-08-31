"""Comprehensive unit and integration tests for Step 22.F Training Run Provenance & Reproducibility Record."""

import json
from pathlib import Path
import pytest
from safetensors.torch import save_file
import torch

from rernggen.data.caption_review import CaptionReviewManager
from rernggen.data.captions import compute_caption_sha256
from rernggen.data.governance import GovernanceManager
from rernggen.data.snapshot import DatasetSnapshotManager
from rernggen.training.provenance import (
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
)


def setup_fixture_snapshot_for_run(
    root_dir: Path,
    dataset_id: str = "fixture_run_ds",
    snapshot_version: str = "dataset_snapshot_v001",
    count: int = 2,
) -> str:
    """Sets up a complete authorized, reviewed, frozen dataset snapshot fixture."""
    ds_dir = root_dir / dataset_id
    latent_dir = ds_dir / "cache" / "latents" / "vae_sd_mse_square256_v001"
    text_dir = ds_dir / "cache" / "text_embeds" / "clip_b32_v001"
    caption_dir = ds_dir / "captions" / "captions_v002"
    manifest_dir = ds_dir / "manifests"

    latent_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    caption_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    latent_recs = []
    text_recs = []
    caption_recs = []
    manifest_recs = []
    all_ids = []

    for i in range(count):
        img_id = f"IMG-{i+1:06d}"
        all_ids.append(img_id)
        save_file({"latent": torch.randn(4, 32, 32)}, latent_dir / f"{img_id}.safetensors")
        save_file({"embedding": torch.randn(512)}, text_dir / f"{img_id}.safetensors")

        latent_recs.append(
            {
                "image_id": img_id,
                "dataset_id": dataset_id,
                "dataset_version": "v001",
                "source_processed_sha256": f"sha_proc_{i}",
                "preprocessing_version": "square256_center_v001",
                "vae_model_id": "mock_vae",
                "vae_revision": "mock_rev",
                "vae_weights_sha256": "mock_w_sha",
                "vae_config_sha256": "mock_c_sha",
                "vae_scaling_factor": 0.18215,
                "posterior_policy": "posterior_mode",
                "latent_shape": [4, 32, 32],
                "latent_dtype": "float32",
                "latent_sha256": f"lat_sha_{i}",
                "latent_relative_path": f"cache/latents/vae_sd_mse_square256_v001/{img_id}.safetensors",
                "min_val": 0.0,
                "max_val": 1.0,
                "mean_val": 0.5,
                "std_val": 0.2,
                "l2_norm": 10.0,
                "training_allowed": None,
                "commercial_allowed": None,
                "license_id": None,
                "cache_version": "vae_sd_mse_square256_v001",
                "status": "CACHED",
            }
        )

        cap_text = f"Scene {i+1}"
        cap_sha = compute_caption_sha256(cap_text)

        text_recs.append(
            {
                "image_id": img_id,
                "dataset_id": dataset_id,
                "dataset_version": "v001",
                "caption_version": "captions_v002",
                "caption_sha256": cap_sha,
                "text_encoder_id": "mock_enc",
                "text_encoder_revision": "mock_rev",
                "text_encoder_weights_sha256": "mock_w_sha",
                "text_encoder_config_sha256": "mock_c_sha",
                "tokenizer_class": "MockTokenizer",
                "tokenizer_config_sha256": "tok_cfg_sha",
                "vocab_sha256": "vocab_sha",
                "merges_sha256": "merges_sha",
                "special_tokens_map_sha256": "special_sha",
                "tokenizer_identity_sha256": "tok_id_sha",
                "max_token_length": 77,
                "pooling_policy": "eos_token",
                "embedding_shape": [512],
                "embedding_dtype": "float32",
                "embedding_sha256": f"emb_sha_{i}",
                "embedding_relative_path": f"cache/text_embeds/clip_b32_v001/{img_id}.safetensors",
                "min_val": 0.0,
                "max_val": 1.0,
                "mean_val": 0.5,
                "std_val": 0.2,
                "l2_norm": 10.0,
                "token_count": 10,
                "truncated": False,
                "training_allowed": None,
                "commercial_allowed": None,
                "license_id": None,
                "cache_version": "clip_b32_v001",
                "status": "CACHED",
            }
        )

        caption_recs.append(
            {
                "image_id": img_id,
                "dataset_id": dataset_id,
                "caption": cap_text,
                "caption_source": "synthetic",
                "caption_version": "captions_v002",
                "caption_sha256": cap_sha,
                "language": "en",
                "review_status": "unreviewed",
                "training_allowed": None,
                "commercial_allowed": None,
                "license_id": None,
            }
        )

        manifest_recs.append(
            {
                "image_id": img_id,
                "dataset_id": dataset_id,
                "relative_path": f"raw/{img_id}.png",
                "file_sha256": f"raw_sha_{i}",
                "width": 256,
                "height": 256,
                "format": "PNG",
                "status": "VALID",
            }
        )

    with open(latent_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
        for r in latent_recs:
            f.write(json.dumps(r) + "\n")
    with open(text_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
        for r in text_recs:
            f.write(json.dumps(r) + "\n")
    with open(caption_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
        for r in caption_recs:
            f.write(json.dumps(r) + "\n")
    with open(manifest_dir / "dataset_manifest.jsonl", "w", encoding="utf-8") as f:
        for r in manifest_recs:
            f.write(json.dumps(r) + "\n")

    # Authorize & Review
    gov_mgr = GovernanceManager(dataset_root=root_dir)
    rev_mgr = CaptionReviewManager(dataset_root=root_dir)
    gov_mgr.authorize_samples(
        dataset_id=dataset_id,
        governance_version="rights_v001",
        image_ids=all_ids,
        training_decision="ALLOW",
        commercial_decision="ALLOW",
        authorization_source="audit_lead",
    )
    rev_mgr.review_samples(
        dataset_id=dataset_id,
        review_version="cap_rev_v001",
        image_ids=all_ids,
        review_status="APPROVED",
        reviewed_by="reviewer_alice",
        review_source="audit",
    )

    # Freeze snapshot
    snap_mgr = DatasetSnapshotManager(dataset_root=root_dir)
    snap = snap_mgr.freeze_snapshot(
        dataset_id=dataset_id,
        snapshot_version=snapshot_version,
        governance_version="rights_v001",
        caption_review_version="cap_rev_v001",
        created_by="engineer_alice",
        creation_source="training_prep",
    )
    return snap.metadata.metadata_sha256


# =============================================================================
# Test 1 - 4: Valid run creation binds exact snapshot provenance
# =============================================================================
def test_valid_run_creation_binds_exact_snapshot_provenance(tmp_path: Path):
    ds_root = tmp_path / "datasets"
    train_root = tmp_path / "training_runs"
    snap_sha = setup_fixture_snapshot_for_run(ds_root, "ds_run_test", "dataset_snapshot_v001", count=2)

    mgr = TrainingRunManager(training_root=train_root, dataset_root=ds_root)
    run = mgr.create_run(
        run_id="run_000001",
        dataset_id="ds_run_test",
        snapshot_version="dataset_snapshot_v001",
        model_family="dit",
        model_config={"hidden_size": 384, "depth": 6},
        training_config={"lr": 1e-4, "batch_size": 4},
        seed=42,
        created_by="lead_engineer",
        creation_source="local_experiment",
    )

    assert run.spec.run_id == "run_000001"
    assert run.spec.dataset_id == "ds_run_test"
    assert run.spec.snapshot_version == "dataset_snapshot_v001"
    assert run.spec.snapshot_metadata_sha256 == snap_sha
    assert run.spec.seed == 42
    assert run.spec.model_family == "dit"
    assert run.spec.run_spec_sha256 is not None
    assert len(run.spec.run_spec_sha256) == 64
    assert run.state.status == TrainingRunStatus.PLANNED.value


# =============================================================================
# Test 5: Expected snapshot metadata SHA mismatch fails closed
# =============================================================================
def test_expected_snapshot_metadata_sha_mismatch_fails_closed(tmp_path: Path):
    ds_root = tmp_path / "datasets"
    train_root = tmp_path / "training_runs"
    setup_fixture_snapshot_for_run(ds_root, "ds_sha_mismatch", "dataset_snapshot_v001", count=2)

    mgr = TrainingRunManager(training_root=train_root, dataset_root=ds_root)
    with pytest.raises(ValueError, match="Snapshot metadata SHA mismatch"):
        mgr.create_run(
            run_id="run_000001",
            dataset_id="ds_sha_mismatch",
            snapshot_version="dataset_snapshot_v001",
            model_family="dit",
            model_config={"hidden_size": 384},
            training_config={"lr": 1e-4},
            seed=42,
            created_by="lead_engineer",
            creation_source="local_experiment",
            expected_snapshot_metadata_sha256="0" * 64,
        )


# =============================================================================
# Test 6: Missing snapshot fails closed
# =============================================================================
def test_missing_snapshot_fails_closed(tmp_path: Path):
    ds_root = tmp_path / "datasets"
    train_root = tmp_path / "training_runs"

    mgr = TrainingRunManager(training_root=train_root, dataset_root=ds_root)
    with pytest.raises(FileNotFoundError, match="not found"):
        mgr.create_run(
            run_id="run_000001",
            dataset_id="nonexistent_dataset",
            snapshot_version="dataset_snapshot_v001",
            model_family="dit",
            model_config={"hidden_size": 384},
            training_config={"lr": 1e-4},
            seed=42,
            created_by="lead_engineer",
            creation_source="local_experiment",
        )


# =============================================================================
# Test 7: Tampered snapshot fails closed
# =============================================================================
def test_tampered_snapshot_fails_closed(tmp_path: Path):
    ds_root = tmp_path / "datasets"
    train_root = tmp_path / "training_runs"
    setup_fixture_snapshot_for_run(ds_root, "ds_snap_tamper", "dataset_snapshot_v001", count=2)

    # Tamper with snapshot metadata on disk
    meta_path = ds_root / "ds_snap_tamper" / "snapshots" / "dataset_snapshot_v001" / "metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["governance_version"] = "TAMPERED_GOV"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    mgr = TrainingRunManager(training_root=train_root, dataset_root=ds_root)
    with pytest.raises(ValueError, match="metadata SHA-256.*does not match"):
        mgr.create_run(
            run_id="run_000001",
            dataset_id="ds_snap_tamper",
            snapshot_version="dataset_snapshot_v001",
            model_family="dit",
            model_config={"hidden_size": 384},
            training_config={"lr": 1e-4},
            seed=42,
            created_by="lead_engineer",
            creation_source="local_experiment",
        )


# =============================================================================
# Test 8: Live dataset directory without snapshot fails
# =============================================================================
def test_live_dataset_without_snapshot_fails(tmp_path: Path):
    ds_root = tmp_path / "datasets"
    train_root = tmp_path / "training_runs"
    (ds_root / "raw_ds" / "manifests").mkdir(parents=True, exist_ok=True)

    mgr = TrainingRunManager(training_root=train_root, dataset_root=ds_root)
    with pytest.raises(FileNotFoundError):
        mgr.create_run(
            run_id="run_000001",
            dataset_id="raw_ds",
            snapshot_version="dataset_snapshot_v001",
            model_family="dit",
            model_config={},
            training_config={},
            seed=42,
            created_by="lead_engineer",
            creation_source="local_experiment",
        )


# =============================================================================
# Test 9 - 16: Determinism, dictionary ordering independence, and digest variation
# =============================================================================
def test_run_spec_digest_determinism_and_ordering_independence():
    spec_dict_1 = {
        "run_id": "run_000001",
        "dataset_id": "ds_1",
        "snapshot_version": "v1",
        "snapshot_metadata_sha256": "sha_meta_1",
        "model_family": "dit",
        "model_config": {"b": 2, "a": 1, "nested": {"z": 10, "y": 20}},
        "training_config": {"lr": 1e-4, "batch_size": 8},
        "seed": 42,
        "git_commit": "abc1234",
        "git_dirty": False,
        "created_at": "2026-08-31T00:00:00Z",
        "created_by": "alice",
        "creation_source": "test",
    }
    # Invert dictionary keys and nested keys
    spec_dict_2 = {
        "creation_source": "test",
        "created_by": "alice",
        "created_at": "2026-08-31T00:00:00Z",
        "git_dirty": False,
        "git_commit": "abc1234",
        "seed": 42,
        "training_config": {"batch_size": 8, "lr": 1e-4},
        "model_config": {"nested": {"y": 20, "z": 10}, "a": 1, "b": 2},
        "model_family": "dit",
        "snapshot_metadata_sha256": "sha_meta_1",
        "snapshot_version": "v1",
        "dataset_id": "ds_1",
        "run_id": "run_000001",
    }

    sha1 = compute_training_run_spec_sha256(spec_dict_1)
    sha2 = compute_training_run_spec_sha256(spec_dict_2)
    assert sha1 == sha2

    # 13. Different seed changes digest
    spec_diff_seed = dict(spec_dict_1, seed=43)
    assert compute_training_run_spec_sha256(spec_diff_seed) != sha1

    # 14. Different model config changes digest
    spec_diff_model = dict(spec_dict_1, model_config={"hidden_size": 512})
    assert compute_training_run_spec_sha256(spec_diff_model) != sha1

    # 15. Different training config changes digest
    spec_diff_train = dict(spec_dict_1, training_config={"lr": 2e-4})
    assert compute_training_run_spec_sha256(spec_diff_train) != sha1

    # 16. Different snapshot changes digest
    spec_diff_snap = dict(spec_dict_1, snapshot_version="v2", snapshot_metadata_sha256="sha_meta_2")
    assert compute_training_run_spec_sha256(spec_diff_snap) != sha1


# =============================================================================
# Test 17 - 19: Seed validation
# =============================================================================
def test_seed_validation_rules():
    assert validate_seed(0) == 0
    assert validate_seed(42) == 42
    assert validate_seed((1 << 32) - 1) == (1 << 32) - 1

    with pytest.raises(ValueError, match="cannot be None"):
        validate_seed(None)
    with pytest.raises(ValueError, match="got bool"):
        validate_seed(True)
    with pytest.raises(ValueError, match="got bool"):
        validate_seed(False)
    with pytest.raises(ValueError, match="must be an integer"):
        validate_seed("42")
    with pytest.raises(ValueError, match="must be an integer"):
        validate_seed(42.0)
    with pytest.raises(ValueError, match="range"):
        validate_seed(-1)
    with pytest.raises(ValueError, match="range"):
        validate_seed(1 << 32)


# =============================================================================
# Test 20 - 22: Run ID validation
# =============================================================================
def test_run_id_validation_rules():
    validate_run_id("run_000001")
    validate_run_id("rernggen-exp-01")

    with pytest.raises(ValueError, match="empty or whitespace"):
        validate_run_id("")
    with pytest.raises(ValueError, match="empty or whitespace"):
        validate_run_id("   ")
    with pytest.raises(ValueError, match="whitespace"):
        validate_run_id(" run_001")
    with pytest.raises(ValueError, match="reserved keyword"):
        validate_run_id("latest")
    with pytest.raises(ValueError, match="reserved keyword"):
        validate_run_id("current")
    with pytest.raises(ValueError, match="invalid characters"):
        validate_run_id("../run_evil")
    with pytest.raises(ValueError, match="invalid characters"):
        validate_run_id("run/child")
    with pytest.raises(ValueError, match="invalid characters"):
        validate_run_id("run\\child")
    with pytest.raises(ValueError, match="invalid characters"):
        validate_run_id("run:1")


# =============================================================================
# Test 23 - 25: Attribution validation
# =============================================================================
def test_attribution_validation_rules():
    validate_attribution("alice", "local_cli")

    with pytest.raises(ValueError, match="created_by is mandatory"):
        validate_attribution("", "local_cli")
    with pytest.raises(ValueError, match="creation_source is mandatory"):
        validate_attribution("alice", "")

    for dummy in ["human", "system", "manual", "unknown", "human_declared", "default", "none", "null"]:
        with pytest.raises(ValueError, match="created_by cannot be placeholder"):
            validate_attribution(dummy, "local_cli")
        with pytest.raises(ValueError, match="creation_source cannot be placeholder"):
            validate_attribution("alice", dummy)


# =============================================================================
# Test 26 - 28: Git provenance collection
# =============================================================================
def test_git_provenance_collection():
    commit, dirty, branch = collect_git_provenance()
    # In this repo, commit should be captured
    if commit is not None:
        assert isinstance(commit, str)
        assert len(commit) >= 7
        assert isinstance(dirty, bool)

    # Outside repo / invalid directory returns (None, None, None) without crashing
    c_out, d_out, b_out = collect_git_provenance(cwd=Path("C:/nonexistent_dir_12345"))
    assert c_out is None
    assert d_out is None
    assert b_out is None


# =============================================================================
# Test 29 - 36: Environment record & Secret exclusion
# =============================================================================
def test_environment_record_and_secret_exclusion(tmp_path: Path):
    rec = collect_environment_record(project_root=tmp_path)
    assert rec.python_version is not None
    assert rec.platform is not None
    assert rec.machine_architecture is not None
    assert rec.device_type in ("cpu", "cuda")
    assert isinstance(rec.gpu_count, int)

    rec_dict = rec.to_dict()
    # Ensure no API keys or environmental leaks exist in environment record
    for key, value in rec_dict.items():
        assert not str(key).upper().endswith("_KEY")
        assert not str(key).upper().endswith("_TOKEN")
        assert not str(key).upper().endswith("_SECRET")
        assert "OPENAI" not in str(key).upper()
        assert "HUGGINGFACE" not in str(key).upper()


# =============================================================================
# Test 37 & 38: Atomic run creation & Overwrite rejection
# =============================================================================
def test_atomic_run_creation_and_overwrite_rejection(tmp_path: Path):
    ds_root = tmp_path / "datasets"
    train_root = tmp_path / "training_runs"
    setup_fixture_snapshot_for_run(ds_root, "ds_atomic", "dataset_snapshot_v001", count=2)

    mgr = TrainingRunManager(training_root=train_root, dataset_root=ds_root)
    run1 = mgr.create_run(
        run_id="run_000001",
        dataset_id="ds_atomic",
        snapshot_version="dataset_snapshot_v001",
        model_family="dit",
        model_config={"hidden_size": 384},
        training_config={"lr": 1e-4},
        seed=42,
        created_by="lead_engineer",
        creation_source="local_experiment",
    )

    assert (train_root / "run_000001" / "spec.json").is_file()
    assert (train_root / "run_000001" / "state.json").is_file()
    assert (train_root / "run_000001" / "environment.json").is_file()

    # Re-creating same run_id must raise FileExistsError
    with pytest.raises(FileExistsError, match="already exists"):
        mgr.create_run(
            run_id="run_000001",
            dataset_id="ds_atomic",
            snapshot_version="dataset_snapshot_v001",
            model_family="dit",
            model_config={"hidden_size": 384},
            training_config={"lr": 1e-4},
            seed=42,
            created_by="lead_engineer",
            creation_source="local_experiment",
        )


# =============================================================================
# Test 39 - 44: Spec tampering detection
# =============================================================================
def test_spec_tampering_detection(tmp_path: Path):
    ds_root = tmp_path / "datasets"
    train_root = tmp_path / "training_runs"
    setup_fixture_snapshot_for_run(ds_root, "ds_tamper_run", "dataset_snapshot_v001", count=2)

    mgr = TrainingRunManager(training_root=train_root, dataset_root=ds_root)
    run = mgr.create_run(
        run_id="run_000001",
        dataset_id="ds_tamper_run",
        snapshot_version="dataset_snapshot_v001",
        model_family="dit",
        model_config={"hidden_size": 384},
        training_config={"lr": 1e-4},
        seed=42,
        created_by="lead_engineer",
        creation_source="local_experiment",
    )

    spec_file = run.run_dir / "spec.json"
    with open(spec_file, "r", encoding="utf-8") as f:
        spec_dict = json.load(f)

    # 40. Seed tampering
    tampered = dict(spec_dict, seed=999)
    with open(spec_file, "w", encoding="utf-8") as f:
        json.dump(tampered, f, indent=2)
    with pytest.raises(ValueError, match="spec integrity error"):
        mgr.load_verified_run("run_000001", verify_integrity=True)

    # 41. Model config tampering
    tampered = dict(spec_dict, model_config={"hidden_size": 1024})
    with open(spec_file, "w", encoding="utf-8") as f:
        json.dump(tampered, f, indent=2)
    with pytest.raises(ValueError, match="spec integrity error"):
        mgr.load_verified_run("run_000001", verify_integrity=True)

    # 42. Training config tampering
    tampered = dict(spec_dict, training_config={"lr": 999.0})
    with open(spec_file, "w", encoding="utf-8") as f:
        json.dump(tampered, f, indent=2)
    with pytest.raises(ValueError, match="spec integrity error"):
        mgr.load_verified_run("run_000001", verify_integrity=True)

    # 43. Snapshot reference tampering
    tampered = dict(spec_dict, snapshot_metadata_sha256="0" * 64)
    with open(spec_file, "w", encoding="utf-8") as f:
        json.dump(tampered, f, indent=2)
    with pytest.raises(ValueError, match="spec integrity error"):
        mgr.load_verified_run("run_000001", verify_integrity=True)

    # 44. run_id mismatch
    tampered = dict(spec_dict, run_id="run_alien")
    with open(spec_file, "w", encoding="utf-8") as f:
        json.dump(tampered, f, indent=2)
    with pytest.raises(ValueError, match="does not match requested run_id"):
        mgr.load_verified_run("run_000001", verify_integrity=False)


# =============================================================================
# Test 45 - 52: State machine transitions & Disallowed transitions
# =============================================================================
def test_state_machine_lifecycle_and_terminal_constraints(tmp_path: Path):
    ds_root = tmp_path / "datasets"
    train_root = tmp_path / "training_runs"
    setup_fixture_snapshot_for_run(ds_root, "ds_state_mach", "dataset_snapshot_v001", count=2)

    mgr = TrainingRunManager(training_root=train_root, dataset_root=ds_root)

    # 45. PLANNED initial state
    run = mgr.create_run(
        run_id="run_000001",
        dataset_id="ds_state_mach",
        snapshot_version="dataset_snapshot_v001",
        model_family="dit",
        model_config={"hidden_size": 384},
        training_config={"lr": 1e-4},
        seed=42,
        created_by="lead_engineer",
        creation_source="local_experiment",
    )
    assert run.state.status == TrainingRunStatus.PLANNED.value
    assert run.state.started_at is None

    # Cannot complete or fail directly from PLANNED
    with pytest.raises(ValueError, match="expected 'RUNNING'"):
        mgr.complete_run("run_000001")
    with pytest.raises(ValueError, match="expected 'RUNNING'"):
        mgr.fail_run("run_000001", failure_reason="fail")

    # 46. PLANNED -> RUNNING
    run = mgr.start_run("run_000001")
    assert run.state.status == TrainingRunStatus.RUNNING.value
    assert run.state.started_at is not None

    # 47. RUNNING -> COMPLETED
    run = mgr.complete_run("run_000001", current_step=1000, last_checkpoint="ckpt_1000.pt")
    assert run.state.status == TrainingRunStatus.COMPLETED.value
    assert run.state.completed_at is not None
    assert run.state.current_step == 1000

    # 50. COMPLETED -> RUNNING rejected
    with pytest.raises(ValueError, match="expected 'PLANNED'"):
        mgr.start_run("run_000001")

    # 48. Test RUNNING -> FAILED
    run2 = mgr.create_run(
        run_id="run_000002",
        dataset_id="ds_state_mach",
        snapshot_version="dataset_snapshot_v001",
        model_family="dit",
        model_config={"hidden_size": 384},
        training_config={"lr": 1e-4},
        seed=43,
        created_by="lead_engineer",
        creation_source="local_experiment",
    )
    mgr.start_run("run_000002")
    run2 = mgr.fail_run("run_000002", failure_reason="CUDA out of memory")
    assert run2.state.status == TrainingRunStatus.FAILED.value
    assert run2.state.failure_reason == "CUDA out of memory"

    # 51. FAILED -> RUNNING rejected
    with pytest.raises(ValueError, match="expected 'PLANNED'"):
        mgr.start_run("run_000002")

    # 49. Test PLANNED -> ABORTED
    run3 = mgr.create_run(
        run_id="run_000003",
        dataset_id="ds_state_mach",
        snapshot_version="dataset_snapshot_v001",
        model_family="dit",
        model_config={"hidden_size": 384},
        training_config={"lr": 1e-4},
        seed=44,
        created_by="lead_engineer",
        creation_source="local_experiment",
    )
    run3 = mgr.abort_run("run_000003", abort_reason="User cancelled")
    assert run3.state.status == TrainingRunStatus.ABORTED.value

    # 52. ABORTED -> RUNNING rejected
    with pytest.raises(ValueError, match="expected 'PLANNED'"):
        mgr.start_run("run_000003")


# =============================================================================
# Test 53 & 54: State mutations leave spec.json & run_spec_sha256 strictly untouched
# =============================================================================
def test_state_mutations_leave_spec_bytes_and_sha_unchanged(tmp_path: Path):
    ds_root = tmp_path / "datasets"
    train_root = tmp_path / "training_runs"
    setup_fixture_snapshot_for_run(ds_root, "ds_spec_immut", "dataset_snapshot_v001", count=2)

    mgr = TrainingRunManager(training_root=train_root, dataset_root=ds_root)
    run = mgr.create_run(
        run_id="run_000001",
        dataset_id="ds_spec_immut",
        snapshot_version="dataset_snapshot_v001",
        model_family="dit",
        model_config={"hidden_size": 384},
        training_config={"lr": 1e-4},
        seed=42,
        created_by="lead_engineer",
        creation_source="local_experiment",
    )

    spec_bytes_initial = (run.run_dir / "spec.json").read_bytes()
    spec_sha_initial = run.spec.run_spec_sha256

    mgr.start_run("run_000001")
    mgr.complete_run("run_000001", current_step=500)

    reloaded = mgr.load_verified_run("run_000001", verify_integrity=True)
    assert (reloaded.run_dir / "spec.json").read_bytes() == spec_bytes_initial
    assert reloaded.spec.run_spec_sha256 == spec_sha_initial
    assert reloaded.state.status == TrainingRunStatus.COMPLETED.value


# =============================================================================
# Test 55: Successor snapshot does NOT mutate historical run
# =============================================================================
def test_successor_snapshot_does_not_mutate_historical_run(tmp_path: Path):
    ds_root = tmp_path / "datasets"
    train_root = tmp_path / "training_runs"
    setup_fixture_snapshot_for_run(ds_root, "ds_hist_run", "dataset_snapshot_v001", count=2)

    mgr = TrainingRunManager(training_root=train_root, dataset_root=ds_root)
    run = mgr.create_run(
        run_id="run_000001",
        dataset_id="ds_hist_run",
        snapshot_version="dataset_snapshot_v001",
        model_family="dit",
        model_config={"hidden_size": 384},
        training_config={"lr": 1e-4},
        seed=42,
        created_by="lead_engineer",
        creation_source="local_experiment",
    )

    # Freeze successor snapshot v002 in dataset
    gov_mgr = GovernanceManager(dataset_root=ds_root)
    rev_mgr = CaptionReviewManager(dataset_root=ds_root)
    snap_mgr = DatasetSnapshotManager(dataset_root=ds_root)

    gov_mgr.authorize_samples(
        dataset_id="ds_hist_run",
        governance_version="rights_v002",
        image_ids=["IMG-000001", "IMG-000002"],
        training_decision="ALLOW",
        commercial_decision="ALLOW",
        authorization_source="audit_lead",
        base_version="rights_v001",
    )
    rev_mgr.review_samples(
        dataset_id="ds_hist_run",
        review_version="cap_rev_v002",
        image_ids=["IMG-000001", "IMG-000002"],
        review_status="APPROVED",
        reviewed_by="reviewer_alice",
        review_source="audit",
        base_version="cap_rev_v001",
    )
    snap_mgr.freeze_snapshot(
        dataset_id="ds_hist_run",
        snapshot_version="dataset_snapshot_v002",
        governance_version="rights_v002",
        caption_review_version="cap_rev_v002",
        created_by="engineer_alice",
        creation_source="training_prep",
        previous_snapshot_version="dataset_snapshot_v001",
    )

    # Historical run still points to dataset_snapshot_v001
    verified_run = mgr.load_verified_run("run_000001", verify_integrity=True)
    assert verified_run.spec.snapshot_version == "dataset_snapshot_v001"


# =============================================================================
# Test 56 - 58: Provenance export compact dictionary
# =============================================================================
def test_provenance_export_compact_dictionary(tmp_path: Path):
    ds_root = tmp_path / "datasets"
    train_root = tmp_path / "training_runs"
    snap_sha = setup_fixture_snapshot_for_run(ds_root, "ds_prov_exp", "dataset_snapshot_v001", count=2)

    mgr = TrainingRunManager(training_root=train_root, dataset_root=ds_root)
    run = mgr.create_run(
        run_id="run_000001",
        dataset_id="ds_prov_exp",
        snapshot_version="dataset_snapshot_v001",
        model_family="dit",
        model_config={"hidden_size": 384},
        training_config={"lr": 1e-4},
        seed=42,
        created_by="lead_engineer",
        creation_source="local_experiment",
    )

    prov = run.to_provenance_dict()
    assert prov["run_id"] == "run_000001"
    assert prov["run_spec_sha256"] == run.spec.run_spec_sha256
    assert prov["dataset_id"] == "ds_prov_exp"
    assert prov["snapshot_version"] == "dataset_snapshot_v001"
    assert prov["snapshot_metadata_sha256"] == snap_sha
    assert prov["model_family"] == "dit"
    assert prov["seed"] == 42


# =============================================================================
# Test 59 & 60: Run creation and verification do NOT mutate snapshot files
# =============================================================================
def test_run_creation_and_verification_do_not_mutate_snapshot(tmp_path: Path):
    ds_root = tmp_path / "datasets"
    train_root = tmp_path / "training_runs"
    setup_fixture_snapshot_for_run(ds_root, "ds_snap_immut", "dataset_snapshot_v001", count=2)

    snap_meta_path = ds_root / "ds_snap_immut" / "snapshots" / "dataset_snapshot_v001" / "metadata.json"
    snap_man_path = ds_root / "ds_snap_immut" / "snapshots" / "dataset_snapshot_v001" / "manifest.jsonl"

    meta_bytes_before = snap_meta_path.read_bytes()
    man_bytes_before = snap_man_path.read_bytes()

    mgr = TrainingRunManager(training_root=train_root, dataset_root=ds_root)
    run = mgr.create_run(
        run_id="run_000001",
        dataset_id="ds_snap_immut",
        snapshot_version="dataset_snapshot_v001",
        model_family="dit",
        model_config={"hidden_size": 384},
        training_config={"lr": 1e-4},
        seed=42,
        created_by="lead_engineer",
        creation_source="local_experiment",
    )

    assert snap_meta_path.read_bytes() == meta_bytes_before
    assert snap_man_path.read_bytes() == man_bytes_before

    mgr.verify_run("run_000001", dataset_root=ds_root)

    assert snap_meta_path.read_bytes() == meta_bytes_before
    assert snap_man_path.read_bytes() == man_bytes_before


# =============================================================================
# Test 61: Run listing filters corrupted metadata when verify_integrity=True
# =============================================================================
def test_run_listing_filters_corrupted_runs(tmp_path: Path):
    ds_root = tmp_path / "datasets"
    train_root = tmp_path / "training_runs"
    setup_fixture_snapshot_for_run(ds_root, "ds_list_runs", "dataset_snapshot_v001", count=2)

    mgr = TrainingRunManager(training_root=train_root, dataset_root=ds_root)
    r1 = mgr.create_run(
        run_id="run_000001",
        dataset_id="ds_list_runs",
        snapshot_version="dataset_snapshot_v001",
        model_family="dit",
        model_config={"hidden_size": 384},
        training_config={"lr": 1e-4},
        seed=42,
        created_by="lead_engineer",
        creation_source="local_experiment",
    )
    r2 = mgr.create_run(
        run_id="run_000002",
        dataset_id="ds_list_runs",
        snapshot_version="dataset_snapshot_v001",
        model_family="dit",
        model_config={"hidden_size": 384},
        training_config={"lr": 1e-4},
        seed=43,
        created_by="lead_engineer",
        creation_source="local_experiment",
    )

    # Corrupt r1 spec directly
    with open(r1.run_dir / "spec.json", "w", encoding="utf-8") as f:
        f.write('{"corrupted": true}')

    # Plain list shows remaining parseable runs
    runs_verified = mgr.list_runs(verify_integrity=True, dataset_id="ds_list_runs")
    assert len(runs_verified) == 1
    assert runs_verified[0].spec.run_id == "run_000002"


# =============================================================================
# Test 62: CLI end-to-end execution
# =============================================================================
def test_training_run_cli_end_to_end(tmp_path: Path):
    import subprocess
    import sys

    ds_root = tmp_path / "datasets"
    train_root = tmp_path / "training_runs"
    setup_fixture_snapshot_for_run(ds_root, "ds_cli_run", "dataset_snapshot_v001", count=2)

    # 1. CLI Create
    create_cmd = [
        sys.executable,
        "-m",
        "rernggen.training",
        "run",
        "create",
        "--run-id",
        "run_cli_001",
        "--dataset-id",
        "ds_cli_run",
        "--snapshot-version",
        "dataset_snapshot_v001",
        "--model-family",
        "dit",
        "--model-config",
        '{"hidden_size": 384, "depth": 6}',
        "--training-config",
        '{"lr": 1e-4, "batch_size": 4}',
        "--seed",
        "42",
        "--created-by",
        "engineer_alice",
        "--creation-source",
        "cli_experiment",
        "--training-root",
        str(train_root),
        "--dataset-root",
        str(ds_root),
    ]
    res_create = subprocess.run(create_cmd, capture_output=True, text=True, check=True)
    assert "TRAINING RUN CREATED: run_cli_001" in res_create.stdout
    assert "Status:                     PLANNED" in res_create.stdout

    # 2. CLI Show
    show_cmd = [
        sys.executable,
        "-m",
        "rernggen.training",
        "run",
        "show",
        "--run-id",
        "run_cli_001",
        "--training-root",
        str(train_root),
    ]
    res_show = subprocess.run(show_cmd, capture_output=True, text=True, check=True)
    assert "TRAINING RUN: run_cli_001" in res_show.stdout

    # 3. CLI List
    list_cmd = [
        sys.executable,
        "-m",
        "rernggen.training",
        "run",
        "list",
        "--training-root",
        str(train_root),
    ]
    res_list = subprocess.run(list_cmd, capture_output=True, text=True, check=True)
    assert "run_cli_001" in res_list.stdout

    # 4. CLI Verify
    verify_cmd = [
        sys.executable,
        "-m",
        "rernggen.training",
        "run",
        "verify",
        "--run-id",
        "run_cli_001",
        "--training-root",
        str(train_root),
        "--dataset-root",
        str(ds_root),
    ]
    res_verify = subprocess.run(verify_cmd, capture_output=True, text=True, check=True)
    assert "TRAINING RUN VERIFICATION: run_cli_001" in res_verify.stdout
    assert "Result:                     PASSED" in res_verify.stdout

    # 5. CLI Start
    start_cmd = [
        sys.executable,
        "-m",
        "rernggen.training",
        "run",
        "start",
        "--run-id",
        "run_cli_001",
        "--training-root",
        str(train_root),
    ]
    res_start = subprocess.run(start_cmd, capture_output=True, text=True, check=True)
    assert "transitioned to RUNNING" in res_start.stdout

    # 6. CLI Complete
    complete_cmd = [
        sys.executable,
        "-m",
        "rernggen.training",
        "run",
        "complete",
        "--run-id",
        "run_cli_001",
        "--current-step",
        "1000",
        "--last-checkpoint",
        "checkpoints/step_1000.pt",
        "--training-root",
        str(train_root),
    ]
    res_complete = subprocess.run(complete_cmd, capture_output=True, text=True, check=True)
    assert "transitioned to COMPLETED" in res_complete.stdout
