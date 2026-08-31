"""Comprehensive unit and integration tests for Pre-Step 22.B Dataset Authorization Governance."""

import json
from pathlib import Path
import pytest
from safetensors.torch import save_file
import torch
from rernggen.data.dataset import (
    GovernanceMode,
    PairedLatentTextDataset,
    create_paired_dataloader,
)
from rernggen.data.governance import (
    GovernanceManager,
    PermissionDecision,
    compute_governance_record_sha256,
    format_permission_decision,
    parse_permission_decision,
)
from rernggen.data.schema import GovernanceRecord


def setup_fixture_paired_dataset(
    root_dir: Path,
    dataset_id: str = "fixture_gov_ds",
    count: int = 4,
) -> Path:
    """Creates a temporary isolated dataset fixture with latents, text embeddings, and captions."""
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

    for i in range(count):
        img_id = f"IMG-{i+1:06d}"
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

        text_recs.append(
            {
                "image_id": img_id,
                "dataset_id": dataset_id,
                "dataset_version": "v001",
                "caption_version": "captions_v002",
                "caption_sha256": f"sha_cap_{i}",
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
                "caption": f"Synthetic scene {i+1}",
                "caption_source": "synthetic",
                "caption_version": "captions_v002",
                "caption_sha256": f"sha_cap_{i}",
                "language": "en",
                "review_status": "reviewed",
                "training_allowed": None,
                "commercial_allowed": None,
                "license_id": None,
            }
        )

        manifest_recs.append(
            {
                "image_id": img_id,
                "dataset_id": dataset_id,
                "dataset_version": "v001",
                "original_filename": f"{img_id}.jpg",
                "source_relative_path": f"originals/{img_id}.jpg",
                "stored_relative_path": f"originals/{img_id}.jpg",
                "sha256": f"sha_orig_{i}",
                "width": 256,
                "height": 256,
                "format": "JPEG",
                "mode": "RGB",
                "training_allowed": None,
                "commercial_allowed": None,
                "license_id": None,
                "status": "IMPORTED",
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

    with open(manifest_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
        for r in manifest_recs:
            f.write(json.dumps(r) + "\n")

    return ds_dir


def test_tristate_parsing_and_formatting():
    """Verify explicit tri-state parsing and rejection of ambiguous decisions."""
    # 1. Parsing
    assert parse_permission_decision("ALLOW") is True
    assert parse_permission_decision("allow") is True
    assert parse_permission_decision(PermissionDecision.ALLOW) is True
    assert parse_permission_decision(True) is True

    assert parse_permission_decision("DENY") is False
    assert parse_permission_decision("deny") is False
    assert parse_permission_decision(PermissionDecision.DENY) is False
    assert parse_permission_decision(False) is False

    assert parse_permission_decision("UNKNOWN") is None
    assert parse_permission_decision("unknown") is None
    assert parse_permission_decision(PermissionDecision.UNKNOWN) is None
    assert parse_permission_decision(None) is None

    # 2. Rejection of invalid / ambiguous values
    with pytest.raises(ValueError, match="Invalid permission decision"):
        parse_permission_decision("MAYBE")

    with pytest.raises(ValueError, match="Invalid permission decision"):
        parse_permission_decision("YES")

    with pytest.raises(ValueError, match="Invalid permission decision type"):
        parse_permission_decision(123)

    # 3. Formatting
    assert format_permission_decision(True) == "ALLOW"
    assert format_permission_decision(False) == "DENY"
    assert format_permission_decision(None) == "UNKNOWN"


def test_mandatory_authorization_source_and_timestamp():
    """Verify authorization_source and authorized_at must be explicitly non-empty."""
    # Empty source rejected
    with pytest.raises(ValueError, match="must have a non-empty authorization_source"):
        GovernanceRecord(
            image_id="IMG-000001",
            dataset_id="test_ds",
            governance_version="rights_v001",
            training_allowed=True,
            commercial_allowed=False,
            authorization_source="",
            authorized_at="2026-08-31T00:00:00Z",
        )

    # Empty timestamp rejected
    with pytest.raises(ValueError, match="must have a non-empty authorized_at timestamp"):
        GovernanceRecord(
            image_id="IMG-000001",
            dataset_id="test_ds",
            governance_version="rights_v001",
            training_allowed=True,
            commercial_allowed=False,
            authorization_source="human_audit",
            authorized_at="",
        )

    # Invalid status rejected
    with pytest.raises(ValueError, match="Invalid governance status"):
        GovernanceRecord(
            image_id="IMG-000001",
            dataset_id="test_ds",
            governance_version="rights_v001",
            training_allowed=True,
            commercial_allowed=False,
            authorization_source="human_audit",
            authorized_at="2026-08-31T00:00:00Z",
            status="INVALID_STATUS",
        )


def test_deterministic_record_sha256():
    """Verify compute_governance_record_sha256 is deterministic and invariant to dict key ordering."""
    rec1 = GovernanceRecord(
        image_id="IMG-000001",
        dataset_id="test_ds",
        governance_version="rights_v001",
        training_allowed=True,
        commercial_allowed=False,
        authorization_source="human_audit",
        authorization_note="Explicit test note",
        authorized_at="2026-08-31T00:00:00Z",
    )
    rec2 = GovernanceRecord(
        image_id="IMG-000001",
        dataset_id="test_ds",
        governance_version="rights_v001",
        training_allowed=True,
        commercial_allowed=False,
        authorization_source="human_audit",
        authorization_note="Explicit test note",
        authorized_at="2026-08-31T00:00:00Z",
    )

    sha1 = compute_governance_record_sha256(rec1)
    sha2 = compute_governance_record_sha256(rec2)
    assert sha1 == sha2
    assert len(sha1) == 64

    # Dict representation with different insertion order produces exact same hash
    d1 = rec1.to_dict()
    d2 = {k: d1[k] for k in reversed(list(d1.keys()))}
    assert compute_governance_record_sha256(d1) == compute_governance_record_sha256(d2)


def test_single_and_multi_image_authorization(tmp_path: Path):
    """Verify single-image and multi-image authorization workflow with mandatory source."""
    gov_mgr = GovernanceManager(dataset_root=tmp_path)
    dataset_id = "test_gov_ds"
    version = "rights_v001"

    # Missing authorization_source in authorize_samples raises ValueError
    with pytest.raises(ValueError, match="authorization_source is required"):
        gov_mgr.authorize_samples(
            dataset_id=dataset_id,
            governance_version=version,
            image_ids="IMG-000001",
            training_decision="ALLOW",
            commercial_decision="DENY",
            authorization_source="",
        )

    # 1. Authorize IMG-000001 as ALLOW, IMG-000002 as DENY in rights_v001
    p, recs = gov_mgr.authorize_samples(
        dataset_id=dataset_id,
        governance_version=version,
        image_ids=["IMG-000001", "IMG-000002"],
        training_decision="ALLOW",
        commercial_decision="DENY",
        authorization_source="human_audit",
        authorization_note="Owner consent granted for training.",
    )
    assert p.exists()
    assert len(recs) == 2
    assert recs[0].training_allowed is True
    assert recs[0].commercial_allowed is False
    assert recs[0].authorization_source == "human_audit"
    assert len(recs[0].authorized_at) > 0


def test_no_accidental_implicit_all_and_empty_id_rejection(tmp_path: Path):
    """Verify that omitting image IDs or passing empty list raises ValueError."""
    gov_mgr = GovernanceManager(dataset_root=tmp_path)
    with pytest.raises(ValueError, match="No valid image IDs specified"):
        gov_mgr.authorize_samples(
            dataset_id="test_ds",
            governance_version="rights_v001",
            image_ids=[],
            training_decision="ALLOW",
            commercial_decision="ALLOW",
            authorization_source="human_audit",
        )

    with pytest.raises(ValueError, match="Cannot authorize 'ALL' without providing explicit all_dataset_ids"):
        gov_mgr.authorize_samples(
            dataset_id="test_ds",
            governance_version="rights_v001",
            image_ids="ALL",
            training_decision="ALLOW",
            commercial_decision="ALLOW",
            authorization_source="human_audit",
            all_dataset_ids=None,
        )


def test_strict_version_immutability_and_superseding(tmp_path: Path):
    """Verify finalized governance versions are strictly immutable and require superseding in new version."""
    gov_mgr = GovernanceManager(dataset_root=tmp_path)
    dataset_id = "test_immut_ds"

    # 1. Create finalized rights_v001
    gov_mgr.authorize_samples(
        dataset_id=dataset_id,
        governance_version="rights_v001",
        image_ids=["IMG-000001", "IMG-000002"],
        training_decision="ALLOW",
        commercial_decision="UNKNOWN",
        authorization_source="initial_audit",
    )

    # 2. Attempting to authorize/save against existing rights_v001 raises FileExistsError
    with pytest.raises(FileExistsError, match="already exists and is finalized"):
        gov_mgr.authorize_samples(
            dataset_id=dataset_id,
            governance_version="rights_v001",
            image_ids=["IMG-000001"],
            training_decision="DENY",
            commercial_decision="DENY",
            authorization_source="new_audit",
        )

    with pytest.raises(FileExistsError, match="strictly immutable"):
        gov_mgr.save_governance(
            dataset_id=dataset_id,
            version="rights_v001",
            records=[
                GovernanceRecord(
                    image_id="IMG-000001",
                    dataset_id=dataset_id,
                    governance_version="rights_v001",
                    training_allowed=False,
                    commercial_allowed=False,
                    authorization_source="new_audit",
                    authorized_at="2026-08-31T00:00:00Z",
                )
            ],
        )

    # 3. Create rights_v002 superseding rights_v001 (changing IMG-000002 to DENY)
    _, recs_v2 = gov_mgr.authorize_samples(
        dataset_id=dataset_id,
        governance_version="rights_v002",
        image_ids=["IMG-000002"],
        training_decision="DENY",
        commercial_decision="DENY",
        authorization_source="human_revocation",
        base_version="rights_v001",
    )
    assert len(recs_v2) == 2
    assert recs_v2[0].image_id == "IMG-000001" and recs_v2[0].training_allowed is True
    assert recs_v2[1].image_id == "IMG-000002" and recs_v2[1].training_allowed is False
    assert recs_v2[1].previous_governance_version == "rights_v001"

    # Version list shows both versions
    versions = gov_mgr.list_versions(dataset_id)
    assert versions == ["rights_v001", "rights_v002"]


def test_atomic_write_and_failure_resilience(tmp_path: Path, monkeypatch):
    """Verify that a write failure leaves the previous valid manifest untouched."""
    gov_mgr = GovernanceManager(dataset_root=tmp_path)
    dataset_id = "test_atomic_ds"
    version = "rights_v001"

    # Create initial valid manifest
    p1, _ = gov_mgr.authorize_samples(
        dataset_id=dataset_id,
        governance_version=version,
        image_ids=["IMG-000001"],
        training_decision="ALLOW",
        commercial_decision="UNKNOWN",
        authorization_source="initial_audit",
    )
    orig_bytes = p1.read_bytes()

    # Simulate filesystem error during atomic replace
    def failing_replace(*args, **kwargs):
        raise OSError("Simulated atomic replace failure!")

    monkeypatch.setattr("os.replace", failing_replace)

    with pytest.raises(OSError, match="Simulated atomic replace failure"):
        gov_mgr.save_governance(
            dataset_id=dataset_id,
            version=version,
            records=[
                GovernanceRecord(
                    image_id="IMG-000001",
                    dataset_id=dataset_id,
                    governance_version=version,
                    training_allowed=False,
                    commercial_allowed=False,
                    authorization_source="test_audit",
                    authorized_at="2026-08-31T00:00:00Z",
                )
            ],
            _allow_test_overwrite=True,
        )

    # Original manifest must remain unchanged
    assert p1.read_bytes() == orig_bytes


def test_duplicate_image_id_rejection(tmp_path: Path):
    """Verify duplicate image IDs in record set are rejected."""
    gov_mgr = GovernanceManager(dataset_root=tmp_path)
    r1 = GovernanceRecord(
        image_id="IMG-000001",
        dataset_id="ds",
        governance_version="v1",
        training_allowed=True,
        commercial_allowed=True,
        authorization_source="src",
        authorized_at="2026-08-31T00:00:00Z",
    )
    r2 = GovernanceRecord(
        image_id="IMG-000001",
        dataset_id="ds",
        governance_version="v1",
        training_allowed=False,
        commercial_allowed=False,
        authorization_source="src",
        authorized_at="2026-08-31T00:00:00Z",
    )
    with pytest.raises(ValueError, match="Duplicate image_id"):
        gov_mgr.save_governance("ds", "v1", [r1, r2])


def test_effective_rights_resolution_and_manifest_order_independence(tmp_path: Path):
    """Verify PairedLatentTextDataset joins governance records strictly by image_id regardless of manifest ordering."""
    ds_dir = setup_fixture_paired_dataset(tmp_path, "ds_order_test", count=4)
    gov_mgr = GovernanceManager(dataset_root=tmp_path)

    # Save with reverse order in manifest to test order-independence
    r1 = GovernanceRecord(image_id="IMG-000004", dataset_id="ds_order_test", governance_version="rights_v001", training_allowed=None, commercial_allowed=None, authorization_source="audit", authorized_at="2026-08-31T00:00:00Z")
    r2 = GovernanceRecord(image_id="IMG-000003", dataset_id="ds_order_test", governance_version="rights_v001", training_allowed=True, commercial_allowed=False, authorization_source="audit", authorized_at="2026-08-31T00:00:00Z")
    r3 = GovernanceRecord(image_id="IMG-000002", dataset_id="ds_order_test", governance_version="rights_v001", training_allowed=False, commercial_allowed=False, authorization_source="audit", authorized_at="2026-08-31T00:00:00Z")
    r4 = GovernanceRecord(image_id="IMG-000001", dataset_id="ds_order_test", governance_version="rights_v001", training_allowed=True, commercial_allowed=True, authorization_source="audit", authorized_at="2026-08-31T00:00:00Z")
    gov_mgr.save_governance("ds_order_test", "rights_v001", [r1, r2, r3, r4])

    dataset = PairedLatentTextDataset(
        dataset_dir=ds_dir,
        governance_version="rights_v001",
        governance_mode=GovernanceMode.DEVELOPMENT_AUDIT,
    )
    assert len(dataset) == 4
    assert dataset.governance_counts == {"allowed": 2, "denied": 1, "unknown": 1}

    # Verify per-sample governance contract
    s0 = dataset[0]  # IMG-000001
    assert s0["image_id"] == "IMG-000001"
    assert s0["governance"]["training_allowed"] is True
    assert s0["governance"]["commercial_allowed"] is True
    assert s0["governance"]["authorization_source"] == "audit"

    s1 = dataset[1]  # IMG-000002
    assert s1["image_id"] == "IMG-000002"
    assert s1["governance"]["training_allowed"] is False

    s2 = dataset[2]  # IMG-000003
    assert s2["image_id"] == "IMG-000003"
    assert s2["governance"]["training_allowed"] is True

    s3 = dataset[3]  # IMG-000004
    assert s3["image_id"] == "IMG-000004"
    assert s3["governance"]["training_allowed"] is None


def test_missing_governance_record_defaults_to_unknown(tmp_path: Path):
    """Verify that omitting an item from the governance manifest keeps its effective rights as unknown (None)."""
    ds_dir = setup_fixture_paired_dataset(tmp_path, "ds_missing_rec", count=3)
    gov_mgr = GovernanceManager(dataset_root=tmp_path)

    # Only authorize IMG-000001; IMG-000002 and IMG-000003 omitted from governance manifest
    gov_mgr.authorize_samples(
        dataset_id="ds_missing_rec",
        governance_version="rights_v001",
        image_ids=["IMG-000001"],
        training_decision="ALLOW",
        commercial_decision="ALLOW",
        authorization_source="audit",
    )

    dataset = PairedLatentTextDataset(
        dataset_dir=ds_dir,
        governance_version="rights_v001",
        governance_mode=GovernanceMode.DEVELOPMENT_AUDIT,
    )
    assert dataset.governance_counts == {"allowed": 1, "denied": 0, "unknown": 2}
    assert dataset[1]["governance"]["training_allowed"] is None
    assert dataset[2]["governance"]["training_allowed"] is None


def test_status_participation_in_effective_authorization(tmp_path: Path):
    """Verify that record status participates in production authorization (REVOKED/SUPERSEDED rejected)."""
    ds_dir = setup_fixture_paired_dataset(tmp_path, "ds_status_test", count=6)
    gov_mgr = GovernanceManager(dataset_root=tmp_path)

    records = [
        # 1. ACTIVE + ALLOW -> ALLOWED (passes gate)
        GovernanceRecord(image_id="IMG-000001", dataset_id="ds_status_test", governance_version="v_status", training_allowed=True, commercial_allowed=True, authorization_source="audit", authorized_at="2026-08-31T00:00:00Z", status="ACTIVE"),
        # 2. ACTIVE + DENY -> DENIED (rejected by gate)
        GovernanceRecord(image_id="IMG-000002", dataset_id="ds_status_test", governance_version="v_status", training_allowed=False, commercial_allowed=False, authorization_source="audit", authorized_at="2026-08-31T00:00:00Z", status="ACTIVE"),
        # 3. ACTIVE + UNKNOWN -> UNKNOWN (rejected by gate)
        GovernanceRecord(image_id="IMG-000003", dataset_id="ds_status_test", governance_version="v_status", training_allowed=None, commercial_allowed=None, authorization_source="audit", authorized_at="2026-08-31T00:00:00Z", status="ACTIVE"),
        # 4. REVOKED + ALLOW -> DENIED / REJECTED (must NEVER pass training)
        GovernanceRecord(image_id="IMG-000004", dataset_id="ds_status_test", governance_version="v_status", training_allowed=True, commercial_allowed=True, authorization_source="audit", authorized_at="2026-08-31T00:00:00Z", status="REVOKED"),
        # 5. SUPERSEDED + ALLOW -> DENIED / REJECTED (must NEVER pass training)
        GovernanceRecord(image_id="IMG-000005", dataset_id="ds_status_test", governance_version="v_status", training_allowed=True, commercial_allowed=True, authorization_source="audit", authorized_at="2026-08-31T00:00:00Z", status="SUPERSEDED"),
        # 6. IMG-000006 is omitted from manifest (missing record -> UNKNOWN / rejected)
    ]
    gov_mgr.save_governance("ds_status_test", "v_status", records)

    dataset = PairedLatentTextDataset(
        dataset_dir=ds_dir,
        governance_version="v_status",
        governance_mode=GovernanceMode.DEVELOPMENT_AUDIT,
    )
    # Expected: allowed=1 (only IMG-000001), denied=3 (IMG-000002, IMG-000004, IMG-000005), unknown=2 (IMG-000003, IMG-000006)
    assert dataset.governance_counts["allowed"] == 1
    assert dataset.governance_counts["denied"] == 3
    assert dataset.governance_counts["unknown"] == 2

    # PRODUCTION_STRICT gate must reject because samples 2, 3, 4, 5, 6 lack valid active authorization
    with pytest.raises(PermissionError, match="Production training gate rejected dataset"):
        PairedLatentTextDataset(
            dataset_dir=ds_dir,
            governance_version="v_status",
            governance_mode=GovernanceMode.PRODUCTION_STRICT,
        )


def test_production_strict_requires_explicit_governance_version(tmp_path: Path):
    """Verify PRODUCTION_STRICT requires explicit governance_version and fails closed if omitted or missing."""
    ds_dir = setup_fixture_paired_dataset(tmp_path, "ds_strict_version", count=2)

    # 1. governance_version=None in PRODUCTION_STRICT raises ValueError
    with pytest.raises(ValueError, match="PRODUCTION_STRICT requires an explicitly specified governance_version"):
        PairedLatentTextDataset(
            dataset_dir=ds_dir,
            governance_version=None,
            governance_mode=GovernanceMode.PRODUCTION_STRICT,
        )

    # 2. Non-existent governance_version in PRODUCTION_STRICT raises FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Required governance manifest for version 'non_existent_version' not found"):
        PairedLatentTextDataset(
            dataset_dir=ds_dir,
            governance_version="non_existent_version",
            governance_mode=GovernanceMode.PRODUCTION_STRICT,
        )


def test_governance_provenance_and_manifest_hash(tmp_path: Path):
    """Verify complete governance provenance and manifest hash are tracked on the dataset instance."""
    ds_dir = setup_fixture_paired_dataset(tmp_path, "ds_prov_test", count=2)
    gov_mgr = GovernanceManager(dataset_root=tmp_path)

    gov_mgr.authorize_samples(
        dataset_id="ds_prov_test",
        governance_version="rights_v001",
        image_ids=["IMG-000001", "IMG-000002"],
        training_decision="ALLOW",
        commercial_decision="UNKNOWN",
        authorization_source="audit_lead",
    )

    dataset = PairedLatentTextDataset(
        dataset_dir=ds_dir,
        governance_version="rights_v001",
        governance_mode=GovernanceMode.PRODUCTION_STRICT,
    )

    prov = dataset.governance_provenance
    assert prov["dataset_id"] == "ds_prov_test"
    assert prov["governance_version"] == "rights_v001"
    assert isinstance(prov["governance_manifest_sha256"], str)
    assert len(prov["governance_manifest_sha256"]) == 64
    assert prov["governance_mode"] == "production_strict"
    assert prov["governance_counts"] == {"allowed": 2, "denied": 0, "unknown": 0}


def test_commercial_and_training_independence(tmp_path: Path):
    """Verify commercial permission is completely independent and does not affect training permission."""
    ds_dir = setup_fixture_paired_dataset(tmp_path, "ds_indep_test", count=2)
    gov_mgr = GovernanceManager(dataset_root=tmp_path)

    # Training = ALLOW, Commercial = DENY
    gov_mgr.authorize_samples(
        dataset_id="ds_indep_test",
        governance_version="rights_v001",
        image_ids=["IMG-000001", "IMG-000002"],
        training_decision="ALLOW",
        commercial_decision="DENY",
        authorization_source="legal_audit",
    )

    # Training gate passes because training_allowed is True and status is ACTIVE
    ds = PairedLatentTextDataset(
        dataset_dir=ds_dir,
        governance_version="rights_v001",
        governance_mode=GovernanceMode.PRODUCTION_STRICT,
    )
    assert ds.governance_counts == {"allowed": 2, "denied": 0, "unknown": 0}
    assert ds[0]["governance"]["training_allowed"] is True
    assert ds[0]["governance"]["commercial_allowed"] is False
