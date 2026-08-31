"""Comprehensive unit and integration tests for Step 22.C Human Caption Review & Invalidation Workflow."""

import json
from pathlib import Path
import pytest
from safetensors.torch import save_file
import torch
from rernggen.data.caption_review import (
    CaptionReviewManager,
    CaptionReviewStatus,
    compute_caption_review_record_sha256,
    parse_caption_review_status,
)
from rernggen.data.captions import CaptionManager, compute_caption_sha256
from rernggen.data.dataset import (
    GovernanceMode,
    PairedLatentTextDataset,
    create_paired_dataloader,
)
from rernggen.data.governance import GovernanceManager
from rernggen.data.schema import CaptionRecord, CaptionReviewRecord, GovernanceRecord


def setup_fixture_paired_dataset_with_captions(
    root_dir: Path,
    dataset_id: str = "fixture_cap_rev_ds",
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

        cap_text = f"Synthetic Khmer scene {i+1}"
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


def test_caption_review_status_parsing():
    """Verify parsing and validation of caption review status values."""
    assert parse_caption_review_status("APPROVED") == "APPROVED"
    assert parse_caption_review_status("approved") == "APPROVED"
    assert parse_caption_review_status(CaptionReviewStatus.APPROVED) == "APPROVED"

    assert parse_caption_review_status("REJECTED") == "REJECTED"
    assert parse_caption_review_status("rejected") == "REJECTED"
    assert parse_caption_review_status(CaptionReviewStatus.REJECTED) == "REJECTED"

    assert parse_caption_review_status("INVALIDATED") == "INVALIDATED"
    assert parse_caption_review_status("invalidated") == "INVALIDATED"
    assert parse_caption_review_status(CaptionReviewStatus.INVALIDATED) == "INVALIDATED"

    assert parse_caption_review_status("PENDING") == "PENDING"
    assert parse_caption_review_status("pending") == "PENDING"
    assert parse_caption_review_status(CaptionReviewStatus.PENDING) == "PENDING"

    with pytest.raises(ValueError, match="Invalid caption review status"):
        parse_caption_review_status("UNKNOWN")

    with pytest.raises(ValueError, match="Invalid caption review status"):
        parse_caption_review_status("GOOD")


def test_caption_review_record_validation():
    """Verify strict field validation on CaptionReviewRecord."""
    # 1. Empty reviewed_by rejected
    with pytest.raises(ValueError, match="must have a non-empty reviewed_by"):
        CaptionReviewRecord(
            image_id="IMG-000001",
            dataset_id="ds",
            review_version="v001",
            caption_sha256="abc123sha",
            review_status="APPROVED",
            reviewed_by="",
            review_source="human_audit",
            reviewed_at="2026-08-31T00:00:00Z",
        )

    # 2. Empty review_source rejected
    with pytest.raises(ValueError, match="must have a non-empty review_source"):
        CaptionReviewRecord(
            image_id="IMG-000001",
            dataset_id="ds",
            review_version="v001",
            caption_sha256="abc123sha",
            review_status="APPROVED",
            reviewed_by="reviewer_alice",
            review_source="   ",
            reviewed_at="2026-08-31T00:00:00Z",
        )

    # 3. Empty reviewed_at rejected
    with pytest.raises(ValueError, match="must have a non-empty reviewed_at timestamp"):
        CaptionReviewRecord(
            image_id="IMG-000001",
            dataset_id="ds",
            review_version="v001",
            caption_sha256="abc123sha",
            review_status="APPROVED",
            reviewed_by="reviewer_alice",
            review_source="human_audit",
            reviewed_at="",
        )

    # 4. Empty caption_sha256 rejected
    with pytest.raises(ValueError, match="must have a non-empty caption_sha256"):
        CaptionReviewRecord(
            image_id="IMG-000001",
            dataset_id="ds",
            review_version="v001",
            caption_sha256="",
            review_status="APPROVED",
            reviewed_by="reviewer_alice",
            review_source="human_audit",
            reviewed_at="2026-08-31T00:00:00Z",
        )

    # 5. Invalid status rejected
    with pytest.raises(ValueError, match="Invalid caption review status"):
        CaptionReviewRecord(
            image_id="IMG-000001",
            dataset_id="ds",
            review_version="v001",
            caption_sha256="abc123sha",
            review_status="UNVERIFIED",
            reviewed_by="reviewer_alice",
            review_source="human_audit",
            reviewed_at="2026-08-31T00:00:00Z",
        )


def test_deterministic_caption_review_record_sha256():
    """Verify deterministic hash computation for review records."""
    rec1 = CaptionReviewRecord(
        image_id="IMG-000001",
        dataset_id="ds",
        review_version="v001",
        caption_sha256="abc123sha",
        review_status="APPROVED",
        reviewed_by="reviewer_alice",
        review_source="human_audit",
        reviewed_at="2026-08-31T00:00:00Z",
        reason="Clear description of Angkor Wat temple",
    )
    rec2 = CaptionReviewRecord(
        image_id="IMG-000001",
        dataset_id="ds",
        review_version="v001",
        caption_sha256="abc123sha",
        review_status="APPROVED",
        reviewed_by="reviewer_alice",
        review_source="human_audit",
        reviewed_at="2026-08-31T00:00:00Z",
        reason="Clear description of Angkor Wat temple",
    )

    sha1 = compute_caption_review_record_sha256(rec1)
    sha2 = compute_caption_review_record_sha256(rec2)
    assert sha1 == sha2
    assert len(sha1) == 64

    # Insertion order invariance
    d1 = rec1.to_dict()
    d2 = {k: d1[k] for k in reversed(list(d1.keys()))}
    assert compute_caption_review_record_sha256(d1) == compute_caption_review_record_sha256(d2)


def test_single_and_multi_caption_review_workflow(tmp_path: Path):
    """Verify review_samples operation creating review manifests bound to caption hashes."""
    ds_dir = setup_fixture_paired_dataset_with_captions(tmp_path, "ds_rev_flow", count=3)
    rev_mgr = CaptionReviewManager(dataset_root=tmp_path)

    # 1. Review IMG-000001 as APPROVED and IMG-000002 as REJECTED
    p, recs = rev_mgr.review_samples(
        dataset_id="ds_rev_flow",
        review_version="caption_review_v001",
        image_ids=["IMG-000001", "IMG-000002"],
        review_status="APPROVED",
        reviewed_by="reviewer_lead",
        review_source="expert_human_review",
        reason="Detailed semantic accuracy verified",
    )
    assert p.exists()
    assert len(recs) == 2
    assert recs[0].image_id == "IMG-000001" and recs[0].caption_sha256 == compute_caption_sha256("Synthetic Khmer scene 1")
    assert recs[0].review_status == "APPROVED"
    assert recs[1].image_id == "IMG-000002" and recs[1].caption_sha256 == compute_caption_sha256("Synthetic Khmer scene 2")

    # 2. Add IMG-000003 as REJECTED in superseding version
    p_v2, recs_v2 = rev_mgr.review_samples(
        dataset_id="ds_rev_flow",
        review_version="caption_review_v002",
        image_ids=["IMG-000003"],
        review_status="REJECTED",
        reviewed_by="reviewer_lead",
        review_source="expert_human_review",
        reason="Caption mentions non-existent character",
        base_version="caption_review_v001",
    )
    assert len(recs_v2) == 3
    assert recs_v2[0].image_id == "IMG-000001" and recs_v2[0].review_status == "APPROVED"
    assert recs_v2[1].image_id == "IMG-000002" and recs_v2[1].review_status == "APPROVED"
    assert recs_v2[2].image_id == "IMG-000003" and recs_v2[2].review_status == "REJECTED"
    assert recs_v2[2].previous_review_version == "caption_review_v001"


def test_review_unknown_sample_id_fails(tmp_path: Path):
    """Verify that attempting to review a sample ID not in caption manifest raises ValueError."""
    setup_fixture_paired_dataset_with_captions(tmp_path, "ds_unknown_sample", count=2)
    rev_mgr = CaptionReviewManager(dataset_root=tmp_path)

    with pytest.raises(ValueError, match="Image ID 'IMG-999999' not found"):
        rev_mgr.review_samples(
            dataset_id="ds_unknown_sample",
            review_version="caption_review_v001",
            image_ids=["IMG-999999"],
            review_status="APPROVED",
            reviewed_by="reviewer_alice",
            review_source="audit",
        )


def test_review_version_immutability(tmp_path: Path):
    """Verify that finalized caption review versions cannot be overwritten."""
    setup_fixture_paired_dataset_with_captions(tmp_path, "ds_immut", count=2)
    rev_mgr = CaptionReviewManager(dataset_root=tmp_path)

    rev_mgr.review_samples(
        dataset_id="ds_immut",
        review_version="caption_review_v001",
        image_ids=["IMG-000001"],
        review_status="APPROVED",
        reviewed_by="reviewer_alice",
        review_source="audit",
    )

    with pytest.raises(FileExistsError, match="already exists and is finalized"):
        rev_mgr.review_samples(
            dataset_id="ds_immut",
            review_version="caption_review_v001",
            image_ids=["IMG-000002"],
            review_status="APPROVED",
            reviewed_by="reviewer_alice",
            review_source="audit",
        )


def test_atomic_review_write_and_failure_resilience(tmp_path: Path, monkeypatch):
    """Verify atomic replace protects existing review manifests from corruption."""
    setup_fixture_paired_dataset_with_captions(tmp_path, "ds_atomic", count=2)
    rev_mgr = CaptionReviewManager(dataset_root=tmp_path)

    p1, _ = rev_mgr.review_samples(
        dataset_id="ds_atomic",
        review_version="caption_review_v001",
        image_ids=["IMG-000001"],
        review_status="APPROVED",
        reviewed_by="reviewer_alice",
        review_source="audit",
    )
    orig_bytes = p1.read_bytes()

    def failing_replace(*args, **kwargs):
        raise OSError("Simulated atomic replace failure!")

    monkeypatch.setattr("os.replace", failing_replace)

    with pytest.raises(OSError, match="Simulated atomic replace failure"):
        rev_mgr.save_reviews(
            dataset_id="ds_atomic",
            version="caption_review_v001",
            records=[
                CaptionReviewRecord(
                    image_id="IMG-000001",
                    dataset_id="ds_atomic",
                    review_version="caption_review_v001",
                    caption_sha256="sha_cap_0",
                    review_status="REJECTED",
                    reviewed_by="reviewer_alice",
                    review_source="audit",
                    reviewed_at="2026-08-31T00:00:00Z",
                )
            ],
            _allow_test_overwrite=True,
        )

    # Manifest remains uncorrupted
    assert p1.read_bytes() == orig_bytes


def test_caption_hash_mismatch_triggers_invalidation(tmp_path: Path):
    """Verify that editing caption text after approval invalidates effective review status."""
    ds_dir = setup_fixture_paired_dataset_with_captions(tmp_path, "ds_inval_test", count=2)
    rev_mgr = CaptionReviewManager(dataset_root=tmp_path)
    cap_mgr = CaptionManager(dataset_root=tmp_path)

    # 1. Approve both samples against initial captions (sha_cap_0, sha_cap_1)
    rev_mgr.review_samples(
        dataset_id="ds_inval_test",
        review_version="caption_review_v001",
        image_ids=["IMG-000001", "IMG-000002"],
        review_status="APPROVED",
        reviewed_by="reviewer_alice",
        review_source="audit",
    )

    # 2. Modify caption text of IMG-000002 in captions_v002
    current_recs = cap_mgr.load_captions(dataset_id="ds_inval_test", version="captions_v002")
    updated_recs = []
    for r in current_recs:
        if r.image_id == "IMG-000002":
            new_r = cap_mgr.create_caption_record(
                image_id="IMG-000002",
                dataset_id="ds_inval_test",
                caption="Heavily edited Khmer story description (new text)",
                caption_version="captions_v002",
            )
            updated_recs.append(new_r)
        else:
            updated_recs.append(r)
    cap_mgr.save_captions("ds_inval_test", updated_recs, version="captions_v002")

    # 3. Load dataset with caption_review_v001
    dataset = PairedLatentTextDataset(
        dataset_dir=ds_dir,
        caption_review_version="caption_review_v001",
        governance_mode=GovernanceMode.DEVELOPMENT_AUDIT,
    )

    # IMG-000001 is approved, IMG-000002 is INVALIDATED due to hash mismatch
    assert dataset.caption_review_counts["approved"] == 1
    assert dataset.caption_review_counts["invalidated"] == 1

    s0 = dataset[0]  # IMG-000001
    assert s0["caption_review"]["review_status"] == "APPROVED"
    assert s0["caption_review"]["caption_hash_match"] is True

    s1 = dataset[1]  # IMG-000002
    assert s1["caption_review"]["review_status"] == "INVALIDATED"
    assert s1["caption_review"]["caption_hash_match"] is False


def test_caption_review_state_matrix_in_dataset(tmp_path: Path):
    """Verify all review states (APPROVED, REJECTED, INVALIDATED, PENDING, missing record) in dataset."""
    ds_dir = setup_fixture_paired_dataset_with_captions(tmp_path, "ds_matrix", count=5)
    rev_mgr = CaptionReviewManager(dataset_root=tmp_path)

    records = [
        # 1. APPROVED + matching hash
        CaptionReviewRecord(image_id="IMG-000001", dataset_id="ds_matrix", review_version="rev_v1", caption_sha256=compute_caption_sha256("Synthetic Khmer scene 1"), review_status="APPROVED", reviewed_by="alice", review_source="audit", reviewed_at="2026-08-31T00:00:00Z"),
        # 2. REJECTED + matching hash
        CaptionReviewRecord(image_id="IMG-000002", dataset_id="ds_matrix", review_version="rev_v1", caption_sha256=compute_caption_sha256("Synthetic Khmer scene 2"), review_status="REJECTED", reviewed_by="alice", review_source="audit", reviewed_at="2026-08-31T00:00:00Z"),
        # 3. INVALIDATED explicitly
        CaptionReviewRecord(image_id="IMG-000003", dataset_id="ds_matrix", review_version="rev_v1", caption_sha256=compute_caption_sha256("Synthetic Khmer scene 3"), review_status="INVALIDATED", reviewed_by="alice", review_source="audit", reviewed_at="2026-08-31T00:00:00Z"),
        # 4. PENDING explicitly
        CaptionReviewRecord(image_id="IMG-000004", dataset_id="ds_matrix", review_version="rev_v1", caption_sha256=compute_caption_sha256("Synthetic Khmer scene 4"), review_status="PENDING", reviewed_by="alice", review_source="audit", reviewed_at="2026-08-31T00:00:00Z"),
        # 5. IMG-000005 is missing from review manifest -> PENDING
    ]
    rev_mgr.save_reviews("ds_matrix", "rev_v1", records)

    dataset = PairedLatentTextDataset(
        dataset_dir=ds_dir,
        caption_review_version="rev_v1",
        governance_mode=GovernanceMode.DEVELOPMENT_AUDIT,
    )

    counts = dataset.caption_review_counts
    assert counts["approved"] == 1
    assert counts["rejected"] == 1
    assert counts["invalidated"] == 1
    assert counts["pending"] == 2  # IMG-000004 and IMG-000005


def test_strict_mode_rejects_unapproved_captions(tmp_path: Path):
    """Verify PRODUCTION_STRICT gate enforces approved and matching caption reviews."""
    ds_dir = setup_fixture_paired_dataset_with_captions(tmp_path, "ds_strict_cap", count=2)
    gov_mgr = GovernanceManager(dataset_root=tmp_path)
    rev_mgr = CaptionReviewManager(dataset_root=tmp_path)

    # 1. Authorize governance for both samples
    gov_mgr.authorize_samples(
        dataset_id="ds_strict_cap",
        governance_version="rights_v001",
        image_ids=["IMG-000001", "IMG-000002"],
        training_decision="ALLOW",
        commercial_decision="ALLOW",
        authorization_source="human_audit",
    )

    # 2. Only approve IMG-000001; IMG-000002 is REJECTED
    rev_mgr.review_samples(
        dataset_id="ds_strict_cap",
        review_version="rev_partial",
        image_ids=["IMG-000001"],
        review_status="APPROVED",
        reviewed_by="alice",
        review_source="audit",
    )
    rev_mgr.review_samples(
        dataset_id="ds_strict_cap",
        review_version="rev_partial",
        image_ids=["IMG-000002"],
        review_status="REJECTED",
        reviewed_by="alice",
        review_source="audit",
        base_version="rev_partial",
        _allow_test_overwrite=True,
    )

    # PRODUCTION_STRICT fails because IMG-000002 is rejected
    with pytest.raises(PermissionError, match="ineligible for training"):
        PairedLatentTextDataset(
            dataset_dir=ds_dir,
            governance_version="rights_v001",
            caption_review_version="rev_partial",
            governance_mode=GovernanceMode.PRODUCTION_STRICT,
        )

    # 3. Approve 100% of samples -> PRODUCTION_STRICT PASSES
    rev_mgr.review_samples(
        dataset_id="ds_strict_cap",
        review_version="rev_all_pass",
        image_ids=["IMG-000001", "IMG-000002"],
        review_status="APPROVED",
        reviewed_by="alice",
        review_source="audit",
    )

    ds_pass = PairedLatentTextDataset(
        dataset_dir=ds_dir,
        governance_version="rights_v001",
        caption_review_version="rev_all_pass",
        governance_mode=GovernanceMode.PRODUCTION_STRICT,
    )
    assert len(ds_pass) == 2
    assert ds_pass.caption_review_counts["approved"] == 2


def test_independence_of_governance_and_caption_review(tmp_path: Path):
    """Verify that rights authorization and caption review remain strictly orthogonal."""
    ds_dir = setup_fixture_paired_dataset_with_captions(tmp_path, "ds_indep", count=2)
    gov_mgr = GovernanceManager(dataset_root=tmp_path)
    rev_mgr = CaptionReviewManager(dataset_root=tmp_path)

    # Rights = DENIED, Caption = APPROVED
    gov_mgr.authorize_samples(
        dataset_id="ds_indep",
        governance_version="rights_v001",
        image_ids=["IMG-000001", "IMG-000002"],
        training_decision="DENY",
        commercial_decision="DENY",
        authorization_source="audit",
    )
    rev_mgr.review_samples(
        dataset_id="ds_indep",
        review_version="cap_rev_v001",
        image_ids=["IMG-000001", "IMG-000002"],
        review_status="APPROVED",
        reviewed_by="alice",
        review_source="audit",
    )

    # Auditing dataset confirms rights denied (2) while captions approved (2)
    dataset = PairedLatentTextDataset(
        dataset_dir=ds_dir,
        governance_version="rights_v001",
        caption_review_version="cap_rev_v001",
        governance_mode=GovernanceMode.DEVELOPMENT_AUDIT,
    )
    assert dataset.governance_counts["denied"] == 2
    assert dataset.governance_counts["allowed"] == 0
    assert dataset.caption_review_counts["approved"] == 2
    assert dataset.caption_review_counts["rejected"] == 0

    # Production gate rejects due to rights
    with pytest.raises(PermissionError, match="ineligible for training"):
        PairedLatentTextDataset(
            dataset_dir=ds_dir,
            governance_version="rights_v001",
            caption_review_version="cap_rev_v001",
            governance_mode=GovernanceMode.PRODUCTION_STRICT,
        )


def test_caption_review_provenance_and_manifest_hash(tmp_path: Path):
    """Verify caption review provenance metadata and manifest hash tracking."""
    ds_dir = setup_fixture_paired_dataset_with_captions(tmp_path, "ds_prov_test", count=2)
    rev_mgr = CaptionReviewManager(dataset_root=tmp_path)

    rev_mgr.review_samples(
        dataset_id="ds_prov_test",
        review_version="caption_review_v001",
        image_ids=["IMG-000001", "IMG-000002"],
        review_status="APPROVED",
        reviewed_by="reviewer_dan",
        review_source="formal_review",
    )

    dataset = PairedLatentTextDataset(
        dataset_dir=ds_dir,
        caption_review_version="caption_review_v001",
        governance_mode=GovernanceMode.DEVELOPMENT_AUDIT,
    )

    prov = dataset.caption_review_provenance
    assert prov["dataset_id"] == "ds_prov_test"
    assert prov["caption_review_version"] == "caption_review_v001"
    assert isinstance(prov["caption_review_manifest_sha256"], str)
    assert len(prov["caption_review_manifest_sha256"]) == 64
    assert prov["caption_review_counts"] == {"approved": 2, "rejected": 0, "pending": 0, "invalidated": 0}


def test_missing_caption_review_record_rejected_in_strict_mode(tmp_path: Path):
    """Verify that a sample missing from caption review manifest fails PRODUCTION_STRICT."""
    ds_dir = setup_fixture_paired_dataset_with_captions(tmp_path, "ds_missing_rev_strict", count=2)
    gov_mgr = GovernanceManager(dataset_root=tmp_path)
    rev_mgr = CaptionReviewManager(dataset_root=tmp_path)

    # Governance authorized for both
    gov_mgr.authorize_samples(
        dataset_id="ds_missing_rev_strict",
        governance_version="rights_v001",
        image_ids=["IMG-000001", "IMG-000002"],
        training_decision="ALLOW",
        commercial_decision="ALLOW",
        authorization_source="audit",
    )

    # Only IMG-000001 is reviewed; IMG-000002 is omitted from review manifest
    rev_mgr.review_samples(
        dataset_id="ds_missing_rev_strict",
        review_version="rev_only_one",
        image_ids=["IMG-000001"],
        review_status="APPROVED",
        reviewed_by="alice",
        review_source="audit",
    )

    with pytest.raises(PermissionError, match="ineligible for training"):
        PairedLatentTextDataset(
            dataset_dir=ds_dir,
            governance_version="rights_v001",
            caption_review_version="rev_only_one",
            governance_mode=GovernanceMode.PRODUCTION_STRICT,
        )


def test_caption_review_cli_execution(tmp_path: Path):
    """Verify that CLI review-captions review, list-versions, and show work end-to-end."""
    import subprocess
    import sys

    setup_fixture_paired_dataset_with_captions(tmp_path, "ds_cli_rev", count=2)

    # 1. CLI review command
    cmd_rev = [
        sys.executable,
        "-m",
        "rernggen.data",
        "review-captions",
        "review",
        "--dataset-id",
        "ds_cli_rev",
        "--review-version",
        "caption_review_v001",
        "--all",
        "--status",
        "APPROVED",
        "--reviewer",
        "reviewer_bob",
        "--source",
        "cli_formal_review",
        "--dataset-root",
        str(tmp_path),
    ]
    res_rev = subprocess.run(cmd_rev, capture_output=True, text=True, check=True)
    assert "CAPTION REVIEW RECORDED" in res_rev.stdout

    # 2. CLI list-versions command
    cmd_list = [
        sys.executable,
        "-m",
        "rernggen.data",
        "review-captions",
        "list-versions",
        "--dataset-id",
        "ds_cli_rev",
        "--dataset-root",
        str(tmp_path),
    ]
    res_list = subprocess.run(cmd_list, capture_output=True, text=True, check=True)
    assert "caption_review_v001" in res_list.stdout

    # 3. CLI show command
    cmd_show = [
        sys.executable,
        "-m",
        "rernggen.data",
        "review-captions",
        "show",
        "--dataset-id",
        "ds_cli_rev",
        "--review-version",
        "caption_review_v001",
        "--dataset-root",
        str(tmp_path),
    ]
    res_show = subprocess.run(cmd_show, capture_output=True, text=True, check=True)
    assert "IMG-000001" in res_show.stdout
    assert "APPROVED" in res_show.stdout
    assert "reviewer_bob" in res_show.stdout

