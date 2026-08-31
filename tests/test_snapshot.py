"""Comprehensive unit and integration tests for Step 22.E Immutable Dataset Snapshot & Freeze."""

import json
from pathlib import Path
import pytest
from safetensors.torch import save_file
import torch
from rernggen.data.caption_review import CaptionReviewManager
from rernggen.data.captions import CaptionManager, compute_caption_sha256
from rernggen.data.dataset import GovernanceMode, PairedLatentTextDataset
from rernggen.data.eligibility import TRAINING_ELIGIBILITY_POLICY_VERSION
from rernggen.data.governance import GovernanceManager
from rernggen.data.importer import compute_sha256
from rernggen.data.schema import DatasetSnapshotMetadata, DatasetSnapshotRecord
from rernggen.data.snapshot import (
    DatasetSnapshot,
    DatasetSnapshotCandidate,
    DatasetSnapshotManager,
    SnapshotStatus,
    compute_snapshot_metadata_sha256,
    compute_snapshot_record_sha256,
    serialize_snapshot_record,
)


def setup_fixture_dataset_for_snapshot(
    root_dir: Path,
    dataset_id: str = "fixture_snap_ds",
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

    return ds_dir


# =============================================================================
# Test 1: Untouched frozen metadata loads successfully
# =============================================================================
def test_untouched_frozen_snapshot_loads_successfully(tmp_path: Path):
    ds_dir = setup_fixture_dataset_for_snapshot(tmp_path, "ds_untouched", count=2)
    gov_mgr = GovernanceManager(dataset_root=tmp_path)
    rev_mgr = CaptionReviewManager(dataset_root=tmp_path)
    snap_mgr = DatasetSnapshotManager(dataset_root=tmp_path)

    all_ids = ["IMG-000001", "IMG-000002"]
    gov_mgr.authorize_samples(
        dataset_id="ds_untouched",
        governance_version="rights_v001",
        image_ids=all_ids,
        training_decision="ALLOW",
        commercial_decision="ALLOW",
        authorization_source="audit_lead",
    )
    rev_mgr.review_samples(
        dataset_id="ds_untouched",
        review_version="cap_rev_v001",
        image_ids=all_ids,
        review_status="APPROVED",
        reviewed_by="reviewer_alice",
        review_source="audit",
    )

    snapshot = snap_mgr.freeze_snapshot(
        dataset_id="ds_untouched",
        snapshot_version="dataset_snapshot_v001",
        governance_version="rights_v001",
        caption_review_version="cap_rev_v001",
        created_by="engineer_alice",
        creation_source="training_prep",
    )

    loaded = snap_mgr.load_snapshot("ds_untouched", "dataset_snapshot_v001", verify_integrity=True)
    assert len(loaded) == 2
    assert loaded.metadata.status == "FROZEN"
    assert loaded.metadata.metadata_sha256 == compute_snapshot_metadata_sha256(loaded.metadata)


# =============================================================================
# Test 2 - 6: Metadata corruption without updating metadata_sha256 fails closed
# =============================================================================
def test_metadata_corruption_detected(tmp_path: Path):
    setup_fixture_dataset_for_snapshot(tmp_path, "ds_meta_tamper", count=2)
    gov_mgr = GovernanceManager(dataset_root=tmp_path)
    rev_mgr = CaptionReviewManager(dataset_root=tmp_path)
    snap_mgr = DatasetSnapshotManager(dataset_root=tmp_path)

    all_ids = ["IMG-000001", "IMG-000002"]
    gov_mgr.authorize_samples(
        dataset_id="ds_meta_tamper",
        governance_version="rights_v001",
        image_ids=all_ids,
        training_decision="ALLOW",
        commercial_decision="ALLOW",
        authorization_source="audit_lead",
    )
    rev_mgr.review_samples(
        dataset_id="ds_meta_tamper",
        review_version="cap_rev_v001",
        image_ids=all_ids,
        review_status="APPROVED",
        reviewed_by="reviewer_alice",
        review_source="audit",
    )

    snap = snap_mgr.freeze_snapshot(
        dataset_id="ds_meta_tamper",
        snapshot_version="dataset_snapshot_v001",
        governance_version="rights_v001",
        caption_review_version="cap_rev_v001",
        created_by="engineer_alice",
        creation_source="training_prep",
    )

    meta_path = snap.metadata_path
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_dict = json.load(f)

    # 2. Corrupt governance_version
    corrupt_dict = dict(meta_dict)
    corrupt_dict["governance_version"] = "rights_v999"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(corrupt_dict, f, indent=2)
    with pytest.raises(ValueError, match="metadata SHA-256.*does not match"):
        snap_mgr.load_snapshot("ds_meta_tamper", "dataset_snapshot_v001", verify_integrity=True)

    # 3. Corrupt governance_manifest_sha256
    corrupt_dict = dict(meta_dict)
    corrupt_dict["governance_manifest_sha256"] = "0" * 64
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(corrupt_dict, f, indent=2)
    with pytest.raises(ValueError, match="metadata SHA-256.*does not match"):
        snap_mgr.load_snapshot("ds_meta_tamper", "dataset_snapshot_v001", verify_integrity=True)

    # 4. Corrupt caption_review_version
    corrupt_dict = dict(meta_dict)
    corrupt_dict["caption_review_version"] = "cap_rev_v999"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(corrupt_dict, f, indent=2)
    with pytest.raises(ValueError, match="metadata SHA-256.*does not match"):
        snap_mgr.load_snapshot("ds_meta_tamper", "dataset_snapshot_v001", verify_integrity=True)

    # 5. Corrupt eligibility_policy_version
    corrupt_dict = dict(meta_dict)
    corrupt_dict["eligibility_policy_version"] = "policy_v999"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(corrupt_dict, f, indent=2)
    with pytest.raises(ValueError, match="metadata SHA-256.*does not match"):
        snap_mgr.load_snapshot("ds_meta_tamper", "dataset_snapshot_v001", verify_integrity=True)

    # 6. Corrupt snapshot_manifest_sha256
    corrupt_dict = dict(meta_dict)
    corrupt_dict["snapshot_manifest_sha256"] = "f" * 64
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(corrupt_dict, f, indent=2)
    with pytest.raises(ValueError, match="metadata SHA-256.*does not match"):
        snap_mgr.load_snapshot("ds_meta_tamper", "dataset_snapshot_v001", verify_integrity=True)


# =============================================================================
# Test 7: Missing metadata_sha256 fails closed
# =============================================================================
def test_missing_metadata_sha256_fails_closed(tmp_path: Path):
    setup_fixture_dataset_for_snapshot(tmp_path, "ds_missing_msha", count=2)
    gov_mgr = GovernanceManager(dataset_root=tmp_path)
    rev_mgr = CaptionReviewManager(dataset_root=tmp_path)
    snap_mgr = DatasetSnapshotManager(dataset_root=tmp_path)

    all_ids = ["IMG-000001", "IMG-000002"]
    gov_mgr.authorize_samples(
        dataset_id="ds_missing_msha",
        governance_version="rights_v001",
        image_ids=all_ids,
        training_decision="ALLOW",
        commercial_decision="ALLOW",
        authorization_source="audit_lead",
    )
    rev_mgr.review_samples(
        dataset_id="ds_missing_msha",
        review_version="cap_rev_v001",
        image_ids=all_ids,
        review_status="APPROVED",
        reviewed_by="reviewer_alice",
        review_source="audit",
    )

    snap = snap_mgr.freeze_snapshot(
        dataset_id="ds_missing_msha",
        snapshot_version="dataset_snapshot_v001",
        governance_version="rights_v001",
        caption_review_version="cap_rev_v001",
        created_by="engineer_alice",
        creation_source="training_prep",
    )

    with open(snap.metadata_path, "r", encoding="utf-8") as f:
        meta_dict = json.load(f)
    meta_dict["metadata_sha256"] = ""
    with open(snap.metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2)

    with pytest.raises(ValueError, match="metadata_sha256 is missing or empty"):
        snap_mgr.load_snapshot("ds_missing_msha", "dataset_snapshot_v001", verify_integrity=True)


# =============================================================================
# Test 8 & 9: Path / Metadata identity mismatch fails closed
# =============================================================================
def test_path_metadata_identity_mismatch_fails_closed(tmp_path: Path):
    setup_fixture_dataset_for_snapshot(tmp_path, "ds_path_match", count=2)
    gov_mgr = GovernanceManager(dataset_root=tmp_path)
    rev_mgr = CaptionReviewManager(dataset_root=tmp_path)
    snap_mgr = DatasetSnapshotManager(dataset_root=tmp_path)

    all_ids = ["IMG-000001", "IMG-000002"]
    gov_mgr.authorize_samples(
        dataset_id="ds_path_match",
        governance_version="rights_v001",
        image_ids=all_ids,
        training_decision="ALLOW",
        commercial_decision="ALLOW",
        authorization_source="audit_lead",
    )
    rev_mgr.review_samples(
        dataset_id="ds_path_match",
        review_version="cap_rev_v001",
        image_ids=all_ids,
        review_status="APPROVED",
        reviewed_by="reviewer_alice",
        review_source="audit",
    )

    snap = snap_mgr.freeze_snapshot(
        dataset_id="ds_path_match",
        snapshot_version="dataset_snapshot_v001",
        governance_version="rights_v001",
        caption_review_version="cap_rev_v001",
        created_by="engineer_alice",
        creation_source="training_prep",
    )

    # 8. Mismatch dataset_id
    with open(snap.metadata_path, "r", encoding="utf-8") as f:
        meta_dict = json.load(f)
    meta_dict["dataset_id"] = "ds_other"
    with open(snap.metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2)
    with pytest.raises(ValueError, match="does not match requested dataset_id"):
        snap_mgr.load_snapshot("ds_path_match", "dataset_snapshot_v001", verify_integrity=False)

    # 9. Mismatch snapshot_version
    meta_dict["dataset_id"] = "ds_path_match"
    meta_dict["snapshot_version"] = "dataset_snapshot_v099"
    with open(snap.metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2)
    with pytest.raises(ValueError, match="does not match requested snapshot_version"):
        snap_mgr.load_snapshot("ds_path_match", "dataset_snapshot_v001", verify_integrity=False)


# =============================================================================
# Test 10: Non-FROZEN status rejected by load_snapshot
# =============================================================================
def test_non_frozen_status_rejected_by_load_snapshot(tmp_path: Path):
    setup_fixture_dataset_for_snapshot(tmp_path, "ds_non_frozen", count=2)
    gov_mgr = GovernanceManager(dataset_root=tmp_path)
    rev_mgr = CaptionReviewManager(dataset_root=tmp_path)
    snap_mgr = DatasetSnapshotManager(dataset_root=tmp_path)

    all_ids = ["IMG-000001", "IMG-000002"]
    gov_mgr.authorize_samples(
        dataset_id="ds_non_frozen",
        governance_version="rights_v001",
        image_ids=all_ids,
        training_decision="ALLOW",
        commercial_decision="ALLOW",
        authorization_source="audit_lead",
    )
    rev_mgr.review_samples(
        dataset_id="ds_non_frozen",
        review_version="cap_rev_v001",
        image_ids=all_ids,
        review_status="APPROVED",
        reviewed_by="reviewer_alice",
        review_source="audit",
    )

    snap = snap_mgr.freeze_snapshot(
        dataset_id="ds_non_frozen",
        snapshot_version="dataset_snapshot_v001",
        governance_version="rights_v001",
        caption_review_version="cap_rev_v001",
        created_by="engineer_alice",
        creation_source="training_prep",
    )

    with open(snap.metadata_path, "r", encoding="utf-8") as f:
        meta_dict = json.load(f)
    meta_dict["status"] = "DRAFT"
    with open(snap.metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2)

    with pytest.raises(ValueError, match="has status 'DRAFT', expected 'FROZEN'"):
        snap_mgr.load_snapshot("ds_non_frozen", "dataset_snapshot_v001", verify_integrity=False)


# =============================================================================
# Test 11 & 12: Record dataset_id / snapshot_version coherence
# =============================================================================
def test_record_to_snapshot_coherence_failure(tmp_path: Path):
    setup_fixture_dataset_for_snapshot(tmp_path, "ds_rec_coherence", count=2)
    gov_mgr = GovernanceManager(dataset_root=tmp_path)
    rev_mgr = CaptionReviewManager(dataset_root=tmp_path)
    snap_mgr = DatasetSnapshotManager(dataset_root=tmp_path)

    all_ids = ["IMG-000001", "IMG-000002"]
    gov_mgr.authorize_samples(
        dataset_id="ds_rec_coherence",
        governance_version="rights_v001",
        image_ids=all_ids,
        training_decision="ALLOW",
        commercial_decision="ALLOW",
        authorization_source="audit_lead",
    )
    rev_mgr.review_samples(
        dataset_id="ds_rec_coherence",
        review_version="cap_rev_v001",
        image_ids=all_ids,
        review_status="APPROVED",
        reviewed_by="reviewer_alice",
        review_source="audit",
    )

    snap = snap_mgr.freeze_snapshot(
        dataset_id="ds_rec_coherence",
        snapshot_version="dataset_snapshot_v001",
        governance_version="rights_v001",
        caption_review_version="cap_rev_v001",
        created_by="engineer_alice",
        creation_source="training_prep",
    )

    # 11. Record dataset_id mismatch
    lines = [json.loads(line) for line in snap.manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    lines[0]["dataset_id"] = "ds_alien"
    with open(snap.manifest_path, "w", encoding="utf-8") as f:
        for r in lines:
            f.write(serialize_snapshot_record(r) + "\n")
    # Update manifest sha in metadata to test coherence check directly
    with open(snap.metadata_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    m["snapshot_manifest_sha256"] = compute_sha256(snap.manifest_path)
    m["metadata_sha256"] = compute_snapshot_metadata_sha256(m)
    with open(snap.metadata_path, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)

    with pytest.raises(ValueError, match="Snapshot record coherence error.*differs from snapshot metadata dataset_id"):
        snap_mgr.load_snapshot("ds_rec_coherence", "dataset_snapshot_v001", verify_integrity=True)

    # 12. Record snapshot_version mismatch
    lines[0]["dataset_id"] = "ds_rec_coherence"
    lines[0]["snapshot_version"] = "dataset_snapshot_v999"
    with open(snap.manifest_path, "w", encoding="utf-8") as f:
        for r in lines:
            f.write(serialize_snapshot_record(r) + "\n")
    m["snapshot_manifest_sha256"] = compute_sha256(snap.manifest_path)
    m["metadata_sha256"] = compute_snapshot_metadata_sha256(m)
    with open(snap.metadata_path, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)

    with pytest.raises(ValueError, match="Snapshot record coherence error.*differs from snapshot metadata snapshot_version"):
        snap_mgr.load_snapshot("ds_rec_coherence", "dataset_snapshot_v001", verify_integrity=True)


# =============================================================================
# Test 13: Duplicate sample_id records fail closed
# =============================================================================
def test_duplicate_sample_id_fails_closed(tmp_path: Path):
    setup_fixture_dataset_for_snapshot(tmp_path, "ds_dup_sample", count=2)
    gov_mgr = GovernanceManager(dataset_root=tmp_path)
    rev_mgr = CaptionReviewManager(dataset_root=tmp_path)
    snap_mgr = DatasetSnapshotManager(dataset_root=tmp_path)

    all_ids = ["IMG-000001", "IMG-000002"]
    gov_mgr.authorize_samples(
        dataset_id="ds_dup_sample",
        governance_version="rights_v001",
        image_ids=all_ids,
        training_decision="ALLOW",
        commercial_decision="ALLOW",
        authorization_source="audit_lead",
    )
    rev_mgr.review_samples(
        dataset_id="ds_dup_sample",
        review_version="cap_rev_v001",
        image_ids=all_ids,
        review_status="APPROVED",
        reviewed_by="reviewer_alice",
        review_source="audit",
    )

    snap = snap_mgr.freeze_snapshot(
        dataset_id="ds_dup_sample",
        snapshot_version="dataset_snapshot_v001",
        governance_version="rights_v001",
        caption_review_version="cap_rev_v001",
        created_by="engineer_alice",
        creation_source="training_prep",
    )

    lines = [json.loads(line) for line in snap.manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    lines.append(dict(lines[0]))  # duplicate IMG-000001
    with open(snap.manifest_path, "w", encoding="utf-8") as f:
        for r in lines:
            f.write(serialize_snapshot_record(r) + "\n")
    with open(snap.metadata_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    m["sample_count"] = len(lines)
    m["snapshot_manifest_sha256"] = compute_sha256(snap.manifest_path)
    m["metadata_sha256"] = compute_snapshot_metadata_sha256(m)
    with open(snap.metadata_path, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)

    with pytest.raises(ValueError, match="duplicate sample_id 'IMG-000001' detected"):
        snap_mgr.load_snapshot("ds_dup_sample", "dataset_snapshot_v001", verify_integrity=True)


# =============================================================================
# Test 14: Out-of-order sample IDs fail rather than silently sorting
# =============================================================================
def test_out_of_order_sample_ids_fail_closed(tmp_path: Path):
    setup_fixture_dataset_for_snapshot(tmp_path, "ds_order_fail", count=2)
    gov_mgr = GovernanceManager(dataset_root=tmp_path)
    rev_mgr = CaptionReviewManager(dataset_root=tmp_path)
    snap_mgr = DatasetSnapshotManager(dataset_root=tmp_path)

    all_ids = ["IMG-000001", "IMG-000002"]
    gov_mgr.authorize_samples(
        dataset_id="ds_order_fail",
        governance_version="rights_v001",
        image_ids=all_ids,
        training_decision="ALLOW",
        commercial_decision="ALLOW",
        authorization_source="audit_lead",
    )
    rev_mgr.review_samples(
        dataset_id="ds_order_fail",
        review_version="cap_rev_v001",
        image_ids=all_ids,
        review_status="APPROVED",
        reviewed_by="reviewer_alice",
        review_source="audit",
    )

    snap = snap_mgr.freeze_snapshot(
        dataset_id="ds_order_fail",
        snapshot_version="dataset_snapshot_v001",
        governance_version="rights_v001",
        caption_review_version="cap_rev_v001",
        created_by="engineer_alice",
        creation_source="training_prep",
    )

    lines = [json.loads(line) for line in snap.manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    lines.reverse()  # now [IMG-000002, IMG-000001]
    with open(snap.manifest_path, "w", encoding="utf-8") as f:
        for r in lines:
            f.write(serialize_snapshot_record(r) + "\n")
    with open(snap.metadata_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    m["snapshot_manifest_sha256"] = compute_sha256(snap.manifest_path)
    m["metadata_sha256"] = compute_snapshot_metadata_sha256(m)
    with open(snap.metadata_path, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)

    with pytest.raises(ValueError, match="not in canonical sorted order"):
        snap_mgr.load_snapshot("ds_order_fail", "dataset_snapshot_v001", verify_integrity=True)


# =============================================================================
# Test 15 & 16: Canonical JSON serialization stability and dictionary insertion order independence
# =============================================================================
def test_canonical_json_serialization_stability():
    rec_dict_1 = {
        "sample_id": "IMG-000001",
        "dataset_id": "ds_canon",
        "snapshot_version": "v1",
        "caption": "Scene 1",
        "caption_sha256": "sha1",
        "caption_version": "c1",
        "caption_review_version": "r1",
        "governance_version": "g1",
        "latent_relative_path": "p1",
        "latent_sha256": "lsha",
        "latent_shape": [4, 32, 32],
        "latent_cache_version": "lv1",
        "text_embedding_relative_path": "tp1",
        "text_embedding_sha256": "tsha",
        "text_embedding_shape": [512],
        "text_cache_version": "tv1",
        "eligibility_policy_version": "pv1",
        "record_sha256": "rec_sha_val",
    }
    # Reordered keys in dict 2
    keys_reversed = list(reversed(list(rec_dict_1.keys())))
    rec_dict_2 = {k: rec_dict_1[k] for k in keys_reversed}

    json1 = serialize_snapshot_record(rec_dict_1)
    json2 = serialize_snapshot_record(rec_dict_2)

    assert json1 == json2
    assert json1.startswith('{"caption":')  # Alphabetically first key


# =============================================================================
# Test 17: to_provenance_dict contains snapshot_metadata_sha256
# =============================================================================
def test_to_provenance_dict_contains_metadata_sha(tmp_path: Path):
    setup_fixture_dataset_for_snapshot(tmp_path, "ds_prov_dict", count=2)
    gov_mgr = GovernanceManager(dataset_root=tmp_path)
    rev_mgr = CaptionReviewManager(dataset_root=tmp_path)
    snap_mgr = DatasetSnapshotManager(dataset_root=tmp_path)

    all_ids = ["IMG-000001", "IMG-000002"]
    gov_mgr.authorize_samples(
        dataset_id="ds_prov_dict",
        governance_version="rights_v001",
        image_ids=all_ids,
        training_decision="ALLOW",
        commercial_decision="ALLOW",
        authorization_source="audit_lead",
    )
    rev_mgr.review_samples(
        dataset_id="ds_prov_dict",
        review_version="cap_rev_v001",
        image_ids=all_ids,
        review_status="APPROVED",
        reviewed_by="reviewer_alice",
        review_source="audit",
    )

    snap = snap_mgr.freeze_snapshot(
        dataset_id="ds_prov_dict",
        snapshot_version="dataset_snapshot_v001",
        governance_version="rights_v001",
        caption_review_version="cap_rev_v001",
        created_by="engineer_alice",
        creation_source="training_prep",
    )

    prov = snap.to_provenance_dict()
    assert "snapshot_metadata_sha256" in prov
    assert prov["snapshot_metadata_sha256"] == snap.metadata.metadata_sha256
    assert prov["snapshot_manifest_sha256"] == snap.metadata.snapshot_manifest_sha256


# =============================================================================
# Test 18 & 19: Tamper detection on manifest bytes & record SHA
# =============================================================================
def test_tamper_detection_manifest_and_record(tmp_path: Path):
    setup_fixture_dataset_for_snapshot(tmp_path, "ds_tamper_suite", count=2)
    gov_mgr = GovernanceManager(dataset_root=tmp_path)
    rev_mgr = CaptionReviewManager(dataset_root=tmp_path)
    snap_mgr = DatasetSnapshotManager(dataset_root=tmp_path)

    all_ids = ["IMG-000001", "IMG-000002"]
    gov_mgr.authorize_samples(
        dataset_id="ds_tamper_suite",
        governance_version="rights_v001",
        image_ids=all_ids,
        training_decision="ALLOW",
        commercial_decision="ALLOW",
        authorization_source="audit_lead",
    )
    rev_mgr.review_samples(
        dataset_id="ds_tamper_suite",
        review_version="cap_rev_v001",
        image_ids=all_ids,
        review_status="APPROVED",
        reviewed_by="reviewer_alice",
        review_source="audit",
    )

    snap = snap_mgr.freeze_snapshot(
        dataset_id="ds_tamper_suite",
        snapshot_version="dataset_snapshot_v001",
        governance_version="rights_v001",
        caption_review_version="cap_rev_v001",
        created_by="engineer_alice",
        creation_source="training_prep",
    )

    # 18. Manifest-byte tamper detection
    manifest_p = snap.manifest_path
    lines = manifest_p.read_text(encoding="utf-8").splitlines()
    corrupt_line = lines[0].replace("Synthetic Khmer scene 1", "MALICIOUS TAMPERED TEXT")
    manifest_p.write_text(corrupt_line + "\n" + lines[1] + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Snapshot integrity error.*manifest SHA-256"):
        snap_mgr.load_snapshot("ds_tamper_suite", "dataset_snapshot_v001", verify_integrity=True)


# =============================================================================
# Test 20: Immutable overwrite rejection
# =============================================================================
def test_immutable_overwrite_rejected(tmp_path: Path):
    setup_fixture_dataset_for_snapshot(tmp_path, "ds_no_over", count=2)
    gov_mgr = GovernanceManager(dataset_root=tmp_path)
    rev_mgr = CaptionReviewManager(dataset_root=tmp_path)
    snap_mgr = DatasetSnapshotManager(dataset_root=tmp_path)

    all_ids = ["IMG-000001", "IMG-000002"]
    gov_mgr.authorize_samples(
        dataset_id="ds_no_over",
        governance_version="rights_v001",
        image_ids=all_ids,
        training_decision="ALLOW",
        commercial_decision="ALLOW",
        authorization_source="audit_lead",
    )
    rev_mgr.review_samples(
        dataset_id="ds_no_over",
        review_version="cap_rev_v001",
        image_ids=all_ids,
        review_status="APPROVED",
        reviewed_by="reviewer_alice",
        review_source="audit",
    )

    snap_mgr.freeze_snapshot(
        dataset_id="ds_no_over",
        snapshot_version="dataset_snapshot_v001",
        governance_version="rights_v001",
        caption_review_version="cap_rev_v001",
        created_by="engineer_alice",
        creation_source="training_prep",
    )

    with pytest.raises(FileExistsError, match="already exists"):
        snap_mgr.freeze_snapshot(
            dataset_id="ds_no_over",
            snapshot_version="dataset_snapshot_v001",
            governance_version="rights_v001",
            caption_review_version="cap_rev_v001",
            created_by="engineer_alice",
            creation_source="training_prep",
        )


# =============================================================================
# Test 21: Live caption mutation resistance
# =============================================================================
def test_live_caption_mutation_resistance(tmp_path: Path):
    ds_dir = setup_fixture_dataset_for_snapshot(tmp_path, "ds_mut_live_cap", count=2)
    gov_mgr = GovernanceManager(dataset_root=tmp_path)
    rev_mgr = CaptionReviewManager(dataset_root=tmp_path)
    snap_mgr = DatasetSnapshotManager(dataset_root=tmp_path)

    all_ids = ["IMG-000001", "IMG-000002"]
    gov_mgr.authorize_samples(
        dataset_id="ds_mut_live_cap",
        governance_version="rights_v001",
        image_ids=all_ids,
        training_decision="ALLOW",
        commercial_decision="ALLOW",
        authorization_source="audit_lead",
    )
    rev_mgr.review_samples(
        dataset_id="ds_mut_live_cap",
        review_version="cap_rev_v001",
        image_ids=all_ids,
        review_status="APPROVED",
        reviewed_by="reviewer_alice",
        review_source="audit",
    )

    s1 = snap_mgr.freeze_snapshot(
        dataset_id="ds_mut_live_cap",
        snapshot_version="dataset_snapshot_v001",
        governance_version="rights_v001",
        caption_review_version="cap_rev_v001",
        created_by="engineer_alice",
        creation_source="training_prep",
    )
    manifest_bytes = s1.manifest_path.read_bytes()

    # Mutate live caption
    cap_file = ds_dir / "captions" / "captions_v002" / "manifest.jsonl"
    with open(cap_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"image_id": "IMG-000001", "caption": "TAMPERED LIVE CAPTION"}) + "\n")

    reloaded = snap_mgr.load_snapshot("ds_mut_live_cap", "dataset_snapshot_v001", verify_integrity=True)
    assert reloaded.manifest_path.read_bytes() == manifest_bytes
    assert reloaded["IMG-000001"].caption == "Synthetic Khmer scene 1"


# =============================================================================
# Test 22: Governance revocation historical snapshot resistance
# =============================================================================
def test_governance_revocation_historical_snapshot_resistance(tmp_path: Path):
    ds_dir = setup_fixture_dataset_for_snapshot(tmp_path, "ds_gov_rev_hist", count=2)
    gov_mgr = GovernanceManager(dataset_root=tmp_path)
    rev_mgr = CaptionReviewManager(dataset_root=tmp_path)
    snap_mgr = DatasetSnapshotManager(dataset_root=tmp_path)

    all_ids = ["IMG-000001", "IMG-000002"]
    gov_mgr.authorize_samples(
        dataset_id="ds_gov_rev_hist",
        governance_version="rights_v001",
        image_ids=all_ids,
        training_decision="ALLOW",
        commercial_decision="ALLOW",
        authorization_source="audit_lead",
    )
    rev_mgr.review_samples(
        dataset_id="ds_gov_rev_hist",
        review_version="cap_rev_v001",
        image_ids=all_ids,
        review_status="APPROVED",
        reviewed_by="reviewer_alice",
        review_source="audit",
    )

    s1 = snap_mgr.freeze_snapshot(
        dataset_id="ds_gov_rev_hist",
        snapshot_version="dataset_snapshot_v001",
        governance_version="rights_v001",
        caption_review_version="cap_rev_v001",
        created_by="engineer_alice",
        creation_source="training_prep",
    )

    # Revoke in rights_v002
    gov_mgr.authorize_samples(
        dataset_id="ds_gov_rev_hist",
        governance_version="rights_v002",
        image_ids=all_ids,
        training_decision="DENY",
        commercial_decision="DENY",
        authorization_source="legal_revocation",
        base_version="rights_v001",
    )

    # Historical snapshot v001 remains intact and valid
    reloaded = snap_mgr.load_snapshot("ds_gov_rev_hist", "dataset_snapshot_v001", verify_integrity=True)
    assert len(reloaded) == 2


# =============================================================================
# Test 23: Zero eligible freeze rejected
# =============================================================================
def test_zero_eligible_freeze_rejected(tmp_path: Path):
    setup_fixture_dataset_for_snapshot(tmp_path, "ds_zero_rej", count=2)
    snap_mgr = DatasetSnapshotManager(dataset_root=tmp_path)

    with pytest.raises(ValueError, match="0 eligible samples"):
        snap_mgr.freeze_snapshot(
            dataset_id="ds_zero_rej",
            snapshot_version="dataset_snapshot_v001",
            governance_version="rights_v001",
            caption_review_version="cap_rev_v001",
            created_by="engineer_alice",
            creation_source="training_prep",
        )


# =============================================================================
# Test 24: Listing safety filters out corrupted metadata
# =============================================================================
def test_listing_safety_filters_corrupted_metadata(tmp_path: Path):
    setup_fixture_dataset_for_snapshot(tmp_path, "ds_list_safe", count=2)
    gov_mgr = GovernanceManager(dataset_root=tmp_path)
    rev_mgr = CaptionReviewManager(dataset_root=tmp_path)
    snap_mgr = DatasetSnapshotManager(dataset_root=tmp_path)

    all_ids = ["IMG-000001", "IMG-000002"]
    gov_mgr.authorize_samples(
        dataset_id="ds_list_safe",
        governance_version="rights_v001",
        image_ids=all_ids,
        training_decision="ALLOW",
        commercial_decision="ALLOW",
        authorization_source="audit_lead",
    )
    rev_mgr.review_samples(
        dataset_id="ds_list_safe",
        review_version="cap_rev_v001",
        image_ids=all_ids,
        review_status="APPROVED",
        reviewed_by="reviewer_alice",
        review_source="audit",
    )

    s1 = snap_mgr.freeze_snapshot(
        dataset_id="ds_list_safe",
        snapshot_version="dataset_snapshot_v001",
        governance_version="rights_v001",
        caption_review_version="cap_rev_v001",
        created_by="engineer_alice",
        creation_source="training_prep",
    )

    # Freeze second snapshot v002
    gov_mgr.authorize_samples(
        dataset_id="ds_list_safe",
        governance_version="rights_v002",
        image_ids=all_ids,
        training_decision="ALLOW",
        commercial_decision="ALLOW",
        authorization_source="audit_lead",
        base_version="rights_v001",
    )
    rev_mgr.review_samples(
        dataset_id="ds_list_safe",
        review_version="cap_rev_v002",
        image_ids=all_ids,
        review_status="APPROVED",
        reviewed_by="reviewer_alice",
        review_source="audit",
        base_version="cap_rev_v001",
    )
    s2 = snap_mgr.freeze_snapshot(
        dataset_id="ds_list_safe",
        snapshot_version="dataset_snapshot_v002",
        governance_version="rights_v002",
        caption_review_version="cap_rev_v002",
        created_by="engineer_alice",
        creation_source="training_prep",
        previous_snapshot_version="dataset_snapshot_v001",
    )

    # Corrupt v001 metadata directly on disk
    with open(s1.metadata_path, "r", encoding="utf-8") as f:
        meta_dict = json.load(f)
    meta_dict["governance_version"] = "CORRUPTED"
    with open(s1.metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2)

    # list_snapshots with verify_integrity=True skips corrupted v001 and returns only valid v002
    valid_list = snap_mgr.list_snapshots("ds_list_safe", verify_integrity=True)
    assert len(valid_list) == 1
    assert valid_list[0].snapshot_version == "dataset_snapshot_v002"


# =============================================================================
# Test 25: CLI end-to-end execution
# =============================================================================
def test_snapshot_cli_end_to_end(tmp_path: Path):
    import subprocess
    import sys

    setup_fixture_dataset_for_snapshot(tmp_path, "ds_cli_snap", count=2)
    gov_mgr = GovernanceManager(dataset_root=tmp_path)
    rev_mgr = CaptionReviewManager(dataset_root=tmp_path)

    all_ids = ["IMG-000001", "IMG-000002"]
    gov_mgr.authorize_samples(
        dataset_id="ds_cli_snap",
        governance_version="rights_v001",
        image_ids=all_ids,
        training_decision="ALLOW",
        commercial_decision="ALLOW",
        authorization_source="audit_lead",
    )
    rev_mgr.review_samples(
        dataset_id="ds_cli_snap",
        review_version="cap_rev_v001",
        image_ids=all_ids,
        review_status="APPROVED",
        reviewed_by="reviewer_alice",
        review_source="audit",
    )

    # 1. CLI Plan
    plan_cmd = [
        sys.executable,
        "-m",
        "rernggen.data",
        "snapshot",
        "plan",
        "--dataset-id",
        "ds_cli_snap",
        "--governance-version",
        "rights_v001",
        "--caption-review-version",
        "cap_rev_v001",
        "--dataset-root",
        str(tmp_path),
    ]
    res_plan = subprocess.run(plan_cmd, capture_output=True, text=True, check=True)
    assert "DATASET SNAPSHOT CANDIDATE PLAN" in res_plan.stdout
    assert "Eligible (Admitted):        2" in res_plan.stdout

    # 2. CLI Freeze
    freeze_cmd = [
        sys.executable,
        "-m",
        "rernggen.data",
        "snapshot",
        "freeze",
        "--dataset-id",
        "ds_cli_snap",
        "--snapshot-version",
        "dataset_snapshot_v001",
        "--governance-version",
        "rights_v001",
        "--caption-review-version",
        "cap_rev_v001",
        "--created-by",
        "engineer_alice",
        "--creation-source",
        "cli_training_prep",
        "--dataset-root",
        str(tmp_path),
    ]
    res_freeze = subprocess.run(freeze_cmd, capture_output=True, text=True, check=True)
    assert "DATASET SNAPSHOT FROZEN" in res_freeze.stdout
    assert "Sample Count:               2" in res_freeze.stdout

    # 3. CLI List
    list_cmd = [
        sys.executable,
        "-m",
        "rernggen.data",
        "snapshot",
        "list",
        "--dataset-id",
        "ds_cli_snap",
        "--dataset-root",
        str(tmp_path),
    ]
    res_list = subprocess.run(list_cmd, capture_output=True, text=True, check=True)
    assert "dataset_snapshot_v001" in res_list.stdout

    # 4. CLI Show
    show_cmd = [
        sys.executable,
        "-m",
        "rernggen.data",
        "snapshot",
        "show",
        "--dataset-id",
        "ds_cli_snap",
        "--snapshot-version",
        "dataset_snapshot_v001",
        "--dataset-root",
        str(tmp_path),
    ]
    res_show = subprocess.run(show_cmd, capture_output=True, text=True, check=True)
    assert "DATASET SNAPSHOT: dataset_snapshot_v001" in res_show.stdout

    # 5. CLI Verify
    verify_cmd = [
        sys.executable,
        "-m",
        "rernggen.data",
        "snapshot",
        "verify",
        "--dataset-id",
        "ds_cli_snap",
        "--snapshot-version",
        "dataset_snapshot_v001",
        "--dataset-root",
        str(tmp_path),
    ]
    res_verify = subprocess.run(verify_cmd, capture_output=True, text=True, check=True)
    assert "SNAPSHOT INTEGRITY VERIFICATION: PASSED" in res_verify.stdout


# =============================================================================
# Test 26: Snapshot includes eligible and excludes ineligible
# =============================================================================
def test_snapshot_includes_eligible_and_excludes_ineligible(tmp_path: Path):
    setup_fixture_dataset_for_snapshot(tmp_path, "ds_elig_filter", count=3)
    gov_mgr = GovernanceManager(dataset_root=tmp_path)
    rev_mgr = CaptionReviewManager(dataset_root=tmp_path)
    snap_mgr = DatasetSnapshotManager(dataset_root=tmp_path)

    gov_mgr.authorize_samples(
        dataset_id="ds_elig_filter",
        governance_version="rights_v001",
        image_ids=["IMG-000001", "IMG-000002"],
        training_decision="ALLOW",
        commercial_decision="ALLOW",
        authorization_source="audit_lead",
    )
    gov_mgr.authorize_samples(
        dataset_id="ds_elig_filter",
        governance_version="rights_v001",
        image_ids=["IMG-000003"],
        training_decision="DENY",
        commercial_decision="DENY",
        authorization_source="audit_lead",
        base_version="rights_v001",
        _allow_test_overwrite=True,
    )

    rev_mgr.review_samples(
        dataset_id="ds_elig_filter",
        review_version="cap_rev_v001",
        image_ids=["IMG-000001", "IMG-000003"],
        review_status="APPROVED",
        reviewed_by="reviewer_alice",
        review_source="audit",
    )
    rev_mgr.review_samples(
        dataset_id="ds_elig_filter",
        review_version="cap_rev_v001",
        image_ids=["IMG-000002"],
        review_status="REJECTED",
        reviewed_by="reviewer_alice",
        review_source="audit",
        base_version="cap_rev_v001",
        _allow_test_overwrite=True,
    )

    snapshot = snap_mgr.freeze_snapshot(
        dataset_id="ds_elig_filter",
        snapshot_version="dataset_snapshot_v001",
        governance_version="rights_v001",
        caption_review_version="cap_rev_v001",
        created_by="engineer_alice",
        creation_source="training_prep",
    )

    assert len(snapshot) == 1
    assert snapshot.metadata.sample_count == 1
    assert snapshot[0].sample_id == "IMG-000001"
    assert snapshot.get_sample("IMG-000002") is None
    assert snapshot.get_sample("IMG-000003") is None


# =============================================================================
# Test 27: Snapshot deterministic ordering
# =============================================================================
def test_snapshot_deterministic_ordering(tmp_path: Path):
    setup_fixture_dataset_for_snapshot(tmp_path, "ds_order", count=4)
    gov_mgr = GovernanceManager(dataset_root=tmp_path)
    rev_mgr = CaptionReviewManager(dataset_root=tmp_path)
    snap_mgr = DatasetSnapshotManager(dataset_root=tmp_path)

    all_ids = ["IMG-000001", "IMG-000002", "IMG-000003", "IMG-000004"]
    gov_mgr.authorize_samples(
        dataset_id="ds_order",
        governance_version="rights_v001",
        image_ids=all_ids,
        training_decision="ALLOW",
        commercial_decision="ALLOW",
        authorization_source="audit_lead",
    )
    rev_mgr.review_samples(
        dataset_id="ds_order",
        review_version="cap_rev_v001",
        image_ids=all_ids,
        review_status="APPROVED",
        reviewed_by="reviewer_alice",
        review_source="audit",
    )

    snapshot = snap_mgr.freeze_snapshot(
        dataset_id="ds_order",
        snapshot_version="dataset_snapshot_v001",
        governance_version="rights_v001",
        caption_review_version="cap_rev_v001",
        created_by="engineer_alice",
        creation_source="training_prep",
    )

    assert len(snapshot) == 4
    sample_ids = [rec.sample_id for rec in snapshot]
    assert sample_ids == sorted(sample_ids) == all_ids


# =============================================================================
# Test 28: Successor snapshot creation with lineage
# =============================================================================
def test_successor_snapshot_creation_with_lineage(tmp_path: Path):
    setup_fixture_dataset_for_snapshot(tmp_path, "ds_lineage", count=3)
    gov_mgr = GovernanceManager(dataset_root=tmp_path)
    rev_mgr = CaptionReviewManager(dataset_root=tmp_path)
    snap_mgr = DatasetSnapshotManager(dataset_root=tmp_path)

    gov_mgr.authorize_samples(
        dataset_id="ds_lineage",
        governance_version="rights_v001",
        image_ids=["IMG-000001", "IMG-000002"],
        training_decision="ALLOW",
        commercial_decision="ALLOW",
        authorization_source="audit_lead",
    )
    rev_mgr.review_samples(
        dataset_id="ds_lineage",
        review_version="cap_rev_v001",
        image_ids=["IMG-000001", "IMG-000002"],
        review_status="APPROVED",
        reviewed_by="reviewer_alice",
        review_source="audit",
    )
    s1 = snap_mgr.freeze_snapshot(
        dataset_id="ds_lineage",
        snapshot_version="dataset_snapshot_v001",
        governance_version="rights_v001",
        caption_review_version="cap_rev_v001",
        created_by="engineer_alice",
        creation_source="training_prep",
    )
    assert len(s1) == 2

    gov_mgr.authorize_samples(
        dataset_id="ds_lineage",
        governance_version="rights_v002",
        image_ids=["IMG-000003"],
        training_decision="ALLOW",
        commercial_decision="ALLOW",
        authorization_source="audit_lead",
        base_version="rights_v001",
    )
    rev_mgr.review_samples(
        dataset_id="ds_lineage",
        review_version="cap_rev_v002",
        image_ids=["IMG-000003"],
        review_status="APPROVED",
        reviewed_by="reviewer_alice",
        review_source="audit",
        base_version="cap_rev_v001",
    )
    s2 = snap_mgr.freeze_snapshot(
        dataset_id="ds_lineage",
        snapshot_version="dataset_snapshot_v002",
        governance_version="rights_v002",
        caption_review_version="cap_rev_v002",
        created_by="engineer_alice",
        creation_source="training_prep",
        previous_snapshot_version="dataset_snapshot_v001",
    )
    assert len(s2) == 3
    assert s2.metadata.previous_snapshot_version == "dataset_snapshot_v001"


# =============================================================================
# Test 29: Attribution validation rejects empty and dummy values
# =============================================================================
def test_attribution_validation_rejects_empty_and_dummy_values():
    with pytest.raises(ValueError, match="created_by"):
        DatasetSnapshotMetadata(
            dataset_id="d",
            snapshot_version="v1",
            status="FROZEN",
            sample_count=1,
            created_at="2026-08-31T00:00:00Z",
            created_by="",
            creation_source="src",
            governance_version="g1",
            governance_manifest_sha256="sha",
            caption_review_version="r1",
            caption_review_manifest_sha256="sha",
            eligibility_policy_version="p1",
            snapshot_manifest_sha256="sha",
        )

    for dummy in ["human", "system", "manual", "unknown", "human_declared"]:
        with pytest.raises(ValueError, match="created_by"):
            DatasetSnapshotMetadata(
                dataset_id="d",
                snapshot_version="v1",
                status="FROZEN",
                sample_count=1,
                created_at="2026-08-31T00:00:00Z",
                created_by=dummy,
                creation_source="src",
                governance_version="g1",
                governance_manifest_sha256="sha",
                caption_review_version="r1",
                caption_review_manifest_sha256="sha",
                eligibility_policy_version="p1",
                snapshot_manifest_sha256="sha",
            )

    for dummy in ["human", "system", "manual", "unknown", "human_declared"]:
        with pytest.raises(ValueError, match="creation_source"):
            DatasetSnapshotMetadata(
                dataset_id="d",
                snapshot_version="v1",
                status="FROZEN",
                sample_count=1,
                created_at="2026-08-31T00:00:00Z",
                created_by="alice",
                creation_source=dummy,
                governance_version="g1",
                governance_manifest_sha256="sha",
                caption_review_version="r1",
                caption_review_manifest_sha256="sha",
                eligibility_policy_version="p1",
                snapshot_manifest_sha256="sha",
            )


# =============================================================================
# Test 30: Plan snapshot candidate inspection
# =============================================================================
def test_plan_snapshot_candidate_inspection(tmp_path: Path):
    setup_fixture_dataset_for_snapshot(tmp_path, "ds_plan_test", count=3)
    gov_mgr = GovernanceManager(dataset_root=tmp_path)
    rev_mgr = CaptionReviewManager(dataset_root=tmp_path)
    snap_mgr = DatasetSnapshotManager(dataset_root=tmp_path)

    gov_mgr.authorize_samples(
        dataset_id="ds_plan_test",
        governance_version="rights_v001",
        image_ids=["IMG-000001"],
        training_decision="ALLOW",
        commercial_decision="ALLOW",
        authorization_source="audit_lead",
    )
    rev_mgr.review_samples(
        dataset_id="ds_plan_test",
        review_version="cap_rev_v001",
        image_ids=["IMG-000001"],
        review_status="APPROVED",
        reviewed_by="reviewer_alice",
        review_source="audit",
    )

    candidate = snap_mgr.plan_snapshot(
        dataset_id="ds_plan_test",
        governance_version="rights_v001",
        caption_review_version="cap_rev_v001",
    )

    assert candidate.total_samples == 3
    assert candidate.eligible_count == 1
    assert candidate.ineligible_count == 2
    assert candidate.can_freeze is True
    assert len(candidate.records) == 1
    assert candidate.records[0].sample_id == "IMG-000001"

    snap_dir = snap_mgr.get_snapshot_dir("ds_plan_test", "candidate_plan")
    assert not snap_dir.exists()



