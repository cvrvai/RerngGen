"""Comprehensive unit and integration tests for Step 22.D Unified Training Eligibility Gate."""

import json
from pathlib import Path
import pytest
from safetensors.torch import save_file
import torch
from rernggen.data.caption_review import CaptionReviewManager
from rernggen.data.captions import CaptionManager, compute_caption_sha256, normalize_caption_text
from rernggen.data.dataset import (
    GovernanceMode,
    PairedLatentTextDataset,
    create_paired_dataloader,
)
from rernggen.data.eligibility import (
    TRAINING_ELIGIBILITY_POLICY_VERSION,
    EligibilityReasonCode,
    TrainingEligibilityDecision,
    TrainingEligibilityEvaluator,
)
from rernggen.data.governance import GovernanceManager
from rernggen.data.schema import (
    CaptionRecord,
    CaptionReviewRecord,
    GovernanceRecord,
    LatentRecord,
    TextEmbeddingRecord,
)


def setup_fixture_paired_dataset(
    root_dir: Path,
    dataset_id: str = "fixture_eligibility_ds",
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
                "caption_sha256": compute_caption_sha256(f"Synthetic Khmer scene {i+1}"),
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
                "caption": f"Synthetic Khmer scene {i+1}",
                "caption_source": "synthetic",
                "caption_version": "captions_v002",
                "caption_sha256": compute_caption_sha256(f"Synthetic Khmer scene {i+1}"),
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


def create_sample_mock_records(img_id: str = "IMG-000001", caption_text: str = "A Khmer folk tale scene"):
    """Helper to generate valid mock schema records for evaluator unit testing."""
    cap_sha = compute_caption_sha256(caption_text)
    gov = GovernanceRecord(
        image_id=img_id,
        dataset_id="test_ds",
        governance_version="rights_v001",
        training_allowed=True,
        commercial_allowed=True,
        authorization_source="human_audit",
        authorized_at="2026-08-31T00:00:00Z",
        status="ACTIVE",
    )
    cap = CaptionRecord(
        image_id=img_id,
        dataset_id="test_ds",
        caption=caption_text,
        caption_source="synthetic",
        caption_version="captions_v002",
        caption_sha256=cap_sha,
        language="en",
    )
    rev = CaptionReviewRecord(
        image_id=img_id,
        dataset_id="test_ds",
        review_version="caption_review_v001",
        caption_sha256=cap_sha,
        review_status="APPROVED",
        reviewed_by="reviewer_alice",
        review_source="human_audit",
        reviewed_at="2026-08-31T00:00:00Z",
    )
    lat = LatentRecord(
        image_id=img_id,
        dataset_id="test_ds",
        dataset_version="v001",
        source_processed_sha256="src_sha",
        preprocessing_version="square256_center_v001",
        vae_model_id="mock_vae",
        vae_revision="mock_rev",
        vae_weights_sha256="w_sha",
        vae_config_sha256="c_sha",
        vae_scaling_factor=0.18215,
        posterior_policy="posterior_mode",
        latent_shape=[4, 32, 32],
        latent_dtype="float32",
        latent_sha256="lat_sha",
        latent_relative_path="lat.safetensors",
        min_val=0.0,
        max_val=1.0,
        mean_val=0.5,
        std_val=0.2,
        l2_norm=10.0,
        cache_version="vae_sd_mse_square256_v001",
        status="CACHED",
    )
    txt = TextEmbeddingRecord(
        image_id=img_id,
        dataset_id="test_ds",
        dataset_version="v001",
        caption_version="captions_v002",
        caption_sha256=cap_sha,
        text_encoder_id="mock_enc",
        text_encoder_revision="mock_rev",
        text_encoder_weights_sha256="w_sha",
        text_encoder_config_sha256="c_sha",
        tokenizer_class="MockTokenizer",
        tokenizer_config_sha256="tok_cfg",
        vocab_sha256="v_sha",
        merges_sha256="m_sha",
        special_tokens_map_sha256="s_sha",
        tokenizer_identity_sha256="tok_id",
        max_token_length=77,
        pooling_policy="eos_token",
        embedding_shape=[512],
        embedding_dtype="float32",
        embedding_sha256="emb_sha",
        embedding_relative_path="emb.safetensors",
        min_val=0.0,
        max_val=1.0,
        mean_val=0.5,
        std_val=0.2,
        l2_norm=10.0,
        token_count=10,
        truncated=False,
        cache_version="clip_b32_v001",
        status="CACHED",
    )
    return gov, cap, rev, lat, txt


# =============================================================================
# Test Matrix 1: ACTIVE + ALLOW, APPROVED + hash match, valid artifact -> ELIGIBLE
# =============================================================================
def test_matrix_1_fully_eligible_sample():
    evaluator = TrainingEligibilityEvaluator()
    gov, cap, rev, lat, txt = create_sample_mock_records()

    decision = evaluator.evaluate_sample(
        sample_id="IMG-000001",
        governance_record=gov,
        caption_record=cap,
        caption_review_record=rev,
        latent_record=lat,
        text_record=txt,
        governance_version="rights_v001",
        caption_review_version="caption_review_v001",
    )

    assert decision.training_allowed is True
    assert decision.reason_codes == [EligibilityReasonCode.ELIGIBLE.value]
    assert decision.governance_effective_status == "ACTIVE_ALLOW"
    assert decision.caption_review_effective_status == "APPROVED"
    assert decision.caption_hash_match is True
    assert decision.artifact_valid is True
    assert decision.policy_version == TRAINING_ELIGIBILITY_POLICY_VERSION


# =============================================================================
# Test Matrix 2: ACTIVE + DENY -> INELIGIBLE (GOVERNANCE_DENIED)
# =============================================================================
def test_matrix_2_governance_denied():
    evaluator = TrainingEligibilityEvaluator()
    gov, cap, rev, lat, txt = create_sample_mock_records()
    gov.training_allowed = False

    decision = evaluator.evaluate_sample(
        sample_id="IMG-000001",
        governance_record=gov,
        caption_record=cap,
        caption_review_record=rev,
        latent_record=lat,
        text_record=txt,
    )

    assert decision.training_allowed is False
    assert EligibilityReasonCode.GOVERNANCE_DENIED.value in decision.reason_codes
    assert decision.governance_effective_status == "ACTIVE_DENY"


# =============================================================================
# Test Matrix 3 & 4: Non-ACTIVE status (REVOKED / SUPERSEDED) -> INELIGIBLE
# =============================================================================
def test_matrix_3_and_4_governance_not_active():
    evaluator = TrainingEligibilityEvaluator()

    # REVOKED
    gov_revoked, cap, rev, lat, txt = create_sample_mock_records()
    gov_revoked.status = "REVOKED"
    gov_revoked.training_allowed = True
    d_rev = evaluator.evaluate_sample(
        sample_id="IMG-000001",
        governance_record=gov_revoked,
        caption_record=cap,
        caption_review_record=rev,
        latent_record=lat,
        text_record=txt,
    )
    assert d_rev.training_allowed is False
    assert EligibilityReasonCode.GOVERNANCE_NOT_ACTIVE.value in d_rev.reason_codes
    assert d_rev.governance_effective_status == "REVOKED"

    # SUPERSEDED
    gov_sup, _, _, _, _ = create_sample_mock_records()
    gov_sup.status = "SUPERSEDED"
    gov_sup.training_allowed = True
    d_sup = evaluator.evaluate_sample(
        sample_id="IMG-000001",
        governance_record=gov_sup,
        caption_record=cap,
        caption_review_record=rev,
        latent_record=lat,
        text_record=txt,
    )
    assert d_sup.training_allowed is False
    assert EligibilityReasonCode.GOVERNANCE_NOT_ACTIVE.value in d_sup.reason_codes
    assert d_sup.governance_effective_status == "SUPERSEDED"


# =============================================================================
# Test Matrix 5, 6, 7: Caption review non-approved states
# =============================================================================
def test_matrix_5_6_7_caption_non_approved_states():
    evaluator = TrainingEligibilityEvaluator()

    # PENDING
    gov, cap, rev, lat, txt = create_sample_mock_records()
    rev.review_status = "PENDING"
    d_pend = evaluator.evaluate_sample(
        sample_id="IMG-000001",
        governance_record=gov,
        caption_record=cap,
        caption_review_record=rev,
        latent_record=lat,
        text_record=txt,
    )
    assert d_pend.training_allowed is False
    assert EligibilityReasonCode.CAPTION_PENDING.value in d_pend.reason_codes
    assert d_pend.caption_review_effective_status == "PENDING"

    # REJECTED
    rev.review_status = "REJECTED"
    d_rej = evaluator.evaluate_sample(
        sample_id="IMG-000001",
        governance_record=gov,
        caption_record=cap,
        caption_review_record=rev,
        latent_record=lat,
        text_record=txt,
    )
    assert d_rej.training_allowed is False
    assert EligibilityReasonCode.CAPTION_REJECTED.value in d_rej.reason_codes
    assert d_rej.caption_review_effective_status == "REJECTED"

    # INVALIDATED
    rev.review_status = "INVALIDATED"
    d_inv = evaluator.evaluate_sample(
        sample_id="IMG-000001",
        governance_record=gov,
        caption_record=cap,
        caption_review_record=rev,
        latent_record=lat,
        text_record=txt,
    )
    assert d_inv.training_allowed is False
    assert EligibilityReasonCode.CAPTION_INVALIDATED.value in d_inv.reason_codes
    assert d_inv.caption_review_effective_status == "INVALIDATED"


# =============================================================================
# Test Matrix 8: Caption hash mismatch -> CAPTION_HASH_MISMATCH
# =============================================================================
def test_matrix_8_caption_hash_mismatch():
    evaluator = TrainingEligibilityEvaluator()
    gov, cap, rev, lat, txt = create_sample_mock_records()

    # Upstream caption text changed after review
    cap.caption = "Completely altered caption text"
    cap.caption_sha256 = compute_caption_sha256(cap.caption)

    decision = evaluator.evaluate_sample(
        sample_id="IMG-000001",
        governance_record=gov,
        caption_record=cap,
        caption_review_record=rev,
        latent_record=lat,
        text_record=txt,
    )

    assert decision.training_allowed is False
    assert decision.caption_hash_match is False
    assert EligibilityReasonCode.CAPTION_HASH_MISMATCH.value in decision.reason_codes
    assert EligibilityReasonCode.CAPTION_INVALIDATED.value not in decision.reason_codes
    assert decision.caption_review_effective_status == "INVALIDATED"


# =============================================================================
# Test Matrix 9 & 10: Artifact missing / invalid
# =============================================================================
def test_matrix_9_and_10_artifact_missing_and_invalid():
    evaluator = TrainingEligibilityEvaluator()
    gov, cap, rev, lat, txt = create_sample_mock_records()

    # Missing latent artifact
    d_miss = evaluator.evaluate_sample(
        sample_id="IMG-000001",
        governance_record=gov,
        caption_record=cap,
        caption_review_record=rev,
        latent_record=None,
        text_record=txt,
    )
    assert d_miss.training_allowed is False
    assert d_miss.artifact_valid is False
    assert EligibilityReasonCode.ARTIFACT_MISSING.value in d_miss.reason_codes

    # Invalid latent artifact shape
    lat.latent_shape = [4, 64, 64]
    d_inv = evaluator.evaluate_sample(
        sample_id="IMG-000001",
        governance_record=gov,
        caption_record=cap,
        caption_review_record=rev,
        latent_record=lat,
        text_record=txt,
    )
    assert d_inv.training_allowed is False
    assert d_inv.artifact_valid is False
    assert EligibilityReasonCode.ARTIFACT_INVALID.value in d_inv.reason_codes


# =============================================================================
# Test Matrix 11 & 12: Multi-failure reason preservation & ordering
# =============================================================================
def test_matrix_11_and_12_multi_failure_preservation():
    evaluator = TrainingEligibilityEvaluator()

    # Missing governance record + missing review record + missing artifacts
    d_all_missing = evaluator.evaluate_sample(
        sample_id="IMG-000001",
        governance_record=None,
        caption_record=None,
        caption_review_record=None,
        latent_record=None,
        text_record=None,
    )

    assert d_all_missing.training_allowed is False
    reasons = d_all_missing.reason_codes

    # Verify exactly the independent gate failures are present (no double counting)
    assert reasons == [
        EligibilityReasonCode.GOVERNANCE_RECORD_MISSING.value,
        EligibilityReasonCode.CAPTION_REVIEW_MISSING.value,
        EligibilityReasonCode.ARTIFACT_MISSING.value,
    ]
    assert EligibilityReasonCode.GOVERNANCE_UNKNOWN.value not in reasons
    assert EligibilityReasonCode.CAPTION_PENDING.value not in reasons

    # Verify deterministic ordering: governance reasons first, then caption, then artifact
    gov_idx = reasons.index(EligibilityReasonCode.GOVERNANCE_RECORD_MISSING.value)
    cap_idx = reasons.index(EligibilityReasonCode.CAPTION_REVIEW_MISSING.value)
    art_idx = reasons.index(EligibilityReasonCode.ARTIFACT_MISSING.value)
    assert gov_idx < cap_idx < art_idx


def test_governance_reason_distinctions():
    """Verify distinct non-overlapping reason codes for all governance conditions."""
    evaluator = TrainingEligibilityEvaluator()
    _, cap, rev, lat, txt = create_sample_mock_records()

    # 1. Missing governance record -> GOVERNANCE_RECORD_MISSING (not GOVERNANCE_UNKNOWN)
    d_miss = evaluator.evaluate_sample(
        sample_id="IMG-000001",
        governance_record=None,
        caption_record=cap,
        caption_review_record=rev,
        latent_record=lat,
        text_record=txt,
    )
    assert d_miss.reason_codes == [EligibilityReasonCode.GOVERNANCE_RECORD_MISSING.value]
    assert EligibilityReasonCode.GOVERNANCE_UNKNOWN.value not in d_miss.reason_codes

    # 2. Existing record with training_allowed=None -> GOVERNANCE_UNKNOWN (not GOVERNANCE_RECORD_MISSING)
    gov_undecided, _, _, _, _ = create_sample_mock_records()
    gov_undecided.training_allowed = None
    d_undecided = evaluator.evaluate_sample(
        sample_id="IMG-000001",
        governance_record=gov_undecided,
        caption_record=cap,
        caption_review_record=rev,
        latent_record=lat,
        text_record=txt,
    )
    assert d_undecided.reason_codes == [EligibilityReasonCode.GOVERNANCE_UNKNOWN.value]
    assert EligibilityReasonCode.GOVERNANCE_RECORD_MISSING.value not in d_undecided.reason_codes

    # 3. Explicit deny -> GOVERNANCE_DENIED
    gov_deny, _, _, _, _ = create_sample_mock_records()
    gov_deny.training_allowed = False
    d_deny = evaluator.evaluate_sample(
        sample_id="IMG-000001",
        governance_record=gov_deny,
        caption_record=cap,
        caption_review_record=rev,
        latent_record=lat,
        text_record=txt,
    )
    assert d_deny.reason_codes == [EligibilityReasonCode.GOVERNANCE_DENIED.value]

    # 4. Status not active -> GOVERNANCE_NOT_ACTIVE
    gov_revoked, _, _, _, _ = create_sample_mock_records()
    gov_revoked.status = "REVOKED"
    d_not_active = evaluator.evaluate_sample(
        sample_id="IMG-000001",
        governance_record=gov_revoked,
        caption_record=cap,
        caption_review_record=rev,
        latent_record=lat,
        text_record=txt,
    )
    assert d_not_active.reason_codes == [EligibilityReasonCode.GOVERNANCE_NOT_ACTIVE.value]


def test_caption_review_reason_distinctions():
    """Verify distinct non-overlapping reason codes for all caption review conditions."""
    evaluator = TrainingEligibilityEvaluator()
    gov, cap, _, lat, txt = create_sample_mock_records()

    # 1. Missing review record -> CAPTION_REVIEW_MISSING (not CAPTION_PENDING)
    d_miss = evaluator.evaluate_sample(
        sample_id="IMG-000001",
        governance_record=gov,
        caption_record=cap,
        caption_review_record=None,
        latent_record=lat,
        text_record=txt,
    )
    assert d_miss.reason_codes == [EligibilityReasonCode.CAPTION_REVIEW_MISSING.value]
    assert EligibilityReasonCode.CAPTION_PENDING.value not in d_miss.reason_codes

    # 2. Existing review record with status=PENDING -> CAPTION_PENDING (not CAPTION_REVIEW_MISSING)
    _, _, rev_pending, _, _ = create_sample_mock_records()
    rev_pending.review_status = "PENDING"
    d_pending = evaluator.evaluate_sample(
        sample_id="IMG-000001",
        governance_record=gov,
        caption_record=cap,
        caption_review_record=rev_pending,
        latent_record=lat,
        text_record=txt,
    )
    assert d_pending.reason_codes == [EligibilityReasonCode.CAPTION_PENDING.value]
    assert EligibilityReasonCode.CAPTION_REVIEW_MISSING.value not in d_pending.reason_codes

    # 3. Existing review record with status=REJECTED -> CAPTION_REJECTED
    _, _, rev_rejected, _, _ = create_sample_mock_records()
    rev_rejected.review_status = "REJECTED"
    d_rej = evaluator.evaluate_sample(
        sample_id="IMG-000001",
        governance_record=gov,
        caption_record=cap,
        caption_review_record=rev_rejected,
        latent_record=lat,
        text_record=txt,
    )
    assert d_rej.reason_codes == [EligibilityReasonCode.CAPTION_REJECTED.value]

    # 4. Existing review record with status=INVALIDATED -> CAPTION_INVALIDATED
    _, _, rev_inv, _, _ = create_sample_mock_records()
    rev_inv.review_status = "INVALIDATED"
    d_inv = evaluator.evaluate_sample(
        sample_id="IMG-000001",
        governance_record=gov,
        caption_record=cap,
        caption_review_record=rev_inv,
        latent_record=lat,
        text_record=txt,
    )
    assert d_inv.reason_codes == [EligibilityReasonCode.CAPTION_INVALIDATED.value]

    # 5. APPROVED record with modified caption text -> CAPTION_HASH_MISMATCH (not CAPTION_INVALIDATED)
    _, cap_mod, rev_app, _, _ = create_sample_mock_records()
    cap_mod.caption = "Modified caption"
    cap_mod.caption_sha256 = compute_caption_sha256(cap_mod.caption)
    d_mismatch = evaluator.evaluate_sample(
        sample_id="IMG-000001",
        governance_record=gov,
        caption_record=cap_mod,
        caption_review_record=rev_app,
        latent_record=lat,
        text_record=txt,
    )
    assert d_mismatch.reason_codes == [EligibilityReasonCode.CAPTION_HASH_MISMATCH.value]
    assert EligibilityReasonCode.CAPTION_INVALIDATED.value not in d_mismatch.reason_codes



# =============================================================================
# Test Matrix 13 & 14: Explicit versions missing in strict mode
# =============================================================================
def test_matrix_13_and_14_missing_explicit_versions_in_strict_mode():
    evaluator = TrainingEligibilityEvaluator()
    gov, cap, rev, lat, txt = create_sample_mock_records()

    # Missing governance version in strict mode
    d_no_gov_v = evaluator.evaluate_sample(
        sample_id="IMG-000001",
        governance_record=gov,
        caption_record=cap,
        caption_review_record=rev,
        latent_record=lat,
        text_record=txt,
        governance_version=None,
        caption_review_version="caption_review_v001",
        require_explicit_versions=True,
    )
    assert d_no_gov_v.training_allowed is False
    assert EligibilityReasonCode.GOVERNANCE_VERSION_MISSING.value in d_no_gov_v.reason_codes

    # Missing caption review version in strict mode
    d_no_cap_v = evaluator.evaluate_sample(
        sample_id="IMG-000001",
        governance_record=gov,
        caption_record=cap,
        caption_review_record=rev,
        latent_record=lat,
        text_record=txt,
        governance_version="rights_v001",
        caption_review_version=None,
        require_explicit_versions=True,
    )
    assert d_no_cap_v.training_allowed is False
    assert EligibilityReasonCode.CAPTION_REVIEW_VERSION_MISSING.value in d_no_cap_v.reason_codes


# =============================================================================
# Test Matrix 15, 16, 17, 18: Provenance tracking and deterministic serialization
# =============================================================================
def test_matrix_15_to_18_provenance_and_serialization(tmp_path: Path):
    ds_dir = setup_fixture_paired_dataset(tmp_path, "ds_prov", count=2)
    gov_mgr = GovernanceManager(dataset_root=tmp_path)
    rev_mgr = CaptionReviewManager(dataset_root=tmp_path)

    gov_mgr.authorize_samples(
        dataset_id="ds_prov",
        governance_version="rights_v001",
        image_ids=["IMG-000001", "IMG-000002"],
        training_decision="ALLOW",
        commercial_decision="ALLOW",
        authorization_source="human_audit",
    )
    rev_mgr.review_samples(
        dataset_id="ds_prov",
        review_version="caption_review_v001",
        image_ids=["IMG-000001", "IMG-000002"],
        review_status="APPROVED",
        reviewed_by="reviewer_dan",
        review_source="audit",
    )

    dataset = PairedLatentTextDataset(
        dataset_dir=ds_dir,
        governance_version="rights_v001",
        caption_review_version="caption_review_v001",
        governance_mode=GovernanceMode.PRODUCTION_STRICT,
    )

    prov = dataset.eligibility_provenance
    assert prov["policy_version"] == TRAINING_ELIGIBILITY_POLICY_VERSION
    assert prov["governance_version"] == "rights_v001"
    assert isinstance(prov["governance_manifest_sha256"], str) and len(prov["governance_manifest_sha256"]) == 64
    assert prov["caption_review_version"] == "caption_review_v001"
    assert isinstance(prov["caption_review_manifest_sha256"], str) and len(prov["caption_review_manifest_sha256"]) == 64
    assert prov["eligibility_counts"] == {"eligible": 2, "ineligible": 0}

    # Decision serialization
    dec = dataset.training_eligibility("IMG-000001")
    d_dict = dec.to_dict()
    assert d_dict["training_allowed"] is True
    assert d_dict["policy_version"] == TRAINING_ELIGIBILITY_POLICY_VERSION
    json_str1 = json.dumps(d_dict, sort_keys=True)
    json_str2 = json.dumps(d_dict, sort_keys=True)
    assert json_str1 == json_str2


# =============================================================================
# Test Matrix 19 & 20: Caption and Rights Orthogonality
# =============================================================================
def test_matrix_19_and_20_orthogonality():
    evaluator = TrainingEligibilityEvaluator()

    # 1. Caption APPROVED does not grant rights (Rights = DENY)
    gov_deny, cap, rev_app, lat, txt = create_sample_mock_records()
    gov_deny.training_allowed = False
    d_deny = evaluator.evaluate_sample(
        sample_id="IMG-000001",
        governance_record=gov_deny,
        caption_record=cap,
        caption_review_record=rev_app,
        latent_record=lat,
        text_record=txt,
    )
    assert d_deny.training_allowed is False
    assert EligibilityReasonCode.GOVERNANCE_DENIED.value in d_deny.reason_codes

    # 2. Rights ALLOW does not approve caption (Caption = REJECTED)
    gov_allow, cap, rev_rej, lat, txt = create_sample_mock_records()
    rev_rej.review_status = "REJECTED"
    d_rej = evaluator.evaluate_sample(
        sample_id="IMG-000001",
        governance_record=gov_allow,
        caption_record=cap,
        caption_review_record=rev_rej,
        latent_record=lat,
        text_record=txt,
    )
    assert d_rej.training_allowed is False
    assert EligibilityReasonCode.CAPTION_REJECTED.value in d_rej.reason_codes


# =============================================================================
# Test Matrix 21: Changing caption text after approval invalidates eligibility
# =============================================================================
def test_matrix_21_changing_caption_invalidates_dataset_eligibility(tmp_path: Path):
    ds_dir = setup_fixture_paired_dataset(tmp_path, "ds_dyn_inval", count=2)
    gov_mgr = GovernanceManager(dataset_root=tmp_path)
    rev_mgr = CaptionReviewManager(dataset_root=tmp_path)
    cap_mgr = CaptionManager(dataset_root=tmp_path)

    gov_mgr.authorize_samples(
        dataset_id="ds_dyn_inval",
        governance_version="rights_v001",
        image_ids=["IMG-000001", "IMG-000002"],
        training_decision="ALLOW",
        commercial_decision="ALLOW",
        authorization_source="human_audit",
    )
    rev_mgr.review_samples(
        dataset_id="ds_dyn_inval",
        review_version="caption_review_v001",
        image_ids=["IMG-000001", "IMG-000002"],
        review_status="APPROVED",
        reviewed_by="reviewer_alice",
        review_source="audit",
    )

    # 1. Modify caption text for IMG-000002 in captions_v002
    current_recs = cap_mgr.load_captions(dataset_id="ds_dyn_inval", version="captions_v002")
    updated = []
    for r in current_recs:
        if r.image_id == "IMG-000002":
            updated.append(
                cap_mgr.create_caption_record(
                    image_id="IMG-000002",
                    dataset_id="ds_dyn_inval",
                    caption="Modified caption description",
                    caption_version="captions_v002",
                )
            )
        else:
            updated.append(r)
    cap_mgr.save_captions("ds_dyn_inval", updated, version="captions_v002")

    # 2. Production strict mode fails because IMG-000002 is invalidated
    with pytest.raises(PermissionError, match="ineligible for training"):
        PairedLatentTextDataset(
            dataset_dir=ds_dir,
            governance_version="rights_v001",
            caption_review_version="caption_review_v001",
            governance_mode=GovernanceMode.PRODUCTION_STRICT,
        )

    # 3. Development audit mode reports 1 eligible, 1 ineligible
    ds_audit = PairedLatentTextDataset(
        dataset_dir=ds_dir,
        governance_version="rights_v001",
        caption_review_version="caption_review_v001",
        governance_mode=GovernanceMode.DEVELOPMENT_AUDIT,
    )
    assert ds_audit.eligibility_counts == {"eligible": 1, "ineligible": 1}
    assert ds_audit.training_eligibility("IMG-000002").training_allowed is False
    assert EligibilityReasonCode.CAPTION_HASH_MISMATCH.value in ds_audit.training_eligibility("IMG-000002").reason_codes


# =============================================================================
# Test Matrix 22: Canonical caption hash helper produces identical results
# =============================================================================
def test_matrix_22_canonical_caption_hash_helper():
    text_raw = "  A Khmer folklore character with   expressive styling.  \n"
    normalized = normalize_caption_text(text_raw)
    assert normalized == "A Khmer folklore character with expressive styling."

    hash1 = compute_caption_sha256(text_raw)
    hash2 = compute_caption_sha256("A Khmer folklore character with expressive styling.")
    assert hash1 == hash2


# =============================================================================
# Test Matrix 23 & 24: Read-only guarantee (Evaluator does not mutate manifests)
# =============================================================================
def test_matrix_23_and_24_evaluator_is_strictly_read_only(tmp_path: Path):
    ds_dir = setup_fixture_paired_dataset(tmp_path, "ds_readonly", count=2)
    gov_mgr = GovernanceManager(dataset_root=tmp_path)
    rev_mgr = CaptionReviewManager(dataset_root=tmp_path)

    p_gov, _ = gov_mgr.authorize_samples(
        dataset_id="ds_readonly",
        governance_version="rights_v001",
        image_ids=["IMG-000001", "IMG-000002"],
        training_decision="ALLOW",
        commercial_decision="ALLOW",
        authorization_source="human_audit",
    )
    p_rev, _ = rev_mgr.review_samples(
        dataset_id="ds_readonly",
        review_version="caption_review_v001",
        image_ids=["IMG-000001", "IMG-000002"],
        review_status="APPROVED",
        reviewed_by="reviewer_alice",
        review_source="audit",
    )

    gov_bytes_before = p_gov.read_bytes()
    rev_bytes_before = p_rev.read_bytes()

    dataset = PairedLatentTextDataset(
        dataset_dir=ds_dir,
        governance_version="rights_v001",
        caption_review_version="caption_review_v001",
        governance_mode=GovernanceMode.DEVELOPMENT_AUDIT,
    )
    _ = dataset.eligibility_summary
    _ = dataset[0]
    _ = dataset[1]

    # Verify byte-for-byte exact equality
    assert p_gov.read_bytes() == gov_bytes_before
    assert p_rev.read_bytes() == rev_bytes_before


# =============================================================================
# Test CLI: python -m rernggen.data eligibility audit
# =============================================================================
def test_eligibility_cli_audit_execution(tmp_path: Path):
    import subprocess
    import sys

    setup_fixture_paired_dataset(tmp_path, "ds_cli_elig", count=2)

    cmd = [
        sys.executable,
        "-m",
        "rernggen.data",
        "eligibility",
        "audit",
        "--dataset-id",
        "ds_cli_elig",
        "--dataset-root",
        str(tmp_path),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "TRAINING ELIGIBILITY AUDIT" in res.stdout
    assert "Total Paired Samples:       2" in res.stdout
    assert "Eligible for Training:      0" in res.stdout
    assert "Ineligible for Training:    2" in res.stdout
    assert "GOVERNANCE_RECORD_MISSING" in res.stdout
    assert "CAPTION_REVIEW_MISSING" in res.stdout
