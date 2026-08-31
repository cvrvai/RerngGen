"""Unified Training Eligibility Gate for RerngGen.

Provides a single authoritative, fail-closed policy engine to determine whether dataset
samples are eligible for model training by synthesizing Rights Authorization, Human Caption
Quality Review, and Training Artifact Integrity.

Policy:
    TRAINING_ELIGIBLE IFF:
        1. Rights Authorized: governance record exists, status == ACTIVE, training_allowed == True
        2. Caption Valid: caption review record exists, status == APPROVED, review caption hash == live caption hash
        3. Artifacts Valid: latent and text embedding records exist and satisfy shape/integrity contract
"""

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
from rernggen.data.captions import compute_caption_sha256
from rernggen.data.schema import (
    CaptionRecord,
    CaptionReviewRecord,
    GovernanceRecord,
    LatentRecord,
    TextEmbeddingRecord,
)

TRAINING_ELIGIBILITY_POLICY_VERSION: str = "training_eligibility_v001"


class EligibilityReasonCode(str, Enum):
    """Machine-readable, deterministic reason codes for training eligibility decisions."""

    ELIGIBLE = "ELIGIBLE"

    # Governance Gate Reasons
    GOVERNANCE_RECORD_MISSING = "GOVERNANCE_RECORD_MISSING"
    GOVERNANCE_NOT_ACTIVE = "GOVERNANCE_NOT_ACTIVE"
    GOVERNANCE_DENIED = "GOVERNANCE_DENIED"
    GOVERNANCE_UNKNOWN = "GOVERNANCE_UNKNOWN"
    GOVERNANCE_VERSION_MISSING = "GOVERNANCE_VERSION_MISSING"

    # Caption Review Gate Reasons
    CAPTION_REVIEW_MISSING = "CAPTION_REVIEW_MISSING"
    CAPTION_PENDING = "CAPTION_PENDING"
    CAPTION_REJECTED = "CAPTION_REJECTED"
    CAPTION_INVALIDATED = "CAPTION_INVALIDATED"
    CAPTION_HASH_MISMATCH = "CAPTION_HASH_MISMATCH"
    CAPTION_REVIEW_VERSION_MISSING = "CAPTION_REVIEW_VERSION_MISSING"

    # Artifact Validity Gate Reasons
    ARTIFACT_MISSING = "ARTIFACT_MISSING"
    ARTIFACT_INVALID = "ARTIFACT_INVALID"


@dataclass
class TrainingEligibilityDecision:
    """Immutable, structured result of evaluating a sample's training eligibility."""

    sample_id: str
    training_allowed: bool
    reason_codes: List[str]
    governance_effective_status: str
    caption_review_effective_status: str
    caption_hash_match: bool
    artifact_valid: bool
    governance_version: Optional[str] = None
    caption_review_version: Optional[str] = None
    governance_manifest_sha256: Optional[str] = None
    caption_review_manifest_sha256: Optional[str] = None
    caption_sha256: Optional[str] = None
    latent_sha256: Optional[str] = None
    policy_version: str = TRAINING_ELIGIBILITY_POLICY_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Converts decision to a JSON-serializable dictionary."""
        return asdict(self)


class TrainingEligibilityEvaluator:
    """Authoritative evaluator enforcing RerngGen training eligibility admission policy."""

    def __init__(self, policy_version: str = TRAINING_ELIGIBILITY_POLICY_VERSION) -> None:
        """Initializes the evaluator with an explicit policy version.

        Args:
            policy_version: Eligibility policy identifier (default: 'training_eligibility_v001').
        """
        self.policy_version = policy_version

    def evaluate_sample(
        self,
        sample_id: str,
        governance_record: Optional[GovernanceRecord] = None,
        caption_record: Optional[CaptionRecord] = None,
        caption_review_record: Optional[CaptionReviewRecord] = None,
        latent_record: Optional[LatentRecord] = None,
        text_record: Optional[TextEmbeddingRecord] = None,
        governance_version: Optional[str] = None,
        caption_review_version: Optional[str] = None,
        governance_manifest_sha256: Optional[str] = None,
        caption_review_manifest_sha256: Optional[str] = None,
        require_explicit_versions: bool = False,
        require_explicit_governance_version: Optional[bool] = None,
        require_explicit_caption_version: Optional[bool] = None,
    ) -> TrainingEligibilityDecision:
        """Evaluates all gates for a single dataset sample.

        Args:
            sample_id: Sample identifier (image_id).
            governance_record: Loaded GovernanceRecord if available.
            caption_record: Loaded CaptionRecord if available.
            caption_review_record: Loaded CaptionReviewRecord if available.
            latent_record: Loaded LatentRecord if available.
            text_record: Loaded TextEmbeddingRecord if available.
            governance_version: Identifier for governance version.
            caption_review_version: Identifier for caption review version.
            governance_manifest_sha256: SHA-256 of governance manifest file.
            caption_review_manifest_sha256: SHA-256 of caption review manifest file.
            require_explicit_versions: If True (production mode), missing version identifiers fail closed.
            require_explicit_governance_version: Explicit override for governance version requirement.
            require_explicit_caption_version: Explicit override for caption review version requirement.

        Returns:
            TrainingEligibilityDecision: Complete structured decision with deterministic reason codes.
        """
        if require_explicit_governance_version is None:
            require_explicit_governance_version = require_explicit_versions
        if require_explicit_caption_version is None:
            require_explicit_caption_version = require_explicit_versions

        reasons: List[str] = []

        # =====================================================================
        # 1. RIGHTS AUTHORIZATION GATE
        # =====================================================================
        gov_pass = False
        if require_explicit_governance_version and not governance_version:
            reasons.append(EligibilityReasonCode.GOVERNANCE_VERSION_MISSING.value)
            gov_effective = "VERSION_MISSING"
        elif governance_record is None:
            reasons.append(EligibilityReasonCode.GOVERNANCE_RECORD_MISSING.value)
            gov_effective = "RECORD_MISSING"
        else:
            if governance_record.status != "ACTIVE":
                reasons.append(EligibilityReasonCode.GOVERNANCE_NOT_ACTIVE.value)
                gov_effective = governance_record.status
            elif governance_record.training_allowed is True:
                gov_effective = "ACTIVE_ALLOW"
                gov_pass = True
            elif governance_record.training_allowed is False:
                reasons.append(EligibilityReasonCode.GOVERNANCE_DENIED.value)
                gov_effective = "ACTIVE_DENY"
            else:
                reasons.append(EligibilityReasonCode.GOVERNANCE_UNKNOWN.value)
                gov_effective = "ACTIVE_UNKNOWN"

        # =====================================================================
        # 2. CAPTION QUALITY & INTEGRITY GATE
        # =====================================================================
        cap_pass = False
        hash_match = False
        live_cap_sha = None

        if caption_record is not None:
            live_cap_sha = compute_caption_sha256(caption_record.caption)

        if require_explicit_caption_version and not caption_review_version:
            reasons.append(EligibilityReasonCode.CAPTION_REVIEW_VERSION_MISSING.value)
            cap_effective = "VERSION_MISSING"
        elif caption_record is None or caption_review_record is None:
            reasons.append(EligibilityReasonCode.CAPTION_REVIEW_MISSING.value)
            cap_effective = "RECORD_MISSING"
        else:
            # Check cryptographic hash match against live canonical caption
            if live_cap_sha is not None and caption_review_record.caption_sha256 == live_cap_sha:
                hash_match = True
            else:
                hash_match = False

            if not hash_match:
                reasons.append(EligibilityReasonCode.CAPTION_HASH_MISMATCH.value)
                cap_effective = "INVALIDATED"
            elif caption_review_record.review_status == "APPROVED":
                cap_effective = "APPROVED"
                cap_pass = True
            elif caption_review_record.review_status == "REJECTED":
                reasons.append(EligibilityReasonCode.CAPTION_REJECTED.value)
                cap_effective = "REJECTED"
            elif caption_review_record.review_status == "INVALIDATED":
                reasons.append(EligibilityReasonCode.CAPTION_INVALIDATED.value)
                cap_effective = "INVALIDATED"
            else:  # PENDING
                reasons.append(EligibilityReasonCode.CAPTION_PENDING.value)
                cap_effective = "PENDING"

        # =====================================================================
        # 3. TRAINING ARTIFACT INTEGRITY GATE
        # =====================================================================
        art_pass = False
        artifact_valid = False
        latent_sha = latent_record.latent_sha256 if latent_record else None

        if latent_record is None or text_record is None:
            reasons.append(EligibilityReasonCode.ARTIFACT_MISSING.value)
        else:
            # Verify artifact shape and cache status contracts
            lat_shape_valid = (latent_record.latent_shape == [4, 32, 32])
            text_shape_valid = (text_record.embedding_shape == [512])
            status_valid = (latent_record.status == "CACHED" and text_record.status == "CACHED")

            if not (lat_shape_valid and text_shape_valid and status_valid):
                reasons.append(EligibilityReasonCode.ARTIFACT_INVALID.value)
            else:
                artifact_valid = True
                art_pass = True

        # =====================================================================
        # 4. DECISION SYNTHESIS & REASON DEDUPLICATION
        # =====================================================================
        training_allowed = (gov_pass and cap_pass and art_pass)

        if training_allowed:
            final_reasons = [EligibilityReasonCode.ELIGIBLE.value]
        else:
            # Deduplicate reasons preserving deterministic order
            seen = set()
            final_reasons = []
            for r in reasons:
                if r not in seen:
                    seen.add(r)
                    final_reasons.append(r)

        return TrainingEligibilityDecision(
            sample_id=sample_id,
            training_allowed=training_allowed,
            reason_codes=final_reasons,
            governance_effective_status=gov_effective,
            caption_review_effective_status=cap_effective,
            caption_hash_match=hash_match,
            artifact_valid=artifact_valid,
            governance_version=governance_version,
            caption_review_version=caption_review_version,
            governance_manifest_sha256=governance_manifest_sha256,
            caption_review_manifest_sha256=caption_review_manifest_sha256,
            caption_sha256=live_cap_sha,
            latent_sha256=latent_sha,
            policy_version=self.policy_version,
        )

    def summarize_decisions(
        self,
        decisions: Dict[str, TrainingEligibilityDecision],
    ) -> Dict[str, Any]:
        """Aggregates a collection of sample eligibility decisions into an audit summary.

        Args:
            decisions: Mapping from sample_id to TrainingEligibilityDecision.

        Returns:
            Dict[str, Any]: Summary dictionary with counts, reason breakdowns, and provenance.
        """
        total = len(decisions)
        eligible = 0
        ineligible = 0
        reason_counts: Dict[str, int] = {}

        gov_version = None
        gov_sha = None
        cap_rev_version = None
        cap_rev_sha = None

        for d in decisions.values():
            if d.training_allowed:
                eligible += 1
            else:
                ineligible += 1

            for rc in d.reason_codes:
                reason_counts[rc] = reason_counts.get(rc, 0) + 1

            if d.governance_version:
                gov_version = d.governance_version
            if d.governance_manifest_sha256:
                gov_sha = d.governance_manifest_sha256
            if d.caption_review_version:
                cap_rev_version = d.caption_review_version
            if d.caption_review_manifest_sha256:
                cap_rev_sha = d.caption_review_manifest_sha256

        return {
            "policy_version": self.policy_version,
            "total_samples": total,
            "eligible_count": eligible,
            "ineligible_count": ineligible,
            "eligibility_counts": {"eligible": eligible, "ineligible": ineligible},
            "reason_counts": reason_counts,
            "governance_version": gov_version,
            "governance_manifest_sha256": gov_sha,
            "caption_review_version": cap_rev_version,
            "caption_review_manifest_sha256": cap_rev_sha,
        }
