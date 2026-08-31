"""Paired Real-Data Dataset and DataLoader for RerngGen DiT training.

Strictly pairs pre-computed [4, 32, 32] VAE latents with [512] frozen text embeddings by image_id,
enforcing governance gates and zero-model-overhead loading during training iterations.
"""

from enum import Enum
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader, Dataset
from rernggen.data.captions import CaptionManager
from rernggen.data.governance import GovernanceManager
from rernggen.data.latent_cache import LatentCacheLoader
from rernggen.data.schema import CaptionRecord, GovernanceRecord, LatentRecord, TextEmbeddingRecord
from rernggen.data.text_cache import TextEmbeddingCacheLoader

logger = logging.getLogger(__name__)


class GovernanceMode(str, Enum):
    """Enforcement policy for dataset rights and training permissions."""

    DEVELOPMENT_AUDIT = "development_audit"
    PRODUCTION_STRICT = "production_strict"


class PairedLatentTextDataset(Dataset):
    """PyTorch Dataset yielding paired VAE latents [4, 32, 32] and frozen text embeddings [512].

    Artifacts are matched strictly by image_id across versioned cache manifests.
    Effective training/commercial rights are resolved from an explicit versioned governance layer.
    Zero VAE, Tokenizer, or Text Encoder instances are executed during dataset operations.
    """

    def __init__(
        self,
        dataset_dir: Union[str, Path],
        latent_cache_version: str = "vae_sd_mse_square256_v001",
        text_cache_version: str = "clip_b32_v001",
        caption_version: str = "captions_v002",
        governance_version: Optional[str] = None,
        caption_review_version: Optional[str] = None,
        governance_mode: Union[str, GovernanceMode] = GovernanceMode.DEVELOPMENT_AUDIT,
    ) -> None:
        """Initializes and audits the paired dataset.

        Args:
            dataset_dir: Directory containing dataset root (e.g. datasets/khmer_story_cartoon_v001).
            latent_cache_version: Subdirectory identifier for latent safetensors.
            text_cache_version: Subdirectory identifier for text embedding safetensors.
            caption_version: Subdirectory identifier for caption manifest.
            governance_version: Optional governance version identifier (e.g. 'rights_v001').
            caption_review_version: Optional caption review version identifier (e.g. 'caption_review_v001').
            governance_mode: 'development_audit' or 'production_strict'.
        """
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.latent_cache_version = latent_cache_version
        self.text_cache_version = text_cache_version
        self.caption_version = caption_version
        self.governance_version = governance_version
        self.caption_review_version = caption_review_version
        self.governance_mode = (
            governance_mode
            if isinstance(governance_mode, GovernanceMode)
            else GovernanceMode(governance_mode)
        )

        # 1. Initialize loaders & managers
        self.latent_loader = LatentCacheLoader(
            dataset_dir=self.dataset_dir,
            cache_version=self.latent_cache_version,
        )
        self.text_loader = TextEmbeddingCacheLoader(
            dataset_dir=self.dataset_dir,
            cache_version=self.text_cache_version,
        )
        self.caption_manager = CaptionManager(
            dataset_root=self.dataset_dir.parent,
        )
        self.governance_manager = GovernanceManager(
            dataset_root=self.dataset_dir.parent,
        )
        from rernggen.data.caption_review import CaptionReviewManager
        self.caption_review_manager = CaptionReviewManager(
            dataset_root=self.dataset_dir.parent,
        )

        # 2. Enforce explicit governance version for PRODUCTION_STRICT
        if self.governance_mode == GovernanceMode.PRODUCTION_STRICT:
            if self.governance_version is None:
                raise ValueError(
                    "PRODUCTION_STRICT requires an explicitly specified governance_version (e.g. 'rights_v001'). "
                    "Unversioned or implicit governance training is strictly forbidden."
                )

        # 3. Load and index manifests strictly by image_id
        self.latent_records: Dict[str, LatentRecord] = {
            r.image_id: r for r in self.latent_loader.load_manifest()
        }
        self.text_records: Dict[str, TextEmbeddingRecord] = {
            r.image_id: r for r in self.text_loader.load_manifest()
        }
        self.caption_records: Dict[str, CaptionRecord] = {
            r.image_id: r
            for r in self.caption_manager.load_captions(
                dataset_id=self.dataset_dir.name,
                version=self.caption_version,
            )
        }

        # Load governance records & manifest provenance hash if version specified
        self.governance_records: Dict[str, GovernanceRecord] = {}
        self.governance_manifest_sha256: Optional[str] = None

        if self.governance_version is not None:
            try:
                gov_list = self.governance_manager.load_governance(
                    dataset_id=self.dataset_dir.name,
                    version=self.governance_version,
                )
                self.governance_records = {r.image_id: r for r in gov_list}
                self.governance_manifest_sha256 = self.governance_manager.compute_manifest_sha256(
                    dataset_id=self.dataset_dir.name,
                    version=self.governance_version,
                )
            except FileNotFoundError:
                if self.governance_mode == GovernanceMode.PRODUCTION_STRICT:
                    raise FileNotFoundError(
                        f"Required governance manifest for version '{self.governance_version}' not found in {self.dataset_dir}."
                    )
                logger.warning(
                    f"Governance version '{self.governance_version}' not found for dataset '{self.dataset_dir.name}'."
                )
                self.governance_records = {}
                self.governance_manifest_sha256 = None

        # Load caption review records & manifest provenance hash if version specified
        self.caption_review_records: Dict[str, CaptionReviewRecord] = {}
        self.caption_review_manifest_sha256: Optional[str] = None

        if self.caption_review_version is not None:
            try:
                rev_list = self.caption_review_manager.load_reviews(
                    dataset_id=self.dataset_dir.name,
                    version=self.caption_review_version,
                )
                self.caption_review_records = {r.image_id: r for r in rev_list}
                self.caption_review_manifest_sha256 = self.caption_review_manager.compute_manifest_sha256(
                    dataset_id=self.dataset_dir.name,
                    version=self.caption_review_version,
                )
            except FileNotFoundError:
                if self.governance_mode == GovernanceMode.PRODUCTION_STRICT:
                    raise FileNotFoundError(
                        f"Required caption review manifest for version '{self.caption_review_version}' not found in {self.dataset_dir}."
                    )
                logger.warning(
                    f"Caption review version '{self.caption_review_version}' not found for dataset '{self.dataset_dir.name}'."
                )
                self.caption_review_records = {}
                self.caption_review_manifest_sha256 = None

        # 4. Intersect IDs to establish strict pairing contract
        latent_ids = set(self.latent_records.keys())
        text_ids = set(self.text_records.keys())
        caption_ids = set(self.caption_records.keys())

        common_ids = latent_ids & text_ids & caption_ids
        if not common_ids:
            raise ValueError(
                f"Zero paired samples found in {self.dataset_dir}. "
                f"Latent IDs: {len(latent_ids)}, Text IDs: {len(text_ids)}, Caption IDs: {len(caption_ids)}."
            )

        self.paired_image_ids = sorted(list(common_ids))
        self.missing_latent_ids = sorted(list((text_ids | caption_ids) - latent_ids))
        self.missing_text_ids = sorted(list((latent_ids | caption_ids) - text_ids))
        self.missing_caption_ids = sorted(list((latent_ids | text_ids) - caption_ids))

        # 5. Initialize Authoritative Eligibility Policy Engine
        from rernggen.data.eligibility import (
            TRAINING_ELIGIBILITY_POLICY_VERSION,
            TrainingEligibilityEvaluator,
        )
        self.eligibility_evaluator = TrainingEligibilityEvaluator(
            policy_version=TRAINING_ELIGIBILITY_POLICY_VERSION,
        )

        self.eligibility_decisions = {}
        unauthorized_sample_ids = []

        for img_id in self.paired_image_ids:
            gov_rec = self.governance_records.get(img_id)
            cap_rec = self.caption_records.get(img_id)
            rev_rec = self.caption_review_records.get(img_id)
            lat_rec = self.latent_records.get(img_id)
            text_rec = self.text_records.get(img_id)

            decision = self.eligibility_evaluator.evaluate_sample(
                sample_id=img_id,
                governance_record=gov_rec,
                caption_record=cap_rec,
                caption_review_record=rev_rec,
                latent_record=lat_rec,
                text_record=text_rec,
                governance_version=self.governance_version,
                caption_review_version=self.caption_review_version,
                governance_manifest_sha256=self.governance_manifest_sha256,
                caption_review_manifest_sha256=self.caption_review_manifest_sha256,
                require_explicit_governance_version=(self.governance_mode == GovernanceMode.PRODUCTION_STRICT),
                require_explicit_caption_version=(self.governance_mode == GovernanceMode.PRODUCTION_STRICT and self.caption_review_version is not None),
            )
            self.eligibility_decisions[img_id] = decision
            if not decision.training_allowed:
                if self.caption_review_version is None:
                    # When caption review version is omitted, gate strictly on governance and artifacts
                    gov_pass = (decision.governance_effective_status == "ACTIVE_ALLOW")
                    art_pass = decision.artifact_valid
                    if not (gov_pass and art_pass):
                        unauthorized_sample_ids.append((img_id, decision.reason_codes))
                else:
                    unauthorized_sample_ids.append((img_id, decision.reason_codes))

        self.eligibility_summary = self.eligibility_evaluator.summarize_decisions(self.eligibility_decisions)
        self.eligibility_counts = self.eligibility_summary["eligibility_counts"]
        self.eligibility_reason_counts = self.eligibility_summary["reason_counts"]

        # 6. Legacy / Compatibility Counts
        self.governance_counts = {"allowed": 0, "denied": 0, "unknown": 0}
        self.caption_review_counts = {"approved": 0, "rejected": 0, "pending": 0, "invalidated": 0}

        for d in self.eligibility_decisions.values():
            if d.governance_effective_status == "ACTIVE_ALLOW":
                self.governance_counts["allowed"] += 1
            elif d.governance_effective_status in ("ACTIVE_DENY", "REVOKED", "SUPERSEDED"):
                self.governance_counts["denied"] += 1
            else:
                self.governance_counts["unknown"] += 1

            if d.caption_review_effective_status == "APPROVED":
                self.caption_review_counts["approved"] += 1
            elif d.caption_review_effective_status == "REJECTED":
                self.caption_review_counts["rejected"] += 1
            elif d.caption_review_effective_status == "INVALIDATED":
                self.caption_review_counts["invalidated"] += 1
            else:
                self.caption_review_counts["pending"] += 1

        # 7. Production Gate Enforcement
        if self.governance_mode == GovernanceMode.PRODUCTION_STRICT:
            if unauthorized_sample_ids:
                raise PermissionError(
                    f"Production training gate rejected dataset: {len(unauthorized_sample_ids)} sample(s) "
                    f"ineligible for training under policy '{self.eligibility_evaluator.policy_version}'. "
                    f"Sample breakdown: {unauthorized_sample_ids[:5]}..."
                )
        else:
            if self.eligibility_counts["ineligible"] > 0:
                logger.info(
                    f"[ELIGIBILITY AUDIT] Dataset '{self.dataset_dir.name}' loaded in development_audit mode. "
                    f"Eligibility state: {self.eligibility_counts}. Reason breakdown: {self.eligibility_reason_counts}."
                )

    def training_eligibility(self, sample_id: str) -> Any:
        """Returns the authoritative TrainingEligibilityDecision for a specific sample."""
        if sample_id not in self.eligibility_decisions:
            raise KeyError(f"Sample '{sample_id}' not found in dataset pairing.")
        return self.eligibility_decisions[sample_id]

    @property
    def eligibility_provenance(self) -> Dict[str, Any]:
        """Returns complete eligibility provenance metadata for training audit trails."""
        return {
            "dataset_id": self.dataset_dir.name,
            "policy_version": self.eligibility_evaluator.policy_version,
            "governance_version": self.governance_version,
            "governance_manifest_sha256": self.governance_manifest_sha256,
            "caption_review_version": self.caption_review_version,
            "caption_review_manifest_sha256": self.caption_review_manifest_sha256,
            "governance_mode": self.governance_mode.value,
            "eligibility_counts": dict(self.eligibility_counts),
            "reason_counts": dict(self.eligibility_reason_counts),
        }

    @property
    def governance_provenance(self) -> Dict[str, Any]:
        """Returns complete governance provenance metadata for training audit trails."""
        return {
            "dataset_id": self.dataset_dir.name,
            "governance_version": self.governance_version,
            "governance_manifest_sha256": self.governance_manifest_sha256,
            "governance_mode": self.governance_mode.value,
            "governance_counts": dict(self.governance_counts),
        }

    @property
    def caption_review_provenance(self) -> Dict[str, Any]:
        """Returns complete caption review provenance metadata for training audit trails."""
        return {
            "dataset_id": self.dataset_dir.name,
            "caption_review_version": self.caption_review_version,
            "caption_review_manifest_sha256": self.caption_review_manifest_sha256,
            "caption_review_counts": dict(self.caption_review_counts),
        }

    def __len__(self) -> int:
        return len(self.paired_image_ids)


    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Loads and returns a paired training sample with effective governance & caption review metadata.

        Args:
            idx (int): Integer index into sorted paired_image_ids.

        Returns:
            Dict[str, Any]: Dictionary containing image_id, latent, text_embedding, caption, governance, and caption_review.
        """
        image_id = self.paired_image_ids[idx]

        lat_rec = self.latent_records[image_id]
        text_rec = self.text_records[image_id]
        cap_rec = self.caption_records[image_id]

        # 1. Load latent directly from safetensors
        latent = self.latent_loader.load_latent(lat_rec)
        if latent.shape != torch.Size([4, 32, 32]):
            raise ValueError(f"Invalid latent shape for {image_id}: expected [4, 32, 32], got {latent.shape}")
        if not torch.all(torch.isfinite(latent)):
            raise ValueError(f"Non-finite values encountered in latent for {image_id}")

        # 2. Load text embedding directly from safetensors
        text_embed = self.text_loader.load_embedding(text_rec)
        if text_embed.shape != torch.Size([512]):
            raise ValueError(f"Invalid text embedding shape for {image_id}: expected [512], got {text_embed.shape}")
        if not torch.all(torch.isfinite(text_embed)):
            raise ValueError(f"Non-finite values encountered in text embedding for {image_id}")

        # 3. Resolve effective governance metadata
        if image_id in self.governance_records:
            gov_rec = self.governance_records[image_id]
            governance_meta = {
                "training_allowed": gov_rec.training_allowed,
                "commercial_allowed": gov_rec.commercial_allowed,
                "license_id": gov_rec.license_id,
                "authorization_source": gov_rec.authorization_source,
                "authorization_note": gov_rec.authorization_note,
                "evidence_reference": gov_rec.evidence_reference,
                "authorized_at": gov_rec.authorized_at,
                "governance_version": gov_rec.governance_version,
                "status": gov_rec.status,
            }
        else:
            governance_meta = {
                "training_allowed": lat_rec.training_allowed,
                "commercial_allowed": lat_rec.commercial_allowed,
                "license_id": lat_rec.license_id,
                "authorization_source": "unspecified",
                "authorization_note": "",
                "evidence_reference": None,
                "authorized_at": "",
                "governance_version": None,
                "status": "UNKNOWN",
            }

        # 4. Resolve effective caption review metadata
        if image_id in self.caption_review_records:
            rev_rec = self.caption_review_records[image_id]
            hash_match = (rev_rec.caption_sha256 == cap_rec.caption_sha256)
            effective_status = rev_rec.review_status if hash_match else "INVALIDATED"
            review_meta = {
                "review_status": effective_status,
                "reviewed_by": rev_rec.reviewed_by,
                "review_source": rev_rec.review_source,
                "reviewed_at": rev_rec.reviewed_at,
                "reason": rev_rec.reason,
                "caption_hash_match": hash_match,
                "review_version": rev_rec.review_version,
            }
        else:
            review_meta = {
                "review_status": "PENDING",
                "reviewed_by": None,
                "review_source": None,
                "reviewed_at": None,
                "reason": "Unreviewed",
                "caption_hash_match": False,
                "review_version": None,
            }

        return {
            "image_id": image_id,
            "latent": latent,
            "text_embedding": text_embed,
            "caption": cap_rec.caption,
            "caption_sha256": cap_rec.caption_sha256,
            "governance": governance_meta,
            "caption_review": review_meta,
            "training_eligibility": self.eligibility_decisions[image_id].to_dict(),
        }


def paired_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collates a list of sample dictionaries into batched PyTorch tensors.

    Args:
        batch (List[Dict[str, Any]]): List of individual sample items from PairedLatentTextDataset.

    Returns:
        Dict[str, Any]: Batched dictionary with:
            - latents: torch.Tensor [B, 4, 32, 32]
            - text_embeddings: torch.Tensor [B, 512]
            - image_ids: List[str]
            - captions: List[str]
            - governance: List[Dict[str, Any]]
            - caption_reviews: List[Dict[str, Any]]
            - training_eligibility: List[Dict[str, Any]]
    """
    latents = torch.stack([item["latent"] for item in batch], dim=0)
    text_embeddings = torch.stack([item["text_embedding"] for item in batch], dim=0)
    image_ids = [item["image_id"] for item in batch]
    captions = [item["caption"] for item in batch]
    governance = [item["governance"] for item in batch]
    caption_reviews = [item.get("caption_review") for item in batch]
    training_eligibility = [item.get("training_eligibility") for item in batch]

    return {
        "latents": latents,
        "text_embeddings": text_embeddings,
        "image_ids": image_ids,
        "captions": captions,
        "governance": governance,
        "caption_reviews": caption_reviews,
        "training_eligibility": training_eligibility,
    }




def create_paired_dataloader(
    dataset: PairedLatentTextDataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: Optional[int] = None,
    drop_last: bool = False,
) -> DataLoader:
    """Constructs a standard PyTorch DataLoader wrapping a PairedLatentTextDataset.

    Args:
        dataset (PairedLatentTextDataset): Initialized paired dataset.
        batch_size (int): Batch size per iteration (default: 4).
        shuffle (bool): Whether to shuffle dataset samples (default: True).
        num_workers (int): Number of worker subprocesses (default: 0).
        seed (Optional[int]): Optional random generator seed for reproducible shuffling.
        drop_last (bool): Whether to drop incomplete final batch (default: False).

    Returns:
        DataLoader: Configured PyTorch DataLoader.
    """
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=paired_collate_fn,
        generator=generator,
        drop_last=drop_last,
    )
