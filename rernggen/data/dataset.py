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
from rernggen.data.latent_cache import LatentCacheLoader
from rernggen.data.schema import CaptionRecord, LatentRecord, TextEmbeddingRecord
from rernggen.data.text_cache import TextEmbeddingCacheLoader

logger = logging.getLogger(__name__)


class GovernanceMode(str, Enum):
    """Enforcement policy for dataset rights and training permissions."""

    DEVELOPMENT_AUDIT = "development_audit"
    PRODUCTION_STRICT = "production_strict"


class PairedLatentTextDataset(Dataset):
    """PyTorch Dataset yielding paired VAE latents [4, 32, 32] and frozen text embeddings [512].

    Artifacts are matched strictly by image_id across versioned cache manifests.
    Zero VAE, Tokenizer, or Text Encoder instances are executed during dataset operations.
    """

    def __init__(
        self,
        dataset_dir: Union[str, Path],
        latent_cache_version: str = "vae_sd_mse_square256_v001",
        text_cache_version: str = "clip_b32_v001",
        caption_version: str = "captions_v002",
        governance_mode: Union[str, GovernanceMode] = GovernanceMode.DEVELOPMENT_AUDIT,
    ) -> None:
        """Initializes and audits the paired dataset.

        Args:
            dataset_dir: Directory containing dataset root (e.g. datasets/khmer_story_cartoon_v001).
            latent_cache_version: Subdirectory identifier for latent safetensors.
            text_cache_version: Subdirectory identifier for text embedding safetensors.
            caption_version: Subdirectory identifier for caption manifest.
            governance_mode: 'development_audit' or 'production_strict'.
        """
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.latent_cache_version = latent_cache_version
        self.text_cache_version = text_cache_version
        self.caption_version = caption_version
        self.governance_mode = (
            governance_mode
            if isinstance(governance_mode, GovernanceMode)
            else GovernanceMode(governance_mode)
        )

        # 1. Initialize loaders
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

        # 2. Load and index manifests strictly by image_id
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

        # 3. Intersect IDs to establish strict pairing contract
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

        # 4. Governance verification
        self.governance_counts = {
            "allowed": 0,
            "denied": 0,
            "unknown": 0,
        }
        unauthorized_ids = []

        for img_id in self.paired_image_ids:
            lat_rec = self.latent_records[img_id]
            training_allowed = lat_rec.training_allowed

            if training_allowed is True:
                self.governance_counts["allowed"] += 1
            elif training_allowed is False:
                self.governance_counts["denied"] += 1
                unauthorized_ids.append(img_id)
            else:
                self.governance_counts["unknown"] += 1
                unauthorized_ids.append(img_id)

        if self.governance_mode == GovernanceMode.PRODUCTION_STRICT:
            if unauthorized_ids:
                raise PermissionError(
                    f"Production training gate rejected dataset: {len(unauthorized_ids)} sample(s) "
                    f"lack explicit training permission (training_allowed is not True). "
                    f"Sample IDs: {unauthorized_ids[:5]}..."
                )
        else:
            if self.governance_counts["unknown"] > 0 or self.governance_counts["denied"] > 0:
                logger.info(
                    f"[GOVERNANCE AUDIT] Dataset '{self.dataset_dir.name}' loaded in development_audit mode. "
                    f"Rights state: {self.governance_counts}."
                )

    def __len__(self) -> int:
        return len(self.paired_image_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Loads and returns a paired training sample.

        Args:
            idx (int): Integer index into sorted paired_image_ids.

        Returns:
            Dict[str, Any]: Dictionary containing image_id, latent, text_embedding, caption, and governance.
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

        return {
            "image_id": image_id,
            "latent": latent,
            "text_embedding": text_embed,
            "caption": cap_rec.caption,
            "governance": {
                "training_allowed": lat_rec.training_allowed,
                "commercial_allowed": lat_rec.commercial_allowed,
                "license_id": lat_rec.license_id,
            },
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
    """
    latents = torch.stack([item["latent"] for item in batch], dim=0)
    text_embeddings = torch.stack([item["text_embedding"] for item in batch], dim=0)
    image_ids = [item["image_id"] for item in batch]
    captions = [item["caption"] for item in batch]
    governance = [item["governance"] for item in batch]

    return {
        "latents": latents,
        "text_embeddings": text_embeddings,
        "image_ids": image_ids,
        "captions": captions,
        "governance": governance,
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
