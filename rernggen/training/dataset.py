"""Training dataset loader and batching adapter for immutable frozen dataset snapshots."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from safetensors.torch import load_file
import torch
from torch.utils.data import DataLoader, Dataset

from rernggen.data.importer import compute_sha256
from rernggen.data.snapshot import DatasetSnapshot, DatasetSnapshotManager


class SnapshotTrainingDataset(Dataset):
    """PyTorch Dataset adapter that consumes verified frozen dataset snapshots and enforces artifact SHA integrity."""

    def __init__(
        self,
        snapshot: Optional[DatasetSnapshot] = None,
        dataset_id: Optional[str] = None,
        snapshot_version: Optional[str] = None,
        dataset_root: Union[str, Path] = "datasets",
    ) -> None:
        self.dataset_root = Path(dataset_root)

        if snapshot is not None:
            self.snapshot = snapshot
        else:
            if not dataset_id or not snapshot_version:
                raise ValueError("Either snapshot or (dataset_id, snapshot_version) must be provided.")
            snap_mgr = DatasetSnapshotManager(dataset_root=self.dataset_root)
            self.snapshot = snap_mgr.load_snapshot(
                dataset_id=dataset_id,
                snapshot_version=snapshot_version,
                verify_integrity=True,
            )

        if self.snapshot.metadata.status != "FROZEN":
            raise ValueError(
                f"Training dataset requires FROZEN snapshot status, got '{self.snapshot.metadata.status}'."
            )

        self.dataset_dir = self.dataset_root / self.snapshot.metadata.dataset_id

    def __len__(self) -> int:
        return len(self.snapshot)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.snapshot[idx]

        # 1. Load and cryptographically verify latent tensor artifact
        latent_path = self.dataset_dir / record.latent_relative_path
        if not latent_path.is_file():
            raise FileNotFoundError(
                f"Latent artifact for sample '{record.sample_id}' not found at '{latent_path}'."
            )

        actual_latent_sha = compute_sha256(latent_path)
        if actual_latent_sha != record.latent_sha256:
            raise ValueError(
                f"Latent artifact SHA-256 mismatch for sample '{record.sample_id}': "
                f"actual '{actual_latent_sha}' vs recorded '{record.latent_sha256}'."
            )

        latent_data = load_file(str(latent_path))
        if "latent" not in latent_data:
            raise KeyError(f"Missing 'latent' tensor in '{latent_path}'.")
        latent_tensor = latent_data["latent"].float()

        if list(latent_tensor.shape) != list(record.latent_shape):
            raise ValueError(
                f"Latent tensor shape {list(latent_tensor.shape)} does not match record shape {record.latent_shape}."
            )

        # 2. Load and cryptographically verify text embedding artifact
        text_path = self.dataset_dir / record.text_embedding_relative_path
        if not text_path.is_file():
            raise FileNotFoundError(
                f"Text embedding artifact for sample '{record.sample_id}' not found at '{text_path}'."
            )

        actual_text_sha = compute_sha256(text_path)
        if actual_text_sha != record.text_embedding_sha256:
            raise ValueError(
                f"Text embedding artifact SHA-256 mismatch for sample '{record.sample_id}': "
                f"actual '{actual_text_sha}' vs recorded '{record.text_embedding_sha256}'."
            )

        text_data = load_file(str(text_path))
        if "embedding" not in text_data:
            raise KeyError(f"Missing 'embedding' tensor in '{text_path}'.")
        text_tensor = text_data["embedding"].float()

        if list(text_tensor.shape) != list(record.text_embedding_shape):
            raise ValueError(
                f"Text embedding tensor shape {list(text_tensor.shape)} does not match record shape {record.text_embedding_shape}."
            )

        return {
            "sample_id": record.sample_id,
            "latent": latent_tensor,
            "text_embedding": text_tensor,
            "caption": record.caption,
            "caption_sha256": record.caption_sha256,
        }


def snapshot_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collates a list of snapshot records into batched PyTorch tensors."""
    sample_ids = [b["sample_id"] for b in batch]
    latents = torch.stack([b["latent"] for b in batch], dim=0)
    text_embeddings = torch.stack([b["text_embedding"] for b in batch], dim=0)
    captions = [b["caption"] for b in batch]
    caption_shas = [b["caption_sha256"] for b in batch]

    return {
        "sample_ids": sample_ids,
        "latent": latents,
        "text_embedding": text_embeddings,
        "captions": captions,
        "caption_shas": caption_shas,
    }


def create_snapshot_dataloader(
    dataset: SnapshotTrainingDataset,
    batch_size: int = 2,
    shuffle: bool = True,
    drop_last: bool = False,
    generator: Optional[torch.Generator] = None,
) -> DataLoader:
    """Creates a deterministic DataLoader for training on a frozen dataset snapshot."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=snapshot_collate_fn,
        generator=generator,
    )
