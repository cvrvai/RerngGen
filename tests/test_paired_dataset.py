"""Comprehensive unit and integration tests for Step 21 Paired Real-Data Dataset & DataLoader."""

import json
from pathlib import Path
import pytest
from safetensors.torch import save_file
import torch
from rernggen.data.dataset import (
    GovernanceMode,
    PairedLatentTextDataset,
    create_paired_dataloader,
    paired_collate_fn,
)
from rernggen.models.text.interface import TextProjection


def setup_synthetic_paired_dataset(
    root_dir: Path,
    dataset_id: str = "test_paired_ds",
    count: int = 4,
    training_allowed: bool = None,
) -> Path:
    """Helper to setup synthetic latent cache, text cache, and caption manifests."""
    ds_dir = root_dir / dataset_id
    latent_dir = ds_dir / "cache" / "latents" / "vae_sd_mse_square256_v001"
    text_dir = ds_dir / "cache" / "text_embeds" / "clip_b32_v001"
    caption_dir = ds_dir / "captions" / "captions_v002"

    latent_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    caption_dir.mkdir(parents=True, exist_ok=True)

    latent_recs = []
    text_recs = []
    caption_recs = []

    for i in range(count):
        img_id = f"IMG-{i+1:06d}"

        # 1. Latent tensor [4, 32, 32]
        latent_tensor = torch.randn(4, 32, 32)
        save_file({"latent": latent_tensor}, latent_dir / f"{img_id}.safetensors")

        latent_recs.append(
            {
                "image_id": img_id,
                "dataset_id": dataset_id,
                "dataset_version": "v001",
                "source_processed_sha256": f"proc_sha_{i}",
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
                "min_val": -1.0,
                "max_val": 1.0,
                "mean_val": 0.0,
                "std_val": 1.0,
                "l2_norm": 10.0,
                "training_allowed": training_allowed,
                "commercial_allowed": None,
                "license_id": None,
                "cache_version": "vae_sd_mse_square256_v001",
                "status": "CACHED",
            }
        )

        # 2. Text embedding tensor [512]
        text_tensor = torch.randn(512)
        save_file({"embedding": text_tensor}, text_dir / f"{img_id}.safetensors")

        text_recs.append(
            {
                "image_id": img_id,
                "dataset_id": dataset_id,
                "dataset_version": "v001",
                "caption_version": "captions_v002",
                "caption_sha256": f"cap_sha_{i}",
                "text_encoder_id": "mock_text_enc",
                "text_encoder_revision": "mock_rev",
                "text_encoder_weights_sha256": "mock_w_sha",
                "text_encoder_config_sha256": "mock_c_sha",
                "tokenizer_class": "CLIPTokenizer",
                "pooling_policy": "eos_token",
                "embedding_shape": [512],
                "embedding_dtype": "float32",
                "embedding_sha256": f"emb_sha_{i}",
                "embedding_relative_path": f"cache/text_embeds/clip_b32_v001/{img_id}.safetensors",
                "min_val": -1.0,
                "max_val": 1.0,
                "mean_val": 0.0,
                "std_val": 1.0,
                "l2_norm": 10.0,
                "token_count": 20,
                "truncated": False,
                "training_allowed": training_allowed,
                "commercial_allowed": None,
                "license_id": None,
                "cache_version": "clip_b32_v001",
                "status": "CACHED",
            }
        )

        # 3. Caption
        caption_recs.append(
            {
                "image_id": img_id,
                "dataset_id": dataset_id,
                "caption": f"Descriptive caption for scene {i+1}",
                "caption_source": "agent_generated",
                "caption_version": "captions_v002",
                "caption_sha256": f"cap_sha_{i}",
                "language": "en",
                "review_status": "unreviewed",
                "training_allowed": training_allowed,
                "commercial_allowed": None,
                "license_id": None,
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

    return ds_dir


def test_paired_dataset_exact_matching_and_ordering_independence(tmp_path: Path):
    """Verify pairing is strictly by image_id, independent of line order in manifests."""
    ds_dir = setup_synthetic_paired_dataset(tmp_path, count=4)

    # Reverse the order of lines in text manifest
    text_manifest_p = ds_dir / "cache" / "text_embeds" / "clip_b32_v001" / "manifest.jsonl"
    with open(text_manifest_p, "r", encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]
    with open(text_manifest_p, "w", encoding="utf-8") as f:
        for l in reversed(lines):
            f.write(l)

    dataset = PairedLatentTextDataset(
        dataset_dir=ds_dir,
        latent_cache_version="vae_sd_mse_square256_v001",
        text_cache_version="clip_b32_v001",
        caption_version="captions_v002",
    )

    assert len(dataset) == 4
    for i in range(4):
        item = dataset[i]
        assert item["image_id"] == f"IMG-{i+1:06d}"
        assert item["latent"].shape == (4, 32, 32)
        assert item["text_embedding"].shape == (512,)
        assert item["caption"] == f"Descriptive caption for scene {i+1}"


def test_paired_dataset_missing_artifact_handling(tmp_path: Path):
    """Verify that dataset only pairs common IDs and tracks missing artifacts."""
    ds_dir = setup_synthetic_paired_dataset(tmp_path, count=3)

    # Delete latent file and record for image 3
    latent_dir = ds_dir / "cache" / "latents" / "vae_sd_mse_square256_v001"
    (latent_dir / "IMG-000003.safetensors").unlink()
    with open(latent_dir / "manifest.jsonl", "r", encoding="utf-8") as f:
        lines = [l for l in f if "IMG-000003" not in l]
    with open(latent_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
        f.writelines(lines)

    dataset = PairedLatentTextDataset(dataset_dir=ds_dir)
    assert len(dataset) == 2
    assert "IMG-000003" in dataset.missing_latent_ids


def test_paired_dataloader_batch_shapes_and_downstream_projection(tmp_path: Path):
    """Verify DataLoader yields [B, 4, 32, 32] latents and [B, 512] embeds, projecting to [B, 384] with gradients."""
    ds_dir = setup_synthetic_paired_dataset(tmp_path, count=6)
    dataset = PairedLatentTextDataset(dataset_dir=ds_dir)

    dataloader = create_paired_dataloader(dataset, batch_size=4, shuffle=False)
    batches = list(dataloader)

    assert len(batches) == 2
    b1 = batches[0]
    assert b1["latents"].shape == (4, 4, 32, 32)
    assert b1["text_embeddings"].shape == (4, 512)
    assert len(b1["image_ids"]) == 4

    b2 = batches[1]
    assert b2["latents"].shape == (2, 4, 32, 32)
    assert b2["text_embeddings"].shape == (2, 512)
    assert len(b2["image_ids"]) == 2

    # Verify downstream TextProjection receives gradients outside Dataset
    text_proj = TextProjection(in_features=512, out_features=384)
    c_text = text_proj(b1["text_embeddings"])
    assert c_text.shape == (4, 384)
    assert c_text.requires_grad is True

    loss = c_text.sum()
    loss.backward()
    assert text_proj.proj.weight.grad is not None


def test_paired_dataset_governance_modes(tmp_path: Path):
    """Verify that development_audit permits unknown rights while production_strict strictly rejects them."""
    # 1. Unknown rights dataset (training_allowed = None)
    ds_unknown = setup_synthetic_paired_dataset(tmp_path, dataset_id="ds_unknown", count=2, training_allowed=None)

    ds_dev = PairedLatentTextDataset(
        dataset_dir=ds_unknown,
        governance_mode=GovernanceMode.DEVELOPMENT_AUDIT,
    )
    assert len(ds_dev) == 2
    assert ds_dev.governance_counts["unknown"] == 2

    with pytest.raises(PermissionError, match="Production training gate rejected dataset"):
        PairedLatentTextDataset(
            dataset_dir=ds_unknown,
            governance_mode=GovernanceMode.PRODUCTION_STRICT,
        )

    # 2. Approved rights dataset (training_allowed = True)
    ds_approved = setup_synthetic_paired_dataset(tmp_path, dataset_id="ds_approved", count=2, training_allowed=True)
    ds_prod = PairedLatentTextDataset(
        dataset_dir=ds_approved,
        governance_mode=GovernanceMode.PRODUCTION_STRICT,
    )
    assert len(ds_prod) == 2
    assert ds_prod.governance_counts["allowed"] == 2


def test_paired_dataset_real_data_integration():
    """Integration test verifying PairedLatentTextDataset over real khmer_story_cartoon_v001 dataset."""
    ds_dir = Path("datasets/khmer_story_cartoon_v001")
    if not ds_dir.exists():
        pytest.skip("Real dataset directory not found.")

    dataset = PairedLatentTextDataset(
        dataset_dir=ds_dir,
        latent_cache_version="vae_sd_mse_square256_v001",
        text_cache_version="clip_b32_v001",
        caption_version="captions_v002",
        governance_mode=GovernanceMode.DEVELOPMENT_AUDIT,
    )

    assert len(dataset) == 12
    assert len(dataset.missing_latent_ids) == 0
    assert len(dataset.missing_text_ids) == 0

    dataloader = create_paired_dataloader(dataset, batch_size=4, shuffle=False)
    batches = list(dataloader)
    assert len(batches) == 3

    for b in batches:
        assert b["latents"].ndim == 4
        assert b["latents"].shape[1:] == (4, 32, 32)
        assert b["text_embeddings"].ndim == 2
        assert b["text_embeddings"].shape[1] == 512
        assert torch.all(torch.isfinite(b["latents"]))
        assert torch.all(torch.isfinite(b["text_embeddings"]))
