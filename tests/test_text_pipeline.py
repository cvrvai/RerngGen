"""Comprehensive unit and integration tests for Step 20 Caption & Frozen Text Conditioning Pipeline."""

import json
from pathlib import Path
import pytest
from safetensors.torch import load_file
import torch
import torch.nn as nn
from rernggen.data.captions import CaptionManager, compute_caption_sha256
from rernggen.data.schema import CaptionRecord
from rernggen.data.text_cache import TextEmbeddingCacheGenerator, TextEmbeddingCacheLoader
from rernggen.models.text.interface import (
    CLIPTextEncoderAdapter,
    MockTextEncoder,
    TextEncoderSpec,
    TextProjection,
)


def test_caption_manifest_serialization_and_hashing(tmp_path: Path):
    """Verify CaptionManager constructs valid SHA-256 hashes and saves/loads atomically."""
    cap_mgr = CaptionManager(dataset_root=tmp_path)
    dataset_id = "test_cap_ds"
    version = "captions_v001"

    rec1 = cap_mgr.create_caption_record(
        image_id="IMG-000001",
        dataset_id=dataset_id,
        caption="A rabbit in traditional Khmer folklore.",
        caption_version=version,
    )
    rec2 = cap_mgr.create_caption_record(
        image_id="IMG-000002",
        dataset_id=dataset_id,
        caption="A crocodile beside a river in ancient Angkor.",
        caption_version=version,
    )

    assert rec1.caption_sha256 == compute_caption_sha256("A rabbit in traditional Khmer folklore.")
    assert rec1.training_allowed is None
    assert rec1.commercial_allowed is None
    assert rec1.language == "en"

    manifest_p = cap_mgr.save_captions(dataset_id=dataset_id, captions=[rec1, rec2], version=version)
    assert manifest_p.exists()

    loaded = cap_mgr.load_captions(dataset_id=dataset_id, version=version)
    assert len(loaded) == 2
    assert loaded[0].image_id == "IMG-000001"
    assert loaded[0].caption == "A rabbit in traditional Khmer folklore."
    assert loaded[1].image_id == "IMG-000002"


def test_caption_empty_rejection(tmp_path: Path):
    """Verify empty or whitespace-only captions are rejected with ValueError."""
    cap_mgr = CaptionManager(dataset_root=tmp_path)
    with pytest.raises(ValueError, match="must be a non-empty string"):
        cap_mgr.create_caption_record(
            image_id="IMG-000001",
            dataset_id="test_ds",
            caption="   ",
        )


def test_mock_text_encoder_and_projection_shapes():
    """Verify frozen text encoder produces [B, 512] and TextProjection maps [B, 512] -> [B, 384]."""
    encoder = MockTextEncoder(output_dim=512)
    proj = TextProjection(in_features=512, out_features=384)

    captions = [
        "A traditional Khmer story illustration.",
        "Another distinct folklore narrative scene.",
    ]

    # 1. Frozen text encoder: [B, 512]
    pooled_embeds = encoder.encode_text(captions)
    assert pooled_embeds.shape == (2, 512)
    assert pooled_embeds.requires_grad is False
    assert torch.all(torch.isfinite(pooled_embeds))

    # Different captions must produce different embeddings
    assert not torch.allclose(pooled_embeds[0], pooled_embeds[1])

    # Same caption must produce identical deterministic embedding
    re_embed = encoder.encode_text([captions[0]])
    assert torch.equal(pooled_embeds[0], re_embed[0])

    # 2. Trainable TextProjection: [B, 512] -> [B, 384]
    c_text = proj(pooled_embeds)
    assert c_text.shape == (2, 384)
    assert c_text.requires_grad is True

    # 3. Verify gradient flow into projection parameters
    loss = c_text.sum()
    loss.backward()
    assert proj.proj.weight.grad is not None
    assert torch.all(torch.isfinite(proj.proj.weight.grad))


def test_text_embedding_cache_generation_and_reload(tmp_path: Path):
    """Verify TextEmbeddingCacheGenerator extracts and caches [512] safetensors matching live encoder."""
    dataset_id = "test_text_cache_v001"
    cap_ver = "captions_v001"
    cache_ver = "clip_b32_v001"

    cap_mgr = CaptionManager(dataset_root=tmp_path)
    records = [
        cap_mgr.create_caption_record("IMG-000001", dataset_id, "Scene one of Khmer tale.", caption_version=cap_ver),
        cap_mgr.create_caption_record("IMG-000002", dataset_id, "Scene two of Khmer tale.", caption_version=cap_ver),
    ]
    cap_mgr.save_captions(dataset_id, records, version=cap_ver)

    encoder = MockTextEncoder(output_dim=512)
    gen = TextEmbeddingCacheGenerator(encoder, cache_version=cache_ver, dataset_root=tmp_path)

    report = gen.generate_cache(dataset_id=dataset_id, caption_version=cap_ver)
    assert report.total_captions_in_dataset == 2
    assert report.embeddings_created == 2
    assert report.valid_cache_hits == 0
    assert report.failures == 0

    loader = TextEmbeddingCacheLoader(dataset_dir=tmp_path / dataset_id, cache_version=cache_ver)
    manifest = loader.load_manifest()
    assert len(manifest) == 2

    for rec in manifest:
        assert rec.embedding_shape == [512]
        assert rec.embedding_dtype == "float32"
        assert rec.training_allowed is None
        assert rec.status == "CACHED"

        cached_emb = loader.load_embedding(rec.image_id)
        assert cached_emb.shape == (512,)
        assert torch.all(torch.isfinite(cached_emb))

        # Check exact equality against live encode
        live_emb = encoder.encode_text(records[0].caption if rec.image_id == "IMG-000001" else records[1].caption).squeeze(0)
        assert torch.equal(cached_emb, live_emb)


def test_text_embedding_cache_idempotency_and_invalidation(tmp_path: Path):
    """Verify that unchanged captions hit cache, while changed caption text invalidates the stale item."""
    dataset_id = "test_text_idemp_v001"
    cap_ver = "captions_v001"
    cache_ver = "clip_b32_v001"

    cap_mgr = CaptionManager(dataset_root=tmp_path)
    records = [
        cap_mgr.create_caption_record("IMG-000001", dataset_id, "Original caption 1.", caption_version=cap_ver),
        cap_mgr.create_caption_record("IMG-000002", dataset_id, "Original caption 2.", caption_version=cap_ver),
    ]
    cap_mgr.save_captions(dataset_id, records, version=cap_ver)

    encoder = MockTextEncoder()
    gen = TextEmbeddingCacheGenerator(encoder, cache_version=cache_ver, dataset_root=tmp_path)

    # 1. Run 1: 2 created
    rep1 = gen.generate_cache(dataset_id=dataset_id, caption_version=cap_ver)
    assert rep1.embeddings_created == 2

    # 2. Run 2: 0 created, 2 cache hits
    rep2 = gen.generate_cache(dataset_id=dataset_id, caption_version=cap_ver)
    assert rep2.embeddings_created == 0
    assert rep2.valid_cache_hits == 2

    # 3. Modify caption 1
    records[0] = cap_mgr.create_caption_record("IMG-000001", dataset_id, "MODIFIED caption 1.", caption_version=cap_ver)
    cap_mgr.save_captions(dataset_id, records, version=cap_ver)

    # 4. Run 3: caption 1 must re-encode, caption 2 must hit cache
    rep3 = gen.generate_cache(dataset_id=dataset_id, caption_version=cap_ver)
    assert rep3.embeddings_created == 1
    assert rep3.valid_cache_hits == 1


def test_text_embedding_cache_atomic_writes(tmp_path: Path, monkeypatch):
    """Verify atomic write resilience against simulated failure."""
    dataset_id = "test_text_atomic"
    cap_ver = "captions_v001"
    cache_ver = "clip_b32_v001"

    cap_mgr = CaptionManager(dataset_root=tmp_path)
    records = [
        cap_mgr.create_caption_record("IMG-000001", dataset_id, "Caption text.", caption_version=cap_ver)
    ]
    cap_mgr.save_captions(dataset_id, records, version=cap_ver)

    encoder = MockTextEncoder()
    gen = TextEmbeddingCacheGenerator(encoder, cache_version=cache_ver, dataset_root=tmp_path)
    gen.generate_cache(dataset_id=dataset_id, caption_version=cap_ver)

    cache_dir = tmp_path / dataset_id / "cache" / "text_embeds" / cache_ver
    orig_bytes = (cache_dir / "IMG-000001.safetensors").read_bytes()

    def failing_save(*args, **kwargs):
        raise IOError("Simulated text embed write failure!")

    monkeypatch.setattr("rernggen.data.text_cache.save_file", failing_save)

    rep = gen.generate_cache(dataset_id=dataset_id, caption_version=cap_ver, force=True)
    assert rep.failures == 1
    assert (cache_dir / "IMG-000001.safetensors").read_bytes() == orig_bytes


def test_real_pretrained_clip_text_encoder_and_cache_end_to_end(tmp_path: Path):
    """Integration test verifying real pretrained CLIPTextModel encoding and caching."""
    model_path = Path("models/text_encoder/openai--clip-text-base-patch32")
    if not (model_path / "model.safetensors").exists():
        pytest.skip("Pretrained CLIP text encoder model not found locally.")

    dataset_id = "test_real_clip_cache"
    cap_ver = "captions_v001"
    cache_ver = "clip_b32_v001"

    cap_mgr = CaptionManager(dataset_root=tmp_path)
    records = [
        cap_mgr.create_caption_record("IMG-000001", dataset_id, "A colorful traditional Khmer cartoon scene.", caption_version=cap_ver),
        cap_mgr.create_caption_record("IMG-000002", dataset_id, "Expressive cartoon animal characters in Cambodian folklore.", caption_version=cap_ver),
    ]
    cap_mgr.save_captions(dataset_id, records, version=cap_ver)

    adapter = CLIPTextEncoderAdapter.from_pretrained(model_path)
    assert adapter.spec.output_dim == 512
    assert adapter.text_model.training is False
    assert all(p.requires_grad is False for p in adapter.text_model.parameters())

    gen = TextEmbeddingCacheGenerator(adapter, cache_version=cache_ver, dataset_root=tmp_path)
    report = gen.generate_cache(dataset_id=dataset_id, caption_version=cap_ver)
    assert report.embeddings_created == 2
    assert report.failures == 0

    loader = TextEmbeddingCacheLoader(dataset_dir=tmp_path / dataset_id, cache_version=cache_ver)
    emb1 = loader.load_embedding("IMG-000001")
    assert emb1.shape == (512,)
    assert torch.all(torch.isfinite(emb1))
