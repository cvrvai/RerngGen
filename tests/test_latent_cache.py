"""Comprehensive unit and integration tests for Step 19 Deterministic Permanent Latent Cache."""

import json
from pathlib import Path
from PIL import Image
import pytest
from safetensors.torch import load_file
import torch
import torchvision.transforms.functional as TF
from rernggen.data.importer import compute_sha256
from rernggen.data.latent_cache import LatentCacheGenerator, LatentCacheLoader
from rernggen.models.vae.interface import AutoencoderKLAdapter, MockVAE, VAESpec


def create_synthetic_processed_dataset(root_dir: Path, dataset_id: str, version: str, count: int = 3) -> Path:
    """Helper to create a synthetic processed dataset with 256x256 images and manifest."""
    proc_dir = root_dir / dataset_id / "processed" / version
    proc_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = proc_dir / "manifest.jsonl"

    records = []
    for i in range(count):
        img_id = f"IMG-{i+1:06d}"
        img_path = proc_dir / f"{img_id}.png"
        img = Image.new("RGB", (256, 256), color=((i * 60) % 256, (i * 90) % 256, (i * 120) % 256))
        img.save(img_path)
        img_sha = compute_sha256(img_path)

        records.append(
            {
                "image_id": img_id,
                "dataset_id": dataset_id,
                "dataset_version": "v001",
                "preprocessing_version": version,
                "output_relative_path": f"processed/{version}/{img_id}.png",
                "processed_sha256": img_sha,
                "original_width": 600,
                "original_height": 600,
                "resized_width": 256,
                "resized_height": 256,
                "crop_left": 0,
                "crop_top": 0,
                "crop_width": 256,
                "crop_height": 256,
                "output_width": 256,
                "output_height": 256,
                "output_mode": "RGB",
                "training_allowed": None,
                "commercial_allowed": None,
                "license_id": None,
                "status": "PROCESSED",
            }
        )

    with open(manifest_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    return proc_dir


def test_latent_cache_generation_and_scaled_contract(tmp_path: Path):
    """Verify that cached latents are [4, 32, 32], scaled by scaling_factor, and match live VAE."""
    dataset_id = "test_cache_v001"
    prep_version = "square256_center_v001"
    cache_version = "vae_sd_mse_square256_v001"

    create_synthetic_processed_dataset(tmp_path, dataset_id, prep_version, count=3)

    mock_vae = MockVAE(latent_channels=4, scaling_factor=0.18215)
    adapter = AutoencoderKLAdapter(mock_vae)

    generator = LatentCacheGenerator(
        vae_adapter=adapter,
        cache_version=cache_version,
        dataset_root=tmp_path,
    )

    report = generator.generate_cache(
        dataset_id=dataset_id,
        preprocessing_version=prep_version,
    )

    assert report.total_images_in_dataset == 3
    assert report.latents_created == 3
    assert report.valid_cache_hits == 0
    assert report.failures == 0
    assert report.total_cache_bytes > 0
    assert report.manifest_path.exists()

    # Verify each latent on disk
    loader = LatentCacheLoader(
        dataset_dir=tmp_path / dataset_id,
        cache_version=cache_version,
    )
    manifest = loader.load_manifest()
    assert len(manifest) == 3

    for rec in manifest:
        assert rec.latent_shape == [4, 32, 32]
        assert rec.latent_dtype == "float32"
        assert rec.vae_scaling_factor == 0.18215
        assert rec.posterior_policy == "posterior_mode"
        assert rec.training_allowed is None
        assert rec.commercial_allowed is None

        # Verify statistics are finite numbers
        assert isinstance(rec.min_val, float)
        assert isinstance(rec.max_val, float)
        assert isinstance(rec.mean_val, float)
        assert isinstance(rec.std_val, float)
        assert isinstance(rec.l2_norm, float)

        # 1. Load latent directly from disk via loader
        cached_z = loader.load_latent(rec.image_id)
        assert cached_z.shape == (4, 32, 32)
        assert torch.all(torch.isfinite(cached_z))

        # 2. Compare against fresh live VAE encode
        img_path = tmp_path / dataset_id / f"processed/{prep_version}/{rec.image_id}.png"
        with Image.open(img_path) as pil_img:
            img_tensor = TF.to_tensor(pil_img).unsqueeze(0)
        x_norm = adapter.normalize_input(img_tensor)
        live_z = adapter.encode(x_norm).squeeze(0)

        # Exact tensor equality between live VAE encode and disk cached tensor
        assert torch.allclose(cached_z, live_z, atol=1e-5), (
            f"Cached latent for {rec.image_id} does not match live VAE encode!"
        )


def test_latent_cache_idempotency_and_no_unnecessary_reencoding(tmp_path: Path):
    """Verify that a second run against an unchanged dataset reuses existing latents with 0 re-encoding."""
    dataset_id = "test_idempotent_v001"
    prep_version = "square256_center_v001"
    cache_version = "vae_sd_mse_square256_v001"

    create_synthetic_processed_dataset(tmp_path, dataset_id, prep_version, count=2)

    adapter = AutoencoderKLAdapter(MockVAE())
    generator = LatentCacheGenerator(
        vae_adapter=adapter,
        cache_version=cache_version,
        dataset_root=tmp_path,
    )

    # 1. First run: 2 encoded
    rep1 = generator.generate_cache(dataset_id=dataset_id, preprocessing_version=prep_version)
    assert rep1.latents_created == 2
    assert rep1.valid_cache_hits == 0

    # 2. Second run: 0 encoded, 2 cache hits
    rep2 = generator.generate_cache(dataset_id=dataset_id, preprocessing_version=prep_version)
    assert rep2.latents_created == 0
    assert rep2.valid_cache_hits == 2
    assert rep2.failures == 0


def test_latent_cache_invalidated_when_source_image_changes(tmp_path: Path):
    """Verify that modifying a processed image's hash invalidates its cache entry and triggers re-encode."""
    dataset_id = "test_invalidation_v001"
    prep_version = "square256_center_v001"
    cache_version = "vae_sd_mse_square256_v001"

    proc_dir = create_synthetic_processed_dataset(tmp_path, dataset_id, prep_version, count=2)

    adapter = AutoencoderKLAdapter(MockVAE())
    generator = LatentCacheGenerator(
        vae_adapter=adapter,
        cache_version=cache_version,
        dataset_root=tmp_path,
    )

    # 1. Initial cache
    generator.generate_cache(dataset_id=dataset_id, preprocessing_version=prep_version)

    # 2. Modify image 1 on disk and update manifest hash
    img1_path = proc_dir / "IMG-000001.png"
    Image.new("RGB", (256, 256), color=(255, 255, 255)).save(img1_path)
    new_sha = compute_sha256(img1_path)

    # Update processed manifest with new SHA
    manifest_path = proc_dir / "manifest.jsonl"
    with open(manifest_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    records[0]["processed_sha256"] = new_sha
    with open(manifest_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # 3. Re-run cache: image 1 should be re-encoded, image 2 should be a cache hit
    rep = generator.generate_cache(dataset_id=dataset_id, preprocessing_version=prep_version)
    assert rep.latents_created == 1
    assert rep.valid_cache_hits == 1


def test_latent_cache_invalidated_when_vae_identity_changes(tmp_path: Path):
    """Verify that any change in VAE weights, config hash, revision, or scaling factor invalidates cache."""
    dataset_id = "test_vae_invalidation_v001"
    prep_version = "square256_center_v001"
    cache_version = "vae_sd_mse_square256_v001"

    create_synthetic_processed_dataset(tmp_path, dataset_id, prep_version, count=2)

    # 1. Initial cache under VAE identity A
    spec_a = VAESpec(
        model_id="mock_vae_a",
        revision="rev_a",
        weights_sha256="weights_hash_aaa",
        config_sha256="config_hash_aaa",
        scaling_factor=0.18215,
        posterior_policy="posterior_mode",
    )
    adapter_a = AutoencoderKLAdapter(MockVAE(), spec=spec_a)
    gen_a = LatentCacheGenerator(adapter_a, cache_version=cache_version, dataset_root=tmp_path)
    rep_a = gen_a.generate_cache(dataset_id=dataset_id, preprocessing_version=prep_version)
    assert rep_a.latents_created == 2

    # 2. Run under VAE identity B (different weights hash) -> MUST NOT accept old latents
    spec_b = VAESpec(
        model_id="mock_vae_b",
        revision="rev_b",
        weights_sha256="weights_hash_bbb",  # Changed!
        config_sha256="config_hash_aaa",
        scaling_factor=0.18215,
        posterior_policy="posterior_mode",
    )
    adapter_b = AutoencoderKLAdapter(MockVAE(), spec=spec_b)
    gen_b = LatentCacheGenerator(adapter_b, cache_version=cache_version, dataset_root=tmp_path)
    rep_b = gen_b.generate_cache(dataset_id=dataset_id, preprocessing_version=prep_version)
    assert rep_b.latents_created == 2, "Changed VAE weights hash must trigger complete re-encoding!"
    assert rep_b.valid_cache_hits == 0

    # 3. Run under VAE identity C (different scaling factor) -> MUST NOT accept old latents
    spec_c = VAESpec(
        model_id="mock_vae_c",
        revision="rev_c",
        weights_sha256="weights_hash_bbb",
        config_sha256="config_hash_aaa",
        scaling_factor=0.25000,  # Changed!
        posterior_policy="posterior_mode",
    )
    mock_c = MockVAE(scaling_factor=0.25)
    adapter_c = AutoencoderKLAdapter(mock_c, spec=spec_c)
    gen_c = LatentCacheGenerator(adapter_c, cache_version=cache_version, dataset_root=tmp_path)
    rep_c = gen_c.generate_cache(dataset_id=dataset_id, preprocessing_version=prep_version)
    assert rep_c.latents_created == 2, "Changed VAE scaling factor must trigger complete re-encoding!"
    assert rep_c.valid_cache_hits == 0


def test_latent_cache_atomic_tensor_and_manifest_writes(tmp_path: Path, monkeypatch):
    """Verify that temporary files live in the target directory and failed serialization never corrupts valid artifacts."""
    dataset_id = "test_atomic_v001"
    prep_version = "square256_center_v001"
    cache_version = "vae_sd_mse_square256_v001"

    create_synthetic_processed_dataset(tmp_path, dataset_id, prep_version, count=2)

    adapter = AutoencoderKLAdapter(MockVAE())
    generator = LatentCacheGenerator(adapter, cache_version=cache_version, dataset_root=tmp_path)

    # 1. Successful first run
    generator.generate_cache(dataset_id=dataset_id, preprocessing_version=prep_version)
    cache_dir = tmp_path / dataset_id / "cache" / "latents" / cache_version
    img1_file = cache_dir / "IMG-000001.safetensors"
    manifest_file = cache_dir / "manifest.jsonl"

    original_img1_bytes = img1_file.read_bytes()
    original_manifest_text = manifest_file.read_text(encoding="utf-8")

    # 2. Simulate failure during safetensors save
    def failing_save_file(*args, **kwargs):
        raise IOError("Simulated disk write error mid-stream!")

    monkeypatch.setattr("rernggen.data.latent_cache.save_file", failing_save_file)

    # Re-run with force=True: should fail gracefully and NOT corrupt existing file
    rep = generator.generate_cache(dataset_id=dataset_id, preprocessing_version=prep_version, force=True)
    assert rep.failures == 2
    assert rep.latents_created == 0

    # Verify original valid file was not corrupted
    assert img1_file.read_bytes() == original_img1_bytes, "Failed save must not corrupt valid target file!"
    assert manifest_file.read_text(encoding="utf-8") == original_manifest_text, "Failed run must not corrupt manifest!"

    # Verify no dangling temporary files left in target directory
    tmp_files = list(cache_dir.glob("*_tmp_*"))
    assert len(tmp_files) == 0, "No temporary files should be left behind on error."


def test_latent_cache_loader_operates_without_vae(tmp_path: Path):
    """Verify that LatentCacheLoader loads tensors cleanly without any VAE dependency."""
    dataset_id = "test_loader_no_vae"
    cache_version = "vae_sd_mse_square256_v001"
    cache_dir = tmp_path / dataset_id / "cache" / "latents" / cache_version
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Save synthetic safetensors latent directly
    from safetensors.torch import save_file

    sample_tensor = torch.randn(4, 32, 32)
    save_file({"latent": sample_tensor}, cache_dir / "IMG-000001.safetensors")

    manifest_record = {
        "image_id": "IMG-000001",
        "dataset_id": dataset_id,
        "dataset_version": "v001",
        "source_processed_sha256": "fake_sha",
        "preprocessing_version": "square256_center_v001",
        "vae_model_id": "mock_vae",
        "vae_revision": "mock_rev",
        "vae_weights_sha256": "mock_w_sha",
        "vae_config_sha256": "mock_c_sha",
        "vae_scaling_factor": 0.18215,
        "posterior_policy": "posterior_mode",
        "latent_shape": [4, 32, 32],
        "latent_dtype": "float32",
        "latent_sha256": "fake_lat_sha",
        "latent_relative_path": f"cache/latents/{cache_version}/IMG-000001.safetensors",
        "min_val": -1.0,
        "max_val": 1.0,
        "mean_val": 0.0,
        "std_val": 1.0,
        "l2_norm": 10.0,
        "cache_version": cache_version,
        "status": "CACHED",
    }
    with open(cache_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(manifest_record) + "\n")

    loader = LatentCacheLoader(
        dataset_dir=tmp_path / dataset_id,
        cache_version=cache_version,
    )
    loaded_tensor = loader.load_latent("IMG-000001")

    assert loaded_tensor.shape == (4, 32, 32)
    assert torch.equal(loaded_tensor, sample_tensor), "Loaded latent must match saved tensor bit-for-bit."


def test_latent_cache_with_real_pretrained_vae(tmp_path: Path):
    """Integration test verifying latent cache generation and reload with real pretrained AutoencoderKL."""
    local_vae_path = Path("models/vae/stabilityai--sd-vae-ft-mse")
    if not (local_vae_path / "diffusion_pytorch_model.safetensors").exists():
        pytest.skip("Local pretrained VAE weights not found.")

    dataset_id = "test_real_vae_cache"
    prep_version = "square256_center_v001"
    cache_version = "vae_sd_mse_square256_v001"

    create_synthetic_processed_dataset(tmp_path, dataset_id, prep_version, count=2)

    adapter = AutoencoderKLAdapter.from_pretrained(local_vae_path)
    generator = LatentCacheGenerator(
        vae_adapter=adapter,
        cache_version=cache_version,
        dataset_root=tmp_path,
    )

    report = generator.generate_cache(
        dataset_id=dataset_id,
        preprocessing_version=prep_version,
    )

    assert report.latents_created == 2
    assert report.failures == 0

    loader = LatentCacheLoader(
        dataset_dir=tmp_path / dataset_id,
        cache_version=cache_version,
    )
    manifest = loader.load_manifest()
    assert len(manifest) == 2

    for rec in manifest:
        z = loader.load_latent(rec)
        assert z.shape == (4, 32, 32)
        assert torch.all(torch.isfinite(z))
        assert rec.vae_scaling_factor == 0.18215
