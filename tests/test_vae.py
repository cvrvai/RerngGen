"""Comprehensive unit and integration tests for Step 18 Frozen VAE Interface & Reconstruction."""

import json
from pathlib import Path
from PIL import Image
import pytest
import torch
import torch.nn as nn
from rernggen.models.vae.interface import AutoencoderKLAdapter, MockVAE, MockVAEConfig, VAESpec
from rernggen.models.vae.validator import ReconstructionValidator, compute_mse_and_psnr


def test_vae_loads_in_eval_mode_and_frozen():
    """Verify that VAE adapter enforces eval() mode and freezes 100% of parameters."""
    mock_vae = MockVAE()
    # Intentionally leave parameters requiring grad before adapter initialization
    for p in mock_vae.parameters():
        p.requires_grad_(True)
    mock_vae.train()

    adapter = AutoencoderKLAdapter(mock_vae)

    assert adapter.vae.training is False, "VAE must be in eval mode!"
    assert all(p.requires_grad is False for p in adapter.vae.parameters()), (
        "All VAE parameters must have requires_grad=False!"
    )


def test_vae_input_normalization_contract():
    """Verify exact mathematical mapping [0, 1] -> [-1, 1] -> [0, 1]."""
    x_zero_one = torch.tensor([[[[0.0, 0.5, 1.0]]]])  # [1, 1, 1, 3]

    # Normalize: 2 * x - 1
    x_norm = AutoencoderKLAdapter.normalize_input(x_zero_one)
    expected_norm = torch.tensor([[[[-1.0, 0.0, 1.0]]]])
    assert torch.allclose(x_norm, expected_norm), "Normalization 2*x - 1 failed!"

    # Unnormalize: clamp((x + 1) / 2, 0, 1)
    x_rec = AutoencoderKLAdapter.unnormalize_output(x_norm)
    assert torch.allclose(x_rec, x_zero_one), "Unnormalization failed to invert normalization!"


def test_vae_encode_decode_shapes_and_scaling_factor():
    """Verify exact tensor shapes [B, 3, 256, 256] -> [B, 4, 32, 32] and scaling factor application."""
    scaling_factor = 0.18215
    mock_vae = MockVAE(latent_channels=4, scaling_factor=scaling_factor)
    adapter = AutoencoderKLAdapter(mock_vae)

    B, C, H, W = 2, 3, 256, 256
    x = torch.randn(B, C, H, W)

    # 1. Test raw latent vs model latent
    z_raw = adapter.encode(x, return_raw=True)
    z_model = adapter.encode(x, return_raw=False)

    assert z_raw.shape == (B, 4, 32, 32)
    assert z_model.shape == (B, 4, 32, 32)
    assert torch.allclose(z_model, z_raw * scaling_factor, atol=1e-5), (
        "z_model must equal z_raw * scaling_factor!"
    )
    assert torch.all(torch.isfinite(z_model)), "Latent contains non-finite values!"

    # 2. Test decode with model latent vs raw latent
    recon_from_model = adapter.decode(z_model, is_model_latent=True)
    recon_from_raw = adapter.decode(z_raw, is_model_latent=False)

    assert recon_from_model.shape == (B, 3, 256, 256)
    assert torch.allclose(recon_from_model, recon_from_raw, atol=1e-5, rtol=1e-4), (
        "Decode from model latent and raw latent must match when is_model_latent is set correctly!"
    )
    assert torch.all(torch.isfinite(recon_from_model)), "Reconstruction contains non-finite values!"


def test_vae_no_gradient_graph_guarantee():
    """Verify that VAE encode and decode strictly run with autograd disabled."""
    mock_vae = MockVAE()
    adapter = AutoencoderKLAdapter(mock_vae)

    # Image tensor with requires_grad=True
    x = torch.randn(2, 3, 256, 256, requires_grad=True)

    z = adapter.encode(x)
    assert z.requires_grad is False
    assert z.grad_fn is None

    recon = adapter.decode(z)
    assert recon.requires_grad is False
    assert recon.grad_fn is None


def test_vae_deterministic_posterior_mode():
    """Verify that encoding is 100% deterministic (posterior mode, not random sample)."""
    mock_vae = MockVAE()
    adapter = AutoencoderKLAdapter(mock_vae)

    x = torch.randn(2, 3, 256, 256)

    z1 = adapter.encode(x)
    z2 = adapter.encode(x)
    assert torch.equal(z1, z2), "Encoding must be strictly deterministic!"

    r1 = adapter.reconstruct(x)
    r2 = adapter.reconstruct(x)
    assert torch.equal(r1, r2), "Reconstruction must be strictly deterministic!"


def test_vae_invalid_shapes_and_channel_errors():
    """Verify error handling for invalid input dimensions and incompatible latent channels."""
    mock_vae = MockVAE(latent_channels=4)
    adapter = AutoencoderKLAdapter(mock_vae)

    # 1. 3D image input error
    with pytest.raises(ValueError, match="Expected 4D image tensor"):
        adapter.encode(torch.randn(3, 256, 256))

    # 2. Wrong image channel count (4 instead of 3)
    with pytest.raises(ValueError, match="Expected 4D image tensor"):
        adapter.encode(torch.randn(1, 4, 256, 256))

    # 3. Spatial dimensions not divisible by 8
    with pytest.raises(ValueError, match="must be divisible by 8"):
        adapter.encode(torch.randn(1, 3, 250, 250))

    # 4. Latent channel mismatch on decode
    with pytest.raises(ValueError, match="Expected 4D latent tensor"):
        adapter.decode(torch.randn(1, 16, 32, 32))

    # 5. Incompatible VAE with 16 latent channels (e.g. FLUX VAE) rejected at init
    class Incompatible16ChannelVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = MockVAEConfig(latent_channels=16)

    with pytest.raises(ValueError, match="Incompatible VAE: expected 4 latent channels, got 16"):
        AutoencoderKLAdapter(Incompatible16ChannelVAE())


def test_vae_batch_sizes_and_dtypes():
    """Verify VAE adapter across varying batch sizes and dtypes."""
    mock_vae = MockVAE()
    adapter = AutoencoderKLAdapter(mock_vae)

    for b in [1, 3, 5]:
        x = torch.randn(b, 3, 256, 256)
        z = adapter.encode(x)
        assert z.shape == (b, 4, 32, 32)
        r = adapter.decode(z)
        assert r.shape == (b, 3, 256, 256)

    for dtype in [torch.float32, torch.float64]:
        adapter.to(dtype=dtype)
        x = torch.randn(2, 3, 256, 256, dtype=dtype)
        z = adapter.encode(x)
        assert z.dtype == dtype


def test_reconstruction_validator_end_to_end(tmp_path: Path):
    """Verify ReconstructionValidator pipeline using mock VAE on synthetic processed dataset."""
    dataset_root = tmp_path / "datasets"
    dataset_id = "test_recon_v001"
    version = "square256_center_v001"

    # Setup synthetic processed directory
    proc_dir = dataset_root / dataset_id / "processed" / version
    proc_dir.mkdir(parents=True, exist_ok=True)

    manifest_records = []
    for i in range(2):
        img_id = f"IMG-{i+1:06d}"
        img_path = proc_dir / f"{img_id}.png"
        img = Image.new("RGB", (256, 256), color=(i * 100, 50, 150))
        img.save(img_path)

        manifest_records.append(
            {
                "image_id": img_id,
                "dataset_id": dataset_id,
                "preprocessing_version": version,
                "output_relative_path": f"processed/{version}/{img_id}.png",
                "original_width": 600,
                "original_height": 600,
            }
        )

    with open(proc_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
        for r in manifest_records:
            f.write(json.dumps(r) + "\n")

    # Run validator with MockVAE
    adapter = AutoencoderKLAdapter(MockVAE())
    validator = ReconstructionValidator(adapter, dataset_root=dataset_root)
    report = validator.validate_dataset(
        dataset_id=dataset_id,
        preprocessing_version=version,
    )

    assert report.images_validated == 2
    assert report.failures == 0
    assert report.report_path.exists()
    assert report.reconstructions_dir.exists()
    assert (report.reconstructions_dir / "IMG-000001.png").exists()
    assert (report.reconstructions_dir / "IMG-000002.png").exists()

    with open(report.report_path, "r", encoding="utf-8") as f:
        saved_report = json.load(f)
    assert saved_report["images_validated"] == 2
    assert "mean" in saved_report["aggregate_psnr"]


def test_real_pretrained_vae_loading_and_reconstruction():
    """Integration test verifying loading the real pretrained AutoencoderKL from local disk."""
    local_vae_path = Path("models/vae/stabilityai--sd-vae-ft-mse")
    if not (local_vae_path / "diffusion_pytorch_model.safetensors").exists():
        pytest.skip("Pretrained VAE weights not cached locally yet.")

    adapter = AutoencoderKLAdapter.from_pretrained(local_vae_path)
    assert adapter.spec.latent_channels == 4
    assert adapter.scaling_factor == 0.18215
    assert adapter.vae.training is False
    assert all(p.requires_grad is False for p in adapter.vae.parameters())

    x = torch.randn(1, 3, 256, 256)
    z_model = adapter.encode(x)
    assert z_model.shape == (1, 4, 32, 32)
    assert torch.all(torch.isfinite(z_model))

    recon = adapter.decode(z_model)
    assert recon.shape == (1, 3, 256, 256)
    assert torch.all(torch.isfinite(recon))
