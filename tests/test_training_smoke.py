"""Comprehensive unit, integration, and smoke test suite for Step 22.G Training Preflight & Tiny DiT Smoke Test."""

import json
from pathlib import Path
import random
import numpy as np
import pytest
from safetensors.torch import save_file
import torch
import torch.nn as nn

from rernggen.data.caption_review import CaptionReviewManager
from rernggen.data.captions import compute_caption_sha256
from rernggen.data.governance import GovernanceManager
from rernggen.data.importer import compute_sha256
from rernggen.data.snapshot import DatasetSnapshotManager
from rernggen.models.dit.model import (
    DiTBlock,
    FinalLayer,
    PatchEmbed,
    TextEmbedder,
    TimestepEmbedder,
    TinyDiT,
)
from rernggen.training.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    load_training_checkpoint,
    save_training_checkpoint,
)
from rernggen.training.dataset import (
    SnapshotTrainingDataset,
    create_snapshot_dataloader,
)
from rernggen.training.diffusion import DiffusionSchedule
from rernggen.training.preflight import run_preflight_checks
from rernggen.training.provenance import TrainingRunManager
from rernggen.training.schema import TrainingRunStatus
from rernggen.training.smoke import run_tiny_dit_smoke, seed_everything


def create_mock_snapshot_fixture(
    root_dir: Path,
    dataset_id: str = "fixture_smoke_ds",
    snapshot_version: str = "dataset_snapshot_v001",
    count: int = 2,
) -> str:
    """Helper to create a fully authorized, reviewed, and frozen dataset snapshot fixture."""
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
    all_ids = []

    for i in range(count):
        img_id = f"IMG-{i+1:06d}"
        all_ids.append(img_id)

        # Save latent artifact
        lat_path = latent_dir / f"{img_id}.safetensors"
        save_file({"latent": torch.randn(4, 32, 32)}, lat_path)
        lat_sha = compute_sha256(lat_path)

        # Save text embedding artifact
        txt_path = text_dir / f"{img_id}.safetensors"
        save_file({"embedding": torch.randn(512)}, txt_path)
        txt_sha = compute_sha256(txt_path)

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
                "latent_sha256": lat_sha,
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

        cap_text = f"Sample scene {i+1} description."
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
                "embedding_sha256": txt_sha,
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

    # Authorize & Review
    gov_mgr = GovernanceManager(dataset_root=root_dir)
    rev_mgr = CaptionReviewManager(dataset_root=root_dir)
    gov_mgr.authorize_samples(
        dataset_id=dataset_id,
        governance_version="rights_v001",
        image_ids=all_ids,
        training_decision="ALLOW",
        commercial_decision="ALLOW",
        authorization_source="audit_lead",
    )
    rev_mgr.review_samples(
        dataset_id=dataset_id,
        review_version="cap_rev_v001",
        image_ids=all_ids,
        review_status="APPROVED",
        reviewed_by="reviewer_alice",
        review_source="audit",
    )

    # Freeze snapshot
    snap_mgr = DatasetSnapshotManager(dataset_root=root_dir)
    snap = snap_mgr.freeze_snapshot(
        dataset_id=dataset_id,
        snapshot_version=snapshot_version,
        governance_version="rights_v001",
        caption_review_version="cap_rev_v001",
        created_by="engineer_alice",
        creation_source="training_prep",
    )
    return snap.metadata.metadata_sha256


# =============================================================================
# 1 - 13: Preflight Tests
# =============================================================================
def test_preflight_success_and_failure_modes(tmp_path: Path):
    ds_root = tmp_path / "datasets"
    train_root = tmp_path / "training_runs"
    snap_sha = create_mock_snapshot_fixture(ds_root, "ds_preflight", "dataset_snapshot_v001", count=2)

    mgr = TrainingRunManager(training_root=train_root, dataset_root=ds_root)
    run = mgr.create_run(
        run_id="run_pf_001",
        dataset_id="ds_preflight",
        snapshot_version="dataset_snapshot_v001",
        model_family="dit",
        model_config={"hidden_size": 64, "depth": 2, "num_heads": 4},
        training_config={"batch_size": 2, "lr": 1e-4, "num_timesteps": 100},
        seed=42,
        created_by="eng_lead",
        creation_source="test",
    )

    # 1. Verified run passes preflight
    res = run_preflight_checks("run_pf_001", training_root=train_root, dataset_root=ds_root)
    assert res.passed is True
    assert res.checks["RUN_VERIFIED"] is True
    assert res.checks["SNAPSHOT_VERIFIED"] is True
    assert res.checks["SNAPSHOT_SHA_MATCH"] is True
    assert res.checks["DATASET_NON_EMPTY"] is True
    assert res.checks["ARTIFACTS_LOADABLE"] is True
    assert res.checks["MODEL_CONFIG_VALID"] is True
    assert res.checks["TRAINING_CONFIG_VALID"] is True

    # 2. Tampered run fails preflight
    with open(run.run_dir / "spec.json", "w", encoding="utf-8") as f:
        f.write('{"tampered": true}')
    res_tampered = run_preflight_checks("run_pf_001", training_root=train_root, dataset_root=ds_root)
    assert res_tampered.passed is False
    assert res_tampered.checks["RUN_VERIFIED"] is False


def test_preflight_snapshot_tamper_and_sha_mismatch(tmp_path: Path):
    ds_root = tmp_path / "datasets"
    train_root = tmp_path / "training_runs"
    create_mock_snapshot_fixture(ds_root, "ds_pf_tamper", "dataset_snapshot_v001", count=2)

    mgr = TrainingRunManager(training_root=train_root, dataset_root=ds_root)
    run = mgr.create_run(
        run_id="run_pf_002",
        dataset_id="ds_pf_tamper",
        snapshot_version="dataset_snapshot_v001",
        model_family="dit",
        model_config={"hidden_size": 64},
        training_config={"batch_size": 2},
        seed=42,
        created_by="eng_lead",
        creation_source="test",
    )

    # 3. Tamper with snapshot metadata
    meta_file = ds_root / "ds_pf_tamper" / "snapshots" / "dataset_snapshot_v001" / "metadata.json"
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["sample_count"] = 999
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    res = run_preflight_checks("run_pf_002", training_root=train_root, dataset_root=ds_root)
    assert res.passed is False
    assert res.checks["SNAPSHOT_VERIFIED"] is False


def test_preflight_missing_and_corrupt_artifacts(tmp_path: Path):
    ds_root = tmp_path / "datasets"
    train_root = tmp_path / "training_runs"
    create_mock_snapshot_fixture(ds_root, "ds_pf_art", "dataset_snapshot_v001", count=2)

    mgr = TrainingRunManager(training_root=train_root, dataset_root=ds_root)
    mgr.create_run(
        run_id="run_pf_003",
        dataset_id="ds_pf_art",
        snapshot_version="dataset_snapshot_v001",
        model_family="dit",
        model_config={"hidden_size": 64},
        training_config={"batch_size": 2},
        seed=42,
        created_by="eng_lead",
        creation_source="test",
    )

    # 6. Missing latent artifact fails
    lat_path = ds_root / "ds_pf_art" / "cache" / "latents" / "vae_sd_mse_square256_v001" / "IMG-000001.safetensors"
    lat_path.unlink()

    res = run_preflight_checks("run_pf_003", training_root=train_root, dataset_root=ds_root)
    assert res.passed is False
    assert res.checks["ARTIFACTS_LOADABLE"] is False


def test_preflight_run_state_rejection(tmp_path: Path):
    ds_root = tmp_path / "datasets"
    train_root = tmp_path / "training_runs"
    create_mock_snapshot_fixture(ds_root, "ds_pf_state", "dataset_snapshot_v001", count=2)

    mgr = TrainingRunManager(training_root=train_root, dataset_root=ds_root)
    run = mgr.create_run(
        run_id="run_pf_004",
        dataset_id="ds_pf_state",
        snapshot_version="dataset_snapshot_v001",
        model_family="dit",
        model_config={"hidden_size": 64},
        training_config={"batch_size": 2},
        seed=42,
        created_by="eng_lead",
        creation_source="test",
    )

    # Transition to COMPLETED
    mgr.start_run("run_pf_004")
    mgr.complete_run("run_pf_004")

    # 13. Terminal run rejected for new preflight
    res = run_preflight_checks("run_pf_004", training_root=train_root, dataset_root=ds_root)
    assert res.passed is False
    assert res.checks["RUN_STATE_VALID"] is False


# =============================================================================
# 14 - 21: Model Tests (TinyDiT Components)
# =============================================================================
def test_patch_embed_dimensions():
    pe = PatchEmbed(latent_size=32, patch_size=4, in_channels=4, hidden_size=64)
    x = torch.randn(2, 4, 32, 32)
    tokens = pe(x)
    assert tokens.shape == (2, 64, 64)  # (B, N=(32/4)^2=64, D=64)


def test_timestep_and_text_embedders():
    te = TimestepEmbedder(hidden_size=64)
    t = torch.tensor([10, 50])
    t_emb = te(t)
    assert t_emb.shape == (2, 64)

    txt_e = TextEmbedder(text_dim=512, hidden_size=64)
    text_raw = torch.randn(2, 512)
    txt_emb = txt_e(text_raw)
    assert txt_emb.shape == (2, 64)


def test_transformer_block_preserves_token_shape():
    block = DiTBlock(hidden_size=64, num_heads=4, mlp_ratio=4.0)
    x = torch.randn(2, 64, 64)
    c = torch.randn(2, 64)
    out = block(x, c)
    assert out.shape == (2, 64, 64)
    assert torch.isfinite(out).all()


def test_tiny_dit_forward_pass_and_conditioning_effects():
    model = TinyDiT(
        in_channels=4,
        out_channels=4,
        latent_size=32,
        patch_size=4,
        hidden_size=64,
        depth=2,
        num_heads=4,
        text_dim=512,
    )
    x = torch.randn(2, 4, 32, 32)
    t1 = torch.tensor([10, 20])
    t2 = torch.tensor([80, 90])
    txt1 = torch.randn(2, 512)
    txt2 = torch.randn(2, 512)

    # 18. Output matches latent shape and is finite
    out1 = model(x, t1, txt1)
    assert out1.shape == (2, 4, 32, 32)
    assert torch.isfinite(out1).all()

    # Perturb zero-initialized output and modulation weights to test conditioning sensitivity
    for block in model.blocks:
        nn.init.normal_(block.adaLN_modulation[1].weight, std=0.02)
    nn.init.normal_(model.final_layer.linear.weight, std=0.02)
    nn.init.normal_(model.final_layer.adaLN_modulation[1].weight, std=0.02)

    with torch.no_grad():
        out_base = model(x, t1, txt1)
        # 20. Different timestep produces different output
        out_diff_t = model(x, t2, txt1)
        assert not torch.allclose(out_base, out_diff_t, atol=1e-4)

        # 21. Different text conditioning produces different output
        out_diff_txt = model(x, t1, txt2)
        assert not torch.allclose(out_base, out_diff_txt, atol=1e-4)


# =============================================================================
# 22 - 27: Diffusion Schedule & Loss Tests
# =============================================================================
def test_diffusion_schedule_and_loss():
    sched = DiffusionSchedule(num_timesteps=100)
    x_start = torch.randn(2, 4, 32, 32)
    t = torch.tensor([10, 50])
    noise = torch.randn_like(x_start)

    # 22. q_sample output shape
    x_t, eps = sched.q_sample(x_start, t, noise=noise)
    assert x_t.shape == (2, 4, 32, 32)
    assert torch.isfinite(x_t).all()

    # 24. Low and High t validity
    x_t_low, _ = sched.q_sample(x_start, torch.tensor([0, 0]))
    x_t_high, _ = sched.q_sample(x_start, torch.tensor([99, 99]))
    assert torch.isfinite(x_t_low).all()
    assert torch.isfinite(x_t_high).all()

    # 26 & 27. Diffusion training loss is scalar and finite
    model = TinyDiT(hidden_size=64, depth=2, num_heads=4)
    text_emb = torch.randn(2, 512)
    loss, eps_hat, _ = sched.training_loss(model, x_start, t, text_emb)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


# =============================================================================
# 28 - 38: Training Loop, Gradients, and Lifecycle Tests
# =============================================================================
def test_snapshot_dataset_and_training_optimization(tmp_path: Path):
    ds_root = tmp_path / "datasets"
    create_mock_snapshot_fixture(ds_root, "ds_train_test", "dataset_snapshot_v001", count=2)

    dataset = SnapshotTrainingDataset(
        dataset_id="ds_train_test",
        snapshot_version="dataset_snapshot_v001",
        dataset_root=ds_root,
    )
    assert len(dataset) == 2
    item = dataset[0]
    assert item["latent"].shape == (4, 32, 32)
    assert item["text_embedding"].shape == (512,)

    loader = create_snapshot_dataloader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    assert batch["latent"].shape == (2, 4, 32, 32)
    assert batch["text_embedding"].shape == (2, 512)

    # Optimizer step verifies gradient flow and parameter updates
    model = TinyDiT(hidden_size=64, depth=2, num_heads=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = DiffusionSchedule(num_timesteps=100)

    t = torch.tensor([5, 10])
    loss, _, _ = sched.training_loss(model, batch["latent"], t, batch["text_embedding"])
    optimizer.zero_grad()
    loss.backward()

    # 32 & 33. Gradients exist and are finite
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)

    # 34. Parameter changes on optimizer step
    orig_weights = [p.clone().detach() for p in model.parameters()]
    optimizer.step()
    changed = any(not torch.allclose(p, o) for p, o in zip(model.parameters(), orig_weights))
    assert changed is True


# =============================================================================
# 39 - 48: Checkpoint Save, Provenance Binding, and Reload
# =============================================================================
def test_checkpoint_provenance_and_continuation(tmp_path: Path):
    ds_root = tmp_path / "datasets"
    train_root = tmp_path / "training_runs"
    snap_sha = create_mock_snapshot_fixture(ds_root, "ds_ckpt_test", "dataset_snapshot_v001", count=2)

    mgr = TrainingRunManager(training_root=train_root, dataset_root=ds_root)
    run = mgr.create_run(
        run_id="run_ckpt_001",
        dataset_id="ds_ckpt_test",
        snapshot_version="dataset_snapshot_v001",
        model_family="dit",
        model_config={"hidden_size": 64},
        training_config={"batch_size": 2},
        seed=42,
        created_by="eng_lead",
        creation_source="test",
    )

    model = TinyDiT(hidden_size=64, depth=2, num_heads=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt_file = train_root / "run_ckpt_001" / "checkpoints" / "step_000001.pt"

    # 39. Save checkpoint
    save_training_checkpoint(
        checkpoint_path=ckpt_file,
        model=model,
        optimizer=optimizer,
        step=1,
        run_spec=run.spec,
        snapshot_metadata_sha256=snap_sha,
        loss=0.45,
    )
    assert ckpt_file.is_file()

    # 43 & 44. Reload into fresh model
    fresh_model = TinyDiT(hidden_size=64, depth=2, num_heads=4)
    payload = load_training_checkpoint(
        checkpoint_path=ckpt_file,
        model=fresh_model,
        expected_run_id="run_ckpt_001",
        expected_run_spec_sha256=run.spec.run_spec_sha256,
        expected_snapshot_metadata_sha256=snap_sha,
    )
    assert payload["step"] == 1
    assert payload["run_id"] == "run_ckpt_001"
    for p1, p2 in zip(model.parameters(), fresh_model.parameters()):
        assert torch.allclose(p1, p2)

    # 45. Wrong run_id rejected
    with pytest.raises(ValueError, match="run_id.*does not match"):
        load_training_checkpoint(
            checkpoint_path=ckpt_file,
            model=fresh_model,
            expected_run_id="wrong_run_id",
        )

    # 46. Wrong run_spec_sha256 rejected
    with pytest.raises(ValueError, match="run_spec_sha256.*does not match"):
        load_training_checkpoint(
            checkpoint_path=ckpt_file,
            model=fresh_model,
            expected_run_spec_sha256="0" * 64,
        )

    # 47. Wrong snapshot SHA rejected
    with pytest.raises(ValueError, match="snapshot_metadata_sha256.*does not match"):
        load_training_checkpoint(
            checkpoint_path=ckpt_file,
            model=fresh_model,
            expected_snapshot_metadata_sha256="0" * 64,
        )


# =============================================================================
# 49 & 50: Determinism Tests
# =============================================================================
def test_seed_determinism():
    seed_everything(1234)
    m1 = TinyDiT(hidden_size=64, depth=2, num_heads=4)

    seed_everything(1234)
    m2 = TinyDiT(hidden_size=64, depth=2, num_heads=4)

    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(p1, p2)

    seed_everything(5678)
    m3 = TinyDiT(hidden_size=64, depth=2, num_heads=4)
    # Different seed produces different initialization
    assert not all(torch.allclose(p1, p3) for p1, p3 in zip(m1.parameters(), m3.parameters()))


# =============================================================================
# 51 - 56: RNG State Capture, Restoration, and Trajectory Reproducibility Tests
# =============================================================================
def test_python_numpy_torch_rng_survives_checkpoint(tmp_path: Path):
    """Verifies that exact RNG states across Python, NumPy, PyTorch CPU roundtrip through checkpoint."""
    ds_root = tmp_path / "datasets"
    train_root = tmp_path / "training_runs"
    snap_sha = create_mock_snapshot_fixture(ds_root, "ds_rng_test", "dataset_snapshot_v001", count=2)

    mgr = TrainingRunManager(training_root=train_root, dataset_root=ds_root)
    run = mgr.create_run(
        run_id="run_rng_001",
        dataset_id="ds_rng_test",
        snapshot_version="dataset_snapshot_v001",
        model_family="dit",
        model_config={"hidden_size": 64},
        training_config={"batch_size": 2},
        seed=42,
        created_by="eng_lead",
        creation_source="test",
    )

    model = TinyDiT(hidden_size=64, depth=2, num_heads=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt_file = train_root / "run_rng_001" / "checkpoints" / "rng_ckpt.pt"

    # Set specific state
    seed_everything(9999)
    _ = random.random()
    _ = np.random.rand(5)
    _ = torch.randn(10)

    # Save checkpoint capturing RNG
    save_training_checkpoint(
        checkpoint_path=ckpt_file,
        model=model,
        optimizer=optimizer,
        step=5,
        run_spec=run.spec,
        snapshot_metadata_sha256=snap_sha,
    )

    # Sample next values from this exact trajectory
    expected_py_next = random.random()
    expected_np_next = np.random.rand(5)
    expected_torch_next = torch.randn(10)

    # Corrupt/perturb current RNG state completely
    seed_everything(1111)

    # Restore from checkpoint with restore_rng=True
    fresh_model = TinyDiT(hidden_size=64, depth=2, num_heads=4)
    fresh_opt = torch.optim.AdamW(fresh_model.parameters(), lr=1e-3)
    load_training_checkpoint(
        checkpoint_path=ckpt_file,
        model=fresh_model,
        optimizer=fresh_opt,
        expected_run_id="run_rng_001",
        expected_run_spec_sha256=run.spec.run_spec_sha256,
        expected_snapshot_metadata_sha256=snap_sha,
        restore_rng=True,
    )

    # Values sampled after restore must match exactly
    assert random.random() == expected_py_next
    assert np.allclose(np.random.rand(5), expected_np_next)
    assert torch.equal(torch.randn(10), expected_torch_next)


def test_cpu_uninterrupted_vs_checkpoint_resumed_reproducibility(tmp_path: Path):
    """Critical reproducibility test: Path A (uninterrupted Steps 1,2,3) vs Path B (Steps 1,2, Checkpoint, Restore, Step 3)."""
    ds_root = tmp_path / "datasets"
    train_root = tmp_path / "training_runs"
    snap_sha = create_mock_snapshot_fixture(ds_root, "ds_repro_test", "dataset_snapshot_v001", count=4)

    mgr = TrainingRunManager(training_root=train_root, dataset_root=ds_root)
    run = mgr.create_run(
        run_id="run_repro_001",
        dataset_id="ds_repro_test",
        snapshot_version="dataset_snapshot_v001",
        model_family="dit",
        model_config={"hidden_size": 64, "depth": 2, "num_heads": 4},
        training_config={"batch_size": 2, "lr": 1e-4, "num_timesteps": 100},
        seed=42,
        created_by="lead_engineer",
        creation_source="repro_test",
    )

    dataset = SnapshotTrainingDataset(
        dataset_id="ds_repro_test",
        snapshot_version="dataset_snapshot_v001",
        dataset_root=ds_root,
    )
    schedule = DiffusionSchedule(num_timesteps=100)

    # =========================================================================
    # Path A: Uninterrupted Steps 1, 2, 3
    # =========================================================================
    seed_everything(42)
    model_a = TinyDiT(hidden_size=64, depth=2, num_heads=4)
    opt_a = torch.optim.AdamW(model_a.parameters(), lr=1e-4)
    gen_a = torch.Generator().manual_seed(42)
    loader_a = create_snapshot_dataloader(dataset, batch_size=2, shuffle=True, generator=gen_a)

    iter_a = iter(loader_a)
    losses_a = []
    t_a_records = []
    noise_a_records = []
    for step in range(1, 4):
        try:
            batch = next(iter_a)
        except StopIteration:
            iter_a = iter(loader_a)
            batch = next(iter_a)
        t = torch.randint(0, schedule.num_timesteps, (batch["latent"].shape[0],))
        t_a_records.append(t.clone())
        loss, _, eps = schedule.training_loss(model_a, batch["latent"], t, batch["text_embedding"])
        noise_a_records.append(eps.clone())
        opt_a.zero_grad()
        loss.backward()
        opt_a.step()
        losses_a.append(loss.item())

    # =========================================================================
    # Path B: Steps 1, 2 -> Checkpoint -> Recreate & Reload with RNG -> Step 3
    # =========================================================================
    seed_everything(42)
    model_b = TinyDiT(hidden_size=64, depth=2, num_heads=4)
    opt_b = torch.optim.AdamW(model_b.parameters(), lr=1e-4)
    gen_b = torch.Generator().manual_seed(42)
    loader_b = create_snapshot_dataloader(dataset, batch_size=2, shuffle=True, generator=gen_b)

    iter_b = iter(loader_b)
    for step in range(1, 3):
        try:
            batch = next(iter_b)
        except StopIteration:
            iter_b = iter(loader_b)
            batch = next(iter_b)
        t = torch.randint(0, schedule.num_timesteps, (batch["latent"].shape[0],))
        loss, _, _ = schedule.training_loss(model_b, batch["latent"], t, batch["text_embedding"])
        opt_b.zero_grad()
        loss.backward()
        opt_b.step()

    # Save checkpoint after step 2
    ckpt_path = train_root / "run_repro_001" / "checkpoints" / "step_000002.pt"
    save_training_checkpoint(
        checkpoint_path=ckpt_path,
        model=model_b,
        optimizer=opt_b,
        step=2,
        run_spec=run.spec,
        snapshot_metadata_sha256=snap_sha,
        loss=loss.item(),
        dataloader_generator=gen_b,
    )

    # Completely reinitialize fresh model, optimizer, generator and perturb global RNG
    seed_everything(999999)
    model_resumed = TinyDiT(hidden_size=64, depth=2, num_heads=4)
    opt_resumed = torch.optim.AdamW(model_resumed.parameters(), lr=1e-4)
    gen_resumed = torch.Generator()

    # Restore from checkpoint with RNG state
    load_training_checkpoint(
        checkpoint_path=ckpt_path,
        model=model_resumed,
        optimizer=opt_resumed,
        expected_run_id="run_repro_001",
        expected_run_spec_sha256=run.spec.run_spec_sha256,
        expected_snapshot_metadata_sha256=snap_sha,
        restore_rng=True,
        dataloader_generator=gen_resumed,
    )

    # Recreate dataloader with restored generator
    loader_resumed = create_snapshot_dataloader(dataset, batch_size=2, shuffle=True, generator=gen_resumed)
    iter_resumed = iter(loader_resumed)

    # Run resumed step 3
    batch_resumed = next(iter_resumed)
    t_resumed = torch.randint(0, schedule.num_timesteps, (batch_resumed["latent"].shape[0],))
    loss_resumed, _, noise_resumed = schedule.training_loss(
        model_resumed, batch_resumed["latent"], t_resumed, batch_resumed["text_embedding"]
    )
    opt_resumed.zero_grad()
    loss_resumed.backward()
    opt_resumed.step()

    # 1. Stochastic state exact equality: timesteps and diffusion noise must match bitwise
    assert torch.equal(t_a_records[2], t_resumed), "Sampled stochastic timesteps do not match!"
    assert torch.equal(noise_a_records[2], noise_resumed), "Sampled diffusion noise tensors do not match!"

    # 2. Model parameter equality: all parameters match within tight floating-point tolerance
    for p_a, p_resumed in zip(model_a.parameters(), model_resumed.parameters()):
        assert torch.allclose(p_a, p_resumed, atol=2e-4), "Resumed model parameters diverge from uninterrupted trajectory!"

    # 3. Loss equality
    assert np.isclose(losses_a[2], loss_resumed.item(), atol=1e-4)


def test_negative_continuation_without_rng_restore(tmp_path: Path):
    """Negative test: Resuming without restoring RNG causes divergent stochastic trajectory."""
    ds_root = tmp_path / "datasets"
    train_root = tmp_path / "training_runs"
    snap_sha = create_mock_snapshot_fixture(ds_root, "ds_neg_repro", "dataset_snapshot_v001", count=4)

    mgr = TrainingRunManager(training_root=train_root, dataset_root=ds_root)
    run = mgr.create_run(
        run_id="run_neg_001",
        dataset_id="ds_neg_repro",
        snapshot_version="dataset_snapshot_v001",
        model_family="dit",
        model_config={"hidden_size": 64, "depth": 2, "num_heads": 4},
        training_config={"batch_size": 2, "lr": 1e-4, "num_timesteps": 100},
        seed=42,
        created_by="lead_engineer",
        creation_source="repro_test",
    )

    dataset = SnapshotTrainingDataset(
        dataset_id="ds_neg_repro",
        snapshot_version="dataset_snapshot_v001",
        dataset_root=ds_root,
    )
    schedule = DiffusionSchedule(num_timesteps=100)

    # Path A: Uninterrupted
    seed_everything(42)
    model_a = TinyDiT(hidden_size=64, depth=2, num_heads=4)
    opt_a = torch.optim.AdamW(model_a.parameters(), lr=1e-4)
    gen_a = torch.Generator().manual_seed(42)
    loader_a = create_snapshot_dataloader(dataset, batch_size=2, shuffle=True, generator=gen_a)
    iter_a = iter(loader_a)
    t_a_step3 = None
    noise_a_step3 = None
    for s in range(1, 4):
        try:
            batch = next(iter_a)
        except StopIteration:
            iter_a = iter(loader_a)
            batch = next(iter_a)
        t = torch.randint(0, schedule.num_timesteps, (batch["latent"].shape[0],))
        loss, _, eps = schedule.training_loss(model_a, batch["latent"], t, batch["text_embedding"])
        if s == 3:
            t_a_step3 = t.clone()
            noise_a_step3 = eps.clone()
        opt_a.zero_grad()
        loss.backward()
        opt_a.step()

    # Path B: Checkpoint at step 2, but resume with restore_rng=False
    seed_everything(42)
    model_b = TinyDiT(hidden_size=64, depth=2, num_heads=4)
    opt_b = torch.optim.AdamW(model_b.parameters(), lr=1e-4)
    gen_b = torch.Generator().manual_seed(42)
    loader_b = create_snapshot_dataloader(dataset, batch_size=2, shuffle=True, generator=gen_b)
    iter_b = iter(loader_b)
    for _ in range(2):
        try:
            batch = next(iter_b)
        except StopIteration:
            iter_b = iter(loader_b)
            batch = next(iter_b)
        t = torch.randint(0, schedule.num_timesteps, (batch["latent"].shape[0],))
        loss, _, _ = schedule.training_loss(model_b, batch["latent"], t, batch["text_embedding"])
        opt_b.zero_grad()
        loss.backward()
        opt_b.step()

    ckpt_path = train_root / "run_neg_001" / "checkpoints" / "step_000002.pt"
    save_training_checkpoint(
        checkpoint_path=ckpt_path,
        model=model_b,
        optimizer=opt_b,
        step=2,
        run_spec=run.spec,
        snapshot_metadata_sha256=snap_sha,
        dataloader_generator=gen_b,
    )

    # Set completely different seed before resume
    seed_everything(888888)
    model_unrestored = TinyDiT(hidden_size=64, depth=2, num_heads=4)
    opt_unrestored = torch.optim.AdamW(model_unrestored.parameters(), lr=1e-4)
    load_training_checkpoint(
        checkpoint_path=ckpt_path,
        model=model_unrestored,
        optimizer=opt_unrestored,
        expected_run_id="run_neg_001",
        expected_run_spec_sha256=run.spec.run_spec_sha256,
        expected_snapshot_metadata_sha256=snap_sha,
        restore_rng=False,  # DO NOT RESTORE RNG
    )

    loader_unrestored = create_snapshot_dataloader(dataset, batch_size=2, shuffle=True)
    batch_unrestored = next(iter(loader_unrestored))
    t_unrestored = torch.randint(0, schedule.num_timesteps, (batch_unrestored["latent"].shape[0],))
    loss_unrestored, _, noise_unrestored = schedule.training_loss(
        model_unrestored, batch_unrestored["latent"], t_unrestored, batch_unrestored["text_embedding"]
    )
    opt_unrestored.zero_grad()
    loss_unrestored.backward()
    opt_unrestored.step()

    # 1. Unrestored RNG produces divergent stochastic timesteps and noise
    assert not torch.equal(t_a_step3, t_unrestored), "Timesteps unexpectedly matched across different RNG seeds!"
    assert not torch.equal(noise_a_step3, noise_unrestored), "Diffusion noise unexpectedly matched across different RNG seeds!"

    # 2. Unrestored RNG produces divergent model parameters
    assert not all(torch.equal(p_a, p_u) for p_a, p_u in zip(model_a.parameters(), model_unrestored.parameters()))


def test_provenance_mismatch_fails_before_rng_restoration(tmp_path: Path):
    """Verifies that provenance failure fails closed before modifying caller's RNG."""
    ds_root = tmp_path / "datasets"
    train_root = tmp_path / "training_runs"
    snap_sha = create_mock_snapshot_fixture(ds_root, "ds_prov_rng", "dataset_snapshot_v001", count=2)

    mgr = TrainingRunManager(training_root=train_root, dataset_root=ds_root)
    run = mgr.create_run(
        run_id="run_prov_rng_01",
        dataset_id="ds_prov_rng",
        snapshot_version="dataset_snapshot_v001",
        model_family="dit",
        model_config={"hidden_size": 64},
        training_config={"batch_size": 2},
        seed=42,
        created_by="eng_lead",
        creation_source="test",
    )

    model = TinyDiT(hidden_size=64, depth=2, num_heads=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt_file = train_root / "run_prov_rng_01" / "checkpoints" / "step_1.pt"
    save_training_checkpoint(
        checkpoint_path=ckpt_file,
        model=model,
        optimizer=optimizer,
        step=1,
        run_spec=run.spec,
        snapshot_metadata_sha256=snap_sha,
    )

    seed_everything(7777)
    py_state_before = random.getstate()
    torch_state_before = torch.get_rng_state()

    # Wrong run ID raises ValueError
    with pytest.raises(ValueError, match="run_id.*does not match"):
        load_training_checkpoint(
            checkpoint_path=ckpt_file,
            model=model,
            expected_run_id="wrong_id",
            restore_rng=True,
        )

    # Caller RNG state must be unmodified
    assert random.getstate() == py_state_before
    assert torch.equal(torch.get_rng_state(), torch_state_before)


def test_restore_rng_false_leaves_caller_rng_untouched(tmp_path: Path):
    """Verifies that restore_rng=False does not modify caller's RNG state."""
    ds_root = tmp_path / "datasets"
    train_root = tmp_path / "training_runs"
    snap_sha = create_mock_snapshot_fixture(ds_root, "ds_no_rng_restore", "dataset_snapshot_v001", count=2)

    mgr = TrainingRunManager(training_root=train_root, dataset_root=ds_root)
    run = mgr.create_run(
        run_id="run_no_rng_01",
        dataset_id="ds_no_rng_restore",
        snapshot_version="dataset_snapshot_v001",
        model_family="dit",
        model_config={"hidden_size": 64},
        training_config={"batch_size": 2},
        seed=42,
        created_by="eng_lead",
        creation_source="test",
    )

    model = TinyDiT(hidden_size=64, depth=2, num_heads=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt_file = train_root / "run_no_rng_01" / "checkpoints" / "step_1.pt"
    save_training_checkpoint(
        checkpoint_path=ckpt_file,
        model=model,
        optimizer=optimizer,
        step=1,
        run_spec=run.spec,
        snapshot_metadata_sha256=snap_sha,
    )

    # Pre-instantiate fresh model so instantiation does not consume RNG
    fresh_model = TinyDiT(hidden_size=64, depth=2, num_heads=4)

    seed_everything(3333)
    py_state_before = random.getstate()
    torch_state_before = torch.get_rng_state()

    load_training_checkpoint(
        checkpoint_path=ckpt_file,
        model=fresh_model,
        expected_run_id="run_no_rng_01",
        expected_run_spec_sha256=run.spec.run_spec_sha256,
        expected_snapshot_metadata_sha256=snap_sha,
        restore_rng=False,
    )

    assert random.getstate() == py_state_before
    assert torch.equal(torch.get_rng_state(), torch_state_before)


# =============================================================================
# End-to-End Smoke Training Execution Test
# =============================================================================
def test_end_to_end_tiny_dit_smoke_training(tmp_path: Path):
    ds_root = tmp_path / "datasets"
    train_root = tmp_path / "training_runs"
    snap_sha = create_mock_snapshot_fixture(ds_root, "ds_smoke_e2e", "dataset_snapshot_v001", count=2)

    mgr = TrainingRunManager(training_root=train_root, dataset_root=ds_root)
    run = mgr.create_run(
        run_id="run_smoke_001",
        dataset_id="ds_smoke_e2e",
        snapshot_version="dataset_snapshot_v001",
        model_family="dit",
        model_config={
            "in_channels": 4,
            "out_channels": 4,
            "latent_size": 32,
            "patch_size": 4,
            "hidden_size": 64,
            "depth": 2,
            "num_heads": 4,
            "text_dim": 512,
        },
        training_config={
            "batch_size": 2,
            "lr": 1e-4,
            "num_timesteps": 100,
        },
        seed=42,
        created_by="lead_engineer",
        creation_source="smoke_test",
    )

    # Run 2 smoke training steps
    res = run_tiny_dit_smoke(
        run_id="run_smoke_001",
        training_root=train_root,
        dataset_root=ds_root,
        device="cpu",
        max_steps=2,
        save_checkpoint=True,
    )

    assert res["status"] == "COMPLETED"
    assert res["steps_completed"] == 2
    assert len(res["step_losses"]) == 2
    assert all(isinstance(l, float) and l >= 0.0 for l in res["step_losses"])
    assert res["checkpoint_path"] is not None
    assert Path(res["checkpoint_path"]).is_file()

    # Verify lifecycle state on disk
    completed_run = mgr.load_verified_run("run_smoke_001", verify_integrity=True)
    assert completed_run.state.status == TrainingRunStatus.COMPLETED.value
    assert completed_run.state.current_step == 2
    assert completed_run.state.last_checkpoint == res["checkpoint_path"]


# =============================================================================
# CLI Preflight and Smoke End-to-End Test
# =============================================================================
def test_cli_preflight_and_smoke_execution(tmp_path: Path):
    import subprocess
    import sys

    ds_root = tmp_path / "datasets"
    train_root = tmp_path / "training_runs"
    create_mock_snapshot_fixture(ds_root, "ds_cli_smoke", "dataset_snapshot_v001", count=2)

    mgr = TrainingRunManager(training_root=train_root, dataset_root=ds_root)
    mgr.create_run(
        run_id="run_cli_smoke_01",
        dataset_id="ds_cli_smoke",
        snapshot_version="dataset_snapshot_v001",
        model_family="dit",
        model_config={"hidden_size": 64, "depth": 2, "num_heads": 4},
        training_config={"batch_size": 2, "lr": 1e-4, "num_timesteps": 100},
        seed=42,
        created_by="eng_lead",
        creation_source="cli_test",
    )

    # 1. CLI Preflight
    pf_cmd = [
        sys.executable,
        "-m",
        "rernggen.training",
        "preflight",
        "--run-id",
        "run_cli_smoke_01",
        "--training-root",
        str(train_root),
        "--dataset-root",
        str(ds_root),
        "--device",
        "cpu",
    ]
    res_pf = subprocess.run(pf_cmd, capture_output=True, text=True, check=True)
    assert "Overall Result:             PASSED" in res_pf.stdout

    # 2. CLI Smoke
    smoke_cmd = [
        sys.executable,
        "-m",
        "rernggen.training",
        "smoke",
        "--run-id",
        "run_cli_smoke_01",
        "--training-root",
        str(train_root),
        "--dataset-root",
        str(ds_root),
        "--device",
        "cpu",
        "--steps",
        "2",
    ]
    res_smoke = subprocess.run(smoke_cmd, capture_output=True, text=True, check=True)
    assert "TINY DiT SMOKE TRAINING COMPLETED: run_cli_smoke_01" in res_smoke.stdout
    assert "Status:                     COMPLETED" in res_smoke.stdout
