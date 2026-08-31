"""Comprehensive unit and integration tests for Step 17 Image Preprocessing."""

import json
from pathlib import Path
from PIL import Image
import pytest
from rernggen.data.importer import DatasetImporter, compute_sha256
from rernggen.data.preprocessor import (
    ImagePreprocessor,
    preprocess_dataset,
    preprocess_image_to_square,
)


def create_test_image_with_mode(
    path: Path,
    size=(600, 600),
    mode="RGB",
    color=(255, 0, 0),
    fmt="PNG",
) -> None:
    """Helper to create synthetic images with varied modes and sizes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new(mode, size, color=color)
    img.save(path, format=fmt)


def test_preprocess_image_square_aspect_ratio_and_coordinates():
    """Verify square image transformation math and center crop coordinates."""
    img = Image.new("RGB", (600, 600), color=(100, 150, 200))
    proc_img, meta = preprocess_image_to_square(img, target_size=256)

    assert proc_img.size == (256, 256)
    assert proc_img.mode == "RGB"
    assert meta["original_width"] == 600
    assert meta["original_height"] == 600
    assert meta["resized_width"] == 256
    assert meta["resized_height"] == 256
    assert meta["crop_left"] == 0
    assert meta["crop_top"] == 0
    assert meta["crop_width"] == 256
    assert meta["crop_height"] == 256


def test_preprocess_image_portrait_no_stretching():
    """Verify portrait image (e.g. 596x825) is scaled by shortest side, NOT stretched."""
    img = Image.new("RGB", (596, 825), color=(50, 100, 150))
    proc_img, meta = preprocess_image_to_square(img, target_size=256)

    assert proc_img.size == (256, 256)
    assert proc_img.mode == "RGB"

    # Original aspect ratio: 825 / 596 = 1.3842
    orig_aspect = 825 / 596.0

    # Shortest side (width 596) must become exactly 256
    assert meta["resized_width"] == 256
    expected_resized_height = int(round(825 * (256.0 / 596.0)))  # ~354
    assert meta["resized_height"] == expected_resized_height

    # Resized aspect ratio must match original aspect ratio within pixel rounding
    resized_aspect = meta["resized_height"] / float(meta["resized_width"])
    assert abs(orig_aspect - resized_aspect) < 0.01, "Aspect ratio was distorted/stretched!"

    # Crop top must be centered: (354 - 256) // 2 = 49
    expected_crop_top = (expected_resized_height - 256) // 2
    assert meta["crop_top"] == expected_crop_top
    assert meta["crop_left"] == 0


def test_preprocess_image_landscape_no_stretching():
    """Verify landscape image (e.g. 1000x500) is scaled by shortest side, NOT stretched."""
    img = Image.new("RGB", (1000, 500), color=(200, 100, 50))
    proc_img, meta = preprocess_image_to_square(img, target_size=256)

    assert proc_img.size == (256, 256)

    # Shortest side (height 500) must become 256
    assert meta["resized_height"] == 256
    assert meta["resized_width"] == 512

    # Crop left must be centered: (512 - 256) // 2 = 128
    assert meta["crop_left"] == 128
    assert meta["crop_top"] == 0


def test_preprocess_color_modes_rgba_grayscale_cmyk():
    """Verify color conversion to RGB for RGBA, Grayscale (L), and CMYK images."""
    # 1. RGBA with alpha transparency
    rgba_img = Image.new("RGBA", (300, 300), color=(255, 0, 0, 128))
    proc_rgba, _ = preprocess_image_to_square(rgba_img, target_size=256)
    assert proc_rgba.mode == "RGB"
    assert proc_rgba.size == (256, 256)

    # 2. Grayscale (L)
    l_img = Image.new("L", (400, 400), color=128)
    proc_l, _ = preprocess_image_to_square(l_img, target_size=256)
    assert proc_l.mode == "RGB"

    # 3. CMYK
    cmyk_img = Image.new("CMYK", (400, 400), color=(0, 100, 100, 0))
    proc_cmyk, _ = preprocess_image_to_square(cmyk_img, target_size=256)
    assert proc_cmyk.mode == "RGB"


def test_preprocess_image_exif_orientation():
    """Verify EXIF orientation tag (e.g. 6: Rotate 90 CW) is normalized before geometry calculation."""
    img = Image.new("RGB", (400, 200), color=(120, 80, 40))
    exif = img.getexif()
    exif[0x0112] = 6  # Orientation tag 6: Rotated 90 CW (transposed to 200x400 portrait)

    proc_img, meta = preprocess_image_to_square(img, target_size=256)

    assert proc_img.size == (256, 256)
    # Effective dimensions after EXIF transpose: 200 width x 400 height
    assert meta["original_width"] == 200
    assert meta["original_height"] == 400
    assert meta["resized_width"] == 256
    assert meta["resized_height"] == 512
    assert meta["crop_top"] == 128
    assert meta["crop_left"] == 0


def test_preprocessor_end_to_end_dataset(tmp_path: Path):
    """Verifies end-to-end dataset preprocessing, metadata serialization, and idempotency."""
    source_dir = tmp_path / "raw_source"
    dataset_root = tmp_path / "datasets"
    dataset_id = "test_khmer_v001"

    # Create 3 diverse source images
    create_test_image_with_mode(source_dir / "sq.jpg", size=(600, 600), fmt="JPEG")
    create_test_image_with_mode(source_dir / "portrait.webp", size=(596, 825), fmt="WEBP")
    create_test_image_with_mode(source_dir / "landscape.png", size=(800, 400), fmt="PNG")

    # Step 16: Import dataset
    importer = DatasetImporter(dataset_id=dataset_id, dataset_root=dataset_root)
    import_report = importer.import_directory(source_dir)
    assert import_report.imported_images == 3

    # Record original file hashes before preprocessing to verify immutability
    originals_dir = dataset_root / dataset_id / "originals"
    original_hashes_before = {p: compute_sha256(p) for p in originals_dir.glob("IMG-*")}

    # Step 17: Preprocess dataset
    preprocessor = ImagePreprocessor(
        target_size=256,
        version="square256_center_v001",
        dataset_root=dataset_root,
    )
    report1 = preprocessor.process_dataset(dataset_id=dataset_id)

    assert report1.total_images_in_dataset == 3
    assert report1.images_processed == 3
    assert report1.images_skipped_idempotent == 0
    assert report1.failures == 0
    assert report1.total_output_bytes > 0
    assert "DATASET PREPROCESSING COMPLETE" in report1.summary()

    # Verify original files remain 100% untouched
    original_hashes_after = {p: compute_sha256(p) for p in originals_dir.glob("IMG-*")}
    assert original_hashes_before == original_hashes_after, "Originals were modified by preprocessor!"

    # Verify processed output directory
    processed_dir = dataset_root / dataset_id / "processed" / "square256_center_v001"
    processed_manifest = processed_dir / "manifest.jsonl"
    assert processed_dir.exists()
    assert processed_manifest.exists()

    # Verify processed files on disk
    processed_files = list(processed_dir.glob("IMG-*.png"))
    assert len(processed_files) == 3

    # Verify processed manifest records and exact schema
    records = []
    with open(processed_manifest, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    assert len(records) == 3
    for rec in records:
        assert rec["dataset_id"] == dataset_id
        assert rec["preprocessing_version"] == "square256_center_v001"
        assert rec["output_width"] == 256
        assert rec["output_height"] == 256
        assert rec["output_mode"] == "RGB"
        assert rec["training_allowed"] is None
        assert rec["commercial_allowed"] is None
        assert rec["license_id"] is None
        assert rec["status"] == "PROCESSED"

        # Verify output file exists and matches recorded sha256
        out_file = dataset_root / dataset_id / rec["output_relative_path"]
        assert out_file.exists()
        assert compute_sha256(out_file) == rec["processed_sha256"]

        # Verify decoded dimensions from disk
        with Image.open(out_file) as img:
            assert img.size == (256, 256)
            assert img.mode == "RGB"

    # Step 17 Idempotency Test: re-run should skip all 3 images with 0 reprocessing
    report2 = preprocessor.process_dataset(dataset_id=dataset_id)
    assert report2.images_processed == 0
    assert report2.images_skipped_idempotent == 3
    assert report2.failures == 0


def test_preprocess_convenience_function(tmp_path: Path):
    """Verify preprocess_dataset convenience function."""
    source_dir = tmp_path / "raw"
    dataset_root = tmp_path / "ds_root"
    dataset_id = "test_helper_v001"

    create_test_image_with_mode(source_dir / "sample.jpg", size=(500, 500), fmt="JPEG")
    importer = DatasetImporter(dataset_id=dataset_id, dataset_root=dataset_root)
    importer.import_directory(source_dir)

    report = preprocess_dataset(
        dataset_id=dataset_id,
        target_size=256,
        dataset_root=dataset_root,
    )
    assert report.images_processed == 1
    assert (dataset_root / dataset_id / "processed" / "square256_center_v001" / "IMG-000001.png").exists()
