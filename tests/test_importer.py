"""Comprehensive unit and integration tests for Step 16 Dataset Importer."""

import json
import os
from pathlib import Path
import subprocess
import sys
from PIL import Image
import pytest
from rernggen.data.importer import DatasetImporter, compute_sha256, import_image_directory


def create_test_image(path: Path, fmt: str = "PNG", size=(64, 64), color=(255, 0, 0)) -> None:
    """Helper to create a valid synthetic image file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=color)
    img.save(path, format=fmt)


def test_importer_comprehensive_suite(tmp_path: Path):
    """Verifies all 17 contract requirements of the dataset importer."""
    source_dir = tmp_path / "source_photos with spaces"
    dataset_root = tmp_path / "target_datasets"
    dataset_id = "khmer_story_cartoon_v001"

    # -------------------------------------------------------------
    # 1. SETUP DIVERSE TEST FILES IN NESTED & UNICODE DIRECTORIES
    # -------------------------------------------------------------
    # 1. PNG in root
    png_file = source_dir / "image_01.png"
    create_test_image(png_file, fmt="PNG", color=(255, 0, 0))

    # 2. JPEG in nested subfolder with spaces
    jpg_file = source_dir / "nested sub folder" / "image_02.jpg"
    create_test_image(jpg_file, fmt="JPEG", color=(0, 255, 0))

    # 3. WEBP in deep folder
    webp_file = source_dir / "a" / "b" / "image_03.webp"
    create_test_image(webp_file, fmt="WEBP", color=(0, 0, 255))

    # 4. Unicode filename (Khmer title)
    unicode_file = source_dir / "រឿង_និទាន_khmer.png"
    create_test_image(unicode_file, fmt="PNG", color=(128, 128, 0))

    # 5. Exact content duplicate in another location (different name)
    duplicate_file = source_dir / "nested sub folder" / "duplicate_of_01.png"
    with open(png_file, "rb") as src, open(duplicate_file, "wb") as dst:
        dst.write(src.read())

    # 6. Same filename as image_01.png but in subfolder with DIFFERENT content
    same_name_diff_content = source_dir / "nested sub folder" / "image_01.png"
    create_test_image(same_name_diff_content, fmt="PNG", color=(100, 200, 50))

    # 7. Corrupt image file (.jpg extension with garbage text bytes)
    corrupt_file = source_dir / "corrupted_file.jpg"
    with open(corrupt_file, "wb") as f:
        f.write(b"NOT_A_VALID_JPEG_HEADER_CORRUPTED_STREAM")

    # 8. Unsupported non-image files (.txt, .json)
    unsupported_txt = source_dir / "readme.txt"
    with open(unsupported_txt, "w", encoding="utf-8") as f:
        f.write("Some text metadata")
    unsupported_json = source_dir / "metadata.json"
    with open(unsupported_json, "w", encoding="utf-8") as f:
        f.write('{"key": "value"}')

    # Record source snapshot before import to prove immutability
    source_snapshot_before = {}
    for p in source_dir.rglob("*"):
        if p.is_file():
            source_snapshot_before[p] = (p.stat().st_size, compute_sha256(p))

    # -------------------------------------------------------------
    # 2. EXECUTE FIRST IMPORT
    # -------------------------------------------------------------
    importer = DatasetImporter(dataset_id=dataset_id, dataset_root=dataset_root)
    report1 = importer.import_directory(source_dir)

    # -------------------------------------------------------------
    # 3. VERIFY METRICS & REPORT
    # -------------------------------------------------------------
    assert report1.files_discovered == 9
    assert report1.supported_candidates == 7  # 7 image extensions
    assert report1.valid_images == 6         # 7 candidate files minus 1 corrupt = 6 valid
    assert report1.imported_images == 5       # 6 valid minus 1 duplicate = 5 imported
    assert report1.duplicate_images_skipped == 1
    assert report1.corrupt_images == 1
    assert report1.unsupported_files == 2     # txt and json
    assert report1.total_bytes_copied > 0
    assert "DATASET IMPORT COMPLETE" in report1.summary()

    # -------------------------------------------------------------
    # 4. VERIFY SOURCE IMMUTABILITY
    # -------------------------------------------------------------
    source_snapshot_after = {}
    for p in source_dir.rglob("*"):
        if p.is_file():
            source_snapshot_after[p] = (p.stat().st_size, compute_sha256(p))

    assert source_snapshot_before == source_snapshot_after, "Source directory files were mutated!"

    # -------------------------------------------------------------
    # 5. VERIFY DIRECTORY STRUCTURE & MANIFEST SCHEMA
    # -------------------------------------------------------------
    target_dir = dataset_root / dataset_id
    originals_dir = target_dir / "originals"
    manifest_file = target_dir / "manifests" / "manifest.jsonl"

    assert (target_dir / "processed").exists()
    assert (target_dir / "captions").exists()
    assert (target_dir / "cache").exists()
    assert originals_dir.exists()
    assert manifest_file.exists()

    # Check imported originals count
    imported_files = list(originals_dir.glob("IMG-*"))
    assert len(imported_files) == 5

    # Check manifest records
    records = []
    with open(manifest_file, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            records.append(rec)

    assert len(records) == 5

    # Validate Schema Fields
    for idx, rec in enumerate(records, start=1):
        expected_id = f"IMG-{idx:06d}"
        assert rec["image_id"] == expected_id
        assert rec["dataset_id"] == dataset_id
        assert rec["dataset_version"] == "v001"
        assert rec["width"] == 64
        assert rec["height"] == 64
        assert rec["format"] in ["PNG", "JPEG", "WEBP"]
        assert rec["mode"] == "RGB"
        assert rec["caption"] is None
        assert rec["tags"] == []
        assert rec["source"] == "local_owned_dataset"
        assert rec["license_id"] is None
        assert rec["training_allowed"] is None
        assert rec["commercial_allowed"] is None
        assert rec["status"] == "IMPORTED"

        # Verify exact byte equality of copied original
        stored_path = target_dir / rec["stored_relative_path"]
        assert stored_path.exists()
        assert compute_sha256(stored_path) == rec["sha256"]

    # -------------------------------------------------------------
    # 6. IDEMPOTENCY & SECOND IMPORT TEST
    # -------------------------------------------------------------
    report2 = importer.import_directory(source_dir)

    assert report2.imported_images == 0, "Second import must import 0 new images."
    assert report2.duplicate_images_skipped == 6, (
        "All valid images must be recognized as existing duplicates on 2nd run."
    )
    assert report2.corrupt_images == 1
    assert report2.unsupported_files == 2

    # Manifest should remain identical with 5 records and same stable IDs
    with open(manifest_file, "r", encoding="utf-8") as f:
        records_after = [json.loads(line) for line in f]
    assert records == records_after, "Manifest records mutated on idempotent re-run!"


def test_importer_convenience_function(tmp_path: Path):
    """Verify import_image_directory functional helper API."""
    source_dir = tmp_path / "src"
    dataset_root = tmp_path / "data_root"
    create_test_image(source_dir / "sample.png")

    report = import_image_directory(
        source_dir=source_dir,
        dataset_id="test_ds_v001",
        dataset_root=dataset_root,
    )
    assert report.imported_images == 1
    assert (dataset_root / "test_ds_v001" / "originals" / "IMG-000001.png").exists()


def test_importer_nonexistent_directory_error(tmp_path: Path):
    """Verify FileNotFoundError on non-existent source directory."""
    importer = DatasetImporter(dataset_root=tmp_path)
    with pytest.raises(FileNotFoundError, match="Source directory does not exist"):
        importer.import_directory(tmp_path / "does_not_exist")


def test_importer_cli_execution(tmp_path: Path):
    """Verify that CLI `python -m rernggen.data import ...` executes properly."""
    source_dir = tmp_path / "cli_source"
    dataset_root = tmp_path / "cli_datasets"
    create_test_image(source_dir / "cli_img.jpg", fmt="JPEG")

    cmd = [
        sys.executable,
        "-m",
        "rernggen.data",
        "import",
        "--source",
        str(source_dir),
        "--dataset-id",
        "cli_dataset_v001",
        "--dataset-root",
        str(dataset_root),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "DATASET IMPORT COMPLETE" in result.stdout
    assert (dataset_root / "cli_dataset_v001" / "manifests" / "manifest.jsonl").exists()
