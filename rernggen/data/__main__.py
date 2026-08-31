"""CLI entrypoint for RerngGen dataset operations.

Usage:
    python -m rernggen.data import --source "C:\\path\\to\\images" --dataset-id "khmer_story_cartoon_v001"
"""

import argparse
import sys
from rernggen.data.importer import DatasetImporter


def main() -> None:
    """CLI parser and dispatcher."""
    parser = argparse.ArgumentParser(
        prog="rernggen.data",
        description="RerngGen Dataset Ingestion & Management CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: import
    import_parser = subparsers.add_parser(
        "import",
        help="Import images from a source directory into a standardized RerngGen dataset.",
    )
    import_parser.add_argument(
        "--source",
        "-s",
        type=str,
        required=True,
        help="Source directory containing images to import.",
    )
    import_parser.add_argument(
        "--dataset-id",
        "-d",
        type=str,
        default="khmer_story_cartoon_v001",
        help="Target dataset ID (default: 'khmer_story_cartoon_v001').",
    )
    import_parser.add_argument(
        "--dataset-root",
        "-r",
        type=str,
        default="datasets",
        help="Root directory for dataset repositories (default: 'datasets').",
    )

    args = parser.parse_args()

    if args.command == "import":
        importer = DatasetImporter(
            dataset_id=args.dataset_id,
            dataset_root=args.dataset_root,
        )
        report = importer.import_directory(source_dir=args.source)
        print(report.summary())
        if report.duplicate_images_skipped > 0:
            print(f"\n[INFO] Skipped {report.duplicate_images_skipped} duplicate file(s).")
        if report.corrupt_images > 0:
            print(f"\n[WARNING] Skipped {report.corrupt_images} corrupt file(s).")
        if report.unsupported_files > 0:
            print(f"\n[INFO] Skipped {report.unsupported_files} unsupported non-image file(s).")


if __name__ == "__main__":
    main()
