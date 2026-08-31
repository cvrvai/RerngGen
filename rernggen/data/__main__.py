"""CLI entrypoint for RerngGen dataset operations.

Usage:
    python -m rernggen.data import --source "C:\\path\\to\\images" --dataset-id "khmer_story_cartoon_v001"
    python -m rernggen.data preprocess --dataset-id "khmer_story_cartoon_v001" --target-size 256
    python -m rernggen.data cache-latents --dataset-id "khmer_story_cartoon_v001" --preprocessing-version "square256_center_v001"
"""

import argparse
import sys
import torch
from rernggen.data.importer import DatasetImporter
from rernggen.data.latent_cache import LatentCacheGenerator
from rernggen.data.preprocessor import ImagePreprocessor
from rernggen.models.vae.interface import AutoencoderKLAdapter


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

    # Subcommand: preprocess
    prep_parser = subparsers.add_parser(
        "preprocess",
        help="Preprocess dataset originals into standardized derivatives (e.g. 256x256 square crops).",
    )
    prep_parser.add_argument(
        "--dataset-id",
        "-d",
        type=str,
        default="khmer_story_cartoon_v001",
        help="Dataset ID to preprocess (default: 'khmer_story_cartoon_v001').",
    )
    prep_parser.add_argument(
        "--target-size",
        "-s",
        type=int,
        default=256,
        help="Target square resolution (default: 256).",
    )
    prep_parser.add_argument(
        "--version",
        "-v",
        type=str,
        default="square256_center_v001",
        help="Version identifier for this derivative (default: 'square256_center_v001').",
    )
    prep_parser.add_argument(
        "--dataset-root",
        "-r",
        type=str,
        default="datasets",
        help="Root directory for dataset repositories (default: 'datasets').",
    )
    prep_parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-processing of images even if already present.",
    )

    # Subcommand: cache-latents
    cache_parser = subparsers.add_parser(
        "cache-latents",
        help="Extract and persist frozen VAE model latents [4, 32, 32] as permanent safetensors.",
    )
    cache_parser.add_argument(
        "--dataset-id",
        "-d",
        type=str,
        default="khmer_story_cartoon_v001",
        help="Dataset ID (default: 'khmer_story_cartoon_v001').",
    )
    cache_parser.add_argument(
        "--preprocessing-version",
        "-p",
        type=str,
        default="square256_center_v001",
        help="Preprocessing derivative version (default: 'square256_center_v001').",
    )
    cache_parser.add_argument(
        "--cache-version",
        "-c",
        type=str,
        default="vae_sd_mse_square256_v001",
        help="Cache version identifier (default: 'vae_sd_mse_square256_v001').",
    )
    cache_parser.add_argument(
        "--model-path",
        "-m",
        type=str,
        default="models/vae/stabilityai--sd-vae-ft-mse",
        help="Local path or HuggingFace ID of AutoencoderKL VAE.",
    )
    cache_parser.add_argument(
        "--dataset-root",
        "-r",
        type=str,
        default="datasets",
        help="Root directory for dataset repositories (default: 'datasets').",
    )
    cache_parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (cuda or cpu).",
    )
    cache_parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-encoding even if already cached.",
    )

    # Subcommand: cache-text-embeds
    text_cache_parser = subparsers.add_parser(
        "cache-text-embeds",
        help="Extract and persist frozen text encoder pooled embeddings [512] as permanent safetensors.",
    )
    text_cache_parser.add_argument(
        "--dataset-id",
        "-d",
        type=str,
        default="khmer_story_cartoon_v001",
        help="Dataset ID (default: 'khmer_story_cartoon_v001').",
    )
    text_cache_parser.add_argument(
        "--caption-version",
        "-c",
        type=str,
        default="captions_v001",
        help="Caption version identifier (default: 'captions_v001').",
    )
    text_cache_parser.add_argument(
        "--cache-version",
        "-v",
        type=str,
        default="clip_b32_v001",
        help="Text embedding cache version identifier (default: 'clip_b32_v001').",
    )
    text_cache_parser.add_argument(
        "--model-path",
        "-m",
        type=str,
        default="models/text_encoder/openai--clip-text-base-patch32",
        help="Local path or HuggingFace ID of CLIP text encoder.",
    )
    text_cache_parser.add_argument(
        "--dataset-root",
        "-r",
        type=str,
        default="datasets",
        help="Root directory for dataset repositories (default: 'datasets').",
    )
    text_cache_parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (cuda or cpu).",
    )
    text_cache_parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-encoding even if already cached.",
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

    elif args.command == "preprocess":
        preprocessor = ImagePreprocessor(
            target_size=args.target_size,
            version=args.version,
            dataset_root=args.dataset_root,
        )
        prep_report = preprocessor.process_dataset(
            dataset_id=args.dataset_id,
            force=args.force,
        )
        print(prep_report.summary())
        if prep_report.failures > 0:
            print(f"\n[WARNING] Encountered {prep_report.failures} processing failure(s).")

    elif args.command == "cache-latents":
        device = torch.device(args.device)
        print(f"Loading VAE from {args.model_path} onto {device}...")
        adapter = AutoencoderKLAdapter.from_pretrained(
            model_id_or_path=args.model_path,
            device=device,
        )
        generator = LatentCacheGenerator(
            vae_adapter=adapter,
            cache_version=args.cache_version,
            dataset_root=args.dataset_root,
        )
        report = generator.generate_cache(
            dataset_id=args.dataset_id,
            preprocessing_version=args.preprocessing_version,
            force=args.force,
            device=device,
        )
        print(report.summary())
        if report.failures > 0:
            print(f"\n[WARNING] Encountered {report.failures} latent encoding failure(s).")

    elif args.command == "cache-text-embeds":
        from rernggen.data.text_cache import TextEmbeddingCacheGenerator
        from rernggen.models.text.interface import CLIPTextEncoderAdapter

        device = torch.device(args.device)
        print(f"Loading CLIP text encoder from {args.model_path} onto {device}...")
        adapter = CLIPTextEncoderAdapter.from_pretrained(
            model_id_or_path=args.model_path,
            device=device,
        )
        generator = TextEmbeddingCacheGenerator(
            text_encoder_adapter=adapter,
            cache_version=args.cache_version,
            dataset_root=args.dataset_root,
        )
        report = generator.generate_cache(
            dataset_id=args.dataset_id,
            caption_version=args.caption_version,
            force=args.force,
            device=device,
        )
        print(report.summary())
        if report.failures > 0:
            print(f"\n[WARNING] Encountered {report.failures} text embedding failure(s).")


if __name__ == "__main__":
    main()
