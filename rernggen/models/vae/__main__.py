"""CLI entrypoint for VAE reconstruction validation.

Usage:
    python -m rernggen.models.vae validate --dataset-id "khmer_story_cartoon_v001" --preprocessing-version "square256_center_v001"
"""

import argparse
from pathlib import Path
import torch
from rernggen.models.vae.interface import AutoencoderKLAdapter
from rernggen.models.vae.validator import ReconstructionValidator


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rernggen.models.vae",
        description="RerngGen VAE Reconstruction & Validation CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    val_parser = subparsers.add_parser(
        "validate",
        help="Run frozen VAE encode/decode reconstruction validation over processed dataset images.",
    )
    val_parser.add_argument(
        "--dataset-id",
        "-d",
        type=str,
        default="khmer_story_cartoon_v001",
        help="Dataset ID to validate (default: 'khmer_story_cartoon_v001').",
    )
    val_parser.add_argument(
        "--preprocessing-version",
        "-v",
        type=str,
        default="square256_center_v001",
        help="Preprocessing derivative version (default: 'square256_center_v001').",
    )
    val_parser.add_argument(
        "--model-path",
        "-m",
        type=str,
        default="models/vae/stabilityai--sd-vae-ft-mse",
        help="Local path or HuggingFace model ID for AutoencoderKL VAE.",
    )
    val_parser.add_argument(
        "--dataset-root",
        "-r",
        type=str,
        default="datasets",
        help="Root directory for dataset repositories (default: 'datasets').",
    )
    val_parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for computation (cuda or cpu).",
    )

    args = parser.parse_args()

    if args.command == "validate":
        device = torch.device(args.device)
        print(f"Loading VAE adapter from {args.model_path} onto {device}...")
        adapter = AutoencoderKLAdapter.from_pretrained(
            model_id_or_path=args.model_path,
            device=device,
        )
        validator = ReconstructionValidator(
            vae_adapter=adapter,
            dataset_root=args.dataset_root,
        )
        report = validator.validate_dataset(
            dataset_id=args.dataset_id,
            preprocessing_version=args.preprocessing_version,
            device=device,
        )
        print(report.summary())


if __name__ == "__main__":
    main()
