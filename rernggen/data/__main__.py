"""CLI entrypoint for RerngGen dataset operations.

Usage:
    python -m rernggen.data import --source "C:\\path\\to\\images" --dataset-id "khmer_story_cartoon_v001"
    python -m rernggen.data preprocess --dataset-id "khmer_story_cartoon_v001" --target-size 256
    python -m rernggen.data cache-latents --dataset-id "khmer_story_cartoon_v001" --preprocessing-version "square256_center_v001"
"""

import argparse
from pathlib import Path
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

    # Subcommand: governance
    gov_parser = subparsers.add_parser(
        "governance",
        help="Manage and audit explicit versioned dataset authorization rights.",
    )
    gov_subparsers = gov_parser.add_subparsers(dest="gov_action", required=True)

    # Subcommand: governance authorize
    auth_parser = gov_subparsers.add_parser(
        "authorize",
        help="Record explicit human authorization decision (ALLOW, DENY, UNKNOWN) for dataset items.",
    )
    auth_parser.add_argument(
        "--dataset-id",
        "-d",
        type=str,
        default="khmer_story_cartoon_v001",
        help="Dataset ID (default: 'khmer_story_cartoon_v001').",
    )
    auth_parser.add_argument(
        "--governance-version",
        "-v",
        type=str,
        required=True,
        help="Target governance version identifier (e.g. 'rights_v001').",
    )
    auth_parser.add_argument(
        "--image-id",
        "-i",
        type=str,
        nargs="*",
        default=[],
        help="Specific image ID(s) to authorize (e.g. --image-id IMG-000001 IMG-000002).",
    )
    auth_parser.add_argument(
        "--all",
        action="store_true",
        help="Explicitly apply authorization to ALL verified images in the dataset.",
    )
    auth_parser.add_argument(
        "--training",
        type=str,
        required=True,
        choices=["ALLOW", "DENY", "UNKNOWN"],
        help="Explicit tri-state training permission decision (ALLOW, DENY, UNKNOWN).",
    )
    auth_parser.add_argument(
        "--commercial",
        type=str,
        required=True,
        choices=["ALLOW", "DENY", "UNKNOWN"],
        help="Explicit tri-state commercial permission decision (ALLOW, DENY, UNKNOWN).",
    )
    auth_parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Mandatory authorization source identifier (e.g. 'human_audit', 'license_verified').",
    )
    auth_parser.add_argument(
        "--note",
        type=str,
        default="",
        help="Human audit rationale / authorization note.",
    )
    auth_parser.add_argument(
        "--evidence-ref",
        type=str,
        default=None,
        help="Optional evidence / license document reference.",
    )
    auth_parser.add_argument(
        "--license-id",
        type=str,
        default=None,
        help="Optional license identifier (e.g. CC0, Custom-Proprietary).",
    )
    auth_parser.add_argument(
        "--base-version",
        type=str,
        default=None,
        help="Optional base governance version to clone and update from.",
    )
    auth_parser.add_argument(
        "--dataset-root",
        "-r",
        type=str,
        default="datasets",
        help="Root directory for datasets (default: 'datasets').",
    )

    # Subcommand: governance list-versions
    list_gov_parser = gov_subparsers.add_parser(
        "list-versions",
        help="List all existing governance versions for a dataset.",
    )
    list_gov_parser.add_argument(
        "--dataset-id",
        "-d",
        type=str,
        default="khmer_story_cartoon_v001",
        help="Dataset ID (default: 'khmer_story_cartoon_v001').",
    )
    list_gov_parser.add_argument(
        "--dataset-root",
        "-r",
        type=str,
        default="datasets",
        help="Root directory for datasets (default: 'datasets').",
    )

    # Subcommand: governance show
    show_gov_parser = gov_subparsers.add_parser(
        "show",
        help="Display all governance records and tri-state decision summaries for a version.",
    )
    show_gov_parser.add_argument(
        "--dataset-id",
        "-d",
        type=str,
        default="khmer_story_cartoon_v001",
        help="Dataset ID (default: 'khmer_story_cartoon_v001').",
    )
    show_gov_parser.add_argument(
        "--governance-version",
        "-v",
        type=str,
        required=True,
        help="Governance version to display (e.g. 'rights_v001').",
    )
    show_gov_parser.add_argument(
        "--dataset-root",
        "-r",
        type=str,
        default="datasets",
        help="Root directory for datasets (default: 'datasets').",
    )

    # Subcommand: review-captions
    rev_parser = subparsers.add_parser(
        "review-captions",
        help="Manage and audit explicit versioned human caption reviews and quality acceptance.",
    )
    rev_subparsers = rev_parser.add_subparsers(dest="rev_action", required=True)

    # Subcommand: review-captions review
    rec_rev_parser = rev_subparsers.add_parser(
        "review",
        help="Record explicit human review decision (APPROVED, REJECTED, INVALIDATED, PENDING) for captions.",
    )
    rec_rev_parser.add_argument(
        "--dataset-id",
        "-d",
        type=str,
        default="khmer_story_cartoon_v001",
        help="Dataset ID (default: 'khmer_story_cartoon_v001').",
    )
    rec_rev_parser.add_argument(
        "--review-version",
        "-v",
        type=str,
        required=True,
        help="Target caption review version identifier (e.g. 'caption_review_v001').",
    )
    rec_rev_parser.add_argument(
        "--image-id",
        "-i",
        type=str,
        nargs="*",
        default=[],
        help="Specific image ID(s) to review (e.g. --image-id IMG-000001 IMG-000002).",
    )
    rec_rev_parser.add_argument(
        "--all",
        action="store_true",
        help="Explicitly apply review decision to ALL verified images in the dataset.",
    )
    rec_rev_parser.add_argument(
        "--status",
        type=str,
        required=True,
        choices=["APPROVED", "REJECTED", "INVALIDATED", "PENDING"],
        help="Explicit review decision (APPROVED, REJECTED, INVALIDATED, PENDING).",
    )
    rec_rev_parser.add_argument(
        "--reviewer",
        type=str,
        required=True,
        help="Mandatory human reviewer identifier (e.g. 'reviewer_alice').",
    )
    rec_rev_parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Mandatory review origin / provenance source (e.g. 'human_audit', 'batch_01').",
    )
    rec_rev_parser.add_argument(
        "--caption-version",
        type=str,
        default="captions_v002",
        help="Target caption version to bind review hashes against (default: 'captions_v002').",
    )
    rec_rev_parser.add_argument(
        "--reason",
        type=str,
        default="",
        help="Audit rationale / review notes / rejection reason.",
    )
    rec_rev_parser.add_argument(
        "--base-version",
        type=str,
        default=None,
        help="Optional base review version to clone and update from.",
    )
    rec_rev_parser.add_argument(
        "--dataset-root",
        "-r",
        type=str,
        default="datasets",
        help="Root directory for datasets (default: 'datasets').",
    )

    # Subcommand: review-captions list-versions
    list_rev_parser = rev_subparsers.add_parser(
        "list-versions",
        help="List all existing caption review versions for a dataset.",
    )
    list_rev_parser.add_argument(
        "--dataset-id",
        "-d",
        type=str,
        default="khmer_story_cartoon_v001",
        help="Dataset ID (default: 'khmer_story_cartoon_v001').",
    )
    list_rev_parser.add_argument(
        "--dataset-root",
        "-r",
        type=str,
        default="datasets",
        help="Root directory for datasets (default: 'datasets').",
    )

    # Subcommand: review-captions show
    show_rev_parser = rev_subparsers.add_parser(
        "show",
        help="Display all caption review records and decisions for a version.",
    )
    show_rev_parser.add_argument(
        "--dataset-id",
        "-d",
        type=str,
        default="khmer_story_cartoon_v001",
        help="Dataset ID (default: 'khmer_story_cartoon_v001').",
    )
    show_rev_parser.add_argument(
        "--review-version",
        "-v",
        type=str,
        required=True,
        help="Caption review version to display (e.g. 'caption_review_v001').",
    )
    show_rev_parser.add_argument(
        "--dataset-root",
        "-r",
        type=str,
        default="datasets",
        help="Root directory for datasets (default: 'datasets').",
    )

    # Subcommand: eligibility
    elig_parser = subparsers.add_parser(
        "eligibility",
        help="Authoritative training eligibility admission audit and evaluation.",
    )
    elig_subparsers = elig_parser.add_subparsers(dest="elig_action", required=True)

    # Subcommand: eligibility audit
    audit_parser = elig_subparsers.add_parser(
        "audit",
        help="Audit dataset samples against unified training eligibility policy.",
    )
    audit_parser.add_argument(
        "--dataset-id",
        "-d",
        type=str,
        default="khmer_story_cartoon_v001",
        help="Dataset ID (default: 'khmer_story_cartoon_v001').",
    )
    audit_parser.add_argument(
        "--governance-version",
        type=str,
        default=None,
        help="Governance version to evaluate (e.g. 'rights_v001').",
    )
    audit_parser.add_argument(
        "--caption-review-version",
        type=str,
        default=None,
        help="Caption review version to evaluate (e.g. 'caption_review_v001').",
    )
    audit_parser.add_argument(
        "--caption-version",
        type=str,
        default="captions_v002",
        help="Caption version identifier (default: 'captions_v002').",
    )
    audit_parser.add_argument(
        "--latent-version",
        type=str,
        default="vae_sd_mse_square256_v001",
        help="Latent cache version (default: 'vae_sd_mse_square256_v001').",
    )
    audit_parser.add_argument(
        "--text-version",
        type=str,
        default="clip_b32_v001",
        help="Text embedding cache version (default: 'clip_b32_v001').",
    )
    audit_parser.add_argument(
        "--mode",
        type=str,
        choices=["development_audit", "production_strict"],
        default="development_audit",
        help="Governance mode for evaluation (default: 'development_audit').",
    )
    audit_parser.add_argument(
        "--dataset-root",
        "-r",
        type=str,
        default="datasets",
        help="Root directory for datasets (default: 'datasets').",
    )

    # Subcommand: snapshot
    snap_parser = subparsers.add_parser(
        "snapshot",
        help="Immutable, reproducible dataset snapshot creation and verification.",
    )
    snap_subparsers = snap_parser.add_subparsers(dest="snap_action", required=True)

    # Subcommand: snapshot plan
    snap_plan_parser = snap_subparsers.add_parser(
        "plan",
        help="Build and inspect a candidate snapshot plan without freezing (read-only).",
    )
    snap_plan_parser.add_argument(
        "--dataset-id",
        "-d",
        type=str,
        default="khmer_story_cartoon_v001",
        help="Dataset ID (default: 'khmer_story_cartoon_v001').",
    )
    snap_plan_parser.add_argument(
        "--snapshot-version",
        type=str,
        default="candidate_plan",
        help="Candidate snapshot version identifier (default: 'candidate_plan').",
    )
    snap_plan_parser.add_argument(
        "--governance-version",
        type=str,
        default=None,
        help="Governance version to evaluate.",
    )
    snap_plan_parser.add_argument(
        "--caption-review-version",
        type=str,
        default=None,
        help="Caption review version to evaluate.",
    )
    snap_plan_parser.add_argument(
        "--caption-version",
        type=str,
        default="captions_v002",
        help="Caption version (default: 'captions_v002').",
    )
    snap_plan_parser.add_argument(
        "--latent-version",
        type=str,
        default="vae_sd_mse_square256_v001",
        help="Latent cache version (default: 'vae_sd_mse_square256_v001').",
    )
    snap_plan_parser.add_argument(
        "--text-version",
        type=str,
        default="clip_b32_v001",
        help="Text cache version (default: 'clip_b32_v001').",
    )
    snap_plan_parser.add_argument(
        "--dataset-root",
        "-r",
        type=str,
        default="datasets",
        help="Root directory for datasets (default: 'datasets').",
    )

    # Subcommand: snapshot freeze
    snap_freeze_parser = snap_subparsers.add_parser(
        "freeze",
        help="Freeze an immutable dataset snapshot binding admitted samples and proofs.",
    )
    snap_freeze_parser.add_argument(
        "--dataset-id",
        "-d",
        type=str,
        default="khmer_story_cartoon_v001",
        help="Dataset ID (default: 'khmer_story_cartoon_v001').",
    )
    snap_freeze_parser.add_argument(
        "--snapshot-version",
        "-v",
        type=str,
        required=True,
        help="Explicit snapshot version identifier (e.g. 'dataset_snapshot_v001').",
    )
    snap_freeze_parser.add_argument(
        "--governance-version",
        type=str,
        required=True,
        help="Explicit governance version (e.g. 'rights_v001').",
    )
    snap_freeze_parser.add_argument(
        "--caption-review-version",
        type=str,
        required=True,
        help="Explicit caption review version (e.g. 'caption_review_v001').",
    )
    snap_freeze_parser.add_argument(
        "--created-by",
        type=str,
        required=True,
        help="Mandatory explicit creator identifier (e.g. 'engineer_lead').",
    )
    snap_freeze_parser.add_argument(
        "--creation-source",
        type=str,
        required=True,
        help="Mandatory explicit creation origin (e.g. 'training_freeze_v001').",
    )
    snap_freeze_parser.add_argument(
        "--previous-version",
        type=str,
        default=None,
        help="Previous snapshot version in lineage if superseding.",
    )
    snap_freeze_parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Contextual notes or rationale for this snapshot.",
    )
    snap_freeze_parser.add_argument(
        "--caption-version",
        type=str,
        default="captions_v002",
        help="Caption version (default: 'captions_v002').",
    )
    snap_freeze_parser.add_argument(
        "--latent-version",
        type=str,
        default="vae_sd_mse_square256_v001",
        help="Latent cache version (default: 'vae_sd_mse_square256_v001').",
    )
    snap_freeze_parser.add_argument(
        "--text-version",
        type=str,
        default="clip_b32_v001",
        help="Text cache version (default: 'clip_b32_v001').",
    )
    snap_freeze_parser.add_argument(
        "--dataset-root",
        "-r",
        type=str,
        default="datasets",
        help="Root directory for datasets (default: 'datasets').",
    )

    # Subcommand: snapshot list
    snap_list_parser = snap_subparsers.add_parser(
        "list",
        help="List all existing snapshots for a dataset.",
    )
    snap_list_parser.add_argument(
        "--dataset-id",
        "-d",
        type=str,
        default="khmer_story_cartoon_v001",
        help="Dataset ID (default: 'khmer_story_cartoon_v001').",
    )
    snap_list_parser.add_argument(
        "--dataset-root",
        "-r",
        type=str,
        default="datasets",
        help="Root directory for datasets (default: 'datasets').",
    )

    # Subcommand: snapshot show
    snap_show_parser = snap_subparsers.add_parser(
        "show",
        help="Display metadata and contents of a frozen snapshot.",
    )
    snap_show_parser.add_argument(
        "--dataset-id",
        "-d",
        type=str,
        default="khmer_story_cartoon_v001",
        help="Dataset ID (default: 'khmer_story_cartoon_v001').",
    )
    snap_show_parser.add_argument(
        "--snapshot-version",
        "-v",
        type=str,
        required=True,
        help="Snapshot version identifier to display.",
    )
    snap_show_parser.add_argument(
        "--dataset-root",
        "-r",
        type=str,
        default="datasets",
        help="Root directory for datasets (default: 'datasets').",
    )

    # Subcommand: snapshot verify
    snap_verify_parser = snap_subparsers.add_parser(
        "verify",
        help="Verify cryptographic tamper integrity of a frozen snapshot.",
    )
    snap_verify_parser.add_argument(
        "--dataset-id",
        "-d",
        type=str,
        default="khmer_story_cartoon_v001",
        help="Dataset ID (default: 'khmer_story_cartoon_v001').",
    )
    snap_verify_parser.add_argument(
        "--snapshot-version",
        "-v",
        type=str,
        required=True,
        help="Snapshot version identifier to verify.",
    )
    snap_verify_parser.add_argument(
        "--dataset-root",
        "-r",
        type=str,
        default="datasets",
        help="Root directory for datasets (default: 'datasets').",
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

    elif args.command == "governance":
        from rernggen.data.governance import GovernanceManager, format_permission_decision

        gov_mgr = GovernanceManager(dataset_root=args.dataset_root)

        if args.gov_action == "list-versions":
            versions = gov_mgr.list_versions(args.dataset_id)
            print(f"\nGovernance versions for dataset '{args.dataset_id}':")
            if not versions:
                print("  (No governance versions found)")
            else:
                for v in versions:
                    print(f"  - {v}")

        elif args.gov_action == "show":
            try:
                records = gov_mgr.load_governance(args.dataset_id, args.governance_version)
            except FileNotFoundError as e:
                print(f"Error: {e}")
                sys.exit(1)

            print(f"\nGovernance records for '{args.dataset_id}' [{args.governance_version}]:")
            print(f"{'Image ID':<12} | {'Training':<10} | {'Commercial':<10} | {'Source':<16} | {'Status'}")
            print("-" * 65)
            counts = {"allowed": 0, "denied": 0, "unknown": 0}
            for r in records:
                t_str = format_permission_decision(r.training_allowed)
                c_str = format_permission_decision(r.commercial_allowed)
                if r.training_allowed is True:
                    counts["allowed"] += 1
                elif r.training_allowed is False:
                    counts["denied"] += 1
                else:
                    counts["unknown"] += 1
                print(f"{r.image_id:<12} | {t_str:<10} | {c_str:<10} | {r.authorization_source:<16} | {r.status}")
            print("-" * 65)
            print(f"Summary: Total: {len(records)} | Allowed: {counts['allowed']} | Denied: {counts['denied']} | Unknown: {counts['unknown']}\n")

        elif args.gov_action == "authorize":
            if not args.image_id and not args.all:
                print("Error: Must specify either --image-id <id...> or --all.")
                sys.exit(1)

            all_ids = None
            if args.all:
                # Discover all dataset image IDs from manifest
                manifest_p = Path(args.dataset_root) / args.dataset_id / "manifests" / "manifest.jsonl"
                if not manifest_p.exists():
                    print(f"Error: Dataset manifest not found at {manifest_p}")
                    sys.exit(1)
                import json
                all_ids = []
                with open(manifest_p, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            all_ids.append(json.loads(line)["image_id"])

            target_images = "ALL" if args.all else args.image_id

            manifest_path, records = gov_mgr.authorize_samples(
                dataset_id=args.dataset_id,
                governance_version=args.governance_version,
                image_ids=target_images,
                training_decision=args.training,
                commercial_decision=args.commercial,
                authorization_source=args.source,
                authorization_note=args.note,
                evidence_reference=args.evidence_ref,
                license_id=args.license_id,
                base_version=args.base_version,
                all_dataset_ids=all_ids,
            )

            print("============================================================")
            print("DATASET AUTHORIZATION RECORDED")
            print("============================================================")
            print(f"Dataset ID:          {args.dataset_id}")
            print(f"Governance Version:  {args.governance_version}")
            print(f"Manifest Path:       {manifest_path}")
            print(f"Target Images:       {len(records)} record(s) in version")
            print(f"Training Decision:   {args.training}")
            print(f"Commercial Decision: {args.commercial}")
            print(f"Source:              {args.source}")
            print(f"Note:                {args.note or '(none)'}")
            print("============================================================")

    elif args.command == "review-captions":
        from rernggen.data.caption_review import CaptionReviewManager

        rev_mgr = CaptionReviewManager(dataset_root=args.dataset_root)

        if args.rev_action == "list-versions":
            versions = rev_mgr.list_versions(args.dataset_id)
            print(f"\nCaption review versions for dataset '{args.dataset_id}':")
            if not versions:
                print("  (No caption review versions found)")
            else:
                for v in versions:
                    print(f"  - {v}")

        elif args.rev_action == "show":
            try:
                records = rev_mgr.load_reviews(args.dataset_id, args.review_version)
            except FileNotFoundError as e:
                print(f"Error: {e}")
                sys.exit(1)

            print(f"\nCaption review records for '{args.dataset_id}' [{args.review_version}]:")
            print(f"{'Image ID':<12} | {'Status':<12} | {'Reviewer':<16} | {'Hash (first 8)':<14} | {'Reason'}")
            print("-" * 75)
            counts = {"approved": 0, "rejected": 0, "pending": 0, "invalidated": 0}
            for r in records:
                st = r.review_status.lower()
                if st in counts:
                    counts[st] += 1
                print(f"{r.image_id:<12} | {r.review_status:<12} | {r.reviewed_by:<16} | {r.caption_sha256[:8]:<14} | {r.reason}")
            print("-" * 75)
            print(f"Summary: Total: {len(records)} | Approved: {counts['approved']} | Rejected: {counts['rejected']} | Pending: {counts['pending']} | Invalidated: {counts['invalidated']}\n")

        elif args.rev_action == "review":
            if not args.image_id and not args.all:
                print("Error: Must specify either --image-id <id...> or --all.")
                sys.exit(1)

            all_ids = None
            if args.all:
                # Discover all dataset image IDs from caption manifest
                caption_p = Path(args.dataset_root) / args.dataset_id / "captions" / args.caption_version / "manifest.jsonl"
                if not caption_p.exists():
                    print(f"Error: Caption manifest not found at {caption_p}")
                    sys.exit(1)
                import json
                all_ids = []
                with open(caption_p, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            all_ids.append(json.loads(line)["image_id"])

            target_images = "ALL" if args.all else args.image_id

            manifest_path, records = rev_mgr.review_samples(
                dataset_id=args.dataset_id,
                review_version=args.review_version,
                image_ids=target_images,
                review_status=args.status,
                reviewed_by=args.reviewer,
                review_source=args.source,
                caption_version=args.caption_version,
                reason=args.reason,
                base_version=args.base_version,
                all_dataset_ids=all_ids,
            )

            print("============================================================")
            print("CAPTION REVIEW RECORDED")
            print("============================================================")
            print(f"Dataset ID:          {args.dataset_id}")
            print(f"Review Version:      {args.review_version}")
            print(f"Manifest Path:       {manifest_path}")
            print(f"Target Images:       {len(records)} record(s) in version")
            print(f"Review Status:       {args.status}")
            print(f"Reviewer:            {args.reviewer}")
            print(f"Source:              {args.source}")
            print(f"Reason / Notes:      {args.reason or '(none)'}")
            print("============================================================")

    elif args.command == "eligibility":
        from rernggen.data.dataset import GovernanceMode, PairedLatentTextDataset

        ds_dir = Path(args.dataset_root) / args.dataset_id
        if not ds_dir.exists():
            print(f"Error: Dataset directory not found: {ds_dir}")
            sys.exit(1)

        gov_mode = GovernanceMode.PRODUCTION_STRICT if args.mode == "production_strict" else GovernanceMode.DEVELOPMENT_AUDIT

        try:
            dataset = PairedLatentTextDataset(
                dataset_dir=ds_dir,
                latent_cache_version=args.latent_version,
                text_cache_version=args.text_version,
                caption_version=args.caption_version,
                governance_version=args.governance_version,
                caption_review_version=args.caption_review_version,
                governance_mode=gov_mode,
            )
        except Exception as e:
            print(f"Error evaluating eligibility: {e}")
            sys.exit(1)

        prov = dataset.eligibility_provenance
        summary = dataset.eligibility_summary

        print("============================================================")
        print("TRAINING ELIGIBILITY AUDIT")
        print("============================================================")
        print(f"Dataset ID:                 {args.dataset_id}")
        print(f"Policy Version:             {prov['policy_version']}")
        print(f"Governance Mode:            {args.mode}")
        print(f"Governance Version:         {prov['governance_version'] or '(none / unversioned)'}")
        print(f"Governance Manifest SHA:    {prov['governance_manifest_sha256'] or '(none)'}")
        print(f"Caption Review Version:     {prov['caption_review_version'] or '(none / unversioned)'}")
        print(f"Caption Review SHA:         {prov['caption_review_manifest_sha256'] or '(none)'}")
        print("-" * 60)
        print(f"Total Paired Samples:       {summary['total_samples']}")
        print(f"Eligible for Training:      {summary['eligible_count']}")
        print(f"Ineligible for Training:    {summary['ineligible_count']}")
        print("-" * 60)
        print("Reason Code Breakdown:")
        if not summary["reason_counts"]:
            print("  (None)")
        else:
            for code, count in summary["reason_counts"].items():
                print(f"  {code:<35}: {count}")
        print("============================================================")

    elif args.command == "snapshot":
        from rernggen.data.snapshot import DatasetSnapshotManager

        mgr = DatasetSnapshotManager(dataset_root=args.dataset_root)

        if args.snap_action == "plan":
            try:
                candidate = mgr.plan_snapshot(
                    dataset_id=args.dataset_id,
                    snapshot_version=args.snapshot_version,
                    governance_version=args.governance_version,
                    caption_review_version=args.caption_review_version,
                    latent_cache_version=args.latent_version,
                    text_cache_version=args.text_version,
                    caption_version=args.caption_version,
                )
                print(candidate.summary())
            except Exception as e:
                print(f"Error planning snapshot: {e}")
                sys.exit(1)

        elif args.snap_action == "freeze":
            try:
                snapshot = mgr.freeze_snapshot(
                    dataset_id=args.dataset_id,
                    snapshot_version=args.snapshot_version,
                    governance_version=args.governance_version,
                    caption_review_version=args.caption_review_version,
                    created_by=args.created_by,
                    creation_source=args.creation_source,
                    latent_cache_version=args.latent_version,
                    text_cache_version=args.text_version,
                    caption_version=args.caption_version,
                    previous_snapshot_version=args.previous_version,
                    notes=args.notes,
                )
                meta = snapshot.metadata
                print("============================================================")
                print("DATASET SNAPSHOT FROZEN")
                print("============================================================")
                print(f"Dataset ID:                 {meta.dataset_id}")
                print(f"Snapshot Version:           {meta.snapshot_version}")
                print(f"Status:                     {meta.status}")
                print(f"Sample Count:               {meta.sample_count}")
                print(f"Created At (UTC):           {meta.created_at}")
                print(f"Created By:                 {meta.created_by}")
                print(f"Creation Source:            {meta.creation_source}")
                print(f"Governance Version:         {meta.governance_version}")
                print(f"Governance Manifest SHA:    {meta.governance_manifest_sha256}")
                print(f"Caption Review Version:     {meta.caption_review_version}")
                print(f"Caption Review SHA:         {meta.caption_review_manifest_sha256}")
                print(f"Eligibility Policy:         {meta.eligibility_policy_version}")
                print(f"Snapshot Manifest SHA:      {meta.snapshot_manifest_sha256}")
                print(f"Previous Version:           {meta.previous_snapshot_version or '(none)'}")
                print(f"Manifest Path:              {snapshot.manifest_path}")
                print("============================================================")
            except Exception as e:
                print(f"Error freezing snapshot: {e}")
                sys.exit(1)

        elif args.snap_action == "list":
            snapshots = mgr.list_snapshots(dataset_id=args.dataset_id)
            print("============================================================")
            print(f"DATASET SNAPSHOTS FOR '{args.dataset_id}'")
            print("============================================================")
            if not snapshots:
                print("  No snapshots found.")
            else:
                for s in snapshots:
                    print(
                        f"  - {s.snapshot_version:<25} | status: {s.status:<8} | samples: {s.sample_count:<4} "
                        f"| gov: {s.governance_version} | rev: {s.caption_review_version} | created: {s.created_at}"
                    )
            print("============================================================")

        elif args.snap_action == "show":
            try:
                snapshot = mgr.load_snapshot(
                    dataset_id=args.dataset_id,
                    snapshot_version=args.snapshot_version,
                    verify_integrity=False,
                )
                meta = snapshot.metadata
                print("============================================================")
                print(f"DATASET SNAPSHOT: {meta.snapshot_version}")
                print("============================================================")
                print(f"Dataset ID:                 {meta.dataset_id}")
                print(f"Status:                     {meta.status}")
                print(f"Sample Count:               {meta.sample_count}")
                print(f"Created At:                 {meta.created_at}")
                print(f"Created By:                 {meta.created_by}")
                print(f"Creation Source:            {meta.creation_source}")
                print(f"Governance Version:         {meta.governance_version}")
                print(f"Governance Manifest SHA:    {meta.governance_manifest_sha256}")
                print(f"Caption Review Version:     {meta.caption_review_version}")
                print(f"Caption Review SHA:         {meta.caption_review_manifest_sha256}")
                print(f"Eligibility Policy:         {meta.eligibility_policy_version}")
                print(f"Snapshot Manifest SHA:      {meta.snapshot_manifest_sha256}")
                print("-" * 60)
                print("Sample Preview (up to 5):")
                for rec in snapshot.records[:5]:
                    print(f"  - {rec.sample_id}: cap_sha={rec.caption_sha256[:12]}... latent={rec.latent_relative_path}")
                if len(snapshot.records) > 5:
                    print(f"  ... and {len(snapshot.records) - 5} more samples.")
                print("============================================================")
            except Exception as e:
                print(f"Error loading snapshot: {e}")
                sys.exit(1)

        elif args.snap_action == "verify":
            try:
                snapshot = mgr.load_snapshot(
                    dataset_id=args.dataset_id,
                    snapshot_version=args.snapshot_version,
                    verify_integrity=True,
                )
                print("============================================================")
                print("SNAPSHOT INTEGRITY VERIFICATION: PASSED")
                print("============================================================")
                print(f"Dataset ID:                 {snapshot.metadata.dataset_id}")
                print(f"Snapshot Version:           {snapshot.metadata.snapshot_version}")
                print(f"Status:                     {snapshot.metadata.status}")
                print(f"Sample Count:               {snapshot.metadata.sample_count} (all records verified)")
                print(f"Manifest SHA-256:           {snapshot.metadata.snapshot_manifest_sha256} (MATCH)")
                print("All cryptographic hashes and record checksums valid.")
                print("============================================================")
            except Exception as e:
                print(f"Snapshot integrity verification FAILED: {e}")
                sys.exit(1)


if __name__ == "__main__":
    main()



