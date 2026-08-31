"""CLI entrypoint for RerngGen training run provenance, preflight validation, and smoke training.

Usage:
    python -m rernggen.training run create --run-id run_000001 --dataset-id my_ds --snapshot-version snapshot_v001 ...
    python -m rernggen.training run show --run-id run_000001
    python -m rernggen.training run verify --run-id run_000001
    python -m rernggen.training run list
    python -m rernggen.training preflight --run-id run_000001
    python -m rernggen.training smoke --run-id run_000001 --device cpu --steps 2
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

from rernggen.training.preflight import run_preflight_checks
from rernggen.training.provenance import TrainingRunManager
from rernggen.training.smoke import run_tiny_dit_smoke


def load_json_or_dict(value: str) -> Dict[str, Any]:
    """Parses a string argument as either a JSON file path or a direct JSON string."""
    p = Path(value)
    if p.is_file():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    try:
        return json.loads(value)
    except Exception as e:
        raise argparse.ArgumentTypeError(
            f"Invalid JSON string or file path for '{value}': {e}"
        )


def main() -> None:
    """CLI dispatcher for training commands."""
    parser = argparse.ArgumentParser(
        prog="rernggen.training",
        description="RerngGen Training Experiment Provenance & Run Management CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand group: run
    run_parser = subparsers.add_parser("run", help="Training run management subcommands")
    run_subparsers = run_parser.add_subparsers(dest="run_command", required=True)

    # 1. run create
    create_parser = run_subparsers.add_parser(
        "create",
        help="Create an immutable training run experiment specification from a verified frozen snapshot.",
    )
    create_parser.add_argument("--run-id", type=str, required=True, help="Unique run identifier.")
    create_parser.add_argument("--dataset-id", "-d", type=str, required=True, help="Dataset identifier.")
    create_parser.add_argument("--snapshot-version", "-s", type=str, required=True, help="Frozen dataset snapshot version.")
    create_parser.add_argument("--model-family", "-m", type=str, default="dit", help="Model family name (e.g. 'dit').")
    create_parser.add_argument(
        "--model-config",
        type=load_json_or_dict,
        default="{}",
        help="JSON string or file path containing model architecture configuration.",
    )
    create_parser.add_argument(
        "--training-config",
        type=load_json_or_dict,
        default="{}",
        help="JSON string or file path containing training hyperparameters configuration.",
    )
    create_parser.add_argument("--seed", type=int, required=True, help="Deterministic random seed [0, 2^32 - 1].")
    create_parser.add_argument("--created-by", type=str, required=True, help="Operator username/identity.")
    create_parser.add_argument("--creation-source", type=str, required=True, help="Operational context description.")
    create_parser.add_argument("--expected-snapshot-metadata-sha256", type=str, default=None, help="Expected snapshot metadata SHA.")
    create_parser.add_argument("--notes", type=str, default=None, help="Optional experiment notes.")
    create_parser.add_argument("--training-root", type=str, default="training_runs", help="Root directory for runs.")
    create_parser.add_argument("--dataset-root", type=str, default="datasets", help="Root directory for datasets.")

    # 2. run show
    show_parser = run_subparsers.add_parser("show", help="Show training run specification and status.")
    show_parser.add_argument("--run-id", type=str, required=True, help="Run identifier.")
    show_parser.add_argument("--training-root", type=str, default="training_runs", help="Root directory for runs.")

    # 3. run list
    list_parser = run_subparsers.add_parser("list", help="List all training runs.")
    list_parser.add_argument("--dataset-id", "-d", type=str, default=None, help="Filter by dataset ID.")
    list_parser.add_argument("--training-root", type=str, default="training_runs", help="Root directory for runs.")
    list_parser.add_argument("--verify", action="store_true", help="Verify integrity while listing.")

    # 4. run verify
    verify_parser = run_subparsers.add_parser("verify", help="Verify training run specification and referenced snapshot integrity.")
    verify_parser.add_argument("--run-id", type=str, required=True, help="Run identifier.")
    verify_parser.add_argument("--training-root", type=str, default="training_runs", help="Root directory for runs.")
    verify_parser.add_argument("--dataset-root", type=str, default="datasets", help="Root directory for datasets.")

    # 5. run start
    start_parser = run_subparsers.add_parser("start", help="Transition run from PLANNED to RUNNING.")
    start_parser.add_argument("--run-id", type=str, required=True, help="Run identifier.")
    start_parser.add_argument("--training-root", type=str, default="training_runs", help="Root directory for runs.")

    # 6. run complete
    complete_parser = run_subparsers.add_parser("complete", help="Transition run from RUNNING to COMPLETED.")
    complete_parser.add_argument("--run-id", type=str, required=True, help="Run identifier.")
    complete_parser.add_argument("--current-step", type=int, default=None, help="Final step count.")
    complete_parser.add_argument("--last-checkpoint", type=str, default=None, help="Path/URI to final checkpoint.")
    complete_parser.add_argument("--training-root", type=str, default="training_runs", help="Root directory for runs.")

    # 7. run fail
    fail_parser = run_subparsers.add_parser("fail", help="Transition run from RUNNING to FAILED.")
    fail_parser.add_argument("--run-id", type=str, required=True, help="Run identifier.")
    fail_parser.add_argument("--reason", type=str, required=True, help="Failure explanation.")
    fail_parser.add_argument("--current-step", type=int, default=None, help="Step at failure.")
    fail_parser.add_argument("--last-checkpoint", type=str, default=None, help="Last valid checkpoint.")
    fail_parser.add_argument("--training-root", type=str, default="training_runs", help="Root directory for runs.")

    # 8. run abort
    abort_parser = run_subparsers.add_parser("abort", help="Transition run from PLANNED/RUNNING to ABORTED.")
    abort_parser.add_argument("--run-id", type=str, required=True, help="Run identifier.")
    abort_parser.add_argument("--reason", type=str, default=None, help="Abort explanation.")
    abort_parser.add_argument("--training-root", type=str, default="training_runs", help="Root directory for runs.")

    # Top-level subcommand: preflight
    preflight_parser = subparsers.add_parser("preflight", help="Execute preflight checks on a training run.")
    preflight_parser.add_argument("--run-id", type=str, required=True, help="Run identifier.")
    preflight_parser.add_argument("--training-root", type=str, default="training_runs", help="Root directory for runs.")
    preflight_parser.add_argument("--dataset-root", type=str, default="datasets", help="Root directory for datasets.")
    preflight_parser.add_argument("--device", type=str, default="cpu", help="Target device ('cpu' or 'cuda').")

    # Top-level subcommand: smoke
    smoke_parser = subparsers.add_parser("smoke", help="Execute an end-to-end TinyDiT smoke training run.")
    smoke_parser.add_argument("--run-id", type=str, required=True, help="Run identifier.")
    smoke_parser.add_argument("--training-root", type=str, default="training_runs", help="Root directory for runs.")
    smoke_parser.add_argument("--dataset-root", type=str, default="datasets", help="Root directory for datasets.")
    smoke_parser.add_argument("--device", type=str, default="cpu", help="Target device ('cpu' or 'cuda').")
    smoke_parser.add_argument("--steps", type=int, default=2, help="Number of smoke training steps.")
    smoke_parser.add_argument("--no-checkpoint", action="store_true", help="Disable checkpoint saving.")

    args = parser.parse_args()

    if args.command == "run":
        mgr = TrainingRunManager(
            training_root=getattr(args, "training_root", "training_runs"),
            dataset_root=getattr(args, "dataset_root", "datasets"),
        )

        if args.run_command == "create":
            run = mgr.create_run(
                run_id=args.run_id,
                dataset_id=args.dataset_id,
                snapshot_version=args.snapshot_version,
                model_family=args.model_family,
                model_config=args.model_config,
                training_config=args.training_config,
                seed=args.seed,
                created_by=args.created_by,
                creation_source=args.creation_source,
                expected_snapshot_metadata_sha256=args.expected_snapshot_metadata_sha256,
                notes=args.notes,
            )
            print("=" * 60)
            print(f"TRAINING RUN CREATED: {run.spec.run_id}")
            print("=" * 60)
            print(f"Status:                     {run.state.status}")
            print(f"Dataset ID:                 {run.spec.dataset_id}")
            print(f"Snapshot Version:           {run.spec.snapshot_version}")
            print(f"Snapshot Metadata SHA:      {run.spec.snapshot_metadata_sha256}")
            print(f"Model Family:               {run.spec.model_family}")
            print(f"Seed:                       {run.spec.seed}")
            print(f"Git Commit:                 {run.spec.git_commit or '(none)'}")
            print(f"Git Dirty:                  {run.spec.git_dirty}")
            print(f"Run Spec SHA-256:           {run.spec.run_spec_sha256}")
            print(f"Run Directory:              {run.run_dir}")
            print("=" * 60)

        elif args.run_command == "show":
            run = mgr.load_run(args.run_id)
            print("=" * 60)
            print(f"TRAINING RUN: {run.spec.run_id}")
            print("=" * 60)
            print(f"Status:                     {run.state.status}")
            print(f"Dataset ID:                 {run.spec.dataset_id}")
            print(f"Snapshot Version:           {run.spec.snapshot_version}")
            print(f"Snapshot Metadata SHA:      {run.spec.snapshot_metadata_sha256}")
            print(f"Model Family:               {run.spec.model_family}")
            print(f"Seed:                       {run.spec.seed}")
            print(f"Git Commit:                 {run.spec.git_commit or '(none)'}")
            print(f"Git Dirty:                  {run.spec.git_dirty}")
            print(f"Started At:                 {run.state.started_at or '(not started)'}")
            print(f"Completed At:               {run.state.completed_at or '(not completed)'}")
            print(f"Current Step:               {run.state.current_step}")
            print(f"Last Checkpoint:            {run.state.last_checkpoint or '(none)'}")
            print(f"Failure Reason:             {run.state.failure_reason or '(none)'}")
            print(f"Run Spec SHA-256:           {run.spec.run_spec_sha256}")
            print("=" * 60)

        elif args.run_command == "list":
            runs = mgr.list_runs(verify_integrity=args.verify, dataset_id=args.dataset_id)
            print("=" * 60)
            print(f"TRAINING RUNS (Total: {len(runs)})")
            print("=" * 60)
            if not runs:
                print("  No training runs found.")
            for r in runs:
                print(
                    f"  [{r.state.status:<9}] {r.spec.run_id:<20} "
                    f"Dataset: {r.spec.dataset_id} | Snap: {r.spec.snapshot_version} | Seed: {r.spec.seed}"
                )
            print("=" * 60)

        elif args.run_command == "verify":
            rep = mgr.verify_run(args.run_id, dataset_root=args.dataset_root)
            print("=" * 60)
            print(f"TRAINING RUN VERIFICATION: {args.run_id}")
            print("=" * 60)
            if rep["valid"]:
                print("Result:                     PASSED")
                print(f"Status:                     {rep['state']['status']}")
                print(f"Run Spec SHA-256:           {rep['spec']['run_spec_sha256']}")
            else:
                print("Result:                     FAILED")
                for err in rep["errors"]:
                    print(f"  Error: {err}")
            print("=" * 60)
            if not rep["valid"]:
                sys.exit(1)

        elif args.run_command == "start":
            run = mgr.start_run(args.run_id)
            print(f"Run '{args.run_id}' transitioned to RUNNING at {run.state.started_at}.")

        elif args.run_command == "complete":
            run = mgr.complete_run(args.run_id, current_step=args.current_step, last_checkpoint=args.last_checkpoint)
            print(f"Run '{args.run_id}' transitioned to COMPLETED at {run.state.completed_at}.")

        elif args.run_command == "fail":
            run = mgr.fail_run(args.run_id, failure_reason=args.reason, current_step=args.current_step, last_checkpoint=args.last_checkpoint)
            print(f"Run '{args.run_id}' transitioned to FAILED (reason: {args.reason}).")

        elif args.run_command == "abort":
            run = mgr.abort_run(args.run_id, abort_reason=args.reason)
            print(f"Run '{args.run_id}' transitioned to ABORTED.")

    elif args.command == "preflight":
        result = run_preflight_checks(
            run_id=args.run_id,
            training_root=args.training_root,
            dataset_root=args.dataset_root,
            device=args.device,
        )
        print("=" * 60)
        print(f"TRAINING PREFLIGHT REPORT: {args.run_id}")
        print("=" * 60)
        print(f"Overall Result:             {'PASSED' if result.passed else 'FAILED'}")
        print(f"Device:                     {result.device}")
        print(f"Sample Count:               {result.sample_count}")
        print(f"Batch Size:                 {result.batch_size}")
        print(f"Random Seed:                {result.seed}")
        print(f"Snapshot Metadata SHA:      {result.snapshot_metadata_sha256[:16]}...")
        print("-" * 60)
        print("Individual Checks:")
        for check, passed in result.checks.items():
            status_str = "PASS" if passed else "FAIL"
            print(f"  [{status_str:4}] {check}")
        if result.warnings:
            print("-" * 60)
            print("Warnings:")
            for w in result.warnings:
                print(f"  - {w}")
        if result.errors:
            print("-" * 60)
            print("Errors:")
            for err in result.errors:
                print(f"  - {err}")
        print("=" * 60)
        if not result.passed:
            sys.exit(1)

    elif args.command == "smoke":
        try:
            res = run_tiny_dit_smoke(
                run_id=args.run_id,
                training_root=args.training_root,
                dataset_root=args.dataset_root,
                device=args.device,
                max_steps=args.steps,
                save_checkpoint=not args.no_checkpoint,
            )
            print("=" * 60)
            print(f"TINY DiT SMOKE TRAINING COMPLETED: {args.run_id}")
            print("=" * 60)
            print(f"Status:                     {res['status']}")
            print(f"Device:                     {res['device']}")
            print(f"Steps Executed:             {res['steps_completed']}")
            print(f"Total Parameters:           {res['parameter_counts']['total_parameters']:,}")
            print(f"Trainable Parameters:       {res['parameter_counts']['trainable_parameters']:,}")
            for idx, loss_val in enumerate(res["step_losses"], start=1):
                print(f"  Step {idx:02d} Loss:              {loss_val:.6f}")
            if res.get("checkpoint_path"):
                print(f"Checkpoint Saved:           {res['checkpoint_path']}")
            print("=" * 60)
        except Exception as e:
            print("=" * 60)
            print(f"TINY DiT SMOKE TRAINING FAILED: {args.run_id}")
            print("=" * 60)
            print(f"Error: {e}")
            print("=" * 60)
            sys.exit(1)


if __name__ == "__main__":
    main()
