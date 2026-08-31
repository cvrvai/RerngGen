"""Explicit Dataset Authorization Governance Engine for RerngGen.

Provides auditable, versioned human authorization tracking for dataset images,
enforcing explicit tri-state decisions (ALLOW, DENY, UNKNOWN) for model training
and commercial deployment without implicit or default permissions.
"""

from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from rernggen.data.schema import GovernanceRecord


class PermissionDecision(str, Enum):
    """Explicit tri-state permission decision for dataset governance."""

    ALLOW = "ALLOW"
    DENY = "DENY"
    UNKNOWN = "UNKNOWN"


def parse_permission_decision(val: Union[str, PermissionDecision, Optional[bool]]) -> Optional[bool]:
    """Parses and validates an explicit permission decision into a tri-state boolean or None.

    Args:
        val: Input decision ('ALLOW', 'DENY', 'UNKNOWN', True, False, None, or PermissionDecision).

    Returns:
        Optional[bool]: True for ALLOW, False for DENY, None for UNKNOWN.

    Raises:
        ValueError: If input is not a recognized explicit decision.
    """
    if isinstance(val, PermissionDecision):
        val = val.value

    if isinstance(val, str):
        val_upper = val.strip().upper()
        if val_upper == "ALLOW":
            return True
        elif val_upper == "DENY":
            return False
        elif val_upper == "UNKNOWN":
            return None
        else:
            raise ValueError(
                f"Invalid permission decision '{val}'. Must be exactly 'ALLOW', 'DENY', or 'UNKNOWN'."
            )
    elif val is True:
        return True
    elif val is False:
        return False
    elif val is None:
        return None
    else:
        raise ValueError(
            f"Invalid permission decision type '{type(val)}'. Must be 'ALLOW', 'DENY', or 'UNKNOWN'."
        )


def format_permission_decision(val: Optional[bool]) -> str:
    """Formats a tri-state boolean value into canonical human-readable string.

    Args:
        val: Tri-state boolean (True, False, or None).

    Returns:
        str: 'ALLOW', 'DENY', or 'UNKNOWN'.
    """
    if val is True:
        return "ALLOW"
    elif val is False:
        return "DENY"
    else:
        return "UNKNOWN"


def compute_governance_record_sha256(record: Union[Dict[str, Any], GovernanceRecord]) -> str:
    """Computes a deterministic SHA-256 hash for a GovernanceRecord excluding the hash itself.

    Args:
        record: GovernanceRecord dataclass or dictionary.

    Returns:
        str: 64-character hexadecimal SHA-256 hash.
    """
    data = record.to_dict() if isinstance(record, GovernanceRecord) else dict(record)
    # Exclude record_sha256 from payload to avoid circularity
    payload_dict = {k: v for k, v in data.items() if k != "record_sha256"}
    serialized = json.dumps(payload_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class GovernanceManager:
    """Manages versioned, cryptographic governance manifests for dataset authorization."""

    def __init__(self, dataset_root: Union[str, Path] = "datasets") -> None:
        """Initializes the governance manager.

        Args:
            dataset_root: Root path containing dataset repositories (default: 'datasets').
        """
        self.dataset_root = Path(dataset_root)

    def get_governance_dir(self, dataset_id: str, version: str) -> Path:
        """Returns the absolute directory path for a governance version.

        Args:
            dataset_id: Dataset identifier.
            version: Governance version identifier (e.g. 'rights_v001').

        Returns:
            Path: Directory path.
        """
        return self.dataset_root / dataset_id / "governance" / version

    def list_versions(self, dataset_id: str) -> List[str]:
        """Lists all existing governance versions for a dataset sorted lexicographically.

        Args:
            dataset_id: Dataset identifier.

        Returns:
            List[str]: List of version directory names containing manifest.jsonl.
        """
        gov_base = self.dataset_root / dataset_id / "governance"
        if not gov_base.exists():
            return []

        versions = []
        for p in sorted(gov_base.iterdir()):
            if p.is_dir() and (p / "manifest.jsonl").exists():
                versions.append(p.name)
        return versions

    def load_governance(self, dataset_id: str, version: str) -> List[GovernanceRecord]:
        """Loads and verifies all governance records from a versioned manifest.

        Args:
            dataset_id: Dataset identifier.
            version: Governance version identifier.

        Returns:
            List[GovernanceRecord]: Parsed and verified records.

        Raises:
            FileNotFoundError: If the manifest file does not exist.
            ValueError: If manifest is corrupt or contains duplicate image IDs.
        """
        manifest_path = self.get_governance_dir(dataset_id, version) / "manifest.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Governance manifest not found: {manifest_path}")

        records: List[GovernanceRecord] = []
        seen_ids = set()

        with open(manifest_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                clean_line = line.strip()
                if not clean_line:
                    continue
                try:
                    data = json.loads(clean_line)
                    rec = GovernanceRecord(**data)
                except Exception as e:
                    raise ValueError(f"Corrupt JSON at line {line_idx} in {manifest_path}: {e}")

                if rec.image_id in seen_ids:
                    raise ValueError(
                        f"Duplicate image_id '{rec.image_id}' found at line {line_idx} in {manifest_path}"
                    )
                seen_ids.add(rec.image_id)
                records.append(rec)

        return records

    def compute_manifest_sha256(self, dataset_id: str, version: str) -> str:
        """Computes SHA-256 hash of a governance version manifest file for training provenance.

        Args:
            dataset_id: Dataset identifier.
            version: Governance version identifier.

        Returns:
            str: 64-character hexadecimal SHA-256 hash.
        """
        manifest_path = self.get_governance_dir(dataset_id, version) / "manifest.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Governance manifest not found: {manifest_path}")
        return hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    def save_governance(
        self,
        dataset_id: str,
        version: str,
        records: Sequence[GovernanceRecord],
        _allow_test_overwrite: bool = False,
    ) -> Path:
        """Saves governance records atomically with cryptographic record hashing.

        Finalized versions are strictly immutable. Standard callers cannot overwrite existing version manifests.

        Args:
            dataset_id: Target dataset identifier.
            version: Target governance version (e.g. 'rights_v001').
            records: Sequence of GovernanceRecord instances.
            _allow_test_overwrite: Internal test flag only. Standard workflows must create a new version.

        Returns:
            Path: Path to saved manifest.jsonl.

        Raises:
            FileExistsError: If version already exists and _allow_test_overwrite is False.
            ValueError: If record list is empty, contains duplicates, or has invalid fields.
        """
        if not records:
            raise ValueError("Cannot save empty governance record list.")

        target_dir = self.get_governance_dir(dataset_id, version)
        manifest_path = target_dir / "manifest.jsonl"

        if manifest_path.exists() and not _allow_test_overwrite:
            raise FileExistsError(
                f"Governance version '{version}' already exists at {manifest_path}. "
                f"Governance versions are finalized and strictly immutable. "
                f"To record modifications, create a new version (e.g. rights_v002) with base_version='{version}'."
            )

        target_dir.mkdir(parents=True, exist_ok=True)

        # Validate uniqueness & recalculate hashes
        seen_ids = set()
        final_records: List[GovernanceRecord] = []

        for r in records:
            if r.image_id in seen_ids:
                raise ValueError(f"Duplicate image_id '{r.image_id}' in governance record set.")
            seen_ids.add(r.image_id)

            if not r.authorization_source or not r.authorization_source.strip():
                raise ValueError(f"GovernanceRecord for '{r.image_id}' missing authorization_source.")
            if not r.authorized_at or not r.authorized_at.strip():
                raise ValueError(f"GovernanceRecord for '{r.image_id}' missing authorized_at timestamp.")
            if r.status not in ("ACTIVE", "SUPERSEDED", "REVOKED"):
                raise ValueError(f"Invalid status '{r.status}' for '{r.image_id}'.")

            # Recompute SHA-256 for deterministic integrity
            record_hash = compute_governance_record_sha256(r)
            r.record_sha256 = record_hash
            final_records.append(r)

        # Atomic write on same filesystem
        tmp_manifest = target_dir / f"manifest_tmp_{os.getpid()}_{time.time_ns()}.jsonl"
        with open(tmp_manifest, "w", encoding="utf-8") as f:
            for r in final_records:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

        os.replace(tmp_manifest, manifest_path)
        return manifest_path

    def authorize_samples(
        self,
        dataset_id: str,
        governance_version: str,
        image_ids: Union[str, Sequence[str]],
        training_decision: Union[str, PermissionDecision, Optional[bool]],
        commercial_decision: Union[str, PermissionDecision, Optional[bool]],
        authorization_source: str,
        authorization_note: str = "",
        evidence_reference: Optional[str] = None,
        license_id: Optional[str] = None,
        base_version: Optional[str] = None,
        all_dataset_ids: Optional[Sequence[str]] = None,
        status: str = "ACTIVE",
        _allow_test_overwrite: bool = False,
    ) -> Tuple[Path, List[GovernanceRecord]]:
        """Applies explicit human authorization decisions to specified dataset records.

        Args:
            dataset_id: Target dataset identifier.
            governance_version: Version name to write into (e.g. 'rights_v001').
            image_ids: Single image ID, list of IDs, or 'ALL' if all_dataset_ids provided.
            training_decision: Explicit tri-state decision for model training ('ALLOW', 'DENY', 'UNKNOWN').
            commercial_decision: Explicit tri-state decision for commercial rights ('ALLOW', 'DENY', 'UNKNOWN').
            authorization_source: Mandatory explicit authority identifier (e.g. 'human_audit', 'license_verified').
            authorization_note: Human audit rationale/note.
            evidence_reference: Optional pointer to license/evidence document.
            license_id: Optional identifier for applicable license.
            base_version: Optional existing version to clone/update records from.
            all_dataset_ids: Optional complete list of image IDs in dataset when executing --all.
            status: Record status ('ACTIVE', 'SUPERSEDED', 'REVOKED'). Default: 'ACTIVE'.
            _allow_test_overwrite: Internal test flag only.

        Returns:
            Tuple[Path, List[GovernanceRecord]]: Manifest path and updated records.
        """
        # 1. Validate mandatory authorization_source
        if not isinstance(authorization_source, str) or not authorization_source.strip():
            raise ValueError(
                "authorization_source is required and must be a non-empty string explicitly identifying the authority."
            )

        if status not in ("ACTIVE", "SUPERSEDED", "REVOKED"):
            raise ValueError(f"Invalid governance status '{status}'. Must be 'ACTIVE', 'SUPERSEDED', or 'REVOKED'.")

        # 2. Parse and strictly validate tri-state decisions
        training_allowed = parse_permission_decision(training_decision)
        commercial_allowed = parse_permission_decision(commercial_decision)

        # 3. Resolve target image IDs
        if isinstance(image_ids, str):
            if image_ids.strip().upper() == "ALL":
                if not all_dataset_ids:
                    raise ValueError(
                        "Cannot authorize 'ALL' without providing explicit all_dataset_ids list."
                    )
                target_ids = list(all_dataset_ids)
            else:
                target_ids = [image_ids.strip()]
        else:
            target_ids = [img.strip() for img in image_ids if img.strip()]

        if not target_ids:
            raise ValueError("No valid image IDs specified for authorization.")

        # 4. Enforce version immutability
        target_dir = self.get_governance_dir(dataset_id, governance_version)
        manifest_path = target_dir / "manifest.jsonl"
        if manifest_path.exists() and not _allow_test_overwrite:
            raise FileExistsError(
                f"Governance version '{governance_version}' already exists and is finalized. "
                f"To record new authorization decisions, specify a new version (e.g. rights_v002) "
                f"with base_version='{governance_version}'."
            )

        # 5. Load baseline records if available
        record_map: Dict[str, GovernanceRecord] = {}
        if base_version is not None:
            for r in self.load_governance(dataset_id, base_version):
                record_map[r.image_id] = r
        elif manifest_path.exists() and _allow_test_overwrite:
            for r in self.load_governance(dataset_id, governance_version):
                record_map[r.image_id] = r

        # 6. Apply updates with explicit timestamp
        timestamp = datetime.now(timezone.utc).isoformat()

        for img_id in target_ids:
            prev_version = record_map[img_id].governance_version if img_id in record_map else base_version
            rec = GovernanceRecord(
                image_id=img_id,
                dataset_id=dataset_id,
                governance_version=governance_version,
                training_allowed=training_allowed,
                commercial_allowed=commercial_allowed,
                authorization_source=authorization_source.strip(),
                authorized_at=timestamp,
                license_id=license_id if license_id is not None else (record_map[img_id].license_id if img_id in record_map else None),
                authorization_note=authorization_note,
                evidence_reference=evidence_reference,
                status=status,
                previous_governance_version=prev_version,
            )
            record_map[img_id] = rec

        # 7. Sort records deterministically by image_id
        sorted_records = [record_map[k] for k in sorted(record_map.keys())]

        # 8. Save atomically
        saved_manifest_path = self.save_governance(
            dataset_id=dataset_id,
            version=governance_version,
            records=sorted_records,
            _allow_test_overwrite=_allow_test_overwrite,
        )

        return saved_manifest_path, sorted_records

