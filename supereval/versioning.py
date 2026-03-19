"""
Dataset versioning — snapshot, list, and restore named versions.

Versions are stored as subdirectories under datasets/<name>/versions/:

    datasets/
      aws-support-qa/
        cases.jsonl          ← current working copy
        dataset.json
        versions/
          v1.0.0/
            cases.jsonl      ← snapshot of cases at tag time
            dataset.json     ← snapshot of meta at tag time
            version.json     ← { version, created_at, case_count, description }
          v1.1.0/
            ...

Version strings must follow semantic versioning: v<MAJOR>.<MINOR>.<PATCH>
(e.g. v1.0.0, v1.2.3).
"""
from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple


VERSION_RE = re.compile(r"^v\d+\.\d+\.\d+$")


class VersionInfo(NamedTuple):
    version: str
    created_at: str
    case_count: int
    description: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _version_dir(dataset_name: str, version: str) -> Path:
    from .storage import dataset_path
    return dataset_path(dataset_name) / "versions" / version


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_version(version: str) -> None:
    """Raise ValueError if version is not a valid vMAJOR.MINOR.PATCH string."""
    if not VERSION_RE.match(version):
        raise ValueError(
            f"Invalid version '{version}'. "
            "Use semantic versioning format: v<MAJOR>.<MINOR>.<PATCH> (e.g. v1.0.0)"
        )


def version_exists(dataset_name: str, version: str) -> bool:
    return _version_dir(dataset_name, version).exists()


def tag_version(
    dataset_name: str,
    version: str,
    description: str = "",
) -> VersionInfo:
    """Snapshot the current cases.jsonl and dataset.json into versions/<version>/.

    Raises ValueError if the version already exists or the format is invalid.
    Raises FileNotFoundError if the dataset does not exist.
    """
    validate_version(version)

    from .storage import dataset_path, load_dataset_meta
    load_dataset_meta(dataset_name)  # raises FileNotFoundError if dataset missing

    if version_exists(dataset_name, version):
        raise ValueError(
            f"Version '{version}' already exists for dataset '{dataset_name}'. "
            "Choose a different version string."
        )

    ds_path = dataset_path(dataset_name)
    vdir = _version_dir(dataset_name, version)
    vdir.mkdir(parents=True, exist_ok=True)

    # Snapshot cases.jsonl
    cases_src = ds_path / "cases.jsonl"
    case_count = 0
    if cases_src.exists():
        shutil.copy2(cases_src, vdir / "cases.jsonl")
        case_count = sum(
            1 for line in cases_src.read_text().splitlines() if line.strip()
        )
    else:
        (vdir / "cases.jsonl").write_text("")

    # Snapshot dataset.json
    meta_src = ds_path / "dataset.json"
    if meta_src.exists():
        shutil.copy2(meta_src, vdir / "dataset.json")

    # Write version metadata
    created_at = datetime.now(timezone.utc).isoformat()
    (vdir / "version.json").write_text(json.dumps({
        "version": version,
        "created_at": created_at,
        "case_count": case_count,
        "description": description,
    }, indent=2))

    return VersionInfo(
        version=version,
        created_at=created_at,
        case_count=case_count,
        description=description,
    )


def list_versions(dataset_name: str) -> list[VersionInfo]:
    """Return all versions for a dataset, sorted by version string."""
    from .storage import dataset_path
    versions_root = dataset_path(dataset_name) / "versions"
    if not versions_root.exists():
        return []

    result: list[VersionInfo] = []
    for vdir in sorted(versions_root.iterdir()):
        if not vdir.is_dir():
            continue
        vinfo_path = vdir / "version.json"
        if not vinfo_path.exists():
            continue
        try:
            d = json.loads(vinfo_path.read_text())
            result.append(VersionInfo(
                version=d["version"],
                created_at=d["created_at"],
                case_count=d["case_count"],
                description=d.get("description", ""),
            ))
        except Exception:
            continue  # skip malformed version.json

    return result


def load_version_cases(dataset_name: str, version: str) -> list:
    """Load test cases from a specific version snapshot.

    Uses the versioned dataset.json to determine the case type; falls back
    to the current dataset.json if the snapshot has none.
    """
    vdir = _version_dir(dataset_name, version)
    if not vdir.exists():
        raise FileNotFoundError(
            f"Version '{version}' not found for dataset '{dataset_name}'."
        )

    # Resolve meta: prefer version snapshot, fall back to current
    versioned_meta_path = vdir / "dataset.json"
    if versioned_meta_path.exists():
        from .models import DatasetMeta
        meta = DatasetMeta.model_validate_json(versioned_meta_path.read_text())
    else:
        from .storage import load_dataset_meta
        meta = load_dataset_meta(dataset_name)

    from .models import TYPE_TO_MODEL
    model = TYPE_TO_MODEL[meta.type]

    cases_path = vdir / "cases.jsonl"
    if not cases_path.exists():
        return []

    cases = []
    for i, line in enumerate(cases_path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            cases.append(model.model_validate_json(line))
        except Exception as exc:
            raise ValueError(
                f"Invalid case on line {i} of version '{version}': {exc}"
            ) from exc
    return cases


def restore_version(dataset_name: str, version: str) -> int:
    """Overwrite the current cases.jsonl with the version snapshot.

    Returns the number of cases restored.
    Raises FileNotFoundError if the version or its cases snapshot does not exist.
    """
    vdir = _version_dir(dataset_name, version)
    if not vdir.exists():
        raise FileNotFoundError(
            f"Version '{version}' not found for dataset '{dataset_name}'."
        )

    cases_src = vdir / "cases.jsonl"
    if not cases_src.exists():
        raise FileNotFoundError(
            f"Version '{version}' has no cases.jsonl snapshot."
        )

    from .storage import dataset_path
    shutil.copy2(cases_src, dataset_path(dataset_name) / "cases.jsonl")
    return sum(1 for line in cases_src.read_text().splitlines() if line.strip())
