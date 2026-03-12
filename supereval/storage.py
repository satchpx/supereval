from __future__ import annotations

import os
from pathlib import Path

from .models import DatasetMeta, DatasetType, TestCase, TYPE_TO_MODEL


def datasets_dir() -> Path:
    """
    Resolves the datasets directory.
    Override with SUPEREVAL_DATASETS_DIR env var; defaults to ./datasets
    relative to the current working directory.
    """
    env = os.environ.get("SUPEREVAL_DATASETS_DIR")
    return Path(env) if env else Path.cwd() / "datasets"


def dataset_path(name: str) -> Path:
    return datasets_dir() / name


def load_dataset_meta(name: str) -> DatasetMeta:
    path = dataset_path(name) / "dataset.json"
    if not path.exists():
        raise FileNotFoundError(f"Dataset '{name}' not found. Run `supereval dataset list` to see available datasets.")
    return DatasetMeta.model_validate_json(path.read_text())


def save_dataset_meta(meta: DatasetMeta) -> None:
    path = dataset_path(meta.name)
    path.mkdir(parents=True, exist_ok=True)
    (path / "dataset.json").write_text(meta.model_dump_json(indent=2))


def load_cases(name: str) -> list[TestCase]:
    meta = load_dataset_meta(name)
    model = TYPE_TO_MODEL[meta.type]
    cases_path = dataset_path(name) / "cases.jsonl"

    if not cases_path.exists():
        return []

    cases = []
    for i, line in enumerate(cases_path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            cases.append(model.model_validate_json(line))
        except Exception as e:
            raise ValueError(f"Invalid case on line {i} in '{name}/cases.jsonl': {e}")
    return cases


def append_cases(name: str, cases: list[TestCase]) -> None:
    cases_path = dataset_path(name) / "cases.jsonl"
    with cases_path.open("a") as f:
        for case in cases:
            f.write(case.model_dump_json() + "\n")


def list_datasets() -> list[DatasetMeta]:
    base = datasets_dir()
    if not base.exists():
        return []
    result = []
    for d in sorted(base.iterdir()):
        if d.is_dir() and (d / "dataset.json").exists():
            try:
                result.append(load_dataset_meta(d.name))
            except Exception:
                continue
    return result
