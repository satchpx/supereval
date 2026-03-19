"""
Storage for RAG datasets.

RAG datasets live under $SUPEREVAL_DATASETS_DIR/rag/<name>/
to avoid collisions with LLM and agent datasets.

Each dataset directory contains:
  dataset.json   — RagDatasetMeta
  cases.jsonl    — one RagTestCase per line
  baseline.json  — committed to git; updated by --update-baseline
"""
from __future__ import annotations

import os
from pathlib import Path

from .models import RagDatasetMeta, RagTestCase


def _base_dir() -> Path:
    env = os.environ.get("SUPEREVAL_DATASETS_DIR")
    return Path(env) if env else Path.cwd() / "datasets"


def rag_datasets_dir() -> Path:
    return _base_dir() / "rag"


def rag_dataset_path(name: str) -> Path:
    return rag_datasets_dir() / name


def save_rag_dataset_meta(meta: RagDatasetMeta) -> None:
    path = rag_dataset_path(meta.name)
    path.mkdir(parents=True, exist_ok=True)
    (path / "dataset.json").write_text(meta.model_dump_json(indent=2))


def load_rag_dataset_meta(name: str) -> RagDatasetMeta:
    path = rag_dataset_path(name) / "dataset.json"
    if not path.exists():
        raise FileNotFoundError(
            f"RAG dataset '{name}' not found. "
            f"Run: supereval rag dataset create {name}"
        )
    return RagDatasetMeta.model_validate_json(path.read_text())


def append_rag_cases(name: str, cases: list[RagTestCase]) -> None:
    path = rag_dataset_path(name) / "cases.jsonl"
    with path.open("a", encoding="utf-8") as f:
        for case in cases:
            f.write(case.model_dump_json() + "\n")


def load_rag_cases(name: str) -> list[RagTestCase]:
    path = rag_dataset_path(name) / "cases.jsonl"
    if not path.exists():
        return []
    cases, errors = [], []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            cases.append(RagTestCase.model_validate_json(line))
        except Exception as e:
            errors.append(f"  Line {i}: {e}")
    if errors:
        raise ValueError(
            f"Validation errors in {path}:\n" + "\n".join(errors)
        )
    return cases


def list_rag_datasets() -> list[RagDatasetMeta]:
    base = rag_datasets_dir()
    if not base.exists():
        return []
    results = []
    for d in sorted(base.iterdir()):
        meta_path = d / "dataset.json"
        if not meta_path.exists():
            continue
        try:
            results.append(RagDatasetMeta.model_validate_json(meta_path.read_text()))
        except Exception:
            continue
    return results
