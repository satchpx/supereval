"""
Storage for agent datasets.

Agent datasets live under $SUPEREVAL_DATASETS_DIR/agents/<name>/
to avoid collisions with LLM datasets.

Each dataset directory contains:
  dataset.json   — AgentDatasetMeta
  cases.jsonl    — one AgentTestCase per line
  baseline.json  — committed to git; updated by --update-baseline
"""
from __future__ import annotations

import os
from pathlib import Path

from .models import AgentDatasetMeta, AgentTestCase


def _base_dir() -> Path:
    env = os.environ.get("SUPEREVAL_DATASETS_DIR")
    return Path(env) if env else Path.cwd() / "datasets"


def agent_datasets_dir() -> Path:
    return _base_dir() / "agents"


def agent_dataset_path(name: str) -> Path:
    return agent_datasets_dir() / name


def save_agent_dataset_meta(meta: AgentDatasetMeta) -> None:
    path = agent_dataset_path(meta.name)
    path.mkdir(parents=True, exist_ok=True)
    (path / "dataset.json").write_text(meta.model_dump_json(indent=2))


def load_agent_dataset_meta(name: str) -> AgentDatasetMeta:
    path = agent_dataset_path(name) / "dataset.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Agent dataset '{name}' not found. "
            f"Run: supereval agent dataset create {name}"
        )
    return AgentDatasetMeta.model_validate_json(path.read_text())


def append_agent_cases(name: str, cases: list[AgentTestCase]) -> None:
    path = agent_dataset_path(name) / "cases.jsonl"
    with path.open("a", encoding="utf-8") as f:
        for case in cases:
            f.write(case.model_dump_json() + "\n")


def load_agent_cases(name: str) -> list[AgentTestCase]:
    path = agent_dataset_path(name) / "cases.jsonl"
    if not path.exists():
        return []
    cases, errors = [], []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            cases.append(AgentTestCase.model_validate_json(line))
        except Exception as e:
            errors.append(f"  Line {i}: {e}")
    if errors:
        raise ValueError(
            f"Validation errors in {path}:\n" + "\n".join(errors)
        )
    return cases


def list_agent_datasets() -> list[AgentDatasetMeta]:
    base = agent_datasets_dir()
    if not base.exists():
        return []
    results = []
    for d in sorted(base.iterdir()):
        meta_path = d / "dataset.json"
        if not meta_path.exists():
            continue
        try:
            results.append(AgentDatasetMeta.model_validate_json(meta_path.read_text()))
        except Exception:
            continue
    return results
