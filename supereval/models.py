from __future__ import annotations

import uuid
from datetime import date
from enum import Enum
from typing import Literal, Union

from pydantic import BaseModel, Field


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


class DatasetType(str, Enum):
    qa = "qa"
    classification = "classification"
    instruction = "instruction"
    rag = "rag"


class Thresholds(BaseModel):
    pass_rate: float = 1.0                  # minimum pass rate required (0.0–1.0)
    max_score_drop: float | None = None     # max allowed pass rate drop vs baseline (e.g. 0.05 = 5%)
    fail_on_regression: bool = True         # fail if any previously-passing case now fails
    max_cost_usd: float | None = None       # fail if total run cost exceeds this (USD)
    max_p95_latency_ms: int | None = None   # fail if p95 latency exceeds this (ms)


class DatasetMeta(BaseModel):
    id: str = Field(default_factory=lambda: _new_id("ds"))
    name: str
    type: DatasetType
    description: str = ""
    tags: list[str] = []
    labels: list[str] | None = None  # valid labels for classification datasets
    thresholds: Thresholds = Field(default_factory=Thresholds)
    created_at: str = Field(default_factory=lambda: date.today().isoformat())
    updated_at: str = Field(default_factory=lambda: date.today().isoformat())
    author: str = ""


# --- Input models ---

class QAInput(BaseModel):
    query: str


class ClassificationInput(BaseModel):
    text: str


class InstructionInput(BaseModel):
    instruction: str
    document: str | None = None


# --- Expected output models ---

class QAExpected(BaseModel):
    ground_truth: str


class ClassificationExpected(BaseModel):
    label: str


class InstructionExpected(BaseModel):
    rubric: str


# --- Test case models ---

class _TestCaseBase(BaseModel):
    id: str = Field(default_factory=lambda: _new_id("tc"))
    description: str = ""
    tags: list[str] = []
    difficulty: Literal["easy", "medium", "hard"] | None = None
    metadata: dict = Field(default_factory=dict)


class QATestCase(_TestCaseBase):
    input: QAInput
    expected: QAExpected


class ClassificationTestCase(_TestCaseBase):
    input: ClassificationInput
    expected: ClassificationExpected


class InstructionTestCase(_TestCaseBase):
    input: InstructionInput
    expected: InstructionExpected


TestCase = Union[QATestCase, ClassificationTestCase, InstructionTestCase]

TYPE_TO_MODEL: dict[DatasetType, type[TestCase]] = {
    DatasetType.qa: QATestCase,
    DatasetType.classification: ClassificationTestCase,
    DatasetType.instruction: InstructionTestCase,
    # DatasetType.rag is handled by supereval.rag.models.RagTestCase, not this map.
    # It lives in a separate subpackage to keep LLM eval and RAG eval concerns apart.
}
