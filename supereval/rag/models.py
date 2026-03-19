"""
Data models for RAG evaluation.

RAG datasets live under $SUPEREVAL_DATASETS_DIR/rag/<name>/.

Key types
---------
RagDatasetMeta  — dataset-level config + thresholds
RagTestCase     — one test case: query + static retrieved_contexts + ground_truth
RagScore        — per-case scoring result (faithfulness + answer_correctness + contains)
RagCaseResult   — test case paired with its scored result
RagRunResult    — all case results for one eval run
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Test case schema
# ---------------------------------------------------------------------------

class RagInput(BaseModel):
    query: str
    retrieved_contexts: list[str] = Field(default_factory=list)


class RagExpected(BaseModel):
    ground_truth: str


class RagTestCase(BaseModel):
    id: str = Field(default_factory=lambda: _new_id("rtc"))
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    difficulty: Literal["easy", "medium", "hard"] | None = None
    input: RagInput
    expected: RagExpected
    metadata: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dataset metadata + thresholds
# ---------------------------------------------------------------------------

class RagThresholds(BaseModel):
    pass_rate: float = 1.0
    fail_on_regression: bool = True
    # Hard fail if the ground_truth string is not found in the answer (icontains).
    # Useful as a cheap fast signal even when no judge is configured.
    require_contains: bool = True
    # Judge-backed thresholds — only enforced when --judge-model is provided.
    # None = scored but never causes a case to fail (informational only).
    min_faithfulness_score: float | None = None
    min_answer_correctness_score: float = 0.5
    # Cost / latency gates (same pattern as LLM + agent eval)
    max_cost_usd: float | None = None
    max_p95_latency_ms: int | None = None
    # Composite score weights (should sum to 1.0; only apply when judge is used)
    faithfulness_weight: float = 0.5
    answer_correctness_weight: float = 0.5


class RagDatasetMeta(BaseModel):
    id: str = Field(default_factory=lambda: _new_id("rds"))
    name: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    thresholds: RagThresholds = Field(default_factory=RagThresholds)
    created_at: str = Field(default_factory=lambda: date.today().isoformat())
    updated_at: str = Field(default_factory=lambda: date.today().isoformat())
    author: str = ""


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass
class RagScore:
    # Core dimensions
    contains_score: float = 0.0          # icontains of ground_truth in answer (0 or 1)
    answer_correctness_score: float = 0.0  # judge or falls back to contains
    faithfulness_score: float = 1.0      # judge; defaults to 1.0 when no judge
    composite_score: float = 0.0         # weighted avg of answer_correctness + faithfulness
    passed: bool = False
    failure_reasons: list[str] = field(default_factory=list)
    # Judge detail (populated when judge is used)
    faithfulness_raw: int | None = None
    faithfulness_label: str = ""
    faithfulness_reason: str = ""
    answer_correctness_raw: int | None = None
    answer_correctness_label: str = ""
    answer_correctness_reason: str = ""


# ---------------------------------------------------------------------------
# Run results
# ---------------------------------------------------------------------------

@dataclass
class RagCaseResult:
    case_id: str
    description: str
    vars: dict          # {"query": ..., "contexts": ...} for store compatibility
    answer: str         # LLM's response
    score: RagScore
    latency_ms: int = 0
    cost_usd: float = 0.0

    @property
    def passed(self) -> bool:
        return self.score.passed

    def to_dict(self) -> dict:
        d: dict = {
            "case_id": self.case_id,
            "description": self.description,
            "vars": self.vars,
            "answer": self.answer,
            "passed": self.passed,
            "composite_score": round(self.score.composite_score, 4),
            "contains_score": round(self.score.contains_score, 4),
            "answer_correctness_score": round(self.score.answer_correctness_score, 4),
            "faithfulness_score": round(self.score.faithfulness_score, 4),
            "failure_reasons": self.score.failure_reasons,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
        }
        if self.score.faithfulness_label:
            d["faithfulness"] = {
                "raw": self.score.faithfulness_raw,
                "label": self.score.faithfulness_label,
                "reason": self.score.faithfulness_reason,
            }
        if self.score.answer_correctness_label:
            d["answer_correctness"] = {
                "raw": self.score.answer_correctness_raw,
                "label": self.score.answer_correctness_label,
                "reason": self.score.answer_correctness_reason,
            }
        return d


@dataclass
class RagRunResult:
    dataset: str
    cases: list[RagCaseResult]
    model_id: str = ""
    run_id: str = field(default_factory=lambda: f"rag_{uuid.uuid4().hex[:8]}")
    ran_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.cases if c.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    @property
    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.cases)

    @property
    def avg_latency_ms(self) -> float:
        if not self.cases:
            return 0.0
        return sum(c.latency_ms for c in self.cases) / len(self.cases)

    @property
    def p50_latency_ms(self) -> float:
        if not self.cases:
            return 0.0
        latencies = sorted(c.latency_ms for c in self.cases)
        return float(latencies[len(latencies) // 2])

    @property
    def p95_latency_ms(self) -> float:
        if not self.cases:
            return 0.0
        latencies = sorted(c.latency_ms for c in self.cases)
        idx = min(int(len(latencies) * 0.95), len(latencies) - 1)
        return float(latencies[idx])

    @property
    def avg_composite_score(self) -> float:
        if not self.cases:
            return 0.0
        return sum(c.score.composite_score for c in self.cases) / len(self.cases)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "dataset": self.dataset,
            "ran_at": self.ran_at,
            "model_id": self.model_id,
            "summary": {
                "total": self.total,
                "passed": self.passed,
                "failed": self.failed,
                "pass_rate": round(self.pass_rate, 4),
                "avg_composite_score": round(self.avg_composite_score, 4),
                "total_cost_usd": round(self.total_cost_usd, 6),
                "avg_latency_ms": round(self.avg_latency_ms, 1),
                "p95_latency_ms": round(self.p95_latency_ms, 1),
            },
            "cases": [c.to_dict() for c in self.cases],
        }
