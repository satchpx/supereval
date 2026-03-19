"""
Data models for agent evaluation.

Agent datasets live under $SUPEREVAL_DATASETS_DIR/agents/<name>/.

Key types
---------
AgentDatasetMeta  — dataset-level config: tool catalog, thresholds
AgentTestCase     — one test case: task input, mock tool responses, expected outcome
Trajectory        — what an agent produced: ordered steps + final answer
TrajectoryScore   — multi-dimensional scoring result
AgentCaseResult   — pairing of a test case with its scored trajectory
AgentRunResult    — all case results for one eval run
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: dict = Field(default_factory=dict)  # JSON Schema


class MockResponse(BaseModel):
    """Canned tool response for a specific test case."""
    response: Any = None
    latency_ms: int = 0
    error: str | None = None  # if set, tool raises RuntimeError(error)


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------

class StepType(str, Enum):
    thought = "thought"
    tool_call = "tool_call"
    observation = "observation"
    answer = "answer"


class ToolCall(BaseModel):
    name: str
    arguments: dict = Field(default_factory=dict)
    result: Any = None
    error: str | None = None
    latency_ms: int = 0


class Step(BaseModel):
    type: StepType
    content: str
    tool_call: ToolCall | None = None
    latency_ms: int = 0
    tokens: int = 0
    cost_usd: float = 0.0


class Trajectory(BaseModel):
    steps: list[Step] = Field(default_factory=list)
    final_answer: str | None = None
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: int = 0
    error: str | None = None  # set if the agent crashed

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [s.tool_call for s in self.steps if s.tool_call is not None]

    @property
    def tool_call_names(self) -> list[str]:
        return [tc.name for tc in self.tool_calls]


# ---------------------------------------------------------------------------
# Test case
# ---------------------------------------------------------------------------

class AnswerMatch(str, Enum):
    exact = "exact"
    contains = "contains"
    regex = "regex"
    llm_judge = "llm_judge"


class MustCallWith(BaseModel):
    """Asserts that a tool was called with at least these arguments (subset match)."""
    tool: str
    args: dict = Field(default_factory=dict)


class AgentExpected(BaseModel):
    answer: str | None = None
    answer_match: AnswerMatch = AnswerMatch.contains
    rubric: str | None = None          # used when answer_match = "llm_judge"
    must_call: list[str] = Field(default_factory=list)
    must_not_call: list[str] = Field(default_factory=list)
    must_call_with: list[MustCallWith] = Field(default_factory=list)
    max_steps: int | None = None
    max_tool_calls: int | None = None


class AgentTask(BaseModel):
    task: str
    context: dict = Field(default_factory=dict)


class AgentTestCase(BaseModel):
    id: str = Field(default_factory=lambda: _new_id("atc"))
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    difficulty: Literal["easy", "medium", "hard"] | None = None
    input: AgentTask
    tools: dict[str, MockResponse] = Field(default_factory=dict)
    expected: AgentExpected = Field(default_factory=AgentExpected)
    metadata: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dataset meta
# ---------------------------------------------------------------------------

class AgentThresholds(BaseModel):
    pass_rate: float = 1.0                       # minimum fraction of cases that must pass
    min_answer_score: float = 0.8
    min_tool_score: float = 1.0
    min_reasoning_score: float | None = None     # None = skip reasoning judge
    fail_on_regression: bool = True
    max_cost_usd: float | None = None
    max_p95_latency_ms: int | None = None
    # Composite score weights (must sum to ~1.0)
    answer_weight: float = 0.40
    tool_weight: float = 0.40
    reasoning_weight: float = 0.15
    efficiency_weight: float = 0.05


class AgentDatasetMeta(BaseModel):
    id: str = Field(default_factory=lambda: _new_id("ads"))
    name: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    tools: list[ToolSpec] = Field(default_factory=list)
    thresholds: AgentThresholds = Field(default_factory=AgentThresholds)
    created_at: str = Field(default_factory=lambda: date.today().isoformat())
    updated_at: str = Field(default_factory=lambda: date.today().isoformat())
    author: str = ""


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryScore:
    answer_score: float = 0.0
    tool_score: float = 0.0
    efficiency_score: float = 1.0
    reasoning_score: float | None = None
    composite_score: float = 0.0
    passed: bool = False
    # Detail
    missing_tools: list[str] = field(default_factory=list)
    forbidden_tools_found: list[str] = field(default_factory=list)
    missing_args: list[str] = field(default_factory=list)
    step_count: int = 0
    tool_call_count: int = 0
    answer_reason: str = ""
    reasoning_reason: str = ""
    failure_reasons: list[str] = field(default_factory=list)
    # Per-metric detail from judge (populated when judge is used)
    correctness_raw: int | None = None
    correctness_label: str = ""
    completeness_raw: int | None = None
    completeness_label: str = ""
    helpfulness_raw: int | None = None
    helpfulness_label: str = ""
    faithfulness_raw: int | None = None
    faithfulness_label: str = ""
    coherence_raw: int | None = None
    coherence_label: str = ""


# ---------------------------------------------------------------------------
# Run results
# ---------------------------------------------------------------------------

@dataclass
class AgentCaseResult:
    case_id: str
    description: str
    vars: dict                    # {"task": ...} — for store compatibility
    trajectory: Trajectory
    score: TrajectoryScore
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
            "passed": self.passed,
            "composite_score": round(self.score.composite_score, 4),
            "answer_score": round(self.score.answer_score, 4),
            "tool_score": round(self.score.tool_score, 4),
            "efficiency_score": round(self.score.efficiency_score, 4),
            "reasoning_score": (
                round(self.score.reasoning_score, 4)
                if self.score.reasoning_score is not None
                else None
            ),
            "failure_reasons": self.score.failure_reasons,
            "step_count": self.score.step_count,
            "tool_call_count": self.score.tool_call_count,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
        }
        # Include per-metric judge detail when available
        if self.score.correctness_label:
            d["correctness"] = {
                "raw": self.score.correctness_raw,
                "label": self.score.correctness_label,
            }
        if self.score.completeness_label:
            d["completeness"] = {
                "raw": self.score.completeness_raw,
                "label": self.score.completeness_label,
            }
        if self.score.helpfulness_label:
            d["helpfulness"] = {
                "raw": self.score.helpfulness_raw,
                "label": self.score.helpfulness_label,
            }
        if self.score.faithfulness_label:
            d["faithfulness"] = {
                "raw": self.score.faithfulness_raw,
                "label": self.score.faithfulness_label,
            }
        if self.score.coherence_label:
            d["logical_coherence"] = {
                "raw": self.score.coherence_raw,
                "label": self.score.coherence_label,
            }
        return d


@dataclass
class AgentRunResult:
    dataset: str
    cases: list[AgentCaseResult]
    runner_id: str = ""           # import path of the AgentRunner used
    run_id: str = field(default_factory=lambda: f"agent_{uuid.uuid4().hex[:8]}")
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

    @property
    def avg_answer_score(self) -> float:
        if not self.cases:
            return 0.0
        return sum(c.score.answer_score for c in self.cases) / len(self.cases)

    @property
    def avg_tool_score(self) -> float:
        if not self.cases:
            return 0.0
        return sum(c.score.tool_score for c in self.cases) / len(self.cases)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "dataset": self.dataset,
            "ran_at": self.ran_at,
            "runner_id": self.runner_id,
            "summary": {
                "total": self.total,
                "passed": self.passed,
                "failed": self.failed,
                "pass_rate": round(self.pass_rate, 4),
                "avg_composite_score": round(self.avg_composite_score, 4),
                "avg_answer_score": round(self.avg_answer_score, 4),
                "avg_tool_score": round(self.avg_tool_score, 4),
                "total_cost_usd": round(self.total_cost_usd, 6),
                "avg_latency_ms": round(self.avg_latency_ms, 1),
                "p95_latency_ms": round(self.p95_latency_ms, 1),
            },
            "cases": [c.to_dict() for c in self.cases],
        }
