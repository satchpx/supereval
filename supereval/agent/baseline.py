"""
Baseline save/load/compare for agent eval runs.

Mirrors the LLM baseline.py pattern. baseline.json is committed to git
alongside cases.jsonl and updated by --update-baseline on push to main.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .models import AgentRunResult
from .storage import agent_dataset_path, load_agent_dataset_meta


def _baseline_path(dataset_name: str) -> Path:
    return agent_dataset_path(dataset_name) / "baseline.json"


def load_agent_baseline(dataset_name: str) -> dict | None:
    path = _baseline_path(dataset_name)
    return json.loads(path.read_text()) if path.exists() else None


def save_agent_baseline(result: AgentRunResult) -> None:
    _baseline_path(result.dataset).write_text(
        json.dumps(result.to_dict(), indent=2)
    )


@dataclass
class AgentRegressionReport:
    passed: bool
    current_pass_rate: float
    baseline_pass_rate: float
    pass_rate_delta: float
    regressions: list[dict] = field(default_factory=list)
    failure_reasons: list[str] = field(default_factory=list)


def compare_agent_to_baseline(result: AgentRunResult) -> AgentRegressionReport | None:
    """
    Compare an AgentRunResult against the stored baseline.
    Returns None if no baseline exists yet (first run).

    Failure conditions (all configurable via dataset thresholds):
      - fail_on_regression:  any case that previously passed now fails
      - pass_rate:           overall pass rate below minimum threshold
      - max_cost_usd:        total run cost exceeds threshold
      - max_p95_latency_ms:  p95 latency exceeds threshold
    """
    baseline = load_agent_baseline(result.dataset)
    if baseline is None:
        return None

    meta = load_agent_dataset_meta(result.dataset)
    thresholds = meta.thresholds

    baseline_pass_rate = baseline.get("summary", {}).get("pass_rate", 1.0)
    baseline_cases = {c["case_id"]: c for c in baseline.get("cases", [])}
    pass_rate_delta = result.pass_rate - baseline_pass_rate

    regressions = [
        {
            "case_id": case.case_id,
            "description": case.description,
            "baseline_score": baseline_cases[case.case_id]["composite_score"],
            "current_score": case.score.composite_score,
            "failure_reasons": case.score.failure_reasons,
        }
        for case in result.cases
        if case.case_id in baseline_cases
        and baseline_cases[case.case_id]["passed"]
        and not case.passed
    ]

    failure_reasons: list[str] = []

    if thresholds.fail_on_regression and regressions:
        failure_reasons.append(
            f"{len(regressions)} previously-passing case(s) now fail"
        )

    if result.pass_rate < thresholds.pass_rate:
        failure_reasons.append(
            f"Pass rate {result.pass_rate:.1%} is below required threshold "
            f"{thresholds.pass_rate:.1%}"
        )

    if (
        thresholds.max_cost_usd is not None
        and result.total_cost_usd > thresholds.max_cost_usd
    ):
        failure_reasons.append(
            f"Total cost ${result.total_cost_usd:.4f} exceeds threshold "
            f"${thresholds.max_cost_usd:.4f}"
        )

    if (
        thresholds.max_p95_latency_ms is not None
        and result.p95_latency_ms > thresholds.max_p95_latency_ms
    ):
        failure_reasons.append(
            f"p95 latency {result.p95_latency_ms:.0f}ms exceeds threshold "
            f"{thresholds.max_p95_latency_ms}ms"
        )

    return AgentRegressionReport(
        passed=len(failure_reasons) == 0,
        current_pass_rate=result.pass_rate,
        baseline_pass_rate=baseline_pass_rate,
        pass_rate_delta=pass_rate_delta,
        regressions=regressions,
        failure_reasons=failure_reasons,
    )
