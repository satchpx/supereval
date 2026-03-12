from __future__ import annotations

import json
from dataclasses import dataclass, field

from .runner import RunResult
from .storage import dataset_path, load_dataset_meta


def _baseline_path(dataset_name: str):
    return dataset_path(dataset_name) / "baseline.json"


def load_baseline(dataset_name: str) -> dict | None:
    path = _baseline_path(dataset_name)
    return json.loads(path.read_text()) if path.exists() else None


def save_baseline(result: RunResult) -> None:
    _baseline_path(result.dataset).write_text(
        json.dumps(result.to_dict(), indent=2)
    )


@dataclass
class RegressionReport:
    passed: bool
    current_pass_rate: float
    baseline_pass_rate: float
    pass_rate_delta: float
    regressions: list[dict] = field(default_factory=list)   # cases that newly fail
    failure_reasons: list[str] = field(default_factory=list)


def compare_to_baseline(result: RunResult) -> RegressionReport | None:
    """
    Compare a RunResult against the stored baseline.
    Returns None if no baseline exists yet (first run).

    Failure conditions (all configurable via dataset thresholds):
      - fail_on_regression:  any case that previously passed now fails
      - pass_rate:           overall pass rate falls below minimum threshold
      - max_score_drop:      pass rate dropped more than N% vs baseline
    """
    baseline = load_baseline(result.dataset)
    if baseline is None:
        return None

    meta = load_dataset_meta(result.dataset)
    thresholds = meta.thresholds

    baseline_pass_rate = baseline.get("summary", {}).get("pass_rate", 1.0)
    baseline_cases = {c["key"]: c for c in baseline.get("cases", [])}
    pass_rate_delta = result.pass_rate - baseline_pass_rate

    # Find cases that previously passed but now fail
    regressions = [
        {
            "vars": case.vars,
            "baseline_score": baseline_cases[case.key]["score"],
            "current_score": case.score,
        }
        for case in result.cases
        if case.key in baseline_cases
        and baseline_cases[case.key]["passed"]
        and not case.passed
    ]

    failure_reasons: list[str] = []

    if thresholds.fail_on_regression and regressions:
        failure_reasons.append(
            f"{len(regressions)} previously-passing case(s) now fail"
        )

    if result.pass_rate < thresholds.pass_rate:
        failure_reasons.append(
            f"Pass rate {result.pass_rate:.1%} is below required threshold {thresholds.pass_rate:.1%}"
        )

    if (
        thresholds.max_score_drop is not None
        and pass_rate_delta < -thresholds.max_score_drop
    ):
        failure_reasons.append(
            f"Pass rate dropped {abs(pass_rate_delta):.1%}, "
            f"exceeds max allowed drop of {thresholds.max_score_drop:.1%}"
        )

    if (
        thresholds.max_cost_usd is not None
        and result.total_cost_usd > thresholds.max_cost_usd
    ):
        failure_reasons.append(
            f"Total cost ${result.total_cost_usd:.4f} exceeds threshold ${thresholds.max_cost_usd:.4f}"
        )

    if (
        thresholds.max_p95_latency_ms is not None
        and result.p95_latency_ms > thresholds.max_p95_latency_ms
    ):
        failure_reasons.append(
            f"p95 latency {result.p95_latency_ms:.0f}ms exceeds threshold {thresholds.max_p95_latency_ms}ms"
        )

    return RegressionReport(
        passed=len(failure_reasons) == 0,
        current_pass_rate=result.pass_rate,
        baseline_pass_rate=baseline_pass_rate,
        pass_rate_delta=pass_rate_delta,
        regressions=regressions,
        failure_reasons=failure_reasons,
    )
