"""Tests for the DuckDB-backed run history store."""
from __future__ import annotations

import pytest

from supereval.runner import CaseResult, RunResult
from supereval.store import get_run, get_run_cases, get_stats, list_runs, record_run


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Each test gets its own DuckDB file so tests don't bleed into each other."""
    monkeypatch.setenv("SUPEREVAL_DB_PATH", str(tmp_path / "test.db"))


def _make_result(
    run_id: str = "run_abc123",
    dataset: str = "test-qa",
    passed: list[bool] | None = None,
    costs: list[float] | None = None,
    latencies: list[int] | None = None,
    total_tokens: int = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> RunResult:
    passed = passed or [True, True, False]
    costs = costs or [0.0] * len(passed)
    latencies = latencies or [100] * len(passed)
    cases = [
        CaseResult(
            vars={"query": f"q{i}"},
            passed=p,
            score=1.0 if p else 0.0,
            cost_usd=c,
            latency_ms=lat,
        )
        for i, (p, c, lat) in enumerate(zip(passed, costs, latencies))
    ]
    return RunResult(
        dataset=dataset,
        cases=cases,
        providers=["test-model"],
        run_id=run_id,
        total_tokens=total_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


class TestRecordRun:
    def test_record_creates_run_row(self):
        result = _make_result()
        record_run(result)
        row = get_run(result.run_id)
        assert row is not None
        assert row["run_id"] == result.run_id
        assert row["dataset"] == "test-qa"
        assert row["total_cases"] == 3
        assert row["passed"] == 2
        assert row["failed"] == 1

    def test_record_stores_case_results(self):
        result = _make_result()
        record_run(result)
        cases = get_run_cases(result.run_id)
        assert len(cases) == 3
        passed_count = sum(1 for c in cases if c["passed"])
        assert passed_count == 2

    def test_record_is_idempotent_on_run_id(self):
        result = _make_result()
        record_run(result)
        record_run(result)  # second call should not raise
        runs = list_runs(dataset="test-qa")
        assert len(runs) == 1

    def test_record_stores_cost(self):
        result = _make_result(costs=[0.001, 0.002, 0.003])
        record_run(result)
        row = get_run(result.run_id)
        assert row["total_cost_usd"] == pytest.approx(0.006, abs=1e-6)

    def test_record_stores_latency(self):
        result = _make_result(latencies=[100, 200, 300])
        record_run(result)
        row = get_run(result.run_id)
        assert row["avg_latency_ms"] == pytest.approx(200.0, abs=1.0)
        assert row["p95_latency_ms"] == pytest.approx(300.0, abs=1.0)

    def test_record_stores_token_counts(self):
        result = _make_result(total_tokens=500, prompt_tokens=300, completion_tokens=200)
        record_run(result)
        row = get_run(result.run_id)
        assert row["total_tokens"] == 500
        assert row["prompt_tokens"] == 300
        assert row["completion_tokens"] == 200

    def test_record_stores_pass_rate(self):
        result = _make_result(passed=[True, True, False])  # 2/3
        record_run(result)
        row = get_run(result.run_id)
        assert row["pass_rate"] == pytest.approx(2 / 3, abs=0.001)


class TestListRuns:
    def test_returns_empty_list_when_no_runs(self):
        assert list_runs() == []

    def test_returns_recorded_runs(self):
        record_run(_make_result(run_id="run_001"))
        record_run(_make_result(run_id="run_002"))
        runs = list_runs()
        assert len(runs) == 2

    def test_filters_by_dataset(self):
        record_run(_make_result(run_id="run_qa", dataset="qa-ds"))
        record_run(_make_result(run_id="run_cls", dataset="cls-ds"))
        qa_runs = list_runs(dataset="qa-ds")
        assert len(qa_runs) == 1
        assert qa_runs[0]["dataset"] == "qa-ds"

    def test_respects_limit(self):
        for i in range(5):
            record_run(_make_result(run_id=f"run_{i:03d}"))
        runs = list_runs(limit=3)
        assert len(runs) == 3

    def test_ordered_most_recent_first(self):
        record_run(_make_result(run_id="run_first"))
        record_run(_make_result(run_id="run_second"))
        runs = list_runs()
        # ran_at is ISO timestamp; second run should come first
        assert runs[0]["run_id"] == "run_second"


class TestGetRun:
    def test_returns_none_when_missing(self):
        assert get_run("nonexistent_run") is None

    def test_returns_full_row(self):
        result = _make_result()
        record_run(result)
        row = get_run(result.run_id)
        assert row is not None
        # Should include all columns including p50
        assert "p50_latency_ms" in row
        assert "prompt_tokens" in row


class TestGetRunCases:
    def test_returns_empty_when_no_cases(self):
        assert get_run_cases("nonexistent_run") == []

    def test_returns_cases_in_order(self):
        result = _make_result()
        record_run(result)
        cases = get_run_cases(result.run_id)
        assert len(cases) == 3
        for c in cases:
            assert "vars" in c
            assert "passed" in c
            assert "score" in c
            assert "latency_ms" in c
            assert "cost_usd" in c

    def test_vars_deserialised_as_dict(self):
        result = _make_result()
        record_run(result)
        cases = get_run_cases(result.run_id)
        assert isinstance(cases[0]["vars"], dict)


class TestGetStats:
    def test_returns_empty_when_no_runs(self):
        stats = get_stats("nonexistent-ds")
        assert stats == {} or not stats.get("run_count")

    def test_aggregates_pass_rate(self):
        # Two runs: 100% and 0% → avg 50%
        record_run(_make_result(run_id="run_a", passed=[True, True]))
        record_run(_make_result(run_id="run_b", passed=[False, False]))
        stats = get_stats("test-qa", last=10)
        assert stats["run_count"] == 2
        assert stats["avg_pass_rate"] == pytest.approx(0.5, abs=0.01)
        assert stats["min_pass_rate"] == pytest.approx(0.0, abs=0.01)
        assert stats["max_pass_rate"] == pytest.approx(1.0, abs=0.01)

    def test_aggregates_cost(self):
        record_run(_make_result(run_id="run_a", costs=[0.01, 0.02]))
        record_run(_make_result(run_id="run_b", costs=[0.05, 0.10]))
        stats = get_stats("test-qa", last=10)
        assert stats["total_cost_usd"] == pytest.approx(0.18, abs=1e-5)

    def test_last_parameter_limits_window(self):
        for i in range(5):
            record_run(_make_result(run_id=f"run_{i:03d}", passed=[True, True]))
        stats = get_stats("test-qa", last=3)
        assert stats["run_count"] == 3
