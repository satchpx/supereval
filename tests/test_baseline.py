"""Tests for baseline save/load/compare logic."""
from __future__ import annotations

import json

import pytest

from supereval.baseline import compare_to_baseline, load_baseline, save_baseline
from supereval.models import Thresholds
from supereval.runner import CaseResult, RunResult
from supereval.storage import load_dataset_meta, save_dataset_meta


def _make_result(dataset: str, passed: list[bool], scores: list[float] | None = None) -> RunResult:
    scores = scores or [1.0 if p else 0.0 for p in passed]
    queries = [f"query_{i}" for i in range(len(passed))]
    cases = [
        CaseResult(vars={"query": q}, passed=p, score=s)
        for q, p, s in zip(queries, passed, scores)
    ]
    return RunResult(dataset=dataset, cases=cases, providers=["test-model"])


class TestSaveLoadBaseline:
    def test_roundtrip(self, qa_meta, sample_run_result):
        save_baseline(sample_run_result)
        loaded = load_baseline("test-qa")
        assert loaded is not None
        assert loaded["run_id"] == sample_run_result.run_id
        assert loaded["summary"]["total"] == 3
        assert loaded["summary"]["passed"] == 2
        assert abs(loaded["summary"]["pass_rate"] - 2 / 3) < 0.01

    def test_load_returns_none_when_missing(self, qa_meta):
        assert load_baseline("test-qa") is None


class TestCompareToBaseline:
    def test_returns_none_when_no_baseline(self, qa_meta):
        result = _make_result("test-qa", [True, True])
        report = compare_to_baseline(result)
        assert report is None

    def test_passes_when_no_regressions(self, qa_meta, sample_run_result):
        # Baseline: 2/3 pass. New run: same or better.
        save_baseline(sample_run_result)
        perfect = _make_result("test-qa", [True, True, True])
        # Override vars to match the baseline keys
        perfect.cases = sample_run_result.cases[:]
        for c in perfect.cases:
            c.passed = True
            c.score = 1.0
        report = compare_to_baseline(perfect)
        assert report is not None
        assert report.passed is True
        assert report.regressions == []

    def test_fails_on_regression(self, qa_meta, sample_run_result):
        save_baseline(sample_run_result)
        # New run: first case regresses from passed → failed
        regressed = RunResult(
            dataset="test-qa",
            providers=["test-model"],
            cases=[
                CaseResult(
                    vars=sample_run_result.cases[0].vars,
                    passed=False,  # was True in baseline
                    score=0.0,
                ),
                CaseResult(
                    vars=sample_run_result.cases[1].vars,
                    passed=True,
                    score=1.0,
                ),
                CaseResult(
                    vars=sample_run_result.cases[2].vars,
                    passed=False,
                    score=0.2,
                ),
            ],
        )
        report = compare_to_baseline(regressed)
        assert report is not None
        assert report.passed is False
        assert len(report.regressions) == 1
        assert "previously-passing" in report.failure_reasons[0]

    def test_fails_when_pass_rate_below_threshold(self, qa_meta):
        from supereval.models import DatasetMeta, DatasetType
        meta = load_dataset_meta("test-qa")
        meta.thresholds.pass_rate = 0.9
        meta.thresholds.fail_on_regression = False
        save_dataset_meta(meta)

        baseline = _make_result("test-qa", [True, True, True, True, True])
        save_baseline(baseline)

        poor = _make_result("test-qa", [True, False, False, False, False])
        # Override keys to match baseline
        for i, c in enumerate(poor.cases):
            c.vars = baseline.cases[i].vars

        report = compare_to_baseline(poor)
        assert report is not None
        assert report.passed is False
        assert any("threshold" in r for r in report.failure_reasons)

    def test_fails_when_max_score_drop_exceeded(self, qa_meta):
        meta = load_dataset_meta("test-qa")
        meta.thresholds.max_score_drop = 0.1
        meta.thresholds.fail_on_regression = False
        meta.thresholds.pass_rate = 0.0
        save_dataset_meta(meta)

        baseline = _make_result("test-qa", [True, True, True, True, True])
        save_baseline(baseline)

        # Drop pass rate by 40% — exceeds the 10% max
        dropped = _make_result("test-qa", [True, False, False, False, False])
        for i, c in enumerate(dropped.cases):
            c.vars = baseline.cases[i].vars

        report = compare_to_baseline(dropped)
        assert report is not None
        assert report.passed is False
        assert any("drop" in r for r in report.failure_reasons)

    def test_passes_when_fail_on_regression_disabled(self, qa_meta, sample_run_result):
        meta = load_dataset_meta("test-qa")
        meta.thresholds.fail_on_regression = False
        meta.thresholds.pass_rate = 0.0
        save_dataset_meta(meta)

        save_baseline(sample_run_result)

        # All cases fail but regression check is disabled
        all_fail = RunResult(
            dataset="test-qa",
            providers=["test-model"],
            cases=[
                CaseResult(vars=c.vars, passed=False, score=0.0)
                for c in sample_run_result.cases
            ],
        )
        report = compare_to_baseline(all_fail)
        assert report is not None
        assert report.passed is True

    def test_fails_when_cost_exceeds_threshold(self, qa_meta):
        meta = load_dataset_meta("test-qa")
        meta.thresholds.max_cost_usd = 0.05
        meta.thresholds.fail_on_regression = False
        meta.thresholds.pass_rate = 0.0
        save_dataset_meta(meta)

        baseline = _make_result("test-qa", [True, True])
        save_baseline(baseline)

        expensive = RunResult(
            dataset="test-qa",
            providers=["test-model"],
            cases=[
                CaseResult(vars=c.vars, passed=True, score=1.0, cost_usd=0.04)
                for c in baseline.cases
            ],
        )
        report = compare_to_baseline(expensive)
        assert report is not None
        assert report.passed is False
        assert any("cost" in r.lower() for r in report.failure_reasons)

    def test_passes_when_cost_within_threshold(self, qa_meta):
        meta = load_dataset_meta("test-qa")
        meta.thresholds.max_cost_usd = 0.50
        meta.thresholds.fail_on_regression = False
        meta.thresholds.pass_rate = 0.0
        save_dataset_meta(meta)

        baseline = _make_result("test-qa", [True, True])
        save_baseline(baseline)

        cheap = RunResult(
            dataset="test-qa",
            providers=["test-model"],
            cases=[
                CaseResult(vars=c.vars, passed=True, score=1.0, cost_usd=0.001)
                for c in baseline.cases
            ],
        )
        report = compare_to_baseline(cheap)
        assert report is not None
        assert report.passed is True

    def test_fails_when_p95_latency_exceeds_threshold(self, qa_meta):
        meta = load_dataset_meta("test-qa")
        meta.thresholds.max_p95_latency_ms = 500
        meta.thresholds.fail_on_regression = False
        meta.thresholds.pass_rate = 0.0
        save_dataset_meta(meta)

        baseline = _make_result("test-qa", [True, True])
        save_baseline(baseline)

        slow = RunResult(
            dataset="test-qa",
            providers=["test-model"],
            cases=[
                CaseResult(vars=c.vars, passed=True, score=1.0, latency_ms=800)
                for c in baseline.cases
            ],
        )
        report = compare_to_baseline(slow)
        assert report is not None
        assert report.passed is False
        assert any("latency" in r.lower() for r in report.failure_reasons)

    def test_passes_when_p95_latency_within_threshold(self, qa_meta):
        meta = load_dataset_meta("test-qa")
        meta.thresholds.max_p95_latency_ms = 1000
        meta.thresholds.fail_on_regression = False
        meta.thresholds.pass_rate = 0.0
        save_dataset_meta(meta)

        baseline = _make_result("test-qa", [True, True])
        save_baseline(baseline)

        fast = RunResult(
            dataset="test-qa",
            providers=["test-model"],
            cases=[
                CaseResult(vars=c.vars, passed=True, score=1.0, latency_ms=200)
                for c in baseline.cases
            ],
        )
        report = compare_to_baseline(fast)
        assert report is not None
        assert report.passed is True

    def test_pass_rate_delta_calculated_correctly(self, qa_meta, sample_run_result):
        save_baseline(sample_run_result)  # pass_rate = 2/3 ≈ 0.667

        better = RunResult(
            dataset="test-qa",
            providers=["test-model"],
            cases=[
                CaseResult(vars=c.vars, passed=True, score=1.0)
                for c in sample_run_result.cases
            ],
        )
        report = compare_to_baseline(better)
        assert report is not None
        assert report.pass_rate_delta > 0
