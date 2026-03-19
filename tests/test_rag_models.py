"""Tests for RAG data models — schema validation, computed properties, serialisation."""
from __future__ import annotations

import pytest
from supereval.rag.models import (
    RagCaseResult,
    RagDatasetMeta,
    RagExpected,
    RagInput,
    RagRunResult,
    RagScore,
    RagTestCase,
    RagThresholds,
)


class TestRagTestCase:
    def test_valid_case(self):
        case = RagTestCase(
            description="S3 bucket region",
            input=RagInput(
                query="What region is my-bucket in?",
                retrieved_contexts=["my-bucket is in us-west-2."],
            ),
            expected=RagExpected(ground_truth="us-west-2"),
        )
        assert case.input.query == "What region is my-bucket in?"
        assert len(case.input.retrieved_contexts) == 1
        assert case.expected.ground_truth == "us-west-2"

    def test_id_auto_generated(self):
        case = RagTestCase(
            input=RagInput(query="q"),
            expected=RagExpected(ground_truth="a"),
        )
        assert case.id.startswith("rtc_")

    def test_empty_contexts_allowed(self):
        case = RagTestCase(
            input=RagInput(query="q", retrieved_contexts=[]),
            expected=RagExpected(ground_truth="a"),
        )
        assert case.input.retrieved_contexts == []

    def test_roundtrip_json(self):
        case = RagTestCase(
            description="test",
            input=RagInput(query="q", retrieved_contexts=["ctx1", "ctx2"]),
            expected=RagExpected(ground_truth="a"),
            tags=["s3"],
            difficulty="easy",
        )
        restored = RagTestCase.model_validate_json(case.model_dump_json())
        assert restored.input.query == case.input.query
        assert restored.input.retrieved_contexts == ["ctx1", "ctx2"]
        assert restored.expected.ground_truth == case.expected.ground_truth


class TestRagDatasetMeta:
    def test_defaults(self):
        meta = RagDatasetMeta(name="my-rag")
        assert meta.name == "my-rag"
        assert meta.id.startswith("rds_")
        assert isinstance(meta.thresholds, RagThresholds)

    def test_thresholds_defaults(self):
        t = RagThresholds()
        assert t.pass_rate == 1.0
        assert t.require_contains is True
        assert t.min_answer_correctness_score == 0.5
        assert t.min_faithfulness_score is None
        assert t.faithfulness_weight == 0.5
        assert t.answer_correctness_weight == 0.5

    def test_roundtrip_json(self):
        meta = RagDatasetMeta(name="my-rag", description="test", tags=["aws"])
        restored = RagDatasetMeta.model_validate_json(meta.model_dump_json())
        assert restored.name == "my-rag"
        assert restored.tags == ["aws"]


class TestRagScore:
    def test_defaults(self):
        s = RagScore()
        assert s.contains_score == 0.0
        assert s.faithfulness_score == 1.0
        assert s.passed is False
        assert s.failure_reasons == []

    def test_passed_when_no_failures(self):
        s = RagScore(
            contains_score=1.0,
            answer_correctness_score=0.8,
            faithfulness_score=0.9,
            composite_score=0.85,
            passed=True,
        )
        assert s.passed is True


class TestRagCaseResult:
    def _make_case(self, passed: bool = True) -> RagCaseResult:
        score = RagScore(
            contains_score=1.0,
            answer_correctness_score=1.0,
            faithfulness_score=1.0,
            composite_score=1.0,
            passed=passed,
        )
        return RagCaseResult(
            case_id="rtc_abc123",
            description="test case",
            vars={"query": "What region?"},
            answer="us-west-2",
            score=score,
            latency_ms=150,
        )

    def test_passed_property(self):
        assert self._make_case(passed=True).passed is True
        assert self._make_case(passed=False).passed is False

    def test_to_dict_includes_required_fields(self):
        d = self._make_case().to_dict()
        assert d["case_id"] == "rtc_abc123"
        assert d["answer"] == "us-west-2"
        assert d["passed"] is True
        assert "composite_score" in d
        assert "faithfulness_score" in d
        assert "answer_correctness_score" in d
        assert "contains_score" in d

    def test_to_dict_includes_judge_detail_when_set(self):
        score = RagScore(
            contains_score=1.0,
            answer_correctness_score=1.0,
            faithfulness_score=1.0,
            composite_score=1.0,
            passed=True,
            faithfulness_raw=4,
            faithfulness_label="all",
            faithfulness_reason="Fully grounded.",
            answer_correctness_raw=2,
            answer_correctness_label="correct",
            answer_correctness_reason="Correct.",
        )
        case = RagCaseResult(
            case_id="rtc_x",
            description="",
            vars={},
            answer="ans",
            score=score,
        )
        d = case.to_dict()
        assert d["faithfulness"]["label"] == "all"
        assert d["answer_correctness"]["label"] == "correct"

    def test_to_dict_omits_judge_detail_when_empty(self):
        d = self._make_case().to_dict()
        assert "faithfulness" not in d
        assert "answer_correctness" not in d


class TestRagRunResult:
    def _make_result(self, pass_flags: list[bool]) -> RagRunResult:
        cases = []
        for i, passed in enumerate(pass_flags):
            score = RagScore(composite_score=1.0 if passed else 0.0, passed=passed)
            cases.append(RagCaseResult(
                case_id=f"rtc_{i}",
                description="",
                vars={},
                answer="ans",
                score=score,
                latency_ms=100 * (i + 1),
            ))
        return RagRunResult(dataset="test-rag", cases=cases, model_id="test-model")

    def test_pass_rate(self):
        r = self._make_result([True, True, False])
        assert r.total == 3
        assert r.passed == 2
        assert r.failed == 1
        assert pytest.approx(r.pass_rate) == 2 / 3

    def test_all_pass(self):
        r = self._make_result([True, True])
        assert r.pass_rate == 1.0

    def test_p95_latency(self):
        r = self._make_result([True, True, True, True])
        assert r.p95_latency_ms > 0

    def test_run_id_starts_with_rag(self):
        r = self._make_result([True])
        assert r.run_id.startswith("rag_")

    def test_to_dict_shape(self):
        r = self._make_result([True, False])
        d = r.to_dict()
        assert d["dataset"] == "test-rag"
        assert d["model_id"] == "test-model"
        assert "summary" in d
        assert "cases" in d
        assert len(d["cases"]) == 2
