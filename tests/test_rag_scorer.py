"""Tests for score_rag_case — all judge calls are mocked."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from supereval.rag.judge import MetricScore, RagJudgeResult
from supereval.rag.models import RagExpected, RagThresholds
from supereval.rag.scorer import score_rag_case


def _make_judge_result(faithfulness_norm: float, correctness_norm: float) -> RagJudgeResult:
    f_labels = {0.0: ("none", 0), 0.25: ("some", 1), 0.5: ("approximately half", 2),
                0.75: ("most", 3), 1.0: ("all", 4)}
    c_labels = {0.0: ("incorrect", 0), 0.5: ("partially correct", 1), 1.0: ("correct", 2)}
    f_label, f_raw = f_labels.get(faithfulness_norm, ("all", 4))
    c_label, c_raw = c_labels.get(correctness_norm, ("correct", 2))
    return RagJudgeResult(
        faithfulness=MetricScore(raw=f_raw, label=f_label, normalized=faithfulness_norm, max_raw=4),
        answer_correctness=MetricScore(raw=c_raw, label=c_label, normalized=correctness_norm, max_raw=2),
        score=(faithfulness_norm + correctness_norm) / 2,
        reason=f"Faithfulness: ok | Answer Correctness: ok",
    )


def _make_judge(faithfulness: float = 1.0, correctness: float = 1.0):
    mock = MagicMock()
    mock.judge.return_value = _make_judge_result(faithfulness, correctness)
    return mock


class TestScoreRagCaseNoJudge:
    def test_contains_match_passes(self):
        score = score_rag_case(
            query="What region?",
            answer="The bucket is in us-west-2.",
            expected=RagExpected(ground_truth="us-west-2"),
            contexts=["my-bucket is in us-west-2."],
            thresholds=RagThresholds(),
        )
        assert score.contains_score == 1.0
        assert score.answer_correctness_score == 1.0  # fallback to contains
        assert score.faithfulness_score == 1.0        # unverified default
        assert score.passed is True

    def test_contains_miss_fails_when_require_contains(self):
        score = score_rag_case(
            query="What region?",
            answer="I don't know.",
            expected=RagExpected(ground_truth="us-west-2"),
            contexts=[],
            thresholds=RagThresholds(require_contains=True),
        )
        assert score.contains_score == 0.0
        assert score.passed is False
        assert any("ground truth" in r for r in score.failure_reasons)

    def test_contains_miss_passes_when_not_required(self):
        # require_contains=False, min_answer_correctness_score=0 → should pass
        score = score_rag_case(
            query="q",
            answer="some other answer",
            expected=RagExpected(ground_truth="us-west-2"),
            contexts=[],
            thresholds=RagThresholds(require_contains=False, min_answer_correctness_score=0.0),
        )
        assert score.passed is True

    def test_composite_equals_answer_correctness_without_judge(self):
        score = score_rag_case(
            query="q",
            answer="the answer is us-west-2",
            expected=RagExpected(ground_truth="us-west-2"),
            contexts=[],
            thresholds=RagThresholds(),
        )
        assert score.composite_score == pytest.approx(score.answer_correctness_score)

    def test_answer_correctness_below_threshold_fails(self):
        # answer misses → correctness = 0.0, threshold = 0.5 → fail
        score = score_rag_case(
            query="q",
            answer="I don't know.",
            expected=RagExpected(ground_truth="us-west-2"),
            contexts=[],
            thresholds=RagThresholds(require_contains=False, min_answer_correctness_score=0.5),
        )
        assert score.passed is False
        assert any("correctness" in r.lower() for r in score.failure_reasons)


class TestScoreRagCaseWithJudge:
    def test_judge_scores_used(self):
        judge = _make_judge(faithfulness=0.75, correctness=1.0)
        score = score_rag_case(
            query="q",
            answer="us-west-2",
            expected=RagExpected(ground_truth="us-west-2"),
            contexts=["my-bucket is in us-west-2."],
            thresholds=RagThresholds(),
            judge=judge,
        )
        assert score.faithfulness_score == pytest.approx(0.75)
        assert score.answer_correctness_score == pytest.approx(1.0)
        assert score.faithfulness_label == "most"
        assert score.answer_correctness_label == "correct"

    def test_composite_is_weighted_average(self):
        # faithfulness=0.5, correctness=1.0, equal weights → composite=0.75
        judge = _make_judge(faithfulness=0.5, correctness=1.0)
        score = score_rag_case(
            query="q",
            answer="us-west-2",
            expected=RagExpected(ground_truth="us-west-2"),
            contexts=["ctx"],
            thresholds=RagThresholds(faithfulness_weight=0.5, answer_correctness_weight=0.5),
            judge=judge,
        )
        assert score.composite_score == pytest.approx(0.75)

    def test_faithfulness_below_threshold_fails(self):
        judge = _make_judge(faithfulness=0.25, correctness=1.0)
        score = score_rag_case(
            query="q",
            answer="ans",
            expected=RagExpected(ground_truth="ans"),
            contexts=["ctx"],
            thresholds=RagThresholds(min_faithfulness_score=0.5),
            judge=judge,
        )
        assert score.passed is False
        assert any("faithfulness" in r.lower() for r in score.failure_reasons)

    def test_faithfulness_none_threshold_never_fails(self):
        judge = _make_judge(faithfulness=0.0, correctness=1.0)
        score = score_rag_case(
            query="q",
            answer="ans",
            expected=RagExpected(ground_truth="ans"),
            contexts=["ctx"],
            thresholds=RagThresholds(min_faithfulness_score=None, require_contains=True),
            judge=judge,
        )
        # Contains passes (ground_truth in answer), faithfulness unthresholded
        assert not any("faithfulness" in r.lower() for r in score.failure_reasons)

    def test_contains_still_checked_with_judge(self):
        judge = _make_judge(faithfulness=1.0, correctness=1.0)
        score = score_rag_case(
            query="q",
            answer="I don't know.",
            expected=RagExpected(ground_truth="us-west-2"),
            contexts=["ctx"],
            thresholds=RagThresholds(require_contains=True),
            judge=judge,
        )
        # Contains fails even though judge scores are high
        assert score.contains_score == 0.0
        assert score.passed is False

    def test_judge_called_once_per_case(self):
        judge = _make_judge()
        score_rag_case(
            query="q",
            answer="ans",
            expected=RagExpected(ground_truth="ans"),
            contexts=["ctx"],
            thresholds=RagThresholds(),
            judge=judge,
        )
        judge.judge.assert_called_once()
