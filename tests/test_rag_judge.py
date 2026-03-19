"""Tests for RagJudge — all Bedrock calls are mocked."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from supereval.rag.judge import MetricScore, RagJudge, RagJudgeResult


def _mock_response(reasoning: str, answer: str) -> dict:
    body_text = json.dumps({"reasoning": reasoning, "answer": answer})
    mock_body = MagicMock()
    mock_body.read.return_value = json.dumps(
        {"content": [{"text": body_text}]}
    ).encode()
    return {"body": mock_body}


def _make_judge() -> RagJudge:
    return RagJudge(model_id="test-model", region="us-east-1")


class TestRagJudge:
    def _two_responses(self, f_reason, f_answer, c_reason, c_answer):
        return [
            _mock_response(f_reason, f_answer),
            _mock_response(c_reason, c_answer),
        ]

    def test_judge_returns_rag_judge_result(self):
        judge = _make_judge()
        responses = self._two_responses("Fully grounded.", "all", "Correct.", "correct")
        with patch.object(judge, "_get_client") as mock_client:
            mock_client.return_value.invoke_model.side_effect = responses
            result = judge.judge(
                query="What region?",
                answer="us-west-2",
                contexts=["my-bucket is in us-west-2."],
                ground_truth="us-west-2",
            )
        assert isinstance(result, RagJudgeResult)
        assert isinstance(result.faithfulness, MetricScore)
        assert isinstance(result.answer_correctness, MetricScore)

    def test_faithfulness_all(self):
        judge = _make_judge()
        responses = self._two_responses("Fully grounded.", "all", "Correct.", "correct")
        with patch.object(judge, "_get_client") as mock_client:
            mock_client.return_value.invoke_model.side_effect = responses
            result = judge.judge("q", "a", ["ctx"], "gt")
        assert result.faithfulness.raw == 4
        assert result.faithfulness.label == "all"
        assert result.faithfulness.normalized == pytest.approx(1.0)
        assert result.faithfulness.max_raw == 4

    def test_faithfulness_some(self):
        judge = _make_judge()
        responses = self._two_responses("Some hallucination.", "some", "Wrong.", "incorrect")
        with patch.object(judge, "_get_client") as mock_client:
            mock_client.return_value.invoke_model.side_effect = responses
            result = judge.judge("q", "a", ["ctx"], "gt")
        assert result.faithfulness.raw == 1
        assert result.faithfulness.normalized == pytest.approx(0.25)

    def test_answer_correctness_correct(self):
        judge = _make_judge()
        responses = self._two_responses("Grounded.", "most", "Correct.", "correct")
        with patch.object(judge, "_get_client") as mock_client:
            mock_client.return_value.invoke_model.side_effect = responses
            result = judge.judge("q", "a", ["ctx"], "gt")
        assert result.answer_correctness.raw == 2
        assert result.answer_correctness.label == "correct"
        assert result.answer_correctness.normalized == pytest.approx(1.0)
        assert result.answer_correctness.max_raw == 2

    def test_answer_correctness_partial(self):
        judge = _make_judge()
        responses = self._two_responses("Grounded.", "all", "Partial.", "partially correct")
        with patch.object(judge, "_get_client") as mock_client:
            mock_client.return_value.invoke_model.side_effect = responses
            result = judge.judge("q", "a", ["ctx"], "gt")
        assert result.answer_correctness.raw == 1
        assert result.answer_correctness.normalized == pytest.approx(0.5)

    def test_score_is_average_of_both_metrics(self):
        judge = _make_judge()
        # faithfulness=most (0.75) + correctness=correct (1.0) → avg 0.875
        responses = self._two_responses(".", "most", ".", "correct")
        with patch.object(judge, "_get_client") as mock_client:
            mock_client.return_value.invoke_model.side_effect = responses
            result = judge.judge("q", "a", ["ctx"], "gt")
        assert result.score == pytest.approx(0.875)

    def test_makes_two_bedrock_calls(self):
        judge = _make_judge()
        responses = self._two_responses(".", "all", ".", "correct")
        with patch.object(judge, "_get_client") as mock_client:
            mock_client.return_value.invoke_model.side_effect = responses
            judge.judge("q", "a", ["ctx"], "gt")
        assert mock_client.return_value.invoke_model.call_count == 2

    def test_contexts_formatted_in_prompt(self):
        judge = _make_judge()
        captured = []

        def capture(**kwargs):
            captured.append(json.loads(kwargs["body"])["messages"][0]["content"])
            mock_body = MagicMock()
            mock_body.read.return_value = json.dumps({
                "content": [{"text": json.dumps({"reasoning": "ok", "answer": "all"})}]
            }).encode()
            return {"body": mock_body}

        with patch.object(judge, "_get_client") as mock_client:
            mock_client.return_value.invoke_model.side_effect = capture
            judge.judge("q", "a", ["context chunk one", "context chunk two"], "gt")
        # First call is faithfulness — should include the contexts
        assert "context chunk one" in captured[0]
        assert "context chunk two" in captured[0]

    def test_empty_contexts_handled(self):
        judge = _make_judge()
        responses = self._two_responses("No contexts.", "none", "Wrong.", "incorrect")
        with patch.object(judge, "_get_client") as mock_client:
            mock_client.return_value.invoke_model.side_effect = responses
            result = judge.judge("q", "a", [], "gt")
        assert isinstance(result, RagJudgeResult)

    def test_malformed_response_falls_back_to_midpoint(self):
        judge = _make_judge()
        bad_body = MagicMock()
        bad_body.read.return_value = json.dumps(
            {"content": [{"text": "not json at all"}]}
        ).encode()
        bad_resp = {"body": bad_body}
        with patch.object(judge, "_get_client") as mock_client:
            mock_client.return_value.invoke_model.side_effect = [bad_resp, bad_resp]
            result = judge.judge("q", "a", ["ctx"], "gt")
        assert 0.0 <= result.score <= 1.0

    def test_strips_markdown_fences(self):
        judge = _make_judge()
        fenced = "```json\n" + json.dumps({"reasoning": "ok", "answer": "all"}) + "\n```"
        fenced_body = MagicMock()
        fenced_body.read.return_value = json.dumps(
            {"content": [{"text": fenced}]}
        ).encode()
        good_body = MagicMock()
        good_body.read.return_value = json.dumps({
            "content": [{"text": json.dumps({"reasoning": "ok", "answer": "correct"})}]
        }).encode()
        with patch.object(judge, "_get_client") as mock_client:
            mock_client.return_value.invoke_model.side_effect = [
                {"body": fenced_body}, {"body": good_body}
            ]
            result = judge.judge("q", "a", ["ctx"], "gt")
        assert result.faithfulness.label == "all"
        assert result.answer_correctness.label == "correct"
