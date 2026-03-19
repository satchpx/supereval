"""Tests for BedrockJudge — all Bedrock calls are mocked."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, call, patch

import pytest
from supereval.agent.judge import (
    AnswerJudgeResult,
    BedrockJudge,
    MetricScore,
    ReasoningJudgeResult,
    _format_trajectory,
)
from supereval.agent.models import Step, StepType, ToolCall, Trajectory


def _make_trajectory(answer: str = "us-west-2") -> Trajectory:
    return Trajectory(
        steps=[
            Step(type=StepType.thought, content="Let me search"),
            Step(
                type=StepType.tool_call,
                content="searching",
                tool_call=ToolCall(
                    name="search",
                    arguments={"query": "bucket region"},
                    result=[{"region": "us-west-2"}],
                ),
            ),
            Step(type=StepType.observation, content="found it"),
            Step(type=StepType.answer, content=answer),
        ],
        final_answer=answer,
    )


def _mock_metric_response(reasoning: str, answer: str):
    """Build a mock boto3 response for a label-based metric call."""
    body_text = json.dumps({"reasoning": reasoning, "answer": answer})
    mock_body = MagicMock()
    mock_body.read.return_value = json.dumps({
        "content": [{"text": body_text}]
    }).encode()
    return {"body": mock_body}


class TestFormatTrajectory:
    def test_formats_thought_steps(self):
        traj = Trajectory(steps=[Step(type=StepType.thought, content="thinking")])
        text = _format_trajectory(traj)
        assert "[thought]" in text
        assert "thinking" in text

    def test_formats_tool_call_with_result(self):
        traj = Trajectory(steps=[
            Step(
                type=StepType.tool_call,
                content="",
                tool_call=ToolCall(name="search", arguments={"q": "s3"}, result=["r"]),
            )
        ])
        text = _format_trajectory(traj)
        assert "search" in text
        assert '"q": "s3"' in text
        assert "r" in text

    def test_formats_tool_call_with_error(self):
        traj = Trajectory(steps=[
            Step(
                type=StepType.tool_call,
                content="",
                tool_call=ToolCall(name="delete", arguments={}, error="AccessDenied"),
            )
        ])
        text = _format_trajectory(traj)
        assert "ERROR: AccessDenied" in text

    def test_empty_trajectory(self):
        text = _format_trajectory(Trajectory())
        assert text == "(no steps)"


class TestBedrockJudge:
    def _judge(self) -> BedrockJudge:
        return BedrockJudge(model_id="test-model", region="us-east-1")

    def _answer_responses(self, c_reasoning, c_answer, comp_reasoning, comp_answer, h_reasoning, h_answer):
        """Build side_effect list for three sequential invoke_model calls (answer judge)."""
        return [
            _mock_metric_response(c_reasoning, c_answer),
            _mock_metric_response(comp_reasoning, comp_answer),
            _mock_metric_response(h_reasoning, h_answer),
        ]

    def _two_responses(self, r1_reasoning, r1_answer, r2_reasoning, r2_answer):
        """Build side_effect list for two sequential invoke_model calls (reasoning judge)."""
        return [
            _mock_metric_response(r1_reasoning, r1_answer),
            _mock_metric_response(r2_reasoning, r2_answer),
        ]

    # ------------------------------------------------------------------
    # judge_answer
    # ------------------------------------------------------------------

    def test_judge_answer_returns_answer_judge_result(self):
        judge = self._judge()
        responses = self._answer_responses(
            "Answer is fully correct.", "correct",
            "Answer covers the question completely.", "yes",
            "Very actionable response.", "helpful",
        )
        with patch.object(judge, "_get_client") as mock_client:
            mock_client.return_value.invoke_model.side_effect = responses
            result = judge.judge_answer(
                task="What region?",
                trajectory=_make_trajectory(),
                expected_answer="us-west-2",
            )
        assert isinstance(result, AnswerJudgeResult)
        assert isinstance(result.correctness, MetricScore)
        assert isinstance(result.completeness, MetricScore)
        assert isinstance(result.helpfulness, MetricScore)

    def test_judge_answer_correctness_correct(self):
        judge = self._judge()
        responses = self._answer_responses(
            "Correct.", "correct",
            "Complete.", "yes",
            "Helpful.", "helpful",
        )
        with patch.object(judge, "_get_client") as mock_client:
            mock_client.return_value.invoke_model.side_effect = responses
            result = judge.judge_answer(
                task="t", trajectory=_make_trajectory(), expected_answer="us-west-2"
            )
        assert result.correctness.raw == 2
        assert result.correctness.label == "correct"
        assert result.correctness.normalized == pytest.approx(1.0)
        assert result.correctness.max_raw == 2

    def test_judge_answer_correctness_partial(self):
        judge = self._judge()
        responses = self._answer_responses(
            "Partially right.", "partially correct",
            "Mostly complete.", "generally yes",
            "Somewhat helpful.", "helpful",
        )
        with patch.object(judge, "_get_client") as mock_client:
            mock_client.return_value.invoke_model.side_effect = responses
            result = judge.judge_answer(
                task="t", trajectory=_make_trajectory(), expected_answer="x"
            )
        assert result.correctness.raw == 1
        assert result.correctness.normalized == pytest.approx(0.5)
        assert result.completeness.raw == 3
        assert result.completeness.normalized == pytest.approx(0.75)

    def test_judge_answer_helpfulness_score(self):
        judge = self._judge()
        responses = self._answer_responses(
            "Correct.", "correct",
            "Complete.", "yes",
            "Very helpful.", "very helpful",
        )
        with patch.object(judge, "_get_client") as mock_client:
            mock_client.return_value.invoke_model.side_effect = responses
            result = judge.judge_answer(
                task="t", trajectory=_make_trajectory(), expected_answer="x"
            )
        assert result.helpfulness.raw == 4
        assert result.helpfulness.label == "very helpful"
        assert result.helpfulness.normalized == pytest.approx(1.0)
        assert result.helpfulness.max_raw == 4

    def test_judge_answer_score_is_average_of_metrics(self):
        judge = self._judge()
        # correctness=correct(1.0) + completeness=neutral/mixed(0.5) + helpfulness=helpful(0.75)
        # avg = (1.0 + 0.5 + 0.75) / 3 = 0.75
        responses = self._answer_responses(
            "ok", "correct", "mixed", "neutral/mixed", "ok", "helpful"
        )
        with patch.object(judge, "_get_client") as mock_client:
            mock_client.return_value.invoke_model.side_effect = responses
            result = judge.judge_answer(
                task="t", trajectory=_make_trajectory(), expected_answer="x"
            )
        assert result.score == pytest.approx(0.75)

    def test_judge_answer_without_expected_answer(self):
        """expected_answer is optional — judge should still run."""
        judge = self._judge()
        responses = self._answer_responses(
            "Best effort.", "partially correct",
            "Incomplete.", "not generally",
            "Somewhat helpful.", "slightly helpful",
        )
        with patch.object(judge, "_get_client") as mock_client:
            mock_client.return_value.invoke_model.side_effect = responses
            result = judge.judge_answer(task="t", trajectory=_make_trajectory())
        assert isinstance(result, AnswerJudgeResult)
        assert result.score >= 0.0

    def test_judge_answer_rubric_in_prompt(self):
        judge = self._judge()
        captured = []

        def capture(**kwargs):
            captured.append(json.loads(kwargs["body"])["messages"][0]["content"])
            mock_body = MagicMock()
            mock_body.read.return_value = json.dumps({
                "content": [{"text": json.dumps({"reasoning": "ok", "answer": "correct"})}]
            }).encode()
            return {"body": mock_body}

        with patch.object(judge, "_get_client") as mock_client:
            mock_client.return_value.invoke_model.side_effect = capture
            judge.judge_answer(
                task="t",
                trajectory=_make_trajectory(),
                expected_answer="x",
                rubric="Must be a valid AWS region.",
            )
        # Rubric should appear in the correctness prompt (first call)
        assert "Must be a valid AWS region." in captured[0]

    def test_judge_answer_malformed_response_returns_midpoint(self):
        judge = self._judge()
        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps({
            "content": [{"text": "not json at all"}]
        }).encode()
        bad_resp = {"body": mock_body}
        with patch.object(judge, "_get_client") as mock_client:
            mock_client.return_value.invoke_model.side_effect = [bad_resp, bad_resp, bad_resp]
            result = judge.judge_answer(
                task="t", trajectory=_make_trajectory(), expected_answer="x"
            )
        # Should not raise; score should be the midpoint fallback
        assert 0.0 <= result.score <= 1.0

    # ------------------------------------------------------------------
    # judge_reasoning
    # ------------------------------------------------------------------

    def test_judge_reasoning_returns_reasoning_judge_result(self):
        judge = self._judge()
        responses = self._two_responses(
            "All grounded in tool results.", "all",
            "Steps follow logically.", "yes",
        )
        with patch.object(judge, "_get_client") as mock_client:
            mock_client.return_value.invoke_model.side_effect = responses
            result = judge.judge_reasoning(task="t", trajectory=_make_trajectory())
        assert isinstance(result, ReasoningJudgeResult)
        assert isinstance(result.faithfulness, MetricScore)
        assert isinstance(result.logical_coherence, MetricScore)

    def test_judge_reasoning_faithfulness_all(self):
        judge = self._judge()
        responses = self._two_responses("Fully grounded.", "all", "Coherent.", "generally yes")
        with patch.object(judge, "_get_client") as mock_client:
            mock_client.return_value.invoke_model.side_effect = responses
            result = judge.judge_reasoning(task="t", trajectory=_make_trajectory())
        assert result.faithfulness.raw == 4
        assert result.faithfulness.label == "all"
        assert result.faithfulness.normalized == pytest.approx(1.0)
        assert result.faithfulness.max_raw == 4

    def test_judge_reasoning_faithfulness_some(self):
        judge = self._judge()
        responses = self._two_responses("Some hallucination.", "some", "Ok logic.", "neutral/mixed")
        with patch.object(judge, "_get_client") as mock_client:
            mock_client.return_value.invoke_model.side_effect = responses
            result = judge.judge_reasoning(task="t", trajectory=_make_trajectory())
        assert result.faithfulness.raw == 1
        assert result.faithfulness.normalized == pytest.approx(0.25)

    def test_judge_reasoning_score_is_average_of_metrics(self):
        judge = self._judge()
        # faithfulness=most (0.75) + coherence=generally yes (0.75) → avg 0.75
        responses = self._two_responses("Mostly faithful.", "most", "Mostly coherent.", "generally yes")
        with patch.object(judge, "_get_client") as mock_client:
            mock_client.return_value.invoke_model.side_effect = responses
            result = judge.judge_reasoning(task="t", trajectory=_make_trajectory())
        assert result.score == pytest.approx(0.75)

    def test_judge_reasoning_makes_two_bedrock_calls(self):
        judge = self._judge()
        responses = self._two_responses("ok", "all", "ok", "yes")
        with patch.object(judge, "_get_client") as mock_client:
            mock_client.return_value.invoke_model.side_effect = responses
            judge.judge_reasoning(task="t", trajectory=_make_trajectory())
        assert mock_client.return_value.invoke_model.call_count == 2

    def test_judge_answer_makes_three_bedrock_calls(self):
        judge = self._judge()
        responses = self._answer_responses("ok", "correct", "ok", "yes", "ok", "helpful")
        with patch.object(judge, "_get_client") as mock_client:
            mock_client.return_value.invoke_model.side_effect = responses
            judge.judge_answer(task="t", trajectory=_make_trajectory(), expected_answer="x")
        assert mock_client.return_value.invoke_model.call_count == 3

    def test_strips_markdown_fences(self):
        judge = self._judge()
        fenced = "```json\n" + json.dumps({"reasoning": "ok", "answer": "correct"}) + "\n```"
        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps({"content": [{"text": fenced}]}).encode()
        good_body = MagicMock()
        good_body.read.return_value = json.dumps({
            "content": [{"text": json.dumps({"reasoning": "ok", "answer": "yes"})}]
        }).encode()
        helpful_body = MagicMock()
        helpful_body.read.return_value = json.dumps({
            "content": [{"text": json.dumps({"reasoning": "ok", "answer": "helpful"})}]
        }).encode()
        with patch.object(judge, "_get_client") as mock_client:
            mock_client.return_value.invoke_model.side_effect = [
                {"body": mock_body}, {"body": good_body}, {"body": helpful_body}
            ]
            result = judge.judge_answer(
                task="t", trajectory=_make_trajectory(), expected_answer="x"
            )
        assert result.correctness.label == "correct"
