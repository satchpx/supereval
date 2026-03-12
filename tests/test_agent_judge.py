"""Tests for BedrockJudge — all Bedrock calls are mocked."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from supereval.agent.judge import BedrockJudge, JudgeResult, _format_trajectory
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


def _mock_bedrock_response(score: float, reason: str):
    """Build a mock boto3 bedrock-runtime response."""
    body_text = json.dumps({"score": score, "reason": reason})
    mock_body = MagicMock()
    mock_body.read.return_value = json.dumps({
        "content": [{"text": body_text}]
    }).encode()
    mock_response = {"body": mock_body}
    return mock_response


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

    def test_judge_answer_returns_score(self):
        judge = self._judge()
        mock_resp = _mock_bedrock_response(0.9, "Correct answer")
        with patch.object(judge, "_get_client") as mock_client:
            mock_client.return_value.invoke_model.return_value = mock_resp
            result = judge.judge_answer(
                task="What region?",
                trajectory=_make_trajectory(),
                expected_answer="us-west-2",
            )
        assert isinstance(result, JudgeResult)
        assert result.score == pytest.approx(0.9)
        assert result.reason == "Correct answer"

    def test_judge_reasoning_returns_score(self):
        judge = self._judge()
        mock_resp = _mock_bedrock_response(0.8, "Good reasoning")
        with patch.object(judge, "_get_client") as mock_client:
            mock_client.return_value.invoke_model.return_value = mock_resp
            result = judge.judge_reasoning(
                task="What region?",
                trajectory=_make_trajectory(),
            )
        assert result.score == pytest.approx(0.8)
        assert "reasoning" in result.reason.lower() or result.reason

    def test_score_clamped_to_0_1(self):
        judge = self._judge()
        # Model returns out-of-range score
        mock_resp = _mock_bedrock_response(1.5, "too high")
        with patch.object(judge, "_get_client") as mock_client:
            mock_client.return_value.invoke_model.return_value = mock_resp
            result = judge.judge_answer(
                task="t", trajectory=_make_trajectory(), expected_answer="x"
            )
        assert result.score <= 1.0

    def test_malformed_response_returns_fallback(self):
        judge = self._judge()
        # Response is not valid JSON
        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps({
            "content": [{"text": "not json at all"}]
        }).encode()
        with patch.object(judge, "_get_client") as mock_client:
            mock_client.return_value.invoke_model.return_value = {"body": mock_body}
            result = judge.judge_answer(
                task="t", trajectory=_make_trajectory(), expected_answer="x"
            )
        assert result.score == pytest.approx(0.5)
        assert "Could not parse" in result.reason

    def test_strips_markdown_fences(self):
        judge = self._judge()
        fenced_text = "```json\n" + json.dumps({"score": 0.7, "reason": "ok"}) + "\n```"
        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps({
            "content": [{"text": fenced_text}]
        }).encode()
        with patch.object(judge, "_get_client") as mock_client:
            mock_client.return_value.invoke_model.return_value = {"body": mock_body}
            result = judge.judge_answer(
                task="t", trajectory=_make_trajectory(), expected_answer="x"
            )
        assert result.score == pytest.approx(0.7)

    def test_rubric_included_in_answer_judge_prompt(self):
        """Verify the rubric is forwarded to Bedrock when provided."""
        judge = self._judge()
        captured_calls = []

        def capture_invoke(**kwargs):
            captured_calls.append(kwargs)
            mock_body = MagicMock()
            mock_body.read.return_value = json.dumps({
                "content": [{"text": json.dumps({"score": 1.0, "reason": "ok"})}]
            }).encode()
            return {"body": mock_body}

        with patch.object(judge, "_get_client") as mock_client:
            mock_client.return_value.invoke_model.side_effect = capture_invoke
            judge.judge_answer(
                task="t",
                trajectory=_make_trajectory(),
                expected_answer="x",
                rubric="Must be a valid AWS region.",
            )

        assert captured_calls
        body = json.loads(captured_calls[0]["body"])
        prompt = body["messages"][0]["content"]
        assert "Must be a valid AWS region." in prompt
