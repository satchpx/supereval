"""Tests for run_rag_eval — all model and judge calls are mocked."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from supereval.rag.models import (
    RagDatasetMeta,
    RagExpected,
    RagInput,
    RagRunResult,
    RagTestCase,
)
from supereval.rag.runner import (
    DEFAULT_PROMPT_TEMPLATE,
    _call_model,
    _format_contexts,
    run_rag_eval,
)
from supereval.rag.storage import append_rag_cases, save_rag_dataset_meta


@pytest.fixture
def rag_dataset(datasets_dir):
    meta = RagDatasetMeta(name="test-rag")
    save_rag_dataset_meta(meta)
    cases = [
        RagTestCase(
            id="rtc_001",
            description="Region lookup",
            input=RagInput(
                query="What region is my-bucket in?",
                retrieved_contexts=["my-bucket is in us-west-2."],
            ),
            expected=RagExpected(ground_truth="us-west-2"),
        ),
        RagTestCase(
            id="rtc_002",
            description="Lambda timeout",
            input=RagInput(
                query="What is the max Lambda timeout?",
                retrieved_contexts=["Lambda max timeout is 15 minutes."],
            ),
            expected=RagExpected(ground_truth="15 minutes"),
        ),
    ]
    append_rag_cases("test-rag", cases)
    return meta


def _make_bedrock_response(text: str) -> dict:
    mock_body = MagicMock()
    mock_body.read.return_value = json.dumps(
        {"content": [{"text": text}]}
    ).encode()
    return {"body": mock_body}


class TestFormatContexts:
    def test_formats_multiple_contexts(self):
        result = _format_contexts(["ctx1", "ctx2"])
        assert "[1] ctx1" in result
        assert "[2] ctx2" in result

    def test_empty_contexts_returns_placeholder(self):
        result = _format_contexts([])
        assert "no context" in result.lower()


class TestCallModel:
    def test_bedrock_model_calls_bedrock(self):
        mock_response = _make_bedrock_response("The answer is us-west-2.")
        with patch("boto3.client") as mock_boto:
            mock_client = MagicMock()
            mock_client.invoke_model.return_value = mock_response
            mock_boto.return_value = mock_client
            text, latency = _call_model("us.anthropic.claude-3-sonnet", "us-east-1", "prompt")
        assert text == "The answer is us-west-2."
        assert latency >= 0
        mock_client.invoke_model.assert_called_once()

    def test_anthropic_prefix_calls_anthropic(self):
        mock_module = MagicMock()
        mock_client = MagicMock()
        content_block = MagicMock()
        content_block.text = "answer"
        mock_client.messages.create.return_value = MagicMock(content=[content_block])
        mock_module.Anthropic.return_value = mock_client

        with patch.dict("sys.modules", {"anthropic": mock_module}):
            text, latency = _call_model("anthropic:claude-sonnet-4-6", "us-east-1", "prompt")
        assert text == "answer"

    def test_openai_prefix_calls_openai(self):
        mock_module = MagicMock()
        mock_client = MagicMock()
        choice = MagicMock()
        choice.message.content = "openai answer"
        mock_client.chat.completions.create.return_value = MagicMock(choices=[choice])
        mock_module.OpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_module}):
            text, latency = _call_model("openai:gpt-4o", "us-east-1", "prompt")
        assert text == "openai answer"


class TestRunRagEval:
    def test_run_returns_rag_run_result(self, rag_dataset):
        responses = [
            _make_bedrock_response("The bucket is in us-west-2."),
            _make_bedrock_response("The maximum Lambda timeout is 15 minutes."),
        ]
        with patch("boto3.client") as mock_boto:
            mock_client = MagicMock()
            mock_client.invoke_model.side_effect = responses
            mock_boto.return_value = mock_client
            result = run_rag_eval("test-rag", model_id="test-model")

        assert isinstance(result, RagRunResult)
        assert result.total == 2
        assert result.model_id == "test-model"

    def test_all_cases_scored(self, rag_dataset):
        responses = [
            _make_bedrock_response("The bucket is in us-west-2."),
            _make_bedrock_response("The maximum Lambda timeout is 15 minutes."),
        ]
        with patch("boto3.client") as mock_boto:
            mock_client = MagicMock()
            mock_client.invoke_model.side_effect = responses
            mock_boto.return_value = mock_client
            result = run_rag_eval("test-rag", model_id="test-model")

        assert len(result.cases) == 2
        assert result.cases[0].case_id == "rtc_001"
        assert result.cases[1].case_id == "rtc_002"

    def test_contains_check_passes_for_correct_answers(self, rag_dataset):
        responses = [
            _make_bedrock_response("us-west-2"),
            _make_bedrock_response("15 minutes"),
        ]
        with patch("boto3.client") as mock_boto:
            mock_client = MagicMock()
            mock_client.invoke_model.side_effect = responses
            mock_boto.return_value = mock_client
            result = run_rag_eval("test-rag", model_id="test-model")

        assert result.cases[0].score.contains_score == 1.0
        assert result.cases[1].score.contains_score == 1.0
        assert result.pass_rate == 1.0

    def test_failed_model_call_records_failure(self, rag_dataset):
        with patch("boto3.client") as mock_boto:
            mock_client = MagicMock()
            mock_client.invoke_model.side_effect = RuntimeError("connection error")
            mock_boto.return_value = mock_client
            result = run_rag_eval("test-rag", model_id="test-model")

        for case in result.cases:
            assert case.passed is False
            assert any("Model call failed" in r for r in case.score.failure_reasons)

    def test_no_cases_raises(self, datasets_dir):
        from supereval.rag.storage import save_rag_dataset_meta
        save_rag_dataset_meta(RagDatasetMeta(name="empty-rag"))
        with pytest.raises(ValueError, match="no cases"):
            run_rag_eval("empty-rag", model_id="test-model")

    def test_prompt_includes_query_and_contexts(self, rag_dataset):
        captured_prompts = []

        def capture_call(**kwargs):
            body = json.loads(kwargs["body"])
            captured_prompts.append(body["messages"][0]["content"])
            return _make_bedrock_response("us-west-2")

        with patch("boto3.client") as mock_boto:
            mock_client = MagicMock()
            mock_client.invoke_model.side_effect = capture_call
            mock_boto.return_value = mock_client
            run_rag_eval("test-rag", model_id="test-model")

        assert "What region is my-bucket in?" in captured_prompts[0]
        assert "us-west-2" in captured_prompts[0]

    def test_custom_prompt_template_used(self, rag_dataset):
        captured = []

        def capture(**kwargs):
            captured.append(json.loads(kwargs["body"])["messages"][0]["content"])
            return _make_bedrock_response("answer")

        with patch("boto3.client") as mock_boto:
            mock_client = MagicMock()
            mock_client.invoke_model.side_effect = capture
            mock_boto.return_value = mock_client
            run_rag_eval(
                "test-rag",
                model_id="test-model",
                prompt_template="CONTEXT: {contexts}\nQUESTION: {query}",
            )

        assert "CONTEXT:" in captured[0]
        assert "QUESTION:" in captured[0]

    def test_with_judge(self, rag_dataset):
        model_responses = [
            _make_bedrock_response("us-west-2"),
            _make_bedrock_response("15 minutes"),
        ]
        judge_response_body = json.dumps({"reasoning": "ok", "answer": "all"})
        judge_body = MagicMock()
        judge_body.read.return_value = json.dumps(
            {"content": [{"text": judge_response_body}]}
        ).encode()
        judge_resp = {"body": judge_body}

        correct_response_body = json.dumps({"reasoning": "ok", "answer": "correct"})
        correct_body = MagicMock()
        correct_body.read.return_value = json.dumps(
            {"content": [{"text": correct_response_body}]}
        ).encode()
        correct_resp = {"body": correct_body}

        # Each case makes 2 judge calls (faithfulness + correctness) and 1 model call
        all_responses = []
        for _ in range(2):  # 2 cases
            all_responses.append(model_responses.pop(0))
            all_responses.append(judge_resp)
            all_responses.append(correct_resp)

        with patch("boto3.client") as mock_boto:
            mock_client = MagicMock()
            mock_client.invoke_model.side_effect = all_responses
            mock_boto.return_value = mock_client
            result = run_rag_eval(
                "test-rag",
                model_id="test-model",
                judge_model="test-judge-model",
            )

        for case in result.cases:
            assert case.score.faithfulness_label != ""
            assert case.score.answer_correctness_label != ""
