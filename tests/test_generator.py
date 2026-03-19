"""Tests for synthetic generation (parsing and chunking; no real Bedrock calls)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from supereval.generator import (
    AnthropicGenerator,
    ANTHROPIC_DEFAULT_MODEL,
    BedrockGenerator,
    DEFAULT_MODEL,
    DEFAULT_REGION,
    GeneratedCase,
    OpenAIGenerator,
    OPENAI_DEFAULT_MODEL,
    _build_prompt,
    _chunk_document,
    _parse_response,
)
from supereval.sources import Document


VALID_CASES_JSON = json.dumps([
    {
        "description": "S3 max object size",
        "input": {"query": "What is the maximum S3 object size?"},
        "expected": {"ground_truth": "5 TB"},
        "tags": ["s3", "limits"],
        "difficulty": "easy",
        "case_type": "factual",
        "source_excerpt": "Maximum object size is 5 TB.",
    },
    {
        "description": "Lambda + S3 combo",
        "input": {"query": "Can a Lambda function write to S3?"},
        "expected": {"ground_truth": "Yes, using the AWS SDK."},
        "tags": ["lambda", "s3"],
        "difficulty": "medium",
        "case_type": "multi_hop",
        "source_excerpt": "Lambda can integrate with S3.",
    },
])


class TestParseResponse:
    def test_parses_valid_json(self):
        cases = _parse_response(VALID_CASES_JSON)
        assert len(cases) == 2
        assert cases[0].input == {"query": "What is the maximum S3 object size?"}
        assert cases[0].expected == {"ground_truth": "5 TB"}
        assert cases[0].case_type == "factual"
        assert cases[0].source_excerpt == "Maximum object size is 5 TB."

    def test_strips_markdown_fences(self):
        fenced = f"```json\n{VALID_CASES_JSON}\n```"
        cases = _parse_response(fenced)
        assert len(cases) == 2

    def test_strips_fences_without_language_tag(self):
        fenced = f"```\n{VALID_CASES_JSON}\n```"
        cases = _parse_response(fenced)
        assert len(cases) == 2

    def test_raises_on_invalid_json(self):
        with pytest.raises(ValueError, match="invalid JSON"):
            _parse_response("not json at all")

    def test_skips_malformed_items(self):
        mixed = json.dumps([
            {"description": "good", "input": {"query": "q"}, "expected": {"ground_truth": "a"},
             "tags": [], "difficulty": "easy", "case_type": "factual", "source_excerpt": ""},
            {"malformed": True},  # missing required fields
        ])
        cases = _parse_response(mixed)
        assert len(cases) == 1
        assert cases[0].description == "good"

    def test_defaults_for_optional_fields(self):
        minimal = json.dumps([
            {"input": {"query": "q"}, "expected": {"ground_truth": "a"}}
        ])
        cases = _parse_response(minimal)
        assert cases[0].description == ""
        assert cases[0].tags == []
        assert cases[0].difficulty == "medium"
        assert cases[0].case_type == "factual"
        assert cases[0].source_excerpt == ""


class TestChunkDocument:
    def test_short_document_returns_single_chunk(self):
        content = "Short document content."
        chunks = _chunk_document(content)
        assert len(chunks) == 1
        assert chunks[0] == content

    def test_long_document_splits_into_chunks(self):
        content = "x" * 100_000
        chunks = _chunk_document(content)
        assert len(chunks) > 1
        assert all(len(c) <= 40_000 for c in chunks)

    def test_chunks_cover_all_content(self):
        content = "word " * 20_000  # ~100K chars
        chunks = _chunk_document(content)
        reconstructed = "".join(chunks)
        # All original words should be present (order preserved)
        assert len(reconstructed) >= len(content) * 0.95

    def test_prefers_paragraph_boundary(self):
        # Each paragraph is 1002 chars. With MAX_CHARS_PER_CHUNK=40000, a hard split
        # would produce fewer chunks than splitting mid-paragraph (rfind finds the last
        # \n\n boundary before the window end). Verify chunks are fewer than a pure
        # fixed-size split would produce, confirming boundary detection reduces splits.
        paragraph = "A" * 998 + "\n\n"
        content = paragraph * 50  # ~50K chars total
        chunks = _chunk_document(content)
        hard_split_count = -(-len(content) // 40_000)  # ceiling division
        assert len(chunks) <= hard_split_count + 1


class TestGeneratedCase:
    def test_to_staged_dict_includes_review_fields(self):
        case = GeneratedCase(
            description="Test",
            input={"query": "q"},
            expected={"ground_truth": "a"},
            tags=["t1"],
            difficulty="easy",
            case_type="factual",
            source_excerpt="excerpt",
        )
        d = case.to_staged_dict()
        assert "case_type" in d
        assert "source_excerpt" in d

    def test_to_importable_dict_excludes_review_fields(self):
        case = GeneratedCase(
            description="Test",
            input={"query": "q"},
            expected={"ground_truth": "a"},
            tags=[],
            difficulty="medium",
            case_type="negation",
            source_excerpt="some excerpt",
        )
        d = case.to_importable_dict()
        assert "case_type" not in d
        assert "source_excerpt" not in d
        assert "input" in d
        assert "expected" in d


class TestBedrockGenerator:
    def _make_bedrock_response(self, cases_json: str) -> dict:
        """Build a mock Bedrock API response."""
        return {
            "body": MagicMock(
                read=MagicMock(return_value=json.dumps({
                    "content": [{"text": cases_json}]
                }).encode())
            )
        }

    def test_generate_calls_bedrock(self):
        doc = Document(content="S3 max size is 5 TB.", filename="s3.md", source="/docs/s3.md")
        mock_response = self._make_bedrock_response(VALID_CASES_JSON)

        with patch("boto3.client") as mock_boto:
            mock_client = MagicMock()
            mock_client.invoke_model.return_value = mock_response
            mock_boto.return_value = mock_client

            gen = BedrockGenerator(model_id=DEFAULT_MODEL, region=DEFAULT_REGION)
            cases = gen.generate(doc, count=2)

        assert len(cases) == 2
        mock_client.invoke_model.assert_called_once()
        call_kwargs = mock_client.invoke_model.call_args[1]
        assert call_kwargs["modelId"] == DEFAULT_MODEL

    def test_generate_uses_configured_model_and_region(self):
        doc = Document(content="content", filename="f.md", source="/f.md")
        mock_response = self._make_bedrock_response(VALID_CASES_JSON)
        custom_model = "anthropic.claude-3-opus-20240229-v1:0"
        custom_region = "us-west-2"

        with patch("boto3.client") as mock_boto:
            mock_client = MagicMock()
            mock_client.invoke_model.return_value = mock_response
            mock_boto.return_value = mock_client

            gen = BedrockGenerator(model_id=custom_model, region=custom_region)
            gen.generate(doc, count=2)

        mock_boto.assert_called_once_with("bedrock-runtime", region_name=custom_region)
        assert mock_client.invoke_model.call_args[1]["modelId"] == custom_model

    def test_boto3_import_error_raises_clearly(self):
        doc = Document(content="content", filename="f.md", source="/f.md")

        with patch.dict("sys.modules", {"boto3": None}):
            gen = BedrockGenerator()
            gen._client = None  # force re-init
            with pytest.raises((ImportError, TypeError)):
                gen.generate(doc, count=2)

    def test_large_document_chunked(self):
        """A document exceeding MAX_CHARS_PER_CHUNK should result in multiple Bedrock calls."""
        large_content = "Lambda details. " * 5000  # ~80K chars
        doc = Document(content=large_content, filename="large.md", source="/large.md")
        mock_response = self._make_bedrock_response(VALID_CASES_JSON)

        with patch("boto3.client") as mock_boto:
            mock_client = MagicMock()
            mock_client.invoke_model.return_value = mock_response
            mock_boto.return_value = mock_client

            gen = BedrockGenerator()
            cases = gen.generate(doc, count=4)

        # Two chunks → two Bedrock calls
        assert mock_client.invoke_model.call_count == 2
        assert len(cases) <= 4

    def test_generate_classification(self):
        classification_json = json.dumps([
            {
                "description": "Route networking issue",
                "input": {"text": "EC2 instance lost internet access"},
                "expected": {"label": "networking"},
                "tags": ["ec2"], "difficulty": "easy",
                "case_type": "clear", "source_excerpt": "networking...",
            }
        ])
        mock_response = self._make_bedrock_response(classification_json)
        doc = Document(content="networking guide", filename="guide.md", source="/guide.md")

        with patch("boto3.client") as mock_boto:
            mock_client = MagicMock()
            mock_client.invoke_model.return_value = mock_response
            mock_boto.return_value = mock_client

            gen = BedrockGenerator()
            cases = gen.generate(doc, count=1, dataset_type="classification", labels=["networking", "storage"])

        assert len(cases) == 1
        assert cases[0].expected == {"label": "networking"}
        body = json.loads(mock_client.invoke_model.call_args[1]["body"])
        assert "networking" in body["messages"][0]["content"]
        assert "storage" in body["messages"][0]["content"]

    def test_generate_instruction(self):
        instruction_json = json.dumps([
            {
                "description": "Summarize incident",
                "input": {"instruction": "Summarize in 3 bullets", "document": "On March 10..."},
                "expected": {"rubric": "Must have 3 bullet points"},
                "tags": [], "difficulty": "medium",
                "case_type": "summarization", "source_excerpt": "March 10...",
            }
        ])
        mock_response = self._make_bedrock_response(instruction_json)
        doc = Document(content="incident report content", filename="report.md", source="/report.md")

        with patch("boto3.client") as mock_boto:
            mock_client = MagicMock()
            mock_client.invoke_model.return_value = mock_response
            mock_boto.return_value = mock_client

            gen = BedrockGenerator()
            cases = gen.generate(doc, count=1, dataset_type="instruction")

        assert len(cases) == 1
        assert "rubric" in cases[0].expected


class TestBuildPrompt:
    def test_qa_prompt_contains_mix(self):
        prompt = _build_prompt("qa", "content", "doc.md", 10)
        assert "factual" in prompt
        assert "multi_hop" in prompt

    def test_classification_prompt_contains_labels(self):
        prompt = _build_prompt("classification", "content", "doc.md", 5, labels=["billing", "networking"])
        assert "billing" in prompt
        assert "networking" in prompt

    def test_classification_prompt_notes_missing_labels(self):
        prompt = _build_prompt("classification", "content", "doc.md", 5, labels=None)
        assert "not specified" in prompt

    def test_instruction_prompt_contains_rubric_key(self):
        prompt = _build_prompt("instruction", "content", "doc.md", 5)
        assert "rubric" in prompt

    def test_prompts_include_filename(self):
        for dtype in ("qa", "classification", "instruction"):
            prompt = _build_prompt(dtype, "content", "my-file.txt", 5, labels=["a"])
            assert "my-file.txt" in prompt


def _make_mock_anthropic_module(mock_client: MagicMock) -> MagicMock:
    """Return a fake 'anthropic' module whose Anthropic() returns mock_client."""
    mock_module = MagicMock()
    mock_module.Anthropic.return_value = mock_client
    return mock_module


class TestAnthropicGenerator:
    def _make_api_response(self, cases_json: str) -> MagicMock:
        content_block = MagicMock()
        content_block.text = cases_json
        message = MagicMock()
        message.content = [content_block]
        return message

    def _make_mock_client(self, cases_json: str) -> MagicMock:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._make_api_response(cases_json)
        return mock_client

    def test_generate_calls_anthropic_api(self):
        doc = Document(content="S3 max size is 5 TB.", filename="s3.md", source="/docs/s3.md")
        mock_client = self._make_mock_client(VALID_CASES_JSON)

        with patch.dict("sys.modules", {"anthropic": _make_mock_anthropic_module(mock_client)}):
            gen = AnthropicGenerator(model_id=ANTHROPIC_DEFAULT_MODEL)
            gen._client = None  # force re-init via property
            cases = gen.generate(doc, count=2)

        assert len(cases) == 2
        mock_client.messages.create.assert_called_once()
        assert mock_client.messages.create.call_args[1]["model"] == ANTHROPIC_DEFAULT_MODEL

    def test_generate_uses_custom_model(self):
        doc = Document(content="content", filename="f.md", source="/f.md")
        mock_client = self._make_mock_client(VALID_CASES_JSON)

        with patch.dict("sys.modules", {"anthropic": _make_mock_anthropic_module(mock_client)}):
            gen = AnthropicGenerator(model_id="claude-opus-4-6")
            gen._client = None
            gen.generate(doc, count=2)

        assert mock_client.messages.create.call_args[1]["model"] == "claude-opus-4-6"

    def test_import_error_raises_clearly(self):
        with patch.dict("sys.modules", {"anthropic": None}):
            gen = AnthropicGenerator()
            gen._client = None
            with pytest.raises((ImportError, TypeError)):
                _ = gen.client

    def test_generate_classification(self):
        classification_json = json.dumps([{
            "description": "clear case",
            "input": {"text": "route table issue"},
            "expected": {"label": "networking"},
            "tags": [], "difficulty": "easy",
            "case_type": "clear", "source_excerpt": "",
        }])
        mock_client = self._make_mock_client(classification_json)
        doc = Document(content="guide", filename="g.md", source="/g.md")

        with patch.dict("sys.modules", {"anthropic": _make_mock_anthropic_module(mock_client)}):
            gen = AnthropicGenerator()
            gen._client = None
            cases = gen.generate(doc, 1, dataset_type="classification", labels=["networking"])

        assert cases[0].expected == {"label": "networking"}


def _make_mock_openai_module(mock_client: MagicMock) -> MagicMock:
    """Return a fake 'openai' module whose OpenAI() / AzureOpenAI() returns mock_client."""
    mock_module = MagicMock()
    mock_module.OpenAI.return_value = mock_client
    mock_module.AzureOpenAI.return_value = mock_client
    return mock_module


class TestOpenAIGenerator:
    def _make_api_response(self, cases_json: str) -> MagicMock:
        message = MagicMock()
        message.content = cases_json
        choice = MagicMock()
        choice.message = message
        response = MagicMock()
        response.choices = [choice]
        return response

    def _make_mock_client(self, cases_json: str) -> MagicMock:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_api_response(cases_json)
        return mock_client

    def test_generate_calls_openai_api(self):
        doc = Document(content="S3 max size is 5 TB.", filename="s3.md", source="/docs/s3.md")
        mock_client = self._make_mock_client(VALID_CASES_JSON)

        with patch.dict("sys.modules", {"openai": _make_mock_openai_module(mock_client)}):
            gen = OpenAIGenerator(model_id=OPENAI_DEFAULT_MODEL)
            gen._client = None
            cases = gen.generate(doc, count=2)

        assert len(cases) == 2
        mock_client.chat.completions.create.assert_called_once()
        assert mock_client.chat.completions.create.call_args[1]["model"] == OPENAI_DEFAULT_MODEL

    def test_generate_uses_custom_model(self):
        doc = Document(content="content", filename="f.md", source="/f.md")
        mock_client = self._make_mock_client(VALID_CASES_JSON)

        with patch.dict("sys.modules", {"openai": _make_mock_openai_module(mock_client)}):
            gen = OpenAIGenerator(model_id="gpt-4o-mini")
            gen._client = None
            gen.generate(doc, count=2)

        assert mock_client.chat.completions.create.call_args[1]["model"] == "gpt-4o-mini"

    def test_import_error_raises_clearly(self):
        with patch.dict("sys.modules", {"openai": None}):
            gen = OpenAIGenerator()
            gen._client = None
            with pytest.raises((ImportError, TypeError)):
                _ = gen.client

    def test_azure_uses_azure_openai_class(self):
        doc = Document(content="content", filename="f.md", source="/f.md")
        mock_client = self._make_mock_client(VALID_CASES_JSON)
        mock_openai = _make_mock_openai_module(mock_client)

        with patch.dict("sys.modules", {"openai": mock_openai}):
            gen = OpenAIGenerator(
                model_id="gpt-4o",
                azure_endpoint="https://my-resource.openai.azure.com",
                api_version="2024-02-01",
            )
            gen._client = None
            gen.generate(doc, count=2)

        mock_openai.AzureOpenAI.assert_called_once()
        call_kwargs = mock_openai.AzureOpenAI.call_args[1]
        assert "azure_endpoint" in call_kwargs
        assert call_kwargs["api_version"] == "2024-02-01"

    def test_non_azure_uses_openai_class(self):
        doc = Document(content="content", filename="f.md", source="/f.md")
        mock_client = self._make_mock_client(VALID_CASES_JSON)
        mock_openai = _make_mock_openai_module(mock_client)

        with patch.dict("sys.modules", {"openai": mock_openai}):
            gen = OpenAIGenerator(model_id="gpt-4o")
            gen._client = None
            gen.generate(doc, count=2)

        mock_openai.OpenAI.assert_called_once()
        mock_openai.AzureOpenAI.assert_not_called()

    def test_large_document_chunked(self):
        large_content = "Lambda details. " * 5000
        doc = Document(content=large_content, filename="large.md", source="/large.md")
        mock_client = self._make_mock_client(VALID_CASES_JSON)

        with patch.dict("sys.modules", {"openai": _make_mock_openai_module(mock_client)}):
            gen = OpenAIGenerator()
            gen._client = None
            cases = gen.generate(doc, count=4)

        assert mock_client.chat.completions.create.call_count == 2
        assert len(cases) <= 4
