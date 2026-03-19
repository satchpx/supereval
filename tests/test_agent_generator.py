"""Tests for agent test case generation from documents."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from supereval.agent.generator import (
    AgentAnthropicGenerator,
    AgentBedrockGenerator,
    AgentOpenAIGenerator,
    GeneratedAgentCase,
    _build_agent_prompt,
    _format_tools,
    _parse_agent_response,
)
from supereval.agent.models import AnswerMatch, MustCallWith, ToolSpec
from supereval.sources import Document


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tool_specs():
    return [
        ToolSpec(name="search_kb", description="Search the knowledge base",
                 parameters={"type": "object", "properties": {"query": {"type": "string"}}}),
        ToolSpec(name="list_buckets", description="List S3 buckets",
                 parameters={}),
    ]


@pytest.fixture
def sample_doc():
    return Document(
        filename="s3-guide.md",
        content="The bucket my-data-bucket is in us-west-2. "
                "Use search_kb to find bucket details.",
        source="docs/s3-guide.md",
    )


VALID_RESPONSE = json.dumps([
    {
        "task": "Which region is my-data-bucket in?",
        "description": "Look up bucket region",
        "expected_answer": "us-west-2",
        "must_call": ["search_kb"],
        "must_call_with": [{"tool": "search_kb", "args": {"query": "my-data-bucket"}}],
        "max_steps": 4,
        "difficulty": "easy",
        "tags": ["s3"],
        "source_excerpt": "The bucket my-data-bucket is in us-west-2.",
    },
    {
        "task": "List all buckets",
        "description": "List available S3 buckets",
        "expected_answer": None,
        "must_call": ["list_buckets"],
        "must_call_with": [],
        "max_steps": 3,
        "difficulty": "easy",
        "tags": ["s3"],
        "source_excerpt": "Use list_buckets to list all buckets.",
    },
])


# ---------------------------------------------------------------------------
# _format_tools
# ---------------------------------------------------------------------------

class TestFormatTools:
    def test_includes_tool_names(self, tool_specs):
        result = _format_tools(tool_specs)
        assert "search_kb" in result
        assert "list_buckets" in result

    def test_includes_descriptions(self, tool_specs):
        result = _format_tools(tool_specs)
        assert "Search the knowledge base" in result

    def test_empty_tools_fallback(self):
        result = _format_tools([])
        assert "no tools defined" in result


# ---------------------------------------------------------------------------
# _build_agent_prompt
# ---------------------------------------------------------------------------

class TestBuildAgentPrompt:
    def test_includes_count(self, tool_specs, sample_doc):
        prompt = _build_agent_prompt(sample_doc.content, sample_doc.filename, 5, tool_specs)
        assert "5" in prompt

    def test_includes_document(self, tool_specs, sample_doc):
        prompt = _build_agent_prompt(sample_doc.content, sample_doc.filename, 5, tool_specs)
        assert "my-data-bucket" in prompt

    def test_includes_tool_name(self, tool_specs, sample_doc):
        prompt = _build_agent_prompt(sample_doc.content, sample_doc.filename, 5, tool_specs)
        assert "search_kb" in prompt

    def test_includes_filename(self, tool_specs, sample_doc):
        prompt = _build_agent_prompt(sample_doc.content, sample_doc.filename, 5, tool_specs)
        assert "s3-guide.md" in prompt


# ---------------------------------------------------------------------------
# _parse_agent_response
# ---------------------------------------------------------------------------

class TestParseAgentResponse:
    def test_parses_valid_json(self):
        cases = _parse_agent_response(VALID_RESPONSE)
        assert len(cases) == 2

    def test_case_fields(self):
        cases = _parse_agent_response(VALID_RESPONSE)
        c = cases[0]
        assert c.task == "Which region is my-data-bucket in?"
        assert c.expected_answer == "us-west-2"
        assert c.must_call == ["search_kb"]
        assert c.must_call_with == [{"tool": "search_kb", "args": {"query": "my-data-bucket"}}]
        assert c.max_steps == 4
        assert c.difficulty == "easy"
        assert c.tags == ["s3"]

    def test_null_expected_answer(self):
        cases = _parse_agent_response(VALID_RESPONSE)
        assert cases[1].expected_answer is None

    def test_strips_markdown_fences(self):
        fenced = f"```json\n{VALID_RESPONSE}\n```"
        cases = _parse_agent_response(fenced)
        assert len(cases) == 2

    def test_empty_string_returns_empty(self):
        assert _parse_agent_response("") == []

    def test_invalid_json_returns_empty(self):
        assert _parse_agent_response("not json") == []

    def test_skips_malformed_items(self):
        bad = json.dumps([
            {"task": "good", "description": "", "must_call": [], "must_call_with": [],
             "max_steps": 3, "difficulty": "easy", "tags": [], "source_excerpt": ""},
            {"no_task_field": True},  # malformed
        ])
        cases = _parse_agent_response(bad)
        assert len(cases) == 1
        assert cases[0].task == "good"


# ---------------------------------------------------------------------------
# GeneratedAgentCase.to_agent_test_case
# ---------------------------------------------------------------------------

class TestToAgentTestCase:
    @pytest.fixture
    def gen_case(self):
        return GeneratedAgentCase(
            task="Which region is my-data-bucket in?",
            description="Region lookup",
            expected_answer="us-west-2",
            must_call=["search_kb"],
            must_call_with=[{"tool": "search_kb", "args": {"query": "my-data-bucket"}}],
            max_steps=4,
            difficulty="easy",
            tags=["s3"],
            source_excerpt="The bucket is in us-west-2.",
        )

    def test_task_mapped(self, gen_case):
        case = gen_case.to_agent_test_case()
        assert case.input.task == "Which region is my-data-bucket in?"

    def test_expected_answer(self, gen_case):
        case = gen_case.to_agent_test_case()
        assert case.expected.answer == "us-west-2"
        assert case.expected.answer_match == AnswerMatch.contains

    def test_must_call(self, gen_case):
        case = gen_case.to_agent_test_case()
        assert case.expected.must_call == ["search_kb"]

    def test_must_call_with(self, gen_case):
        case = gen_case.to_agent_test_case()
        assert len(case.expected.must_call_with) == 1
        assert case.expected.must_call_with[0].tool == "search_kb"
        assert case.expected.must_call_with[0].args == {"query": "my-data-bucket"}

    def test_max_steps(self, gen_case):
        case = gen_case.to_agent_test_case()
        assert case.expected.max_steps == 4

    def test_metadata(self, gen_case):
        case = gen_case.to_agent_test_case()
        assert case.description == "Region lookup"
        assert case.difficulty == "easy"
        assert case.tags == ["s3"]

    def test_tools_block_is_empty(self, gen_case):
        case = gen_case.to_agent_test_case()
        assert case.tools == {}

    def test_none_expected_answer(self):
        gen = GeneratedAgentCase(
            task="q", description="", expected_answer=None,
            must_call=[], must_call_with=[], max_steps=3,
            difficulty="easy", tags=[], source_excerpt="",
        )
        case = gen.to_agent_test_case()
        assert case.expected.answer is None

    def test_must_call_with_missing_tool_field_ignored(self):
        gen = GeneratedAgentCase(
            task="q", description="", expected_answer=None,
            must_call=[], must_call_with=[{"args": {"k": "v"}}],  # missing "tool"
            max_steps=3, difficulty="easy", tags=[], source_excerpt="",
        )
        case = gen.to_agent_test_case()
        assert case.expected.must_call_with == []


# ---------------------------------------------------------------------------
# AgentBedrockGenerator
# ---------------------------------------------------------------------------

class TestAgentBedrockGenerator:
    def test_calls_bedrock_and_returns_cases(self, sample_doc, tool_specs):
        mock_response = {
            "body": MagicMock(read=lambda: json.dumps({
                "content": [{"text": VALID_RESPONSE}]
            }).encode())
        }
        with patch("boto3.client") as mock_boto:
            mock_boto.return_value.invoke_model.return_value = mock_response
            gen = AgentBedrockGenerator()
            cases = gen.generate(sample_doc, count=2, tools=tool_specs)

        assert len(cases) == 2
        assert cases[0].task == "Which region is my-data-bucket in?"

    def test_tools_passed_to_prompt(self, sample_doc, tool_specs):
        mock_response = {
            "body": MagicMock(read=lambda: json.dumps({
                "content": [{"text": VALID_RESPONSE}]
            }).encode())
        }
        with patch("boto3.client") as mock_boto:
            mock_boto.return_value.invoke_model.return_value = mock_response
            gen = AgentBedrockGenerator()
            gen.generate(sample_doc, count=2, tools=tool_specs)

        call_kwargs = mock_boto.return_value.invoke_model.call_args
        body = json.loads(call_kwargs[1]["body"])
        prompt = body["messages"][0]["content"]
        assert "search_kb" in prompt


# ---------------------------------------------------------------------------
# AgentAnthropicGenerator
# ---------------------------------------------------------------------------

def _make_mock_anthropic_module(mock_client):
    mock_module = MagicMock()
    mock_module.Anthropic.return_value = mock_client
    return mock_module


class TestAgentAnthropicGenerator:
    def test_calls_anthropic_and_returns_cases(self, sample_doc, tool_specs):
        content_block = MagicMock()
        content_block.text = VALID_RESPONSE
        mock_message = MagicMock()
        mock_message.content = [content_block]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message

        with patch.dict("sys.modules", {"anthropic": _make_mock_anthropic_module(mock_client)}):
            gen = AgentAnthropicGenerator()
            gen._client = None
            cases = gen.generate(sample_doc, count=2, tools=tool_specs)

        assert len(cases) == 2
