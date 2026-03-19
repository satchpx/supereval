"""
Synthetic agent test case generation using Claude.

Generates task scenarios from documentation. Unlike the trace importer (which
converts real execution traces into test cases), this module uses an LLM to
*invent* plausible tasks grounded in the document — tasks that naturally require
the agent's available tools to answer.

The generated cases have an empty `tools` block (no mock responses). Mock
responses cannot be inferred from docs alone; add them by:
  1. Running your agent against the generated tasks to collect traces.
  2. Importing those traces with `supereval agent generate --from traces.jsonl`.

Supported backends: Amazon Bedrock (default), Anthropic API, OpenAI, Azure OpenAI
"""
from __future__ import annotations

import json
import math
import textwrap
from dataclasses import dataclass, field

from ..generator import (
    DEFAULT_MODEL,
    DEFAULT_REGION,
    ANTHROPIC_DEFAULT_MODEL,
    OPENAI_DEFAULT_MODEL,
    _chunk_document,
)
from ..sources import Document


_AGENT_PROMPT = textwrap.dedent("""
    You are an expert at creating evaluation test cases for AI agents.

    Given the documentation below and the available tools, generate exactly {count}
    realistic task scenarios that test an agent's ability to use the tools correctly.

    Available tools:
    {tool_descriptions}

    Rules:
    - Each task must require the agent to call at least one listed tool.
    - expected_answer must be grounded in the document; use null if not determinable.
    - must_call must contain only tool names from the Available tools list above.
    - must_call_with provides concrete argument examples — include only when the document
      supplies clear values (e.g. a bucket name, a user ID, a search term).
    - max_steps should be a realistic upper bound (typically 3–8).
    - Vary difficulty: easy (single tool call, direct answer), medium (two calls or
      light inference), hard (multi-step or cross-referencing multiple facts).
    - source_excerpt must be a verbatim quote from the document, max 200 characters.

    Return ONLY a valid JSON array — no markdown, no explanation. Each element:
    {{
      "task": "a realistic user task or question",
      "description": "what this test case evaluates",
      "expected_answer": "the correct final answer, or null",
      "must_call": ["tool_name1", "tool_name2"],
      "must_call_with": [{{"tool": "tool_name", "args": {{"param": "value"}}}}],
      "max_steps": 5,
      "difficulty": "easy|medium|hard",
      "tags": ["tag1", "tag2"],
      "source_excerpt": "verbatim excerpt from the document (max 200 chars)"
    }}

    Document ({filename}):
    {document}
""").strip()


def _format_tools(tools: list) -> str:
    """Render a list of ToolSpec objects as a readable block for the prompt."""
    if not tools:
        return "  (no tools defined — all tasks will use generic tool usage)"
    lines = []
    for t in tools:
        lines.append(f"  Tool: {t.name}")
        lines.append(f"  Description: {t.description}")
        if t.parameters:
            lines.append(f"  Parameters: {json.dumps(t.parameters)}")
        lines.append("")
    return "\n".join(lines).strip()


def _build_agent_prompt(text: str, filename: str, count: int, tools: list) -> str:
    return _AGENT_PROMPT.format(
        count=count,
        tool_descriptions=_format_tools(tools),
        filename=filename,
        document=text,
    )


@dataclass
class GeneratedAgentCase:
    task: str
    description: str
    expected_answer: str | None
    must_call: list[str]
    must_call_with: list[dict]          # [{"tool": ..., "args": {...}}]
    max_steps: int
    difficulty: str
    tags: list[str]
    source_excerpt: str

    def to_staged_dict(self) -> dict:
        """Full dict for the staging file — includes source_excerpt for review."""
        return {
            "task": self.task,
            "description": self.description,
            "expected_answer": self.expected_answer,
            "must_call": self.must_call,
            "must_call_with": self.must_call_with,
            "max_steps": self.max_steps,
            "difficulty": self.difficulty,
            "tags": self.tags,
            "source_excerpt": self.source_excerpt,
        }

    def to_agent_test_case(self):
        """Convert to AgentTestCase.

        The `tools` block (mock responses) is intentionally empty — mock
        responses must be added separately from traces or manually.
        """
        from .models import (
            AgentExpected,
            AgentTask,
            AgentTestCase,
            AnswerMatch,
            MustCallWith,
        )
        must_call_with = [
            MustCallWith(tool=m["tool"], args=m.get("args", {}))
            for m in self.must_call_with
            if "tool" in m
        ]
        return AgentTestCase(
            description=self.description,
            tags=self.tags,
            difficulty=self.difficulty,
            input=AgentTask(task=self.task),
            tools={},
            expected=AgentExpected(
                answer=self.expected_answer,
                answer_match=AnswerMatch.contains,
                must_call=self.must_call,
                must_call_with=must_call_with,
                max_steps=self.max_steps,
            ),
        )


def _parse_agent_response(raw: str) -> list[GeneratedAgentCase]:
    """Parse the LLM's JSON array response into GeneratedAgentCase objects.

    Strips markdown fences. Skips malformed items rather than failing entirely.
    """
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

    try:
        items = json.loads(text)
    except json.JSONDecodeError:
        return []

    if not isinstance(items, list):
        return []

    cases: list[GeneratedAgentCase] = []
    for item in items:
        try:
            cases.append(GeneratedAgentCase(
                task=item["task"],
                description=item.get("description", ""),
                expected_answer=item.get("expected_answer"),
                must_call=item.get("must_call", []),
                must_call_with=item.get("must_call_with", []),
                max_steps=int(item.get("max_steps", 6)),
                difficulty=item.get("difficulty", "medium"),
                tags=item.get("tags", []),
                source_excerpt=item.get("source_excerpt", ""),
            ))
        except (KeyError, TypeError):
            continue

    return cases


# ---------------------------------------------------------------------------
# Generator classes (same backend pattern as generator.py)
# ---------------------------------------------------------------------------

class AgentBedrockGenerator:
    """Generate agent task scenarios using Claude on Amazon Bedrock."""

    def __init__(self, model_id: str = DEFAULT_MODEL, region: str = DEFAULT_REGION):
        self.model_id = model_id
        self.region = region
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client("bedrock-runtime", region_name=self.region)
            except ImportError:
                raise ImportError("boto3 is required. Install it with: pip install boto3")
        return self._client

    def generate(
        self,
        document: Document,
        count: int,
        tools: list,
    ) -> list[GeneratedAgentCase]:
        chunks = _chunk_document(document.content)
        if len(chunks) == 1:
            return self._generate_from_chunk(chunks[0], document.filename, count, tools)

        cases: list[GeneratedAgentCase] = []
        per_chunk = max(1, math.ceil(count / len(chunks)))
        for chunk in chunks:
            remaining = count - len(cases)
            if remaining <= 0:
                break
            cases.extend(
                self._generate_from_chunk(chunk, document.filename, min(per_chunk, remaining), tools)
            )
        return cases[:count]

    def _generate_from_chunk(
        self, text: str, filename: str, count: int, tools: list
    ) -> list[GeneratedAgentCase]:
        prompt = _build_agent_prompt(text, filename, count, tools)
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 8192,
            "messages": [{"role": "user", "content": prompt}],
        })
        response = self.client.invoke_model(modelId=self.model_id, body=body)
        raw = json.loads(response["body"].read())
        return _parse_agent_response(raw["content"][0]["text"])


class AgentAnthropicGenerator:
    """Generate agent task scenarios using the Anthropic API directly."""

    def __init__(self, model_id: str = ANTHROPIC_DEFAULT_MODEL, api_key: str | None = None):
        self.model_id = model_id
        self.api_key = api_key
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package is required. "
                    "Install with: pip install 'supereval[anthropic]'"
                )
        return self._client

    def generate(
        self,
        document: Document,
        count: int,
        tools: list,
    ) -> list[GeneratedAgentCase]:
        chunks = _chunk_document(document.content)
        if len(chunks) == 1:
            return self._generate_from_chunk(chunks[0], document.filename, count, tools)

        cases: list[GeneratedAgentCase] = []
        per_chunk = max(1, math.ceil(count / len(chunks)))
        for chunk in chunks:
            remaining = count - len(cases)
            if remaining <= 0:
                break
            cases.extend(
                self._generate_from_chunk(chunk, document.filename, min(per_chunk, remaining), tools)
            )
        return cases[:count]

    def _generate_from_chunk(
        self, text: str, filename: str, count: int, tools: list
    ) -> list[GeneratedAgentCase]:
        prompt = _build_agent_prompt(text, filename, count, tools)
        message = self.client.messages.create(
            model=self.model_id,
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}],
        )
        return _parse_agent_response(message.content[0].text)


class AgentOpenAIGenerator:
    """Generate agent task scenarios using the OpenAI API or Azure OpenAI."""

    def __init__(
        self,
        model_id: str = OPENAI_DEFAULT_MODEL,
        api_key: str | None = None,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
    ):
        self.model_id = model_id
        self.api_key = api_key
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import openai
                if self.azure_endpoint:
                    self._client = openai.AzureOpenAI(
                        api_key=self.api_key,
                        azure_endpoint=self.azure_endpoint,
                        api_version=self.api_version or "2024-02-01",
                    )
                else:
                    self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "openai package is required. "
                    "Install with: pip install 'supereval[openai]'"
                )
        return self._client

    def generate(
        self,
        document: Document,
        count: int,
        tools: list,
    ) -> list[GeneratedAgentCase]:
        chunks = _chunk_document(document.content)
        if len(chunks) == 1:
            return self._generate_from_chunk(chunks[0], document.filename, count, tools)

        cases: list[GeneratedAgentCase] = []
        per_chunk = max(1, math.ceil(count / len(chunks)))
        for chunk in chunks:
            remaining = count - len(cases)
            if remaining <= 0:
                break
            cases.extend(
                self._generate_from_chunk(chunk, document.filename, min(per_chunk, remaining), tools)
            )
        return cases[:count]

    def _generate_from_chunk(
        self, text: str, filename: str, count: int, tools: list
    ) -> list[GeneratedAgentCase]:
        prompt = _build_agent_prompt(text, filename, count, tools)
        response = self.client.chat.completions.create(
            model=self.model_id,
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}],
        )
        return _parse_agent_response(response.choices[0].message.content)
