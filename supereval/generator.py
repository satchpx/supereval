"""
Synthetic test case generation using Claude.

Supported generators : Amazon Bedrock (default), Anthropic API, OpenAI / Azure OpenAI
Supported types      : qa, classification, instruction
"""
from __future__ import annotations

import json
import math
import textwrap
from dataclasses import dataclass, field

from .sources import Document

# Default Bedrock model. Cross-region inference profiles (e.g. us.anthropic.*) also work.
DEFAULT_MODEL = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
DEFAULT_REGION = "us-east-1"

# Default model when using the Anthropic API directly
ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-4-6"

# Default model when using the OpenAI API
OPENAI_DEFAULT_MODEL = "gpt-4o"

# Max characters per chunk sent to the model (~10K tokens; keeps cost predictable)
MAX_CHARS_PER_CHUNK = 40_000

CASE_TYPES = ["factual", "multi_hop", "negation", "out_of_scope", "ambiguous"]

# Target distribution across case types for a diverse dataset
CASE_MIX: dict[str, float] = {
    "factual":      0.40,
    "multi_hop":    0.20,
    "negation":     0.15,
    "out_of_scope": 0.15,
    "ambiguous":    0.10,
}

_QA_GENERATION_PROMPT = textwrap.dedent("""
    You are an expert at creating evaluation test cases for RAG (Retrieval-Augmented Generation) systems.

    Given the document below, generate exactly {count} diverse Q&A test cases that rigorously test
    a RAG system's ability to handle real-world queries. Aim for this distribution:
    {mix}

    Case type definitions:
    - factual:      Direct factual question with a specific, verifiable answer in the document.
    - multi_hop:    Requires combining two or more distinct facts from the document to answer.
    - negation:     Asks what something does NOT do, support, or include.
    - out_of_scope: Cannot be answered from this document alone. Use ground_truth:
                    "This information is not covered in the provided documentation."
    - ambiguous:    Has multiple plausible interpretations. Address the most likely one in ground_truth.

    Rules:
    - All answers must be grounded in the document (except out_of_scope).
    - Do not invent facts not present in the document.
    - Vary difficulty: mix easy (direct lookup), medium (inference), and hard (multi-step).
    - source_excerpt must be a verbatim quote from the document, max 200 characters.

    Return ONLY a valid JSON array — no markdown, no explanation. Each element:
    {{
      "description": "one-line description of what this case tests",
      "input": {{"query": "the question"}},
      "expected": {{"ground_truth": "the correct answer"}},
      "tags": ["tag1", "tag2"],
      "difficulty": "easy|medium|hard",
      "case_type": "factual|multi_hop|negation|out_of_scope|ambiguous",
      "source_excerpt": "verbatim excerpt from the document (max 200 chars)"
    }}

    Document ({filename}):
    ---
    {document}
    ---
""").strip()


_CLASSIFICATION_PROMPT = textwrap.dedent("""
    You are an expert at creating evaluation test cases for text classification systems.

    Given the document below, generate exactly {count} diverse classification test cases
    that test an AI classifier's ability to assign the correct label to a text input.

    Valid labels: {labels}

    Rules:
    - Each input "text" must be realistic — similar to what a user might submit.
    - Cover a range of difficulty: obvious cases, edge cases, and ambiguous borderline examples.
    - Do not invent facts not present in the document; base text inputs on its content.
    - source_excerpt must be a verbatim quote from the document, max 200 characters.

    Return ONLY a valid JSON array — no markdown, no explanation. Each element:
    {{
      "description": "one-line description of what this case tests",
      "input": {{"text": "the text to classify"}},
      "expected": {{"label": "one of the valid labels"}},
      "tags": ["tag1", "tag2"],
      "difficulty": "easy|medium|hard",
      "case_type": "clear|ambiguous|edge_case",
      "source_excerpt": "verbatim excerpt from the document (max 200 chars)"
    }}

    Document ({filename}):
    ---
    {document}
    ---
""").strip()

_INSTRUCTION_PROMPT = textwrap.dedent("""
    You are an expert at creating evaluation test cases for instruction-following AI systems.

    Given the document below, generate exactly {count} diverse instruction-following test cases
    that test an AI's ability to execute specific tasks based on the document's content.

    Task types to cover: summarization, extraction, formatting, analysis, comparison.

    Rules:
    - Each test case has an instruction and optionally a short document excerpt as input.
    - The rubric must define clear, verifiable criteria (e.g. "Must contain exactly 3 bullet points").
    - Do not invent facts not present in the document.
    - source_excerpt must be a verbatim quote from the document, max 200 characters.

    Return ONLY a valid JSON array — no markdown, no explanation. Each element:
    {{
      "description": "one-line description of what this case tests",
      "input": {{
        "instruction": "the task instruction",
        "document": "relevant excerpt from the document, or null if not needed"
      }},
      "expected": {{
        "rubric": "clear pass/fail criteria for a correct response"
      }},
      "tags": ["tag1", "tag2"],
      "difficulty": "easy|medium|hard",
      "case_type": "summarization|extraction|formatting|analysis|comparison",
      "source_excerpt": "verbatim excerpt from the document (max 200 chars)"
    }}

    Document ({filename}):
    ---
    {document}
    ---
""").strip()


_RAG_GENERATION_PROMPT = textwrap.dedent("""
    You are an expert at creating evaluation test cases for RAG (Retrieval-Augmented Generation) systems.

    Given the document below, generate exactly {count} RAG test cases. Each case must include:
    - A realistic question a user might ask
    - The correct answer (ground_truth) — grounded in the document
    - 2–4 retrieved_contexts: verbatim excerpts from the document that together contain
      the information needed to answer the question
    - Optionally 1 distractor context (a plausible but unhelpful excerpt) in ~30% of cases
      to test faithfulness

    Rules:
    - retrieved_contexts must be verbatim excerpts from the document (max 300 characters each)
    - ground_truth must be fully answerable from the retrieved_contexts
    - Do not invent facts not present in the document
    - Vary difficulty: mix direct lookup, inference, and multi-hop questions

    Return ONLY a valid JSON array — no markdown, no explanation. Each element:
    {{
      "description": "one-line description of what this case tests",
      "input": {{
        "query": "the question",
        "retrieved_contexts": ["verbatim excerpt 1", "verbatim excerpt 2"]
      }},
      "expected": {{"ground_truth": "the correct answer"}},
      "tags": ["tag1", "tag2"],
      "difficulty": "easy|medium|hard",
      "case_type": "direct|inference|multi_hop",
      "source_excerpt": "verbatim excerpt from the document (max 200 chars)"
    }}

    Document ({filename}):
    ---
    {document}
    ---
""").strip()


def _build_prompt(
    dataset_type: str,
    text: str,
    filename: str,
    count: int,
    labels: list[str] | None = None,
) -> str:
    if dataset_type == "classification":
        return _CLASSIFICATION_PROMPT.format(
            count=count,
            labels=", ".join(labels) if labels else "(not specified)",
            filename=filename,
            document=text,
        )
    if dataset_type == "instruction":
        return _INSTRUCTION_PROMPT.format(count=count, filename=filename, document=text)
    if dataset_type == "rag":
        return _RAG_GENERATION_PROMPT.format(count=count, filename=filename, document=text)
    # qa (default)
    mix_lines = "\n".join(
        f"  - {k}: ~{int(v * count)} cases ({int(v * 100)}%)"
        for k, v in CASE_MIX.items()
    )
    return _QA_GENERATION_PROMPT.format(
        count=count, mix=mix_lines, filename=filename, document=text
    )


@dataclass
class GeneratedCase:
    description: str
    input: dict
    expected: dict
    tags: list[str]
    difficulty: str
    case_type: str
    source_excerpt: str

    def to_staged_dict(self) -> dict:
        """Full dict including review-helper fields. Written to the staging file."""
        return {
            "description": self.description,
            "input": self.input,
            "expected": self.expected,
            "tags": self.tags,
            "difficulty": self.difficulty,
            "case_type": self.case_type,
            "source_excerpt": self.source_excerpt,
        }

    def to_importable_dict(self) -> dict:
        """Dict without generation-only fields. Safe to pass to dataset add-cases."""
        return {
            "description": self.description,
            "input": self.input,
            "expected": self.expected,
            "tags": self.tags,
            "difficulty": self.difficulty,
        }


class BedrockGenerator:
    """
    Generates test cases by invoking Claude on Amazon Bedrock.

    Uses standard boto3 credential resolution — AWS credentials must be configured
    via environment variables, ~/.aws/credentials, or an IAM role.
    """

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
                raise ImportError(
                    "boto3 is required for generation. Install it with: pip install boto3"
                )
        return self._client

    def generate(
        self,
        document: Document,
        count: int,
        dataset_type: str = "qa",
        labels: list[str] | None = None,
    ) -> list[GeneratedCase]:
        """Generate `count` test cases from a document, chunking if necessary."""
        chunks = _chunk_document(document.content)

        if len(chunks) == 1:
            return self._generate_from_chunk(
                chunks[0], document.filename, count, dataset_type, labels
            )

        # Distribute cases across chunks proportionally
        cases: list[GeneratedCase] = []
        per_chunk = max(1, math.ceil(count / len(chunks)))
        for chunk in chunks:
            remaining = count - len(cases)
            if remaining <= 0:
                break
            chunk_count = min(per_chunk, remaining)
            cases.extend(
                self._generate_from_chunk(
                    chunk, document.filename, chunk_count, dataset_type, labels
                )
            )

        return cases[:count]

    def _generate_from_chunk(
        self,
        text: str,
        filename: str,
        count: int,
        dataset_type: str = "qa",
        labels: list[str] | None = None,
    ) -> list[GeneratedCase]:
        prompt = _build_prompt(dataset_type, text, filename, count, labels)
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 8192,
            "messages": [{"role": "user", "content": prompt}],
        })
        response = self.client.invoke_model(modelId=self.model_id, body=body)
        raw = json.loads(response["body"].read())
        content = raw["content"][0]["text"]
        return _parse_response(content)


class AnthropicGenerator:
    """
    Generates test cases by calling the Anthropic API directly.

    Useful outside AWS environments. Reads ANTHROPIC_API_KEY from the environment
    unless an explicit api_key is passed.

    Install: pip install 'supereval[anthropic]'
    """

    def __init__(
        self,
        model_id: str = ANTHROPIC_DEFAULT_MODEL,
        api_key: str | None = None,
    ):
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
                    "Install it with: pip install 'supereval[anthropic]'"
                )
        return self._client

    def generate(
        self,
        document: Document,
        count: int,
        dataset_type: str = "qa",
        labels: list[str] | None = None,
    ) -> list[GeneratedCase]:
        chunks = _chunk_document(document.content)

        if len(chunks) == 1:
            return self._generate_from_chunk(
                chunks[0], document.filename, count, dataset_type, labels
            )

        cases: list[GeneratedCase] = []
        per_chunk = max(1, math.ceil(count / len(chunks)))
        for chunk in chunks:
            remaining = count - len(cases)
            if remaining <= 0:
                break
            chunk_count = min(per_chunk, remaining)
            cases.extend(
                self._generate_from_chunk(
                    chunk, document.filename, chunk_count, dataset_type, labels
                )
            )

        return cases[:count]

    def _generate_from_chunk(
        self,
        text: str,
        filename: str,
        count: int,
        dataset_type: str = "qa",
        labels: list[str] | None = None,
    ) -> list[GeneratedCase]:
        prompt = _build_prompt(dataset_type, text, filename, count, labels)
        message = self.client.messages.create(
            model=self.model_id,
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}],
        )
        return _parse_response(message.content[0].text)


class OpenAIGenerator:
    """
    Generates test cases using the OpenAI API or Azure OpenAI.

    For OpenAI: reads OPENAI_API_KEY from the environment automatically.
    For Azure:  set AZURE_OPENAI_ENDPOINT and OPENAI_API_VERSION env vars,
                or pass azure_endpoint / api_version explicitly.

    Install: pip install 'supereval[openai]'
    """

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
                    "Install it with: pip install 'supereval[openai]'"
                )
        return self._client

    def generate(
        self,
        document: Document,
        count: int,
        dataset_type: str = "qa",
        labels: list[str] | None = None,
    ) -> list[GeneratedCase]:
        chunks = _chunk_document(document.content)

        if len(chunks) == 1:
            return self._generate_from_chunk(
                chunks[0], document.filename, count, dataset_type, labels
            )

        cases: list[GeneratedCase] = []
        per_chunk = max(1, math.ceil(count / len(chunks)))
        for chunk in chunks:
            remaining = count - len(cases)
            if remaining <= 0:
                break
            chunk_count = min(per_chunk, remaining)
            cases.extend(
                self._generate_from_chunk(
                    chunk, document.filename, chunk_count, dataset_type, labels
                )
            )

        return cases[:count]

    def _generate_from_chunk(
        self,
        text: str,
        filename: str,
        count: int,
        dataset_type: str = "qa",
        labels: list[str] | None = None,
    ) -> list[GeneratedCase]:
        prompt = _build_prompt(dataset_type, text, filename, count, labels)
        response = self.client.chat.completions.create(
            model=self.model_id,
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}],
        )
        return _parse_response(response.choices[0].message.content)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _chunk_document(content: str) -> list[str]:
    """Split content into chunks that fit within MAX_CHARS_PER_CHUNK."""
    if len(content) <= MAX_CHARS_PER_CHUNK:
        return [content]

    chunks: list[str] = []
    start = 0
    while start < len(content):
        end = start + MAX_CHARS_PER_CHUNK
        if end < len(content):
            # Prefer breaking on a paragraph boundary
            boundary = content.rfind("\n\n", start, end)
            if boundary > start:
                end = boundary
        chunks.append(content[start:end].strip())
        start = end

    return [c for c in chunks if c]


def _parse_response(raw_text: str) -> list[GeneratedCase]:
    """Parse LLM JSON response into GeneratedCase objects. Tolerates markdown fences."""
    text = raw_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        start = 1
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[start:end])

    try:
        items = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Generator returned invalid JSON: {e}\n\nRaw output (first 500 chars):\n{raw_text[:500]}"
        )

    cases: list[GeneratedCase] = []
    for item in items:
        try:
            cases.append(GeneratedCase(
                description=item.get("description", ""),
                input=item["input"],
                expected=item["expected"],
                tags=item.get("tags", []),
                difficulty=item.get("difficulty", "medium"),
                case_type=item.get("case_type", "factual"),
                source_excerpt=item.get("source_excerpt", ""),
            ))
        except (KeyError, TypeError):
            continue  # Skip malformed items; partial results beat a full failure

    return cases
