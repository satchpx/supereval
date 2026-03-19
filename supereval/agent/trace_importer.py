"""
Convert agent execution traces into AgentTestCase objects.

Traces capture what an agent actually did (thoughts, tool calls, answers)
and are the raw material for bootstrapping an agent eval dataset. Instead
of writing cases by hand, export traces from your production agent, feed
them here, and get structured, importable test cases.

Supported input format
----------------------
A JSONL file where each line is a JSON object with these fields:

  {
    "task": "Which region is my-data-bucket in?",          # required
    "steps": [                                              # required
      {"type": "thought",  "content": "I should search."},
      {"type": "tool_call", "tool": "search_kb",
       "arguments": {"query": "my-data-bucket"},
       "result": {"region": "us-west-2"}},
      {"type": "answer",   "content": "The bucket is in us-west-2."}
    ],
    "final_answer": "The bucket is in us-west-2.",          # optional
    "metadata": {                                           # optional
      "description": "Look up S3 bucket region",
      "difficulty":  "easy",
      "tags":        ["s3", "region"]
    }
  }

Step types
----------
  thought    — agent reasoning (no tool interaction)
  tool_call  — agent called a tool; requires "tool" + optional "arguments"/"result"
  observation — tool response recorded separately (result attached to the preceding tool_call)
  answer     — agent's final answer

Inference rules (trace → AgentTestCase)
----------------------------------------
  input.task          ← trace.task
  expected.answer     ← trace.final_answer, or content of last "answer" step
  expected.must_call  ← ordered unique tool names from all tool_call steps
  tools               ← for each unique tool name, the last result seen for that tool
  expected.max_steps  ← ceil(len(steps) * 1.5), minimum 3
  expected.max_tool_calls ← tool_call_count + 1 (minimum 1)
  description / tags / difficulty ← from metadata block
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .models import AgentExpected, AgentTask, AgentTestCase, AnswerMatch, MockResponse


# ---------------------------------------------------------------------------
# Input trace models
# ---------------------------------------------------------------------------

class TraceStep(BaseModel):
    type: Literal["thought", "tool_call", "observation", "answer"]
    content: str = ""
    tool: str | None = None          # tool name (tool_call steps only)
    arguments: dict = Field(default_factory=dict)
    result: Any = None               # tool result (tool_call steps only)


class InputTrace(BaseModel):
    task: str
    steps: list[TraceStep]
    final_answer: str | None = None
    metadata: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_traces(path: Path) -> list[InputTrace]:
    """Parse a JSONL file and return a list of InputTrace objects.

    Skips blank lines. Raises ValueError with a line number on the first
    parse or validation error.
    """
    traces: list[InputTrace] = []
    raw_lines = path.read_text().splitlines()
    for lineno, line in enumerate(raw_lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            traces.append(InputTrace.model_validate(obj))
        except Exception as exc:
            raise ValueError(f"Line {lineno}: {exc}") from exc
    return traces


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def trace_to_case(trace: InputTrace) -> AgentTestCase:
    """Convert a single InputTrace into an AgentTestCase.

    All inference is deterministic — no LLM calls.
    """
    # Collect tool call information from steps
    tool_call_steps = [s for s in trace.steps if s.type == "tool_call" and s.tool]
    tool_call_count = len(tool_call_steps)

    # must_call: unique tool names in order of first occurrence
    seen: set[str] = set()
    must_call: list[str] = []
    for s in tool_call_steps:
        if s.tool not in seen:
            must_call.append(s.tool)
            seen.add(s.tool)

    # tools mock block: last result seen per tool name
    tool_mocks: dict[str, MockResponse] = {}
    for s in tool_call_steps:
        tool_mocks[s.tool] = MockResponse(response=s.result)

    # expected answer: explicit final_answer preferred, else last answer step
    answer: str | None = trace.final_answer
    if answer is None:
        answer_steps = [s for s in trace.steps if s.type == "answer" and s.content]
        if answer_steps:
            answer = answer_steps[-1].content

    # step / tool limits with generous headroom
    max_steps = max(3, math.ceil(len(trace.steps) * 1.5))
    max_tool_calls = max(1, tool_call_count + 1)

    # metadata fields
    meta = trace.metadata
    description: str = meta.get("description", "")
    tags: list[str] = meta.get("tags", [])
    difficulty = meta.get("difficulty", None)

    return AgentTestCase(
        description=description,
        tags=tags,
        difficulty=difficulty,
        input=AgentTask(task=trace.task),
        tools=tool_mocks,
        expected=AgentExpected(
            answer=answer,
            answer_match=AnswerMatch.contains,
            must_call=must_call,
            max_steps=max_steps,
            max_tool_calls=max_tool_calls,
        ),
    )
