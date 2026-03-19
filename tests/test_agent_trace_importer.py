"""Tests for agent trace parsing and AgentTestCase inference."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from supereval.agent.models import AnswerMatch
from supereval.agent.trace_importer import InputTrace, TraceStep, parse_traces, trace_to_case


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_traces(tmp_path: Path, traces: list[dict]) -> Path:
    p = tmp_path / "traces.jsonl"
    p.write_text("\n".join(json.dumps(t) for t in traces) + "\n")
    return p


SIMPLE_TRACE = {
    "task": "Which region is my-data-bucket in?",
    "steps": [
        {"type": "thought", "content": "I should search."},
        {
            "type": "tool_call",
            "tool": "search_kb",
            "arguments": {"query": "my-data-bucket"},
            "result": {"region": "us-west-2"},
        },
        {"type": "answer", "content": "The bucket is in us-west-2."},
    ],
    "final_answer": "us-west-2",
    "metadata": {"description": "S3 region lookup", "difficulty": "easy", "tags": ["s3"]},
}


# ---------------------------------------------------------------------------
# parse_traces
# ---------------------------------------------------------------------------

class TestParseTraces:
    def test_parses_single_trace(self, tmp_path):
        path = _write_traces(tmp_path, [SIMPLE_TRACE])
        traces = parse_traces(path)
        assert len(traces) == 1
        assert traces[0].task == "Which region is my-data-bucket in?"

    def test_parses_multiple_traces(self, tmp_path):
        path = _write_traces(tmp_path, [SIMPLE_TRACE, SIMPLE_TRACE])
        traces = parse_traces(path)
        assert len(traces) == 2

    def test_skips_blank_lines(self, tmp_path):
        p = tmp_path / "traces.jsonl"
        p.write_text("\n" + json.dumps(SIMPLE_TRACE) + "\n\n")
        traces = parse_traces(p)
        assert len(traces) == 1

    def test_raises_on_invalid_json(self, tmp_path):
        p = tmp_path / "traces.jsonl"
        p.write_text("not json\n")
        with pytest.raises(ValueError, match="Line 1"):
            parse_traces(p)

    def test_raises_on_missing_task(self, tmp_path):
        bad = {"steps": []}
        path = _write_traces(tmp_path, [bad])
        with pytest.raises(ValueError, match="Line 1"):
            parse_traces(path)

    def test_empty_file_returns_empty_list(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        assert parse_traces(p) == []


# ---------------------------------------------------------------------------
# trace_to_case — basic fields
# ---------------------------------------------------------------------------

class TestTraceToCaseBasicFields:
    def test_task_maps_to_input(self):
        trace = InputTrace.model_validate(SIMPLE_TRACE)
        case = trace_to_case(trace)
        assert case.input.task == "Which region is my-data-bucket in?"

    def test_final_answer_used_when_present(self):
        trace = InputTrace.model_validate(SIMPLE_TRACE)
        case = trace_to_case(trace)
        assert case.expected.answer == "us-west-2"

    def test_answer_step_used_when_no_final_answer(self):
        t = {**SIMPLE_TRACE, "final_answer": None}
        trace = InputTrace.model_validate(t)
        case = trace_to_case(trace)
        assert case.expected.answer == "The bucket is in us-west-2."

    def test_answer_is_none_when_no_answer_anywhere(self):
        t = {"task": "q?", "steps": [{"type": "thought", "content": "hmm"}]}
        trace = InputTrace.model_validate(t)
        case = trace_to_case(trace)
        assert case.expected.answer is None

    def test_answer_match_defaults_to_contains(self):
        trace = InputTrace.model_validate(SIMPLE_TRACE)
        case = trace_to_case(trace)
        assert case.expected.answer_match == AnswerMatch.contains

    def test_metadata_mapped(self):
        trace = InputTrace.model_validate(SIMPLE_TRACE)
        case = trace_to_case(trace)
        assert case.description == "S3 region lookup"
        assert case.difficulty == "easy"
        assert case.tags == ["s3"]

    def test_empty_metadata_gives_defaults(self):
        t = {**SIMPLE_TRACE, "metadata": {}}
        trace = InputTrace.model_validate(t)
        case = trace_to_case(trace)
        assert case.description == ""
        assert case.tags == []
        assert case.difficulty is None


# ---------------------------------------------------------------------------
# trace_to_case — tool inference
# ---------------------------------------------------------------------------

class TestTraceToCaseTool:
    def test_must_call_from_tool_steps(self):
        trace = InputTrace.model_validate(SIMPLE_TRACE)
        case = trace_to_case(trace)
        assert case.expected.must_call == ["search_kb"]

    def test_must_call_unique_ordered(self):
        t = {
            "task": "q",
            "steps": [
                {"type": "tool_call", "tool": "a", "result": 1},
                {"type": "tool_call", "tool": "b", "result": 2},
                {"type": "tool_call", "tool": "a", "result": 3},  # duplicate
            ],
        }
        trace = InputTrace.model_validate(t)
        case = trace_to_case(trace)
        assert case.expected.must_call == ["a", "b"]

    def test_tool_mock_uses_last_result(self):
        t = {
            "task": "q",
            "steps": [
                {"type": "tool_call", "tool": "search", "result": "first"},
                {"type": "tool_call", "tool": "search", "result": "second"},
            ],
        }
        trace = InputTrace.model_validate(t)
        case = trace_to_case(trace)
        assert case.tools["search"].response == "second"

    def test_tool_mock_result_none_when_absent(self):
        t = {
            "task": "q",
            "steps": [{"type": "tool_call", "tool": "search"}],
        }
        trace = InputTrace.model_validate(t)
        case = trace_to_case(trace)
        assert case.tools["search"].response is None

    def test_no_tools_when_no_tool_calls(self):
        t = {
            "task": "q",
            "steps": [{"type": "thought", "content": "thinking"}],
        }
        trace = InputTrace.model_validate(t)
        case = trace_to_case(trace)
        assert case.tools == {}
        assert case.expected.must_call == []

    def test_tool_call_step_without_tool_field_is_ignored(self):
        # tool_call step missing the 'tool' name — should not crash or add to must_call
        t = {
            "task": "q",
            "steps": [{"type": "tool_call", "content": "something"}],
        }
        trace = InputTrace.model_validate(t)
        case = trace_to_case(trace)
        assert case.expected.must_call == []


# ---------------------------------------------------------------------------
# trace_to_case — step/tool limits
# ---------------------------------------------------------------------------

class TestTraceToCaseLimits:
    def test_max_steps_generous_headroom(self):
        # 3 steps → ceil(3 * 1.5) = 5
        trace = InputTrace.model_validate(SIMPLE_TRACE)
        case = trace_to_case(trace)
        assert case.expected.max_steps == 5

    def test_max_steps_minimum_three(self):
        t = {"task": "q", "steps": [{"type": "answer", "content": "a"}]}
        trace = InputTrace.model_validate(t)
        case = trace_to_case(trace)
        assert case.expected.max_steps == 3  # ceil(1 * 1.5) = 2, but minimum is 3

    def test_max_tool_calls_is_count_plus_one(self):
        # SIMPLE_TRACE has 1 tool call → max_tool_calls = 2
        trace = InputTrace.model_validate(SIMPLE_TRACE)
        case = trace_to_case(trace)
        assert case.expected.max_tool_calls == 2

    def test_max_tool_calls_minimum_one_when_no_tools(self):
        t = {"task": "q", "steps": [{"type": "thought", "content": "ok"}]}
        trace = InputTrace.model_validate(t)
        case = trace_to_case(trace)
        assert case.expected.max_tool_calls == 1
