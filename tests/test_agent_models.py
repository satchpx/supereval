"""Tests for agent data models: validation, properties, serialisation."""
from __future__ import annotations

import pytest
from supereval.agent.models import (
    AgentCaseResult,
    AgentExpected,
    AgentRunResult,
    AgentTask,
    AgentTestCase,
    AgentThresholds,
    AnswerMatch,
    MockResponse,
    MustCallWith,
    Step,
    StepType,
    ToolCall,
    ToolSpec,
    Trajectory,
    TrajectoryScore,
)


class TestTrajectory:
    def test_tool_calls_property(self):
        tc = ToolCall(name="search", arguments={"q": "s3"}, result=["r1"])
        steps = [
            Step(type=StepType.thought, content="thinking"),
            Step(type=StepType.tool_call, content="searching", tool_call=tc),
            Step(type=StepType.observation, content="saw result"),
        ]
        traj = Trajectory(steps=steps)
        assert len(traj.tool_calls) == 1
        assert traj.tool_calls[0].name == "search"

    def test_tool_call_names_property(self):
        traj = Trajectory(steps=[
            Step(type=StepType.tool_call, content="",
                 tool_call=ToolCall(name="a", result=None)),
            Step(type=StepType.tool_call, content="",
                 tool_call=ToolCall(name="b", result=None)),
        ])
        assert traj.tool_call_names == ["a", "b"]

    def test_empty_trajectory(self):
        traj = Trajectory()
        assert traj.tool_calls == []
        assert traj.final_answer is None
        assert traj.error is None

    def test_error_trajectory(self):
        traj = Trajectory(error="timeout")
        assert traj.error == "timeout"


class TestAgentExpected:
    def test_defaults(self):
        e = AgentExpected()
        assert e.answer is None
        assert e.answer_match == AnswerMatch.contains
        assert e.must_call == []
        assert e.must_not_call == []
        assert e.must_call_with == []
        assert e.max_steps is None

    def test_must_call_with_validation(self):
        e = AgentExpected(
            must_call_with=[MustCallWith(tool="search", args={"query": "s3"})]
        )
        assert e.must_call_with[0].tool == "search"
        assert e.must_call_with[0].args == {"query": "s3"}


class TestAgentTestCase:
    def test_auto_id(self):
        case = AgentTestCase(
            input=AgentTask(task="test"),
            expected=AgentExpected(),
        )
        assert case.id.startswith("atc_")

    def test_tools_default_empty(self):
        case = AgentTestCase(input=AgentTask(task="t"), expected=AgentExpected())
        assert case.tools == {}

    def test_roundtrip_json(self):
        case = AgentTestCase(
            input=AgentTask(task="What is S3?"),
            tools={"search": MockResponse(response=["result"], latency_ms=50)},
            expected=AgentExpected(answer="storage", must_call=["search"]),
        )
        json_str = case.model_dump_json()
        loaded = AgentTestCase.model_validate_json(json_str)
        assert loaded.input.task == "What is S3?"
        assert loaded.tools["search"].response == ["result"]
        assert loaded.expected.must_call == ["search"]


class TestAgentThresholds:
    def test_defaults(self):
        t = AgentThresholds()
        assert t.pass_rate == 1.0
        assert t.min_answer_score == 0.8
        assert t.min_tool_score == 1.0
        assert t.min_reasoning_score is None
        assert t.fail_on_regression is True
        assert t.max_cost_usd is None
        assert t.max_p95_latency_ms is None

    def test_weights_sum_to_one(self):
        t = AgentThresholds()
        total = t.answer_weight + t.tool_weight + t.reasoning_weight + t.efficiency_weight
        assert abs(total - 1.0) < 1e-9


class TestAgentRunResult:
    def _make_result(self, passed_flags: list[bool]) -> AgentRunResult:
        cases = [
            AgentCaseResult(
                case_id=f"c{i}",
                description="",
                vars={"task": f"task{i}"},
                trajectory=Trajectory(),
                score=TrajectoryScore(
                    passed=p,
                    composite_score=1.0 if p else 0.0,
                    answer_score=1.0 if p else 0.0,
                    tool_score=1.0,
                ),
                latency_ms=100 * (i + 1),
                cost_usd=0.01 * (i + 1),
            )
            for i, p in enumerate(passed_flags)
        ]
        return AgentRunResult(dataset="test-agent", cases=cases)

    def test_pass_rate(self):
        r = self._make_result([True, True, False])
        assert r.pass_rate == pytest.approx(2 / 3, abs=0.001)

    def test_total_cost(self):
        r = self._make_result([True, False])
        assert r.total_cost_usd == pytest.approx(0.03, abs=1e-6)

    def test_latency_percentiles(self):
        r = self._make_result([True, False, True])
        # latencies: 100, 200, 300
        assert r.avg_latency_ms == pytest.approx(200.0)
        assert r.p95_latency_ms >= 200

    def test_to_dict_shape(self):
        r = self._make_result([True, False])
        d = r.to_dict()
        assert "run_id" in d
        assert "summary" in d
        assert d["summary"]["total"] == 2
        assert d["summary"]["passed"] == 1
        assert len(d["cases"]) == 2

    def test_avg_scores(self):
        r = self._make_result([True, False])
        assert r.avg_composite_score == pytest.approx(0.5)
        assert r.avg_answer_score == pytest.approx(0.5)
