"""Tests for MockToolExecutor, load_runner, and run_agent_eval."""
from __future__ import annotations

import pytest
from supereval.agent.models import (
    AgentExpected,
    AgentTask,
    AgentTestCase,
    MockResponse,
    Step,
    StepType,
    ToolCall,
    ToolSpec,
    Trajectory,
)
from supereval.agent.runner import MockToolExecutor, load_runner, run_agent_eval
from supereval.agent.storage import append_agent_cases, save_agent_dataset_meta


# ---------------------------------------------------------------------------
# A minimal in-process AgentRunner for testing
# ---------------------------------------------------------------------------

class DummyAgent:
    """
    Calls every available tool once (no args), then returns 'done' as the answer.
    Used as a predictable stand-in for a real agent.
    """
    def run(self, task: str, tool_executor) -> Trajectory:
        steps = [Step(type=StepType.thought, content=f"solving: {task}")]
        for spec in tool_executor.available_tools():
            result = tool_executor.execute(spec.name, {})
            steps.append(Step(
                type=StepType.tool_call,
                content=f"called {spec.name}",
                tool_call=ToolCall(name=spec.name, arguments={}, result=result),
            ))
            steps.append(Step(type=StepType.observation, content=str(result)))
        steps.append(Step(type=StepType.answer, content="done"))
        return Trajectory(steps=steps, final_answer="done")


class CrashingAgent:
    """Always raises an exception."""
    def run(self, task: str, tool_executor) -> Trajectory:
        raise RuntimeError("agent exploded")


class NoAnswerAgent:
    """Returns a trajectory with no final_answer."""
    def run(self, task: str, tool_executor) -> Trajectory:
        return Trajectory(steps=[Step(type=StepType.thought, content="hmm")])


# ---------------------------------------------------------------------------
# MockToolExecutor
# ---------------------------------------------------------------------------

class TestMockToolExecutor:
    def _specs(self):
        return [ToolSpec(name="search", description="Search")]

    def test_returns_canned_response(self):
        mocks = {"search": MockResponse(response=["result1"])}
        executor = MockToolExecutor(mocks=mocks, tool_specs=self._specs())
        assert executor.execute("search", {}) == ["result1"]

    def test_raises_on_undefined_tool(self):
        executor = MockToolExecutor(mocks={}, tool_specs=self._specs())
        with pytest.raises(ValueError, match="not defined"):
            executor.execute("nonexistent", {})

    def test_raises_runtime_error_on_mock_error(self):
        mocks = {"search": MockResponse(error="AccessDenied")}
        executor = MockToolExecutor(mocks=mocks, tool_specs=self._specs())
        with pytest.raises(RuntimeError, match="AccessDenied"):
            executor.execute("search", {})

    def test_available_tools_returns_specs(self):
        specs = self._specs()
        executor = MockToolExecutor(mocks={}, tool_specs=specs)
        assert executor.available_tools() == specs

    def test_none_response_allowed(self):
        mocks = {"search": MockResponse(response=None)}
        executor = MockToolExecutor(mocks=mocks, tool_specs=self._specs())
        assert executor.execute("search", {}) is None


# ---------------------------------------------------------------------------
# load_runner
# ---------------------------------------------------------------------------

class TestLoadRunner:
    def test_rejects_path_without_colon(self):
        with pytest.raises(ValueError, match="module.path:ClassName"):
            load_runner("mymodule.MyClass")

    def test_rejects_missing_module(self):
        with pytest.raises(ImportError):
            load_runner("nonexistent_module_xyz:MyClass")

    def test_rejects_missing_class(self):
        with pytest.raises(AttributeError):
            load_runner("supereval.agent.runner:NoSuchClass")

    def test_rejects_non_runner(self):
        # str doesn't implement AgentRunner
        with pytest.raises(TypeError, match="AgentRunner"):
            load_runner("builtins:str")


# ---------------------------------------------------------------------------
# run_agent_eval
# ---------------------------------------------------------------------------

class TestRunAgentEval:
    def test_raises_on_empty_dataset(self, agent_meta):
        with pytest.raises(ValueError, match="no cases"):
            run_agent_eval("test-agent", DummyAgent())

    def test_runs_all_cases(self, agent_dataset_with_cases):
        result = run_agent_eval("test-agent", DummyAgent())
        assert result.total == 2

    def test_crash_agent_produces_failed_case(self, agent_meta, sample_agent_cases):
        append_agent_cases("test-agent", [sample_agent_cases[0]])
        result = run_agent_eval("test-agent", CrashingAgent())
        assert result.total == 1
        assert result.cases[0].passed is False
        assert result.cases[0].trajectory.error is not None

    def test_runner_id_stored(self, agent_dataset_with_cases):
        result = run_agent_eval("test-agent", DummyAgent(), runner_id="tests:DummyAgent")
        assert result.runner_id == "tests:DummyAgent"

    def test_case_latency_recorded(self, agent_dataset_with_cases):
        result = run_agent_eval("test-agent", DummyAgent())
        for case in result.cases:
            assert case.latency_ms >= 0

    def test_no_answer_agent_fails_answer_check(self, agent_meta, sample_agent_cases):
        append_agent_cases("test-agent", [sample_agent_cases[0]])
        result = run_agent_eval("test-agent", NoAnswerAgent())
        # case[0] expects "us-west-2" in answer — NoAnswerAgent gives None
        assert result.cases[0].score.answer_score == 0.0
