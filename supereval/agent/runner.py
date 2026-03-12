"""
Agent runner: protocol definition, mock tool executor, and eval orchestrator.

AgentRunner protocol
--------------------
Customers implement this for their agent framework:

    class MyAgent:
        def run(self, task: str, tool_executor: ToolExecutor) -> Trajectory:
            ...

Pass it to `supereval agent run` via --runner myapp.agents:MyAgent.

MockToolExecutor
----------------
Wraps the per-case tool mock definitions from AgentTestCase.tools.
The agent calls execute() thinking it's talking to real tools; it gets
back the canned responses defined in the test case.

run_agent_eval
--------------
Orchestrates: load dataset → iterate cases → run agent with mocks → score.
"""
from __future__ import annotations

import importlib
import time
from typing import Any, Protocol, runtime_checkable

from .judge import BedrockJudge
from .models import (
    AgentCaseResult,
    AgentRunResult,
    MockResponse,
    ToolCall,
    ToolSpec,
    Trajectory,
    Step,
    StepType,
)
from .scorer import score_trajectory
from .storage import load_agent_cases, load_agent_dataset_meta


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

@runtime_checkable
class ToolExecutor(Protocol):
    def execute(self, tool_name: str, arguments: dict) -> Any: ...
    def available_tools(self) -> list[ToolSpec]: ...


@runtime_checkable
class AgentRunner(Protocol):
    def run(self, task: str, tool_executor: ToolExecutor) -> Trajectory: ...


# ---------------------------------------------------------------------------
# Mock tool executor
# ---------------------------------------------------------------------------

class MockToolExecutor:
    """Returns canned responses from an AgentTestCase's tools block."""

    def __init__(
        self,
        mocks: dict[str, MockResponse],
        tool_specs: list[ToolSpec],
    ):
        self._mocks = mocks
        self._specs = tool_specs

    def execute(self, tool_name: str, arguments: dict) -> Any:
        if tool_name not in self._mocks:
            available = list(self._mocks.keys())
            raise ValueError(
                f"Tool '{tool_name}' is not defined in this test case. "
                f"Available: {available}"
            )
        mock = self._mocks[tool_name]
        if mock.error:
            raise RuntimeError(mock.error)
        return mock.response

    def available_tools(self) -> list[ToolSpec]:
        return self._specs


# ---------------------------------------------------------------------------
# Runner loading
# ---------------------------------------------------------------------------

def load_runner(import_path: str) -> AgentRunner:
    """
    Load an AgentRunner from a 'module.path:ClassName' string.

    Example: myapp.agents:ReActAgent
    """
    if ":" not in import_path:
        raise ValueError(
            f"--runner must be 'module.path:ClassName', got: {import_path!r}"
        )
    module_path, class_name = import_path.rsplit(":", 1)
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Could not import module '{module_path}': {e}"
        ) from e
    if not hasattr(module, class_name):
        raise AttributeError(
            f"Module '{module_path}' has no attribute '{class_name}'"
        )
    cls = getattr(module, class_name)
    instance = cls()
    if not isinstance(instance, AgentRunner):
        raise TypeError(
            f"{import_path} does not implement AgentRunner "
            f"(needs run(task, tool_executor) -> Trajectory)"
        )
    return instance


# ---------------------------------------------------------------------------
# Eval orchestrator
# ---------------------------------------------------------------------------

def run_agent_eval(
    dataset_name: str,
    runner: AgentRunner,
    runner_id: str = "",
    judge_model: str | None = None,
    judge_region: str = "us-east-1",
) -> AgentRunResult:
    """
    Run an agent eval against every case in a dataset.

    Each case provides a MockToolExecutor so the agent runs deterministically
    against canned tool responses rather than live systems.
    """
    meta = load_agent_dataset_meta(dataset_name)
    cases = load_agent_cases(dataset_name)

    if not cases:
        raise ValueError(
            f"Agent dataset '{dataset_name}' has no cases. "
            f"Add cases with: supereval agent dataset add-cases {dataset_name} --from cases.jsonl"
        )

    judge = (
        BedrockJudge(model_id=judge_model, region=judge_region)
        if judge_model
        else None
    )

    case_results: list[AgentCaseResult] = []

    for case in cases:
        tool_executor = MockToolExecutor(
            mocks=case.tools,
            tool_specs=meta.tools,
        )

        try:
            t0 = time.monotonic()
            trajectory = runner.run(
                task=case.input.task,
                tool_executor=tool_executor,
            )
            elapsed_ms = int((time.monotonic() - t0) * 1000)
        except Exception as exc:
            # Agent crashed entirely — record an error trajectory
            trajectory = Trajectory(error=str(exc))
            elapsed_ms = 0

        traj_score = score_trajectory(
            trajectory=trajectory,
            expected=case.expected,
            thresholds=meta.thresholds,
            task=case.input.task,
            judge=judge,
        )

        case_results.append(
            AgentCaseResult(
                case_id=case.id,
                description=case.description,
                vars={"task": case.input.task},
                trajectory=trajectory,
                score=traj_score,
                latency_ms=trajectory.total_latency_ms or elapsed_ms,
                cost_usd=trajectory.total_cost_usd,
            )
        )

    return AgentRunResult(
        dataset=dataset_name,
        cases=case_results,
        runner_id=runner_id,
    )
