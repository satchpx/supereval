"""Tests for multi-dimensional trajectory scoring."""
from __future__ import annotations

import pytest
from supereval.agent.models import (
    AgentExpected,
    AgentThresholds,
    AnswerMatch,
    MockResponse,
    MustCallWith,
    Step,
    StepType,
    ToolCall,
    Trajectory,
)
from supereval.agent.scorer import score_trajectory


def _make_trajectory(
    final_answer: str | None = "done",
    tool_names: list[str] | None = None,
    tool_args: dict[str, dict] | None = None,
    step_count: int | None = None,
    error: str | None = None,
) -> Trajectory:
    """Build a minimal Trajectory for scorer tests."""
    if error:
        return Trajectory(error=error)

    steps = []
    for name in (tool_names or []):
        args = (tool_args or {}).get(name, {})
        steps.append(Step(
            type=StepType.tool_call,
            content=f"calling {name}",
            tool_call=ToolCall(name=name, arguments=args, result="ok"),
        ))

    # Pad to desired step_count with thought steps
    while step_count and len(steps) < step_count:
        steps.insert(0, Step(type=StepType.thought, content="thinking"))

    if final_answer is not None:
        steps.append(Step(type=StepType.answer, content=final_answer))

    return Trajectory(steps=steps, final_answer=final_answer)


def _thresholds(**kwargs) -> AgentThresholds:
    return AgentThresholds(**kwargs)


# ---------------------------------------------------------------------------
# Answer scoring
# ---------------------------------------------------------------------------

class TestAnswerScoring:
    def test_contains_match_pass(self):
        traj = _make_trajectory(final_answer="The answer is us-west-2")
        score = score_trajectory(
            traj,
            AgentExpected(answer="us-west-2", answer_match=AnswerMatch.contains),
            _thresholds(),
        )
        assert score.answer_score == 1.0

    def test_contains_match_fail(self):
        traj = _make_trajectory(final_answer="I don't know")
        score = score_trajectory(
            traj,
            AgentExpected(answer="us-west-2", answer_match=AnswerMatch.contains),
            _thresholds(),
        )
        assert score.answer_score == 0.0

    def test_contains_is_case_insensitive(self):
        traj = _make_trajectory(final_answer="US-WEST-2")
        score = score_trajectory(
            traj,
            AgentExpected(answer="us-west-2", answer_match=AnswerMatch.contains),
            _thresholds(),
        )
        assert score.answer_score == 1.0

    def test_exact_match_pass(self):
        traj = _make_trajectory(final_answer="us-west-2")
        score = score_trajectory(
            traj,
            AgentExpected(answer="us-west-2", answer_match=AnswerMatch.exact),
            _thresholds(),
        )
        assert score.answer_score == 1.0

    def test_exact_match_fail(self):
        traj = _make_trajectory(final_answer="The answer is us-west-2")
        score = score_trajectory(
            traj,
            AgentExpected(answer="us-west-2", answer_match=AnswerMatch.exact),
            _thresholds(),
        )
        assert score.answer_score == 0.0

    def test_regex_match_pass(self):
        traj = _make_trajectory(final_answer="region: us-west-2")
        score = score_trajectory(
            traj,
            AgentExpected(answer=r"us-\w+-\d+", answer_match=AnswerMatch.regex),
            _thresholds(),
        )
        assert score.answer_score == 1.0

    def test_regex_match_fail(self):
        traj = _make_trajectory(final_answer="unknown")
        score = score_trajectory(
            traj,
            AgentExpected(answer=r"us-\w+-\d+", answer_match=AnswerMatch.regex),
            _thresholds(),
        )
        assert score.answer_score == 0.0

    def test_no_expected_answer_full_score(self):
        traj = _make_trajectory(final_answer="whatever")
        score = score_trajectory(
            traj,
            AgentExpected(answer=None),
            _thresholds(),
        )
        assert score.answer_score == 1.0

    def test_llm_judge_falls_back_to_contains_when_no_judge(self):
        traj = _make_trajectory(final_answer="us-west-2")
        score = score_trajectory(
            traj,
            AgentExpected(answer="us-west-2", answer_match=AnswerMatch.llm_judge),
            _thresholds(),
            judge=None,
        )
        assert score.answer_score == 1.0
        assert "fell back" in score.answer_reason

    def test_answer_below_threshold_adds_failure_reason(self):
        traj = _make_trajectory(final_answer="wrong")
        score = score_trajectory(
            traj,
            AgentExpected(answer="correct", answer_match=AnswerMatch.contains),
            _thresholds(min_answer_score=0.5),
        )
        assert score.passed is False
        assert any("Answer score" in r for r in score.failure_reasons)


# ---------------------------------------------------------------------------
# Tool scoring
# ---------------------------------------------------------------------------

class TestToolScoring:
    def test_required_tool_called(self):
        traj = _make_trajectory(tool_names=["search"])
        score = score_trajectory(
            traj,
            AgentExpected(must_call=["search"]),
            _thresholds(),
        )
        assert score.tool_score == 1.0
        assert score.missing_tools == []

    def test_required_tool_not_called(self):
        traj = _make_trajectory(tool_names=[])
        score = score_trajectory(
            traj,
            AgentExpected(must_call=["search"]),
            _thresholds(),
        )
        assert score.tool_score == 0.0
        assert "search" in score.missing_tools
        assert any("search" in r for r in score.failure_reasons)

    def test_forbidden_tool_not_called(self):
        traj = _make_trajectory(tool_names=["search"])
        score = score_trajectory(
            traj,
            AgentExpected(must_not_call=["delete"]),
            _thresholds(),
        )
        assert score.forbidden_tools_found == []
        assert score.tool_score == 1.0

    def test_forbidden_tool_called(self):
        traj = _make_trajectory(tool_names=["search", "delete"])
        score = score_trajectory(
            traj,
            AgentExpected(must_not_call=["delete"]),
            _thresholds(),
        )
        assert "delete" in score.forbidden_tools_found
        assert any("delete" in r for r in score.failure_reasons)

    def test_must_call_with_args_satisfied(self):
        traj = _make_trajectory(
            tool_names=["search"],
            tool_args={"search": {"query": "s3"}},
        )
        score = score_trajectory(
            traj,
            AgentExpected(
                must_call_with=[MustCallWith(tool="search", args={"query": "s3"})]
            ),
            _thresholds(),
        )
        assert score.missing_args == []
        assert score.tool_score == 1.0

    def test_must_call_with_args_not_satisfied(self):
        traj = _make_trajectory(
            tool_names=["search"],
            tool_args={"search": {"query": "ec2"}},
        )
        score = score_trajectory(
            traj,
            AgentExpected(
                must_call_with=[MustCallWith(tool="search", args={"query": "s3"})]
            ),
            _thresholds(),
        )
        assert len(score.missing_args) == 1
        assert score.tool_score < 1.0

    def test_no_tool_checks_full_score(self):
        traj = _make_trajectory(tool_names=["search"])
        score = score_trajectory(traj, AgentExpected(), _thresholds())
        assert score.tool_score == 1.0

    def test_partial_tool_score(self):
        # 2 required tools, 1 missing → tool_score = 0.5
        traj = _make_trajectory(tool_names=["search"])
        score = score_trajectory(
            traj,
            AgentExpected(must_call=["search", "get_item"]),
            _thresholds(min_tool_score=0.0),
        )
        assert score.tool_score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Step / tool-call limits
# ---------------------------------------------------------------------------

class TestLimits:
    def test_step_limit_exceeded(self):
        traj = _make_trajectory(tool_names=["a", "b", "c"], step_count=10)
        score = score_trajectory(
            traj,
            AgentExpected(max_steps=5),
            _thresholds(),
        )
        assert any("Step count" in r for r in score.failure_reasons)

    def test_step_limit_not_exceeded(self):
        traj = _make_trajectory(tool_names=["a"], step_count=3)
        score = score_trajectory(
            traj,
            AgentExpected(max_steps=5),
            _thresholds(),
        )
        assert not any("Step count" in r for r in score.failure_reasons)

    def test_tool_call_limit_exceeded(self):
        traj = _make_trajectory(tool_names=["a", "b", "c"])
        score = score_trajectory(
            traj,
            AgentExpected(max_tool_calls=2),
            _thresholds(),
        )
        assert any("Tool call count" in r for r in score.failure_reasons)


# ---------------------------------------------------------------------------
# Efficiency scoring
# ---------------------------------------------------------------------------

class TestEfficiencyScoring:
    def test_single_step_is_perfect(self):
        # No padding + no tools → just 1 answer step (step_count=1)
        traj = _make_trajectory(tool_names=[])
        score = score_trajectory(
            traj,
            AgentExpected(max_steps=5),
            _thresholds(),
        )
        assert score.efficiency_score == pytest.approx(1.0)

    def test_at_max_steps_is_zero(self):
        # 4 thought steps padded + 1 answer step = 5 steps = max_steps → efficiency 0.0
        traj = _make_trajectory(tool_names=[], step_count=4)
        score = score_trajectory(
            traj,
            AgentExpected(max_steps=5),
            _thresholds(),
        )
        assert score.efficiency_score == pytest.approx(0.0)

    def test_no_max_steps_is_full(self):
        traj = _make_trajectory(tool_names=["a", "b", "c"])
        score = score_trajectory(traj, AgentExpected(), _thresholds())
        assert score.efficiency_score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------

class TestCompositeScore:
    def test_perfect_run(self):
        # No max_steps → efficiency_score defaults to 1.0, composite reaches 1.0
        traj = _make_trajectory(final_answer="us-west-2", tool_names=["search"])
        score = score_trajectory(
            traj,
            AgentExpected(answer="us-west-2", must_call=["search"]),
            _thresholds(),
        )
        assert score.composite_score == pytest.approx(1.0, abs=0.01)
        assert score.passed is True

    def test_composite_bounded_zero_to_one(self):
        traj = _make_trajectory(final_answer="wrong")
        score = score_trajectory(
            traj,
            AgentExpected(answer="right", must_call=["missing_tool"]),
            _thresholds(min_answer_score=0.0, min_tool_score=0.0),
        )
        assert 0.0 <= score.composite_score <= 1.0

    def test_crash_trajectory_fails(self):
        traj = _make_trajectory(error="timeout")
        score = score_trajectory(traj, AgentExpected(), _thresholds())
        assert score.passed is False
        assert any("Agent error" in r for r in score.failure_reasons)
