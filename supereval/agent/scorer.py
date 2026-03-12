"""
Multi-dimensional trajectory scoring.

Dimensions
----------
answer_score     — correctness of the final answer (exact / contains / regex / llm_judge)
tool_score       — required tools called, forbidden tools not called, args matched
efficiency_score — how close to optimal step count (informational by default)
reasoning_score  — LLM judge on thought chain quality (optional; requires BedrockJudge)
composite_score  — weighted average of the above

A case passes when all threshold checks pass and there are no failure_reasons.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .models import (
    AgentExpected,
    AgentThresholds,
    AnswerMatch,
    Trajectory,
    TrajectoryScore,
)

if TYPE_CHECKING:
    from .judge import BedrockJudge


def score_trajectory(
    trajectory: Trajectory,
    expected: AgentExpected,
    thresholds: AgentThresholds,
    task: str = "",
    judge: "BedrockJudge | None" = None,
) -> TrajectoryScore:
    score = TrajectoryScore()
    tool_calls = trajectory.tool_calls
    tool_names_called = {tc.name for tc in tool_calls}
    score.step_count = len(trajectory.steps)
    score.tool_call_count = len(tool_calls)

    # --- Agent crashed ---
    if trajectory.error:
        score.failure_reasons.append(f"Agent error: {trajectory.error}")
        return score

    # -----------------------------------------------------------------------
    # Answer scoring
    # -----------------------------------------------------------------------
    if expected.answer is not None:
        answer_text = trajectory.final_answer or ""

        if expected.answer_match == AnswerMatch.exact:
            score.answer_score = (
                1.0 if answer_text.strip() == expected.answer.strip() else 0.0
            )
        elif expected.answer_match == AnswerMatch.contains:
            score.answer_score = (
                1.0 if expected.answer.lower() in answer_text.lower() else 0.0
            )
        elif expected.answer_match == AnswerMatch.regex:
            score.answer_score = (
                1.0 if re.search(expected.answer, answer_text) else 0.0
            )
        elif expected.answer_match == AnswerMatch.llm_judge:
            if judge:
                result = judge.judge_answer(
                    task=task,
                    trajectory=trajectory,
                    expected_answer=expected.answer,
                    rubric=expected.rubric,
                )
                score.answer_score = result.score
                score.answer_reason = result.reason
            else:
                # Graceful fallback when no judge is configured
                score.answer_score = (
                    1.0
                    if expected.answer.lower() in answer_text.lower()
                    else 0.0
                )
                score.answer_reason = (
                    "llm_judge requested but no --judge-model configured; "
                    "fell back to contains match"
                )
    else:
        score.answer_score = 1.0  # no expected answer = full marks

    if score.answer_score < thresholds.min_answer_score:
        msg = (
            f"Answer score {score.answer_score:.2f} below threshold "
            f"{thresholds.min_answer_score:.2f}"
        )
        if score.answer_reason:
            msg += f": {score.answer_reason}"
        score.failure_reasons.append(msg)

    # -----------------------------------------------------------------------
    # Tool scoring
    # -----------------------------------------------------------------------
    # Required tools
    score.missing_tools = [
        t for t in expected.must_call if t not in tool_names_called
    ]
    for t in score.missing_tools:
        score.failure_reasons.append(f"Required tool not called: '{t}'")

    # Forbidden tools
    score.forbidden_tools_found = [
        t for t in expected.must_not_call if t in tool_names_called
    ]
    for t in score.forbidden_tools_found:
        score.failure_reasons.append(f"Forbidden tool was called: '{t}'")

    # Argument checks (subset match — all specified keys must match)
    for req in expected.must_call_with:
        matching = [tc for tc in tool_calls if tc.name == req.tool]
        if not matching:
            score.missing_args.append(f"{req.tool}(not called)")
            score.failure_reasons.append(
                f"Tool '{req.tool}' required by must_call_with but was never called"
            )
            continue
        satisfied = any(
            all(tc.arguments.get(k) == v for k, v in req.args.items())
            for tc in matching
        )
        if not satisfied:
            desc = ", ".join(f"{k}={v!r}" for k, v in req.args.items())
            score.missing_args.append(f"{req.tool}({desc})")
            score.failure_reasons.append(
                f"Tool '{req.tool}' called but never with required args: {desc}"
            )

    total_tool_checks = (
        len(expected.must_call)
        + len(expected.must_not_call)
        + len(expected.must_call_with)
    )
    if total_tool_checks == 0:
        score.tool_score = 1.0
    else:
        failures = (
            len(score.missing_tools)
            + len(score.forbidden_tools_found)
            + len(score.missing_args)
        )
        score.tool_score = max(0.0, 1.0 - failures / total_tool_checks)

    if score.tool_score < thresholds.min_tool_score:
        # Individual reasons already appended above; nothing extra needed
        pass

    # -----------------------------------------------------------------------
    # Step / tool-call limits
    # -----------------------------------------------------------------------
    if expected.max_steps is not None and score.step_count > expected.max_steps:
        score.failure_reasons.append(
            f"Step count {score.step_count} exceeds max {expected.max_steps}"
        )

    if (
        expected.max_tool_calls is not None
        and score.tool_call_count > expected.max_tool_calls
    ):
        score.failure_reasons.append(
            f"Tool call count {score.tool_call_count} exceeds max "
            f"{expected.max_tool_calls}"
        )

    # -----------------------------------------------------------------------
    # Efficiency scoring (informational; 1 step = perfect)
    # -----------------------------------------------------------------------
    if expected.max_steps and expected.max_steps > 1:
        over = max(0, score.step_count - 1)
        score.efficiency_score = max(0.0, 1.0 - over / (expected.max_steps - 1))
    else:
        score.efficiency_score = 1.0

    # -----------------------------------------------------------------------
    # Reasoning scoring (optional LLM judge)
    # -----------------------------------------------------------------------
    if judge is not None and thresholds.min_reasoning_score is not None:
        result = judge.judge_reasoning(task=task, trajectory=trajectory)
        score.reasoning_score = result.score
        score.reasoning_reason = result.reason
        if score.reasoning_score < thresholds.min_reasoning_score:
            score.failure_reasons.append(
                f"Reasoning score {score.reasoning_score:.2f} below threshold "
                f"{thresholds.min_reasoning_score:.2f}: {result.reason}"
            )

    # -----------------------------------------------------------------------
    # Composite score (weighted average; reasoning weight dropped if not run)
    # -----------------------------------------------------------------------
    rs = score.reasoning_score if score.reasoning_score is not None else None
    aw = thresholds.answer_weight
    tw = thresholds.tool_weight
    rw = thresholds.reasoning_weight if rs is not None else 0.0
    ew = thresholds.efficiency_weight
    total_w = aw + tw + rw + ew

    score.composite_score = (
        score.answer_score * aw
        + score.tool_score * tw
        + (rs if rs is not None else 0.0) * rw
        + score.efficiency_score * ew
    ) / total_w if total_w > 0 else 0.0

    score.passed = len(score.failure_reasons) == 0
    return score
