"""
LLM-as-judge scoring via Amazon Bedrock.

Used for two scoring dimensions:
  - Answer judge:    is the agent's final answer correct?
  - Reasoning judge: is the thought chain coherent and hallucination-free?

Both judges return a JudgeResult with a float score (0.0–1.0) and a short reason.
Uses the same Bedrock invoke pattern as generator.py.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from .models import Trajectory

# Re-use the same defaults as the generator
_DEFAULT_MODEL = "anthropic.claude-3-5-sonnet-20241022-v2:0"
_DEFAULT_REGION = "us-east-1"

_ANSWER_JUDGE_PROMPT = """\
You are an impartial evaluator assessing whether an AI agent correctly answered a task.

## Task
{task}

## Agent trajectory
{trajectory_text}

## Agent's final answer
{final_answer}

## Expected answer
{expected_answer}
{rubric_section}
Score the final answer from 0.0 to 1.0:
- 1.0 = Fully correct and complete
- 0.7 = Mostly correct, minor gaps or imprecision
- 0.4 = Partially correct, significant gaps
- 0.0 = Wrong or no answer given

Respond with a JSON object only: {{"score": <float>, "reason": "<one sentence>"}}"""

_REASONING_JUDGE_PROMPT = """\
You are an impartial evaluator assessing the quality of an AI agent's reasoning trajectory.

## Task
{task}

## Agent trajectory
{trajectory_text}

Evaluate on four dimensions:
1. Coherence — does each step logically follow from the previous?
2. Accuracy  — does the agent correctly interpret tool results without hallucinating?
3. Efficiency — does the agent avoid unnecessary or redundant tool calls?
4. Correctness — does the agent avoid logical errors or wrong conclusions?

Score the trajectory from 0.0 to 1.0:
- 1.0 = Excellent: coherent, accurate, efficient, no errors
- 0.7 = Good: minor inefficiencies or small errors
- 0.4 = Flawed: hallucinations or significant errors
- 0.0 = Incoherent or entirely wrong

Respond with a JSON object only: {{"score": <float>, "reason": "<one sentence>"}}"""


@dataclass
class JudgeResult:
    score: float    # 0.0–1.0
    reason: str


def _format_trajectory(trajectory: Trajectory) -> str:
    lines = []
    for i, step in enumerate(trajectory.steps, 1):
        if step.tool_call:
            tc = step.tool_call
            lines.append(
                f"Step {i} [tool_call] {tc.name}({json.dumps(tc.arguments)})"
            )
            if tc.error:
                lines.append(f"  → ERROR: {tc.error}")
            else:
                lines.append(f"  → {json.dumps(tc.result)}")
        else:
            lines.append(f"Step {i} [{step.type.value}] {step.content}")
    return "\n".join(lines) if lines else "(no steps)"


class BedrockJudge:
    def __init__(
        self,
        model_id: str | None = None,
        region: str = _DEFAULT_REGION,
    ):
        self._model_id = model_id or _DEFAULT_MODEL
        self._region = region
        self._client = None

    def _get_client(self):
        if self._client is None:
            import boto3
            self._client = boto3.client(
                "bedrock-runtime", region_name=self._region
            )
        return self._client

    def _call(self, prompt: str) -> JudgeResult:
        client = self._get_client()
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": prompt}],
        })
        response = client.invoke_model(
            modelId=self._model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        text = json.loads(response["body"].read())["content"][0]["text"].strip()

        # Strip markdown fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:])
            if text.endswith("```"):
                text = text[:-3].strip()

        try:
            parsed = json.loads(text)
            return JudgeResult(
                score=max(0.0, min(1.0, float(parsed.get("score", 0.0)))),
                reason=str(parsed.get("reason", "")),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return JudgeResult(
                score=0.5,
                reason=f"Could not parse judge response: {text[:120]}",
            )

    def judge_answer(
        self,
        task: str,
        trajectory: Trajectory,
        expected_answer: str,
        rubric: str | None = None,
    ) -> JudgeResult:
        rubric_section = f"\n## Rubric\n{rubric}" if rubric else ""
        prompt = _ANSWER_JUDGE_PROMPT.format(
            task=task,
            trajectory_text=_format_trajectory(trajectory),
            final_answer=trajectory.final_answer or "(no answer given)",
            expected_answer=expected_answer,
            rubric_section=rubric_section,
        )
        return self._call(prompt)

    def judge_reasoning(self, task: str, trajectory: Trajectory) -> JudgeResult:
        prompt = _REASONING_JUDGE_PROMPT.format(
            task=task,
            trajectory_text=_format_trajectory(trajectory),
        )
        return self._call(prompt)
