"""
LLM-as-judge scoring via Amazon Bedrock.

Implements four focused metrics modelled on AWS Bedrock's built-in judge suite:

  Answer judge (2 calls):
    - Correctness      — is the final answer correct?  (3-point scale)
    - Completeness     — does the answer cover everything needed?  (5-point scale)

  Reasoning judge (2 calls):
    - Faithfulness     — did the agent hallucinate beyond tool results?  (5-point scale)
    - Logical Coherence — did each reasoning step follow from the previous?  (5-point scale)

Each metric uses a dedicated prompt with a metric-appropriate ordinal scale.
Scores are surfaced both as native labels ("partially correct", "most faithful")
and as normalized 0.0–1.0 floats for composite scoring in scorer.py.

Uses the same Bedrock invoke pattern as generator.py.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from .models import Trajectory

_DEFAULT_MODEL = "anthropic.claude-3-5-sonnet-20241022-v2:0"
_DEFAULT_REGION = "us-east-1"

# ---------------------------------------------------------------------------
# Ordinal scale definitions
# ---------------------------------------------------------------------------

# Correctness: 3-point
_CORRECTNESS_LABELS: dict[str, tuple[int, float]] = {
    "incorrect":          (0, 0.0),
    "partially correct":  (1, 0.5),
    "correct":            (2, 1.0),
}

# Completeness / Logical Coherence: 5-point Likert
_LIKERT5_LABELS: dict[str, tuple[int, float]] = {
    "not at all":     (0, 0.00),
    "not generally":  (1, 0.25),
    "neutral/mixed":  (2, 0.50),
    "generally yes":  (3, 0.75),
    "yes":            (4, 1.00),
}

# Faithfulness: 5-point
_FAITHFULNESS_LABELS: dict[str, tuple[int, float]] = {
    "none":                 (0, 0.00),
    "some":                 (1, 0.25),
    "approximately half":   (2, 0.50),
    "most":                 (3, 0.75),
    "all":                  (4, 1.00),
}

# Helpfulness: 5-point
_HELPFULNESS_LABELS: dict[str, tuple[int, float]] = {
    "not helpful":       (0, 0.00),
    "slightly helpful":  (1, 0.25),
    "moderately helpful": (2, 0.50),
    "helpful":           (3, 0.75),
    "very helpful":      (4, 1.00),
}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_CORRECTNESS_PROMPT = """\
You are an impartial evaluator assessing whether an AI agent's final answer is correct.

## Task
{task}

## Agent's final answer
{final_answer}
{expected_section}
{rubric_section}
Is the agent's final answer correct?

Respond with a JSON object only:
{{"reasoning": "<one or two sentences>", "answer": "<incorrect|partially correct|correct>"}}"""


_COMPLETENESS_PROMPT = """\
You are an impartial evaluator assessing whether an AI agent's answer is complete.

## Task
{task}

## Agent's final answer
{final_answer}
{expected_section}
Does the agent's answer address every part of the task and include all necessary information?

Respond with a JSON object only:
{{"reasoning": "<one or two sentences>", "answer": "<not at all|not generally|neutral/mixed|generally yes|yes>"}}"""


_HELPFULNESS_PROMPT = """\
You are an impartial evaluator assessing whether an AI agent's response is helpful to the user.

## Task
{task}

## Agent's final answer
{final_answer}
{expected_section}
How helpful is the agent's response for accomplishing the user's task? \
Consider relevance, actionability, and whether it addresses the user's actual need.

Respond with a JSON object only:
{{"reasoning": "<one or two sentences>", "answer": "<not helpful|slightly helpful|moderately helpful|helpful|very helpful>"}}"""


_FAITHFULNESS_PROMPT = """\
You are an impartial evaluator assessing whether an AI agent's answer is faithful to the \
information it gathered — i.e., whether it hallucinated facts not present in the tool results.

## Task
{task}

## Agent trajectory (tool calls and results)
{trajectory_text}

## Agent's final answer
{final_answer}

How much of the agent's final answer is grounded in (faithful to) the tool results above, \
rather than fabricated or hallucinated?

Respond with a JSON object only:
{{"reasoning": "<one or two sentences>", "answer": "<none|some|approximately half|most|all>"}}"""


_COHERENCE_PROMPT = """\
You are an impartial evaluator assessing whether an AI agent's reasoning steps follow \
logically from one another.

## Task
{task}

## Agent trajectory
{trajectory_text}

Does each step in the agent's reasoning follow logically from the previous step? \
Are there gaps, contradictions, or non-sequiturs in the chain of thought?

Respond with a JSON object only:
{{"reasoning": "<one or two sentences>", "answer": "<not at all|not generally|neutral/mixed|generally yes|yes>"}}"""


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class MetricScore:
    raw: int            # native ordinal value (e.g. 0, 1, 2)
    label: str          # human-readable label ("partially correct", "most faithful", etc.)
    normalized: float   # 0.0–1.0 for composite scoring
    max_raw: int        # denominator (2 for correctness, 4 for 5-point scales)


@dataclass
class AnswerJudgeResult:
    correctness: MetricScore    # 3-point (0–2)
    completeness: MetricScore   # 5-point (0–4)
    helpfulness: MetricScore    # 5-point (0–4)
    score: float                # normalized composite (avg of all three) — for scorer.py compat
    reason: str                 # combined reasoning summary


@dataclass
class ReasoningJudgeResult:
    faithfulness: MetricScore       # 5-point (0–4)
    logical_coherence: MetricScore  # 5-point (0–4)
    score: float                    # normalized composite (avg of both) — for scorer.py compat
    reason: str                     # combined reasoning summary


# ---------------------------------------------------------------------------
# Trajectory formatter (unchanged)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# BedrockJudge
# ---------------------------------------------------------------------------

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

    def _invoke(self, prompt: str) -> str:
        """Send a prompt to Bedrock and return the raw text response."""
        client = self._get_client()
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 512,
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
        return text

    def _call_metric(
        self,
        prompt: str,
        label_map: dict[str, tuple[int, float]],
        max_raw: int,
    ) -> tuple[MetricScore, str]:
        """
        Invoke Bedrock and parse a label-based metric response.

        Returns (MetricScore, reasoning_text).
        Falls back to the midpoint label on parse failure.
        """
        try:
            text = self._invoke(prompt)
            parsed = json.loads(text)
            label = str(parsed.get("answer", "")).strip().lower()
            reasoning = str(parsed.get("reasoning", ""))

            if label in label_map:
                raw, normalized = label_map[label]
                return MetricScore(raw=raw, label=label, normalized=normalized, max_raw=max_raw), reasoning

            # Fuzzy match — accept if the label is a substring of a known key or vice versa
            for known_label, (raw, normalized) in label_map.items():
                if known_label in label or label in known_label:
                    return MetricScore(raw=raw, label=known_label, normalized=normalized, max_raw=max_raw), reasoning

            # Fallback to midpoint
            mid_label = list(label_map.keys())[len(label_map) // 2]
            mid_raw, mid_norm = label_map[mid_label]
            return (
                MetricScore(raw=mid_raw, label=mid_label, normalized=mid_norm, max_raw=max_raw),
                f"Could not parse label '{label}' — defaulted to '{mid_label}'",
            )

        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            mid_label = list(label_map.keys())[len(label_map) // 2]
            mid_raw, mid_norm = label_map[mid_label]
            return (
                MetricScore(raw=mid_raw, label=mid_label, normalized=mid_norm, max_raw=max_raw),
                f"Could not parse judge response: {exc}",
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def judge_answer(
        self,
        task: str,
        trajectory: Trajectory,
        expected_answer: str | None = None,
        rubric: str | None = None,
    ) -> AnswerJudgeResult:
        """
        Score the agent's final answer on two dimensions:
          - Correctness  (3-point: incorrect / partially correct / correct)
          - Completeness (5-point Likert)

        expected_answer is optional. When omitted, correctness is evaluated
        against the rubric alone (or as best-effort if neither is provided).
        """
        final_answer = trajectory.final_answer or "(no answer given)"
        expected_section = (
            f"\n## Expected answer\n{expected_answer}" if expected_answer else ""
        )
        rubric_section = f"\n## Rubric\n{rubric}" if rubric else ""

        correctness_prompt = _CORRECTNESS_PROMPT.format(
            task=task,
            final_answer=final_answer,
            expected_section=expected_section,
            rubric_section=rubric_section,
        )
        completeness_prompt = _COMPLETENESS_PROMPT.format(
            task=task,
            final_answer=final_answer,
            expected_section=expected_section,
        )
        helpfulness_prompt = _HELPFULNESS_PROMPT.format(
            task=task,
            final_answer=final_answer,
            expected_section=expected_section,
        )

        correctness, c_reasoning = self._call_metric(
            correctness_prompt, _CORRECTNESS_LABELS, max_raw=2
        )
        completeness, comp_reasoning = self._call_metric(
            completeness_prompt, _LIKERT5_LABELS, max_raw=4
        )
        helpfulness, h_reasoning = self._call_metric(
            helpfulness_prompt, _HELPFULNESS_LABELS, max_raw=4
        )

        score = (correctness.normalized + completeness.normalized + helpfulness.normalized) / 3
        reason = (
            f"Correctness: {c_reasoning} | "
            f"Completeness: {comp_reasoning} | "
            f"Helpfulness: {h_reasoning}"
        )

        return AnswerJudgeResult(
            correctness=correctness,
            completeness=completeness,
            helpfulness=helpfulness,
            score=round(score, 4),
            reason=reason,
        )

    def judge_reasoning(
        self,
        task: str,
        trajectory: Trajectory,
    ) -> ReasoningJudgeResult:
        """
        Score the agent's reasoning trajectory on two dimensions:
          - Faithfulness      (5-point: none / some / approximately half / most / all)
          - Logical Coherence (5-point Likert)
        """
        trajectory_text = _format_trajectory(trajectory)

        faithfulness_prompt = _FAITHFULNESS_PROMPT.format(
            task=task,
            trajectory_text=trajectory_text,
            final_answer=trajectory.final_answer or "(no answer given)",
        )
        coherence_prompt = _COHERENCE_PROMPT.format(
            task=task,
            trajectory_text=trajectory_text,
        )

        faithfulness, f_reasoning = self._call_metric(
            faithfulness_prompt, _FAITHFULNESS_LABELS, max_raw=4
        )
        coherence, coh_reasoning = self._call_metric(
            coherence_prompt, _LIKERT5_LABELS, max_raw=4
        )

        score = (faithfulness.normalized + coherence.normalized) / 2
        reason = f"Faithfulness: {f_reasoning} | Coherence: {coh_reasoning}"

        return ReasoningJudgeResult(
            faithfulness=faithfulness,
            logical_coherence=coherence,
            score=round(score, 4),
            reason=reason,
        )
