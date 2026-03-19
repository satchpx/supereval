"""
LLM-as-judge scoring for RAG evaluation via Amazon Bedrock.

Implements two focused metrics:

  Faithfulness      — does the answer stay grounded in the retrieved contexts?
                      (5-point scale; detects hallucination beyond the provided docs)
  Answer Correctness — is the final answer correct vs the ground truth?
                      (3-point scale)

Uses the same Bedrock invoke pattern as generator.py and agent/judge.py.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

_DEFAULT_MODEL = "anthropic.claude-3-5-sonnet-20241022-v2:0"
_DEFAULT_REGION = "us-east-1"

# ---------------------------------------------------------------------------
# Ordinal scale definitions (mirrors agent/judge.py for consistency)
# ---------------------------------------------------------------------------

_FAITHFULNESS_LABELS: dict[str, tuple[int, float]] = {
    "none":               (0, 0.00),
    "some":               (1, 0.25),
    "approximately half": (2, 0.50),
    "most":               (3, 0.75),
    "all":                (4, 1.00),
}

_CORRECTNESS_LABELS: dict[str, tuple[int, float]] = {
    "incorrect":         (0, 0.0),
    "partially correct": (1, 0.5),
    "correct":           (2, 1.0),
}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_FAITHFULNESS_PROMPT = """\
You are an impartial evaluator assessing whether an AI assistant's answer is \
faithful to the retrieved context documents — i.e., whether it makes claims \
not grounded in those documents.

## Question
{query}

## Retrieved context documents
{contexts}

## Assistant's answer
{answer}

How much of the assistant's answer is grounded in (faithful to) the retrieved \
context documents above, rather than fabricated or hallucinated?

Respond with a JSON object only:
{{"reasoning": "<one or two sentences>", "answer": "<none|some|approximately half|most|all>"}}"""


_CORRECTNESS_PROMPT = """\
You are an impartial evaluator assessing whether an AI assistant's answer is correct.

## Question
{query}

## Expected answer (ground truth)
{ground_truth}

## Assistant's answer
{answer}

Is the assistant's answer correct compared to the expected answer?

Respond with a JSON object only:
{{"reasoning": "<one or two sentences>", "answer": "<incorrect|partially correct|correct>"}}"""

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class MetricScore:
    raw: int
    label: str
    normalized: float
    max_raw: int


@dataclass
class RagJudgeResult:
    faithfulness: MetricScore
    answer_correctness: MetricScore
    score: float        # avg of both normalized scores — for scorer.py compat
    reason: str         # combined reasoning summary


# ---------------------------------------------------------------------------
# RagJudge
# ---------------------------------------------------------------------------

class RagJudge:
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
        try:
            text = self._invoke(prompt)
            parsed = json.loads(text)
            label = str(parsed.get("answer", "")).strip().lower()
            reasoning = str(parsed.get("reasoning", ""))

            if label in label_map:
                raw, normalized = label_map[label]
                return MetricScore(raw=raw, label=label, normalized=normalized, max_raw=max_raw), reasoning

            # Fuzzy match
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

    def judge(
        self,
        query: str,
        answer: str,
        contexts: list[str],
        ground_truth: str,
    ) -> RagJudgeResult:
        """
        Score a RAG answer on two dimensions:
          - Faithfulness     (5-point: none / some / approximately half / most / all)
          - Answer Correctness (3-point: incorrect / partially correct / correct)

        Makes two sequential Bedrock calls.
        """
        contexts_text = "\n\n".join(
            f"[{i + 1}] {ctx}" for i, ctx in enumerate(contexts)
        ) if contexts else "(no contexts provided)"

        faithfulness_prompt = _FAITHFULNESS_PROMPT.format(
            query=query,
            contexts=contexts_text,
            answer=answer,
        )
        correctness_prompt = _CORRECTNESS_PROMPT.format(
            query=query,
            ground_truth=ground_truth,
            answer=answer,
        )

        faithfulness, f_reason = self._call_metric(
            faithfulness_prompt, _FAITHFULNESS_LABELS, max_raw=4
        )
        correctness, c_reason = self._call_metric(
            correctness_prompt, _CORRECTNESS_LABELS, max_raw=2
        )

        score = (faithfulness.normalized + correctness.normalized) / 2
        reason = f"Faithfulness: {f_reason} | Answer Correctness: {c_reason}"

        return RagJudgeResult(
            faithfulness=faithfulness,
            answer_correctness=correctness,
            score=round(score, 4),
            reason=reason,
        )
