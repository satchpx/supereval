"""
RAG case scoring.

Dimensions
----------
contains_score           — icontains of ground_truth in answer (0 or 1); cheap fast signal
answer_correctness_score — judge (3-point) or falls back to contains when no judge
faithfulness_score       — judge (5-point); defaults to 1.0 when no judge configured
composite_score          — weighted average of answer_correctness + faithfulness

Failure conditions
------------------
- require_contains=True and contains_score == 0.0
- answer_correctness_score < thresholds.min_answer_correctness_score
- faithfulness_score < thresholds.min_faithfulness_score (only when judge is provided)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .models import RagExpected, RagScore, RagThresholds

if TYPE_CHECKING:
    from .judge import RagJudge


def score_rag_case(
    query: str,
    answer: str,
    expected: RagExpected,
    contexts: list[str],
    thresholds: RagThresholds,
    judge: "RagJudge | None" = None,
) -> RagScore:
    score = RagScore()

    # -----------------------------------------------------------------------
    # Contains check (always run — cheap signal, no judge needed)
    # -----------------------------------------------------------------------
    score.contains_score = (
        1.0 if expected.ground_truth.lower() in answer.lower() else 0.0
    )
    if thresholds.require_contains and score.contains_score == 0.0:
        score.failure_reasons.append(
            f"Answer does not contain expected ground truth: {expected.ground_truth!r}"
        )

    # -----------------------------------------------------------------------
    # Judge-backed scoring
    # -----------------------------------------------------------------------
    if judge is not None:
        result = judge.judge(
            query=query,
            answer=answer,
            contexts=contexts,
            ground_truth=expected.ground_truth,
        )
        score.faithfulness_score = result.faithfulness.normalized
        score.faithfulness_raw = result.faithfulness.raw
        score.faithfulness_label = result.faithfulness.label
        score.faithfulness_reason = result.reason.split(" | ")[0].replace("Faithfulness: ", "")

        score.answer_correctness_score = result.answer_correctness.normalized
        score.answer_correctness_raw = result.answer_correctness.raw
        score.answer_correctness_label = result.answer_correctness.label
        score.answer_correctness_reason = result.reason.split(" | Answer Correctness: ")[-1]

        # Enforce faithfulness threshold when judge is configured
        if (
            thresholds.min_faithfulness_score is not None
            and score.faithfulness_score < thresholds.min_faithfulness_score
        ):
            score.failure_reasons.append(
                f"Faithfulness score {score.faithfulness_score:.2f} below threshold "
                f"{thresholds.min_faithfulness_score:.2f}: {score.faithfulness_reason}"
            )

        # Enforce answer correctness threshold
        if score.answer_correctness_score < thresholds.min_answer_correctness_score:
            score.failure_reasons.append(
                f"Answer correctness score {score.answer_correctness_score:.2f} below threshold "
                f"{thresholds.min_answer_correctness_score:.2f}: {score.answer_correctness_reason}"
            )

        # Composite: weighted average of both judge dimensions
        total_w = thresholds.faithfulness_weight + thresholds.answer_correctness_weight
        score.composite_score = (
            score.faithfulness_score * thresholds.faithfulness_weight
            + score.answer_correctness_score * thresholds.answer_correctness_weight
        ) / total_w if total_w > 0 else 0.0

    else:
        # No judge: answer_correctness falls back to contains; faithfulness unverified (1.0)
        score.answer_correctness_score = score.contains_score

        if score.answer_correctness_score < thresholds.min_answer_correctness_score:
            score.failure_reasons.append(
                f"Answer correctness score {score.answer_correctness_score:.2f} below threshold "
                f"{thresholds.min_answer_correctness_score:.2f} "
                f"(no judge configured; using contains match)"
            )

        # Composite equals answer_correctness when faithfulness is unverified
        score.composite_score = score.answer_correctness_score

    score.passed = len(score.failure_reasons) == 0
    return score
