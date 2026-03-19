"""
RAG eval runner.

Calls the configured LLM with each test case's query + retrieved_contexts,
then scores the response using RagJudge (if configured) and the contains check.

Supported model backends
------------------------
  Bedrock (default)  — any Bedrock model ID, e.g. us.anthropic.claude-3-5-sonnet-20241022-v2:0
  Anthropic API      — prefix with "anthropic:", e.g. anthropic:claude-sonnet-4-6
  OpenAI API         — prefix with "openai:", e.g. openai:gpt-4o

Default prompt template
-----------------------
Answer the question using ONLY the provided context documents. If the answer
is not in the context, say "I don't know."
"""
from __future__ import annotations

import json
import time
import textwrap

from .judge import RagJudge
from .models import RagCaseResult, RagRunResult
from .scorer import score_rag_case
from .storage import load_rag_cases, load_rag_dataset_meta

DEFAULT_PROMPT_TEMPLATE = textwrap.dedent("""
    Answer the question using ONLY the provided context documents.
    If the answer cannot be found in the context, say "I don't know."

    Context documents:
    {contexts}

    Question: {query}
""").strip()


def _format_contexts(contexts: list[str]) -> str:
    if not contexts:
        return "(no context documents provided)"
    return "\n\n".join(f"[{i + 1}] {ctx}" for i, ctx in enumerate(contexts))


def _call_bedrock(model_id: str, region: str, prompt: str) -> tuple[str, int]:
    """Call a Bedrock model. Returns (answer_text, latency_ms)."""
    import boto3
    client = boto3.client("bedrock-runtime", region_name=region)
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2048,
        "messages": [{"role": "user", "content": prompt}],
    })
    t0 = time.monotonic()
    response = client.invoke_model(
        modelId=model_id,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    latency_ms = int((time.monotonic() - t0) * 1000)
    raw = json.loads(response["body"].read())
    text = raw["content"][0]["text"].strip()
    return text, latency_ms


def _call_anthropic(model_id: str, prompt: str) -> tuple[str, int]:
    """Call the Anthropic API. Returns (answer_text, latency_ms)."""
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "anthropic package is required for the 'anthropic:' backend. "
            "Install with: pip install 'supereval[anthropic]'"
        )
    client = anthropic.Anthropic()
    t0 = time.monotonic()
    message = client.messages.create(
        model=model_id,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    latency_ms = int((time.monotonic() - t0) * 1000)
    return message.content[0].text.strip(), latency_ms


def _call_openai(model_id: str, prompt: str) -> tuple[str, int]:
    """Call the OpenAI API. Returns (answer_text, latency_ms)."""
    try:
        import openai
    except ImportError:
        raise ImportError(
            "openai package is required for the 'openai:' backend. "
            "Install with: pip install 'supereval[openai]'"
        )
    client = openai.OpenAI()
    t0 = time.monotonic()
    response = client.chat.completions.create(
        model=model_id,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    latency_ms = int((time.monotonic() - t0) * 1000)
    return response.choices[0].message.content.strip(), latency_ms


def _call_model(
    model_id: str,
    region: str,
    prompt: str,
) -> tuple[str, int]:
    """
    Dispatch to the right backend based on the model_id prefix.

    anthropic:<model>  → Anthropic API
    openai:<model>     → OpenAI API
    <anything else>    → Amazon Bedrock
    """
    if model_id.startswith("anthropic:"):
        return _call_anthropic(model_id[len("anthropic:"):], prompt)
    if model_id.startswith("openai:"):
        return _call_openai(model_id[len("openai:"):], prompt)
    return _call_bedrock(model_id, region, prompt)


def run_rag_eval(
    dataset_name: str,
    model_id: str,
    region: str = "us-east-1",
    prompt_template: str | None = None,
    judge_model: str | None = None,
    judge_region: str = "us-east-1",
) -> RagRunResult:
    """
    Run a RAG eval against every case in a dataset.

    For each case, calls the LLM with query + static retrieved_contexts,
    then scores the response for faithfulness and answer correctness.
    """
    meta = load_rag_dataset_meta(dataset_name)
    cases = load_rag_cases(dataset_name)

    if not cases:
        raise ValueError(
            f"RAG dataset '{dataset_name}' has no cases. "
            f"Add cases with: supereval rag dataset add-cases {dataset_name} --from cases.jsonl"
        )

    template = prompt_template or DEFAULT_PROMPT_TEMPLATE
    judge = RagJudge(model_id=judge_model, region=judge_region) if judge_model else None

    case_results: list[RagCaseResult] = []

    for case in cases:
        contexts_text = _format_contexts(case.input.retrieved_contexts)
        prompt = template.format(
            query=case.input.query,
            contexts=contexts_text,
        )

        try:
            answer, latency_ms = _call_model(model_id, region, prompt)
        except Exception as exc:
            # Model call failed — record as a failed case with empty answer
            answer = ""
            latency_ms = 0
            rag_score = score_rag_case(
                query=case.input.query,
                answer=answer,
                expected=case.expected,
                contexts=case.input.retrieved_contexts,
                thresholds=meta.thresholds,
                judge=None,
            )
            rag_score.failure_reasons.insert(0, f"Model call failed: {exc}")
            rag_score.passed = False
            case_results.append(RagCaseResult(
                case_id=case.id,
                description=case.description,
                vars={"query": case.input.query},
                answer=answer,
                score=rag_score,
                latency_ms=latency_ms,
            ))
            continue

        rag_score = score_rag_case(
            query=case.input.query,
            answer=answer,
            expected=case.expected,
            contexts=case.input.retrieved_contexts,
            thresholds=meta.thresholds,
            judge=judge,
        )

        case_results.append(RagCaseResult(
            case_id=case.id,
            description=case.description,
            vars={"query": case.input.query},
            answer=answer,
            score=rag_score,
            latency_ms=latency_ms,
        ))

    return RagRunResult(
        dataset=dataset_name,
        cases=case_results,
        model_id=model_id,
    )
