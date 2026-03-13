from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .export import _build_tests
from .models import DatasetType
from .storage import load_dataset_meta

# Default prompt templates used when the customer provides no config or --prompt.
# Each template uses the same variable names as the test case vars.
DEFAULT_PROMPTS: dict[DatasetType, str] = {
    DatasetType.qa: (
        "Answer the following question accurately and concisely.\n\n"
        "Question: {{query}}"
    ),
    DatasetType.classification: (
        "Classify the following text into exactly one of these categories: {labels}.\n"
        "Respond with only the category name, nothing else.\n\n"
        "Text: {{{{text}}}}"   # outer braces escaped; {labels} filled at build time
    ),
    DatasetType.instruction: (
        "{{instruction}}"
        "{{% if document %}}\n\n---\n{{document}}{{% endif %}}"
    ),
}


def _vars_key(vars_dict: dict) -> str:
    """Stable short hash of a vars dict — used to match cases across runs."""
    canonical = json.dumps(vars_dict, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


class CaseResult:
    def __init__(
        self,
        vars: dict,
        passed: bool,
        score: float,
        latency_ms: int = 0,
        cost_usd: float = 0.0,
    ):
        self.vars = vars
        self.passed = passed
        self.score = score
        self.latency_ms = latency_ms
        self.cost_usd = cost_usd
        self.key = _vars_key(vars)

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "vars": self.vars,
            "passed": self.passed,
            "score": round(self.score, 4),
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
        }


class RunResult:
    def __init__(
        self,
        dataset: str,
        cases: list[CaseResult],
        providers: list[str],
        run_id: str | None = None,
        total_tokens: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        provider_results: dict[str, list[CaseResult]] | None = None,
        case_keys_ordered: list[str] | None = None,
    ):
        self.run_id = run_id or f"run_{uuid.uuid4().hex[:8]}"
        self.dataset = dataset
        self.cases = cases
        self.providers = providers
        self.ran_at = datetime.now(timezone.utc).isoformat()
        self.total_tokens = total_tokens
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        # Per-provider breakdown for comparison table (populated on multi-provider runs)
        self.provider_results: dict[str, list[CaseResult]] = provider_results or {}
        self.case_keys_ordered: list[str] = case_keys_ordered or []

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.cases if c.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    @property
    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.cases)

    @property
    def avg_latency_ms(self) -> float:
        if not self.cases:
            return 0.0
        return sum(c.latency_ms for c in self.cases) / len(self.cases)

    @property
    def p50_latency_ms(self) -> float:
        if not self.cases:
            return 0.0
        latencies = sorted(c.latency_ms for c in self.cases)
        return float(latencies[len(latencies) // 2])

    @property
    def p95_latency_ms(self) -> float:
        if not self.cases:
            return 0.0
        latencies = sorted(c.latency_ms for c in self.cases)
        idx = min(int(len(latencies) * 0.95), len(latencies) - 1)
        return float(latencies[idx])

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "dataset": self.dataset,
            "ran_at": self.ran_at,
            "providers": self.providers,
            "summary": {
                "total": self.total,
                "passed": self.passed,
                "failed": self.failed,
                "pass_rate": round(self.pass_rate, 4),
                "total_cost_usd": round(self.total_cost_usd, 6),
                "total_tokens": self.total_tokens,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "avg_latency_ms": round(self.avg_latency_ms, 1),
                "p50_latency_ms": round(self.p50_latency_ms, 1),
                "p95_latency_ms": round(self.p95_latency_ms, 1),
            },
            "cases": [c.to_dict() for c in self.cases],
        }


def _resolve_prompt(dataset_name: str, prompt: str | None) -> str:
    """Return the prompt string to use, filling in dataset-level values."""
    meta = load_dataset_meta(dataset_name)
    if prompt:
        return prompt
    template = DEFAULT_PROMPTS[meta.type]
    if meta.type == DatasetType.classification:
        labels = ", ".join(meta.labels) if meta.labels else "unknown"
        # Replace {labels} placeholder (not a Jinja var) with the actual label list
        template = template.format(labels=labels)
    return template


def _build_promptfoo_config(
    dataset_name: str,
    models: list[str],
    prompt: str | None,
    customer_config: dict | None,
) -> dict:
    """
    Build the full Promptfoo config dict.

    Tier 3 — customer_config provided:  inject tests, preserve everything else.
    Tier 2 — models + custom prompt:    auto-generate providers + prompts.
    Tier 1 — models only:               auto-generate providers + default prompt.
    """
    tests = _build_tests(dataset_name)

    if customer_config:
        config = dict(customer_config)
        config["tests"] = tests
        return config

    return {
        "prompts": [_resolve_prompt(dataset_name, prompt)],
        "providers": models,
        "tests": tests,
    }


def _promptfoo_cmd() -> list[str]:
    """Return the base promptfoo command, preferring global install over npx."""
    if shutil.which("promptfoo"):
        return ["promptfoo"]
    if shutil.which("npx"):
        return ["npx", "promptfoo@latest"]
    raise RuntimeError(
        "promptfoo not found. Install it with: npm install -g promptfoo"
    )


def _parse_promptfoo_output(raw: dict, providers: list[str], dataset_name: str) -> RunResult:
    """
    Parse Promptfoo's JSON output into a RunResult.

    Promptfoo nests results under raw["results"]["results"].
    For multi-provider runs, a case passes only if ALL providers pass it.
    Cases are matched by a hash of their vars.
    """
    results_block = raw.get("results", raw)
    raw_results: list[dict] = (
        results_block.get("results", [])
        if isinstance(results_block, dict)
        else results_block
    )

    # Extract token usage from stats block
    stats = results_block.get("stats", {}) if isinstance(results_block, dict) else {}
    token_usage = stats.get("tokenUsage", {})
    total_tokens = int(token_usage.get("total", 0))
    prompt_tokens = int(token_usage.get("prompt", 0))
    completion_tokens = int(token_usage.get("completion", 0))

    # Aggregate per vars-key across providers: strictest (AND) semantics
    # Also track per-provider results for comparison table
    cases_by_key: dict[str, dict] = {}
    per_provider: dict[str, dict[str, CaseResult]] = {}  # provider_id -> {vars_key -> CaseResult}
    case_keys_ordered: list[str] = []

    for item in raw_results:
        vars_ = {k: v for k, v in item.get("vars", {}).items()}
        key = _vars_key(vars_)
        passed = bool(item.get("success", False))
        score = float(item.get("score", 1.0 if passed else 0.0))
        latency = int(item.get("latencyMs", 0))
        cost = float(item.get("cost", 0.0))

        # Extract provider ID from Promptfoo item
        prov_field = item.get("provider", "")
        if isinstance(prov_field, dict):
            provider_id = prov_field.get("id", "unknown")
        else:
            provider_id = str(prov_field) if prov_field else "unknown"

        # Aggregated (AND) pass/fail across providers
        if key not in cases_by_key:
            cases_by_key[key] = {
                "vars": vars_,
                "passed": passed,
                "score": score,
                "latency_ms": latency,
                "cost_usd": cost,
                "count": 1,
            }
            case_keys_ordered.append(key)
        else:
            entry = cases_by_key[key]
            entry["passed"] = entry["passed"] and passed
            entry["score"] = min(entry["score"], score)
            entry["latency_ms"] += latency
            entry["cost_usd"] += cost
            entry["count"] += 1

        # Per-provider breakdown
        if provider_id not in per_provider:
            per_provider[provider_id] = {}
        per_provider[provider_id][key] = CaseResult(
            vars=vars_, passed=passed, score=score, latency_ms=latency, cost_usd=cost
        )

    cases = [
        CaseResult(
            vars=v["vars"],
            passed=v["passed"],
            score=v["score"],
            latency_ms=v["latency_ms"] // v["count"],
            cost_usd=v["cost_usd"],
        )
        for v in cases_by_key.values()
    ]

    # Convert per_provider to ordered lists matching case_keys_ordered
    provider_results: dict[str, list[CaseResult]] = {
        pid: [pcases[k] for k in case_keys_ordered if k in pcases]
        for pid, pcases in per_provider.items()
    }

    return RunResult(
        dataset=dataset_name,
        cases=cases,
        providers=providers,
        total_tokens=total_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        provider_results=provider_results,
        case_keys_ordered=case_keys_ordered,
    )


def run_eval(
    dataset_name: str,
    models: list[str] | None = None,
    prompt: str | None = None,
    config_path: Path | None = None,
) -> RunResult:
    """
    Run a Promptfoo eval against a dataset and return structured results.

    Tier 1: only --model(s) provided  → auto-generate full config with default prompt
    Tier 2: --model(s) + --prompt     → auto-generate config with custom prompt
    Tier 3: --config provided         → inject tests into customer's config
    """
    customer_config: dict | None = None
    if config_path:
        customer_config = yaml.safe_load(config_path.read_text())

    providers = list(models or [])
    if customer_config and not providers:
        providers = [
            p if isinstance(p, str) else p.get("id", str(p))
            for p in customer_config.get("providers", [])
        ]

    if not providers:
        raise ValueError(
            "No models specified. Use --model or provide a --config with providers."
        )

    config = _build_promptfoo_config(dataset_name, providers, prompt, customer_config)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = Path(tmpdir) / "promptfoo.yaml"
        results_file = Path(tmpdir) / "results.json"

        config_file.write_text(
            yaml.dump(config, default_flow_style=False, allow_unicode=True, sort_keys=False)
        )

        cmd = _promptfoo_cmd() + [
            "eval",
            "--config", str(config_file),
            "--output", str(results_file),
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)

        if not results_file.exists():
            raise RuntimeError(
                f"promptfoo eval failed (exit {proc.returncode}).\n"
                f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
            )

        raw = json.loads(results_file.read_text())

    return _parse_promptfoo_output(raw, providers, dataset_name)
