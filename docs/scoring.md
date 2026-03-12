# How Scoring Works

## LLM eval scoring

supereval builds a Promptfoo config from your dataset and runs `promptfoo eval`. The asserts on each test case are determined by the dataset type.

### `qa` datasets

Each case gets two asserts:

| Assert | Type | What it checks |
|---|---|---|
| Contains | `icontains` | Response contains the `ground_truth` text (case-insensitive) |
| LLM judge | `llm-rubric` | An LLM evaluator confirms the response correctly addresses the ground truth |

**Both must pass for the case to pass.** This catches responses that happen to mention the right term but get the meaning wrong (fails the judge), and responses where the LLM judge is lenient but the exact answer is missing (fails the contains check).

Example: `ground_truth` = `"256 KB"`. A response of `"SQS messages are limited to 256 kilobytes"` passes the judge but fails the contains check. A response of `"The limit is 256 KB, but this was recently increased"` passes the contains check but may fail the judge.

The LLM judge used by Promptfoo is configured separately (see [Custom Promptfoo config](./custom-config.md)). By default Promptfoo uses its own hosted grading model.

### `classification` datasets

Each case gets one assert:

| Assert | Type | What it checks |
|---|---|---|
| Contains | `icontains` | Response contains the expected `label` (case-insensitive) |

The default prompt instructs the model to respond with only the category name, so a contains check is the right signal. If your prompt produces verbose output, add a custom assert — see [Custom Promptfoo config](./custom-config.md).

### `instruction` datasets

Each case gets one assert:

| Assert | Type | What it checks |
|---|---|---|
| LLM judge | `llm-rubric` | An LLM evaluator scores the response against the `rubric` string |

The rubric is passed verbatim to the judge. Write rubrics as clear pass/fail criteria:

```
"Must contain exactly 3 bullet points. Each bullet must address one of: root cause, impact, resolution."
```

Vague rubrics produce unreliable scores. See [Writing good test cases](./writing-test-cases.md) for rubric guidance.

---

## Pass/fail semantics

A case **passes** if all asserts pass. Promptfoo grades each assert as pass/fail; a case's **score** is the fraction of asserts that pass (0.0–1.0).

The case-level `passed` field in results is `true` only when score = 1.0.

### Multiple models

When you run with `--model A --model B`, each case is run against both models. A case passes only if **all** models pass it. The score for a case is the minimum score across providers. This ensures a regression on any model is surfaced.

---

## Thresholds

Thresholds are set per-dataset in `dataset.json` and evaluated by `--compare-baseline`. They do not affect the per-case pass/fail — they affect whether the overall run fails CI.

```json
{
  "thresholds": {
    "pass_rate": 0.90,
    "fail_on_regression": true,
    "max_score_drop": 0.05,
    "max_cost_usd": 1.00,
    "max_p95_latency_ms": 3000
  }
}
```

| Field | Default | Behaviour |
|---|---|---|
| `pass_rate` | `1.0` | Run fails if fewer than this fraction of cases pass |
| `fail_on_regression` | `true` | Run fails if any previously-passing case now fails |
| `max_score_drop` | `null` | Run fails if pass rate dropped more than this vs baseline |
| `max_cost_usd` | `null` | Run fails if total cost exceeds this amount |
| `max_p95_latency_ms` | `null` | Run fails if p95 latency exceeds this threshold |

`fail_on_regression: true` is the most important setting. With `pass_rate: 0.9`, you might not notice a single case regressing if everything else stays green. With `fail_on_regression: true`, any individual regression breaks the build regardless of the aggregate pass rate.

---

## Agent eval scoring

Agent evals are scored independently of Promptfoo — supereval's own scorer runs after each trajectory completes.

### Four dimensions

| Dimension | Score range | How it's computed |
|---|---|---|
| `answer_score` | 0.0–1.0 | Compares `trajectory.final_answer` against `expected.answer` |
| `tool_score` | 0.0–1.0 | Fraction of tool checks satisfied |
| `efficiency_score` | 0.0–1.0 | Normalized step count vs `max_steps` |
| `reasoning_score` | 0.0–1.0 | LLM judge on the thought chain (only when `--judge-model` is set) |

### Answer scoring methods

Set `answer_match` on the test case `expected` block:

| Method | Behaviour |
|---|---|
| `contains` (default) | `final_answer` contains the expected string (case-insensitive) |
| `exact` | `final_answer` equals the expected string exactly (case-insensitive, stripped) |
| `regex` | `final_answer` matches the expected regex pattern |
| `llm_judge` | Bedrock judge evaluates correctness; requires `--judge-model` |

When `answer_match: llm_judge` is set but no `--judge-model` is provided, scoring falls back to `contains` and the result notes the fallback.

### Tool scoring

Tool score is `checks_passed / total_checks`. Each test case can define:

- `must_call: ["tool_a", "tool_b"]` — penalises 1/total for each missing call
- `must_not_call: ["dangerous_tool"]` — penalises 1/total for each forbidden call
- `must_call_with: [{"tool": "search", "args": {"query": "s3"}}]` — penalises 1/total for each call that doesn't match the expected arguments

If no tool checks are defined, `tool_score` defaults to 1.0.

### Efficiency scoring

```
efficiency = max(0, 1 - (step_count - 1) / (max_steps - 1))
```

A single-step trajectory scores 1.0. A trajectory that uses all `max_steps` scores 0.0. If `max_steps` is not set, efficiency defaults to 1.0 (informational only).

### Composite score

```
composite = weighted_average(answer, tool, efficiency, reasoning)
```

Default weights: answer 40%, tool 40%, efficiency 5%, reasoning 15%. When no judge is configured, the reasoning weight is redistributed to the other dimensions so a perfect run still reaches 1.0.

### Agent thresholds

```json
{
  "thresholds": {
    "min_answer_score": 0.8,
    "min_tool_score": 1.0,
    "min_reasoning_score": null,
    "pass_rate": 1.0,
    "fail_on_regression": true
  }
}
```

A case fails if any dimension score falls below its threshold. `min_reasoning_score: null` means reasoning is scored but never causes a failure — useful while you're calibrating the judge.
