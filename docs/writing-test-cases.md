# Writing Good Test Cases

## How many cases do you need?

| Stage | Recommended count | Purpose |
|---|---|---|
| Initial baseline | 20–50 | Enough to detect obvious regressions; fast to run in CI |
| Production eval | 100–200 | Statistically meaningful pass rate deltas |
| Comprehensive coverage | 200+ | Coverage across all major topics/labels/task types |

Start small. A tight set of high-quality cases is more useful than a large set of vague ones. Add cases as you discover real failures in production.

## Difficulty distribution

Aim for a spread across difficulty levels. An all-easy dataset will pass on almost any model and won't catch real regressions.

| Difficulty | Target share | Purpose |
|---|---|---|
| `easy` | ~30% | Sanity check — basic facts, obvious classifications |
| `medium` | ~50% | Core coverage — typical queries, inference required |
| `hard` | ~20% | Stress test — edge cases, ambiguity, multi-hop reasoning |

If your dataset is all easy, you'll have a 100% baseline that never catches anything meaningful.

---

## `qa` datasets

### What makes a good case

**Be specific in `ground_truth`.** The scorer checks that the model's response contains your ground truth string. If your ground truth is vague, almost any response will pass.

```jsonl
// Bad — too vague, anything passes
{"input": {"query": "What is Lambda?"}, "expected": {"ground_truth": "a compute service"}}

// Good — specific and verifiable
{"input": {"query": "What is the maximum execution timeout for a Lambda function?"}, "expected": {"ground_truth": "15 minutes"}}
```

**One answer per case.** If the ground truth has multiple valid phrasings, pick the most canonical one. Avoid `OR` constructions in ground truth — they confuse both the contains check and the LLM judge.

**Avoid ambiguous questions.** If the question has different correct answers depending on context (region, account tier, current date), either add the context to the question or pick a different question.

### Case types to include

Generated cases use a specific mix, but when writing manually, aim to cover:

| Type | Example |
|---|---|
| Factual | "What is the max SQS message size?" → "256 KB" |
| Multi-hop | "What's the max Lambda timeout when triggered from SQS?" (requires knowing both services) |
| Negation | "Does S3 Standard support automatic replication across regions?" → "No, use S3 CRR for cross-region replication" |
| Out-of-scope | Questions the system shouldn't claim to answer — verify the model says it doesn't know |
| Ambiguous | Questions with plausible but wrong common answers |

Negation and out-of-scope cases are the most valuable stress tests — they catch confidently wrong answers.

---

## `classification` datasets

### Labels matter

Define your labels when creating the dataset:

```bash
supereval dataset create ticket-router \
  --type classification \
  --labels "billing,networking,storage,iam,compute"
```

The default prompt injects these labels and instructs the model to respond with exactly one of them. Without labels the prompt says "unknown" which produces unusable results.

### What makes a good case

**Cover all labels.** Don't have 40 billing cases and 2 security cases — you won't know if the security classifier is broken.

**Include ambiguous cases.** Real input is messy. Add cases that could plausibly belong to two categories and define which one is correct.

**Include edge cases.** Very short input, input in a different language, input that doesn't clearly belong to any label.

```jsonl
// Clear case
{"input": {"text": "I was charged twice for my EC2 instance this month"}, "expected": {"label": "billing"}, "difficulty": "easy"}

// Ambiguous case — EC2 networking issue, could be compute or networking
{"input": {"text": "My EC2 instance can't reach the internet"}, "expected": {"label": "networking"}, "difficulty": "medium", "tags": ["ec2", "ambiguous"]}

// Edge case — very short
{"input": {"text": "invoice"}, "expected": {"label": "billing"}, "difficulty": "hard"}
```

---

## `instruction` datasets

### Writing rubrics

The rubric is the most important field in an instruction test case. A vague rubric produces inconsistent scores.

**Good rubrics are objective and enumerable:**

```json
"rubric": "Must contain exactly 3 bullet points. Each bullet must cover one of: root cause, customer impact, resolution steps. Must not exceed 100 words total."
```

**Bad rubrics are subjective:**

```json
"rubric": "Should be a good summary"
```

**Checklist for a strong rubric:**
- Does it specify the required format? (bullet points, numbered list, paragraph)
- Does it specify required content? (which topics must be covered)
- Does it specify constraints? (word count, tone, what to exclude)
- Can a human verify pass/fail in 10 seconds?

### Using the `document` field

If the instruction operates on a specific piece of text, put it in `document`. Don't embed long documents in the `instruction` itself — it clutters the eval output and makes it harder to vary the document across cases.

```jsonl
{
  "input": {
    "instruction": "Summarize this incident report in 3 bullet points covering root cause, impact, and resolution.",
    "document": "On March 10th at 14:32 UTC, a misconfigured IAM policy denied S3 PutObject calls from the data pipeline role. 847 records failed to be written over a 23-minute window. The policy was reverted at 14:55 UTC. No data was lost — the pipeline replayed the failed writes successfully."
  },
  "expected": {
    "rubric": "Must contain exactly 3 bullet points. Must mention: (1) misconfigured IAM policy as root cause, (2) number of failed records (847) or duration (23 minutes) as impact, (3) policy revert and successful replay as resolution."
  }
}
```

---

## Agent eval test cases

### Mock environments

Each case's `tools` block defines the mock environment. Think of it as "what the world looks like for this specific task":

```json
"tools": {
  "search_kb": {
    "response": [{"bucket": "my-data-bucket", "region": "us-west-2"}],
    "latency_ms": 80
  },
  "delete_bucket": {
    "error": "AccessDenied"
  }
}
```

- Tools not listed in the `tools` block will raise a `ValueError` when called — this tests that the agent doesn't call unexpected tools.
- Set `"error"` on any tool the agent should not use successfully. This lets you verify `must_not_call` behaviour.
- Use `latency_ms` to simulate realistic tool response times if latency accuracy matters.

### Tool checks

Be explicit about what the agent should and shouldn't do:

```json
"expected": {
  "answer": "us-west-2",
  "answer_match": "contains",
  "must_call": ["search_kb"],
  "must_not_call": ["delete_bucket", "list_all_buckets"],
  "must_call_with": [{"tool": "search_kb", "args": {"query": "my-data-bucket"}}],
  "max_steps": 6
}
```

`must_call_with` does a subset check — the actual arguments must contain all the specified key-value pairs, but can have additional keys.

### Sizing `max_steps`

`max_steps` is informational by default (efficiency is scored but doesn't fail the case). To make it a hard limit, set `min_efficiency_score` in the dataset thresholds. A reasonable starting value is 2× the expected number of steps for an ideal solution.

---

## General guidance

**Test for real failures, not just happy paths.** Your model will encounter edge cases in production. Make sure your dataset includes questions where the correct answer is "I don't know", inputs that are ambiguous, and adversarial inputs that might trigger hallucination.

**Keep cases independent.** Each case should be self-contained. Don't write cases where knowing the answer to case 3 is required to evaluate case 4.

**Review staged cases before importing.** The `source_excerpt` field in staged output shows exactly which sentence in the document the question came from. If the excerpt doesn't support the answer, delete the case.

**Version your dataset with git.** `cases.jsonl` and `baseline.json` are committed to the repository. The git history is your audit trail for when cases were added or changed.
