# Getting Started Tutorial

This tutorial walks through a complete supereval workflow from scratch using an AWS support chatbot as the example scenario.

## What you'll build

An eval dataset for a chatbot that answers questions about AWS services. By the end you'll have:
- A dataset with test cases
- A passing baseline captured
- A regression caught automatically

## Prerequisites

```bash
pip install -e .
npm install -g promptfoo
# AWS credentials configured (env vars, ~/.aws/credentials, or IAM role)
```

---

## Step 1 — Create a dataset

```bash
supereval dataset create aws-support-qa \
  --type qa \
  --description "Factual Q&A for AWS support chatbot" \
  --tags "aws,support"
```

This creates `datasets/aws-support-qa/dataset.json`. Nothing else yet.

---

## Step 2 — Write your first test cases

Create a file called `cases.jsonl`. Each line is one test case:

```jsonl
{"input": {"query": "What is the maximum size of an SQS message?"}, "expected": {"ground_truth": "256 KB"}, "description": "SQS message size limit", "difficulty": "easy", "tags": ["sqs", "limits"]}
{"input": {"query": "How long does S3 strong consistency apply?"}, "expected": {"ground_truth": "S3 provides strong consistency for all operations"}, "description": "S3 consistency model", "difficulty": "medium", "tags": ["s3"]}
{"input": {"query": "What is the maximum execution timeout for a Lambda function?"}, "expected": {"ground_truth": "15 minutes"}, "description": "Lambda max timeout", "difficulty": "easy", "tags": ["lambda", "limits"]}
{"input": {"query": "Which AWS service provides managed Kafka?"}, "expected": {"ground_truth": "Amazon MSK"}, "description": "Managed Kafka service", "difficulty": "easy", "tags": ["msk", "kafka"]}
{"input": {"query": "What does the AWS Shared Responsibility Model say about OS patching for EC2?"}, "expected": {"ground_truth": "The customer is responsible for patching the OS on EC2 instances"}, "description": "Shared responsibility - EC2 OS", "difficulty": "medium", "tags": ["ec2", "security"]}
```

Add them to the dataset:

```bash
supereval dataset add-cases aws-support-qa --from cases.jsonl
```

```
Added 5 case(s) to 'aws-support-qa'
```

Verify:

```bash
supereval dataset show aws-support-qa
```

```
aws-support-qa  ds_3f8a1b2c
  Type:        qa
  Description: Factual Q&A for AWS support chatbot
  Tags:        aws, support
  Cases:       5
  Created:     2025-11-01

Preview (5 of 5 cases):

  tc_4a2b3c  SQS message size limit
  input:    {'query': 'What is the maximum size of an SQS message?'}
  expected: {'ground_truth': '256 KB'}
  ...
```

---

## Step 3 — Run your first eval

```bash
supereval run aws-support-qa --model anthropic:claude-opus-4-6
```

supereval builds a Promptfoo config, runs the model against each case, and prints results:

```
Running eval: aws-support-qa
  Models:  anthropic:claude-opus-4-6

Results  (run run_7d4e9f2a)
  Passed:    5/5
  Pass rate: 100.0%
  Cost:      $0.0031
  Tokens:    1,842  (prompt: 1,204  completion: 638)
  Latency:   avg 823ms  p50 791ms  p95 1241ms
```

If a case fails, you'll see:

```
Results  (run run_2b5a8c1d)
  Passed:    4/5
  Pass rate: 80.0%

  Failed cases:
    ✗  {'query': 'How long does S3 strong consistency apply?'}
```

---

## Step 4 — Understand what "pass" means

For a `qa` case to pass, the model's response must satisfy **both**:

1. **Contains check** — the response contains the `ground_truth` text (case-insensitive)
2. **LLM judge** — an LLM evaluator confirms the response correctly addresses the ground truth

Both must pass. This catches responses that mention the right term but get the meaning wrong.

See [scoring.md](./scoring.md) for the full breakdown.

---

## Step 5 — Capture a baseline

Once you're happy with your initial pass rate, save it as the baseline:

```bash
supereval run aws-support-qa --model anthropic:claude-opus-4-6 --update-baseline
```

This writes `datasets/aws-support-qa/baseline.json`. Commit it to git:

```bash
git add datasets/aws-support-qa/baseline.json
git commit -m "add eval baseline for aws-support-qa"
```

---

## Step 6 — Detect a regression

Now imagine you change a prompt or switch models. Run with `--compare-baseline`:

```bash
supereval run aws-support-qa \
  --model anthropic:claude-opus-4-6 \
  --prompt "You are a concise AWS assistant. {{query}}" \
  --compare-baseline
```

If a previously-passing case now fails:

```
Baseline comparison  (baseline: 100.0%  current: 80.0%  delta: -20.0%)
  FAILED — regressions detected:
    • Regression: 1 case(s) that previously passed now fail
    • Pass rate dropped below baseline (80.0% < 100.0%)

  Regressed cases:
    ✗  {'query': 'What is the maximum execution timeout...'}
       (score: 1.00 → 0.00)
```

The exit code is non-zero, which fails the CI build.

---

## Step 7 — Generate more cases from your documents

Once you have real documentation, generate cases from it:

```bash
supereval generate aws-support-qa --from docs/aws-kb/ --count 30
```

This writes `aws-support-qa_staged.jsonl`. Open it and review each case — the `source_excerpt` field shows which part of the document each question came from, so you can verify the answer is grounded.

```json
{
  "description": "DynamoDB on-demand capacity pricing",
  "input": {"query": "How does DynamoDB on-demand pricing work?"},
  "expected": {"ground_truth": "You pay per read/write request unit with no minimum capacity"},
  "case_type": "factual",
  "source_excerpt": "On-demand capacity mode charges you for the data reads and writes...",
  "difficulty": "medium",
  "tags": ["dynamodb", "pricing"]
}
```

Or review each case interactively before importing — keep, edit, skip, or quit at any point:

```bash
supereval generate aws-support-qa --from docs/aws-kb/ --count 30 --interactive
```

Remove any cases that look wrong, then import:

```bash
supereval dataset add-cases aws-support-qa --from aws-support-qa_staged.jsonl
```

---

## Step 8 — Set up CI

Push baseline.json to git and add a CI workflow. With GitHub Actions, copy `.github/workflows/eval.yml` to your repo. With CodeBuild, copy `codebuild/buildspec.yml`.

On every PR the workflow runs `--compare-baseline` and fails if there's a regression. On merge to main it runs `--update-baseline` and commits the new baseline back.

---

## Next steps

- [Scoring explained](./scoring.md) — understand exactly what "pass" means for each dataset type, including RAG eval metrics
- [Writing good test cases](./writing-test-cases.md) — guidance on case quality and coverage for all dataset types
- [Custom Promptfoo config](./custom-config.md) — add custom scoring logic or model parameters
- [MCP server setup](./mcp.md) — connect supereval to Claude Code, Kiro, or any AI coding assistant so non-technical teammates can run evals via natural language
- **Dataset versioning** — snapshot your dataset at any point with `supereval dataset version tag v1.0.0 aws-support-qa --description "initial baseline"`. Restore a previous version with `supereval dataset version restore v1.0.0 aws-support-qa`.

### Evaluating a RAG pipeline?

If you have a retrieval-augmented generation system, use the `rag` dataset type instead. The workflow is similar:

```bash
# 1. Create a RAG dataset
supereval rag dataset create aws-rag-eval \
  --description "RAG eval for AWS support chatbot"

# 2. Add cases (each case includes the retrieved contexts)
supereval rag dataset add-cases aws-rag-eval --from rag-cases.jsonl

# 3. Run eval — model is called directly (no Promptfoo required)
supereval rag run aws-rag-eval --model us.anthropic.claude-3-5-sonnet-20241022-v2:0

# 4. Add a Bedrock judge for faithfulness + answer correctness scoring
supereval rag run aws-rag-eval \
  --model us.anthropic.claude-3-5-sonnet-20241022-v2:0 \
  --judge-model us.anthropic.claude-3-5-sonnet-20241022-v2:0

# 5. Baseline and regression detection work the same way
supereval rag run aws-rag-eval --model <model> --update-baseline
supereval rag run aws-rag-eval --model <model> --compare-baseline
```

See [Writing good test cases](./writing-test-cases.md#rag-datasets) for the `rag` case format and guidance on including distractor contexts.
