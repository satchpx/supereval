# supereval

**supereval** is a dataset registry and eval runner for LLM applications and AI agents, built on top of [Promptfoo](https://www.promptfoo.dev).

## How It Works

supereval covers three evaluation modes:

**LLM eval** — test prompt quality and model behaviour against typed datasets (Q&A, classification, instruction-following). Promptfoo handles model calls and scoring; you never write Promptfoo config by hand.

**RAG eval** — test retrieval-augmented generation pipelines against datasets with committed context documents. Each test case includes the query, the retrieved context chunks, and the expected ground truth. supereval calls the model directly and scores each case on contains accuracy, faithfulness to context (no hallucination), and answer correctness — all via a Bedrock judge.

**Agent eval** — test tool-use agents end-to-end against datasets with mock tool environments. Each test case defines the task, canned tool responses, and expected outcome. The agent runs deterministically against mock tools; supereval scores the trajectory across four dimensions: answer correctness, tool usage, efficiency, and reasoning quality.

## Install

```bash
pip install -e .                       # core install
pip install -e '.[pdf]'               # add PDF support
pip install -e '.[anthropic]'         # add Anthropic API generator backend
pip install -e '.[openai]'            # add OpenAI / Azure OpenAI generator backend
pip install -e '.[mcp]'               # add MCP server (Claude Code, Kiro integration)
npm install -g promptfoo              # required for supereval run
```

AWS credentials must be configured for Bedrock features (`supereval generate`, Bedrock models in `supereval run`). Standard boto3 credential resolution applies: environment variables, `~/.aws/credentials`, or an IAM role.

## Quickstart

```bash
# 1. Create a dataset
supereval dataset create aws-support-qa \
  --type qa \
  --description "Factual Q&A for AWS support chatbot" \
  --tags "aws,support"

# 2a. Generate test cases from your documents (recommended)
supereval generate aws-support-qa --from docs/kb/ --count 30
#     → writes aws-support-qa_staged.jsonl for your review

# 2b. Or write cases manually and add them
supereval dataset add-cases aws-support-qa --from cases.jsonl

# 3. Run an eval
supereval run aws-support-qa --model bedrock:us.anthropic.claude-sonnet-4-6

# 4. Capture a baseline
supereval run aws-support-qa --model bedrock:us.anthropic.claude-sonnet-4-6 --update-baseline

# 5. On your next change, check for regressions
supereval run aws-support-qa --model bedrock:us.anthropic.claude-sonnet-4-6 --compare-baseline
```

## Documentation

| Guide | Contents |
|---|---|
| [Getting started tutorial](./tutorial.md) | End-to-end walkthrough: create a dataset, run an eval, capture a baseline, catch a regression |
| [How scoring works](./scoring.md) | What "pass" means for each dataset type; RAG scoring metrics; agent scoring dimensions; threshold semantics |
| [Writing good test cases](./writing-test-cases.md) | Case quality, difficulty distribution, rubric writing, RAG context design, agent mock environments |
| [MCP server setup](./mcp.md) | Connect supereval to Claude Code, Kiro, and other AI tools; environment variables; troubleshooting |
| [Custom Promptfoo config](./custom-config.md) | Tier 3 config, custom asserts, model parameters, custom grading providers |
| [Agent runner examples](./agent-examples.md) | ReAct/Bedrock, LangChain, and Bedrock Agents runners explained with tradeoffs |
| [Troubleshooting](./troubleshooting.md) | Common errors and fixes: credentials, dataset validation, all-cases-failing, DuckDB, agent crashes |
| [Upgrading](./upgrading.md) | Schema migrations, dataset format stability, baseline.json format, environment separation |

---

## Dataset Types

Each dataset has a `type` that defines the shape of its test cases.

### `qa` — question with a known correct answer
```json
{
  "description": "SQS message size limit",
  "input": { "query": "What is the maximum size of an SQS message?" },
  "expected": { "ground_truth": "256 KB" },
  "tags": ["sqs", "limits"],
  "difficulty": "easy"
}
```

### `classification` — input maps to a label
```json
{
  "description": "Route a networking incident",
  "input": { "text": "My EC2 instance lost internet access after modifying the route table" },
  "expected": { "label": "networking" },
  "tags": ["ec2", "routing"]
}
```

### `instruction` — open-ended task scored by rubric
```json
{
  "description": "Summarize an incident report",
  "input": {
    "instruction": "Summarize this incident report in 3 bullet points",
    "document": "On March 10th at 14:32 UTC, a misconfigured IAM policy..."
  },
  "expected": {
    "rubric": "Must contain exactly 3 bullet points. Must include root cause, impact, and resolution."
  }
}
```

### `rag` — question answered from retrieved context documents
```json
{
  "description": "S3 bucket region lookup",
  "input": {
    "query": "What region is my-data-bucket in?",
    "retrieved_contexts": [
      "The bucket my-data-bucket was created in us-west-2 in January 2024.",
      "S3 bucket names are globally unique but buckets exist in a specific region."
    ]
  },
  "expected": { "ground_truth": "us-west-2" },
  "tags": ["s3", "region"],
  "difficulty": "easy"
}
```

## Running Evals

### Tier 1 — no Promptfoo knowledge required
```bash
supereval run my-dataset --model bedrock:us.anthropic.claude-sonnet-4-6
```
supereval auto-generates the prompt template based on the dataset type.

### Tier 2 — custom prompt
```bash
supereval run my-dataset \
  --model bedrock:us.anthropic.claude-sonnet-4-6 \
  --prompt "You are an AWS expert. Answer concisely: {{query}}"
```

### Tier 3 — bring your own Promptfoo config
Provide a `promptfoo.yaml` with `prompts` and `providers`. supereval injects the test cases automatically — you never write the `tests` section by hand.
```bash
supereval run my-dataset --config promptfoo.yaml
```

### Multiple models (model comparison)
```bash
supereval run my-dataset \
  --model bedrock:us.anthropic.claude-sonnet-4-6 \
  --model bedrock:us.amazon.nova-pro-v1:0
```
A case passes only if **all** models pass it.

## Baseline & Regression Detection

```bash
# Save current results as baseline (run once on main after initial setup)
supereval run my-dataset --model <model> --update-baseline

# Compare against baseline (run on PRs)
supereval run my-dataset --model <model> --compare-baseline

# Save results to a file
supereval run my-dataset --model <model> --compare-baseline --output results.json
```

`baseline.json` is committed to git alongside your test cases. The CI workflow updates it automatically on every push to main.

## Configuring Thresholds

Thresholds are set per-dataset in `dataset.json`. Defaults are zero tolerance.

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

| Field | Default | Meaning |
|---|---|---|
| `pass_rate` | `1.0` | Minimum fraction of cases that must pass |
| `fail_on_regression` | `true` | Fail if any previously-passing case now fails |
| `max_score_drop` | `null` | Max allowed pass rate drop vs baseline (e.g. `0.05` = 5%) |
| `max_cost_usd` | `null` | Fail if total run cost exceeds this amount (USD) |
| `max_p95_latency_ms` | `null` | Fail if p95 response latency exceeds this threshold (ms) |

## CI Setup

### GitHub Actions

Copy `.github/workflows/eval.yml` to your repository. Configure the dataset name and model:

```yaml
env:
  DATASET: aws-support-qa
  MODEL: bedrock:us.anthropic.claude-sonnet-4-6
```

**What the workflow does:**

- **On PR** — runs eval, compares to baseline, fails the build on regression
- **On push to main** — runs eval, updates `baseline.json`, commits it back

### AWS CodeBuild / CodePipeline

Copy `codebuild/buildspec.yml` to your repository. Set these environment variables in your CodeBuild project:

```yaml
DATASET: aws-support-qa
MODEL: bedrock:us.anthropic.claude-sonnet-4-6
SUPEREVAL_MODE: pr    # or: main
```

### GitLab CI

Copy `gitlab/eval.yml` to `.gitlab-ci.yml` (or include it). Set `DATASET`, `MODEL`, and any API keys as masked CI/CD variables in your project settings.

### CircleCI

Copy `circleci/config.yml` to `.circleci/config.yml`. Set `DATASET`, `MODEL`, and any API keys as environment variables in your CircleCI project settings.

## Cost & Latency Tracking

Every `supereval run` records cost, token usage, and latency to a local DuckDB database.

```bash
supereval history list
supereval history list --dataset aws-support-qa
supereval history show <run-id>
supereval history stats aws-support-qa --last 20
```

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `SUPEREVAL_DATASETS_DIR` | `./datasets` | Root directory for dataset files |
| `SUPEREVAL_DB_PATH` | `./supereval.db` | Path to the DuckDB run history database |

## Choosing an Eval Tool

| Tool | Best For | Not Great For |
|---|---|---|
| **[Promptfoo](https://www.promptfoo.dev)** | Prompt iteration, model comparison, CI integration | Production tracing, complex agent eval |
| **[Langfuse](https://langfuse.com)** | Production observability, scoring live traces | Offline/pre-deployment eval workflows |
| **[DeepEval](https://docs.confident-ai.com)** | Python-native evals with rich built-in metrics | Teams preferring config-over-code |
| **[RAGAS](https://docs.ragas.io)** | RAG eval with embedding-based metrics (context recall, precision) | Teams that don't want an external dependency or embedding model |
| **[Braintrust](https://www.braintrust.dev)** | Polished UI, team collaboration, dataset management | Fully self-hosted / offline-only setups |
