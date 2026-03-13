# supereval Roadmap

Tracks what is currently supported, what is planned, and what is out of scope.

---

## Document Sources (for `supereval generate`)

| Source | Status | Notes |
|---|---|---|
| Local files (`.txt`, `.md`) | Supported | |
| Local files (`.pdf`) | Supported | Requires `pip install 'supereval[pdf]'` |
| Local directory (recursive) | Supported | All supported file types under a path |
| Amazon S3 | Supported | `s3://bucket/prefix` — `S3Source` in `sources.py`; requires boto3 + s3:ListObjectsV2/GetObject |
| URLs | Planned | HTTP fetch + text extraction |
| Amazon Kendra | Planned | Pull documents from a Kendra index |
| Bedrock Knowledge Base | Planned | Pull source documents from a KB |

---

## Dataset Types

| Type | Registry | Generation | Notes |
|---|---|---|---|
| `qa` | Supported | Supported | Query + ground truth |
| `classification` | Supported | Supported | Label-aware generation; requires `--labels` set on dataset |
| `instruction` | Supported | Supported | Rubric-based generation |
| `rag` | Planned | Planned | Extends `qa` with expected source documents |
| `multiturn` | Planned | Planned | Conversation datasets |
| `agent` | Supported | Planned | Trajectory + tool-call evaluation; see Agent Eval section |

---

## Generator Backends (for `supereval generate`)

| Backend | Status | Notes |
|---|---|---|
| Amazon Bedrock | Supported | Default. Requires `boto3` + Bedrock model access |
| Anthropic API | Supported | `--backend anthropic`; requires `pip install 'supereval[anthropic]'` + `ANTHROPIC_API_KEY` |
| OpenAI | Planned | For customers evaluating GPT models as generators |
| Azure OpenAI | Planned | |

---

## CI Systems (for `supereval run` + `.github/workflows/`)

| System | Status | Notes |
|---|---|---|
| GitHub Actions | Supported | Template at `.github/workflows/eval.yml` |
| AWS CodeBuild / CodePipeline | Supported | `codebuild/buildspec.yml`; set `SUPEREVAL_MODE=pr|main` |
| GitLab CI | Planned | |
| CircleCI | Planned | |

---

## Result Storage (for `supereval run`)

| Storage | Status | Notes |
|---|---|---|
| `baseline.json` committed to git | Supported | Simple, reproducible, reviewable in PRs |
| Per-run results JSON (`--output`) | Supported | Written locally; not persisted automatically |
| DuckDB local store | Supported | `supereval history list/show/stats`; `SUPEREVAL_DB_PATH` to override location |
| S3 results store | Planned | Team-shared history; needed for dashboards |
| Langfuse integration | Planned | Post-deployment monitoring alongside offline evals |

---

## Cost & Latency Tracking

| Feature | Status | Notes |
|---|---|---|
| Per-run latency in results JSON | Supported (via Promptfoo) | |
| Per-run token cost in results JSON | Supported (via Promptfoo) | |
| Persistent cost/latency history | Supported | DuckDB store; `supereval history` commands |
| Cost regression alerts | Supported | `max_cost_usd` threshold in `dataset.json` |
| Latency p95 tracking | Supported | `max_p95_latency_ms` threshold; p50/p95 stored per run |

---

## Agent Eval (supereval agent)

### Core (v1)

| Feature | Status | Notes |
|---|---|---|
| `agent` dataset type with tool catalog | Supported | `dataset.json` holds tool specs (name, description, JSON Schema) |
| Per-case mock tool environments | Supported | Canned responses + errors; keeps test cases self-contained and reproducible |
| `AgentRunner` protocol | Supported | Framework-agnostic; customers implement `run(task, tool_executor) -> Trajectory` |
| `MockToolExecutor` | Supported | Returns canned responses from test case definition |
| Full trajectory model | Supported | Steps typed as thought / tool_call / observation / answer |
| Answer scoring | Supported | exact / contains / regex / LLM judge |
| Tool-call scoring | Supported | must_call, must_not_call, must_call_with checks |
| Efficiency scoring | Supported | Normalized step count vs. max_steps (informational by default) |
| Reasoning scoring via LLM judge | Supported | Bedrock judge evaluates thought chain for coherence and hallucinations |
| Configurable score thresholds | Supported | Per-dataset min_answer_score, min_tool_score, min_reasoning_score |
| Baseline & regression detection | Supported | Same pattern as LLM evals; `--compare-baseline` / `--update-baseline` |
| `supereval agent run --runner module:Class` | Supported | Python import path to customer's AgentRunner implementation |
| Cost & latency tracking | Supported | Unified `supereval.db` with LLM runs; `run_type` column distinguishes them |
| `supereval agent history` subcommand | Supported | list / show / stats; reuses store.py |
| Reference AgentRunner examples | Supported | `examples/`: custom ReAct/Bedrock, Bedrock Agents, LangChain |

### Judge improvements (planned)

Informed by comparison with AWS Bedrock's built-in LLM-as-judge metric suite (12 metrics across quality, safety, and style dimensions).

| Feature | Notes |
|---|---|
| Split `judge_reasoning` into Faithfulness + Logical Coherence | Faithfulness = did the agent hallucinate from tool results?; Logical Coherence = did each step follow from the previous? Currently conflated into one call |
| Add Helpfulness / Relevance dimension to `judge_answer` | Especially valuable for instruction-type datasets where correctness alone is insufficient |
| Adopt per-metric ordinal scoring scales | Use metric-appropriate scales (binary for safety checks, 3-point for correctness, 5-point for quality, 7-point for helpfulness) instead of a single continuous 0.0–1.0 float for all dimensions |
| Make ground truth optional in answer judging | Currently `expected_answer` is always required; judge should degrade gracefully to rubric-only scoring when no ground truth is available |

### Out of scope for v1 (prioritize later)

| Feature | Notes |
|---|---|
| Multi-turn conversation state | Each test case is a single task invocation; stateful conversation datasets are v2 |
| Parallel tool calls | Tool calls are sequential for now; parallel execution adds replay complexity |
| Synthetic generation for agent cases | Writing realistic mock environments requires domain knowledge; needs separate design |
| Real framework adapters in the package | Reference impls live in `examples/` only; not installed with `supereval` |
| Trajectory similarity scoring | Comparing step sequences across runs (edit distance, embedding similarity) |
| Bedrock Agents native integration | Polling `GetAgentExecution`, parsing Bedrock Agents trace format |
| Agent dataset generation from existing traces | Import real agent traces to bootstrap test cases |
| AWS Step Functions / multi-agent orchestration | Evaluating pipelines of agents, not single agents |

---

## Other Features

| Feature | Status | Notes |
|---|---|---|
| `supereval providers` | Supported | Lists Bedrock model IDs, Anthropic API model IDs, and Promptfoo provider IDs |
| Multi-dataset CI runs | Planned | Run all datasets in one pipeline step |
| Dataset versioning (tags) | Planned | Semantic version tags beyond git history |
| Interactive case review | Planned | `supereval generate --interactive` for approve/edit/skip flow |
