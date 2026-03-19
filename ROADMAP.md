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
| URLs | Supported | HTTP/HTTPS fetch + HTML stripping; auto-detected from `--from` |
| Amazon Kendra | Coming Soon | Pull documents from a Kendra index |
| Bedrock Knowledge Base | Coming Soon | Pull source documents from a KB |

---

## Dataset Types

| Type | Registry | Generation | Notes |
|---|---|---|---|
| `qa` | Supported | Supported | Query + ground truth |
| `classification` | Supported | Supported | Label-aware generation; requires `--labels` set on dataset |
| `instruction` | Supported | Supported | Rubric-based generation |
| `rag` | Supported | Supported | Extends `qa` with retrieved_contexts; Faithfulness + Answer Correctness scoring via Bedrock judge; static contexts (Option A) |
| `rag` (live retrieval) | Planned | — | Option B: contexts fetched live from your retriever at eval time; tests the full RAG pipeline end-to-end |
| `multiturn` | Coming Soon | Coming Soon | Conversation datasets |
| `agent` | Supported | Planned | Trajectory + tool-call evaluation; see Agent Eval section |

---

## Generator Backends (for `supereval generate`)

| Backend | Status | Notes |
|---|---|---|
| Amazon Bedrock | Supported | Default. Requires `boto3` + Bedrock model access |
| Anthropic API | Supported | `--backend anthropic`; requires `pip install 'supereval[anthropic]'` + `ANTHROPIC_API_KEY` |
| OpenAI | Supported | `--backend openai`; requires `pip install 'supereval[openai]'` + `OPENAI_API_KEY` |
| Azure OpenAI | Supported | `--backend azure-openai`; requires `pip install 'supereval[openai]'`; set `AZURE_OPENAI_ENDPOINT` or pass `--azure-endpoint` |

---

## CI Systems (for `supereval run` + `.github/workflows/`)

| System | Status | Notes |
|---|---|---|
| GitHub Actions | Supported | Template at `.github/workflows/eval.yml` |
| AWS CodeBuild / CodePipeline | Supported | `codebuild/buildspec.yml`; set `SUPEREVAL_MODE=pr|main` |
| GitLab CI | Supported | Template at `gitlab/eval.yml` |
| CircleCI | Supported | Template at `circleci/config.yml` |

---

## Result Storage (for `supereval run`)

| Storage | Status | Notes |
|---|---|---|
| `baseline.json` committed to git | Supported | Simple, reproducible, reviewable in PRs |
| Per-run results JSON (`--output`) | Supported | Written locally; not persisted automatically |
| DuckDB local store | Supported | `supereval history list/show/stats`; `SUPEREVAL_DB_PATH` to override location |
| S3 results store | Planned | Team-shared history; needed for dashboards |
| Langfuse integration | Coming Soon | Post-deployment monitoring alongside offline evals; trace supereval runs to Langfuse for production comparison |

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

### Judge improvements

Informed by comparison with AWS Bedrock's built-in LLM-as-judge metric suite (12 metrics across quality, safety, and style dimensions).

| Feature | Status | Notes |
|---|---|---|
| Split `judge_reasoning` into Faithfulness + Logical Coherence | Supported | Two focused Bedrock calls; Faithfulness = did agent hallucinate from tool results?; Logical Coherence = did each step follow from the previous? |
| Adopt per-metric ordinal scoring scales | Supported | 3-point for Correctness, 5-point for Completeness / Faithfulness / Coherence; native labels ("partially correct", "most faithful") surfaced alongside normalized 0.0–1.0 float |
| Make ground truth optional in answer judging | Supported | `expected_answer` is now optional; judge degrades gracefully to rubric-only scoring when omitted |
| Add Helpfulness / Relevance dimension to `judge_answer` | Supported | 5-point scale; third Bedrock call alongside Correctness + Completeness; normalized score averaged into composite |

### Out of scope for v1 (prioritize later)

| Feature | Notes |
|---|---|
| Multi-turn conversation state | Each test case is a single task invocation; stateful conversation datasets are v2 |
| Parallel tool calls | Tool calls are sequential for now; parallel execution adds replay complexity |
| Synthetic generation for agent cases | **Supported** (`supereval agent generate --from docs/`) — LLM generates task scenarios from documentation; `tools={}` block left empty, fill via trace import |
| Real framework adapters in the package | Reference impls live in `examples/` only; not installed with `supereval` |
| Trajectory similarity scoring | Comparing step sequences across runs (edit distance, embedding similarity) |
| Bedrock Agents native integration | Polling `GetAgentExecution`, parsing Bedrock Agents trace format |
| Agent dataset generation from existing traces | **Supported** (`supereval agent generate --from traces.jsonl`) — imports real execution traces; infers tool mocks from step results |
| AWS Step Functions / multi-agent orchestration | Evaluating pipelines of agents, not single agents |

---

## Other Features

| Feature | Status | Notes |
|---|---|---|
| `supereval providers` | Supported | Lists Bedrock model IDs, Anthropic API model IDs, and Promptfoo provider IDs |
| Multi-dataset CI runs | Supported | `supereval run-all`; exits non-zero if any dataset fails |
| Dataset versioning (tags) | Supported | `supereval dataset version tag/list/show/restore`; semantic version snapshots stored under `datasets/<name>/versions/` |
| Interactive case review | Supported | `supereval generate --interactive` and `supereval agent generate --interactive`; keep/edit/skip/quit per case |
