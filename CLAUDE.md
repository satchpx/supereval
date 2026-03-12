# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Structure

```
supereval/                ← repo root (flat)
  supereval/              ← Python package (the CLI)
    cli.py                ← Typer entry point; all commands defined here
    models.py             ← Pydantic models: DatasetMeta, Thresholds, test case types
    storage.py            ← read/write datasets and cases; resolves SUPEREVAL_DATASETS_DIR
    export.py             ← transforms datasets to Promptfoo YAML; _build_tests() is shared
    runner.py             ← builds Promptfoo config, invokes promptfoo eval, parses output
    baseline.py           ← load/save/compare baseline.json; RegressionReport
    store.py              ← DuckDB-backed run history; record_run, record_agent_run, list_runs (run_type filter), get_run, get_stats
    sources.py            ← DocumentSource protocol + LocalFileSource + S3Source
    generator.py          ← BedrockGenerator + AnthropicGenerator; _build_prompt (type-aware); _parse_response
    agent/                ← agent eval subpackage
      models.py           ← AgentDatasetMeta, AgentTestCase, Trajectory, Step, TrajectoryScore, AgentRunResult
      storage.py          ← agent datasets at $SUPEREVAL_DATASETS_DIR/agents/<name>/
      scorer.py           ← score_trajectory(): answer, tool, efficiency, reasoning dimensions
      judge.py            ← BedrockJudge: judge_answer(), judge_reasoning(); same Bedrock pattern as generator
      runner.py           ← AgentRunner + ToolExecutor protocols, MockToolExecutor, load_runner(), run_agent_eval()
      baseline.py         ← save/load/compare agent baselines; AgentRegressionReport
      cli.py              ← agent_app: dataset/run/history subcommands; registered in cli.py
  datasets/               ← test data (committed to git)
    <dataset-name>/
      dataset.json        ← DatasetMeta (type, thresholds, tags, etc.)
      cases.jsonl         ← one test case per line
      baseline.json       ← committed; updated on push to main via CI
  examples/               ← agent runner reference implementations (not installed with package)
    react_bedrock.py      ← ReActAgent: custom ReAct loop on Bedrock
    langchain.py          ← LangChainAgentRunner: wraps LangChain agent executor
    bedrock_agents.py     ← BedrockAgentRunner: integration testing with deployed Bedrock Agents
  tests/
    conftest.py           ← shared fixtures: datasets_dir (tmp + env var), qa/classification/instruction meta, sample cases/results
    test_models.py        ← Pydantic schema validation
    test_storage.py       ← save/load/list datasets and cases
    test_export.py        ← _build_tests() and export_to_promptfoo() for all types
    test_sources.py       ← LocalFileSource + S3Source: files, directories, URI parsing, error cases
    test_generator.py     ← _parse_response(), _chunk_document(), BedrockGenerator, AnthropicGenerator (mocked)
    test_baseline.py      ← compare_to_baseline() for all threshold combinations (incl. cost/latency)
    test_runner.py        ← config building (all 3 tiers), output parsing, cost/latency extraction
    test_store.py         ← DuckDB store: record, list, get, stats; uses SUPEREVAL_DB_PATH isolation
    test_cli.py           ← every CLI command via Typer CliRunner; history subcommands; --no-record
    test_agent_models.py  ← Trajectory properties, AgentRunResult computed fields, serialisation
    test_agent_runner.py  ← MockToolExecutor, load_runner, run_agent_eval with DummyAgent fixture
    test_agent_scorer.py  ← all scoring dimensions: answer/tool/efficiency/composite
    test_agent_judge.py   ← BedrockJudge with mocked Bedrock responses; format_trajectory
    test_agent_cli.py     ← agent dataset/run/history commands via CliRunner
  codebuild/
    buildspec.yml         ← AWS CodeBuild/CodePipeline template; set SUPEREVAL_MODE=pr|main
  .github/workflows/
    eval.yml              ← GitHub Actions: compare baseline on PR, update on main
  generate_diagram.py     ← regenerates decision_tree.png (graphviz)
  pyproject.toml          ← defines `supereval` CLI entrypoint; boto3 dep; [pdf], [anthropic], [dev] optional extras
  ROADMAP.md              ← tracks supported vs. planned features
```

## Development Setup

```bash
pip install -e '.[dev]'         # installs supereval CLI + pytest + pytest-mock
pip install -e '.[pdf]'         # optional: add PDF document support
pip install -e '.[anthropic]'   # optional: Anthropic API generator backend
npm install -g promptfoo        # required for supereval run
```

## Running Tests

```bash
python3 -m pytest                          # run all tests
python3 -m pytest tests/test_baseline.py  # run a single file
python3 -m pytest -k "test_fails"         # run tests matching a pattern
python3 -m pytest -v                       # verbose output
```

## Key Commands

```bash
# Dataset management
supereval dataset create <name> --type qa|classification|instruction --description "..."
supereval dataset add-cases <name> --from cases.jsonl
supereval dataset list
supereval dataset show <name>
supereval dataset validate <name>
supereval dataset export <name> --output promptfoo-tests.yaml

# Running evals
supereval run <dataset> --model anthropic:claude-opus-4-6          # Tier 1: auto-config
supereval run <dataset> --model anthropic:claude-opus-4-6 \        # Tier 2: custom prompt
  --prompt "You are an AWS expert. Answer: {{query}}"
supereval run <dataset> --config promptfoo.yaml                     # Tier 3: own config

# Baseline management
supereval run <dataset> --model <m> --compare-baseline             # fails on regression
supereval run <dataset> --model <m> --update-baseline              # save new baseline
supereval run <dataset> --model <m> --compare-baseline --output results.json

# Synthetic generation (qa, classification, instruction datasets; stages for review by default)
supereval generate <dataset> --from docs/my-file.md --count 20           # local file, Bedrock
supereval generate <dataset> --from s3://bucket/docs/ --count 20         # S3 source
supereval generate <dataset> --from docs/ --backend anthropic            # Anthropic API
supereval generate <dataset> --from docs/ --count 20 --auto-import       # skip review

# Provider discovery
supereval providers    # lists Bedrock model IDs, Anthropic API IDs, Promptfoo provider IDs

# Run history (backed by DuckDB; SUPEREVAL_DB_PATH controls file location)
supereval history list [--dataset <name>] [--limit N]
supereval history show <run-id> [--cases]
supereval history stats <dataset> [--last N]
supereval run <dataset> --model <m> --no-record   # skip persisting to history

# Agent eval
supereval agent dataset create <name> [--tool-specs tools.json]
supereval agent dataset add-cases <name> --from cases.jsonl
supereval agent run <dataset> --runner myapp.agents:MyAgent
supereval agent run <dataset> --runner myapp.agents:MyAgent --judge-model anthropic.claude-3-5-sonnet-20241022-v2:0
supereval agent run <dataset> --runner myapp.agents:MyAgent --compare-baseline
supereval agent run <dataset> --runner myapp.agents:MyAgent --update-baseline
supereval agent history list / show / stats   # same pattern as LLM history

# Regenerate decision tree diagram
python3 generate_diagram.py
```

## Architecture: How the Pieces Fit Together

**Dataset types** (`models.py`): `qa`, `classification`, `instruction`. The `DatasetMeta.type` field
drives which Pydantic model validates each case and which export/prompt logic applies.

**Export vs. runner** (`export.py`, `runner.py`): `_build_tests()` in `export.py` is the shared
function that converts a dataset to a Promptfoo `tests` list. `export_to_promptfoo()` wraps it into
YAML for the CLI. `runner.py` calls `_build_tests()` directly when building the config before invoking
`promptfoo eval`.

**Three-tier config resolution** (`runner.py`): When `supereval run` is called, the runner determines
which tier applies — Tier 3 (customer config) → inject tests only; Tier 2 (models + prompt) → build
full config with custom prompt; Tier 1 (models only) → build full config with the default prompt
template for the dataset type. All three paths produce the same shape of config passed to `promptfoo eval`.

**Baseline flow** (`baseline.py`): `baseline.json` is committed to git alongside `cases.jsonl`. On PRs,
`--compare-baseline` loads this file and checks thresholds. On push to main, `--update-baseline`
overwrites it. The GitHub Actions workflow commits the updated baseline back automatically.

**Thresholds** (`models.py`): Stored in `dataset.json` under `"thresholds"`. Defaults are zero
tolerance (`pass_rate=1.0`, `fail_on_regression=true`). Configurable per dataset.

**Cost & latency tracking** (`store.py`, `runner.py`): Every `supereval run` automatically persists
a summary row to DuckDB (`supereval.db` or `SUPEREVAL_DB_PATH`). `CaseResult` carries `cost_usd`
and `latency_ms`; `RunResult` carries `total_tokens`, `prompt_tokens`, `completion_tokens` and
computed properties `total_cost_usd`, `avg_latency_ms`, `p50_latency_ms`, `p95_latency_ms`.
These are populated from Promptfoo's `stats.tokenUsage` block and per-result `cost`/`latencyMs` fields.
Use `--no-record` to skip persistence. The `history` subcommand exposes `list`, `show`, and `stats`.

**Cost/latency thresholds** (`baseline.py`, `models.py`): `Thresholds` includes `max_cost_usd` and
`max_p95_latency_ms`. When set, `compare_to_baseline()` adds a failure reason if either threshold is
breached, causing `--compare-baseline` to exit non-zero in CI. Same fields exist in `AgentThresholds`.

**Agent eval subpackage** (`agent/`): Plugged into the main CLI via `app.add_typer(agent_app, name="agent")`.
Agent datasets live at `$SUPEREVAL_DATASETS_DIR/agents/<name>/` (separate from LLM datasets at
`$SUPEREVAL_DATASETS_DIR/<name>/`). `AgentRunner` and `ToolExecutor` are runtime-checkable protocols
in `agent/runner.py`. `load_runner("module:Class")` dynamically imports the customer's implementation.
`MockToolExecutor` wraps per-case canned responses. `score_trajectory()` in `agent/scorer.py` produces
a `TrajectoryScore` with four dimensions; the composite score is a weighted average with reasoning
weight zeroed out when no judge is provided. `BedrockJudge` in `agent/judge.py` follows the same
boto3 invoke pattern as `BedrockGenerator`. Agent runs are stored in the shared DuckDB `runs` table
with `run_type='agent'`; `list_runs()` accepts an optional `run_type` filter. Reference implementations
for real frameworks live in `examples/` — not installed with the package.

**Datasets directory resolution** (`storage.py`): Defaults to `./datasets` relative to `cwd`.
Override with `SUPEREVAL_DATASETS_DIR`. Agent datasets use `$SUPEREVAL_DATASETS_DIR/agents/<name>/`.
DB defaults to `./supereval.db`; override with `SUPEREVAL_DB_PATH`.

**Testing conventions** (`tests/`): All tests use `monkeypatch.setenv("SUPEREVAL_DATASETS_DIR", ...)` via
the `datasets_dir` fixture in `conftest.py` — no test touches the real `./datasets` directory. DuckDB
tests use `monkeypatch.setenv("SUPEREVAL_DB_PATH", str(tmp_path / "test.db"))` (autouse in
`test_store.py`) for isolation. Bedrock calls are mocked via `unittest.mock.patch("boto3.client")`.
Promptfoo subprocess calls are mocked via `patch("supereval.runner.subprocess.run")`. Agent tests use
a `DummyAgent` (calls all available tools once, returns "done") for deterministic runner tests, and mock
`load_runner` / `run_agent_eval` for CLI tests. The `agent_meta`, `sample_agent_cases`, and
`agent_dataset_with_cases` fixtures are in `conftest.py`.

**Synthetic generation** (`sources.py`, `generator.py`): `DocumentSource` is a Protocol — `LocalFileSource`
handles local files/dirs, `S3Source` handles `s3://bucket/prefix` URIs (boto3 list+get). `_build_prompt()`
selects the prompt by `dataset_type` (`qa`, `classification`, `instruction`). Classification requires
`meta.labels` to be set on the dataset; instruction generates rubrics. Both `BedrockGenerator` and
`AnthropicGenerator` share the same interface (`generate(doc, count, dataset_type, labels)`). The CLI
`generate` command auto-detects `s3://` in `--from`, selects the backend via `--backend bedrock|anthropic`,
and defaults the model ID based on the backend. Staged output includes `source_excerpt`/`case_type` for
review convenience; ignored by Pydantic on import.
