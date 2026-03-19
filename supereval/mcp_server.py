"""
MCP server for supereval.

Exposes supereval's dataset management, eval running, generation, history,
and versioning as MCP tools so AI coding assistants (Claude Code, Kiro, etc.)
can operate supereval on behalf of users via natural language.

Usage
-----
Start the server (stdio transport, used by all MCP clients):

    supereval mcp

Then register it in your MCP client config. Claude Code (~/.claude.json):

    {
      "mcpServers": {
        "supereval": {
          "command": "supereval",
          "args": ["mcp"]
        }
      }
    }

Kiro (.kiro/settings/mcp.json):

    {
      "mcpServers": {
        "supereval": {
          "command": "supereval",
          "args": ["mcp"]
        }
      }
    }
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path


def _get_mcp():
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        raise ImportError(
            "mcp package is required. Install with: pip install 'supereval[mcp]'"
        )
    return FastMCP


def _fmt_error(exc: Exception) -> str:
    return f"Error: {exc}"


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def create_server():
    FastMCP = _get_mcp()

    mcp = FastMCP(
        "supereval",
        instructions=(
            "supereval manages LLM and agent evaluation datasets. "
            "Use it to create datasets, add test cases, run evals against models, "
            "track history, and generate synthetic test cases from documentation. "
            "Always list_datasets first to see what exists before running evals."
        ),
    )

    # -----------------------------------------------------------------------
    # Dataset management
    # -----------------------------------------------------------------------

    @mcp.tool()
    def list_datasets() -> str:
        """List all LLM evaluation datasets (qa, classification, instruction).
        Returns dataset names, types, case counts, and descriptions."""
        from .storage import list_datasets as _list
        datasets = _list()
        if not datasets:
            return "No datasets found. Create one with create_dataset()."
        lines = ["LLM Datasets:", ""]
        for ds in datasets:
            case_label = f"{_count_cases(ds.name)} cases"
            tags = f"  tags: {', '.join(ds.tags)}" if ds.tags else ""
            lines.append(f"  {ds.name}  [{ds.type.value}]  {case_label}")
            if ds.description:
                lines.append(f"    {ds.description}")
            if tags:
                lines.append(tags)
        return "\n".join(lines)

    @mcp.tool()
    def show_dataset(name: str) -> str:
        """Show full details for a dataset including thresholds and a case preview.

        Args:
            name: Dataset name (e.g. 'aws-support-qa')
        """
        try:
            from .storage import load_dataset_meta, load_cases, dataset_path
            meta = load_dataset_meta(name)
            try:
                cases = load_cases(name)
            except Exception:
                cases = []

            lines = [
                f"Dataset: {meta.name}  (id: {meta.id})",
                f"  Type:        {meta.type.value}",
                f"  Description: {meta.description or '(none)'}",
                f"  Tags:        {', '.join(meta.tags) or '(none)'}",
                f"  Cases:       {len(cases)}",
                f"  Created:     {meta.created_at}",
                f"  Updated:     {meta.updated_at}",
                "",
                "  Thresholds:",
                f"    pass_rate:            {_pct(meta.thresholds.pass_rate)}",
                f"    fail_on_regression:   {meta.thresholds.fail_on_regression}",
            ]
            if meta.thresholds.max_cost_usd is not None:
                lines.append(f"    max_cost_usd:         ${meta.thresholds.max_cost_usd:.4f}")
            if meta.thresholds.max_p95_latency_ms is not None:
                lines.append(f"    max_p95_latency_ms:   {meta.thresholds.max_p95_latency_ms}ms")
            if meta.labels:
                lines.append(f"  Labels:      {', '.join(meta.labels)}")

            if cases:
                lines.append("")
                lines.append(f"  Preview (first {min(3, len(cases))} of {len(cases)} cases):")
                for c in cases[:3]:
                    lines.append(f"    [{c.id}] {c.description or '(no description)'}")
                    lines.append(f"      input: {json.dumps(c.input.model_dump())}")

            # Baseline
            bl_path = dataset_path(name) / "baseline.json"
            if bl_path.exists():
                try:
                    bl = json.loads(bl_path.read_text())
                    summary = bl.get("summary", {})
                    lines.append("")
                    lines.append(
                        f"  Baseline:    {_pct(summary.get('pass_rate', 0))} pass rate  "
                        f"({summary.get('passed', '?')}/{summary.get('total', '?')} cases)"
                    )
                except Exception:
                    pass

            return "\n".join(lines)
        except FileNotFoundError as e:
            return _fmt_error(e)
        except Exception as e:
            return _fmt_error(e)

    @mcp.tool()
    def create_dataset(
        name: str,
        type: str,
        description: str = "",
        tags: str = "",
        labels: str = "",
    ) -> str:
        """Create a new LLM evaluation dataset.

        Args:
            name: Unique dataset name, e.g. 'aws-support-qa'
            type: Dataset type — 'qa', 'classification', or 'instruction'
            description: Human-readable description of what this dataset tests
            tags: Comma-separated tags, e.g. 'aws,support,chatbot'
            labels: Comma-separated class labels for classification datasets only,
                    e.g. 'billing,networking,storage'. Required for type=classification.
        """
        try:
            from .models import DatasetMeta, DatasetType
            from .storage import dataset_path, save_dataset_meta

            if dataset_path(name).exists():
                return f"Error: Dataset '{name}' already exists. Use show_dataset('{name}') to inspect it."

            try:
                ds_type = DatasetType(type)
            except ValueError:
                return f"Error: Invalid type '{type}'. Must be one of: qa, classification, instruction"

            if ds_type == DatasetType.classification and not labels:
                return (
                    "Error: Classification datasets require labels. "
                    "Pass labels='billing,networking,storage' (or whatever your categories are)."
                )

            tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
            label_list = [l.strip() for l in labels.split(",") if l.strip()] if labels else None

            meta = DatasetMeta(
                name=name,
                type=ds_type,
                description=description,
                tags=tag_list,
                labels=label_list,
            )
            save_dataset_meta(meta)
            return (
                f"Created dataset '{name}' (type: {type})\n"
                f"  Next: add test cases with add_cases('{name}', '/path/to/cases.jsonl')\n"
                f"  Or generate cases with generate_cases('{name}', '/path/to/docs/')"
            )
        except Exception as e:
            return _fmt_error(e)

    @mcp.tool()
    def add_cases(dataset: str, file_path: str) -> str:
        """Add test cases to a dataset from a JSONL file (one JSON object per line).

        Args:
            dataset: Dataset name
            file_path: Absolute path to a .jsonl file containing test cases
        """
        try:
            from .models import TYPE_TO_MODEL
            from .storage import load_dataset_meta, append_cases

            meta = load_dataset_meta(dataset)
            model = TYPE_TO_MODEL.get(meta.type)
            if model is None:
                return f"Error: Dataset type '{meta.type.value}' does not support add_cases via this tool."

            path = Path(file_path)
            if not path.exists():
                return f"Error: File not found: {file_path}"

            cases = []
            errors = []
            for i, line in enumerate(path.read_text().splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    cases.append(model.model_validate_json(line))
                except Exception as e:
                    errors.append(f"  Line {i}: {e}")

            if errors and not cases:
                return "Error: All lines failed validation:\n" + "\n".join(errors[:5])

            append_cases(dataset, cases)
            result = f"Added {len(cases)} case(s) to '{dataset}'"
            if errors:
                result += f"\n  Warning: {len(errors)} line(s) skipped due to validation errors"
            return result
        except FileNotFoundError as e:
            return _fmt_error(e)
        except Exception as e:
            return _fmt_error(e)

    @mcp.tool()
    def validate_dataset(name: str) -> str:
        """Validate all test cases in a dataset against the expected schema.

        Args:
            name: Dataset name
        """
        try:
            from .storage import load_cases
            cases = load_cases(name)
            return f"Dataset '{name}' is valid — {len(cases)} case(s) passed schema validation."
        except FileNotFoundError as e:
            return _fmt_error(e)
        except ValueError as e:
            return f"Validation failed:\n{e}"
        except Exception as e:
            return _fmt_error(e)

    # -----------------------------------------------------------------------
    # Running evals
    # -----------------------------------------------------------------------

    @mcp.tool()
    def run_eval(
        dataset: str,
        model: str,
        prompt: str = "",
        compare_baseline_flag: bool = False,
        update_baseline_flag: bool = False,
    ) -> str:
        """Run an LLM evaluation on a dataset against a model. Requires promptfoo installed (npm install -g promptfoo).

        Args:
            dataset: Dataset name to evaluate
            model: Model provider string, e.g. 'anthropic:claude-haiku-4-5-20251001',
                   'anthropic:claude-opus-4-6', 'bedrock:us.anthropic.claude-3-5-sonnet-20241022-v2:0'
            prompt: Optional custom prompt template. Use {{query}}, {{text}}, or {{instruction}}
                    as placeholders. Leave empty to use the default prompt for the dataset type.
            compare_baseline_flag: If True, compare results against the saved baseline and
                                    report any regressions.
            update_baseline_flag: If True, save these results as the new baseline.
        """
        try:
            from .runner import run_eval as _run
            from .store import record_run
            from .baseline import compare_to_baseline, save_baseline

            result = _run(
                dataset_name=dataset,
                models=[model],
                prompt=prompt or None,
            )
            record_run(result)

            lines = [
                f"Eval complete: {dataset}",
                f"  Run ID:    {result.run_id}",
                f"  Model:     {model}",
                f"  Passed:    {result.passed}/{result.total}",
                f"  Pass rate: {_pct(result.pass_rate)}",
            ]
            if result.total_cost_usd:
                lines.append(f"  Cost:      ${result.total_cost_usd:.4f}")
            if result.avg_latency_ms:
                lines.append(
                    f"  Latency:   avg {result.avg_latency_ms:.0f}ms  "
                    f"p95 {result.p95_latency_ms:.0f}ms"
                )

            failed = [c for c in result.cases if not c.passed]
            if failed:
                lines.append(f"\n  Failed cases ({len(failed)}):")
                for c in failed[:5]:
                    lines.append(f"    ✗  {json.dumps(c.vars)}")
                if len(failed) > 5:
                    lines.append(f"    ... and {len(failed) - 5} more")

            if compare_baseline_flag:
                report = compare_to_baseline(result)
                if report is None:
                    lines.append("\n  No baseline found — run with update_baseline_flag=True to set one.")
                elif report.passed:
                    lines.append(
                        f"\n  Baseline comparison: PASSED "
                        f"(baseline {_pct(report.baseline_pass_rate)} → current {_pct(report.current_pass_rate)})"
                    )
                else:
                    lines.append(f"\n  Baseline comparison: FAILED")
                    for reason in report.failure_reasons:
                        lines.append(f"    • {reason}")

            if update_baseline_flag:
                save_baseline(result)
                lines.append(f"\n  Baseline updated to {_pct(result.pass_rate)} pass rate.")

            return "\n".join(lines)
        except FileNotFoundError as e:
            return _fmt_error(e)
        except Exception as e:
            return f"Error running eval: {e}\n{traceback.format_exc()}"

    @mcp.tool()
    def run_all_evals(model: str, prompt: str = "") -> str:
        """Run all LLM datasets against a model. Useful for a full regression sweep.

        Args:
            model: Model provider string, e.g. 'anthropic:claude-haiku-4-5-20251001'
            prompt: Optional custom prompt template applied to all datasets
        """
        try:
            from .storage import list_datasets as _list
            from .runner import run_eval as _run
            from .store import record_run

            datasets = _list()
            if not datasets:
                return "No datasets found."

            lines = [f"Running {len(datasets)} dataset(s) against {model}:", ""]
            overall_passed = overall_total = 0

            for ds in datasets:
                try:
                    result = _run(
                        dataset_name=ds.name,
                        models=[model],
                        prompt=prompt or None,
                    )
                    record_run(result)
                    overall_passed += result.passed
                    overall_total += result.total
                    status = "PASS" if result.pass_rate == 1.0 else "FAIL"
                    lines.append(
                        f"  [{status}] {ds.name}  "
                        f"{result.passed}/{result.total}  {_pct(result.pass_rate)}"
                    )
                except Exception as e:
                    lines.append(f"  [ERROR] {ds.name}: {e}")

            lines.append("")
            if overall_total:
                lines.append(
                    f"Overall: {overall_passed}/{overall_total}  "
                    f"{_pct(overall_passed / overall_total)}"
                )
            return "\n".join(lines)
        except Exception as e:
            return _fmt_error(e)

    # -----------------------------------------------------------------------
    # Synthetic generation
    # -----------------------------------------------------------------------

    @mcp.tool()
    def generate_cases(
        dataset: str,
        from_path: str,
        count: int = 10,
        backend: str = "bedrock",
        model: str = "",
        auto_import: bool = False,
    ) -> str:
        """Generate synthetic test cases from documents using an LLM.

        Reads local files, directories, S3 URIs (s3://bucket/prefix), or URLs
        and uses an LLM to generate test cases grounded in that content.
        By default, cases are staged to <dataset>_staged.jsonl for review.
        Set auto_import=True to import directly without staging.

        Args:
            dataset: Dataset name to generate cases for
            from_path: Source — local file/dir, s3://bucket/prefix, or https:// URL
            count: Number of cases to generate (default 10)
            backend: LLM backend — 'bedrock' (default), 'anthropic', 'openai', 'azure-openai'
            model: Override the default model for the chosen backend
            auto_import: If True, import cases directly instead of staging for review
        """
        try:
            from .storage import load_dataset_meta, append_cases
            from .sources import LocalFileSource, S3Source, URLSource
            from .generator import (
                BedrockGenerator, AnthropicGenerator, OpenAIGenerator,
                DEFAULT_MODEL, DEFAULT_REGION, ANTHROPIC_DEFAULT_MODEL, OPENAI_DEFAULT_MODEL,
            )
            from .models import TYPE_TO_MODEL, DatasetType

            meta = load_dataset_meta(dataset)

            # Build document source
            if from_path.startswith("s3://"):
                source = S3Source(from_path)
            elif from_path.startswith("http://") or from_path.startswith("https://"):
                source = URLSource(from_path)
            else:
                source = LocalFileSource(Path(from_path))

            docs = list(source.load())
            if not docs:
                return f"Error: No documents found at '{from_path}'"

            # Build generator
            backend_lower = backend.lower()
            if backend_lower == "bedrock":
                gen = BedrockGenerator(model_id=model or DEFAULT_MODEL)
            elif backend_lower == "anthropic":
                gen = AnthropicGenerator(model_id=model or ANTHROPIC_DEFAULT_MODEL)
            elif backend_lower in ("openai", "azure-openai"):
                gen = OpenAIGenerator(model_id=model or OPENAI_DEFAULT_MODEL)
            else:
                return f"Error: Unknown backend '{backend}'. Use: bedrock, anthropic, openai, azure-openai"

            import math
            per_doc = max(1, math.ceil(count / len(docs)))
            all_cases = []
            for doc in docs:
                remaining = count - len(all_cases)
                if remaining <= 0:
                    break
                generated = gen.generate(
                    doc,
                    count=min(per_doc, remaining),
                    dataset_type=meta.type.value,
                    labels=meta.labels or [],
                )
                all_cases.extend(generated)
            all_cases = all_cases[:count]

            if not all_cases:
                return "No cases were generated. Check that the documents contain relevant content."

            # Convert to test case models
            case_model = TYPE_TO_MODEL.get(meta.type)
            if case_model is None:
                return f"Error: Generation not supported for dataset type '{meta.type.value}'"

            valid_cases = []
            for gc in all_cases:
                try:
                    valid_cases.append(case_model.model_validate(gc.to_case_dict()))
                except Exception:
                    continue

            if auto_import:
                append_cases(dataset, valid_cases)
                return (
                    f"Generated and imported {len(valid_cases)} case(s) into '{dataset}'.\n"
                    f"  Source: {from_path}  ({len(docs)} document(s))\n"
                    f"  Backend: {backend}"
                )
            else:
                staged_path = Path(f"{dataset}_staged.jsonl")
                with staged_path.open("w") as f:
                    for c in all_cases:
                        f.write(json.dumps(c.to_staged_dict()) + "\n")
                return (
                    f"Generated {len(all_cases)} case(s) — staged for review.\n"
                    f"  Staged file: {staged_path.resolve()}\n"
                    f"  Source: {from_path}  ({len(docs)} document(s))\n"
                    f"  Backend: {backend}\n\n"
                    f"Review the staged file, then import with:\n"
                    f"  add_cases('{dataset}', '{staged_path.resolve()}')\n"
                    f"Or re-run with auto_import=True to skip staging."
                )
        except FileNotFoundError as e:
            return _fmt_error(e)
        except Exception as e:
            return _fmt_error(e)

    # -----------------------------------------------------------------------
    # Run history
    # -----------------------------------------------------------------------

    @mcp.tool()
    def list_history(dataset: str = "", limit: int = 10) -> str:
        """List recent eval runs from history.

        Args:
            dataset: Filter by dataset name (leave empty for all datasets)
            limit: Maximum number of runs to return (default 10)
        """
        try:
            from .store import list_runs
            runs = list_runs(dataset=dataset or None, limit=limit)
            if not runs:
                msg = f"No runs found"
                if dataset:
                    msg += f" for dataset '{dataset}'"
                return msg + "."

            lines = ["Recent runs:", ""]
            for r in runs:
                providers = json.loads(r["providers"]) if isinstance(r["providers"], str) else r["providers"]
                provider_str = ", ".join(providers) if providers else "(none)"
                lines.append(
                    f"  {r['run_id']}  [{r['run_type']}]  {r['dataset']}"
                )
                lines.append(
                    f"    {r['ran_at'][:19]}  {r['passed']}/{r['total_cases']} passed  "
                    f"{_pct(r['pass_rate'])}  {provider_str}"
                )
                if r["total_cost_usd"]:
                    lines.append(f"    cost: ${r['total_cost_usd']:.4f}  p95: {r['p95_latency_ms']:.0f}ms")
            return "\n".join(lines)
        except Exception as e:
            return _fmt_error(e)

    @mcp.tool()
    def show_run(run_id: str, include_cases: bool = False) -> str:
        """Show details for a specific eval run.

        Args:
            run_id: Run ID from list_history (e.g. 'run_7d4e9f2a')
            include_cases: If True, include per-case results (can be long)
        """
        try:
            from .store import get_run, get_run_cases
            run = get_run(run_id)
            if run is None:
                return f"Run '{run_id}' not found."

            providers = json.loads(run["providers"]) if isinstance(run["providers"], str) else run["providers"]
            lines = [
                f"Run: {run['run_id']}",
                f"  Dataset:   {run['dataset']}",
                f"  Type:      {run['run_type']}",
                f"  Ran at:    {run['ran_at'][:19]}",
                f"  Models:    {', '.join(providers) or '(none)'}",
                f"  Results:   {run['passed']}/{run['total_cases']} passed  {_pct(run['pass_rate'])}",
            ]
            if run["total_cost_usd"]:
                lines.append(f"  Cost:      ${run['total_cost_usd']:.4f}")
            if run["avg_latency_ms"]:
                lines.append(
                    f"  Latency:   avg {run['avg_latency_ms']:.0f}ms  "
                    f"p50 {run['p50_latency_ms']:.0f}ms  "
                    f"p95 {run['p95_latency_ms']:.0f}ms"
                )
            if run["total_tokens"]:
                lines.append(
                    f"  Tokens:    {run['total_tokens']}  "
                    f"(prompt: {run['prompt_tokens']}  completion: {run['completion_tokens']})"
                )

            if include_cases:
                cases = get_run_cases(run_id)
                lines.append(f"\n  Cases ({len(cases)}):")
                for c in cases:
                    icon = "✓" if c["passed"] else "✗"
                    lines.append(
                        f"    {icon}  score={c['score']:.2f}  "
                        f"{json.dumps(c['vars'])}"
                    )
            return "\n".join(lines)
        except Exception as e:
            return _fmt_error(e)

    @mcp.tool()
    def get_stats(dataset: str, last: int = 10) -> str:
        """Get aggregated pass rate, cost, and latency stats for a dataset over recent runs.

        Args:
            dataset: Dataset name
            last: Number of most recent runs to aggregate (default 10)
        """
        try:
            from .store import get_stats as _stats
            s = _stats(dataset, last=last)
            if not s or s.get("run_count") == 0:
                return f"No run history found for dataset '{dataset}'."

            lines = [
                f"Stats for '{dataset}' (last {s['run_count']} run(s)):",
                f"  Pass rate:    avg {_pct(s['avg_pass_rate'])}  "
                f"min {_pct(s['min_pass_rate'])}  max {_pct(s['max_pass_rate'])}",
            ]
            if s["total_cost_usd"]:
                lines.append(
                    f"  Cost:         total ${s['total_cost_usd']:.4f}  "
                    f"avg/run ${s['avg_cost_per_run']:.4f}  "
                    f"max/run ${s['max_cost_per_run']:.4f}"
                )
            if s["avg_latency_ms"]:
                lines.append(
                    f"  Latency:      avg {s['avg_latency_ms']:.0f}ms  "
                    f"max p95 {s['max_p95_latency_ms']:.0f}ms"
                )
            if s["total_tokens"]:
                lines.append(f"  Total tokens: {int(s['total_tokens'])}")
            return "\n".join(lines)
        except Exception as e:
            return _fmt_error(e)

    # -----------------------------------------------------------------------
    # Dataset versioning
    # -----------------------------------------------------------------------

    @mcp.tool()
    def version_tag(dataset: str, version: str, description: str = "") -> str:
        """Snapshot the current dataset as a named version. Version must follow
        semantic versioning: v<MAJOR>.<MINOR>.<PATCH> (e.g. v1.0.0).

        Args:
            dataset: Dataset name
            version: Version tag, e.g. 'v1.0.0'
            description: Optional note about what changed in this version
        """
        try:
            from .versioning import tag_version
            info = tag_version(dataset, version, description=description)
            return (
                f"Tagged '{dataset}' as {version}\n"
                f"  Cases:       {info.case_count}\n"
                f"  Created at:  {info.created_at[:19]}\n"
                f"  Description: {info.description or '(none)'}"
            )
        except (ValueError, FileNotFoundError) as e:
            return _fmt_error(e)
        except Exception as e:
            return _fmt_error(e)

    @mcp.tool()
    def version_list(dataset: str) -> str:
        """List all saved versions for a dataset.

        Args:
            dataset: Dataset name
        """
        try:
            from .versioning import list_versions
            versions = list_versions(dataset)
            if not versions:
                return f"No versions found for '{dataset}'. Create one with version_tag()."
            lines = [f"Versions for '{dataset}':", ""]
            for v in versions:
                lines.append(f"  {v.version}  {v.case_count} cases  {v.created_at[:19]}")
                if v.description:
                    lines.append(f"    {v.description}")
            return "\n".join(lines)
        except Exception as e:
            return _fmt_error(e)

    @mcp.tool()
    def version_restore(dataset: str, version: str) -> str:
        """Restore a dataset to a previously tagged version. This overwrites
        the current cases.jsonl with the version snapshot.

        Args:
            dataset: Dataset name
            version: Version to restore, e.g. 'v1.0.0'
        """
        try:
            from .versioning import restore_version
            count = restore_version(dataset, version)
            return (
                f"Restored '{dataset}' to {version} — {count} case(s) now in the working copy.\n"
                f"  The version snapshot is unchanged; the working cases.jsonl has been overwritten."
            )
        except (ValueError, FileNotFoundError) as e:
            return _fmt_error(e)
        except Exception as e:
            return _fmt_error(e)

    # -----------------------------------------------------------------------
    # Agent datasets
    # -----------------------------------------------------------------------

    @mcp.tool()
    def list_agent_datasets() -> str:
        """List all agent evaluation datasets."""
        try:
            from .agent.storage import list_agent_datasets as _list
            datasets = _list()
            if not datasets:
                return "No agent datasets found. Create one with create_agent_dataset()."
            lines = ["Agent Datasets:", ""]
            for ds in datasets:
                case_count = _count_agent_cases(ds.name)
                tool_names = [t.name for t in (ds.tools or [])]
                lines.append(f"  {ds.name}  {case_count} cases")
                if ds.description:
                    lines.append(f"    {ds.description}")
                if tool_names:
                    lines.append(f"    tools: {', '.join(tool_names)}")
            return "\n".join(lines)
        except Exception as e:
            return _fmt_error(e)

    @mcp.tool()
    def show_agent_dataset(name: str) -> str:
        """Show full details for an agent evaluation dataset.

        Args:
            name: Agent dataset name
        """
        try:
            from .agent.storage import load_agent_dataset_meta, load_agent_cases
            meta = load_agent_dataset_meta(name)
            try:
                cases = load_agent_cases(name)
            except Exception:
                cases = []

            lines = [
                f"Agent Dataset: {meta.name}  (id: {meta.id})",
                f"  Description: {meta.description or '(none)'}",
                f"  Cases:       {len(cases)}",
                f"  Created:     {meta.created_at}",
            ]
            if meta.tools:
                lines.append(f"  Tools ({len(meta.tools)}):")
                for t in meta.tools:
                    lines.append(f"    {t.name} — {t.description}")

            if cases:
                lines.append(f"\n  Preview (first {min(3, len(cases))} cases):")
                for c in cases[:3]:
                    lines.append(f"    [{c.id}] {c.description or c.input.task[:60]}")
                    lines.append(f"      must_call: {c.expected.must_call}")
                    if c.expected.answer:
                        lines.append(f"      answer:    {c.expected.answer[:80]}")

            return "\n".join(lines)
        except FileNotFoundError as e:
            return _fmt_error(e)
        except Exception as e:
            return _fmt_error(e)

    @mcp.tool()
    def create_agent_dataset(name: str, description: str = "") -> str:
        """Create a new agent evaluation dataset.

        Args:
            name: Unique dataset name, e.g. 'aws-agent-eval'
            description: What this dataset tests
        """
        try:
            from .agent.models import AgentDatasetMeta
            from .agent.storage import agent_dataset_path, save_agent_dataset_meta

            if agent_dataset_path(name).exists():
                return f"Error: Agent dataset '{name}' already exists."

            meta = AgentDatasetMeta(name=name, description=description)
            save_agent_dataset_meta(meta)
            return (
                f"Created agent dataset '{name}'\n"
                f"  Next: add test cases with agent_add_cases() or\n"
                f"  generate cases with agent_generate_cases()"
            )
        except Exception as e:
            return _fmt_error(e)

    @mcp.tool()
    def agent_run_eval(
        dataset: str,
        runner: str,
        judge_model: str = "",
        compare_baseline_flag: bool = False,
        update_baseline_flag: bool = False,
    ) -> str:
        """Run an agent evaluation. Requires a custom AgentRunner implementation.

        Args:
            dataset: Agent dataset name
            runner: Python import path to your AgentRunner class,
                    e.g. 'myapp.agents:MyAgent'
            judge_model: Optional Bedrock model ID for LLM-as-judge reasoning scoring,
                         e.g. 'anthropic.claude-3-5-sonnet-20241022-v2:0'
            compare_baseline_flag: Compare results against saved baseline
            update_baseline_flag: Save results as new baseline
        """
        try:
            from .agent.runner import load_runner, run_agent_eval
            from .agent.storage import load_agent_dataset_meta, load_agent_cases
            from .agent.judge import BedrockJudge
            from .agent.baseline import compare_to_agent_baseline, save_agent_baseline
            from .store import record_agent_run

            meta = load_agent_dataset_meta(dataset)
            cases = load_agent_cases(dataset)
            if not cases:
                return f"No cases found in agent dataset '{dataset}'."

            agent_runner = load_runner(runner)
            judge = BedrockJudge(model_id=judge_model) if judge_model else None

            result = run_agent_eval(
                dataset_name=dataset,
                cases=cases,
                runner=agent_runner,
                judge=judge,
            )
            record_agent_run(result)

            lines = [
                f"Agent eval complete: {dataset}",
                f"  Run ID:    {result.run_id}",
                f"  Runner:    {runner}",
                f"  Passed:    {result.passed}/{result.total}",
                f"  Pass rate: {_pct(result.pass_rate)}",
            ]
            if result.avg_latency_ms:
                lines.append(f"  Latency:   avg {result.avg_latency_ms:.0f}ms")

            failed = [c for c in result.cases if not c.passed]
            if failed:
                lines.append(f"\n  Failed cases ({len(failed)}):")
                for c in failed[:5]:
                    lines.append(f"    ✗  [{c.case_id}] composite={c.score.composite_score:.2f}")

            if compare_baseline_flag:
                report = compare_to_agent_baseline(result)
                if report is None:
                    lines.append("\n  No baseline — run with update_baseline_flag=True to set one.")
                elif report.passed:
                    lines.append(f"\n  Baseline comparison: PASSED")
                else:
                    lines.append(f"\n  Baseline comparison: FAILED")
                    for reason in report.failure_reasons:
                        lines.append(f"    • {reason}")

            if update_baseline_flag:
                save_agent_baseline(result)
                lines.append(f"\n  Baseline updated.")

            return "\n".join(lines)
        except FileNotFoundError as e:
            return _fmt_error(e)
        except Exception as e:
            return f"Error running agent eval: {e}\n{traceback.format_exc()}"

    # -----------------------------------------------------------------------
    # Internal helpers (not MCP tools)
    # -----------------------------------------------------------------------

    return mcp


def _count_cases(dataset_name: str) -> int:
    try:
        from .storage import dataset_path
        p = dataset_path(dataset_name) / "cases.jsonl"
        if not p.exists():
            return 0
        return sum(1 for line in p.read_text().splitlines() if line.strip())
    except Exception:
        return 0


def _count_agent_cases(dataset_name: str) -> int:
    try:
        from .agent.storage import agent_dataset_path
        p = agent_dataset_path(dataset_name) / "cases.jsonl"
        if not p.exists():
            return 0
        return sum(1 for line in p.read_text().splitlines() if line.strip())
    except Exception:
        return 0


def run():
    """Entry point: start the MCP server (stdio transport)."""
    mcp = create_server()
    mcp.run()


if __name__ == "__main__":
    run()
