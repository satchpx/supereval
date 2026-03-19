"""
CLI commands for supereval agent eval.

Registered under the 'agent' sub-app in supereval/cli.py.

Commands
--------
supereval agent dataset create / list / show / add-cases / validate
supereval agent run
supereval agent history list / show / stats
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from .baseline import (
    compare_agent_to_baseline,
    load_agent_baseline,
    save_agent_baseline,
)
from .models import AgentDatasetMeta, AgentTestCase, ToolSpec
from .runner import load_runner, run_agent_eval
from .generator import (
    AgentAnthropicGenerator,
    AgentBedrockGenerator,
    AgentOpenAIGenerator,
    GeneratedAgentCase,
)
from .trace_importer import parse_traces, trace_to_case
from .storage import (
    agent_dataset_path,
    agent_datasets_dir,
    append_agent_cases,
    list_agent_datasets,
    load_agent_cases,
    load_agent_dataset_meta,
    save_agent_dataset_meta,
)

agent_app = typer.Typer(
    help="Agent evaluation — dataset management and eval runs",
    no_args_is_help=True,
)
agent_dataset_app = typer.Typer(
    help="Manage agent evaluation datasets",
    no_args_is_help=True,
)
agent_history_app = typer.Typer(
    help="Browse agent run history and cost/latency stats",
    no_args_is_help=True,
)
agent_app.add_typer(agent_dataset_app, name="dataset")
agent_app.add_typer(agent_history_app, name="history")

console = Console()


# ---------------------------------------------------------------------------
# dataset sub-commands
# ---------------------------------------------------------------------------

@agent_dataset_app.command("create")
def agent_dataset_create(
    name: str = typer.Argument(..., help="Dataset name"),
    tool_specs: Optional[Path] = typer.Option(
        None, "--tool-specs",
        help="JSON file containing a list of ToolSpec objects "
             "(name, description, parameters). Can be added later.",
    ),
    description: str = typer.Option("", "--description", "-d"),
    tags: Optional[str] = typer.Option(None, "--tags", help="Comma-separated tags"),
    author: str = typer.Option("", "--author"),
):
    """Create a new empty agent evaluation dataset."""
    if agent_dataset_path(name).exists():
        rprint(f"[red]Agent dataset '{name}' already exists.[/red]")
        raise typer.Exit(1)

    tools: list[ToolSpec] = []
    if tool_specs:
        if not tool_specs.exists():
            rprint(f"[red]Tool specs file not found: {tool_specs}[/red]")
            raise typer.Exit(1)
        try:
            raw = json.loads(tool_specs.read_text())
            tools = [ToolSpec.model_validate(t) for t in raw]
        except Exception as e:
            rprint(f"[red]Failed to parse tool specs: {e}[/red]")
            raise typer.Exit(1)

    meta = AgentDatasetMeta(
        name=name,
        description=description,
        tags=[t.strip() for t in tags.split(",")] if tags else [],
        tools=tools,
        author=author,
    )
    save_agent_dataset_meta(meta)
    rprint(
        f"[green]Created agent dataset '{name}' at "
        f"{agent_dataset_path(name)}[/green]"
    )
    if not tools:
        rprint(
            "[yellow]Tip: add tool definitions with --tool-specs tools.json[/yellow]"
        )


@agent_dataset_app.command("list")
def agent_dataset_list():
    """List all agent datasets."""
    datasets = list_agent_datasets()
    if not datasets:
        rprint(
            f"[yellow]No agent datasets found in {agent_datasets_dir()}[/yellow]"
        )
        rprint("Run [bold]supereval agent dataset create[/bold] to get started.")
        raise typer.Exit()

    table = Table(title=f"Agent datasets  ({agent_datasets_dir()})")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Tools", justify="right")
    table.add_column("Cases", justify="right")
    table.add_column("Tags")
    table.add_column("Description")

    for ds in datasets:
        try:
            count = str(len(load_agent_cases(ds.name)))
        except Exception:
            count = "[red]?[/red]"
        table.add_row(
            ds.name,
            str(len(ds.tools)),
            count,
            ", ".join(ds.tags) or "—",
            ds.description or "—",
        )
    console.print(table)


@agent_dataset_app.command("show")
def agent_dataset_show(
    name: str = typer.Argument(..., help="Dataset name"),
    limit: int = typer.Option(5, "--limit", "-n", help="Number of cases to preview"),
):
    """Show details and a preview of cases for an agent dataset."""
    meta = load_agent_dataset_meta(name)
    cases = load_agent_cases(name)

    rprint(f"\n[bold cyan]{meta.name}[/bold cyan]  [dim]{meta.id}[/dim]")
    rprint(f"  Description: {meta.description or '—'}")
    rprint(f"  Tags:        {', '.join(meta.tags) or '—'}")
    rprint(f"  Cases:       {len(cases)}")
    rprint(f"  Author:      {meta.author or '—'}")

    if meta.tools:
        rprint(f"\n[bold]Tools ({len(meta.tools)}):[/bold]")
        for t in meta.tools:
            rprint(f"  [cyan]{t.name}[/cyan] — {t.description}")

    t = meta.thresholds
    rprint(f"\n[bold]Thresholds:[/bold]")
    rprint(f"  pass_rate:           {t.pass_rate:.0%}")
    rprint(f"  min_answer_score:    {t.min_answer_score:.2f}")
    rprint(f"  min_tool_score:      {t.min_tool_score:.2f}")
    rprint(f"  min_reasoning_score: {t.min_reasoning_score or '—'}")
    rprint(f"  fail_on_regression:  {t.fail_on_regression}")

    if not cases:
        rprint("\n[yellow]No cases yet. Add with: supereval agent dataset add-cases[/yellow]")
        return

    rprint(f"\n[bold]Preview ({min(limit, len(cases))} of {len(cases)} cases):[/bold]")
    for case in cases[:limit]:
        rprint(
            f"\n  [cyan]{case.id}[/cyan]"
            + (f"  {case.description}" if case.description else "")
        )
        rprint(f"  task:         {case.input.task}")
        rprint(f"  tools mocked: {', '.join(case.tools.keys()) or '—'}")
        e = case.expected
        if e.answer:
            rprint(f"  expected:     {e.answer!r}  ({e.answer_match.value})")
        if e.must_call:
            rprint(f"  must_call:    {e.must_call}")
        if e.must_not_call:
            rprint(f"  must_not_call:{e.must_not_call}")


@agent_dataset_app.command("add-cases")
def agent_dataset_add_cases(
    name: str = typer.Argument(..., help="Dataset name"),
    from_file: Path = typer.Option(..., "--from", help="Path to a JSONL file of AgentTestCase objects"),
):
    """Add test cases from a JSONL file."""
    load_agent_dataset_meta(name)  # validates dataset exists

    if not from_file.exists():
        rprint(f"[red]File not found: {from_file}[/red]")
        raise typer.Exit(1)

    raw_lines = [l.strip() for l in from_file.read_text().splitlines() if l.strip()]
    if not raw_lines:
        rprint("[yellow]File is empty — nothing to add.[/yellow]")
        raise typer.Exit()

    cases, errors = [], []
    for i, line in enumerate(raw_lines, 1):
        try:
            cases.append(AgentTestCase.model_validate_json(line))
        except Exception as e:
            errors.append(f"  Line {i}: {e}")

    if errors:
        rprint(f"[red]{len(errors)} validation error(s) — no cases added:[/red]")
        for err in errors:
            rprint(err)
        raise typer.Exit(1)

    append_agent_cases(name, cases)
    rprint(f"[green]Added {len(cases)} case(s) to '{name}'[/green]")


@agent_dataset_app.command("validate")
def agent_dataset_validate(
    name: str = typer.Argument(..., help="Dataset name"),
):
    """Validate all cases in an agent dataset against its schema."""
    load_agent_dataset_meta(name)
    try:
        cases = load_agent_cases(name)
        rprint(f"[green]OK — {len(cases)} valid case(s) in '{name}'[/green]")
    except ValueError as e:
        rprint(f"[red]{e}[/red]")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# generate command (from traces)
# ---------------------------------------------------------------------------

def _agent_interactive_review(cases: list[AgentTestCase]) -> list[AgentTestCase]:
    """Walk through generated agent cases one-by-one for review.

    Actions: [k]eep / [e]dit (opens $EDITOR as JSON) / [s]kip / [q]uit
    Returns the list of approved AgentTestCase objects.
    """
    import click
    from rich.panel import Panel
    from rich.syntax import Syntax

    approved: list[AgentTestCase] = []
    total = len(cases)

    for i, case in enumerate(cases, 1):
        rprint(
            f"\n[bold]Case {i}/{total}[/bold]"
            + (f"  [dim]{case.difficulty}[/dim]" if case.difficulty else "")
        )
        case_dict = json.loads(case.model_dump_json())
        rprint(Panel(
            Syntax(json.dumps(case_dict, indent=2), "json", theme="ansi_dark"),
            title=f"[cyan]{case.description or case.input.task[:60]}[/cyan]",
            expand=False,
        ))

        while True:
            raw = typer.prompt("[k]eep  [e]dit  [s]kip  [q]uit", default="k").strip().lower()
            if raw in ("k", "keep", ""):
                approved.append(case)
                break
            elif raw in ("e", "edit"):
                edited_text = click.edit(
                    json.dumps(case_dict, indent=2),
                    extension=".json",
                )
                if edited_text is None:
                    rprint("[yellow]No changes — keeping original.[/yellow]")
                    approved.append(case)
                else:
                    try:
                        approved.append(AgentTestCase.model_validate_json(edited_text))
                        rprint("[green]Updated.[/green]")
                    except Exception as exc:
                        rprint(f"[red]Validation error: {exc} — keeping original.[/red]")
                        approved.append(case)
                break
            elif raw in ("s", "skip"):
                rprint("[yellow]Skipped.[/yellow]")
                break
            elif raw in ("q", "quit"):
                rprint(
                    f"[yellow]Quit at case {i}/{total}. "
                    f"Keeping {len(approved)} approved so far.[/yellow]"
                )
                return approved
            else:
                rprint("[red]Enter k, e, s, or q.[/red]")

    return approved


@agent_app.command("generate")
def agent_generate(
    dataset: str = typer.Argument(..., help="Agent dataset name to generate cases for"),
    from_path: str = typer.Option(
        ..., "--from",
        help="Source to generate from: a JSONL traces file, a local file/directory, "
             "or an s3://bucket/prefix URI.",
    ),
    count: int = typer.Option(
        20, "--count", "-n",
        help="Number of cases to generate (document mode only; ignored for traces).",
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Staging file path (default: <dataset>_staged.jsonl).",
    ),
    backend: str = typer.Option(
        "bedrock", "--backend",
        help="Generator backend for document mode: bedrock (default), anthropic, "
             "openai, or azure-openai. Ignored when --from is a .jsonl traces file.",
    ),
    model: Optional[str] = typer.Option(
        None, "--model",
        help="Model ID for the selected backend (defaults per backend).",
    ),
    region: str = typer.Option(
        "us-east-1", "--region",
        help="AWS region (Bedrock only).",
    ),
    azure_endpoint: Optional[str] = typer.Option(
        None, "--azure-endpoint",
        help="Azure OpenAI endpoint URL (azure-openai backend only). "
             "Can also be set via AZURE_OPENAI_ENDPOINT env var.",
    ),
    auto_import: bool = typer.Option(
        False, "--auto-import",
        help="Import generated cases directly without staging.",
    ),
    interactive: bool = typer.Option(
        False, "--interactive",
        help="Review each case interactively before staging or importing. "
             "[k]eep, [e]dit (opens $EDITOR), [s]kip, or [q]uit.",
    ),
):
    """
    Generate agent test cases — from execution traces or from documents.

    \b
    Mode 1 — From traces (.jsonl file):
      supereval agent generate my-dataset --from production_traces.jsonl

    \b
    Mode 2 — From documents (LLM generates task scenarios using the dataset's tool catalog):
      supereval agent generate my-dataset --from docs/ --count 20
      supereval agent generate my-dataset --from docs/ --backend anthropic --count 10
      supereval agent generate my-dataset --from s3://bucket/docs/ --count 20

    \b
    In document mode, the dataset's tool catalog drives what tools the LLM includes
    in must_call. The tools block (mock responses) is left empty — add mock responses
    from traces or manually before running an eval.

    \b
    Both modes support --auto-import (skip staging) and --interactive (review each case).
    """
    import math as _math
    meta = load_agent_dataset_meta(dataset)

    from_path_obj = Path(from_path)

    # -----------------------------------------------------------------------
    # Mode detection: .jsonl file → traces; anything else → document generation
    # -----------------------------------------------------------------------
    if from_path_obj.suffix.lower() == ".jsonl" and not from_path.startswith("s3://"):
        # ---- Trace import mode ----
        if not from_path_obj.exists():
            rprint(f"[red]File not found: {from_path}[/red]")
            raise typer.Exit(1)

        rprint(f"\n[bold]Parsing traces from:[/bold] {from_path}")
        try:
            traces = parse_traces(from_path_obj)
        except ValueError as exc:
            rprint(f"[red]{exc}[/red]")
            raise typer.Exit(1)

        if not traces:
            rprint("[yellow]No traces found in file.[/yellow]")
            raise typer.Exit(0)

        rprint(f"  Parsed {len(traces)} trace(s)")
        cases: list[AgentTestCase] = [trace_to_case(t) for t in traces]
        rprint(f"\n[bold]Generated {len(cases)} case(s) from traces[/bold]")

    else:
        # ---- Document generation mode ----
        from ..sources import LocalFileSource, S3Source, URLSource

        rprint(f"\n[bold]Loading documents from:[/bold] {from_path}")
        try:
            if from_path.startswith("s3://"):
                source = S3Source(from_path, region=region)
            elif from_path.startswith(("http://", "https://")):
                source = URLSource(from_path)
            else:
                local = Path(from_path)
                if not local.exists():
                    rprint(f"[red]Path not found: {from_path}[/red]")
                    raise typer.Exit(1)
                source = LocalFileSource(local)
            documents = source.load()
        except (FileNotFoundError, ValueError, ImportError) as exc:
            rprint(f"[red]{exc}[/red]")
            raise typer.Exit(1)

        rprint(f"  Loaded {len(documents)} document(s): "
               f"{', '.join(d.filename for d in documents)}")

        if meta.tools:
            rprint(f"  Tool catalog: {', '.join(t.name for t in meta.tools)}")
        else:
            rprint(
                "[yellow]  Warning: no tools defined on this dataset. "
                "Add tools with `supereval agent dataset create --tool-specs tools.json` "
                "or the generated tasks may not reference specific tools.[/yellow]"
            )

        # Resolve backend and model
        from ..generator import DEFAULT_MODEL, ANTHROPIC_DEFAULT_MODEL, OPENAI_DEFAULT_MODEL
        _model_defaults = {
            "bedrock": DEFAULT_MODEL,
            "anthropic": ANTHROPIC_DEFAULT_MODEL,
            "openai": OPENAI_DEFAULT_MODEL,
            "azure-openai": OPENAI_DEFAULT_MODEL,
        }
        if backend not in _model_defaults:
            rprint(
                f"[red]Unknown backend '{backend}'. "
                "Choose bedrock, anthropic, openai, or azure-openai.[/red]"
            )
            raise typer.Exit(1)

        resolved_model = model or _model_defaults[backend]
        rprint(f"\n[bold]Generating {count} case(s)[/bold] via {backend} ({resolved_model})")

        if backend == "bedrock":
            generator = AgentBedrockGenerator(model_id=resolved_model, region=region)
        elif backend == "anthropic":
            generator = AgentAnthropicGenerator(model_id=resolved_model)
        elif backend == "azure-openai":
            import os
            endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
            if not endpoint:
                rprint(
                    "[red]Azure OpenAI requires an endpoint. "
                    "Pass --azure-endpoint or set AZURE_OPENAI_ENDPOINT.[/red]"
                )
                raise typer.Exit(1)
            generator = AgentOpenAIGenerator(model_id=resolved_model, azure_endpoint=endpoint)
        else:
            generator = AgentOpenAIGenerator(model_id=resolved_model)

        generated: list[GeneratedAgentCase] = []
        per_doc = max(1, _math.ceil(count / len(documents)))
        for doc in documents:
            doc_count = min(per_doc, count - len(generated))
            if doc_count <= 0:
                break
            rprint(f"  Generating {doc_count} case(s) from [cyan]{doc.filename}[/cyan]...")
            try:
                batch = generator.generate(doc, doc_count, tools=meta.tools)
                generated.extend(batch)
                rprint(f"    [green]Got {len(batch)} case(s)[/green]")
            except Exception as exc:
                rprint(f"    [red]Failed: {exc}[/red]")
                raise typer.Exit(1)

        generated = generated[:count]
        rprint(f"\n[bold]Generated {len(generated)} case(s) total[/bold]")
        cases = [g.to_agent_test_case() for g in generated]

    # -----------------------------------------------------------------------
    # Interactive review (shared between both modes)
    # -----------------------------------------------------------------------
    if interactive:
        rprint(
            f"\n[bold]Interactive review[/bold] — {len(cases)} case(s). "
            "Edit opens $EDITOR (or nano/vi)."
        )
        cases = _agent_interactive_review(cases)
        rprint(f"[bold]Review complete:[/bold] {len(cases)} case(s) approved")
        if not cases:
            rprint("[yellow]No cases approved — nothing to import or stage.[/yellow]")
            raise typer.Exit(0)

    # -----------------------------------------------------------------------
    # Stage or import
    # -----------------------------------------------------------------------
    if auto_import:
        append_agent_cases(dataset, cases)
        rprint(f"[green]Imported {len(cases)} case(s) into '{dataset}'[/green]")
    else:
        staging_path = output or Path(f"{dataset}_staged.jsonl")
        staging_path.write_text("\n".join(c.model_dump_json() for c in cases) + "\n")
        rprint(f"\n[green]Staged to:[/green] {staging_path}")
        rprint("\nNext steps:")
        rprint(f"  1. Review [cyan]{staging_path}[/cyan]")
        rprint(f"  2. Add mock tool responses to each case's 'tools' block")
        rprint(
            f"  3. Import: [bold]supereval agent dataset add-cases "
            f"{dataset} --from {staging_path}[/bold]"
        )


# ---------------------------------------------------------------------------
# run command
# ---------------------------------------------------------------------------

@agent_app.command("run")
def agent_run(
    dataset: str = typer.Argument(..., help="Agent dataset name"),
    runner_path: str = typer.Option(
        ..., "--runner", "-r",
        help="Python import path to AgentRunner implementation, e.g. myapp.agents:MyAgent",
    ),
    judge_model: Optional[str] = typer.Option(
        None, "--judge-model",
        help="Bedrock model ID for LLM-as-judge answer/reasoning scoring. "
             "Required for answer_match=llm_judge or min_reasoning_score.",
    ),
    judge_region: str = typer.Option("us-east-1", "--judge-region"),
    compare_baseline: bool = typer.Option(
        False, "--compare-baseline",
        help="Compare results against the stored baseline and fail on regression.",
    ),
    update_baseline: bool = typer.Option(
        False, "--update-baseline",
        help="Save the current run results as the new baseline.",
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Save full run results as JSON to this path.",
    ),
    no_record: bool = typer.Option(
        False, "--no-record",
        help="Do not persist this run to the history store.",
    ),
):
    """
    Run an agent eval against a dataset.

    \b
    The --runner flag accepts a Python import path to your AgentRunner:
      supereval agent run my-dataset --runner myapp.agents:ReActAgent

    \b
    See examples/ for reference implementations.
    """
    rprint(f"\n[bold]Running agent eval:[/bold] {dataset}")
    rprint(f"  Runner:  {runner_path}")
    if judge_model:
        rprint(f"  Judge:   {judge_model} ({judge_region})")

    # Load runner
    try:
        runner = load_runner(runner_path)
    except (ValueError, ImportError, AttributeError, TypeError) as e:
        rprint(f"\n[red]Failed to load runner: {e}[/red]")
        raise typer.Exit(1)

    # Run eval
    try:
        result = run_agent_eval(
            dataset_name=dataset,
            runner=runner,
            runner_id=runner_path,
            judge_model=judge_model,
            judge_region=judge_region,
        )
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        rprint(f"\n[red]Error: {e}[/red]")
        raise typer.Exit(1)

    # Persist to history store
    if not no_record:
        try:
            from ..store import record_agent_run
            record_agent_run(result)
        except Exception as e:
            rprint(
                f"[yellow]Warning: could not persist run to history store: {e}[/yellow]"
            )

    # Results summary
    rprint(f"\n[bold]Results[/bold]  (run {result.run_id})")
    color = "green" if result.failed == 0 else "red"
    rprint(f"  Passed:            [{color}]{result.passed}/{result.total}[/{color}]")
    rprint(f"  Pass rate:         [{color}]{result.pass_rate:.1%}[/{color}]")
    rprint(f"  Avg composite:     {result.avg_composite_score:.3f}")
    rprint(f"  Avg answer score:  {result.avg_answer_score:.3f}")
    rprint(f"  Avg tool score:    {result.avg_tool_score:.3f}")
    if result.total_cost_usd > 0:
        rprint(f"  Cost:              ${result.total_cost_usd:.4f}")
    if result.avg_latency_ms > 0:
        rprint(
            f"  Latency:           avg {result.avg_latency_ms:.0f}ms  "
            f"p95 {result.p95_latency_ms:.0f}ms"
        )

    if result.failed > 0:
        rprint("\n  [bold]Failed cases:[/bold]")
        for case in result.cases:
            if not case.passed:
                rprint(f"    [red]✗[/red]  [{case.case_id}] {case.description or case.vars.get('task', '')}")
                for reason in case.score.failure_reasons:
                    rprint(f"         • {reason}")

    # Baseline comparison
    exit_code = 0
    if compare_baseline:
        report = compare_agent_to_baseline(result)
        if report is None:
            rprint(
                "\n[yellow]No baseline found — run with --update-baseline to create one.[/yellow]"
            )
        else:
            sign = "+" if report.pass_rate_delta >= 0 else ""
            rprint(
                f"\n[bold]Baseline comparison[/bold]  "
                f"(baseline: {report.baseline_pass_rate:.1%}  "
                f"current: {report.current_pass_rate:.1%}  "
                f"delta: {sign}{report.pass_rate_delta:.1%})"
            )
            if report.passed:
                rprint("  [green]PASSED — no regressions detected[/green]")
            else:
                rprint("  [red]FAILED — regressions detected:[/red]")
                for reason in report.failure_reasons:
                    rprint(f"    [red]•[/red] {reason}")
                if report.regressions:
                    rprint("\n  [bold]Regressed cases:[/bold]")
                    for reg in report.regressions:
                        rprint(
                            f"    [red]✗[/red]  {reg['case_id']}  "
                            f"(score: {reg['baseline_score']:.2f} → {reg['current_score']:.2f})"
                        )
                exit_code = 1

    if update_baseline:
        save_agent_baseline(result)
        rprint(f"\n[green]Baseline updated for '{dataset}'[/green]")

    if output:
        output.write_text(json.dumps(result.to_dict(), indent=2))
        rprint(f"[green]Results saved to {output}[/green]")

    raise typer.Exit(exit_code)


# ---------------------------------------------------------------------------
# history sub-commands
# ---------------------------------------------------------------------------

@agent_history_app.command("list")
def agent_history_list(
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d"),
    limit: int = typer.Option(20, "--limit", "-n"),
):
    """List recent agent eval runs."""
    from ..store import list_runs
    runs = list_runs(dataset=dataset, limit=limit, run_type="agent")
    if not runs:
        rprint("[yellow]No agent runs recorded yet. Run `supereval agent run` to start.[/yellow]")
        raise typer.Exit()

    table = Table(title="Agent run history")
    table.add_column("Run ID", style="cyan", no_wrap=True)
    table.add_column("Dataset", style="magenta")
    table.add_column("Ran at", no_wrap=True)
    table.add_column("Runner")
    table.add_column("Cases", justify="right")
    table.add_column("Pass rate", justify="right")
    table.add_column("Cost ($)", justify="right")
    table.add_column("p95 (ms)", justify="right")

    for r in runs:
        pr = r["pass_rate"]
        color = "green" if pr >= 1.0 else ("yellow" if pr >= 0.8 else "red")
        providers = json.loads(r["providers"]) if isinstance(r["providers"], str) else r["providers"]
        runner_label = providers[0] if providers else "—"
        table.add_row(
            r["run_id"],
            r["dataset"],
            r["ran_at"][:19].replace("T", " "),
            runner_label,
            str(r["total_cases"]),
            f"[{color}]{pr:.1%}[/{color}]",
            f"{r['total_cost_usd']:.4f}" if r["total_cost_usd"] else "—",
            f"{r['p95_latency_ms']:.0f}" if r["p95_latency_ms"] else "—",
        )
    console.print(table)


@agent_history_app.command("show")
def agent_history_show(
    run_id: str = typer.Argument(..., help="Run ID to inspect"),
    steps: bool = typer.Option(False, "--steps", help="Show trajectory steps for each case"),
):
    """Show details for a specific agent run."""
    from ..store import get_run, get_run_cases
    run = get_run(run_id)
    if run is None:
        rprint(f"[red]Run '{run_id}' not found.[/red]")
        raise typer.Exit(1)

    providers = json.loads(run["providers"]) if isinstance(run["providers"], str) else run["providers"]
    pr = run["pass_rate"]
    color = "green" if pr >= 1.0 else ("yellow" if pr >= 0.8 else "red")

    rprint(f"\n[bold cyan]{run['run_id']}[/bold cyan]")
    rprint(f"  Dataset:   {run['dataset']}")
    rprint(f"  Ran at:    {run['ran_at'][:19].replace('T', ' ')} UTC")
    rprint(f"  Runner:    {providers[0] if providers else '—'}")
    rprint(f"  Cases:     {run['total_cases']}  (passed: {run['passed']}  failed: {run['failed']})")
    rprint(f"  Pass rate: [{color}]{pr:.1%}[/{color}]")
    if run["total_cost_usd"]:
        rprint(f"  Cost:      ${run['total_cost_usd']:.4f}")
    if run["avg_latency_ms"]:
        rprint(
            f"  Latency:   avg {run['avg_latency_ms']:.0f}ms  "
            f"p95 {run['p95_latency_ms']:.0f}ms"
        )

    case_rows = get_run_cases(run_id)
    if case_rows:
        rprint(f"\n[bold]Case results ({len(case_rows)}):[/bold]")
        for c in case_rows:
            icon = "[green]✓[/green]" if c["passed"] else "[red]✗[/red]"
            rprint(
                f"  {icon}  {c['vars'].get('task', c['vars'])}  "
                f"score={c['score']:.3f}  {c['latency_ms']}ms"
            )


@agent_history_app.command("stats")
def agent_history_stats(
    dataset: str = typer.Argument(..., help="Dataset name"),
    last: int = typer.Option(10, "--last", "-n"),
):
    """Show aggregate stats for an agent dataset."""
    from ..store import get_stats
    stats = get_stats(dataset=dataset, last=last)
    if not stats or not stats.get("run_count"):
        rprint(f"[yellow]No agent runs found for dataset '{dataset}'.[/yellow]")
        raise typer.Exit()

    rprint(f"\n[bold]Stats for '{dataset}'[/bold]  (last {stats['run_count']} run(s))")
    rprint(
        f"  Pass rate:  avg {stats['avg_pass_rate']:.1%}  "
        f"min {stats['min_pass_rate']:.1%}  "
        f"max {stats['max_pass_rate']:.1%}"
    )
    if stats["total_cost_usd"]:
        rprint(
            f"  Cost:       total ${stats['total_cost_usd']:.4f}  "
            f"avg/run ${stats['avg_cost_per_run']:.4f}"
        )
    if stats["avg_latency_ms"]:
        rprint(
            f"  Latency:    avg {stats['avg_latency_ms']:.0f}ms  "
            f"max p95 {stats['max_p95_latency_ms']:.0f}ms"
        )
