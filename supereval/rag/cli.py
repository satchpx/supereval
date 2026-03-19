"""
CLI commands for supereval RAG eval.

Registered under the 'rag' sub-app in supereval/cli.py.

Commands
--------
supereval rag dataset create / list / show / add-cases / validate
supereval rag run
supereval rag history list / show / stats
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
    compare_rag_to_baseline,
    load_rag_baseline,
    save_rag_baseline,
)
from .models import RagDatasetMeta, RagTestCase
from .runner import DEFAULT_PROMPT_TEMPLATE, run_rag_eval
from .storage import (
    append_rag_cases,
    list_rag_datasets,
    load_rag_cases,
    load_rag_dataset_meta,
    rag_dataset_path,
    rag_datasets_dir,
    save_rag_dataset_meta,
)

rag_app = typer.Typer(
    help="RAG evaluation — dataset management and eval runs",
    no_args_is_help=True,
)
rag_dataset_app = typer.Typer(
    help="Manage RAG evaluation datasets",
    no_args_is_help=True,
)
rag_history_app = typer.Typer(
    help="Browse RAG run history and cost/latency stats",
    no_args_is_help=True,
)
rag_app.add_typer(rag_dataset_app, name="dataset")
rag_app.add_typer(rag_history_app, name="history")

console = Console()


# ---------------------------------------------------------------------------
# dataset sub-commands
# ---------------------------------------------------------------------------

@rag_dataset_app.command("create")
def rag_dataset_create(
    name: str = typer.Argument(..., help="Dataset name"),
    description: str = typer.Option("", "--description", "-d"),
    tags: Optional[str] = typer.Option(None, "--tags", help="Comma-separated tags"),
    author: str = typer.Option("", "--author"),
):
    """Create a new empty RAG evaluation dataset."""
    if rag_dataset_path(name).exists():
        rprint(f"[red]RAG dataset '{name}' already exists.[/red]")
        raise typer.Exit(1)

    meta = RagDatasetMeta(
        name=name,
        description=description,
        tags=[t.strip() for t in tags.split(",")] if tags else [],
        author=author,
    )
    save_rag_dataset_meta(meta)
    rprint(
        f"[green]Created RAG dataset '{name}' at "
        f"{rag_dataset_path(name)}[/green]"
    )
    rprint(
        "[dim]Add cases with: supereval rag dataset add-cases "
        f"{name} --from cases.jsonl[/dim]"
    )


@rag_dataset_app.command("list")
def rag_dataset_list():
    """List all RAG datasets."""
    datasets = list_rag_datasets()
    if not datasets:
        rprint(
            f"[yellow]No RAG datasets found in {rag_datasets_dir()}[/yellow]"
        )
        rprint("Run [bold]supereval rag dataset create[/bold] to get started.")
        raise typer.Exit()

    table = Table(title=f"RAG datasets  ({rag_datasets_dir()})")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Cases", justify="right")
    table.add_column("Tags")
    table.add_column("Description")

    for ds in datasets:
        try:
            count = str(len(load_rag_cases(ds.name)))
        except Exception:
            count = "[red]?[/red]"
        table.add_row(
            ds.name,
            count,
            ", ".join(ds.tags) or "—",
            ds.description or "—",
        )
    console.print(table)


@rag_dataset_app.command("show")
def rag_dataset_show(
    name: str = typer.Argument(..., help="Dataset name"),
    limit: int = typer.Option(5, "--limit", "-n", help="Number of cases to preview"),
):
    """Show details and a preview of cases for a RAG dataset."""
    meta = load_rag_dataset_meta(name)
    cases = load_rag_cases(name)

    rprint(f"\n[bold cyan]{meta.name}[/bold cyan]  [dim]{meta.id}[/dim]")
    rprint(f"  Description: {meta.description or '—'}")
    rprint(f"  Tags:        {', '.join(meta.tags) or '—'}")
    rprint(f"  Cases:       {len(cases)}")
    rprint(f"  Author:      {meta.author or '—'}")

    t = meta.thresholds
    rprint(f"\n[bold]Thresholds:[/bold]")
    rprint(f"  pass_rate:                    {t.pass_rate:.0%}")
    rprint(f"  require_contains:             {t.require_contains}")
    rprint(f"  min_answer_correctness_score: {t.min_answer_correctness_score:.2f}")
    rprint(f"  min_faithfulness_score:       {t.min_faithfulness_score or '—'}")
    rprint(f"  fail_on_regression:           {t.fail_on_regression}")

    if not cases:
        rprint("\n[yellow]No cases yet. Add with: supereval rag dataset add-cases[/yellow]")
        return

    rprint(f"\n[bold]Preview ({min(limit, len(cases))} of {len(cases)} cases):[/bold]")
    for case in cases[:limit]:
        rprint(
            f"\n  [cyan]{case.id}[/cyan]"
            + (f"  {case.description}" if case.description else "")
        )
        rprint(f"  query:             {case.input.query}")
        rprint(f"  contexts:          {len(case.input.retrieved_contexts)} chunk(s)")
        rprint(f"  ground_truth:      {case.expected.ground_truth!r}")


@rag_dataset_app.command("add-cases")
def rag_dataset_add_cases(
    name: str = typer.Argument(..., help="Dataset name"),
    from_file: Path = typer.Option(..., "--from", help="Path to a JSONL file of RagTestCase objects"),
):
    """Add test cases from a JSONL file."""
    load_rag_dataset_meta(name)  # validates dataset exists

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
            cases.append(RagTestCase.model_validate_json(line))
        except Exception as e:
            errors.append(f"  Line {i}: {e}")

    if errors:
        rprint(f"[red]{len(errors)} validation error(s) — no cases added:[/red]")
        for err in errors:
            rprint(err)
        raise typer.Exit(1)

    append_rag_cases(name, cases)
    rprint(f"[green]Added {len(cases)} case(s) to '{name}'[/green]")


@rag_dataset_app.command("validate")
def rag_dataset_validate(
    name: str = typer.Argument(..., help="Dataset name"),
):
    """Validate all cases in a RAG dataset against its schema."""
    load_rag_dataset_meta(name)
    try:
        cases = load_rag_cases(name)
        rprint(f"[green]OK — {len(cases)} valid case(s) in '{name}'[/green]")
    except ValueError as e:
        rprint(f"[red]{e}[/red]")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# run command
# ---------------------------------------------------------------------------

@rag_app.command("run")
def rag_run(
    dataset: str = typer.Argument(..., help="RAG dataset name"),
    model: str = typer.Option(
        ..., "--model", "-m",
        help=(
            "Model to generate answers. Bedrock model ID by default. "
            "Prefix with 'anthropic:' for Anthropic API, 'openai:' for OpenAI. "
            "Examples: us.anthropic.claude-3-5-sonnet-20241022-v2:0  "
            "anthropic:claude-sonnet-4-6  openai:gpt-4o"
        ),
    ),
    region: str = typer.Option("us-east-1", "--region", help="AWS region (Bedrock only)"),
    prompt: Optional[str] = typer.Option(
        None, "--prompt",
        help=(
            "Custom prompt template. Must contain {query} and {contexts} placeholders. "
            "Defaults to the built-in RAG prompt."
        ),
    ),
    judge_model: Optional[str] = typer.Option(
        None, "--judge-model",
        help="Bedrock model ID for LLM-as-judge faithfulness + answer correctness scoring.",
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
    Run a RAG eval against a dataset.

    \b
    The model answers each test case using the pre-committed retrieved_contexts.
    Scoring: icontains check always runs; faithfulness + answer correctness
    require --judge-model.

    \b
    Examples:
      supereval rag run my-rag-dataset --model us.anthropic.claude-3-5-sonnet-20241022-v2:0
      supereval rag run my-rag-dataset --model anthropic:claude-sonnet-4-6 \\
        --judge-model anthropic.claude-3-5-sonnet-20241022-v2:0
    """
    rprint(f"\n[bold]Running RAG eval:[/bold] {dataset}")
    rprint(f"  Model:  {model}")
    if judge_model:
        rprint(f"  Judge:  {judge_model} ({judge_region})")
    else:
        rprint(
            "  [dim]No --judge-model set; scoring via icontains only. "
            "Add --judge-model for faithfulness + answer correctness.[/dim]"
        )

    try:
        result = run_rag_eval(
            dataset_name=dataset,
            model_id=model,
            region=region,
            prompt_template=prompt,
            judge_model=judge_model,
            judge_region=judge_region,
        )
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        rprint(f"\n[red]Error: {e}[/red]")
        raise typer.Exit(1)

    # Persist to history store
    if not no_record:
        try:
            from ..store import record_rag_run
            record_rag_run(result)
        except Exception as e:
            rprint(
                f"[yellow]Warning: could not persist run to history store: {e}[/yellow]"
            )

    # Results summary
    rprint(f"\n[bold]Results[/bold]  (run {result.run_id})")
    color = "green" if result.failed == 0 else "red"
    rprint(f"  Passed:          [{color}]{result.passed}/{result.total}[/{color}]")
    rprint(f"  Pass rate:       [{color}]{result.pass_rate:.1%}[/{color}]")
    rprint(f"  Avg composite:   {result.avg_composite_score:.3f}")
    if result.avg_latency_ms > 0:
        rprint(
            f"  Latency:         avg {result.avg_latency_ms:.0f}ms  "
            f"p95 {result.p95_latency_ms:.0f}ms"
        )

    if result.failed > 0:
        rprint("\n  [bold]Failed cases:[/bold]")
        for case in result.cases:
            if not case.passed:
                rprint(
                    f"    [red]✗[/red]  [{case.case_id}] "
                    f"{case.description or case.vars.get('query', '')}"
                )
                for reason in case.score.failure_reasons:
                    rprint(f"         • {reason}")

    # Baseline comparison
    exit_code = 0
    if compare_baseline:
        report = compare_rag_to_baseline(result)
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
        save_rag_baseline(result)
        rprint(f"\n[green]Baseline updated for '{dataset}'[/green]")

    if output:
        output.write_text(json.dumps(result.to_dict(), indent=2))
        rprint(f"[green]Results saved to {output}[/green]")

    raise typer.Exit(exit_code)


# ---------------------------------------------------------------------------
# history sub-commands
# ---------------------------------------------------------------------------

@rag_history_app.command("list")
def rag_history_list(
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d"),
    limit: int = typer.Option(20, "--limit", "-n"),
):
    """List recent RAG eval runs."""
    from ..store import list_runs
    runs = list_runs(dataset=dataset, limit=limit, run_type="rag")
    if not runs:
        rprint("[yellow]No RAG runs recorded yet. Run `supereval rag run` to start.[/yellow]")
        raise typer.Exit()

    table = Table(title="RAG run history")
    table.add_column("Run ID", style="cyan", no_wrap=True)
    table.add_column("Dataset", style="magenta")
    table.add_column("Ran at", no_wrap=True)
    table.add_column("Model")
    table.add_column("Cases", justify="right")
    table.add_column("Pass rate", justify="right")
    table.add_column("p95 (ms)", justify="right")

    for r in runs:
        pr = r["pass_rate"]
        color = "green" if pr >= 1.0 else ("yellow" if pr >= 0.8 else "red")
        providers = json.loads(r["providers"]) if isinstance(r["providers"], str) else r["providers"]
        model_label = providers[0] if providers else "—"
        table.add_row(
            r["run_id"],
            r["dataset"],
            r["ran_at"][:19].replace("T", " "),
            model_label,
            str(r["total_cases"]),
            f"[{color}]{pr:.1%}[/{color}]",
            f"{r['p95_latency_ms']:.0f}" if r["p95_latency_ms"] else "—",
        )
    console.print(table)


@rag_history_app.command("show")
def rag_history_show(
    run_id: str = typer.Argument(..., help="Run ID to inspect"),
    cases: bool = typer.Option(False, "--cases", help="Show per-case results"),
):
    """Show details for a specific RAG run."""
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
    rprint(f"  Model:     {providers[0] if providers else '—'}")
    rprint(f"  Cases:     {run['total_cases']}  (passed: {run['passed']}  failed: {run['failed']})")
    rprint(f"  Pass rate: [{color}]{pr:.1%}[/{color}]")
    if run["avg_latency_ms"]:
        rprint(
            f"  Latency:   avg {run['avg_latency_ms']:.0f}ms  "
            f"p95 {run['p95_latency_ms']:.0f}ms"
        )

    if cases:
        case_rows = get_run_cases(run_id)
        if case_rows:
            rprint(f"\n[bold]Case results ({len(case_rows)}):[/bold]")
            for c in case_rows:
                icon = "[green]✓[/green]" if c["passed"] else "[red]✗[/red]"
                rprint(
                    f"  {icon}  {c['vars'].get('query', c['vars'])}  "
                    f"score={c['score']:.3f}  {c['latency_ms']}ms"
                )


@rag_history_app.command("stats")
def rag_history_stats(
    dataset: str = typer.Argument(..., help="Dataset name"),
    last: int = typer.Option(10, "--last", "-n"),
):
    """Show aggregate stats for a RAG dataset."""
    from ..store import get_stats
    stats = get_stats(dataset=dataset, last=last)
    if not stats or not stats.get("run_count"):
        rprint(f"[yellow]No RAG runs found for dataset '{dataset}'.[/yellow]")
        raise typer.Exit()

    rprint(f"\n[bold]Stats for '{dataset}'[/bold]  (last {stats['run_count']} run(s))")
    rprint(
        f"  Pass rate:  avg {stats['avg_pass_rate']:.1%}  "
        f"min {stats['min_pass_rate']:.1%}  "
        f"max {stats['max_pass_rate']:.1%}"
    )
    if stats["avg_latency_ms"]:
        rprint(
            f"  Latency:    avg {stats['avg_latency_ms']:.0f}ms  "
            f"max p95 {stats['max_p95_latency_ms']:.0f}ms"
        )
