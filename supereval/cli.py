from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from .baseline import compare_to_baseline, save_baseline
from .export import export_to_promptfoo
from .generator import (
    AnthropicGenerator,
    ANTHROPIC_DEFAULT_MODEL,
    BedrockGenerator,
    DEFAULT_MODEL,
    DEFAULT_REGION,
    OpenAIGenerator,
    OPENAI_DEFAULT_MODEL,
)
from .models import DatasetMeta, DatasetType, TYPE_TO_MODEL
from .runner import run_eval, _vars_key
from .sources import LocalFileSource, S3Source, URLSource
from .storage import (
    append_cases,
    dataset_path,
    datasets_dir,
    list_datasets,
    load_cases,
    load_dataset_meta,
    save_dataset_meta,
)
from .store import get_run, get_run_cases, get_stats, list_runs, record_run
from .agent.cli import agent_app
from .rag.cli import rag_app

def _version_callback(value: bool):
    if value:
        from importlib.metadata import version
        typer.echo(f"supereval {version('supereval')}")
        raise typer.Exit()


app = typer.Typer(
    help="supereval — LLM and Agent evaluation dataset registry",
    no_args_is_help=True,
)


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
):
    pass
dataset_app = typer.Typer(help="Manage evaluation datasets", no_args_is_help=True)
version_app = typer.Typer(help="Tag and restore dataset versions", no_args_is_help=True)
dataset_app.add_typer(version_app, name="version")
history_app = typer.Typer(help="Browse run history and cost/latency stats", no_args_is_help=True)
providers_app = typer.Typer(help="List, explore, and test model providers", no_args_is_help=True)
app.add_typer(dataset_app, name="dataset")
app.add_typer(history_app, name="history")
app.add_typer(providers_app, name="providers")
app.add_typer(agent_app, name="agent")
app.add_typer(rag_app, name="rag")

console = Console()


def _short_provider(provider_id: str) -> str:
    """Shorten a provider ID for display in narrow table columns."""
    # bedrock:us.anthropic.claude-sonnet-4-6  -> claude-sonnet-4-6
    # bedrock:us.amazon.nova-pro-v1:0         -> nova-pro-v1:0
    # anthropic:claude-opus-4-6               -> claude-opus-4-6
    # Split only on the FIRST colon so trailing :0 in model IDs is preserved.
    if ":" in provider_id:
        _, name = provider_id.split(":", 1)
    else:
        name = provider_id
    # strip cross-region prefix (us. eu. ap.)
    if name.startswith(("us.", "eu.", "ap.")):
        name = name[3:]
    # strip vendor prefix (anthropic. amazon. meta.)
    for prefix in ("anthropic.", "amazon.", "meta.", "mistral."):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name[:28]


def _short_case_label(vars_: dict) -> str:
    """Extract a short display label from a test case's vars dict."""
    for key in ("instruction", "query", "text"):
        val = vars_.get(key, "")
        if val:
            # If it contains a 'Question:' line, use just that
            if "Question:" in val:
                val = val.split("Question:")[-1].strip()
            val = val.replace("\n", " ")
            return val[:60] + ("…" if len(val) > 60 else "")
    return str(vars_)[:60]


def _print_comparison_table(result, console: Console) -> None:
    """Print a side-by-side per-provider comparison table. Only shown for multi-provider runs."""
    if len(result.provider_results) < 2:
        return

    providers = list(result.provider_results.keys())
    short_names = [_short_provider(p) for p in providers]

    table = Table(
        title="Model Comparison",
        show_lines=True,
        title_style="bold",
    )
    table.add_column("Case", style="cyan", max_width=62, no_wrap=False)
    for name in short_names:
        table.add_column(name, justify="center", min_width=22)

    # One row per test case
    for key in result.case_keys_ordered:
        # Use vars from whichever provider has this key
        label = ""
        for pid in providers:
            cases_map = {_vars_key(c.vars): c for c in result.provider_results.get(pid, [])}
            if key in cases_map:
                label = _short_case_label(cases_map[key].vars)
                break

        row = [label]
        for pid in providers:
            cases_map = {_vars_key(c.vars): c for c in result.provider_results.get(pid, [])}
            c = cases_map.get(key)
            if c is None:
                row.append("—")
            else:
                icon = "[green]✓[/green]" if c.passed else "[red]✗[/red]"
                lat = f"{c.latency_ms}ms" if c.latency_ms else "—"
                row.append(f"{icon}  {c.score:.2f}  {lat}")
        table.add_row(*row)

    # Summary row
    summary = ["[bold]Summary[/bold]"]
    for pid in providers:
        cases = result.provider_results.get(pid, [])
        if not cases:
            summary.append("—")
            continue
        passed = sum(1 for c in cases if c.passed)
        pr = passed / len(cases)
        color = "green" if pr >= 0.8 else ("yellow" if pr >= 0.6 else "red")
        cost = sum(c.cost_usd for c in cases)
        avg_lat = sum(c.latency_ms for c in cases) / len(cases)
        cost_str = f"  ${cost:.4f}" if cost > 0 else ""
        summary.append(f"[{color}][bold]{pr:.0%}[/bold][/{color}]{cost_str}  avg {avg_lat:.0f}ms")
    table.add_row(*summary)

    console.print()
    console.print(table)
    console.print(
        "[dim]  score = rubric quality (0.0–1.0 as judged by LLM); "
        "✓/✗ = pass/fail against all assertions[/dim]"
    )


@dataset_app.command("create")
def dataset_create(
    name: str = typer.Argument(..., help="Dataset name (used as the directory name)"),
    type_: DatasetType = typer.Option(..., "--type", "-t", help="Dataset type: qa | classification | instruction"),
    description: str = typer.Option("", "--description", "-d", help="Short description"),
    tags: Optional[str] = typer.Option(None, "--tags", help="Comma-separated tags"),
    author: str = typer.Option("", "--author", help="Author name"),
    labels: Optional[str] = typer.Option(
        None, "--labels", help="Comma-separated valid labels (classification datasets only)"
    ),
):
    """Create a new empty dataset."""
    if dataset_path(name).exists():
        rprint(f"[red]Dataset '{name}' already exists.[/red]")
        raise typer.Exit(1)

    tag_list = [t.strip() for t in tags.split(",")] if tags else []
    label_list = [l.strip() for l in labels.split(",")] if labels else None

    if type_ == DatasetType.classification and not label_list:
        rprint("[yellow]Tip: use --labels to define valid labels for this classification dataset.[/yellow]")

    meta = DatasetMeta(
        name=name,
        type=type_,
        description=description,
        tags=tag_list,
        labels=label_list,
        author=author,
    )
    save_dataset_meta(meta)
    rprint(f"[green]Created dataset '{name}' (type: {type_.value}) at {dataset_path(name)}[/green]")


@dataset_app.command("list")
def dataset_list():
    """List all datasets."""
    datasets = list_datasets()

    if not datasets:
        rprint(f"[yellow]No datasets found in {datasets_dir()}[/yellow]")
        rprint("Run [bold]supereval dataset create[/bold] to get started.")
        raise typer.Exit()

    table = Table(title=f"Datasets  ({datasets_dir()})")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Type", style="magenta")
    table.add_column("Cases", justify="right")
    table.add_column("Tags")
    table.add_column("Description")

    for ds in datasets:
        try:
            count = str(len(load_cases(ds.name)))
        except Exception:
            count = "[red]?[/red]"
        table.add_row(
            ds.name,
            ds.type.value,
            count,
            ", ".join(ds.tags) or "—",
            ds.description or "—",
        )

    console.print(table)


@dataset_app.command("show")
def dataset_show(
    name: str = typer.Argument(..., help="Dataset name"),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of cases to preview"),
):
    """Show details and a preview of cases for a dataset."""
    meta = load_dataset_meta(name)
    cases = load_cases(name)

    rprint(f"\n[bold cyan]{meta.name}[/bold cyan]  [dim]{meta.id}[/dim]")
    rprint(f"  Type:        {meta.type.value}")
    rprint(f"  Description: {meta.description or '—'}")
    rprint(f"  Tags:        {', '.join(meta.tags) or '—'}")
    if meta.labels:
        rprint(f"  Labels:      {', '.join(meta.labels)}")
    rprint(f"  Cases:       {len(cases)}")
    rprint(f"  Created:     {meta.created_at}")
    rprint(f"  Author:      {meta.author or '—'}")

    if not cases:
        rprint("\n[yellow]No cases yet. Run `supereval dataset add-cases` to add some.[/yellow]")
        return

    preview = cases[:limit]
    rprint(f"\n[bold]Preview ({len(preview)} of {len(cases)} cases):[/bold]")
    for case in preview:
        rprint(f"\n  [cyan]{case.id}[/cyan]" + (f"  {case.description}" if case.description else ""))
        rprint(f"  input:    {case.input.model_dump()}")
        rprint(f"  expected: {case.expected.model_dump()}")
        if case.tags:
            rprint(f"  tags:     {', '.join(case.tags)}")
        if case.difficulty:
            rprint(f"  difficulty: {case.difficulty}")


@dataset_app.command("add-cases")
def dataset_add_cases(
    name: str = typer.Argument(..., help="Dataset name"),
    from_file: Path = typer.Option(..., "--from", help="Path to a JSONL file of test cases"),
):
    """Add test cases from a JSONL file."""
    meta = load_dataset_meta(name)
    model = TYPE_TO_MODEL[meta.type]

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
            cases.append(model.model_validate_json(line))
        except Exception as e:
            errors.append(f"  Line {i}: {e}")

    if errors:
        rprint(f"[red]{len(errors)} validation error(s) — no cases were added:[/red]")
        for err in errors:
            rprint(err)
        raise typer.Exit(1)

    append_cases(name, cases)
    rprint(f"[green]Added {len(cases)} case(s) to '{name}'[/green]")


@dataset_app.command("update")
def dataset_update(
    name: str = typer.Argument(..., help="Dataset name"),
    description: Optional[str] = typer.Option(None, "--description", "-d", help="New description"),
    tags: Optional[str] = typer.Option(None, "--tags", help="New comma-separated tags (replaces existing)"),
    author: Optional[str] = typer.Option(None, "--author", help="New author name"),
    labels: Optional[str] = typer.Option(None, "--labels", help="New comma-separated labels (classification datasets only; replaces existing)"),
):
    """Update dataset metadata. Only the fields you specify are changed."""
    meta = load_dataset_meta(name)
    changed = []

    if description is not None:
        meta.description = description
        changed.append("description")
    if tags is not None:
        meta.tags = [t.strip() for t in tags.split(",") if t.strip()]
        changed.append("tags")
    if author is not None:
        meta.author = author
        changed.append("author")
    if labels is not None:
        meta.labels = [l.strip() for l in labels.split(",") if l.strip()]
        changed.append("labels")

    if not changed:
        rprint("[yellow]Nothing to update — specify at least one of --description, --tags, --author, --labels.[/yellow]")
        raise typer.Exit()

    from datetime import date
    meta.updated_at = date.today().isoformat()
    save_dataset_meta(meta)
    rprint(f"[green]Updated '{name}':[/green] {', '.join(changed)}")


@dataset_app.command("delete")
def dataset_delete(
    name: str = typer.Argument(..., help="Dataset name to delete"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """Delete a dataset and all its files (cases, baseline, results)."""
    import shutil

    path = dataset_path(name)
    if not path.exists():
        rprint(f"[red]Dataset '{name}' not found.[/red]")
        raise typer.Exit(1)

    if not yes:
        typer.confirm(f"Delete dataset '{name}' at {path}? This cannot be undone.", abort=True)

    shutil.rmtree(path)
    rprint(f"[green]Deleted dataset '{name}'[/green]")


@dataset_app.command("validate")
def dataset_validate(
    name: str = typer.Argument(..., help="Dataset name"),
):
    """Validate all cases in a dataset against its schema."""
    meta = load_dataset_meta(name)
    try:
        cases = load_cases(name)
        rprint(f"[green]OK — {len(cases)} valid case(s) in '{name}' (type: {meta.type.value})[/green]")
    except ValueError as e:
        rprint(f"[red]{e}[/red]")
        raise typer.Exit(1)


@app.command("run")
def run(
    dataset: str = typer.Argument(..., help="Dataset name to evaluate"),
    model: Optional[List[str]] = typer.Option(
        None, "--model", "-m",
        help="Model provider ID (e.g. anthropic:claude-opus-4-6). Repeat for multiple models.",
    ),
    prompt: Optional[str] = typer.Option(
        None, "--prompt", "-p",
        help="Prompt template. Use {{query}}, {{text}}, or {{instruction}} as placeholders. "
             "Omit to use the default template for the dataset type.",
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c",
        help="Path to a Promptfoo YAML config with prompts + providers. "
             "supereval will inject the test cases automatically.",
    ),
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
    Run an eval against a dataset and optionally compare to baseline.

    \b
    Tier 1 — no Promptfoo knowledge needed:
      supereval run my-dataset --model anthropic:claude-opus-4-6

    \b
    Tier 2 — custom prompt, no config file:
      supereval run my-dataset --model anthropic:claude-opus-4-6 \\
        --prompt "You are an AWS expert. Answer concisely: {{query}}"

    \b
    Tier 3 — bring your own Promptfoo config (prompts + providers):
      supereval run my-dataset --config promptfoo.yaml
    """
    if not config and not model:
        rprint("[red]Specify at least one --model or a --config file.[/red]")
        rprint("  Example: supereval run my-dataset --model anthropic:claude-opus-4-6")
        raise typer.Exit(1)

    rprint(f"\n[bold]Running eval:[/bold] {dataset}")
    if model:
        rprint(f"  Models:  {', '.join(model)}")
    if config:
        rprint(f"  Config:  {config}")
    if prompt:
        rprint(f"  Prompt:  {prompt[:80]}{'...' if len(prompt) > 80 else ''}")

    try:
        result = run_eval(
            dataset_name=dataset,
            models=list(model) if model else None,
            prompt=prompt,
            config_path=config,
        )
    except (RuntimeError, ValueError) as e:
        rprint(f"\n[red]Error: {e}[/red]")
        raise typer.Exit(1)

    # --- Record run in history store ---
    if not no_record:
        try:
            record_run(result)
        except Exception as e:
            rprint(f"[yellow]Warning: could not persist run to history store: {e}[/yellow]")

    # --- Results summary ---
    rprint(f"\n[bold]Results[/bold]  (run {result.run_id})")
    status_color = "green" if result.failed == 0 else "red"
    rprint(f"  Passed:    [{status_color}]{result.passed}/{result.total}[/{status_color}]")
    rprint(f"  Pass rate: [{status_color}]{result.pass_rate:.1%}[/{status_color}]")
    if result.total_cost_usd > 0:
        rprint(f"  Cost:      ${result.total_cost_usd:.4f}")
    if result.total_tokens > 0:
        rprint(
            f"  Tokens:    {result.total_tokens:,}"
            f"  (prompt: {result.prompt_tokens:,}  completion: {result.completion_tokens:,})"
        )
    if result.avg_latency_ms > 0:
        rprint(
            f"  Latency:   avg {result.avg_latency_ms:.0f}ms  "
            f"p50 {result.p50_latency_ms:.0f}ms  "
            f"p95 {result.p95_latency_ms:.0f}ms"
        )

    if result.failed > 0:
        rprint("\n  [bold]Failed cases:[/bold]")
        for case in result.cases:
            if not case.passed:
                rprint(f"    [red]✗[/red]  {case.vars}")

    # --- Multi-provider comparison table ---
    _print_comparison_table(result, console)

    # --- Baseline comparison ---
    exit_code = 0
    if compare_baseline:
        report = compare_to_baseline(result)
        if report is None:
            rprint("\n[yellow]No baseline found — run with --update-baseline to create one.[/yellow]")
        else:
            delta_sign = "+" if report.pass_rate_delta >= 0 else ""
            rprint(
                f"\n[bold]Baseline comparison[/bold]  "
                f"(baseline: {report.baseline_pass_rate:.1%}  "
                f"current: {report.current_pass_rate:.1%}  "
                f"delta: {delta_sign}{report.pass_rate_delta:.1%})"
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
                            f"    [red]✗[/red]  {reg['vars']}  "
                            f"(score: {reg['baseline_score']:.2f} → {reg['current_score']:.2f})"
                        )
                exit_code = 1

    # --- Update baseline ---
    if update_baseline:
        save_baseline(result)
        rprint(f"\n[green]Baseline updated for '{dataset}'[/green]")

    # --- Save full results ---
    if output:
        import json
        output.write_text(json.dumps(result.to_dict(), indent=2))
        rprint(f"[green]Results saved to {output}[/green]")

    raise typer.Exit(exit_code)


@app.command("run-all")
def run_all(
    model: Optional[List[str]] = typer.Option(
        None, "--model", "-m",
        help="Model provider ID(s). Repeat for multiple models.",
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c",
        help="Promptfoo YAML config. Applied to all datasets.",
    ),
    compare_baseline: bool = typer.Option(
        False, "--compare-baseline",
        help="Compare each dataset against its baseline.",
    ),
    update_baseline: bool = typer.Option(
        False, "--update-baseline",
        help="Update baselines for all datasets after running.",
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o",
        help="Directory to write per-dataset JSON results. Created if it doesn't exist.",
    ),
    no_record: bool = typer.Option(
        False, "--no-record",
        help="Do not persist runs to the history store.",
    ),
):
    """
    Run evals for all datasets in one pipeline step.

    \b
    Exit code is 0 only when every dataset passes.

    \b
    Example:
      supereval run-all --model anthropic:claude-opus-4-6 --compare-baseline
    """
    import json as _json

    if not config and not model:
        rprint("[red]Specify at least one --model or a --config file.[/red]")
        raise typer.Exit(1)

    datasets = list_datasets()
    if not datasets:
        rprint(f"[yellow]No datasets found in {datasets_dir()}[/yellow]")
        raise typer.Exit()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    rprint(f"\n[bold]Running all {len(datasets)} dataset(s)[/bold]")
    if model:
        rprint(f"  Models: {', '.join(model)}")

    all_passed = True
    summary_rows: list[tuple] = []

    for ds in datasets:
        rprint(f"\n[cyan]{ds.name}[/cyan]")
        try:
            result = run_eval(
                dataset_name=ds.name,
                models=list(model) if model else None,
                config_path=config,
            )
        except Exception as e:
            rprint(f"  [red]ERROR: {e}[/red]")
            summary_rows.append((ds.name, "[red]ERROR[/red]", "—", "—"))
            all_passed = False
            continue

        if not no_record:
            try:
                record_run(result)
            except Exception:
                pass

        if update_baseline:
            save_baseline(result)

        if compare_baseline:
            from .baseline import compare_to_baseline
            report = compare_to_baseline(result)
            if report is not None and not report.passed:
                all_passed = False
                rprint(f"  [red]REGRESSION — pass rate delta: {report.pass_rate_delta:+.1%}[/red]")
            elif report is None:
                rprint("  [yellow]No baseline found[/yellow]")
        elif result.pass_rate < 1.0:
            all_passed = False

        color = "green" if result.pass_rate == 1.0 else ("yellow" if result.pass_rate >= 0.8 else "red")
        icon = "✓" if result.pass_rate == 1.0 else "✗"
        cost_str = f"${result.total_cost_usd:.4f}" if result.total_cost_usd > 0 else "—"
        lat_str = f"{result.p95_latency_ms:.0f}ms p95" if result.avg_latency_ms > 0 else "—"
        rprint(
            f"  [{color}]{icon}[/{color}]  "
            f"{result.passed}/{result.total} passed  "
            f"({result.pass_rate:.1%})"
        )
        summary_rows.append((
            ds.name,
            f"[{color}]{result.pass_rate:.1%}[/{color}]",
            cost_str,
            lat_str,
        ))

        if output_dir:
            out = output_dir / f"{ds.name}.json"
            out.write_text(_json.dumps(result.to_dict(), indent=2))

    # Summary table
    rprint()
    table = Table(title="run-all summary", show_lines=False)
    table.add_column("Dataset", style="cyan")
    table.add_column("Pass rate", justify="right")
    table.add_column("Cost", justify="right", style="dim")
    table.add_column("Latency", justify="right", style="dim")
    for row in summary_rows:
        table.add_row(*row)
    console.print(table)

    if not all_passed:
        rprint("\n[red]One or more datasets failed.[/red]")
        raise typer.Exit(1)

    rprint("\n[green]All datasets passed.[/green]")


@dataset_app.command("export")
def dataset_export(
    name: str = typer.Argument(..., help="Dataset name"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file (default: stdout)"),
):
    """Export a dataset to Promptfoo YAML format."""
    result = export_to_promptfoo(name)
    if output:
        output.write_text(result)
        rprint(f"[green]Exported '{name}' to {output}[/green]")
    else:
        typer.echo(result)


# ---------------------------------------------------------------------------
# dataset version sub-commands
# ---------------------------------------------------------------------------

@version_app.command("tag")
def dataset_version_tag(
    name: str = typer.Argument(..., help="Dataset name"),
    version: str = typer.Argument(..., help="Version string, e.g. v1.0.0"),
    description: str = typer.Option("", "--description", "-d", help="Optional tag note"),
):
    """Snapshot the current dataset cases and metadata as a named version.

    \b
    Example:
      supereval dataset version tag aws-support-qa v1.0.0 --description "Initial release"
    """
    from .versioning import tag_version
    try:
        info = tag_version(name, version, description=description)
    except (ValueError, FileNotFoundError) as exc:
        rprint(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    rprint(f"[green]Tagged '{name}' as {version} ({info.case_count} case(s))[/green]")
    if description:
        rprint(f"  Note: {description}")


@version_app.command("list")
def dataset_version_list(
    name: str = typer.Argument(..., help="Dataset name"),
):
    """List all versions for a dataset."""
    from .versioning import list_versions
    load_dataset_meta(name)  # validate dataset exists
    versions = list_versions(name)

    if not versions:
        rprint(f"[yellow]No versions found for '{name}'. Tag one with:[/yellow]")
        rprint(f"  supereval dataset version tag {name} v1.0.0")
        raise typer.Exit()

    table = Table(title=f"Versions — {name}")
    table.add_column("Version", style="cyan", no_wrap=True)
    table.add_column("Cases", justify="right")
    table.add_column("Tagged at", no_wrap=True)
    table.add_column("Description")

    for v in versions:
        table.add_row(
            v.version,
            str(v.case_count),
            v.created_at[:19].replace("T", " ") + " UTC",
            v.description or "—",
        )
    console.print(table)


@version_app.command("show")
def dataset_version_show(
    name: str = typer.Argument(..., help="Dataset name"),
    version: str = typer.Argument(..., help="Version string, e.g. v1.0.0"),
    limit: int = typer.Option(5, "--limit", "-n", help="Number of cases to preview"),
):
    """Preview cases stored in a specific version snapshot."""
    from .versioning import load_version_cases
    import json as _json

    try:
        cases = load_version_cases(name, version)
    except (FileNotFoundError, ValueError) as exc:
        rprint(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    rprint(f"\n[bold cyan]{name}[/bold cyan] @ [bold]{version}[/bold]  "
           f"({len(cases)} case(s))")

    if not cases:
        rprint("[yellow]No cases in this version.[/yellow]")
        return

    rprint(f"\n[bold]Preview ({min(limit, len(cases))} of {len(cases)} cases):[/bold]")
    for case in cases[:limit]:
        rprint(f"\n  [cyan]{case.id}[/cyan]"
               + (f"  {case.description}" if case.description else ""))
        rprint("  " + _json.dumps(case.model_dump(), indent=2)[:300])


@version_app.command("restore")
def dataset_version_restore(
    name: str = typer.Argument(..., help="Dataset name"),
    version: str = typer.Argument(..., help="Version string to restore, e.g. v1.0.0"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """Restore a dataset's cases.jsonl from a version snapshot.

    \b
    This overwrites the current cases.jsonl with the snapshot from <version>.
    The operation is safe — the version snapshot itself is never modified.

    \b
    Example:
      supereval dataset version restore aws-support-qa v1.0.0
    """
    from .versioning import restore_version

    if not yes:
        typer.confirm(
            f"Restore '{name}' to {version}? This will overwrite the current cases.jsonl.",
            abort=True,
        )

    try:
        count = restore_version(name, version)
    except FileNotFoundError as exc:
        rprint(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    rprint(f"[green]Restored '{name}' to {version} ({count} case(s))[/green]")


def _interactive_review(cases: list, dataset_type: str) -> list:
    """Walk through generated cases one-by-one for keep / edit / skip / quit review.

    Returns the list of approved GeneratedCase objects.
    """
    import json as _json
    import click
    from rich.panel import Panel
    from rich.syntax import Syntax
    from .generator import GeneratedCase

    approved = []
    total = len(cases)

    for i, case in enumerate(cases, 1):
        rprint(
            f"\n[bold]Case {i}/{total}[/bold]  "
            f"[dim]{case.case_type} · {case.difficulty}[/dim]"
        )
        rprint(Panel(
            Syntax(_json.dumps(case.to_staged_dict(), indent=2), "json", theme="ansi_dark"),
            title=f"[cyan]{case.description or '(no description)'}[/cyan]",
            expand=False,
        ))

        while True:
            raw = typer.prompt("[k]eep  [e]dit  [s]kip  [q]uit", default="k").strip().lower()
            if raw in ("k", "keep", ""):
                approved.append(case)
                break
            elif raw in ("e", "edit"):
                edited_text = click.edit(
                    _json.dumps(case.to_staged_dict(), indent=2),
                    extension=".json",
                )
                if edited_text is None:
                    rprint("[yellow]No changes — keeping original.[/yellow]")
                    approved.append(case)
                else:
                    try:
                        d = _json.loads(edited_text)
                        approved.append(GeneratedCase(
                            description=d.get("description", case.description),
                            input=d.get("input", case.input),
                            expected=d.get("expected", case.expected),
                            tags=d.get("tags", case.tags),
                            difficulty=d.get("difficulty", case.difficulty),
                            case_type=d.get("case_type", case.case_type),
                            source_excerpt=d.get("source_excerpt", case.source_excerpt),
                        ))
                        rprint("[green]Updated.[/green]")
                    except Exception as exc:
                        rprint(f"[red]Parse error: {exc} — keeping original.[/red]")
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


class _GeneratorBackend(str):
    bedrock = "bedrock"
    anthropic = "anthropic"


@app.command("generate")
def generate(
    dataset: str = typer.Argument(..., help="Dataset name to generate cases for"),
    from_path: str = typer.Option(..., "--from", help="Local file/directory or s3://bucket/prefix"),
    count: int = typer.Option(20, "--count", "-n", help="Number of test cases to generate"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Staging file path (default: <dataset>_staged.jsonl). "
             "Review this file before importing with `supereval dataset add-cases`.",
    ),
    model: Optional[str] = typer.Option(
        None, "--model",
        help="Model ID to use for generation. "
             "Bedrock: e.g. anthropic.claude-3-5-sonnet-20241022-v2:0. "
             "Anthropic API: e.g. claude-sonnet-4-6.",
    ),
    region: str = typer.Option(DEFAULT_REGION, "--region", help="AWS region (Bedrock only)"),
    backend: str = typer.Option(
        "bedrock", "--backend",
        help="Generator backend: bedrock (default), anthropic, openai, or azure-openai.",
    ),
    azure_endpoint: Optional[str] = typer.Option(
        None, "--azure-endpoint",
        help="Azure OpenAI endpoint URL (azure-openai backend only). "
             "Can also be set via AZURE_OPENAI_ENDPOINT env var.",
    ),
    crawl: bool = typer.Option(
        False, "--crawl",
        help="Follow links under the same URL prefix (URL sources only). "
             "Combines with --max-pages to cap how many pages are fetched.",
    ),
    max_pages: int = typer.Option(
        20, "--max-pages",
        help="Maximum number of pages to fetch when --crawl is set (default: 20).",
    ),
    auto_import: bool = typer.Option(
        False, "--auto-import",
        help="Import generated cases directly without staging. Skips the review step.",
    ),
    interactive: bool = typer.Option(
        False, "--interactive",
        help="Review each generated case interactively before staging or importing. "
             "For each case: [k]eep, [e]dit (opens $EDITOR), [s]kip, or [q]uit.",
    ),
):
    """
    Generate synthetic test cases from documents using Claude.

    \b
    Supported dataset types: qa, classification, instruction

    \b
    From a local file or directory (Bedrock):
      supereval generate my-dataset --from docs/ --count 20

    \b
    From S3 (Bedrock):
      supereval generate my-dataset --from s3://my-bucket/docs/ --count 20

    \b
    From a URL (single page):
      supereval generate my-dataset --from https://example.com/docs/page.html

    \b
    From a URL with crawling (follows links under the same prefix):
      supereval generate my-dataset --from https://docs.aws.amazon.com/lambda/latest/dg/ --crawl
      supereval generate my-dataset --from https://docs.aws.amazon.com/lambda/latest/dg/ --crawl --max-pages 30

    \b
    Using the Anthropic API directly:
      supereval generate my-dataset --from docs/ --backend anthropic

    \b
    Using OpenAI:
      supereval generate my-dataset --from docs/ --backend openai --model gpt-4o

    \b
    Cases are staged for review by default. To import directly:
      supereval generate my-dataset --from docs/ --auto-import
    """
    import json as _json

    meta = load_dataset_meta(dataset)

    # For classification, labels are required
    if meta.type == DatasetType.classification and not meta.labels:
        rprint(
            f"[red]Classification generation requires labels. "
            f"Recreate '{dataset}' with --labels 'label1,label2,...'[/red]"
        )
        raise typer.Exit(1)

    if backend not in ("bedrock", "anthropic", "openai", "azure-openai"):
        rprint(
            f"[red]Unknown backend '{backend}'. "
            "Choose 'bedrock', 'anthropic', 'openai', or 'azure-openai'.[/red]"
        )
        raise typer.Exit(1)

    # Load documents
    rprint(f"\n[bold]Loading documents from:[/bold] {from_path}")
    try:
        if from_path.startswith("s3://"):
            source = S3Source(from_path, region=region)
        elif from_path.startswith(("http://", "https://")):
            source = URLSource(from_path, crawl=crawl, max_pages=max_pages)
        else:
            local = Path(from_path)
            if not local.exists():
                rprint(f"[red]Path not found: {from_path}[/red]")
                raise typer.Exit(1)
            source = LocalFileSource(local)
        documents = source.load()
    except (FileNotFoundError, ValueError, ImportError) as e:
        rprint(f"[red]{e}[/red]")
        raise typer.Exit(1)

    if crawl:
        rprint(f"  Crawled {len(documents)} page(s): {', '.join(d.filename for d in documents)}")
    else:
        rprint(f"  Loaded {len(documents)} document(s): {', '.join(d.filename for d in documents)}")

    # Resolve model default based on backend
    _model_defaults = {
        "bedrock": DEFAULT_MODEL,
        "anthropic": ANTHROPIC_DEFAULT_MODEL,
        "openai": OPENAI_DEFAULT_MODEL,
        "azure-openai": OPENAI_DEFAULT_MODEL,
    }
    resolved_model = model or _model_defaults[backend]

    _backend_labels = {
        "bedrock": f"{resolved_model} ({region})",
        "anthropic": f"Anthropic API ({resolved_model})",
        "openai": f"OpenAI API ({resolved_model})",
        "azure-openai": f"Azure OpenAI ({resolved_model})",
    }
    rprint(f"\n[bold]Generating {count} {meta.type.value} cases[/bold] using {_backend_labels[backend]}")

    if meta.type == DatasetType.qa:
        rprint("  Case mix: 40% factual · 20% multi-hop · 15% negation · 15% out-of-scope · 10% ambiguous")
    elif meta.type == DatasetType.classification:
        rprint(f"  Labels: {', '.join(meta.labels)}")

    if backend == "bedrock":
        generator = BedrockGenerator(model_id=resolved_model, region=region)
    elif backend == "anthropic":
        generator = AnthropicGenerator(model_id=resolved_model)
    elif backend == "azure-openai":
        import os
        endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            rprint(
                "[red]Azure OpenAI requires an endpoint. "
                "Pass --azure-endpoint or set AZURE_OPENAI_ENDPOINT.[/red]"
            )
            raise typer.Exit(1)
        generator = OpenAIGenerator(model_id=resolved_model, azure_endpoint=endpoint)
    else:  # openai
        generator = OpenAIGenerator(model_id=resolved_model)

    all_cases = []
    per_doc = max(1, math.ceil(count / len(documents)))
    for doc in documents:
        doc_count = min(per_doc, count - len(all_cases))
        if doc_count <= 0:
            break
        rprint(f"  Generating {doc_count} cases from [cyan]{doc.filename}[/cyan]...")
        try:
            cases = generator.generate(
                doc, doc_count,
                dataset_type=meta.type.value,
                labels=meta.labels,
            )
            all_cases.extend(cases)
            rprint(f"    [green]Got {len(cases)} case(s)[/green]")
        except Exception as e:
            rprint(f"    [red]Failed: {e}[/red]")
            raise typer.Exit(1)

    all_cases = all_cases[:count]
    rprint(f"\n[bold]Generated {len(all_cases)} case(s) total[/bold]")

    if interactive:
        rprint(f"\n[bold]Interactive review[/bold] — {len(all_cases)} case(s). "
               "Edit opens $EDITOR (or nano/vi).")
        all_cases = _interactive_review(all_cases, meta.type.value)
        rprint(f"[bold]Review complete:[/bold] {len(all_cases)} case(s) approved")
        if not all_cases:
            rprint("[yellow]No cases approved — nothing to import or stage.[/yellow]")
            raise typer.Exit(0)

    if auto_import:
        type_model = TYPE_TO_MODEL[meta.type]
        valid_cases = [type_model.model_validate(c.to_importable_dict()) for c in all_cases]
        append_cases(dataset, valid_cases)
        rprint(f"[green]Imported {len(valid_cases)} case(s) into '{dataset}'[/green]")
    else:
        staging_path = output or Path(f"{dataset}_staged.jsonl")
        lines = [_json.dumps(c.to_staged_dict()) for c in all_cases]
        staging_path.write_text("\n".join(lines) + "\n")
        rprint(f"\n[green]Staged to:[/green] {staging_path}")
        rprint("\nNext steps:")
        rprint(f"  1. Review [cyan]{staging_path}[/cyan]  (check source_excerpt to verify accuracy)")
        rprint(f"  2. Remove or edit any cases that look wrong")
        rprint(f"  3. Import: [bold]supereval dataset add-cases {dataset} --from {staging_path}[/bold]")


# ---------------------------------------------------------------------------
# history subcommand
# ---------------------------------------------------------------------------

@history_app.command("list")
def history_list(
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d", help="Filter by dataset name"),
    limit: int = typer.Option(20, "--limit", "-n", help="Number of runs to show"),
):
    """List recent eval runs."""
    runs = list_runs(dataset=dataset, limit=limit)
    if not runs:
        rprint("[yellow]No runs recorded yet. Run `supereval run` to start.[/yellow]")
        raise typer.Exit()

    table = Table(title="Run history")
    table.add_column("Run ID", style="cyan", no_wrap=True)
    table.add_column("Dataset", style="magenta")
    table.add_column("Ran at", no_wrap=True)
    table.add_column("Providers")
    table.add_column("Cases", justify="right")
    table.add_column("Pass rate", justify="right")
    table.add_column("Cost ($)", justify="right")
    table.add_column("p95 (ms)", justify="right")

    import json
    for r in runs:
        pass_rate = r["pass_rate"]
        color = "green" if pass_rate >= 1.0 else ("yellow" if pass_rate >= 0.8 else "red")
        providers = json.loads(r["providers"]) if isinstance(r["providers"], str) else r["providers"]
        table.add_row(
            r["run_id"],
            r["dataset"],
            r["ran_at"][:19].replace("T", " "),
            ", ".join(providers),
            str(r["total_cases"]),
            f"[{color}]{pass_rate:.1%}[/{color}]",
            f"{r['total_cost_usd']:.4f}" if r["total_cost_usd"] else "—",
            f"{r['p95_latency_ms']:.0f}" if r["p95_latency_ms"] else "—",
        )

    console.print(table)


@history_app.command("show")
def history_show(
    run_id: str = typer.Argument(..., help="Run ID to inspect"),
    cases: bool = typer.Option(False, "--cases", help="Show individual case results"),
):
    """Show details for a specific run."""
    run = get_run(run_id)
    if run is None:
        rprint(f"[red]Run '{run_id}' not found.[/red]")
        raise typer.Exit(1)

    import json
    providers = json.loads(run["providers"]) if isinstance(run["providers"], str) else run["providers"]
    pass_rate = run["pass_rate"]
    color = "green" if pass_rate >= 1.0 else ("yellow" if pass_rate >= 0.8 else "red")

    rprint(f"\n[bold cyan]{run['run_id']}[/bold cyan]")
    rprint(f"  Dataset:    {run['dataset']}")
    rprint(f"  Ran at:     {run['ran_at'][:19].replace('T', ' ')} UTC")
    rprint(f"  Providers:  {', '.join(providers)}")
    rprint(f"  Cases:      {run['total_cases']}  (passed: {run['passed']}  failed: {run['failed']})")
    rprint(f"  Pass rate:  [{color}]{pass_rate:.1%}[/{color}]")
    if run["total_cost_usd"]:
        rprint(f"  Cost:       ${run['total_cost_usd']:.4f}")
    if run["total_tokens"]:
        rprint(f"  Tokens:     {run['total_tokens']:,}")
    if run["avg_latency_ms"]:
        rprint(
            f"  Latency:    avg {run['avg_latency_ms']:.0f}ms  "
            f"p50 {run['p50_latency_ms']:.0f}ms  "
            f"p95 {run['p95_latency_ms']:.0f}ms"
        )

    if cases:
        case_rows = get_run_cases(run_id)
        if not case_rows:
            rprint("\n[yellow]No case results stored for this run.[/yellow]")
            return
        rprint(f"\n[bold]Case results ({len(case_rows)}):[/bold]")
        for c in case_rows:
            icon = "[green]✓[/green]" if c["passed"] else "[red]✗[/red]"
            extras = f"  score={c['score']:.3f}"
            if c["latency_ms"]:
                extras += f"  {c['latency_ms']}ms"
            if c["cost_usd"]:
                extras += f"  ${c['cost_usd']:.5f}"
            rprint(f"  {icon}  {c['vars']}{extras}")


@providers_app.command("list")
def providers_list():
    """Show known model IDs for supereval generate and supereval run."""

    # --- Bedrock generate model IDs ---
    bedrock_models = [
        ("us.anthropic.claude-3-5-haiku-20241022-v1:0",   "Claude 3.5 Haiku",   "Default for generate"),
        ("us.anthropic.claude-sonnet-4-6",                 "Claude Sonnet 4.6",  "Recommended"),
        ("us.anthropic.claude-haiku-4-5-20251001-v1:0",   "Claude Haiku 4.5",   ""),
        ("us.anthropic.claude-opus-4-6-v1",               "Claude Opus 4.6",    "Most capable"),
        ("us.amazon.nova-pro-v1:0",                        "Amazon Nova Pro",    ""),
        ("us.amazon.nova-lite-v1:0",                       "Amazon Nova Lite",   ""),
        ("us.amazon.nova-micro-v1:0",                      "Amazon Nova Micro",  "Cheapest"),
        ("us.deepseek.r1-v1:0",                            "DeepSeek R1",        ""),
    ]
    bedrock_table = Table(title="Bedrock model IDs  (supereval generate --model <id>)")
    bedrock_table.add_column("Model ID", style="cyan", no_wrap=True)
    bedrock_table.add_column("Name")
    bedrock_table.add_column("Notes", style="dim")
    for row in bedrock_models:
        bedrock_table.add_row(*row)
    console.print(bedrock_table)

    # --- Anthropic API model IDs ---
    anthropic_models = [
        ("claude-sonnet-4-6",          "Claude Sonnet 4.6",  "Recommended (Anthropic API)"),
        ("claude-opus-4-6",            "Claude Opus 4.6",    "Most capable"),
        ("claude-haiku-4-5-20251001",  "Claude Haiku 4.5",   "Fast, cost-efficient"),
        ("claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet",  ""),
        ("claude-3-5-haiku-20241022",  "Claude 3.5 Haiku",   ""),
    ]
    anthropic_table = Table(title="Anthropic API model IDs  (supereval generate --backend anthropic --model <id>)")
    anthropic_table.add_column("Model ID", style="cyan", no_wrap=True)
    anthropic_table.add_column("Name")
    anthropic_table.add_column("Notes", style="dim")
    for row in anthropic_models:
        anthropic_table.add_row(*row)
    console.print(anthropic_table)

    # --- Promptfoo provider IDs for supereval run ---
    run_models = [
        ("bedrock:us.anthropic.claude-sonnet-4-6",               "Claude Sonnet 4.6",  "Recommended"),
        ("bedrock:us.anthropic.claude-haiku-4-5-20251001-v1:0",  "Claude Haiku 4.5",   ""),
        ("bedrock:us.anthropic.claude-opus-4-6-v1",              "Claude Opus 4.6",    ""),
        ("bedrock:us.amazon.nova-pro-v1:0",                       "Amazon Nova Pro",    ""),
        ("bedrock:us.amazon.nova-lite-v1:0",                      "Amazon Nova Lite",   ""),
        ("bedrock:us.deepseek.r1-v1:0",                           "DeepSeek R1",        ""),
        ("anthropic:claude-sonnet-4-6",                           "Claude Sonnet 4.6",  "Requires ANTHROPIC_API_KEY"),
        ("openai:gpt-4o",                                         "GPT-4o",             "Requires OPENAI_API_KEY"),
    ]
    run_table = Table(title="Provider IDs  (supereval run --model <id>)")
    run_table.add_column("Provider ID", style="cyan", no_wrap=True)
    run_table.add_column("Name")
    run_table.add_column("Notes", style="dim")
    for row in run_models:
        run_table.add_row(*row)
    console.print(run_table)
    rprint("\n[dim]Use 'supereval providers bedrock' to list models available in your account.[/dim]")
    rprint("[dim]Use 'supereval providers test <model-id>' to verify a model is reachable.[/dim]")


@providers_app.command("bedrock")
def providers_bedrock(
    region: str = typer.Option(DEFAULT_REGION, "--region", "-r", help="AWS region to query."),
    provider: Optional[str] = typer.Option(
        None, "--provider", "-p",
        help="Filter by provider name (e.g. Anthropic, Amazon, Meta).",
    ),
):
    """List foundation models available in your Bedrock account."""
    import boto3
    try:
        client = boto3.client("bedrock", region_name=region)
        kwargs = {}
        if provider:
            kwargs["byProvider"] = provider
        resp = client.list_foundation_models(**kwargs)
    except Exception as e:
        rprint(f"[red]Failed to query Bedrock: {e}[/red]")
        raise typer.Exit(1)

    models = resp.get("modelSummaries", [])
    if not models:
        rprint("[yellow]No models returned. Check your credentials and region.[/yellow]")
        raise typer.Exit()

    table = Table(title=f"Bedrock foundation models  ({region})", show_lines=False)
    table.add_column("Model ID", style="cyan", no_wrap=True)
    table.add_column("Provider")
    table.add_column("Name")
    table.add_column("Status", justify="center")
    table.add_column("Input", style="dim")
    table.add_column("Output", style="dim")

    for m in sorted(models, key=lambda x: x.get("modelId", "")):
        status = m.get("modelLifecycle", {}).get("status", "")
        color = "green" if status == "ACTIVE" else "yellow" if status == "LEGACY" else "dim"
        input_mods = "+".join(m.get("inputModalities", []))
        output_mods = "+".join(m.get("outputModalities", []))
        table.add_row(
            m.get("modelId", ""),
            m.get("providerName", ""),
            m.get("modelName", ""),
            f"[{color}]{status}[/{color}]",
            input_mods,
            output_mods,
        )

    console.print(table)
    rprint(f"\n[dim]{len(models)} model(s) listed. Use --provider to filter (e.g. --provider Anthropic).[/dim]")


@providers_app.command("test")
def providers_test(
    model_id: str = typer.Argument(
        ...,
        help="Model ID to test. Accepts bare Bedrock ID (us.anthropic.claude-sonnet-4-6) "
             "or prefixed form (bedrock:us.anthropic.claude-sonnet-4-6).",
    ),
    region: str = typer.Option(DEFAULT_REGION, "--region", "-r", help="AWS region."),
    prompt: str = typer.Option(
        "Reply with exactly one word: hello.",
        "--prompt", "-p",
        help="Test prompt to send.",
    ),
):
    """Send a test prompt to a model and verify it is reachable."""
    import time
    import boto3

    # Normalise: strip bedrock: prefix if present
    bare_id = model_id
    if model_id.startswith("bedrock:"):
        bare_id = model_id[len("bedrock:"):]
    elif ":" in model_id and not model_id.startswith("us.") \
            and not model_id.startswith("eu.") \
            and not model_id.startswith("ap.") \
            and not model_id.startswith("anthropic.") \
            and not model_id.startswith("amazon.") \
            and not model_id.startswith("meta.") \
            and not model_id.startswith("us."):
        # Looks like a non-Bedrock provider (anthropic:, openai:) — not supported here
        rprint(f"[red]'providers test' only supports Bedrock models.[/red]")
        rprint(f"  Strip the prefix and use the bare model ID, e.g.: [cyan]{model_id.split(':', 1)[-1]}[/cyan]")
        raise typer.Exit(1)

    rprint(f"\nTesting [cyan]{bare_id}[/cyan]  (region: {region})")
    rprint(f"  Prompt: [dim]{prompt}[/dim]")

    try:
        client = boto3.client("bedrock-runtime", region_name=region)
        import json as _json
        t0 = time.monotonic()
        response = client.converse(
            modelId=bare_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 64},
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
    except Exception as e:
        rprint(f"\n  [red]FAILED[/red]  {e}")
        raise typer.Exit(1)

    output = response.get("output", {}).get("message", {}).get("content", [])
    reply = output[0].get("text", "").strip() if output else "(no text)"
    usage = response.get("usage", {})
    input_tokens = usage.get("inputTokens", 0)
    output_tokens = usage.get("outputTokens", 0)

    rprint(f"\n  [green]OK[/green]  {latency_ms}ms  "
           f"[dim](in: {input_tokens} tok  out: {output_tokens} tok)[/dim]")
    rprint(f"  Response: [italic]{reply[:120]}[/italic]")


@history_app.command("stats")
def history_stats(
    dataset: str = typer.Argument(..., help="Dataset name"),
    last: int = typer.Option(10, "--last", "-n", help="Number of recent runs to aggregate"),
):
    """Show aggregate cost/latency/pass-rate stats for a dataset."""
    stats = get_stats(dataset=dataset, last=last)
    if not stats or not stats.get("run_count"):
        rprint(f"[yellow]No runs found for dataset '{dataset}'.[/yellow]")
        raise typer.Exit()

    rprint(f"\n[bold]Stats for '{dataset}'[/bold]  (last {stats['run_count']} run(s))")
    rprint(f"  Pass rate:   avg {stats['avg_pass_rate']:.1%}  "
           f"min {stats['min_pass_rate']:.1%}  "
           f"max {stats['max_pass_rate']:.1%}")
    if stats["total_cost_usd"]:
        rprint(f"  Cost:        total ${stats['total_cost_usd']:.4f}  "
               f"avg/run ${stats['avg_cost_per_run']:.4f}  "
               f"max/run ${stats['max_cost_per_run']:.4f}")
    if stats["total_tokens"]:
        rprint(f"  Tokens:      {int(stats['total_tokens']):,} total")
    if stats["avg_latency_ms"]:
        rprint(f"  Latency:     avg {stats['avg_latency_ms']:.0f}ms  "
               f"max p95 {stats['max_p95_latency_ms']:.0f}ms")


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

@app.command("mcp")
def mcp_serve():
    """Start the supereval MCP server (stdio transport).

    Exposes supereval as a set of tools that AI coding assistants
    (Claude Code, Kiro, etc.) can call on behalf of users.

    \b
    Install the dependency first:
      pip install 'supereval[mcp]'

    \b
    Then add to your Claude Code config (~/.claude.json):
      {
        "mcpServers": {
          "supereval": {
            "command": "supereval",
            "args": ["mcp"]
          }
        }
      }

    \b
    Or for Kiro (.kiro/settings/mcp.json):
      {
        "mcpServers": {
          "supereval": {
            "command": "supereval",
            "args": ["mcp"]
          }
        }
      }
    """
    try:
        from .mcp_server import run as _run_mcp
    except ImportError:
        rprint("[red]mcp package not installed.[/red]")
        rprint("Install with: [bold]pip install 'supereval[mcp]'[/bold]")
        raise typer.Exit(1)
    _run_mcp()
