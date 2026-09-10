"""The `anchor` command line.

Thin wrappers only. Every subcommand resolves settings once and hands them to a
library function, so that anything reachable from the CLI is equally reachable
from a test or a sweep.
"""

from __future__ import annotations

import json
import logging

import typer
from rich.console import Console

from anchor import db
from anchor.config import get_settings
from anchor.ingest.pipeline import ingest as run_ingest

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
db_app = typer.Typer(no_args_is_help=True, help="Database lifecycle.")
app.add_typer(db_app, name="db")

console = Console()


@db_app.command("init")
def db_init() -> None:
    """Apply the schema. Idempotent."""
    settings = get_settings()
    db.init_schema(settings)
    console.print(f"[green]schema applied[/green] -> {settings.pg_dsn}")


@db_app.command("check")
def db_check() -> None:
    """Report row counts per table, so a broken ingest is visible immediately."""
    settings = get_settings()
    tables = ("documents", "chunks", "symbols", "golden_queries", "runs", "answers", "claims")
    with db.connect(settings) as conn:
        for table in tables:
            # Interpolated, not parameterised: an identifier cannot be a bind
            # parameter. `tables` is a literal tuple above, never user input.
            row = conn.execute(f"SELECT count(*) AS n FROM {table}").fetchone()
            count = row["n"] if row else 0
            console.print(f"{table:>16}  {count}")


@app.command("ingest")
def ingest_cmd(
    commit: str = typer.Option(
        "",
        help="Pin to this vLLM commit. Defaults to the tip of the configured branch.",
    ),
    no_docstrings: bool = typer.Option(
        False, "--no-docstrings", help="Markup only. Much faster for a smoke run."
    ),
) -> None:
    """Clone vLLM at a pinned commit, parse the docs, and version-tag them."""
    settings = get_settings()
    if commit:
        settings.ingest.vllm_commit = commit
    if no_docstrings:
        settings.ingest.include_docstrings = False

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = run_ingest(settings)

    console.print(f"[bold]vLLM {report.vllm_version}[/bold] @ {report.commit_sha[:12]}")
    console.print(f"  markup documents     {report.markup_documents}")
    console.print(f"  docstring documents  {report.docstring_documents}")
    console.print(f"  inserted             {report.inserted}")
    console.print(f"  updated              {report.updated}")
    console.print(f"  skipped (too short)  {report.skipped_short}")
    console.print(f"  skipped (no docs)    {report.skipped_unparseable}")
    if report.total == 0:
        console.print("[yellow]no documents ingested; check ingest.doc_globs[/yellow]")


@app.command("config")
def show_config() -> None:
    """Print the run config exactly as it would be written to runs.config_json."""
    console.print_json(json.dumps(get_settings().run_config()))


if __name__ == "__main__":
    app()
