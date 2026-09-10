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
from anchor.chunk.pipeline import chunk_corpus
from anchor.config import ChunkStrategy, get_settings
from anchor.evaluate.chunk_integrity import BUCKETS
from anchor.evaluate.integrity_run import measure_integrity
from anchor.ingest.pipeline import ingest as run_ingest

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
db_app = typer.Typer(no_args_is_help=True, help="Database lifecycle.")
app.add_typer(db_app, name="db")

console = Console()


def _configure_logging() -> None:
    """Our own progress at INFO, without httpx logging every model download."""
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("anchor").setLevel(logging.INFO)


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

    _configure_logging()
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


@app.command("chunk")
def chunk_cmd(
    strategy: str = typer.Option(
        "",
        help="Only this strategy: fixed or structure_aware. Default builds both.",
    ),
) -> None:
    """Chunk every ingested document. Builds both strategies by default.

    Both have to exist over the same documents at the same time, or the recall
    comparison between them is not like-for-like.
    """
    settings = get_settings()
    targets = [ChunkStrategy(strategy)] if strategy else list(ChunkStrategy)

    _configure_logging()
    report = chunk_corpus(settings, targets)

    console.print(f"[bold]chunked[/bold] {report.documents} documents")
    for name, count in report.chunks_by_strategy.items():
        code = report.code_chunks_by_strategy[name]
        console.print(f"  {name:>16}  {count} chunks  ({code} code-only)")
    if report.missing_documents:
        console.print(
            f"[yellow]{report.missing_documents} files had no document row; "
            "run `anchor ingest` first[/yellow]"
        )


@app.command("integrity")
def integrity_cmd() -> None:
    """Measure how many fenced code blocks survive each chunking strategy whole.

    The objective half of the week-1 chunking comparison: no LLM judge, no
    proxy, just whether a command can still be copied out of a retrieved chunk.
    """
    _configure_logging()
    results = measure_integrity(get_settings())

    console.print("[bold]code-block integrity[/bold]")
    for result in results:
        console.print(
            f"  {result.strategy:>16}  {result.intact}/{result.code_blocks} intact"
            f"  ({result.rate:.1%})"
        )

    console.print()
    console.print("[bold]by code-block size, in tokens[/bold]")
    header = "  " + "size".ljust(10) + "".join(r.strategy.rjust(20) for r in results)
    console.print(header)
    for name, _, _ in BUCKETS:
        cells = ""
        for result in results:
            blocks, intact = result.by_size.get(name, (0, 0))
            rate = result.rate_for(name)
            cells += ("-" if rate is None else f"{intact}/{blocks} ({rate:.0%})").rjust(20)
        console.print("  " + name.ljust(10) + cells)


@app.command("config")
def show_config() -> None:
    """Print the run config exactly as it would be written to runs.config_json."""
    console.print_json(json.dumps(get_settings().run_config()))


if __name__ == "__main__":
    app()
