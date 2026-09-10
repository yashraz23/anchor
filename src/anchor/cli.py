"""The `anchor` command line.

Thin wrappers only. Every subcommand resolves settings once and hands them to a
library function, so that anything reachable from the CLI is equally reachable
from a test or a sweep.
"""

from __future__ import annotations

import json
import logging
import sys
import textwrap
from pathlib import Path

import typer
from rich.console import Console

from anchor import db
from anchor.chunk.pipeline import chunk_corpus
from anchor.config import REPO_ROOT, ChunkStrategy, RetrievalMode, get_settings
from anchor.evaluate.chunk_integrity import BUCKETS
from anchor.evaluate.integrity_run import measure_integrity
from anchor.index.pipeline import build_index
from anchor.ingest.pipeline import ingest as run_ingest

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
db_app = typer.Typer(no_args_is_help=True, help="Database lifecycle.")
app.add_typer(db_app, name="db")
golden_app = typer.Typer(no_args_is_help=True, help="Golden-set curation workflow.")
app.add_typer(golden_app, name="golden")

console = Console()


def _configure_logging() -> None:
    """Our own progress at INFO, without httpx logging every model download."""
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("anchor").setLevel(logging.INFO)


@app.callback()
def _main() -> None:
    """Force UTF-8 output before any subcommand runs.

    The corpus is real documentation and contains emoji. A Windows console
    defaults to a legacy codepage that cannot encode them, so printing a
    retrieved span dies with UnicodeEncodeError partway through the results.
    errors="replace" so one unencodable glyph degrades to a placeholder rather
    than losing the span it appeared in.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


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
    commit: str = typer.Option("", help="Pin to this exact vLLM commit."),
    latest: bool = typer.Option(
        False,
        "--latest",
        help="Advance the corpus to the tip of the branch. Requires re-chunking.",
    ),
    no_docstrings: bool = typer.Option(
        False, "--no-docstrings", help="Markup only. Much faster for a smoke run."
    ),
) -> None:
    """Clone vLLM at a pinned commit, parse the docs, and version-tag them.

    Re-running stays on the commit already ingested. Silently advancing to the
    branch tip would orphan every chunk and embedding built against the previous
    one, and would invalidate any measurement already recorded, since vLLM's main
    moves several times a day. Advancing is therefore explicit: `--latest`.
    """
    settings = get_settings()
    if commit:
        settings.ingest.vllm_commit = commit
    elif not latest:
        with db.connect(settings) as conn:
            existing = db.latest_ingested_commit(conn)
        if existing:
            settings.ingest.vllm_commit = existing
            console.print(f"[dim]staying on ingested commit {existing[:12]}[/dim]")
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


@app.command("index")
def index_cmd() -> None:
    """Embed every chunk into pgvector and build its tsvector.

    Resumable: both passes select only rows still missing their column, so an
    interrupted run picks up rather than re-embedding the corpus.
    """
    _configure_logging()
    report = build_index(get_settings())
    console.print(f"  embedded          {report.embedded}")
    console.print(f"  tsvectors built   {report.tsv_built}")
    console.print(f"  already embedded  {report.already_embedded}")


@app.command("search")
def search_cmd(
    query: str = typer.Argument(..., help="The question to retrieve for."),
    mode: str = typer.Option("", help="dense, sparse or hybrid. Default from config."),
    strategy: str = typer.Option("", help="Chunking strategy to search."),
) -> None:
    """Retrieve for one query and print the spans, for eyeballing quality."""
    _configure_logging()
    settings = get_settings()
    chosen = ChunkStrategy(strategy) if strategy else settings.chunk.strategy

    from anchor.index.embed import Embedder
    from anchor.retrieve.search import search

    embedder = Embedder(
        settings.index.embedding_model,
        normalize=settings.index.normalize_embeddings,
        query_instruction=settings.index.query_instruction,
    )
    with db.connect(settings) as conn:
        hits = search(
            conn,
            query,
            settings,
            embedder,
            strategy=chosen.value,
            mode=RetrievalMode(mode) if mode else None,
        )

    console.print(f"[bold]{len(hits)} spans[/bold] for {query!r} ({chosen.value})")
    console.print()
    for i, hit in enumerate(hits, start=1):
        # markup=False on every line carrying corpus text. Retrieved spans are
        # documentation, full of square brackets that rich would otherwise parse
        # as style tags and reject.
        flag = "  (code)" if hit.is_code_block else ""
        console.print(
            f"span {i}  {hit.cite()}  score={hit.score:.4f}{flag}",
            markup=False,
            style="bold",
        )
        body = hit.text if len(hit.text) <= 300 else hit.text[:300] + "..."
        console.print(textwrap.indent(body, "    "), markup=False)
        console.print()


@golden_app.command("harvest")
def golden_harvest(
    labels: str = typer.Option("usage,documentation", help="Comma-separated issue labels."),
    pages: int = typer.Option(3, help="Pages of 100 issues per label."),
) -> None:
    """Fetch candidate questions from real vLLM issues. Requires the gh CLI.

    Produces candidates only. Whether a question is answerable from the corpus
    is a curation judgement, made against evidence, not here.
    """
    _configure_logging()
    from anchor.evaluate.harvest import fetch_candidates, write_candidates

    seen: set[int] = set()
    unique = []
    for label in [x.strip() for x in labels.split(",") if x.strip()]:
        got = fetch_candidates(label=label, pages=pages)
        console.print(f"  {label:>16}  {len(got)} issues")
        for candidate in got:
            if candidate.number not in seen:
                seen.add(candidate.number)
                unique.append(candidate)

    out = REPO_ROOT / "data" / "golden" / "candidates.jsonl"
    console.print(f"wrote {write_candidates(out, unique)} unique candidates -> {out}")


@golden_app.command("propose")
def golden_propose(
    limit: int = typer.Option(60, help="How many candidates to gather evidence for."),
    out: str = typer.Option("", help="Where to write the report. Defaults to stdout."),
) -> None:
    """Run retrieval over candidates and write the spans, for human review.

    Decides nothing. Auto-accepting whatever retrieval returned would make the
    golden set a transcript of current behaviour, and a set that agrees with the
    system by construction cannot measure it.
    """
    _configure_logging()
    from anchor.evaluate.harvest import read_candidates
    from anchor.evaluate.propose import gather_evidence, render_evidence

    candidates = read_candidates(REPO_ROOT / "data" / "golden" / "candidates.jsonl")
    if not candidates:
        console.print("[yellow]no candidates; run `anchor golden harvest` first[/yellow]")
        return

    report = render_evidence(gather_evidence(candidates, get_settings(), limit=limit))
    if out:
        Path(out).write_text(report, encoding="utf-8")
        console.print(f"wrote evidence -> {out}")
    else:
        console.print(report, markup=False)


@golden_app.command("validate")
def golden_validate() -> None:
    """Check the golden set parses and that every expected document exists.

    A malformed entry does not crash anything downstream, it just quietly
    produces a wrong number, which is the one failure this project cannot ship.
    """
    _configure_logging()
    from anchor.evaluate.golden import load_golden

    settings = get_settings()
    queries = load_golden(settings.evaluate.golden_path)
    console.print(f"[bold]{len(queries)} golden queries[/bold]")

    missing: list[str] = []
    with db.connect(settings) as conn:
        commit = db.latest_ingested_commit(conn)
        for query in queries:
            for path in query.expected_source_paths:
                row = conn.execute(
                    "SELECT 1 FROM documents WHERE source_path = %s AND commit_sha = %s",
                    (path, commit),
                ).fetchone()
                if row is None:
                    missing.append(f"{query.id} -> {path}")

    if missing:
        console.print("[red]expected documents absent from the corpus:[/red]")
        for item in missing:
            console.print(f"  {item}")
        raise typer.Exit(1)
    console.print("[green]every expected document is present in the corpus[/green]")


@app.command("eval-retrieval")
def eval_retrieval_cmd(
    out: str = typer.Option("", help="Write the markdown table here as well."),
) -> None:
    """Run the recall@k sweep over the golden set and print the table.

    Writes a runs row with the full config and the git SHA, so the numbers can
    be traced back to what produced them.
    """
    _configure_logging()
    from anchor.evaluate.sweep import render_report, run_retrieval_sweep

    settings = get_settings()
    report = run_retrieval_sweep(settings)
    rendered = render_report(report, settings)

    console.print(rendered, markup=False)
    if out:
        Path(out).write_text(f"{rendered}\n", encoding="utf-8")
        console.print()
        console.print(f"wrote {out}")


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
