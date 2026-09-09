"""The `anchor` command line.

Thin wrappers only. Every subcommand resolves settings once and hands them to a
library function, so that anything reachable from the CLI is equally reachable
from a test or a sweep.
"""

from __future__ import annotations

import json

import typer
from rich.console import Console

from anchor import db
from anchor.config import get_settings

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


@app.command("config")
def show_config() -> None:
    """Print the run config exactly as it would be written to runs.config_json."""
    console.print_json(json.dumps(get_settings().run_config()))


if __name__ == "__main__":
    app()
