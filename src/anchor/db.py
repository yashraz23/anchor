"""Postgres access.

One datastore holds both the dense and the sparse side of retrieval, so there is
no second service to keep in sync and hybrid search is a single SQL statement.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import resources
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import DictRow, dict_row

from anchor.config import Settings


@contextmanager
def connect(settings: Settings) -> Iterator[psycopg.Connection[DictRow]]:
    """A connection with the pgvector type adapters registered.

    Without register_vector, a Python list round-trips as a Postgres array and
    the vector cast fails at query time rather than at write time.
    """
    with psycopg.connect(settings.pg_dsn, row_factory=dict_row) as conn:
        register_vector(conn)
        yield conn


def schema_sql() -> str:
    """The schema DDL, read from the packaged file."""
    return resources.files("anchor").joinpath("schema.sql").read_text(encoding="utf-8")


def init_schema(settings: Settings) -> None:
    """Apply the schema. Idempotent."""
    with connect(settings) as conn:
        conn.execute(schema_sql())
        conn.commit()


def set_ef_search(conn: psycopg.Connection[DictRow], ef_search: int) -> None:
    """Set the HNSW search-time breadth for this session.

    ef_search trades recall against latency and is a sweep dimension, so it is
    set per session from config rather than baked into the index.
    """
    conn.execute("SET LOCAL hnsw.ef_search = %s", (ef_search,))


def git_sha(repo_root: Path | None = None) -> str:
    """The current commit, or 'unknown' outside a git checkout.

    Every run records this. A results table that cannot be traced back to a
    commit is not reproducible.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return out.stdout.strip()
