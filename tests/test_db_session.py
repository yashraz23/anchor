"""Session-level database settings.

Marked `db` and skipped without a database, like the other integration tests.
Guards a failure mode that only appears against a real Postgres: `SET` is parsed
before bind parameters are substituted, so the obvious spelling of this is a
syntax error no unit test would catch.
"""

from __future__ import annotations

from collections.abc import Iterator

import psycopg
import pytest
from psycopg.rows import DictRow

from anchor import db
from anchor.config import get_settings

pytestmark = pytest.mark.db


@pytest.fixture
def conn() -> Iterator[psycopg.Connection[DictRow]]:
    try:
        with db.connect(get_settings()) as connection:
            yield connection
            connection.rollback()
    except psycopg.OperationalError as exc:
        pytest.skip(f"no database reachable: {exc}")


def test_set_ef_search_applies(conn: psycopg.Connection[DictRow]) -> None:
    db.set_ef_search(conn, 137)
    row = conn.execute("SELECT current_setting('hnsw.ef_search') AS v").fetchone()
    assert row is not None
    assert row["v"] == "137"


def test_set_ef_search_accepts_the_configured_default(
    conn: psycopg.Connection[DictRow],
) -> None:
    """The value retrieval actually passes has to be accepted."""
    configured = get_settings().index.hnsw_ef_search
    db.set_ef_search(conn, configured)
    row = conn.execute("SELECT current_setting('hnsw.ef_search') AS v").fetchone()
    assert row is not None
    assert row["v"] == str(configured)
