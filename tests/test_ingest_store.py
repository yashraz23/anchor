"""Store tests against a live Postgres.

Marked `db` and skipped when no database is reachable, so a laptop without
Docker running still gets a green local suite. CI always has the service, so
these always run there.

Every test rolls back, leaving the corpus untouched.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
from psycopg.rows import DictRow

from anchor import db
from anchor.config import get_settings
from anchor.ingest.documents import ParsedDocument, content_hash
from anchor.ingest.repo import RepoSnapshot
from anchor.ingest.store import upsert_document

pytestmark = pytest.mark.db

SHA = "1111111111111111111111111111111111111111"


@pytest.fixture
def conn() -> Iterator[psycopg.Connection[DictRow]]:
    settings = get_settings()
    try:
        with db.connect(settings) as connection:
            yield connection
            connection.rollback()
    except psycopg.OperationalError as exc:
        pytest.skip(f"no database reachable: {exc}")


def _doc(text: str = "Pass --max-num-seqs to cap concurrent sequences.") -> ParsedDocument:
    return ParsedDocument(
        source_path="docs/test_only_engine_args.md",
        title="Engine Arguments",
        text=text,
        content_hash=content_hash(text),
    )


def _snapshot(version: str = "v0.28.1rc0") -> RepoSnapshot:
    return RepoSnapshot(
        root=Path(),
        commit_sha=SHA,
        vllm_version=version,
        repo_url="https://github.com/vllm-project/vllm.git",
    )


def test_upsert_reports_insert_then_update(conn: psycopg.Connection[DictRow]) -> None:
    """Re-ingesting the same commit must not duplicate rows."""
    first_id, inserted = upsert_document(conn, _doc(), _snapshot())
    assert inserted is True

    second_id, inserted_again = upsert_document(conn, _doc(), _snapshot())
    assert second_id == first_id
    assert inserted_again is False


def test_upsert_refreshes_the_version_on_conflict(conn: psycopg.Connection[DictRow]) -> None:
    """A row must not keep a stale version once resolution improves.

    A clone fetched without tags reports 'unknown'; a later run resolves the
    real tag. Leaving the first answer in place would silently poison staleness
    detection, since every chunk inherits this field through document_id.
    """
    doc_id, _ = upsert_document(conn, _doc(), _snapshot(version="unknown"))
    upsert_document(conn, _doc(), _snapshot(version="v0.28.1rc0"))

    row = conn.execute("SELECT vllm_version FROM documents WHERE id = %s", (doc_id,)).fetchone()
    assert row is not None
    assert row["vllm_version"] == "v0.28.1rc0"


def test_upsert_records_a_commit_pinned_url(conn: psycopg.Connection[DictRow]) -> None:
    doc_id, _ = upsert_document(conn, _doc(), _snapshot())
    row = conn.execute("SELECT url FROM documents WHERE id = %s", (doc_id,)).fetchone()
    assert row is not None
    assert row["url"] == (
        f"https://github.com/vllm-project/vllm/blob/{SHA}/docs/test_only_engine_args.md"
    )


def test_the_same_path_at_two_commits_is_two_rows(conn: psycopg.Connection[DictRow]) -> None:
    """The corpus is versioned, so one path exists once per commit ingested."""
    other = RepoSnapshot(
        root=Path(),
        commit_sha="2222222222222222222222222222222222222222",
        vllm_version="v0.27.0",
        repo_url="https://github.com/vllm-project/vllm.git",
    )
    first_id, _ = upsert_document(conn, _doc(), _snapshot())
    second_id, inserted = upsert_document(conn, _doc(), other)
    assert inserted is True
    assert second_id != first_id
