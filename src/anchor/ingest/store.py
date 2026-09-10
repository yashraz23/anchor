"""Write parsed documents to Postgres."""

from __future__ import annotations

import psycopg
from psycopg.rows import DictRow

from anchor.ingest.documents import ParsedDocument
from anchor.ingest.repo import RepoSnapshot

# ON CONFLICT on (source_path, commit_sha): re-running an ingest at the same
# commit is a no-op that returns the existing id, so ingest is idempotent and
# safe to re-run after an interrupted chunking pass.
_UPSERT = """
INSERT INTO documents (source_path, url, title, vllm_version, commit_sha, content_hash)
VALUES (%(source_path)s, %(url)s, %(title)s, %(vllm_version)s, %(commit_sha)s, %(content_hash)s)
ON CONFLICT (source_path, commit_sha) DO UPDATE
    SET title = EXCLUDED.title,
        url = EXCLUDED.url,
        content_hash = EXCLUDED.content_hash,
        -- Updated too, not just set on insert. Version resolution can improve
        -- between runs at the same commit (a clone without tags reports
        -- unknown, a later one resolves the tag), and a row left holding the
        -- older answer would silently poison staleness detection.
        vllm_version = EXCLUDED.vllm_version,
        fetched_at = now()
RETURNING id, (xmax = 0) AS inserted
"""


def upsert_document(
    conn: psycopg.Connection[DictRow],
    doc: ParsedDocument,
    snapshot: RepoSnapshot,
) -> tuple[int, bool]:
    """Insert or update one document. Returns its id and whether it was new.

    The `xmax = 0` trick distinguishes an insert from an update in the same
    statement, which is what lets the ingest report how much actually changed
    rather than just how many rows it touched.
    """
    row = conn.execute(
        _UPSERT,
        {
            "source_path": doc.source_path,
            "url": snapshot.blob_url(doc.source_path),
            "title": doc.title,
            "vllm_version": snapshot.vllm_version,
            "commit_sha": snapshot.commit_sha,
            "content_hash": doc.content_hash,
        },
    ).fetchone()
    if row is None:  # pragma: no cover - RETURNING always yields a row here
        raise RuntimeError(f"upsert returned no row for {doc.source_path}")
    return int(row["id"]), bool(row["inserted"])
