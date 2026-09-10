"""Sync the golden set into Postgres and resolve expected chunk ids.

The YAML file is the source of truth and stores document paths. This resolves
those paths to chunk ids for one (strategy, commit) pair and writes both into
`golden_queries`, so `answers` has something to reference and so the ids exist
where a SQL query can reach them.

Resolution happens here rather than in the file because chunk ids change every
time anything is re-chunked. Committing them would produce a version-controlled
file that silently goes stale, which is the failure mode this whole project is
about.
"""

from __future__ import annotations

import logging

import psycopg
from psycopg.rows import DictRow

from anchor.config import Settings
from anchor.evaluate.golden import GoldenQuery, load_golden

logger = logging.getLogger(__name__)

_UPSERT = """
INSERT INTO golden_queries
    (question, source_url, expected_answer, expected_chunk_ids, category, difficulty)
VALUES (%(question)s, %(source_url)s, %(expected_answer)s, %(expected_chunk_ids)s,
        %(category)s, %(difficulty)s)
ON CONFLICT (question) DO UPDATE
    SET source_url = EXCLUDED.source_url,
        expected_answer = EXCLUDED.expected_answer,
        expected_chunk_ids = EXCLUDED.expected_chunk_ids,
        category = EXCLUDED.category,
        difficulty = EXCLUDED.difficulty
RETURNING id
"""


def resolve_chunk_ids(
    conn: psycopg.Connection[DictRow],
    query: GoldenQuery,
    strategy: str,
    commit_sha: str,
) -> list[int]:
    """Every chunk id belonging to this query's expected documents."""
    if not query.expected_source_paths:
        return []
    rows = conn.execute(
        """
        SELECT c.id
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE d.source_path = ANY(%s) AND d.commit_sha = %s AND c.strategy = %s
        ORDER BY c.id
        """,
        (list(query.expected_source_paths), commit_sha, strategy),
    ).fetchall()
    return [int(row["id"]) for row in rows]


def sync_golden(settings: Settings) -> dict[str, int]:
    """Write the golden set into Postgres. Returns question id by golden id."""
    from anchor import db

    queries = load_golden(settings.evaluate.golden_path)
    ids: dict[str, int] = {}

    with db.connect(settings) as conn:
        commit = db.latest_ingested_commit(conn)
        if commit is None:
            raise RuntimeError("no documents ingested; run `anchor ingest` first")
        strategy = settings.chunk.strategy.value

        for query in queries:
            chunk_ids = resolve_chunk_ids(conn, query, strategy, commit)
            if not chunk_ids:
                # Loud, because it silently zeroes this question's recall and
                # would otherwise look like a retrieval failure.
                logger.warning("%s resolved to no chunks under strategy %s", query.id, strategy)
            row = conn.execute(
                _UPSERT,
                {
                    "question": query.question,
                    "source_url": query.source_url,
                    "expected_answer": query.expected_answer,
                    "expected_chunk_ids": chunk_ids,
                    "category": query.category.value,
                    "difficulty": query.difficulty.value,
                },
            ).fetchone()
            if row is not None:
                ids[query.id] = int(row["id"])
        conn.commit()

    return ids
