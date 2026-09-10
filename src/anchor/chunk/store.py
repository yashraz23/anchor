"""Write chunks to Postgres."""

from __future__ import annotations

import psycopg
from psycopg.rows import DictRow

from anchor.chunk.base import Chunk

_INSERT = """
INSERT INTO chunks
    (document_id, strategy, ordinal, text, heading_path, is_code_block, token_count)
VALUES
    (%(document_id)s, %(strategy)s, %(ordinal)s, %(text)s, %(heading_path)s,
     %(is_code_block)s, %(token_count)s)
"""


def document_id_for(
    conn: psycopg.Connection[DictRow], source_path: str, commit_sha: str
) -> int | None:
    """The document row for one file at one commit, if it was ingested."""
    row = conn.execute(
        "SELECT id FROM documents WHERE source_path = %s AND commit_sha = %s",
        (source_path, commit_sha),
    ).fetchone()
    return int(row["id"]) if row else None


def replace_chunks(
    conn: psycopg.Connection[DictRow],
    document_id: int,
    strategy: str,
    chunks: list[Chunk],
) -> int:
    """Replace this document's chunks under one strategy. Returns how many.

    Delete-then-insert rather than upsert on (document_id, strategy, ordinal).
    Re-chunking at a different token budget produces a different number of
    chunks, and an upsert would leave the tail of the previous run behind as
    orphan rows that are still retrievable. The other strategy's chunks are
    untouched, so the two can be rebuilt independently.
    """
    conn.execute(
        "DELETE FROM chunks WHERE document_id = %s AND strategy = %s",
        (document_id, strategy),
    )
    for chunk in chunks:
        conn.execute(
            _INSERT,
            {
                "document_id": document_id,
                "strategy": strategy,
                "ordinal": chunk.ordinal,
                "text": chunk.text,
                "heading_path": chunk.heading_path,
                "is_code_block": chunk.is_code_block,
                "token_count": chunk.token_count,
            },
        )
    return len(chunks)
