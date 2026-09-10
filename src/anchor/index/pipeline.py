"""Indexing: fill in the dense and sparse halves of every chunk row.

Both live in the same row, which is the reason this project uses one datastore.
Hybrid retrieval is then a single SQL statement with no second service to keep
in sync and no cross-store consistency problem.

Resumable by construction: both passes select only rows that are still missing
their column, so an interrupted run picks up where it stopped rather than
re-embedding the corpus.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import psycopg
from psycopg.rows import DictRow

from anchor import db
from anchor.config import Settings
from anchor.index.embed import Embedder

logger = logging.getLogger(__name__)


@dataclass
class IndexReport:
    embedded: int = 0
    tsv_built: int = 0
    already_embedded: int = 0


def build_tsvectors(conn: psycopg.Connection[DictRow], fts_config: str) -> int:
    """Build the sparse side for every chunk that lacks it.

    One statement over the whole table rather than a row at a time: this is pure
    SQL with no model in the loop, so there is no reason to pay per-row round
    trips for it.

    The config name is interpolated because a Postgres text-search
    configuration is an identifier and cannot be a bind parameter. It is
    validated against the catalogue first so the value cannot be anything but a
    configuration that exists.
    """
    exists = conn.execute("SELECT 1 FROM pg_ts_config WHERE cfgname = %s", (fts_config,)).fetchone()
    if exists is None:
        raise ValueError(f"unknown Postgres text-search configuration: {fts_config!r}")

    cursor = conn.execute(
        f"UPDATE chunks SET tsv = to_tsvector('{fts_config}', text) WHERE tsv IS NULL"
    )
    return cursor.rowcount


def embed_chunks(
    conn: psycopg.Connection[DictRow],
    embedder: Embedder,
    batch_size: int,
) -> int:
    """Embed every chunk that has no vector yet."""
    rows = conn.execute(
        "SELECT id, text FROM chunks WHERE embedding IS NULL ORDER BY id"
    ).fetchall()
    if not rows:
        return 0

    done = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        vectors = embedder.embed_passages([str(row["text"]) for row in batch], batch_size)
        with conn.cursor() as cur:
            cur.executemany(
                "UPDATE chunks SET embedding = %s WHERE id = %s",
                [(vector, int(row["id"])) for row, vector in zip(batch, vectors, strict=True)],
            )
        done += len(batch)
        # Commit per batch so an interrupted run keeps the work it finished.
        conn.commit()
        if done % (batch_size * 20) == 0:
            logger.info("embedded %d/%d chunks", done, len(rows))

    return done


def build_index(settings: Settings) -> IndexReport:
    """Fill in embeddings and tsvectors for the whole corpus."""
    report = IndexReport()
    cfg = settings.index

    embedder = Embedder(
        cfg.embedding_model,
        normalize=cfg.normalize_embeddings,
        query_instruction=cfg.query_instruction,
    )
    if embedder.dimension != cfg.embedding_dim:
        # The column is vector(384). A mismatch has to fail here, with a clear
        # message, rather than as an opaque cast error on the first insert.
        raise ValueError(
            f"{cfg.embedding_model} produces {embedder.dimension}-dim vectors "
            f"but index.embedding_dim is {cfg.embedding_dim} and the schema "
            "column is fixed at that width"
        )

    with db.connect(settings) as conn:
        row = conn.execute(
            "SELECT count(*) AS n FROM chunks WHERE embedding IS NOT NULL"
        ).fetchone()
        report.already_embedded = int(row["n"]) if row else 0

        report.tsv_built = build_tsvectors(conn, cfg.fts_config)
        conn.commit()
        logger.info("built %d tsvectors", report.tsv_built)

        report.embedded = embed_chunks(conn, embedder, cfg.embedding_batch_size)
        conn.commit()

    return report
