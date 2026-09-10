"""Dense, sparse, and hybrid retrieval over pgvector plus Postgres full text."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import psycopg
from psycopg.rows import DictRow

from anchor import db
from anchor.config import RetrievalMode, Settings
from anchor.index.embed import Embedder
from anchor.retrieve.fusion import reciprocal_rank_fusion

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Hit:
    chunk_id: int
    text: str
    score: float
    heading_path: str | None
    is_code_block: bool
    source_path: str
    url: str | None
    vllm_version: str

    def cite(self) -> str:
        """How this span is labelled when handed to the generator."""
        where = self.heading_path or self.source_path
        return f"{where} ({self.vllm_version})"


# `<=>` is cosine distance in pgvector, so smaller is closer. Vectors are
# normalised at write time, which is what makes cosine the right operator and
# lets the HNSW index built with vector_cosine_ops actually get used.
_DENSE = """
SELECT c.id, c.text, c.heading_path, c.is_code_block,
       d.source_path, d.url, d.vllm_version,
       1 - (c.embedding <=> %(query)s) AS score
FROM chunks c
JOIN documents d ON d.id = c.document_id
WHERE c.strategy = %(strategy)s AND c.embedding IS NOT NULL
ORDER BY c.embedding <=> %(query)s
LIMIT %(limit)s
"""

# websearch_to_tsquery over plainto_tsquery: it tolerates the punctuation real
# questions carry (quotes, "or", a stray hyphen) instead of erroring or
# silently dropping the query.
_SPARSE = """
SELECT c.id, c.text, c.heading_path, c.is_code_block,
       d.source_path, d.url, d.vllm_version,
       ts_rank_cd(c.tsv, websearch_to_tsquery(%(config)s, %(query)s)) AS score
FROM chunks c
JOIN documents d ON d.id = c.document_id
WHERE c.strategy = %(strategy)s
  AND c.tsv @@ websearch_to_tsquery(%(config)s, %(query)s)
ORDER BY score DESC, c.id
LIMIT %(limit)s
"""


def _to_hit(row: DictRow) -> Hit:
    return Hit(
        chunk_id=int(row["id"]),
        text=str(row["text"]),
        score=float(row["score"]),
        heading_path=row["heading_path"],
        is_code_block=bool(row["is_code_block"]),
        source_path=str(row["source_path"]),
        url=row["url"],
        vllm_version=str(row["vllm_version"]),
    )


def dense_search(
    conn: psycopg.Connection[DictRow],
    query_vector: np.ndarray,
    strategy: str,
    limit: int,
    ef_search: int,
) -> list[Hit]:
    db.set_ef_search(conn, ef_search)
    rows = conn.execute(
        _DENSE, {"query": query_vector, "strategy": strategy, "limit": limit}
    ).fetchall()
    return [_to_hit(row) for row in rows]


def sparse_search(
    conn: psycopg.Connection[DictRow],
    query: str,
    strategy: str,
    limit: int,
    fts_config: str,
) -> list[Hit]:
    rows = conn.execute(
        _SPARSE,
        {"query": query, "strategy": strategy, "limit": limit, "config": fts_config},
    ).fetchall()
    return [_to_hit(row) for row in rows]


def search(
    conn: psycopg.Connection[DictRow],
    query: str,
    settings: Settings,
    embedder: Embedder,
    strategy: str,
    mode: RetrievalMode | None = None,
) -> list[Hit]:
    """Retrieve for one query under the configured mode.

    Hybrid fuses the two lists with RRF and then, when reranking is on, hands
    the fused top-k to a cross-encoder. The bi-encoder scores query and passage
    independently, so it cannot see their interaction; the cross-encoder reads
    both together and is the only stage that can. It is also far too slow to run
    over the corpus, which is exactly why it runs last over a short list.
    """
    cfg = settings.retrieve
    mode = mode or cfg.mode

    if mode is RetrievalMode.DENSE:
        hits = dense_search(
            conn,
            embedder.embed_query(query),
            strategy,
            cfg.dense_top_k,
            settings.index.hnsw_ef_search,
        )
    elif mode is RetrievalMode.SPARSE:
        hits = sparse_search(conn, query, strategy, cfg.sparse_top_k, settings.index.fts_config)
    else:
        dense = dense_search(
            conn,
            embedder.embed_query(query),
            strategy,
            cfg.dense_top_k,
            settings.index.hnsw_ef_search,
        )
        sparse = sparse_search(conn, query, strategy, cfg.sparse_top_k, settings.index.fts_config)
        by_id = {hit.chunk_id: hit for hit in [*dense, *sparse]}
        fused = reciprocal_rank_fusion(
            {
                "dense": [h.chunk_id for h in dense],
                "sparse": [h.chunk_id for h in sparse],
            },
            k=cfg.rrf_k,
            top_k=cfg.fused_top_k,
        )
        # The RRF score replaces the retriever scores: they are not comparable
        # to each other and the fused ranking is what downstream stages consume.
        hits = [Hit(**{**by_id[f.chunk_id].__dict__, "score": f.score}) for f in fused]

    if cfg.use_rerank and mode is not RetrievalMode.SPARSE:
        from anchor.retrieve.rerank import rerank

        hits = rerank(query, hits, settings)

    return hits[: cfg.rerank_top_n] if cfg.use_rerank else hits
