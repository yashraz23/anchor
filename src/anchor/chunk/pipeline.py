"""Chunking orchestration.

Both strategies are written by default. They have to exist side by side in the
same database, over the same documents, or the recall comparison between them is
not like-for-like. `chunk.strategy` in config selects which one *retrieval*
reads; it does not decide which ones get built.

Document text is not stored in `documents`, so this re-parses the checkout
rather than reading text back out of Postgres. That keeps one parser in the
system instead of two, and the parse is cheap next to embedding.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from anchor import db
from anchor.chunk.base import Chunk
from anchor.chunk.fixed import chunk_fixed
from anchor.chunk.store import document_id_for, replace_chunks
from anchor.chunk.structure import chunk_structure_aware
from anchor.config import ChunkStrategy, Settings
from anchor.ingest.pipeline import IngestReport, parse_corpus
from anchor.ingest.repo import prepare_checkout
from anchor.tokenizer import HFTokenizer, Tokenizer

logger = logging.getLogger(__name__)


@dataclass
class ChunkReport:
    commit_sha: str
    documents: int = 0
    missing_documents: int = 0
    chunks_by_strategy: dict[str, int] = field(default_factory=dict)
    code_chunks_by_strategy: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.chunks_by_strategy.values())


def build_chunks(
    text: str, strategy: ChunkStrategy, settings: Settings, tokenizer: Tokenizer
) -> list[Chunk]:
    """Apply one strategy. The only place the two are dispatched between."""
    cfg = settings.chunk
    if strategy is ChunkStrategy.FIXED:
        return chunk_fixed(
            text,
            tokenizer,
            target_tokens=cfg.fixed_target_tokens,
            overlap_tokens=cfg.fixed_overlap_tokens,
        )
    return chunk_structure_aware(
        text,
        tokenizer,
        max_tokens=cfg.max_tokens,
        min_tokens=cfg.min_tokens,
        prepend_heading_path=cfg.prepend_heading_path,
        never_split_code_blocks=cfg.never_split_code_blocks,
    )


def chunk_corpus(settings: Settings, strategies: list[ChunkStrategy] | None = None) -> ChunkReport:
    """Chunk every ingested document under each strategy."""
    targets = strategies or list(ChunkStrategy)

    # Pin to what was ingested, never to the tip of the branch.
    with db.connect(settings) as conn:
        commit = settings.ingest.vllm_commit or db.latest_ingested_commit(conn)
    if commit is None:
        raise RuntimeError("no documents ingested; run `anchor ingest` first")

    snapshot = prepare_checkout(settings.ingest, commit=commit)
    tokenizer = HFTokenizer(settings.chunk.tokenizer_model)

    report = ChunkReport(commit_sha=snapshot.commit_sha)
    for strategy in targets:
        report.chunks_by_strategy[strategy.value] = 0
        report.code_chunks_by_strategy[strategy.value] = 0

    # parse_corpus reports on its own work; chunking only needs the documents.
    ingest_report = IngestReport(commit_sha=snapshot.commit_sha, vllm_version=snapshot.vllm_version)

    with db.connect(settings) as conn:
        for doc in parse_corpus(snapshot, settings, ingest_report):
            document_id = document_id_for(conn, doc.source_path, snapshot.commit_sha)
            if document_id is None:
                # Ingest has not been run at this commit, or was interrupted.
                report.missing_documents += 1
                continue

            report.documents += 1
            for strategy in targets:
                chunks = build_chunks(doc.text, strategy, settings, tokenizer)
                replace_chunks(conn, document_id, strategy.value, chunks)
                report.chunks_by_strategy[strategy.value] += len(chunks)
                report.code_chunks_by_strategy[strategy.value] += sum(
                    1 for c in chunks if c.is_code_block
                )

            if report.documents % 200 == 0:
                # Commit periodically so a long run is resumable and does not
                # hold one transaction open across the whole corpus.
                conn.commit()
                logger.info("chunked %d documents", report.documents)

        conn.commit()

    return report
