"""Run the code-block integrity metric over the ingested corpus."""

from __future__ import annotations

import logging

from anchor import db
from anchor.config import ChunkStrategy, Settings
from anchor.evaluate.chunk_integrity import (
    BUCKETS,
    IntegrityResult,
    bucket_for,
    chunk_texts_for,
    code_block_bodies,
)
from anchor.ingest.pipeline import IngestReport, parse_corpus
from anchor.ingest.repo import prepare_checkout
from anchor.tokenizer import HFTokenizer

logger = logging.getLogger(__name__)


def measure_integrity(
    settings: Settings, strategies: list[ChunkStrategy] | None = None
) -> list[IntegrityResult]:
    """Code-block integrity per strategy, over every ingested document.

    Both strategies are measured against the same documents and the same code
    blocks, which is the whole reason both are persisted at once.
    """
    targets = strategies or list(ChunkStrategy)

    with db.connect(settings) as conn:
        commit = settings.ingest.vllm_commit or db.latest_ingested_commit(conn)
        if commit is None:
            raise RuntimeError("no documents ingested; run `anchor ingest` first")

        snapshot = prepare_checkout(settings.ingest, commit=commit)
        report = IngestReport(commit_sha=commit, vllm_version=snapshot.vllm_version)

        tokenizer = HFTokenizer(settings.chunk.tokenizer_model)
        totals = dict.fromkeys(targets, 0)
        intact = dict.fromkeys(targets, 0)
        by_size: dict[ChunkStrategy, dict[str, list[int]]] = {
            s: {name: [0, 0] for name, _, _ in BUCKETS} for s in targets
        }

        for doc in parse_corpus(snapshot, settings, report):
            bodies = code_block_bodies(doc.text)
            if not bodies:
                continue
            row = conn.execute(
                "SELECT id FROM documents WHERE source_path = %s AND commit_sha = %s",
                (doc.source_path, commit),
            ).fetchone()
            if row is None:
                continue
            document_id = int(row["id"])
            # Bucket each body once; the two strategies see identical blocks.
            buckets = [bucket_for(tokenizer.count(body)) for body in bodies]

            for strategy in targets:
                texts = chunk_texts_for(conn, document_id, strategy.value)
                totals[strategy] += len(bodies)
                for body, bucket in zip(bodies, buckets, strict=True):
                    survived = any(body in text for text in texts)
                    by_size[strategy][bucket][0] += 1
                    if survived:
                        intact[strategy] += 1
                        by_size[strategy][bucket][1] += 1

    return [
        IntegrityResult(
            strategy=strategy.value,
            code_blocks=totals[strategy],
            intact=intact[strategy],
            by_size={name: (counts[0], counts[1]) for name, counts in by_size[strategy].items()},
        )
        for strategy in targets
    ]
