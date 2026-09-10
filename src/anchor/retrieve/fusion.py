"""Reciprocal Rank Fusion.

    score(d) = sum over lists of 1 / (k + rank(d))

Chosen over weighted score fusion because it needs no normalisation between a
cosine similarity and a BM25-style rank, and no tuned weights to defend. It
consumes only the order of each list, never the scores, so the two retrievers
cannot be put on a common scale incorrectly, because they are never put on one
at all.

`k` damps the influence of the very top ranks: with k = 60 the gap between rank
1 and rank 2 is small, so one list cannot dominate the fusion on its own. That
is the value from the original paper and it is a config knob, not a constant.

Pure and dependency-free on purpose. Fusion is the easiest part of a hybrid
retriever to get subtly wrong, so it is tested directly on rank lists rather
than only through the database.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Fused:
    chunk_id: int
    score: float
    # Rank this document held in each input list, for the ones it appeared in.
    # Kept for debugging a surprising fusion, which is otherwise opaque.
    ranks: dict[str, int]


def reciprocal_rank_fusion(
    ranked_lists: Mapping[str, Sequence[int]],
    k: int,
    top_k: int | None = None,
) -> list[Fused]:
    """Fuse ranked lists of chunk ids into one ranking.

    Ranks are 1-based: the first element of each list is rank 1. Documents
    missing from a list contribute nothing from it rather than a penalty, which
    is what lets a document found by only one retriever still surface.

    Ties break on the lower chunk id, so a fusion is deterministic and a run is
    reproducible from its config.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    scores: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}

    for source, ids in ranked_lists.items():
        for position, chunk_id in enumerate(ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + position)
            ranks.setdefault(chunk_id, {})[source] = position

    fused = [
        Fused(chunk_id=chunk_id, score=score, ranks=ranks[chunk_id])
        for chunk_id, score in scores.items()
    ]
    fused.sort(key=lambda f: (-f.score, f.chunk_id))
    return fused[:top_k] if top_k is not None else fused


def recall_at_k(retrieved: Iterable[int], relevant: Iterable[int], k: int) -> float | None:
    """Fraction of relevant chunks found in the top k.

    None when nothing is relevant, rather than 0.0 or 1.0. A query with no
    ground truth is unanswerable by this metric, and averaging it in as either
    value would quietly move the headline number.
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return None
    top = list(retrieved)[:k]
    return len(relevant_set.intersection(top)) / len(relevant_set)
