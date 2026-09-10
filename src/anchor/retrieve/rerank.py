"""Cross-encoder reranking.

The bi-encoder scores a query and a passage independently and compares the two
vectors, so it never sees how they interact. A cross-encoder reads the pair
together in one forward pass, which is why it is more precise and also why it
cannot run over a corpus: cost is one pass per candidate, not one lookup.

So it runs last, over the fused shortlist only. That ordering is the entire
argument for the stage, and it is the reason `fused_top_k` and `rerank_top_n`
are separate knobs: the first sets how much work the reranker does, the second
how much context the generator gets.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from functools import lru_cache
from typing import TYPE_CHECKING

from anchor.config import Settings

if TYPE_CHECKING:  # pragma: no cover
    from anchor.retrieve.search import Hit

logger = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def _load(model_name: str):  # type: ignore[no-untyped-def]
    """Load and cache the cross-encoder. Loading dwarfs scoring a shortlist."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def rerank(query: str, hits: list[Hit], settings: Settings) -> list[Hit]:
    """Re-order `hits` by cross-encoder relevance, most relevant first.

    The returned Hits carry the cross-encoder score in place of the retriever
    or fusion score. Those scales are unrelated, and keeping the old number
    beside a new ordering is how a results table ends up lying.
    """
    if not hits:
        return []

    model = _load(settings.retrieve.rerank_model)
    scores = model.predict(
        [(query, hit.text) for hit in hits],
        batch_size=settings.retrieve.rerank_batch_size,
        show_progress_bar=False,
    )
    rescored = [replace(hit, score=float(score)) for hit, score in zip(hits, scores, strict=True)]
    # Ties break on chunk id so a run is reproducible from its config.
    rescored.sort(key=lambda h: (-h.score, h.chunk_id))
    return rescored
