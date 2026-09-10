"""Span attribution: does the cited span actually support the claim?

A citation is an assertion by the generator, not evidence. This scores it. Each
claim is embedded and compared against the spans it cited, using the same
encoder as retrieval, so the score lives on a scale the rest of the system
already speaks.

What this measures, stated precisely because the abstention policy is built on
it: semantic similarity between a claim and its cited span. That is a proxy for
support, not support itself. A claim can be close to its span and still invert
its meaning, which is exactly the case the LLM judge exists to catch. Attribution
is the cheap first pass that decides where the expensive judge is worth
spending.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from anchor.ground.claims import Claim
from anchor.index.embed import PassageEncoder
from anchor.retrieve.search import Hit

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Attribution:
    """A claim, its best supporting span, and how well that span matches."""

    claim: Claim
    # 1-based span number, matching the citation numbering the answer used.
    best_span: int | None
    score: float
    # None when the claim cited nothing, so there was no span to score against.
    supported: bool = False


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two vectors.

    Computed explicitly rather than assuming normalisation. Claim vectors come
    from the same encoder as the corpus, but relying on that invariant here
    would make this function silently wrong if it were ever reused.
    """
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def attribute_claims(
    claims: list[Claim],
    spans: list[Hit],
    embedder: PassageEncoder,
    support_threshold: float,
) -> list[Attribution]:
    """Score every claim against the spans it cited.

    Only cited spans are scored, not all of them. Searching every span for the
    best match would quietly repair the generator's citation errors: a claim
    that cites the wrong span would score well against some other one and pass,
    and the citation requirement would stop meaning anything.
    """
    if not claims:
        return []

    claim_vectors = embedder.embed_passages([c.text for c in claims], batch_size=32)
    span_vectors = (
        embedder.embed_passages([s.text for s in spans], batch_size=32)
        if spans
        else np.empty((0, embedder.dimension), dtype=np.float32)
    )

    results: list[Attribution] = []
    for claim, vector in zip(claims, claim_vectors, strict=True):
        best_span: int | None = None
        best_score = 0.0

        for number in claim.cited_spans:
            index = number - 1
            if not 0 <= index < len(span_vectors):
                # A citation to a span that was never supplied. Left unscored;
                # the answer-level check already reports it, and inventing a
                # score here would hide it.
                continue
            score = cosine(vector, span_vectors[index])
            if best_span is None or score > best_score:
                best_span, best_score = number, score

        results.append(
            Attribution(
                claim=claim,
                best_span=best_span,
                score=best_score,
                supported=best_span is not None and best_score >= support_threshold,
            )
        )
    return results


def support_fraction(attributions: list[Attribution]) -> float | None:
    """Share of judgeable claims whose cited span clears the support threshold.

    Sentences about the evidence itself are excluded from the denominator. "The
    spans do not cover this" asserts nothing about vLLM, and counting it as
    unsupported would score the model's honesty as hallucination.

    None when nothing is judgeable. An answer with nothing to check is not fully
    supported, and scoring it 1.0 would let an empty answer look perfect to the
    abstention policy.
    """
    judged = [a for a in attributions if not a.claim.about_context]
    if not judged:
        return None
    return sum(1 for a in judged if a.supported) / len(judged)
