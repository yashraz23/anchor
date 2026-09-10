"""Gather retrieval evidence for candidate questions, to support curation.

Curation has to answer one question per candidate: is this answerable from the
corpus at all? Guessing produces a golden set that measures the wrong thing in
both directions. Including an unanswerable question makes retrieval look broken
when it is not, and quietly dropping a hard-but-answerable one inflates every
score.

So this runs the real retrieval pipeline over each candidate and writes what
came back. A human then decides, with the evidence in front of them, rather than
from the title alone.

Deliberately does not decide anything itself. Auto-accepting whatever retrieval
returned would make the golden set a transcript of current behaviour, and a set
that agrees with the system by construction cannot measure it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from anchor import db
from anchor.config import Settings
from anchor.evaluate.harvest import Candidate
from anchor.index.embed import Embedder
from anchor.retrieve.search import Hit, search

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Evidence:
    """What the corpus offers for one candidate question."""

    candidate: Candidate
    hits: list[Hit]

    @property
    def best_score(self) -> float:
        return self.hits[0].score if self.hits else 0.0

    @property
    def source_paths(self) -> list[str]:
        """Distinct documents behind the hits, best first."""
        seen: list[str] = []
        for hit in self.hits:
            if hit.source_path not in seen:
                seen.append(hit.source_path)
        return seen


def question_text(candidate: Candidate) -> str:
    """The phrasing to evaluate.

    The title is preferred over the body: it is how the user summarised their own
    problem, it is short enough to embed without the surrounding stack traces and
    logs, and it is closer to what someone would actually type into a search box.
    """
    return candidate.title.strip() or candidate.question.strip()


def gather_evidence(
    candidates: list[Candidate],
    settings: Settings,
    limit: int | None = None,
) -> list[Evidence]:
    """Run retrieval for each candidate and collect the spans."""
    chosen = candidates[:limit] if limit else candidates
    embedder = Embedder(
        settings.index.embedding_model,
        normalize=settings.index.normalize_embeddings,
        query_instruction=settings.index.query_instruction,
    )

    out: list[Evidence] = []
    with db.connect(settings) as conn:
        for i, candidate in enumerate(chosen, start=1):
            question = question_text(candidate)
            if not question:
                continue
            hits = search(
                conn,
                question,
                settings,
                embedder,
                strategy=settings.chunk.strategy.value,
            )
            out.append(Evidence(candidate=candidate, hits=hits))
            if i % 20 == 0:
                logger.info("gathered evidence for %d/%d candidates", i, len(chosen))
    return out


def render_evidence(evidence: list[Evidence], span_chars: int = 240) -> str:
    """A reviewable report, one block per candidate."""
    lines: list[str] = []
    for item in evidence:
        candidate = item.candidate
        lines.append("=" * 78)
        lines.append(f"#{candidate.number}  {question_text(candidate)}")
        lines.append(f"  {candidate.url}")
        lines.append(f"  best_score={item.best_score:.4f}")
        for rank, hit in enumerate(item.hits, start=1):
            body = " ".join(hit.text.split())[:span_chars]
            lines.append(f"  [{rank}] {hit.score:.4f}  {hit.source_path}")
            lines.append(f"      {hit.heading_path or '-'}")
            lines.append(f"      {body}")
        lines.append("")
    return "\n".join(lines)
