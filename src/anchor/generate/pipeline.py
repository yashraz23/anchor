"""Answer one question: retrieve, assemble, generate.

Grounding verification lives in `ground/` and runs over the result of this. The
split matters: generation must not be allowed to decide whether its own answer
was supported, and keeping the two apart is what stops that creeping in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from anchor import db
from anchor.config import Settings
from anchor.generate.backends import Backend, Completion, make_backend
from anchor.generate.prompt import build_prompt, invalid_citations, parse_citations
from anchor.index.embed import Embedder
from anchor.retrieve.search import Hit, search
from anchor.tokenizer import HFTokenizer

logger = logging.getLogger(__name__)


@dataclass
class Answer:
    """One generated answer and everything needed to audit it."""

    question: str
    text: str
    spans: list[Hit] = field(default_factory=list)
    completion: Completion | None = None
    abstained: bool = False
    abstain_reason: str = ""

    @property
    def cited(self) -> set[int]:
        return parse_citations(self.text)

    @property
    def invalid_cited(self) -> set[int]:
        """Cited span numbers that were never supplied."""
        return invalid_citations(self.text, len(self.spans))

    @property
    def retrieved_chunk_ids(self) -> list[int]:
        return [hit.chunk_id for hit in self.spans]

    @property
    def uncited_spans(self) -> set[int]:
        """Supplied spans the answer never cited, 1-based.

        Not a fault on its own. It is a signal about retrieval depth: if most
        spans go uncited across the golden set, rerank_top_n is larger than the
        answer needs and the extra context is only costing input tokens.
        """
        return set(range(1, len(self.spans) + 1)) - self.cited


def answer_question(
    question: str,
    settings: Settings,
    embedder: Embedder | None = None,
    backend: Backend | None = None,
) -> Answer:
    """Retrieve context and generate a cited answer.

    Abstains without calling the model when retrieval returns nothing. There is
    no answer to be had from an empty context, and paying for a token to be told
    so is waste.
    """
    embedder = embedder or Embedder(
        settings.index.embedding_model,
        normalize=settings.index.normalize_embeddings,
        query_instruction=settings.index.query_instruction,
    )
    tokenizer = HFTokenizer(settings.chunk.tokenizer_model)

    with db.connect(settings) as conn:
        hits = search(conn, question, settings, embedder, strategy=settings.chunk.strategy.value)

    if not hits:
        return Answer(
            question=question,
            text=settings.ground.abstain_message,
            abstained=True,
            abstain_reason="retrieval returned no spans",
        )

    prompt = build_prompt(question, hits, tokenizer, settings.generate.max_context_tokens)
    backend = backend or make_backend(settings)
    completion = backend.complete(prompt.system, prompt.user)

    if completion.stop_reason == "refusal":
        # HTTP 200 with no usable text. Recorded as an abstention rather than
        # an empty answer, so it cannot be scored as a grounded response.
        return Answer(
            question=question,
            text=settings.ground.abstain_message,
            spans=list(prompt.spans),
            completion=completion,
            abstained=True,
            abstain_reason="the model declined the request",
        )

    answer = Answer(
        question=question,
        text=completion.text,
        spans=list(prompt.spans),
        completion=completion,
    )
    if answer.invalid_cited:
        # A model citing [7] when six spans were supplied invented a source.
        logger.warning(
            "answer cites spans that were never supplied: %s", sorted(answer.invalid_cited)
        )
    return answer
