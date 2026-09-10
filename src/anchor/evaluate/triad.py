"""The RAG triad: faithfulness, answer relevance, context relevance.

Implemented here rather than with `ragas`, which the spec originally named. Every
published ragas version, up to 0.4.3, hard-requires `langchain`,
`langchain-community`, `langchain-openai` and `openai`. The project's stated
constraint is that LangChain does not enter the tree, because the retrieval
logic being visible is the point. The pinned version also failed to import, on a
`langchain_community.chat_models.vertexai` incompatibility.

So the triad is computed directly. The three metrics are a page of prompt each,
the judge machinery already exists, and the result is that every number in the
README comes from code in this repository rather than from a framework whose
version drift silently changes it.

The faithfulness leg is not recomputed here: the grounding layer already
measures it against cited spans, and with the symbol oracle it does so more
strictly than a generic judge could. This module supplies the other two legs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from anchor.config import Settings, cost_usd
from anchor.retrieve.search import Hit

logger = logging.getLogger(__name__)

TRIAD_SYSTEM = """You score how well a retrieval system served one question \
about vLLM.

You are given the question, the answer, and the numbered context spans that were \
retrieved for it.

Score two things:

1. answer_relevance: does the answer address the question that was asked?
   - 1.0 it answers the question directly
   - 0.5 it addresses the topic but not the specific question
   - 0.0 it answers a different question, or says nothing useful

   An honest "the documentation does not cover this" is NOT irrelevant. If the \
answer correctly reports that the context cannot answer the question, score it \
1.0. Penalising that would reward guessing.

2. span_relevance: for each span, whether it is relevant to the question.
   A span is relevant if it contains information that helps answer the question, \
even partially. Judge each span on its own."""


class SpanRelevance(BaseModel):
    span: int = Field(description="The span number.")
    relevant: bool


class TriadScores(BaseModel):
    answer_relevance: Literal["0.0", "0.5", "1.0"]
    spans: list[SpanRelevance]
    reason: str = Field(description="One short sentence.")


@dataclass(frozen=True)
class TriadResult:
    """Scores for one question."""

    answer_relevance: float
    relevant_spans: int
    total_spans: int
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def context_relevance(self) -> float | None:
        """Share of retrieved spans that were useful.

        None when nothing was retrieved. Zero spans is not zero relevance, it is
        an absent measurement, and averaging it in as 0.0 would blame the
        reranker for a retrieval failure upstream.
        """
        return self.relevant_spans / self.total_spans if self.total_spans else None


def build_triad_prompt(question: str, answer: str, spans: list[Hit]) -> str:
    parts = [f"Question: {question}", "", f"Answer: {answer}", "", "Context spans:"]
    for i, span in enumerate(spans, start=1):
        parts.append(f"[{i}] {span.text}")
    return "\n".join(parts)


class TriadScorer:
    """Scores the two legs the grounding layer does not already cover."""

    def __init__(self, settings: Settings) -> None:
        import anthropic

        self.settings = settings
        key = settings.anthropic_api_key
        self._client = (
            anthropic.Anthropic(api_key=key.get_secret_value())
            if key is not None
            else anthropic.Anthropic()
        )

    def score(self, question: str, answer: str, spans: list[Hit]) -> TriadResult:
        response = self._client.messages.parse(
            model=self.settings.evaluate.ragas_model,
            max_tokens=self.settings.generate.max_tokens,
            system=TRIAD_SYSTEM,
            messages=[{"role": "user", "content": build_triad_prompt(question, answer, spans)}],
            output_format=TriadScores,
        )
        parsed = response.parsed_output
        if parsed is None:
            return TriadResult(
                answer_relevance=0.0,
                relevant_spans=0,
                total_spans=len(spans),
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )

        relevant = sum(1 for s in parsed.spans if s.relevant and 1 <= s.span <= len(spans))
        return TriadResult(
            answer_relevance=float(parsed.answer_relevance),
            relevant_spans=relevant,
            total_spans=len(spans),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


def mean_or_none(values: list[float | None]) -> float | None:
    """Mean of the values that exist, or None when none do."""
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def triad_cost(results: list[TriadResult], model: str) -> float | None:
    total_in = sum(r.input_tokens for r in results)
    total_out = sum(r.output_tokens for r in results)
    return cost_usd(model, total_in, total_out)
