"""Prompt assembly.

Context spans are numbered and the model is required to cite the numbers inline.
That requirement is not cosmetic: it is what makes the grounding layer possible
at all. A claim carrying a span number can be checked against that span; a claim
with no citation is, by construction, a claim the model could not attribute.

Everything here is pure. The prompt is the single largest lever on answer
quality, so it has to be inspectable in a test rather than reconstructed by
reading API calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from anchor.retrieve.search import Hit
from anchor.tokenizer import Tokenizer

# A citation is a bracketed span number, or several separated by commas: [2] or
# [1, 3]. Deliberately strict: matching bare numbers would collide with the
# version strings and shapes that fill this corpus.
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

SYSTEM_PROMPT = """You answer questions about vLLM using only the numbered \
context spans provided.

Rules:
- Every factual claim you make must cite the span it came from, inline, as [1] \
or [2, 3]. Put the citation immediately after the claim it supports.
- Use only the spans. Do not use anything you know about vLLM that is not in \
them, even if you are confident it is correct.
- If the spans do not contain the answer, say so plainly and do not guess. A \
short honest non-answer is worth more than a plausible invented one.
- Quote exact flag names, config keys and defaults from the spans rather than \
paraphrasing them.
- Be concise. Answer the question that was asked."""


@dataclass(frozen=True)
class Prompt:
    """An assembled prompt, plus the spans that survived the context budget."""

    system: str
    user: str
    spans: tuple[Hit, ...]

    @property
    def span_count(self) -> int:
        return len(self.spans)


def format_span(index: int, hit: Hit) -> str:
    """One numbered span, labelled with where it came from.

    The label carries the vLLM version, so a model reading two spans from
    different releases can say which one it used, and staleness stays visible
    in the answer rather than being flattened away.
    """
    return f"[{index}] {hit.cite()}\n{hit.text}"


def fit_spans(hits: list[Hit], tokenizer: Tokenizer, max_context_tokens: int) -> list[Hit]:
    """Take spans in rank order until the context budget is spent.

    Truncating the last span instead would be worse: a half-shown code block is
    exactly the failure the structure-aware chunker exists to avoid, and it
    would be reintroduced here at the final step. A span either fits whole or is
    dropped.

    Later spans are still considered after one is dropped, since a small span
    further down may fit where a large one did not, and dropping it too would
    discard context for nothing.
    """
    kept: list[Hit] = []
    used = 0
    for hit in hits:
        cost = tokenizer.count(hit.text)
        if used + cost > max_context_tokens:
            continue
        kept.append(hit)
        used += cost
    return kept


def build_prompt(
    question: str,
    hits: list[Hit],
    tokenizer: Tokenizer,
    max_context_tokens: int,
) -> Prompt:
    """Assemble the prompt for one question."""
    spans = fit_spans(hits, tokenizer, max_context_tokens)
    if spans:
        body = "\n\n".join(format_span(i, hit) for i, hit in enumerate(spans, start=1))
    else:
        # Said explicitly rather than left as an empty section. An empty context
        # block reads as a formatting bug to the model; this reads as a fact.
        body = "(no context spans were retrieved)"

    user = f"Context spans:\n\n{body}\n\nQuestion: {question}"
    return Prompt(system=SYSTEM_PROMPT, user=user, spans=tuple(spans))


def parse_citations(answer: str) -> set[int]:
    """Span numbers cited anywhere in the answer.

    Used by the grounding layer to route each claim to the span it names, and to
    notice claims that cite nothing at all.
    """
    found: set[int] = set()
    for match in _CITATION.finditer(answer):
        for part in match.group(1).split(","):
            found.add(int(part.strip()))
    return found


def invalid_citations(answer: str, span_count: int) -> set[int]:
    """Cited span numbers that do not exist.

    A model citing [7] when six spans were supplied has invented a source. That
    is a hallucination of a particularly checkable kind, so it is worth
    detecting here rather than waiting for the judge.
    """
    return {n for n in parse_citations(answer) if n < 1 or n > span_count}
