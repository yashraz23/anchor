"""The RAG triad's two remaining legs.

No API calls. What matters here is how the scores are defined, particularly the
two cases where an absent measurement must not be scored as a bad one.
"""

from __future__ import annotations

from anchor.evaluate.triad import (
    TRIAD_SYSTEM,
    TriadResult,
    build_triad_prompt,
    mean_or_none,
)
from anchor.retrieve.search import Hit


def _span(text: str) -> Hit:
    return Hit(
        chunk_id=1,
        text=text,
        score=1.0,
        heading_path=None,
        is_code_block=False,
        source_path="docs/x.md",
        url=None,
        vllm_version="v0.28.1rc0",
    )


def test_context_relevance_is_a_fraction_of_retrieved_spans() -> None:
    result = TriadResult(answer_relevance=1.0, relevant_spans=2, total_spans=5)
    assert result.context_relevance == 0.4


def test_context_relevance_with_nothing_retrieved_is_none_not_zero() -> None:
    """Zero spans is an absent measurement, not zero relevance. Averaging it in
    as 0.0 would blame the reranker for a retrieval failure upstream."""
    result = TriadResult(answer_relevance=0.0, relevant_spans=0, total_spans=0)
    assert result.context_relevance is None


def test_the_prompt_contains_question_answer_and_numbered_spans() -> None:
    prompt = build_triad_prompt("Why?", "Because [1].", [_span("a span"), _span("b")])
    assert "Question: Why?" in prompt
    assert "Answer: Because [1]." in prompt
    assert "[1] a span" in prompt
    assert "[2] b" in prompt


def test_the_rubric_does_not_penalise_an_honest_non_answer() -> None:
    """Scoring "the docs do not cover this" as irrelevant would reward
    guessing, which is the opposite of what the abstention policy is for."""
    assert "does not cover this" in TRIAD_SYSTEM
    assert "reward guessing" in TRIAD_SYSTEM


def test_mean_ignores_absent_values() -> None:
    assert mean_or_none([1.0, None, 0.0]) == 0.5


def test_mean_of_nothing_present_is_none() -> None:
    assert mean_or_none([None, None]) is None
    assert mean_or_none([]) is None
