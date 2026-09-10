"""Prompt assembly and citation parsing.

The prompt is the largest single lever on answer quality, so it is asserted in
tests rather than reconstructed by reading API calls. Nothing here touches the
network: every function under test is pure.
"""

from __future__ import annotations

from anchor.generate.prompt import (
    SYSTEM_PROMPT,
    build_prompt,
    fit_spans,
    format_span,
    invalid_citations,
    parse_citations,
)
from anchor.retrieve.search import Hit
from tests.conftest import FakeTokenizer


def _hit(text: str, path: str = "docs/x.md", chunk_id: int = 1) -> Hit:
    return Hit(
        chunk_id=chunk_id,
        text=text,
        score=0.9,
        heading_path="Engine Arguments > GPU memory",
        is_code_block=False,
        source_path=path,
        url=f"https://github.com/vllm-project/vllm/blob/abc/{path}",
        vllm_version="v0.28.1rc0",
    )


# --------------------------------------------------------------------------- #
# span formatting                                                              #
# --------------------------------------------------------------------------- #
def test_span_is_numbered_and_labelled_with_its_version() -> None:
    """The version travels with the span so staleness stays visible in the
    answer rather than being flattened away."""
    out = format_span(3, _hit("some text"))
    assert out.startswith("[3] Engine Arguments > GPU memory (v0.28.1rc0)")
    assert "some text" in out


def test_spans_are_numbered_from_one() -> None:
    prompt = build_prompt("q", [_hit("a"), _hit("b")], FakeTokenizer(), max_context_tokens=100)
    assert "[1] " in prompt.user
    assert "[2] " in prompt.user
    assert "[0] " not in prompt.user


# --------------------------------------------------------------------------- #
# context budget                                                               #
# --------------------------------------------------------------------------- #
def test_fit_spans_drops_whole_spans_rather_than_truncating() -> None:
    """Truncating the last span would reintroduce the exact failure the
    structure-aware chunker exists to prevent, at the final step."""
    tokenizer = FakeTokenizer()
    hits = [_hit("one two three"), _hit("four five six")]
    kept = fit_spans(hits, tokenizer, max_context_tokens=3)
    assert len(kept) == 1
    assert kept[0].text == "one two three"


def test_fit_spans_keeps_a_later_span_that_still_fits() -> None:
    """Dropping everything after the first overflow would discard context for
    nothing when a smaller span further down would have fitted."""
    tokenizer = FakeTokenizer()
    hits = [_hit("a"), _hit("way too many tokens here to fit"), _hit("b")]
    kept = fit_spans(hits, tokenizer, max_context_tokens=3)
    assert [h.text for h in kept] == ["a", "b"]


def test_fit_spans_within_budget_keeps_everything() -> None:
    hits = [_hit("a"), _hit("b")]
    assert fit_spans(hits, FakeTokenizer(), max_context_tokens=100) == hits


def test_build_prompt_reports_the_spans_that_survived() -> None:
    prompt = build_prompt("q", [_hit("a"), _hit("much longer span here")], FakeTokenizer(), 1)
    assert prompt.span_count == 1
    assert prompt.spans[0].text == "a"


# --------------------------------------------------------------------------- #
# empty context                                                                #
# --------------------------------------------------------------------------- #
def test_no_spans_is_stated_rather_than_left_blank() -> None:
    """An empty context block reads as a formatting bug to the model."""
    prompt = build_prompt("q", [], FakeTokenizer(), 100)
    assert "(no context spans were retrieved)" in prompt.user
    assert prompt.span_count == 0


def test_the_question_is_in_the_prompt() -> None:
    prompt = build_prompt("what is enforce_eager", [_hit("a")], FakeTokenizer(), 100)
    assert "what is enforce_eager" in prompt.user


def test_system_prompt_demands_citation_and_permits_refusal() -> None:
    """Both rules are load-bearing for the grounding layer downstream."""
    assert "cite" in SYSTEM_PROMPT.lower()
    assert "do not guess" in SYSTEM_PROMPT.lower()


# --------------------------------------------------------------------------- #
# citations                                                                    #
# --------------------------------------------------------------------------- #
def test_parse_a_single_citation() -> None:
    assert parse_citations("Set it to 0.9 [2].") == {2}


def test_parse_a_grouped_citation() -> None:
    assert parse_citations("Both flags matter [1, 3].") == {1, 3}
    assert parse_citations("Tight grouping [1,3].") == {1, 3}


def test_parse_several_citations() -> None:
    assert parse_citations("First [1]. Second [2]. Again [1].") == {1, 2}


def test_parse_ignores_bare_numbers() -> None:
    """This corpus is full of version strings and shapes; matching unbracketed
    numbers would produce citations nobody wrote."""
    assert parse_citations("vLLM 0.28.1 supports tensor shape 4, 8.") == set()


def test_parse_on_an_answer_with_no_citations() -> None:
    assert parse_citations("I could not ground this answer.") == set()


def test_invalid_citations_catches_an_invented_source() -> None:
    """Citing [7] when six spans were supplied is a checkable hallucination,
    so it is caught here rather than left to the judge."""
    assert invalid_citations("As shown [7].", span_count=6) == {7}
    assert invalid_citations("As shown [6].", span_count=6) == set()


def test_invalid_citations_catches_zero() -> None:
    assert invalid_citations("See [0].", span_count=3) == {0}
