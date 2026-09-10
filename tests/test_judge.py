"""LLM judge: prompt construction and agreement accounting.

No API calls. What is tested is the part that decides what the judge is allowed
to see and how its verdicts are compared against the cheap check, both of which
are policy rather than model behaviour.
"""

from __future__ import annotations

from anchor.config import Verdict
from anchor.ground.claims import Claim
from anchor.ground.judge import agreement, build_judge_prompt
from anchor.retrieve.search import Hit


def _claim(ordinal: int, text: str, cited: tuple[int, ...]) -> Claim:
    return Claim(ordinal=ordinal, text=text, cited_spans=cited)


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


def test_the_prompt_pairs_a_claim_with_the_span_it_cited() -> None:
    prompt = build_judge_prompt(
        [_claim(0, "It caps the batch size.", (1,))],
        [_span("max_num_seqs caps sequences per iteration.")],
    )
    assert "Claim 0: It caps the batch size." in prompt
    assert "Cited span [1]: max_num_seqs caps" in prompt


def test_only_cited_spans_are_shown() -> None:
    """Showing all of them would let the judge find support elsewhere in the
    context and pass a wrongly-attributed claim, which is the exact error the
    citation requirement exists to surface."""
    prompt = build_judge_prompt(
        [_claim(0, "A claim.", (1,))],
        [_span("the cited one"), _span("an unrelated span")],
    )
    assert "the cited one" in prompt
    assert "an unrelated span" not in prompt


def test_an_uncited_claim_is_marked_as_such() -> None:
    prompt = build_judge_prompt([_claim(0, "A claim.", ())], [_span("x")])
    assert "cited no span" in prompt


def test_a_citation_to_a_missing_span_is_shown_as_missing() -> None:
    """The judge must not be handed a silently empty span for an invented
    citation; it would read as absence of evidence rather than a bad citation."""
    prompt = build_judge_prompt([_claim(0, "A claim.", (9,))], [_span("x")])
    assert "no such span was supplied" in prompt


def test_several_claims_are_numbered_in_the_prompt() -> None:
    prompt = build_judge_prompt(
        [_claim(0, "First.", (1,)), _claim(1, "Second.", (1,))], [_span("s")]
    )
    assert "Claim 0:" in prompt
    assert "Claim 1:" in prompt


# --------------------------------------------------------------------------- #
# agreement                                                                    #
# --------------------------------------------------------------------------- #
def test_agreement_counts_matches() -> None:
    judge = {0: Verdict.SUPPORTED, 1: Verdict.UNSUPPORTED}
    cheap = {0: True, 1: False}
    assert agreement(judge, cheap) == (2, 0, 0)


def test_the_judge_can_be_stricter_than_similarity() -> None:
    """The case similarity cannot see: a claim close to its span that inverts
    its meaning."""
    assert agreement({0: Verdict.UNSUPPORTED}, {0: True}) == (0, 1, 0)


def test_contradicted_counts_as_not_supported_for_the_comparison() -> None:
    """Attribution has only two outcomes, so the judge's finer verdicts collapse
    when the two are compared."""
    assert agreement({0: Verdict.CONTRADICTED}, {0: True}) == (0, 1, 0)


def test_similarity_can_be_stricter_than_the_judge() -> None:
    assert agreement({0: Verdict.SUPPORTED}, {0: False}) == (0, 0, 1)


def test_claims_the_cheap_check_never_scored_are_skipped() -> None:
    assert agreement({0: Verdict.SUPPORTED, 5: Verdict.SUPPORTED}, {0: True}) == (
        1,
        0,
        0,
    )


def test_agreement_on_nothing() -> None:
    assert agreement({}, {}) == (0, 0, 0)
