"""The abstention tradeoff.

This is the project's headline result, so the definitions matter more than the
arithmetic. Two in particular: what counts as a delivered claim, and what an
undefined rate renders as.
"""

from __future__ import annotations

from anchor.ground.attribution import Attribution
from anchor.ground.claims import Claim
from anchor.ground.tradeoff import (
    ScoredAnswer,
    evaluate_at,
    format_rate,
    render_curve,
    sweep,
)


def _attr(supported: bool, cited: bool = True, about_context: bool = False) -> Attribution:
    return Attribution(
        claim=Claim(
            ordinal=0,
            text="a claim",
            cited_spans=(1,) if cited else (),
            about_context=about_context,
        ),
        best_span=1 if cited else None,
        score=0.9 if supported else 0.1,
        supported=supported,
    )


def _answer(qid: str, supported: int, unsupported: int, **kw: bool) -> ScoredAnswer:
    return ScoredAnswer(
        query_id=qid,
        attributions=[_attr(True)] * supported + [_attr(False)] * unsupported,
        **kw,
    )


def test_a_fully_supported_answer_is_delivered_at_every_threshold() -> None:
    scored = [_answer("q", supported=3, unsupported=0)]
    for t in (0.0, 0.5, 1.0):
        assert evaluate_at(scored, t, 0.55).answered == 1


def test_a_partly_supported_answer_is_withheld_as_strictness_rises() -> None:
    """Two of three claims supported: delivered at 0.6, withheld at 0.8."""
    scored = [_answer("q", supported=2, unsupported=1)]
    assert evaluate_at(scored, 0.6, 0.55).answered == 1
    assert evaluate_at(scored, 0.8, 0.55).abstained == 1


def test_an_answer_with_no_claims_is_withheld() -> None:
    """Delivering it would let an empty response count as fully grounded."""
    point = evaluate_at([_answer("q", supported=0, unsupported=0)], 0.0, 0.55)
    assert point.abstained == 1
    assert point.answered == 0


def test_an_uncited_claim_is_reported_separately_from_a_wrongly_cited_one() -> None:
    """Folding the two together put the headline rate at 37% when the strict
    figure was under 1%. A claim citing the wrong span asserted something the
    evidence contradicts; a claim citing nothing is merely unattributable."""
    scored = [
        ScoredAnswer(
            query_id="q",
            attributions=[_attr(False, cited=True), _attr(False, cited=False)],
        )
    ]
    point = evaluate_at(scored, 0.0, 0.55)
    assert point.unsupported_delivered == 1
    assert point.uncited_delivered == 1


def test_a_remark_about_the_evidence_counts_as_neither() -> None:
    """It asserts nothing about vLLM, so it is not a faithfulness failure."""
    scored = [
        ScoredAnswer(
            query_id="q",
            attributions=[_attr(True), _attr(False, cited=False, about_context=True)],
        )
    ]
    point = evaluate_at(scored, 0.0, 0.55)
    assert point.unsupported_delivered == 0
    assert point.uncited_delivered == 0


def test_unsupported_claims_are_counted_only_when_delivered() -> None:
    """A claim inside a withheld answer never reached anyone. Counting it would
    make abstention look like it fixed nothing."""
    scored = [_answer("q", supported=1, unsupported=3)]
    delivered = evaluate_at(scored, 0.0, 0.55)
    withheld = evaluate_at(scored, 0.9, 0.55)
    assert delivered.unsupported_delivered == 3
    assert withheld.unsupported_delivered == 0


def test_answer_rate_and_unsupported_rate() -> None:
    scored = [
        _answer("a", supported=2, unsupported=0),
        _answer("b", supported=0, unsupported=2),
    ]
    point = evaluate_at(scored, 0.5, 0.55)
    assert point.answered == 1
    assert point.answer_rate == 0.5
    assert point.unsupported_rate == 0.0


def test_strictness_trades_against_completeness() -> None:
    """The shape of the curve is the whole result: answering less, but with a
    smaller share of unsupported claims among what is delivered."""
    scored = [
        _answer("a", supported=3, unsupported=0),
        _answer("b", supported=1, unsupported=2),
    ]
    loose = evaluate_at(scored, 0.0, 0.55)
    strict = evaluate_at(scored, 0.9, 0.55)

    assert loose.answered > strict.answered
    assert loose.unsupported_rate is not None and strict.unsupported_rate is not None
    assert loose.unsupported_rate > strict.unsupported_rate


def test_total_abstention_leaves_the_rate_undefined_not_zero() -> None:
    """0% would read as 'no unsupported claims', the opposite of what total
    abstention means."""
    point = evaluate_at([_answer("q", supported=0, unsupported=2)], 1.0, 0.55)
    assert point.answered == 0
    assert point.unsupported_rate is None
    assert format_rate(point.unsupported_rate) == "-"


def test_oracle_contradiction_is_recorded_when_delivered() -> None:
    scored = [_answer("q", supported=2, unsupported=0, oracle_contradicted=True)]
    assert evaluate_at(scored, 0.0, 0.55).contradicted_delivered == 2


def test_sweep_returns_one_point_per_threshold() -> None:
    points = sweep([_answer("q", supported=1, unsupported=1)], [0.0, 0.5, 1.0], 0.55)
    assert [p.threshold for p in points] == [0.0, 0.5, 1.0]


def test_render_curve_is_markdown() -> None:
    table = render_curve(sweep([_answer("q", supported=1, unsupported=0)], [0.5], 0.55))
    lines = table.splitlines()
    assert lines[0].startswith("| Abstention threshold |")
    assert lines[1].startswith("|---|")
    assert "0.50" in lines[2]
