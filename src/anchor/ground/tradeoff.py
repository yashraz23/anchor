"""The strictness-versus-completeness tradeoff.

The headline deliverable. Sweeping the abstention threshold trades how often the
system answers at all against how well-grounded its answers are, and the curve
is the result the project exists to produce.

Nothing here calls a model. Attribution scores are computed once per answer and
then re-thresholded, so the entire curve costs one pass rather than one pass per
point. That is the reason answers and their scores are persisted rather than
recomputed: a threshold sweep should be free.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from anchor.ground.attribution import Attribution, support_fraction

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScoredAnswer:
    """One answer's claims, scored once and ready to be re-thresholded."""

    query_id: str
    attributions: list[Attribution]
    # Verdict from the symbol oracle, which is threshold-independent: a flag
    # either exists or it does not, so it does not move as strictness changes.
    oracle_contradicted: bool = False

    @property
    def claim_count(self) -> int:
        return len(self.attributions)


@dataclass(frozen=True)
class TradeoffPoint:
    """The system's behaviour at one abstention threshold."""

    threshold: float
    answered: int
    abstained: int
    # Claims that went out to the user, i.e. those inside answers not withheld.
    delivered_claims: int
    # Cited a span, but that span does not support the claim. A faithfulness
    # failure in the strict sense: the answer pointed at evidence and the
    # evidence does not say it.
    unsupported_delivered: int
    # Cited nothing at all. Unattributable rather than contradicted, and a
    # different failure from the one above. Folding the two together put the
    # headline rate at 37% when the strict figure was under 1%.
    uncited_delivered: int
    contradicted_delivered: int

    @property
    def total(self) -> int:
        return self.answered + self.abstained

    @property
    def answer_rate(self) -> float | None:
        """Share of questions answered rather than withheld. Completeness."""
        return self.answered / self.total if self.total else None

    @property
    def unsupported_rate(self) -> float | None:
        """Share of delivered claims that cited a span which does not support them.

        The strict hallucination proxy. Measured over *delivered* claims only,
        because a claim inside a withheld answer never reached anyone, and
        counting it would make abstention look like it fixed nothing.
        """
        return self.unsupported_delivered / self.delivered_claims if self.delivered_claims else None

    @property
    def uncited_rate(self) -> float | None:
        """Share of delivered claims carrying no citation at all.

        Reported separately from the rate above on purpose. A claim that cites
        the wrong span asserted something the evidence contradicts; a claim with
        no citation is merely unattributable. Both are failures against the
        prompt, which required a citation on every factual claim, but they are
        not the same failure and one number cannot stand for both.
        """
        return self.uncited_delivered / self.delivered_claims if self.delivered_claims else None


def evaluate_at(
    scored: Sequence[ScoredAnswer], threshold: float, support_threshold: float
) -> TradeoffPoint:
    """Behaviour of the whole system at one abstention threshold.

    An answer is delivered when the fraction of its claims clearing
    `support_threshold` is at least `threshold`. An answer with no claims is
    withheld: there is nothing to stand behind, and delivering it would let an
    empty response count as a fully grounded one.
    """
    answered = abstained = 0
    delivered_claims = unsupported = uncited = contradicted = 0

    for item in scored:
        fraction = support_fraction(item.attributions)
        deliver = fraction is not None and fraction >= threshold

        if deliver:
            answered += 1
            delivered_claims += item.claim_count
            unsupported += sum(
                1
                for a in item.attributions
                if a.claim.is_cited and not a.supported and not a.claim.about_context
            )
            uncited += sum(
                1 for a in item.attributions if not a.claim.is_cited and not a.claim.about_context
            )
            if item.oracle_contradicted:
                contradicted += item.claim_count
        else:
            abstained += 1

    return TradeoffPoint(
        threshold=threshold,
        answered=answered,
        abstained=abstained,
        delivered_claims=delivered_claims,
        unsupported_delivered=unsupported,
        uncited_delivered=uncited,
        contradicted_delivered=contradicted,
    )


def sweep(
    scored: Sequence[ScoredAnswer],
    thresholds: Sequence[float],
    support_threshold: float,
) -> list[TradeoffPoint]:
    """The curve: one point per abstention threshold."""
    return [evaluate_at(scored, t, support_threshold) for t in thresholds]


def format_rate(value: float | None) -> str:
    """Render a rate, or a placeholder when it does not exist.

    A rate over zero delivered claims is undefined, not zero. Printing 0% would
    read as "no unsupported claims", which is the opposite of what total
    abstention means.
    """
    return "-" if value is None else f"{value:.0%}"


def render_curve(points: Sequence[TradeoffPoint]) -> str:
    """The tradeoff table, as Markdown ready for the README."""
    lines = [
        "| Abstention threshold | Answered | Abstained | Answer rate | "
        "Delivered claims | Cited but unsupported | Uncited |",
        "|---|---|---|---|---|---|---|",
    ]
    for point in points:
        lines.append(
            f"| {point.threshold:.2f} | {point.answered} | {point.abstained} | "
            f"{format_rate(point.answer_rate)} | {point.delivered_claims} | "
            f"{point.unsupported_delivered} ({format_rate(point.unsupported_rate)}) | "
            f"{point.uncited_delivered} ({format_rate(point.uncited_rate)}) |"
        )
    return "\n".join(lines)
