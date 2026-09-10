"""Answer a question and decide whether to stand behind the answer.

The product surface of everything else in this package. Generation produces a
cited answer; the grounding layer scores it; the abstention policy decides
whether it goes out. The threshold is the same config value the tradeoff curve
sweeps, so the README's curve describes the behaviour of this function, not a
separate offline calculation.

Withholding happens after generation, never instead of it. The check needs the
answer's claims to exist before it can score them, so the cost is already paid
by the time the decision is made. That is a real cost of the design and the
reason cheaper signals are used first where they exist.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from anchor import db
from anchor.config import Settings, Verdict
from anchor.generate.backends import Backend
from anchor.generate.pipeline import Answer, answer_question
from anchor.ground.attribution import Attribution, attribute_claims, support_fraction
from anchor.ground.claims import extract_claims
from anchor.ground.verify import verify_with_oracle
from anchor.index.embed import Embedder
from anchor.retrieve.search import Hit

logger = logging.getLogger(__name__)


@dataclass
class GroundedAnswer:
    """An answer plus the evidence for trusting it."""

    question: str
    answer: Answer
    attributions: list[Attribution] = field(default_factory=list)
    support: float | None = None
    oracle_contradicted: bool = False
    oracle_missing: tuple[str, ...] = ()
    withheld: bool = False
    reason: str = ""

    @property
    def text(self) -> str:
        """What the user sees.

        A withheld answer returns the abstention message, not the ungrounded
        text. Returning both would defeat the point: anything printed will be
        read, whatever caveat sits beside it.
        """
        return self.answer.text

    @property
    def spans(self) -> list[Hit]:
        return self.answer.spans

    @property
    def claim_count(self) -> int:
        return len(self.attributions)

    @property
    def supported_claims(self) -> int:
        return sum(1 for a in self.attributions if a.supported)


def grounded_answer(
    question: str,
    settings: Settings,
    embedder: Embedder | None = None,
    backend: Backend | None = None,
) -> GroundedAnswer:
    """Answer, verify, and apply the abstention policy."""
    embedder = embedder or Embedder(
        settings.index.embedding_model,
        normalize=settings.index.normalize_embeddings,
        query_instruction=settings.index.query_instruction,
    )

    answer = answer_question(question, settings, embedder, backend)
    result = GroundedAnswer(question=question, answer=answer)

    if answer.abstained:
        result.withheld = True
        result.reason = answer.abstain_reason
        return result

    claims = extract_claims(answer.text)
    result.attributions = attribute_claims(
        claims, answer.spans, embedder, settings.ground.support_threshold
    )
    result.support = support_fraction(result.attributions)

    # The oracle runs even when attribution is content, because it can veto.
    # A claim naming a flag that does not exist is wrong regardless of how
    # closely it resembles its span.
    if settings.ground.use_symbol_oracle:
        with db.connect(settings) as conn:
            row = conn.execute("SELECT DISTINCT vllm_version FROM symbols LIMIT 1").fetchone()
            if row is not None:
                verdict = verify_with_oracle(
                    conn, answer.text, settings.ground, str(row["vllm_version"])
                )
                result.oracle_contradicted = verdict.verdict is Verdict.CONTRADICTED
                result.oracle_missing = tuple(m.name for m in verdict.missing)

    if result.oracle_contradicted:
        result.withheld = True
        result.reason = (
            "the answer names something that does not exist in vLLM at this "
            f"version: {', '.join(result.oracle_missing)}"
        )
    elif result.support is None:
        result.withheld = True
        result.reason = "the answer made no checkable claim"
    elif result.support < settings.ground.abstain_threshold:
        result.withheld = True
        result.reason = (
            f"only {result.support:.0%} of claims are supported by their cited "
            f"span, below the {settings.ground.abstain_threshold:.0%} threshold"
        )

    if result.withheld:
        result.answer.text = settings.ground.abstain_message
        result.answer.abstained = True
        result.answer.abstain_reason = result.reason

    return result
