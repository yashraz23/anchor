"""LLM-as-judge over (claim, cited span) pairs.

Span attribution measures similarity, which is a proxy for support. This asks
the question directly: does this span actually say this? It is the only verifier
that can catch a claim sitting close to its span while inverting its meaning,
which is precisely the failure similarity cannot see.

It runs second, and only where the oracle has nothing to say. The oracle is
exact and free; the judge is neither, so it is spent on what is left.

One call per answer rather than one per claim. The spans are shared across an
answer's claims, so per-claim calls would resend the same context many times
over for no extra signal.

Every verdict is persisted with `verifier = 'llm_judge'`. Phase 2 distils this
model into a local one, and these rows are its training data, accumulated as a
byproduct of work already done.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from anchor.config import Settings, Verdict, cost_usd
from anchor.ground.claims import Claim
from anchor.retrieve.search import Hit
from anchor.track import record_generation

logger = logging.getLogger(__name__)

JUDGE_SYSTEM = """You check whether a numbered source span supports a claim \
taken from an answer about vLLM.

For each claim you are given the span it cited. Judge only whether that span \
supports it. Do not use anything you know about vLLM.

Verdicts:
- supported: the span states the claim, or states something the claim follows \
directly from.
- unsupported: the span does not state the claim. It says nothing either way.
- contradicted: the span states something incompatible with the claim. Reserve \
this for a real conflict, not for absence.
- unverifiable: the claim is not a factual assertion that a span could support, \
for example a remark about the evidence itself.

"Unsupported" and "contradicted" are different and the distinction matters. A \
span that is merely silent is not a contradiction.

Be strict about specifics. If a claim names a flag, a default or a number that \
the span does not state, it is unsupported even if the surrounding topic \
matches."""


class ClaimJudgement(BaseModel):
    """One verdict, as returned by the judge."""

    ordinal: int = Field(description="The claim number being judged.")
    verdict: Literal["supported", "unsupported", "contradicted", "unverifiable"]
    reason: str = Field(description="One short sentence citing what the span says.")


class JudgeResponse(BaseModel):
    judgements: list[ClaimJudgement]


@dataclass(frozen=True)
class JudgeResult:
    """Verdicts for one answer, plus what the call cost."""

    verdicts: dict[int, Verdict]
    reasons: dict[int, str]
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def cost(self) -> float | None:
        return None


class Judge(Protocol):
    def judge(self, claims: list[Claim], spans: list[Hit]) -> JudgeResult: ...


def build_judge_prompt(claims: list[Claim], spans: list[Hit]) -> str:
    """Render the claims and the spans each one cited.

    Only cited spans are shown per claim. Showing all of them would let the
    judge find support somewhere else in the context and mark a
    wrongly-attributed claim as supported, which is exactly the error the
    citation requirement exists to surface.
    """
    lines: list[str] = []
    for claim in claims:
        lines.append(f"Claim {claim.ordinal}: {claim.text}")
        if not claim.cited_spans:
            lines.append("  (this claim cited no span)")
        for number in claim.cited_spans:
            index = number - 1
            if 0 <= index < len(spans):
                lines.append(f"  Cited span [{number}]: {spans[index].text}")
            else:
                lines.append(f"  Cited span [{number}]: (no such span was supplied)")
        lines.append("")
    return "\n".join(lines)


class AnthropicJudge:
    """The judge backed by the Anthropic API, using structured outputs.

    Structured outputs rather than free-text parsing: the verdict set is closed,
    and a judge whose output occasionally fails to parse would drop claims from
    the denominator silently.
    """

    def __init__(self, settings: Settings) -> None:
        import anthropic

        self.settings = settings
        key = settings.anthropic_api_key
        self._client = (
            anthropic.Anthropic(api_key=key.get_secret_value())
            if key is not None
            else anthropic.Anthropic()
        )

    def judge(self, claims: list[Claim], spans: list[Hit]) -> JudgeResult:
        if not claims:
            return JudgeResult(verdicts={}, reasons={})

        prompt = build_judge_prompt(claims, spans)
        started = time.perf_counter()
        response = self._client.messages.parse(
            model=self.settings.ground.judge_model,
            max_tokens=self.settings.generate.max_tokens,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_format=JudgeResponse,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        parsed = response.parsed_output
        verdicts: dict[int, Verdict] = {}
        reasons: dict[int, str] = {}
        if parsed is not None:
            for judgement in parsed.judgements:
                verdicts[judgement.ordinal] = Verdict(judgement.verdict)
                reasons[judgement.ordinal] = judgement.reason

        result = JudgeResult(
            verdicts=verdicts,
            reasons=reasons,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        # Judge calls are traced alongside answer calls on purpose. The judge is
        # the most expensive model in the project and the one whose behaviour is
        # hardest to argue about from a summary statistic; its prompts are worth
        # being able to read one by one.
        record_generation(
            self.settings,
            name="judge",
            model=self.settings.ground.judge_model,
            prompt=prompt,
            output=repr(parsed) if parsed is not None else "",
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=latency_ms,
            cost_usd=judge_cost(result, self.settings.ground.judge_model),
            metadata={"claims": len(claims), "spans": len(spans)},
        )
        return result


def judge_cost(result: JudgeResult, model: str) -> float | None:
    return cost_usd(model, result.input_tokens, result.output_tokens)


def agreement(
    judge_verdicts: dict[int, Verdict], attribution_supported: dict[int, bool]
) -> tuple[int, int, int]:
    """How often the judge and span attribution agree.

    Returns (agree, judge_stricter, attribution_stricter). Attribution has only
    two outcomes, so the judge's `contradicted` and `unsupported` both count as
    "not supported" for the comparison. The point is not to score the judge but
    to put a number on how much the cheap check misses.
    """
    agree = judge_stricter = attribution_stricter = 0
    for ordinal, verdict in judge_verdicts.items():
        if ordinal not in attribution_supported:
            continue
        judge_says_supported = verdict is Verdict.SUPPORTED
        cheap_says_supported = attribution_supported[ordinal]
        if judge_says_supported == cheap_says_supported:
            agree += 1
        elif cheap_says_supported:
            judge_stricter += 1
        else:
            attribution_stricter += 1
    return agree, judge_stricter, attribution_stricter
