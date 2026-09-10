"""Split an answer into atomic claims, each carrying the spans it cited.

Extraction is deterministic rather than model-driven. The generator was already
required to cite a span number after every factual claim, so the (claim, span)
pairs the grounding layer needs are present in the text and do not have to be
re-derived by a second model.

That buys three things. Runs are reproducible, since nothing here samples. It
costs nothing, so re-scoring a stored answer under a different threshold is
free. And the judge is not asked to both invent the claims and rule on them,
which would let one model's segmentation choices silently set the denominator of
the faithfulness metric.

The cost is granularity: a sentence asserting two things gets one verdict. That
is a real limitation and it is why `claims` stores the sentence text, so a
disagreement can be traced back to what was actually judged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from anchor.generate.prompt import _FENCED_CODE, parse_citations

# Abbreviations that end in a period without ending a sentence. Short and
# specific: a broad list would suppress real boundaries.
_ABBREVIATIONS = ("e.g.", "i.e.", "etc.", "cf.", "vs.", "approx.")

# A sentence boundary: terminal punctuation, then whitespace, then something
# that starts a new sentence. Markdown emphasis and inline code count, since
# answers routinely open a sentence with `**bold**` or a backticked name.
_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[*`\"'-]|\*\*)")

# Markdown list and heading markers, stripped so a bullet reads as a claim
# rather than as punctuation.
_LIST_MARKER = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_HEADING = re.compile(r"^\s*#{1,6}\s+")

# Sentences whose subject is the retrieved context rather than vLLM: "the spans
# do not cover this", "I cannot say from the provided context". These are the
# model declining, which is the behaviour the abstention policy exists to
# encourage. Counting them as unsupported claims would score honest hedging as
# hallucination and push the system toward confident guessing.
_CONTEXT_REMARK = re.compile(
    r"\b(?:provided\s+|retrieved\s+|given\s+)?"
    r"(?:spans?|context|documentation provided)\b",
    re.I,
)
_FIRST_PERSON_LIMIT = re.compile(
    r"\b(?:I(?:'d| would)? (?:can'?t|cannot|could not|couldn'?t|need|rather not)"
    r"|there is nothing in|I do not have|I don'?t have)\b",
    re.I,
)


@dataclass(frozen=True)
class Claim:
    """One atomic assertion from an answer."""

    ordinal: int
    text: str
    cited_spans: tuple[int, ...]
    # True when the sentence talks about the evidence rather than about vLLM.
    about_context: bool = False

    @property
    def is_cited(self) -> bool:
        """Whether the claim attributes itself to any span.

        An uncited claim is not automatically false, but it is unattributable by
        construction: the generator was told to cite every factual claim, so its
        failing to do so is itself the signal.
        """
        return bool(self.cited_spans)


def _protect_abbreviations(text: str) -> str:
    """Hide periods that do not end a sentence, so the splitter ignores them."""
    out = text
    for abbreviation in _ABBREVIATIONS:
        out = out.replace(abbreviation, abbreviation.replace(".", "\x00"))
    # A period between digits is a decimal or a version, never a boundary:
    # "25.6k" and "vLLM 0.28.1" both appear in real answers.
    return re.sub(r"(?<=\d)\.(?=\d)", "\x00", out)


def _restore(text: str) -> str:
    return text.replace("\x00", ".")


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences, tolerating the shapes answers actually use."""
    pieces: list[str] = []
    for block in text.split("\n\n"):
        for line in block.splitlines():
            stripped = _HEADING.sub("", _LIST_MARKER.sub("", line)).strip()
            if not stripped:
                continue
            protected = _protect_abbreviations(stripped)
            pieces.extend(_restore(part).strip() for part in _BOUNDARY.split(protected))
    return [p for p in pieces if p]


def is_substantive(sentence: str) -> bool:
    """Whether a fragment carries an assertion worth verifying.

    Filters out the debris of markdown prose: a lone bold heading, a stray
    bullet, a fragment left behind by a stripped code block. Counting those as
    claims would inflate the denominator of every faithfulness rate with text
    that asserts nothing.
    """
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", sentence)
    if len(words) < 4:
        return False
    # A short fragment ending in a colon introduces the next thing rather than
    # asserting anything: "What the spans do say:", "Notes:".
    return not (sentence.rstrip().endswith(":") and len(words) <= 8)


def is_context_remark(sentence: str) -> bool:
    """Whether a sentence is about the evidence rather than about vLLM.

    "The provided spans do not cover S3 credentials" asserts nothing about vLLM;
    it is the model declining, which is what the abstention policy is for.
    Treating it as an unsupported claim would score honest hedging as
    hallucination and reward confident guessing instead.
    """
    return bool(_CONTEXT_REMARK.search(sentence) or _FIRST_PERSON_LIMIT.search(sentence))


def extract_claims(answer: str) -> list[Claim]:
    """Every substantive claim in an answer, with the spans it cited.

    Fenced code blocks are removed: a block is an illustration, not an
    assertion, and asking whether a snippet is "supported" is not a question a
    verifier can answer. Inline code is *kept*, because it carries the meaning
    of the sentence around it. Stripping it left claims like "pass your own stat
    loggers into  ." with the subject deleted.

    Citations are parsed by parse_citations, which does its own code stripping,
    so keeping inline code here cannot reintroduce `output[0]` as a citation.
    """
    claims: list[Claim] = []
    for sentence in split_sentences(_FENCED_CODE.sub(" ", answer)):
        if not is_substantive(sentence):
            continue
        claims.append(
            Claim(
                ordinal=len(claims),
                text=sentence,
                cited_spans=tuple(sorted(parse_citations(sentence))),
                about_context=is_context_remark(sentence),
            )
        )
    return claims
