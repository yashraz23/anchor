"""Span attribution.

A fake embedder keeps these offline and deterministic. What is tested here is
the attribution *policy*, not the encoder: which spans get scored, what happens
when a citation is missing or invalid, and how the support fraction is defined.
Those decisions are what the abstention threshold acts on.
"""

from __future__ import annotations

import numpy as np

from anchor.ground.attribution import (
    Attribution,
    attribute_claims,
    cosine,
    support_fraction,
)
from anchor.ground.claims import Claim
from anchor.retrieve.search import Hit


class FakeEmbedder:
    """Embeds text by a keyword, so similarity is controllable by hand."""

    @property
    def dimension(self) -> int:
        return 3

    def embed_passages(self, texts: list[str], batch_size: int) -> np.ndarray:
        out = []
        for text in texts:
            if "alpha" in text:
                out.append([1.0, 0.0, 0.0])
            elif "beta" in text:
                out.append([0.0, 1.0, 0.0])
            else:
                out.append([0.0, 0.0, 1.0])
        return np.array(out, dtype=np.float32)


def _claim(text: str, cited: tuple[int, ...], ordinal: int = 0) -> Claim:
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


# --------------------------------------------------------------------------- #
# cosine                                                                       #
# --------------------------------------------------------------------------- #
def test_cosine_of_identical_vectors() -> None:
    v = np.array([1.0, 2.0, 3.0])
    assert cosine(v, v) == 1.0


def test_cosine_of_orthogonal_vectors() -> None:
    assert cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 0.0


def test_cosine_of_a_zero_vector_is_zero_not_nan() -> None:
    """A NaN here would propagate into the support fraction and quietly
    poison the abstention decision."""
    assert cosine(np.array([0.0, 0.0]), np.array([1.0, 0.0])) == 0.0


# --------------------------------------------------------------------------- #
# attribution                                                                  #
# --------------------------------------------------------------------------- #
def test_a_claim_matching_its_cited_span_is_supported() -> None:
    results = attribute_claims(
        [_claim("alpha claim", (1,))], [_span("alpha span")], FakeEmbedder(), 0.55
    )
    assert results[0].supported
    assert results[0].best_span == 1


def test_a_claim_citing_an_unrelated_span_is_not_supported() -> None:
    results = attribute_claims(
        [_claim("alpha claim", (1,))], [_span("beta span")], FakeEmbedder(), 0.55
    )
    assert not results[0].supported


def test_only_cited_spans_are_scored() -> None:
    """Searching all spans for the best match would quietly repair the
    generator's citation errors and make the citation requirement meaningless."""
    results = attribute_claims(
        [_claim("alpha claim", (2,))],
        [_span("alpha span"), _span("beta span")],
        FakeEmbedder(),
        0.55,
    )
    assert results[0].best_span == 2
    assert not results[0].supported


def test_the_best_of_several_cited_spans_wins() -> None:
    results = attribute_claims(
        [_claim("alpha claim", (1, 2))],
        [_span("beta span"), _span("alpha span")],
        FakeEmbedder(),
        0.55,
    )
    assert results[0].best_span == 2
    assert results[0].supported


def test_an_uncited_claim_has_no_span_and_is_unsupported() -> None:
    results = attribute_claims(
        [_claim("alpha claim", ())], [_span("alpha span")], FakeEmbedder(), 0.55
    )
    assert results[0].best_span is None
    assert not results[0].supported


def test_a_citation_to_a_span_that_does_not_exist_is_left_unscored() -> None:
    """Inventing a score would hide an invalid citation the answer-level check
    already reports."""
    results = attribute_claims(
        [_claim("alpha claim", (9,))], [_span("alpha span")], FakeEmbedder(), 0.55
    )
    assert results[0].best_span is None
    assert not results[0].supported


def test_no_claims_yields_no_attributions() -> None:
    assert attribute_claims([], [_span("alpha")], FakeEmbedder(), 0.55) == []


def test_the_threshold_is_applied() -> None:
    """The knob the whole tradeoff curve is swept over."""
    claims = [_claim("alpha claim", (1,))]
    spans = [_span("alpha span")]
    assert attribute_claims(claims, spans, FakeEmbedder(), 0.99)[0].supported
    assert not attribute_claims(claims, spans, FakeEmbedder(), 1.01)[0].supported


# --------------------------------------------------------------------------- #
# support fraction                                                             #
# --------------------------------------------------------------------------- #
def _attr(supported: bool) -> Attribution:
    return Attribution(claim=_claim("x", ()), best_span=1, score=1.0, supported=supported)


def test_support_fraction_counts_supported_claims() -> None:
    assert support_fraction([_attr(True), _attr(False)]) == 0.5


def test_support_fraction_of_no_claims_is_none_not_one() -> None:
    """Scoring an empty answer 1.0 would let it look perfect to the abstention
    policy."""
    assert support_fraction([]) is None
