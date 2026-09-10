"""Claim extraction.

The fixture is a verbatim excerpt from a generated answer, because the shapes
that break a sentence splitter are exactly the ones real answers contain:
decimals, version numbers, "e.g.", inline code, bold, arrows and code blocks.

Segmentation sets the denominator of every faithfulness rate, so over-splitting
inflates it with fragments that assert nothing and under-splitting hides
unsupported claims inside supported ones.
"""

from __future__ import annotations

from anchor.ground.claims import (
    Claim,
    extract_claims,
    is_context_remark,
    is_substantive,
    split_sentences,
)

REAL_ANSWER = """**`max_model_len`** — the model context length, covering both \
prompt and output [3]. If you don't specify it, it's automatically derived from \
the model config [3]. When passed as `--max-model-len` it accepts human-readable \
formats (e.g. `1k` -> 1000, `25.6k` -> 25,600) [3].

**`max_num_seqs`** — the maximum batch size [1].

Both are levers for reducing memory usage [1], e.g.:

```python
from vllm import LLM
llm = LLM(model="x", max_model_len=2048)
print(out.outputs[0].text)
```
"""


def test_sentences_split_on_terminal_punctuation() -> None:
    sentences = split_sentences("First claim [1]. Second claim [2].")
    assert len(sentences) == 2


def test_a_decimal_is_not_a_sentence_boundary() -> None:
    """ "25.6k" and "vLLM 0.28.1" both appear in real answers."""
    assert len(split_sentences("It accepts 25.6k as a value [1].")) == 1
    assert len(split_sentences("vLLM 0.28.1 supports it [2].")) == 1


def test_an_abbreviation_is_not_a_sentence_boundary() -> None:
    assert len(split_sentences("Use a suffix, e.g. 1k, for brevity [1].")) == 1


def test_a_sentence_may_open_with_markdown_emphasis() -> None:
    """Answers routinely start a sentence with bold or a backticked name."""
    sentences = split_sentences("First [1]. **`max_num_seqs`** is the batch size [2].")
    assert len(sentences) == 2


def test_list_markers_are_stripped() -> None:
    sentences = split_sentences("- The first point here [1]\n- The second point [2]")
    assert sentences[0].startswith("The first point")
    assert sentences[1].startswith("The second point")


def test_headings_are_stripped() -> None:
    assert split_sentences("## Some heading here")[0] == "Some heading here"


# --------------------------------------------------------------------------- #
# substance                                                                    #
# --------------------------------------------------------------------------- #
def test_a_short_fragment_is_not_a_claim() -> None:
    """Counting markdown debris would inflate the denominator of every
    faithfulness rate with text that asserts nothing."""
    assert not is_substantive("**Notes:**")
    assert not is_substantive("[1]")
    assert not is_substantive("e.g.:")


def test_a_real_sentence_is_substantive() -> None:
    assert is_substantive("The maximum batch size is controlled by this [1].")


# --------------------------------------------------------------------------- #
# extraction                                                                   #
# --------------------------------------------------------------------------- #
def test_claims_carry_their_citations() -> None:
    claims = extract_claims(REAL_ANSWER)
    assert claims
    assert all(isinstance(c, Claim) for c in claims)
    first = claims[0]
    assert first.cited_spans == (3,)
    assert first.is_cited


def test_code_blocks_are_not_claims() -> None:
    """A code block is an illustration, not an assertion, and asking whether a
    snippet is 'supported' is not a question a verifier can answer."""
    claims = extract_claims(REAL_ANSWER)
    joined = " ".join(c.text for c in claims)
    assert "from vllm import LLM" not in joined
    assert "print(out.outputs" not in joined


def test_indexing_inside_a_code_block_is_not_cited_as_a_span() -> None:
    """out.outputs[0] would otherwise become a citation to span zero."""
    claims = extract_claims(REAL_ANSWER)
    assert all(0 not in c.cited_spans for c in claims)


def test_ordinals_are_contiguous() -> None:
    claims = extract_claims(REAL_ANSWER)
    assert [c.ordinal for c in claims] == list(range(len(claims)))


def test_an_uncited_claim_is_recorded_as_uncited() -> None:
    """Not automatically false, but unattributable by construction: the
    generator was told to cite every factual claim."""
    claims = extract_claims("The scheduler batches requests as they arrive.")
    assert len(claims) == 1
    assert claims[0].cited_spans == ()
    assert not claims[0].is_cited


def test_a_claim_citing_several_spans() -> None:
    claims = extract_claims("Both flags reduce memory use [1, 3].")
    assert claims[0].cited_spans == (1, 3)


def test_an_empty_answer_yields_no_claims() -> None:
    assert extract_claims("") == []


def test_an_answer_that_is_only_code_yields_no_claims() -> None:
    assert extract_claims("```python\nx = 1\n```\n") == []


# --------------------------------------------------------------------------- #
# remarks about the evidence                                                   #
# --------------------------------------------------------------------------- #
# All fixtures below are verbatim from real answers. Before these were handled,
# 120 of 313 extracted claims cited nothing and every one counted as
# unsupported, which put the headline unsupported rate at 40%. Much of that was
# the model correctly declining, scored as if it had hallucinated.
def test_a_remark_about_the_spans_is_not_a_claim_about_vllm() -> None:
    assert is_context_remark("The spans don't cover S3 credential/endpoint configuration.")
    assert is_context_remark("The provided spans don't include the client-side request format.")


def test_a_first_person_limitation_is_a_context_remark() -> None:
    assert is_context_remark("I can't say how those are supplied.")
    assert is_context_remark("There is nothing in them about weight loading.")


def test_a_claim_about_vllm_is_not_a_context_remark() -> None:
    """The exclusion must not swallow real assertions."""
    assert not is_context_remark("max_num_seqs caps the batch size [1].")
    assert not is_context_remark("vLLM allocates the KV cache at startup [2].")


def test_context_remarks_are_flagged_on_the_claim() -> None:
    claims = extract_claims("The spans do not cover this topic at all.")
    assert claims[0].about_context


def test_a_lead_in_label_is_not_a_claim() -> None:
    """ "What the spans do say:" introduces the next thing; it asserts nothing."""
    assert not is_substantive("What the spans do say:")
    assert not is_substantive("What the spans do cover:")


def test_a_long_sentence_ending_in_a_colon_is_still_a_claim() -> None:
    assert is_substantive(
        "Both flags reduce memory usage and can be combined as follows, "
        "which is the recommended approach:"
    )


# --------------------------------------------------------------------------- #
# inline code is part of the claim                                             #
# --------------------------------------------------------------------------- #
def test_inline_code_is_kept_in_the_claim_text() -> None:
    """Stripping it left claims like "pass your own stat loggers into  ."
    with the subject deleted."""
    claims = extract_claims("Pass your own stat loggers into `StatLoggerBase` [1].")
    assert "StatLoggerBase" in claims[0].text


def test_inline_code_does_not_become_a_citation() -> None:
    """parse_citations does its own stripping, so keeping inline code in the
    claim text cannot reintroduce output[0] as a citation to span zero."""
    claims = extract_claims("Read it via `output.outputs[0].text` [2].")
    assert claims[0].cited_spans == (2,)
