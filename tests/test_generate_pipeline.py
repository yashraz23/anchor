"""Answer pipeline, driven by a fake backend.

No API calls: the point of these is the control flow around generation, which is
where the cost and correctness decisions live.
"""

from __future__ import annotations

from anchor.generate.backends import Completion
from anchor.generate.pipeline import Answer
from anchor.retrieve.search import Hit


def _hit(chunk_id: int) -> Hit:
    return Hit(
        chunk_id=chunk_id,
        text=f"span {chunk_id}",
        score=0.9,
        heading_path="H",
        is_code_block=False,
        source_path="docs/x.md",
        url=None,
        vllm_version="v0.28.1rc0",
    )


def _completion(text: str, **kwargs: object) -> Completion:
    defaults = {
        "model": "claude-opus-5",
        "input_tokens": 1000,
        "output_tokens": 100,
        "latency_ms": 1200,
    }
    defaults.update(kwargs)
    return Completion(text=text, **defaults)  # type: ignore[arg-type]


def test_cost_is_priced_from_usage() -> None:
    """1000 in and 100 out on Opus 5 at $5 / $25 per million."""
    completion = _completion("x")
    expected = 1000 / 1_000_000 * 5.0 + 100 / 1_000_000 * 25.0
    assert completion.cost_usd == expected


def test_cost_of_an_unpriced_model_is_none_not_zero() -> None:
    """A local vLLM model has no price on file. Recording 0.0 would make the
    phase 3 cost comparison silently claim the model was free."""
    assert _completion("x", model="Qwen2.5-7B-Instruct").cost_usd is None


def test_answer_reports_cited_spans() -> None:
    answer = Answer(
        question="q",
        text="Set it to 0.9 [1]. Also see [3].",
        spans=[_hit(10), _hit(11), _hit(12)],
    )
    assert answer.cited == {1, 3}
    assert answer.invalid_cited == set()


def test_answer_flags_a_citation_that_was_never_supplied() -> None:
    answer = Answer(question="q", text="As shown [4].", spans=[_hit(10)])
    assert answer.invalid_cited == {4}


def test_uncited_spans_are_reported() -> None:
    """Not a fault on its own. It is a signal that rerank_top_n may be larger
    than answers actually use, which costs input tokens on every query."""
    answer = Answer(question="q", text="Only this one [2].", spans=[_hit(1), _hit(2), _hit(3)])
    assert answer.uncited_spans == {1, 3}


def test_retrieved_chunk_ids_are_recorded_in_rank_order() -> None:
    answer = Answer(question="q", text="", spans=[_hit(7), _hit(3)])
    assert answer.retrieved_chunk_ids == [7, 3]


def test_an_abstention_carries_its_reason() -> None:
    answer = Answer(
        question="q", text="could not ground", abstained=True, abstain_reason="no spans"
    )
    assert answer.abstained is True
    assert answer.abstain_reason == "no spans"
    assert answer.cited == set()
