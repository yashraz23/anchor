"""API surface.

Driven through FastAPI's TestClient with the grounding service stubbed, so what
is tested is the contract: that the response always carries the evidence, and
that a withheld answer says so rather than returning ungrounded prose.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from anchor.api import app as api_module
from anchor.generate.backends import Completion
from anchor.generate.pipeline import Answer
from anchor.ground.attribution import Attribution
from anchor.ground.claims import Claim
from anchor.ground.service import GroundedAnswer
from anchor.retrieve.search import Hit


def _hit(text: str) -> Hit:
    return Hit(
        chunk_id=1,
        text=text,
        score=0.9,
        heading_path="Engine Arguments",
        is_code_block=False,
        source_path="docs/engine_args.md",
        url="https://github.com/vllm-project/vllm/blob/abc/docs/engine_args.md",
        vllm_version="v0.28.1rc0",
    )


def _grounded(withheld: bool) -> GroundedAnswer:
    answer = Answer(
        question="q",
        text="withheld message" if withheld else "max_num_seqs caps the batch [1].",
        spans=[_hit("a span"), _hit("another span")],
        completion=Completion(
            text="x",
            model="claude-opus-5",
            input_tokens=1000,
            output_tokens=100,
            latency_ms=1234,
        ),
        abstained=withheld,
    )
    return GroundedAnswer(
        question="q",
        answer=answer,
        attributions=[
            Attribution(claim=Claim(0, "a claim", (1,)), best_span=1, score=0.9, supported=True)
        ],
        support=1.0 if not withheld else 0.2,
        withheld=withheld,
        reason="only 20% of claims are supported" if withheld else "",
    )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    holder: dict[str, bool] = {"withheld": False}

    def fake(question: str, settings, embedder=None, backend=None) -> GroundedAnswer:  # type: ignore[no-untyped-def]
        return _grounded(holder["withheld"])

    monkeypatch.setattr(api_module, "grounded_answer", fake)
    with TestClient(api_module.app) as c:
        c.holder = holder  # type: ignore[attr-defined]
        yield c


def test_an_answer_carries_its_spans(client: TestClient) -> None:
    body = client.post("/ask", json={"question": "what is max_num_seqs"}).json()
    assert body["answer"].startswith("max_num_seqs")
    assert len(body["spans"]) == 2
    assert body["spans"][0]["number"] == 1


def test_the_response_says_which_spans_were_cited(client: TestClient) -> None:
    """An endpoint returning only prose would hide exactly what this project
    exists to measure."""
    body = client.post("/ask", json={"question": "what is max_num_seqs"}).json()
    assert body["spans"][0]["cited"] is True
    assert body["spans"][1]["cited"] is False


def test_the_response_carries_cost_and_latency(client: TestClient) -> None:
    body = client.post("/ask", json={"question": "what is max_num_seqs"}).json()
    assert body["latency_ms"] == 1234
    assert body["cost_usd"] == pytest.approx(1000 / 1e6 * 5 + 100 / 1e6 * 25)


def test_a_withheld_answer_says_so_and_why(client: TestClient) -> None:
    client.holder["withheld"] = True  # type: ignore[attr-defined]
    body = client.post("/ask", json={"question": "something unanswerable"}).json()
    assert body["withheld"] is True
    assert "20%" in body["reason"]


def test_a_withheld_answer_does_not_return_the_ungrounded_text(
    client: TestClient,
) -> None:
    """Anything printed will be read, whatever caveat sits beside it."""
    client.holder["withheld"] = True  # type: ignore[attr-defined]
    body = client.post("/ask", json={"question": "something unanswerable"}).json()
    assert "max_num_seqs caps" not in body["answer"]


def test_an_empty_question_is_rejected(client: TestClient) -> None:
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_support_may_be_absent_without_being_zero(client: TestClient) -> None:
    body = client.post("/ask", json={"question": "what is max_num_seqs"}).json()
    assert body["support"] == 1.0
    assert body["claims"] == 1
    assert body["supported_claims"] == 1
