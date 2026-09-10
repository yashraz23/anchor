"""FastAPI surface.

Thin. Every endpoint resolves settings, calls one library function and shapes
the result; nothing here decides anything the library does not already decide,
so the API and the evaluation harness cannot drift apart in what they consider
a grounded answer.

The response always carries the evidence: which spans were used, how many claims
were supported, and why an answer was withheld when it was. An endpoint that
returned only prose would hide exactly what this project exists to measure.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from anchor import db
from anchor.config import Settings, get_settings
from anchor.ground.service import GroundedAnswer, grounded_answer
from anchor.index.embed import Embedder

logger = logging.getLogger(__name__)

# Held across requests. Loading the encoder takes seconds and the cross-encoder
# more; paying that per request would dominate latency and hide the retrieval
# cost the project is trying to measure.
_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    settings = get_settings()
    _state["settings"] = settings
    _state["embedder"] = Embedder(
        settings.index.embedding_model,
        normalize=settings.index.normalize_embeddings,
        query_instruction=settings.index.query_instruction,
    )
    logger.info("models loaded")
    yield
    _state.clear()


app = FastAPI(
    title="anchor",
    summary="Grounding-verified RAG over vLLM documentation.",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class SpanOut(BaseModel):
    number: int
    source_path: str
    heading_path: str | None
    url: str | None
    vllm_version: str
    cited: bool
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    withheld: bool
    reason: str = ""
    # Share of claims whose cited span supports them. None when the answer made
    # no checkable claim, which is not the same as zero support.
    support: float | None = None
    claims: int = 0
    supported_claims: int = 0
    spans: list[SpanOut] = []
    latency_ms: int | None = None
    cost_usd: float | None = None


class HealthResponse(BaseModel):
    ok: bool
    documents: int
    chunks: int
    symbols: int
    embedded: int


def _to_response(result: GroundedAnswer) -> AskResponse:
    cited = result.answer.cited
    completion = result.answer.completion
    return AskResponse(
        question=result.question,
        answer=result.text,
        withheld=result.withheld,
        reason=result.reason,
        support=result.support,
        claims=result.claim_count,
        supported_claims=result.supported_claims,
        spans=[
            SpanOut(
                number=i,
                source_path=hit.source_path,
                heading_path=hit.heading_path,
                url=hit.url,
                vllm_version=hit.vllm_version,
                cited=i in cited,
                text=hit.text,
            )
            for i, hit in enumerate(result.spans, start=1)
        ],
        latency_ms=completion.latency_ms if completion else None,
        cost_usd=completion.cost_usd if completion else None,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Corpus state, not just process liveness.

    A server that answers requests against an empty index is up and useless, so
    the check reports what it can actually retrieve from.
    """
    settings: Settings = _state.get("settings") or get_settings()
    try:
        with db.connect(settings) as conn:
            row = conn.execute(
                """
                SELECT (SELECT count(*) FROM documents) AS documents,
                       (SELECT count(*) FROM chunks) AS chunks,
                       (SELECT count(*) FROM symbols) AS symbols,
                       (SELECT count(embedding) FROM chunks) AS embedded
                """
            ).fetchone()
    except Exception as exc:
        # Broad on purpose: any failure reaching the database means this
        # instance cannot serve, and the caller needs 503 rather than a 500
        # with a stack trace. The error is reported, never swallowed.
        raise HTTPException(status_code=503, detail=f"database unavailable: {exc}") from exc

    if row is None:  # pragma: no cover
        raise HTTPException(status_code=503, detail="database returned no counts")

    return HealthResponse(
        ok=int(row["embedded"]) > 0,
        documents=int(row["documents"]),
        chunks=int(row["chunks"]),
        symbols=int(row["symbols"]),
        embedded=int(row["embedded"]),
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Answer a question, or explain why the answer was withheld."""
    settings: Settings = _state.get("settings") or get_settings()
    result = grounded_answer(request.question, settings, embedder=_state.get("embedder"))
    return _to_response(result)
