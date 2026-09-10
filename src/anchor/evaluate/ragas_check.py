"""Cross-check the hand-written triad against ragas.

The triad in `triad.py` was written first, when LangChain was out of bounds.
LangChain is now permitted, so ragas runs too, and the point of keeping both is
not redundancy: an in-house metric that nobody has calibrated is a number
without a reference class. Running the standard implementation over the same
answers says how far the two land apart, which is the only way to know whether
the custom one is measuring what it claims.

Ragas is driven by Claude rather than OpenAI, so both triads see the same model
and a disagreement is about the metric rather than about which model judged.

Maintenance note, because it will bite someone: ragas 0.4.3 still imports
`langchain_community.chat_models.vertexai`, which langchain-community removed in
0.4. pyproject pins below that. It is the running cost of this dependency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from anchor.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class RagasScores:
    """Ragas scores for one question."""

    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None


@dataclass
class RagasReport:
    run_id: int
    scores: list[RagasScores] = field(default_factory=list)
    failures: int = 0

    @staticmethod
    def _mean(values: list[float | None]) -> float | None:
        present = [v for v in values if v is not None]
        return sum(present) / len(present) if present else None

    @property
    def faithfulness(self) -> float | None:
        return self._mean([s.faithfulness for s in self.scores])

    @property
    def answer_relevancy(self) -> float | None:
        return self._mean([s.answer_relevancy for s in self.scores])

    @property
    def context_precision(self) -> float | None:
        return self._mean([s.context_precision for s in self.scores])


def build_ragas_llm(settings: Settings):  # type: ignore[no-untyped-def]
    """Wrap Claude for ragas.

    Ragas defaults to OpenAI. Pointing it at the same model the rest of the
    project uses is what makes the comparison a comparison of metrics rather
    than of models.
    """
    from langchain_anthropic import ChatAnthropic
    from ragas.llms import LangchainLLMWrapper

    key = settings.anthropic_api_key
    chat = ChatAnthropic(
        model_name=settings.ground.judge_model,
        timeout=120,
        stop=None,
        # Opus 5 rejects sampling parameters outright: sending temperature is a
        # 400, not a warning. LangChain sends one by default and ragas
        # overwrites it per call, so both have to be suppressed.
        temperature=None,
        **({"api_key": key.get_secret_value()} if key is not None else {}),
    )
    return LangchainLLMWrapper(chat, bypass_temperature=True)


def build_ragas_embeddings(settings: Settings):  # type: ignore[no-untyped-def]
    """Wrap the same encoder the corpus was indexed with.

    Answer relevancy is an embedding metric. Scoring it with a different encoder
    than the one that built the index would measure the encoder swap rather than
    the answer.
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    return LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=settings.index.embedding_model)
    )


async def _score_one(sample, metrics) -> RagasScores:  # type: ignore[no-untyped-def]
    faithfulness, relevancy, precision = metrics
    return RagasScores(
        faithfulness=float(await faithfulness.single_turn_ascore(sample)),
        answer_relevancy=float(await relevancy.single_turn_ascore(sample)),
        context_precision=float(await precision.single_turn_ascore(sample)),
    )


def run_ragas(settings: Settings, run_id: int | None = None) -> RagasReport:
    """Score a stored answer run with ragas."""
    import asyncio

    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import (
        AnswerRelevancy,
        Faithfulness,
        LLMContextPrecisionWithoutReference,
    )

    from anchor import db
    from anchor.ground.ground_run import _spans_for

    llm = build_ragas_llm(settings)
    embeddings = build_ragas_embeddings(settings)
    metrics = (
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        LLMContextPrecisionWithoutReference(llm=llm),
    )

    with db.connect(settings) as conn:
        if run_id is None:
            row = conn.execute("SELECT max(run_id) AS r FROM answers").fetchone()
            if row is None or row["r"] is None:
                raise RuntimeError("no answers stored; run `anchor answer-golden` first")
            run_id = int(row["r"])

        rows = conn.execute(
            """
            SELECT a.answer_text, a.retrieved_chunk_ids, g.question
            FROM answers a JOIN golden_queries g ON g.id = a.query_id
            WHERE a.run_id = %s AND NOT a.abstained
            ORDER BY a.id
            """,
            (run_id,),
        ).fetchall()

        samples = [
            SingleTurnSample(
                user_input=str(r["question"]),
                response=str(r["answer_text"] or ""),
                retrieved_contexts=[
                    h.text for h in _spans_for(conn, list(r["retrieved_chunk_ids"] or []))
                ],
            )
            for r in rows
        ]

    report = RagasReport(run_id=run_id)

    async def run_all() -> None:
        for i, sample in enumerate(samples, start=1):
            try:
                report.scores.append(await _score_one(sample, metrics))
            except Exception as exc:
                # One metric failure must not lose the whole run. Recorded as a
                # failure rather than as a zero: a score that could not be
                # computed is missing, not bad.
                report.failures += 1
                logger.warning("ragas failed on sample %d: %s", i, exc)
            logger.info("ragas scored %d/%d", i, len(samples))

    asyncio.run(run_all())
    return report
