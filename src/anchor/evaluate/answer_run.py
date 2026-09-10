"""Generate an answer for every golden question, into one recorded run.

This is what the grounding layer consumes. Answers are persisted rather than
held in memory so that claim extraction, the symbol oracle and the judge can all
run over the same fixed set of answers, and so a sweep can be re-scored later
without paying to regenerate it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from anchor import db
from anchor.config import Settings
from anchor.evaluate.golden import load_golden
from anchor.evaluate.link import sync_golden
from anchor.generate.backends import make_backend
from anchor.generate.pipeline import answer_question
from anchor.generate.store import save_answer
from anchor.index.embed import Embedder

logger = logging.getLogger(__name__)


@dataclass
class AnswerRunReport:
    run_id: int
    answered: int = 0
    abstained: int = 0
    total_cost_usd: float = 0.0
    unpriced: int = 0
    latencies_ms: list[int] = field(default_factory=list)
    invalid_citation_ids: list[str] = field(default_factory=list)
    uncited_span_count: int = 0
    supplied_span_count: int = 0

    @property
    def mean_latency_ms(self) -> float | None:
        return sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else None

    @property
    def cost_per_answer(self) -> float | None:
        total = self.answered + self.abstained
        return self.total_cost_usd / total if total else None

    @property
    def uncited_fraction(self) -> float | None:
        """Share of supplied spans that answers never cited.

        A high value means rerank_top_n is larger than answers actually use, and
        the surplus is paid for in input tokens on every single query.
        """
        return (
            self.uncited_span_count / self.supplied_span_count if self.supplied_span_count else None
        )


def run_answers(settings: Settings) -> AnswerRunReport:
    """Answer every golden question and persist the results."""
    queries = load_golden(settings.evaluate.golden_path)
    if not queries:
        raise RuntimeError("the golden set is empty")

    query_ids = sync_golden(settings)
    embedder = Embedder(
        settings.index.embedding_model,
        normalize=settings.index.normalize_embeddings,
        query_instruction=settings.index.query_instruction,
    )
    backend = make_backend(settings)

    with db.connect(settings) as conn:
        row = conn.execute(
            "INSERT INTO runs (config_json, git_sha, notes) VALUES (%s, %s, %s) RETURNING id",
            (json.dumps(settings.run_config()), db.git_sha(), json.dumps({"kind": "answers"})),
        ).fetchone()
        if row is None:  # pragma: no cover
            raise RuntimeError("failed to open a run")
        run_id = int(row["id"])
        conn.commit()

        report = AnswerRunReport(run_id=run_id)
        for i, query in enumerate(queries, start=1):
            answer = answer_question(query.question, settings, embedder, backend)

            if answer.abstained:
                report.abstained += 1
            else:
                report.answered += 1
            if answer.invalid_cited:
                report.invalid_citation_ids.append(query.id)

            report.supplied_span_count += len(answer.spans)
            report.uncited_span_count += len(answer.uncited_spans)

            completion = answer.completion
            if completion is not None:
                report.latencies_ms.append(completion.latency_ms)
                if completion.cost_usd is None:
                    report.unpriced += 1
                else:
                    report.total_cost_usd += completion.cost_usd

            save_answer(conn, run_id, query_ids[query.id], answer)
            # Committed per answer: generation costs real money, so an
            # interrupted run must keep everything it already paid for.
            conn.commit()
            logger.info("answered %d/%d  %s", i, len(queries), query.id)

        conn.execute("UPDATE runs SET finished_at = now() WHERE id = %s", (run_id,))
        conn.commit()

    return report
