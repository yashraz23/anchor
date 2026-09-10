"""Score the triad over every answer in a run."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from anchor import db
from anchor.config import Settings
from anchor.evaluate.triad import TriadResult, TriadScorer, mean_or_none, triad_cost
from anchor.ground.ground_run import _spans_for

logger = logging.getLogger(__name__)


@dataclass
class TriadReport:
    run_id: int
    results: list[TriadResult] = field(default_factory=list)

    @property
    def answer_relevance(self) -> float | None:
        return mean_or_none([r.answer_relevance for r in self.results])

    @property
    def context_relevance(self) -> float | None:
        return mean_or_none([r.context_relevance for r in self.results])

    @property
    def fully_relevant_contexts(self) -> int:
        """Questions where every retrieved span was useful."""
        return sum(1 for r in self.results if r.context_relevance == 1.0)


def run_triad(settings: Settings, run_id: int | None = None) -> TriadReport:
    """Score answer relevance and context relevance across a run."""
    scorer = TriadScorer(settings)

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

        report = TriadReport(run_id=run_id)
        for i, row in enumerate(rows, start=1):
            spans = _spans_for(conn, list(row["retrieved_chunk_ids"] or []))
            report.results.append(
                scorer.score(str(row["question"]), str(row["answer_text"] or ""), spans)
            )
            logger.info("scored %d/%d", i, len(rows))

    return report


def report_cost(report: TriadReport, settings: Settings) -> float | None:
    return triad_cost(report.results, settings.evaluate.ragas_model)
