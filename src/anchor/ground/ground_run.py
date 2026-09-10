"""Score stored answers and produce the tradeoff curve.

Reads the answers a generation run already paid for, extracts claims, attributes
them to their cited spans, asks the oracle about anything checkable, and writes
one `claims` row per claim. Then sweeps the abstention threshold over the scores
already in memory.

No generation happens here. The whole curve is a re-thresholding of one scoring
pass, which is why the sweep is free and why phase 2 will find thousands of
(claim, span, verdict) rows waiting for it as a byproduct.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import psycopg
from psycopg.rows import DictRow

from anchor import db
from anchor.config import Settings, Verdict, Verifier
from anchor.ground.attribution import Attribution, attribute_claims
from anchor.ground.claims import extract_claims
from anchor.ground.judge import AnthropicJudge, Judge, agreement, judge_cost
from anchor.ground.tradeoff import ScoredAnswer, TradeoffPoint, sweep
from anchor.ground.verify import verify_with_oracle
from anchor.index.embed import Embedder
from anchor.retrieve.search import Hit

logger = logging.getLogger(__name__)

_INSERT_CLAIM = """
INSERT INTO claims
    (answer_id, claim_text, supporting_chunk_id, support_score, verdict, verifier)
VALUES (%(answer_id)s, %(claim_text)s, %(supporting_chunk_id)s, %(support_score)s,
        %(verdict)s, %(verifier)s)
"""


@dataclass
class GroundReport:
    run_id: int
    answers: int = 0
    claims: int = 0
    uncited_claims: int = 0
    oracle_checked: int = 0
    oracle_contradicted: int = 0
    judged_claims: int = 0
    judge_cost_usd: float = 0.0
    # Judge verdict counts, and how often it agrees with span attribution.
    judge_verdicts: dict[str, int] = field(default_factory=dict)
    agree: int = 0
    judge_stricter: int = 0
    attribution_stricter: int = 0
    points: list[TradeoffPoint] = field(default_factory=list)


def _spans_for(conn: psycopg.Connection[DictRow], chunk_ids: list[int]) -> list[Hit]:
    """Rebuild the spans an answer was given, in the order it saw them.

    Ordered by the stored id list rather than by chunk id, because the citation
    numbers in the answer refer to that ordering. Re-sorting would silently
    renumber every citation.
    """
    if not chunk_ids:
        return []
    rows = conn.execute(
        """
        SELECT c.id, c.text, c.heading_path, c.is_code_block,
               d.source_path, d.url, d.vllm_version
        FROM chunks c JOIN documents d ON d.id = c.document_id
        WHERE c.id = ANY(%s)
        """,
        (chunk_ids,),
    ).fetchall()
    by_id = {int(r["id"]): r for r in rows}

    spans: list[Hit] = []
    for chunk_id in chunk_ids:
        row = by_id.get(chunk_id)
        if row is None:
            continue
        spans.append(
            Hit(
                chunk_id=chunk_id,
                text=str(row["text"]),
                score=0.0,
                heading_path=row["heading_path"],
                is_code_block=bool(row["is_code_block"]),
                source_path=str(row["source_path"]),
                url=row["url"],
                vllm_version=str(row["vllm_version"]),
            )
        )
    return spans


def _verdict_for(attribution: Attribution, oracle_contradicted: bool) -> tuple[str, str]:
    """The verdict and the verifier that produced it.

    The oracle outranks attribution: it consulted the source, while attribution
    only measured similarity. A claim naming a flag that does not exist is
    contradicted no matter how well it matches its span.
    """
    if oracle_contradicted:
        return Verdict.CONTRADICTED.value, Verifier.SYMBOL_ORACLE.value
    if not attribution.claim.is_cited:
        return Verdict.UNVERIFIABLE.value, Verifier.SPAN_ATTRIBUTION.value
    verdict = Verdict.SUPPORTED if attribution.supported else Verdict.UNSUPPORTED
    return verdict.value, Verifier.SPAN_ATTRIBUTION.value


def ground_run(
    settings: Settings,
    run_id: int | None = None,
    judge: Judge | None = None,
) -> GroundReport:
    """Score every answer in a run and sweep the abstention threshold.

    The judge is optional because it costs money. Without it the
    cited-but-unsupported column is a similarity floor; with it, it is a
    semantic verdict, and only the judge can catch a claim that sits close to
    its span while stating a different number.
    """
    if judge is None and settings.ground.use_llm_judge:
        judge = AnthropicJudge(settings)
    embedder = Embedder(
        settings.index.embedding_model,
        normalize=settings.index.normalize_embeddings,
        query_instruction=settings.index.query_instruction,
    )

    with db.connect(settings) as conn:
        if run_id is None:
            row = conn.execute("SELECT max(run_id) AS r FROM answers").fetchone()
            if row is None or row["r"] is None:
                raise RuntimeError("no answers stored; run `anchor answer-golden` first")
            run_id = int(row["r"])

        version_row = conn.execute("SELECT DISTINCT vllm_version FROM symbols LIMIT 1").fetchone()
        if version_row is None:
            raise RuntimeError("the symbol oracle is empty; run `anchor oracle-build`")
        version = str(version_row["vllm_version"])

        rows = conn.execute(
            """
            SELECT a.id, a.answer_text, a.retrieved_chunk_ids, g.question
            FROM answers a JOIN golden_queries g ON g.id = a.query_id
            WHERE a.run_id = %s AND NOT a.abstained
            ORDER BY a.id
            """,
            (run_id,),
        ).fetchall()

        report = GroundReport(run_id=run_id)
        scored: list[ScoredAnswer] = []

        # Re-scoring is idempotent: a previous pass over this run is cleared so
        # claims are not duplicated when a threshold or model is changed.
        conn.execute(
            "DELETE FROM claims WHERE answer_id IN (SELECT id FROM answers WHERE run_id = %s)",
            (run_id,),
        )

        for row in rows:
            answer_id = int(row["id"])
            answer_text = str(row["answer_text"] or "")
            spans = _spans_for(conn, list(row["retrieved_chunk_ids"] or []))

            claims = extract_claims(answer_text)
            attributions = attribute_claims(
                claims, spans, embedder, settings.ground.support_threshold
            )

            judged: dict[int, Verdict] = {}
            if judge is not None and claims:
                result = judge.judge(claims, spans)
                judged = dict(result.verdicts)
                report.judged_claims += len(result.verdicts)
                cost = judge_cost(result, settings.ground.judge_model)
                if cost is not None:
                    report.judge_cost_usd += cost
                for judged_verdict in result.verdicts.values():
                    key = judged_verdict.value
                    report.judge_verdicts[key] = report.judge_verdicts.get(key, 0) + 1
                agreed, stricter, looser = agreement(
                    result.verdicts,
                    {a.claim.ordinal: a.supported for a in attributions},
                )
                report.agree += agreed
                report.judge_stricter += stricter
                report.attribution_stricter += looser

            oracle = verify_with_oracle(conn, answer_text, settings.ground, version)
            contradicted = oracle.verdict is Verdict.CONTRADICTED
            if oracle.checkable:
                report.oracle_checked += 1
            if contradicted:
                report.oracle_contradicted += 1

            for attribution in attributions:
                verdict, verifier = _verdict_for(attribution, contradicted)
                # The judge outranks similarity where it has an opinion: it read
                # the span, while attribution only measured distance to it. The
                # oracle still outranks both, since it consulted the source.
                if not contradicted and attribution.claim.ordinal in judged:
                    verdict = judged[attribution.claim.ordinal].value
                    verifier = Verifier.LLM_JUDGE.value
                supporting = (
                    spans[attribution.best_span - 1].chunk_id
                    if attribution.best_span is not None and attribution.best_span - 1 < len(spans)
                    else None
                )
                conn.execute(
                    _INSERT_CLAIM,
                    {
                        "answer_id": answer_id,
                        "claim_text": attribution.claim.text,
                        "supporting_chunk_id": supporting,
                        "support_score": attribution.score,
                        "verdict": verdict,
                        "verifier": verifier,
                    },
                )
                if not attribution.claim.is_cited:
                    report.uncited_claims += 1

            report.answers += 1
            report.claims += len(attributions)
            scored.append(
                ScoredAnswer(
                    query_id=str(row["question"])[:60],
                    attributions=attributions,
                    oracle_contradicted=contradicted,
                )
            )

        conn.commit()

    thresholds = [round(i / 10, 2) for i in range(0, 11)]
    report.points = sweep(scored, thresholds, settings.ground.support_threshold)
    return report
