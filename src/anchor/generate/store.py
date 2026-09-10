"""Persist generated answers.

Cost and latency are written on every row from the first run, not reconstructed
later. Per-request cost is a question that gets asked directly, and the answer
should be a measurement rather than an estimate.
"""

from __future__ import annotations

import psycopg
from psycopg.rows import DictRow

from anchor.generate.pipeline import Answer

_UPSERT = """
INSERT INTO answers
    (run_id, query_id, answer_text, retrieved_chunk_ids, latency_ms,
     input_tokens, output_tokens, cost_usd, abstained)
VALUES (%(run_id)s, %(query_id)s, %(answer_text)s, %(retrieved_chunk_ids)s,
        %(latency_ms)s, %(input_tokens)s, %(output_tokens)s, %(cost_usd)s,
        %(abstained)s)
ON CONFLICT (run_id, query_id) DO UPDATE
    SET answer_text = EXCLUDED.answer_text,
        retrieved_chunk_ids = EXCLUDED.retrieved_chunk_ids,
        latency_ms = EXCLUDED.latency_ms,
        input_tokens = EXCLUDED.input_tokens,
        output_tokens = EXCLUDED.output_tokens,
        cost_usd = EXCLUDED.cost_usd,
        abstained = EXCLUDED.abstained
RETURNING id
"""


def save_answer(
    conn: psycopg.Connection[DictRow],
    run_id: int,
    query_id: int,
    answer: Answer,
) -> int:
    """Write one answer and return its id."""
    completion = answer.completion
    row = conn.execute(
        _UPSERT,
        {
            "run_id": run_id,
            "query_id": query_id,
            "answer_text": answer.text,
            "retrieved_chunk_ids": answer.retrieved_chunk_ids,
            # An abstention that never called the model has no usage at all.
            # These stay NULL rather than becoming zeros, so an abstained row
            # cannot be averaged in as a free, instant answer.
            "latency_ms": completion.latency_ms if completion else None,
            "input_tokens": completion.input_tokens if completion else None,
            "output_tokens": completion.output_tokens if completion else None,
            "cost_usd": completion.cost_usd if completion else None,
            "abstained": answer.abstained,
        },
    ).fetchone()
    if row is None:  # pragma: no cover - RETURNING always yields a row
        raise RuntimeError("failed to save answer")
    return int(row["id"])
