"""Run the retrieval sweep across chunking strategies and retrieval modes.

Every run writes a row to `runs` holding the full config and the git SHA it was
produced at. A results table that cannot be traced back to a commit and a
configuration is not reproducible, and a number in the README that cannot be
reproduced is worse than no number.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from anchor import db
from anchor.config import ChunkStrategy, RetrievalMode, Settings
from anchor.evaluate.golden import GoldenQuery, load_golden
from anchor.evaluate.retrieval_eval import (
    ConfigOutcome,
    evaluate_config,
    paired_comparison,
    render_table,
    sign_test_p_value,
)
from anchor.index.embed import Embedder

logger = logging.getLogger(__name__)

# The cells of the comparison, in the order the README table presents them.
# Reranking is skipped for sparse: the cross-encoder rescoring a keyword list is
# a different experiment, and it is not the one the table is claiming to run.
MODES: tuple[tuple[RetrievalMode, bool], ...] = (
    (RetrievalMode.DENSE, False),
    (RetrievalMode.SPARSE, False),
    (RetrievalMode.HYBRID, False),
    (RetrievalMode.HYBRID, True),
)


@dataclass
class SweepReport:
    run_id: int
    git_sha: str
    queries: int
    results: list[ConfigOutcome]


def _record_run(settings: Settings, summary: dict[str, object]) -> int:
    """Insert the runs row and return its id."""
    with db.connect(settings) as conn:
        row = conn.execute(
            "INSERT INTO runs (config_json, git_sha, notes) VALUES (%s, %s, %s) RETURNING id",
            (
                json.dumps(settings.run_config()),
                db.git_sha(),
                # The measured metrics travel with the config that produced
                # them. The schema has no retrieval-results table, and adding
                # one to hold four numbers per cell would be more machinery than
                # the result justifies.
                json.dumps(summary),
            ),
        ).fetchone()
        conn.commit()
    if row is None:  # pragma: no cover - RETURNING always yields a row
        raise RuntimeError("failed to record the run")
    return int(row["id"])


def run_retrieval_sweep(settings: Settings) -> SweepReport:
    """Evaluate every (strategy, mode) cell over the golden set."""
    queries: list[GoldenQuery] = load_golden(settings.evaluate.golden_path)
    if not queries:
        raise RuntimeError(
            f"the golden set at {settings.evaluate.golden_path} is empty; "
            "run `anchor golden harvest` and curate before evaluating"
        )

    depth = max(settings.evaluate.recall_at)
    embedder = Embedder(
        settings.index.embedding_model,
        normalize=settings.index.normalize_embeddings,
        query_instruction=settings.index.query_instruction,
    )

    results: list[ConfigOutcome] = []
    for strategy in ChunkStrategy:
        for mode, rerank in MODES:
            logger.info(
                "evaluating %s / %s%s", strategy.value, mode.value, " + rerank" if rerank else ""
            )
            results.append(
                evaluate_config(queries, settings, embedder, strategy, mode, rerank, depth)
            )

    ks = list(settings.evaluate.recall_at)
    summary = {
        "queries": len(queries),
        "recall_at": ks,
        "cells": [
            {
                "strategy": r.strategy,
                "mode": r.mode,
                "rerank": r.rerank,
                "recall": r.recall_row(ks),
                "never_found": r.never_found(),
            }
            for r in results
        ],
    }
    run_id = _record_run(settings, summary)
    _track(settings, run_id, results, ks, len(queries))

    return SweepReport(run_id=run_id, git_sha=db.git_sha(), queries=len(queries), results=results)


def _track(
    settings: Settings,
    run_id: int,
    results: list[ConfigOutcome],
    ks: list[int],
    queries: int,
) -> None:
    """Publish the recall grid to Weights & Biases.

    The database row remains the record of truth; this exists so twelve cells
    across several runs can be compared without writing a query. The run id is
    logged with it, which is what ties a chart in the UI back to the row that
    holds the configuration that produced it.
    """
    from anchor.track import experiment, log_metrics, log_table

    with experiment(
        settings, "retrieval-sweep", {"run_id": run_id, "git_sha": db.git_sha(), "queries": queries}
    ) as run:
        log_table(
            run,
            "recall_grid",
            ["strategy", "mode", "rerank", *(f"recall@{k}" for k in ks), "never_found"],
            [
                [r.strategy, r.mode, r.rerank, *r.recall_row(ks), len(r.never_found())]
                for r in results
            ],
        )
        # The full-stack cell is the one the README quotes, so it is also what
        # the run summary sorts on.
        best = next(
            (
                r
                for r in results
                if r.mode == "hybrid" and r.rerank and r.strategy == "structure_aware"
            ),
            None,
        )
        if best is not None:
            log_metrics(
                run, {f"best/recall@{k}": v for k, v in zip(ks, best.recall_row(ks), strict=True)}
            )


def render_report(report: SweepReport, settings: Settings) -> str:
    """The full markdown block for the README."""
    ks = list(settings.evaluate.recall_at)
    lines = [
        render_table(report.results, ks),
        "",
        f"Run {report.run_id} at commit `{report.git_sha[:12]}`, "
        f"{report.queries} golden questions.",
    ]

    # The two full-stack cells, compared question by question. At this sample
    # size a difference of means is one or two questions moving; the paired
    # record says how many questions actually changed and in which direction.
    best = {(r.strategy, r.rerank): r for r in report.results if r.mode == "hybrid"}
    aware = best.get(("structure_aware", True))
    fixed = best.get(("fixed", True))
    if aware is not None and fixed is not None:
        lines.append("")
        lines.append("Paired, structure_aware vs fixed (both hybrid + rerank), per question:")
        lines.append("")
        lines.append("| k | structure_aware better | fixed better | same | sign test |")
        lines.append("|---|---|---|---|---|")
        for k in ks:
            pair = paired_comparison(aware, fixed, k)
            p_value = sign_test_p_value(pair.a_wins, pair.b_wins)
            shown = "-" if p_value is None else f"p = {p_value:.4f}"
            lines.append(f"| {k} | {pair.a_wins} | {pair.b_wins} | {pair.ties} | {shown} |")

    # Questions no configuration ever retrieves are the ones worth naming: they
    # are corpus or retriever failures, not ranking noise.
    everywhere = set.intersection(*(set(r.never_found()) for r in report.results))
    if everywhere:
        lines += [
            "",
            "Never retrieved by any configuration:",
            *(f"- `{qid}`" for qid in sorted(everywhere)),
        ]
    return "\n".join(lines)
