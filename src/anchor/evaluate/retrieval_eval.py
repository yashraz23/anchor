"""recall@k over the golden set.

Recall is measured at the **document** level: of the top k retrieved chunks, did
any come from a document the golden entry names? That is what the golden set can
honestly support, because entries store source paths rather than chunk ids, and
chunk ids are stable only within one (strategy, commit) pair.

It is also the fairer comparison. The two chunking strategies cut the same
document into different numbers of pieces, so a chunk-level score would reward
whichever strategy happens to produce more chunks per document rather than
whichever one retrieves the right material.

Every function that computes a number is pure and separately tested. The parts
that touch Postgres only gather ranked paths.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from anchor import db
from anchor.config import ChunkStrategy, RetrievalMode, Settings
from anchor.evaluate.golden import GoldenQuery
from anchor.index.embed import Embedder
from anchor.retrieve.search import search

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueryOutcome:
    """One golden question under one retrieval configuration."""

    query_id: str
    # Source path of each retrieved chunk, in rank order. Duplicates are kept:
    # rank k means the k-th chunk, not the k-th distinct document.
    retrieved_paths: tuple[str, ...]
    expected_paths: tuple[str, ...]

    def recall_at(self, k: int) -> float | None:
        return recall_at_k(self.retrieved_paths, self.expected_paths, k)

    def first_hit_rank(self) -> int | None:
        """1-based rank of the first correct document, or None if never found."""
        expected = set(self.expected_paths)
        for rank, path in enumerate(self.retrieved_paths, start=1):
            if path in expected:
                return rank
        return None


def recall_at_k(
    retrieved_paths: Sequence[str], expected_paths: Sequence[str], k: int
) -> float | None:
    """Fraction of expected documents found among the top k retrieved chunks.

    None when nothing is expected, rather than 0.0 or 1.0. A question with no
    ground truth is unanswerable by this metric, and folding it in as either
    value would quietly move the headline number.
    """
    expected = set(expected_paths)
    if not expected:
        return None
    found = expected.intersection(retrieved_paths[:k])
    return len(found) / len(expected)


def mean_recall(outcomes: Sequence[QueryOutcome], k: int) -> float | None:
    """Mean recall@k across questions, ignoring those with no ground truth."""
    scores = [s for s in (o.recall_at(k) for o in outcomes) if s is not None]
    return sum(scores) / len(scores) if scores else None


@dataclass
class ConfigOutcome:
    """Results for one (chunking strategy, retrieval mode, rerank) cell."""

    strategy: str
    mode: str
    rerank: bool
    outcomes: list[QueryOutcome] = field(default_factory=list)

    @property
    def label(self) -> str:
        base = {"dense": "dense", "sparse": "sparse", "hybrid": "hybrid (RRF)"}[self.mode]
        return f"{base} + rerank" if self.rerank else base

    def recall_row(self, ks: Sequence[int]) -> list[float | None]:
        return [mean_recall(self.outcomes, k) for k in ks]

    def never_found(self) -> list[str]:
        """Questions where no expected document appeared at any depth.

        The most useful column in practice: these are the queries where the
        corpus or the retriever fails outright, not merely ranks poorly.
        """
        return [o.query_id for o in self.outcomes if o.first_hit_rank() is None]


def evaluate_config(
    queries: Sequence[GoldenQuery],
    settings: Settings,
    embedder: Embedder,
    strategy: ChunkStrategy,
    mode: RetrievalMode,
    rerank: bool,
    depth: int,
) -> ConfigOutcome:
    """Run every golden question under one configuration."""
    # A copy, so one cell of the sweep cannot leak its settings into the next.
    run_settings = settings.model_copy(deep=True)
    run_settings.retrieve.use_rerank = rerank

    result = ConfigOutcome(strategy=strategy.value, mode=mode.value, rerank=rerank)
    with db.connect(run_settings) as conn:
        for query in queries:
            hits = search(
                conn,
                query.question,
                run_settings,
                embedder,
                strategy=strategy.value,
                mode=mode,
                limit=depth,
            )
            result.outcomes.append(
                QueryOutcome(
                    query_id=query.id,
                    retrieved_paths=tuple(hit.source_path for hit in hits),
                    expected_paths=query.expected_source_paths,
                )
            )
    return result


def format_number(value: float | None) -> str:
    """Render a metric, or an em-free placeholder when it does not exist."""
    return "-" if value is None else f"{value:.2f}"


def render_table(results: Sequence[ConfigOutcome], ks: Sequence[int]) -> str:
    """The recall@k table, as Markdown ready for the README."""
    header = "| Chunking | Retrieval | " + " | ".join(f"recall@{k}" for k in ks) + " |"
    divider = "|---|---|" + "---|" * len(ks)
    lines = [header, divider]
    for result in results:
        cells = " | ".join(format_number(v) for v in result.recall_row(ks))
        lines.append(f"| `{result.strategy}` | {result.label} | {cells} |")
    return "\n".join(lines)


@dataclass(frozen=True)
class Paired:
    """Head-to-head record between two configurations on the same questions."""

    a_wins: int
    b_wins: int
    ties: int

    @property
    def decided(self) -> int:
        return self.a_wins + self.b_wins


def paired_comparison(a: ConfigOutcome, b: ConfigOutcome, k: int) -> Paired:
    """Per-question wins between two configurations at depth k.

    Both cells run the same questions, so pairing is available and it is far
    more informative than the difference of two means on a set this small. A
    gap of 0.05 across 21 questions is one question changing rank; a 6-2 split
    across the questions that actually differ is a claim with something behind
    it.

    Questions where neither configuration has ground truth are excluded rather
    than counted as ties, since they carry no information either way.
    """
    by_id = {o.query_id: o for o in b.outcomes}
    a_wins = b_wins = ties = 0
    for outcome in a.outcomes:
        other = by_id.get(outcome.query_id)
        if other is None:
            continue
        left, right = outcome.recall_at(k), other.recall_at(k)
        if left is None or right is None:
            continue
        if left > right:
            a_wins += 1
        elif right > left:
            b_wins += 1
        else:
            ties += 1
    return Paired(a_wins=a_wins, b_wins=b_wins, ties=ties)


def sign_test_p_value(wins: int, losses: int) -> float | None:
    """Two-sided exact sign test on a paired win/loss record.

    The appropriate test here, and deliberately a weak one. It assumes only that
    under the null hypothesis each question that separates the two
    configurations is a coin flip, which is the least this comparison can assume.
    Ties carry no directional information and are excluded, which is what the
    sign test does by construction.

    Returns None when nothing was decided, since a test on no evidence has no
    p-value rather than a p-value of 1.
    """
    n = wins + losses
    if n == 0:
        return None
    extreme = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(extreme + 1))
    p_value: float = min(1.0, 2.0 * tail / (2**n))
    return p_value
