"""recall@k tests.

Written before the harness was run, per the convention for anything in
evaluate/. Every metric here is pure, so the numbers that land in the README
come from logic specified first rather than fitted to a result.
"""

from __future__ import annotations

from anchor.evaluate.retrieval_eval import (
    ConfigOutcome,
    QueryOutcome,
    format_number,
    mean_recall,
    paired_comparison,
    recall_at_k,
    render_table,
)


def _outcome(retrieved: list[str], expected: list[str], qid: str = "q") -> QueryOutcome:
    return QueryOutcome(
        query_id=qid, retrieved_paths=tuple(retrieved), expected_paths=tuple(expected)
    )


# --------------------------------------------------------------------------- #
# recall@k                                                                     #
# --------------------------------------------------------------------------- #
def test_recall_counts_only_the_top_k() -> None:
    paths = ["a.md", "b.md", "target.md"]
    assert recall_at_k(paths, ["target.md"], 2) == 0.0
    assert recall_at_k(paths, ["target.md"], 3) == 1.0


def test_recall_with_several_expected_documents() -> None:
    assert recall_at_k(["a.md", "b.md"], ["a.md", "z.md"], 2) == 0.5


def test_recall_is_none_when_nothing_is_expected() -> None:
    """Scoring an entry with no ground truth as 0.0 or 1.0 would quietly move
    the headline number in the README."""
    assert recall_at_k(["a.md"], [], 5) is None


def test_rank_k_means_the_kth_chunk_not_the_kth_document() -> None:
    """Two chunks from one document occupy two ranks.

    Collapsing duplicates first would silently deepen the cut and inflate
    recall at small k.
    """
    paths = ["a.md", "a.md", "target.md"]
    assert recall_at_k(paths, ["target.md"], 2) == 0.0
    assert recall_at_k(paths, ["target.md"], 3) == 1.0


def test_duplicate_expected_paths_do_not_inflate_the_denominator() -> None:
    assert recall_at_k(["a.md"], ["a.md", "a.md"], 1) == 1.0


# --------------------------------------------------------------------------- #
# aggregation                                                                  #
# --------------------------------------------------------------------------- #
def test_mean_recall_averages_across_questions() -> None:
    outcomes = [
        _outcome(["hit.md"], ["hit.md"], "a"),
        _outcome(["miss.md"], ["hit.md"], "b"),
    ]
    assert mean_recall(outcomes, 1) == 0.5


def test_mean_recall_skips_questions_with_no_ground_truth() -> None:
    """One ungradable question must not drag the average toward zero."""
    outcomes = [
        _outcome(["hit.md"], ["hit.md"], "a"),
        _outcome(["x.md"], [], "ungradable"),
    ]
    assert mean_recall(outcomes, 1) == 1.0


def test_mean_recall_of_nothing_is_none() -> None:
    assert mean_recall([], 5) is None


def test_first_hit_rank_is_one_based() -> None:
    assert _outcome(["a.md", "target.md"], ["target.md"]).first_hit_rank() == 2


def test_first_hit_rank_is_none_when_never_found() -> None:
    assert _outcome(["a.md"], ["target.md"]).first_hit_rank() is None


def test_never_found_lists_outright_failures() -> None:
    """The most useful column: where the retriever fails, not merely ranks low."""
    result = ConfigOutcome(strategy="fixed", mode="hybrid", rerank=False)
    result.outcomes = [
        _outcome(["hit.md"], ["hit.md"], "found"),
        _outcome(["x.md", "y.md"], ["target.md"], "lost"),
    ]
    assert result.never_found() == ["lost"]


# --------------------------------------------------------------------------- #
# rendering                                                                    #
# --------------------------------------------------------------------------- #
def test_label_distinguishes_rerank() -> None:
    assert ConfigOutcome("fixed", "hybrid", rerank=False).label == "hybrid (RRF)"
    assert ConfigOutcome("fixed", "hybrid", rerank=True).label == "hybrid (RRF) + rerank"
    assert ConfigOutcome("fixed", "dense", rerank=False).label == "dense"


def test_format_number_marks_a_missing_metric() -> None:
    """A metric that does not exist must never render as 0.00."""
    assert format_number(None) == "-"
    assert format_number(0.5) == "0.50"


def test_render_table_is_markdown_with_one_row_per_config() -> None:
    result = ConfigOutcome(strategy="fixed", mode="dense", rerank=False)
    result.outcomes = [_outcome(["hit.md"], ["hit.md"])]
    table = render_table([result], [1, 5])

    lines = table.splitlines()
    assert lines[0].startswith("| Chunking | Retrieval | recall@1 | recall@5 |")
    assert lines[1].startswith("|---|---|")
    assert "`fixed`" in lines[2]
    assert "1.00" in lines[2]


# --------------------------------------------------------------------------- #
# paired comparison                                                            #
# --------------------------------------------------------------------------- #
def _cell(name: str, outcomes: list[QueryOutcome]) -> ConfigOutcome:
    cell = ConfigOutcome(strategy=name, mode="hybrid", rerank=False)
    cell.outcomes = outcomes
    return cell


def test_paired_comparison_counts_per_question_wins() -> None:
    a = _cell("a", [_outcome(["hit.md"], ["hit.md"], "q1"), _outcome(["x.md"], ["hit.md"], "q2")])
    b = _cell("b", [_outcome(["x.md"], ["hit.md"], "q1"), _outcome(["hit.md"], ["hit.md"], "q2")])
    result = paired_comparison(a, b, 1)
    assert (result.a_wins, result.b_wins, result.ties) == (1, 1, 0)


def test_paired_comparison_counts_ties() -> None:
    a = _cell("a", [_outcome(["hit.md"], ["hit.md"], "q1")])
    b = _cell("b", [_outcome(["hit.md"], ["hit.md"], "q1")])
    assert paired_comparison(a, b, 1).ties == 1


def test_paired_comparison_pairs_by_query_id_not_position() -> None:
    """Cells could be ordered differently; pairing by index would compare
    unrelated questions and produce a number that looks fine and means nothing."""
    a = _cell("a", [_outcome(["hit.md"], ["hit.md"], "q1"), _outcome(["x.md"], ["hit.md"], "q2")])
    b = _cell("b", [_outcome(["x.md"], ["hit.md"], "q2"), _outcome(["hit.md"], ["hit.md"], "q1")])
    result = paired_comparison(a, b, 1)
    assert (result.a_wins, result.b_wins, result.ties) == (0, 0, 2)


def test_paired_comparison_excludes_questions_without_ground_truth() -> None:
    """They carry no information either way, so counting them as ties would
    dilute the head-to-head record toward parity."""
    a = _cell("a", [_outcome(["x.md"], [], "ungradable")])
    b = _cell("b", [_outcome(["y.md"], [], "ungradable")])
    result = paired_comparison(a, b, 1)
    assert (result.a_wins, result.b_wins, result.ties, result.decided) == (0, 0, 0, 0)
