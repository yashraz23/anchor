"""Reciprocal Rank Fusion tests.

Written before the retrieval pipeline was run, per the convention for anything
in retrieve/. Fusion consumes only ranks, so it is fully testable on hand-built
lists with no database and no model.
"""

from __future__ import annotations

import pytest

from anchor.retrieve.fusion import recall_at_k, reciprocal_rank_fusion


def test_a_document_ranked_first_by_both_wins() -> None:
    fused = reciprocal_rank_fusion({"dense": [7, 1, 2], "sparse": [7, 3, 4]}, k=60)
    assert fused[0].chunk_id == 7


def test_score_is_the_sum_of_reciprocal_ranks() -> None:
    fused = reciprocal_rank_fusion({"dense": [1], "sparse": [1]}, k=60)
    assert fused[0].score == pytest.approx(2 / 61)


def test_a_document_in_one_list_only_still_surfaces() -> None:
    """Absence contributes nothing rather than a penalty, which is what lets a
    keyword-only match survive fusion with a dense list that missed it."""
    fused = reciprocal_rank_fusion({"dense": [1, 2], "sparse": [99]}, k=60)
    assert 99 in [f.chunk_id for f in fused]


def test_consensus_beats_a_single_first_place() -> None:
    """The property that makes RRF worth using.

    Document 5 is second in both lists; document 1 is first in one and absent
    from the other. Agreement across retrievers outranks one strong opinion.
    """
    fused = reciprocal_rank_fusion({"dense": [1, 5], "sparse": [9, 5]}, k=60)
    assert fused[0].chunk_id == 5


def test_k_damps_the_top_rank_advantage() -> None:
    """Small k makes rank 1 dominant; large k flattens the curve.

    The agreeing document has to sit *deep* in both lists for this to show. Two
    second places always beat one first place, at every positive k, because
    2/(k+2) > 1/(k+1) reduces to k > 0. So the contest here is one first place
    against agreement at rank 5:

        k = 1   1/2   = 0.500  beats  2/6  = 0.333
        k = 60  1/61  = 0.016  loses  2/65 = 0.031

    That crossover is the knob's whole effect, and why it is a config value
    rather than a constant.
    """
    lists = {"dense": [1, 2, 3, 4, 5], "sparse": [9, 8, 7, 6, 5]}
    assert reciprocal_rank_fusion(lists, k=1)[0].chunk_id == 1
    assert reciprocal_rank_fusion(lists, k=60)[0].chunk_id == 5


def test_two_second_places_always_beat_one_first_place() -> None:
    """The inequality above, asserted directly across the plausible range."""
    lists = {"dense": [1, 5], "sparse": [9, 5]}
    for k in (1, 10, 60, 200):
        assert reciprocal_rank_fusion(lists, k=k)[0].chunk_id == 5


def test_ranks_are_recorded_for_debugging() -> None:
    fused = reciprocal_rank_fusion({"dense": [4, 1], "sparse": [1]}, k=60)
    top = next(f for f in fused if f.chunk_id == 1)
    assert top.ranks == {"dense": 2, "sparse": 1}


def test_ties_break_deterministically_on_chunk_id() -> None:
    """A run has to be reproducible from its config."""
    first = reciprocal_rank_fusion({"a": [3, 1, 2]}, k=60)
    second = reciprocal_rank_fusion({"a": [3, 1, 2]}, k=60)
    assert [f.chunk_id for f in first] == [f.chunk_id for f in second]

    tied = reciprocal_rank_fusion({"a": [5], "b": [2]}, k=60)
    assert [f.chunk_id for f in tied] == [2, 5]


def test_top_k_truncates() -> None:
    fused = reciprocal_rank_fusion({"a": [1, 2, 3, 4, 5]}, k=60, top_k=2)
    assert len(fused) == 2


def test_empty_input() -> None:
    assert reciprocal_rank_fusion({"dense": [], "sparse": []}, k=60) == []


def test_k_must_be_positive() -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        reciprocal_rank_fusion({"a": [1]}, k=0)


# --------------------------------------------------------------------------- #
# recall@k                                                                     #
# --------------------------------------------------------------------------- #
def test_recall_counts_only_the_top_k() -> None:
    assert recall_at_k([1, 2, 3, 4], relevant=[3], k=2) == 0.0
    assert recall_at_k([1, 2, 3, 4], relevant=[3], k=3) == 1.0


def test_recall_with_several_relevant_chunks() -> None:
    assert recall_at_k([1, 2, 3], relevant=[1, 9], k=3) == 0.5


def test_recall_is_none_when_nothing_is_relevant() -> None:
    """A query with no ground truth is unanswerable by this metric.

    Scoring it 0.0 or 1.0 would quietly move the headline number in the README.
    """
    assert recall_at_k([1, 2], relevant=[], k=2) is None
