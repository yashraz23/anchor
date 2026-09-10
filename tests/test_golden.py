"""Golden-set schema tests.

The validation here is deliberately strict. A malformed golden entry does not
crash anything downstream, it just quietly produces a wrong number, and wrong
numbers in the README are the one failure mode this project cannot tolerate.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from anchor.evaluate.golden import (
    Category,
    Difficulty,
    GoldenError,
    GoldenQuery,
    dump_golden,
    load_golden,
    parse_entry,
)

VALID = {
    "id": "vllm-6641-max-num-seqs",
    "question": "What do max_num_seqs and max_model_len do?",
    "source_url": "https://github.com/vllm-project/vllm/issues/6641",
    "category": "config",
    "difficulty": "easy",
    "expected_source_paths": ["vllm/config/scheduler.py"],
    "expected_answer": "max_num_seqs caps sequences per iteration.",
}


def test_parse_a_valid_entry() -> None:
    query = parse_entry(dict(VALID), 0)
    assert query.category is Category.CONFIG
    assert query.difficulty is Difficulty.EASY
    assert query.expected_source_paths == ("vllm/config/scheduler.py",)


@pytest.mark.parametrize(
    "missing", ["id", "question", "source_url", "category", "difficulty", "expected_source_paths"]
)
def test_every_required_field_is_enforced(missing: str) -> None:
    entry = dict(VALID)
    del entry[missing]
    with pytest.raises(GoldenError, match=missing):
        parse_entry(entry, 0)


def test_an_empty_required_field_is_rejected() -> None:
    """Present-but-blank is the likelier mistake than absent."""
    entry = dict(VALID) | {"expected_source_paths": []}
    with pytest.raises(GoldenError, match="expected_source_paths"):
        parse_entry(entry, 0)


def test_source_url_must_be_a_github_link() -> None:
    """The whole argument for this set is that the questions are real. An entry
    that cannot be traced back to an issue cannot be defended."""
    entry = dict(VALID) | {"source_url": "https://example.invalid/made-up"}
    with pytest.raises(GoldenError, match="source_url"):
        parse_entry(entry, 0)


def test_an_unknown_category_is_rejected() -> None:
    entry = dict(VALID) | {"category": "vibes"}
    with pytest.raises(GoldenError):
        parse_entry(entry, 0)


def test_an_unknown_difficulty_is_rejected() -> None:
    entry = dict(VALID) | {"difficulty": "spicy"}
    with pytest.raises(GoldenError):
        parse_entry(entry, 0)


def test_load_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "golden.yaml"
    path.write_text(yaml.safe_dump({"queries": [dict(VALID), dict(VALID)]}), encoding="utf-8")
    with pytest.raises(GoldenError, match="duplicate"):
        load_golden(path)


def test_load_an_empty_set_is_not_an_error(tmp_path: Path) -> None:
    """The harness has to be runnable before the set is full."""
    path = tmp_path / "golden.yaml"
    path.write_text("queries: []\n", encoding="utf-8")
    assert load_golden(path) == []


def test_load_a_missing_file_is_not_an_error(tmp_path: Path) -> None:
    assert load_golden(tmp_path / "nope.yaml") == []


def test_round_trip(tmp_path: Path) -> None:
    query = parse_entry(dict(VALID), 0)
    path = tmp_path / "golden.yaml"
    path.write_text(dump_golden([query]), encoding="utf-8")
    assert load_golden(path) == [query]


def test_dump_preserves_unicode() -> None:
    query = GoldenQuery(
        id="x",
        question="Why does 🚀 appear in the docs?",
        source_url="https://github.com/vllm-project/vllm/issues/1",
        category=Category.CONCEPT,
        difficulty=Difficulty.EASY,
        expected_source_paths=("README.md",),
    )
    assert "🚀" in dump_golden([query])


def test_the_checked_in_golden_set_is_valid() -> None:
    """The real file, validated on every run.

    This is the guard that matters: it fails the build the moment a hand-edited
    entry is malformed, rather than at eval time when a number is already wrong.
    """
    load_golden(Path(__file__).resolve().parents[1] / "data" / "golden" / "golden.yaml")
