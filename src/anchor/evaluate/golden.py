"""The golden set: schema, loading, and validation.

Every entry must cite the issue it came from. An entry without a `source_url`
cannot be defended as a real user question, and the whole argument for this set
is that it is a real query distribution rather than a synthetic one.

`expected_source_paths` holds document paths, not chunk ids. Chunk ids are only
stable within one (strategy, commit) pair, so pinning them in a version
controlled file would silently rot the moment anything is re-chunked. The link
step resolves paths to ids at evaluation time instead.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml


class Category(StrEnum):
    CLI_FLAG = "cli_flag"
    CONFIG = "config"
    API = "api"
    CONCEPT = "concept"
    TROUBLESHOOTING = "troubleshooting"
    PERF = "perf"


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class GoldenError(ValueError):
    """Raised when the golden set is malformed."""


@dataclass(frozen=True)
class GoldenQuery:
    """One evaluated question.

    `question` is the phrasing evaluated, normalised from the issue title or
    body. It stays close to how the user actually asked: rewriting it into the
    documentation's vocabulary would erase the very gap this set exists to
    measure.
    """

    id: str
    question: str
    source_url: str
    category: Category
    difficulty: Difficulty
    # Documents that contain the answer. Retrieval is scored on finding these.
    expected_source_paths: tuple[str, ...]
    # Written from the documentation, never from a model's output.
    expected_answer: str = ""
    notes: str = ""
    expected_chunk_ids: tuple[int, ...] = field(default_factory=tuple)


def _require(entry: dict[str, Any], key: str, index: int) -> Any:
    if key not in entry or entry[key] in (None, "", []):
        raise GoldenError(f"golden entry {index}: missing required field {key!r}")
    return entry[key]


def parse_entry(entry: dict[str, Any], index: int) -> GoldenQuery:
    """Validate and build one entry, failing loudly on anything malformed."""
    try:
        category = Category(_require(entry, "category", index))
    except ValueError as exc:
        raise GoldenError(f"golden entry {index}: {exc}") from exc
    try:
        difficulty = Difficulty(_require(entry, "difficulty", index))
    except ValueError as exc:
        raise GoldenError(f"golden entry {index}: {exc}") from exc

    url = str(_require(entry, "source_url", index))
    if not url.startswith("https://github.com/"):
        raise GoldenError(
            f"golden entry {index}: source_url must be a GitHub issue or discussion, got {url!r}"
        )

    return GoldenQuery(
        id=str(_require(entry, "id", index)),
        question=str(_require(entry, "question", index)),
        source_url=url,
        category=category,
        difficulty=difficulty,
        expected_source_paths=tuple(
            str(p) for p in _require(entry, "expected_source_paths", index)
        ),
        expected_answer=str(entry.get("expected_answer", "")),
        notes=str(entry.get("notes", "")),
        expected_chunk_ids=tuple(int(i) for i in entry.get("expected_chunk_ids", ())),
    )


def load_golden(path: Path) -> list[GoldenQuery]:
    """Load and validate the golden set.

    An empty or missing file is not an error: the set is built up over time, and
    the evaluation harness has to be runnable before it is full.
    """
    if not path.is_file():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = raw.get("queries") or []
    queries = [parse_entry(entry, i) for i, entry in enumerate(entries)]

    seen: set[str] = set()
    for query in queries:
        if query.id in seen:
            raise GoldenError(f"duplicate golden id {query.id!r}")
        seen.add(query.id)
    return queries


def _plain(value: Any) -> Any:
    """Reduce a field to something yaml.safe_dump can represent.

    A StrEnum is a str subclass, but PyYAML's safe representer dispatches on the
    exact type and refuses the subclass, so the conversion has to be explicit.
    """
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    return value


def dump_golden(queries: list[GoldenQuery]) -> str:
    """Serialise the golden set back to YAML."""
    payload = {"queries": [{k: _plain(v) for k, v in asdict(q).items()} for q in queries]}
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=88)
