"""Ingest orchestration: checkout, parse, store."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from anchor import db
from anchor.config import Settings
from anchor.ingest.documents import ParsedDocument, parse_docstrings, parse_markup
from anchor.ingest.repo import RepoSnapshot, prepare_checkout
from anchor.ingest.store import upsert_document

logger = logging.getLogger(__name__)


@dataclass
class IngestReport:
    """What one ingest actually did. Printed by the CLI and worth logging."""

    commit_sha: str
    vllm_version: str
    markup_documents: int = 0
    docstring_documents: int = 0
    inserted: int = 0
    updated: int = 0
    skipped_short: int = 0
    skipped_unparseable: int = 0

    @property
    def total(self) -> int:
        return self.markup_documents + self.docstring_documents


def _read(path: Path) -> str:
    # vLLM's tree contains files with mixed encodings; a decode error must not
    # abort an ingest of thousands of files.
    return path.read_text(encoding="utf-8", errors="replace")


def _iter_matches(root: Path, globs: tuple[str, ...]) -> Iterator[Path]:
    """Files matching any glob, de-duplicated and ordered deterministically.

    Ordering matters: `documents.id` is assigned in insertion order, and a
    reproducible run needs the same file to get the same id.
    """
    seen: set[Path] = set()
    for pattern in globs:
        for path in sorted(root.glob(pattern)):
            if path.is_file() and path not in seen:
                seen.add(path)
                yield path


def parse_corpus(
    snapshot: RepoSnapshot,
    settings: Settings,
    report: IngestReport,
) -> Iterator[ParsedDocument]:
    """Every document in the corpus, markup first then docstrings."""
    root = snapshot.root
    ingest = settings.ingest

    for path in _iter_matches(root, ingest.doc_globs):
        raw = _read(path)
        source_path = path.relative_to(root).as_posix()
        doc = parse_markup(source_path, raw)
        if len(doc.text) < ingest.min_document_chars:
            report.skipped_short += 1
            continue
        report.markup_documents += 1
        yield doc

    if not ingest.include_docstrings:
        return

    for path in _iter_matches(root, ingest.docstring_globs):
        source_path = path.relative_to(root).as_posix()
        parsed = parse_docstrings(source_path, _read(path))
        if parsed is None:
            report.skipped_unparseable += 1
            continue
        if len(parsed.text) < ingest.min_document_chars:
            report.skipped_short += 1
            continue
        report.docstring_documents += 1
        yield parsed


def ingest(settings: Settings) -> IngestReport:
    """Run the full ingest and return what it did.

    The commit is resolved once, up front, and every document written in this
    run carries it. Resolving per document would let a fetch racing mid-run
    produce a corpus that spans two versions of vLLM.
    """
    snapshot = prepare_checkout(settings.ingest)
    logger.info("ingesting vLLM %s at %s", snapshot.vllm_version, snapshot.commit_sha[:12])

    report = IngestReport(
        commit_sha=snapshot.commit_sha,
        vllm_version=snapshot.vllm_version,
    )

    with db.connect(settings) as conn:
        for doc in parse_corpus(snapshot, settings, report):
            _, inserted = upsert_document(conn, doc, snapshot)
            if inserted:
                report.inserted += 1
            else:
                report.updated += 1
        conn.commit()

    return report
