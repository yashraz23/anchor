"""Populate the `symbols` table from the pinned vLLM checkout."""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field

from anchor import db
from anchor.config import Settings, SymbolKind
from anchor.ground.oracle import (
    Symbol,
    class_annotations,
    derive_flags,
    extract_symbols,
    find_flag_source_classes,
)
from anchor.ingest.repo import prepare_checkout

logger = logging.getLogger(__name__)

_UPSERT = """
INSERT INTO symbols (name, kind, source_file, signature, vllm_version)
VALUES (%(name)s, %(kind)s, %(source_file)s, %(signature)s, %(vllm_version)s)
ON CONFLICT (name, kind, vllm_version) DO UPDATE
    SET source_file = EXCLUDED.source_file,
        signature = EXCLUDED.signature
"""


@dataclass
class OracleReport:
    vllm_version: str
    commit_sha: str
    files_scanned: int = 0
    derived_flags: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.by_kind.values())


def build_oracle(settings: Settings) -> OracleReport:
    """Walk the checkout and record every symbol it declares.

    Pinned to the ingested commit, like every stage after ingest. An oracle
    built from a different tree than the corpus would judge answers against
    symbols the retrieved documents never described.
    """
    with db.connect(settings) as conn:
        commit = settings.ingest.vllm_commit or db.latest_ingested_commit(conn)
    if commit is None:
        raise RuntimeError("no documents ingested; run `anchor ingest` first")

    snapshot = prepare_checkout(settings.ingest, commit=commit)
    report = OracleReport(vllm_version=snapshot.vllm_version, commit_sha=snapshot.commit_sha)
    for kind in SymbolKind:
        report.by_kind[kind.value] = 0

    # De-duplicated in memory before writing. The same class name occurs in many
    # modules, and the table is keyed on (name, kind, version), so writing every
    # occurrence would be thousands of redundant upserts.
    seen: set[tuple[str, str]] = set()

    # Accumulated across the whole tree, because a flag-generating class can
    # inherit fields from a base declared in another module.
    annotations: dict[str, tuple[list[str], list[str]]] = {}
    flag_sources: set[str] = set()

    with db.connect(settings) as conn:
        for path in sorted(snapshot.root.glob("vllm/**/*.py")):
            if not path.is_file():
                continue
            source_file = path.relative_to(snapshot.root).as_posix()
            report.files_scanned += 1

            source = path.read_text(encoding="utf-8", errors="replace")
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            annotations.update(class_annotations(tree))
            flag_sources |= find_flag_source_classes(tree)

            for symbol in extract_symbols(source, source_file):
                key = (symbol.name, symbol.kind.value)
                if key in seen:
                    continue
                seen.add(key)
                conn.execute(
                    _UPSERT,
                    {
                        "name": symbol.name,
                        "kind": symbol.kind.value,
                        "source_file": symbol.source_file,
                        "signature": symbol.signature,
                        "vllm_version": snapshot.vllm_version,
                    },
                )
                report.by_kind[symbol.kind.value] += 1

            if report.files_scanned % 500 == 0:
                conn.commit()
                logger.info("scanned %d files", report.files_scanned)

        # Second pass: flags vLLM generates from dataclass fields rather than
        # declaring as literals. Resolved only now, because a generating class
        # may inherit from a base in a module scanned later.
        for flag in sorted(derive_flags(flag_sources, annotations)):
            key = (flag, SymbolKind.CLI_FLAG.value)
            if key in seen:
                continue
            seen.add(key)
            derived = Symbol(
                name=flag, kind=SymbolKind.CLI_FLAG, source_file="(derived from fields)"
            )
            conn.execute(
                _UPSERT,
                {
                    "name": derived.name,
                    "kind": derived.kind.value,
                    "source_file": derived.source_file,
                    "signature": derived.signature,
                    "vllm_version": snapshot.vllm_version,
                },
            )
            report.by_kind[SymbolKind.CLI_FLAG.value] += 1
            report.derived_flags += 1

        conn.commit()

    return report
