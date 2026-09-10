"""Route a claim to a verifier and reach a verdict.

Claims mentioning something checkable go to the symbol oracle, which is exact.
Everything else falls through to span attribution and the LLM judge. The split
is the point of the project: a flag either exists or it does not, and that part
of faithfulness should never be decided by a model's opinion.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import psycopg
from psycopg.rows import DictRow

from anchor.config import GroundSettings, SymbolKind, Verdict, Verifier
from anchor.ground.jurisdiction import flags_in_vllm_commands

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SymbolMention:
    """A checkable name found in a claim."""

    name: str
    kind: SymbolKind


@dataclass(frozen=True)
class OracleVerdict:
    verdict: Verdict
    verifier: Verifier
    mentions: tuple[SymbolMention, ...] = ()
    missing: tuple[SymbolMention, ...] = ()

    @property
    def checkable(self) -> bool:
        return bool(self.mentions)


# Inline code is where flags and config keys are written in a real answer. Prose
# mentions are matched too, but a name inside backticks is the strong signal.
_INLINE_CODE = re.compile(r"`([^`\n]+)`")


def find_flags(text: str, pattern: str) -> set[str]:
    """CLI flags mentioned in a claim.

    A trailing `=value` is stripped: `--gpu-memory-utilization=0.9` mentions the
    flag, and the value is not part of its name.
    """
    flags: set[str] = set()
    for match in re.finditer(pattern, text):
        flags.add(match.group(0).split("=")[0].rstrip(".,;:"))
    return flags


def find_config_keys(text: str) -> set[str]:
    """Config keys mentioned in a claim, taken from inline code only.

    Restricted to backticked spans on purpose. Matching bare snake_case in prose
    would sweep up ordinary English written with underscores and, worse, any
    identifier appearing in a sentence, producing confident verdicts about names
    the writer never claimed existed.
    """
    keys: set[str] = set()
    for match in _INLINE_CODE.finditer(text):
        token = match.group(1).strip()
        # A bare identifier, not an expression or a call.
        if re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+", token):
            keys.add(token)
    return keys


def extract_mentions(text: str, settings: GroundSettings) -> list[SymbolMention]:
    """Everything in a claim the oracle is entitled to check.

    Only flags inside a vLLM invocation. See ground.jurisdiction for why this is
    narrow: judging every flag in an answer produced a 42% "not found" rate on
    real answers, almost all of it flags belonging to pip and numactl that the
    model had used correctly.

    Config keys are deliberately not adjudicated. A backticked identifier is as
    likely to be a config *value*, a variable from the model's own example, or
    another library's parameter as it is a vLLM field, and the oracle cannot
    tell the difference from the name alone. Those claims go to the LLM judge,
    which is the fallback the design already has.
    """
    return [
        SymbolMention(name=flag, kind=SymbolKind.CLI_FLAG)
        for flag in sorted(flags_in_vllm_commands(text, settings.cli_flag_pattern))
    ]


def symbol_exists(
    conn: psycopg.Connection[DictRow], mention: SymbolMention, vllm_version: str
) -> bool:
    row = conn.execute(
        "SELECT 1 FROM symbols WHERE name = %s AND kind = %s AND vllm_version = %s",
        (mention.name, mention.kind.value, vllm_version),
    ).fetchone()
    return row is not None


def verify_with_oracle(
    conn: psycopg.Connection[DictRow],
    claim: str,
    settings: GroundSettings,
    vllm_version: str,
) -> OracleVerdict:
    """Check every checkable name in a claim against the source tree.

    A claim naming a symbol that does not exist is `contradicted`, not merely
    unsupported: the source says otherwise. A claim the oracle cannot check is
    `unverifiable` here and falls through to the judge, rather than being
    counted as supported by default.
    """
    mentions = extract_mentions(claim, settings)
    if not mentions:
        return OracleVerdict(verdict=Verdict.UNVERIFIABLE, verifier=Verifier.SYMBOL_ORACLE)

    missing = tuple(m for m in mentions if not symbol_exists(conn, m, vllm_version))
    return OracleVerdict(
        verdict=Verdict.CONTRADICTED if missing else Verdict.SUPPORTED,
        verifier=Verifier.SYMBOL_ORACLE,
        mentions=tuple(mentions),
        missing=missing,
    )
