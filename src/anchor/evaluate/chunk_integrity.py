"""Code-block integrity: does a chunking strategy keep commands runnable?

The metric is deliberately not LLM-judged and not a proxy. For every fenced code
block in the corpus, ask a yes-or-no question: does some chunk under this
strategy contain that block's body verbatim? A block split across two chunks
answers no, and neither half is a runnable command.

This is the objective half of the week-1 comparison. recall@k needs the golden
set, but whether a strategy destroys code blocks is measurable today, and it is
the mechanism that should explain the recall difference when it arrives.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import psycopg
from psycopg.rows import DictRow

from anchor.chunk.blocks import BlockKind, parse_blocks

# Blocks are bucketed by length because the aggregate rate hides the effect.
# A fixed window only breaks a block when the block is long enough to straddle a
# boundary, so a corpus of mostly one-line commands makes the naive strategy look
# far better than it is on the multi-line examples people actually copy.
BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("<=32", 0, 32),
    ("33-64", 33, 64),
    ("65-128", 65, 128),
    ("129-256", 129, 256),
    (">256", 257, 10**9),
)


def bucket_for(token_count: int) -> str:
    for name, low, high in BUCKETS:
        if low <= token_count <= high:
            return name
    raise ValueError(f"no bucket for {token_count}")


@dataclass(frozen=True)
class IntegrityResult:
    strategy: str
    code_blocks: int
    intact: int
    # bucket name -> (blocks, intact)
    by_size: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def rate(self) -> float:
        """Fraction of code blocks that survive whole. Zero blocks reads as 0.0."""
        return self.intact / self.code_blocks if self.code_blocks else 0.0

    def rate_for(self, bucket: str) -> float | None:
        """Intact rate within one size bucket, or None when it holds no blocks.

        None rather than 0.0, for the same reason an unpriced model costs None:
        an empty bucket must not read as total failure in a results table.
        """
        blocks, intact = self.by_size.get(bucket, (0, 0))
        return intact / blocks if blocks else None


def code_block_bodies(text: str) -> list[str]:
    """The body of every fenced code block, without its fence lines.

    The fences are dropped because they are markup, not content: a chunker may
    legitimately keep the body and lose a delimiter. What must survive is the
    command itself.
    """
    bodies: list[str] = []
    for block in parse_blocks(text):
        if block.kind is not BlockKind.CODE:
            continue
        lines = block.text.splitlines()
        # First line is the opening fence; the last is the closing one when the
        # block was terminated.
        inner = lines[1:-1] if len(lines) > 2 and lines[-1].strip() else lines[1:]
        body = "\n".join(inner).strip()
        if body:
            bodies.append(body)
    return bodies


def count_intact(bodies: list[str], chunk_texts: list[str]) -> int:
    """How many bodies appear whole inside at least one chunk."""
    return sum(1 for body in bodies if any(body in chunk for chunk in chunk_texts))


def chunk_texts_for(
    conn: psycopg.Connection[DictRow], document_id: int, strategy: str
) -> list[str]:
    rows = conn.execute(
        "SELECT text FROM chunks WHERE document_id = %s AND strategy = %s ORDER BY ordinal",
        (document_id, strategy),
    ).fetchall()
    return [str(row["text"]) for row in rows]
