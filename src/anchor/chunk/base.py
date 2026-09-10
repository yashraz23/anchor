"""The chunk record produced by every strategy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """One retrievable span.

    `ordinal` is the chunk's position within its document under one strategy, so
    (document_id, strategy, ordinal) identifies it uniquely and re-chunking is
    idempotent.
    """

    ordinal: int
    text: str
    token_count: int
    heading_path: str | None = None
    # True when the span is entirely a fenced code block. Recorded because the
    # comparison this project exists to make is about code blocks specifically:
    # they are what users ask about and what fixed-size chunking destroys.
    is_code_block: bool = False
