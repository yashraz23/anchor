"""Structure-aware chunking.

Splits on heading boundaries and treats a fenced code block as atomic. Two rules
carry the whole strategy:

1. A code block is never split, even when it alone exceeds max_tokens. Half a
   `vllm serve` invocation is not a worse answer than the whole one, it is a
   wrong one, and an over-long chunk merely gets truncated at its tail by the
   encoder rather than being made unrunnable.
2. A chunk carries its heading path. "Set this to 0.9" is not retrievable
   guidance; "Engine Arguments > GPU memory: set this to 0.9" is.

Both are testable claims, which is why they are rules rather than heuristics.
"""

from __future__ import annotations

from anchor.chunk.base import Chunk
from anchor.chunk.blocks import Block, BlockKind, heading_path, parse_blocks
from anchor.tokenizer import Tokenizer


def _emit(
    parts: list[Block],
    stack: list[Block],
    tokenizer: Tokenizer,
    prepend_heading: bool,
    ordinal: int,
) -> Chunk | None:
    """Turn accumulated blocks into a chunk, or None when there is nothing."""
    body = "\n\n".join(part.text for part in parts).strip()
    if not body:
        return None

    path = heading_path(stack) if stack else None
    text = f"{path}\n\n{body}" if (prepend_heading and path) else body
    return Chunk(
        ordinal=ordinal,
        text=text,
        token_count=tokenizer.count(text),
        heading_path=path,
        # Only when the span is nothing but code. A code block with its
        # surrounding explanation is a prose chunk that happens to contain code.
        is_code_block=all(part.kind is BlockKind.CODE for part in parts),
    )


def chunk_structure_aware(
    text: str,
    tokenizer: Tokenizer,
    max_tokens: int,
    min_tokens: int,
    prepend_heading_path: bool = True,
    never_split_code_blocks: bool = True,
) -> list[Chunk]:
    """Split `text` on structure, packing blocks up to `max_tokens`.

    A heading closes the current chunk and updates the heading stack. Blocks
    accumulate until adding the next one would exceed `max_tokens`, at which
    point the chunk is emitted and the next one begins.

    `min_tokens` prevents a stub chunk: a section too small to stand alone stays
    with the section that follows it, rather than becoming a fragment that wins
    retrieval on a keyword and then answers nothing.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if min_tokens < 0 or min_tokens >= max_tokens:
        raise ValueError("min_tokens must be non-negative and below max_tokens")

    blocks = parse_blocks(text)
    chunks: list[Chunk] = []
    stack: list[Block] = []
    pending: list[Block] = []
    pending_tokens = 0
    ordinal = 0

    def flush() -> None:
        nonlocal pending, pending_tokens, ordinal
        chunk = _emit(pending, stack, tokenizer, prepend_heading_path, ordinal)
        if chunk is not None:
            chunks.append(chunk)
            ordinal += 1
        pending = []
        pending_tokens = 0

    for block in blocks:
        if block.kind is BlockKind.HEADING:
            # A section boundary closes the current chunk, unless what is
            # pending is too small to stand on its own, in which case it rides
            # along into the next section.
            if pending_tokens >= min_tokens:
                flush()
            # Pop to the parent of this level, then push. A level-2 heading
            # replaces the previous level-2 and everything under it.
            while stack and stack[-1].level >= block.level:
                stack.pop()
            stack.append(block)
            continue

        block_tokens = tokenizer.count(block.text)
        is_atomic_code = block.kind is BlockKind.CODE and never_split_code_blocks

        if pending and pending_tokens + block_tokens > max_tokens:
            flush()

        if block_tokens > max_tokens and not is_atomic_code:
            # Oversized prose under one heading, with no structure left to split
            # on. Fall back to a token window rather than emitting a chunk the
            # encoder would silently truncate.
            for piece in _split_oversized(block.text, tokenizer, max_tokens):
                pending = [Block(kind=BlockKind.TEXT, text=piece)]
                pending_tokens = tokenizer.count(piece)
                flush()
            continue

        pending.append(block)
        pending_tokens += block_tokens

        # An atomic code block already over the limit cannot grow further, so
        # close it immediately rather than appending prose it would push out.
        if is_atomic_code and pending_tokens >= max_tokens:
            flush()

    flush()
    return chunks


def _split_oversized(text: str, tokenizer: Tokenizer, max_tokens: int) -> list[str]:
    """Cut over-long prose at token boundaries, preserving the source text."""
    offsets = tokenizer.offsets(text)
    pieces: list[str] = []
    for start in range(0, len(offsets), max_tokens):
        window = offsets[start : start + max_tokens]
        piece = text[window[0][0] : window[-1][1]].strip()
        if piece:
            pieces.append(piece)
    return pieces
