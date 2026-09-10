"""Fixed-size chunking: the naive baseline.

A sliding window of `target_tokens` with `overlap_tokens` of overlap, blind to
document structure. It is in the project to be beaten, and the point of keeping
it is that "structure-aware chunking is better" is a claim worth a number rather
than an assertion.

Its known failure mode is deliberate and measured, not worked around: a window
boundary lands wherever the token count says, so a fenced code block gets cut in
half and neither half is a runnable command.
"""

from __future__ import annotations

from anchor.chunk.base import Chunk
from anchor.tokenizer import Tokenizer


def chunk_fixed(
    text: str,
    tokenizer: Tokenizer,
    target_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    """Split `text` into overlapping windows of roughly `target_tokens`.

    Windows are cut at token boundaries but sliced out of the original string by
    character offset, so the stored text is the source text exactly. Decoding
    token ids back to text would lowercase it and normalise its whitespace,
    which would corrupt every code example in the corpus.
    """
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    if not 0 <= overlap_tokens < target_tokens:
        # Equal would make the window never advance and loop forever.
        raise ValueError("overlap_tokens must be non-negative and below target_tokens")

    offsets = tokenizer.offsets(text)
    if not offsets:
        return []

    stride = target_tokens - overlap_tokens
    chunks: list[Chunk] = []
    start = 0
    ordinal = 0

    while start < len(offsets):
        window = offsets[start : start + target_tokens]
        char_start = window[0][0]
        char_end = window[-1][1]
        span = text[char_start:char_end].strip()
        if span:
            chunks.append(
                Chunk(
                    ordinal=ordinal,
                    text=span,
                    token_count=len(window),
                    # No structure was consulted, so there is no heading to
                    # record. That absence is itself part of the comparison.
                    heading_path=None,
                    is_code_block=False,
                )
            )
            ordinal += 1
        if start + target_tokens >= len(offsets):
            break
        start += stride

    return chunks
