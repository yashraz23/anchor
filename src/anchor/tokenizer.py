"""Tokenization for chunking.

Chunk sizes are counted in the embedding model's own tokens, not in words or
characters, because the binding constraint is that encoder's 512-token input
limit. A chunk measured in the wrong units silently overflows and gets truncated
at embed time, and a truncated chunk is one whose tail can never be retrieved.

The chunkers depend on the `Tokenizer` protocol rather than on transformers, so
they are unit-testable without downloading a model.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol

Offsets = list[tuple[int, int]]


class Tokenizer(Protocol):
    """The slice of tokenizer behaviour the chunkers actually need."""

    def count(self, text: str) -> int:
        """Number of tokens in `text`, excluding special tokens."""
        ...

    def offsets(self, text: str) -> Offsets:
        """Character (start, end) spans of each token in `text`.

        Offsets rather than token ids on purpose. Splitting by decoding token ids
        back to text is lossy with a WordPiece vocabulary: it lowercases, drops
        the distinction between whitespace runs, and mangles anything outside the
        vocabulary. Slicing the original string at token boundaries splits on
        exact token counts while preserving the source text byte for byte, which
        matters because this text is what gets shown to a user and fed to the
        generator.
        """
        ...


class HFTokenizer:
    """A `Tokenizer` backed by a Hugging Face fast tokenizer."""

    def __init__(self, model_name: str) -> None:
        self._tok = _load(model_name)

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._tok(text, add_special_tokens=False)["input_ids"])

    def offsets(self, text: str) -> Offsets:
        if not text:
            return []
        encoded = self._tok(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            # Long documents are chunked precisely so they do not have to be
            # truncated; truncating here would discard the tail of every file
            # over the model limit before chunking ever saw it.
            truncation=False,
        )
        return [(int(start), int(end)) for start, end in encoded["offset_mapping"]]


@lru_cache(maxsize=4)
def _load(model_name: str):  # type: ignore[no-untyped-def]
    """Load and cache a fast tokenizer.

    Cached because chunking calls this once per document and loading is far
    slower than tokenizing.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if not tok.is_fast:  # pragma: no cover - every model in the stack is fast
        raise RuntimeError(
            f"{model_name} has no fast tokenizer, so character offsets are unavailable"
        )
    return tok
