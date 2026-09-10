"""Shared fixtures.

The fake tokenizer keeps chunking tests offline, fast, and readable: one token
per whitespace-delimited word means an expected token count can be checked by
eye, and a failure points at the chunker rather than at a model download.

The real tokenizer is exercised separately, in the slow-marked test that proves
offsets round-trip against the actual bge vocabulary.
"""

from __future__ import annotations

import re

import pytest

from anchor.tokenizer import Offsets

_WORD = re.compile(r"\S+")


class FakeTokenizer:
    """One token per whitespace-delimited word."""

    def count(self, text: str) -> int:
        return len(_WORD.findall(text))

    def offsets(self, text: str) -> Offsets:
        return [(m.start(), m.end()) for m in _WORD.finditer(text)]


@pytest.fixture
def tokenizer() -> FakeTokenizer:
    return FakeTokenizer()
