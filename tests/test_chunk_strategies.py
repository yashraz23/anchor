"""Chunking strategy tests.

The headline assertion is the last one: on the same document, fixed-size
chunking splits a code block and structure-aware chunking does not. That is the
comparison the project's first result rests on, so it is pinned by a test rather
than left to be observed once and asserted about later.
"""

from __future__ import annotations

import pytest

from anchor.chunk.base import Chunk
from anchor.chunk.fixed import chunk_fixed
from anchor.chunk.structure import chunk_structure_aware
from tests.conftest import FakeTokenizer

DOC = """# Engine Arguments

Tune the vLLM server with the flags below to trade throughput against latency.

## GPU memory

```bash
vllm serve meta-llama/Llama-3.1-8B --gpu-memory-utilization 0.9 --max-num-seqs 256
```

Raise the fraction when the server has the card to itself.

## Scheduling

Concurrency is capped by the scheduler and by available KV cache blocks.
"""


# --------------------------------------------------------------------------- #
# fixed                                                                        #
# --------------------------------------------------------------------------- #
def test_fixed_windows_overlap(tokenizer: FakeTokenizer) -> None:
    text = " ".join(str(i) for i in range(20))
    chunks = chunk_fixed(text, tokenizer, target_tokens=8, overlap_tokens=3)
    assert chunks[0].token_count == 8
    # A stride of 5 means chunk two starts at token 5, so tokens 5 to 7 repeat.
    assert chunks[0].text.split()[5:] == chunks[1].text.split()[:3]


def test_fixed_preserves_source_text_exactly(tokenizer: FakeTokenizer) -> None:
    """Slicing by offset, not decoding token ids, is what keeps case and
    whitespace intact. A decoded WordPiece round-trip would lowercase this."""
    text = "Serve  meta-llama/Llama-3.1-8B with --max-num-seqs 256"
    chunks = chunk_fixed(text, tokenizer, target_tokens=100, overlap_tokens=0)
    assert chunks[0].text == text


def test_fixed_records_no_heading(tokenizer: FakeTokenizer) -> None:
    chunks = chunk_fixed(DOC, tokenizer, target_tokens=20, overlap_tokens=5)
    assert all(c.heading_path is None for c in chunks)


def test_fixed_rejects_an_overlap_that_would_not_advance(
    tokenizer: FakeTokenizer,
) -> None:
    with pytest.raises(ValueError, match="overlap_tokens"):
        chunk_fixed("a b c", tokenizer, target_tokens=4, overlap_tokens=4)


def test_fixed_on_empty_text(tokenizer: FakeTokenizer) -> None:
    assert chunk_fixed("", tokenizer, target_tokens=10, overlap_tokens=2) == []


def test_fixed_ordinals_are_contiguous(tokenizer: FakeTokenizer) -> None:
    chunks = chunk_fixed(DOC, tokenizer, target_tokens=15, overlap_tokens=4)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


# --------------------------------------------------------------------------- #
# structure_aware                                                              #
# --------------------------------------------------------------------------- #
def test_structure_aware_never_splits_a_code_block(tokenizer: FakeTokenizer) -> None:
    """The rule the whole strategy exists for.

    max_tokens is set below the size of the code block on purpose: even then the
    command has to survive whole.
    """
    chunks = chunk_structure_aware(DOC, tokenizer, max_tokens=6, min_tokens=0)
    code = [c for c in chunks if "vllm serve" in c.text]
    assert len(code) == 1
    assert "--gpu-memory-utilization 0.9" in code[0].text
    assert "--max-num-seqs 256" in code[0].text


def test_structure_aware_carries_the_heading_path(tokenizer: FakeTokenizer) -> None:
    chunks = chunk_structure_aware(DOC, tokenizer, max_tokens=40, min_tokens=0)
    paths = {c.heading_path for c in chunks}
    assert "Engine Arguments > GPU memory" in paths
    assert "Engine Arguments > Scheduling" in paths


def test_structure_aware_prepends_the_heading_to_the_text(
    tokenizer: FakeTokenizer,
) -> None:
    """The heading is embedded with the chunk, not just stored beside it."""
    chunks = chunk_structure_aware(
        DOC, tokenizer, max_tokens=40, min_tokens=0, prepend_heading_path=True
    )
    scheduling = next(c for c in chunks if (c.heading_path or "").endswith("Scheduling"))
    assert scheduling.text.startswith("Engine Arguments > Scheduling")


def test_structure_aware_can_omit_the_prepended_heading(
    tokenizer: FakeTokenizer,
) -> None:
    chunks = chunk_structure_aware(
        DOC, tokenizer, max_tokens=40, min_tokens=0, prepend_heading_path=False
    )
    scheduling = next(c for c in chunks if (c.heading_path or "").endswith("Scheduling"))
    assert not scheduling.text.startswith("Engine Arguments")
    assert scheduling.heading_path == "Engine Arguments > Scheduling"


def test_a_sibling_heading_replaces_its_predecessor(tokenizer: FakeTokenizer) -> None:
    """A level-2 heading pops the previous level-2, it does not nest under it."""
    chunks = chunk_structure_aware(DOC, tokenizer, max_tokens=40, min_tokens=0)
    assert all("GPU memory > Scheduling" not in (c.heading_path or "") for c in chunks)


def test_code_only_span_is_flagged(tokenizer: FakeTokenizer) -> None:
    chunks = chunk_structure_aware(DOC, tokenizer, max_tokens=6, min_tokens=0)
    code = next(c for c in chunks if "vllm serve" in c.text)
    assert code.is_code_block is True


def test_prose_containing_code_is_not_flagged_as_code(
    tokenizer: FakeTokenizer,
) -> None:
    """A code block packed together with its explanation is a prose chunk."""
    chunks = chunk_structure_aware(DOC, tokenizer, max_tokens=200, min_tokens=0)
    mixed = next(c for c in chunks if "Raise the fraction" in c.text)
    assert mixed.is_code_block is False


def test_oversized_prose_is_split_rather_than_truncated(
    tokenizer: FakeTokenizer,
) -> None:
    """Prose has no atomicity claim, so a wall of text is windowed instead of
    being emitted as a chunk the encoder would silently truncate."""
    text = "# H\n\n" + " ".join(str(i) for i in range(100))
    chunks = chunk_structure_aware(text, tokenizer, max_tokens=10, min_tokens=0)
    assert len(chunks) > 1
    assert all(c.token_count <= 15 for c in chunks)


def test_structure_aware_ordinals_are_contiguous(tokenizer: FakeTokenizer) -> None:
    chunks = chunk_structure_aware(DOC, tokenizer, max_tokens=20, min_tokens=0)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_min_tokens_keeps_a_stub_with_the_next_section(
    tokenizer: FakeTokenizer,
) -> None:
    """A one-line section should not become a chunk that wins on a keyword and
    then answers nothing."""
    text = "## Tiny\n\nOne line.\n\n## Real\n\n" + " ".join(str(i) for i in range(30))
    chunks = chunk_structure_aware(text, tokenizer, max_tokens=60, min_tokens=10)
    assert any("One line." in c.text and "0 1 2" in c.text for c in chunks)


# --------------------------------------------------------------------------- #
# the comparison                                                               #
# --------------------------------------------------------------------------- #
def test_fixed_splits_the_command_that_structure_aware_keeps_whole(
    tokenizer: FakeTokenizer,
) -> None:
    """The project's first expected finding, pinned as a test.

    On the same document at the same budget, the naive strategy cuts the serve
    command across two chunks so neither is runnable, while the structure-aware
    one keeps it intact.
    """
    budget = 6
    fixed = chunk_fixed(DOC, tokenizer, target_tokens=budget, overlap_tokens=0)
    aware = chunk_structure_aware(DOC, tokenizer, max_tokens=budget, min_tokens=0)

    def whole(chunks: list[Chunk]) -> int:
        return sum(
            1 for c in chunks if "--gpu-memory-utilization" in c.text and "--max-num-seqs" in c.text
        )

    assert whole(fixed) == 0
    assert whole(aware) == 1
