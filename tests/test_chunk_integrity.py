"""Tests for the code-block integrity metric.

Written before the measurement was run, so the number the README will carry
comes from logic that was specified first rather than fitted to a result.
"""

from __future__ import annotations

from anchor.chunk.fixed import chunk_fixed
from anchor.chunk.structure import chunk_structure_aware
from anchor.evaluate.chunk_integrity import (
    IntegrityResult,
    code_block_bodies,
    count_intact,
)
from tests.conftest import FakeTokenizer

DOC = """# Serving

Start the server like this.

```bash
vllm serve meta-llama/Llama-3.1-8B --gpu-memory-utilization 0.9 --max-num-seqs 256
```

Then send it a request.
"""


def test_code_block_bodies_drops_the_fences() -> None:
    bodies = code_block_bodies(DOC)
    assert len(bodies) == 1
    assert bodies[0].startswith("vllm serve")
    assert "```" not in bodies[0]


def test_code_block_bodies_ignores_prose() -> None:
    assert code_block_bodies("# H\n\nJust text.\n") == []


def test_code_block_bodies_handles_an_unterminated_fence() -> None:
    bodies = code_block_bodies("```bash\nvllm serve\n")
    assert bodies == ["vllm serve"]


def test_count_intact_requires_the_whole_body() -> None:
    bodies = ["vllm serve --max-num-seqs 256"]
    assert count_intact(bodies, ["vllm serve --max-num-seqs 256 extra"]) == 1
    # Split across two chunks: neither half is a runnable command.
    assert count_intact(bodies, ["vllm serve --max", "-num-seqs 256"]) == 0


def test_rate_is_zero_when_there_are_no_code_blocks() -> None:
    assert IntegrityResult(strategy="fixed", code_blocks=0, intact=0).rate == 0.0


def test_rate_is_a_fraction() -> None:
    assert IntegrityResult(strategy="fixed", code_blocks=4, intact=1).rate == 0.25


def test_structure_aware_beats_fixed_on_the_same_document(
    tokenizer: FakeTokenizer,
) -> None:
    """The metric has to be able to tell the two strategies apart at all."""
    bodies = code_block_bodies(DOC)
    fixed = [c.text for c in chunk_fixed(DOC, tokenizer, 6, 0)]
    aware = [c.text for c in chunk_structure_aware(DOC, tokenizer, 6, 0)]

    assert count_intact(bodies, fixed) == 0
    assert count_intact(bodies, aware) == len(bodies)
