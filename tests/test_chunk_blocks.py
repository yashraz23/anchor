"""Block parsing tests.

The case that matters most is a `#` comment inside a shell fence. vLLM's docs
are full of that shape, and reading one as a heading is what splits a command in
half.
"""

from __future__ import annotations

from anchor.chunk.blocks import BlockKind, heading_path, parse_blocks

DOC = """# Engine Arguments

Tune the server with these flags.

## GPU memory

```bash
# serve a model
vllm serve meta-llama/Llama-3.1-8B --max-num-seqs 256
```

Set the fraction to 0.9.
"""


def test_hash_inside_a_fence_is_not_a_heading() -> None:
    blocks = parse_blocks(DOC)
    headings = [b.text for b in blocks if b.kind is BlockKind.HEADING]
    assert headings == ["Engine Arguments", "GPU memory"]
    assert "serve a model" not in headings


def test_a_fence_is_one_block_with_its_language() -> None:
    code = [b for b in parse_blocks(DOC) if b.kind is BlockKind.CODE]
    assert len(code) == 1
    assert code[0].info == "bash"
    assert "vllm serve" in code[0].text
    assert code[0].text.startswith("```bash")
    assert code[0].text.rstrip().endswith("```")


def test_heading_levels_are_recorded() -> None:
    blocks = parse_blocks(DOC)
    levels = [b.level for b in blocks if b.kind is BlockKind.HEADING]
    assert levels == [1, 2]


def test_tilde_fences_are_supported() -> None:
    blocks = parse_blocks("~~~python\nx = 1\n~~~\n")
    assert [b.kind for b in blocks] == [BlockKind.CODE]


def test_a_shorter_run_does_not_close_a_longer_fence() -> None:
    """A block can contain a literal fence, which is how docs show markdown."""
    text = "````markdown\n```bash\nvllm serve\n```\n````\n"
    blocks = parse_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].kind is BlockKind.CODE
    assert "vllm serve" in blocks[0].text


def test_an_unterminated_fence_runs_to_the_end() -> None:
    """Reinterpreting it as prose would put stray '#' lines back in play."""
    blocks = parse_blocks("Intro.\n\n```bash\n# a comment\nvllm serve\n")
    assert [b.kind for b in blocks] == [BlockKind.TEXT, BlockKind.CODE]
    assert "# a comment" in blocks[1].text


def test_an_indented_fence_is_not_a_fence() -> None:
    """Four spaces makes it indented code, which CommonMark treats as content."""
    blocks = parse_blocks("Intro.\n\n    ```\n    not a fence\n")
    assert all(b.kind is not BlockKind.CODE for b in blocks)


def test_heading_path_renders_a_breadcrumb() -> None:
    blocks = [b for b in parse_blocks(DOC) if b.kind is BlockKind.HEADING]
    assert heading_path(blocks) == "Engine Arguments > GPU memory"
