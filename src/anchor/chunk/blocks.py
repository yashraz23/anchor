"""Split a Markdown document into blocks.

The only structure the chunkers need is: where the headings are, and where the
fenced code blocks are. Everything else is prose.

Fences are tracked first and headings second, which is the whole point. A `#`
inside a shell block is a comment, not a section title, and a chunker that reads
it as a heading will split a command in half. vLLM's docs are full of exactly
that shape:

    ```bash
    # serve a model
    vllm serve meta-llama/Llama-3.1-8B --max-num-seqs 256
    ```
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

# Up to three leading spaces is still a fence or a heading; four makes it an
# indented code block, which CommonMark treats as content.
_FENCE = re.compile(r"^ {0,3}(?P<char>`{3,}|~{3,})(?P<info>[^`]*)$")
_HEADING = re.compile(r"^ {0,3}(?P<hashes>#{1,6})\s+(?P<title>.+?)\s*#*\s*$")


class BlockKind(StrEnum):
    HEADING = "heading"
    CODE = "code"
    TEXT = "text"


@dataclass(frozen=True)
class Block:
    kind: BlockKind
    text: str
    # Heading depth, 1 to 6. Zero for anything that is not a heading.
    level: int = 0
    # The language tag on a fence, when there is one. Useful later for deciding
    # whether a code span is a CLI invocation worth handing to the oracle.
    info: str = ""


def _close_fence(line: str, opening: str) -> bool:
    """Whether `line` closes a fence opened with `opening`.

    A closing fence uses the same character and is at least as long. A shorter
    run does not close it, which is what lets a block contain a literal fence.
    """
    match = _FENCE.match(line)
    if match is None:
        return False
    char = match.group("char")
    return char[0] == opening[0] and len(char) >= len(opening) and not match.group("info").strip()


def parse_blocks(text: str) -> list[Block]:
    """Parse `text` into headings, fenced code blocks, and prose.

    An unterminated fence runs to the end of the document rather than being
    discarded or reinterpreted as prose. Truncated examples do occur in real
    docs, and reinterpreting one would put stray `#` comment lines back in play
    as headings.
    """
    blocks: list[Block] = []
    lines = text.splitlines()
    buffer: list[str] = []

    def flush_text() -> None:
        if buffer:
            body = "\n".join(buffer).strip()
            if body:
                blocks.append(Block(kind=BlockKind.TEXT, text=body))
            buffer.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        fence = _FENCE.match(line)

        if fence is not None:
            flush_text()
            opening = fence.group("char")
            info = fence.group("info").strip()
            fenced = [line]
            i += 1
            while i < len(lines):
                fenced.append(lines[i])
                if _close_fence(lines[i], opening):
                    i += 1
                    break
                i += 1
            blocks.append(Block(kind=BlockKind.CODE, text="\n".join(fenced), info=info))
            continue

        heading = _HEADING.match(line)
        if heading is not None:
            flush_text()
            blocks.append(
                Block(
                    kind=BlockKind.HEADING,
                    text=heading.group("title").strip(),
                    level=len(heading.group("hashes")),
                )
            )
            i += 1
            continue

        buffer.append(line)
        i += 1

    flush_text()
    return blocks


def heading_path(stack: list[Block], separator: str = " > ") -> str:
    """Render a heading stack as a breadcrumb.

    Stored on the chunk and optionally prepended to its text, so a chunk carries
    the context of where it sat in the document. A retrieved span reading
    "set this to 0.9" is useless without "Engine Arguments > GPU memory".
    """
    return separator.join(block.text for block in stack)
