"""Parsing tests. Fixtures are inline strings, so nothing here touches the
network, the filesystem, or Postgres.

The docstring cases use shapes that actually occur in vLLM's tree: a module with
a public class carrying documented methods, a private helper that must not be
picked up, and a file that does not parse.
"""

from __future__ import annotations

from anchor.ingest.documents import (
    content_hash,
    markdown_title,
    parse_docstrings,
    parse_markup,
    rst_title,
    strip_front_matter,
)

MD_WITH_FRONT_MATTER = """---
title: Engine Arguments
weight: 3
---

# Engine Arguments

Pass `--max-num-seqs` to cap concurrent sequences.
"""

RST_UNDERLINED = """Quickstart
==========

Install vLLM and serve a model.
"""

RST_OVERLINED = """==========
Deployment
==========

Serve with the OpenAI-compatible server.
"""


def test_strip_front_matter_removes_only_the_leading_block() -> None:
    out = strip_front_matter(MD_WITH_FRONT_MATTER)
    assert "weight: 3" not in out
    assert "Engine Arguments" in out
    assert "--max-num-seqs" in out


def test_strip_front_matter_leaves_a_document_without_one_untouched() -> None:
    text = "# Title\n\nBody.\n"
    assert strip_front_matter(text) == text


def test_strip_front_matter_does_not_eat_a_later_horizontal_rule() -> None:
    text = "# Title\n\nBefore.\n\n---\n\nAfter.\n"
    assert strip_front_matter(text) == text


def test_markdown_title() -> None:
    assert markdown_title("# Engine Arguments\n\nBody") == "Engine Arguments"
    assert markdown_title("## Only a subheading\n") is None


def test_rst_title_underlined() -> None:
    assert rst_title(RST_UNDERLINED) == "Quickstart"


def test_rst_title_overlined() -> None:
    assert rst_title(RST_OVERLINED) == "Deployment"


def test_rst_title_ignores_an_underline_shorter_than_its_title() -> None:
    """A short rule under a long line is a table border, not a section title."""
    assert rst_title("A very long section heading\n===\n") is None


def test_parse_markup_uses_front_matter_stripped_text_for_the_hash() -> None:
    doc = parse_markup("docs/engine_args.md", MD_WITH_FRONT_MATTER)
    assert doc.title == "Engine Arguments"
    assert doc.source_path == "docs/engine_args.md"
    assert "weight: 3" not in doc.text
    assert doc.content_hash == content_hash(doc.text)


def test_parse_markup_falls_back_to_the_filename() -> None:
    doc = parse_markup("docs/multi-node_serving.md", "No heading here.\n")
    assert doc.title == "multi node serving"


def test_parse_markup_dispatches_on_suffix() -> None:
    doc = parse_markup("docs/quickstart.rst", RST_UNDERLINED)
    assert doc.title == "Quickstart"


MODULE = '''
"""Engine entrypoints."""


class LLM:
    """An offline inference engine."""

    def generate(self, prompts: list[str], n: int = 1) -> list[str]:
        """Run generation over prompts."""

    def _internal(self) -> None:
        """Private and must not be ingested."""


def _helper() -> None:
    """Private module function."""


async def serve(host: str = "0.0.0.0") -> None:
    """Start the OpenAI-compatible server."""
'''


def test_parse_docstrings_collects_public_symbols_with_signatures() -> None:
    doc = parse_docstrings("vllm/entrypoints/llm.py", MODULE)
    assert doc is not None
    assert doc.title == "vllm.entrypoints.llm"
    assert "Engine entrypoints." in doc.text
    assert "## class LLM" in doc.text
    assert "An offline inference engine." in doc.text
    # The signature is carried alongside the prose so a retrieved span shows the
    # API shape, which is what the symbol oracle later checks against.
    assert "def generate(self, prompts: list[str], n: int=1) -> list[str]" in doc.text
    assert "async def serve(host: str='0.0.0.0') -> None" in doc.text


def test_parse_docstrings_excludes_private_symbols() -> None:
    doc = parse_docstrings("vllm/entrypoints/llm.py", MODULE)
    assert doc is not None
    assert "_internal" not in doc.text
    assert "_helper" not in doc.text
    assert "Private and must not be ingested." not in doc.text


def test_parse_docstrings_returns_none_for_an_undocumented_module() -> None:
    assert parse_docstrings("vllm/x.py", "class A:\n    pass\n") is None


def test_parse_docstrings_returns_none_on_a_syntax_error() -> None:
    """One unparseable file must not fail an ingest of thousands."""
    assert parse_docstrings("vllm/broken.py", "def f(:\n") is None


def test_content_hash_is_stable_and_content_sensitive() -> None:
    assert content_hash("abc") == content_hash("abc")
    assert content_hash("abc") != content_hash("abd")


# --------------------------------------------------------------------------- #
# attribute docstrings                                                         #
# --------------------------------------------------------------------------- #
# vLLM documents its engine arguments and config keys almost entirely as bare
# string expressions after a field assignment. ast.get_docstring cannot see
# those, and the rendered engine-arguments page is generated at docs build time
# rather than committed, so without this the authoritative description of every
# flag is missing from the corpus entirely.
CONFIG_MODULE = '''
"""Scheduler configuration."""


class SchedulerConfig:
    """Configuration for the scheduler."""

    max_num_seqs: int = Field(default=DEFAULT_MAX_NUM_SEQS, ge=1)
    """Maximum number of sequences to be processed in a single iteration."""

    enable_chunked_prefill: bool = True
    """Whether to enable chunked prefill."""

    _private_knob: int = 3
    """Private and must not be ingested."""

    undocumented: int = 5

    def method(self) -> None:
        """A real method docstring."""
'''


def test_attribute_docstrings_are_captured() -> None:
    doc = parse_docstrings("vllm/config/scheduler.py", CONFIG_MODULE)
    assert doc is not None
    assert "### SchedulerConfig.max_num_seqs" in doc.text
    assert "Maximum number of sequences to be processed" in doc.text
    assert "enable_chunked_prefill" in doc.text


def test_attribute_declaration_carries_type_and_default() -> None:
    """Half the answer to "what does this flag do" is its type and default."""
    doc = parse_docstrings("vllm/config/scheduler.py", CONFIG_MODULE)
    assert doc is not None
    assert "max_num_seqs: int = Field(default=DEFAULT_MAX_NUM_SEQS, ge=1)" in doc.text


def test_private_attributes_are_excluded() -> None:
    doc = parse_docstrings("vllm/config/scheduler.py", CONFIG_MODULE)
    assert doc is not None
    assert "_private_knob" not in doc.text
    assert "Private and must not be ingested." not in doc.text


def test_an_undocumented_attribute_is_not_emitted() -> None:
    doc = parse_docstrings("vllm/config/scheduler.py", CONFIG_MODULE)
    assert doc is not None
    assert "undocumented" not in doc.text


def test_attribute_docs_alone_make_a_module_worth_ingesting() -> None:
    """A config class with only field docs still has to produce a document."""
    source = 'class C:\n    x: int = 1\n    """The x knob."""\n'
    doc = parse_docstrings("vllm/config/x.py", source)
    assert doc is not None
    assert "The x knob." in doc.text


def test_a_string_not_following_an_assignment_is_not_an_attribute_doc() -> None:
    source = 'class C:\n    """Class doc."""\n\n    x: int = 1\n'
    doc = parse_docstrings("vllm/config/x.py", source)
    assert doc is not None
    assert "### C.x" not in doc.text
