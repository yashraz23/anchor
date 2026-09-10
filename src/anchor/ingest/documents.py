"""Parse source files into documents.

Parsing is deliberately kept free of I/O beyond reading a file, and free of any
database dependency, so every function here is unit-testable against a fixture
string. The pipeline module is what touches Postgres.
"""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

# Punctuation that reStructuredText accepts as a section underline.
_RST_UNDERLINE = re.compile(r"^([=\-`:.'\"~^_*+#])\1{2,}\s*$")
_MD_H1 = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
_FRONT_MATTER = re.compile(r"\A---\r?\n.*?\r?\n---\r?\n", re.DOTALL)


@dataclass(frozen=True)
class ParsedDocument:
    """One source file, ready to be written to `documents`.

    Version and commit are attached by the pipeline from the RepoSnapshot, not
    here, so that parsing stays independent of how the checkout was obtained.
    """

    source_path: str
    title: str | None
    text: str
    content_hash: str


def content_hash(text: str) -> str:
    """Stable hash of document text.

    Lets a re-ingest at a new commit skip documents whose content did not
    actually change, which matters because vLLM's docs churn far less than its
    commit history suggests.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def strip_front_matter(text: str) -> str:
    """Remove a leading YAML front-matter block.

    Front matter is metadata for the docs renderer. Left in place it becomes
    retrievable text that answers no question anyone asks.
    """
    return _FRONT_MATTER.sub("", text, count=1)


def markdown_title(text: str) -> str | None:
    """The first level-one ATX heading, if there is one.

    Only `# ` counts. A setext underline is rare in these docs and a false
    positive title is worse than no title, since the title is prepended to
    chunks and therefore ends up in the embedded text.
    """
    match = _MD_H1.search(text)
    return match.group(1).strip() if match else None


def rst_title(text: str) -> str | None:
    """The first reStructuredText section title.

    Handles both the underlined form and the overlined-and-underlined form. The
    underline has to be at least as long as the title, which is what
    distinguishes a real title from a horizontal rule or a table border.
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or not _RST_UNDERLINE.match(line):
            continue

        # Overline form: the title sits between two rules of the same character.
        following = lines[i + 1].strip() if i + 1 < len(lines) else ""
        after = lines[i + 2] if i + 2 < len(lines) else ""
        if following and _RST_UNDERLINE.match(after) and len(stripped) >= len(following):
            return following

        # Underline form: the title is the line above.
        if i > 0:
            candidate = lines[i - 1].strip()
            if candidate and len(stripped) >= len(candidate):
                return candidate
    return None


def parse_markup(source_path: str, raw: str) -> ParsedDocument:
    """Parse a Markdown or reStructuredText file."""
    text = strip_front_matter(raw).strip()
    suffix = Path(source_path).suffix.lower()
    title = rst_title(text) if suffix == ".rst" else markdown_title(text)
    if title is None:
        # Fall back to the filename so a document is never anonymous in results.
        title = Path(source_path).stem.replace("-", " ").replace("_", " ")
    return ParsedDocument(
        source_path=source_path,
        title=title,
        text=text,
        content_hash=content_hash(text),
    )


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
    returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"{prefix}{node.name}({ast.unparse(node.args)}){returns}"


def _is_public(name: str) -> bool:
    return not name.startswith("_")


def _target_name(stmt: ast.AST) -> str | None:
    """The attribute name bound by an assignment, if it is a plain name."""
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        return stmt.target.id
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        target = stmt.targets[0]
        if isinstance(target, ast.Name):
            return target.id
    return None


def attribute_docs(node: ast.ClassDef) -> list[tuple[str, str, str]]:
    """Attribute docstrings on a class, as (name, declaration, docstring).

    An attribute docstring is a bare string expression directly after an
    assignment::

        max_num_seqs: int = Field(default=DEFAULT_MAX_NUM_SEQS, ge=1)
        \"\"\"Maximum number of sequences processed in a single iteration.\"\"\"

    `ast.get_docstring` does not see these, because they belong to no function
    or class. Missing them is not a minor gap for this corpus: vLLM's engine
    arguments and config keys are documented almost entirely this way, and its
    rendered engine-arguments page is generated from them at docs build time
    rather than committed. Without this the authoritative description of every
    flag is absent from the corpus.
    """
    found: list[tuple[str, str, str]] = []
    body = node.body
    for i, stmt in enumerate(body[:-1]):
        name = _target_name(stmt)
        if name is None or not _is_public(name):
            continue
        following = body[i + 1]
        if not isinstance(following, ast.Expr):
            continue
        if not (
            isinstance(following.value, ast.Constant) and isinstance(following.value.value, str)
        ):
            continue
        # The declaration carries the type and default, which is half the answer
        # to "what does this flag do and what is it set to by default".
        found.append((name, ast.unparse(stmt), following.value.value.strip()))
    return found


def parse_docstrings(source_path: str, raw: str) -> ParsedDocument | None:
    """Assemble the public docstrings of one Python module into a document.

    Only public symbols are included. Private helpers are not what a user asks
    about, and including them dilutes the corpus with text that can never be the
    right answer.

    Each symbol is rendered with its signature above its docstring, so that the
    retrieved span carries the API shape and not just the prose. Returns None
    when the module has nothing public and documented, and on a syntax error,
    because one unparseable file must not fail an ingest of thousands.
    """
    try:
        tree = ast.parse(raw)
    except SyntaxError:
        return None

    module_name = Path(source_path).with_suffix("").as_posix().replace("/", ".")
    parts: list[str] = [f"# {module_name}"]

    module_doc = ast.get_docstring(tree)
    if module_doc:
        parts.append(module_doc.strip())

    documented = False
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and _is_public(node.name):
            class_doc = ast.get_docstring(node)
            if class_doc:
                parts.append(f"## class {node.name}\n\n{class_doc.strip()}")
                documented = True
            for field_name, declaration, field_doc in attribute_docs(node):
                parts.append(
                    f"### {node.name}.{field_name}\n\n```python\n{declaration}\n```\n\n{field_doc}"
                )
                documented = True
            for member in node.body:
                if not isinstance(member, ast.FunctionDef | ast.AsyncFunctionDef):
                    continue
                if not _is_public(member.name):
                    continue
                member_doc = ast.get_docstring(member)
                if member_doc:
                    sig = _signature(member)
                    parts.append(
                        f"### {node.name}.{member.name}\n\n```python\n{sig}\n```\n\n{member_doc.strip()}"
                    )
                    documented = True
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and _is_public(node.name):
            func_doc = ast.get_docstring(node)
            if func_doc:
                parts.append(
                    f"## {node.name}\n\n```python\n{_signature(node)}\n```\n\n{func_doc.strip()}"
                )
                documented = True

    if not documented and not module_doc:
        return None

    text = "\n\n".join(parts).strip()
    return ParsedDocument(
        source_path=source_path,
        title=module_name,
        text=text,
        content_hash=content_hash(text),
    )
