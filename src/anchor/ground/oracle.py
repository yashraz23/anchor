"""The symbol oracle.

This is what makes a subset of faithfulness checks ground-truthed rather than
LLM-judged. `--max-num-seqs` either exists in vLLM's argument parser at a given
commit or it does not, and no judge's opinion changes that.

Symbols are extracted from the source tree by walking the AST, not by importing
vLLM or running `--help`. Importing would need the package installed with its
CUDA dependencies, and running `--help` would need a working GPU environment.
Neither is available in CI, and the oracle has to run there.

What the oracle can and cannot say, stated plainly because it bounds every
verdict built on it:

- **Present** is certain. The symbol is in the source at the pinned commit.
- **Absent** means "not found in the scanned tree at this commit". That is
  strong evidence of a hallucination for a flag, since flags are declared as
  string literals in one place, but it is evidence rather than proof: a symbol
  built dynamically at runtime would be missed.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass

from anchor.config import SymbolKind

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Symbol:
    """One extracted symbol, ready for the `symbols` table."""

    name: str
    kind: SymbolKind
    source_file: str
    signature: str = ""


def _is_public(name: str) -> bool:
    return not name.startswith("_")


def extract_cli_flags(tree: ast.AST, source_file: str) -> list[Symbol]:
    """Every `--flag` passed as a literal to an add_argument call.

    vLLM declares its command-line surface as string literals, so this recovers
    the flag list exactly. A call can declare several at once, as with a short
    and long form, so every leading positional string is taken rather than only
    the first.
    """
    found: list[Symbol] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name != "add_argument":
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if arg.value.startswith("--"):
                    found.append(
                        Symbol(
                            name=arg.value,
                            kind=SymbolKind.CLI_FLAG,
                            source_file=source_file,
                        )
                    )
            else:
                # A non-literal positional means the flag name is computed. Stop
                # rather than guess: a wrong flag in the oracle is worse than a
                # missing one, because it would license a false "supported".
                break
    return found


def extract_config_keys(tree: ast.AST, source_file: str) -> list[Symbol]:
    """Annotated public fields of public classes.

    These are vLLM's configuration surface. Undocumented fields are included:
    the oracle answers whether a key exists, which is independent of whether
    anyone wrote prose about it.
    """
    found: list[Symbol] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or not _is_public(node.name):
            continue
        for stmt in node.body:
            if not isinstance(stmt, ast.AnnAssign) or not isinstance(stmt.target, ast.Name):
                continue
            field = stmt.target.id
            if not _is_public(field):
                continue
            found.append(
                Symbol(
                    name=field,
                    kind=SymbolKind.CONFIG_KEY,
                    source_file=source_file,
                    signature=ast.unparse(stmt),
                )
            )
    return found


def extract_api_symbols(tree: ast.Module, source_file: str) -> list[Symbol]:
    """Public top-level classes and functions.

    Only top-level. A method is reached through its class, and flattening
    methods into the same namespace would let the oracle confirm a name that
    exists on some unrelated object.
    """
    found: list[Symbol] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and _is_public(node.name):
            found.append(
                Symbol(
                    name=node.name,
                    kind=SymbolKind.CLASS,
                    source_file=source_file,
                    signature=f"class {node.name}",
                )
            )
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and _is_public(node.name):
            prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
            found.append(
                Symbol(
                    name=node.name,
                    kind=SymbolKind.FUNCTION,
                    source_file=source_file,
                    signature=f"{prefix}{node.name}({ast.unparse(node.args)})",
                )
            )
    return found


def extract_symbols(source: str, source_file: str) -> list[Symbol]:
    """Every symbol in one module. Returns nothing on a syntax error.

    One unparseable file must not fail an oracle build over thousands.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    return [
        *extract_cli_flags(tree, source_file),
        *extract_config_keys(tree, source_file),
        *extract_api_symbols(tree, source_file),
    ]


def find_flag_source_classes(tree: ast.AST) -> set[str]:
    """Classes whose fields vLLM turns into CLI flags automatically.

    vLLM declares its command line two ways. Most of it is explicit string
    literals, which `extract_cli_flags` recovers. The rest is generated in a
    loop::

        frontend_kwargs = get_kwargs(cls)
        for key, value in frontend_kwargs.items():
            group.add_argument(*extra, f"--{key.replace('_', '-')}", **value)

    Nothing there is a literal, so a literals-only oracle reports every one of
    those flags as non-existent. `--tool-call-parser`, `--chat-template` and
    `--enable-auto-tool-choice` are all real and all invisible to the first
    mechanism, and the oracle called them hallucinations until this landed.

    Detected structurally rather than by name: a function that both calls
    get_kwargs and passes an f-string to add_argument is generating flags from
    that class's fields. `get_kwargs(cls)` inside a classmethod resolves to the
    enclosing class.
    """
    sources: set[str] = set()
    for class_node in ast.walk(tree):
        if not isinstance(class_node, ast.ClassDef):
            continue
        for func in class_node.body:
            if not isinstance(func, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            sources |= _flag_sources_in_function(func, class_node.name)
    return sources


def _flag_sources_in_function(
    func: ast.FunctionDef | ast.AsyncFunctionDef, enclosing_class: str
) -> set[str]:
    generates = False
    targets: set[str] = set()

    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        name = (
            node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        )
        if name == "add_argument" and any(isinstance(arg, ast.JoinedStr) for arg in node.args):
            generates = True
        elif name == "get_kwargs" and node.args:
            first = node.args[0]
            if isinstance(first, ast.Name):
                targets.add(enclosing_class if first.id == "cls" else first.id)

    return targets if generates else set()


def class_annotations(tree: ast.AST) -> dict[str, tuple[list[str], list[str]]]:
    """Public annotated field names and base-class names, keyed by class name.

    Bases are kept because a subclass inherits its parent's fields, and so
    inherits its parent's flags: FrontendArgs extends BaseFrontendArgs and
    exposes everything the base declares.
    """
    out: dict[str, tuple[list[str], list[str]]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        fields = [
            stmt.target.id
            for stmt in node.body
            if isinstance(stmt, ast.AnnAssign)
            and isinstance(stmt.target, ast.Name)
            and _is_public(stmt.target.id)
        ]
        bases = [b.id for b in node.bases if isinstance(b, ast.Name)]
        out[node.name] = (fields, bases)
    return out


def flag_name(field: str) -> str:
    """The flag vLLM generates for a field name."""
    return f"--{field.replace('_', '-')}"


def derive_flags(
    sources: set[str], annotations: dict[str, tuple[list[str], list[str]]]
) -> set[str]:
    """Flags implied by the fields of every flag-source class, bases included."""
    flags: set[str] = set()

    def collect(class_name: str, seen: set[str]) -> None:
        if class_name in seen or class_name not in annotations:
            return
        seen.add(class_name)
        fields, bases = annotations[class_name]
        flags.update(flag_name(f) for f in fields)
        for base in bases:
            collect(base, seen)

    for source in sources:
        collect(source, set())
    return flags
