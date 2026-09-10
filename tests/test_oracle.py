"""Symbol oracle extraction.

The oracle is the reason a subset of faithfulness checks are ground-truthed
rather than judged, so a wrong entry is worse than a missing one: a spurious
symbol licenses a false "supported" verdict that nothing downstream will catch.
The tests lean accordingly.

Fixtures use the shapes that actually occur in vLLM's tree.
"""

from __future__ import annotations

from anchor.config import SymbolKind
from anchor.ground.oracle import (
    extract_api_symbols,
    extract_cli_flags,
    extract_config_keys,
    extract_symbols,
)

ARG_UTILS = '''
class EngineArgs:
    """Engine arguments."""

    @staticmethod
    def add_cli_args(parser):
        model_group = parser.add_argument_group("model")
        model_group.add_argument("--model", **kwargs["model"])
        model_group.add_argument("--max-num-seqs", type=int, default=None)
        parser.add_argument("-q", "--quiet", action="store_true")
        parser.add_argument(computed_name, type=str)
'''

CONFIG_MODULE = '''
class SchedulerConfig:
    """Scheduler config."""

    max_num_seqs: int = Field(default=128, ge=1)
    """Maximum sequences per iteration."""

    enable_chunked_prefill: bool = True

    _private: int = 0

    untyped = 5

    def method(self) -> None:
        """A method, not a config key."""


class _PrivateConfig:
    hidden: int = 1
'''


# --------------------------------------------------------------------------- #
# CLI flags                                                                    #
# --------------------------------------------------------------------------- #
def test_flags_are_recovered_from_add_argument_literals() -> None:
    names = {s.name for s in extract_cli_flags_from(ARG_UTILS)}
    assert "--model" in names
    assert "--max-num-seqs" in names


def test_several_flags_from_one_call_are_all_recovered() -> None:
    """A short and long form are declared together; taking only the first
    positional would silently drop the long form."""
    assert "--quiet" in {s.name for s in extract_cli_flags_from(ARG_UTILS)}


def test_a_computed_flag_name_is_skipped_not_guessed() -> None:
    """A wrong flag in the oracle is worse than a missing one: it would license
    a false 'supported' verdict that nothing downstream catches."""
    names = {s.name for s in extract_cli_flags_from(ARG_UTILS)}
    assert not any(n.startswith("computed") for n in names)


def test_short_flags_are_not_recorded_as_long_ones() -> None:
    assert "-q" not in {s.name for s in extract_cli_flags_from(ARG_UTILS)}


def test_flags_are_tagged_as_cli_flags() -> None:
    flags = extract_cli_flags_from(ARG_UTILS)
    assert flags and all(s.kind is SymbolKind.CLI_FLAG for s in flags)


def extract_cli_flags_from(source: str):  # type: ignore[no-untyped-def]
    import ast

    return extract_cli_flags(ast.parse(source), "vllm/engine/arg_utils.py")


# --------------------------------------------------------------------------- #
# config keys                                                                  #
# --------------------------------------------------------------------------- #
def _config_keys(source: str):  # type: ignore[no-untyped-def]
    import ast

    return extract_config_keys(ast.parse(source), "vllm/config/scheduler.py")


def test_annotated_public_fields_become_config_keys() -> None:
    names = {s.name for s in _config_keys(CONFIG_MODULE)}
    assert "max_num_seqs" in names
    assert "enable_chunked_prefill" in names


def test_an_undocumented_field_is_still_a_config_key() -> None:
    """The oracle answers whether a key exists, which is independent of whether
    anyone wrote prose about it."""
    assert "enable_chunked_prefill" in {s.name for s in _config_keys(CONFIG_MODULE)}


def test_private_fields_and_classes_are_excluded() -> None:
    names = {s.name for s in _config_keys(CONFIG_MODULE)}
    assert "_private" not in names
    assert "hidden" not in names


def test_an_unannotated_assignment_is_not_a_config_key() -> None:
    """Without an annotation it is a class constant, not a configuration field."""
    assert "untyped" not in {s.name for s in _config_keys(CONFIG_MODULE)}


def test_a_method_is_not_a_config_key() -> None:
    assert "method" not in {s.name for s in _config_keys(CONFIG_MODULE)}


def test_config_key_carries_its_declaration() -> None:
    key = next(s for s in _config_keys(CONFIG_MODULE) if s.name == "max_num_seqs")
    assert "int" in key.signature


# --------------------------------------------------------------------------- #
# API symbols                                                                  #
# --------------------------------------------------------------------------- #
API_MODULE = '''
class LLM:
    """Engine."""

    def generate(self): ...


class _Hidden: ...


def serve(host: str = "0.0.0.0") -> None: ...


async def shutdown() -> None: ...


def _helper() -> None: ...
'''


def _api(source: str):  # type: ignore[no-untyped-def]
    import ast

    return extract_api_symbols(ast.parse(source), "vllm/entrypoints/llm.py")


def test_public_classes_and_functions_are_extracted() -> None:
    by_kind = {(s.name, s.kind) for s in _api(API_MODULE)}
    assert ("LLM", SymbolKind.CLASS) in by_kind
    assert ("serve", SymbolKind.FUNCTION) in by_kind
    assert ("shutdown", SymbolKind.FUNCTION) in by_kind


def test_private_symbols_are_excluded() -> None:
    names = {s.name for s in _api(API_MODULE)}
    assert "_Hidden" not in names
    assert "_helper" not in names


def test_methods_are_not_flattened_into_the_top_level() -> None:
    """Flattening would let the oracle confirm a name that exists only on some
    unrelated object."""
    assert "generate" not in {s.name for s in _api(API_MODULE)}


# --------------------------------------------------------------------------- #
# whole-module extraction                                                      #
# --------------------------------------------------------------------------- #
def test_extract_symbols_combines_every_kind() -> None:
    kinds = {s.kind for s in extract_symbols(ARG_UTILS, "vllm/engine/arg_utils.py")}
    assert SymbolKind.CLI_FLAG in kinds
    assert SymbolKind.CLASS in kinds


def test_extract_symbols_survives_a_syntax_error() -> None:
    """One unparseable file must not fail an oracle build over thousands."""
    assert extract_symbols("def f(:\n", "vllm/broken.py") == []


# --------------------------------------------------------------------------- #
# derived flags                                                                #
# --------------------------------------------------------------------------- #
# vLLM generates part of its command line from dataclass fields in a loop, with
# no string literal anywhere. A literals-only oracle called --tool-call-parser,
# --chat-template and --enable-auto-tool-choice hallucinations. They are real.
DERIVED = '''
class BaseFrontendArgs:
    """Frontend args."""

    tool_call_parser: str | None = None
    chat_template: str | None = None
    _private: int = 0


class FrontendArgs(BaseFrontendArgs):
    """More frontend args."""

    enable_auto_tool_choice: bool = False

    @classmethod
    def add_cli_args(cls, parser):
        frontend_kwargs = get_kwargs(cls)
        group = parser.add_argument_group(title="Frontend")
        for key, value in frontend_kwargs.items():
            group.add_argument(*value.pop("flags", []), f"--{key.replace('_', '-')}", **value)
        return parser


class NotAFlagSource:
    """Fields here are config, not command line."""

    internal_knob: int = 3
'''


def _derived(source: str) -> set[str]:
    import ast

    from anchor.ground.oracle import class_annotations, derive_flags, find_flag_source_classes

    tree = ast.parse(source)
    return derive_flags(find_flag_source_classes(tree), class_annotations(tree))


def test_flags_are_derived_from_a_generating_class() -> None:
    assert "--enable-auto-tool-choice" in _derived(DERIVED)


def test_derived_flags_include_inherited_fields() -> None:
    """FrontendArgs extends BaseFrontendArgs, so it exposes the base's flags."""
    flags = _derived(DERIVED)
    assert "--tool-call-parser" in flags
    assert "--chat-template" in flags


def test_a_class_that_does_not_generate_flags_contributes_none() -> None:
    """Deriving from every config class would invent flags that do not exist,
    and a false 'supported' hides a real hallucination."""
    assert "--internal-knob" not in _derived(DERIVED)


def test_private_fields_do_not_become_flags() -> None:
    assert "--private" not in _derived(DERIVED)


def test_flag_name_conversion() -> None:
    from anchor.ground.oracle import flag_name

    assert flag_name("tool_call_parser") == "--tool-call-parser"
    assert flag_name("headless") == "--headless"


def test_a_literal_only_module_derives_nothing() -> None:
    """arg_utils declares its flags as literals; deriving there too would
    double-count and could invent flags for fields it deliberately omits."""
    assert _derived(ARG_UTILS) == set()
