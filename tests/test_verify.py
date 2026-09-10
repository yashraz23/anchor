"""Claim routing: what the oracle will and will not check.

The precision rules here matter more than the recall ones. A mention the oracle
should not have picked up produces a confident verdict about a name the writer
never claimed existed, and nothing downstream catches that.
"""

from __future__ import annotations

from anchor.config import GroundSettings, SymbolKind
from anchor.ground.verify import extract_mentions, find_config_keys, find_flags

CFG = GroundSettings()


def _flags(text: str) -> set[str]:
    return find_flags(text, CFG.cli_flag_pattern)


# --------------------------------------------------------------------------- #
# CLI flags                                                                    #
# --------------------------------------------------------------------------- #
def test_a_flag_is_found_in_prose() -> None:
    assert _flags("Pass --max-num-seqs to cap it.") == {"--max-num-seqs"}


def test_a_flag_with_a_value_keeps_only_the_flag() -> None:
    assert _flags("Use --gpu-memory-utilization=0.9 here.") == {"--gpu-memory-utilization"}


def test_trailing_punctuation_is_not_part_of_the_flag() -> None:
    """Without this the oracle looks up '--enforce-eager.' and reports a
    contradiction for a flag that plainly exists."""
    assert _flags("Set --enforce-eager.") == {"--enforce-eager"}
    assert _flags("Either --foo, or --bar;") == {"--foo", "--bar"}


def test_several_flags_in_one_claim() -> None:
    assert _flags("--max-num-seqs and --max-model-len") == {
        "--max-num-seqs",
        "--max-model-len",
    }


def test_prose_with_no_flags() -> None:
    assert _flags("vLLM allocates a KV cache at startup.") == set()


# --------------------------------------------------------------------------- #
# config keys                                                                  #
# --------------------------------------------------------------------------- #
def test_a_backticked_config_key_is_found() -> None:
    assert find_config_keys("Set `max_num_seqs` to 256.") == {"max_num_seqs"}


def test_bare_snake_case_in_prose_is_ignored() -> None:
    """Matching prose would produce confident verdicts about names the writer
    never claimed were symbols."""
    assert find_config_keys("The max_num_seqs setting caps it.") == set()


def test_an_expression_in_backticks_is_not_a_config_key() -> None:
    assert find_config_keys("Call `llm.generate(prompts)` to run it.") == set()
    assert find_config_keys("Read `output.outputs[0].text`.") == set()


def test_a_single_word_in_backticks_is_not_a_config_key() -> None:
    """A config key has an underscore; a bare word is usually a class or a
    filename and would collide with unrelated symbols."""
    assert find_config_keys("The `LLM` class.") == set()
    assert find_config_keys("See `vllm`.") == set()


def test_a_dotted_path_in_backticks_is_not_a_config_key() -> None:
    assert find_config_keys("In `vllm.config.scheduler`.") == set()


# --------------------------------------------------------------------------- #
# routing                                                                      #
# --------------------------------------------------------------------------- #
def test_only_flags_in_a_vllm_command_are_adjudicated() -> None:
    mentions = extract_mentions("vllm serve m --enforce-eager", CFG)
    assert {(m.name, m.kind) for m in mentions} == {("--enforce-eager", SymbolKind.CLI_FLAG)}


def test_a_flag_in_prose_is_left_to_the_judge() -> None:
    """Narrower than it could be, on purpose. Judging every flag in an answer
    produced a 42% "not found" rate on real answers, nearly all of it flags
    belonging to other tools that the model had used correctly."""
    assert extract_mentions("Pass --enforce-eager to disable compilation.", CFG) == []


def test_config_keys_are_not_adjudicated_by_the_oracle() -> None:
    """A backticked identifier is as likely to be a config value, a variable
    from the model's own example, or another library's parameter as it is a
    vLLM field, and the oracle cannot tell from the name alone."""
    assert extract_mentions("Set `max_num_seqs` to 256.", CFG) == []


def test_a_claim_with_nothing_checkable_yields_no_mentions() -> None:
    """It falls through to the judge rather than being counted as supported."""
    assert extract_mentions("vLLM batches requests continuously.", CFG) == []
