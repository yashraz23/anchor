"""What the oracle is entitled to rule on.

Every fixture is a shape taken from a real generated answer. The oracle's
verdicts carry the authority of ground truth, so a false contradiction is the
expensive error and these tests are all about precision.
"""

from __future__ import annotations

from anchor.config import GroundSettings
from anchor.ground.jurisdiction import flags_in_vllm_commands, vllm_command_lines

PATTERN = GroundSettings().cli_flag_pattern


def _flags(text: str) -> set[str]:
    return flags_in_vllm_commands(text, PATTERN)


def test_a_flag_on_a_vllm_command_is_in_jurisdiction() -> None:
    assert _flags("vllm serve my-model --max-num-seqs 256") == {"--max-num-seqs"}


def test_a_pip_flag_is_not_a_claim_about_vllm() -> None:
    """The model used --editable correctly, for pip. Reporting it as a
    non-existent vLLM flag would be a false contradiction."""
    assert _flags("pip install --editable .") == set()


def test_a_numactl_flag_is_not_a_claim_about_vllm() -> None:
    assert _flags("numactl --cpunodebind=0 --membind=0 python x.py") == set()


def test_continuation_lines_are_part_of_the_command() -> None:
    """Real serve commands wrap, and the interesting flags are rarely on the
    first line."""
    text = (
        "vllm serve meta-llama/Llama-3.1-8B \\n"
        "  --tool-call-parser hermes \\n"
        "  --max-model-len 4096\n"
    )
    assert _flags(text) == {"--tool-call-parser", "--max-model-len"}


def test_the_command_ends_when_continuation_stops() -> None:
    text = "vllm serve m --max-num-seqs 8\n\npip install --editable .\n"
    assert _flags(text) == {"--max-num-seqs"}


def test_a_shell_prompt_prefix_is_tolerated() -> None:
    assert _flags("$ vllm serve m --enforce-eager") == {"--enforce-eager"}


def test_an_env_prefix_is_tolerated() -> None:
    assert _flags("VLLM_LOGGING_LEVEL=DEBUG vllm serve m --enforce-eager") == {"--enforce-eager"}


def test_python_module_invocation_counts() -> None:
    assert _flags("python -m vllm.entrypoints.openai.api_server --port 8000") == {"--port"}


def test_prose_mentioning_vllm_is_not_a_command() -> None:
    """Otherwise every answer qualifies, since they are all about vLLM."""
    assert _flags("In vLLM you can pass --made-up-flag to do it.") == set()


def test_a_trailing_backslash_is_not_part_of_the_flag() -> None:
    assert _flags("vllm serve m --enforce-eager \\n  --port 8000") == {
        "--enforce-eager",
        "--port",
    }


def test_a_value_attached_with_equals_is_stripped() -> None:
    assert _flags("vllm serve m --gpu-memory-utilization=0.9") == {"--gpu-memory-utilization"}


def test_command_lines_are_reported() -> None:
    assert vllm_command_lines("echo hi\nvllm serve m\n") == ["vllm serve m"]
