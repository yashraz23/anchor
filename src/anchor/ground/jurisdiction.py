"""What the symbol oracle is entitled to rule on.

The oracle's value is that it is exact, so it must only judge names that are
actually claims about vLLM. Its first version judged every `--flag` and every
backticked identifier in an answer, and on 33 real answers 42% of those came
back "not found in source". Almost none were hallucinations. They were:

- flags belonging to other tools: `pip install --editable`, `numactl
  --cpunodebind`, which the model used correctly
- *values* rather than field names: `fp8_e5m2` is a value of `kv_cache_dtype`
- variables the model defined in its own illustrative code: `a_val`, `b_val`
- parameters of other libraries' APIs: `max_new_tokens` is HuggingFace's

An oracle that reports those as contradictions is worse than no oracle, because
its verdicts carry the authority of ground truth. So jurisdiction is narrow and
explicit: a flag counts as a claim about vLLM only when it appears in something
that invokes vLLM.

The cost of this choice is recall, and it is the right way round. A hallucinated
flag outside a command line is missed here and falls through to the LLM judge,
which is exactly the fallback the design already has. A false contradiction has
no such safety net.
"""

from __future__ import annotations

import re

# A line invoking vLLM: the executable, or a module invocation. Matched at the
# start of a line so prose merely mentioning the word does not qualify.
_VLLM_COMMAND = re.compile(
    r"^\s*(?:\$\s*)?(?:[A-Za-z0-9_]+=\S+\s+)*"
    r"(?:vllm\b|python\s+-m\s+vllm\b)",
    re.MULTILINE,
)

# A continuation line inside a shell command, i.e. the previous line ended in a
# backslash. vLLM invocations in the docs are almost always wrapped this way.
_CONTINUES = re.compile(r"\\\s*$")


def vllm_command_lines(text: str) -> list[str]:
    """Lines that form part of a vLLM invocation.

    Includes continuation lines, since a real serve command spans several and
    the flags of interest are usually on them rather than the first.
    """
    lines = text.splitlines()
    out: list[str] = []
    in_command = False

    for line in lines:
        starts = bool(_VLLM_COMMAND.match(line))
        if starts or in_command:
            out.append(line)
            in_command = bool(_CONTINUES.search(line))
        else:
            in_command = False
    return out


def flags_in_vllm_commands(text: str, pattern: str) -> set[str]:
    """CLI flags the answer attributes to vLLM.

    A flag on a `pip` or `numactl` line is not a claim about vLLM's interface,
    so the oracle has nothing to say about it.
    """
    flags: set[str] = set()
    for line in vllm_command_lines(text):
        for match in re.finditer(pattern, line):
            flags.add(match.group(0).split("=")[0].rstrip(".,;:\\"))
    return flags
