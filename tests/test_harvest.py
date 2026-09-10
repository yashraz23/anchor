"""Harvesting tests.

Fixtures are real vLLM issue shapes, copied from issues 54060 and 54947. No
network: every function under test is pure, and fetching is the only part that
shells out.
"""

from __future__ import annotations

import json
from pathlib import Path

from anchor.evaluate.harvest import (
    Candidate,
    extract_question,
    parse_issue,
    read_candidates,
    strip_title_prefix,
    write_candidates,
)

TEMPLATED_BODY = """### Your current environment

```text
vLLM from 0.19.1 to 0.25
PyTorch 2.4.0
CUDA 12.4
```


### How would you like to use vllm

Why did the input tokens double for the same request when running
Qwen3-Omni-30B-A3B-Instruct after upgrading vLLM from 0.19.1 to 0.25?
we use use_audio_in_video

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues.
"""

FREEFORM_BODY = """I deployed a model inference service using vLLM and sent
requests to it. When I closed the client connection mid-request, the server
continued to generate tokens.
"""


def test_strip_title_prefix() -> None:
    assert strip_title_prefix("[Usage]: How do I set max_num_seqs") == ("How do I set max_num_seqs")
    assert strip_title_prefix("[Doc] Missing page") == "Missing page"
    assert strip_title_prefix("No prefix here") == "No prefix here"


def test_extract_question_prefers_the_template_section() -> None:
    out = extract_question(TEMPLATED_BODY)
    assert out.startswith("Why did the input tokens double")
    assert "use_audio_in_video" in out


def test_extract_question_drops_the_environment_dump() -> None:
    """The dump is hundreds of tokens of versions that would dominate any
    embedding of the question itself."""
    out = extract_question(TEMPLATED_BODY)
    assert "PyTorch 2.4.0" not in out
    assert "CUDA 12.4" not in out


def test_extract_question_drops_the_checkbox_boilerplate() -> None:
    out = extract_question(TEMPLATED_BODY)
    assert "Before submitting" not in out
    assert "searched for relevant issues" not in out


def test_extract_question_falls_back_to_the_whole_body() -> None:
    """A plain body is still a real question. Dropping it would bias the set
    toward people who filled in the template."""
    out = extract_question(FREEFORM_BODY)
    assert "closed the client connection" in out


def test_extract_question_handles_crlf() -> None:
    out = extract_question(TEMPLATED_BODY.replace("\n", "\r\n"))
    assert out.startswith("Why did the input tokens double")


def test_extract_question_on_empty_body() -> None:
    assert extract_question("") == ""


def test_parse_issue() -> None:
    payload = {
        "number": 54060,
        "html_url": "https://github.com/vllm-project/vllm/issues/54060",
        "title": "[Usage]: Qwen3-Omni input tokens doubled",
        "body": TEMPLATED_BODY,
        "labels": [{"name": "usage"}, {"name": "stale"}],
        "created_at": "2026-01-02T03:04:05Z",
        "state": "open",
    }
    candidate = parse_issue(payload)
    assert candidate.number == 54060
    assert candidate.title == "Qwen3-Omni input tokens doubled"
    assert candidate.raw_title.startswith("[Usage]")
    assert candidate.labels == ("usage", "stale")
    assert "input tokens double" in candidate.question


def test_parse_issue_tolerates_a_null_body() -> None:
    """GitHub returns null, not an empty string, for an issue with no body."""
    candidate = parse_issue(
        {"number": 1, "html_url": "u", "title": "[Usage]: t", "body": None, "labels": []}
    )
    assert candidate.question == ""


def test_candidates_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "candidates.jsonl"
    original = [
        Candidate(
            number=1,
            url="https://example.invalid/1",
            title="A question",
            raw_title="[Usage]: A question",
            question="How do I do the thing?",
            labels=("usage",),
            created_at="2026-01-01T00:00:00Z",
            state="open",
        )
    ]
    assert write_candidates(path, original) == 1
    assert read_candidates(path) == original


def test_read_candidates_on_a_missing_file(tmp_path: Path) -> None:
    assert read_candidates(tmp_path / "nope.jsonl") == []


def test_written_candidates_are_valid_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    write_candidates(
        path,
        [
            Candidate(
                number=2,
                url="u",
                title="unicode: 🚀 café",
                raw_title="[Usage]: unicode: 🚀 café",
                question="q",
                labels=(),
                created_at="",
                state="open",
            )
        ],
    )
    line = path.read_text(encoding="utf-8").strip()
    assert json.loads(line)["title"] == "unicode: 🚀 café"
