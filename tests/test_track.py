"""Tests for the two trackers.

Both are observers, so the property that matters most is not what they record
but what they refuse to break. A tracker that raises inside an evaluation would
destroy a run that cost real money, and these tests exist mainly to hold that
line.

The Langfuse tests drive a stub client rather than a server. That is a real
limit and worth naming: they prove the payload this project builds is the one
it means to build, not that a Langfuse instance accepted it.
"""

from __future__ import annotations

from typing import Any

import pytest

from anchor.config import Settings
from anchor.track import experiment, log_metrics, record_generation, tracing


@pytest.fixture(autouse=True)
def _reset_tracer() -> Any:
    """The Langfuse client is a process-wide singleton, resolved once.

    Without this every test after the first would reuse whatever the first one
    resolved, and the ordering would decide the results.
    """
    tracing._client = None
    tracing._resolved = False
    yield
    tracing._client = None
    tracing._resolved = False


class _StubObservation:
    def __init__(self) -> None:
        self.ended = False

    def end(self) -> None:
        self.ended = True


class _StubClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.flushed = 0

    def start_observation(self, **kwargs: Any) -> _StubObservation:
        self.calls.append(kwargs)
        return _StubObservation()

    def flush(self) -> None:
        self.flushed += 1


def _settings(**env: str) -> Settings:
    # _env_file=None keeps the developer's own .env out of the test: this suite
    # must behave the same on a machine that happens to have real credentials.
    return Settings(_env_file=None, **env)  # type: ignore[call-arg]


# --------------------------------------------------------------------------- #
# Credentials                                                                  #
# --------------------------------------------------------------------------- #
def test_blank_secret_is_treated_as_absent() -> None:
    """`KEY=` in a .env means unset, which is how compose passes a missing var."""
    settings = _settings(WANDB_API_KEY="", LANGFUSE_PUBLIC_KEY="   ")
    assert settings.wandb_api_key is None
    assert settings.langfuse_public_key is None


def test_present_secret_survives() -> None:
    settings = _settings(WANDB_API_KEY="abc123")
    assert settings.wandb_api_key is not None
    assert settings.wandb_api_key.get_secret_value() == "abc123"


def test_run_config_excludes_telemetry() -> None:
    """Telemetry cannot change an answer, so it must not enter the run config.

    Two runs differing only in whether a tracker watched are the same
    experiment. Recording them as different configurations would defeat the one
    job `runs.config_json` has.
    """
    assert "track" not in _settings().run_config()


# --------------------------------------------------------------------------- #
# Langfuse                                                                     #
# --------------------------------------------------------------------------- #
def test_no_keys_means_no_tracing() -> None:
    """No offline mode exists for Langfuse, so it disables rather than pretend."""
    assert tracing._tracer(_settings()) is None


def test_record_generation_passes_usage_and_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubClient()
    monkeypatch.setattr(tracing, "_tracer", lambda _s: stub)

    record_generation(
        _settings(),
        name="answer",
        model="claude-opus-5",
        prompt="p",
        output="o",
        input_tokens=100,
        output_tokens=20,
        latency_ms=1234,
        cost_usd=0.5,
    )

    (call,) = stub.calls
    assert call["as_type"] == "generation"
    assert call["model"] == "claude-opus-5"
    assert call["usage_details"] == {"input": 100, "output": 20, "total": 120}
    assert call["cost_details"] == {"total": 0.5}
    assert call["metadata"]["latency_ms"] == 1234


def test_unpriced_model_sends_no_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown price is missing, never free.

    Sending 0.0 here would put a self-hosted vLLM run in the dashboard as
    costing nothing, which reads as measured rather than absent.
    """
    stub = _StubClient()
    monkeypatch.setattr(tracing, "_tracer", lambda _s: stub)

    record_generation(
        _settings(),
        name="answer",
        model="some-local-model",
        prompt="p",
        output="o",
        input_tokens=1,
        output_tokens=1,
        latency_ms=1,
        cost_usd=None,
    )

    assert stub.calls[0]["cost_details"] is None


def test_tracing_failure_never_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole point: a broken tracer must not kill a paid evaluation."""

    class _Exploding:
        def start_observation(self, **_kwargs: Any) -> None:
            raise RuntimeError("langfuse is down")

    monkeypatch.setattr(tracing, "_tracer", lambda _s: _Exploding())

    record_generation(
        _settings(),
        name="answer",
        model="claude-opus-5",
        prompt="p",
        output="o",
        input_tokens=1,
        output_tokens=1,
        latency_ms=1,
    )


def test_disabled_flag_wins_over_present_keys() -> None:
    settings = _settings(LANGFUSE_PUBLIC_KEY="pk", LANGFUSE_SECRET_KEY="sk")
    settings.track.use_langfuse = False
    assert tracing._tracer(settings) is None


# --------------------------------------------------------------------------- #
# Weights & Biases                                                             #
# --------------------------------------------------------------------------- #
def test_experiment_is_a_noop_when_disabled() -> None:
    settings = _settings()
    settings.track.use_wandb = False
    with experiment(settings, "test-job") as run:
        assert run.disabled
        # Logging through the null run must be safe, so call sites never have
        # to branch on whether tracking is on.
        log_metrics(run, {"anything": 1.0})


def test_experiment_survives_a_broken_wandb(monkeypatch: pytest.MonkeyPatch) -> None:
    """An init failure yields the null run instead of raising."""
    import wandb

    def _explode(**_kwargs: Any) -> None:
        raise RuntimeError("wandb is down")

    monkeypatch.setattr(wandb, "init", _explode)
    with experiment(_settings(), "test-job") as run:
        assert getattr(run, "disabled", False)


def test_log_metrics_drops_missing_values() -> None:
    """None means could-not-compute, and must not reach the dashboard as zero."""
    sent: dict[str, Any] = {}

    class _Summary:
        def update(self, values: dict[str, Any]) -> None:
            sent.update(values)

    class _Run:
        disabled = False
        summary = _Summary()

    log_metrics(_Run(), {"present": 1.0, "missing": None})
    assert sent == {"present": 1.0}
