"""Config is a sweep surface, so its behaviour is tested like any other module.

The defaults asserted here are experiment variables. A test failing after a
default changes is the intended alarm, not noise: results already in the README
were produced under the old value.
"""

from __future__ import annotations

import pytest

from anchor.config import (
    MODEL_PRICING,
    ChunkStrategy,
    RetrievalMode,
    Settings,
    cost_usd,
)


def test_defaults_match_the_documented_experiment_baseline() -> None:
    s = Settings()
    assert s.chunk.strategy is ChunkStrategy.STRUCTURE_AWARE
    assert s.chunk.fixed_target_tokens == 512
    assert s.chunk.fixed_overlap_tokens == 64
    assert s.index.embedding_model == "BAAI/bge-small-en-v1.5"
    assert s.index.embedding_dim == 384
    assert s.retrieve.mode is RetrievalMode.HYBRID
    assert s.retrieve.rrf_k == 60
    assert s.retrieve.rerank_top_n == 5


def test_nested_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """A sweep sets knobs through the environment; nested delimiters must work."""
    monkeypatch.setenv("ANCHOR_RETRIEVE__DENSE_TOP_K", "17")
    monkeypatch.setenv("ANCHOR_CHUNK__STRATEGY", "fixed")
    s = Settings()
    assert s.retrieve.dense_top_k == 17
    assert s.chunk.strategy is ChunkStrategy.FIXED


def test_run_config_is_json_serialisable_and_excludes_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-appear")
    config = Settings().run_config()
    blob = json.dumps(config)
    assert "sk-ant-should-not-appear" not in blob
    assert set(config) == {
        "ingest",
        "chunk",
        "index",
        "retrieve",
        "generate",
        "ground",
        "evaluate",
    }


def test_cost_usd_known_model() -> None:
    # 1M in and 1M out on Opus 5 at $5 / $25.
    assert cost_usd("claude-opus-5", 1_000_000, 1_000_000) == pytest.approx(30.0)


def test_cost_usd_unknown_model_is_none_not_zero() -> None:
    """An unknown price must read as missing in the results table, never free."""
    assert cost_usd("some-local-model", 1000, 1000) is None


def test_every_priced_model_has_positive_rates() -> None:
    for name, price in MODEL_PRICING.items():
        assert price.input_per_mtok > 0, name
        assert price.output_per_mtok > 0, name
