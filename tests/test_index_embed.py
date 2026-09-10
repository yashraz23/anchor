"""Embedding tests.

The query/passage asymmetry is the piece worth pinning: bge prefixes queries and
not passages, nothing errors when that is wrong, and recall just drops. The
prefix logic is pure, so it is tested without loading a model.

The model itself is exercised in the slow-marked test below, which is the only
place a real encoder is loaded.
"""

from __future__ import annotations

import pytest

from anchor.config import Settings
from anchor.index.embed import with_query_instruction

INSTRUCTION = "Represent this sentence for searching relevant passages: "


def test_query_gets_the_instruction_prefix() -> None:
    out = with_query_instruction("what is --max-num-seqs", INSTRUCTION)
    assert out == INSTRUCTION + "what is --max-num-seqs"


def test_an_empty_instruction_is_a_no_op() -> None:
    """Dropping the prefix has to be expressible, because it is a sweep knob."""
    assert with_query_instruction("query text", "") == "query text"


def test_the_configured_instruction_is_bge_s() -> None:
    """A wrong prefix silently costs recall, so the default is asserted."""
    assert Settings().index.query_instruction == INSTRUCTION


@pytest.mark.slow
def test_real_encoder_dimension_and_normalisation() -> None:
    """The real model, checked on the two properties the schema depends on.

    The column is vector(384) and the HNSW index uses cosine, which is only
    correct if vectors come out L2-normalised.
    """
    import numpy as np

    from anchor.index.embed import Embedder

    settings = Settings()
    embedder = Embedder(
        settings.index.embedding_model,
        normalize=settings.index.normalize_embeddings,
        query_instruction=settings.index.query_instruction,
    )
    assert embedder.dimension == settings.index.embedding_dim

    vectors = embedder.embed_passages(["vllm serve --max-num-seqs 256"], batch_size=8)
    assert vectors.shape == (1, settings.index.embedding_dim)
    assert np.linalg.norm(vectors[0]) == pytest.approx(1.0, abs=1e-4)

    query = embedder.embed_query("how do I cap concurrent sequences")
    assert query.shape == (settings.index.embedding_dim,)
