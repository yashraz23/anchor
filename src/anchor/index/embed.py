"""Dense embedding.

bge is an asymmetric model: queries get an instruction prefix and passages do
not. Getting that backwards, or applying the prefix to both sides, costs
measurable recall, so the two paths are separate methods rather than one method
with a flag that is easy to pass wrong.

Vectors are L2-normalised at write time, which is what lets the database use
cosine distance against an HNSW index built with `vector_cosine_ops`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - import cost is why this is guarded
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def with_query_instruction(text: str, instruction: str) -> str:
    """Prefix a query with bge's retrieval instruction.

    Pure and separately tested, because this asymmetry is the single easiest
    thing to get silently wrong in a bge pipeline: nothing errors, recall just
    drops.
    """
    return f"{instruction}{text}" if instruction else text


class Embedder:
    """A `bge`-style bi-encoder over passages and queries."""

    def __init__(
        self,
        model_name: str,
        *,
        normalize: bool = True,
        query_instruction: str = "",
        device: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.normalize = normalize
        self.query_instruction = query_instruction
        self._model = SentenceTransformer(model_name, device=device)

    @property
    def dimension(self) -> int:
        # sentence-transformers 6 renamed this; support both so the pinned
        # range in pyproject stays honest rather than silently 6-only.
        getter = getattr(self._model, "get_embedding_dimension", None) or (
            self._model.get_sentence_embedding_dimension
        )
        dim = getter()
        if dim is None:  # pragma: no cover - every model in the stack reports one
            raise RuntimeError(f"{self.model_name} does not report an output dimension")
        return int(dim)

    def embed_passages(self, texts: list[str], batch_size: int) -> NDArray[np.float32]:
        """Embed chunk text. No instruction prefix: these are the passage side."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        """Embed one query, with the instruction prefix applied."""
        prepared = with_query_instruction(text, self.query_instruction)
        vector = self._model.encode(
            [prepared],
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        result: NDArray[np.float32] = np.asarray(vector, dtype=np.float32)[0]
        return result
