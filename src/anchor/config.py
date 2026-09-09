"""Central configuration for anchor.

Every tunable value in the system lives here. Nothing downstream is allowed to
inline a magic number, because every knob in this file is a potential sweep
dimension and every sweep dimension has to be recorded in the `runs` table
alongside its results.

Settings load from environment variables and `.env`, prefixed `ANCHOR_` and
nested with a double underscore::

    ANCHOR_RETRIEVE__DENSE_TOP_K=100
    ANCHOR_CHUNK__STRATEGY=fixed
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Enums shared across the pipeline                                             #
# --------------------------------------------------------------------------- #
class ChunkStrategy(StrEnum):
    FIXED = "fixed"
    STRUCTURE_AWARE = "structure_aware"


class RetrievalMode(StrEnum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class GeneratorBackend(StrEnum):
    ANTHROPIC = "anthropic"
    VLLM = "vllm"


class Verdict(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    UNVERIFIABLE = "unverifiable"


class Verifier(StrEnum):
    SYMBOL_ORACLE = "symbol_oracle"
    SPAN_ATTRIBUTION = "span_attribution"
    LLM_JUDGE = "llm_judge"


class SymbolKind(StrEnum):
    CLI_FLAG = "cli_flag"
    CLASS = "class"
    FUNCTION = "function"
    CONFIG_KEY = "config_key"


# --------------------------------------------------------------------------- #
# Model pricing, USD per 1M tokens. Used to fill answers.cost_usd.             #
# Source: Anthropic public pricing. Update deliberately -- these numbers end    #
# up in the README cost table and get defended in interviews.                   #
# --------------------------------------------------------------------------- #
class TokenPrice(BaseModel):
    input_per_mtok: float
    output_per_mtok: float


MODEL_PRICING: dict[str, TokenPrice] = {
    "claude-opus-5": TokenPrice(input_per_mtok=5.00, output_per_mtok=25.00),
    "claude-sonnet-5": TokenPrice(input_per_mtok=2.00, output_per_mtok=10.00),
    "claude-haiku-4-5": TokenPrice(input_per_mtok=1.00, output_per_mtok=5.00),
}


# --------------------------------------------------------------------------- #
# Section settings                                                             #
# --------------------------------------------------------------------------- #
class IngestSettings(BaseModel):
    """Corpus acquisition. Every document records the version it came from."""

    vllm_repo_url: str = "https://github.com/vllm-project/vllm.git"
    # Pinned at ingest time and written to documents.commit_sha. Empty means
    # "resolve the tip of vllm_branch and pin to whatever that is right now".
    vllm_commit: str = ""
    vllm_branch: str = "main"
    checkout_dir: Path = REPO_ROOT / "data" / "vllm"

    # Globs relative to the vLLM checkout root.
    doc_globs: tuple[str, ...] = ("docs/**/*.md", "docs/**/*.rst", "README.md")
    include_docstrings: bool = True
    docstring_globs: tuple[str, ...] = ("vllm/**/*.py",)

    # Docs shorter than this after stripping front matter are dropped as stubs.
    min_document_chars: int = 200


class ChunkSettings(BaseModel):
    """Both strategies are persisted so they compare on identical queries."""

    strategy: ChunkStrategy = ChunkStrategy.STRUCTURE_AWARE

    # Token counts use the embedding model's own tokenizer, not a generic one,
    # because the real constraint is that encoder's 512-token input limit.
    tokenizer_model: str = "BAAI/bge-small-en-v1.5"

    # `fixed`: the naive baseline.
    fixed_target_tokens: int = 512
    fixed_overlap_tokens: int = 64

    # `structure_aware`: split on heading boundaries.
    max_tokens: int = 512
    min_tokens: int = 64
    # A fenced code block is never split, even when it exceeds max_tokens. This
    # is the point of the second strategy: users ask about code blocks.
    never_split_code_blocks: bool = True
    # Headings are prepended so a chunk carries its own context.
    prepend_heading_path: bool = True


class IndexSettings(BaseModel):
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    embedding_batch_size: int = 64
    normalize_embeddings: bool = True
    # bge wants an instruction prefix on the query side only. Dropping it costs
    # measurable recall, so it is a knob rather than a constant.
    query_instruction: str = "Represent this sentence for searching relevant passages: "

    # pgvector HNSW. Build params are per index; ef_search is per session.
    hnsw_m: int = 16
    hnsw_ef_construction: int = 64
    hnsw_ef_search: int = 100

    # Postgres full-text configuration used to build chunks.tsv.
    fts_config: str = "english"


class RetrieveSettings(BaseModel):
    mode: RetrievalMode = RetrievalMode.HYBRID

    dense_top_k: int = 50
    sparse_top_k: int = 50

    # Reciprocal Rank Fusion: score = sum over lists of 1 / (rrf_k + rank).
    # 60 is the value from the original RRF paper. It damps the influence of the
    # top ranks so that one list cannot dominate the fusion.
    rrf_k: int = 60
    fused_top_k: int = 20

    use_rerank: bool = True
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_batch_size: int = 32
    # Final context size handed to the generator.
    rerank_top_n: int = 5


class GenerateSettings(BaseModel):
    backend: GeneratorBackend = GeneratorBackend.ANTHROPIC

    # Anthropic path.
    anthropic_model: str = "claude-opus-5"
    max_tokens: int = 4096
    # Opus 5 rejects temperature and top_p outright, so generation determinism
    # comes from a fixed prompt and a fixed effort level, not from sampling.
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    thinking_display: Literal["omitted", "summarized"] = "omitted"

    # Local vLLM path (phase 3). OpenAI-compatible server.
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model: str = "Qwen2.5-7B-Instruct"
    vllm_temperature: float = 0.0
    vllm_seed: int = 1337

    # Prompt assembly. Spans are numbered and the model must cite the numbers.
    cite_spans_inline: bool = True
    max_context_tokens: int = 8000


class GroundSettings(BaseModel):
    """The grounding verification layer, the distinguishing feature."""

    claim_model: str = "claude-opus-5"
    max_claims_per_answer: int = 20

    # Span attribution: similarity between a claim and its best supporting span.
    # A claim below this is not considered attributable to the context.
    support_threshold: float = 0.55

    # Abstention policy. When the fraction of claims clearing support_threshold
    # falls below this, the answer is withheld. This is THE sweep dimension:
    # sweeping it produces the strictness-versus-completeness tradeoff curve.
    abstain_threshold: float = 0.80
    abstain_message: str = "I could not ground this answer in the retrieved vLLM documentation."

    # The symbol oracle. First item on the cut list if the schedule slips.
    use_symbol_oracle: bool = True
    # Patterns that route a claim to the oracle instead of the LLM judge.
    cli_flag_pattern: str = r"--[a-z0-9][a-z0-9-]*"
    config_key_pattern: str = r"\b[A-Z][A-Z0-9_]{3,}\b"

    use_llm_judge: bool = True
    judge_model: str = "claude-opus-5"


class EvaluateSettings(BaseModel):
    golden_path: Path = REPO_ROOT / "data" / "golden" / "golden.yaml"
    # recall@k is reported at each of these.
    recall_at: tuple[int, ...] = (1, 3, 5, 10, 20)
    seed: int = 1337

    run_ragas: bool = True
    ragas_model: str = "claude-opus-5"

    wandb_project: str = "anchor"
    wandb_entity: str = ""
    # Both trackers are optional. The harness no-ops when keys are absent so
    # that CI and offline runs work unchanged.
    use_wandb: bool = True
    use_langfuse: bool = True


class ApiSettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    request_timeout_s: float = 120.0


# --------------------------------------------------------------------------- #
# Root                                                                         #
# --------------------------------------------------------------------------- #
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ANCHOR_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )

    pg_dsn: str = "postgresql://anchor:anchor@localhost:5433/anchor"

    anthropic_api_key: SecretStr | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    wandb_api_key: SecretStr | None = Field(default=None, alias="WANDB_API_KEY")
    langfuse_public_key: SecretStr | None = Field(default=None, alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: SecretStr | None = Field(default=None, alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(default="https://cloud.langfuse.com", alias="LANGFUSE_HOST")

    model_cache_dir: Path = REPO_ROOT / "data" / "cache"

    ingest: IngestSettings = IngestSettings()
    chunk: ChunkSettings = ChunkSettings()
    index: IndexSettings = IndexSettings()
    retrieve: RetrieveSettings = RetrieveSettings()
    generate: GenerateSettings = GenerateSettings()
    ground: GroundSettings = GroundSettings()
    evaluate: EvaluateSettings = EvaluateSettings()
    api: ApiSettings = ApiSettings()

    def run_config(self) -> dict[str, object]:
        """The exact dict written to runs.config_json.

        Secrets are excluded. A result must be reproducible from this alone, so
        anything that can change an answer belongs in one of the sections above.
        """
        return {
            "ingest": self.ingest.model_dump(mode="json"),
            "chunk": self.chunk.model_dump(mode="json"),
            "index": self.index.model_dump(mode="json"),
            "retrieve": self.retrieve.model_dump(mode="json"),
            "generate": self.generate.model_dump(mode="json"),
            "ground": self.ground.model_dump(mode="json"),
            "evaluate": self.evaluate.model_dump(mode="json"),
        }


def get_settings() -> Settings:
    """A fresh Settings. Not cached, because sweeps mutate the environment."""
    return Settings()


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Cost of one call, or None when the model has no price on file.

    Returning None rather than 0.0 is deliberate. An unknown price has to show
    up as missing in the results table, never as free.
    """
    price = MODEL_PRICING.get(model)
    if price is None:
        return None
    return (
        input_tokens / 1_000_000 * price.input_per_mtok
        + output_tokens / 1_000_000 * price.output_per_mtok
    )
