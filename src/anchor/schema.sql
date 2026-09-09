-- anchor schema.
--
-- Single source of truth for the data model. Applied two ways:
--   * docker-compose mounts this into the Postgres init directory
--   * `anchor db init` applies it to an already-running database
-- Both paths are idempotent, so re-running is safe.
--
-- NOTE ON DIMENSIONS: vector(384) matches index.embedding_dim for
-- BAAI/bge-small-en-v1.5. Changing the embedding model is a migration, not a
-- config flip, and the HNSW index has to be rebuilt with it.

CREATE EXTENSION IF NOT EXISTS vector;

-- --------------------------------------------------------------------------
-- Corpus
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    id           BIGSERIAL PRIMARY KEY,
    source_path  TEXT        NOT NULL,
    url          TEXT,
    title        TEXT,
    -- Version provenance. Every chunk inherits this through document_id, which
    -- is what makes staleness detection possible.
    vllm_version TEXT        NOT NULL,
    commit_sha   TEXT        NOT NULL,
    content_hash TEXT        NOT NULL,
    fetched_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source_path, commit_sha)
);

CREATE TABLE IF NOT EXISTS chunks (
    id           BIGSERIAL PRIMARY KEY,
    document_id  BIGINT      NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    -- Both chunking strategies live in this table at the same time so they can
    -- be compared on identical queries. Never filter a retrieval without it.
    strategy     TEXT        NOT NULL,
    ordinal      INT         NOT NULL,
    text         TEXT        NOT NULL,
    heading_path TEXT,
    is_code_block BOOLEAN    NOT NULL DEFAULT FALSE,
    token_count  INT         NOT NULL,
    embedding    vector(384),
    tsv          tsvector,
    UNIQUE (document_id, strategy, ordinal)
);

-- Dense. HNSW build params mirror index.hnsw_m / hnsw_ef_construction.
-- Cosine, because embeddings are L2-normalised at write time.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Sparse.
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);

-- Strategy-scoped lookups during evaluation.
CREATE INDEX IF NOT EXISTS chunks_strategy_document ON chunks (strategy, document_id);

-- --------------------------------------------------------------------------
-- The oracle
-- --------------------------------------------------------------------------
-- Populated by walking the vLLM AST and parsing `--help`. This is what makes a
-- subset of faithfulness checks ground-truthed instead of LLM-judged.
CREATE TABLE IF NOT EXISTS symbols (
    id           BIGSERIAL PRIMARY KEY,
    name         TEXT NOT NULL,
    kind         TEXT NOT NULL CHECK (kind IN ('cli_flag', 'class', 'function', 'config_key')),
    source_file  TEXT,
    signature    TEXT,
    vllm_version TEXT NOT NULL,
    UNIQUE (name, kind, vllm_version)
);

CREATE INDEX IF NOT EXISTS symbols_name_kind ON symbols (name, kind);

-- --------------------------------------------------------------------------
-- Evaluation
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS golden_queries (
    id                BIGSERIAL PRIMARY KEY,
    question          TEXT NOT NULL,
    source_url        TEXT,
    expected_answer   TEXT,
    expected_chunk_ids BIGINT[] NOT NULL DEFAULT '{}',
    category          TEXT,
    difficulty        TEXT,
    UNIQUE (question)
);

CREATE TABLE IF NOT EXISTS runs (
    id          BIGSERIAL PRIMARY KEY,
    config_json JSONB       NOT NULL,
    git_sha     TEXT        NOT NULL,
    notes       TEXT,
    started_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS answers (
    id            BIGSERIAL PRIMARY KEY,
    run_id        BIGINT NOT NULL REFERENCES runs (id) ON DELETE CASCADE,
    query_id      BIGINT NOT NULL REFERENCES golden_queries (id) ON DELETE CASCADE,
    answer_text   TEXT,
    retrieved_chunk_ids BIGINT[] NOT NULL DEFAULT '{}',
    latency_ms    INT,
    input_tokens  INT,
    output_tokens INT,
    -- NULL means the price was unknown, not that the call was free.
    cost_usd      NUMERIC(12, 6),
    abstained     BOOLEAN NOT NULL DEFAULT FALSE,
    UNIQUE (run_id, query_id)
);

CREATE TABLE IF NOT EXISTS claims (
    id                  BIGSERIAL PRIMARY KEY,
    answer_id           BIGINT NOT NULL REFERENCES answers (id) ON DELETE CASCADE,
    claim_text          TEXT   NOT NULL,
    supporting_chunk_id BIGINT REFERENCES chunks (id) ON DELETE SET NULL,
    support_score       REAL,
    verdict             TEXT CHECK (verdict IN ('supported', 'unsupported', 'contradicted', 'unverifiable')),
    verifier            TEXT CHECK (verifier IN ('symbol_oracle', 'span_attribution', 'llm_judge')),
    -- Phase 2 harvests (claim_text, supporting span, verdict) from here as
    -- distillation training data, so keep the rows.
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS claims_answer ON claims (answer_id);
CREATE INDEX IF NOT EXISTS claims_verifier_verdict ON claims (verifier, verdict);
