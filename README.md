# anchor

Retrieval-augmented generation over vLLM and LLM-serving documentation, with a
**grounding verification layer**: every claim in a generated answer is traced to
a retrieved source span, and claims of a verifiable class (CLI flags, API
symbols, config keys) are checked against the **actual vLLM source code** rather
than against the retrieved text alone.

The deliverable is not the chatbot. It is a measured tradeoff curve between
grounding strictness and answer completeness.

**Why this corpus.** Technical documentation gives an objective oracle.
`--max-num-seqs` either exists in vLLM's CLI or it does not. That makes a subset
of the faithfulness checks ground-truthed instead of LLM-judged, which is
unusual for a RAG evaluation project. A second property falls out of the same
choice: vLLM releases fast and its docs go stale, so chunks are version-tagged
and the system can flag retrieved guidance that describes outdated behaviour.

> **Status: week 1, in progress.** Every results table below is empty on
> purpose. Numbers appear here only once the harness has produced them.

---

## Results

### Retrieval: recall@k by chunking strategy and retrieval mode

Both chunking strategies are indexed at the same time and queried identically,
so the comparison is like-for-like.

| Chunking | Retrieval | recall@1 | recall@3 | recall@5 | recall@10 | recall@20 |
|---|---|---|---|---|---|---|
| fixed | dense | TODO | TODO | TODO | TODO | TODO |
| fixed | sparse | TODO | TODO | TODO | TODO | TODO |
| fixed | hybrid (RRF) | TODO | TODO | TODO | TODO | TODO |
| fixed | hybrid + rerank | TODO | TODO | TODO | TODO | TODO |
| structure_aware | dense | TODO | TODO | TODO | TODO | TODO |
| structure_aware | sparse | TODO | TODO | TODO | TODO | TODO |
| structure_aware | hybrid (RRF) | TODO | TODO | TODO | TODO | TODO |
| structure_aware | hybrid + rerank | TODO | TODO | TODO | TODO | TODO |

### Grounding: strictness versus completeness

The headline chart. Sweeping the abstention threshold trades hallucination rate
against how many questions get answered at all.

| Abstention threshold | Answered | Abstained | Unsupported claims | Contradicted claims | Cost / query |
|---|---|---|---|---|---|
| TODO | TODO | TODO | TODO | TODO | TODO |

### Verifier agreement

How often the symbol oracle and the LLM judge reach the same verdict on the
claims both can see. Where they disagree, the oracle is right by construction,
which puts a number on the LLM judge's error rate.

| Verifier | Claims judged | Agreement with oracle | Notes |
|---|---|---|---|
| TODO | TODO | TODO | TODO |

---

## How it works

1. **Ingest.** Clone vLLM at a pinned commit. Parse the docs sources, the
   README, and public docstrings. Record version and commit on every document.
2. **Chunk.** Two strategies, both persisted:
   - `fixed`, roughly 512 tokens with 64 overlap, the naive baseline
   - `structure_aware`, split on heading boundaries, never splitting a fenced
     code block
3. **Index.** Embed with `bge-small-en-v1.5`, write to pgvector with an HNSW
   index, and build the `tsvector` in the same row.
4. **Retrieve.** Dense top-k and sparse top-k, fused with Reciprocal Rank
   Fusion, then reranked with a `bge-reranker-base` cross-encoder.
5. **Generate.** Assemble a prompt of numbered context spans and require inline
   citation of the span IDs.
6. **Ground.** Extract atomic claims, attribute each to a span, route verifiable
   claims to the symbol oracle and the rest to span attribution plus an LLM
   judge, then apply the abstention policy.
7. **Evaluate.** Run the golden set across configs, log to Weights & Biases, and
   write the tables above.

Token counts use the embedding model's own tokenizer rather than a generic one,
because the binding constraint is that encoder's 512-token input limit.

### Design decisions worth asking about

- **One datastore for dense and sparse.** Postgres with pgvector holds the
  embeddings and the full-text index in the same row, so hybrid retrieval is a
  single SQL statement and there is no second service to keep in sync.
- **RRF over weighted score fusion.** Rank fusion needs no score normalisation
  between a cosine similarity and a BM25 score, and no tuned weights to justify.
- **Cosine distance.** Embeddings are L2-normalised at write time.
- **`cost_usd` and `latency_ms` recorded from the first run.** Per-request cost
  is a question that gets asked, and the answer should be measured.
- **An unknown price stores as NULL, never 0.** Missing has to look missing.

---

## Quickstart

Requires Docker and [uv](https://docs.astral.sh/uv/).

```bash
cp .env.example .env      # then fill in ANTHROPIC_API_KEY
uv sync --all-extras
docker compose up -d db   # Postgres 16 + pgvector on host port 5433
uv run anchor db init     # idempotent
uv run anchor db check    # row counts per table
uv run anchor config      # the exact JSON written to runs.config_json
```

Run the checks the way CI does:

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy src tests
uv run pytest
```

## Configuration

Every tunable value lives in [`src/anchor/config.py`](src/anchor/config.py).
Nothing downstream inlines a magic number, because each knob is a potential
sweep dimension and each sweep dimension has to be recorded alongside its
results. Override through the environment with an `ANCHOR_` prefix and a double
underscore for nesting:

```bash
ANCHOR_RETRIEVE__DENSE_TOP_K=100 ANCHOR_CHUNK__STRATEGY=fixed uv run anchor config
```

Every eval run writes a row to `runs` holding that config and the git SHA, so a
table in this README can be traced back to the commit that produced it.

## Layout

| Path | What lives there |
|---|---|
| `src/anchor/config.py` | Every knob in the system |
| `src/anchor/schema.sql` | The data model, applied by compose and by the CLI |
| `src/anchor/ingest/` | Fetch vLLM docs, parse, version-tag |
| `src/anchor/chunk/` | The two chunking strategies |
| `src/anchor/index/` | Embedding and writes to pgvector |
| `src/anchor/retrieve/` | Dense, sparse, hybrid fusion, reranking |
| `src/anchor/generate/` | Prompt assembly, Claude and vLLM backends |
| `src/anchor/ground/` | Claim extraction, span attribution, symbol oracle |
| `src/anchor/evaluate/` | Golden set, ragas, judge, sweep runner |
| `data/golden/` | The golden set, version-controlled |

## Roadmap

- **Week 1.** Ingestion with version tagging, both chunking strategies, hybrid
  retrieval and reranking. Exit criterion: the recall@k table above is filled.
- **Week 2.** A golden set of 60 to 80 questions sourced from real vLLM GitHub
  issues and discussions, claim extraction, span attribution, the symbol oracle,
  the ragas triad, and evals in CI. Exit criterion: the tradeoff curve.
- **Week 3.** FastAPI, Docker, a Gradio demo on Hugging Face Spaces, and a
  documentation pull request against vLLM using gaps the harness exposed.

Phase 2 distils the LLM judge into a LoRA-tuned local model and reports
agreement against cost. Phase 3 serves the generator on k3s with a quantized
model and reports sustained throughput and P99 latency.
