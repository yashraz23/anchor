# CLAUDE.md — `anchor`

> Dropped at the repo root. Claude Code reads it automatically as project context.

## What this project is

`anchor` is a retrieval-augmented generation system over **vLLM / LLM-serving documentation** whose distinguishing feature is a **grounding verification layer**: every claim in a generated answer is traced to a retrieved source span, and claims of a verifiable class (CLI flags, API symbols, config keys) are checked against the **actual vLLM source code**, not just the retrieved text.

The headline deliverable is not the chatbot. It is a measured **tradeoff curve** between grounding strictness and answer completeness.

**Why the corpus matters:** technical documentation provides an objective oracle. `--max-num-seqs` either exists in vLLM's CLI or it does not. This makes a subset of faithfulness checks ground-truthed rather than LLM-judged, which is rare for a RAG evaluation project.

**Secondary property:** vLLM releases fast and its docs go stale. Chunks are version-tagged so the system can detect when retrieved guidance describes outdated behavior.

---

## Tech stack (and why each was chosen)

| Layer | Choice | Rationale |
|---|---|---|
| Language | Python 3.11 | Baseline |
| Env / deps | `uv` | Fast, lockfile-based, modern default |
| Datastore | Postgres 16 + **pgvector** | One datastore for dense *and* sparse; extends existing Postgres experience |
| Sparse retrieval | Postgres FTS (`tsvector`) | Keeps hybrid search in a single SQL query — no second service |
| Dense embeddings | `BAAI/bge-small-en-v1.5` | Runs on CPU, 384-dim, strong quality/cost ratio |
| Fusion | Reciprocal Rank Fusion (RRF) | Standard, explainable, no tuning weights to justify |
| Reranking | `BAAI/bge-reranker-base` (cross-encoder) | Measurable precision lift over bi-encoder alone |
| Generation | Anthropic Claude API | Primary path |
| Local serving | **vLLM** (`Qwen2.5-7B-Instruct` or similar) | Optional generator backend — makes vLLM a stack item you *used*, not just profiled |
| Eval | `ragas` + custom harness | Faithfulness / answer relevance / context relevance |
| Experiment tracking | **Weights & Biases** (free tier) | Config sweeps produce dozens of runs; W&B is the industry default |
| LLM observability | **Langfuse** | Already know it from Spektra — reuse, don't relearn |
| API | FastAPI + Pydantic v2 | Standard |
| Demo | Gradio on Hugging Face Spaces | Free hosting, simpler than Streamlit for Spaces |
| Container | Docker + docker-compose | Postgres + app in one `up` |
| CI | GitHub Actions | Evals must run in CI |
| Test / lint | pytest, ruff, mypy | Non-negotiable — see conventions |
| Fine-tuning *(phase 2)* | PEFT / **LoRA** on `Qwen2.5-1.5B-Instruct` | Distil the LLM judge into a local model — a measurable cost result, not a technique demo |
| Orchestration *(phase 3)* | **k3s** on local RTX 5070 Ti + HPA | Real GPU serving with load numbers; free, because it's your own hardware |

**Explicitly out of scope:** multi-tenancy, auth, streaming UI polish, managed cloud K8s (costs money for no extra signal over local k3s).

---

## Repo structure

```
anchor/
├── CLAUDE.md
├── README.md              # results tables + tradeoff chart — the hiring artifact
├── pyproject.toml
├── docker-compose.yml     # postgres+pgvector, app
├── Dockerfile
├── .github/workflows/evals.yml
├── src/anchor/
│   ├── config.py          # pydantic-settings, all knobs in one place
│   ├── ingest/            # fetch vLLM docs, parse, version-tag
│   ├── chunk/             # strategies: fixed, structure_aware
│   ├── index/             # embed + write to pgvector, build tsvector
│   ├── retrieve/          # dense, sparse, hybrid (RRF), rerank
│   ├── generate/          # prompt assembly, Claude / vLLM backends
│   ├── ground/            # claim extraction, span attribution, symbol oracle
│   ├── evaluate/          # golden set, ragas, judge, sweep runner
│   ├── api/               # FastAPI
│   └── demo/              # Gradio
├── data/golden/           # golden set YAML/JSONL, version-controlled
├── scripts/               # one-off CLIs (ingest, sweep, report)
└── tests/
```

---

## Data model

```
documents(id, source_path, url, title, vllm_version, commit_sha, fetched_at)

chunks(id, document_id, strategy, ordinal, text, heading_path,
       is_code_block, token_count, embedding vector(384), tsv tsvector)
  -- indexes: HNSW on embedding, GIN on tsv, btree on (strategy, document_id)

symbols(id, name, kind, source_file, signature, vllm_version)
  -- kind: cli_flag | class | function | config_key
  -- THE ORACLE. Populated by AST-walking vLLM source + parsing `--help`.

golden_queries(id, question, source_url, expected_answer,
               expected_chunk_ids[], category, difficulty)

runs(id, config_json, git_sha, started_at)

answers(id, run_id, query_id, answer_text, latency_ms,
        input_tokens, output_tokens, cost_usd, abstained)

claims(id, answer_id, claim_text, supporting_chunk_id, support_score,
       verdict, verifier)
  -- verifier: symbol_oracle | span_attribution | llm_judge
  -- verdict: supported | unsupported | contradicted | unverifiable
```

Track `cost_usd` and `latency_ms` from day one. Interviewers ask you to estimate per-request cost and then halve it; you want real numbers.

---

## Pipeline

1. **Ingest** — clone vLLM at a pinned commit; parse docs (Sphinx/MkDocs sources) + README + public docstrings. Record version and commit on every document.
2. **Chunk** — two strategies, both persisted so they can be compared on identical queries:
   - `fixed`: ~512 tokens, 64 overlap (naive baseline)
   - `structure_aware`: split on heading boundaries; **never split a fenced code block**
3. **Index** — embed with bge-small, write to pgvector (HNSW); build `tsvector` in the same row.
4. **Retrieve** — dense top-k ∪ sparse top-k → RRF → cross-encoder rerank → top-n.
5. **Generate** — assemble prompt with numbered context spans; require the model to cite span IDs inline.
6. **Ground** —
   - extract atomic claims from the answer
   - attribute each to a span
   - route verifiable claims (flags, symbols, config keys) to the **symbol oracle**; everything else to span attribution + LLM judge
   - apply the abstention policy at the configured threshold
7. **Evaluate** — run the golden set across configs; log to W&B; write results tables.

---

## Milestones

### Week 1 (Sept 9–15) — retrieval works and is measured
- Ingestion with version tagging
- Both chunking strategies implemented
- Hybrid retrieval + reranking
- **Exit criterion:** a `recall@k` table comparing chunking × retrieval configs, in the README

Expected first real finding: fixed-size chunking mangles code blocks, and code blocks are what users actually ask about.

### Week 2 (Sept 16–22) — grounding + evals
- Golden set of 60–80 questions **sourced from real vLLM GitHub issues and discussions** — a real query distribution, not synthetic generation. This is a question you will be asked in interviews; have the good answer.
- Claim extraction + span attribution
- Symbol oracle
- Ragas triad + LLM-as-judge
- Evals running in GitHub Actions
- **Exit criterion:** the tradeoff curve — hallucination rate vs. abstention/completeness across grounding thresholds

### Week 3 (Sept 23–30) — ship
- FastAPI + Docker
- Gradio demo on HF Spaces, linked from resume
- README with both tables and the chart
- **File the vLLM documentation PR** using gaps the eval harness exposed

### Cut list, in order, if behind
1. Symbol oracle — fall back to LLM-judge-only faithfulness
2. Version/staleness detection
3. Second chunking strategy

**Never cut:** the golden set, or evals in CI. They are the point of the project.

---

## Phase 2 (October): LoRA judge distillation

**Do not start until phase 1 is shipped and linked on the resume.**

The problem this solves is real, not manufactured: every eval sweep calls Claude to judge faithfulness on every claim, and that cost scales with the number of configs you test. Distil it.

1. **Harvest training data** from phase 1 — every `claims` row already stores the judge verdict, the claim, and the supporting span. A few thousand rows accumulate naturally across sweeps.
2. **LoRA-tune** `Qwen2.5-1.5B-Instruct` with PEFT on `(claim, span) → verdict`. QLoRA (4-bit) if VRAM is tight.
3. **Measure what matters:** agreement with the Claude judge (Cohen's κ, not raw accuracy — the classes are imbalanced), per-judgment cost, and latency.
4. **Report the tradeoff:** agreement vs. cost. This is the second chart in the README.

Why this framing beats fine-tuning the reranker: 60–80 golden questions is far too little data to fine-tune a retriever on, and "I fine-tuned on 80 examples" is an answer that loses you the room. Judge distillation has thousands of labels generated as a byproduct of work you already did.

**Resume keywords earned:** LoRA, QLoRA, PEFT, knowledge distillation, model evaluation, inference cost optimization.

## Phase 3 (early November): k3s serving path

Pairs with NCA-GENL study — Triton, TensorRT-LLM, and NIM are on that exam, and this is the same problem space.

1. **k3s** on the local RTX 5070 Ti with the NVIDIA device plugin for GPU scheduling
2. Serve the generator via **vLLM** as a Deployment; `anchor` API as a second Deployment
3. Quantize the served model (AWQ or GPTQ) and record the memory and throughput delta — this reuses the methodology from your inference profiling project
4. **HPA** driven by a custom metric (queue depth or in-flight requests, not CPU — say why in the README)
5. **Load test** with k6 or locust. Record **RPM sustained and P99 latency**, before and after quantization.

The deliverable is the numbers, not the YAML. A resume bullet reading "served a quantized model via vLLM on k3s at N RPM, X ms P99" is only worth something if you measured N and X yourself.

**Resume keywords earned:** Kubernetes, HPA, GPU scheduling, vLLM serving, quantization (AWQ/GPTQ), load testing, P99 latency, throughput optimization.

**Honest risk on both phases:** each adds an interview surface you must be able to defend. If someone asks why you chose κ over accuracy, or why HPA on queue depth rather than CPU, you need the answer. That is the price of the keywords, and it's the reason these are sequenced after phase 1 rather than folded into it.

---

## Conventions

- Type hints everywhere; `mypy` clean.
- `ruff` for lint + format. No bare `except`.
- Every retrieval and grounding function gets a unit test with a fixed fixture. Randomness pinned by seed.
- All tunable values live in `config.py` via pydantic-settings — **no magic numbers inline**, because every knob becomes a sweep dimension.
- Every eval run writes a row to `runs` with the git SHA. Results must be reproducible from the table.
- Commits: conventional commits (`feat:`, `fix:`, `eval:`).
- Secrets in `.env`, never committed. `.env.example` checked in.

## Instructions for Claude Code

- **Ask before adding a dependency.** The stack above was chosen deliberately; adding LangChain or LlamaIndex would hide exactly the retrieval logic this project exists to demonstrate.
- **Never fabricate evaluation numbers.** If a metric hasn't been computed, leave it blank or `TODO`. Numbers in the README get defended in interviews.
- Prefer small, reviewable diffs over large rewrites.
- When implementing retrieval, grounding, or fusion logic, **explain the approach in the PR body or a comment** — the human needs to be able to defend every design decision out loud.
- Write the test before the implementation for anything in `retrieve/`, `ground/`, or `evaluate/`.
- Do not silently change chunking, embedding, or threshold defaults — those are experiment variables.

---

## Resume keywords this project earns

RAG · hybrid retrieval · reciprocal rank fusion · cross-encoder reranking · pgvector · embeddings · chunking strategy · Ragas · LLM-as-judge · golden-set evaluation · hallucination measurement · guardrails · abstention policy · Weights & Biases · Langfuse · vLLM · FastAPI · Docker · CI for evals · Hugging Face Spaces

---

## Environment notes (local, discovered at setup)

- Python 3.11.16 is installed and pinned via `uv python install 3.11`. The system Python is 3.14, which the project deliberately does not use.
- Postgres runs on host port **5433**, not 5432, so it cannot collide with any other local Postgres.
- The local GPU is an **RTX 5070 Ti Laptop with 12 GB VRAM**. Phase 3 must plan around 12 GB, not the 16 GB of the desktop part: `Qwen2.5-7B-Instruct` needs quantization (AWQ or GPTQ) to serve comfortably, which is convenient because quantization is a phase 3 deliverable anyway.
- `gh` is installed and authenticated as `yashraz23`.

## Decisions taken that differ from this document

- **ragas is not used.** Every published version through 0.4.3 hard-requires
  `langchain`, `langchain-community`, `langchain-openai` and `openai`, and the
  installed version failed to import. This file names ragas *and* forbids
  LangChain; those cannot both hold. The triad is implemented in
  `anchor/evaluate/triad.py`. Reversing this means accepting LangChain.
- **Claim extraction is deterministic**, not model-driven: the generator already
  cites a span per claim, so the pairs are in the text. Keeps runs reproducible
  and re-thresholding free.
- **The symbol oracle rules only on CLI flags inside a vLLM invocation**, and
  does not adjudicate config keys. Judging every flag and backticked identifier
  gave a 42% false "not found" rate on real answers.
- **CI evals run over `tests/fixtures/corpus`**, not the real corpus, which
  would take ~15 minutes per run to build.
