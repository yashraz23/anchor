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

> **Status: week 2, in progress.** Tables still marked TODO are empty on
> purpose. Numbers appear here only once the harness has produced them.

---

## Results

### Chunking: code-block integrity

Measured, not judged. For every fenced code block in the corpus, does some chunk
under this strategy contain that block's body verbatim? A block split across two
chunks fails, because neither half is a command anyone can run.

Corpus: vLLM v0.28.1rc0 at commit `65f3fca5`, 1722 documents, 6245 code blocks.
(Document count later rose to 1729 once attribute docstrings were captured; the
integrity numbers below predate that and are unaffected by it, since it added
no code blocks.)

| Chunking | Code blocks intact | Rate |
|---|---|---|
| `fixed` (512 tokens, 64 overlap) | 6095 / 6245 | 97.6% |
| `structure_aware` | 6245 / 6245 | 100% |

The aggregate rate is the least interesting cut of this, and on its own it is
misleading. A 512-token window only breaks a block long enough to straddle a
window boundary, and most blocks in these docs are one-line invocations. Split
by block length, the effect is stark:

| Block size (tokens) | Blocks | `fixed` intact | `structure_aware` intact |
|---|---|---|---|
| ≤ 32 | 3161 | 100% | 100% |
| 33 to 64 | 1744 | 100% | 100% |
| 65 to 128 | 966 | 95% | 100% |
| 129 to 256 | 316 | 79% | 100% |
| > 256 | 58 | **31%** | 100% |

So the honest finding is narrower and more useful than "fixed-size chunking
mangles code blocks". It mangles the *long* ones: the multi-line configs and
full serve invocations, which are exactly the blocks a user wants to copy whole
and the ones a short snippet cannot substitute for. Short blocks are unaffected,
and any evaluation reporting only the aggregate would have missed this.

Why the boundary behaves that way: with a 512-token window and 64-token overlap
the stride is 448, so a block starting at offset `a` survives only if its length
fits in `512 - (a mod 448)`. Worst-case alignment leaves 65 tokens. Every block
above that is at the mercy of where it happens to land.

**Caveat, measured:** 11 of 6245 blocks (0.18%) exceed the encoder's 512-token
input limit, the largest at 913 tokens. Structure-aware chunking keeps those
whole in storage, but bge-small still truncates them at embed time, so "intact"
means retrievable as text, not fully embedded.

### Corpus: where vLLM's documentation actually lives

Building retrieval over this corpus surfaced something worth recording, because
it changes what "ingest the docs" has to mean.

vLLM's engine arguments and config keys are documented almost entirely as
**attribute docstrings**: a bare string expression following a field
assignment.

```python
max_num_seqs: int = Field(default=DEFAULT_MAX_NUM_SEQS, ge=1)
"""Maximum number of sequences to be processed in a single iteration."""
```

Python's `ast.get_docstring` cannot see these, because they belong to no
function or class. An ingester that walks docstrings the obvious way silently
drops every one. In `vllm/config/` alone that is **532 field descriptions**.

It compounds: the rendered engine-arguments page is generated from those fields
at docs build time and is not committed. The source tree holds only a stub
ending in an mkdocs include directive, and 29 doc pages are stubs of that shape.
So the authoritative description of every engine flag is absent from both the
docs sources *and* a naive docstring pass.

Measured effect on the query "what does max_num_seqs do and what is its
default":

| | Before | After |
|---|---|---|
| Chunks mentioning `max_num_seqs` | 12 | 18 |
| Top reranked span | wrong document | `SchedulerConfig.max_num_seqs`, score 0.99 |

This is also the clearest argument yet for the symbol oracle in week 2. The
ground truth for flags and config keys is in the source tree, not the prose.

### Retrieval: an open weakness, to be quantified

On a natural-language phrasing of the same question, "how do I limit the number
of concurrent sequences the server handles", the pipeline does **not** surface
that chunk. Its cosine similarity is 0.654 against a corpus best of 0.709, and
78 chunks outrank it. Raising the candidate pool to 200 does not rescue it: the
cross-encoder scores every candidate below 0.11, so it is not confident in any
of them.

The cause is a vocabulary gap between how a user asks and how the docs are
written. Recording it rather than tuning it away: `dense_top_k` and the
reranking knobs are experiment variables, and the golden set is what decides
them. The golden questions come from real GitHub issues, so this gap is exactly
the distribution they will test.

### The golden set

60 to 80 questions, every one filed by a real person on vLLM's issue tracker.
Nothing is synthesised, and that is the point: a model-generated set is written
in the vocabulary of the documentation it was generated from, so it flatters
retrieval and measures the wrong thing.

Built by harvesting issues labelled `usage` and `documentation`, sorted by
reaction count, then curating against retrieval evidence rather than titles.

| | |
|---|---|
| Candidates harvested | 563 |
| Curated so far | 33 |
| Target | 60-80 |

Two curation rules are worth stating, because they are what stop the set
measuring the wrong thing:

- **Questions the system currently fails are kept.** A set built only from what
  retrieval already finds is a transcript of current behaviour and cannot
  measure anything. "How to disable logging" is in the set precisely because the
  pipeline gets it wrong today.
- **Questions no documentation can answer are dropped, not marked hard.**
  "Can I get the loss of model directly?" was rejected on that basis, as was a
  Whisper-timestamps question whose only match is a code sample about
  implementing a model. Keeping them would make retrieval look broken when the
  corpus, not the retriever, is the limit.
- **A rejection can itself be wrong, so rejections get re-checked.** A
  sparse-embeddings question was dropped in the first pass and reinstated once
  `specific_models.md` turned out to document it, serve command included.

Entries store expected *document paths*, not chunk ids: chunk ids are stable
only within one (strategy, commit) pair, so pinning them in a version-controlled
file would rot on the next re-chunk.

### Generation: cost and citation behaviour

Answers are generated with `claude-opus-5` over the top spans from the best
retrieval configuration, and every claim must cite the span number it came from.
Measured over the full golden set, run 5:

| | |
|---|---|
| Answers | 33 |
| Cost per answer | **$0.0280** |
| Total cost | $0.9251 |
| Mean latency | 8.4 s |
| Mean input / output tokens | 2284 / 665 |
| Spans supplied that were cited | 127 / 165 (77%) |
| Answers citing a span that did not exist | 0 |
| Answers citing nothing at all | 0 |

Cost per request is a question that gets asked directly, so it is measured
rather than estimated. Abstained rows store NULL usage rather than zeros, so an
abstention can never be averaged in as a free, instant answer.

**23% of supplied spans are never cited.** That is a retrieval signal, not a
generation fault: `rerank_top_n` is 5, and answers typically use fewer, so the
surplus is paid for in input tokens on every single query. It is a config knob,
so the sweep decides it, not a hand-tuned guess.

**A bug worth recording, because it would have become a false headline.** The
first run reported that 4 of 33 answers cited a span that was never supplied,
which reads as a checkable hallucination. Every one was the citation parser
mistaking bracketed integers in code for citations:

```
output.outputs[0].text
cudagraph_capture_sizes=[1, 2, 4, 8, 16]
```

Both are real fragments from generated answers. Citations are only meaningful in
prose, so fenced blocks and inline code spans are now stripped before parsing.
After the fix the true count is zero, and the eight test cases guarding it are
taken verbatim from those answers. Publishing the unfixed number would have
been a hallucination claim about the model that was really a bug in the
measurement.

### The symbol oracle

The reason a subset of faithfulness checks here are ground-truthed rather than
LLM-judged. `--max-num-seqs` either exists in vLLM's argument parser at a pinned
commit or it does not, and no judge's opinion changes that.

Symbols are recovered by walking the AST of the pinned checkout, never by
importing vLLM or running `--help`: both need the package installed with CUDA,
and the oracle has to run in CI.

| Kind | Symbols |
|---|---|
| CLI flags | 433 |
| Config keys | 3603 |
| Classes | 5366 |
| Functions | 3967 |
| **Total** | **13369** |

Built from 2392 files at v0.28.1rc0. Verified: real flags resolve, and an
invented `--enable-turbo-mode` in a serve command is returned as
`contradicted`, not merely unsupported, because the source says otherwise.

**vLLM declares its command line two ways, and missing one produced false
accusations.** Most flags are string literals in `add_argument`. The rest are
generated in a loop with no literal anywhere:

```python
frontend_kwargs = get_kwargs(cls)
for key, value in frontend_kwargs.items():
    group.add_argument(*extra, f"--{key.replace('_', '-')}", **value)
```

A literals-only oracle reported `--tool-call-parser`, `--chat-template` and
`--enable-auto-tool-choice` as non-existent. All three are real. The second
mechanism is now detected structurally, by finding functions that both call
`get_kwargs` and pass an f-string to `add_argument`, and fields are inherited
through base classes because a subclass inherits its parent's flags.

**Jurisdiction is narrow, on purpose.** The first version judged every `--flag`
and every backticked identifier in an answer. Across 33 real answers that
produced 106 checkable mentions with a 42% "not found" rate, and almost none
were hallucinations:

| What it actually was | Example |
|---|---|
| Another tool's flag, used correctly | `pip install --editable`, `numactl --cpunodebind` |
| A config *value*, not a field name | `fp8_e5m2` is a value of `kv_cache_dtype` |
| A variable from the model's own example | `a_val`, `b_val`, `target_token` |
| Another library's API parameter | `max_new_tokens` is HuggingFace's |

An oracle that reports those as contradictions is worse than no oracle, because
its verdicts carry the authority of ground truth. So it now rules only on flags
appearing inside an actual vLLM invocation, and does not adjudicate config keys
at all: a backticked identifier is as likely to be a value, a local variable, or
another library's parameter, and the name alone cannot distinguish them.

After narrowing, the same 33 answers yield 7 adjudicable flag claims and **0
contradictions**. Every flag the answers attributed to vLLM is real.

The cost is recall, and that is the right way round. A hallucinated flag outside
a command line falls through to the LLM judge, which is the fallback the design
already has. A false contradiction has no such safety net.

### Retrieval: recall@k by chunking strategy and retrieval mode

Both chunking strategies are indexed at the same time and queried identically,
so the comparison is like-for-like. Recall is measured at the **document**
level: of the top k retrieved chunks, did any come from a document the golden
entry names? The two strategies cut the same document into different numbers of
pieces, so a chunk-level score would reward whichever produces more chunks
rather than whichever retrieves the right material.

| Chunking | Retrieval | recall@1 | recall@3 | recall@5 | recall@10 | recall@20 |
|---|---|---|---|---|---|---|
| `fixed` | dense | 0.23 | 0.47 | 0.64 | 0.74 | 0.74 |
| `fixed` | sparse | 0.15 | 0.24 | 0.24 | 0.24 | 0.27 |
| `fixed` | hybrid (RRF) | 0.30 | 0.55 | 0.68 | 0.76 | 0.79 |
| `fixed` | hybrid (RRF) + rerank | 0.35 | 0.64 | 0.70 | 0.77 | 0.79 |
| `structure_aware` | dense | 0.45 | 0.61 | 0.67 | 0.74 | 0.91 |
| `structure_aware` | sparse | 0.18 | 0.24 | 0.24 | 0.24 | 0.27 |
| `structure_aware` | hybrid (RRF) | 0.48 | 0.67 | 0.71 | 0.79 | 0.94 |
| `structure_aware` | **hybrid (RRF) + rerank** | **0.55** | **0.89** | **0.91** | **0.92** | **0.94** |

Run 3, commit `fd75f95`, 33 golden questions. Reproduce with
`uv run anchor eval-retrieval`.

Every cell runs the same questions, so the paired record is the honest
comparison and a two-sided exact sign test is the appropriate check. It assumes
only that a question separating the two configurations is a coin flip under the
null, which is the least this comparison can assume. Ties carry no directional
information and are excluded, which is what the sign test does by construction.

| k | `structure_aware` better | `fixed` better | same | sign test |
|---|---|---|---|---|
| 1 | 11 | 1 | 21 | p = 0.0063 |
| 3 | 11 | 0 | 22 | p = 0.0010 |
| 5 | 9 | 0 | 24 | p = 0.0039 |
| 10 | 7 | 0 | 26 | p = 0.0156 |
| 20 | 7 | 0 | 26 | p = 0.0156 |

**Structure-aware chunking wins 11 questions and loses none at k=3.** With 33
questions that is significant at every depth measured, and the direction never
reverses. It is a single eval set on one corpus, so it is evidence about vLLM's
documentation rather than a general claim about chunking.

What the table supports:

- **Sparse alone is weak here, at 0.27 recall@20.** Not a bug. These are
  natural-language issue titles that share almost no vocabulary with the docs,
  and keyword search cannot match "concurrent sequences" to `max_num_seqs`. It
  earns its place inside the hybrid, not on its own.
- **Reranking helps structure-aware far more than fixed**, lifting recall@3 from
  0.67 to 0.89 against 0.55 to 0.64. A cross-encoder scores a query against a
  passage, and a fixed-size window that starts and ends mid-sentence is a worse
  passage to score. Chunking and reranking are not independent choices, which is
  not visible from either row alone.
- **Two questions are never retrieved by any configuration.** Both were flagged
  as expected failures during curation, before the harness existed:
  `vllm-6660-disable-logging`, whose answer lives only in `vllm/envs.py` because
  the env-var page is an mkdocs stub, and
  `vllm-23108-gpt-oss-builtin-python-tool`.
- **The absolute numbers fell when the set grew from 21 to 33.** `fixed` at
  recall@1 went 0.45 to 0.35. The second curation batch added harder questions,
  so the set got harder rather than the system getting worse. This is the
  expected direction while a golden set is still being built, and it is the
  reason the run id and commit are printed next to every table.

### Grounding: strictness versus completeness

The headline result. Sweeping the abstention threshold trades how often the
system answers at all against how well-grounded the answers it does give are.
Over 33 golden questions and 306 extracted claims, run 5:

| Abstention threshold | Answered | Abstained | Answer rate | Delivered claims | Cited but unsupported | Uncited |
|---|---|---|---|---|---|---|
| 0.00 | 32 | 1 | 97% | 303 | 1 (0%) | 64 (21%) |
| 0.30 | 31 | 2 | 94% | 293 | 1 (0%) | 57 (19%) |
| 0.50 | 31 | 2 | 94% | 293 | 1 (0%) | 57 (19%) |
| 0.60 | 26 | 7 | 79% | 238 | 0 (0%) | 37 (16%) |
| 0.70 | 21 | 12 | 64% | 186 | 0 (0%) | 21 (11%) |
| 0.80 | 15 | 18 | 45% | 122 | 0 (0%) | 7 (6%) |
| 0.90 | 11 | 22 | 33% | 72 | 0 (0%) | 0 (0%) |

Reproduce with `uv run anchor ground`. The sweep calls no model: claims are
scored once and re-thresholded, so the whole curve costs one pass over answers
already paid for.

**The two failure columns are separate on purpose, and merging them produced a
number four times too large.** A claim that cites a span which does not support
it asserted something its own evidence contradicts. A claim that cites nothing
is merely unattributable. The first version of this table folded both into one
"unsupported" figure and reported **37%**. Split apart:

| | Claims |
|---|---|
| Supported by their cited span | 193 |
| Cited a span that does not support them | **1** |
| Cited nothing at all | 112 |

The strict faithfulness failure rate is 1 claim in 306. The 37% was almost
entirely claims with no citation, which is a different and much less alarming
problem, and publishing it as a hallucination rate would have been wrong.

**Honest hedging is not a hallucination.** Many uncited sentences were the model
correctly declining: "The provided spans don't cover S3 credential
configuration", "I'd need documentation covering sparse embeddings to answer
this". Those assert nothing about vLLM. Counting them as unsupported claims
scores the exact behaviour the abstention policy exists to encourage, and would
push the system toward confident guessing. They are now detected and excluded
from the faithfulness denominator, which moved the uncited rate from 23% to 21%
and, more importantly, stopped penalising the right answer.

**What the curve shows.** Raising the threshold from 0.00 to 0.90 cuts the
answer rate from 97% to 33% and drives uncited claims from 21% to zero. The
knee is around 0.60 to 0.70, where the system still answers 64% to 79% of
questions while more than halving the uncited rate. There is no threshold at
which strictly unsupported claims are a problem, because there was only ever one.

**How the columns are computed.** The curve thresholds on span attribution,
which is free, so the whole sweep is a re-thresholding of one pass. The LLM
judge scores the same claims independently and agrees on 98% of them; see
verifier agreement below. The symbol oracle covers the checkable subset exactly
and found no contradictions across the 3 answers it had jurisdiction over.

### Verifier agreement

Three verifiers, deliberately ordered cheapest-first. The symbol oracle is exact
and free but only covers checkable names. Span attribution is cheap but measures
similarity, which is a proxy for support. The LLM judge reads the span and the
claim together and is the only one that can catch a claim sitting close to its
span while stating something different.

Over 306 claims from 33 answers, run 5:

| Verifier | Claims judged | Cost | Result |
|---|---|---|---|
| Symbol oracle | 3 answers in jurisdiction | free | 0 contradictions |
| Span attribution | 306 | free | 193 supported |
| LLM judge | 306 | $1.30 | 194 supported, 31 unsupported, 81 unverifiable |

**The two agree on 301 of 306 claims (98%)** on the binary question of whether a
claim is supported. The judge was stricter on 2, similarity on 3. That number is
the phase 2 argument in miniature: a free check already reproduces a $1.30 one
almost everywhere, and the interesting question is whether the remaining 2%
matters.

**No claim cited a span that failed to support it.** Both verifiers agree, and
this is the strongest single result in the grounding layer: across 306 claims,
zero cases of the answer pointing at evidence that does not say what it claims.
Every one of the judge's 31 `unsupported` verdicts is on a claim that carried no
citation at all.

What the judge adds is a distinction similarity cannot make. It splits the 112
uncited claims into 31 genuine assertions that should have carried a citation
and 81 that are not factual assertions about vLLM at all, mostly the model
noting the limits of its own evidence.

**Verified on a constructed case, because the real answers contain none.** Given
a span reading "Default 128" and the claim "max_num_seqs defaults to 512", the
judge returns `contradicted` with the reason "The span states the default is 128,
not 512". Cosine similarity between those two is high, since they differ by one
token. This is the failure mode the judge exists for, and the reason the
cited-but-unsupported column is a semantic verdict rather than a similarity
floor.

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

uv run anchor ingest      # clone vLLM at a pinned commit, parse, version-tag
uv run anchor chunk       # build both chunking strategies over the corpus
uv run anchor index       # embed into pgvector and build the tsvectors
uv run anchor integrity   # the code-block integrity table above

uv run anchor search "what does max_num_seqs do and what is its default"

uv run anchor golden harvest    # candidate questions from real vLLM issues (needs gh)
uv run anchor golden propose    # retrieval evidence for each, for human review
uv run anchor golden validate   # every expected document exists in the corpus

uv run anchor eval-retrieval    # the recall@k sweep; writes a row to `runs`

uv run anchor ask "what does max_num_seqs do"   # one cited answer
uv run anchor golden sync                       # golden set into Postgres
uv run anchor answer-golden --yes               # answer the whole set; costs money
uv run anchor oracle-build                      # populate the symbol oracle
```

Re-running `ingest` stays on the commit already ingested. Advancing to the
branch tip is explicit (`--latest`), because it orphans every chunk and
embedding built against the previous commit.

Every stage after `ingest` pins itself to the commit the corpus was ingested at,
read back from the database. vLLM's main branch moves several times a day, so a
stage that re-resolved the branch tip would check out a tree that no longer
matches the `documents` rows.

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
