# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A verbatim knowledge-graph RAG system for the Digital Personal Data Protection Act, 2023 (India), plus a FastAPI + SSE chatbot POC on top of it. Everything lives under `KG/`. Read `KG/FLOW.md` before touching the pipeline — it is the authoritative, numbers-grounded explanation of every stage (PDF extraction, graph construction, chunking, retrieval, why there are no embeddings by default). `KG/README.md` has the quickstart and file reference. `KG/TARGET.md` is the production-hardening roadmap and the source of truth for *why* the pluggable-provider, hybrid-retrieval, and audit-log machinery below exists — read it before changing any of them.

## Commands

All commands run from `KG/`.

```bash
# One-time setup
pip install pdfplumber networkx rank_bm25 pyyaml neo4j snowballstemmer fastapi uvicorn sse_starlette
ollama pull qwen2.5:3b-instruct   # or qwen2.5:7b-instruct for materially better answers; see providers below
pip install sentence-transformers torch   # optional, only for --hybrid / DPDP_HYBRID=1
pip install langfuse                      # optional, only if LANGFUSE_PUBLIC_KEY/SECRET_KEY are set

# Full rebuild (PDF -> graph -> search index)
python build.py --neo4j     # PDF -> graph; --neo4j also loads into Neo4j (needs .env)
python build.py             # same, without touching Neo4j
python index.py             # graph -> 142 BM25-searchable chunks (fast, no LLM)
python index.py --plain     # also regenerates plain_language.json (~3h local CPU inference — see below)

# Tests
python test_build.py        # 15 checks: structural invariants + eval.yaml retrieval scoring (101 cases)
pytest test_build.py         # same, if you prefer the pytest runner
python eval_answers.py      # answer-quality eval (answer_eval.yaml): quote-exactness, citation
                             # resolution, rupee-figure accuracy, abstention — separate from
                             # test_build.py because it costs a real generation call per case

# CLI query (no server)
python ask.py "what is the fine if customer data leaks?"
python ask.py --retrieval-only "penalty for a data leak"   # skip the LLM call, just show what was retrieved
python ask.py --parent-doc "..."   # search sub-section precision, answer from the whole section
python ask.py --hybrid "our processor's data centre sits in Singapore"   # dense+BM25+rerank, opt-in
python ask.py --decompose "processor abroad lost a child's records, what do we owe and to whom"
python ask.py --provider claude --model claude-sonnet-5 "..."   # override DPDP_PROVIDER for one run

# Web app
uvicorn api:app --port 8000 --reload    # http://127.0.0.1:8000
```

There is no single-test flag — `test_build.py` is a flat script of `test_*` functions run via `main()`, not pytest-parametrized. To isolate one check, comment out the others in the `tests` list comprehension at the bottom of the file, or run `python -c "import test_build; test_build.test_penalty_join()"` for one function directly.

## Non-obvious setup

- **`.env`** (gitignored, copy from `.env.example`) holds Neo4j credentials and Ollama config (`NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`, `OLLAMA_HOST`, `DPDP_MODEL`). `build.py` loads it itself via a 6-line stdlib parser — no `python-dotenv` dependency.
- **Neo4j Aura quirk**: some Aura instances reject the usual `neo4j`/`neo4j` user+database pair and require both set to the instance ID instead. Check which pair your instance accepts before assuming a connection failure is a bug.
- **`plain_language.json`** lives at the repo root (`KG/`), not under `out/`. It costs ~3h of local CPU inference to regenerate and is deliberately excluded from `.gitignore` (unlike everything else in `out/` and `review/`, which are disposable and rebuilt in seconds). Never `rm -rf out` expecting it to be safe — it isn't in `out/` precisely because of a past incident where it was.
- **`logs/`** (gitignored) holds `audit_log.jsonl`, the FastAPI backend's append-only per-request record (question, retrieval trace, full answer, citation resolution, timings, `build_id`). Also outside `out/` for the same reason `plain_language.json` is — it must survive a rebuild.
- **`llm.py` is a pluggable provider** (`OllamaProvider` / `ClaudeProvider`), selected by `DPDP_PROVIDER` (default `ollama`, no key needed). Every caller (`ask.py`, `api.py`, `index.py`) goes through the same five module functions (`chat`, `chat_stream`, `check`, `models`, `warn_if_small`) regardless of provider — never call the Ollama HTTP API or the `anthropic` SDK directly from anywhere else.
- **Ollama must be running** (`ollama serve`) for `index.py --plain` and any non-`--retrieval-only` answer path when `DPDP_PROVIDER=ollama`. With `DPDP_PROVIDER=claude`, `ANTHROPIC_API_KEY` must be set instead — no local server needed. Retrieval-only paths (`build.py`, `index.py` without `--plain`, `ask.py --retrieval-only`, `test_build.py`) never call any provider.
- **A 3B local model is the current default** (`DPDP_MODEL=qwen2.5:3b-instruct`) and is the known weak link — it has misread Schedule penalty figures and glossed over legal nuance (e.g. conflating "family" with "nominated person" under §14). Retrieval is not the bottleneck; the model is. `answer_eval.yaml` has permanent regression cases for both errors. Prefer `DPDP_PROVIDER=claude` or `qwen2.5:7b-instruct` for anything beyond a quick smoke test.
- **The embedder was measured, not assumed.** `DPDP_EMBED_MODEL` overrides `hybrid.py`'s default. Three-way comparison on `eval.yaml`'s 101 cases: `all-MiniLM-L6-v2` (80MB) **92/101**, `law-ai/InLegalBERT` (534MB) **89/101**, `BAAI/bge-base-en-v1.5` (439MB) **92/101**. The legal-domain model lost because it is a raw MLM checkpoint never contrastively fine-tuned for sentence similarity — domain pretraining is a different axis from retrieval-readiness. MiniLM retained. Don't re-litigate this without re-running the comparison; at 142 chunks the cross-encoder reranker, not the embedder, is doing the discrimination.
- **Hybrid retrieval (`hybrid.py`) is off by default** (`DPDP_HYBRID=1` or `ask.py --hybrid` to enable). Needs `sentence-transformers` + `torch` and two locally-cached HF models (`sentence-transformers/all-MiniLM-L6-v2`, `cross-encoder/ms-marco-MiniLM-L-6-v2`) — `hybrid.py` forces `HF_HUB_OFFLINE=1` by default so a slow/absent network doesn't stall every startup; set `DPDP_HF_ONLINE=1` the first time a model isn't cached yet. Measured, not assumed better: 92/101 on `eval.yaml` vs BM25-only's 90/101, cutting `known_miss` cases from 11 to 5, but with its own new misses on a few previously-easy exact-match questions — see the comparison methodology and the code comments in `ask.py`'s `retrieve()` before changing the reranker's candidate text; a plausible-looking "improvement" there was measured to make things *worse* (85/101) and was reverted.
- **Graph authority (`build.py`'s `add_authority`)** stores a PageRank score per node, computed over `REFERENCES` (1.0) + `PENALISED_BY` (0.5) only. `MENTIONS` is excluded deliberately and this was measured: including it at *any* weight makes every top-authority node a Definition, because all 605 `MENTIONS` edges point into the 28 definitions — that measures term usage, not how load-bearing a provision is. Consumed in `ask.py` purely as a within-tier tie-breaker during expansion, never to reorder edge types. Honest caveat: it does **not** fix §40(2)-style hub-flooding (that arrives via edge-type priority, which authority never overrides), and `eval.yaml` cannot measure it at all since the eval scores seeds only (`hop == 0`) while authority affects `hop >= 1`.
- **Query decomposition (`decompose.py`) is off by default** (`DPDP_DECOMPOSE=1` or `ask.py --decompose`). Splits a compound question via one LLM call and retrieves each part separately. The original question is always retrieved first and merges are additive, so a bad split can only add noise, never remove a provision the plain query would have found. Measured limitation: on the local 3B model the split is unreliable for scenario-shaped questions — it dropped a cross-border thread in one case and invented an out-of-scope sub-question in another. Re-measure with `DPDP_PROVIDER=claude` before judging the technique.
- **Entailment checking (`entailment.py`) is off by default** (`DPDP_ENTAILMENT=1`, needs `cross-encoder/nli-deberta-v3-base`, ~749MB). Checks whether the cited provisions actually *support* each sentence of the answer — the gap `check_citations()` cannot cover, since it only verifies a provision exists and was retrieved. Validated on both errors this project recorded: the §14 "family can step in" bug scores 0.004 (`unsupported`) and the §17(3) "startups are automatically exempt" trap 0.001, while their correct counterparts score 0.998 and 0.994 — in every case `check_citations()` had said `verified`. Results appear as a `claims` field on the `citations` SSE event and in the audit log. Two traps if you touch it: (1) fetch the model with explicit `allow_patterns` — an unfiltered `snapshot_download` pulls 9 ONNX builds plus duplicate weights (~3.6GB) and will fill the disk; (2) `HF_HUB_OFFLINE=1` makes the model unloadable because sentence-transformers calls the hub's `model_info` during construction, and `os.environ.pop()` cannot fix it (the flag is read into a module constant at import), which is why `entailment.py` patches `huggingface_hub.constants.HF_HUB_OFFLINE` around the load. Note also that citation-attribution prefixes ("§29(1) states that …") are stripped before checking — without that, near-verbatim quotes score ~0.000 and correct answers get flagged.
- **Langfuse tracing (`observability.py`) is off unless `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are both set** (`LANGFUSE_BASE_URL` optional, defaults to Langfuse Cloud). When on, every `/api/chat` request becomes one Langfuse trace with three nested observations — `retrieve` (retriever), `generate` (generation), `verify_citations` (evaluator) — matching the three SSE stages and `audit_log.jsonl`'s own record shape. `observability.trace()`/`observability.step()` are context managers that yield `None` and do nothing when disabled — callers never branch on `observability.ENABLED` directly, and the `langfuse` package is only imported the first time one of them actually needs it. Verified resilient to a misconfigured or unreachable Langfuse endpoint: `check()` (surfaced in `/api/health`'s `tracing` field) reports the failure without raising, and a request still completes normally (span export just logs a warning and gives up) — tracing failures must never take down the actual assistant.
- **The abstention gate in `api.py`** (`should_abstain`, threshold `ABSTAIN_THRESHOLD`) is a first-line filter, not a complete out-of-domain classifier — it catches clearly unrelated questions but not adjacent-domain ones (GDPR/HIPAA-style questions score as high as genuine DPDP questions on BM25 because they share real legal vocabulary). This is documented, not hidden, in the code comment next to the threshold.

## Architecture

The pipeline is five scripts, each a stage, each disposable/regeneratable except the two "expensive" cached outputs (`out/dpdp_graph.json` family and `plain_language.json`):

```
build.py  → out/dpdp_tree.json, out/dpdp_graph.json, out/schedule.json,
            out/dpdp.gexf, out/graph.html, out/load.cypher, review/*.md
            (+ optionally pushes to Neo4j via --neo4j)
index.py  → out/index.json (142 BM25 chunks); --plain also writes plain_language.json
llm.py    → pluggable provider (Ollama default / Claude) — chat, chat_stream, check, models
hybrid.py → optional dense retrieval: embeddings + RRF fusion + cross-encoder rerank
decompose.py → optional query decomposition for compound questions
entailment.py → optional NLI check: does the cited text actually support the claim?
observability.py → optional Langfuse tracing, on only if credentials are set
ask.py    → CLI: question -> vocab.yaml expansion -> BM25(+hybrid) -> 2-hop graph
            expansion -> cited answer
api.py    → FastAPI: same retrieval as ask.py, streamed over SSE, plus citation
            verification, abstention gate, and audit_log.jsonl
eval_answers.py → answer-quality eval (answer_eval.yaml): quote-exactness, citation
            resolution, rupee-figure accuracy, abstention — deterministic, no judge model
web/index.html → single-file chat UI, no build step, consumes the SSE stream
```

### The two non-negotiable properties (enforced by code, not convention)

1. **Verbatim storage.** Every node's `text` field is the Act's exact words. `build.py` reassembles the *entire* document from the graph in document order and diffs it character-for-character against the raw extracted PDF text; a single dropped/reordered/invented character fails the build (`test_lossless_round_trip`). LLM-generated text (glosses, generated questions) is a strictly separate field, never merged into `text`, and is never quoted or cited in an answer — only used to *find* the right verbatim chunk.
2. **The graph is authoritative for structured facts.** Penalty amounts, cross-references, and citation validity are never trusted from LLM output — they are read from the graph and either rendered directly (penalty table) or used to verify what the model claimed (citation status: `verified` / `out_of_context` / `unresolved`). See "Citation verification" below.

### Extraction (`build.py`)

The Gazette PDF has three columns per page (marginalia-left, body, marginalia-right) that `pdfplumber`'s naive text extraction merges into one garbled stream. `build.py` derives column boundaries and the indentation ladder (the depth signal that resolves clause markers like `(i)`, which is ambiguous between "sub-clause `(i)`" and "letter-`i`" without positional context) from the PDF's own word coordinates at runtime — nothing is hand-tuned to this specific document. Three typesetting conventions that can't be geometrically derived (headnote-to-section attachment, "bare" cross-reference resolution, illustration boundaries) are written to `review/*.md` for human sign-off and can be corrected via `overrides.yaml` without touching parser code.

### Graph model

Nodes: Act → Chapter → Section → SubSection/Clause/SubClause → Illustration, plus `Definition` nodes (28, one per defined term in §2) and `Penalty` nodes (7, one per Schedule row). Edges include structural containment, `DEFINES`, `MENTIONS` (any provision referencing a defined term), `REFERENCES` (cross-references between sections, resolved via regex + the "bare reference means current section" convention), and `PENALISED_BY` (the join between an obligation and its Schedule penalty — the single edge type that makes the graph worth having, since no embedding-based retrieval can infer that §8(5)'s security-safeguards duty maps to Schedule entry 1's ₹250-crore penalty from text similarity alone).

### Retrieval — BM25 by default, hybrid opt-in

Retrieval is BM25 (`rank_bm25`) over the 142 chunks by default, not a vector store. `FLOW.md` has the full rationale; the short version: the corpus is small (142 chunks), exact statutory terms (section numbers, defined terms, rupee figures) matter more than semantic similarity for this domain, and the vocabulary gap between layperson questions and statutory language is closed by two cheaper, more auditable mechanisms instead:
- `vocab.yaml` — a hand-maintained synonym map (e.g. "leak" → "personal data breach") applied before BM25 scoring.
- `plain_language.json` — LLM-generated plain-English questions per chunk (indexed, never cited), so a layperson's phrasing has something to match against even when it shares no vocabulary with the statute.

`hybrid.py` (opt-in, `DPDP_HYBRID=1` or `ask.py --hybrid`) adds dense embeddings fused with BM25 via Reciprocal Rank Fusion, then reranks the fused shortlist with a cross-encoder — implemented per FLOW.md's own specified upgrade path, and gated on the eval set rather than shipped by default. Never dense-only: BM25 stays in the fusion because dense retrieval alone regresses badly on exact section numbers and rupee figures.

After retrieval, `ask.retrieve()` performs **up to two hops** of graph expansion from the top hits (following `REFERENCES`, `PENALISED_BY`, `DEFINES`, `MENTIONS`), with each hop's contribution decayed (`HOP_DECAY`) so distant provisions are still reachable without flooding the context — this is the "Graph" part of GraphRAG. Expansion edges are rolled up to the nearest *chunked* ancestor on both sides (`chunk_for`) — this matters because a short section's `REFERENCES`/`MENTIONS` edges live on its un-chunked children, not the section chunk itself; without the rollup, expansion could never leave a short section at all. `retrieve(..., parent_doc=True)` searches at sub-section precision but promotes each hit to its containing section for generation — better precision at retrieval time, fuller context at generation time.

`eval.yaml` (101 question/expected-provision pairs, grown from an original 26 — every case checked against real retrieval before being added, `known_miss` cases are confirmed real gaps, not guesses) is the regression harness for retrieval quality; `test_build.py` fails the build if retrieval accuracy drops below its floor. `answer_eval.yaml` + `eval_answers.py` is the companion harness one layer downstream — it proves the model *used* what was retrieved correctly (verbatim quotes, resolved citations, correct rupee figures, honest abstention), which `eval.yaml` cannot check since it only scores retrieval.

### Citation verification (`api.py`)

The FastAPI backend's most important behavior: after the LLM streams an answer, every `§N(x)` / "Schedule entry N" citation it wrote is parsed out and checked against the graph, then labeled `verified` (provision exists and was in the retrieved context), `out_of_context` (provision exists but wasn't retrieved — the model likely hallucinated it from training data), or `unresolved` (no such provision exists at all). This is what lets the frontend show citation trust status instead of asking the user to take the model's word for it. Penalty amounts shown in the UI are read directly from graph `Penalty` nodes, never extracted from the LLM's prose.

### SSE streaming shape

`POST /api/chat` streams event types in order: `retrieval` (which chunks/graph nodes were pulled in, before generation starts — lets the UI show progress immediately since the LLM hasn't started yet, now also carrying `build_id`), then either `abstain` (retrieval was too weak to answer from — see `should_abstain`, a deterministic BM25-score gate — or the request ends here with no generation call made) or `token`×N (incremental answer text), `citations` (the verification result described above, sent after the full answer is available), `done`. `web/index.html` is a single static file with no build tooling — edit it directly, no bundler/transpile step involved.

### Audit trail and versioning

Every request writes one line to `logs/audit_log.jsonl` (`api.audit_log`) — question, retrieval trace, full answer, per-citation resolution status, timings, model/provider, and `build_id`. `build_id` (`api.compute_build_id`) is a content hash of the graph, not a version number anyone has to remember to bump — it changes automatically the moment the Act (or, later, the Rules) is amended and rebuilt, so a logged or displayed answer's `build_id` tells you exactly which text it was generated against.
