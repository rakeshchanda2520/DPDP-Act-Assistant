# FLOW — what every folder and file does, and how a question becomes an answer

This is the map. `README.md` is the pitch; this is the wiring diagram —
every folder, every script, and the exact sequence of calls from a PDF on
disk to a cited answer on screen.

```
Compliance Assistant/
├── data/               source PDF, hand-reviewed overrides, generated artifacts
├── kg_build/           PDF -> validated tree -> graph -> Neo4j + chunks.json
├── backend/            Neo4j + chunks -> retrieval -> model -> checked answer
├── frontend/           one static HTML file, no build step
├── review/             human sign-off output from the last build
├── logs/               append-only audit trail (created at first request)
└── .env / requirements.txt / README.md / FLOW.md
```

There are exactly two runtime flows in this system: **the build**, run by a
human on a schedule (once, or whenever the Act is amended), and **the
question**, run by FastAPI on every request. They share code (`indexing.py`,
`prompt.py`'s citation format) but never run in the same process — the build
never touches Neo4j credentials it doesn't already have from `.env`, and the
backend never touches the PDF.

---

## `data/` — everything an install needs to seed itself

| File | Written by | Read by | What it is |
|---|---|---|---|
| `dpdp_act_2023.pdf` | you (source) | `kg_build` | the Gazette of India PDF — the only ground truth in the system |
| `overrides.yaml` | a human reviewer | `kg_build` | corrections to the three typesetting conventions the parser cannot derive geometrically (see "Phase 1" below) |
| `plain_language.json` | a one-off LLM generation pass (not part of this rewrite's scope) | `kg_build` → `chunks.json` | a per-chunk gloss + generated layperson questions, indexed for retrieval, **never quoted or cited** |
| `vocab.yaml` | a human, incrementally | `backend/retrieval.py` at every request | layperson phrase → statutory term map, e.g. `"leak" → "personal data breach"` |
| `chunks.json` | `kg_build` (generated) | `backend/indexing.py` at startup | the 142 searchable chunks — the **only** artifact besides Neo4j the backend reads |

`chunks.json` is deliberately **not** graph data. It is a derived search
index (BM25 documents + the plain-language layer), and a graph database is
the wrong place to keep a document generated for full-text search. The graph
lives in Neo4j; the search index lives beside the source data as a file.

---

## `kg_build/` — the build, in five phases

Invoked as `python -m kg_build` (add `--neo4j` to publish). Nothing here runs
at request time; this package is not imported by `backend/`.

### `kg_build/extract.py` — Phase 1: PDF → validated tree

This is the file the whole system's trust rests on, and it is the one piece
of code in this rewrite **ported unchanged** rather than rewritten — its
existing round-trip guarantee was too load-bearing to risk restructuring.

**1a. Geometry** (`Geometry.derive`). The Gazette prints three columns per
page — marginalia left, body, marginalia right — and naive PDF text
extraction interleaves them into gibberish. Column boundaries and the
*indent ladder* (the set of horizontal positions markers can start at) are
measured from the actual word coordinates on the page, not hard-coded. The
indent ladder is what makes a marker like `(i)` unambiguous: is it
sub-clause `(i)`, or the letter `i` in an `(h)/(i)/(j)` sequence? Only which
rung of the ladder it sits on answers that — the token alone cannot.

**1b. Extraction** (`extract`). Walks the body column page by page, using the
geometry to assign each line a depth, and separately extracts margin
fragments (headnotes) and the Schedule table (`parse_schedule_page`, which
reads the table structurally rather than as running text).

**1c. Parsing** (`parse`, `candidates`, `disambiguate`). Builds the `Tree` —
Act → Chapter → Section → SubSection → Clause → SubClause → Illustration —
by matching each line's marker against `MARKERS` and using `disambiguate` to
resolve depth ambiguity against the current stack and the indent ladder.

**1d. Cross-references** (`find_xrefs`, `scan_references`). Regex-scans every
provision's text for references to other sections (`REFERENCES` edges) and
to other Acts entirely (`ExternalAct` nodes, deliberately **not** linked into
the DPDP graph — a citation to the IT Act, 2000 is not a DPDP provision).

**1e. Headnotes** (`attach_headnotes`). Which margin note belongs to which
section is a typesetting convention, not something geometry alone resolves
— this is the first of three places `overrides.yaml` can correct a decision
without touching parser code.

**1f. The gate** (`validate`). Reassembles the *entire* document from the
parsed tree, in document order, and diffs it **character-for-character**
against the raw extracted text. `kg_build/__main__.py` checks this return
value before writing anything: a build that fails this check produces no
`chunks.json` and is never pushed to Neo4j. `first_divergence` and
`normalise` exist purely to make a failure's error message point at the
exact character that differs, not just say "mismatch".

This phase also writes `review/` — the same reviewer sign-off files the
original build produced, so a human can inspect the three non-derivable
decisions (headnote attachment, bare cross-reference scope, illustration
boundaries) before trusting a build.

### `kg_build/graph.py` — Phase 2: tree → graph → authority → Neo4j

`networkx` exists **only** in this file, and only at build time. Three
responsibilities:

**2a. `build_graph`** turns the validated tree into a `networkx.MultiDiGraph`:
one node per provision (with its verbatim `text`), containment edges
(`HAS_SECTION`, `HAS_SUBSECTION`, …), `DEFINES` edges from §2's definitions,
`MENTIONS` edges (exact word-boundary match of every defined term against
every provision — exhaustive, deterministic, the kind of recall an
embedding cannot guarantee), `REFERENCES` edges from the cross-reference
scan, and `PENALISED_BY` edges linking a duty to its Schedule row. This last
edge type is *the reason the graph exists*: §8(5) states a duty and carries
no rupee figure; the Schedule row carries the figure and describes no duty.
No text similarity connects them — only this edge does.

**2b. `add_authority`**, called as the final step inside `build_graph`
itself (not left for the caller to remember — a real bug in an earlier draft
of this rewrite was computing it twice, once here and once redundantly in
`__main__.py`; fixed by making `build_graph` own it completely). Runs
PageRank over `REFERENCES` + `PENALISED_BY` edges only — `MENTIONS` is
deliberately excluded, and that exclusion was *measured*, not assumed:
including it at any weight makes every top-authority node a `Definition`,
because all ~600 `MENTIONS` edges point *into* the ~28 definitions. That
measures how often a term is *used*, not how load-bearing a provision *is* —
the opposite of what the signal is for. With it excluded, the ranking is
immediately sensible: §29(1) (right of appeal), the Schedule, §33
(penalties) come out on top. Every node gets an `authority` float, stored
directly on the graph node.

**2c. `push_to_neo4j`** replaces the graph in Neo4j. `MATCH (n:Provision)
DETACH DELETE n` clears the previous build, then nodes are batch-created via
`UNWIND` (one round trip per *kind*, not per node — 404 nodes as 404 separate
statements was slow for no reason), and edges likewise batched per *type*.
Every **value** written is parameterised (`$batch`, `$edges`); the small set
of **identifiers** Cypher cannot parameterise — relationship types, the
per-kind label — are validated against a strict `^[A-Za-z][A-Za-z0-9_]*$`
pattern before being formatted into the query string, since they originate
from this build but a value reaching a query string unchecked is exactly how
injection happens the day someone makes them configurable.

`NODE_PROPERTIES` is an explicit allow-list of what gets written per node.
This is where the real bug this rewrite found and fixed lived: the original
build filtered properties with `isinstance(v, (str, int, bool))`, which
**silently dropped `authority`** — it's a `float`. PageRank was computed on
every build and thrown away before it ever reached the database. Confirmed
fixed by querying Neo4j directly after a push and checking `authority > 0`
on real nodes.

### `kg_build/chunks.py` — Phase 3: graph → 142 searchable chunks

`build_chunks` walks the tree and emits one `Chunk` (the same dataclass
`backend/indexing.py` defines — imported, not duplicated) per Section,
Definition, and Schedule row, plus **sub-section chunks for long sections**
(`LONG_SECTION_WORDS = 220`): §8 and §17 each cover several unrelated duties,
and a single section-sized chunk would retrieve the wrong sub-topic for a
specific question. `_render` reassembles a chunk's verbatim text from the
tree, indentation intact; `_header` attaches context the provision's own
body never states — its chapter, its cross-references, its Schedule
entry — so that context survives into the retrieval index and the prompt
even though it isn't part of the Act's own words.

`attach_plain_language` merges the generated gloss and questions from
`data/plain_language.json` onto each chunk **after** the verbatim structure
is fixed — kept as a separate step so it's obvious in the code that this
text is additive and optional, never part of what a chunk verbatim-quotes.

### `kg_build/__main__.py` — the pipeline, in the order that matters

```
1. read overrides.yaml
2. Geometry.derive(pdf)               -- measure columns and indent ladder
3. extract(geometry)                  -- body lines, margins, Schedule, dropped-page counts
4. parse(lines)                       -- build the Tree
5. reassemble_margins(margins, geo)   -- headnote candidates + inline citations
6. attach_headnotes(tree, ..., overrides)
7. parse_definitions(tree)
8. find_xrefs(tree, overrides)
9. act_metadata(tree)
10. validate(tree, ...)               -- THE GATE. Stop here on failure.
11. build_graph(...)                  -- includes add_authority internally
12. build_chunks(...)
13. attach_plain_language(...)
14. save_chunks(..., data/chunks.json)
15. [--neo4j only] push_to_neo4j(...)
```

Step 10 is not one gate among many — it is *the* gate. Nothing after it runs
on a tree that failed the round-trip check.

---

## `backend/` — the question, in six stages

Everything here is loaded **once**, at process startup (`app.py`'s
`lifespan`), and reused for every request. The corpus does not change
between builds; rebuilding a BM25 index or reopening a Neo4j connection on
every question would be latency spent for nothing.

### `backend/config.py` — read once, trusted everywhere else

Every environment variable this system uses is read in exactly one place.
`load_dotenv` is a six-line `.env` parser with `os.environ.setdefault` —
deliberately *not* overwriting a variable already in the environment, so a
value injected by a container or CI wins over a stale file on disk.
`NEO4J_URI` / `NEO4J_PASSWORD` are `required=True`: a missing credential
raises immediately at import, which means the process refuses to start
rather than accepting requests it cannot serve. `public_settings()` is a
hand-written allow-list of the *only* four values (`provider`, `model`,
`abstain_threshold`, `max_context_chars`) any HTTP response may ever include
— adding a new setting above does not make it public; it has to be added
here too, on purpose. `DPDP_CORS_ORIGINS=*` is checked and **refused
outright** at import time, not silently honoured.

### `backend/graph_store.py` — the runtime source of truth

`load_graph()` runs two Cypher queries — all provisions, then all edges of
the traversable types (`REFERENCES`, `PENALISED_BY`, `DEFINES`, `MENTIONS`,
`HAS_ENTRY`) — and materialises them into an in-memory `Graph` dataclass:
a `dict[str, Provision]` plus an edge list plus a `parent_of` map built from
the `HAS_*` containment edges (used to resolve a citation to a clause up to
the section chunk that actually contains it). `_build_id` hashes the
provisions and edges into a 12-character id, stamped on every SSE event and
every audit record — not a version number a human has to remember to bump,
but a fact that changes automatically the moment a rebuild changes anything,
so "which answers predate the amendment?" is answerable by comparing hashes.

### `backend/indexing.py` — the `Chunk`, and the tokenizer both sides share

The same dataclass `kg_build/chunks.py` populates. `document` is what BM25
actually scores — verbatim text *plus* the plain-language layer concatenated
— while `verbatim` alone is what an answer is permitted to quote. `tokenize`
stems with Snowball but only touches alphabetic tokens over three
characters, so `"§8(5)"`, `"250"`, and `"2023"` — exactly the tokens a legal
question turns on — always survive intact into the index untouched.

### `backend/retrieval.py` — vocabulary bridge → BM25 → two-hop graph walk

The `Retriever` is built once (`BM25Okapi` over all 142 chunks' `.document`)
and its adjacency map (`_build_adjacency`) is precomputed once too, not
rebuilt per query. Three things happen inside `retrieve()`, in order:

**Step 1 — `expand_query`.** Every phrase in `vocab.yaml` found in the
question (tolerating regular inflections — `"leak"` matches `"leaked"`) adds
its statutory equivalents to the query text before BM25 ever runs. This is
what lets "our vendor leaked customer numbers" reach `"personal data
breach"` without either word appearing in the question. Intent triggers
(`penalty_lookup`, etc.) are collected in the same pass and used to boost
whole chunk kinds or chapters (`INTENT_BOOST = 1.6`).

**Step 2 — BM25 seeding.** The top `k` (default 6) non-zero-scoring chunks
become hop-0 seeds. One deliberate exception: if the `penalty_lookup` intent
fired, **every** Schedule row is forcibly included regardless of score —
ranking seven rows against each other is the wrong problem when the honest
answer is the whole table, and ranking actively fails on entry 5 (the Data
Principal's own ₹10,000 duty), the one row that shares no vocabulary with
"customer" or "personal data".

**Step 3 — `_expand`, the two-hop graph walk.** From the seeds, follow
`EXPAND_PRIORITY`-ordered edges (`PENALISED_BY`/`PENALISES` first,
`REFERENCES`/`CITED_BY` next, `DEFINES`, then `MENTIONS` last) up to two
hops, with each hop's contribution multiplied by `HOP_DECAY = 0.6` so a
second-hop provision is reachable but ranked below a first-hop one.
`MENTIONS` — exhaustive by construction — is barred from the *second* hop
entirely, or it would drag in a large slice of the Act through terms merely
mentioned two edges away. `REVERSIBLE` edges (`PENALISED_BY`, `REFERENCES`)
are also walked *backwards*: a penalty question lands on the Schedule row,
and the useful next hop is *up* to the duty that carries it, the opposite
direction from how the edge is stored. `authority` (from Phase 2b) is used
purely as a **tie-break within one priority tier** — it can never make a
merely-mentioned provision outrank a genuinely cited one; it only decides
which of several equally-ranked candidates is more load-bearing.

`should_abstain` runs against the result, separately, back in `app.py` — see
Stage 2 below.

### `backend/citations.py` — the trust core

`check()` regex-scans the model's answer for every `§N(x)` / `Schedule entry
N` it wrote (`RE_CITATION`), resolves each against `graph.provisions`, and
labels it:

- **`verified`** — exists, and the same node (or a parent/child of it) was
  actually retrieved for this question.
- **`out_of_context`** — exists in the Act, but was **not** retrieved. The
  model recalled it from training rather than reading it in front of it.
- **`unresolved`** — no such provision. If it's structurally close to a real
  one (a model writing `§8(5)(z)` when only `§8(5)` exists), the note names
  the nearest real provision instead of just saying "invented".

`penalty_facts()` is the second half of the trust story: for every retrieved
`Penalty` chunk, the amount is read straight from `Provision.penalty` in the
graph and returned as structured data — **it never passes through the
model's own words**, because a small model has already misread a Schedule
figure in this exact corpus.

### `backend/llm.py` — one interface, two providers

`Provider` is an `ABC` with `check`/`complete`/`stream`. `OllamaProvider`
(default, stdlib `urllib` only, no key needed) and `ClaudeProvider` (the
official `anthropic` SDK, imported lazily inside its methods so a deployment
that never selects it doesn't need the package installed) both implement it
identically. Swapping providers is `DPDP_PROVIDER=claude` in `.env` — no
code change anywhere else in the system. Every provider error is caught and
re-raised as `LLMError` with a short, safe message; the real exception is
logged, never handed to the caller — a provider's own error text can echo
prompt fragments or internal hostnames.

### `backend/prompt.py` — the contract the citation checker depends on

`SYSTEM_PROMPT` is not incidental string data — the instruction *"cite every
provision you rely on, in the form §8(5) or Schedule entry 2"* is what makes
`citations.RE_CITATION` able to find anything at all. Change the format here
without changing that regex and verification silently stops working; this
is why the prompt lives in its own module next to the checker rather than
buried inline in `app.py`.

### `backend/observability.py` — optional Langfuse tracing

Enabled only when both `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are
set (`config.TRACING_ENABLED`) — same shape as `llm.py`'s provider choice:
real behaviour when configured, zero import cost when not (`langfuse` is
imported lazily inside `_get_client()`, mirroring `ClaudeProvider`'s lazy
`import anthropic`). Nothing else in the codebase imports `langfuse`
directly; every call goes through `trace()` / `step()` here.

`trace()` opens the root span for one request; `step(name, as_type=...)`
opens a child observation nested under it. Both are no-ops (`yield None`)
when tracing is off, so call sites are unconditional —
`with observability.trace(...) as t:` — and only need to guard direct use of
the returned object with `if t:`, never a separate `if config.TRACING_ENABLED`
scattered through `app.py`.

The exception handling here is the one subtle part: only a failure to
*create* the Langfuse observation degrades to a no-op (logged as a warning).
Once the caller's `with` block is running, any exception it raises is handed
to the span (so Langfuse records the failure) and then re-raised unchanged —
a broad `try/except` wrapped around the `yield` would otherwise also catch
the *caller's* real exceptions (a `@contextmanager` receives them at the
`yield` point), silently mislabeling an application bug as "tracing
unavailable." Verified against a live span with an unreachable Langfuse host:
the caller's exception still propagated correctly. A failed span export
elsewhere is just a logged warning — tracing is an assurance layer, never a
dependency of the actual answer.

`check()` mirrors `llm.check()`'s contract (`None` if fine, an error string
otherwise) so `/api/health` can report Langfuse reachability the same way it
reports the LLM provider's.

### `backend/app.py` — the HTTP layer and the request lifecycle

`lifespan` (an `asynccontextmanager`, not the older `@app.on_event`) runs
once at process start: `load_graph()`, build the `Retriever`, open the
`AuditLog`. If Neo4j is unreachable or `chunks.json` is missing, the process
fails here — loudly, before accepting a single request — rather than on the
first question someone asks.

A `security_headers` middleware runs on **every** response: CSP
(`connect-src 'self'` specifically blocks a script injected into the page
from exfiltrating an answer to another host), `X-Frame-Options: DENY`,
`X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`. A global
`@app.exception_handler(Exception)` guarantees no unhandled exception ever
reaches a client as a stack trace — it's logged with a request id, and the
client gets that id plus `"internal error"` unless `DPDP_DEBUG=1`.

**`POST /api/chat`** wraps its whole body in `observability.trace(...)`, with
each of the three stages below nested as its own `observability.step(...)` —
`retrieve` (`as_type="retriever"`), `generate` (`"generation"`),
`verify_citations` (`"evaluator"`) — so a trace in the Langfuse UI, the
browser's network tab, and one line in `logs/audit.jsonl` all describe the
same request the same way. Every terminal outcome updates and flushes the
root span before the SSE stream ends, whether tracing is on or a no-op.
Streams over Server-Sent Events in exactly this order:

```
1. retrieval   sent the instant retrieval finishes — the evidence, before
               generation has even started, so the wait is legible
   ↓
   [gate] should_abstain? -> "abstain" event, stop here, no model call spent
   [gate] llm.check() failed? -> "error" event, stop here
   ↓
2. token ×N    answer fragments, as the model produces them
   ↓
3. citations   every citation checked against the graph, plus penalty
               amounts read straight from it
4. done        timings, model, provider, build_id
```

The blocking provider call runs on a worker thread
(`loop.run_in_executor`) and feeds the streaming coroutine through an
`asyncio.Queue`, so the event loop keeps serving other requests — including
the health check — while one answer streams for up to several minutes on a
local model. Every terminal outcome (`no_results`, `abstained`,
`llm_unavailable`, `generation_error`, `answered`) writes exactly one record
to the audit log via `backend/audit.py`, always including `build_id`, so any
answer served by this system points back to the exact snapshot of the Act it
was generated from.

**`GET /api/health`** returns only what `config.public_settings()` allows,
plus provision/chunk counts and the build id — no host, no credential, no
path, ever.

**`GET /api/provision/{node_id}`** is a plain dictionary lookup against
`graph.provisions` — an unknown or hostile `node_id` can only ever miss and
404; there is no filesystem or query path it can reach.

**`GET /`** serves `frontend/index.html` from a fixed `Path`, never a
request-supplied one — directory traversal has no surface here because no
path input exists to traverse with.

---

## `frontend/` — one file, no build step

`index.html` is the entire client: markup, CSS, and the SSE-consuming
JavaScript in one file, fetched by `GET /`. It opens `POST /api/chat`, reads
the SSE stream frame by frame (handling the CRLF line-ending quirk
`sse_starlette` emits), and renders each event as it arrives — the
retrieval trace as soon as it's sent, tokens as they stream in, then the
citation cards (colour- and icon-coded by status) and the penalty table once
the model finishes. Nothing here calls Neo4j or the LLM provider directly;
every fact on the page came from the backend's own verification.

---

## The two flows, end to end

### Build (human-triggered, occasional)

```
data/dpdp_act_2023.pdf
   │  kg_build.extract:  Geometry.derive → extract → parse → find_xrefs → attach_headnotes
   ▼
validated Tree  ──[validate() FAILS → STOP, nothing written]──
   │  kg_build.graph:  build_graph (+ add_authority)
   ▼
networkx MultiDiGraph  (404 nodes, ~1090 edges, each with an `authority` score)
   │                                        │  kg_build.graph: push_to_neo4j
   │  kg_build.chunks: build_chunks          ▼
   │  + attach_plain_language            Neo4j  (:Provision nodes, typed edges)
   ▼
data/chunks.json  (142 chunks: verbatim + header + plain-language layer)
```

### Question (one HTTP request)

```
POST /api/chat {"question": "..."}
   │  backend.retrieval.Retriever.retrieve()
   │    1. expand_query()        vocab.yaml phrase → statutory term
   │    2. BM25 over chunks.json  → top-k seeds (+ full Schedule if penalty intent)
   │    3. _expand()              two-hop walk over Neo4j-loaded Graph.edges,
   │                              PageRank-tie-broken, decayed per hop
   ▼
Result[]  ──SSE "retrieval"──▶ frontend (evidence shown before generation starts)
   │
   │  [gate] should_abstain(results, ABSTAIN_THRESHOLD)?
   │    yes → SSE "abstain", audit "outcome": "abstained", STOP
   ▼
backend.retrieval.build_context()  → prompt = verbatim provisions + question
   │  backend.llm.stream()  (Ollama or Claude, via one Provider interface)
   ▼
answer text  ──SSE "token"×N──▶ frontend (streamed as produced)
   │
   │  backend.citations.check(answer, retrieved_ids, graph)
   │    every §N(x) / Schedule entry N → verified | out_of_context | unresolved
   │  backend.citations.penalty_facts(results, graph)
   │    penalty amounts read from Neo4j directly — never from the model
   ▼
SSE "citations" + "done"  ──▶ frontend renders citation cards + penalty table
   │
   ▼
backend.audit.AuditLog.write()  →  logs/audit.jsonl
   {request_id, question, answer, citations, build_id, elapsed_ms, ...}
```

Every arrow in the second diagram is something a reader can independently
verify: the retrieval trace names which chunks and which graph edges were
walked; the citation stage names, for every claim, whether the Act actually
says it and whether that provision was in front of the model when it wrote
it; the penalty table is read from the same database the citations were
checked against, not from the model's own arithmetic. Nothing in this system
asks to be trusted without also showing its work.
