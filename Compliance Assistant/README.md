# DPDP Compliance Assistant

Answers plain-English questions about India's **Digital Personal Data
Protection Act, 2023**, quoting the statute verbatim and proving every
citation against a knowledge graph.

```
kg_build/     PDF  ->  validated tree  ->  graph  ->  Neo4j + chunks.json
backend/      Neo4j + chunks  ->  retrieval  ->  model  ->  checked answer
frontend/     one static page, no build step
```

---

## The two properties everything rests on

**1. The Act's words are never altered.** `kg_build` reassembles the entire
document from the parsed tree and diffs it character-for-character against the
raw extracted text. A build that fails that check writes nothing and publishes
nothing. Generated text (the plain-language layer) lives in separate fields,
is used only to *find* the right provision, and is never quoted or cited.

**2. The graph is authoritative for structured facts.** Penalty amounts are
read from Neo4j and rendered directly — never routed through the model, which
has misread a Schedule figure in this corpus. Every citation the model writes
is resolved against the graph and labelled:

| Status | Meaning |
|---|---|
| `verified` | the provision exists **and** was in the retrieved context |
| `out_of_context` | it exists but was **not** retrieved — recalled, not read |
| `unresolved` | no such provision. The model invented it. |

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # fill in Neo4j credentials
```

Then build the corpus and publish it:

```bash
python -m kg_build            # parse, validate, write data/chunks.json
python -m kg_build --neo4j    # ... and replace the graph in Neo4j
```

Run the service (it serves the frontend too, at `http://127.0.0.1:8000`):

```bash
uvicorn backend.app:app --port 8000
```

A local model via [Ollama](https://ollama.com) is the default and needs no
key: `ollama pull qwen2.5:3b-instruct`. Set `DPDP_PROVIDER=claude` with
`ANTHROPIC_API_KEY` for materially better answers.

Tracing to [Langfuse](https://langfuse.com) is optional and off by default.
Set both `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` in `.env` to turn it
on — nothing else changes. Each request becomes one trace with three nested
observations (`retrieve`, `generate`, `verify_citations`), the same three
stages already streamed over SSE and written to `logs/audit.jsonl`. A
misconfigured or unreachable Langfuse host degrades to a logged warning, never
a failed answer.

---

## Layout

| Path | Role |
|---|---|
| `kg_build/extract.py` | PDF → tree. Column geometry, indent ladder, round-trip validation. |
| `kg_build/graph.py` | Tree → networkx graph, PageRank, push to Neo4j. |
| `kg_build/chunks.py` | Graph → 142 searchable chunks on the Act's own boundaries. |
| `backend/graph_store.py` | Neo4j repository. The runtime source of truth. |
| `backend/retrieval.py` | Vocabulary bridge → BM25 → two-hop graph expansion. |
| `backend/citations.py` | Citation verification and graph-sourced penalties. |
| `backend/llm.py` | Pluggable provider (Ollama / Claude). |
| `backend/observability.py` | Optional Langfuse tracing, no-op unless both keys are set. |
| `backend/app.py` | FastAPI, SSE streaming, security headers. |
| `data/vocab.yaml` | Layperson → statutory vocabulary. Hand-maintained, auditable. |

`networkx` appears **only** in `kg_build`. The serving path talks to Neo4j.

---

## Retrieval, and why it is not a vector store

BM25 over 142 chunks. In law the exact token *is* the answer — `§8(5)`,
`250 crore`, `Significant Data Fiduciary` — and embeddings blur precisely the
distinctions that decide a question. The gap they would close (a layperson's
words sharing no root with the statute's) is closed instead by two cheaper,
auditable mechanisms: `data/vocab.yaml`, and a generated plain-language layer
attached to each chunk at build time.

After retrieval, up to **two hops** of graph expansion pull in what keyword
matching alone would miss. This is the part a flat index cannot do: *"what's
the penalty for failing to notify a breach?"* needs §8(6), which contains no
rupee figure, joined to the Schedule row, which says nothing about
notification. That join is an edge, not a similarity.

Questions the corpus plainly does not cover are refused **before** a
generation call, on a deterministic score threshold rather than the model's
own judgement of its competence.

---

## Security

| Concern | Handling |
|---|---|
| Credentials | Read only in `config.py`. `public_settings()` is a hand-written allow-list — no config value becomes public by being added. |
| Startup | Missing Neo4j credentials abort the process, rather than surfacing as a 500 on the first question. |
| Error responses | Internal detail is logged with a request id; clients get the id and a generic message unless `DPDP_DEBUG=1`. |
| Input | Bounded by Pydantic (2–800 chars) before any work; control characters stripped so they cannot corrupt SSE framing or the audit log. |
| Cypher | All values parameterised. Labels and relationship types (which Cypher cannot parameterise) are validated against a strict identifier pattern before interpolation. |
| Path traversal | The frontend is one fixed file. No filesystem path is ever taken from a request. |
| CORS | Same-origin by default. `*` is refused outright, not silently honoured. |
| Headers | CSP (`connect-src 'self'` blocks exfiltration of answers), `X-Frame-Options: DENY`, `nosniff`, `no-referrer`. |
| API surface | `/docs` and `/openapi.json` are disabled unless `DPDP_DEBUG=1`. |
| Audit log | Append-only, `0600` on POSIX, outside `data/` so a rebuild cannot delete it. |

There is **no authentication** — the API is open to anyone who can reach the
port. That is unchanged from the previous version and is the main gap before
this faces untrusted users.

---

## Not legal advice

Answers are informational. Verify every citation against the Act before
relying on it. A small local model is the default and is the weakest component
by a distance — it has misread both a penalty figure and the scope of §14 in
this corpus.
