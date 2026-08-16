# DPDP Act 2023 — Knowledge Graph + RAG

A verbatim knowledge graph of the **Digital Personal Data Protection Act, 2023**
built straight from the Gazette PDF, and a retrieval layer that answers
plain-English questions from people who have never read a statute.

Runs entirely on your machine — **local Ollama for the LLM, local Neo4j for the
graph. No API key, nothing leaves the box.**

```bash
pip install pdfplumber networkx rank_bm25 pyyaml neo4j snowballstemmer
ollama pull qwen2.5:7b-instruct        # 3b works but misreads figures — see below

python build.py --neo4j    # PDF -> graph -> Neo4j at neo4j://127.0.0.1:7687
python index.py --plain    # graph -> 142 chunks + local-LLM question index
python test_build.py       # 15 checks, incl. the retrieval eval set

uvicorn api:app --port 8000        # then open http://127.0.0.1:8000
python ask.py "what is the fine if customer data leaks?"   # or use the CLI
```

Configuration is all environment variables:

| Variable | Default |
|---|---|
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` | `neo4j://127.0.0.1:7687` / `neo4j` / *(set it)* |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` |
| `DPDP_MODEL` | `qwen2.5:3b-instruct` |
| `DPDP_MAX_CONTEXT_CHARS` | `10000` — raise for a bigger model |

---

## The web app

`uvicorn api:app --port 8000` → <http://127.0.0.1:8000>. FastAPI backend,
one static HTML file, no build step.

It streams over SSE in four stages so nothing sits behind a spinner:

| Stage | When | What it shows |
|---|---|---|
| `retrieval` | ~1.5s | which provisions matched, and whether each came from BM25 or the graph |
| `token` | as generated | the answer, word by word |
| `citations` | on completion | every `§` the model wrote, **checked against the graph** |
| `done` | end | model, elapsed time, context size |

### Citations are checked, not just displayed

This is the part that matters. A model can write "§8(5)" whether or not §8(5)
says what it claims, so the backend resolves every citation against the verbatim
graph and labels it:

| Label | Meaning |
|---|---|
| **verified** | the provision exists **and** was in the retrieved context |
| **out of context** | it exists, but was never retrieved — the model recalled it from training. Suspect. |
| **unresolved** | no such provision. The model invented it. |

Each one expands to the Act's own words, so a reader can check the claim against
the text without leaving the page. Tested against a deliberately bad answer:

```
unresolved      §8(5)(z)          no such provision; nearest is §8(5)
unresolved      §99(4)            no such provision in this Act
out_of_context  §12(1)            exists but was not retrieved for this question
verified        §8(5)             "A Data Fiduciary shall protect personal data…"
verified        Schedule entry 2  "Breach in observing the obligation to give the Board…"
```

Penalty amounts are rendered from the graph in a separate table and never pass
through the model at all.

### Endpoints

| | |
|---|---|
| `POST /api/chat` | `{question, k}` → SSE stream |
| `GET /api/provision/{id}` | verbatim text of one provision (`s-8-5`, `pen-2`, `def-child`) |
| `GET /api/health` | model, chunk count, node count |

---

## Three guarantees

**1. Verbatim.** Every `text` field is the Act's exact words. `build.py`
reassembles the entire document from the graph and diffs it against the extracted
body stream; a single character out of place fails the build. LLM-written
plain-English text lives in a separate, clearly-labelled layer that is never
quoted and never cited.

**2. Nothing hand-entered.** Column boundaries, the indentation ladder, the
Schedule table, the section headnotes and the Act's own metadata are all measured
or read from the PDF at runtime. There is no data file to drift out of sync with
the source. The only human input is `overrides.yaml`, which starts empty.

**3. No silent conventions.** Three typesetting conventions cannot be derived
from geometry. Each is resolved, then written to `review/` for sign-off:

| Convention | Where to check |
|---|---|
| A headnote belongs to the section it is typeset level with | `review/headnotes.md` — all 44 currently align at **0.0pt** |
| A bare `sub-section (5)` means the section it sits in | `review/crossrefs.md` — every edge, with its rule |
| An illustration explains the nearest section or sub-section | `review/illustrations.md` — shows both candidates |

---

## Why the PDF looked unparseable

It isn't irregular. `extract_text()` was merging three separate columns into one
stream and dumping the margin notes at the end of each page.

```
x0 < 109                109 ≤ x0, x1 ≤ 486               x1 > 486
┌──────────────┐  ┌────────────────────────────────┐  ┌──────────────┐
│  marginalia  │  │           BODY TEXT            │  │  marginalia  │
│ (even pages) │  │                                │  │ (odd pages)  │
│ "Definitions"│  │  2. In this Act, unless the    │  │ "Consent."   │
│ "45 of 1860."│  │  context otherwise requires,—  │  │ "24 of 1997."│
└──────────────┘  └────────────────────────────────┘  └──────────────┘
```

Both divides are found by gap analysis on the x-histograms (16pt and 14pt gaps),
so the numbers above are measured, not typed.

**The indentation is the hierarchy.** Body lines snap to a ladder at
`117.4 + 24.1k`, derived at runtime from the marker-line histogram:

| Rung | Opens |
|---|---|
| 117.4 | continuation of the line above |
| 141.5 | section / sub-section — `4.`, `(1)` |
| 165.6 | clause — `(a)`, `(za)` |
| 189.7 | sub-clause — `(i)`, `(vii)` |
| 213.8 | item — `(A)`, `(B)` |

This is what resolves `(i)` — letter-i clause or roman numeral one? Geometry
answers it where the rungs differ; where both are legal on the same rung, the
sibling rule decides (a clause list opens at `(a)`, a sub-clause list at `(i)`,
and a list never changes type part-way). Section 2 has **both** — clause `(i)`
"Data Fiduciary" *and* sub-clauses `(i)`/`(ii)` under clause `(j)` — and all 28
clauses parse correctly.

**The marginalia is not noise — it is the section headnotes.** "Definitions.",
"Grounds for processing personal data without consent.", "Power to amend
Schedule." Forty-four authoritative one-line summaries, which most pipelines
discard. They are attached to their sections and embedded for search.

---

## What is in the graph

404 nodes, 1088 edges.

| Node | Count | | Edge | Meaning |
|---|---:|---|---|---|
| Section | 44 | | `HAS_*` | structural containment |
| SubSection | 117 | | `REFERENCES` | one provision cites another |
| Clause | 137 | | `DEFINES` | §2 clause → defined term |
| SubClause | 33 | | `MENTIONS` | any provision → a defined term it uses |
| Definition | 28 | | `PENALISED_BY` | duty → Schedule entry |
| Illustration | 11 | | `HAS_ENTRY` | Schedule → its 7 rows |
| Penalty | 7 | | | |

**Section 2 hands you the ontology.** The Act defines all 28 of its own terms,
so the entity layer is extracted by regex — no model, no hallucination surface,
100% recall on `MENTIONS`.

### The edge that justifies the graph

```
§8(5)  "reasonable security safeguards"      ← contains no rupee figure
   │ PENALISED_BY
   ▼
Schedule entry 1  "up to two hundred and fifty crore rupees"
                                              ← contains no word about security
```

Ask a flat vector store *"what's the fine if customer data leaks?"* and it
retrieves the Schedule rows and stops. Here, retrieval lands on the Schedule and
the graph walks **back** along `PENALISED_BY` to the duties:

```
   [0] Schedule entry 2                    (bm25 38.6)
   [0] Schedule entry 1                    (bm25 36.3)
   [0] Definition of "personal data breach"
   [1] Section 8(6)   ← Schedule entry 2 —PENALISES→
   [1] Section 8(5)   ← Schedule entry 1 —PENALISES→
```

The join lives in the citation, not in the semantics. `test_penalty_join`
locks it in.

Section 44 amends four other statutes, so `section 81` there means the IT Act,
not this one. Nine such citations are classified external and deliberately **not**
linked — a wrong internal link would be a confident wrong answer.

---

## Seeing the graph

Three exports from the same build:

`python build.py --neo4j` loads it directly through the driver — batched with
`UNWIND`, and only **after** validation passes, so a graph that failed the
round-trip check never reaches a database anyone will query. Each node carries
its kind as a second label, which is what makes the Browser legend useful.

Open <http://localhost:7474> and try:

```cypher
// every duty that carries a penalty, with the amount
MATCH (s)-[:PENALISED_BY]->(p) RETURN s.id, s.headnote, p.penalty ORDER BY s.id;

// everything section 17(1) switches off
MATCH (:Provision {id:'s-17-1'})-[:REFERENCES]->(t) RETURN t.id, t.text;

// the obligations chapter, three levels deep — the picture worth looking at
MATCH p=(:Chapter {id:'ch-II'})-[:HAS_SECTION|HAS_SUBSECTION|HAS_CLAUSE*1..3]->()
RETURN p LIMIT 200;
```

`out/load.cypher` is still written for `cypher-shell` if you prefer that route.

- `out/graph.html` — standalone viewer, no server. Double-click it. Filter by
  edge type, search node text.
- `out/dpdp.gexf` — open in Gephi for layout and community detection.

---

## Answering non-lawyers

Real users ask *"can I text my customers an offer?"* The Act never says
"customer", "text", or "offer". Three layers close that gap:

**1. Vocabulary bridge** (`vocab.yaml`, ~150 entries, no model).
`customer → Data Principal`, `leak/hack/stole → personal data breach`,
`delete my data → erasure`. Deterministic and auditable — a wrong hit traces to
one line and is fixed without touching code. Matching tolerates `-s/-ed/-ing`.

**2. Plain-language index layer** (`python index.py --plain`).
For each of the 142 chunks, the local model writes a plain-English gloss and ~8
questions a layperson would actually type — **1116 questions, generated once**
(~3h on a 3B CPU model, resumable, temperature 0 so it is reproducible,
cached to `plain_language.json`). Those go
into the search document, so a user's own words match *a question* rather than
having to match legalese. It is why "someone died…" now reaches §14: the
generated question says *"what happens after the customer dies"*.

**2b. The Schedule is answered exhaustively, not ranked.** There are seven
penalty rows; when a question is a penalty lookup, all seven go into the
context. Ranking them was actively wrong — entry 5 (the customer's *own* duties,
₹10,000) is the only row that never says "Data Principal", so expanding
"customer" into that term pushed the one row about the customer to last place.

**2a. Stemming.** Tokens are indexed under both surface form and Snowball stem,
so "died" reaches "dies". Anything containing a digit is left untouched — `8(5)`,
`250` and `2023` are exactly the tokens a legal question turns on.

**3. Answer format built for someone with no legal training** — short answer,
why, the verbatim quotes with citations, what to do, and the penalty.

> The plain layer never reaches the user as law. Retrieval uses it; answers quote
> the verbatim text.

### Chunking

142 chunks on the Act's own boundaries, never by token count — a fixed window
would cut clause lists in half and discard the structure the parser just
recovered. Sections over 220 words also get per-sub-section chunks, because §8
and §17 each cover several unrelated duties.

Every chunk carries a header the body never states:

```
[Chapter II — OBLIGATIONS OF DATA FIDUCIARY]
Section 8(5) — Data Fiduciary shall protect personal data in its possession...
Penalised by: Schedule entry 1
```

---

## Files

```
build.py       PDF → tree → graph → out/ + review/   (no API key)
index.py       graph → 142 chunks → BM25 index; --plain adds the LLM layer
ask.py         question → vocab → BM25 → graph expansion → cited answer (CLI)
api.py         FastAPI: SSE streaming + citation checking
web/index.html the chat UI — one file, no build step
llm.py         Ollama client (stdlib HTTP), streaming and structured output
test_build.py  14 checks; run after every build
vocab.yaml     the layperson→statute dictionary — grow this as questions miss
overrides.yaml human corrections after reading review/ (starts empty)
plain_language.json  the generated question index — committed, NOT under out/,
               because it costs ~3h of local inference to rebuild
out/           disposable: dpdp_tree.json, dpdp_graph.json, schedule.json,
               index.json, load.cypher, graph.html, dpdp.gexf
review/        headnotes, crossrefs, illustrations, schedule, geometry, roundtrip
```

`ANTHROPIC_API_KEY` is needed only for `index.py --plain` and the answer step of
`ask.py`. Everything else — build, graph, exports, retrieval, tests — runs offline.

---

## Known limits

- **Retrieval is BM25 only.** With the vocabulary bridge and the question index
  this is strong on a 142-chunk corpus, and it never blurs section numbers or
  rupee figures the way embeddings do. Add a dense retriever and fuse if recall
  measurably falls short — the chunk format is already right for it.
- **`eval.yaml` scores 25/26**, with the floor enforced by `test_build.py`, so
  tuning `vocab.yaml` can no longer silently trade one question's accuracy for
  another's. It has already earned its keep three times: it caught two wrong
  expectations of mine, a missing-stemmer bug, and a ±2 score swing between
  plain-layer regenerations (now fixed with temperature 0). Grow it whenever a real question misses. The one known miss —
  *"what happens if we ignore a Board order?"* — reaches §29 (appeals) instead
  of §33 (penalties); it is recorded, not hidden.
- **The local model is the weak link, not retrieval.** On `qwen2.5:3b-instruct`
  answers are directionally right but lose nuance — it read §14 as "family can
  step in" when the section only grants rights to a *nominated* person, and it
  misquoted a Schedule figure until penalties were pulled from the graph. Use a
  7B+ model for anything user-facing.
- **This Act only.** DPDP Rules 2025 and later amendments change the operative
  picture. The graph carries `act_number` and `assent_date` so a second document
  can be layered in without a rebuild.
- **Not legal advice.** It quotes and cites the Act; it does not apply it.
