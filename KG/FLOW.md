# FLOW — how the whole solution works

End-to-end walkthrough of the DPDP Act assistant: what happens to the PDF, how
the graph is built, how retrieval works, and — the question worth answering
directly up front — **why there are no embeddings anywhere in this system.**

Every number below is measured from the current build, not estimated.

---

## The pipeline at a glance

```
 dpdp_act_2023.pdf   21 pages, 56,768 characters of operative text
        │
        │  build.py ── stage 1   geometry: split 3 columns, drop headers
        ▼
 873 body lines + 152 margin fragments + 1 table
        │
        │  build.py ── stage 2   parse the indentation ladder
        ▼
 tree: 404 nodes (Act → Chapter → Section → SubSection → Clause → …)
        │
        │  build.py ── stage 3   derive relationships
        ▼
 graph: 404 nodes, 1,088 edges  ──►  Neo4j / GEXF / HTML / JSON
        │
        │  index.py ── stage 4   chunk on the Act's own boundaries
        ▼
 142 chunks  +  stage 5: 1,116 generated layperson questions
        │
        │  index.py ── stage 6   BM25 index (no embeddings)
        ▼
 searchable corpus
        │
        │  ask.py / api.py ── stages 7-9   retrieve → expand → answer → verify
        ▼
 cited answer + citation audit + graph-sourced penalty table
```

Two rules hold across every stage:

1. **Verbatim.** The Act's words are never paraphrased, summarised or
   regenerated anywhere in the storage path. A round-trip check reassembles the
   whole document from the graph and diffs it against the extracted text; one
   character out and the build fails.
2. **Model output is never authoritative.** Anything an LLM writes is a
   retrieval aid or a presentation layer. Quotes, citations and penalty amounts
   all come from the graph.

---

## Stage 1 — PDF to clean text

### The actual problem

The naive read of this PDF looks unparseable, and the reason is not the Act — it
is that the Gazette sets **three columns per page** and `extract_text()` merges
them into one stream, dumping the margin notes at the end of each page:

```
...personal data breach under sub-section (5) of section 8.
Definitions.                 ← a margin note, now stranded mid-document
24 of 1997.                  ← a citation from a completely different page region
```

The columns never overlap horizontally:

```
 x0 < 109              109 ≤ x0 , x1 ≤ 486            x1 > 486
┌──────────────┐  ┌────────────────────────────────┐  ┌──────────────┐
│  marginalia  │  │           BODY TEXT            │  │  marginalia  │
│ (even pages) │  │                                │  │ (odd pages)  │
│ "Definitions"│  │  2. In this Act, unless the    │  │ "Consent."   │
│ "45 of 1860."│  │  context otherwise requires,—  │  │ "24 of 1997."│
└──────────────┘  └────────────────────────────────┘  └──────────────┘
                  top < 85 = running header, discarded
```

### How the boundaries are found

`pdfplumber` gives a box per word. Histogram every word's `x0` and `x1`, then
look for the widest gap next to the densest cluster:

| Boundary | Method | Value |
|---|---|---|
| body left | largest gap in the `x0` histogram left of the mode | **109.0** (16pt gap) |
| body right | first gap in the `x1` histogram right of the mode | **486.0** (14pt gap) |
| line grouping | 40% of modal baseline spacing (12pt) | **4.8pt** |

**Nothing is hard-coded.** Run it on a different Gazette PDF and it re-derives
its own numbers. The tolerance exists because the Gazette drops small-caps
fragments ~3pt below their baseline — without it, the `OF` in
`OBLIGATIONS OF DATA FIDUCIARY` becomes its own line.

Three regions are removed and accounted for:

- **19 running headers** — matched on content (`THE GAZETTE OF INDIA
  EXTRAORDINARY`), not on a y-coordinate.
- **18 lines of page-1 masthead** — the bilingual gazette banner. The boundary
  is the Act's own title line, not a page number.
- **34 lines on the Schedule page** — that page is a table, parsed separately.

### The marginalia is not noise

This is the highest-value thing in the file and most pipelines discard it. Those
margin notes are the **section headnotes** — "Definitions.", "Grounds for
processing personal data without consent.", "Power to amend Schedule." Forty-four
authoritative one-line summaries written by the drafters. They are reattached to
their sections and indexed for search.

Two complications, both handled: headnotes wrap across up to five fragments and
must be rejoined by vertical proximity, and act citations (`45 of 1860.`)
sometimes **share a baseline** with headnote text — §25 is set as
`"Members and officers to be public"` / `"45 of 1860. servants."` — so citations
are stripped per-fragment, not per-block.

### Why this approach

| Alternative | Why not |
|---|---|
| **Manual bounding boxes** (the original instinct) | ~100 boxes by hand, hours of work, must be redone for every document, and it still gives you *regions* — not the parent-child structure you actually need |
| **OCR** | The PDF has a clean text layer. OCR would *introduce* errors into a document where a wrong digit is a wrong legal answer |
| **LLM extraction** | Non-deterministic, unauditable, and it would paraphrase a statute. Structure recovery here is a solved geometry problem |
| **`pdftotext -layout`** | Preserves visual layout as spaces; you then re-parse columns from whitespace, which is strictly harder than reading the coordinates that are already there |

### Better approach?

For this document, no — the geometry is exact and free. Where this would need
replacing: a **scanned** statute (needs OCR, and then a layout model like
LayoutLMv3 or Surya to recover the columns), or a document with genuinely
irregular layout such as multi-column tables mid-flow. For a portfolio of Indian
Gazette PDFs, this same code should generalise with no changes.

---

## Stage 2 — Text to hierarchy

### The indentation *is* the structure

Body lines snap to a ladder, derived at runtime by histogramming the left edge
of every line that begins with a marker token:

| Rung | Opens | Marker |
|---|---|---|
| 117.4 | continuation of the line above | *(none)* |
| 141.5 | section / sub-section | `4.` `(1)` |
| 165.6 | clause | `(a)` `(za)` |
| 189.7 | sub-clause | `(i)` `(vii)` |
| 213.8 | item | `(A)` `(B)` |

Base 117.4, step 24.1, derived from the peak spacing. Justified text drifts a
few points; rungs are 24pt apart, so snapping is never ambiguous.

Note the **hanging indent**: a node's first line sits one rung deeper than its
continuation lines. So a line only opens a new node if it *also* starts with a
marker token — otherwise it is continuation text appended to whatever is open.

### The `(i)` problem

This is the single hardest thing in the parse, and it is why geometry matters.

`(i)` is both **letter-i** (a clause) and **roman numeral one** (a sub-clause).
Section 2 contains *both*: clause `(i)` defines "Data Fiduciary", while clauses
`(j)`, `(o)`, `(p)`, `(s)` each contain sub-clauses `(i)`, `(ii)`, …

- **Where the rungs differ**, geometry settles it: a clause at 165.6 and a
  sub-clause at 189.7 are simply different tokens.
- **Where both are legal on the same rung** (§5(1) sets its sub-clauses at
  clause depth), a sequence rule settles it: *a clause list always opens at
  `(a)`, a sub-clause list always opens at `(i)`, and a list never changes type
  part-way.* So the first marker under a parent decides, and the rest inherit.

A regex-only parser gets this wrong. Early in development mine did: `(i)` matched
the clause pattern first and returned, silently swallowing **26 sub-clauses** as
continuation text. The round-trip check did not catch it — the text was all still
there, just in the wrong node — which is exactly why `test_sub_clauses_survive`
exists.

### Other real-world quirks handled

| Quirk | Handling |
|---|---|
| `13. (1)A Data Principal…` — no space after the marker | markers use `\s*` not `\s` |
| `6. (1) The consent…` — section and sub-section on one line | split into two nodes, both verbatim |
| `Illustration.` / `Illustrations.` — centred, no rung | detected by content; attached to the nearest **section or sub-section** (a provision), with the deeper node recorded as `follows` |
| Schedule table | column gutters found by coverage analysis; bounded by the `(1) (2) (3)` header row and the `————` rule |

### The guarantee

```
rebuilt = " ".join(node.prefix + node.text for node in document_order)
assert normalise(rebuilt) == normalise(extracted_body_stream)
```

If a single character was invented, dropped or reordered, the build fails. This
is what makes "verbatim" a property rather than a promise.

### Why this approach / alternatives

| Alternative | Why not |
|---|---|
| Pure regex on text | Cannot resolve `(i)`; no depth signal at all |
| Indentation from leading spaces | `pdftotext` space counts are unstable; coordinates are exact |
| LLM structure extraction | Non-deterministic, no round-trip guarantee, ~$0 benefit here |
| Existing legal parsers (Akoma Ntoso / LegalDocML) | Right long-term target as an *output format*. As an *input* parser, none handle Indian Gazette layout, and you would still write this stage |

**Better approach:** emit **Akoma Ntoso XML** as an additional export. It is the
international standard for legislation, and it would make this graph
interoperable with other legal tooling. Purely additive — the parse stays.

---

## Stage 3 — Hierarchy to graph

### Nodes (404)

| Kind | Count | Kind | Count |
|---|---:|---|---:|
| Clause | 137 | Illustration | 11 |
| SubSection | 117 | ExternalAct | 9 |
| Section | 44 | Chapter | 9 |
| SubClause | 33 | Penalty | 7 |
| Definition | 28 | IllustrationItem | 4 |
| Item | 2 | Act / Preamble / Schedule | 3 |

### Edges (1,088)

| Edge | Count | How it is derived | Confidence |
|---|---:|---|---|
| `MENTIONS` | 605 | exact word-boundary match of a defined term in any provision | exact |
| `HAS_*` | 224 | parent-child from the indentation parse | exact |
| `REFERENCES` | 84 | regex over verbatim citation text | 58 exact, 26 by convention |
| `DEFINES` | 28 | §2 clause → the term it defines | exact |
| `HAS_ENTRY` | 7 | Schedule → its rows | exact |
| `PENALISED_BY` | 6 | the Schedule row's own text names the duty | exact |

### 3a. Structural edges — free

Falls out of the parse. `HAS_CHAPTER`, `HAS_SECTION`, `HAS_SUBSECTION`,
`HAS_CLAUSE`, `HAS_SUBCLAUSE`, `HAS_ITEM`, `HAS_ILLUSTRATION`.

### 3b. `DEFINES` and `MENTIONS` — the Act hands you its own ontology

Most knowledge-graph projects spend their effort *inventing* an ontology and then
running NER to populate it. **Section 2 of this Act defines all 28 of its own
terms** — Data Principal, Data Fiduciary, Consent Manager, personal data breach,
processing, child, and so on.

So the entity layer is extracted with a regex over §2:

```
"Data Fiduciary" means any person who alone or in conjunction with…
 └── term ──┘   └ relation ┘
```

Then every one of those 28 terms is string-matched across all 404 nodes to build
605 `MENTIONS` edges. **Deterministic, exhaustive, and with zero hallucination
surface** — no model is involved and recall is 100% by construction. An
LLM-based entity extractor would be slower, more expensive, and *less* accurate
here.

`"processing" in relation to personal data, means…` — the qualifier between term
and verb — is handled by taking the first quoted phrase as the term and the first
`means|includes` after it as the relation.

### 3c. `REFERENCES` — the edges between sections

**This is the most valuable edge type and the hardest to get right.** 56 of the
84 references cross a section boundary:

```
s-2-c    -> s-18       'section 18'
s-2-l    -> s-10-2-a   'clause (a) of sub-section (2) of section 10'
s-5-1    -> s-6        'section 6'
s-17-1   -> s-8-5      'sub-sections (1) and (5) of section 8'
```

The Act cites in three shapes, and one pattern handles all of them:

| Shape | Example | Resolves to |
|---|---|---|
| chained | `clause (a) of sub-section (2) of section 10` | `s-10-2-a` |
| list | `sub-sections (1) and (5) of section 8` | `s-8-1`, `s-8-5` |
| bare | `sub-section (5)` *(no section named)* | the enclosing section |

The bare form relies on a **drafting convention** — a bare `sub-section (N)`
means the section it sits in. That is the single riskiest inference in the build,
so all 26 such edges are written to `review/crossrefs.md` labelled
`relative-to-own-section` for sign-off, and any one of them can be removed via
`overrides.yaml` without touching code.

### Citations that point *out* of the Act

§44 amends four other statutes. Naively, `in section 81` there would link to
DPDP §81 — except there is no §81, and the reference actually means the
Information Technology Act, 2000. Two guards:

1. A citation followed by `of the <Name> Act, <year>` is external.
2. A section number **greater than the Act's own highest section** (44) is
   external. Derived from the parsed tree, not hard-coded.

Nine such citations are classified external and deliberately **not** linked. When
the statute name is quoted rather than named — §44(2)(b) quotes "the Patents Act,
1970" while amending the IT Act — quoted spans are stripped before looking for
the statute being amended.

**A wrong internal link would be a confidently-wrong answer with a real-looking
citation.** That is a worse failure than no link at all.

### 3d. `PENALISED_BY` — the join that justifies the graph

```
§8(5)  "…taking reasonable security safeguards to prevent personal data breach."
   │                                          ← contains no rupee figure
   │ PENALISED_BY
   ▼
Schedule entry 1  "May extend to two hundred and fifty crore rupees."
                                              ← contains no word about security
```

Six of these edges, each derived from the Schedule row's *own* text (`…under
sub-section (5) of section 8`) using the same reference scanner. Entry 7 is the
residual catch-all and correctly gets no edge.

**Why this matters:** ask any flat retrieval system *"what's the fine if customer
data leaks?"* and it returns the Schedule rows and stops — none of them explain
what the duty was. Or it returns §8(5) and stops — which never states an amount.
The connection exists only in the citation, not in the semantics. No embedding
model can infer it, because the two texts share almost no vocabulary.

Locked down by `test_penalty_join`.

---

## Stage 4 — Chunking

142 chunks, built on the **Act's own boundaries**, never by token count.

| Chunk type | Count | Rule |
|---|---:|---|
| Section | 44 | one per section, with all children rendered in hierarchy |
| SubSection | 63 | added for sections over 220 words |
| Definition | 28 | one per §2 defined term |
| Penalty | 7 | one per Schedule row |

Chunk sizes: median **48 words**, mean 107, max 818.

### Why not fixed-size chunks

A 512-token sliding window would cut `(a)/(b)/(c)` clause lists in half, split a
sub-section from the sub-section it depends on, and discard the entire hierarchy
that stage 2 just spent all that effort recovering. Legal text is *already*
chunked — by its drafters, into numbered provisions that are designed to be
citable in isolation. Fighting that is strictly worse.

### Dual granularity

§8 and §17 each carry several unrelated duties. A whole-section chunk retrieves
the wrong one for a specific question, so long sections get **both** a
whole-section chunk and per-sub-section chunks. `§8(5)` competes on its own
merits for a security question; `§8` wins for "what are our general obligations".

### Every chunk carries context its body never states

```
[Chapter II — OBLIGATIONS OF DATA FIDUCIARY]
Section 8(5) — General obligations of Data Fiduciary
Penalised by: Schedule entry 1
---
(5) A Data Fiduciary shall protect personal data in its possession…
```

The chapter, the headnote, the citations and the penalty link are all injected
into the chunk. That header is searchable *and* it reaches the model's prompt —
so a chunk knows what it is even when read in isolation.

### Better approach?

For a single statute, this is right. Two upgrades worth considering:

1. **Parent-document retrieval** — search the small sub-section chunks, but pass
   the whole section to the model. Better precision with full context. This
   codebase is one function away from it (`chunk_for` already walks to parents).
2. **Contextual retrieval** (Anthropic's variant) — prepend an LLM-written
   situating sentence to each chunk before indexing. Reported ~35% retrieval-
   failure reduction. **The plain-language layer in stage 5 is essentially this**,
   done with questions instead of statements.

---

## Stage 5 — The plain-language layer

**The problem:** users are not lawyers. They type *"can I text my customers an
offer?"* The Act says *"processing"*, *"Data Principal"*, *"specified purpose"*.
There is often **zero lexical overlap** between the question and the answer.

**The fix:** for each of the 142 chunks, a local model writes a plain-English
gloss and ~8 questions a layperson would actually type. **1,116 questions,
generated once**, indexed alongside the verbatim text.

The user's words now match *a question* rather than having to match legalese.
This is why "someone died, can their family get their account information?"
reaches §14 — the generated question says *"what happens after the customer
dies"*.

### Three properties that make this safe

1. **Never citable.** The plain layer exists only in the search document. Answers
   quote the verbatim field. A generated gloss can never become the quoted law.
2. **Temperature 0.** Two runs at 0.3 produced different question sets that
   scored **two eval points apart**. A retrieval index must be reproducible.
3. **Committed, not regenerated.** It costs ~3h of local CPU inference, so it
   lives at the repo root outside the disposable `out/` directory. *(I learned
   this by deleting it with `rm -rf out`.)*

### Better approach?

This is the standard technique (variously "HyDE-at-index-time", "doc2query",
"synthetic query generation") and it is well-founded. The real upgrade is a
**bigger generator model** — 7B+ writes more natural and more varied questions
than 3B — and the cost is one-time.

---

## Stage 6 — Retrieval: the embeddings question

### Straight answer: there are no embeddings in this system

**No vectors. No cosine similarity. No ANN index. No vector database.**

Retrieval is **Okapi BM25** — a lexical, probabilistic ranking function
(`rank_bm25`, ~200 lines of pure Python).

### What BM25 actually does

For each query term, score every document by three factors:

```
score(D,Q) = Σ  IDF(qᵢ) ·      f(qᵢ,D) · (k₁+1)
            qᵢ∈Q          ─────────────────────────────────────
                          f(qᵢ,D) + k₁·(1 − b + b·|D|/avgdl)
```

| Factor | Effect |
|---|---|
| **IDF** | rare terms count for more — "Fiduciary" outweighs "the" |
| **TF saturation** (`k₁`) | the 10th occurrence of a word adds far less than the 2nd |
| **Length normalisation** (`b`) | a short definition isn't penalised against a long section |

It matches **words**, not meaning.

### Why not embeddings — six reasons, in order of weight

**1. Exact tokens are the whole game in law.** A legal question turns on
`§8(5)`, `250 crore`, `Significant Data Fiduciary`, `eighteen years`. Embeddings
are *designed* to blur near-neighbours: `§8(5)` and `§8(6)` land in almost the
same place in vector space, and they are different duties with different
penalties. BM25 treats them as different tokens, which is correct.

**2. The corpus is 142 chunks.** Dense retrieval's advantage grows with corpus
size and ambiguity. At this scale BM25 scans everything in **~900ms** including
building the index from scratch each query. There is no recall problem to solve.

**3. The vocabulary gap is closed more cheaply elsewhere.** The usual argument
for embeddings is "the user says *leak*, the doc says *personal data breach*".
That is fixed here by two deterministic layers — the vocabulary bridge (stage 6a)
and 1,116 generated questions (stage 5) — both of which are *auditable*. A bad
match traces to one line in a YAML file. A bad cosine score traces to nothing.

**4. Auditability.** In a compliance setting the question "why did it show me
this?" must have an answer. "The query contained *safeguards*, which appears 3
times in §8(5), and *safeguards* is rare across the corpus" is an answer. "The
vectors were 0.83 apart" is not.

**5. Zero infrastructure.** No embedding model to download, no GPU, no vector
store, no dimension/version drift between index and query time. The whole system
runs on a laptop with Ollama and 200KB of Python dependencies.

**6. The hard cases here are *structural*, not semantic.** "What's the penalty
for a security failure?" is not a similarity problem — it is a **join** across
two documents that share no vocabulary. The graph solves it. A better embedding
model would not.

### What we do instead of embeddings

**6a. Vocabulary bridge** — ~170 entries in `vocab.yaml`, no model:

```yaml
customer:  [Data Principal, personal data]
leak:      [personal data breach]
hacker:    [personal data breach]
kid:       [child, children]
fine:      [penalty, Schedule]
outdated:  [correction, right to correction]
```

Matched with inflection tolerance (`-s/-es/-ed/-ing`), longest phrase first.
Deterministic, auditable, and instantly fixable when a real question misses.

**6b. Snowball stemming** — every token is indexed under both its surface form
and its stem, so *died* reaches *dies*. **Tokens containing digits are never
stemmed** — `8(5)`, `250` and `2023` must survive intact.

> This one was a real bug. My first hand-rolled stemmer refused to touch *died*
> and *dies* (both 4 letters — any rule safe enough to protect *data* also
> protected them), so §14 was never retrieved for a question about death.
> Snowball maps both to *die*. §14 went from absent to top hit at 55.5.

**6c. Intent boosts** — five intent categories detected by trigger phrases, each
boosting a kind or chapter by 1.6×:

| Intent | Boosts |
|---|---|
| `penalty_lookup` | Penalty + Schedule chunks |
| `rights_lookup` | Chapter III |
| `obligation_lookup` | Chapter II |
| `definition_lookup` | Definition chunks |
| `exemption_lookup` | Chapter V |

**6d. Exhaustive Schedule.** There are only **seven** penalty rows. When the
question is a penalty lookup, all seven go into the context — ranking them
against each other was actively harmful. Entry 5 (the customer's *own* duties,
₹10,000) is the only row that never says "Data Principal", so expanding
"customer" into that term pushed **the one row about the customer to last place.**
Completeness beats a cleverer score when the table is seven rows long.

### When embeddings *would* win, and exactly how to add them

Be clear about the limits. BM25 fails when the user's words share **no**
morphological root with the target and the vocabulary bridge has no entry:

- *"our SaaS vendor stores data in **Singapore**"* → should reach §16 (processing
  outside India). "Singapore" appears nowhere in the Act, and a country list is
  not a scalable fix. **An embedding would probably get this.**
- The one recorded eval miss: *"what happens if we **ignore a Board order**?"* →
  reaches §29 (appeals) instead of §33 (penalties).

Add them when: the corpus grows past ~1,000 chunks (adding the DPDP Rules, RBI
circulars, the IT Act), or the eval set drops below ~85% on paraphrase-heavy
questions.

**How, concretely — hybrid, never dense-only:**

```python
# 1. Embed each chunk's (verbatim + plain-English + questions) text.
#    BGE-M3 or e5-large; both run locally via Ollama or sentence-transformers.
# 2. Keep BM25 exactly as it is.
# 3. Fuse with Reciprocal Rank Fusion — no score normalisation needed:
#        RRF(d) = Σ 1 / (60 + rank_in_that_list(d))
# 4. Feed the fused top-k into the SAME graph expansion, unchanged.
```

RRF beats score-blending because BM25 scores and cosine scores are not on
comparable scales. And **BM25 must stay in the mix** — dense-only retrieval
regresses badly on exact section numbers and rupee amounts, which is precisely
what legal users search for.

You already have **Neo4j Aura**, which ships a native vector index — so the
embeddings could live on the same nodes as the graph, and one Cypher query could
do similarity search *and* traversal together. That is the natural next step for
this architecture.

---

## Stage 7 — Graph expansion

This is what makes it a **Graph**RAG rather than a search box.

```
query
  ├─ vocabulary bridge + stemming
  └─ BM25 over (verbatim + gloss + questions)
        ▼  top-6 seeds
   ONE-HOP EXPANSION, priority ordered
        ├─ 0  PENALISED_BY / PENALISES   duty ⇄ penalty
        ├─ 1  REFERENCES / CITED_BY      cited and citing provisions
        ├─ 2  DEFINES                    definitions of terms used
        ├─ 3  HAS_ENTRY                  Schedule rows
        └─ 4  MENTIONS                   (last — exhaustive, therefore noisy)
        ▼  capped at 8 added
   deduped, ranked by hop distance
```

### Two design decisions that took iteration

**Priority ordering.** `MENTIONS` is exhaustive by construction — §8(5) mentions
six defined terms — so an unranked expansion buried the *cited sections* under
six definitions. Cited provisions are almost always more load-bearing than
mentioned terms.

**Reverse traversal.** `PENALISED_BY` points duty → penalty. But a user asking
about a *fine* lands on the Schedule and needs to go **up** to the duty — the
opposite direction. Without reverse edges, "what's the fine if data leaks?"
returned penalty rows with no explanation of what was breached. Now:

```
[0] Schedule entry 1                     (bm25 36.3)
[0] Schedule entry 2                     (bm25 38.6)
[1] Section 8(5)   ← Schedule entry 1 —PENALISES→
[1] Section 8(6)   ← Schedule entry 2 —PENALISES→
```

Citations to a *clause* resolve up to the nearest chunked provision, so a
reference to §10(2)(a) surfaces the section that contains it.

### Better approach?

- **Two-hop with decay** for multi-step questions (*"if our processor in
  Singapore leaks children's data, what do we owe?"* touches §8, §9, §16 and the
  Schedule). Currently one hop; two hops need a decay factor to avoid pulling in
  half the Act.
- **Learned edge weighting** — which edge types actually help, per intent,
  measured against the eval set rather than hand-assigned.
- **Cypher-side expansion** now that the graph is in Neo4j: one query doing
  seeds + traversal + ranking server-side.

---

## Stage 8 — Answer generation and citation verification

### Generation

The retrieved provisions are assembled into a prompt (~8–10k chars, capped for a
local model) with a system prompt that requires: answer only from the supplied
provisions, quote exactly, cite everything, write for a non-lawyer, never state a
rupee amount not copied character-for-character.

### Verification — the part that matters

**A model can write `§8(5)` whether or not §8(5) says what it claims.** So every
citation in the answer is parsed out and resolved against the graph:

| Status | Meaning |
|---|---|
| **verified** | the provision exists **and** was in the retrieved context |
| **out of context** | it exists, but was never retrieved — the model recalled it from training. Suspect. |
| **unresolved** | no such provision. Invented. |

Tested against a deliberately bad answer:

```
unresolved      §8(5)(z)          no such provision; nearest is §8(5)
unresolved      §99(4)            no such provision in this Act
out_of_context  §12(1)            exists but was not retrieved for this question
verified        §8(5)             "A Data Fiduciary shall protect personal data…"
verified        Schedule entry 2  "Breach in observing the obligation to give the Board…"
```

Each expands to the Act's own words in the UI. **This is the accuracy proof** —
the user does not have to trust the model, they can check it.

### Penalty amounts bypass the model entirely

Asked *"what's the fine for weak security?"*, the 3B model answered **₹200 crore**.
The correct answer is **₹250 crore** — it read the figure off the adjacent
Schedule row.

The fix was not a bigger model. Penalty amounts are structured data the graph
already resolved, so they are rendered from the graph in their own table,
labelled *"read directly from the graph — these never pass through the model"*.

**General principle: if the graph knows a fact exactly, never let a language
model restate it.**

---

## Stage 9 — The backend

```
Browser ──POST /api/chat──► FastAPI
                              │
                              ├─ 1. ask.retrieve()  in a thread   ~900 ms
                              │      └─► SSE "retrieval"  (provisions + why)
                              │
                              ├─ 2. llm.chat_stream() in an executor,
                              │     fragments pushed onto an asyncio.Queue
                              │      └─► SSE "token" × N   (~194 for a typical answer)
                              │
                              ├─ 3. check_citations(answer, retrieved_ids)
                              │      └─► SSE "citations" + graph penalty table
                              │
                              └─ 4. SSE "done"  (model, elapsed, context size)
```

### Why streaming, and why in four stages

A local 3B model takes 25–60s to finish a statutory answer. Waiting on a blank
screen reads as broken. But the *retrieval* completes in under a second — so it
is sent first. The user sees which provisions were found, and how, while the
model is still thinking. The wait becomes legible instead of empty.

Ollama is blocking and synchronous, so generation runs in an executor thread and
pushes fragments onto an `asyncio.Queue` that the SSE generator drains. The event
loop is never blocked.

**One real bug worth recording:** the first live run streamed to completion and
rendered *nothing*. `sse_starlette` terminates lines with `\r\n`, so the
frontend's `\n\n` frame split never matched — no error, no warning, just silence.
CRLF is now normalised at the decode boundary.

### Endpoints

| | |
|---|---|
| `POST /api/chat` | `{question, k}` → SSE stream |
| `GET /api/provision/{id}` | verbatim text of any provision — what a citation click opens |
| `GET /api/health` | model, chunk count, node count |

The index and graph load once at startup; BM25 is rebuilt per query (142 docs —
milliseconds), but JSON parsing does not belong in the request path.

---

## Validation

Correctness is enforced, not asserted. `test_build.py` — 15 checks:

| Check | Guards against |
|---|---|
| **lossless round-trip** | any text invented, dropped or reordered |
| 44 sections, numbered 1–44 | a missed or duplicated section |
| §2 has 28 clauses | the `(i)` ambiguity regressing |
| ≥30 sub-clauses survive | sub-clauses being swallowed as continuation text |
| every section has a chapter + headnote | column-split drift |
| no headnote or page header inside body text | marginalia leaking into operative text |
| 7 Schedule rows, all populated | table parse failure |
| `PENALISED_BY` joins §8(5)→₹250cr etc. | the graph's headline claim |
| all cross-references resolve | dangling edges |
| §81/§87 not linked internally | external citations being mislinked |
| **eval.yaml ≥ 25/26** | retrieval regressions |

`eval.yaml` holds 26 real-user questions with expected provisions. It has already
caught three separate problems: two wrong expectations of mine, the missing
stemmer, and the ±2 score swing from non-deterministic generation.

---

## Honest weaknesses

| Weakness | Severity | Fix |
|---|---|---|
| **The local 3B model** is the weakest component by far. It misread a Schedule figure, and read §14 as "family can step in" when only a *nominated* person qualifies | **High** | `ollama pull qwen2.5:7b-instruct` — retrieval is not the problem |
| No embeddings → paraphrase questions with no shared root can miss (*"Singapore"* → §16) | Medium | hybrid BM25 + dense with RRF |
| One-hop expansion only | Medium | two hops with decay for compound questions |
| 26 eval questions is a small set | Medium | grow to ~100; the harness is already there |
| One-hop `MENTIONS` is noisy (605 edges) | Low | already deprioritised and capped |
| DPDP Rules 2025 not included | Contextual | the graph is versioned; layer them in without a rebuild |

---

## If I were rebuilding this for production

In priority order:

1. **7B+ answering model.** Single biggest quality gain available. Nothing else
   comes close.
2. **Hybrid retrieval** — add dense vectors on the Neo4j nodes, fuse with RRF,
   keep BM25.
3. **Grow the eval set to ~100 questions** with a legal reviewer confirming the
   expected citations.
4. **Cross-encoder reranking** of the top ~20 before the graph expansion.
5. **Layer in the DPDP Rules 2025** and cross-link them to the Act's sections —
   this is where the graph model really pays off, because the Rules are almost
   entirely a web of references back into the Act.
6. **Akoma Ntoso export** for interoperability with other legal tooling.

What I would **not** change: verbatim storage with a round-trip check,
graph-sourced penalty amounts, citation verification, and the review/sign-off
files for the three typesetting conventions. Those are what make the output
trustworthy rather than merely plausible.
