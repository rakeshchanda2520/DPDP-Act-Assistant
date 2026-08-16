# DPDP Act 2023 — Knowledge Graph + RAG Plan

**Source:** `DPDP ACT 2023.pdf` (Gazette of India, No. 22 of 2023), 21 pages.
**Verdict up front:** don't draw bounding boxes by hand. You don't need to. The document
is not irregular — it only *looks* irregular because the extractor is mixing three
different text columns into one stream. Fix that with two x-coordinate thresholds
(20 lines of code) and the document becomes one of the most regular, machine-parseable
legal texts you will ever handle.

---

## 1. What the PDF actually contains (measured, not assumed)

I extracted the word-level bounding boxes to check. Facts:

| Property | Value |
|---|---|
| Pages | 21 |
| Text layer | Native (digital). **No OCR needed.** |
| Page size | 595 × 842 pt (A4) |
| Chapters | 9 (I–IX) |
| Sections | 44 (numbered `1.` … `44.`) |
| Illustrations | 11 (worked examples embedded in sections) |
| Schedule | 1, on page 21 — a real 3-column table (7 rows of penalties) |
| Encoding quirk | Curly quotes/dashes come out as `�`. One `.replace()` fixes it. |

### The three text columns

Every body page has exactly three kinds of text, and they occupy **disjoint horizontal
bands**:

```
x0 < 120                 141 ≤ x0 … x1 ≤ 478               x1 > 480
┌──────────────┐  ┌────────────────────────────────┐  ┌──────────────┐
│  marginalia  │  │                                │  │  marginalia  │
│ (verso/even  │  │          BODY TEXT             │  │ (recto/odd   │
│   pages)     │  │                                │  │   pages)     │
│              │  │                                │  │              │
│ "Definitions"│  │  2. In this Act, unless the    │  │ "Application │
│ "Notice."    │  │  context otherwise requires,—  │  │  of Act."    │
│ "24 of 1997."│  │      (a) "Appellate Tribunal"  │  │ "24 of 1997."│
└──────────────┘  └────────────────────────────────┘  └──────────────┘
        top < 85  =  running header ("THE GAZETTE OF INDIA EXTRAORDINARY")
```

`pypdf.extract_text()` flattens all of this into one stream and dumps the margin notes
at the **end of the page**, which is why your text looks like:

```
...personal data breach under sub-section (5) of section 8.
Definitions.
24 of 1997.
```

That is the entire source of the "irregularity". It's a column problem, not a
structure problem.

### The indentation ladder is quantized

Body-line starting x0 snaps to four discrete values. This gives you the hierarchy
depth **geometrically**, independent of regex:

| x0 (pt) | Level | Marker style |
|---|---|---|
| **142** | section opener / sub-section / continuation line | `4. (1)` or `(2)` |
| **166** | clause | `(a)`, `(b)`, `(za)` |
| **190** | sub-clause | `(i)`, `(ii)`, `(vii)` |
| **214** | item | `(A)`, `(B)` |
| **~271 (centered)** | heading | `CHAPTER II`, `Illustration.`, `THE SCHEDULE` |

Because the text is justified, continuation lines drift to 143–165. **Snap to nearest
of {142, 166, 190, 214}** and only trust the level when the line also *starts with a
marker token*. Otherwise it's a continuation → append to the previous node.

---

## 2. Why manual bounding boxes are the wrong approach

Your instinct — "isolate regions spatially" — is **correct**. Your proposed
implementation is not. Compare:

| | Manual boxes | Automatic x-thresholds (recommended) |
|---|---|---|
| Effort | 21 pages × ~5 regions = ~100 boxes drawn by hand | 2 numbers: `140` and `479` |
| Time | 3–5 hours, error-prone | 20 lines of code |
| Reusable on DPDP Rules 2025 / IT Act / RBI circulars? | No, redo from scratch | Yes, same two thresholds work on any Gazette PDF |
| Captures the hierarchy? | No — boxes give you *regions*, not *parent-child relations*. You'd still need a parser. | Yes — x0 **is** the depth signal |
| Auditability | You'd have to trust your own hand-drawn boxes | Reproducible, diffable, testable |

Manual boxes solve the easy half of the problem (separating columns) at high cost, and
solve none of the hard half (nesting, cross-references). **Skip them.**

The one place a spatial approach genuinely earns its keep: **page 21, the Schedule
table**. That is 7 rows. Use `pdfplumber.extract_table()`, and if it comes out ugly,
just type the 7 rows into a YAML file by hand. Faster than debugging a table parser.

---

## 3. The ontology is handed to you by the Act itself

This is the second reason a KG is a good fit here. Most documents force you to *invent*
an ontology. Section 2 of the DPDP Act **defines every entity explicitly** (28
definitions, `(a)` through `(zb)`). Your node types are literally the defined terms:

`Data Principal`, `Data Fiduciary`, `Significant Data Fiduciary`, `Data Processor`,
`Consent Manager`, `Data Protection Officer`, `Board`, `Appellate Tribunal`,
`Central Government`, `child`, `person`, `personal data`, `personal data breach`,
`processing`, `specified purpose`, `certain legitimate uses`, `notification`, `State`.

You do **not** need an LLM to discover them. You extract them with a regex over Section
2, then string-match those terms across the rest of the Act to create `MENTIONS` edges.
Zero hallucination risk, zero token cost, 100% recall. This is the single biggest
lever in the whole plan.

---

## 4. Target graph schema

### Node types

| Label | Key | Properties | Count |
|---|---|---|---|
| `Act` | `dpdp-2023` | title, act_no, assent_date | 1 |
| `Chapter` | `ch-II` | roman, heading | 9 |
| `Section` | `s-8` | number, headnote (from marginalia!), text | 44 |
| `SubSection` | `s-8-5` | number, text | ~120 |
| `Clause` | `s-8-5-a` | letter, text | ~200 |
| `SubClause` | `s-2-j-i` | roman, text | ~60 |
| `Illustration` | `ill-s5-1` | text, parent_ref | 11 |
| `Definition` | `def-data-fiduciary` | term, text, defined_in | 28 |
| `Entity` | `ent-data-fiduciary` | canonical_name, aliases | ~18 |
| `Penalty` | `pen-3` | sl_no, breach_desc, max_amount_inr | 7 |
| `ExternalAct` | `ext-it-act-2000` | name, year, act_no | ~7 |

> The **marginalia is not noise — it is the section headnote**, and it's gold.
> "Definitions.", "Notice.", "Grounds for processing personal data without consent.",
> "Power to amend Schedule." Attach it as `Section.headnote`. It is a hand-written,
> authoritative one-line summary of each section, perfect for embedding and for
> graph-node labels. Most pipelines throw this away.

### Edge types

| Edge | From → To | How it's derived |
|---|---|---|
| `HAS_CHAPTER` / `HAS_SECTION` / `HAS_SUBSECTION` / `HAS_CLAUSE` | structural parent → child | indentation parse |
| `ILLUSTRATES` | Illustration → Section/SubSection | position in tree |
| `REFERENCES` | any node → Section/SubSection/Clause | **regex over body text** |
| `DEFINES` | Section 2 clause → Definition | parse of §2 |
| `MENTIONS` | any node → Entity | string match on defined terms |
| `PENALISED_BY` | Section/SubSection → Penalty | Schedule column (2) parse |
| `AMENDS` | Section 44 sub-node → ExternalAct | §44 parse |
| `OBLIGATION_OF` / `RIGHT_OF` | Section → Entity | chapter-level rule (Ch. II = obligations of DF, Ch. III = rights & duties of DP) |

### The `REFERENCES` edge is why you want a graph at all

The Act is dense with internal cross-references, and this is **exactly where flat vector
RAG fails**. Real chain from the document:

```
§8(5) "reasonable security safeguards"
   │ PENALISED_BY
   ▼
Schedule entry 2  ──►  "May extend to two hundred and fifty crore rupees"

§9  "additional obligations in relation to children"
   │ REFERENCES
   ▼
§2(f) "child" = under 18       ──► and PENALISED_BY ──► Schedule entry 4 (₹200 cr)
```

Ask a vector-RAG "what's the penalty for a security-safeguard failure?" and it retrieves
§8 — which never mentions a rupee figure — and the Schedule row, which never mentions the
word "security" in a way that embeds close to the question. **The join lives in the
citation, not in the semantics.** That's the graph edge.

Regexes to build it (run over every node's raw text before you strip anything):

```python
RE_SECTION   = r'section\s+(\d+[A-Z]?)'
RE_SUBSEC    = r'sub-section\s*\((\d+)\)'
RE_CLAUSE    = r'clause\s*\(([a-z]{1,2})\)'
RE_SUBCLAUSE = r'sub-clause\s*\(([ivx]+)\)'
RE_SCHEDULE  = r'\bthe Schedule\b'
RE_EXT_ACT   = r'([A-Z][A-Za-z ,()]+Act),?\s+(\d{4})'
```

Resolution rule: a bare `sub-section (5)` with no `of section N` resolves to the
**current section** (the node's own ancestor). `sub-section (5) of section 8` resolves
absolutely. Handle both; the relative form is common.

---

## 5. Implementation phases

### Phase 0 — Setup (15 min)

```bash
pip install pdfplumber networkx rank-bm25 chromadb anthropic pyyaml
```

Skip Neo4j for now. 44 sections ≈ ~500 nodes. NetworkX in memory + a JSON dump is
plenty, and you can query it in one line. Move to Neo4j only if you later merge the
DPDP Rules, RBI circulars, and IT Act into one multi-document graph — *then* it earns
its operational cost.

### Phase 1 — Column-aware extraction

The core of the whole thing. Validated against the actual PDF geometry:

```python
import pdfplumber, collections, re

BODY_L, BODY_R, HEADER_TOP = 140, 479, 85
LEVELS = [142, 166, 190, 214]          # section/subsec, clause, subclause, item

def snap(x0):
    return min(range(len(LEVELS)), key=lambda i: abs(LEVELS[i] - x0))

def extract(path):
    pages = []
    with pdfplumber.open(path) as pdf:
        for pno, page in enumerate(pdf.pages, 1):
            lines, margin = [], []
            rows = collections.defaultdict(list)
            for w in page.extract_words():
                rows[round(w["top"])].append(w)
            for top in sorted(rows):
                if top < HEADER_TOP:            # running header — drop
                    continue
                ws = sorted(rows[top], key=lambda w: w["x0"])
                body = [w for w in ws if w["x0"] >= BODY_L and w["x1"] <= BODY_R]
                side = [w for w in ws if w not in body]
                if side:
                    margin.append((top, " ".join(w["text"] for w in side)))
                if body:
                    lines.append({
                        "page": pno, "top": top,
                        "x0": body[0]["x0"],
                        "level": snap(body[0]["x0"]),
                        "centered": body[0]["x0"] > 240,
                        "text": " ".join(w["text"] for w in body),
                    })
            pages.append({"page": pno, "lines": lines, "margin": margin})
    return pages

FIXES = {"�": "—"}   # inspect and refine: — “ ” ’
```

**Checks to run immediately after this step** (do not skip — these are your unit tests):

```python
assert len([l for l in all_lines if re.match(r'^\d+\.\s', l["text"])]) == 44
assert sum(l["text"].startswith("CHAPTER") for l in all_lines) == 9
assert sum(l["text"].startswith("Illustration") for l in all_lines) == 11
# and: no line in body contains a marginalia headnote string
```

Page 1 (bilingual gazette masthead) and page 21 (Schedule table) will fail these
heuristics. **Handle both by hand** — page 1 is metadata you type once, page 21 is
7 table rows. Do not build a parser for 2 pages.

### Phase 2 — Marginalia → section headnotes

Margin fragments arrive as wrapped pieces (`"Grounds for"`, `"processing"`, `"personal
data."`). Two rules to reassemble:

1. Group consecutive margin fragments whose `top` values are within ~12 pt (one line
   height) of each other → one headnote.
2. Fragments matching `^\d+ of \d{4}\.$` are **external act citations**, not headnotes.
   Route them to `ExternalAct` nodes.
3. Attach each headnote to the section whose opener line has the nearest `top` on that
   page. (Gazette convention: the headnote is typeset level with the section's first
   line.)

Expect ~44 headnotes. Eyeball the list once; it takes two minutes and catches every
misalignment.

### Phase 3 — Build the tree

Single pass over `lines`, maintaining a stack:

```
if line is CHAPTER heading      → close section, push Chapter
elif ^(\d+)\.                   → new Section (level 0)
elif ^\((\d+)\) at level 0      → new SubSection
elif ^\(([a-z]{1,2})\) at lvl 1 → new Clause
elif ^\(([ivx]+)\) at level 2   → new SubClause
elif ^\(([A-Z])\) at level 3    → new Item
elif text == "Illustration."    → open Illustration block, absorb until dedent
else                            → append to current node (continuation line)
```

Ambiguity warning: `(i)` is both roman-one and the letter-i clause. **Resolve by
level (x0), not by the token.** This is precisely why you keep the geometry — the
character alone is genuinely ambiguous and every text-only parser gets it wrong.

Serialize the tree to `dpdp_tree.json`. This file is your ground truth; everything
downstream reads it.

### Phase 4 — Build the graph

```python
import networkx as nx
G = nx.MultiDiGraph()
# 1. walk tree      → nodes + structural edges
# 2. parse §2       → Definition + Entity nodes, DEFINES edges
# 3. regex all text → REFERENCES edges (resolve relative refs against ancestors)
# 4. string-match   → MENTIONS edges (entity aliases from §2)
# 5. load schedule  → Penalty nodes + PENALISED_BY edges
nx.write_gexf(G, "dpdp.gexf")   # opens in Gephi for a visual sanity check
```

**Sanity metrics** — if these are off, your parse is wrong:
- Nodes: ~450–550. Edges: ~900–1200.
- Zero orphan Sections (every section has a chapter parent).
- `REFERENCES` edges: expect 150+. Every one must resolve to an existing node —
  log unresolved targets; a nonzero count means a numbering bug.
- All 7 Penalty nodes reachable from a Section.

### Phase 5 — Chunking for retrieval

**Chunk at the Section level, not by token count.** 44 sections, longest is maybe 900
tokens — comfortably inside any embedding model's window. Fixed-size chunking would cut
`(a)`/`(b)` clause lists in half and destroy exactly the structure you just spent four
phases recovering.

Each chunk text, assembled from the graph:

```
[Chapter II — OBLIGATIONS OF DATA FIDUCIARY]
Section 8 — Reasonable security safeguards.        ← headnote from marginalia
Cites: §5, §6(1), §10 · Penalised by: Schedule 2, 3
---
<full section text, hierarchy preserved with indentation>
---
Illustration: <text>
```

That header block does real work: it lets the embedding capture *context the section
body never states*, and it survives into the LLM's prompt so answers can cite properly.

Two extra granularities, cheap to add, worth it:
- **Sub-section chunks** for long sections (§8, §10, §27, §33) — improves precision.
- **Definition chunks** (28 of them, one per defined term) — users ask "what counts as a
  child under the Act?" and this hits exactly.

Store `node_id` as metadata on every vector so a hit maps straight back into the graph.

### Phase 6 — Retrieval: hybrid + graph expansion

```
query
  ├─► BM25          (exact: "section 33", "250 crore", "Consent Manager")
  └─► embeddings    (semantic: "what if we leak customer data?")
        │
        ▼  reciprocal-rank fusion → top 5 seed nodes
        ▼
   GRAPH EXPANSION (1 hop, this is the whole point)
        ├─ + parent Chapter heading            (context)
        ├─ + every REFERENCES target           (the cited sections)
        ├─ + every DEFINES target for terms used (the definitions)
        ├─ + PENALISED_BY targets              (the consequences)
        └─ + ILLUSTRATES children              (the worked examples)
        ▼
   dedupe, cap ~8k tokens, rank by hop distance
        ▼
   LLM answer with mandatory citations (§N(x)(y))
```

BM25 is not optional here. Legal queries are full of exact tokens — section numbers,
rupee amounts, capitalized defined terms — that embeddings blur. Hybrid is a known
double-digit improvement on statutory corpora and it's ~10 lines with `rank_bm25`.

---

## 6. "Should I build a KG, or is plain RAG enough?"

Honest answer, both halves:

**Plain vector RAG would work acceptably for ~70% of queries on this document.** It's
21 pages. Naive chunking + a decent embedding model answers "what is a Data Fiduciary"
just fine. If you needed something working this afternoon, that's the move.

**The graph earns its cost on the other 30%, which is the 30% that matters in BFSI:**

| Query | Flat RAG | Graph |
|---|---|---|
| "What is personal data?" | ✅ | ✅ (no gain) |
| "Penalty for failing to notify a data breach?" | ❌ §8(6) has no amount; Schedule row has no context | ✅ `PENALISED_BY` edge joins them |
| "Every obligation on a Significant Data Fiduciary" | ⚠️ finds §10, misses §§8, 9 which also bind SDFs | ✅ traverse `OBLIGATION_OF` |
| "What does §33(1) depend on?" | ❌ | ✅ follow `REFERENCES` |
| "What changed in the IT Act 2000?" | ⚠️ | ✅ `AMENDS` edges from §44 |
| "Show me every provision mentioning 'child'" | ⚠️ recall gaps | ✅ `MENTIONS`, exhaustive |

The decisive argument isn't retrieval quality — it's **auditability**. In a compliance
setting you must be able to answer *"which provision did this come from, and what does
it depend on?"* The graph gives you a citation path. A cosine score does not. For a
regulator-facing or audit-facing system, that alone justifies the build.

**Also relevant:** the marginal cost here is genuinely low. There's no LLM extraction
step, no entity-resolution model, no hallucination surface. The graph is built by
regex over a text whose ontology is printed in Section 2. That is an unusually
favourable cost/benefit — most KG projects are not this cheap, and you should be
skeptical of KGs by default. This one is worth it.

**Recommendation: build both, in this order.**
1. Ship Phase 1–3 + Section-level chunks + hybrid search. Working RAG, ~1 day.
2. Add the graph and expansion on top. It's additive — same chunks, better neighbours.

Don't build the graph first and the RAG second. You'll over-model the ontology before
you know which queries actually need it.

---

## 7. Effort estimate

| Phase | Time | Risk |
|---|---|---|
| 0. Setup | 15 min | none |
| 1. Column extraction | 2–3 h | low — geometry already verified |
| 2. Headnote reassembly | 1–2 h | medium — fragment grouping needs eyeballing |
| 3. Tree parser | 3–4 h | medium — `(i)` ambiguity, illustration boundaries |
| 4. Graph build | 3–4 h | low — regex + string match |
| 5. Chunk + embed | 1–2 h | low |
| 6. Hybrid + expansion | 3–4 h | low |
| 7. Eval set (40 Q&A) | 2–3 h | the part everyone skips; don't |
| **Total** | **~2.5 focused days** | |

---

## 8. Pitfalls, ranked by how likely they are to bite you

1. **Trusting `extract_text()`.** It is what made this look hard. Word boxes only.
2. **`(i)` = clause-i vs roman-one.** Disambiguate by x0. Text-only parsers fail here.
3. **Throwing away marginalia.** It's the headnotes. Highest-value metadata in the file.
4. **Fixed-size chunking.** Splits clause lists, destroys the hierarchy. Chunk by section.
5. **Page 21 Schedule.** Column-major reading order — comes out scrambled. Type it manually.
6. **Page 1 masthead.** Bilingual gazette boilerplate. Exclude, hand-enter the metadata.
7. **`�` mojibake.** Curly quotes/em-dashes. Normalize before regex or your
   `"defined term"` patterns silently miss.
8. **Unresolved cross-references.** Always log them. A nonzero count = a real parse bug,
   not an edge case.
9. **The Act as ground truth.** DPDP Rules 2025 and Gazette amendments change the
   operative picture. Version your graph (`Act.version`) so you can layer the Rules in
   later without a rebuild.
10. **Modelling before querying.** Write your 40 eval questions *first*. They will tell
    you which edge types you actually need — probably fewer than the schema above.

---

## 9. Suggested repo layout

```
dpdp-kg/
├── data/
│   ├── DPDP ACT 2023.pdf
│   ├── schedule.yaml          # 7 penalty rows, hand-entered
│   └── act_meta.yaml          # page-1 metadata, hand-entered
├── src/
│   ├── extract.py             # Phase 1: column-aware word boxes
│   ├── headnotes.py           # Phase 2: marginalia reassembly
│   ├── parse.py               # Phase 3: → dpdp_tree.json
│   ├── graph.py               # Phase 4: → dpdp.gexf / dpdp_graph.json
│   ├── index.py               # Phase 5: chunks + BM25 + vectors
│   └── retrieve.py            # Phase 6: hybrid + graph expansion
├── out/
│   ├── dpdp_tree.json
│   └── dpdp_graph.json
├── tests/
│   └── test_structure.py      # the 44/9/11 assertions + ref resolution
└── eval/
    └── questions.yaml         # 40 Q&A with expected §citations
```

---

## 10. Immediate next step

Run Phase 1 on the PDF and print the 44 section openers with their reassembled
headnotes. If that list is clean, the remaining phases are mechanical. If it isn't,
the fix is a threshold adjustment, not a redesign.

Then write the eval questions — before the graph schema, not after.
