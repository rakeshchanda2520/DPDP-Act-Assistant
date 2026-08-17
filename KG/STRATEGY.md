# STRATEGY — legal-RAG techniques from the research literature

> **Implementation status (measured, not assumed).** Four of the nine items
> below have now been built; the findings corrected two of this document's
> own predictions, which are marked inline where that happened.
>
> | § | Item | Status |
> |---|---|---|
> | 1 | InLegalBERT embeddings | **Blocked** — needs a ~450MB model download; disk was at 99% |
> | 2 | NLI citation verification | **Built and validated** (`entailment.py`), off by default. Catches both recorded real bugs (§14 at 0.004, §17(3) at 0.001) where citation-existence checking said `verified`; one false positive found and fixed — see §2 |
> | 3 | Query decomposition | **Built** (`decompose.py`), off by default. Measured: mechanism works, but split quality on the local 3B model is unreliable — see the correction in §3 |
> | 4 | PageRank authority | **Built** (`build.py` `add_authority`, consumed in `ask.py`). Measured: works as a within-tier tie-breaker, but does **not** fix hub-flooding as this document predicted — see the correction in §4 |
> | 5 | Conformal abstention | **Not started** — needs ~100 labelled calibration questions; 6 exist |
> | 6 | Contextual retrieval | **Not started** — schema change is trivial, but needs a ~3h `plain_language.json` regeneration to take effect |
> | 7 | Temporal graph | **Deliberately not started** — design-only until the Rules extraction exists, per this document's own recommendation |
> | 8 | Self-consistency | **Not started** |
> | 9 | Multi-agent verification | **Partially covered** by §2's separate-verifier design |
>
> Nothing built here is on by default. Every one is a documented env-var
> opt-in, for the same reason `hybrid.py` is: unmeasured is not the same as
> better, and this project's own history includes a plausible-sounding
> retrieval "improvement" that measured *worse* and was reverted.

This is a **research digest and plan**. It complements `TARGET.md`: that
document is about *production hardening* (model, latency, ops, eval depth)
using techniques already decided on. This document is about *answer quality
and grounding* techniques from the wider legal-AI research literature — what
they are, why they'd help *this specific system*, and what they'd cost.

Everything below is sourced from real 2024–2026 research and shipped legal-AI
products, not invented. Sources are linked inline and collected at the bottom.

---

## Where this project already stands, so nothing below gets re-pitched

Before reading the gaps, it's worth being clear about what this system
**already does that the research literature treats as hard, unsolved, or
optional** — so the list below is genuinely new ground, not a rediscovery of
what `build.py`/`ask.py`/`api.py` already do:

| Already implemented | What the literature calls this |
|---|---|
| Verbatim storage + character-exact round-trip proof | "Grounding" — most systems assert it; this one *proves* it |
| Citation resolved against the graph, labelled verified/out-of-context/unresolved | Regex-based citation-existence checking (one layer of what §2 below adds) |
| `PENALISED_BY` graph edge joining duty → penalty | A hand-built instance of "incorporating legal structure into RAG" (§ below cites the general research) |
| `vocab.yaml` + `plain_language.json` (layperson↔statute vocabulary bridge) | A hand-built precursor to "Contextual Retrieval" (§6) |
| BM25 + optional hybrid dense+rerank, gated on eval | Standard hybrid retrieval, already shipped |
| `should_abstain()` BM25-threshold gate | A first-line version of "uncertainty-aware abstention" (§5 below is the upgrade path) |
| `audit_log.jsonl` + content-hashed `build_id` | Basic provenance/versioning — §7 below is the deeper version for when the Rules land |

The honest research finding worth sitting with first: **a 2025 peer-reviewed
study of leading commercial legal-AI research tools found they still produced
incorrect or unsupported statements on 17–33% of queries**, even with RAG in
place — RAG cuts the error rate sharply versus a raw chatbot, but every
citation still needs independent verification; "RAG eliminates hallucination"
is a marketing claim the research does not support ([Magesh et al. 2025,
*Journal of Empirical Legal Studies*](https://onlinelibrary.wiley.com/doi/full/10.1111/jels.12413)). That is the frame for everything below: these
are all evidence-based ways to push that error rate down further, not ways to
make it zero.

---

## The findings, in priority order

Each entry: what it is, what the research shows, why it specifically helps
*this* DPDP system, and roughly what it costs.

---

### 1. Domain-specific legal embeddings (InLegalBERT) — replace the generic embedder in `hybrid.py`

**What it is.** `hybrid.py` currently embeds chunks with
`sentence-transformers/all-MiniLM-L6-v2` — a general-purpose English sentence
embedder, trained on web/forum/QA text, not law. **InLegalBERT**
(`law-ai/InLegalBERT`, [Hugging Face](https://huggingface.co/law-ai/InLegalBERT))
is a BERT-base model further pre-trained on 5.4 million Indian legal
documents (27GB corpus, 300k training steps), and has been directly evaluated
on **legal statute retrieval** as one of its benchmark tasks, alongside
rhetorical-role labelling and judgment prediction, achieving state-of-the-art
results on 4 of 5 evaluation datasets ([Paul et al., *Pre-trained Language
Models for the Legal Domain*, arXiv 2209.06049](https://arxiv.org/pdf/2209.06049); [InLegalBERT model card](https://huggingface.co/law-ai/InLegalBERT)).

**Why it helps here specifically.** MiniLM's training data has essentially
zero exposure to the register this project retrieves in — "Data Fiduciary",
"specified purpose", "reasonable security safeguards" are ordinary English
words used in a specific legal sense, and a generic embedder has no reason to
cluster them the way a lawyer would. This is a plausible root cause of the
hybrid-mode regressions this project already measured and documented (§6
citing §5 instead of §6 for "can I text customers marketing offers?" —
`ask.py`'s `retrieve()` comments). A model that has actually seen 5.4M Indian
legal documents, including presumably statutory language patterns close to
the DPDP Act's own drafting style, should discriminate "Consent" from
"Notice" far better on vocabulary it was trained to specialise in.

**Cost / how to try it.** Almost free to test: `hybrid.py`'s `EMBED_MODEL`
constant is a single string; InLegalBERT is a mean-pooling encoder (not a
native sentence-transformers model, so it needs a small wrapper — mean-pool
the last hidden state, or use the `sentence-transformers` `models.Transformer`
+ `models.Pooling` composition) rather than a one-line drop-in, but it is
still an afternoon of work. **Gate it exactly like the hybrid mode itself is
already gated**: run it through `eval.yaml`'s 101 cases and compare against
the current 92/101 hybrid score before deciding to switch. Given this
project's own measured experience that a plausible-sounding embedding change
made things *worse* until measured, don't skip the measurement step here
either.

---

### 2. NLI-based citation-content verification — catch a *correct* citation with a *wrong* claim

**What it is.** `api.py`'s `check_citations()` currently verifies that a
cited provision **exists** and **was retrieved** — `verified` /
`out_of_context` / `unresolved`. It does **not** verify that what the model
*said* about that provision is actually true of its text. The research
technique that closes this gap is **NLI-based grounding verification**: for
each claim (or each cited sentence) in the answer, run a cross-encoder
entailment model (commonly DeBERTa-v3, at a calibrated threshold like ~0.65)
against the cited provision's verbatim text, and check whether the text
*entails* the claim ([Google Cloud grounding docs](https://docs.cloud.google.com/generative-ai-app-builder/docs/check-grounding);
survey: [Sadat et al., *Grounding and Evaluation for LLMs*, arXiv
2407.12858](https://arxiv.org/pdf/2407.12858)).

**Why it helps here specifically.** This is the single most direct fix for
the exact documented failure this project already treats as its canonical
regression case: the §14 "family can step in" error. Today's
`check_citations()` would mark `§14` as `verified` — it *was* retrieved, it
*does* exist — even though the claim built on top of it is wrong. An
entailment check would compare the claim "family can access the account"
against §14's actual text ("nominate ... any other individual") and correctly
flag it as **not entailed**, independent of whether the citation ID itself
resolves. This is a materially stronger trust signal than citation-ID
resolution alone, and it directly targets the error class this project's own
`answer_eval.yaml` was built to catch — except NLI catches it automatically,
without needing a hand-written `must_not_say: ["family can"]` regression case
for every future error of this shape.

**Concrete implementation for this codebase:** a small cross-encoder (e.g.
`cross-encoder/nli-deberta-v3-base`, comparable size to the reranker already
added in `hybrid.py`) run per sentence of the answer against the verbatim
text of whichever provision that sentence cites, added as a fourth citation
status alongside `verified`/`out_of_context`/`unresolved`: something like
`unsupported` (citation resolves and was retrieved, but the entailment score
is below threshold). Cheap at this scale — a typical answer is 5–8 sentences,
each one entailment check against a short provision, all local, no network
call.

**Cost.** A day: model integration mirrors the reranker already built in
`hybrid.py`, sentence-splitting the answer is a regex, and the eval-set
extension is `answer_eval.yaml` cases specifically targeting known
misstatement patterns (which this project already has two of).

#### ✅ Built and validated — it catches the exact bug it was built for

Implemented as `entailment.py` (`DPDP_ENTAILMENT=1`), wired into `api.py`'s
citation stage: each answer sentence is scored against the verbatim text of
the provisions it cites, and sentences below the entailment threshold come
back in a new `claims` field on the `citations` SSE event and in the audit
log. Answer-format scaffolding (`Short answer:`, `Why:`, …) is stripped
before checking, since those are not legal claims.

The design detail worth keeping: a claim counts as supported if **any** cited
provision entails it, so an answer legitimately drawing on three provisions
isn't penalised because its second sentence is grounded in the third
provision rather than the first. Entailment label order is read from the
model's own `id2label` config rather than assumed — it is not alphabetical
and not consistent across NLI checkpoints, and guessing it wrong silently
inverts the entire check.

**Measured on the two errors this project actually recorded, and the
separation is decisive:**

| Claim, checked against the provision it cites | P(entail) | Verdict |
|---|---:|---|
| §14 — *"family members can exercise their rights"* — **the real 3B bug** | **0.004** | `unsupported` ✅ |
| §14 — *"may nominate another individual…"* (correct) | **0.998** | `supported` ✅ |
| §17(3) — *"startups are automatically exempt"* — the exemption trap | **0.001** | `unsupported` ✅ |
| §17(3) — *"the Central Government may notify…"* (correct) | **0.994** | `supported` ✅ |

Both are cases where `check_citations()` returns `verified` — the provision
exists and was retrieved — so this is genuinely new signal, not a
restatement of a check the system already had.

**One real false positive found and fixed, worth recording.** A near-verbatim
quote — *"Section 29(1) states that any person aggrieved by an order… may
prefer an appeal"* — initially scored **0.000**, i.e. it looked like a
hallucination. Cause: it is a *meta-statement* ("the document says X")
rather than an assertion of X, and NLI models do not treat the two as
equivalent — the premise never refers to itself. This matters more than a
curiosity, because `ask.SYSTEM` explicitly instructs the model to write in
exactly that form (`The law says: §N(x) — "…"`), so unstripped it would
have flagged well-formed *correct* answers as unsupported — the worst
possible failure for a trust feature. Stripping citation-attribution
prefixes before checking took that same claim from 0.000 to **0.996**, with
the §14 bug still correctly caught at 0.004.

**Two implementation traps, both real, both hit here:**
1. An unfiltered `snapshot_download` pulls every format variant — 9 ONNX
   builds plus duplicate `.bin` and `.safetensors` weights, ~3.6GB — and
   filled the disk. Pass explicit `allow_patterns` (config + safetensors +
   tokenizer ≈ 749MB).
2. `HF_HUB_OFFLINE=1` makes this model *permanently unloadable*, because
   sentence-transformers calls the hub's `model_info` endpoint during
   construction even with everything cached. `os.environ.pop()` does not fix
   it — `huggingface_hub` reads the flag into a module constant at import
   time — so `entailment.py` patches `huggingface_hub.constants.HF_HUB_OFFLINE`
   for the duration of the load and restores it afterwards.

---

### 3. Query decomposition for compound questions

**What it is.** Splitting a multi-part question into independent
sub-questions before retrieval, retrieving for each separately, then merging
— rather than hoping one BM25/hybrid pass surfaces everything a compound
question needs. Recent legal-QA-specific work (**Decompose-and-Refine**,
[arXiv 2605.24454](https://arxiv.org/abs/2605.24454); **KoBLEX**, multi-hop
open legal QA, [arXiv 2509.01324](https://arxiv.org/html/2509.01324)) shows
this is now a standard technique specifically because "many statutory
questions require multi-hop reasoning across multiple legal issues,
substantially increasing the risk of hallucination" when handled as one flat
retrieval.

**Why it helps here specifically.** This project's own `eval.yaml` already
has a "compound questions" section built to stress-test exactly this — three
cases like *"our data processor in another country lost a child's records,
what do we owe and to whom"* that touch §9, §16, and a Schedule entry at
once. Today they pass only because the **OR-semantics** convention (any one
of the listed provisions counts as a hit) is generous — the eval comment even
says so explicitly. A real user asking that question needs *all three*
threads answered, and one BM25/hybrid pass over the whole compound sentence
dilutes the signal for each individual sub-topic (the query vector/token set
is now an average of three concerns, matching each one less precisely than it
would alone). Query decomposition — split into "what are our duties when a
child's data is involved (§9)", "what applies when a processor is abroad
(§16)", "what's the penalty (Schedule)" — retrieves each sub-question at full
strength, then the results merge exactly the way `ask.retrieve()`'s existing
seed+expansion structure already expects.

**Concrete implementation:** a cheap LLM call (works fine on the local 3B
model, since decomposition is pattern-matching, not legal reasoning) that
outputs a JSON list of sub-questions when the question contains coordinating
structure ("and", "what do we owe **and to whom**", multiple named
obligations); run `ask.retrieve()` once per sub-question; union and dedupe
results before `build_context()`. This is a natural extension of
`ask.expand_query()`'s existing role rather than a new stage.

**Cost.** ~1 day. The harder part is honestly the eval question — how do you
score "did it get all three sub-topics" rather than "did it get at least
one"? That requires tightening the compound-question eval cases from
OR-semantics to AND-semantics once decomposition ships, which the current
`eval.yaml` comment already flags as future work.

#### ✅ Built — and the "a small model is adequate" claim above is wrong

Implemented as `decompose.py` (`DPDP_DECOMPOSE=1` or `ask.py --decompose`).
The *merge* mechanism works exactly as designed — retrieving the motivating
question pulled 12 seeds instead of 6, correctly adding §8's security and
breach duties.

But this section claimed decomposition is "pattern-matching, not legal
reasoning, so a small local model is adequate here." **Measured on the local
3B model, that is not true.** Across four compound test questions:

| Question shape | Result |
|---|---|
| Clean coordination (*"delete their data **and also** complain to the Board"*) | **Good** — two faithful, standalone sub-questions |
| Narrative scenario (*"processor in another country lost a child's records, what do we owe and to whom"*) | **Poor** — produced only `"what do we owe?"`, silently dropping both the cross-border and children's-data threads, so §16 still was not retrieved |
| Multi-clause scenario (*"transfer data abroad without telling anyone and it leaks"*) | **Bad** — invented *"How does international data protection differ from domestic regulations?"*, a topic wholly outside this Act, in direct violation of the prompt's explicit "never invent a topic" rule |

So decomposition is only as good as the splitter, and a 3B model is not a
reliable splitter for scenario-shaped legal questions — which is the same
conclusion the rest of this project keeps reaching about that model, just
arrived at from a new direction. This is precisely why the design keeps the
original question first and merges additively: a bad split adds noise to
the context but can never remove a provision the undecomposed query would
have found, so the failure mode is degraded-to-baseline rather than broken.

**Re-evaluate this with `DPDP_PROVIDER=claude` before judging the technique
itself** — the research it comes from assumes a competent decomposer, and
what was measured here is the 3B model's limitation, not the method's.

---

### 4. Provision authority scoring (PageRank over the citation graph) — a free signal this project already has the data for

**What it is.** Network-centrality measures (PageRank, HITS hub/authority
scores) computed over a legal citation graph predict a provision's practical
importance better than raw citation counts — this is well-established in case
law analysis ([PageRank-related methods for citation networks](https://www.researchgate.net/publication/278702375_PageRank-Related_Methods_for_Analyzing_Citation_Networks);
applied to a 100M-document Ukrainian court citation graph: [arXiv
2605.15362](https://arxiv.org/pdf/2605.15362)), and topic-sensitive variants
extend it to rank by both authority and subject relevance simultaneously.

**Why it helps here specifically.** This project already has the exact input
this technique needs and is currently leaving it unused: 605 `MENTIONS`
edges, 84 `REFERENCES` edges, all sitting in `out/dpdp_graph.json`. A
PageRank pass over `REFERENCES` (weighted more than `MENTIONS`, which is
exhaustive-by-construction and already deprioritised in `EXPAND_PRIORITY`)
would give every provision a static importance score — computed once at
`build.py` time, not per query. Two concrete uses for it, both already
flagged as open problems in this project's own code comments:

- **Hub-node flooding in two-hop expansion.** The diagnostic run during this
  session's Phase 5 work found `§40(2)` ("power to make rules") has an
  enormous fan-out — dozens of `REFERENCES` edges — and at a generous
  `MAX_EXPANDED` budget it drowns out everything else reachable from a seed.
  A PageRank-derived importance score lets `EXPAND_PRIORITY` weight
  candidates by *how load-bearing the destination provision is*, not just
  which edge type points to it — so a hub node like §40(2) doesn't get
  free priority just because it has many edges.
  a hub node like §40(2) doesn't automatically flood the pool just because
  it has many outgoing edges.
- **A tie-breaker for BM25/hybrid ranking.** When two provisions score
  similarly on lexical or semantic relevance, the one more central to the
  Act's own cross-reference structure is the more likely intended answer —
  a cheap, deterministic, auditable tie-breaker in exactly the spirit of this
  project's existing intent-boost mechanism in `vocab.yaml`.

**Cost.** Half a day — `networkx` (already a `build.py` dependency) has
`pagerank()` built in; this is a ~15-line addition to `build.py`'s existing
graph-construction pass, stored as a `pagerank` field on each graph node, no
new dependency.

#### ✅ Built — and two things this section got wrong

Implemented as `build.py`'s `add_authority()` (stored as an `authority`
field per node) and consumed in `ask.py`'s expansion as a within-tier
tie-breaker. Two corrections from actually measuring it:

**1. `MENTIONS` had to be excluded entirely, not merely down-weighted.**
This section proposed weighting `REFERENCES` above `MENTIONS`. That is not
enough: with `MENTIONS` included at *any* weight, every single top-authority
node came out a Definition, because all 605 `MENTIONS` edges point *into*
the 28 definitions. The result measured "how often is this term used", not
"how load-bearing is this provision" — the opposite of what expansion needs.
With `MENTIONS` dropped and only `REFERENCES` (1.0) + `PENALISED_BY` (0.5)
counted, the ranking is immediately sensible: §29(1) right of appeal, §33
penalties, §6 consent, the Schedule.

**2. It does NOT fix the §40(2) hub-flooding problem, which was the main
motivation given above.** §40(2) does score low (0.0021, the floor) exactly
as hoped — but it still gets pulled into expansion anyway, because it
arrives via a `CITED_BY` edge whose *tier* (`EXPAND_PRIORITY`) authority
never overrides. Authority only reorders candidates *within* one tier; it
cannot demote across tiers, and deliberately so — letting it do that would
mean a merely-mentioned provision could outrank a genuinely cited one.
Hub-flooding is a distinct problem needing a distinct fix (penalising a
*source* node's out-degree, rather than ranking a *target* node's
in-degree), which is not what PageRank measures.

**3. `eval.yaml` structurally cannot measure this change.** The eval scores
seeds only (`hop == 0`); authority only affects expansion (`hop >= 1`).
Before/after scores are identical (90/101 BM25, 92/101 hybrid) and that is
expected, not evidence of either success or failure. Kept because the signal
is principled, free, and useful to have on the nodes — but an honest
assessment is that its practical benefit here is currently *unproven*, and
would need an expansion-quality eval (which does not exist yet) to
establish.

---

### 5. Calibrated abstention (conformal prediction) — upgrade path for `should_abstain()`

**What it is.** The abstention gate this project shipped in Phase 3 is a
single BM25-score threshold, and its own code comment is honest that it only
catches the clearly-unrelated end (income tax filing, "capital of France")
and not adjacent-domain questions (GDPR, HIPAA) that share real legal
vocabulary with the Act. **Conformal prediction** is the research-grade
upgrade: rather than picking a threshold by eyeballing 8 probe questions (as
this project's calibration did, honestly documented as such), conformal
methods convert *any* uncertainty score into a threshold with a **provable,
statistically certified error-rate guarantee** on a held-out calibration set
— e.g. "abstain such that at most 5% of *answered* questions are wrong, with
95% confidence" ([Uncertainty-Aware Abstention with Provable Alignment
Guarantees, arXiv 2607.04430](https://arxiv.org/html/2607.04430v1)).

**Why it helps here specifically.** This project's own calibration honestly
documented its own limitation in the code (`api.py`'s `ABSTAIN_THRESHOLD`
comment): the threshold of 15.0 was picked from 8 hand-run probes, not a
principled statistical procedure, and known to miss GDPR/HIPAA/RBI-KYC-style
questions that score *higher* than genuine in-scope questions. Conformal
prediction doesn't fix the underlying signal problem (BM25 score still can't
distinguish "shares vocabulary" from "is actually in scope" — that's a
retrieval problem, not a calibration problem, and dense/hybrid retrieval or
domain embeddings from §1 are the actual fix for the signal itself) — but it
replaces "a threshold I picked by hand and admitted has a real gap" with "a
threshold with a mathematically stated error bound," which is a categorically
stronger claim to make to a compliance stakeholder, and it composes with
whatever underlying score (BM25, hybrid RRF, or a future combined score) ends
up feeding it.

**Cost.** A day, mostly in building the calibration set — needs maybe 100+
labelled in-scope/out-of-scope questions (this project's `answer_eval.yaml`
already has 6 abstain cases; conformal calibration wants closer to 100 for a
meaningful guarantee) rather than the 8 currently used. The algorithm itself
is a few lines once the calibration set exists.

---

### 6. Formalised Contextual Retrieval (Anthropic's technique) — systematise what `plain_language.json` already does informally

**What it is.** Anthropic's **Contextual Retrieval**: before embedding (and
before BM25-indexing) each chunk, prepend a short (50–100 token) LLM-written
sentence situating that chunk within its source document — "this chunk is
from §8(5) of the DPDP Act, in the chapter on Data Fiduciary obligations,
about security safeguards for personal data breaches." Anthropic reports a
**49% reduction in retrieval failures** from this alone, with follow-on
implementations reporting 5–15% precision gains ([Anthropic engineering
blog](https://www.anthropic.com/engineering/contextual-retrieval)).

**Why it helps here specifically — and why this is a refinement, not a new
idea.** FLOW.md's own Stage 5 already explicitly names this: *"the plain-
language layer in stage 5 is essentially [Anthropic's contextual retrieval],
done with questions instead of statements."* That's a fair characterisation,
but it's not the same technique, and the difference is worth closing. The
current `plain_language.json` generates a *gloss* (what the provision means)
and *questions* (what a layperson might ask) — both **content-focused**.
Anthropic's technique generates *situating* context — **position-focused**:
where does this chunk sit in the larger document, what does it depend on,
what's above and below it. `index.py`'s `chunk_header()` already assembles
some of this deterministically (chapter, headnote, cites, penalty link) — the
gap is that this structural header is **not** part of what gets embedded for
dense retrieval in the same way, and it is entirely non-LLM-generated (purely
templated), whereas Anthropic's technique specifically found value in an
LLM's *prose synthesis* of the position, not just concatenated metadata
fields.

**Concrete, scoped improvement:** rather than treating this as a new stage,
extend `index.py`'s existing `generate_plain()` LLM call (already running
once per chunk, already cached in `plain_language.json`) to also produce one
situating sentence in the Anthropic style, and prepend it to what
`build_index()` assembles into `docs[i]` — the same string BM25 tokenizes and
`hybrid.chunk_embeddings()` embeds. This reuses the existing one-time LLM
pass rather than adding a new one, and the "40k tokens" cost concern
Anthropic's post raises doesn't apply here — 142 chunks, not thousands.

**Cost.** Half a day — one new field in the existing `PLAIN_SCHEMA`, one
line in `build_index()`'s doc-assembly. Gate on `eval.yaml` before keeping,
same discipline as every other retrieval change in this project.

---

### 7. Temporal/versioned legal-norm graph modeling — the principled way to add the DPDP Rules 2025 (and handle future amendments)

**What it is.** This is the most structurally significant finding, and it
speaks directly to the deferred Phase 7 (DPDP Rules) and to a gap TARGET.md
only mentions in passing ("which answers are now stale?"). Recent research
(**"An Ontology-Driven Graph RAG for Legal Norms"**, [arXiv
2505.00039](https://arxiv.org/html/2505.00039v3)) proposes a four-layer,
FRBRoo-inspired model built specifically to handle legal documents that are
"characterized by a formal hierarchy, a dense web of cross-references, and
continuous diachronic evolution through amendments, repeals, and
consolidations" — exactly this project's situation once the Rules exist
alongside the Act and either gets amended.

The model, adapted to this codebase's vocabulary:

- **Norm** — the Act as a whole (or, separately, the Rules as a whole):
  today's implicit root, made explicit.
- **Component** — a persistent conceptual entity for each provision (§8(5)
  is "the same provision" across any future amendment) — this project's
  `s-8-5` node IDs already are this, just without an explicit temporal axis.
- **Temporal Version** — the *exact text* of a Component as it existed at a
  specific date, with a date-stamped identifier. **This is the piece
  entirely absent today**: `out/dpdp_tree.json` has exactly one version of
  every node, forever, with no way to represent "this is what §16 said before
  the Rules operationalised it" or "this is what changed."
- **Action nodes** — explicit entities representing a legislative change
  (an amendment, the Rules implementing a section of the Act), linking an old
  Temporal Version → new Temporal Version, with a generated explanation of
  *why*.

**Why this is the right shape for the Rules specifically, not just future
amendments.** The DPDP Rules 2025 are — as this project's own `TARGET.md`
already notes — "almost entirely a web of references back into the Act."
That is precisely what the FRBRoo model's inter-Norm relationship edges are
for: model the Rules as a second Norm, and represent "Rule 4 implements §8(5)
of the Act" as a first-class typed edge, not a informally-worded
cross-reference string. Critically, the paper's **aggregation-based
versioning** — a new Temporal Version is created *only* for the specific
Component that actually changed, with everything else reusing its prior
version rather than being duplicated — means adding the Rules doesn't require
re-processing or re-embedding the unchanged parts of the Act, and a future
amendment to, say, §9 alone wouldn't require rebuilding all 404 nodes.

**Why this also closes TARGET.md's own open question.** TARGET.md's
`compute_build_id()` note already says "which answers are now stale?
answerable... because every answer carries the graph build id it was
generated from" — that's a coarse, whole-graph version stamp. The FRBRoo
model gives **per-provision** versioning: an audit-logged answer that cited
§16 could point to the *exact dated version* of §16 that was current when the
answer was given, which is a materially stronger provenance claim than "here
is which overall build this came from."

**Cost — and why this belongs in planning, not this pass.** This is
genuinely the largest item in this document, and is correctly scoped as its
own project rather than a bolt-on: it changes the node-ID scheme
(`s-8-5` → something date-qualified), touches `build.py`, `index.py`,
`ask.py`, and the graph schema simultaneously, and needs the Rules extraction
itself (already flagged as its own multi-day project) done first, since there
is nothing to version against yet. **Recommended sequencing: do the Rules
extraction first as a second, separate Norm with today's flat schema; only
introduce the Temporal Version layer when a second point-in-time actually
exists to model** — i.e., when the Act or Rules are next amended for real,
not speculatively ahead of that event. Building temporal machinery for a
single point in time is exactly the kind of premature abstraction this
project's own engineering discipline (see `TARGET.md`'s "what we deliberately
do not change" list) would reject.

---

### 8. Self-consistency sampling for the highest-stakes claims (penalty figures, citations)

**What it is.** Sample the same answer multiple times at nonzero temperature
and check agreement across samples — if a claim (a rupee figure, a citation)
is consistent across, say, 3–5 independent generations, it's far more likely
factual than a claim that varies each time; hallucinated content tends to be
*inconsistent* across resamples in a way that faithfully-grounded content is
not ([Self-Consistency survey, EmergentMind](https://www.emergentmind.com/topics/self-consistency-technique);
applied to reliability: [arXiv 2505.09031](https://arxiv.org/pdf/2505.09031)).

**Why it helps here specifically — and where it's redundant.** For **penalty
amounts**, this technique is entirely unnecessary: this project already does
something strictly better — it reads the figure directly from the graph and
never lets the model state it at all (`penalty_facts()` in both `ask.py` and
`api.py`, explicitly justified in `FLOW.md` as "if the graph knows a fact
exactly, never let a language model restate it"). Self-consistency is a
mitigation for uncertainty; graph-grounding is an elimination of it — don't
downgrade to the weaker technique where the stronger one already applies.

Where it **would** add value is exactly the class of claim the graph
*cannot* verify because it's not structured data: substantive interpretive
claims like "family can access the account" (the §14 error) or "this
exemption is automatic" (the §17(3) startup-exemption trap already in
`answer_eval.yaml`). These are prose judgements, not facts the graph holds,
so self-consistency is a legitimate (if probabilistic, not certain) signal:
if 3 independent generations at temperature 0.7 disagree about whether family
members can access a deceased user's account, that disagreement itself is
informative and worth surfacing.

**Cost and honest tradeoff.** 3–5x generation cost and latency for whichever
claims get sampled this way — directly in tension with TARGET.md's latency
work (Phase 1's whole point was getting to 3–6 seconds). The realistic
scoping: apply this selectively, only to answers touching sections already
known to be interpretively subtle (§14's nominate-vs-family distinction is
already flagged; a short, hand-maintained list of "sections known to need
extra scrutiny" — the same spirit as `vocab.yaml`'s manually-curated,
auditable approach — rather than blanket 5x sampling on every question).

---

### 9. Multi-agent cross-validation (the pattern behind CoCounsel's Deep Research)

**What it is.** Rather than one model generating and citing in a single
pass, dispatch a **separate verifier pass** — potentially a different, cheaper
model — whose only job is to check "does this specific citation actually
support this specific claim," independent of the model that wrote the
answer. This is the architecture behind Thomson Reuters' CoCounsel Deep
Research: *"the system dispatches specialized agents — one searches case
law, another analyzes statutes, a third synthesizes findings... they
cross-validate results"* ([GC.ai comparison](https://gc.ai/blog/harvey-vs-cocounsel)).

**Why it helps here specifically.** This project already has almost the
right shape for a lightweight version of this, and §2 (NLI verification)
above is really a specific, cheap instance of it: a small, purpose-built
verifier (the entailment cross-encoder) checking the output of a larger
generator (the answer model), rather than asking the same model to
self-check. The general pattern is worth naming explicitly because it
generalises past citation-content checking to other checks this project could
add the same way — e.g., a second cheap pass that specifically checks "does
this answer's 'What to do' section only recommend things the Act actually
requires, not generic security-advice padding," using the same
generator/verifier split.

**Cost and honest tradeoff.** The value of a *separate* verifier over
asking the same model to double-check itself is that a different model (or
a much smaller, specialized one) doesn't share the first model's blind spots
— but that only holds if the verifier is genuinely a different
model/architecture, not the identical model asked twice (research on
self-consistency and self-verification finds a model re-checking its own
work is a weaker signal than an independent verifier). For this project,
the natural free instance of this is already available: **the pluggable
provider from TARGET.md's Phase 1 means Claude can generate while a small
local Ollama model verifies**, or vice versa — genuinely different
architectures, not the same model asked twice. This costs a second (cheap,
small-model) inference call per answer, not a second expensive one.

---

## How these interact with what's already planned (`TARGET.md`)

None of the above compete with TARGET.md's phases — they sit alongside them:

| This document | Relationship to TARGET.md |
|---|---|
| §1 InLegalBERT | A drop-in swap inside TARGET.md's already-shipped hybrid retrieval (`hybrid.py`) |
| §2 NLI verification | Extends TARGET.md's citation verification (`check_citations`) with a content check, not just an existence check |
| §3 Query decomposition | Directly strengthens TARGET.md's two-hop expansion work — decomposition is what makes each hop's seed actually relevant |
| §4 PageRank authority | Uses graph data TARGET.md's `build.py` already produces; feeds into the same `EXPAND_PRIORITY` mechanism |
| §5 Conformal abstention | The statistically-rigorous version of TARGET.md's already-shipped, self-admittedly-approximate `should_abstain()` |
| §6 Contextual retrieval | A refinement of TARGET.md/FLOW.md's existing `plain_language.json` mechanism, not a new one |
| §7 Temporal graph | The principled architecture for TARGET.md's deferred Phase 7 (DPDP Rules) — **do this design work before, not during, the Rules extraction** |
| §8 Self-consistency | A targeted supplement where TARGET.md's graph-grounding principle *can't* apply (prose judgement, not structured fact) |
| §9 Multi-agent verification | Free to attempt once TARGET.md's Phase 1 pluggable provider makes a second, different model available |

---

## Recommended next-step ordering, if you want to act on this

Given effort and what depends on what:

1. **§4 PageRank** (half a day, no new dependency, immediately useful for
   the hub-flooding problem already observed) and **§1 InLegalBERT** (half a
   day, direct fix for a documented hybrid-mode regression) — do these
   together, they're both cheap and both measured the same way (`eval.yaml`).
2. **§2 NLI citation verification** (one day) — the highest trust-per-effort
   item; it is the direct fix for this project's own canonical bug (§14).
3. **§6 Contextual retrieval refinement** (half a day) — cheap, reuses
   existing LLM calls, gate on eval like everything else.
4. **§3 Query decomposition** (one day) + **§5 Conformal abstention** (one
   day, needs a bigger calibration set first) as a pair once the above are
   measured.
5. **§8 / §9** (self-consistency, multi-agent verification) — selective,
   applied only to the small set of sections already known to be
   interpretively subtle, not blanket policy — evaluate cost/benefit against
   TARGET.md's latency budget before deciding scope.
6. **§7 Temporal graph modeling** — design-only until the Rules extraction
   itself (TARGET.md's Phase 7) is underway; don't build the versioning
   machinery before there's a second version to model.

---

## Sources

- [Magesh et al. 2025, "Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools," *Journal of Empirical Legal Studies*](https://onlinelibrary.wiley.com/doi/full/10.1111/jels.12413)
- [InLegalBERT model card, Hugging Face](https://huggingface.co/law-ai/InLegalBERT)
- [Paul et al., "Pre-trained Language Models for the Legal Domain," arXiv 2209.06049](https://arxiv.org/pdf/2209.06049)
- [Google Cloud, "Check grounding with RAG" docs](https://docs.cloud.google.com/generative-ai-app-builder/docs/check-grounding)
- [Sadat et al., "Grounding and Evaluation for Large Language Models," arXiv 2407.12858](https://arxiv.org/pdf/2407.12858)
- ["Decompose-and-Refine: Structured Legal Question Answering with Parametric Retrieval," arXiv 2605.24454](https://arxiv.org/abs/2605.24454)
- ["KoBLEX: Open Legal Question Answering with Multi-hop Reasoning," arXiv 2509.01324](https://arxiv.org/html/2509.01324)
- ["PageRank-Related Methods for Analyzing Citation Networks"](https://www.researchgate.net/publication/278702375_PageRank-Related_Methods_for_Analyzing_Citation_Networks)
- ["Automatic Construction of a Legal Citation Graph from 100 Million Ukrainian Court Decisions," arXiv 2605.15362](https://arxiv.org/pdf/2605.15362)
- ["Uncertainty-Aware Abstention in Large Language Models with Provable Alignment Guarantees," arXiv 2607.04430](https://arxiv.org/html/2607.04430v1)
- [Anthropic, "Introducing Contextual Retrieval"](https://www.anthropic.com/engineering/contextual-retrieval)
- ["An Ontology-Driven Graph RAG for Legal Norms: A Hierarchical, Temporal and Deterministic Approach," arXiv 2505.00039](https://arxiv.org/html/2505.00039v3)
- ["Temporal Misgrounding in Legal RAG: A Versioned-Corpus Benchmark for French Tax Law," arXiv 2608.09393](https://arxiv.org/html/2608.09393)
- ["Self-Consistency: Ensemble Methods for LLMs," EmergentMind](https://www.emergentmind.com/topics/self-consistency-technique)
- ["Improving the Reliability of LLMs: Combining CoT, RAG, Self-Consistency, and Self-Verification," arXiv 2505.09031](https://arxiv.org/pdf/2505.09031)
- [GC.ai, "Harvey vs CoCounsel: 2026 Side-by-Side Comparison"](https://gc.ai/blog/harvey-vs-cocounsel)
- ["Incorporating Legal Structure in Retrieval-Augmented Generation: A Case Study on Copyright Fair Use," arXiv 2505.02164](https://arxiv.org/pdf/2505.02164)
- ["IL-TUR: Benchmark for Indian Legal Text Understanding and Reasoning," arXiv 2407.05399 / ACL 2024](https://arxiv.org/abs/2407.05399)
- [LegalGraphRAG: Multi-Agent Graph Retrieval-Augmented Generation for Reliable Legal Reasoning, ACL Anthology 2026](https://aclanthology.org/2026.acl-long.1738/)
- [KanoonGPT — AI Legal Research for Indian Bare Acts](https://kanoongpt.in/about)
