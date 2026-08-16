"""
Answer a plain-English question about the DPDP Act.

    query
      ├─ vocabulary bridge      "customer" -> Data Principal, "leak" -> personal data breach
      └─ BM25 over verbatim + plain-language layer
              │
              ▼  top seeds
         GRAPH EXPANSION (one hop — the whole reason the graph exists)
              ├─ REFERENCES   the provisions this one cites
              ├─ DEFINES      the definitions of terms it uses
              ├─ PENALISED_BY the Schedule entry it maps to
              └─ parent chapter for context
              ▼
         answer, quoting the verbatim text and citing every provision

The expansion step is what a flat vector store cannot do. "What's the penalty
for failing to notify a breach?" needs section 8(6) — which contains no rupee
figure — joined to Schedule entry 3, which contains no word about notification.
The join lives in the citation, not in the semantics.

Run:  python ask.py "can I text my customers an offer?"
      python ask.py --retrieval-only "penalty for a data leak"
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml
from rank_bm25 import BM25Okapi

import llm

# Windows consoles default to cp1252 and would crash on the Act's curly
# quotes and em-dashes. Never let an encoding detail fail a run.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
OUT = ROOT / "out"
# A local 3B-8B model degrades fast on a wall of statute, so the context budget
# is much tighter than it would be for a frontier model. Raise both together if
# you switch to a larger model.
MAX_CONTEXT_CHARS = int(os.environ.get("DPDP_MAX_CONTEXT_CHARS", "10000"))
ANSWER_NUM_CTX = int(os.environ.get("DPDP_ANSWER_NUM_CTX", "16384"))

# Expansion order matters. A provision the seed *cites* is almost always more
# load-bearing than a term it merely *mentions* — and MENTIONS is exhaustive by
# construction, so an unranked expansion buries the cited sections under six
# definitions. Lower number wins; the cap keeps the prompt focused.
EXPAND_PRIORITY = {"PENALISED_BY": 0, "PENALISES": 0, "REFERENCES": 1,
                   "CITED_BY": 1, "DEFINES": 2, "HAS_ENTRY": 3, "MENTIONS": 4}
MAX_EXPANDED = 8

# Some edges must be walked backwards as well as forwards. A question about a
# fine lands on the Schedule, and from there the useful hop is *up* to the duty
# that carries the penalty — the opposite direction from how the edge is stored.
REVERSIBLE = {"PENALISED_BY": "PENALISES", "REFERENCES": "CITED_BY"}

SYSTEM = """You answer questions about the Digital Personal Data Protection \
Act, 2023 (India) for people who are not lawyers — compliance staff, engineers, \
product managers, and members of the public.

You are given provisions of the Act, verbatim, retrieved for this question.

Rules:
- Answer ONLY from the provisions supplied. If they do not settle the question, \
say so plainly and name what would.
- Quote the Act's exact words when you state what it requires. Never paraphrase \
a quote inside quotation marks.
- Cite every provision you rely on, in the form §8(5) or Schedule entry 2.
- NEVER state a rupee amount unless you are copying it character for character from the Schedule entry in front of you. If two entries carry different amounts, say which entry you are quoting.
- Write for someone with no legal training. Use "customer" and "your company" \
rather than "Data Principal" and "Data Fiduciary" in your own sentences — but \
keep the Act's terms inside quotes.
- You are not giving legal advice. Where the answer turns on facts you do not \
have, say which facts decide it.

Format your answer exactly like this, omitting any section that does not apply:

Short answer:  one or two sentences.

Why:           the reasoning, in plain words.

The law says:  §N(x) — "<exact quote>"
               (one line per provision, quoting the Act)

What to do:    concrete steps, if the question is about what someone should do.

Penalty:       the Schedule entry and amount, if a penalty was retrieved."""


# Imported rather than redefined: the query and the documents must be tokenized
# by exactly the same function or BM25 silently stops matching.
from index import tokenize  # noqa: E402


def expand_query(query: str, vocab: dict) -> tuple[str, list[str], list[str]]:
    """Rewrite the question into statutory vocabulary before retrieval.

    Deterministic and auditable: every expansion is a line in vocab.yaml, so a
    wrong hit can be traced to a rule and fixed without touching code.
    """
    low = query.lower()
    added, hits = [], []
    for phrase, statutory in sorted(vocab.get("terms", {}).items(), key=lambda kv: -len(kv[0])):
        # Tolerate the usual inflections so vocab.yaml can stay singular:
        # "phone number" matches "phone numbers", "leak" matches "leaked".
        # Irregular forms ("stole") still need their own entry.
        if re.search(rf"\b{re.escape(phrase)}(?:s|es|ed|ing)?\b", low):
            hits.append(phrase)
            added += statutory
    intents = [name for name, cfg in (vocab.get("intents") or {}).items()
               if any(t in low for t in cfg.get("triggers", []))]
    return f"{query} {' '.join(added)}", sorted(set(hits)), intents


def load():
    index = json.loads((OUT / "index.json").read_text(encoding="utf-8"))
    graph = json.loads((OUT / "dpdp_graph.json").read_text(encoding="utf-8"))
    vocab = yaml.safe_load((ROOT / "vocab.yaml").read_text(encoding="utf-8"))
    return index, graph, vocab


def retrieve(index: dict, graph: dict, vocab: dict, query: str, k: int) -> tuple[list, dict]:
    expanded, hits, intents = expand_query(query, vocab)
    bm25 = BM25Okapi([tokenize(d) for d in index["docs"]])
    scores = bm25.get_scores(tokenize(expanded))

    # Intent hints nudge whole kinds or chapters up — "how much is the fine"
    # should reach the Schedule even though it shares few words with it.
    boost_kinds, boost_chapters = set(), set()
    for name in intents:
        cfg = vocab["intents"][name]
        boost_kinds.update(cfg.get("boost_kinds", []))
        if cfg.get("boost_chapter"):
            boost_chapters.add(cfg["boost_chapter"])
    for i, c in enumerate(index["chunks"]):
        if c["kind"] in boost_kinds or c["chapter"] in boost_chapters:
            scores[i] *= 1.6

    ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
    seeds = [index["chunks"][i] | {"score": round(float(scores[i]), 3), "hop": 0}
             for i in ranked[:k] if scores[i] > 0]

    # The Schedule is seven rows. When someone asks about a penalty, ranking
    # them against each other is the wrong problem — the honest answer is the
    # whole table, and it costs a few hundred tokens.
    #
    # Ranking actively fails here: entry 5 (the Data Principal's own duties,
    # ₹10,000) is the only row that never says "Data Principal" or "personal
    # data", so expanding "customer" into those terms pushes the one row about
    # the customer to last place. Completeness beats a cleverer score.
    if "penalty_lookup" in intents:
        chosen = {s["id"] for s in seeds}
        for i, chunk in enumerate(index["chunks"]):
            if chunk["kind"] == "Penalty" and chunk["id"] not in chosen:
                seeds.append(chunk | {"score": round(float(scores[i]), 3), "hop": 0})

    # --- graph expansion --------------------------------------------------- #
    by_node = {c["node_id"]: c for c in index["chunks"]}
    parent = {n["id"]: None for n in graph["nodes"]}
    adj: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for e in graph["links"]:
        if e["type"].startswith("HAS_"):
            parent[e["target"]] = e["source"]
        if e["type"] in EXPAND_PRIORITY:
            adj[e["source"]].append((e["target"], e["type"]))
        if e["type"] in REVERSIBLE:
            adj[e["target"]].append((e["source"], REVERSIBLE[e["type"]]))

    def chunk_for(node_id: str):
        """Walk up to the nearest provision that is itself a chunk — a citation
        to a clause should surface the section that contains it."""
        seen = set()
        while node_id and node_id not in seen:
            if node_id in by_node:
                return by_node[node_id]
            seen.add(node_id)
            node_id = parent.get(node_id)
        return None

    picked = {c["id"]: c for c in seeds}
    candidates = []
    for rank, seed in enumerate(seeds):
        for target, etype in adj.get(seed["node_id"], []):
            neighbour = chunk_for(target)
            if neighbour and neighbour["id"] not in picked:
                candidates.append((EXPAND_PRIORITY[etype], rank, neighbour, seed, etype))

    for _prio, _rank, neighbour, seed, etype in sorted(candidates, key=lambda t: (t[0], t[1])):
        if len(picked) - len(seeds) >= MAX_EXPANDED:
            break
        picked.setdefault(neighbour["id"], neighbour | {
            "score": 0.0, "hop": 1, "via": f"{seed['label']} —{etype}→"})

    results = sorted(picked.values(), key=lambda c: (c["hop"], -c["score"]))
    return results, {"expanded": expanded, "vocab_hits": hits, "intents": intents}


def penalty_facts(results: list[dict], graph: dict) -> list[str]:
    """Read the penalty amounts straight out of the graph.

    A 3B model asked "what is the fine for weak security?" answered "two hundred
    crore" — it took the figure off the adjacent Schedule row. The amount and the
    provision it attaches to are *structured data* that the build already
    resolved and `test_penalty_join` already locks down, so there is no reason to
    let a language model re-derive them. These lines are printed verbatim from
    the graph, whatever the model says above them.
    """
    labels = {n["id"]: n for n in graph["nodes"]}
    duty_of: dict[str, list[str]] = defaultdict(list)
    for e in graph["links"]:
        if e["type"] == "PENALISED_BY":
            duty_of[e["target"]].append(e["source"])

    facts = []
    for r in results:
        if r["kind"] != "Penalty":
            continue
        node = labels.get(r["node_id"], {})
        duties = ", ".join(f"§{d[2:].replace('-', '(') + ')' * d[2:].count('-')}"
                           for d in sorted(duty_of.get(r["node_id"], []))) or "—"
        facts.append(f"  {r['label']:<18} {node.get('penalty', '?')}\n"
                     f"  {'':<18} applies to: {duties}")
    return facts


def build_context(results: list[dict]) -> str:
    parts, total = [], 0
    for c in results:
        block = f"=== {c['label']} ===\n{c['header']}\n---\n{c['verbatim']}\n"
        if total + len(block) > MAX_CONTEXT_CHARS:
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("question", nargs="+")
    ap.add_argument("-k", type=int, default=6, help="seed chunks before expansion")
    ap.add_argument("--retrieval-only", action="store_true",
                    help="show what was retrieved and why; call no model")
    ap.add_argument("--model", help="override DPDP_MODEL for this run")
    args = ap.parse_args()
    if args.model:
        llm.MODEL = args.model
    question = " ".join(args.question)

    index, graph, vocab = load()
    results, trace = retrieve(index, graph, vocab, question, args.k)

    if not results:
        print("nothing retrieved — the question may be outside this Act.")
        return 1

    print(f"question : {question}")
    if trace["vocab_hits"]:
        print(f"vocabulary: {', '.join(trace['vocab_hits'])}")
    if trace["intents"]:
        print(f"intent    : {', '.join(trace['intents'])}")
    print(f"retrieved : {sum(1 for r in results if r['hop'] == 0)} seeds "
          f"+ {sum(1 for r in results if r['hop'] == 1)} via graph")
    for r in results:
        via = f"   ← {r['via']}" if r.get("via") else f"   (bm25 {r['score']})"
        print(f"   [{r['hop']}] {r['label']}{via}")

    if args.retrieval_only:
        return 0

    if error := llm.check():
        print(f"\n{error}\n(retrieval above still works — use --retrieval-only)",
              file=sys.stderr)
        return 1
    llm.warn_if_small()

    context = build_context(results)
    print(f"\nasking {llm.MODEL} ({len(context)} chars of statute)…", flush=True)
    try:
        answer = llm.chat(f"{context}\n\nQuestion: {question}",
                          system=SYSTEM, num_ctx=ANSWER_NUM_CTX, temperature=0.1)
    except RuntimeError as e:
        print(f"\n{e}", file=sys.stderr)
        return 1

    print("\n" + "-" * 70)
    print(str(answer).strip())

    if facts := penalty_facts(results, graph):
        print("\nPenalty amounts, read from the graph (not from the model):")
        print("\n".join(facts))

    print("-" * 70)
    print(f"answered by {llm.MODEL} from {len(results)} retrieved provisions. "
          f"Small models misread figures — trust the block above, and check "
          f"quotes against out/dpdp_tree.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
