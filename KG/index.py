"""
Turn the verbatim graph into a searchable index.

Two layers, kept strictly apart:

  VERBATIM   Section / sub-section / definition / penalty chunks whose text is
             the Act's own words. This is what an answer may quote.

  PLAIN      An LLM-written plain-English gloss and a set of layperson
             questions for each chunk. Never quoted, never cited, never shown
             as law. It exists only so that "can I text my customers an offer?"
             lands on section 6 instead of missing entirely.

The plain layer is what makes this usable by people who don't read statutes:
the vocabulary gap is bridged at index time, once, instead of at query time on
every question.

Run:  python index.py            # verbatim chunks + BM25 only (no LLM)
      python index.py --plain    # also generate the plain-language layer

The plain layer runs on a local Ollama model — see llm.py.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import snowballstemmer
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

LONG_SECTION_TOKENS = 220     # sections above this also get per-sub-section chunks


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #

def render(tree: dict, node_id: str, depth: int = 0) -> str:
    """Reassemble a provision and everything beneath it, hierarchy intact.

    Chunking by token count would cut clause lists in half and throw away the
    structure the parser just recovered, so chunks follow the Act's own
    boundaries instead.
    """
    node = tree[node_id]
    lines = []
    if node["text"]:
        lines.append("  " * depth + f"{node['prefix']} {node['text']}".strip())
    elif node["prefix"] and node["kind"] != "Section":
        lines.append("  " * depth + node["prefix"])
    for child in node["children"]:
        lines.append(render(tree, child, depth + 1))
    return "\n".join(l for l in lines if l.strip())


def chunk_header(tree: dict, node: dict, refs: list[str], penalties: list[str]) -> str:
    """Context the provision's own body never states, carried into the chunk so
    it survives both retrieval and the prompt."""
    bits = []
    if node.get("chapter") and node["chapter"] in tree:
        bits.append(f"[{tree[node['chapter']]['label']}]")
    bits.append(node["label"] + (f" — {node['headnote']}" if node.get("headnote") else ""))
    if refs:
        bits.append("Cites: " + ", ".join(sorted(set(refs))))
    if penalties:
        bits.append("Penalised by: " + ", ".join(sorted(set(penalties))))
    return "\n".join(bits)


def build_chunks(tree: dict, graph: dict, schedule: list[dict]) -> list[dict]:
    refs: dict[str, list[str]] = {}
    pens: dict[str, list[str]] = {}
    labels = {n["id"]: n.get("label", n["id"]) for n in graph["nodes"]}
    for e in graph["links"]:
        if e["type"] == "REFERENCES":
            refs.setdefault(e["source"], []).append(labels.get(e["target"], e["target"]))
        elif e["type"] == "PENALISED_BY":
            pens.setdefault(e["source"], []).append(labels.get(e["target"], e["target"]))

    chunks: list[dict] = []

    def add(node_id: str, kind: str, body: str) -> None:
        node = tree.get(node_id, {})
        header = chunk_header(tree, node, refs.get(node_id, []), pens.get(node_id, []))
        chunks.append({
            "id": node_id, "kind": kind, "node_id": node_id,
            "label": node.get("label", node_id),
            "headnote": node.get("headnote") or "",
            "chapter": node.get("chapter") or "",
            "page": node.get("page", 0),
            "verbatim": body,
            "header": header,
        })

    for nid, node in tree.items():
        if node["kind"] == "Section":
            body = render(tree, nid)
            add(nid, "Section", body)
            # Long sections also get sub-section chunks: §8 and §17 each cover
            # several unrelated duties, and a whole-section chunk retrieves the
            # wrong one for a specific question.
            if len(body.split()) > LONG_SECTION_TOKENS:
                for cid in node["children"]:
                    if tree[cid]["kind"] == "SubSection":
                        add(cid, "SubSection", render(tree, cid))

    for n in graph["nodes"]:
        if n.get("kind") == "Definition":
            chunks.append({
                "id": n["id"], "kind": "Definition", "node_id": n["id"],
                "label": f"Definition of “{n['label']}”", "headnote": "", "chapter": "ch-I",
                "page": 0, "verbatim": n.get("text", ""),
                "header": f"[Chapter I — PRELIMINARY]\nSection 2 defines “{n['label']}”",
            })

    for row in schedule:
        rid = f"pen-{row['sl_no']}"
        chunks.append({
            "id": rid, "kind": "Penalty", "node_id": rid,
            "label": f"Schedule entry {row['sl_no']}", "headnote": "", "chapter": "",
            "page": 21,
            "verbatim": f"{row['breach']}\nPenalty: {row['penalty']}",
            "header": "[THE SCHEDULE — see section 33(1)]\n"
                      f"Entry {row['sl_no']}: monetary penalty",
        })

    return chunks


# --------------------------------------------------------------------------- #
# Plain-language layer (LLM — clearly separated, never citable)
# --------------------------------------------------------------------------- #

PLAIN_SYSTEM = """You are helping index the Digital Personal Data Protection Act, 2023 \
so that non-lawyers can find the right provision.

You will be given one provision, verbatim. Produce two things:

1. plain_english — 1-3 sentences explaining what this provision actually requires \
or permits, in the words an ordinary business person would use. Say "customer" \
rather than "Data Principal" where that is what is meant. No legal citations. \
Do not add requirements the text does not contain.

2. questions — 8 questions a non-lawyer would realistically ask that THIS \
provision answers. Write them the way a bank operations manager, a startup \
founder, or a customer would actually type them. Use everyday words: "leak" not \
"personal data breach", "delete my data" not "erasure". Vary the phrasing. \
Do not use section numbers.

These are retrieval aids only. They are never shown to anyone as a statement of \
the law — the verbatim text is always what gets quoted."""


PLAIN_SCHEMA = {
    "type": "object",
    "properties": {
        "plain_english": {"type": "string"},
        "questions": {"type": "array", "items": {"type": "string"},
                      "minItems": 6, "maxItems": 10},
    },
    "required": ["plain_english", "questions"],
}


def generate_plain(chunks: list[dict], cache_path: Path) -> dict:
    """One local call per chunk, cached to disk after every result.

    Incremental: the cache is written after every result, so an interrupted run
    resumes exactly where it stopped.

    Runs at temperature 0. This file is a retrieval index, and at 0.3 two runs
    over the same 142 chunks produced different question sets that scored two
    points apart on eval.yaml — regenerating it must not quietly move retrieval
    quality.

    The provision text is truncated for this step only. Section 2 is 28
    definitions in one chunk and took ten minutes on a 3B model; the gloss and
    the questions only need the topic, and every definition already has its own
    chunk. Retrieval and answering still see the full verbatim text.
    """
    PLAIN_INPUT_CHARS = 2500
    if error := llm.check():
        print(f"cannot generate the plain-language layer:\n  {error}", file=sys.stderr)
        return json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    llm.warn_if_small()

    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    todo = [c for c in chunks if c["id"] not in cache]
    print(f"plain-language layer via {llm.MODEL}: "
          f"{len(cache)} cached, {len(todo)} to generate")

    start = time.time()
    for i, chunk in enumerate(todo, 1):
        try:
            body = chunk["verbatim"][:PLAIN_INPUT_CHARS]
            result = llm.chat(f"{chunk['header']}\n\n{body}",
                              system=PLAIN_SYSTEM, schema=PLAIN_SCHEMA, temperature=0.0)
            if not isinstance(result, dict) or not result.get("questions"):
                raise RuntimeError("no questions returned")
            cache[chunk["id"]] = {"plain_english": str(result.get("plain_english", "")),
                                  "questions": [str(q) for q in result["questions"]]}
        except RuntimeError as e:
            print(f"  ! {chunk['id']}: {e}", file=sys.stderr)
            continue
        # Write through on every success — a long local run must be resumable.
        # Atomically: this file is rewritten every few seconds for hours, and a
        # plain write_text truncates first, so a reader (or a crash) in that
        # window sees a half-written cache.
        tmp = cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, cache_path)
        rate = (time.time() - start) / i
        print(f"  {i}/{len(todo)}  {chunk['label'][:48]:<48} "
              f"eta {(len(todo) - i) * rate / 60:4.1f}m", flush=True)

    return cache


# --------------------------------------------------------------------------- #
# Search index
# --------------------------------------------------------------------------- #

_STEMMER = snowballstemmer.stemmer("english")


def stem(token: str) -> str | None:
    """Snowball stem, or None if the token should be left alone.

    Hand-rolled suffix stripping got this wrong: "died" and "dies" are both
    four letters, so any rule conservative enough to protect "data" and "use"
    also refused to touch them, and the two never matched each other. Snowball
    maps both to "die".

    The `isalpha` guard is the important part — "8(5)", "250" and "2023" are
    exactly the tokens a legal question turns on, and they must survive intact.
    """
    if not token.isalpha() or len(token) <= 3:
        return None
    stemmed = _STEMMER.stemWord(token)
    return stemmed if stemmed != token else None


def tokenize(text: str) -> list[str]:
    """Tokens for BM25, with each word indexed under both its surface form and
    its stem.

    Keeping both means an exact match still scores highest while "someone died"
    can still reach a question written as "what happens when a customer dies".
    Section numbers and rupee figures survive untouched — `stem` refuses to
    touch anything with a digit in it.
    """
    words = re.findall(r"[a-z0-9]+(?:\([a-z0-9]+\))?", text.lower())
    return words + [s for w in words if (s := stem(w))]


def build_index(chunks: list[dict], plain: dict) -> dict:
    docs = []
    for c in chunks:
        p = plain.get(c["id"], {})
        # The searchable document is verbatim text PLUS the plain-language layer:
        # a user's own words can match the questions even when they match nothing
        # in the statute itself.
        docs.append(" ".join([
            c["label"], c["headnote"], c["header"], c["verbatim"],
            p.get("plain_english", ""), " ".join(p.get("questions", [])),
        ]))
    return {
        "chunks": [{**c, **plain.get(c["id"], {})} for c in chunks],
        "docs": docs,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plain", action="store_true",
                    help="generate the plain-language layer via Ollama")
    args = ap.parse_args()

    tree = json.loads((OUT / "dpdp_tree.json").read_text(encoding="utf-8"))
    graph = json.loads((OUT / "dpdp_graph.json").read_text(encoding="utf-8"))
    schedule = json.loads((OUT / "schedule.json").read_text(encoding="utf-8"))

    chunks = build_chunks(tree, graph, schedule)
    print("chunks: " + " | ".join(
        f"{k} {sum(1 for c in chunks if c['kind'] == k)}"
        for k in ("Section", "SubSection", "Definition", "Penalty")))

    # NOT under out/: that directory is disposable and gets wiped on a clean
    # rebuild, and this file costs hours of local inference to regenerate.
    cache_path = ROOT / "plain_language.json"
    plain = generate_plain(chunks, cache_path) if args.plain else (
        json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {})
    if not plain:
        print("no plain-language layer — run `python index.py --plain` for "
              "layperson-question matching")

    index = build_index(chunks, plain)
    (OUT / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
    # Sanity: the index must be able to find its own corpus.
    bm25 = BM25Okapi([tokenize(d) for d in index["docs"]])
    top = index["chunks"][max(range(len(index["docs"])),
                              key=lambda i: bm25.get_scores(tokenize("personal data breach"))[i])]
    print(f"wrote out/index.json ({len(chunks)} chunks); "
          f"smoke query 'personal data breach' -> {top['label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
