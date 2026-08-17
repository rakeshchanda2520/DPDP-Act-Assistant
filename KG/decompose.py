"""
Optional query decomposition for compound questions.

A real compliance question is often several questions wearing one coat:

    "our data processor in another country lost a child's records,
     what do we owe and to whom?"

That touches §9 (children), §16 (processing outside India), §8 (security and
breach duties) and a Schedule entry. Retrieved as ONE query, the BM25 token
set and the dense query vector are both an average of three separate
concerns, and each individual thread matches less well than it would alone —
so the weakest thread (usually the one with the least statutory vocabulary
overlap) drops out entirely.

Decomposition retrieves each sub-question at full strength and merges the
results. Statutory-QA research reports this matters specifically because
multi-issue legal questions "substantially increase the risk of
hallucination" when the supporting provisions are not all retrieved.

Two deliberate design choices:

  * The split is done by the LLM, but the SPLIT IS NOT TRUSTED as legal
    reasoning — it is pattern-matching on sentence structure, which is why a
    small local model is adequate here even though the same model is the
    weak link for answering. Every sub-question still goes through the exact
    same deterministic `ask.retrieve()` as before.
  * The ORIGINAL question is always retrieved too, and always ranked first.
    Decomposition can only ever ADD provisions, never remove one the
    undecomposed query would have found — so a bad split degrades to today's
    behaviour instead of losing a correct result.

Off by default (DPDP_DECOMPOSE=1, or ask.py --decompose): it costs one extra
LLM round-trip before retrieval even starts, which is a real latency cost on
a local model and pointless on simple questions.
"""
from __future__ import annotations

import os
import re

ENABLED = os.environ.get("DPDP_DECOMPOSE", "") not in ("", "0", "false")

# Cheap pre-filter: only pay for an LLM call when the question actually looks
# compound. Costing a round-trip on "who is a data fiduciary?" would be pure
# latency for nothing.
RE_COMPOUND = re.compile(
    r"\band\b|\balso\b|,\s*(?:and\s+)?(?:what|who|how|when|do|does|can|is|are)\b"
    r"|\bwhat do we owe\b|\bto whom\b|\bas well as\b|\bboth\b|;", re.IGNORECASE)

MIN_WORDS = 9          # shorter than this is not a compound question worth splitting
MAX_SUB_QUESTIONS = 4  # a legal question decomposing past this is usually a bad split

SYSTEM = """You split a compound question about a law into the separate \
questions it actually contains, so each can be looked up on its own.

Rules:
- Output ONLY the sub-questions, one per line. No numbering, no preamble, no \
explanation.
- Each sub-question must stand alone and be answerable by itself. Replace \
pronouns with what they refer to.
- Split ONLY where the question genuinely asks about separate topics. If it \
asks one thing, output that one question unchanged.
- Never invent a topic the original question did not raise.
- Output at most 4 lines."""


def looks_compound(question: str) -> bool:
    return len(question.split()) >= MIN_WORDS and bool(RE_COMPOUND.search(question))


def split(question: str, llm_module) -> list[str]:
    """Return sub-questions, ALWAYS including the original first.

    `llm_module` is injected rather than imported so this stays testable
    without a running model, and so the caller's provider/model overrides
    (ask.py's --provider/--model) apply here automatically.

    Any failure — model down, empty output, a "split" that just echoes the
    question — falls back to `[question]`, i.e. exactly today's behaviour.
    """
    if not looks_compound(question):
        return [question]

    try:
        raw = llm_module.chat(question, system=SYSTEM, temperature=0.0)
    except RuntimeError:
        return [question]

    subs = []
    for line in str(raw).splitlines():
        # Models like to number things despite being told not to.
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if len(line.split()) >= 3 and line.lower() != question.lower():
            subs.append(line)

    return [question] + subs[:MAX_SUB_QUESTIONS]


def retrieve_decomposed(retrieve_fn, index, graph, vocab, question, k,
                        parent_doc=False, hybrid=None, llm_module=None):
    """Retrieve for each sub-question and merge, preserving `retrieve()`'s
    exact return shape so callers cannot tell the difference.

    Merge rule: first occurrence wins. Because the original question is
    always retrieved first, its ranking is preserved intact and sub-question
    results only ever append — the property that makes a bad decomposition
    harmless rather than harmful.
    """
    subs = split(question, llm_module) if llm_module else [question]
    if len(subs) == 1:
        return retrieve_fn(index, graph, vocab, question, k, parent_doc, hybrid)

    merged: dict[str, dict] = {}
    trace = {"expanded": "", "vocab_hits": [], "intents": [], "sub_questions": subs}
    for i, sub in enumerate(subs):
        results, sub_trace = retrieve_fn(index, graph, vocab, sub, k, parent_doc, hybrid)
        for r in results:
            if r["id"] not in merged:
                # Record which sub-question surfaced this, so the retrieval
                # trace stays as auditable as the single-query path.
                merged[r["id"]] = r if i == 0 else r | {"via_sub_question": sub}
        if i == 0:
            trace["expanded"] = sub_trace["expanded"]
        trace["vocab_hits"] = sorted(set(trace["vocab_hits"]) | set(sub_trace["vocab_hits"]))
        trace["intents"] = sorted(set(trace["intents"]) | set(sub_trace["intents"]))

    results = sorted(merged.values(), key=lambda c: (c["hop"], -c.get("weight", 0.0), -c["score"]))
    return results, trace
