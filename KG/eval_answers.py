"""
Answer-quality eval — the half of correctness eval.yaml doesn't cover.

eval.yaml proves the right provision was RETRIEVED. Nothing checked, until
this file, that the model then USED it correctly. Retrieval finding s-14
cleanly does not stop a model from writing "family can step in" when the text
says "nominate ... any other individual" — that error passed every check in
eval.yaml and reached a user. See answer_eval.yaml for the case set and the
full rationale.

Every check is deterministic: string and graph comparisons against the same
verbatim tree/graph the rest of the system trusts. No judge model, no extra
API cost beyond the one answer call per case. Costs real time and (with a
hosted provider) real money, so this is a separate command from
test_build.py, not folded into the build.

Run:  python eval_answers.py
      python eval_answers.py --provider claude --model claude-sonnet-5
      python eval_answers.py -k "s-14"     # substring-filter cases by question
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

import ask
import llm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent

RE_QUOTE = re.compile(r"[“\"]([^”\"]{8,})[”\"]")
RE_RUPEE = re.compile(
    r"((?:[a-z\s-]*(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
    r"hundred|thousand|lakh|crore)[a-z\s-]*|\d[\d,]*)\s*(?:rupees|crore|lakh))",
    re.IGNORECASE)


def normalise(text: str) -> str:
    """Collapse whitespace/quote-style differences that aren't substantive."""
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip().lower()


def check_quotes_exact(answer: str, context: str) -> list[str]:
    """Every quoted span in the answer must appear verbatim in the context the
    model was actually given. A paraphrase inside quotation marks is the most
    dangerous formatting failure this system can produce — it looks like a
    citation.

    Checked against the full context block, not a per-citation tree lookup:
    `tree[node_id]["text"]` is empty for Section-kind nodes (their content
    lives in child nodes and is only assembled at chunk-render time), so a
    citation to a section would falsely appear to have no matchable text.
    The context block is what was actually rendered and actually shown to the
    model, so it is the correct — and simpler — source of truth here."""
    haystack = normalise(context)
    problems = []
    for q in RE_QUOTE.findall(answer):
        if normalise(q) not in haystack:
            problems.append(q.strip())
    return problems


def check_rupee_figures(answer: str, penalty_nodes: dict[str, str]) -> list[str]:
    """Every rupee figure the model states must match the graph's own wording
    for some retrieved Penalty node. Catches the ₹200cr-for-₹250cr error class
    without trusting the model to get arithmetic or reading right."""
    known = {normalise(v) for v in penalty_nodes.values()}
    problems = []
    for m in RE_RUPEE.findall(answer):
        if not any(normalise(m) in k or k in normalise(m) for k in known):
            problems.append(m.strip())
    return problems


def load_answer_cases() -> list[dict]:
    return yaml.safe_load((ROOT / "answer_eval.yaml").read_text(encoding="utf-8"))["cases"]


def run_case(index, graph, vocab, tree, case: dict) -> dict:
    # Reused, not duplicated: this must test what production actually does,
    # not a stripped-down version of it. In particular, api.should_abstain is
    # a deterministic gate in front of generation — measuring the model's OWN
    # abstention judgement instead would understate how the real system
    # behaves on out-of-scope questions, which defeats the point of this file.
    from api import RE_CITATION, citation_id, label_for, should_abstain

    results, _trace = ask.retrieve(index, graph, vocab, case["q"], case.get("k", 6))
    retrieved_ids = {r["node_id"] for r in results}
    context = ask.build_context(results)

    if reason := should_abstain(results):
        if case.get("abstain"):
            return {"q": case["q"], "ok": True, "problems": [], "answer": f"[abstained: {reason}]"}
        return {"q": case["q"], "ok": False,
               "problems": [f"gate abstained when an answer was expected: {reason}"],
               "answer": f"[abstained: {reason}]"}

    try:
        answer = str(llm.chat(f"{context}\n\nQuestion: {case['q']}",
                              system=ask.SYSTEM, temperature=0.1))
    except RuntimeError as e:
        return {"q": case["q"], "ok": False, "problems": [f"generation failed: {e}"]}

    # Resolve every citation the same way api.py does for the live UI.
    nodes_by_id = {n["id"]: n for n in graph["nodes"]}
    citations = []
    for match in RE_CITATION.finditer(answer):
        section, parts, schedule = match.groups()
        node_id = f"pen-{schedule}" if schedule else citation_id(section, parts)
        source = tree.get(node_id) or nodes_by_id.get(node_id)
        in_context = node_id in retrieved_ids or any(
            r.startswith(node_id + "-") or node_id.startswith(r + "-") for r in retrieved_ids)
        status = "unresolved" if source is None else ("verified" if in_context else "out_of_context")
        citations.append({"id": node_id, "status": status})

    problems = []

    if case.get("abstain"):
        if citations:
            problems.append(f"expected abstention, but cited {[c['id'] for c in citations]}")
        if RE_RUPEE.search(answer):
            problems.append("expected abstention, but stated a rupee figure")
    else:
        cited_ids = {c["id"] for c in citations}
        unresolved = [c["id"] for c in citations if c["status"] == "unresolved"]
        if unresolved:
            problems.append(f"unresolved citations (invented): {unresolved}")

        if wants := case.get("cites_any"):
            if not any(any(w == cid or cid.startswith(w + "-") for cid in cited_ids) for w in wants):
                problems.append(f"cited none of {wants} (cited {sorted(cited_ids)})")

        if entries := case.get("penalty_entries"):
            penalty_nodes = {n["id"]: n.get("penalty", "") for n in graph["nodes"]
                             if n["id"].startswith("pen-")
                             and int(n["id"][4:]) in entries}
            if bad := check_rupee_figures(answer, penalty_nodes):
                problems.append(f"rupee figure not matching entry {entries}: {bad}")

        if bad := check_quotes_exact(answer, context):
            problems.append(f"quote not verbatim in the retrieved context: {bad}")

    for phrase in case.get("must_say", []):
        if phrase.lower() not in answer.lower():
            problems.append(f"must contain {phrase!r}")
    for phrase in case.get("must_not_say", []):
        if phrase.lower() in answer.lower():
            problems.append(f"must NOT contain {phrase!r}")

    return {"q": case["q"], "ok": not problems, "problems": problems, "answer": answer}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-k", "--filter", default="", help="only run cases whose question contains this substring")
    ap.add_argument("--provider", choices=["ollama", "claude"])
    ap.add_argument("--model")
    ap.add_argument("-v", "--verbose", action="store_true", help="print full answers")
    args = ap.parse_args()

    if args.provider:
        llm.PROVIDER = args.provider
        if not args.model:
            llm.MODEL = llm._DEFAULT_MODEL[args.provider]
    if args.model:
        llm.MODEL = args.model

    if error := llm.check():
        print(error, file=sys.stderr)
        return 1

    index, graph, vocab = ask.load()
    import json
    tree = json.loads((ROOT / "out" / "dpdp_tree.json").read_text(encoding="utf-8"))

    cases = [c for c in load_answer_cases() if args.filter.lower() in c["q"].lower()]
    print(f"running {len(cases)} answer-quality cases via {llm.PROVIDER}/{llm.MODEL}\n")

    passed = 0
    for i, case in enumerate(cases, 1):
        result = run_case(index, graph, vocab, tree, case)
        mark = "PASS" if result["ok"] else "FAIL"
        print(f"[{i}/{len(cases)}] {mark}  {case['q'][:70]}")
        if not result["ok"]:
            for p in result["problems"]:
                print(f"          - {p}")
        if args.verbose:
            print(f"          answer: {result.get('answer', '')[:300]}")
        passed += result["ok"]

    print(f"\n{passed}/{len(cases)} passed")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
