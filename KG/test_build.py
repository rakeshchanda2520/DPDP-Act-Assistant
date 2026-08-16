"""
Checks that fail if the extraction drifts.

The valuable ones are not the counts — they are `test_lossless_round_trip`
(nothing was invented, lost or reordered) and `test_penalty_join` (the edge a
flat vector store cannot make). Everything else is a tripwire.

Run:  python test_build.py        (no framework needed)
      pytest test_build.py        (also works)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import build

OUT = Path(__file__).parent / "out"


def load(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #

def test_section_numbering_is_complete():
    tree = load("dpdp_tree.json")
    nums = sorted(int(k.split("-")[1]) for k, v in tree.items() if v["kind"] == "Section")
    assert nums == list(range(1, 45)), f"sections are not 1..44: {nums}"


def test_chapters_and_illustrations():
    tree = load("dpdp_tree.json")
    kinds = {}
    for v in tree.values():
        kinds[v["kind"]] = kinds.get(v["kind"], 0) + 1
    assert kinds["Chapter"] == 9, kinds
    assert kinds["Illustration"] == 11, kinds
    assert kinds["Section"] == 44, kinds


def test_every_section_sits_in_a_chapter_and_has_a_headnote():
    tree = load("dpdp_tree.json")
    for k, v in tree.items():
        if v["kind"] == "Section":
            assert v["chapter"], f"{k} has no chapter"
            assert v["headnote"], f"{k} has no headnote recovered from the margin"


def test_section_2_defines_28_terms():
    """(a) through (zb). If this drops, the letter-i clause regressed into a
    roman sub-clause — the ambiguity the indentation ladder exists to resolve."""
    tree = load("dpdp_tree.json")
    clauses = [k for k, v in tree.items() if v["parent"] == "s-2"]
    assert len(clauses) == 28, f"section 2 has {len(clauses)} clauses, expected 28"
    assert "s-2-i" in clauses and "s-2-zb" in clauses


def test_sub_clauses_survive():
    """`(i)` matches the clause pattern as well as the roman one. If depth
    disambiguation breaks, sub-clauses silently become continuation text."""
    tree = load("dpdp_tree.json")
    n = sum(1 for v in tree.values() if v["kind"] == "SubClause")
    assert n >= 30, f"only {n} sub-clauses — the (i) ambiguity has regressed"


# --------------------------------------------------------------------------- #
# Verbatim guarantee
# --------------------------------------------------------------------------- #

def test_lossless_round_trip():
    """The whole document, reassembled from the graph, must equal the extracted
    body stream character for character. This is the guarantee that nothing was
    invented, dropped or reordered."""
    report = (Path(__file__).parent / "review" / "roundtrip.txt").read_text(encoding="utf-8")
    assert "identical    : True" in report, report


def test_no_marginalia_leaked_into_body():
    """Headnotes live in the margin. If the column split drifts, they appear
    mid-sentence in the operative text."""
    tree = load("dpdp_tree.json")
    for k, v in tree.items():
        if v["headnote"] and v["text"]:
            assert v["headnote"] not in v["text"], f"headnote leaked into {k}"


def test_no_running_headers_in_any_node():
    tree = load("dpdp_tree.json")
    for k, v in tree.items():
        assert "GAZETTE OF INDIA" not in (v["text"] or ""), f"page header leaked into {k}"


# --------------------------------------------------------------------------- #
# The Schedule and the join that makes the graph worth building
# --------------------------------------------------------------------------- #

def test_schedule_recovered_from_the_table():
    schedule = load("schedule.json")
    assert [r["sl_no"] for r in schedule] == [1, 2, 3, 4, 5, 6, 7]
    for row in schedule:
        assert row["breach"].startswith("Breach"), row
        assert row["penalty"], row
    assert "two hundred and fifty crore" in schedule[0]["penalty"]
    assert "ten thousand" in schedule[4]["penalty"]


def test_penalty_join():
    """Section 8(5) requires security safeguards and names no amount. Schedule
    entry 1 names 250 crore and never says "security". The PENALISED_BY edge is
    the join, and it is the single clearest reason this is a graph."""
    graph = load("dpdp_graph.json")
    edges = {(e["source"], e["target"]) for e in graph["links"] if e["type"] == "PENALISED_BY"}
    assert ("s-8-5", "pen-1") in edges, "s-8(5) -> Schedule 1 missing"
    assert ("s-8-6", "pen-2") in edges, "s-8(6) -> Schedule 2 missing"
    assert ("s-9", "pen-3") in edges, "s-9 (children) -> Schedule 3 missing"
    assert ("s-10", "pen-4") in edges, "s-10 (SDF) -> Schedule 4 missing"
    assert ("s-15", "pen-5") in edges, "s-15 (duties) -> Schedule 5 missing"


def test_cross_references_all_resolve():
    graph = load("dpdp_graph.json")
    ids = {n["id"] for n in graph["nodes"]}
    refs = [e for e in graph["links"] if e["type"] == "REFERENCES"]
    assert len(refs) > 50, f"only {len(refs)} cross-references — the scanner regressed"
    for e in refs:
        assert e["target"] in ids, f"dangling reference {e['source']} -> {e['target']}"


def test_external_acts_are_not_linked_to_dpdp_sections():
    """Section 44 amends four other statutes. "section 81" there means the IT
    Act, not this Act — linking it internally would be a wrong answer with a
    confident citation."""
    graph = load("dpdp_graph.json")
    ids = {n["id"] for n in graph["nodes"]}
    assert "s-81" not in ids and "s-87" not in ids
    crossrefs = (Path(__file__).parent / "review" / "crossrefs.md").read_text(encoding="utf-8")
    assert "Information Technology Act, 2000" in crossrefs
    assert "None." in crossrefs.split("## UNRESOLVED")[-1], "unresolved references remain"


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #

def test_retrieval_finds_the_penalty_for_a_security_failure():
    """The end-to-end question a flat vector store gets wrong."""
    if not (OUT / "index.json").exists():
        print("  (skipped: run `python index.py` first)")
        return
    import ask
    index, graph, vocab = ask.load()
    results, _trace = ask.retrieve(index, graph, vocab,
                                   "what is the fine if customer data leaks?", k=6)
    labels = {r["label"] for r in results}
    assert any(l.startswith("Schedule entry") for l in labels), labels
    assert any("Section 8" in l for l in labels), labels


def test_retrieval_eval_set():
    """Score eval.yaml and fail below its floor.

    This is the guard that stops retrieval tuning from being guesswork: without
    it, adding a vocab entry to fix one question can quietly break three others
    and nobody finds out. Cases marked `known_miss` are reported but excluded
    from the floor — they are the honest backlog, not silent failures.
    """
    if not (OUT / "index.json").exists():
        print("  (skipped: run `python index.py` first)")
        return
    import ask
    spec = build.yaml.safe_load(
        (Path(__file__).parent / "eval.yaml").read_text(encoding="utf-8"))
    index, graph, vocab = ask.load()

    def satisfied(want: str, got: list[str]) -> bool:
        """A sub-section counts as a hit for its section.

        Asking "do we have to tell customers what we collect?" and getting
        §5(1) is a better result than getting all of §5, not a worse one — so
        `want: s-5` is satisfied by `s-5` or any `s-5-*`.
        """
        return any(g == want or g.startswith(want + "-") for g in got)

    hits, misses, known = 0, [], []
    for case in spec["cases"]:
        results, _trace = ask.retrieve(index, graph, vocab, case["q"], case.get("k", 6))
        seeds = [r["id"] for r in results if r["hop"] == 0]
        wants = case["want"] if isinstance(case["want"], list) else [case["want"]]
        found = any(satisfied(w, seeds) for w in wants)
        if found:
            hits += 1
        elif case.get("known_miss"):
            known.append(case["q"])
        else:
            misses.append(f"{case['want']} <- {case['q']}")

    print(f"  ...  eval {hits}/{len(spec['cases'])} "
          f"(floor {spec['floor']}, {len(known)} known misses)")
    for m in misses:
        print(f"       MISS {m}")
    assert hits >= spec["floor"], (
        f"retrieval scored {hits}, below the floor of {spec['floor']}. "
        f"New misses: {misses}")


def test_vocabulary_bridge_translates_layperson_words():
    vocab = build.yaml.safe_load(
        (Path(__file__).parent / "vocab.yaml").read_text(encoding="utf-8"))
    import ask
    expanded, hits, _intents = ask.expand_query(
        "a hacker stole our customers phone numbers", vocab)
    assert "personal data breach" in expanded, expanded
    assert "Data Principal" in expanded, expanded


# --------------------------------------------------------------------------- #

def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed.append((name, str(e)))
            print(f"  FAIL  {name}\n        {e}")
        except Exception as e:  # noqa: BLE001 - a missing artefact is a test failure
            failed.append((name, repr(e)))
            print(f"  ERROR {name}\n        {e!r}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
