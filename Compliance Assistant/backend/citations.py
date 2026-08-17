"""
Citation verification — the point of the whole system.

A language model can write "§8(5)" whether or not §8(5) says what it claims.
Every citation in an answer is therefore resolved against the graph and
labelled:

    verified        the provision exists AND was in the retrieved context
    out_of_context  it exists but was NOT retrieved — recalled from training
                    rather than read. Treat with suspicion.
    unresolved      no such provision. The model invented it.

Penalty amounts bypass the model entirely: they are read from the graph and
rendered directly. A small model has already misread a Schedule figure in
this corpus, and an amount is structured data the build already resolved —
there is no reason to let a model restate it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .graph_store import Graph

# "§8(5)", "§ 8 (5)(a)", "section 8(5)", "Schedule entry 2", "Schedule 2".
RE_CITATION = re.compile(
    r"(?:§\s*|\bsections?\s+)(\d{1,2})((?:\s*\(\s*[0-9a-zA-Z]{1,3}\s*\))*)"
    r"|\bSchedule\s+(?:entry\s+)?(\d)\b",
    re.IGNORECASE)
RE_PART = re.compile(r"\(\s*([0-9a-zA-Z]{1,3})\s*\)")

STATUS_ORDER = {"unresolved": 0, "out_of_context": 1, "verified": 2}


@dataclass
class Citation:
    id: str
    label: str
    status: str
    text: str = ""
    headnote: str = ""
    note: str = ""

    def to_dict(self) -> dict:
        return {"id": self.id, "label": self.label, "status": self.status,
                "text": self.text, "headnote": self.headnote, "note": self.note}


def node_id_for(section: str, parts: str) -> str:
    return "-".join(["s", section] + RE_PART.findall(parts or ""))


def label_for(node_id: str) -> str:
    if node_id.startswith("pen-"):
        return f"Schedule entry {node_id[4:]}"
    bits = node_id.split("-")[1:]
    if not bits:
        return node_id
    return f"§{bits[0]}" + "".join(f"({b})" for b in bits[1:])


def _in_context(node_id: str, retrieved: set[str]) -> bool:
    """A citation counts as retrieved if the exact node was retrieved, or if
    retrieval covered a parent or child of it — quoting §8(5) from a chunk
    that held all of §8 is not an out-of-context citation."""
    return node_id in retrieved or any(
        r.startswith(node_id + "-") or node_id.startswith(r + "-")
        for r in retrieved)


def check(answer: str, retrieved_node_ids: set[str], graph: Graph) -> list[Citation]:
    """Resolve every citation the answer makes. Ordered worst-first, so a
    reader sees what needs attention before what is fine."""
    seen: dict[str, Citation] = {}

    for match in RE_CITATION.finditer(answer):
        section, parts, schedule = match.groups()
        node_id = f"pen-{schedule}" if schedule else node_id_for(section, parts)
        if node_id in seen:
            continue

        provision = graph.provisions.get(node_id)
        if provision is None:
            # A model writing §8(5)(z) is still pointing at §8(5); naming the
            # nearest real provision is more useful than "invented".
            parts_ = node_id.split("-")
            parent = "-".join(parts_[:-1])
            note = ("no such provision in this Act"
                    if len(parts_) <= 2 or parent not in graph.provisions
                    else f"no such provision; nearest is {label_for(parent)}")
            seen[node_id] = Citation(node_id, label_for(node_id), "unresolved", note=note)
            continue

        text = provision.text
        if node_id.startswith("pen-"):
            # A Schedule row is only meaningful with its amount attached.
            text = f"{provision.text}  —  {provision.penalty}".strip(" —")

        verified = _in_context(node_id, retrieved_node_ids)
        seen[node_id] = Citation(
            id=node_id,
            label=label_for(node_id),
            status="verified" if verified else "out_of_context",
            text=text.strip(),
            headnote=provision.headnote,
            note="" if verified else
                 "this provision exists but was not retrieved for this question",
        )

    return sorted(seen.values(), key=lambda c: (STATUS_ORDER[c.status], c.id))


def penalty_facts(results, graph: Graph) -> list[dict]:
    """Penalty amounts read from the graph, never from the model."""
    duty_of = graph.penalised_by()
    facts = []
    for r in results:
        if r.chunk.kind != "Penalty":
            continue
        provision = graph.provisions.get(r.chunk.node_id)
        if provision is None:
            continue
        facts.append({
            "entry": r.chunk.label,
            "amount": provision.penalty,
            "applies_to": [label_for(d) for d in sorted(duty_of.get(r.chunk.node_id, []))],
        })
    return facts
