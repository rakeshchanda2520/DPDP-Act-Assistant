"""
Graph -> searchable chunks.

Chunks follow the Act's own boundaries. The drafters already divided this
document into citable units; a fixed token window would cut clause lists in
half and throw away the structure the parser just recovered.

Long sections get sub-section chunks as well as a whole-section chunk: §8 and
§17 each cover several unrelated duties, and a single section-sized chunk
retrieves the wrong one for a specific question.
"""
from __future__ import annotations

import logging

import networkx as nx

from backend.indexing import Chunk

log = logging.getLogger(__name__)

# Above this word count a section is also split into its sub-sections.
LONG_SECTION_WORDS = 220


def _render(tree: dict, node_id: str, depth: int = 0) -> str:
    """Reassemble a provision and everything beneath it, hierarchy intact."""
    node = tree[node_id]
    lines: list[str] = []
    if node["text"]:
        lines.append("  " * depth + f"{node['prefix']} {node['text']}".strip())
    elif node["prefix"] and node["kind"] != "Section":
        lines.append("  " * depth + node["prefix"])
    for child in node["children"]:
        lines.append(_render(tree, child, depth + 1))
    return "\n".join(line for line in lines if line.strip())


def _header(tree: dict, node: dict, refs: list[str], penalties: list[str]) -> str:
    """Context the provision's own body never states, carried into the chunk
    so it survives both retrieval and the prompt."""
    bits: list[str] = []
    chapter = node.get("chapter")
    if chapter and chapter in tree:
        bits.append(f"[{tree[chapter]['label']}]")
    headnote = node.get("headnote")
    bits.append(node["label"] + (f" — {headnote}" if headnote else ""))
    if refs:
        bits.append("Cites: " + ", ".join(sorted(set(refs))))
    if penalties:
        bits.append("Penalised by: " + ", ".join(sorted(set(penalties))))
    return "\n".join(bits)


def build_chunks(tree: dict, graph: nx.MultiDiGraph, schedule: list[dict]) -> list[Chunk]:
    labels = {n: a.get("label", n) for n, a in graph.nodes(data=True)}
    refs: dict[str, list[str]] = {}
    penalties: dict[str, list[str]] = {}
    for src, dst, attrs in graph.edges(data=True):
        if attrs["type"] == "REFERENCES":
            refs.setdefault(src, []).append(labels.get(dst, dst))
        elif attrs["type"] == "PENALISED_BY":
            penalties.setdefault(src, []).append(labels.get(dst, dst))

    chunks: list[Chunk] = []

    def add(node_id: str, kind: str, body: str) -> None:
        node = tree.get(node_id, {})
        chunks.append(Chunk(
            id=node_id, node_id=node_id, kind=kind,
            label=node.get("label", node_id),
            verbatim=body,
            header=_header(tree, node, refs.get(node_id, []), penalties.get(node_id, [])),
            headnote=node.get("headnote") or "",
            chapter=node.get("chapter") or "",
            page=node.get("page", 0),
        ))

    for node_id, node in tree.items():
        if node["kind"] != "Section":
            continue
        body = _render(tree, node_id)
        add(node_id, "Section", body)
        if len(body.split()) > LONG_SECTION_WORDS:
            for child_id in node["children"]:
                if tree[child_id]["kind"] == "SubSection":
                    add(child_id, "SubSection", _render(tree, child_id))

    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("kind") == "Definition":
            chunks.append(Chunk(
                id=node_id, node_id=node_id, kind="Definition",
                label=f"Definition of “{attrs['label']}”",
                verbatim=attrs.get("text", ""),
                header=f"[Chapter I — PRELIMINARY]\nSection 2 defines “{attrs['label']}”",
                chapter="ch-I",
            ))

    for row in schedule:
        node_id = f"pen-{row['sl_no']}"
        chunks.append(Chunk(
            id=node_id, node_id=node_id, kind="Penalty",
            label=f"Schedule entry {row['sl_no']}",
            verbatim=f"{row['breach']}\nPenalty: {row['penalty']}",
            header=("[THE SCHEDULE — see section 33(1)]\n"
                    f"Entry {row['sl_no']}: monetary penalty"),
            page=21,
        ))

    log.info("chunks: %s", " | ".join(
        f"{kind} {sum(1 for c in chunks if c.kind == kind)}"
        for kind in ("Section", "SubSection", "Definition", "Penalty")))
    return chunks


def attach_plain_language(chunks: list[Chunk], cache: dict) -> list[Chunk]:
    """Merge the generated plain-language layer onto the chunks.

    Kept strictly separate from `verbatim`: this text is a retrieval aid so a
    layperson's phrasing has something to match, and it is never quoted,
    cited, or shown as a statement of the law.
    """
    merged = []
    for chunk in chunks:
        entry = cache.get(chunk.id, {})
        merged.append(Chunk(
            **{**chunk.to_dict(),
               "plain_english": entry.get("plain_english", ""),
               "questions": list(entry.get("questions", []))}))
    return merged
