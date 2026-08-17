"""
Tree -> graph, and graph -> Neo4j.

networkx exists only in this module and only at build time. It is the right
tool for constructing the graph and computing PageRank once; it has no place
in the serving path, where Neo4j is the source of truth.

The edge that justifies the whole graph is PENALISED_BY: §8(5) states a duty
and carries no rupee figure, the Schedule row carries the figure and describes
no duty. No amount of text similarity connects them — the join is structural.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict

import networkx as nx

# The graph is built from the parser's own types and patterns; importing them
# keeps one definition of each rather than a drifting copy.
from .extract import (RE_ACT_NUMBER, RE_ASSENT_DATE, RE_EXTERNAL_CITATION,
                      PDF, Tree, scan_references, slug)

log = logging.getLogger(__name__)

# Weights for the authority computation. REFERENCES is a drafted, deliberate
# cross-reference and is the real signal of what the Act treats as
# load-bearing. PENALISED_BY is a genuine structural join and counts for half.
#
# MENTIONS is EXCLUDED, and that was measured rather than assumed: at any
# weight it makes every top-authority node a Definition, because all 605
# MENTIONS edges point *into* the 28 definitions. That measures how often a
# term is used, not how load-bearing a provision is — the opposite of what
# retrieval needs. Excluded, the ranking is immediately sensible: §29(1)
# (right of appeal), §33 (penalties), §6 (consent), the Schedule.
AUTHORITY_WEIGHTS = {"REFERENCES": 1.0, "PENALISED_BY": 0.5}

# Node properties Neo4j should carry. An explicit allow-list: the original
# push filtered by `isinstance(v, (str, int, bool))`, which silently dropped
# `authority` because it is a float — the property existed in the build and
# never reached the database.
NODE_PROPERTIES = ("id", "kind", "label", "text", "headnote", "chapter",
                   "penalty", "page", "authority", "verbatim", "relation")


def add_authority(graph: nx.MultiDiGraph) -> None:
    """Store a PageRank score per node, over the Act's own citation structure.

    Centrality predicts practical importance better than a raw count of
    inbound references: a provision cited by heavily-cited provisions matters
    more than one cited as often by peripheral ones. Computed once here and
    consumed at query time as a tie-break within an expansion priority tier.

    Structural HAS_* edges are excluded deliberately — they encode
    containment, not endorsement, and including them would just rediscover
    the document outline and rank long sections above important ones.
    """
    weighted = nx.DiGraph()
    weighted.add_nodes_from(graph.nodes())
    for src, dst, data in graph.edges(data=True):
        weight = AUTHORITY_WEIGHTS.get(data.get("type"))
        if not weight:
            continue
        if weighted.has_edge(src, dst):
            weighted[src][dst]["weight"] += weight
        else:
            weighted.add_edge(src, dst, weight=weight)

    scores = nx.pagerank(weighted, weight="weight") if weighted.number_of_edges() else {}
    for node_id in graph.nodes():
        graph.nodes[node_id]["authority"] = round(scores.get(node_id, 0.0), 6)


def push_to_neo4j(graph: nx.MultiDiGraph, uri: str, user: str, password: str,
                  database: str = "neo4j") -> tuple[int, int]:
    """Replace the graph in Neo4j with this build.

    Batched with UNWIND rather than one statement per element: 404 nodes and
    1088 edges as ~1500 round trips is slow for no reason.

    On interpolation: relationship types and labels cannot be parameterised
    in Cypher, so they are formatted into the query string. They are validated
    against a strict identifier pattern first — they originate from this
    build, but a value reaching a query string unchecked is exactly how
    injection happens when someone later makes them configurable. Every
    *value* is parameterised.
    """
    from neo4j import GraphDatabase

    safe = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

    nodes = []
    for node_id, attrs in graph.nodes(data=True):
        row = {"id": node_id}
        for key in NODE_PROPERTIES:
            if key == "id":
                continue
            value = attrs.get(key)
            if isinstance(value, (str, int, float, bool)):
                row[key] = value
        nodes.append(row)

    by_type: dict[str, list[dict]] = defaultdict(list)
    for src, dst, attrs in graph.edges(data=True):
        by_type[attrs["type"]].append({
            "src": src, "dst": dst,
            "quote": attrs.get("quote", ""),
            "resolution": attrs.get("resolution", ""),
        })

    by_kind: dict[str, list[dict]] = defaultdict(list)
    for row in nodes:
        by_kind[row.get("kind") or "Node"].append(row)

    for name in list(by_kind) + list(by_type):
        if not safe.match(name):
            raise ValueError(f"refusing to build Cypher with unsafe identifier: {name!r}")

    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            session.run("MATCH (n:Provision) DETACH DELETE n")
            session.run("CREATE CONSTRAINT dpdp_id IF NOT EXISTS "
                        "FOR (n:Provision) REQUIRE n.id IS UNIQUE")
            # A second label per kind is what makes the Neo4j Browser legend
            # and kind-scoped queries useful.
            for kind, batch in by_kind.items():
                session.run(
                    f"UNWIND $batch AS row CREATE (n:Provision:{kind}) SET n = row",
                    batch=batch)
            for etype, edges in by_type.items():
                session.run(
                    f"UNWIND $edges AS e "
                    f"MATCH (a:Provision {{id: e.src}}), (b:Provision {{id: e.dst}}) "
                    f"CREATE (a)-[r:{etype}]->(b) "
                    f"SET r.quote = e.quote, r.resolution = e.resolution",
                    edges=edges)
            counts = session.run(
                "MATCH (n:Provision) WITH count(n) AS nodes "
                "MATCH (:Provision)-[r]->() RETURN nodes, count(r) AS rels"
            ).single()

    log.info("pushed to Neo4j: %d nodes, %d relationships",
             counts["nodes"], counts["rels"])
    return counts["nodes"], counts["rels"]


def build_graph(tree: Tree, defs, xrefs, citations, schedule, meta) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    max_section = max(int(n.id.split("-")[1]) for n in tree.nodes.values() if n.kind == "Section")

    for n in tree.nodes.values():
        G.add_node(n.id, label=n.label, kind=n.kind, text=n.text, headnote=n.headnote or "",
                   page=n.page, chapter=n.chapter or "", verbatim=True)
        if n.parent:
            G.add_edge(n.parent, n.id, key=f"HAS_{n.kind.upper()}", type=f"HAS_{n.kind.upper()}")

    for d in defs:
        G.add_node(d["id"], label=d["term"], kind="Definition", text=d["text"],
                   relation=d["relation"], verbatim=True)
        G.add_edge(d["defined_in"], d["id"], key="DEFINES", type="DEFINES")

    # MENTIONS: exact word-boundary match of each defined term. Deterministic and
    # exhaustive -- this is the recall that embeddings cannot guarantee.
    for n in tree.nodes.values():
        if not n.text:
            continue
        for d in defs:
            if d["defined_in"] != n.id and re.search(rf"\b{re.escape(d['term'])}\b", n.text):
                G.add_edge(n.id, d["id"], key="MENTIONS", type="MENTIONS")

    G.add_node("schedule", label="The Schedule", kind="Schedule",
               text="[See section 33 (1)]", verbatim=True)
    for row in schedule:
        rid = f"pen-{row['sl_no']}"
        G.add_node(rid, label=f"Schedule entry {row['sl_no']}", kind="Penalty",
                   text=row["breach"], penalty=row["penalty"], verbatim=True)
        G.add_edge("schedule", rid, key="HAS_ENTRY", type="HAS_ENTRY")
        # Which provision a penalty binds is stated in the breach text itself.
        for ref in scan_references(row["breach"], owner=None, max_section=max_section):
            if ref["target"] in G:
                G.add_edge(ref["target"], rid, key="PENALISED_BY", type="PENALISED_BY")

    for x in xrefs:
        G.add_edge(x["src"], x["dst"], key="REFERENCES", type="REFERENCES",
                   quote=x["quote"], resolution=x["resolution"])

    for c in citations:
        m = RE_EXTERNAL_CITATION.match(c["text"])
        G.add_node(f"ext-{m.group(2)}-{m.group(1)}", kind="ExternalAct",
                   label=f"Act {m.group(1)} of {m.group(2)}", text=c["text"],
                   page=c["page"], verbatim=True)

    G.graph.update(meta)
    # PageRank is computed here, once, as the last step of graph
    # construction — not left to the caller to remember, and not run twice.
    add_authority(G)
    return G


def act_metadata(tree: Tree) -> dict:
    """Act number and assent date, read off the title block rather than typed in."""
    head = tree.nodes["preamble"].text
    num = RE_ACT_NUMBER.search(head)
    date = RE_ASSENT_DATE.search(head)
    return {"title": "The Digital Personal Data Protection Act, 2023",
            "act_number": f"{num.group(1)} of {num.group(2)}" if num else "",
            "assent_date": date.group(1) if date else "",
            "source": PDF.name}

